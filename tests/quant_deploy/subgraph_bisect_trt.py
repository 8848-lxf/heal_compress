from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from deployment_equivalence import TensorRTEngineRunner, _load_model_context
from diagnose_scatternd_issue import hash_array
from export_lidar_pyramid_onnx import INPUT_NAMES, _extract_inputs, _first_real_sample
from quant_deploy_utils import ensure_quant_deploy_run_dirs, find_trtexec, read_json, save_json


BISECT_CANDIDATES = [
    ("pillar_features", "/encoder_m1/pillar_vfe/Squeeze_output_0"),
    ("spatial_features", "/encoder_m1/scatter/Reshape_1_output_0"),
    ("feat0", "/layer0/layer0.2/relu_4/Relu_output_0"),
    ("feat1", "/layer1/layer1.4/relu_2/Relu_output_0"),
    ("feat2", "/layer2/layer2.7/relu_2/Relu_output_0"),
    ("warp0_feature", "/GridSample_output_0"),
    ("warp0_score", "/GridSample_1_output_0"),
    ("warp1_feature", "/GridSample_2_output_0"),
    ("warp1_score", "/GridSample_3_output_0"),
    ("warp2_feature", "/GridSample_4_output_0"),
    ("warp2_score", "/GridSample_5_output_0"),
    ("fused0", "/ReduceSum_output_0"),
    ("fused1", "/ReduceSum_1_output_0"),
    ("fused2", "/ReduceSum_2_output_0"),
    ("pyramid_decoded", "/Concat_9_output_0"),
    ("shrink_conv", "/shrink_conv/layers.0/double_conv/double_conv.3/Relu_output_0"),
    ("cls_head", "cls_preds"),
    ("reg_head", "reg_preds"),
    ("dir_head", "dir_preds"),
]


def classify_bisect_stage(tensor_name: str) -> str:
    lowered = tensor_name.lower()
    if "pillar_vfe" in lowered:
        return "PillarVFE"
    if "/scatter/" in lowered:
        return "PointPillarScatter"
    if "gridsample" in lowered:
        return "GridSample/BEV warp"
    if "reducesum" in lowered or "concat_9" in lowered:
        return "Pyramid fusion"
    if "layer0" in lowered or "layer1" in lowered or "layer2" in lowered:
        return "BEV backbone / pyramid backbone"
    if "shrink_conv" in lowered:
        return "shrink_conv"
    if "cls" in lowered or "reg" in lowered or "dir" in lowered:
        return "heads"
    return "unknown"


def find_first_mismatch_stage(comparisons: list[dict[str, Any]], threshold: float = 1.0e-3) -> str | None:
    for item in comparisons:
        value = item.get("max_abs_error")
        if value is not None and float(value) > threshold:
            return item.get("stage")
    return None


def _onnx_io_names(onnx_path: Path) -> dict[str, list[str]]:
    import onnx

    model = onnx.load(str(onnx_path))
    return {
        "inputs": [value.name for value in model.graph.input],
        "outputs": [value.name for value in model.graph.output],
    }


def _available_tensors(onnx_path: Path) -> set[str]:
    import onnx

    model = onnx.load(str(onnx_path))
    tensors: set[str] = set()
    for value in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        tensors.add(value.name)
    for node in model.graph.node:
        tensors.update(node.output)
    return tensors


def make_augmented_onnx(onnx_path: Path, output_path: Path) -> list[dict[str, str]]:
    import onnx
    from onnx import TensorProto, helper, shape_inference

    inferred = shape_inference.infer_shapes(onnx.load(str(onnx_path)))
    existing_outputs = {value.name for value in inferred.graph.output}
    available = _available_tensors(onnx_path)
    selected: list[dict[str, str]] = []
    for alias, tensor_name in BISECT_CANDIDATES:
        if tensor_name not in available:
            continue
        selected.append({"alias": alias, "tensor_name": tensor_name, "stage": classify_bisect_stage(tensor_name)})
        if tensor_name not in existing_outputs:
            inferred.graph.output.append(helper.make_tensor_value_info(tensor_name, TensorProto.FLOAT, None))
            existing_outputs.add(tensor_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(inferred, str(output_path))
    return selected


def _profile_args(profile_shapes: dict[str, Any]) -> list[str]:
    parts = {}
    for key in ("min", "opt", "max"):
        items = []
        for name, shape in profile_shapes.items():
            dims = "x".join(str(int(v)) for v in shape[key])
            items.append(f"{name}:{dims}")
        parts[key] = ",".join(items)
    return [f"--minShapes={parts['min']}", f"--optShapes={parts['opt']}", f"--maxShapes={parts['max']}"]


def build_augmented_engine(
    augmented_onnx: Path,
    engine_path: Path,
    layerinfo_path: Path,
    profile_shapes: dict[str, Any],
    trt_root: str | None,
    trtexec_path: str | None,
    timeout: int,
) -> dict[str, Any]:
    trtexec = trtexec_path or find_trtexec(trt_root=trt_root, explicit_trtexec=None)
    if not trtexec:
        return {"success": False, "error": "trtexec not found"}
    cmd = [
        trtexec,
        f"--onnx={augmented_onnx}",
        f"--saveEngine={engine_path}",
        "--profilingVerbosity=detailed",
        "--dumpLayerInfo",
        f"--exportLayerInfo={layerinfo_path}",
        "--verbose",
        "--noTF32",
    ]
    cmd.extend(_profile_args(profile_shapes))
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    log_path = engine_path.parent / "build_subgraph_bisect_fp32.log"
    log_path.write_text(" ".join(cmd) + "\n\n" + proc.stdout + "\n\n" + proc.stderr, encoding="utf-8")
    return {
        "success": proc.returncode == 0 and engine_path.exists(),
        "returncode": proc.returncode,
        "command": cmd,
        "log_path": str(log_path),
        "engine_path": str(engine_path),
        "error": None if proc.returncode == 0 and engine_path.exists() else (proc.stderr or proc.stdout)[-2000:],
    }


def _to_numpy_dict(outputs: dict[str, Any]) -> dict[str, np.ndarray]:
    result = {}
    for name, value in outputs.items():
        if torch.is_tensor(value):
            result[name] = value.detach().float().cpu().numpy()
        else:
            result[name] = np.asarray(value, dtype=np.float32)
    return result


def _compare_arrays(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    if reference.shape != candidate.shape:
        return {
            "same_shape": False,
            "reference_shape": list(reference.shape),
            "candidate_shape": list(candidate.shape),
            "max_abs_error": None,
            "mean_abs_error": None,
            "relative_error": None,
        }
    ref = reference.astype(np.float32, copy=False)
    cand = candidate.astype(np.float32, copy=False)
    diff = np.abs(ref - cand)
    denom = max(float(np.mean(np.abs(ref))) if ref.size else 0.0, 1.0e-12)
    return {
        "same_shape": True,
        "reference_shape": list(ref.shape),
        "candidate_shape": list(cand.shape),
        "max_abs_error": float(diff.max()) if diff.size else 0.0,
        "mean_abs_error": float(diff.mean()) if diff.size else 0.0,
        "relative_error": float(diff.mean() / denom) if diff.size else 0.0,
    }


def _run_ort(onnx_path: Path, tensors_by_name: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    import onnxruntime as ort

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    available = ort.get_available_providers()
    providers = [p for p in providers if p in available] or available
    session = ort.InferenceSession(str(onnx_path), providers=providers)
    inputs = {name: tensor.detach().cpu().numpy() for name, tensor in tensors_by_name.items()}
    values = session.run(None, inputs)
    names = [output.name for output in session.get_outputs()]
    return {name: value for name, value in zip(names, values)}


def _trt_io_names(engine_path: Path, device: torch.device) -> dict[str, Any]:
    runner = TensorRTEngineRunner(engine_path, device)
    trt = runner.trt
    names = []
    inputs = []
    outputs = []
    for index in range(runner.engine.num_io_tensors):
        name = runner.engine.get_tensor_name(index)
        mode = runner.engine.get_tensor_mode(name)
        item = {"index": index, "name": name, "mode": str(mode)}
        names.append(item)
        if mode == trt.TensorIOMode.INPUT:
            inputs.append(name)
        else:
            outputs.append(name)
    return {"all_tensors": names, "inputs": inputs, "outputs": outputs}


def run_bisect(args: argparse.Namespace) -> dict[str, Any]:
    dirs = ensure_quant_deploy_run_dirs(args.output_root)
    onnx_path = Path(args.onnx_path)
    augmented_onnx = dirs["debug"] / "subgraph_bisect_augmented.onnx"
    selected = make_augmented_onnx(onnx_path, augmented_onnx)
    profile_shapes = read_json(dirs["configs"] / "profile_shapes.json", default={}) or {}
    engine_path = dirs["debug"] / "subgraph_bisect_fp32.engine"
    layerinfo_path = dirs["debug"] / "subgraph_bisect_layerinfo.json"
    build_report = build_augmented_engine(
        augmented_onnx,
        engine_path,
        layerinfo_path,
        profile_shapes,
        args.trt_root,
        args.trtexec_path,
        args.timeout,
    )

    hypes, device, _model, modality = _load_model_context(args)
    sample = _first_real_sample(hypes, device)
    tensors, _agent_modalities = _extract_inputs(sample, modality)
    tensors_by_name = {name: tensor for name, tensor in zip(INPUT_NAMES, tensors)}
    input_hashes = {name: hash_array(tensor) for name, tensor in tensors_by_name.items()}

    onnx_io = _onnx_io_names(augmented_onnx)
    trt_io = _trt_io_names(engine_path, device) if build_report["success"] else {"inputs": [], "outputs": [], "all_tensors": []}
    io_report = {
        "onnx_inputs": onnx_io["inputs"],
        "onnx_outputs": onnx_io["outputs"],
        "trt_inputs": trt_io["inputs"],
        "trt_outputs": trt_io["outputs"],
        "trt_all_tensors": trt_io["all_tensors"],
        "runtime_feed_keys": list(tensors_by_name.keys()),
        "runtime_output_keys": trt_io["outputs"],
        "name_based_io": True,
        "input_name_match": sorted(onnx_io["inputs"]) == sorted(trt_io["inputs"]),
        "output_mapping_note": "TRT outputs are read by tensor name from execute_async_v3, never by positional outputs[0]/outputs[1]/outputs[2].",
    }
    save_json(io_report, dirs["debug"] / "trt_io_binding_report.json")
    save_json({"output_names": trt_io["outputs"], "selected_intermediate_outputs": selected}, dirs["configs"] / "output_name_mapping.json")

    comparisons = []
    if build_report["success"]:
        ort_outputs = _run_ort(augmented_onnx, tensors_by_name)
        runner = TensorRTEngineRunner(engine_path, device)
        trt_outputs = _to_numpy_dict(runner.run(tensors_by_name))
        for item in selected:
            name = item["tensor_name"]
            if name not in ort_outputs or name not in trt_outputs:
                comparisons.append({**item, "missing": True, "in_ort": name in ort_outputs, "in_trt": name in trt_outputs})
                continue
            comparisons.append(
                {
                    **item,
                    **_compare_arrays(np.asarray(ort_outputs[name]), np.asarray(trt_outputs[name])),
                }
            )

    first_mismatch = find_first_mismatch_stage(comparisons, threshold=args.threshold)
    report = {
        "onnx_path": str(onnx_path),
        "augmented_onnx_path": str(augmented_onnx),
        "build_report": build_report,
        "io_mapping_bug_found": not bool(io_report["input_name_match"]),
        "io_report_path": str(dirs["debug"] / "trt_io_binding_report.json"),
        "output_name_mapping_path": str(dirs["configs"] / "output_name_mapping.json"),
        "input_hashes": input_hashes,
        "selected_outputs": selected,
        "comparisons": comparisons,
        "first_mismatch_stage": first_mismatch,
        "threshold": args.threshold,
        "plugin_assessment": plugin_assessment(first_mismatch, comparisons),
    }
    save_json(report, dirs["debug"] / "subgraph_bisect_report.json")
    update_summary_with_bisect(dirs, report)
    return report


def plugin_assessment(first_mismatch_stage: str | None, comparisons: list[dict[str, Any]]) -> dict[str, Any]:
    need_bevwarp = first_mismatch_stage == "GridSample/BEV warp"
    need_scatter = first_mismatch_stage == "PointPillarScatter"
    need_inverse = first_mismatch_stage == "Inverse/Solve"
    plugin_needed = bool(need_bevwarp or need_scatter or need_inverse)
    recommended = None
    reason = None
    if need_bevwarp:
        recommended = "BEVWarpDynamicTRT"
        reason = "First mismatch appears at GridSample/BEV warp stage."
    elif need_scatter:
        recommended = "PointPillarScatterTRT or VoxelScatterBEVTRT"
        reason = "First mismatch appears at PointPillarScatter stage."
    elif need_inverse:
        recommended = "Inverse3x3TRT or SE3InverseTRT"
        reason = "First mismatch appears at inverse/solve matrix transform stage."
    return {
        "plugin_needed": plugin_needed if plugin_needed else False,
        "plugin_needed_reason": reason,
        "need_bevwarp_plugin": need_bevwarp,
        "need_bevpool_plugin": False,
        "need_inverse_plugin": need_inverse,
        "need_pointpillar_scatter_plugin": need_scatter,
        "recommended_plugin": recommended,
    }


def update_summary_with_bisect(dirs: dict[str, Path], report: dict[str, Any]) -> None:
    from quant_deploy_utils import write_summary_files

    summary = read_json(dirs["summary"] / "summary_all.json", default={}) or {}
    assessment = report["plugin_assessment"]
    summary.update(
        {
            "plugin_needed": assessment["plugin_needed"],
            "plugin_needed_reason": assessment["plugin_needed_reason"],
            "need_bevwarp_plugin": assessment["need_bevwarp_plugin"],
            "need_bevpool_plugin": assessment["need_bevpool_plugin"],
            "need_inverse_plugin": assessment["need_inverse_plugin"],
            "need_pointpillar_scatter_plugin": assessment["need_pointpillar_scatter_plugin"],
            "first_mismatch_stage": report["first_mismatch_stage"],
            "io_mapping_bug_found": report["io_mapping_bug_found"],
            "gridsample_trt_equivalent": None,
            "scatternd_trt_equivalent": None,
            "recommended_plugin": assessment["recommended_plugin"],
        }
    )
    write_summary_files(summary, dirs)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bisect ONNXRuntime vs TensorRT FP32 mismatches by intermediate tensors.")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--onnx_path", required=True)
    parser.add_argument("--hypes_yaml", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--heal_repo", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--trt_root", default=None)
    parser.add_argument("--trtexec_path", default=None)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--threshold", type=float, default=1.0e-3)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    report = run_bisect(parse_args(argv))
    print(json.dumps({"first_mismatch_stage": report["first_mismatch_stage"], **report["plugin_assessment"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
