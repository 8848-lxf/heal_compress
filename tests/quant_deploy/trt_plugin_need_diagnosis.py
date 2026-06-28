from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from deployment_equivalence import TensorRTEngineRunner, _load_model_context
from diagnose_scatternd_issue import hash_array
from export_lidar_pyramid_onnx import INPUT_NAMES, _extract_inputs, _first_real_sample
from quant_deploy_utils import ensure_quant_deploy_run_dirs, find_trtexec, read_json, save_json, write_summary_files
from subgraph_bisect_trt import _compare_arrays, _profile_args, build_augmented_engine, classify_bisect_stage


GRID_SAMPLE_CASES = [
    {
        "alias": "grid0_feature",
        "feature": "/layer0/layer0.2/relu_4/Relu_output_0",
        "grid": "/Cast_output_0",
        "output": "/GridSample_output_0",
        "height": 128,
        "width": 256,
    },
    {
        "alias": "grid1_feature",
        "feature": "/layer1/layer1.4/relu_2/Relu_output_0",
        "grid": "/Cast_1_output_0",
        "output": "/GridSample_2_output_0",
        "height": 64,
        "width": 128,
    },
    {
        "alias": "grid2_feature",
        "feature": "/layer2/layer2.7/relu_2/Relu_output_0",
        "grid": "/Cast_2_output_0",
        "output": "/GridSample_4_output_0",
        "height": 32,
        "width": 64,
    },
]

SCATTER_CASE = {
    "alias": "pointpillar_scatter",
    "data": "/encoder_m1/scatter/Constant_output_0",
    "indices": "/encoder_m1/scatter/Concat_output_0",
    "updates": "/encoder_m1/scatter/Reshape_output_0",
    "output": "/encoder_m1/scatter/ScatterND_output_0",
    "reshaped_output": "/encoder_m1/scatter/Reshape_1_output_0",
}

BACKBONE_DETAIL_OUTPUTS = [
    ("layer0_block0_conv1", "/layer0/layer0.0/conv1/Conv_output_0"),
    ("layer0_block0_relu", "/layer0/layer0.0/relu/Relu_output_0"),
    ("layer0_block0_conv2", "/layer0/layer0.0/conv2/Conv_output_0"),
    ("layer0_block0_downsample", "/layer0/layer0.0/downsample/downsample.0/Conv_output_0"),
    ("layer0_block0_add", "/layer0/layer0.0/Add_output_0"),
    ("layer0_block0_relu1", "/layer0/layer0.0/relu_1/Relu_output_0"),
    ("layer0_block1_relu", "/layer0/layer0.1/relu/Relu_output_0"),
    ("layer0_block1_relu1", "/layer0/layer0.1/relu_1/Relu_output_0"),
    ("layer0_block2_relu", "/layer0/layer0.2/relu/Relu_output_0"),
    ("layer0_block2_relu1", "/layer0/layer0.2/relu_1/Relu_output_0"),
    ("pyramid0_concat", "/Concat_2_output_0"),
    ("pyramid0_conv1", "/layer0/layer0.0/conv1_1/Conv_output_0"),
    ("pyramid0_relu2", "/layer0/layer0.0/relu_2/Relu_output_0"),
    ("pyramid0_conv2", "/layer0/layer0.0/conv2_1/Conv_output_0"),
    ("pyramid0_relu3", "/layer0/layer0.0/relu_3/Relu_output_0"),
    ("pyramid0_conv3", "/layer0/layer0.0/conv3/Conv_output_0"),
    ("pyramid0_add", "/layer0/layer0.0_1/Add_output_0"),
    ("feat0", "/layer0/layer0.2/relu_4/Relu_output_0"),
]


def make_io_mapping_report(
    onnx_inputs: list[str],
    onnx_outputs: list[str],
    trt_inputs: list[str],
    trt_outputs: list[str],
    runtime_feed_keys: list[str],
    runtime_output_keys: list[str],
) -> dict[str, Any]:
    input_name_match = sorted(onnx_inputs) == sorted(trt_inputs)
    output_name_match = sorted(onnx_outputs) == sorted(trt_outputs)
    runtime_feed_name_match = sorted(runtime_feed_keys) == sorted(onnx_inputs)
    runtime_output_name_match = sorted(runtime_output_keys) == sorted(trt_outputs)
    return {
        "onnx_inputs": list(onnx_inputs),
        "onnx_outputs": list(onnx_outputs),
        "trt_inputs": list(trt_inputs),
        "trt_outputs": list(trt_outputs),
        "runtime_feed_keys": list(runtime_feed_keys),
        "runtime_output_keys": list(runtime_output_keys),
        "name_based_io": True,
        "input_name_match": input_name_match,
        "output_name_match": output_name_match,
        "runtime_feed_name_match": runtime_feed_name_match,
        "runtime_output_name_match": runtime_output_name_match,
        "io_mapping_bug_found": not (
            input_name_match and output_name_match and runtime_feed_name_match and runtime_output_name_match
        ),
        "output_mapping_note": "TRT tensors are bound and returned by tensor name; positional outputs[0]/outputs[1]/outputs[2] semantics are not used.",
    }


def assess_plugin_need(
    first_mismatch_stage: str | None,
    gridsample_trt_equivalent: bool | None,
    scatternd_trt_equivalent: bool | None,
    inverse_or_solve_present: bool,
) -> dict[str, Any]:
    need_bevwarp = first_mismatch_stage == "GridSample/BEV warp" and gridsample_trt_equivalent is False
    need_scatter = first_mismatch_stage == "PointPillarScatter" and scatternd_trt_equivalent is False
    need_inverse = first_mismatch_stage == "Inverse/Solve" and bool(inverse_or_solve_present)
    plugin_needed = bool(need_bevwarp or need_scatter or need_inverse)

    recommended = None
    reason = None
    if need_bevwarp:
        recommended = "BEVWarpDynamicTRT"
        reason = "Minimal GridSample/BEV warp subgraph is not equivalent between TensorRT FP32 and ONNXRuntime FP32."
    elif need_scatter:
        recommended = "PointPillarScatterTRT or VoxelScatterBEVTRT"
        reason = "Minimal PointPillarScatter/ScatterND subgraph is not equivalent between TensorRT FP32 and ONNXRuntime FP32."
    elif need_inverse:
        recommended = "Inverse3x3TRT or SE3InverseTRT"
        reason = "An inverse/solve matrix transform exists in the LiDAR ONNX graph and is the first mismatch stage."
    elif first_mismatch_stage:
        reason = f"First mismatch stage is {first_mismatch_stage}, which does not map to an existing camera plugin recommendation."

    return {
        "plugin_needed": plugin_needed,
        "plugin_needed_reason": reason,
        "need_bevwarp_plugin": need_bevwarp,
        "need_bevpool_plugin": False,
        "need_inverse_plugin": need_inverse,
        "need_pointpillar_scatter_plugin": need_scatter,
        "recommended_plugin": recommended,
    }


def _onnx_model_io(onnx_path: Path) -> dict[str, list[str]]:
    import onnx

    model = onnx.load(str(onnx_path))
    return {
        "inputs": [value.name for value in model.graph.input],
        "outputs": [value.name for value in model.graph.output],
    }


def _available_tensors(onnx_path: Path) -> set[str]:
    import onnx

    model = onnx.load(str(onnx_path))
    names = {value.name for value in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info)}
    for node in model.graph.node:
        names.update(node.output)
    return names


def _has_inverse_or_solve(onnx_path: Path) -> bool:
    import onnx

    model = onnx.load(str(onnx_path))
    return any(node.op_type.lower() in {"inverse", "solve"} for node in model.graph.node)


def _augment_outputs(onnx_path: Path, output_path: Path, output_names: list[str]) -> list[str]:
    import onnx
    from onnx import TensorProto, helper, shape_inference

    model = shape_inference.infer_shapes(onnx.load(str(onnx_path)))
    available = _available_tensors(onnx_path)
    existing = {value.name for value in model.graph.output}
    value_info_by_name = {
        value.name: value
        for value in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info)
    }
    selected = []
    for name in output_names:
        if name not in available:
            continue
        selected.append(name)
        if name not in existing:
            if name in value_info_by_name:
                model.graph.output.append(value_info_by_name[name])
            else:
                model.graph.output.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, None))
            existing.add(name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output_path))
    return selected


def _run_ort(onnx_path: Path, feeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    import onnxruntime as ort

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    available = ort.get_available_providers()
    providers = [provider for provider in providers if provider in available] or available
    session = ort.InferenceSession(str(onnx_path), providers=providers)
    values = session.run(None, feeds)
    return {output.name: value for output, value in zip(session.get_outputs(), values)}


def _trt_io_names(engine_path: Path, device: torch.device) -> dict[str, list[str]]:
    runner = TensorRTEngineRunner(engine_path, device)
    trt = runner.trt
    inputs = []
    outputs = []
    all_tensors = []
    for index in range(runner.engine.num_io_tensors):
        name = runner.engine.get_tensor_name(index)
        mode = runner.engine.get_tensor_mode(name)
        item = {"index": index, "name": name, "mode": str(mode)}
        all_tensors.append(item)
        if mode == trt.TensorIOMode.INPUT:
            inputs.append(name)
        else:
            outputs.append(name)
    return {"inputs": inputs, "outputs": outputs, "all_tensors": all_tensors}


def _build_engine_for_onnx(
    onnx_path: Path,
    engine_path: Path,
    layerinfo_path: Path,
    profile_shapes: dict[str, Any],
    trt_root: str | None,
    trtexec_path: str | None,
    timeout: int,
    skip_inference: bool = False,
) -> dict[str, Any]:
    trtexec = trtexec_path or find_trtexec(trt_root=trt_root, explicit_trtexec=None)
    if not trtexec:
        return {"success": False, "error": "trtexec not found"}
    cmd = [
        trtexec,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        "--profilingVerbosity=detailed",
        "--dumpLayerInfo",
        f"--exportLayerInfo={layerinfo_path}",
        "--verbose",
        "--noTF32",
    ]
    if profile_shapes:
        cmd.extend(_profile_args(profile_shapes))
    if skip_inference:
        cmd.append("--skipInference")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    log_path = engine_path.with_suffix(".build.log")
    log_path.write_text(" ".join(cmd) + "\n\n" + proc.stdout + "\n\n" + proc.stderr, encoding="utf-8")
    return {
        "success": proc.returncode == 0 and engine_path.exists(),
        "returncode": proc.returncode,
        "command": cmd,
        "log_path": str(log_path),
        "engine_path": str(engine_path),
        "error": None if proc.returncode == 0 and engine_path.exists() else (proc.stderr or proc.stdout)[-2000:],
    }


def _make_gridsample_onnx(path: Path, case: dict[str, Any], feature: np.ndarray, grid: np.ndarray) -> None:
    import onnx
    from onnx import TensorProto, helper

    node = helper.make_node(
        "GridSample",
        ["feature", "grid"],
        ["output"],
        name=f"{case['alias']}_GridSample",
        mode="bilinear",
        padding_mode="zeros",
        align_corners=0,
    )
    graph = helper.make_graph(
        [node],
        f"{case['alias']}_graph",
        [
            helper.make_tensor_value_info("feature", TensorProto.FLOAT, list(feature.shape)),
            helper.make_tensor_value_info("grid", TensorProto.FLOAT, list(grid.shape)),
        ],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [feature.shape[0], feature.shape[1], grid.shape[1], grid.shape[2]])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def _make_scatternd_onnx(path: Path, data: np.ndarray, indices: np.ndarray, updates: np.ndarray) -> None:
    import onnx
    from onnx import TensorProto, helper

    node = helper.make_node("ScatterND", ["data", "indices", "updates"], ["output"], name="PointPillarScatterND")
    graph = helper.make_graph(
        [node],
        "pointpillar_scatternd_graph",
        [
            helper.make_tensor_value_info("data", TensorProto.FLOAT, list(data.shape)),
            helper.make_tensor_value_info("indices", TensorProto.INT64, list(indices.shape)),
            helper.make_tensor_value_info("updates", TensorProto.FLOAT, list(updates.shape)),
        ],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, list(data.shape))],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def _fixed_profile_from_arrays(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    return {name: {"min": list(value.shape), "opt": list(value.shape), "max": list(value.shape)} for name, value in arrays.items()}


def _to_torch_inputs(arrays: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: torch.from_numpy(value).to(device) for name, value in arrays.items()}


def _hashes(arrays: dict[str, np.ndarray]) -> dict[str, str]:
    return {name: hash_array(value) for name, value in arrays.items()}


def run_minimal_gridsample_checks(
    dirs: dict[str, Path],
    full_outputs: dict[str, np.ndarray],
    device: torch.device,
    trt_root: str | None,
    trtexec_path: str | None,
    timeout: int,
    threshold: float,
) -> dict[str, Any]:
    comparisons = []
    for case in GRID_SAMPLE_CASES:
        missing = [name for name in (case["feature"], case["grid"], case["output"]) if name not in full_outputs]
        item: dict[str, Any] = {"alias": case["alias"], "case": case, "missing_inputs": missing}
        if missing:
            comparisons.append(item)
            continue

        feature = np.asarray(full_outputs[case["feature"]], dtype=np.float32)
        grid = np.asarray(full_outputs[case["grid"]], dtype=np.float32)
        expected = np.asarray(full_outputs[case["output"]], dtype=np.float32)
        mini_onnx = dirs["debug"] / f"{case['alias']}_minimal_gridsample.onnx"
        engine_path = dirs["debug"] / f"{case['alias']}_minimal_gridsample_fp32.engine"
        layerinfo_path = dirs["debug"] / f"{case['alias']}_minimal_gridsample_layerinfo.json"
        _make_gridsample_onnx(mini_onnx, case, feature, grid)
        feeds = {"feature": feature, "grid": grid}
        build_report = _build_engine_for_onnx(
            mini_onnx,
            engine_path,
            layerinfo_path,
            {},
            trt_root,
            trtexec_path,
            timeout,
        )
        ort_outputs = _run_ort(mini_onnx, feeds)
        item.update(
            {
                "onnx_path": str(mini_onnx),
                "engine_path": str(engine_path),
                "build_report": build_report,
                "input_hashes": _hashes(feeds),
                "ort_vs_full_graph_ort": _compare_arrays(expected, ort_outputs["output"]),
            }
        )
        if build_report.get("success"):
            trt_outputs = TensorRTEngineRunner(engine_path, device).run(_to_torch_inputs(feeds, device))
            trt_np = trt_outputs["output"].detach().float().cpu().numpy()
            item["trt_vs_ort"] = _compare_arrays(ort_outputs["output"], trt_np)
        comparisons.append(item)

    max_errors = [
        float(item["trt_vs_ort"]["max_abs_error"])
        for item in comparisons
        if item.get("trt_vs_ort", {}).get("max_abs_error") is not None
    ]
    report = {
        "op": "GridSample",
        "threshold": threshold,
        "comparisons": comparisons,
        "gridsample_trt_equivalent": bool(max_errors) and max(max_errors) <= threshold,
        "max_abs_error": max(max_errors) if max_errors else None,
    }
    if not max_errors:
        report["gridsample_trt_equivalent"] = None
    save_json(report, dirs["debug"] / "gridsample_trt_check.json")
    return report


def run_minimal_scatternd_check(
    dirs: dict[str, Path],
    full_outputs: dict[str, np.ndarray],
    device: torch.device,
    trt_root: str | None,
    trtexec_path: str | None,
    timeout: int,
    threshold: float,
) -> dict[str, Any]:
    missing = [name for name in (SCATTER_CASE["data"], SCATTER_CASE["indices"], SCATTER_CASE["updates"], SCATTER_CASE["output"]) if name not in full_outputs]
    report: dict[str, Any] = {"op": "ScatterND", "case": SCATTER_CASE, "threshold": threshold, "missing_inputs": missing}
    if missing:
        report["scatternd_trt_equivalent"] = None
        save_json(report, dirs["debug"] / "scatternd_trt_check.json")
        return report

    data = np.asarray(full_outputs[SCATTER_CASE["data"]], dtype=np.float32)
    indices = np.asarray(full_outputs[SCATTER_CASE["indices"]], dtype=np.int64)
    updates = np.asarray(full_outputs[SCATTER_CASE["updates"]], dtype=np.float32)
    expected = np.asarray(full_outputs[SCATTER_CASE["output"]], dtype=np.float32)
    feeds = {"data": data, "indices": indices, "updates": updates}
    mini_onnx = dirs["debug"] / "pointpillar_scatternd_minimal.onnx"
    engine_path = dirs["debug"] / "pointpillar_scatternd_minimal_fp32.engine"
    layerinfo_path = dirs["debug"] / "pointpillar_scatternd_minimal_layerinfo.json"
    _make_scatternd_onnx(mini_onnx, data, indices, updates)
    build_report = _build_engine_for_onnx(
        mini_onnx,
        engine_path,
        layerinfo_path,
        {},
        trt_root,
        trtexec_path,
        timeout,
        skip_inference=True,
    )
    ort_outputs = _run_ort(mini_onnx, feeds)
    report.update(
        {
            "onnx_path": str(mini_onnx),
            "engine_path": str(engine_path),
            "build_report": build_report,
            "input_hashes": _hashes(feeds),
            "ort_vs_full_graph_ort": _compare_arrays(expected, ort_outputs["output"]),
        }
    )
    engine_ready = engine_path.exists()
    if engine_ready:
        trt_outputs = TensorRTEngineRunner(engine_path, device).run(_to_torch_inputs(feeds, device))
        trt_np = trt_outputs["output"].detach().float().cpu().numpy()
        report["trt_vs_ort"] = _compare_arrays(ort_outputs["output"], trt_np)
        max_abs = report["trt_vs_ort"]["max_abs_error"]
        report["scatternd_trt_equivalent"] = max_abs is not None and float(max_abs) <= threshold
    else:
        report["scatternd_trt_equivalent"] = None
    save_json(report, dirs["debug"] / "scatternd_trt_check.json")
    return report


def run_detailed_backbone_bisect(
    args: argparse.Namespace,
    dirs: dict[str, Path],
    tensors_by_name: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, Any]:
    onnx_path = Path(args.onnx_path)
    selected_names = _augment_outputs(
        onnx_path,
        dirs["debug"] / "backbone_detail_augmented.onnx",
        [name for _alias, name in BACKBONE_DETAIL_OUTPUTS],
    )
    profile_shapes = read_json(dirs["configs"] / "profile_shapes.json", default={}) or {}
    engine_path = dirs["debug"] / "backbone_detail_fp32.engine"
    build_report = build_augmented_engine(
        dirs["debug"] / "backbone_detail_augmented.onnx",
        engine_path,
        dirs["debug"] / "backbone_detail_layerinfo.json",
        profile_shapes,
        args.trt_root,
        args.trtexec_path,
        args.timeout,
    )
    report: dict[str, Any] = {"selected_outputs": selected_names, "build_report": build_report, "comparisons": []}
    if not build_report.get("success"):
        save_json(report, dirs["debug"] / "backbone_detail_bisect_report.json")
        return report

    feeds = {name: tensor.detach().cpu().numpy() for name, tensor in tensors_by_name.items()}
    ort_outputs = _run_ort(dirs["debug"] / "backbone_detail_augmented.onnx", feeds)
    trt_outputs = TensorRTEngineRunner(engine_path, device).run(tensors_by_name)
    trt_np = {name: value.detach().float().cpu().numpy() for name, value in trt_outputs.items()}
    for alias, name in BACKBONE_DETAIL_OUTPUTS:
        if name not in selected_names:
            continue
        report["comparisons"].append(
            {
                "alias": alias,
                "tensor_name": name,
                "stage": classify_bisect_stage(name),
                **_compare_arrays(ort_outputs[name], trt_np[name]),
            }
        )
    report["first_mismatch_tensor"] = next(
        (item for item in report["comparisons"] if (item.get("max_abs_error") or 0.0) > args.threshold),
        None,
    )
    save_json(report, dirs["debug"] / "backbone_detail_bisect_report.json")
    return report


def run_diagnosis(args: argparse.Namespace) -> dict[str, Any]:
    dirs = ensure_quant_deploy_run_dirs(args.output_root)
    onnx_path = Path(args.onnx_path)

    hypes, device, _model, modality = _load_model_context(args)
    sample = _first_real_sample(hypes, device)
    tensors, _agent_modalities = _extract_inputs(sample, modality)
    tensors_by_name = {name: tensor for name, tensor in zip(INPUT_NAMES, tensors)}
    feeds = {name: tensor.detach().cpu().numpy() for name, tensor in tensors_by_name.items()}

    full_intermediate_names = []
    for case in GRID_SAMPLE_CASES:
        full_intermediate_names.extend([case["feature"], case["grid"], case["output"]])
    full_intermediate_names.extend(
        [SCATTER_CASE["data"], SCATTER_CASE["indices"], SCATTER_CASE["updates"], SCATTER_CASE["output"], SCATTER_CASE["reshaped_output"]]
    )
    augmented_full = dirs["debug"] / "plugin_need_intermediates.onnx"
    selected_full_outputs = _augment_outputs(onnx_path, augmented_full, full_intermediate_names)
    full_outputs = _run_ort(augmented_full, feeds)

    trt_engine_path = Path(args.fp32_engine_path) if args.fp32_engine_path else dirs["engine_fp32"] / "lidar_pyramid_fp32.engine"
    onnx_io = _onnx_model_io(onnx_path)
    trt_io = _trt_io_names(trt_engine_path, device) if trt_engine_path.exists() else {"inputs": [], "outputs": [], "all_tensors": []}
    io_report = make_io_mapping_report(
        onnx_inputs=onnx_io["inputs"],
        onnx_outputs=onnx_io["outputs"],
        trt_inputs=trt_io["inputs"],
        trt_outputs=trt_io["outputs"],
        runtime_feed_keys=list(tensors_by_name.keys()),
        runtime_output_keys=trt_io["outputs"],
    )
    io_report["trt_all_tensors"] = trt_io["all_tensors"]
    save_json(io_report, dirs["debug"] / "trt_io_binding_report.json")
    save_json({"output_names": trt_io["outputs"], "mapping": {name: name for name in trt_io["outputs"]}}, dirs["configs"] / "output_name_mapping.json")

    gridsample_report = run_minimal_gridsample_checks(dirs, full_outputs, device, args.trt_root, args.trtexec_path, args.timeout, args.threshold)
    scatter_report = run_minimal_scatternd_check(dirs, full_outputs, device, args.trt_root, args.trtexec_path, args.timeout, args.threshold)
    backbone_report = run_detailed_backbone_bisect(args, dirs, tensors_by_name, device)

    previous_bisect = read_json(dirs["debug"] / "subgraph_bisect_report.json", default={}) or {}
    first_stage = previous_bisect.get("first_mismatch_stage")
    detail_first = (backbone_report or {}).get("first_mismatch_tensor") or {}
    if detail_first.get("tensor_name") == "/Concat_2_output_0":
        first_stage = "BEV warp grid construction"
    if first_stage == "BEV backbone / pyramid backbone":
        grid_max = gridsample_report.get("max_abs_error")
        if grid_max is not None and float(grid_max) > args.threshold:
            first_stage = "GridSample/BEV warp"

    inverse_present = _has_inverse_or_solve(onnx_path)
    assessment = assess_plugin_need(
        first_mismatch_stage=first_stage,
        gridsample_trt_equivalent=gridsample_report.get("gridsample_trt_equivalent"),
        scatternd_trt_equivalent=scatter_report.get("scatternd_trt_equivalent"),
        inverse_or_solve_present=inverse_present,
    )
    report = {
        "onnx_path": str(onnx_path),
        "fp32_engine_path": str(trt_engine_path),
        "selected_full_intermediate_outputs": selected_full_outputs,
        "input_hashes": {name: hash_array(value) for name, value in feeds.items()},
        "io_mapping_bug_found": io_report["io_mapping_bug_found"],
        "first_mismatch_stage": first_stage,
        "original_subgraph_first_mismatch_stage": previous_bisect.get("first_mismatch_stage"),
        "inverse_or_solve_present": inverse_present,
        "gridsample_trt_equivalent": gridsample_report.get("gridsample_trt_equivalent"),
        "scatternd_trt_equivalent": scatter_report.get("scatternd_trt_equivalent"),
        "gridsample_report_path": str(dirs["debug"] / "gridsample_trt_check.json"),
        "scatternd_report_path": str(dirs["debug"] / "scatternd_trt_check.json"),
        "backbone_detail_report_path": str(dirs["debug"] / "backbone_detail_bisect_report.json"),
        "plugin_assessment": assessment,
    }
    save_json(report, dirs["debug"] / "plugin_need_diagnosis_report.json")
    update_summary_with_plugin_diagnosis(dirs, report)
    return report


def update_summary_with_plugin_diagnosis(dirs: dict[str, Path], report: dict[str, Any]) -> None:
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
            "recommended_plugin": assessment["recommended_plugin"],
            "first_mismatch_stage": report["first_mismatch_stage"],
            "io_mapping_bug_found": report["io_mapping_bug_found"],
            "gridsample_trt_equivalent": report["gridsample_trt_equivalent"],
            "scatternd_trt_equivalent": report["scatternd_trt_equivalent"],
        }
    )
    write_summary_files(summary, dirs)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decide whether TensorRT mismatch requires a plugin, and which plugin.")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--onnx_path", required=True)
    parser.add_argument("--fp32_engine_path", default=None)
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
    report = run_diagnosis(parse_args(argv))
    print(
        json.dumps(
            {
                "io_mapping_bug_found": report["io_mapping_bug_found"],
                "first_mismatch_stage": report["first_mismatch_stage"],
                "gridsample_trt_equivalent": report["gridsample_trt_equivalent"],
                "scatternd_trt_equivalent": report["scatternd_trt_equivalent"],
                **report["plugin_assessment"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
