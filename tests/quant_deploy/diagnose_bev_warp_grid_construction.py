from __future__ import annotations

import argparse
import json
import subprocess
import sys
import traceback
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
from subgraph_bisect_trt import _profile_args


TARGET_THETA_TENSOR = "/Concat_2_output_0"
TARGET_GRID_TENSORS = ["/Cast_output_0", "/Cast_1_output_0", "/Cast_2_output_0"]
KEY_OP_TYPES = {
    "Gather",
    "Slice",
    "Cast",
    "Div",
    "Mul",
    "Add",
    "Unsqueeze",
    "Reshape",
    "Concat",
    "MatMul",
    "Expand",
    "Shape",
    "ConstantOfShape",
    "Where",
    "Equal",
    "Transpose",
    "Constant",
}


def _tensor_shape_dtype_from_value_info(value_info: Any) -> dict[str, Any]:
    import onnx

    tensor_type = value_info.type.tensor_type
    dtype = None
    if tensor_type.elem_type:
        dtype = onnx.TensorProto.DataType.Name(tensor_type.elem_type)
    shape = []
    if tensor_type.HasField("shape"):
        for dim in tensor_type.shape.dim:
            if dim.HasField("dim_value"):
                shape.append(int(dim.dim_value))
            elif dim.HasField("dim_param"):
                shape.append(dim.dim_param)
            else:
                shape.append(None)
    return {"dtype": dtype, "shape": shape}


def _tensor_shape_dtype_from_initializer(initializer: Any) -> dict[str, Any]:
    import onnx

    return {
        "dtype": onnx.TensorProto.DataType.Name(initializer.data_type),
        "shape": [int(dim) for dim in initializer.dims],
    }


def _value_info_map(model: Any) -> dict[str, dict[str, Any]]:
    info: dict[str, dict[str, Any]] = {}
    for value in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        info[value.name] = _tensor_shape_dtype_from_value_info(value)
    for initializer in model.graph.initializer:
        info.setdefault(initializer.name, _tensor_shape_dtype_from_initializer(initializer))
    return info


def _array_summary(array: np.ndarray, *, max_values: int = 16) -> dict[str, Any]:
    arr = np.asarray(array)
    flat = arr.reshape(-1) if arr.size else arr
    summary: dict[str, Any] = {
        "dtype": str(arr.dtype),
        "shape": list(arr.shape),
        "numel": int(arr.size),
    }
    if arr.size and np.issubdtype(arr.dtype, np.number):
        numeric = arr.astype(np.float64, copy=False)
        summary.update(
            {
                "min": float(np.min(numeric)),
                "max": float(np.max(numeric)),
                "mean": float(np.mean(numeric)),
            }
        )
    if arr.size <= max_values:
        summary["values"] = flat.tolist()
    else:
        summary["first_values"] = flat[:max_values].tolist()
    return summary


def _attribute_to_json(attribute: Any) -> Any:
    import onnx
    from onnx import numpy_helper

    value = onnx.helper.get_attribute_value(attribute)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, onnx.TensorProto):
        return {"tensor": _array_summary(numpy_helper.to_array(value))}
    if isinstance(value, onnx.GraphProto):
        return {"graph_name": value.name, "num_nodes": len(value.node)}
    if isinstance(value, (list, tuple)):
        result = []
        for item in value:
            if isinstance(item, bytes):
                result.append(item.decode("utf-8", errors="replace"))
            elif isinstance(item, onnx.TensorProto):
                result.append({"tensor": _array_summary(numpy_helper.to_array(item))})
            else:
                result.append(item)
        return result
    return value


def _node_attributes(node: Any) -> dict[str, Any]:
    return {attribute.name: _attribute_to_json(attribute) for attribute in node.attribute}


def _make_index_maps(model: Any) -> tuple[dict[str, Any], dict[str, int]]:
    producer_by_output: dict[str, Any] = {}
    node_index_by_output: dict[str, int] = {}
    for index, node in enumerate(model.graph.node):
        for output in node.output:
            producer_by_output[output] = node
            node_index_by_output[output] = index
    return producer_by_output, node_index_by_output


def build_upstream_trace(model: Any, target_output: str) -> dict[str, Any]:
    """Trace target_output backwards until graph inputs, initializers, or Constant nodes."""
    producer_by_output, node_index_by_output = _make_index_maps(model)
    info_by_name = _value_info_map(model)
    graph_inputs = {value.name for value in model.graph.input}
    graph_outputs = {value.name for value in model.graph.output}
    initializers = {value.name for value in model.graph.initializer}
    visited_tensors: set[str] = set()
    visited_nodes: set[int] = set()
    boundary_tensors: dict[str, dict[str, Any]] = {}

    def add_boundary(name: str, kind: str) -> None:
        boundary_tensors[name] = {
            "name": name,
            "kind": kind,
            "is_graph_input": name in graph_inputs,
            "is_graph_output": name in graph_outputs,
            "is_initializer": name in initializers,
            "inferred": info_by_name.get(name),
        }

    def visit_tensor(name: str) -> None:
        if not name or name in visited_tensors:
            return
        visited_tensors.add(name)
        if name in initializers:
            add_boundary(name, "initializer")
            return
        node = producer_by_output.get(name)
        if node is None:
            add_boundary(name, "graph_input" if name in graph_inputs else "external_or_missing")
            return
        node_index = node_index_by_output[name]
        visited_nodes.add(node_index)
        if node.op_type == "Constant":
            return
        for input_name in node.input:
            visit_tensor(input_name)

    visit_tensor(target_output)

    nodes = []
    for node_index in sorted(visited_nodes):
        node = model.graph.node[node_index]
        nodes.append(
            {
                "index": int(node_index),
                "name": node.name,
                "op_type": node.op_type,
                "inputs": list(node.input),
                "outputs": list(node.output),
                "attributes": _node_attributes(node),
                "is_constant_node": node.op_type == "Constant",
                "outputs_inferred": {name: info_by_name.get(name) for name in node.output},
                "inputs_inferred": {name: info_by_name.get(name) for name in node.input},
                "input_kinds": {
                    name: (
                        "initializer"
                        if name in initializers
                        else "graph_input"
                        if name in graph_inputs
                        else "node_output"
                        if name in producer_by_output
                        else "external_or_missing"
                    )
                    for name in node.input
                },
            }
        )

    return {
        "target_output": target_output,
        "num_nodes": len(nodes),
        "nodes": nodes,
        "boundary_tensors": boundary_tensors,
        "visited_tensors": sorted(visited_tensors),
    }


def collect_trace_output_names(model: Any, trace_targets: list[str]) -> list[str]:
    producer_by_output, _node_index_by_output = _make_index_maps(model)
    selected: list[str] = []
    seen: set[str] = set()
    for target in trace_targets:
        trace = build_upstream_trace(model, target)
        for node in trace["nodes"]:
            if node["op_type"] not in KEY_OP_TYPES:
                continue
            for output_name in node["outputs"]:
                if output_name in producer_by_output and output_name not in seen:
                    selected.append(output_name)
                    seen.add(output_name)
    for target in [TARGET_THETA_TENSOR, *TARGET_GRID_TENSORS]:
        if target in producer_by_output and target not in seen:
            selected.append(target)
            seen.add(target)
    return selected


def augment_outputs(onnx_path: Path, output_path: Path, output_names: list[str]) -> list[str]:
    import onnx
    from onnx import TensorProto, helper, shape_inference

    inferred = shape_inference.infer_shapes(onnx.load(str(onnx_path)))
    info_by_name = {
        value.name: value
        for value in list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output)
    }
    available = {value.name for value in list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output)}
    for node in inferred.graph.node:
        available.update(node.output)
    existing = {value.name for value in inferred.graph.output}
    selected: list[str] = []
    for name in output_names:
        if name not in available or name in selected:
            continue
        selected.append(name)
        if name in existing:
            continue
        if name in info_by_name:
            inferred.graph.output.append(info_by_name[name])
        else:
            inferred.graph.output.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, None))
        existing.add(name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(inferred, str(output_path))
    return selected


def _run_ort(onnx_path: Path, feeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    import onnxruntime as ort

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    available = ort.get_available_providers()
    providers = [provider for provider in providers if provider in available] or available
    session = ort.InferenceSession(str(onnx_path), providers=providers)
    values = session.run(None, feeds)
    return {output.name: value for output, value in zip(session.get_outputs(), values)}


def _to_torch_inputs(arrays: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: torch.from_numpy(value).to(device) for name, value in arrays.items()}


def _normalize_dtype_name(dtype: Any) -> str:
    text = str(dtype).lower()
    if text.startswith("torch."):
        text = text.split(".", 1)[1]
    if text.startswith("trt."):
        text = text.split(".", 1)[1]
    if "float32" in text or text == "float":
        return "float32"
    if "float16" in text or text == "half":
        return "float16"
    if "float64" in text or text == "double":
        return "float64"
    if "int64" in text:
        return "int64"
    if "int32" in text or text == "int":
        return "int32"
    if "bool" in text:
        return "bool"
    return text


def _trt_dtype_name(dtype: Any) -> str:
    import tensorrt as trt

    if dtype == trt.float32:
        return "float32"
    if dtype == trt.float16:
        return "float16"
    if dtype == trt.int32:
        return "int32"
    if dtype == trt.int64:
        return "int64"
    if dtype == trt.bool:
        return "bool"
    return _normalize_dtype_name(dtype)


def engine_io_report(engine_path: Path, device: torch.device) -> dict[str, Any]:
    runner = TensorRTEngineRunner(engine_path, device)
    trt = runner.trt
    inputs: dict[str, dict[str, Any]] = {}
    outputs: dict[str, dict[str, Any]] = {}
    all_tensors = []
    for index in range(runner.engine.num_io_tensors):
        name = runner.engine.get_tensor_name(index)
        mode = runner.engine.get_tensor_mode(name)
        dtype = _trt_dtype_name(runner.engine.get_tensor_dtype(name))
        item = {"index": int(index), "name": name, "mode": str(mode), "dtype": dtype}
        all_tensors.append(item)
        if mode == trt.TensorIOMode.INPUT:
            inputs[name] = item
        else:
            outputs[name] = item
    return {"inputs": inputs, "outputs": outputs, "all_tensors": all_tensors}


def assess_dtype_binding(
    engine_input_dtypes: dict[str, str],
    runtime_feed_dtypes: dict[str, str],
) -> dict[str, Any]:
    inputs: dict[str, dict[str, Any]] = {}
    mismatched = []
    for name, engine_dtype in sorted(engine_input_dtypes.items()):
        feed_dtype = runtime_feed_dtypes.get(name)
        normalized_engine = _normalize_dtype_name(engine_dtype)
        normalized_feed = _normalize_dtype_name(feed_dtype)
        matches = normalized_engine == normalized_feed
        if not matches:
            mismatched.append(name)
        inputs[name] = {
            "engine_dtype": normalized_engine,
            "runtime_feed_dtype": normalized_feed,
            "matches": matches,
        }
    return {
        "inputs": inputs,
        "mismatched_inputs": mismatched,
        "dtype_binding_mismatch_found": bool(mismatched),
    }


def _cast_array_to_engine_dtype(array: np.ndarray, engine_dtype: str) -> np.ndarray:
    dtype = _normalize_dtype_name(engine_dtype)
    if dtype == "float32":
        return array.astype(np.float32, copy=False)
    if dtype == "float16":
        return array.astype(np.float16, copy=False)
    if dtype == "float64":
        return array.astype(np.float64, copy=False)
    if dtype == "int32":
        return array.astype(np.int32, copy=False)
    if dtype == "int64":
        return array.astype(np.int64, copy=False)
    if dtype == "bool":
        return array.astype(np.bool_, copy=False)
    return array


def cast_feeds_to_engine_dtypes(feeds: dict[str, np.ndarray], engine_input_dtypes: dict[str, str]) -> dict[str, np.ndarray]:
    return {
        name: _cast_array_to_engine_dtype(value, engine_input_dtypes.get(name, str(value.dtype)))
        for name, value in feeds.items()
    }


def _to_numpy_outputs(outputs: dict[str, Any]) -> dict[str, np.ndarray]:
    converted: dict[str, np.ndarray] = {}
    for name, value in outputs.items():
        if torch.is_tensor(value):
            converted[name] = value.detach().cpu().numpy()
        else:
            converted[name] = np.asarray(value)
    return converted


def array_stats(array: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(array)
    result: dict[str, Any] = {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "numel": int(arr.size),
    }
    if arr.size and np.issubdtype(arr.dtype, np.number):
        numeric = arr.astype(np.float64, copy=False)
        result.update(
            {
                "min": float(np.min(numeric)),
                "max": float(np.max(numeric)),
                "mean": float(np.mean(numeric)),
                "std": float(np.std(numeric)),
            }
        )
    else:
        result.update({"min": None, "max": None, "mean": None, "std": None})
    return result


def _top_errors(reference: np.ndarray, candidate: np.ndarray, topk: int) -> list[dict[str, Any]]:
    if reference.shape != candidate.shape or reference.size == 0:
        return []
    ref = reference.astype(np.float64, copy=False)
    cand = candidate.astype(np.float64, copy=False)
    diff = np.abs(ref - cand).reshape(-1)
    if diff.size == 0:
        return []
    count = min(int(topk), int(diff.size))
    indices = np.argpartition(-diff, np.arange(count))[:count]
    indices = indices[np.argsort(-diff[indices])]
    result = []
    for flat_index in indices:
        multi_index = np.unravel_index(int(flat_index), reference.shape)
        result.append(
            {
                "index": [int(v) for v in multi_index],
                "flat_index": int(flat_index),
                "abs_error": float(diff[flat_index]),
                "ort_value": float(ref[multi_index]),
                "trt_value": float(cand[multi_index]),
            }
        )
    return result


def compare_named_arrays(
    tensor_names: list[str],
    reference: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
    *,
    topk: int = 10,
) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    for name in tensor_names:
        item: dict[str, Any] = {"tensor_name": name}
        if name not in reference or name not in candidate:
            item.update({"missing": True, "in_ort": name in reference, "in_trt": name in candidate})
            comparisons.append(item)
            continue
        ref = np.asarray(reference[name])
        cand = np.asarray(candidate[name])
        same_shape = ref.shape == cand.shape
        item.update(
            {
                "missing": False,
                "same_shape": same_shape,
                "reference_shape": list(ref.shape),
                "candidate_shape": list(cand.shape),
                "reference_dtype": str(ref.dtype),
                "candidate_dtype": str(cand.dtype),
                "reference_stats": array_stats(ref),
                "candidate_stats": array_stats(cand),
            }
        )
        if same_shape:
            ref_float = ref.astype(np.float64, copy=False)
            cand_float = cand.astype(np.float64, copy=False)
            diff = np.abs(ref_float - cand_float)
            denom = max(float(np.mean(np.abs(ref_float))) if ref_float.size else 0.0, 1.0e-12)
            item.update(
                {
                    "max_abs_error": float(np.max(diff)) if diff.size else 0.0,
                    "mean_abs_error": float(np.mean(diff)) if diff.size else 0.0,
                    "relative_error": float(np.mean(diff) / denom) if diff.size else 0.0,
                    "top_errors": _top_errors(ref, cand, topk),
                }
            )
        else:
            item.update(
                {
                    "max_abs_error": None,
                    "mean_abs_error": None,
                    "relative_error": None,
                    "top_errors": [],
                }
            )
        comparisons.append(item)
    return comparisons


def first_bad_comparison(comparisons: list[dict[str, Any]], threshold: float) -> dict[str, Any] | None:
    for item in comparisons:
        value = item.get("max_abs_error")
        if value is not None and float(value) > threshold:
            return item
        if item.get("missing") or item.get("same_shape") is False:
            return item
    return None


def _build_engine(
    onnx_path: Path,
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
        "error": None if proc.returncode == 0 and engine_path.exists() else (proc.stderr or proc.stdout)[-4000:],
    }


def _node_by_output(model: Any) -> dict[str, dict[str, Any]]:
    info_by_name = _value_info_map(model)
    result: dict[str, dict[str, Any]] = {}
    for index, node in enumerate(model.graph.node):
        payload = {
            "index": int(index),
            "name": node.name,
            "op_type": node.op_type,
            "inputs": list(node.input),
            "outputs": list(node.output),
            "attributes": _node_attributes(node),
            "outputs_inferred": {name: info_by_name.get(name) for name in node.output},
        }
        for output in node.output:
            result[output] = payload
    return result


def _suspected_reason(first_bad: dict[str, Any] | None, first_bad_node: dict[str, Any] | None) -> str | None:
    if not first_bad or not first_bad_node:
        return None
    op_type = first_bad_node.get("op_type")
    name = first_bad.get("tensor_name")
    dtype_pair = (first_bad.get("reference_dtype"), first_bad.get("candidate_dtype"))
    if op_type in {"Gather", "Slice"}:
        return "Gather/Slice index or axis semantics differ between TensorRT full graph and ONNXRuntime."
    if op_type == "Cast" or dtype_pair[0] != dtype_pair[1]:
        return "Cast or dtype conversion differs; check double/float and int64/int32 conversion in TensorRT."
    if op_type in {"Div", "Mul", "Add"}:
        return "Arithmetic output diverges; compare constant operands and TensorRT constant folding."
    if op_type in {"Shape", "ConstantOfShape", "Reshape", "Expand", "Where"}:
        return "Dynamic shape or ConstantOfShape/Reshape/Expand handling diverges in TensorRT optimization."
    if op_type == "Concat" and name == TARGET_THETA_TENSOR:
        return "BEV warp theta construction diverges before GridSample; likely TensorRT optimization of the Gather/constant arithmetic subgraph."
    if op_type == "MatMul":
        return "Grid construction MatMul diverges; verify expanded base grid and sliced theta inputs."
    return f"First divergent tensor is produced by {op_type}; inspect its inputs in the same report."


def _engine_cast_suspected_reason(dtype_report: dict[str, Any], cast_first_bad: dict[str, Any] | None) -> str | None:
    if dtype_report.get("dtype_binding_mismatch_found") and cast_first_bad is None:
        mismatched = ", ".join(dtype_report.get("mismatched_inputs") or [])
        return (
            f"TensorRT converted ONNX input dtype(s) to engine dtype(s), but runtime feeds used original dtype(s): {mismatched}. "
            "Binding buffers with the engine dtype makes ORT and TRT align, so this is a runtime dtype binding bug, not a plugin issue."
        )
    if dtype_report.get("dtype_binding_mismatch_found"):
        mismatched = ", ".join(dtype_report.get("mismatched_inputs") or [])
        return (
            f"Runtime dtype mismatch exists for {mismatched}, but errors remain after engine-dtype casting; inspect the first remaining bad tensor."
        )
    return None


def update_summary(dirs: dict[str, Path], diff_report: dict[str, Any]) -> None:
    summary = read_json(dirs["summary"] / "summary_all.json", default={}) or {}
    summary.update(
        {
            "first_mismatch_stage": "BEV warp grid construction",
            "bev_warp_grid_first_bad_tensor": diff_report.get("first_bad_tensor"),
            "bev_warp_grid_first_bad_node": diff_report.get("first_bad_node"),
            "bev_warp_grid_first_bad_op_type": diff_report.get("first_bad_op_type"),
            "bev_warp_grid_suspected_reason": diff_report.get("suspected_reason"),
            "plugin_needed": False,
            "need_bevwarp_plugin": False,
            "need_pointpillar_scatter_plugin": False,
            "need_bevpool_plugin": False,
            "need_inverse_plugin": False,
            "recommended_plugin": None,
            "trt_dtype_binding_mismatch_found": diff_report.get("dtype_binding_mismatch_found"),
            "trt_dtype_binding_mismatched_inputs": diff_report.get("dtype_binding_mismatched_inputs"),
            "trt_engine_cast_first_bad_tensor": diff_report.get("engine_cast_first_bad_tensor"),
        }
    )
    write_summary_files(summary, dirs)


def run_diagnosis(args: argparse.Namespace) -> dict[str, Any]:
    import onnx
    from onnx import shape_inference

    dirs = ensure_quant_deploy_run_dirs(args.output_root)
    onnx_path = Path(args.onnx_path)
    inferred_model = shape_inference.infer_shapes(onnx.load(str(onnx_path)))
    node_lookup = _node_by_output(inferred_model)

    trace_targets = [TARGET_THETA_TENSOR, *TARGET_GRID_TENSORS]
    trace = build_upstream_trace(inferred_model, TARGET_THETA_TENSOR)
    trace["additional_augmented_trace_targets"] = TARGET_GRID_TENSORS
    save_json(trace, dirs["debug"] / "bev_warp_grid_construction_trace.json")

    selected_outputs = collect_trace_output_names(inferred_model, trace_targets)
    augmented_onnx = dirs["debug"] / "bev_warp_grid_construction_augmented.onnx"
    selected_outputs = augment_outputs(onnx_path, augmented_onnx, selected_outputs)

    profile_shapes = read_json(dirs["configs"] / "profile_shapes.json", default={}) or {}
    engine_path = dirs["debug"] / "bev_warp_grid_construction_fp32.engine"
    build_report = _build_engine(
        augmented_onnx,
        engine_path,
        dirs["debug"] / "bev_warp_grid_construction_layerinfo.json",
        profile_shapes,
        args.trt_root,
        args.trtexec_path,
        args.timeout,
    )

    diff_report: dict[str, Any] = {
        "onnx_path": str(onnx_path),
        "augmented_onnx_path": str(augmented_onnx),
        "engine_path": str(engine_path),
        "trace_report_path": str(dirs["debug"] / "bev_warp_grid_construction_trace.json"),
        "selected_outputs": selected_outputs,
        "build_report": build_report,
        "threshold": args.threshold,
        "comparisons": [],
    }
    try:
        hypes, device, _model, modality = _load_model_context(args)
        sample = _first_real_sample(hypes, device)
        tensors, _agent_modalities = _extract_inputs(sample, modality)
        tensors_by_name = {name: tensor for name, tensor in zip(INPUT_NAMES, tensors)}
        feeds = {name: tensor.detach().cpu().numpy() for name, tensor in tensors_by_name.items()}
        diff_report["input_hashes"] = {name: hash_array(value) for name, value in feeds.items()}
        diff_report["input_shapes"] = {name: list(value.shape) for name, value in feeds.items()}

        ort_outputs = _run_ort(augmented_onnx, feeds)
        diff_report["ort_provider_note"] = "CUDAExecutionProvider is used when available, otherwise CPUExecutionProvider."
        if build_report.get("success"):
            engine_io = engine_io_report(engine_path, device)
            engine_input_dtypes = {name: item["dtype"] for name, item in engine_io["inputs"].items()}
            runtime_feed_dtypes = {name: str(value.dtype) for name, value in feeds.items()}
            dtype_report = assess_dtype_binding(engine_input_dtypes, runtime_feed_dtypes)
            diff_report["engine_io_report"] = engine_io
            diff_report["runtime_feed_dtypes"] = runtime_feed_dtypes
            diff_report["dtype_binding_report"] = dtype_report
            diff_report["dtype_binding_mismatch_found"] = dtype_report["dtype_binding_mismatch_found"]
            diff_report["dtype_binding_mismatched_inputs"] = dtype_report["mismatched_inputs"]

            trt_outputs = _to_numpy_outputs(
                TensorRTEngineRunner(engine_path, device).run(
                    _to_torch_inputs(feeds, device),
                    cast_inputs_to_engine_dtype=False,
                )
            )
            comparisons = compare_named_arrays(selected_outputs, ort_outputs, trt_outputs, topk=args.topk)
            diff_report["comparisons"] = comparisons
            first_bad = first_bad_comparison(comparisons, args.threshold)
            first_node = node_lookup.get(first_bad["tensor_name"]) if first_bad else None

            engine_cast_feeds = cast_feeds_to_engine_dtypes(feeds, engine_input_dtypes)
            engine_cast_feed_dtypes = {name: str(value.dtype) for name, value in engine_cast_feeds.items()}
            engine_cast_hashes = {name: hash_array(value) for name, value in engine_cast_feeds.items()}
            engine_cast_trt_outputs = _to_numpy_outputs(
                TensorRTEngineRunner(engine_path, device).run(_to_torch_inputs(engine_cast_feeds, device))
            )
            engine_cast_comparisons = compare_named_arrays(selected_outputs, ort_outputs, engine_cast_trt_outputs, topk=args.topk)
            engine_cast_first_bad = first_bad_comparison(engine_cast_comparisons, args.threshold)
            engine_cast_first_node = node_lookup.get(engine_cast_first_bad["tensor_name"]) if engine_cast_first_bad else None
            diff_report["engine_cast_feed_dtypes"] = engine_cast_feed_dtypes
            diff_report["engine_cast_input_hashes"] = engine_cast_hashes
            diff_report["engine_cast_comparisons"] = engine_cast_comparisons
            diff_report["engine_cast_first_bad_tensor"] = engine_cast_first_bad.get("tensor_name") if engine_cast_first_bad else None
            diff_report["engine_cast_first_bad_node"] = engine_cast_first_node.get("name") if engine_cast_first_node else None
            diff_report["engine_cast_first_bad_op_type"] = engine_cast_first_node.get("op_type") if engine_cast_first_node else None
            diff_report["engine_cast_suspected_reason"] = _engine_cast_suspected_reason(dtype_report, engine_cast_first_bad)

            suspected_reason = diff_report["engine_cast_suspected_reason"] or _suspected_reason(first_bad, first_node)
            diff_report.update(
                {
                    "first_bad_tensor": first_bad.get("tensor_name") if first_bad else None,
                    "first_bad_node": first_node.get("name") if first_node else None,
                    "first_bad_op_type": first_node.get("op_type") if first_node else None,
                    "first_bad_node_inputs": first_node.get("inputs") if first_node else None,
                    "first_bad_node_attributes": first_node.get("attributes") if first_node else None,
                    "suspected_reason": suspected_reason,
                }
            )
        else:
            diff_report["error"] = "TensorRT debug engine build failed; ORT outputs were generated but TRT comparison was skipped."
    except Exception as exc:
        diff_report.update({"error": str(exc), "traceback": traceback.format_exc()})

    save_json(diff_report, dirs["debug"] / "bev_warp_grid_construction_diff.json")
    update_summary(dirs, diff_report)
    return diff_report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trace and diff BEV warp grid/theta construction between ONNXRuntime and TensorRT.")
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
    parser.add_argument("--topk", type=int, default=10)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    report = run_diagnosis(parse_args(argv))
    print(
        json.dumps(
            {
                "build_success": report.get("build_report", {}).get("success"),
                "first_bad_tensor": report.get("first_bad_tensor"),
                "first_bad_node": report.get("first_bad_node"),
                "first_bad_op_type": report.get("first_bad_op_type"),
                "suspected_reason": report.get("suspected_reason"),
                "diff_report": "debug/bev_warp_grid_construction_diff.json",
            },
            indent=2,
        )
    )
    return 0 if report.get("build_report", {}).get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
