from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from quant_deploy_utils import ensure_quant_deploy_run_dirs, read_json, save_json, write_summary_files


SEMANTIC_OPS = {
    "Split",
    "Slice",
    "Gather",
    "GatherElements",
    "GatherND",
    "Reshape",
    "Where",
    "NonZero",
    "Range",
    "TopK",
    "Scatter",
    "ScatterElements",
    "ScatterND",
    "GridSample",
}
KEEPALIVE_OPS = {"Cast", "ReduceSum", "Mul", "Add", "Identity"}


def _attribute_value(attr: Any) -> Any:
    import onnx
    from onnx import numpy_helper

    if attr.type == onnx.AttributeProto.INT:
        return int(attr.i)
    if attr.type == onnx.AttributeProto.FLOAT:
        return float(attr.f)
    if attr.type == onnx.AttributeProto.STRING:
        return attr.s.decode("utf-8", errors="replace")
    if attr.type == onnx.AttributeProto.INTS:
        return [int(v) for v in attr.ints]
    if attr.type == onnx.AttributeProto.FLOATS:
        return [float(v) for v in attr.floats]
    if attr.type == onnx.AttributeProto.TENSOR:
        arr = numpy_helper.to_array(attr.t)
        return {"dtype": str(arr.dtype), "shape": list(arr.shape), "values": arr.reshape(-1)[:16].tolist()}
    return str(attr)


def node_to_dict(node: Any, *, zero_constant_input: bool = False) -> dict[str, Any]:
    return {
        "name": node.name or "",
        "op_type": node.op_type,
        "inputs": list(node.input),
        "outputs": list(node.output),
        "attributes": {attr.name: _attribute_value(attr) for attr in getattr(node, "attribute", [])},
        "zero_constant_input": bool(zero_constant_input),
    }


def tensor_type_info(value_info: Any) -> dict[str, Any]:
    from onnx import TensorProto

    tensor_type = value_info.type.tensor_type
    shape: list[Any] = []
    dynamic_axes: dict[str, str] = {}
    for idx, dim in enumerate(tensor_type.shape.dim):
        if getattr(dim, "dim_param", ""):
            shape.append(dim.dim_param)
            dynamic_axes[str(idx)] = dim.dim_param
        else:
            shape.append(int(getattr(dim, "dim_value", 0)))
    return {
        "name": value_info.name,
        "dtype": TensorProto.DataType.Name(tensor_type.elem_type),
        "shape": shape,
        "dynamic_axes": dynamic_axes,
    }


def _initializer_arrays(model: Any) -> dict[str, np.ndarray]:
    from onnx import numpy_helper

    return {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}


def _constant_node_arrays(model: Any) -> dict[str, np.ndarray]:
    from onnx import numpy_helper

    constants: dict[str, np.ndarray] = {}
    for node in model.graph.node:
        if node.op_type != "Constant" or not node.output:
            continue
        for attr in node.attribute:
            if attr.name == "value":
                constants[node.output[0]] = numpy_helper.to_array(attr.t)
    return constants


def _is_zero_tensor(name: str, constants: dict[str, np.ndarray]) -> bool:
    arr = constants.get(name)
    if arr is None:
        return False
    return bool(arr.size == 1 and float(arr.reshape(-1)[0]) == 0.0)


def trace_downstream(model: Any, source_tensor: str) -> list[dict[str, Any]]:
    consumers: dict[str, list[Any]] = defaultdict(list)
    for node in model.graph.node:
        for input_name in node.input:
            consumers[input_name].append(node)
    constants = {}
    constants.update(_initializer_arrays(model))
    constants.update(_constant_node_arrays(model))

    queue: deque[str] = deque([source_tensor])
    seen_tensors = {source_tensor}
    seen_nodes: set[int] = set()
    traced: list[dict[str, Any]] = []
    while queue:
        tensor = queue.popleft()
        for node in consumers.get(tensor, []):
            node_key = id(node)
            if node_key in seen_nodes:
                continue
            seen_nodes.add(node_key)
            zero_constant_input = any(_is_zero_tensor(input_name, constants) for input_name in node.input if input_name != tensor)
            item = node_to_dict(node, zero_constant_input=zero_constant_input)
            item["depth_order"] = len(traced)
            traced.append(item)
            for output_name in node.output:
                if output_name not in seen_tensors:
                    seen_tensors.add(output_name)
                    queue.append(output_name)
    return traced


def _node_name_text(node: dict[str, Any]) -> str:
    text = " ".join([node.get("name", ""), node.get("op_type", ""), " ".join(node.get("inputs", [])), " ".join(node.get("outputs", []))])
    return text.lower()


def classify_record_len_usage(nodes: list[dict[str, Any]], graph_outputs: set[str]) -> dict[str, Any]:
    op_types = [node.get("op_type") for node in nodes]
    semantic_nodes = [
        node
        for node in nodes
        if node.get("op_type") in SEMANTIC_OPS and not (node.get("op_type") == "Gather" and "shape" in _node_name_text(node))
    ]
    output_add_nodes = [
        node
        for node in nodes
        if node.get("op_type") == "Add" and bool(set(node.get("outputs", [])) & set(graph_outputs))
    ]
    zero_mul_nodes = [node for node in nodes if node.get("op_type") == "Mul" and node.get("zero_constant_input")]
    op_set = set(op_types)
    text = "\n".join(_node_name_text(node) for node in nodes)
    keepalive_only = bool(nodes) and bool(zero_mul_nodes) and bool(output_add_nodes) and not semantic_nodes and set(op_types).issubset(KEEPALIVE_OPS)
    return {
        "record_len_semantics_erased_in_onnx": bool(keepalive_only or (bool(nodes) and not semantic_nodes and not any(op in op_set for op in {"Split", "Slice", "Gather", "Reshape", "GridSample"}))),
        "keepalive_only": bool(keepalive_only),
        "semantic_consumer_count": len(semantic_nodes),
        "participates_in_split": "Split" in op_set,
        "participates_in_slice": "Slice" in op_set,
        "participates_in_gather": any(op in op_set for op in {"Gather", "GatherElements", "GatherND"}),
        "participates_in_reshape": "Reshape" in op_set,
        "participates_in_valid_agent_mask": "valid_agent_mask" in text or "mask" in text and any(op in op_set for op in {"Where", "NonZero", "Greater", "Less", "Equal"}),
        "participates_in_multi_agent_regroup": any(token in text for token in ["regroup", "split", "record_len", "agent"]) and bool(semantic_nodes),
        "participates_in_pyramid_fusion": "pyramid" in text or "fusion" in text,
        "participates_in_bev_warp": "gridsample" in "".join(op_types).lower() or "warp" in text or "grid" in text,
        "participates_in_weighted_fusion": "softmax" in "".join(op_types).lower() or "weighted" in text or "score" in text,
        "downstream_op_types": op_types,
    }


def count_record_len_distribution(frames: list[dict[str, Any]]) -> dict[str, int]:
    counter = Counter()
    for frame in frames:
        value = frame.get("record_len")
        if value is None:
            continue
        counter[str(int(value))] += 1
    return dict(sorted(counter.items(), key=lambda item: int(item[0])))


def _read_evaluation_record_len_distribution(dirs: dict[str, Path]) -> dict[str, int]:
    per_frame = read_json(dirs["debug"] / "per_frame_ap_and_error_report.json", default={}) or {}
    frames = per_frame.get("frames") or []
    if frames:
        return count_record_len_distribution(frames)
    wrapper = read_json(dirs["evaluation"] / "wrapper_equivalence.json", default={}) or {}
    by_record_len = wrapper.get("by_record_len") or {}
    return {str(key): int(value.get("num_frames", 0)) for key, value in sorted(by_record_len.items(), key=lambda item: int(item[0]))}


def _export_dummy_info(dirs: dict[str, Path]) -> dict[str, Any]:
    tensor_shapes = read_json(dirs["debug"] / "tensor_shapes_report.json", default={}) or {}
    input_shapes = tensor_shapes.get("input_shapes") or {}
    io_names = read_json(dirs["onnx_fp32"] / "input_output_names.json", default={}) or {}
    profile_shapes = read_json(dirs["configs"] / "profile_shapes.json", default={}) or {}
    pair_shape = input_shapes.get("pairwise_t_matrix") or (profile_shapes.get("pairwise_t_matrix") or {}).get("opt")
    record_shape = input_shapes.get("record_len") or (profile_shapes.get("record_len") or {}).get("opt")
    agent_len = io_names.get("agent_modality_list_length")
    pair_agents = int(pair_shape[1]) if isinstance(pair_shape, list) and len(pair_shape) > 1 else None
    return {
        "export_dummy_record_len": None,
        "export_dummy_record_len_shape": record_shape,
        "export_dummy_num_agents": agent_len if agent_len is not None else pair_agents,
        "export_dummy_pairwise_max_cav": pair_agents,
        "agent_modality_list_length": agent_len,
        "input_shapes": input_shapes,
        "profile_shapes": profile_shapes,
    }


def analyze_record_len_onnx_usage(onnx_path: str | Path, output_root: str | Path) -> dict[str, Any]:
    import onnx

    dirs = ensure_quant_deploy_run_dirs(output_root)
    model = onnx.load(str(onnx_path))
    graph_outputs = {output.name for output in model.graph.output}
    graph_inputs = [tensor_type_info(value_info) for value_info in model.graph.input]
    record_input = next((item for item in graph_inputs if item["name"] == "record_len"), None)
    downstream = trace_downstream(model, "record_len")
    usage = classify_record_len_usage(downstream, graph_outputs)
    export_info = _export_dummy_info(dirs)
    evaluation_dist = _read_evaluation_record_len_distribution(dirs)
    suspected_static = export_info.get("export_dummy_num_agents") or export_info.get("export_dummy_pairwise_max_cav")
    if suspected_static is None:
        suspected_static = max((int(key) for key in evaluation_dist), default=None)
    report = {
        "onnx_path": str(onnx_path),
        "graph_inputs": graph_inputs,
        "record_len_input": record_input,
        "record_len_is_graph_input": record_input is not None,
        "record_len_downstream_consumer_count": len(downstream),
        "record_len_downstream_consumers": downstream,
        **usage,
        **export_info,
        "evaluation_record_len_distribution": evaluation_dist,
        "suspected_static_agent_count": suspected_static,
        "graph_outputs": sorted(graph_outputs),
        "notes": [
            "record_len is considered semantically erased if it only flows through a zero keepalive path into graph outputs.",
            "export_dummy_record_len value was not saved by the original export; shape and agent count are recovered from tensor_shapes/input_output/profile metadata.",
        ],
    }
    save_json(report, dirs["debug"] / "record_len_onnx_usage_report.json")
    summary = read_json(dirs["summary"] / "summary_all.json", default={}) or {}
    summary.update(
        {
            "record_len_is_graph_input": report["record_len_is_graph_input"],
            "record_len_downstream_consumer_count": report["record_len_downstream_consumer_count"],
            "record_len_semantics_erased_in_onnx": report["record_len_semantics_erased_in_onnx"],
            "export_dummy_record_len": report["export_dummy_record_len"],
            "evaluation_record_len_distribution": report["evaluation_record_len_distribution"],
            "suspected_static_agent_count": report["suspected_static_agent_count"],
        }
    )
    write_summary_files(summary, dirs)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trace record_len usage in exported lidar_pyramid ONNX.")
    parser.add_argument("--onnx_path", required=True)
    parser.add_argument("--output_root", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = analyze_record_len_onnx_usage(args.onnx_path, args.output_root)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
