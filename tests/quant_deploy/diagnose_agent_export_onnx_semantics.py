from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from diagnose_record_len_onnx_usage import tensor_type_info, trace_downstream
from quant_deploy_utils import ensure_quant_deploy_run_dirs, read_json, save_json, write_summary_files


SEMANTIC_OPS = {
    "Gather",
    "Slice",
    "ScatterND",
    "ScatterElements",
    "Where",
    "Mul",
    "Div",
    "Add",
    "Softmax",
    "GridSample",
    "Reshape",
    "Concat",
    "Unsqueeze",
}


def _trace_downstream_from_any(model: Any, source_tensors: list[str]) -> list[dict[str, Any]]:
    consumers: dict[str, list[Any]] = defaultdict(list)
    for node in model.graph.node:
        for input_name in node.input:
            consumers[input_name].append(node)
    queue: deque[str] = deque(source_tensors)
    seen_tensors = set(source_tensors)
    seen_nodes: set[int] = set()
    traced: list[dict[str, Any]] = []
    while queue:
        tensor = queue.popleft()
        for node in consumers.get(tensor, []):
            key = id(node)
            if key in seen_nodes:
                continue
            seen_nodes.add(key)
            item = {
                "name": node.name or "",
                "op_type": node.op_type,
                "inputs": list(node.input),
                "outputs": list(node.output),
                "attributes": {attr.name: str(attr) for attr in node.attribute},
                "depth_order": len(traced),
            }
            traced.append(item)
            for output in node.output:
                if output not in seen_tensors:
                    seen_tensors.add(output)
                    queue.append(output)
    return traced


def _graph_inputs(model: Any) -> list[dict[str, Any]]:
    return [tensor_type_info(value_info) for value_info in model.graph.input]


def _contains_keepalive_only(nodes: list[dict[str, Any]]) -> bool:
    if not nodes:
        return False
    op_types = {node.get("op_type") for node in nodes}
    return op_types.issubset({"Cast", "ReduceSum", "Mul", "Add", "Identity"}) and "Mul" in op_types and "Add" in op_types


def analyze_agent_export_semantics(onnx_path: str | Path, output_root: str | Path, mode: str) -> dict[str, Any]:
    import onnx

    dirs = ensure_quant_deploy_run_dirs(output_root)
    model = onnx.load(str(onnx_path))
    inputs = _graph_inputs(model)
    input_names = {item["name"] for item in inputs}
    node_text = "\n".join(
        " ".join([node.name or "", node.op_type, " ".join(node.input), " ".join(node.output)])
        for node in model.graph.node
    ).lower()
    report: dict[str, Any] = {
        "onnx_path": str(onnx_path),
        "pyramid_forward_export_mode": mode,
        "graph_inputs": inputs,
        "semantics_erased_in_onnx": False,
        "valid_for_multi_agent_deployment": mode in {"dynamic_agent_dim", "padded_agent_static"},
        "agent_dim_enters_fusion": False,
        "pairwise_dynamic_n_enters_warp": False,
        "valid_agent_mask_enters_fusion": False,
        "record_len_keepalive_only": None,
        "failure_reason": None,
    }

    if "record_len" in input_names:
        downstream = trace_downstream(model, "record_len")
        report["record_len_downstream_consumers"] = downstream
        report["record_len_keepalive_only"] = _contains_keepalive_only(downstream)
    else:
        report["record_len_downstream_consumers"] = []
        report["record_len_keepalive_only"] = False

    if mode == "dynamic_agent_dim":
        pairwise_nodes = _trace_downstream_from_any(model, ["pairwise_t_matrix"])
        report["pairwise_t_matrix_downstream_consumers"] = pairwise_nodes
        op_types = {node["op_type"] for node in pairwise_nodes}
        report["pairwise_dynamic_n_enters_warp"] = "GridSample" in op_types or "gridsample" in node_text or any(op in op_types for op in {"Slice", "Gather", "MatMul", "Concat"})
        report["agent_dim_enters_fusion"] = report["pairwise_dynamic_n_enters_warp"] and "Softmax" in {node.op_type for node in model.graph.node}
        report["semantics_erased_in_onnx"] = not report["pairwise_dynamic_n_enters_warp"]
        if report["semantics_erased_in_onnx"]:
            report["failure_reason"] = "pairwise_t_matrix dynamic agent dimension does not reach BEV warp/fusion subgraph"

    if mode == "padded_agent_static":
        mask_nodes = _trace_downstream_from_any(model, ["valid_agent_mask"]) if "valid_agent_mask" in input_names else []
        report["valid_agent_mask_downstream_consumers"] = mask_nodes
        op_types = {node["op_type"] for node in mask_nodes}
        report["valid_agent_mask_is_graph_input"] = "valid_agent_mask" in input_names
        report["valid_agent_mask_enters_fusion"] = bool(mask_nodes) and bool(op_types & SEMANTIC_OPS) and ("Softmax" in {node.op_type for node in model.graph.node} or "gridsample" in node_text)
        report["semantics_erased_in_onnx"] = not report["valid_agent_mask_enters_fusion"]
        if report["semantics_erased_in_onnx"]:
            report["failure_reason"] = "valid_agent_mask does not reach feature zeroing, confidence mask, or weighted fusion"

    if report["record_len_keepalive_only"]:
        report["semantics_erased_in_onnx"] = True
        report["failure_reason"] = "record_len only flows through zero keepalive path"
    report["valid_for_multi_agent_deployment"] = bool(report["valid_for_multi_agent_deployment"] and not report["semantics_erased_in_onnx"])

    if mode == "dynamic_agent_dim":
        save_json(report, dirs["debug"] / "dynamic_agent_dim_onnx_semantics_report.json")
        save_json(report, dirs["debug"] / "dynamic_agent_dim_record_len_usage_report.json")
    elif mode == "padded_agent_static":
        save_json(report, dirs["debug"] / "padded_agent_static_onnx_semantics_report.json")
        save_json(report, dirs["debug"] / "padded_agent_static_record_len_usage_report.json")
        save_json(report, dirs["debug"] / "valid_agent_mask_onnx_usage_report.json")

    summary = read_json(dirs["summary"] / "summary_all.json", default={}) or {}
    summary[f"{mode}_onnx_semantics"] = {
        "semantics_erased_in_onnx": report["semantics_erased_in_onnx"],
        "valid_for_multi_agent_deployment": report["valid_for_multi_agent_deployment"],
        "failure_reason": report["failure_reason"],
    }
    if summary:
        write_summary_files(summary, dirs)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check lidar_pyramid A/B agent export ONNX semantics.")
    parser.add_argument("--onnx_path", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--mode", required=True, choices=["dynamic_agent_dim", "padded_agent_static"])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = analyze_agent_export_semantics(args.onnx_path, args.output_root, args.mode)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report.get("valid_for_multi_agent_deployment") else 2


if __name__ == "__main__":
    raise SystemExit(main())
