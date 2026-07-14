"""Resolve FP16 merge-output contracts from the canonical ONNX graph."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from ..types import CanonicalPrecisionMappingResult, stable_json_hash


def fp16_merge_cast_name(merge_node_name: str, input_index: int) -> str:
    token = stable_json_hash({"merge": str(merge_node_name)})[:12]
    return f"__merge_fp16__{token}__input{int(input_index):02d}__Cast"


def apply_fp16_merge_output_contract(
    model_or_path: Any,
    mapping: CanonicalPrecisionMappingResult,
) -> tuple[CanonicalPrecisionMappingResult, dict[str, Any]]:
    """Force actual residual/concat producers to expose FP16 outputs.

    Precision-coupling groups own weighted compute precision, but they cannot
    identify the exact ONNX producer at a functional Add/Concat boundary.  This
    graph pass stops at the nearest canonical weighted producer on every merge
    branch.  INT8 compute is preserved while its output is dequantized to FP16;
    downstream weighted layers may quantize again at their own input boundary.
    """

    import onnx

    model = onnx.load(str(model_or_path)) if isinstance(model_or_path, (str, Path)) else model_or_path
    producers = {str(output): node for node in model.graph.node for output in node.output}
    consumers: dict[str, list[Any]] = {}
    for graph_node in model.graph.node:
        for input_name in graph_node.input:
            consumers.setdefault(str(input_name), []).append(graph_node)
    initializers = {str(row.name) for row in model.graph.initializer}
    entries = {str(row.canonical_node_name): row for row in mapping.entries}

    def nearest_upstream(tensor_name: str, seen: set[str] | None = None) -> set[str]:
        visited = set(seen or ())
        if tensor_name in visited:
            return set()
        visited.add(tensor_name)
        producer = producers.get(str(tensor_name))
        if producer is None:
            return set()
        canonical = str(producer.name)
        if canonical in entries:
            return {canonical}
        result: set[str] = set()
        for input_name in producer.input:
            if str(input_name) not in initializers:
                result.update(nearest_upstream(str(input_name), visited))
        return result

    targets: set[str] = set()
    merges: list[dict[str, Any]] = []
    for node in model.graph.node:
        if str(node.op_type) not in {"Add", "Concat"}:
            continue
        branches = [sorted(nearest_upstream(str(input_name))) for input_name in node.input]
        weighted_branch_count = sum(bool(branch) for branch in branches)
        if weighted_branch_count < 2:
            continue
        branch_targets = sorted({name for branch in branches for name in branch})
        targets.update(branch_targets)
        post_merge_activation_nodes = []
        if len(node.output) == 1:
            direct_consumers = consumers.get(str(node.output[0]), [])
            if len(direct_consumers) == 1 and str(direct_consumers[0].op_type) == "Relu":
                post_merge_activation_nodes.append(str(direct_consumers[0].name))
        merges.append(
            {
                "merge_op_name": str(node.name),
                "merge_op_type": str(node.op_type),
                "input_tensors": [str(value) for value in node.input],
                "nearest_weighted_producers": [
                    [
                        {
                            "canonical_node_name": name,
                            "module_path": entries[name].module_path,
                            "precision_group": entries[name].precision_group,
                            "compute_precision": entries[name].realized_request_precision,
                        }
                        for name in branch
                    ]
                    for branch in branches
                ],
                "weighted_input_branch_count": weighted_branch_count,
                "output_contract": "FP16_merge_then_optional_downstream_requantization",
                "input_cast_nodes": [
                    fp16_merge_cast_name(str(node.name), index)
                    for index, _ in enumerate(node.input)
                ],
                "post_merge_activation_nodes": post_merge_activation_nodes,
            }
        )
    resolved_entries = [
        replace(row, realized_output_precision="fp16")
        if row.canonical_node_name in targets
        else row
        for row in mapping.entries
    ]
    resolved = CanonicalPrecisionMappingResult(
        entries=resolved_entries,
        profile_id=mapping.profile_id,
        profile_hash=mapping.profile_hash,
        origin_map_hash=mapping.origin_map_hash,
        policy_version="canonical-precision-mapping-fp16-merge-output-v1",
        auxiliary_layer_precisions={
            **dict(mapping.auxiliary_layer_precisions),
            **{
                str(row["merge_op_name"]): "fp16"
                for row in merges
            },
            **{
                cast_name: "fp16"
                for row in merges
                for cast_name in row["input_cast_nodes"]
            },
            **{
                activation_name: "fp16"
                for row in merges
                for activation_name in row["post_merge_activation_nodes"]
            },
        },
        auxiliary_layer_output_types={
            **dict(mapping.auxiliary_layer_output_types),
            **{
                str(row["merge_op_name"]): "fp16"
                for row in merges
            },
            **{
                cast_name: "fp16"
                for row in merges
                for cast_name in row["input_cast_nodes"]
            },
            **{
                activation_name: "fp16"
                for row in merges
                for activation_name in row["post_merge_activation_nodes"]
            },
        },
    )
    report = {
        "policy": "A_fp16_merge",
        "status": "resolved",
        "resolved_merge_count": len(merges),
        "fp16_output_canonical_count": len(targets),
        "fp16_output_canonical_nodes": sorted(targets),
        "fp16_output_modules": sorted(entries[name].module_path for name in targets),
        "merges": merges,
        "mapping_hash_before": mapping.mapping_hash,
        "mapping_hash_after": resolved.mapping_hash,
    }
    report["contract_hash"] = stable_json_hash(report)
    return resolved, report
