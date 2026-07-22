"""Resolve fixed and adaptive merge-output contracts on canonical ONNX graphs."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

from ..types import CanonicalPrecisionMappingResult, stable_json_hash


def fp16_merge_cast_name(merge_node_name: str, input_index: int) -> str:
    token = stable_json_hash({"merge": str(merge_node_name)})[:12]
    return f"__merge_fp16__{token}__input{int(input_index):02d}__Cast"


def adaptive_merge_cast_name(merge_node_name: str, input_index: int, precision: str) -> str:
    token = stable_json_hash({"merge": str(merge_node_name)})[:12]
    return (
        f"__merge_adaptive__{token}__input{int(input_index):02d}__"
        f"{str(precision).lower()}__Cast"
    )


_PRECISION_RANK = {"int8": 0, "fp16": 1, "fp32": 2}


def _promoted_precision(values: Sequence[str]) -> str:
    normalized = [str(value).lower() for value in values]
    unknown = sorted(set(normalized) - set(_PRECISION_RANK))
    if unknown or not normalized:
        raise RuntimeError(f"adaptive_merge_precision_invalid:{unknown or normalized}")
    return max(normalized, key=lambda value: _PRECISION_RANK[value])


def apply_adaptive_merge_output_contract(
    model_or_path: Any,
    mapping: CanonicalPrecisionMappingResult,
    *,
    runtime_relations: Sequence[Any] = (),
) -> tuple[CanonicalPrecisionMappingResult, dict[str, Any]]:
    """Derive each activation merge precision from its realized branch outputs.

    Equal branch precisions remain unchanged. Mixed branches are promoted with
    ``INT8 < FP16 < FP32``. Runtime relations remain the source of search-space
    coupling; this canonical graph pass resolves their concrete exported ONNX
    node boundaries and records any unmatched relations for fail-closed QA.
    """

    import onnx

    model = onnx.load(str(model_or_path)) if isinstance(model_or_path, (str, Path)) else model_or_path
    producers = {str(output): node for node in model.graph.node for output in node.output}
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

    supported_merges = {"Add", "Concat", "Mul", "Where", "MatMul"}
    merges: list[dict[str, Any]] = []
    for node in model.graph.node:
        if str(node.op_type) not in supported_merges:
            continue
        branches = [sorted(nearest_upstream(str(input_name))) for input_name in node.input]
        weighted_branches = [branch for branch in branches if branch]
        if len(weighted_branches) < 2:
            continue
        branch_precisions = [
            _promoted_precision([
                str(entries[name].realized_request_precision)
                for name in branch
            ])
            for branch in weighted_branches
        ]
        merge_precision = _promoted_precision(branch_precisions)
        canonical_targets = sorted({name for branch in weighted_branches for name in branch})
        member_modules = sorted({entries[name].module_path for name in canonical_targets})
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
                "member_modules": member_modules,
                "weighted_input_branch_count": len(weighted_branches),
                "branch_precisions": branch_precisions,
                "all_branch_precisions_equal": len(set(branch_precisions)) == 1,
                "derived_merge_precision": merge_precision,
                "promotion_order": ["int8", "fp16", "fp32"],
                "input_cast_nodes": (
                    [
                        adaptive_merge_cast_name(str(node.name), index, merge_precision)
                        for index, _value in enumerate(node.input)
                    ]
                    if merge_precision in {"fp16", "fp32"}
                    else []
                ),
            }
        )

    # Keep each weighted producer's own output contract equal to its compute
    # precision. Promotion is edge-local at the concrete merge. This matters
    # for fan-out: one INT8 producer may feed both an all-INT8 merge and a
    # mixed FP16 merge without globally forcing every consumer to FP16.
    resolved_entries = [
        replace(row, realized_output_precision=str(row.realized_request_precision).lower())
        for row in mapping.entries
    ]
    auxiliary_precisions = {
        **dict(mapping.auxiliary_layer_precisions),
        **{str(row["merge_op_name"]): str(row["derived_merge_precision"]) for row in merges},
    }
    auxiliary_outputs = {
        **dict(mapping.auxiliary_layer_output_types),
        **{str(row["merge_op_name"]): str(row["derived_merge_precision"]) for row in merges},
    }
    resolved = CanonicalPrecisionMappingResult(
        entries=resolved_entries,
        profile_id=mapping.profile_id,
        profile_hash=mapping.profile_hash,
        origin_map_hash=mapping.origin_map_hash,
        policy_version="canonical-precision-mapping-adaptive-runtime-merge-v1",
        auxiliary_layer_precisions=auxiliary_precisions,
        auxiliary_layer_output_types=auxiliary_outputs,
    )

    runtime_rows = [
        relation.to_dict() if hasattr(relation, "to_dict") else dict(relation)
        for relation in runtime_relations
    ]
    canonical_member_sets = [set(row["member_modules"]) for row in merges]
    runtime_relation_matches = []
    for relation in runtime_rows:
        members = set(str(value) for value in relation.get("member_modules", []))
        matches = [
            str(row["merge_op_name"])
            for row, canonical_members in zip(merges, canonical_member_sets)
            if members and (members == canonical_members or members.issubset(canonical_members))
        ]
        runtime_relation_matches.append(
            {
                "relation_id": str(relation.get("relation_id", "")),
                "relation_kind": str(relation.get("relation_kind", "")),
                "member_modules": sorted(members),
                "canonical_merge_nodes": matches,
                "resolved": bool(matches),
            }
        )
    report = {
        "policy": "adaptive_upcast_merge",
        "status": "resolved",
        "promotion_order": ["int8", "fp16", "fp32"],
        "resolved_merge_count": len(merges),
        "merges": merges,
        "runtime_relation_count": len(runtime_rows),
        "runtime_relation_matches": runtime_relation_matches,
        "unresolved_runtime_relation_ids": sorted(
            row["relation_id"] for row in runtime_relation_matches if not row["resolved"]
        ),
        "mapping_hash_before": mapping.mapping_hash,
        "mapping_hash_after": resolved.mapping_hash,
    }
    report["contract_hash"] = stable_json_hash(report)
    return resolved, report


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
