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

_RUNTIME_RELATION_ONNX_OPS = {
    "concat": {"Concat"},
    "residual_add": {"Add"},
    "elementwise_multiply": {"Mul"},
    "conditional_select": {"Where"},
    "matrix_merge": {"MatMul"},
    "stack": {"Concat"},
}


def _promoted_precision(values: Sequence[str]) -> str:
    normalized = [str(value).lower() for value in values]
    unknown = sorted(set(normalized) - set(_PRECISION_RANK))
    if unknown or not normalized:
        raise RuntimeError(f"adaptive_merge_precision_invalid:{unknown or normalized}")
    return max(normalized, key=lambda value: _PRECISION_RANK[value])


def inferred_tensor_element_types(model: Any) -> dict[str, int]:
    """Return ONNX element types without mutating the caller's model."""

    import onnx

    try:
        typed_model = onnx.shape_inference.infer_shapes(
            model,
            check_type=False,
            strict_mode=False,
            data_prop=False,
        )
    except Exception:
        typed_model = model
    result: dict[str, int] = {
        str(initializer.name): int(initializer.data_type)
        for initializer in typed_model.graph.initializer
    }
    for value in (
        list(typed_model.graph.input)
        + list(typed_model.graph.output)
        + list(typed_model.graph.value_info)
    ):
        tensor_type = value.type.tensor_type
        if int(tensor_type.elem_type) > 0:
            result[str(value.name)] = int(tensor_type.elem_type)
    return result


def adaptive_merge_dtype_audit(
    model: Any,
    node: Any,
    *,
    tensor_element_types: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Prove that a candidate merge carries floating activations, not shapes."""

    from onnx import TensorProto

    types = tensor_element_types or inferred_tensor_element_types(model)
    floating_types = {
        int(TensorProto.FLOAT),
        int(TensorProto.FLOAT16),
        int(TensorProto.DOUBLE),
        int(TensorProto.BFLOAT16),
    }
    data_input_indices = (
        list(range(1, len(node.input)))
        if str(node.op_type) == "Where"
        else list(range(len(node.input)))
    )
    data_input_types = [types.get(str(node.input[index])) for index in data_input_indices]
    output_types = [types.get(str(value)) for value in node.output]

    def dtype_name(value: int | None) -> str:
        if value is None:
            return "UNKNOWN"
        try:
            return str(TensorProto.DataType.Name(int(value)))
        except ValueError:
            return f"TYPE_{int(value)}"

    reason = ""
    if any(value is None for value in data_input_types):
        reason = "unknown_data_input_dtype"
    elif any(int(value) not in floating_types for value in data_input_types if value is not None):
        reason = "non_floating_data_input"
    elif not output_types or any(value is None for value in output_types):
        reason = "unknown_output_dtype"
    elif any(int(value) not in floating_types for value in output_types if value is not None):
        reason = "non_floating_output"
    return {
        "safe_floating_activation_merge": not reason,
        "exclusion_reason": reason,
        "data_input_indices": data_input_indices,
        "data_input_dtypes": [dtype_name(value) for value in data_input_types],
        "output_dtypes": [dtype_name(value) for value in output_types],
    }


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
    tensor_element_types = inferred_tensor_element_types(model)
    runtime_rows = [
        relation.to_dict() if hasattr(relation, "to_dict") else dict(relation)
        for relation in runtime_relations
    ]

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
    merge_candidates: list[dict[str, Any]] = []
    excluded_merges: list[dict[str, Any]] = []
    for node in model.graph.node:
        if str(node.op_type) not in supported_merges:
            continue
        branches = [sorted(nearest_upstream(str(input_name))) for input_name in node.input]
        weighted_branches = [branch for branch in branches if branch]
        if len(weighted_branches) < 2:
            continue
        canonical_targets = sorted({name for branch in weighted_branches for name in branch})
        member_modules = sorted({entries[name].module_path for name in canonical_targets})
        dtype_audit = adaptive_merge_dtype_audit(
            model,
            node,
            tensor_element_types=tensor_element_types,
        )
        candidate_identity = {
            "merge_op_name": str(node.name),
            "merge_op_type": str(node.op_type),
            "input_tensors": [str(value) for value in node.input],
            "dtype_audit": dtype_audit,
            "member_modules": member_modules,
        }
        if not dtype_audit["safe_floating_activation_merge"]:
            excluded_merges.append(candidate_identity)
            continue
        branch_precisions = [
            _promoted_precision([
                str(entries[name].realized_request_precision)
                for name in branch
            ])
            for branch in weighted_branches
        ]
        merge_precision = _promoted_precision(branch_precisions)
        merge_candidates.append(
            {
                **candidate_identity,
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
                        for index in dtype_audit["data_input_indices"]
                    ]
                    if merge_precision in {"fp16", "fp32"}
                    else []
                ),
            }
        )

    runtime_relation_matches = []
    merge_relation_ids: dict[str, list[str]] = {
        str(row["merge_op_name"]): [] for row in merge_candidates
    }
    for relation in runtime_rows:
        relation_id = str(relation.get("relation_id", ""))
        relation_kind = str(relation.get("relation_kind", "")).lower()
        members = set(str(value) for value in relation.get("member_modules", []))
        compatible_ops = _RUNTIME_RELATION_ONNX_OPS.get(relation_kind, set())
        matches = []
        for row in merge_candidates:
            canonical_members = set(row["member_modules"])
            if (
                str(row["merge_op_type"]) in compatible_ops
                and members
                and (members == canonical_members or members.issubset(canonical_members))
            ):
                merge_name = str(row["merge_op_name"])
                matches.append(merge_name)
                merge_relation_ids[merge_name].append(relation_id)
        non_activation_matches = []
        if not matches:
            for row in excluded_merges:
                dtype_audit = dict(row.get("dtype_audit", {}) or {})
                exclusion_reason = str(dtype_audit.get("exclusion_reason", ""))
                canonical_members = set(row.get("member_modules", []))
                if (
                    str(row.get("merge_op_type", "")) in compatible_ops
                    and members
                    and (members == canonical_members or members.issubset(canonical_members))
                    and exclusion_reason
                    in {"non_floating_data_input", "non_floating_output"}
                ):
                    non_activation_matches.append(str(row["merge_op_name"]))
        resolved_as_non_activation = bool(non_activation_matches) and not matches
        runtime_relation_matches.append(
            {
                "relation_id": relation_id,
                "relation_kind": relation_kind,
                "compatible_onnx_ops": sorted(compatible_ops),
                "member_modules": sorted(members),
                "canonical_merge_nodes": matches,
                "non_activation_shape_nodes": sorted(non_activation_matches),
                "resolution": (
                    "floating_activation_merge"
                    if matches
                    else "excluded_proven_non_floating_shape_merge"
                    if resolved_as_non_activation
                    else "unresolved"
                ),
                "resolved": bool(matches or resolved_as_non_activation),
            }
        )

    if runtime_rows:
        merges = []
        for row in merge_candidates:
            relation_ids = sorted(set(merge_relation_ids[str(row["merge_op_name"])]))
            if not relation_ids:
                excluded_merges.append(
                    {
                        "merge_op_name": str(row["merge_op_name"]),
                        "merge_op_type": str(row["merge_op_type"]),
                        "input_tensors": list(row["input_tensors"]),
                        "dtype_audit": dict(row["dtype_audit"]),
                        "exclusion_reason": "no_compatible_runtime_relation",
                    }
                )
                continue
            row["runtime_relation_ids"] = relation_ids
            merges.append(row)
    else:
        merges = merge_candidates

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
        policy_version="canonical-precision-mapping-adaptive-runtime-merge-v2",
        auxiliary_layer_precisions=auxiliary_precisions,
        auxiliary_layer_output_types=auxiliary_outputs,
    )

    unresolved_relation_ids = sorted(
        row["relation_id"] for row in runtime_relation_matches if not row["resolved"]
    )
    non_activation_relation_ids = sorted(
        row["relation_id"]
        for row in runtime_relation_matches
        if row["resolution"] == "excluded_proven_non_floating_shape_merge"
    )
    report = {
        "policy": "adaptive_upcast_merge",
        "status": "resolved",
        "promotion_order": ["int8", "fp16", "fp32"],
        "resolved_merge_count": len(merges),
        "merges": merges,
        "excluded_merge_count": len(excluded_merges),
        "excluded_merges": excluded_merges,
        "runtime_relation_count": len(runtime_rows),
        "runtime_relation_matches": runtime_relation_matches,
        "unresolved_runtime_relation_ids": unresolved_relation_ids,
        "non_activation_runtime_relation_ids": non_activation_relation_ids,
        "mapping_hash_before": mapping.mapping_hash,
        "mapping_hash_after": resolved.mapping_hash,
    }
    report["contract_hash"] = stable_json_hash(report)
    if unresolved_relation_ids:
        raise RuntimeError(
            "adaptive_merge_runtime_relations_unresolved:"
            + ",".join(unresolved_relation_ids)
        )
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
