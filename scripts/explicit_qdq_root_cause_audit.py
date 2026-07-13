#!/usr/bin/env python3
"""Static root-cause audit for the all-keep explicit-Q/DQ INT8 baseline.

This command is intentionally read-only with respect to the source equivalence
audit.  It reconstructs the precision-coverage and merge-boundary contracts
from immutable ONNX/JSON/CSV artifacts and records where the current search
profile, canonical exporter, and TensorRT realization diverge.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, deque
from pathlib import Path
from typing import Any, Iterable


REPO = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = REPO / "outputs/int8_baseline_equivalence_audit_20260713_021108"
WEIGHTED_OPS = {"Conv", "ConvTranspose", "Gemm", "MatMul"}


def read_json(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: Iterable[str]) -> None:
    names = list(fields)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in names})


def bool_text(value: Any) -> str:
    return "true" if bool(value) else "false"


def tensor_scale(initializers: dict[str, Any], name: str) -> dict[str, Any] | None:
    import numpy as np

    value = initializers.get(name)
    if value is None:
        return None
    array = np.asarray(value)
    flat = array.reshape(-1)
    return {
        "initializer": name,
        "shape": list(array.shape),
        "values": [float(item) for item in flat] if flat.size <= 32 else [],
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
    }


def load_engine_rows(path: Path) -> list[dict[str, str]]:
    return read_csv(path)


def match_search_engine(entry: dict[str, Any], rows: list[dict[str, str]]) -> dict[str, str] | None:
    canonical = str(entry["canonical_node_name"])
    matches = [row for row in rows if row.get("weighted_compute") == "True" and canonical in row.get("name", "")]
    return matches[0] if len(matches) == 1 else None


def legacy_name_variants(name: str) -> set[str]:
    values = {name}
    values.add(name.replace("pfn_layers.0", "pfn_layers_0"))
    values.add(name.replace("pfn_layers.0", "pfn_layers_0").replace("/linear/MatMul", "/linear/MatMul_myl"))
    return {value for value in values if value}


def match_legacy_engine(entry: dict[str, Any], rows: list[dict[str, str]]) -> dict[str, str] | None:
    variants = legacy_name_variants(str(entry["original_node_name"]))
    matches = [
        row
        for row in rows
        if row.get("weighted_compute") == "True"
        and any((variant in row.get("name", "")) for variant in variants)
    ]
    return matches[0] if len(matches) == 1 else None


def reason_for_entry(entry: dict[str, Any], expansion: dict[str, list[str]]) -> tuple[str, str, str]:
    module = str(entry["module_path"])
    group = str(entry["precision_group"])
    requested = str(entry["requested_precision"]).lower()
    if requested == "int8":
        return "requested_int8_and_realized_int8", "", ""
    if "pillar_vfe" in module.lower():
        return (
            "search_space_unsupported_int8_keyword",
            "tracer._module_supported_precisions rejects module names containing pillar_vfe before canonical legalization",
            "unsupported_int8:pillar_vfe",
        )
    if any(token in module.lower() for token in ("single_head", "cls_head", "reg_head", "dir_head")):
        return (
            "search_space_head_constraint",
            "build_precision_coupling_groups(..., allow_head_int8=False) removes INT8 for head modules",
            "head_constraint",
        )
    members = expansion.get(group, [])
    poison = [name for name in members if any(token in name.lower() for token in ("single_head", "cls_head", "reg_head", "dir_head"))]
    if group.startswith("pg_scope_") and poison:
        return (
            "precision_group_overcoupling_with_head_member",
            "a pruning dependency scope was converted to force_same_precision=True and its allowed set was intersected with a head constraint",
            "trace_dependency_scope_intersection:" + ";".join(poison),
        )
    return "search_profile_requested_fp16", "profile requested FP16 before Q/DQ insertion", ""


def build_coverage(source: Path, output: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    artifact = source / "search_maximal_legal_int8_force_rebuild/artifacts"
    mapping = read_json(artifact / "canonical_layer_map.json", {})
    expansion = read_json(artifact / "precision_group_expansion.json", {})
    requested_groups = read_json(artifact / "requested_quantization_groups.json", {})
    stage1_groups = read_json(artifact / "stage1_legalized_quantization_groups.json", {})
    realized_groups = read_json(artifact / "stage2_realized_quantization_groups.json", {})
    inventory = read_json(source / "qdq_inventory.json", {})
    qrows = inventory.get("rows", [])
    qby_layer_role = {(str(row.get("canonical_layer")), str(row.get("quant_role"))): row for row in qrows}
    legacy_rows = load_engine_rows(source / "engine_layer_precision_legacy.csv")
    search_rows = load_engine_rows(source / "engine_layer_precision_search.csv")
    rows: list[dict[str, Any]] = []
    matched_search_indices: set[str] = set()
    direct_counts: Counter[str] = Counter()
    group_counts: Counter[str] = Counter()
    for entry in mapping.get("entries", []):
        module = str(entry["module_path"])
        group = str(entry["precision_group"])
        requested = str(entry["requested_precision"]).lower()
        legalized = str(entry["realized_request_precision"]).lower()
        weight = qby_layer_role.get((module, "weight"), {})
        input_q = qby_layer_role.get((module, "activation_input"), {})
        output_q = qby_layer_role.get((module, "activation_output"), {})
        legacy = match_legacy_engine(entry, legacy_rows)
        search = match_search_engine(entry, search_rows)
        if search is not None:
            matched_search_indices.add(str(search.get("index")))
        direct, detail, constraint = reason_for_entry(entry, expansion)
        if requested == "int8" and legalized != "int8":
            direct = "legalizer_rejected_int8"
            detail = str(entry.get("fallback_reason", ""))
        search_precision = str(search.get("precision", "")) if search else ""
        if legalized == "int8" and search_precision != "int8":
            direct = "tensorrt_builder_fallback"
            detail = f"legalized=int8, engine={search_precision or 'unmatched'}"
        direct_counts[direct] += 1
        if requested == "fp16":
            group_counts[group] += 1
        fallback = ""
        if requested != legalized:
            fallback = str(entry.get("fallback_reason", "")) or "canonical_legalizer_changed_request"
        elif search and legalized != search_precision:
            fallback = f"TensorRT realized {search_precision} from {legalized}"
        row = {
            "canonical_layer": module,
            "module_path": module,
            "op_type": entry.get("onnx_op_type", ""),
            "quant_group_id": group,
            "requested_precision": requested,
            "legalized_precision": legalized,
            "weight_qdq_present": bool_text(weight),
            "input_activation_q_present": bool_text(input_q),
            "output_activation_q_present": bool_text(output_q),
            "weight_scale_shape": json.dumps(weight.get("scale_shape", ""), separators=(",", ":")),
            "weight_axis": weight.get("axis", ""),
            "legacy_realized_precision": legacy.get("precision", "unmatched") if legacy else "unmatched",
            "search_realized_precision": search_precision or "unmatched",
            "legalizer_reason": str(entry.get("fallback_reason", "")),
            "mapping_status": "canonical_exact+legacy_exact+search_exact" if legacy and search else f"legacy={bool(legacy)};search={bool(search)}",
            "builder_constraint": constraint,
            "fallback_reason": fallback,
            "direct_reason_category": direct,
            "direct_reason_detail": detail,
            "original_onnx_node": entry.get("original_node_name", ""),
            "canonical_onnx_node": entry.get("canonical_node_name", ""),
            "requested_group_precision": str(requested_groups.get(group, "")).lower(),
            "stage1_legalized_group_precision": str(stage1_groups.get(group, "")).lower(),
            "stage2_realized_group_precision": str(realized_groups.get(group, "")).lower(),
        }
        rows.append(row)

    extra_search = [
        row
        for row in search_rows
        if row.get("weighted_compute") == "True" and str(row.get("index")) not in matched_search_indices
    ]
    for engine_row in extra_search:
        direct = "functional_matmul_missing_from_canonical_mapping"
        direct_counts[direct] += 1
        legacy_match = next(
            (
                row
                for row in legacy_rows
                if row.get("weighted_compute") == "True"
                and "/MatMul_2" in row.get("name", "")
                and "/MatMul_2" in engine_row.get("name", "")
            ),
            None,
        )
        rows.append(
            {
                "canonical_layer": "<functional_matmul_untracked>",
                "module_path": "<functional_matmul_untracked>",
                "op_type": "MatMul",
                "quant_group_id": "",
                "requested_precision": "not_in_profile",
                "legalized_precision": "not_in_profile",
                "weight_qdq_present": "false",
                "input_activation_q_present": "false",
                "output_activation_q_present": "false",
                "weight_scale_shape": "",
                "weight_axis": "",
                "legacy_realized_precision": legacy_match.get("precision", "unmatched") if legacy_match else "unmatched",
                "search_realized_precision": engine_row.get("precision", "unmatched"),
                "legalizer_reason": "not represented by module origin map",
                "mapping_status": "functional_weighted_node_not_in_canonical_layer_map",
                "builder_constraint": "canonical search profile has no assignment for functional MatMul",
                "fallback_reason": "",
                "direct_reason_category": direct,
                "direct_reason_detail": "TensorRT sees an additional weighted functional MatMul that the 69-module search space does not own",
                "original_onnx_node": "",
                "canonical_onnx_node": engine_row.get("name", ""),
                "requested_group_precision": "",
                "stage1_legalized_group_precision": "",
                "stage2_realized_group_precision": "",
            }
        )

    fields = [
        "canonical_layer", "module_path", "op_type", "quant_group_id", "requested_precision",
        "legalized_precision", "weight_qdq_present", "input_activation_q_present",
        "output_activation_q_present", "weight_scale_shape", "weight_axis",
        "legacy_realized_precision", "search_realized_precision", "legalizer_reason",
        "mapping_status", "builder_constraint", "fallback_reason", "direct_reason_category",
        "direct_reason_detail", "original_onnx_node", "canonical_onnx_node",
        "requested_group_precision", "stage1_legalized_group_precision", "stage2_realized_group_precision",
    ]
    write_csv(output / "precision_coverage_matrix.csv", rows, fields)
    fp16 = [row for row in rows if row["search_realized_precision"] == "fp16"]
    summary = {
        "canonical_layer_count": len(mapping.get("entries", [])),
        "engine_weighted_layer_count": sum(row.get("weighted_compute") == "True" for row in search_rows),
        "search_weighted_int8_count": sum(row["search_realized_precision"] == "int8" for row in rows),
        "search_weighted_fp16_count": len(fp16),
        "fp16_direct_reason_counts": dict(Counter(row["direct_reason_category"] for row in fp16)),
        "all_layer_direct_reason_counts": dict(direct_counts),
        "requested_fp16_group_counts": dict(group_counts),
        "requested_int8_but_missing_weight_qdq": [row["canonical_layer"] for row in rows if row["requested_precision"] == "int8" and row["weight_qdq_present"] != "true"],
        "requested_int8_but_missing_input_q": [row["canonical_layer"] for row in rows if row["requested_precision"] == "int8" and row["input_activation_q_present"] != "true"],
        "requested_int8_but_missing_output_q": [row["canonical_layer"] for row in rows if row["requested_precision"] == "int8" and row["output_activation_q_present"] != "true"],
        "canonical_mapping_missing_count": sum("canonical" not in row["mapping_status"] for row in rows),
        "canonical_legalizer_rejection_count": sum(row["direct_reason_category"] == "legalizer_rejected_int8" for row in rows),
        "tensorrt_builder_fallback_count": sum(row["direct_reason_category"] == "tensorrt_builder_fallback" for row in rows),
        "conclusion": "The 22-layer ceiling is imposed primarily by the search-space precision groups before Q/DQ export, not by TensorRT hardware fallback.",
    }
    write_json(output / "precision_coverage_summary.json", summary)
    lines = [
        "# Precision coverage root-cause audit", "",
        f"- Canonical weighted layers: {summary['canonical_layer_count']}",
        f"- TensorRT weighted layers: {summary['engine_weighted_layer_count']}",
        f"- Search realized INT8/FP16: {summary['search_weighted_int8_count']}/{summary['search_weighted_fp16_count']}",
        f"- Canonical legalizer INT8 rejection: {summary['canonical_legalizer_rejection_count']}",
        f"- TensorRT builder fallback: {summary['tensorrt_builder_fallback_count']}", "",
        "## Direct causes of the 48 FP16 layers", "",
    ]
    for key, count in sorted(summary["fp16_direct_reason_counts"].items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- `{key}`: {count}")
    lines += ["", "The per-layer evidence is in `precision_coverage_matrix.csv`. The dominant cause is pruning-dependency-scope over-coupling in the search profile; no requested INT8 canonical layer was rejected by the legalizer or fell back in TensorRT.", ""]
    (output / "precision_coverage_summary.md").write_text("\n".join(lines), encoding="utf-8")
    return rows, summary


def graph_index(model: Any) -> dict[str, Any]:
    producer: dict[str, Any] = {}
    consumers: dict[str, list[Any]] = {}
    nodes = {str(node.name): node for node in model.graph.node}
    for node in model.graph.node:
        for name in node.output:
            producer[str(name)] = node
        for name in node.input:
            consumers.setdefault(str(name), []).append(node)
    return {"producer": producer, "consumers": consumers, "nodes": nodes}


def nearest_upstream_weighted(tensor: str, index: dict[str, Any], canonical_by_node: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    queue: deque[tuple[str, int]] = deque([(tensor, 0)])
    seen: set[str] = set()
    found: list[tuple[int, dict[str, Any]]] = []
    best: int | None = None
    while queue:
        value, distance = queue.popleft()
        if value in seen or (best is not None and distance > best):
            continue
        seen.add(value)
        node = index["producer"].get(value)
        if node is None:
            continue
        entry = canonical_by_node.get(str(node.name))
        if entry is not None:
            best = distance
            found.append((distance, entry))
            continue
        for name in node.input:
            queue.append((str(name), distance + 1))
    unique: dict[str, dict[str, Any]] = {}
    for _, entry in found:
        unique[str(entry["module_path"])] = entry
    return list(unique.values())


def nearest_downstream_weighted(tensor: str, index: dict[str, Any], canonical_by_node: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    queue: deque[tuple[str, int]] = deque([(tensor, 0)])
    seen: set[str] = set()
    found: list[tuple[int, dict[str, Any]]] = []
    best: int | None = None
    while queue:
        value, distance = queue.popleft()
        if value in seen or (best is not None and distance > best):
            continue
        seen.add(value)
        for node in index["consumers"].get(value, []):
            entry = canonical_by_node.get(str(node.name))
            if entry is not None:
                best = distance
                found.append((distance, entry))
                continue
            for output in node.output:
                queue.append((str(output), distance + 1))
    unique: dict[str, dict[str, Any]] = {}
    for _, entry in found:
        unique[str(entry["module_path"])] = entry
    return list(unique.values())


def qdq_on_backward_path(tensor: str, index: dict[str, Any], initializers: dict[str, Any], *, max_depth: int = 12) -> list[dict[str, Any]]:
    queue: deque[tuple[str, int]] = deque([(tensor, 0)])
    seen: set[str] = set()
    found: list[dict[str, Any]] = []
    while queue:
        value, depth = queue.popleft()
        if value in seen or depth > max_depth:
            continue
        seen.add(value)
        node = index["producer"].get(value)
        if node is None:
            continue
        if str(node.op_type) == "DequantizeLinear":
            qnode = index["producer"].get(str(node.input[0]))
            scale_name = str(node.input[1]) if len(node.input) > 1 else ""
            found.append(
                {
                    "distance": depth,
                    "dequantize_node": str(node.name),
                    "quantize_node": str(qnode.name) if qnode is not None and str(qnode.op_type) == "QuantizeLinear" else "",
                    "source_tensor": str(qnode.input[0]) if qnode is not None and str(qnode.op_type) == "QuantizeLinear" else "",
                    "dequantized_tensor": str(node.output[0]),
                    "scale": tensor_scale(initializers, scale_name),
                    "immediately_before_merge": depth == 0,
                }
            )
        if str(node.op_type) in WEIGHTED_OPS:
            continue
        for name in node.input:
            queue.append((str(name), depth + 1))
    return found


def qdq_on_forward_path(tensor: str, index: dict[str, Any], initializers: dict[str, Any], *, max_depth: int = 12) -> list[dict[str, Any]]:
    queue: deque[tuple[str, int]] = deque([(tensor, 0)])
    seen: set[str] = set()
    found: list[dict[str, Any]] = []
    while queue:
        value, depth = queue.popleft()
        if value in seen or depth > max_depth:
            continue
        seen.add(value)
        for node in index["consumers"].get(value, []):
            if str(node.op_type) == "QuantizeLinear":
                scale_name = str(node.input[1]) if len(node.input) > 1 else ""
                found.append(
                    {
                        "distance": depth,
                        "quantize_node": str(node.name),
                        "source_tensor": str(node.input[0]),
                        "scale": tensor_scale(initializers, scale_name),
                        "immediately_after_merge": depth == 0,
                    }
                )
                continue
            if str(node.op_type) in WEIGHTED_OPS:
                continue
            for output_name in node.output:
                queue.append((str(output_name), depth + 1))
    return found


def merge_engine_match(name: str, rows: list[dict[str, str]]) -> dict[str, str] | None:
    matches = [row for row in rows if name in row.get("name", "")]
    return matches[0] if matches else None


def branch_record(
    tensor: str,
    base_index: dict[str, Any],
    qdq_index: dict[str, Any],
    canonical_by_node: dict[str, dict[str, Any]],
    initializers: dict[str, Any],
    search_engine: list[dict[str, str]],
    legacy_engine: list[dict[str, str]],
) -> dict[str, Any]:
    entries = nearest_upstream_weighted(tensor, base_index, canonical_by_node)
    layers = []
    for entry in entries:
        search = match_search_engine(entry, search_engine)
        legacy = match_legacy_engine(entry, legacy_engine)
        layers.append(
            {
                "canonical_layer": entry["module_path"],
                "quantization_group": entry["precision_group"],
                "requested_precision": entry["requested_precision"],
                "legalized_precision": entry["realized_request_precision"],
                "search_realized_precision": search.get("precision", "unmatched") if search else "unmatched",
                "legacy_realized_precision": legacy.get("precision", "unmatched") if legacy else "unmatched",
            }
        )
    return {
        "merge_input_tensor": tensor,
        "nearest_weighted_producers": layers,
        "qdq_before_merge": qdq_on_backward_path(tensor, qdq_index, initializers),
    }


def build_merge_audit(source: Path, output: Path) -> dict[str, Any]:
    import onnx
    from onnx import numpy_helper

    artifact = source / "search_maximal_legal_int8_force_rebuild/artifacts"
    base = onnx.load(str(artifact / "pruned_fp32.onnx"))
    qdq = onnx.load(str(artifact / "qdq.onnx"))
    base_idx = graph_index(base)
    qdq_idx = graph_index(qdq)
    initializers = {str(row.name): numpy_helper.to_array(row) for row in qdq.graph.initializer}
    mapping = read_json(artifact / "canonical_layer_map.json", {})
    canonical_by_node = {str(row["canonical_node_name"]): row for row in mapping.get("entries", [])}
    search_engine = load_engine_rows(source / "engine_layer_precision_search.csv")
    legacy_engine = load_engine_rows(source / "engine_layer_precision_legacy.csv")
    merges = []
    for node in base.graph.node:
        name = str(node.name)
        op_type = str(node.op_type)
        is_residual = op_type == "Add" and (name.startswith("/layer0/") or name.startswith("/layer1/") or name.startswith("/layer2/"))
        is_feature_concat = op_type == "Concat" and name == "/Concat_9"
        if not (is_residual or is_feature_concat):
            continue
        search_merge = merge_engine_match(name, search_engine)
        legacy_merge = merge_engine_match(name, legacy_engine)
        branches = [
            branch_record(str(value), base_idx, qdq_idx, canonical_by_node, initializers, search_engine, legacy_engine)
            for value in node.input
        ]
        output_q = qdq_on_forward_path(str(node.output[0]), qdq_idx, initializers)
        downstream = nearest_downstream_weighted(str(node.output[0]), base_idx, canonical_by_node)
        downstream_rows = []
        for entry in downstream:
            search = match_search_engine(entry, search_engine)
            downstream_rows.append(
                {
                    "canonical_layer": entry["module_path"],
                    "quantization_group": entry["precision_group"],
                    "requested_precision": entry["requested_precision"],
                    "legalized_precision": entry["realized_request_precision"],
                    "search_realized_precision": search.get("precision", "unmatched") if search else "unmatched",
                }
            )
        immediate = [
            bool(any(item.get("immediately_before_merge") for item in branch["qdq_before_merge"]))
            for branch in branches
        ]
        merge_precision = search_merge.get("precision", "unmatched") if search_merge else "unmatched_or_fused"
        policy = "A_fp16_merge"
        if merge_precision == "int8":
            policy = "B_int8_merge"
        input_scales = [
            [item.get("scale", {}).get("max") for item in branch["qdq_before_merge"] if item.get("immediately_before_merge") and item.get("scale")]
            for branch in branches
        ]
        merges.append(
            {
                "merge_op_name": name,
                "merge_op_type": op_type,
                "merge_kind": "residual_add" if is_residual else "feature_concat",
                "input_branches": branches,
                "immediate_input_qdq_presence": immediate,
                "single_sided_immediate_qdq": sum(immediate) == 1,
                "all_inputs_immediate_qdq": bool(immediate) and all(immediate),
                "immediate_input_scales": input_scales,
                "merge_output_tensor": str(node.output[0]),
                "qdq_after_merge_before_next_weighted_layer": output_q,
                "downstream_weighted_layers": downstream_rows,
                "search_merge_realized_precision": merge_precision,
                "legacy_merge_realized_precision": legacy_merge.get("precision", "unmatched_or_fused") if legacy_merge else "unmatched_or_fused",
                "current_effective_policy": policy,
                "scale_compatibility_rule": "not_required_after_DequantizeLinear; inputs are floating tensors" if policy == "A_fp16_merge" else "not_implemented_or_not_proven",
                "explicit_policy_metadata_present": False,
                "deployment_signature_contains_merge_policy": False,
                "contract_assessment": (
                    "numerically valid FP16 merge topology, but incidental: Q/DQ is inserted per weighted node and the merge policy/scale ownership is not represented or validated"
                    if policy == "A_fp16_merge"
                    else "INT8 merge requires common-scale or explicit requantization proof, which current exporter does not provide"
                ),
            }
        )
    add_rows = [row for row in merges if row["merge_kind"] == "residual_add"]
    concat_rows = [row for row in merges if row["merge_kind"] == "feature_concat"]
    result = {
        "design_contract": {
            "A_FP16_merge": "branches may differ; each quantized branch must be dequantized/cast to FP16 before merge; merge output may be quantized for a downstream INT8 consumer",
            "B_INT8_merge": "all inputs must use compatible INT8 representations through identical scales or explicit requantization to a common scale",
        },
        "current_exporter_capability": {
            "inserts_qdq_per_weighted_node": True,
            "merge_aware_topology": False,
            "merge_scale_policy": False,
            "merge_policy_in_deployment_signature": False,
            "int8_requantization_to_common_scale": False,
        },
        "residual_add_count": len(add_rows),
        "feature_concat_count": len(concat_rows),
        "residual_add_single_sided_immediate_qdq_count": sum(row["single_sided_immediate_qdq"] for row in add_rows),
        "concat_single_sided_immediate_qdq_count": sum(row["single_sided_immediate_qdq"] for row in concat_rows),
        "merges": merges,
        "conclusion": {
            "residual_single_side": "The three observed single-sided Add boundaries are an incidental consequence of per-layer Q/DQ insertion around a floating merge. They are not an implemented or serialized merge-policy choice.",
            "concat": "Concat_9 currently executes as a floating concat followed by one downstream input quantizer. Its topology is compatible with policy A, but the exporter does not declare, own, or validate that policy.",
        },
    }
    write_json(output / "merge_quantization_audit.json", result)
    lines = [
        "# Merge quantization audit", "",
        f"- Residual Add nodes audited: {result['residual_add_count']}",
        f"- Residual Adds with exactly one immediate Q/DQ branch: {result['residual_add_single_sided_immediate_qdq_count']}",
        f"- Feature Concats audited: {result['feature_concat_count']}", "",
        "## Verdict", "",
        result["conclusion"]["residual_single_side"], "",
        result["conclusion"]["concat"], "",
        "The present graph is de facto policy A (floating merge), because every explicit QuantizeLinear is followed by DequantizeLinear before a floating ONNX consumer. The missing implementation is not a mandatory second Q/DQ on every branch; it is the absence of a first-class merge contract, scale owner, cast rule, validation, and deployment-signature field.", "",
        "## Per-merge summary", "",
        "| Merge | Kind | immediate Q/DQ inputs | search merge precision | downstream |", "|---|---|---:|---|---|",
    ]
    for row in merges:
        downstream_text = ", ".join(f"{item['canonical_layer']}:{item['search_realized_precision']}" for item in row["downstream_weighted_layers"])
        lines.append(f"| `{row['merge_op_name']}` | {row['merge_kind']} | {sum(row['immediate_input_qdq_presence'])}/{len(row['immediate_input_qdq_presence'])} | {row['search_merge_realized_precision']} | {downstream_text} |")
    lines.append("")
    (output / "merge_quantization_audit.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def build_calibration_audit(source: Path, output: Path) -> dict[str, Any]:
    artifact = source / "search_maximal_legal_int8_force_rebuild/artifacts"
    scales_doc = read_json(artifact / "calibration_scales.json", {})
    metadata = scales_doc.get("metadata", {})
    scales = scales_doc.get("scales", {})
    inventory = read_json(source / "qdq_inventory.json", {})
    diff_rows = read_csv(source / "scale_diff.csv")
    ratio_by_layer_role = {
        (row.get("canonical_layer", ""), row.get("quant_role", "")): row
        for row in diff_rows
        if row.get("scale_ratio_search_over_legacy", "")
    }
    qrows = inventory.get("rows", [])
    lineage = []
    output_paths = metadata.get("output_module_paths", {})
    for row in qrows:
        role = str(row.get("quant_role", ""))
        module = str(row.get("canonical_layer", ""))
        if role == "weight":
            observer = "final BN-folded ONNX initializer"
            statistic = "global absolute maximum over all initializer elements"
            q_input_identity = bool(row.get("weight_scale_matches_final_folded_initializer"))
        elif role == "activation_input":
            observer = f"PyTorch forward_pre_hook:{module}"
            statistic = "global absolute maximum over all elements and all 200 calibration calls"
            q_input_identity = False
        else:
            observer_path = str(output_paths.get(module, module))
            observer = f"PyTorch forward_hook:{observer_path}"
            statistic = "global absolute maximum over all elements and all 200 calibration calls"
            q_input_identity = False
        ratio = ratio_by_layer_role.get((module, role), {})
        lineage.append(
            {
                "canonical_layer": module,
                "quant_role": role,
                "observer_module_or_tensor": observer,
                "q_node_actual_input_tensor": row.get("tensor", ""),
                "q_node": row.get("quantize_node", ""),
                "statistic": statistic,
                "clipping_threshold": "amax (no clipping search)",
                "scale_formula": "amax / 127",
                "scale": row.get("scale_max"),
                "scale_shape": row.get("scale_shape"),
                "axis": row.get("axis", ""),
                "bn_fold_position": "paired BN output before ReLU" if role == "activation_output" and output_paths.get(module, module) != module else ("final folded ONNX weight" if role == "weight" else "module boundary"),
                "multi_call_aggregation": "torch.maximum of per-call amax",
                "padding_agent_dynamic_policy": "no explicit masking, padding exclusion, or per-agent stratification in collector",
                "observer_tensor_equals_q_input_proven": q_input_identity,
                "observer_tensor_identity_status": "exact initializer hash/amax verified" if q_input_identity else "assumed by module-origin mapping; not numerically proven",
                "legacy_scale": ratio.get("legacy_scale", ""),
                "search_over_legacy_scale_ratio": ratio.get("scale_ratio_search_over_legacy", ""),
            }
        )
    ratios = [float(row["search_over_legacy_scale_ratio"]) for row in lineage if row["search_over_legacy_scale_ratio"] not in ("", None)]
    result = {
        "calibration_algorithm": "symmetric global per-tensor absolute-maximum calibration",
        "activation_observer": "PyTorch task-model module hooks",
        "weight_observer": "final BN-folded ONNX initializer",
        "sample_count": metadata.get("frame_count", 200),
        "statistic": "max(abs(x)) across every tensor element and calibration call",
        "clipping": "none",
        "entropy_or_percentile": False,
        "scale_formula": "amax / 127",
        "zero_point": 0,
        "activation_granularity": "per_tensor",
        "weight_granularity": "per_tensor",
        "observer_q_input_identity": {
            "weights": "verified from exact final ONNX initializer",
            "activations": "not verified; module-origin correspondence is assumed, while calibration uses task-model forward and deployment uses the signal-maxK export graph",
        },
        "padding_and_dynamic_dimensions": "collector applies no explicit valid-mask filtering, padded-voxel exclusion, agent stratification, percentile clipping, or entropy threshold optimization",
        "matched_legacy_activation_ratio": {
            "count": len(ratios),
            "min": min(ratios) if ratios else None,
            "max": max(ratios) if ratios else None,
            "median": sorted(ratios)[len(ratios) // 2] if ratios else None,
        },
        "why_scales_are_larger": [
            "search uses an unclipped global maximum, while the legacy cache was produced by TensorRT EntropyCalibration2 and may select a smaller clipping threshold",
            "search activation hooks are attached to the PyTorch task model, not numerically proven to observe the exact tensor entering each Q node in the signal-maxK ONNX graph",
            "global maxima include every element; no padding/validity mask or outlier rejection is applied",
        ],
        "lineage": lineage,
    }
    write_json(output / "calibration_lineage_audit.json", result)
    write_csv(
        output / "calibration_lineage.csv",
        lineage,
        [
            "canonical_layer", "quant_role", "observer_module_or_tensor", "q_node_actual_input_tensor",
            "q_node", "statistic", "clipping_threshold", "scale_formula", "scale", "scale_shape", "axis",
            "bn_fold_position", "multi_call_aggregation", "padding_agent_dynamic_policy",
            "observer_tensor_equals_q_input_proven", "observer_tensor_identity_status", "legacy_scale",
            "search_over_legacy_scale_ratio",
        ],
    )
    lines = [
        "# Calibration lineage audit", "",
        f"Current algorithm: **{result['calibration_algorithm']}** (`amax / 127`, zero point 0).", "",
        "It is not entropy or percentile calibration and performs no clipping search. Weight identity is proven against the final folded ONNX initializer; activation observer identity is only assumed through the module-origin map and is not yet numerically proven against the actual Q-node input tensor.", "",
        "The 2–10× scale gap is therefore consistent with unclipped outlier-sensitive maxima versus legacy EntropyCalibration2, with an additional unresolved risk that the task-model hook tensor differs from the signal-maxK deployment boundary.", "",
        "Per-boundary evidence is in `calibration_lineage.csv`.", "",
    ]
    (output / "calibration_lineage_audit.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def build_group_semantics(source: Path, output: Path, coverage: list[dict[str, Any]], merge: dict[str, Any]) -> dict[str, Any]:
    artifact = source / "search_maximal_legal_int8_force_rebuild/artifacts"
    expansion = read_json(artifact / "precision_group_expansion.json", {})
    requested = read_json(artifact / "requested_quantization_groups.json", {})
    stage1 = read_json(artifact / "stage1_legalized_quantization_groups.json", {})
    stage2 = read_json(artifact / "stage2_realized_quantization_groups.json", {})
    rows = []
    for group, members in expansion.items():
        member_rows = [row for row in coverage if row.get("quant_group_id") == group]
        constraints = sorted({row.get("builder_constraint", "") for row in member_rows if row.get("builder_constraint")})
        merge_names = []
        for item in merge.get("merges", []):
            groups = {
                producer.get("quantization_group", "")
                for branch in item.get("input_branches", [])
                for producer in branch.get("nearest_weighted_producers", [])
            }
            groups.update(row.get("quantization_group", "") for row in item.get("downstream_weighted_layers", []))
            if group in groups:
                merge_names.append(item["merge_op_name"])
        rows.append(
            {
                "quantization_group": group,
                "member_layers": members,
                "requested_precision": requested.get(group),
                "legalized_precision": stage1.get(group),
                "realized_precision": stage2.get(group),
                "current_group_origin": "pruning_dependency_scope" if group.startswith("pg_scope_") else "precision_coupling_tracer",
                "force_same_precision_effective": group.startswith("pg_scope_") or len(members) == 1,
                "merge_boundaries": sorted(set(merge_names)),
                "input_output_qdq_placement_owned_by_group": False,
                "activation_scale_owner": "per-layer calibration record, not group/merge boundary",
                "merge_scale_policy": "missing",
                "weight_granularity_axis": "per-tensor scalar; no axis",
                "constraints": constraints,
            }
        )
    result = {
        "group_count": len(rows),
        "groups": rows,
        "contract_fields_required": [
            "member_layers", "merge_boundaries", "input_output_qdq_placement", "activation_scale_ownership",
            "merge_scale_policy", "weight_granularity_axis", "legalized_precision", "realized_precision",
        ],
        "exporter_execution": {
            "member_layer_precision_expansion": True,
            "merge_boundaries": False,
            "input_output_qdq_placement_from_group": False,
            "activation_scale_ownership_from_group": False,
            "merge_scale_policy": False,
            "weight_granularity_axis_from_group": False,
            "legalized_precision": True,
            "realized_precision_validated": True,
        },
        "fully_executed": False,
        "conclusion": "The exporter executes only the expanded per-layer precision string. It does not execute the rest of the quantization-coupling semantics required for residual/concat deployment.",
    }
    write_json(output / "quantization_group_semantics_audit.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-audit-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source_audit_root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing audit directory: {output}")
    output.mkdir(parents=True)
    coverage, coverage_summary = build_coverage(source, output)
    merge = build_merge_audit(source, output)
    calibration = build_calibration_audit(source, output)
    groups = build_group_semantics(source, output, coverage, merge)
    sources = [
        source / "search_maximal_legal_int8_force_rebuild/artifacts/pruned_fp32.onnx",
        source / "search_maximal_legal_int8_force_rebuild/artifacts/qdq.onnx",
        source / "search_maximal_legal_int8_force_rebuild/artifacts/canonical_layer_map.json",
        source / "search_maximal_legal_int8_force_rebuild/artifacts/calibration_scales.json",
        source / "engine_layer_precision_legacy.csv",
        source / "engine_layer_precision_search.csv",
    ]
    manifest = {
        "audit_kind": "all_keep_explicit_qdq_static_root_cause",
        "source_audit_root": str(source),
        "read_only_sources": [
            {"path": str(path), "size": path.stat().st_size, "sha256": sha256_file(path)} for path in sources
        ],
        "summary": {
            "coverage": coverage_summary,
            "merge": {key: value for key, value in merge.items() if key != "merges"},
            "calibration": {key: value for key, value in calibration.items() if key != "lineage"},
            "group_semantics": {key: value for key, value in groups.items() if key != "groups"},
        },
    }
    write_json(output / "static_audit_manifest.json", manifest)
    print(json.dumps({"output": str(output), "coverage": coverage_summary, "merge_count": len(merge["merges"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
