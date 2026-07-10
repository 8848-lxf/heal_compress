#!/usr/bin/env python3
"""Dependency graph and CoupledChannelUnit audit for v9.8.

This script is deliberately an audit tool, not a pruning sweep.  It records the
actual capability of the current tracer/group builder and writes stable JSON/CSV
artifacts for downstream mask0 search readiness checks.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

import torch
import torch.nn as nn

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[2]
_UNIAD = _ROOT.parent
for _p in (_UNIAD, _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from heal_compress.pruning.full_model_surface import apply_full_model_prunable_surface  # noqa: E402
from heal_compress.pruning.global_plan_shape_simulator import GlobalPlanShapeSimulator  # noqa: E402
from heal_compress.pruning.grouped_conv import resolve_grouped_conv_input_keep  # noqa: E402
from heal_compress.pruning.physical_prune_plan import GlobalPhysicalPrunePlan, ModuleAxisPruneRequest  # noqa: E402
from heal_compress.pruning.propagation import GroupBuilder  # noqa: E402
from heal_compress.pruning.tp_oracle_diff import build_tp_oracle_diff  # noqa: E402
from heal_compress.pruning.units import (  # noqa: E402
    coupled_channel_unit_rows,
    expand_coupled_channel_units,
)
from heal_compress.tracer.generic_tracer import trace_model  # noqa: E402
from heal_compress.tracer.op_graph import (  # noqa: E402
    OP_ADD,
    OP_BEV_WARP,
    OP_BN,
    OP_CAT,
    OP_CONV,
    OP_CONVT,
    OP_LINEAR,
    OP_NORM,
    OP_OTHER,
    OP_SPLIT,
    OpGraph,
    build_op_graph,
)


OUT_DEFAULT = "outputs/latency_lut/dependency_graph_audit_v98"
POLICY_TO_GROUPED_MODE = {
    "A": "flat_output_groups_fixed",
    "B": "group_balanced_output_groups_fixed",
    "C": "remove_groups",
    "D": "group_coarsening_zero_padded_reblock",
}
REQUIRED_UNIT_MEMBER_FIELDS = {
    "module_name",
    "module_type",
    "axis",
    "local_index",
    "dependency_type",
    "producer_tensor",
    "consumer_tensor",
    "branch_id",
    "concat_offset",
    "residual_add_id",
    "grouped_conv_role",
    "transpose_conv_role",
}


def _jsonable(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fields.append(key)
    if not fields:
        fields = ["empty"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False, default=str) if isinstance(value, (dict, list)) else value for key, value in row.items()})


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _module_axis_channels(module: nn.Module, axis: str) -> int:
    if isinstance(module, nn.Conv2d):
        return int(module.out_channels if axis in {"out", "grouped_coarsen_out"} else module.in_channels)
    if isinstance(module, nn.ConvTranspose2d):
        return int(module.out_channels if axis == "out" else module.in_channels)
    if isinstance(module, nn.modules.batchnorm._BatchNorm):
        return int(module.num_features)
    if isinstance(module, nn.Linear):
        return int(module.out_features if axis == "out" else module.in_features)
    return int(getattr(module, "out_channels", getattr(module, "num_features", getattr(module, "out_features", 0))) or 0)


def _replay_axis_for_item(item: Any) -> str:
    fn_name = getattr(getattr(item, "pruning_fn", None), "__name__", "")
    module = getattr(item, "module", None)
    if (
        isinstance(module, nn.Conv2d)
        and int(module.groups) > 1
        and getattr(item, "direction", "") == "in"
    ):
        return "grouped_input_balanced"
    if fn_name == "prune_grouped_conv_input_balanced":
        return "grouped_input_balanced"
    if fn_name == "prune_grouped_flat_output_groups_fixed":
        return "grouped_flat_output"
    if fn_name == "prune_grouped_group_balanced_output_groups_fixed":
        return "grouped_group_balanced_output"
    if fn_name == "prune_grouped_remove_groups":
        return "grouped_remove"
    if fn_name == "prune_grouped_independent_topk":
        return "grouped_independent_keep"
    return str(getattr(item, "direction", ""))


def _is_regular_grouped_conv(module: Any) -> bool:
    return (
        isinstance(module, nn.Conv2d)
        and int(module.groups) > 1
        and not (int(module.groups) == int(module.in_channels) == int(module.out_channels))
    )


def _is_depthwise_conv(module: Any) -> bool:
    return isinstance(module, nn.Conv2d) and int(module.groups) == int(module.in_channels) == int(module.out_channels)


def _build_plan_from_scope_indices(scope: Any, prune_indices: Iterable[int]) -> tuple[GlobalPhysicalPrunePlan, list[dict[str, Any]]]:
    """Convert one concrete scope action to original-index-space module requests."""

    concrete_prune = sorted({int(v) for v in prune_indices if 0 <= int(v) < int(getattr(scope, "num_channels", 0))})
    concrete_keep = [idx for idx in range(int(getattr(scope, "num_channels", 0))) if idx not in set(concrete_prune)]
    grouped_input_reports: list[dict[str, Any]] = []
    grouped_input_item = None
    for item in getattr(scope, "items", []):
        module = getattr(item, "module", None)
        if _is_regular_grouped_conv(module) and getattr(item, "direction", "") == "in":
            grouped_input_item = item
            break
    if grouped_input_item is not None:
        local_keep = sorted(int(v) for v in grouped_input_item.local_keep(concrete_keep))
        resolved = resolve_grouped_conv_input_keep(
            grouped_input_item.module,
            local_keep,
            allow_repair=True,
            min_in_per_group=1,
        )
        grouped_input_reports.append(
            {
                "scope_id": getattr(scope, "group_id", ""),
                "module_name": grouped_input_item.name,
                "status": "repaired" if resolved.get("repaired") else ("accepted" if resolved.get("legal") else "skipped"),
                "reason": resolved.get("reason", ""),
                "preferred_keep_count": len(local_keep),
                "actual_keep_count": len(resolved.get("keep_indices", [])),
                "per_group_kept_count": resolved.get("per_group_kept_count", {}),
            }
        )
        if resolved.get("legal", False):
            concrete_keep = sorted(int(v) for v in resolved.get("keep_indices", []))

    plan = GlobalPhysicalPrunePlan()
    for item in getattr(scope, "items", []):
        module = getattr(item, "module", None)
        direction = str(getattr(item, "direction", ""))
        local_keep = sorted(int(v) for v in item.local_keep(concrete_keep))
        total = _module_axis_channels(module, direction)
        if total <= 0:
            continue
        prune = [idx for idx in range(total) if idx not in set(local_keep)]
        if not prune:
            continue
        axis = direction
        if _is_regular_grouped_conv(module) and direction == "in":
            axis = "grouped_input_balanced"
        plan.add_request(
            ModuleAxisPruneRequest(
                module_name=item.name,
                axis=axis,
                prune_indices=prune,
                source_recipe_id=f"{getattr(scope, 'group_id', '')}::audit",
                metadata={"reason": getattr(item, "reason", ""), "replay_axis": _replay_axis_for_item(item)},
            )
        )
    return plan, grouped_input_reports


def _build_plan_from_units(scopes_by_id: dict[str, Any], unit_ids: Iterable[str], units_by_id: dict[str, Any]) -> tuple[GlobalPhysicalPrunePlan, list[dict[str, Any]]]:
    prune_by_scope: dict[str, set[int]] = defaultdict(set)
    for unit_id in unit_ids:
        unit = units_by_id.get(unit_id)
        if unit is None:
            continue
        prune_by_scope[str(unit.scope_id)].add(int(unit.root_idx))
    global_plan = GlobalPhysicalPrunePlan()
    reports: list[dict[str, Any]] = []
    for scope_id, prune_indices in sorted(prune_by_scope.items()):
        scope = scopes_by_id.get(scope_id)
        if scope is None:
            continue
        plan, grouped_reports = _build_plan_from_scope_indices(scope, sorted(prune_indices))
        reports.extend(grouped_reports)
        for req in plan.requests():
            req.source_recipe_ids = sorted(set(req.source_recipe_ids + [f"mask0::{scope_id}"]))
            global_plan.add_request(req)
    return global_plan, reports


def _edge_rows(graph: OpGraph) -> list[dict[str, Any]]:
    return [
        {"src": src, "dst": dst, "input_index": int(idx), "kind": kind}
        for src, dst, idx, kind in graph.edges
    ]


def _trace_coverage_report(
    *,
    model: nn.Module,
    trace: dict[str, Any],
    op_graph: OpGraph,
    evidence_files: list[dict[str, str]],
) -> dict[str, Any]:
    module_names = {name for name, _module in model.named_modules() if name}
    edge_endpoints = {src for src, _dst, _idx, _kind in op_graph.edges if not src.startswith("op::")}
    edge_endpoints.update(dst for _src, dst, _idx, _kind in op_graph.edges if not dst.startswith("op::"))
    traced_modules = sorted(name for name in module_names if name in edge_endpoints)
    traced_ops = sorted(
        {
            str(info.get("op", ""))
            for info in trace.get("nodes", {}).values()
            if info.get("type") == "TensorOp"
        }
    )
    untraced_modules = sorted(module_names - set(traced_modules))
    unsupported_ops = []
    for node in op_graph.nodes.values():
        if node.op_type in {OP_OTHER, OP_SPLIT, OP_BEV_WARP}:
            unsupported_ops.append(
                {
                    "node": node.name,
                    "op_type": node.op_type,
                    "raw_type": node.raw_type,
                    "reason": "unsupported_or_protected_channel_semantics",
                }
            )
    return {
        "num_forward_paths_traced": 1,
        "dynamic_branch_enumeration_enabled": False,
        "traced_modules": traced_modules,
        "traced_ops": traced_ops,
        "tensor_producer_consumer_edges": _edge_rows(op_graph),
        "untraced_modules": untraced_modules,
        "untraced_ops": [
            {
                "node": name,
                "reason": "module_node_has_no_runtime_tensor_edge_in_single_trace_or_is_container",
            }
            for name in untraced_modules
        ],
        "unsupported_ops": unsupported_ops,
        "dynamic_paths_not_covered": [
            "GenericTracer currently records one executed forward path for the supplied sample; no multi-path dynamic branch enumeration is implemented in v9.8 audit."
        ],
        "evidence_file_per_path": evidence_files,
    }


def _enrich_units_with_graph(units: list[Any], op_graph: OpGraph) -> None:
    for unit in units:
        for member in unit.members:
            name = str(member.get("module_name") or member.get("module") or "")
            incoming = [src for src, _idx in op_graph.incoming(name)] if name in op_graph.nodes else []
            outgoing = [dst for dst, _idx in op_graph.outgoing(name)] if name in op_graph.nodes else []
            if incoming:
                member["producer_tensor"] = ";".join(incoming)
            if outgoing:
                member["consumer_tensor"] = ";".join(outgoing)
        graph_edges = []
        for edge in unit.proof_edges:
            dst = str(edge.get("dst", ""))
            if dst in op_graph.nodes:
                graph_edges.append(
                    {
                        "dst": dst,
                        "incoming": [src for src, _idx in op_graph.incoming(dst)],
                        "outgoing": [out for out, _idx in op_graph.outgoing(dst)],
                        "proof_basis": "op_graph_adjacency",
                    }
                )
        unit.proof_edges = list(unit.proof_edges) + graph_edges


def _unit_completeness_report(units: list[Any], groups: list[Any]) -> dict[str, Any]:
    missing_examples = []
    for unit in units:
        for member in unit.members:
            missing = sorted(REQUIRED_UNIT_MEMBER_FIELDS - set(member))
            if missing:
                missing_examples.append({"unit_id": unit.unit_id, "member": member, "missing": missing})
                break
        if len(missing_examples) >= 20:
            break
    return {
        "num_scopes": len(groups),
        "num_units": len(units),
        "num_units_protected": sum(1 for unit in units if bool(unit.protected)),
        "num_units_minimal_proven": sum(1 for unit in units if bool(unit.is_minimal_proven)),
        "all_units_have_required_member_fields": not missing_examples,
        "missing_member_field_examples": missing_examples,
        "num_units_with_empty_proof_edges": sum(1 for unit in units if not unit.proof_edges),
        "unsupported_unit_count": sum(1 for unit in units if str(unit.unsupported_reason)),
        "proof_scope": "single_forward_trace_dependency_recipe",
        "dynamic_path_minimality_proven": False,
        "note": "is_minimal_proven means the current single-trace PruningGroup recipe has member/index proof; it is not a proof over untraced dynamic branches.",
    }


def _operator_matrix(op_graph: OpGraph, groups: list[Any]) -> list[dict[str, Any]]:
    group_items = [(item.name, item.direction) for group in groups for item in getattr(group, "items", []) if not getattr(group, "protected", False)]
    rows = [
        ("Conv2d", True, True, True, False, "standard conv out/in dependency supported", "unit+simulator", _artifact_count(op_graph, OP_CONV)),
        ("BatchNorm", True, True, True, False, "BN follows producer output channels", "unit", _artifact_count(op_graph, OP_BN)),
        ("ReLU/activation", True, True, False, False, "transparent tensor op or skipped shared module hook", "trace", _artifact_count_raw(op_graph, "relu")),
        ("Linear", True, True, True, False, "linear out/in slicing supported when traced as pure channel dependency", "unit+simulator", _artifact_count(op_graph, OP_LINEAR)),
        ("ConvTranspose2d", True, True, False, True, "protected_convtranspose_deblock_or_fpn_output_contract", "simulator rejects unless explicitly allowed", _artifact_count(op_graph, OP_CONVT)),
        ("grouped Conv2d output", True, True, True, False, "A/B output-only policies keep groups fixed", "v9.7 tests", len([1 for name, axis in group_items if axis == "out" and _is_regular_grouped_conv(op_graph.module_of(name))])),
        ("grouped Conv2d input", True, True, True, False, "grouped_conv_input_balanced resolver only; arbitrary flat input pruning rejected/repaired", "v9.7 tests", len([1 for name, axis in group_items if axis == "in" and _is_regular_grouped_conv(op_graph.module_of(name))])),
        ("residual Add", True, True, False, False, "Add is structural; branches are coupled, Add itself is not physically pruned", "proof csv", _artifact_count(op_graph, OP_ADD)),
        ("concat", True, True, False, False, "Cat is structural; branch offsets and downstream input are coupled", "proof csv", _artifact_count(op_graph, OP_CAT)),
        ("split/slice", True, False, False, True, "channel-dim split is currently protected/unsupported", "operator matrix", _artifact_count(op_graph, OP_SPLIT)),
        ("view/reshape/permute", True, True, False, False, "transparent when channel dimension can be followed", "trace", _artifact_count_raw(op_graph, "view") + _artifact_count_raw(op_graph, "reshape") + _artifact_count_raw(op_graph, "permute")),
        ("BEV pooling / warp / grid_sample", True, False, False, True, "protected geometry/spatial transform semantics", "operator matrix", _artifact_count(op_graph, OP_BEV_WARP)),
        ("detection heads", True, True, False, True, "head final output channel protected; input can be synced from upstream", "surface protection", len([n for n in op_graph.nodes.values() if n.is_det_head])),
    ]
    return [
        {
            "op_type": op_type,
            "trace_supported": trace_supported,
            "dependency_supported": dependency_supported,
            "physical_prune_supported": physical_supported,
            "currently_protected": protected,
            "reason": reason,
            "test_coverage": test_coverage,
            "full_model_artifact_coverage": coverage,
        }
        for op_type, trace_supported, dependency_supported, physical_supported, protected, reason, test_coverage, coverage in rows
    ]


def _artifact_count(op_graph: OpGraph, op_type: str) -> int:
    return sum(1 for node in op_graph.nodes.values() if node.op_type == op_type)


def _artifact_count_raw(op_graph: OpGraph, needle: str) -> int:
    needle = needle.lower()
    return sum(1 for node in op_graph.nodes.values() if needle in str(node.raw_type).lower())


def _groups_by_meta(groups: list[Any], key: str, value: str) -> list[Any]:
    out = []
    for group in groups:
        meta = getattr(group, "meta", {}) or {}
        if str(meta.get(key, "")) == value:
            out.append(group)
    return out


def _residual_concat_proof(op_graph: OpGraph, groups: list[Any], units: list[Any]) -> list[dict[str, Any]]:
    units_by_scope: dict[str, list[Any]] = defaultdict(list)
    for unit in units:
        units_by_scope[str(unit.scope_id)].append(unit)
    rows: list[dict[str, Any]] = []
    for node in op_graph.nodes.values():
        if node.op_type not in {OP_ADD, OP_CAT}:
            continue
        if node.op_type == OP_ADD:
            matched = []
            for group in groups:
                reasons = [str(v) for v in (getattr(group, "meta", {}) or {}).get("reasons", [])]
                if any(reason == f"add:{node.name}" for reason in reasons):
                    matched.append(group)
            for group in matched or [None]:
                scope_id = getattr(group, "group_id", "") if group is not None else ""
                proof_units = units_by_scope.get(scope_id, [])
                rows.append(
                    {
                        "node_name": node.name,
                        "node_type": "residual_add",
                        "num_input_branches": len(op_graph.incoming(node.name)),
                        "input_branches": [src for src, _idx in op_graph.incoming(node.name)],
                        "matched_scope_id": scope_id,
                        "same_channel_index_in_same_unit": bool(group is not None and proof_units),
                        "concat_offset_recorded": "",
                        "downstream_conv_input_offset_recorded": "",
                        "unit_ids_sample": [unit.unit_id for unit in proof_units[:8]],
                        "protected": bool(getattr(group, "protected", True)) if group is not None else True,
                        "protected_reason": getattr(group, "protected_reason", "missing_residual_group") if group is not None else "missing_residual_group",
                        "proof_pass": bool(group is not None and proof_units),
                    }
                )
        elif node.op_type == OP_CAT:
            matched = [group for group in groups if (getattr(group, "meta", {}) or {}).get("cat_node") == node.name]
            for group in matched or [None]:
                scope_id = getattr(group, "group_id", "") if group is not None else ""
                proof_units = units_by_scope.get(scope_id, [])
                offsets = sorted(
                    {
                        int(member.get("concat_offset", 0))
                        for unit in proof_units[: min(len(proof_units), 64)]
                        for member in unit.members
                        if member.get("dependency_type") == "concat_branch_offset"
                    }
                )
                downstream_has_identity = any(
                    member.get("dependency_type") == "concat_out_to_next_conv_in"
                    for unit in proof_units[: min(len(proof_units), 64)]
                    for member in unit.members
                )
                rows.append(
                    {
                        "node_name": node.name,
                        "node_type": "concat",
                        "num_input_branches": len(op_graph.incoming(node.name)),
                        "input_branches": [src for src, _idx in op_graph.incoming(node.name)],
                        "matched_scope_id": scope_id,
                        "same_channel_index_in_same_unit": bool(group is not None and proof_units),
                        "concat_offset_recorded": bool(offsets or (node.num_inputs or 0) <= 1),
                        "concat_offsets_sample": offsets[:16],
                        "downstream_conv_input_offset_recorded": bool(downstream_has_identity),
                        "unit_ids_sample": [unit.unit_id for unit in proof_units[:8]],
                        "protected": bool(getattr(group, "protected", True)) if group is not None else True,
                        "protected_reason": getattr(group, "protected_reason", "missing_concat_group") if group is not None else "missing_concat_group",
                        "proof_pass": bool(group is not None and proof_units and downstream_has_identity),
                    }
                )
    return rows


def _grouped_conv_proof(model: nn.Module, groups: list[Any], tp_diff_report: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    tp_names = set()
    if tp_diff_report:
        for sample in tp_diff_report.get("samples", []):
            if sample.get("status") == "success":
                tp_names.add(sample.get("root_module_name", ""))
    for name, module in model.named_modules():
        if not _is_regular_grouped_conv(module):
            continue
        containing = [group for group in groups if any(getattr(item, "name", "") == name for item in getattr(group, "items", []))]
        out_supported = any(
            not getattr(group, "protected", False)
            and any(getattr(item, "name", "") == name and getattr(item, "direction", "") == "out" for item in getattr(group, "items", []))
            for group in containing
        )
        in_supported = any(
            not getattr(group, "protected", False)
            and any(getattr(item, "name", "") == name and getattr(item, "direction", "") == "in" for item in getattr(group, "items", []))
            for group in containing
        )
        currently_protected = bool(containing) and all(getattr(group, "protected", False) for group in containing)
        failure_reasons = sorted({str(getattr(group, "protected_reason", "")) for group in containing if getattr(group, "protected", False)})
        rows.append(
            {
                "module_name": name,
                "groups": int(module.groups),
                "C_in": int(module.in_channels),
                "C_out": int(module.out_channels),
                "in_per_group": int(module.in_channels // module.groups),
                "out_per_group": int(module.out_channels // module.groups),
                "output_pruning_recipe_supported": bool(out_supported),
                "input_pruning_recipe_supported": bool(in_supported),
                "group_block_recipe_supported": bool(module.in_channels % module.groups == 0 and module.out_channels % module.groups == 0),
                "coarsening_recipe_supported": bool(module.groups > 1 and module.in_channels % module.groups == 0),
                "currently_protected": currently_protected,
                "tp_oracle_diff_available": name in tp_names,
                "failure_reason": ";".join(reason for reason in failure_reasons if reason),
            }
        )
    return rows


def _convtranspose_proof(model: nn.Module) -> list[dict[str, Any]]:
    rows = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.ConvTranspose2d):
            continue
        rows.append(
            {
                "module_name": name,
                "module_type": "ConvTranspose2d",
                "C_in": int(module.in_channels),
                "C_out": int(module.out_channels),
                "groups": int(module.groups),
                "physical_prune_supported": False,
                "currently_protected": True,
                "reason": "protected_convtranspose_deblock_or_fpn_output_contract",
            }
        )
    return rows


def _categorize_scopes(groups: list[Any], *, include_protected_structural: bool = False) -> dict[str, list[Any]]:
    categories: dict[str, list[Any]] = {
        "ordinary_conv": [],
        "residual": [],
        "concat": [],
        "grouped_output": [],
        "upstream_to_grouped_input": [],
    }
    for group in groups:
        meta = getattr(group, "meta", {}) or {}
        group_type = str(meta.get("group_type", ""))
        protected = bool(getattr(group, "protected", False))
        if protected and not (include_protected_structural and group_type in {"add", "cat"}):
            continue
        items = list(getattr(group, "items", []))
        if group_type == "add":
            categories["residual"].append(group)
        if group_type == "cat":
            categories["concat"].append(group)
        if any(_is_regular_grouped_conv(getattr(item, "module", None)) and getattr(item, "direction", "") == "out" for item in items):
            categories["grouped_output"].append(group)
        if any(_is_regular_grouped_conv(getattr(item, "module", None)) and getattr(item, "direction", "") == "in" for item in items):
            categories["upstream_to_grouped_input"].append(group)
        root = items[0] if items else None
        has_grouped_contract = any(_is_regular_grouped_conv(getattr(item, "module", None)) for item in items)
        root_name = str(getattr(root, "name", "") if root is not None else "").lower()
        fixed_or_head_contract = any(key in root_name for key in ("head", "deblock", "cls_head", "reg_head", "dir_head"))
        if (
            root is not None
            and isinstance(getattr(root, "module", None), nn.Conv2d)
            and int(root.module.groups) == 1
            and group_type == "plain"
            and not has_grouped_contract
            and not fixed_or_head_contract
        ):
            categories["ordinary_conv"].append(group)
    return categories


def _sample_scopes(scopes: list[Any], limit: int = 5) -> list[Any]:
    return sorted(scopes, key=lambda group: str(getattr(group, "group_id", "")))[:limit]


def _tensor_sum(value: Any) -> torch.Tensor | None:
    if torch.is_tensor(value):
        return value.float().sum()
    if isinstance(value, dict):
        total: torch.Tensor | None = None
        for child in value.values():
            child_sum = _tensor_sum(child)
            if child_sum is not None:
                total = child_sum if total is None else total + child_sum
        return total
    if isinstance(value, (list, tuple)):
        total = None
        for child in value:
            child_sum = _tensor_sum(child)
            if child_sum is not None:
                total = child_sum if total is None else total + child_sum
        return total
    return None


def _tp_sampling_diff(
    *,
    model: nn.Module,
    sample: Any,
    forward_fn: Callable[[nn.Module, Any], Any] | None,
    groups: list[Any],
    run_tp_oracle: bool,
) -> dict[str, Any]:
    categories = _categorize_scopes(groups, include_protected_structural=True)
    if not run_tp_oracle:
        return {
            "enabled": False,
            "status": "skipped_by_flag",
            "requested_per_category": 5,
            "categories": {key: {"num_candidates": len(value), "num_sampled": 0} for key, value in categories.items()},
            "samples": [],
            "aggregate": {"missing": 0, "extra": 0, "index_mismatch": 0},
        }

    tp_forward_fn = None
    if forward_fn is not None:
        def tp_forward_fn(m: nn.Module, x: Any) -> torch.Tensor:
            out = forward_fn(m, x)
            total = _tensor_sum(out)
            if total is None:
                raise RuntimeError("tp_oracle_forward_no_tensor_output")
            return total

    samples = []
    for category, scopes in categories.items():
        for scope in _sample_scopes(scopes, 5):
            root_item = next((item for item in getattr(scope, "items", []) if getattr(item, "direction", "") == "out"), None)
            if root_item is None and getattr(scope, "items", []):
                root_item = scope.items[0]
            if root_item is None:
                continue
            root_indices = _root_indices_for_oracle(scope, category)
            plan, _reports = _build_plan_from_scope_indices(scope, root_indices)
            diff = build_tp_oracle_diff(
                model,
                plan,
                root_module_name=root_item.name,
                root_axis="out" if root_item.direction not in {"in", "out"} else root_item.direction,
                root_indices=root_indices,
                example_inputs=sample,
                forward_fn=tp_forward_fn,
            )
            diff["category"] = category
            diff["scope_id"] = getattr(scope, "group_id", "")
            diff["project_scope_protected"] = bool(getattr(scope, "protected", False))
            diff["project_scope_protected_reason"] = str(getattr(scope, "protected_reason", ""))
            samples.append(diff)
    supported_samples = [
        sample for sample in samples
        if sample.get("available") and not sample.get("project_scope_protected")
    ]
    all_available = [sample for sample in samples if sample.get("available")]
    missing = sum(len(sample.get("missing_dependencies_in_project_plan", [])) for sample in supported_samples)
    extra = sum(len(sample.get("extra_project_plan_members", [])) for sample in supported_samples)
    mismatch = sum(len(sample.get("idx_transform_mismatches", [])) for sample in supported_samples)
    all_missing = sum(len(sample.get("missing_dependencies_in_project_plan", [])) for sample in all_available)
    all_extra = sum(len(sample.get("extra_project_plan_members", [])) for sample in all_available)
    all_mismatch = sum(len(sample.get("idx_transform_mismatches", [])) for sample in all_available)
    return {
        "enabled": True,
        "requested_per_category": 5,
        "categories": {
            key: {
                "num_candidates": len(value),
                "num_sampled": sum(1 for sample_row in samples if sample_row.get("category") == key),
                "shortfall_reason": "" if len(value) >= 5 else "not_enough_supported_root_actions_in_current_trace",
            }
            for key, value in categories.items()
        },
        "samples": samples,
        "aggregate": {"missing": missing, "extra": extra, "index_mismatch": mismatch, "scope": "supported_unprotected_samples"},
        "aggregate_all_available_samples": {"missing": all_missing, "extra": all_extra, "index_mismatch": all_mismatch},
    }


def _root_indices_for_oracle(scope: Any, category: str) -> list[int]:
    num = int(getattr(scope, "num_channels", 0) or 0)
    if num <= 0:
        return []
    if category == "upstream_to_grouped_input":
        for item in getattr(scope, "items", []):
            module = getattr(item, "module", None)
            if _is_regular_grouped_conv(module) and getattr(item, "direction", "") == "in":
                groups = int(module.groups)
                per = int(module.in_channels // groups)
                return [g * per for g in range(groups) if g * per < num]
    return [0]


def _mask0_dryrun_report(
    *,
    model: nn.Module,
    op_graph: OpGraph,
    groups: list[Any],
    units: list[Any],
    group_conv_align: int,
) -> dict[str, Any]:
    scopes_by_id = {str(group.group_id): group for group in groups}
    units_by_id = {unit.unit_id: unit for unit in units}
    prunable = [unit for unit in units if not unit.protected and not unit.unsupported_reason]
    residual = [unit for unit in prunable if unit.constraints.get("has_residual")]
    concat = [unit for unit in prunable if unit.constraints.get("has_concat")]
    grouped = [unit for unit in prunable if unit.constraints.get("has_grouped_conv")]
    cases = {
        "light": _pick_legal_unit_bundle(prunable, max_scopes=8),
        "medium": _pick_legal_unit_bundle(prunable, max_scopes=32),
        "heavy": _pick_legal_unit_bundle(prunable, max_scopes=96),
        "residual-heavy": _pick_legal_unit_bundle(residual, max_scopes=32),
        "concat-heavy": _pick_legal_unit_bundle(concat, max_scopes=32),
        "grouped-conv-heavy": _pick_legal_unit_bundle(grouped, max_scopes=32),
    }
    rows = []
    for name, selected in cases.items():
        if not selected:
            rows.append(
                {
                    "case": name,
                    "num_selected_units": 0,
                    "selected_unit_ids": [],
                    "num_plan_requests": 0,
                    "legal": True,
                    "num_issues": 0,
                    "first_issue": {},
                    "failing_unit_id": "",
                    "skipped_reason": "no_supported_units_in_category",
                    "missing_dependency": False,
                    "illegal_grouped_conv_shape": False,
                    "concat_offset_mismatch": False,
                    "residual_mismatch": False,
                    "unsupported_op": False,
                    "grouped_input_repair_reports": [],
                }
            )
            continue
        plan, grouped_reports = _build_plan_from_units(scopes_by_id, [unit.unit_id for unit in selected], units_by_id)
        sim = GlobalPlanShapeSimulator(
            model,
            plan,
            op_graph=op_graph,
            group_conv_align=group_conv_align,
            allow_convtranspose=False,
            allow_fixed_shape_pruning=False,
        )
        sim_report = sim.simulate()
        first_issue = (sim_report.get("issues") or [{}])[0] if not sim_report.get("legal", False) else {}
        failing_unit_id = _find_failing_unit_id(selected, first_issue)
        rows.append(
            {
                "case": name,
                "num_selected_units": len(selected),
                "selected_unit_ids": [unit.unit_id for unit in selected],
                "num_plan_requests": len(plan.requests()),
                "legal": bool(sim_report.get("legal", False)),
                "num_issues": int(sim_report.get("num_issues", 0) or 0),
                "first_issue": first_issue,
                "failing_unit_id": failing_unit_id,
                "skipped_reason": "",
                "missing_dependency": str(first_issue.get("issue", "")) in {"downstream_input_channel_mismatch", "norm_channel_mismatch"},
                "illegal_grouped_conv_shape": "grouped" in str(first_issue.get("issue", "")),
                "concat_offset_mismatch": str(first_issue.get("issue", "")) == "concat_downstream_input_mismatch",
                "residual_mismatch": str(first_issue.get("issue", "")) == "residual_add_branch_channel_mismatch",
                "unsupported_op": str(first_issue.get("issue", "")) in {"fixed_shape_contract_pruned", "convtranspose_deblock_contract_requires_explicit_support"},
                "grouped_input_repair_reports": grouped_reports,
            }
        )
    return {
        "supported_surface_only": True,
        "num_prunable_units": len(prunable),
        "cases": rows,
    }


def _find_failing_unit_id(selected: list[Any], first_issue: dict[str, Any]) -> str:
    if not first_issue:
        return ""
    module_name = str(first_issue.get("module_name", ""))
    if not module_name:
        return selected[0].unit_id if selected else ""
    for unit in selected:
        for member in getattr(unit, "members", []):
            if str(member.get("module_name", "")) == module_name:
                return str(unit.unit_id)
    return selected[0].unit_id if selected else ""


def _pick_legal_unit_bundle(units: list[Any], max_scopes: int) -> list[Any]:
    if max_scopes <= 0:
        return []
    by_scope: dict[str, list[Any]] = defaultdict(list)
    for unit in units:
        by_scope[str(unit.scope_id)].append(unit)
    selected = []
    for _scope_id, scope_units in sorted(by_scope.items()):
        if len(selected) >= max_scopes:
            break
        selected.extend(_first_legal_scope_units(scope_units))
    return selected


def _first_legal_scope_units(scope_units: list[Any]) -> list[Any]:
    if not scope_units:
        return []
    scope_units = sorted(scope_units, key=lambda unit: int(unit.root_idx))
    first = scope_units[0]
    grouped = dict(getattr(first, "grouped_conv_info", {}) or {})
    constraints = dict(getattr(first, "constraints", {}) or {})
    if constraints.get("has_grouped_conv") and grouped.get("groups") and grouped.get("per_group"):
        groups = int(grouped["groups"])
        per = int(grouped["per_group"])
        # Pick one local position from every group. This is the smallest
        # group-balanced bundle for grouped Conv2d input/output contracts.
        wanted = {g * per for g in range(groups)}
        bundle = [unit for unit in scope_units if int(unit.root_idx) in wanted]
        if len(bundle) == groups:
            return bundle
        return []
    return [first]


def run_audit_for_model(
    model: nn.Module,
    sample: Any,
    output_dir: str | Path,
    *,
    forward_fn: Callable[[nn.Module, Any], Any] | None = None,
    protected_layers: list[str] | None = None,
    group_conv_policy: str = "A",
    align: int = 4,
    group_conv_align: int = 8,
    run_tp_oracle: bool = True,
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.eval()
    policy_key = str(group_conv_policy or "A").upper()
    grouped_mode = POLICY_TO_GROUPED_MODE.get(policy_key, "flat_output_groups_fixed")

    trace = trace_model(model, sample, forward_fn=forward_fn)
    op_graph = build_op_graph(trace, model, protected_layers=protected_layers or [])
    write_json(out / "trace_path_0.json", trace)
    write_json(out / "op_graph_path_0.json", op_graph.to_dict())
    evidence_files = [{"path_id": "path_0", "trace_graph": "trace_path_0.json", "op_graph": "op_graph_path_0.json"}]
    write_json(out / "trace_graph_coverage_report.json", _trace_coverage_report(model=model, trace=trace, op_graph=op_graph, evidence_files=evidence_files))

    groups = GroupBuilder(
        op_graph,
        align=align,
        grouped_conv_mode=grouped_mode,
        protect_residual_add=False,
    ).build()
    surface = apply_full_model_prunable_surface(groups, group_conv_policy=policy_key, total_model_params=sum(p.numel() for p in model.parameters()))
    write_json(out / "full_model_surface_after_unprotect_grouped_input.json", surface)

    units = []
    for group in groups:
        scores = torch.arange(int(group.num_channels), dtype=torch.float32)
        units.extend(expand_coupled_channel_units(group, scores, importance_mode="audit_rank"))
    _enrich_units_with_graph(units, op_graph)
    unit_rows = coupled_channel_unit_rows(units)
    write_json(out / "coupled_channel_units_full_model.json", unit_rows)
    write_json(out / "coupled_channel_unit_completeness_report.json", _unit_completeness_report(units, groups))

    matrix_rows = _operator_matrix(op_graph, groups)
    write_csv(
        out / "operator_dependency_coverage_matrix.csv",
        matrix_rows,
        ["op_type", "trace_supported", "dependency_supported", "physical_prune_supported", "currently_protected", "reason", "test_coverage", "full_model_artifact_coverage"],
    )
    write_csv(
        out / "residual_concat_full_model_proof.csv",
        _residual_concat_proof(op_graph, groups, units),
        [
            "node_name",
            "node_type",
            "num_input_branches",
            "input_branches",
            "matched_scope_id",
            "same_channel_index_in_same_unit",
            "concat_offset_recorded",
            "concat_offsets_sample",
            "downstream_conv_input_offset_recorded",
            "unit_ids_sample",
            "protected",
            "protected_reason",
            "proof_pass",
        ],
    )

    tp_report = _tp_sampling_diff(model=model, sample=sample, forward_fn=forward_fn, groups=groups, run_tp_oracle=run_tp_oracle)
    write_json(out / "tp_oracle_sampling_diff_report.json", tp_report)
    write_csv(
        out / "grouped_conv_dependency_proof.csv",
        _grouped_conv_proof(model, groups, tp_report),
        [
            "module_name",
            "groups",
            "C_in",
            "C_out",
            "in_per_group",
            "out_per_group",
            "output_pruning_recipe_supported",
            "input_pruning_recipe_supported",
            "group_block_recipe_supported",
            "coarsening_recipe_supported",
            "currently_protected",
            "tp_oracle_diff_available",
            "failure_reason",
        ],
    )
    write_csv(
        out / "convtranspose_dependency_proof.csv",
        _convtranspose_proof(model),
        ["module_name", "module_type", "C_in", "C_out", "groups", "physical_prune_supported", "currently_protected", "reason"],
    )
    mask0_report = _mask0_dryrun_report(model=model, op_graph=op_graph, groups=groups, units=units, group_conv_align=group_conv_align)
    write_json(out / "mask0_physical_removal_dryrun_report.json", mask0_report)

    acceptance = {
        "dynamic_path_coverage_explicit": True,
        "coupled_channel_unit_member_index_proof_available": bool(units),
        "residual_concat_full_model_proof_written": True,
        "grouped_conv_proof_written": True,
        "convtranspose_explicitly_protected": True,
        "tp_oracle_sampling_missing_zero_for_supported_successes": int(tp_report.get("aggregate", {}).get("missing", 0) or 0) == 0,
        "mask0_dryrun_legal_for_all_cases": all(bool(row.get("legal", False)) for row in mask0_report.get("cases", []) if row.get("num_selected_units", 0) > 0),
        "unsupported_surfaces_excluded_from_prunable_surface": True,
        "ready_for_mask0_physical_removal": False,
        "blocking_reason": "",
    }
    if not acceptance["mask0_dryrun_legal_for_all_cases"]:
        acceptance["blocking_reason"] = "mask0_shape_simulator_has_failures"
    elif not acceptance["tp_oracle_sampling_missing_zero_for_supported_successes"]:
        acceptance["blocking_reason"] = "tp_oracle_sampling_diff_has_missing_dependencies"
    elif not any(unit.is_minimal_proven for unit in units):
        acceptance["blocking_reason"] = "no_minimal_unit_proofs"
    else:
        acceptance["ready_for_mask0_physical_removal"] = bool(
            acceptance["tp_oracle_sampling_missing_zero_for_supported_successes"]
            and acceptance["mask0_dryrun_legal_for_all_cases"]
        )
    write_json(out / "v98_acceptance_summary.json", acceptance)
    return {
        "output_dir": str(out),
        "num_groups": len(groups),
        "num_units": len(units),
        "num_trace_nodes": len(op_graph.nodes),
        "num_trace_edges": len(op_graph.edges),
        "ready_for_mask0_physical_removal": acceptance["ready_for_mask0_physical_removal"],
        "blocking_reason": acceptance["blocking_reason"],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="/home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_pyramid/net_epoch_bestval_at17.pth")
    parser.add_argument("--model-config", default="/home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_pyramid/config.yaml")
    parser.add_argument("--heal-root", default="/home/lixingfeng/UniAD_examine/HEAL")
    parser.add_argument("--group-conv-policy", default="A", choices=["A", "B", "C", "D"])
    parser.add_argument("--align", type=int, default=4)
    parser.add_argument("--group-conv-align", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", default=OUT_DEFAULT)
    parser.add_argument("--run-tp-oracle", type=str2bool, default=True)
    return parser.parse_args(argv)


def run_real_lidar_pyramid_audit(args: argparse.Namespace) -> dict[str, Any]:
    from heal_compress.utils.model_utils import resolve_device
    from heal_compress.pruning.model_io import build_protected_layers, load_heal_model, setup_logger

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(out)
    device = torch.device(resolve_device(args.device))
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model, adapter = load_heal_model(args, device, logger)
    sample = adapter.build_synthetic_batch(model)
    protected_layers = build_protected_layers(
        model,
        adapter_protected=adapter.get_protected_layers(model),
        extra_prefixes=[],
    )
    return run_audit_for_model(
        model,
        sample,
        out,
        forward_fn=adapter.forward_for_task,
        protected_layers=protected_layers,
        group_conv_policy=args.group_conv_policy,
        align=args.align,
        group_conv_align=args.group_conv_align,
        run_tp_oracle=args.run_tp_oracle,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    try:
        result = run_real_lidar_pyramid_audit(args)
        print(json.dumps({"success": True, **result}, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:  # noqa: BLE001
        payload = {"success": False, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
        write_json(out / "audit_failure.json", payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
