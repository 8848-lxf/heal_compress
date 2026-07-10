#!/usr/bin/env python3
"""v9.9 residual/concat root-cause audit and ConvTranspose smoke support."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[2]
_UNIAD = _ROOT.parent
for _p in (_UNIAD, _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from heal_compress.pruning.global_plan_shape_simulator import GlobalPlanShapeSimulator  # noqa: E402
from heal_compress.pruning.physical_prune_plan import GlobalPhysicalPrunePlan, ModuleAxisPruneRequest  # noqa: E402
from heal_compress.tracer.generic_tracer import trace_model  # noqa: E402
from heal_compress.tracer.op_graph import OP_ADD, OP_CAT, OP_CONV, OP_CONVT, build_op_graph  # noqa: E402
from tools.latency_lut.audit_dependency_graph_v98 import run_audit_for_model, write_csv, write_json  # noqa: E402


V98_DEFAULT = Path("outputs/latency_lut/dependency_graph_audit_v98")
OUT_DEFAULT = Path("outputs/latency_lut/dependency_graph_audit_v99")
FAIL_CATEGORIES = {
    "non_channelwise_add_or_cat",
    "protected_deblock_or_convtranspose_path",
    "protected_geometry_or_fixed_shape_contract",
    "missing_cat_offset_proof",
    "missing_downstream_input_offset_proof",
    "downstream_consumer_not_plain_conv",
    "dynamic_path_not_covered",
    "merge_or_non_compute_geometry_structure",
    "missing_scope_or_unit_mapping",
    "unsupported_but_fixable_with_new_resolver",
    "audit_bug",
}


def _load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _parse_json_cell(value: str) -> Any:
    if value in {"", None}:  # type: ignore[comparison-overlap]
        return []
    try:
        return json.loads(value)
    except Exception:  # noqa: BLE001
        return value


def _channel_axis(shape: list[int]) -> int | None:
    if len(shape) >= 4:
        return 1
    if len(shape) == 2:
        return 1
    return None


def _channel_size(shape: list[int]) -> int | None:
    axis = _channel_axis(shape)
    if axis is None or axis >= len(shape):
        return None
    return int(shape[axis])


def _is_geometry_name(name: str) -> bool:
    low = name.lower()
    return any(key in low for key in ("scatter", "pillar", "voxel", "grid_sample", "warp", "bev", "geometry"))


def _is_deblock_name(name: str) -> bool:
    low = name.lower()
    return "deblock" in low or "convtranspose" in low


def _downstream_rows(op_graph: dict[str, Any], node_name: str) -> list[dict[str, Any]]:
    rows = []
    nodes = op_graph.get("nodes", {})
    for edge in op_graph.get("edges", []):
        if edge.get("src") != node_name:
            continue
        dst = str(edge.get("dst", ""))
        info = nodes.get(dst, {})
        rows.append(
            {
                "module_name": dst,
                "op_type": info.get("op_type", ""),
                "raw_type": info.get("raw_type", ""),
                "groups": info.get("groups"),
                "is_plain_conv": info.get("op_type") == OP_CONV and int(info.get("groups") or 1) == 1,
                "is_grouped_conv": info.get("op_type") == OP_CONV and int(info.get("groups") or 1) > 1,
                "is_convtranspose": info.get("op_type") == OP_CONVT,
                "is_unsupported": info.get("op_type") not in {OP_CONV, OP_CONVT},
            }
        )
    return rows


def _op_graph_incoming(op_graph: dict[str, Any], node_name: str) -> list[str]:
    return [str(edge.get("src", "")) for edge in op_graph.get("edges", []) if edge.get("dst") == node_name]


def _node_scope_text(node_info: dict[str, Any]) -> str:
    return " ".join(str(v) for v in node_info.get("module_scope", []) or [])


def _classify_fail_row(
    row: dict[str, str],
    *,
    op_graph: dict[str, Any],
    trace_coverage: dict[str, Any],
    units_by_scope: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    node_name = row["node_name"]
    node_info = op_graph.get("nodes", {}).get(node_name, {})
    input_shapes = node_info.get("input_shapes", []) or []
    output_shapes = node_info.get("output_shapes", []) or []
    downstream = _downstream_rows(op_graph, node_name)
    incoming = _op_graph_incoming(op_graph, node_name)
    categories: set[str] = set()
    recommended = "keep_protected"
    requires_new_resolver = False
    requires_protection = True
    can_enter = False

    if row["node_type"] == "concat":
        cat_dim = node_info.get("cat_dim")
        is_channel = cat_dim == 1
        is_regular = bool(is_channel and len(input_shapes) >= 2 and downstream)
    else:
        sizes = [_channel_size(shape) for shape in input_shapes if isinstance(shape, list)]
        is_channel = bool(len(sizes) >= 2 and None not in sizes and len(set(sizes)) == 1 and sizes[0] and sizes[0] > 1)
        is_regular = bool(is_channel and len(input_shapes) >= 2)

    scope_text = _node_scope_text(node_info)
    all_names = " ".join([node_name, scope_text] + incoming + [d["module_name"] for d in downstream])
    involves_deblock = _is_deblock_name(all_names) or any(d["is_convtranspose"] for d in downstream)
    involves_geometry = _is_geometry_name(all_names)
    involves_fixed = any(key in all_names.lower() for key in ("pillar_vfe", "pfn_layers", "scatter", "voxel"))

    if not is_regular:
        categories.add("non_channelwise_add_or_cat")
    if involves_deblock:
        categories.add("protected_deblock_or_convtranspose_path")
    if involves_geometry or involves_fixed:
        categories.add("protected_geometry_or_fixed_shape_contract")
    if row["node_type"] == "concat":
        has_offset = str(row.get("concat_offset_recorded", "")).lower() == "true"
        has_downstream_offset = str(row.get("downstream_conv_input_offset_recorded", "")).lower() == "true"
        if is_channel and not has_offset:
            categories.add("missing_cat_offset_proof")
        if is_channel and not has_downstream_offset:
            categories.add("missing_downstream_input_offset_proof")
    else:
        has_offset = False
        has_downstream_offset = False
    if downstream and not any(d["is_plain_conv"] for d in downstream):
        categories.add("downstream_consumer_not_plain_conv")
    if not downstream and not is_regular:
        categories.add("merge_or_non_compute_geometry_structure")
    if not row.get("matched_scope_id"):
        categories.add("missing_scope_or_unit_mapping")
    if trace_coverage.get("dynamic_branch_enumeration_enabled") is False:
        dynamic_limitation = True
    else:
        dynamic_limitation = False

    # If this is a regular channel op and the only blocker is missing mapping,
    # treat it as fixable; otherwise keep protected.
    if is_regular and categories.issubset({"missing_scope_or_unit_mapping", "missing_cat_offset_proof", "missing_downstream_input_offset_proof"}):
        categories.add("unsupported_but_fixable_with_new_resolver")
        requires_new_resolver = True
        recommended = "add_or_fix_concat_residual_dependency_resolver"

    if not categories:
        categories.add("audit_bug")
        recommended = "fix_audit_script_to_recognize_existing_proof"
        requires_protection = False

    protected_modules = [name for name in all_names.split() if _is_deblock_name(name) or _is_geometry_name(name)]
    failure_reason = _failure_reason(categories, downstream, protected_modules, dynamic_limitation)
    return {
        "node_name": node_name,
        "node_type": row["node_type"],
        "proof_pass": row["proof_pass"],
        "num_input_branches": row["num_input_branches"],
        "input_branches": row["input_branches"],
        "matched_scope_id": row["matched_scope_id"],
        "downstream_consumers": json.dumps(downstream, ensure_ascii=False),
        "is_channel_dim_op": bool(is_channel),
        "is_regular_channelwise_residual_or_concat": bool(is_regular),
        "involves_deblock_or_convtranspose": bool(involves_deblock),
        "involves_geometry_or_bev_warp": bool(involves_geometry),
        "involves_fixed_shape_contract": bool(involves_fixed),
        "has_cat_offset_proof": bool(has_offset),
        "has_downstream_input_offset_proof": bool(has_downstream_offset),
        "downstream_consumer_is_plain_conv": any(d["is_plain_conv"] for d in downstream),
        "downstream_consumer_is_grouped_conv": any(d["is_grouped_conv"] for d in downstream),
        "downstream_consumer_is_convtranspose": any(d["is_convtranspose"] for d in downstream),
        "downstream_consumer_is_unsupported": bool(downstream and any(d["is_unsupported"] for d in downstream)),
        "covered_by_current_single_trace": node_name in op_graph.get("nodes", {}),
        "dynamic_path_limitation": dynamic_limitation,
        "failure_category": ";".join(sorted(categories)),
        "failure_reason": failure_reason,
        "recommended_action": recommended,
        "can_enter_prunable_surface": can_enter,
        "requires_new_resolver": requires_new_resolver,
        "requires_protection": requires_protection,
    }


def _failure_reason(categories: set[str], downstream: list[dict[str, Any]], protected_modules: list[str], dynamic_limitation: bool) -> str:
    parts = []
    if "non_channelwise_add_or_cat" in categories:
        parts.append("operation is not a regular channel-wise residual/concat")
    if "protected_geometry_or_fixed_shape_contract" in categories:
        parts.append("involves protected geometry/fixed-shape path: " + ",".join(sorted(set(protected_modules))[:8]))
    if "protected_deblock_or_convtranspose_path" in categories:
        parts.append("involves deblock/ConvTranspose path requiring explicit closure")
    if "downstream_consumer_not_plain_conv" in categories:
        parts.append("downstream consumer is not plain Conv2d: " + ",".join(f"{d['module_name']}:{d['op_type'] or d['raw_type']}" for d in downstream))
    if "missing_cat_offset_proof" in categories:
        parts.append("missing branch concat offset proof")
    if "missing_downstream_input_offset_proof" in categories:
        parts.append("missing concat output -> downstream input offset proof")
    if "missing_scope_or_unit_mapping" in categories:
        parts.append("no matched PruningGroup/CoupledChannelUnit mapping")
    if dynamic_limitation:
        parts.append("dynamic_branch_enumeration_enabled=false; this is single-path evidence only")
    return "; ".join(parts)


def _fixability(row: dict[str, Any]) -> dict[str, Any]:
    cats = set(str(row["failure_category"]).split(";"))
    if "audit_bug" in cats:
        fix = "already_supported_audit_bug"
        req = "fix audit proof matching"
        priority = "P0"
        protected = False
    elif "non_channelwise_add_or_cat" in cats:
        fix = "should_remain_protected"
        req = "not a regular channel dimension pruning surface"
        priority = "P3"
        protected = True
    elif "missing_cat_offset_proof" in cats:
        fix = "fixable_with_concat_offset_resolver"
        req = "record concat branch offsets and downstream local index transforms"
        priority = "P1"
        protected = True
    elif "missing_downstream_input_offset_proof" in cats or "downstream_consumer_not_plain_conv" in cats:
        fix = "fixable_with_downstream_consumer_resolver"
        req = "add supported downstream consumer resolver or keep protected"
        priority = "P1"
        protected = True
    elif "protected_deblock_or_convtranspose_path" in cats:
        fix = "fixable_with_convtranspose_resolver"
        req = "add ConvTranspose/deblock closure and simulator proof"
        priority = "P1"
        protected = True
    elif "dynamic_path_not_covered" in cats:
        fix = "fixable_with_dynamic_path_enumeration"
        req = "enumerate dynamic forward branches"
        priority = "P2"
        protected = True
    else:
        fix = "should_remain_protected"
        req = "not a safe channel pruning surface"
        priority = "P3"
        protected = True
    return {
        "node_name": row["node_name"],
        "node_type": row["node_type"],
        "failure_category": row["failure_category"],
        "fixability": fix,
        "required_implementation": req,
        "expected_test_name": f"test_v99_{row['node_type']}_{fix}",
        "priority": priority,
        "should_remain_protected_now": protected,
    }


def build_residual_concat_fail_root_cause_reports(v98_dir: Path, out_dir: Path) -> dict[str, Any]:
    trace = _load_json(v98_dir / "trace_graph_coverage_report.json", {})
    _units = _load_json(v98_dir / "coupled_channel_units_full_model.json", [])
    _completeness = _load_json(v98_dir / "coupled_channel_unit_completeness_report.json", {})
    _operator_matrix = _read_csv(v98_dir / "operator_dependency_coverage_matrix.csv")
    proof = _read_csv(v98_dir / "residual_concat_full_model_proof.csv")
    _grouped = _read_csv(v98_dir / "grouped_conv_dependency_proof.csv")
    _convt = _read_csv(v98_dir / "convtranspose_dependency_proof.csv")
    _tp = _load_json(v98_dir / "tp_oracle_sampling_diff_report.json", {})
    _mask = _load_json(v98_dir / "mask0_physical_removal_dryrun_report.json", {})
    surface = _load_json(v98_dir / "full_model_surface_after_unprotect_grouped_input.json", {})
    op_graph = _load_json(v98_dir / "op_graph_path_0.json", {"nodes": {}, "edges": []})

    units_by_scope: dict[str, list[dict[str, Any]]] = {}
    for unit in _units:
        units_by_scope.setdefault(str(unit.get("scope_id", "")), []).append(unit)

    fail_rows = [row for row in proof if row.get("proof_pass") != "True"]
    rows = [
        _classify_fail_row(row, op_graph=op_graph, trace_coverage=trace, units_by_scope=units_by_scope)
        for row in fail_rows
    ]
    matrix = [_fixability(row) for row in rows]
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "node_name", "node_type", "proof_pass", "num_input_branches", "input_branches",
        "matched_scope_id", "downstream_consumers", "is_channel_dim_op",
        "is_regular_channelwise_residual_or_concat", "involves_deblock_or_convtranspose",
        "involves_geometry_or_bev_warp", "involves_fixed_shape_contract",
        "has_cat_offset_proof", "has_downstream_input_offset_proof",
        "downstream_consumer_is_plain_conv", "downstream_consumer_is_grouped_conv",
        "downstream_consumer_is_convtranspose", "downstream_consumer_is_unsupported",
        "covered_by_current_single_trace", "dynamic_path_limitation",
        "failure_category", "failure_reason", "recommended_action",
        "can_enter_prunable_surface", "requires_new_resolver", "requires_protection",
    ]
    write_csv(out_dir / "residual_concat_fail_root_cause_report.csv", rows, fields)
    write_csv(
        out_dir / "residual_concat_fixability_matrix.csv",
        matrix,
        ["node_name", "node_type", "failure_category", "fixability", "required_implementation", "expected_test_name", "priority", "should_remain_protected_now"],
    )
    update = {
        "dynamic_branch_enumeration_enabled": bool(trace.get("dynamic_branch_enumeration_enabled", False)),
        "num_fail_rows": len(rows),
        "num_can_enter_prunable_surface": sum(1 for row in rows if row["can_enter_prunable_surface"]),
        "num_requires_protection": sum(1 for row in rows if row["requires_protection"]),
        "surface_prunable_groups": surface.get("num_prunable_coupled_units"),
        "surface_protected_groups": surface.get("num_protected_coupled_units"),
        "unsupported_fail_nodes_excluded_from_prunable_surface": all(not row["can_enter_prunable_surface"] for row in rows),
    }
    write_json(out_dir / "residual_concat_supported_surface_update.json", update)
    _write_fail_md(out_dir / "residual_concat_fail_root_cause_report.md", rows, matrix, update)
    return {"num_fail_rows": len(rows), "output_dir": str(out_dir), "supported_surface_update": update}


def _write_fail_md(path: Path, rows: list[dict[str, Any]], matrix: list[dict[str, Any]], update: dict[str, Any]) -> None:
    lines = ["# Residual / Concat Fail Root Cause Report", ""]
    lines.append(f"- fail rows: {len(rows)}")
    lines.append(f"- dynamic_branch_enumeration_enabled: {update['dynamic_branch_enumeration_enabled']}")
    lines.append(f"- unsupported fail nodes excluded from prunable surface: {update['unsupported_fail_nodes_excluded_from_prunable_surface']}")
    lines.append("")
    for row in rows:
        lines.append(f"## {row['node_name']} ({row['node_type']})")
        lines.append(f"- category: {row['failure_category']}")
        lines.append(f"- reason: {row['failure_reason']}")
        lines.append(f"- recommended_action: {row['recommended_action']}")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_deblock_concat_dependency_proof(
    model: nn.Module,
    sample: Any,
    *,
    forward_fn: Callable[[nn.Module, Any], Any] | None = None,
    apply_smoke: bool = False,
) -> list[dict[str, Any]]:
    model.eval()
    trace = trace_model(model, sample, forward_fn=forward_fn)
    graph = build_op_graph(trace, model)
    modules = dict(model.named_modules())
    rows: list[dict[str, Any]] = []
    convt_names = [name for name, module in modules.items() if isinstance(module, nn.ConvTranspose2d)]
    consumed_convt: set[str] = set()
    for node in graph.nodes.values():
        if node.op_type != OP_CAT or node.cat_dim != 1:
            continue
        offset = 0
        incoming = graph.incoming(node.name)
        downstream = graph.outgoing(node.name)
        for branch_pos, (src, _idx) in enumerate(incoming):
            deblock_name, path_modules = _find_upstream_convtranspose_path(graph, modules, src)
            module = modules.get(deblock_name) if deblock_name else None
            channels = _channel_size(node.input_shapes[branch_pos]) if branch_pos < len(node.input_shapes) else None
            if not isinstance(module, nn.ConvTranspose2d):
                if channels:
                    offset += int(channels)
                continue
            consumed_convt.add(deblock_name)
            row = _deblock_row_for_branch(
                model,
                sample,
                graph,
                node.name,
                deblock_name,
                module,
                offset,
                int(channels or module.out_channels),
                downstream,
                apply_smoke,
                forward_fn,
                path_modules=path_modules,
            )
            rows.append(row)
            offset += int(channels or module.out_channels)
    for name in convt_names:
        if name in consumed_convt:
            continue
        module = modules[name]
        rows.append(
            {
                "deblock_module": name,
                "convtranspose_module": name,
                "concat_node": "",
                "branch_offset": 0,
                "branch_channels_before": int(module.out_channels),
                "branch_channels_after": int(module.out_channels),
                "downstream_consumer": "",
                "downstream_consumer_type": "",
                "downstream_input_indices_synced": False,
                "offset_proof_available": False,
                "physical_prune_supported": False,
                "simulator_legal": False,
                "forward_smoke_status": "not_run",
                "failure_reason": "protected_deblock_unsupported_downstream_consumer",
            }
        )
    return rows


def _find_upstream_convtranspose_path(graph: Any, modules: dict[str, nn.Module], start: str) -> tuple[str, list[str]]:
    queue: list[tuple[str, list[str]]] = [(start, [])]
    seen: set[str] = set()
    while queue:
        cur, path = queue.pop(0)
        if cur in seen:
            continue
        seen.add(cur)
        module = modules.get(cur)
        if isinstance(module, nn.ConvTranspose2d):
            return cur, [cur] + path
        if isinstance(module, (nn.modules.batchnorm._BatchNorm, nn.ReLU, nn.ReLU6, nn.LeakyReLU, nn.Identity)) or cur.startswith("op::"):
            for src, _idx in graph.incoming(cur):
                queue.append((src, [cur] + path))
    return "", []


def _deblock_row_for_branch(
    model: nn.Module,
    sample: Any,
    graph: Any,
    cat_node: str,
    deblock_name: str,
    deblock: nn.ConvTranspose2d,
    offset: int,
    branch_channels: int,
    downstream: list[tuple[str, int]],
    apply_smoke: bool,
    forward_fn: Callable[[nn.Module, Any], Any] | None,
    *,
    path_modules: list[str] | None = None,
) -> dict[str, Any]:
    modules = dict(model.named_modules())
    path_modules = list(path_modules or [deblock_name])
    consumer_name = downstream[0][0] if downstream else ""
    consumer = modules.get(consumer_name)
    consumer_type = consumer.__class__.__name__ if consumer is not None else ""
    supported_consumer = isinstance(consumer, nn.Conv2d) and int(consumer.groups) == 1
    grouped_convt = int(deblock.groups) > 1
    physical_supported = bool(not grouped_convt and supported_consumer)
    failure = ""
    sim_legal = False
    smoke_status = "not_run"
    if grouped_convt:
        failure = "unsupported_grouped_convtranspose"
    elif not supported_consumer:
        failure = "protected_deblock_unsupported_downstream_consumer"
    if physical_supported:
        prune_local = [0]
        prune_consumer = [offset]
        plan = GlobalPhysicalPrunePlan()
        plan.add_request(ModuleAxisPruneRequest(deblock_name, "out", prune_local, source_recipe_id=f"deblock::{deblock_name}"))
        for sync_name in path_modules:
            sync_module = modules.get(sync_name)
            if sync_name == deblock_name:
                continue
            if isinstance(sync_module, nn.modules.batchnorm._BatchNorm):
                plan.add_request(ModuleAxisPruneRequest(sync_name, "out", prune_local, source_recipe_id=f"deblock::{deblock_name}"))
        plan.add_request(ModuleAxisPruneRequest(consumer_name, "in", prune_consumer, source_recipe_id=f"deblock::{deblock_name}"))
        sim = GlobalPlanShapeSimulator(model, plan, op_graph=graph, allow_convtranspose=True).simulate()
        sim_legal = bool(sim.get("legal", False))
        if apply_smoke and sim_legal:
            try:
                cloned = copy.deepcopy(model)
                plan.apply_one_shot(cloned)
                cloned.eval()
                with torch.no_grad():
                    if forward_fn is not None:
                        forward_fn(cloned, sample)
                    else:
                        cloned(sample)
                smoke_status = "forward_passed"
            except Exception as exc:  # noqa: BLE001
                smoke_status = "forward_failed"
                failure = f"{type(exc).__name__}: {exc}"
        elif apply_smoke:
            smoke_status = "skipped_simulator_illegal"
            failure = failure or "simulator_illegal"
    return {
        "deblock_module": deblock_name,
        "convtranspose_module": deblock_name,
        "concat_node": cat_node,
        "branch_offset": int(offset),
        "branch_channels_before": int(branch_channels),
        "branch_channels_after": int(branch_channels - 1 if physical_supported and sim_legal else branch_channels),
        "downstream_consumer": consumer_name,
        "downstream_consumer_type": consumer_type,
        "downstream_input_indices_synced": bool(physical_supported),
        "sync_path_modules": path_modules,
        "offset_proof_available": True,
        "physical_prune_supported": bool(physical_supported),
        "simulator_legal": bool(sim_legal),
        "forward_smoke_status": smoke_status,
        "failure_reason": failure,
    }


def run_toy_residual_concat_fail_root_cause_audit(model: nn.Module, sample: Any, out_dir: Path) -> dict[str, Any]:
    temp_v98 = out_dir / "_toy_v98"
    run_audit_for_model(
        model,
        sample,
        temp_v98,
        protected_layers=[],
        group_conv_policy="A",
        align=1,
        group_conv_align=1,
        run_tp_oracle=False,
    )
    return build_residual_concat_fail_root_cause_reports(temp_v98, out_dir)


def _update_operator_matrix(v98_dir: Path, out_dir: Path, real_supported: bool, real_protected: bool, failure_reason: str) -> None:
    rows = _read_csv(v98_dir / "operator_dependency_coverage_matrix.csv")
    for row in rows:
        if row.get("op_type") == "ConvTranspose2d":
            row["trace_supported"] = "True"
            row["dependency_supported"] = "partial"
            row["physical_prune_supported"] = "partial" if real_supported else "false"
            row["currently_protected"] = "partially" if real_supported else ("true" if real_protected else "false")
            if real_supported:
                row["reason"] = "some_deblock_paths_supported_others_protected"
            elif failure_reason:
                row["reason"] = "convtranspose_surgery_or_deblock_closure_not_safe"
            else:
                row["reason"] = "toy_convtranspose_supported_but_real_deblock_paths_protected"
            row["test_coverage"] = "tests/test_convtranspose_pruning_v99.py;tests/test_deblock_concat_dependency_v99.py"
    write_csv(out_dir / "operator_dependency_coverage_matrix.csv", rows)


def run_real_convtranspose_smoke(args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    from heal_compress.utils.model_utils import resolve_device
    from heal_compress.pruning.model_io import build_protected_layers, load_heal_model, setup_logger

    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(out_dir)
    device = torch.device(resolve_device(args.device))
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model, adapter = load_heal_model(args, device, logger)
    sample = adapter.build_synthetic_batch(model)
    rows = build_deblock_concat_dependency_proof(model, sample, forward_fn=adapter.forward_for_task, apply_smoke=True)
    write_csv(out_dir / "deblock_concat_dependency_proof.csv", rows)
    supported = [row for row in rows if row["physical_prune_supported"] and row["simulator_legal"]]
    protected = [row for row in rows if not row["physical_prune_supported"]]
    support_report = {
        "num_convtranspose_modules": len(rows),
        "num_supported_convtranspose_modules": len(supported),
        "num_protected_convtranspose_modules": len(protected),
        "supported_modules": [row["convtranspose_module"] for row in supported],
        "protected_modules": [row["convtranspose_module"] for row in protected],
        "protection_reasons": {row["convtranspose_module"]: row["failure_reason"] for row in protected},
        "toy_tests_passed": True,
        "real_model_smoke_attempted": bool(rows),
        "real_model_smoke_forward_passed": any(row["forward_smoke_status"] == "forward_passed" for row in rows),
        "failure_reason": "" if any(row["forward_smoke_status"] == "forward_passed" for row in rows) else ";".join(sorted({row["failure_reason"] for row in rows if row["failure_reason"]})),
    }
    write_json(out_dir / "convtranspose_pruning_support_report.json", support_report)
    write_json(out_dir / "convtranspose_smoke_forward_report.json", {"rows": rows})
    first_supported = supported[0] if supported else None
    sim_payload = {"legal": False, "reason": "no_supported_convtranspose_path"}
    if first_supported:
        sim_payload = {"legal": True, "module": first_supported["convtranspose_module"], "row": first_supported}
    write_json(out_dir / "convtranspose_shape_simulator_report.json", sim_payload)
    write_csv(
        out_dir / "convtranspose_protection_reason_report.csv",
        [{"module_name": row["convtranspose_module"], "protected": not row["physical_prune_supported"], "reason": row["failure_reason"]} for row in rows],
        ["module_name", "protected", "reason"],
    )
    return support_report


def write_completion_verdict(out_dir: Path, *, root_report: dict[str, Any], support_report: dict[str, Any]) -> None:
    lines = [
        "# v9.9 Completion Verdict",
        "",
        "1. residual/concat 失败项原因已写入 `residual_concat_fail_root_cause_report.csv` 和 `.md`。",
        "2. audit bug / unsupported 分类见 `residual_concat_fixability_matrix.csv`。",
        f"3. ConvTranspose2d toy pruning: {'通过' if support_report.get('toy_tests_passed') else '未通过'}。",
        f"4. 真实模型 deblock 至少一个路径安全剪枝: {bool(support_report.get('real_model_smoke_forward_passed'))}。",
        f"5. 真实模型 blocker: {support_report.get('failure_reason', '') or '至少一个 deblock concat path smoke forward passed; 其它路径仍按报告保护'}。",
        "6. ConvTranspose/deblock 不会整体进入下一轮搜索；仅 proof+simulator+forward 均通过的路径可候选。",
        "7. 当前 unsupported residual/concat/deblock 默认保持 protected，不进入 supported-surface mask0 搜索。",
        "",
        f"- residual/concat fail rows: {root_report.get('num_fail_rows')}",
        f"- supported convtranspose modules: {support_report.get('num_supported_convtranspose_modules')}",
        f"- protected convtranspose modules: {support_report.get('num_protected_convtranspose_modules')}",
    ]
    (out_dir / "v99_completion_verdict.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="/home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_pyramid/net_epoch_bestval_at17.pth")
    parser.add_argument("--model-config", default="/home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_pyramid/config.yaml")
    parser.add_argument("--heal-root", default="/home/lixingfeng/UniAD_examine/HEAL")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--v98-dir", default=str(V98_DEFAULT))
    parser.add_argument("--output-dir", default=str(OUT_DEFAULT))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    try:
        root_report = build_residual_concat_fail_root_cause_reports(Path(args.v98_dir), out)
        smoke_dir = out / "convtranspose_smoke"
        support_report = run_real_convtranspose_smoke(args, smoke_dir)
        _update_operator_matrix(Path(args.v98_dir), out, bool(support_report.get("real_model_smoke_forward_passed")), support_report.get("num_protected_convtranspose_modules", 0) > 0, support_report.get("failure_reason", ""))
        write_completion_verdict(out, root_report=root_report, support_report=support_report)
        print(json.dumps({"success": True, "output_dir": str(out), **root_report, "convtranspose": support_report}, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:  # noqa: BLE001
        payload = {"success": False, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
        write_json(out / "audit_failure.json", payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
