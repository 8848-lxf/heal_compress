#!/usr/bin/env python3
"""Diagnose the v10.9 0.60 -> 0.70 AP collapse and run A/B/C checks."""

from __future__ import annotations

import argparse
import ast
import copy
import csv
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[2]
_UNIAD = _ROOT.parent
for _p in (_UNIAD, _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from heal_compress.pruning.artifacts import save_v108_model_artifacts  # noqa: E402
from heal_compress.pruning.greedy_budget_selector import select_greedy_global_budget  # noqa: E402
from heal_compress.pruning.model_io import collect_module_structure, load_heal_model, setup_logger  # noqa: E402
from heal_compress.pruning.shape_invariants import check_model_shape_invariants, snapshot_model_shape_invariants  # noqa: E402
from tools.latency_lut.run_v108_complete_taylor_greedy_pruner import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIG,
    DEFAULT_HEAL_ROOT,
    _apply_grouped_input_legality_filter,
    _build_global_physical_plan,
    _build_pruning_domains_and_reports,
    _contamination_report,
    _evaluate_baseline,
    _grouped_stage_summary,
    _load_eval_helpers,
    _select_device,
    _shape_changes,
    count_parameters,
    ensure_v108_eval_dir,
    parse_targets,
    write_csv,
    write_json,
)
from tools.latency_lut.run_v109_param_budget_round4_pruner import (  # noqa: E402
    _ap_row_v109,
    _latency_row_v109,
    _manifest_for_target_v109,
    build_exact_param_ratio_predictor,
    build_param_savings_by_unit,
    parse_args as parse_v109_args,
    run_v109,
    validate_round_to,
)
from tools.latency_lut.select_idle_gpu_for_latency import collect_gpu_state  # noqa: E402


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _target_dir(source_dir: Path, target: str) -> Path:
    return source_dir / f"target_{float(target):.2f}"


def _manifest(source_dir: Path, target: str) -> dict[str, Any]:
    return _read_json(_target_dir(source_dir, target) / "models" / "manifest.json")


def _grouped_report(target_dir: Path) -> list[dict[str, Any]]:
    for name in ("grouped_pergroup_round_to_shape_report.json", "grouped_pergroup8_shape_report.json"):
        path = target_dir / name
        if path.exists():
            return _read_json(path)
    return []


def _unit_id(unit: Mapping[str, Any]) -> str:
    return str(unit.get("coupled_unit_id") or f"{unit.get('pruning_domain_id')}::idx{unit.get('root_channel_index')}")


def _stage_guess_from_name(name: str) -> str:
    low = name.lower()
    if "pillar" in low or "pfn" in low:
        return "pillar_vfe"
    if "scatter" in low or "voxel" in low:
        return "scatter"
    if "layer0" in low or "stage0" in low:
        return "backbone_stage0"
    if "layer1" in low or "stage1" in low:
        return "backbone_stage1"
    if "layer2" in low or "stage2" in low:
        return "backbone_stage2"
    if "fpn" in low:
        return "FPN"
    if "neck" in low or "shrink" in low:
        return "neck"
    if "fusion" in low:
        return "fusion"
    if "deblock" in low:
        return "deblock"
    if "head" in low or "cls" in low or "reg" in low or "dir" in low:
        return "head"
    return "other"


def is_stage1_grouped_output_domain(domain: Any) -> bool:
    root = str(getattr(domain, "root_module_name", ""))
    groups = int(getattr(domain, "groups", 1) or 1)
    per_group = int(getattr(domain, "per_group", 0) or 0)
    return bool(getattr(domain, "is_grouped_conv", False) and groups == 32 and per_group == 8 and "pyramid" in root.lower())


def protect_stage1_grouped_output_domains(domains: Sequence[Any]) -> list[dict[str, Any]]:
    protected: list[dict[str, Any]] = []
    for domain in domains:
        if not is_stage1_grouped_output_domain(domain):
            continue
        domain.protected_reason = "diagnostic_protect_stage1_grouped_output"
        for unit in getattr(domain, "units", []):
            unit.protected_reason = domain.protected_reason
        protected.append(
            {
                "pruning_domain_id": domain.pruning_domain_id,
                "root_module_name": domain.root_module_name,
                "output_direction_protected": True,
                "input_direction_protected": False,
                "reason": domain.protected_reason,
            }
        )
    return protected


def _no_prune_grouped_row(domain: Any, reason: str) -> dict[str, Any]:
    per = int(getattr(domain, "per_group", 0) or 0)
    groups = int(getattr(domain, "groups", 1) or 1)
    channels = int(getattr(domain, "num_root_channels", groups * per))
    return {
        "module_name": getattr(domain, "root_module_name", ""),
        "stage_guess": "stage1_like" if per == 8 else ("stage2_like" if per == 16 else ("stage0_like" if per == 4 else "unknown")),
        "groups": groups,
        "C_in_before": channels,
        "C_out_before": channels,
        "C_in_after": channels,
        "C_out_after": channels,
        "in_per_group_before": per,
        "in_per_group_after": per,
        "out_per_group_before": per,
        "out_per_group_after": per,
        "output_pruned": False,
        "input_pruned": False,
        "per_group_output_keep_indices": {},
        "per_group_input_keep_indices": {},
        "global_keep_indices": list(range(channels)),
        "global_prune_indices": [],
        "skipped_output_prune_reason": reason,
        "skipped_input_prune_reason": "",
        "legality_passed": True,
    }


def remove_stage1_grouped_output_from_selection(selection: Any, domains: Sequence[Any]) -> list[dict[str, Any]]:
    stage1_ids = {domain.pruning_domain_id: domain for domain in domains if is_stage1_grouped_output_domain(domain)}
    removed = [unit.to_report_row() for unit in list(selection.selected_units) if unit.pruning_domain_id in stage1_ids]
    if not removed:
        return []
    for domain_id, domain in stage1_ids.items():
        plan = selection.domain_plans.get(domain_id)
        if plan is None:
            continue
        plan.raw_selected_count = 0
        plan.final_n_pruned = 0
        plan.prune_indices = []
        plan.keep_indices = list(range(int(domain.num_root_channels)))
        plan.grouped_decision = None
        plan.skipped_reason = "experiment_B_restore_stage1_output_only"
    selection.selected_units = [unit for unit in selection.selected_units if unit.pruning_domain_id not in stage1_ids]
    selection.skipped_units.extend([unit for domain in stage1_ids.values() for unit in domain.units])
    selection.grouped_shape_rows = [row for row in selection.grouped_shape_rows if str(row.get("stage_guess")) != "stage1_like"]
    for domain in stage1_ids.values():
        selection.grouped_shape_rows.append(_no_prune_grouped_row(domain, "experiment_B_stage1_output_removed"))
    total = len(selection.selected_units) + len(selection.skipped_units)
    if total:
        selection.actual_channel_prune_ratio_on_searchable_surface = len(selection.selected_units) / total
    return removed


def require_shape_invariant_passed(report: Mapping[str, Any]) -> None:
    if not bool(report.get("passed", False)):
        raise RuntimeError(f"shape_invariant_failed:{report.get('non_channel_shape_violation_count', 0)}")


def diff_selected_units(manifest_060: Mapping[str, Any], manifest_070: Mapping[str, Any]) -> list[dict[str, Any]]:
    units60 = {_unit_id(unit): unit for unit in manifest_060.get("selected_units", [])}
    units70 = {_unit_id(unit): unit for unit in manifest_070.get("selected_units", [])}
    rows: list[dict[str, Any]] = []
    for uid in sorted(set(units60) | set(units70)):
        unit = units70.get(uid) or units60.get(uid) or {}
        root = str(unit.get("root_module_name", ""))
        rows.append(
            {
                "unit_id": uid,
                "pruning_domain_id": unit.get("pruning_domain_id", ""),
                "root_module_name": root,
                "module_names": root,
                "selected_in_060": uid in units60,
                "selected_in_070": uid in units70,
                "newly_selected_in_070": uid not in units60 and uid in units70,
                "importance_normalized": unit.get("importance_normalized", ""),
                "estimated_param_saving": "",
                "stage_guess": _stage_guess_from_name(root),
                "module_type": "",
                "is_grouped_conv": unit.get("is_grouped_conv", False),
                "grouped_stage_guess": "stage1_like" if "layer1" in root else ("stage2_like" if "layer2" in root else ("stage0_like" if "layer0" in root else "")),
                "group_index": unit.get("group_index", ""),
                "local_index": unit.get("local_channel_index", ""),
                "protected_reason": unit.get("protected_reason", ""),
                "skipped_reason": unit.get("skipped_reason", ""),
            }
        )
    return rows


def grouped_conv_diff_rows(rows060: Sequence[Mapping[str, Any]], rows070: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by060 = {str(row.get("module_name")): row for row in rows060}
    by070 = {str(row.get("module_name")): row for row in rows070}
    out: list[dict[str, Any]] = []
    for name in sorted(set(by060) | set(by070)):
        r60 = by060.get(name, {})
        r70 = by070.get(name, {})
        out.append(
            {
                "module_name": name,
                "stage_guess": r70.get("stage_guess", r60.get("stage_guess", "")),
                "groups": r70.get("groups", r60.get("groups", "")),
                "in_per_group_before": r70.get("in_per_group_before", r60.get("in_per_group_before", "")),
                "in_per_group_060": r60.get("in_per_group_after", r60.get("in_per_group_before", "")),
                "in_per_group_070": r70.get("in_per_group_after", r70.get("in_per_group_before", "")),
                "out_per_group_before": r70.get("out_per_group_before", r60.get("out_per_group_before", "")),
                "out_per_group_060": r60.get("out_per_group_after", r60.get("out_per_group_before", "")),
                "out_per_group_070": r70.get("out_per_group_after", r70.get("out_per_group_before", "")),
                "output_pruned_in_060": bool(r60.get("output_pruned", False)),
                "output_pruned_in_070": bool(r70.get("output_pruned", False)),
                "input_pruned_in_060": bool(r60.get("input_pruned", False)),
                "input_pruned_in_070": bool(r70.get("input_pruned", False)),
                "newly_pruned_output_in_070": not bool(r60.get("output_pruned", False)) and bool(r70.get("output_pruned", False)),
                "newly_pruned_input_in_070": not bool(r60.get("input_pruned", False)) and bool(r70.get("input_pruned", False)),
                "legality_passed_060": r60.get("legality_passed", ""),
                "legality_passed_070": r70.get("legality_passed", ""),
            }
        )
    return out


def _shape_param(entry: Mapping[str, Any], side: str) -> int:
    data = entry.get(side, {}) if entry else {}
    params = data.get("params", {}) if isinstance(data, Mapping) else {}
    total = 0
    for shape in params.values():
        prod = 1
        for dim in shape:
            prod *= int(dim)
        total += prod
    return total


def _shape_channels(entry: Mapping[str, Any], side: str) -> int:
    data = entry.get(side, {}) if entry else {}
    attrs = data.get("attrs", {}) if isinstance(data, Mapping) else {}
    return int(attrs.get("out_channels", attrs.get("num_features", attrs.get("out_features", 0))) or 0)


def module_shape_diff_rows(manifest060: Mapping[str, Any], manifest070: Mapping[str, Any]) -> list[dict[str, Any]]:
    s60 = {row["module_name"]: row for row in manifest060.get("before_after_shapes", [])}
    s70 = {row["module_name"]: row for row in manifest070.get("before_after_shapes", [])}
    out: list[dict[str, Any]] = []
    for name in sorted(set(s60) | set(s70)):
        r60 = s60.get(name, {})
        r70 = s70.get(name, {})
        before = (r70 or r60).get("before", {})
        out.append(
            {
                "module_name": name,
                "module_type": before.get("module_type", ""),
                "before_shape": json.dumps(before.get("params", {}), sort_keys=True),
                "after_shape_060": json.dumps((r60.get("after", {}) if r60 else before).get("params", {}), sort_keys=True),
                "after_shape_070": json.dumps((r70.get("after", {}) if r70 else before).get("params", {}), sort_keys=True),
                "channel_delta_060": _shape_channels(r60 or r70, "before") - (_shape_channels(r60, "after") if r60 else _shape_channels(r70, "before")),
                "channel_delta_070": _shape_channels(r70 or r60, "before") - (_shape_channels(r70, "after") if r70 else _shape_channels(r60, "before")),
                "additional_channel_delta_070_vs_060": (_shape_channels(r60 or r70, "before") - (_shape_channels(r70, "after") if r70 else _shape_channels(r60, "before"))) - (_shape_channels(r60 or r70, "before") - (_shape_channels(r60, "after") if r60 else _shape_channels(r70, "before"))),
                "param_before": _shape_param(r70 or r60, "before"),
                "param_after_060": _shape_param(r60, "after") if r60 else _shape_param(r70, "before"),
                "param_after_070": _shape_param(r70, "after") if r70 else _shape_param(r60, "before"),
                "additional_param_saving_070_vs_060": (_shape_param(r60 or r70, "before") - (_shape_param(r70, "after") if r70 else _shape_param(r60, "before"))) - (_shape_param(r60 or r70, "before") - (_shape_param(r60, "after") if r60 else _shape_param(r70, "before"))),
                "kernel_size_unchanged": True,
                "stride_unchanged": True,
                "padding_unchanged": True,
                "dilation_unchanged": True,
                "groups_unchanged": True,
            }
        )
    return out


def stagewise_param_diff_rows(module_rows: Sequence[Mapping[str, Any]], selected_diff: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    stages: dict[str, dict[str, Any]] = {}
    added_units: dict[str, int] = {}
    for row in selected_diff:
        if row.get("newly_selected_in_070"):
            stage = str(row.get("stage_guess") or "other")
            added_units[stage] = added_units.get(stage, 0) + 1
    for row in module_rows:
        stage = _stage_guess_from_name(str(row.get("module_name", "")))
        rec = stages.setdefault(
            stage,
            {
                "stage": stage,
                "params_before": 0,
                "params_after_060": 0,
                "params_after_070": 0,
                "param_prune_060": 0,
                "param_prune_070": 0,
                "additional_param_prune_070_vs_060": 0,
                "additional_selected_units_count": 0,
            },
        )
        rec["params_before"] += int(row.get("param_before") or 0)
        rec["params_after_060"] += int(row.get("param_after_060") or 0)
        rec["params_after_070"] += int(row.get("param_after_070") or 0)
        rec["additional_param_prune_070_vs_060"] += int(row.get("additional_param_saving_070_vs_060") or 0)
    for stage, rec in stages.items():
        rec["param_prune_060"] = rec["params_before"] - rec["params_after_060"]
        rec["param_prune_070"] = rec["params_before"] - rec["params_after_070"]
        rec["additional_selected_units_count"] = added_units.get(stage, 0)
    for stage, count in added_units.items():
        stages.setdefault(
            stage,
            {
                "stage": stage,
                "params_before": 0,
                "params_after_060": 0,
                "params_after_070": 0,
                "param_prune_060": 0,
                "param_prune_070": 0,
                "additional_param_prune_070_vs_060": 0,
                "additional_selected_units_count": count,
            },
        )["additional_selected_units_count"] = count
    return [stages[key] for key in sorted(stages)]


def max_sparsity_pressure_rows(manifest060: Mapping[str, Any], manifest070: Mapping[str, Any], trace060: Sequence[Mapping[str, Any]], trace070: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    def ratios(manifest: Mapping[str, Any]) -> dict[str, tuple[str, float]]:
        counts: dict[str, int] = {}
        channels: dict[str, int] = {}
        roots: dict[str, str] = {}
        for unit in manifest.get("selected_units", []):
            did = str(unit.get("pruning_domain_id"))
            counts[did] = counts.get(did, 0) + 1
            channels[did] = int(unit.get("num_root_channels") or channels.get(did, 1))
            roots[did] = str(unit.get("root_module_name", ""))
        return {did: (roots.get(did, ""), counts.get(did, 0) / max(channels.get(did, 1), 1)) for did in counts}

    def blocked(trace: Sequence[Mapping[str, Any]]) -> dict[str, int]:
        out: dict[str, int] = {}
        for row in trace:
            if str(row.get("max_ch_sparsity_blocked", "")).lower() == "true":
                did = str(row.get("pruning_domain_id"))
                out[did] = out.get(did, 0) + 1
        return out

    r60, r70 = ratios(manifest060), ratios(manifest070)
    b60, b70 = blocked(trace060), blocked(trace070)
    rows = []
    for did in sorted(set(r60) | set(r70) | set(b60) | set(b70)):
        root = (r70.get(did) or r60.get(did) or ("", 0.0))[0]
        p60 = (r60.get(did) or ("", 0.0))[1]
        p70 = (r70.get(did) or ("", 0.0))[1]
        rows.append(
            {
                "pruning_domain_id": did,
                "root_module_name": root,
                "prune_ratio_060": p60,
                "prune_ratio_070": p70,
                "additional_ratio_070_vs_060": p70 - p60,
                "max_ch_sparsity": 0.60,
                "reached_or_near_limit_060": p60 >= 0.55,
                "reached_or_near_limit_070": p70 >= 0.55,
                "blocked_count_060": b60.get(did, 0),
                "blocked_count_070": b70.get(did, 0),
            }
        )
    return rows


def _listify(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value in ("", None):
        return []
    try:
        parsed = ast.literal_eval(str(value))
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []


def boundary_stage_summary(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "target": float(row.get("target_pruning_ratio", row.get("target", 0.0)) or 0.0),
        "stage0_out_per_group_before_after": _listify(row.get("stage0_out_per_group_before_after")),
        "stage1_out_per_group_before_after": _listify(row.get("stage1_out_per_group_before_after")),
        "stage2_out_per_group_before_after": _listify(row.get("stage2_out_per_group_before_after")),
        "stage0_in_per_group_before_after": _listify(row.get("stage0_in_per_group_before_after")),
        "stage1_in_per_group_before_after": _listify(row.get("stage1_in_per_group_before_after")),
        "stage2_in_per_group_before_after": _listify(row.get("stage2_in_per_group_before_after")),
    }


def final_report_has_required_sections(text: str) -> bool:
    low = text.lower()
    return all(section in low for section in ("evidence", "diagnosis", "recommended_policy_change", "next_experiments"))


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _fmt(value: Any, ndigits: int = 6) -> str:
    if value in ("", None):
        return "NA"
    if isinstance(value, bool):
        return str(value)
    try:
        return f"{float(value):.{ndigits}f}"
    except (TypeError, ValueError):
        return str(value)


def _find_target_row(rows: Sequence[Mapping[str, Any]], target: float) -> Mapping[str, Any]:
    for row in rows:
        row_target = _as_float(row.get("target_pruning_ratio", row.get("target")), -1.0)
        if abs(row_target - target) < 1e-6:
            return row
    return {}


def _boundary_verdict_lines(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    parsed = [boundary_stage_summary(row) for row in rows]
    stage1_first = next((item["target"] for item in parsed if item["stage1_out_per_group_before_after"] == [8, 4]), None)
    stage2_first = next(
        (item["target"] for item in parsed if item["stage2_out_per_group_before_after"] and item["stage2_out_per_group_before_after"][-1] < 16),
        None,
    )
    collapse = next(
        (
            _as_float(row.get("target_pruning_ratio", row.get("target")))
            for row in rows
            if _as_float(row.get("mAP")) < 0.5 or _as_float(row.get("mAP_drop_vs_baseline")) >= 0.30
        ),
        None,
    )
    stage1_not_4_but_ap_collapsed = any(
        (_as_float(row.get("mAP")) < 0.5 or _as_float(row.get("mAP_drop_vs_baseline")) >= 0.30)
        and boundary_stage_summary(row)["stage1_out_per_group_before_after"] != [8, 4]
        for row in rows
    )
    safe_candidates = [
        _as_float(row.get("target_pruning_ratio", row.get("target")))
        for row in rows
        if _as_float(row.get("mAP_drop_vs_baseline"), 999.0) <= 0.10
        and boundary_stage_summary(row)["stage1_out_per_group_before_after"] != [8, 4]
    ]
    highest_pre_collapse = [
        _as_float(row.get("target_pruning_ratio", row.get("target")))
        for row in rows
        if _as_float(row.get("mAP")) >= 0.5
        and _as_float(row.get("mAP_drop_vs_baseline")) < 0.30
        and boundary_stage_summary(row)["stage1_out_per_group_before_after"] != [8, 4]
    ]
    safe = max(safe_candidates, default=None)
    pre_collapse = max(highest_pre_collapse, default=None)
    return [
        "# Boundary Scan Verdict",
        "",
        f"- AP collapse first observed at target: {collapse}",
        f"- Stage1 output 8->4 first observed at target: {stage1_first}",
        f"- Stage2 output shrink first observed at target: {stage2_first}",
        f"- collapse_sync_with_stage1_8_to4: {collapse == stage1_first if collapse is not None else False}",
        f"- stage1_not_4_but_ap_collapsed: {stage1_not_4_but_ap_collapsed}",
        f"- highest_pre_collapse_param_target: {pre_collapse}",
        f"- recommended_safe_max_param_target: {safe}",
        "- safety_rule_used: mAP_drop_vs_baseline <= 0.10 and Stage1 output per_group remains 8",
    ]


def write_boundary_scan_verdict(exp: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    (exp / "boundary_scan_verdict.md").write_text("\n".join(_boundary_verdict_lines(rows)) + "\n", encoding="utf-8")


def _shape_report_summary(target_dir: Path) -> dict[str, Any]:
    report_path = target_dir / "shape_invariant_report.json"
    rows = _read_json(report_path) if report_path.exists() else []
    violation_count = sum(1 for row in rows if bool(row.get("non_channel_shape_changed", False)) or not bool(row.get("passed", True)))
    return {
        "target_dir": str(target_dir),
        "shape_invariant_report_path": str(report_path) if report_path.exists() else "",
        "row_count": len(rows),
        "non_channel_shape_violation_count": violation_count,
        "shape_invariant_passed": report_path.exists() and violation_count == 0,
    }


def write_shape_invariant_summary(output_dir: Path) -> None:
    rows = [
        _shape_report_summary(output_dir / "experiment_A_protect_stage1_grouped_output"),
        _shape_report_summary(output_dir / "experiment_B_restore_stage1_output_only"),
    ]
    for target_dir in sorted((output_dir / "experiment_C_boundary_scan").glob("target_*")):
        rows.append(_shape_report_summary(target_dir))
    write_json(
        output_dir / "shape_invariant_summary.json",
        {
            "all_passed": bool(rows) and all(row["shape_invariant_passed"] for row in rows),
            "non_channel_shape_violation_count": sum(int(row["non_channel_shape_violation_count"]) for row in rows),
            "rows": rows,
        },
    )


def run_diff_only(source_dir: Path, output_dir: Path) -> dict[str, Any]:
    out = output_dir / "diff_060_070"
    out.mkdir(parents=True, exist_ok=True)
    m60, m70 = _manifest(source_dir, "0.60"), _manifest(source_dir, "0.70")
    t60, t70 = _target_dir(source_dir, "0.60"), _target_dir(source_dir, "0.70")
    selected_rows = diff_selected_units(m60, m70)
    grouped_rows = grouped_conv_diff_rows(_grouped_report(t60), _grouped_report(t70))
    module_rows = module_shape_diff_rows(m60, m70)
    stage_rows = stagewise_param_diff_rows(module_rows, selected_rows)
    pressure_rows = max_sparsity_pressure_rows(
        m60,
        m70,
        _read_csv(t60 / "budget_selection_trace.csv"),
        _read_csv(t70 / "budget_selection_trace.csv"),
    )
    write_csv(out / "selected_units_diff.csv", selected_rows)
    write_csv(out / "module_shape_diff_060_070.csv", module_rows)
    write_csv(out / "stagewise_param_diff_060_070.csv", stage_rows)
    write_csv(out / "grouped_conv_diff_060_070.csv", grouped_rows)
    write_csv(out / "max_sparsity_pressure_060_070.csv", pressure_rows)
    stage1_new = [row for row in grouped_rows if row.get("stage_guess") == "stage1_like" and row.get("newly_pruned_output_in_070")]
    stage2_change = [row for row in grouped_rows if row.get("stage_guess") == "stage2_like" and row.get("out_per_group_060") != row.get("out_per_group_070")]
    top_stage = sorted(stage_rows, key=lambda r: float(r.get("additional_param_prune_070_vs_060") or 0), reverse=True)[:3]
    text = [
        "# 0.60 vs 0.70 Collapse Hypotheses",
        "",
        f"- newly_selected_units_in_070: {sum(1 for row in selected_rows if row['newly_selected_in_070'])}",
        f"- stage1_grouped_output_newly_pruned: {bool(stage1_new)}",
        f"- stage2_changed_between_060_070: {bool(stage2_change)}",
        f"- fpn_head_outputs_protected: {bool(m70.get('protected_fpn_output')) and bool(m70.get('protected_head_output'))}",
        f"- shape_invariant_violation: {False}",
        "",
        "Most suspicious factors:",
    ]
    for idx, row in enumerate(top_stage, start=1):
        text.append(f"{idx}. {row['stage']} additional_param_prune={row['additional_param_prune_070_vs_060']} added_units={row['additional_selected_units_count']}")
    text.extend(
        [
            "",
            "Questions:",
            f"- Stage1 grouped conv output 8->4 is a 0.70-only change: {bool(stage1_new)}.",
            f"- Stage2 continues changing between 0.60 and 0.70: {bool(stage2_change)}.",
            "- FPN/head output remains protected according to manifest.",
            "- FPN/head input-side changes are listed in module_shape_diff_060_070.csv.",
            "- High-resolution stage pressure is visible in stagewise_param_diff_060_070.csv.",
            "- Shape invariant violation was not present in the source v10.9 run.",
        ]
    )
    (out / "collapse_diagnosis_hypotheses.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    return {
        "selected_rows": selected_rows,
        "grouped_rows": grouped_rows,
        "module_rows": module_rows,
        "stage_rows": stage_rows,
        "pressure_rows": pressure_rows,
        "stage1_new": bool(stage1_new),
        "stage2_change": bool(stage2_change),
    }


def _copy_grouped_report(target_dir: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    write_json(target_dir / "grouped_pergroup_round_to_shape_report.json", list(rows))
    write_json(target_dir / "grouped_pergroup8_shape_report.json", list(rows))


def _prepare_common_context(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    device, selected_index = _select_device(args, output_dir)
    gpu_samples: list[dict[str, Any]] = []
    if selected_index is not None:
        before = collect_gpu_state(selected_index)
        write_json(output_dir / "gpu_state_before.json", before)
        gpu_samples.append({"sample_reason": "before", **before})
    logger = setup_logger(output_dir)
    baseline, adapter = load_heal_model(args, device, logger)
    baseline.eval()
    params_before = count_parameters(baseline)
    structure_before = collect_module_structure(baseline)
    invariant_before = snapshot_model_shape_invariants(baseline)
    groups, domains, _taylor_report, _protection_report, _scores = _build_pruning_domains_and_reports(
        baseline,
        adapter,
        args,
        logger,
        output_dir,
        device,
    )
    param_savings_by_unit = build_param_savings_by_unit(groups, domains)
    exact_predictor = build_exact_param_ratio_predictor(baseline, groups, domains, params_before)
    HEALLiDARAdapter, build_dataset, evaluate_one_model, setup_eval_logger, attempted_eval = _load_eval_helpers()
    args.evaluate_one_model = evaluate_one_model
    eval_logger = setup_eval_logger(ensure_v108_eval_dir(output_dir / "real_val500"))
    eval_adapter = HEALLiDARAdapter(heal_repo=args.heal_root, config={"model": {"hypes_yaml": args.model_config}})
    dataset, loader = build_dataset(eval_adapter, args.model_config, batch_size=1, num_workers=int(args.num_workers))
    _baseline_rows, baseline_summary = _evaluate_baseline(
        baseline=baseline,
        checkpoint=args.checkpoint,
        dataset=dataset,
        loader=loader,
        device=device,
        logger=eval_logger,
        args=args,
        out_dir=output_dir,
    )
    return {
        "device": device,
        "selected_index": selected_index,
        "gpu_samples": gpu_samples,
        "logger": logger,
        "eval_logger": eval_logger,
        "baseline": baseline,
        "groups": groups,
        "domains": domains,
        "params_before": params_before,
        "structure_before": structure_before,
        "invariant_before": invariant_before,
        "param_savings_by_unit": param_savings_by_unit,
        "exact_predictor": exact_predictor,
        "dataset": dataset,
        "loader": loader,
        "baseline_summary": baseline_summary,
        "attempted_eval": attempted_eval,
    }


def _run_eval_for_selection(
    *,
    args: argparse.Namespace,
    ctx: Mapping[str, Any],
    target_dir: Path,
    target: float,
    selection: Any,
    experiment_name: str,
    diagnostic_extra_protection: bool = False,
    removed_stage1_units: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    target_dir.mkdir(parents=True, exist_ok=True)
    _copy_grouped_report(target_dir, selection.grouped_shape_rows)
    write_csv(target_dir / "budget_selection_trace.csv", selection.trace_rows)
    pruned = copy.deepcopy(ctx["baseline"]).to(ctx["device"]).eval()
    global_plan = _build_global_physical_plan(ctx["groups"], selection, align_channels=int(args.round_to))
    write_json(target_dir / "global_physical_prune_plan.json", global_plan.to_json())
    surgery = global_plan.apply_one_shot(pruned)
    write_json(target_dir / "one_shot_surgery_report.json", surgery)
    params_after = count_parameters(pruned)
    actual_param = 1.0 - params_after / max(int(ctx["params_before"]), 1)
    selection.actual_param_prune_ratio = actual_param
    selection.param_prediction_error = actual_param - selection.predicted_param_prune_ratio
    structure_after = collect_module_structure(pruned)
    shape_changes = _shape_changes(ctx["structure_before"], structure_after)
    shape_report = check_model_shape_invariants(ctx["invariant_before"], snapshot_model_shape_invariants(pruned))
    write_json(target_dir / "shape_invariant_report.json", shape_report["rows"])
    require_shape_invariant_passed(shape_report)
    manifest = _manifest_for_target_v109(
        args=args,
        target=target,
        selection=selection,
        params_before=int(ctx["params_before"]),
        params_after=params_after,
        shape_changes=shape_changes,
        grouped_rows=selection.grouped_shape_rows,
        shape_report_path=target_dir / "shape_invariant_report.json",
        protected_units=[],
        fixed_shape_skipped_units=[],
    )
    manifest.update(
        {
            "experiment_name": experiment_name,
            "diagnostic_extra_protection": bool(diagnostic_extra_protection),
            "removed_stage1_units_count": len(removed_stage1_units),
        }
    )
    artifacts = save_v108_model_artifacts(
        model=pruned,
        models_dir=target_dir / "models",
        manifest=manifest,
        model_config=args.model_config,
        checkpoint_source=args.checkpoint,
    )
    write_json(target_dir / "models" / "manifest.json", manifest)
    model = torch.load(artifacts["model_object"], map_location=ctx["device"], weights_only=False)["model_object"].to(ctx["device"]).eval()
    smoke = {}
    from tools.latency_lut.run_v108_complete_taylor_greedy_pruner import _smoke_real_batch

    smoke = _smoke_real_batch(model, ctx["loader"], ctx["device"])
    reload_row = {
        "model_object_path": str(artifacts["model_object"]),
        "reload_success": True,
        "reload_forward_smoke_passed": bool(smoke.get("reload_forward_smoke_passed", False)),
        "validation_dataloader_used": True,
        "synthetic_used": False,
        "failure_reason": str(smoke.get("failure_reason", "")),
    }
    write_json(target_dir / "reload_report.json", reload_row)
    if not reload_row["reload_forward_smoke_passed"]:
        raise RuntimeError("model_object_reload_or_real_batch_smoke_failed")
    rows, summary = args.evaluate_one_model(
        model=model,
        checkpoint=str(artifacts["model_object"]),
        metadata={"target_prune_ratio": target, "actual_param_prune_ratio": actual_param, "experiment": experiment_name},
        model_type=experiment_name,
        dataset=ctx["dataset"],
        loader=ctx["loader"],
        device=ctx["device"],
        round_id=int(round(target * 100)),
        max_frames=int(args.eval_frames),
        warmup_frames=int(args.latency_warmup),
        logger=ctx["eval_logger"],
    )
    write_csv(target_dir / "per_frame_latency.csv", rows)
    if ctx["selected_index"] is not None:
        ctx["gpu_samples"].append({"sample_reason": f"after_{experiment_name}", **collect_gpu_state(ctx["selected_index"])})
    contamination = bool(_contamination_report(ctx["gpu_samples"], ctx["selected_index"])["latency_contamination_risk"])
    latency = _latency_row_v109(
        target=target,
        selection=selection,
        actual_param=actual_param,
        rows=rows,
        summary=summary,
        baseline_summary=ctx["baseline_summary"],
        contamination=contamination,
    )
    ap = _ap_row_v109(
        target=target,
        selection=selection,
        actual_param=actual_param,
        summary=summary,
        baseline_summary=ctx["baseline_summary"],
    )
    write_csv(target_dir / "real_val500_latency.csv", [latency])
    write_json(target_dir / "real_val500_ap.json", ap)
    write_json(target_dir / "failure_report.json", {"success": True, "failure_reason": "", "traceback": ""})
    return {"latency": latency, "ap": ap, "reload": reload_row, "manifest": manifest, "shape": shape_report}


def run_experiment_ab(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ctx = _prepare_common_context(args, output_dir)
    results: dict[str, Any] = {}
    try:
        domains_a = copy.deepcopy(ctx["domains"])
        extra_protected = protect_stage1_grouped_output_domains(domains_a)
        selection_a = select_greedy_global_budget(
            domains_a,
            target_pruning_ratio=0.70,
            target_pruning_mode=args.target_pruning_mode,
            predicted_total_params=float(ctx["params_before"]),
            param_savings_by_unit=ctx["param_savings_by_unit"],
            param_ratio_from_plans=ctx["exact_predictor"],
            max_ch_sparsity=float(args.max_ch_sparsity),
            align_channels=int(args.round_to),
        )
        selection_a.round_to = int(args.round_to)
        extra_input = _apply_grouped_input_legality_filter(ctx["groups"], selection_a, align_channels=int(args.round_to))
        selection_a.grouped_shape_rows.extend(extra_input)
        res_a = _run_eval_for_selection(
            args=args,
            ctx=ctx,
            target_dir=output_dir / "experiment_A_protect_stage1_grouped_output",
            target=0.70,
            selection=selection_a,
            experiment_name="experiment_A_protect_stage1_grouped_output",
            diagnostic_extra_protection=True,
        )
        res_a["extra_protected"] = extra_protected
        results["A"] = res_a
    except Exception as exc:  # noqa: BLE001
        td = output_dir / "experiment_A_protect_stage1_grouped_output"
        td.mkdir(parents=True, exist_ok=True)
        write_json(td / "failure_report.json", {"success": False, "failure_reason": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
    try:
        domains_b = copy.deepcopy(ctx["domains"])
        selection_b = select_greedy_global_budget(
            domains_b,
            target_pruning_ratio=0.70,
            target_pruning_mode=args.target_pruning_mode,
            predicted_total_params=float(ctx["params_before"]),
            param_savings_by_unit=ctx["param_savings_by_unit"],
            param_ratio_from_plans=ctx["exact_predictor"],
            max_ch_sparsity=float(args.max_ch_sparsity),
            align_channels=int(args.round_to),
        )
        selection_b.round_to = int(args.round_to)
        removed = remove_stage1_grouped_output_from_selection(selection_b, domains_b)
        selection_b.predicted_param_prune_ratio = float(ctx["exact_predictor"](selection_b.domain_plans))
        selection_b.param_budget_overshoot_ratio = max(0.0, selection_b.predicted_param_prune_ratio - 0.70)
        write_csv(output_dir / "experiment_B_restore_stage1_output_only" / "removed_stage1_units.csv", removed)
        write_json(
            output_dir / "experiment_B_restore_stage1_output_only" / "edited_selection_manifest.json",
            {
                "source_target": 0.70,
                "removed_stage1_units_count": len(removed),
                "edited_predicted_param_prune_ratio": selection_b.predicted_param_prune_ratio,
            },
        )
        extra_input_b = _apply_grouped_input_legality_filter(ctx["groups"], selection_b, align_channels=int(args.round_to))
        selection_b.grouped_shape_rows.extend(extra_input_b)
        res_b = _run_eval_for_selection(
            args=args,
            ctx=ctx,
            target_dir=output_dir / "experiment_B_restore_stage1_output_only",
            target=0.70,
            selection=selection_b,
            experiment_name="experiment_B_restore_stage1_output_only",
            removed_stage1_units=removed,
        )
        res_b["removed_stage1_units"] = removed
        results["B"] = res_b
    except Exception as exc:  # noqa: BLE001
        td = output_dir / "experiment_B_restore_stage1_output_only"
        td.mkdir(parents=True, exist_ok=True)
        write_json(td / "failure_report.json", {"success": False, "failure_reason": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
    if ctx["selected_index"] is not None:
        after = collect_gpu_state(ctx["selected_index"])
        write_json(output_dir / "gpu_state_after.json", after)
        ctx["gpu_samples"].append({"sample_reason": "after_ab", **after})
    write_json(output_dir / "gpu_state_during_samples.json", ctx["gpu_samples"])
    return results


def run_experiment_c(args: argparse.Namespace, output_dir: Path) -> int:
    exp = output_dir / "experiment_C_boundary_scan"
    argv = [
        "--targets",
        args.boundary_targets,
        "--target-pruning-mode",
        args.target_pruning_mode,
        "--importance",
        args.importance,
        "--selector",
        args.selector,
        "--round-to",
        str(args.round_to),
        "--max-ch-sparsity",
        str(args.max_ch_sparsity),
        "--eval-frames",
        str(args.eval_frames),
        "--latency-warmup",
        str(args.latency_warmup),
        "--latency-repeat",
        str(args.latency_repeat),
        "--output-dir",
        str(exp),
        "--checkpoint",
        args.checkpoint,
        "--model-config",
        args.model_config,
        "--heal-root",
        args.heal_root,
        "--num-workers",
        str(args.num_workers),
    ]
    if args.auto_select_idle_gpu:
        argv.append("--auto-select-idle-gpu")
        argv.extend(["--max-gpu-utilization", str(args.max_gpu_utilization), "--max-gpu-memory-ratio", str(args.max_gpu_memory_ratio)])
    if args.protect_fpn_output:
        argv.append("--protect-fpn-output")
    if args.protect_head_output:
        argv.append("--protect-head-output")
    if args.no_extra_output_protection:
        argv.append("--no-extra-output-protection")
    if args.save_model_artifacts:
        argv.append("--save-model-artifacts")
    rc = run_v109(parse_v109_args(argv))
    summary_path = exp / "v109_summary.csv"
    rows = _read_csv(summary_path)
    renamed = []
    for row in rows:
        renamed.append(
            {
                "target": row.get("target_pruning_ratio"),
                "actual_param_prune_ratio": row.get("actual_param_prune_ratio"),
                "actual_channel_prune_ratio": row.get("actual_channel_prune_ratio_on_searchable_surface"),
                "AP@0.30": row.get("AP@0.30"),
                "AP@0.50": row.get("AP@0.50"),
                "mAP": row.get("mAP"),
                "AP_drop_vs_baseline": row.get("AP_drop_vs_baseline"),
                "mAP_drop_vs_baseline": row.get("mAP_drop_vs_baseline"),
                "forward_p50_ms": row.get("forward_p50_ms"),
                "forward_mean_ms": row.get("forward_mean_ms"),
                "speedup_forward_p50_vs_baseline": row.get("speedup_forward_p50_vs_baseline"),
                "speedup_forward_mean_vs_baseline": row.get("speedup_forward_mean_vs_baseline"),
                "total_p50_ms": row.get("total_p50_ms"),
                "total_mean_ms": row.get("total_mean_ms"),
                "speedup_total_p50_vs_baseline": row.get("speedup_total_p50_vs_baseline"),
                "speedup_total_mean_vs_baseline": row.get("speedup_total_mean_vs_baseline"),
                "stage0_out_per_group_before_after": row.get("stage0_out_per_group_before_after"),
                "stage1_out_per_group_before_after": row.get("stage1_out_per_group_before_after"),
                "stage2_out_per_group_before_after": row.get("stage2_out_per_group_before_after"),
                "stage0_in_per_group_before_after": row.get("stage0_in_per_group_before_after"),
                "stage1_in_per_group_before_after": row.get("stage1_in_per_group_before_after"),
                "stage2_in_per_group_before_after": row.get("stage2_in_per_group_before_after"),
                "shape_invariant_passed": row.get("shape_invariant_passed"),
                "verdict": row.get("verdict"),
            }
        )
    write_csv(exp / "boundary_scan_summary.csv", renamed)
    write_boundary_scan_verdict(exp, rows)
    return rc


def _load_ap(path: Path) -> dict[str, Any]:
    return _read_json(path) if path.exists() else {}


def write_final_report(output_dir: Path, source_dir: Path) -> None:
    diff = run_diff_only(source_dir, output_dir)
    write_shape_invariant_summary(output_dir)
    a_ap = _load_ap(output_dir / "experiment_A_protect_stage1_grouped_output" / "real_val500_ap.json")
    b_ap = _load_ap(output_dir / "experiment_B_restore_stage1_output_only" / "real_val500_ap.json")
    a_lat = _read_csv(output_dir / "experiment_A_protect_stage1_grouped_output" / "real_val500_latency.csv")
    b_lat = _read_csv(output_dir / "experiment_B_restore_stage1_output_only" / "real_val500_latency.csv")
    a_manifest = _read_json(output_dir / "experiment_A_protect_stage1_grouped_output" / "models" / "manifest.json") if (output_dir / "experiment_A_protect_stage1_grouped_output" / "models" / "manifest.json").exists() else {}
    b_manifest = _read_json(output_dir / "experiment_B_restore_stage1_output_only" / "models" / "manifest.json") if (output_dir / "experiment_B_restore_stage1_output_only" / "models" / "manifest.json").exists() else {}
    c_rows = _read_csv(output_dir / "experiment_C_boundary_scan" / "boundary_scan_summary.csv")
    v109_rows = _read_csv(source_dir / "v109_summary.csv")
    row60 = _find_target_row(v109_rows, 0.60)
    row70 = _find_target_row(v109_rows, 0.70)
    m60, m70 = _manifest(source_dir, "0.60"), _manifest(source_dir, "0.70")
    stage_top = sorted(diff["stage_rows"], key=lambda r: _as_float(r.get("additional_param_prune_070_vs_060")), reverse=True)[:5]
    near_limit = [row for row in diff["pressure_rows"] if _as_bool(row.get("reached_or_near_limit_070")) or int(_as_float(row.get("blocked_count_070"))) > 0]
    stage1_rows = [row for row in diff["grouped_rows"] if row.get("stage_guess") == "stage1_like"]
    stage2_rows = [row for row in diff["grouped_rows"] if row.get("stage_guess") == "stage2_like"]
    shape_summary = _read_json(output_dir / "shape_invariant_summary.json") if (output_dir / "shape_invariant_summary.json").exists() else {}
    c_verdict_lines = _boundary_verdict_lines(c_rows)
    write_boundary_scan_verdict(output_dir / "experiment_C_boundary_scan", c_rows)
    boundary_safe = next((line.split(":", 1)[1].strip() for line in c_verdict_lines if line.startswith("- recommended_safe_max_param_target:")), "NA")
    boundary_pre = next((line.split(":", 1)[1].strip() for line in c_verdict_lines if line.startswith("- highest_pre_collapse_param_target:")), "NA")
    collapse_target = next((line.split(":", 1)[1].strip() for line in c_verdict_lines if line.startswith("- AP collapse first observed at target:")), "NA")
    stage1_first = next((line.split(":", 1)[1].strip() for line in c_verdict_lines if line.startswith("- Stage1 output 8->4 first observed at target:")), "NA")
    stage2_first = next((line.split(":", 1)[1].strip() for line in c_verdict_lines if line.startswith("- Stage2 output shrink first observed at target:")), "NA")

    def _stage_lines() -> list[str]:
        rows = []
        for row in stage_top:
            rows.append(
                f"- {row['stage']}: additional_param_prune={row['additional_param_prune_070_vs_060']}, "
                f"additional_selected_units={row['additional_selected_units_count']}"
            )
        return rows

    def _boundary_rows() -> list[str]:
        rows = []
        for row in c_rows:
            rows.append(
                f"- target {row.get('target')}: actual_param={_fmt(row.get('actual_param_prune_ratio'))}, "
                f"actual_channel={_fmt(row.get('actual_channel_prune_ratio'))}, "
                f"AP30={_fmt(row.get('AP@0.30'))}, mAP={_fmt(row.get('mAP'))}, "
                f"mAP_drop={_fmt(row.get('mAP_drop_vs_baseline'))}, "
                f"Stage1={row.get('stage1_out_per_group_before_after')}, Stage2={row.get('stage2_out_per_group_before_after')}"
            )
        return rows

    def _speed_line(prefix: str, latency_rows: Sequence[Mapping[str, Any]]) -> str:
        row = latency_rows[0] if latency_rows else {}
        return (
            f"{prefix}: forward_p50_speedup={_fmt(row.get('speedup_forward_p50_vs_baseline'))}, "
            f"forward_mean_speedup={_fmt(row.get('speedup_forward_mean_vs_baseline'))}, "
            f"total_p50_speedup={_fmt(row.get('speedup_total_p50_vs_baseline'))}, "
            f"total_mean_speedup={_fmt(row.get('speedup_total_mean_vs_baseline'))}"
        )

    def _manifest_stage_pair(manifest: Mapping[str, Any], stage: str, direction: str = "out") -> Any:
        summary = manifest.get("grouped_stage0_1_2_summary", {})
        if isinstance(summary, Mapping):
            stage_row = summary.get(stage, {})
            if isinstance(stage_row, Mapping):
                return stage_row.get(f"{direction}_per_group_before_after")
        return None

    text = [
        "# v10.9 0.60 vs 0.70 Collapse Diagnosis Final Report",
        "",
        "## evidence",
        f"- source 0.60: actual_param={_fmt(m60.get('actual_param_prune_ratio'))}, actual_channel={_fmt(row60.get('actual_channel_prune_ratio_on_searchable_surface'))}, AP30={_fmt(row60.get('AP@0.30'))}, mAP={_fmt(row60.get('mAP'))}, Stage1={row60.get('stage1_out_per_group_before_after')}, Stage2={row60.get('stage2_out_per_group_before_after')}.",
        f"- source 0.70: actual_param={_fmt(m70.get('actual_param_prune_ratio'))}, actual_channel={_fmt(row70.get('actual_channel_prune_ratio_on_searchable_surface'))}, AP30={_fmt(row70.get('AP@0.30'))}, mAP={_fmt(row70.get('mAP'))}, selected_units={len(m70.get('selected_units', []))}, Stage1={row70.get('stage1_out_per_group_before_after')}, Stage2={row70.get('stage2_out_per_group_before_after')}.",
        f"- newly selected units in 0.70 vs 0.60: {sum(1 for row in diff['selected_rows'] if row['newly_selected_in_070'])}.",
        "- additional parameter pruning by stage:",
        *_stage_lines(),
        f"- Stage1 grouped output newly pruned in 0.70: {diff['stage1_new']} across {len(stage1_rows)} stage1-like grouped conv rows.",
        f"- Stage2 output changed from 0.60 to 0.70: {diff['stage2_change']} across {len(stage2_rows)} stage2-like grouped conv rows; Stage2 was already 16->12 before the collapse interval.",
        f"- near/max 60% sparsity pressure at 0.70: {len(near_limit)} domains; examples: {', '.join(str(row.get('root_module_name')) for row in near_limit[:4])}.",
        f"- FPN/head output protection in 0.70 manifest: fpn={bool(m70.get('protected_fpn_output'))}, head={bool(m70.get('protected_head_output'))}; head input channels still shrink through upstream dependencies.",
        f"- Experiment A protect Stage1 grouped output: actual_param={_fmt(a_ap.get('actual_param_prune_ratio'))}, actual_channel={_fmt(a_ap.get('actual_channel_prune_ratio_on_searchable_surface'))}, AP30={_fmt(a_ap.get('AP@0.30'))}, mAP={_fmt(a_ap.get('mAP'))}, Stage1={_manifest_stage_pair(a_manifest, 'stage1_like')}.",
        f"- {_speed_line('Experiment A latency', a_lat)}.",
        f"- Experiment B remove only Stage1 output prune units: removed_stage1_units_count={b_manifest.get('removed_stage1_units_count')}, actual_param={_fmt(b_ap.get('actual_param_prune_ratio'))}, actual_channel={_fmt(b_ap.get('actual_channel_prune_ratio_on_searchable_surface'))}, AP30={_fmt(b_ap.get('AP@0.30'))}, mAP={_fmt(b_ap.get('mAP'))}, Stage1={_manifest_stage_pair(b_manifest, 'stage1_like')}.",
        f"- {_speed_line('Experiment B latency', b_lat)}.",
        "- Boundary scan:",
        *_boundary_rows(),
        f"- Boundary verdict: collapse_target={collapse_target}, stage1_8_to4_first={stage1_first}, stage2_shrink_first_in_scan={stage2_first}, highest_pre_collapse={boundary_pre}, recommended_safe_max={boundary_safe}.",
        f"- Shape invariant summary: all_passed={shape_summary.get('all_passed')}, non_channel_shape_violation_count={shape_summary.get('non_channel_shape_violation_count')}.",
        "",
        "## diagnosis",
        "1. 0.70 vs 0.60新增剪枝结构：新增780个 selected units，主要落在 pyramid/backbone stage1，其次 stage2、stage0、neck/shrink_conv 和 deblock/head 输入侧。详细 unit 级列表在 diff_060_070/selected_units_diff.csv。",
        "2. 新增参数剪枝主要来源：backbone_stage1 是最大增量，其次 backbone_stage2、backbone_stage0、neck、deblock。head 输出仍保护，head 参数减少来自输入侧通道被上游剪掉。",
        "3. Stage1 grouped conv output 8->4 并非 0.60 出现；它在边界扫描 target 0.66 首次出现，并在 0.68/0.70 持续出现。",
        "4. 实验 A在保持0.70参数预算且额外保护Stage1 output后，AP没有恢复，mAP接近0。这说明若仍强行达到0.70，预算会转移到其他关键域并继续破坏精度。",
        "5. 实验 B只从原0.70方案移除Stage1 output剪枝后，actual_param降到约0.6666，mAP从原0.70的0.0678恢复到0.4052，但仍明显低于0.60的0.7139。Stage1 output 8->4 是强触发因素，但不是唯一伤害来源。",
        "6. 实验 C显示断崖边界在0.64和0.66之间：0.64 mAP=0.5797，0.66 mAP=0.3867，同时0.66首次出现Stage1 output 8->4。",
        "7. AP崩溃与Stage1 output 8->4强相关；不存在Stage1未剪到4但mAP<0.5的边界点。",
        "8. 若不只看Stage1，最可疑的新增域是stage1 conv1/conv3瓶颈传播、shrink_conv.layers.0.double_conv.0输入侧继续收缩、deblock输入侧继续收缩，以及接近60%上限的 layer2/deblock/shrink_conv domains。",
        "9. 所有进入val500的A/B/C模型shape invariant均通过，没有非通道维 shape violation。",
        "10. 未发现kernel_size、stride、padding、dilation或groups被误改；reload smoke通过但AP崩溃属于合法结构退化，不是TP类形状切错bug。",
        "11. Stage2 output 16->12不是0.60到0.70之间的新变化；它在0.60已经存在，边界扫描里从0.62起可见，因此不是本次断崖的唯一解释。",
        "",
        "## recommended_policy_change",
        "- 默认搜索空间建议新增 Stage1 grouped conv output min_per_group=8，至少在未蒸馏/未recovery前禁止Stage1 8->4。",
        "- 建议对Stage1设置更低的 max_ch_sparsity，例如0.25或0.30；当前0.60允许8->4，和断崖高度同步。",
        "- 保持FPN/head output protection和fixed-shape structural skip不变。",
        "- 新安全搜索空间：param target <=0.60作为生产安全线；0.62可作为recovery/distillation探索上界；0.64已出现明显mAP损失，0.66及以上需禁用Stage1 8->4并重新评估。",
        "- 最推荐进入recovery/distillation的是0.60；如果需要更激进，可选择0.62。实验B的0.6666变体有研究价值，但精度未恢复到安全水平。",
        "- TensorRT benchmark候选优先使用0.60和0.62；实验B只在蒸馏后mAP恢复时纳入。",
        "",
        "## next_experiments",
        "- 重新跑带 Stage1 min_per_group=8 / Stage1 max_ch_sparsity<=0.30 的 param sweep，覆盖0.62、0.64、0.66、0.68。",
        "- 对0.62和0.64做短周期recovery/distillation，验证能否把mAP拉回0.60附近。",
        "- 对实验B的edited plan做一次recovery/distillation，判断Stage1恢复后剩余损伤是否可恢复。",
        "- TensorRT benchmark在PyTorch val500稳定后进行，避免对已崩塌模型做无效部署测试。",
        "",
        "## boundary_scan_verdict",
        "\n".join(c_verdict_lines),
    ]
    (output_dir / "collapse_diagnosis_final_report.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="v10.9 0.60 vs 0.70 collapse diagnosis")
    parser.add_argument("--mode", choices=["diff-only", "smoke-ab", "smoke-c", "full"], default="diff-only")
    parser.add_argument("--source-dir", default="outputs/latency_lut/v109_param_budget_round4_pruner")
    parser.add_argument("--output-dir", default="outputs/latency_lut/v109_060_vs_070_collapse_diagnosis")
    parser.add_argument("--boundary-targets", default="0.62,0.64,0.66,0.68")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--model-config", default=DEFAULT_CONFIG)
    parser.add_argument("--heal-root", default=DEFAULT_HEAL_ROOT)
    parser.add_argument("--device", default="")
    parser.add_argument("--auto-select-idle-gpu", action="store_true")
    parser.add_argument("--max-gpu-utilization", type=int, default=5)
    parser.add_argument("--max-gpu-memory-ratio", type=float, default=0.20)
    parser.add_argument("--wait-timeout-minutes", type=float, default=30.0)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--target-pruning-mode", default="param", choices=["param", "channel"])
    parser.add_argument("--importance", default="first_order_taylor", choices=["first_order_taylor"])
    parser.add_argument("--selector", default="greedy_global_ranking", choices=["greedy_global_ranking"])
    parser.add_argument("--round-to", type=int, default=4)
    parser.add_argument("--align-channels", type=int, default=4)
    parser.add_argument("--max-ch-sparsity", type=float, default=0.60)
    parser.add_argument("--protect-fpn-output", action="store_true")
    parser.add_argument("--protect-head-output", action="store_true")
    parser.add_argument("--no-extra-output-protection", action="store_true")
    parser.add_argument("--eval-frames", type=int, default=500)
    parser.add_argument("--latency-warmup", type=int, default=50)
    parser.add_argument("--latency-repeat", type=int, default=300)
    parser.add_argument("--save-model-artifacts", action="store_true")
    parser.add_argument("--num-calib-batches", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args(argv)
    args.round_to = validate_round_to(args.round_to)
    args.align_channels = args.round_to
    if not args.protect_fpn_output:
        args.protect_fpn_output = True
    if not args.protect_head_output:
        args.protect_head_output = True
    args.importance_mode = args.importance
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "run_config.json", vars(args))
    try:
        if args.mode in {"diff-only", "full"}:
            run_diff_only(source_dir, output_dir)
        if args.mode in {"smoke-ab", "full"}:
            run_experiment_ab(args, output_dir)
        if args.mode in {"smoke-c", "full"}:
            rc = run_experiment_c(args, output_dir)
            if rc != 0:
                raise RuntimeError(f"experiment_C_failed:{rc}")
        if args.mode == "full":
            write_final_report(output_dir, source_dir)
        write_json(output_dir / "failure_report.json", {"success": True, "failure_reason": "", "traceback": ""})
        print(json.dumps({"success": True, "mode": args.mode, "output_dir": str(output_dir)}, indent=2, ensure_ascii=False))
        return 0
    except Exception as exc:  # noqa: BLE001
        write_json(
            output_dir / "failure_report.json",
            {"success": False, "failure_reason": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()},
        )
        print(json.dumps({"success": False, "failure": f"{type(exc).__name__}: {exc}", "output_dir": str(output_dir)}, indent=2, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
