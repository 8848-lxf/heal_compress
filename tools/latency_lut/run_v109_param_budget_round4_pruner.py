#!/usr/bin/env python3
"""v10.9 param-budget Taylor greedy pruner with configurable round_to."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[2]
_UNIAD = _ROOT.parent
for _p in (_UNIAD, _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from heal_compress.pruning.artifacts import load_v108_model_object, save_v108_model_artifacts  # noqa: E402
from heal_compress.pruning.greedy_budget_selector import select_greedy_global_budget  # noqa: E402
from heal_compress.pruning.model_io import collect_module_structure, load_heal_model, setup_logger  # noqa: E402
from heal_compress.pruning.shape_invariants import (  # noqa: E402
    check_model_shape_invariants,
    snapshot_model_shape_invariants,
)
from tools.latency_lut.run_v108_complete_taylor_greedy_pruner import (  # noqa: E402
    BASELINE_LABEL,
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIG,
    DEFAULT_HEAL_ROOT,
    EVAL_ENTRYPOINT,
    _apply_grouped_input_legality_filter,
    _build_global_physical_plan,
    _build_pruning_domains_and_reports,
    _channels_for_item,
    _contamination_report,
    _evaluate_baseline,
    _grouped_stage_summary,
    _load_eval_helpers,
    _select_device,
    _shape_changes,
    _skip_reason_counts,
    _smoke_real_batch,
    count_parameters,
    ensure_v108_eval_dir,
    parse_targets,
    percentile,
    target_dir_name,
    write_csv,
    write_json,
)
from tools.latency_lut.select_idle_gpu_for_latency import collect_gpu_state  # noqa: E402


SUPPORTED_ROUND_TO = {4, 8, 16, 32, 64, 128}


def validate_round_to(value: int | str) -> int:
    round_to = int(value)
    if round_to not in SUPPORTED_ROUND_TO or round_to <= 0 or (round_to & (round_to - 1)) != 0:
        raise ValueError(f"round_to_must_be_supported_power_of_two:{round_to}")
    return round_to


def _json_load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _param_cost_per_local_channel(module: nn.Module, direction: str) -> float:
    if isinstance(module, nn.Conv2d):
        kh, kw = int(module.weight.shape[2]), int(module.weight.shape[3])
        if direction == "out":
            return float(int(module.weight.shape[1]) * kh * kw + (1 if module.bias is not None else 0))
        out_per_group = int(module.out_channels // max(int(module.groups), 1))
        return float(out_per_group * kh * kw)
    if isinstance(module, nn.ConvTranspose2d):
        kh, kw = int(module.weight.shape[2]), int(module.weight.shape[3])
        if direction == "out":
            in_per_group = int(module.in_channels // max(int(module.groups), 1))
            return float(in_per_group * kh * kw + (1 if module.bias is not None else 0))
        out_per_group = int(module.out_channels // max(int(module.groups), 1))
        return float(out_per_group * kh * kw)
    if isinstance(module, nn.modules.batchnorm._BatchNorm):
        return float((1 if module.weight is not None else 0) + (1 if module.bias is not None else 0))
    if isinstance(module, nn.Linear):
        if direction == "out":
            return float(int(module.in_features) + (1 if module.bias is not None else 0))
        return float(int(module.out_features))
    return 0.0


def build_param_savings_by_unit(groups: Sequence[Any], domains: Sequence[Any]) -> dict[str, dict[int, float]]:
    """Estimate per-root-channel parameter savings without changing ranking order."""

    groups_by_id = {str(group.group_id): group for group in groups}
    savings: dict[str, dict[int, float]] = {}
    for domain in domains:
        scope = groups_by_id.get(str(domain.pruning_domain_id))
        if scope is None:
            continue
        domain_savings: dict[int, float] = {}
        all_group_indices = list(range(int(domain.num_root_channels)))
        for unit in domain.units:
            root_idx = int(unit.root_channel_index)
            group_keep = [idx for idx in all_group_indices if idx != root_idx]
            total = 0.0
            for item in getattr(scope, "items", []):
                local_total = _channels_for_item(item)
                if local_total <= 0:
                    continue
                local_keep = set(int(v) for v in item.local_keep(group_keep))
                local_pruned = [idx for idx in range(local_total) if idx not in local_keep]
                total += len(local_pruned) * _param_cost_per_local_channel(item.module, str(item.direction))
            domain_savings[root_idx] = total
        savings[str(domain.pruning_domain_id)] = domain_savings
    return savings


def _intersect_keep(store: dict[str, dict[str, set[int]]], module_name: str, axis: str, keep: Sequence[int]) -> None:
    axes = store.setdefault(module_name, {})
    keep_set = {int(v) for v in keep}
    if axis in axes:
        axes[axis] = axes[axis].intersection(keep_set)
    else:
        axes[axis] = keep_set


def _module_param_count_with_axes(module: nn.Module, axes: Mapping[str, set[int]]) -> int:
    if isinstance(module, nn.Conv2d):
        out_ch = len(axes["out"]) if "out" in axes else int(module.out_channels)
        in_ch = len(axes["in"]) if "in" in axes else int(module.in_channels)
        groups = int(module.groups)
        kh, kw = int(module.weight.shape[2]), int(module.weight.shape[3])
        weight = out_ch * max(in_ch // max(groups, 1), 0) * kh * kw
        bias = out_ch if module.bias is not None else 0
        return int(weight + bias)
    if isinstance(module, nn.ConvTranspose2d):
        out_ch = len(axes["out"]) if "out" in axes else int(module.out_channels)
        in_ch = len(axes["in"]) if "in" in axes else int(module.in_channels)
        groups = int(module.groups)
        kh, kw = int(module.weight.shape[2]), int(module.weight.shape[3])
        weight = in_ch * max(out_ch // max(groups, 1), 0) * kh * kw
        bias = out_ch if module.bias is not None else 0
        return int(weight + bias)
    if isinstance(module, nn.modules.batchnorm._BatchNorm):
        num = len(axes["out"]) if "out" in axes else int(module.num_features)
        return int((num if module.weight is not None else 0) + (num if module.bias is not None else 0))
    if isinstance(module, nn.GroupNorm):
        num = len(axes["out"]) if "out" in axes else int(module.num_channels)
        return int((num if module.weight is not None else 0) + (num if module.bias is not None else 0))
    if isinstance(module, nn.Linear):
        out_f = len(axes["out"]) if "out" in axes else int(module.out_features)
        in_f = len(axes["in"]) if "in" in axes else int(module.in_features)
        return int(out_f * in_f + (out_f if module.bias is not None else 0))
    return int(sum(param.numel() for param in module.parameters(recurse=False)))


def build_exact_param_ratio_predictor(
    model: nn.Module,
    groups: Sequence[Any],
    domains: Sequence[Any],
    params_before: int,
) -> Any:
    """Create a module-level parameter predictor for current domain plans."""

    modules = dict(model.named_modules())
    groups_by_id = {str(group.group_id): group for group in groups}
    domains_by_id = {str(domain.pruning_domain_id): domain for domain in domains}
    base_module_params = {name: int(sum(param.numel() for param in module.parameters(recurse=False))) for name, module in modules.items()}

    def _predict(domain_plans: Mapping[str, Any]) -> float:
        keep_by_module_axis: dict[str, dict[str, set[int]]] = {}
        for domain_id, plan in domain_plans.items():
            domain = domains_by_id.get(str(domain_id))
            scope = groups_by_id.get(str(domain_id))
            if domain is None or scope is None:
                continue
            prune_indices = sorted({int(v) for v in getattr(plan, "prune_indices", [])})
            if not prune_indices:
                continue
            keep_indices = list(getattr(plan, "keep_indices", []) or [])
            if not keep_indices:
                keep_indices = [idx for idx in range(int(domain.num_root_channels)) if idx not in set(prune_indices)]
            for item in getattr(scope, "items", []):
                module = item.module
                module_name = str(item.name)
                local_keep = sorted(int(v) for v in item.local_keep(keep_indices))
                if isinstance(module, nn.Conv2d) and int(module.groups) > 1 and not (
                    int(module.groups) == int(module.in_channels) == int(module.out_channels)
                ):
                    if str(item.direction) == "out":
                        _intersect_keep(keep_by_module_axis, module_name, "out", local_keep)
                        if int(module.in_channels) == int(module.out_channels):
                            _intersect_keep(keep_by_module_axis, module_name, "in", local_keep)
                    elif str(item.direction) == "in":
                        _intersect_keep(keep_by_module_axis, module_name, "in", local_keep)
                elif str(item.direction) == "out":
                    _intersect_keep(keep_by_module_axis, module_name, "out", local_keep)
                elif str(item.direction) == "in":
                    _intersect_keep(keep_by_module_axis, module_name, "in", local_keep)
        predicted_after = 0
        for name, module in modules.items():
            if name == "":
                continue
            axes = keep_by_module_axis.get(name, {})
            if axes:
                predicted_after += _module_param_count_with_axes(module, axes)
            else:
                predicted_after += base_module_params.get(name, 0)
        return 1.0 - predicted_after / max(int(params_before), 1)

    return _predict


def _unit_rows(domains: Sequence[Any], predicate: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for domain in domains:
        for unit in domain.units:
            if predicate(domain, unit):
                rows.append(unit.to_report_row())
    return rows


def _latency_row_v109(
    *,
    target: float,
    selection: Any,
    actual_param: float,
    rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    baseline_summary: Mapping[str, Any],
    contamination: bool,
) -> dict[str, Any]:
    forward_vals = [float(row.get("forward_time_ms", 0.0) or 0.0) for row in rows if row.get("success")]
    total_vals = [float(row.get("total_time_ms", 0.0) or 0.0) for row in rows if row.get("success")]
    f_p50 = percentile(forward_vals, 0.50)
    f_mean = round(sum(forward_vals) / len(forward_vals), 6) if forward_vals else 0.0
    t_p50 = percentile(total_vals, 0.50)
    t_mean = round(sum(total_vals) / len(total_vals), 6) if total_vals else 0.0
    base_f_p50 = float(baseline_summary.get("forward_time_p50_ms", 0.0) or 0.0)
    base_f_mean = float(baseline_summary.get("forward_time_mean_ms", 0.0) or 0.0)
    base_t_p50 = float(baseline_summary.get("total_time_p50_ms", 0.0) or 0.0)
    base_t_mean = float(baseline_summary.get("total_time_mean_ms", 0.0) or 0.0)
    return {
        "target": float(target),
        "target_pruning_mode": selection.target_pruning_mode,
        "round_to": getattr(selection, "round_to", ""),
        "actual_channel_prune_ratio_on_searchable_surface": selection.actual_channel_prune_ratio_on_searchable_surface,
        "predicted_param_prune_ratio": selection.predicted_param_prune_ratio,
        "actual_param_prune_ratio": actual_param,
        "param_prediction_error": actual_param - selection.predicted_param_prune_ratio,
        "evaluated_frames": int(summary.get("actual_frames", 0) or 0),
        "skipped_frames": int(summary.get("missing_frames", 0) or 0),
        "skip_reason_counts": _skip_reason_counts(rows),
        "forward_latency_p50": f_p50,
        "forward_latency_mean": f_mean,
        "forward_latency_p90": percentile(forward_vals, 0.90),
        "forward_latency_p95": percentile(forward_vals, 0.95),
        "total_latency_p50": t_p50,
        "total_latency_mean": t_mean,
        "total_latency_p90": percentile(total_vals, 0.90),
        "total_latency_p95": percentile(total_vals, 0.95),
        "speedup_forward_p50_vs_baseline": base_f_p50 / f_p50 if f_p50 > 0 else 0.0,
        "speedup_forward_mean_vs_baseline": base_f_mean / f_mean if f_mean > 0 else 0.0,
        "speedup_total_p50_vs_baseline": base_t_p50 / t_p50 if t_p50 > 0 else 0.0,
        "speedup_total_mean_vs_baseline": base_t_mean / t_mean if t_mean > 0 else 0.0,
        "latency_contamination_risk": bool(contamination),
    }


def _ap_row_v109(
    *,
    target: float,
    selection: Any,
    actual_param: float,
    summary: Mapping[str, Any],
    baseline_summary: Mapping[str, Any],
    failure_reason: str = "",
) -> dict[str, Any]:
    ap30 = float(summary.get("AP_0_30", 0.0) or 0.0)
    ap50 = float(summary.get("AP_0_50", 0.0) or 0.0)
    ap70 = float(summary.get("AP_0_70", 0.0) or 0.0)
    b_ap30 = float(baseline_summary.get("AP_0_30", 0.0) or 0.0)
    b_map = sum(float(baseline_summary.get(k, 0.0) or 0.0) for k in ("AP_0_30", "AP_0_50", "AP_0_70")) / 3.0
    map3 = (ap30 + ap50 + ap70) / 3.0
    return {
        "target": float(target),
        "target_pruning_mode": selection.target_pruning_mode,
        "actual_channel_prune_ratio_on_searchable_surface": selection.actual_channel_prune_ratio_on_searchable_surface,
        "predicted_param_prune_ratio": selection.predicted_param_prune_ratio,
        "actual_param_prune_ratio": actual_param,
        "param_prediction_error": actual_param - selection.predicted_param_prune_ratio,
        "evaluated_frames": int(summary.get("actual_frames", 0) or 0),
        "AP@0.30": round(ap30, 6),
        "AP@0.50": round(ap50, 6),
        "mAP": round(map3, 6),
        "AP_drop_vs_baseline": round(b_ap30 - ap30, 6),
        "mAP_drop_vs_baseline": round(b_map - map3, 6),
        "metric_helper_used": EVAL_ENTRYPOINT,
        "failure_reason": failure_reason or str(summary.get("first_failure", "") or ""),
    }


def _run_target_eval_v109(
    *,
    target: float,
    target_dir: Path,
    artifact_path: Path,
    selection: Any,
    actual_param: float,
    dataset: Any,
    loader: Any,
    device: torch.device,
    logger: Any,
    args: argparse.Namespace,
    baseline_summary: Mapping[str, Any],
    gpu_samples: list[dict[str, Any]],
    selected_index: int | None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    model = load_v108_model_object(artifact_path, device=device)
    smoke = _smoke_real_batch(model, loader, device)
    reload_row = {
        "model_object_path": str(artifact_path),
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
        checkpoint=str(artifact_path),
        metadata={"target_prune_ratio": target, "actual_param_prune_ratio": actual_param},
        model_type=f"target_{target:.2f}",
        dataset=dataset,
        loader=loader,
        device=device,
        round_id=int(round(target * 100)),
        max_frames=int(args.eval_frames),
        warmup_frames=int(args.latency_warmup),
        logger=logger,
    )
    write_csv(target_dir / "per_frame_latency.csv", rows)
    if selected_index is not None:
        gpu_samples.append({"sample_reason": f"after_target_{target:.2f}", **collect_gpu_state(selected_index)})
    contamination = bool(_contamination_report(gpu_samples, selected_index)["latency_contamination_risk"])
    latency = _latency_row_v109(
        target=target,
        selection=selection,
        actual_param=actual_param,
        rows=rows,
        summary=summary,
        baseline_summary=baseline_summary,
        contamination=contamination,
    )
    ap = _ap_row_v109(
        target=target,
        selection=selection,
        actual_param=actual_param,
        summary=summary,
        baseline_summary=baseline_summary,
    )
    write_csv(target_dir / "real_val500_latency.csv", [latency])
    write_json(target_dir / "real_val500_ap.json", ap)
    return latency, ap, reload_row


def _manifest_for_target_v109(
    *,
    args: argparse.Namespace,
    target: float,
    selection: Any,
    params_before: int,
    params_after: int,
    shape_changes: Sequence[Mapping[str, Any]],
    grouped_rows: Sequence[Mapping[str, Any]],
    shape_report_path: Path,
    protected_units: Sequence[Mapping[str, Any]],
    fixed_shape_skipped_units: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    actual_param = 1.0 - params_after / max(params_before, 1)
    return {
        "target_pruning_mode": args.target_pruning_mode,
        "target_pruning_ratio": float(target),
        "round_to": int(args.round_to),
        "max_ch_sparsity": float(args.max_ch_sparsity),
        "actual_channel_prune_ratio_on_searchable_surface": selection.actual_channel_prune_ratio_on_searchable_surface,
        "predicted_param_prune_ratio": selection.predicted_param_prune_ratio,
        "actual_param_prune_ratio": actual_param,
        "param_prediction_error": actual_param - selection.predicted_param_prune_ratio,
        "param_budget_overshoot_ratio": max(0.0, actual_param - float(target)) if args.target_pruning_mode == "param" else 0.0,
        "before_after_shapes": list(shape_changes),
        "shape_invariant_report_path": str(shape_report_path),
        "grouped_stage0_1_2_summary": _grouped_stage_summary(grouped_rows),
        "selected_units": [unit.to_report_row() for unit in selection.selected_units],
        "skipped_units": [unit.to_report_row() for unit in selection.skipped_units],
        "protected_units": list(protected_units),
        "fixed_shape_skipped_units": list(fixed_shape_skipped_units),
        "max_ch_sparsity_blocked_count": selection.max_ch_sparsity_blocked_count,
        "skipped_because_pergroup_below_round_to_count": selection.skipped_because_pergroup_below_round_to_count,
        "unreachable": selection.unreachable,
        "params_before": int(params_before),
        "params_after": int(params_after),
        "importance": "first_order_taylor",
        "selector": "greedy_global_ranking",
        "protected_fpn_output": True,
        "protected_head_output": True,
        "additional_output_protection": False,
        "requires_architecture_patch": True,
    }


def _fmt(value: Any, digits: int = 6) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _load_v108_rows() -> list[dict[str, str]]:
    path = Path("outputs/latency_lut/v108_complete_taylor_greedy_pruner/v108_summary.csv")
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _verdict_lines_v109(summary_rows: Sequence[Mapping[str, Any]], failure: str) -> list[str]:
    lines = [
        "# v10.9 Param Budget Round4 Pruner Verdict",
        "",
        f"failure: {failure or 'none'}",
        "",
        "## Answers",
    ]
    if not summary_rows:
        lines.append("No target rows were completed.")
        return lines
    mode_values = sorted({str(row.get("target_pruning_mode", "")) for row in summary_rows})
    round_values = sorted({str(row.get("round_to", "")) for row in summary_rows})
    lines.append(f"1. target_pruning_mode is implemented; parser default is param; observed modes={mode_values}.")
    lines.append("2. --round-to validates supported powers of two: 4/8/16/32/64/128; --align-channels remains a fallback.")
    lines.append(f"3. this run used target_pruning_mode={mode_values} and round_to={round_values}.")
    lines.append("4. actual_param_prune_ratio by target:")
    for row in summary_rows:
        lines.append(f"- target {_fmt(row.get('target_pruning_ratio'), 2)}: actual_param={_fmt(row.get('actual_param_prune_ratio'))}, error_vs_target={float(row.get('actual_param_prune_ratio') or 0.0) - float(row.get('target_pruning_ratio') or 0.0):.6f}")
    lines.append("5. actual_channel_prune_ratio by target:")
    for row in summary_rows:
        lines.append(f"- target {_fmt(row.get('target_pruning_ratio'), 2)}: channel={_fmt(row.get('actual_channel_prune_ratio_on_searchable_surface'))}")
    lines.append("6. param prediction error:")
    for row in summary_rows:
        lines.append(f"- target {_fmt(row.get('target_pruning_ratio'), 2)}: predicted={_fmt(row.get('predicted_param_prune_ratio'))}, actual={_fmt(row.get('actual_param_prune_ratio'))}, error={_fmt(row.get('param_prediction_error'))}")
    last = summary_rows[-1]
    lines.append(f"7. Stage0 output per_group: {last.get('stage0_out_per_group_before_after')} (expected unchanged for round_to=4).")
    lines.append(f"8. Stage1 output per_group: {last.get('stage1_out_per_group_before_after')} (8->4 indicates round_to=4 grouped output pruning).")
    lines.append(f"9. Stage2 output per_group: {last.get('stage2_out_per_group_before_after')}; 16->4 is blocked by max_ch_sparsity=0.60 when selected.")
    lines.append(f"10. grouped input per_group: Stage0 {last.get('stage0_in_per_group_before_after')}, Stage1 {last.get('stage1_in_per_group_before_after')}, Stage2 {last.get('stage2_in_per_group_before_after')}; legality is captured by shape/grouped reports.")
    shape_bad = [row.get("target_pruning_ratio") for row in summary_rows if str(row.get("shape_invariant_passed")) != "True"]
    lines.append(f"11. non-channel shape changes: {'none' if not shape_bad else shape_bad}.")
    kernel_bad = [row.get("target_pruning_ratio") for row in summary_rows if int(row.get("non_channel_shape_violation_count") or 0) > 0]
    lines.append(f"12. kernel/stride/padding/dilation/groups mis-prune bug: {'not found' if not kernel_bad else kernel_bad}.")
    lines.append(f"13. forward p50 speedup targets: {[ _fmt(r.get('target_pruning_ratio'), 2) for r in summary_rows if float(r.get('speedup_forward_p50_vs_baseline') or 0.0) > 1.0 ] or 'none'}.")
    lines.append(f"14. forward mean speedup targets: {[ _fmt(r.get('target_pruning_ratio'), 2) for r in summary_rows if float(r.get('speedup_forward_mean_vs_baseline') or 0.0) > 1.0 ] or 'none'}.")
    lines.append(f"15. total p50 speedup targets: {[ _fmt(r.get('target_pruning_ratio'), 2) for r in summary_rows if float(r.get('speedup_total_p50_vs_baseline') or 0.0) > 1.0 ] or 'none'}; total mean speedup targets: {[ _fmt(r.get('target_pruning_ratio'), 2) for r in summary_rows if float(r.get('speedup_total_mean_vs_baseline') or 0.0) > 1.0 ] or 'none'}.")
    lines.append("16. AP/mAP drops:")
    for row in summary_rows:
        lines.append(f"- target {_fmt(row.get('target_pruning_ratio'), 2)}: AP30_drop={_fmt(row.get('AP_drop_vs_baseline'))}, mAP_drop={_fmt(row.get('mAP_drop_vs_baseline'))}")
    v108_rows = _load_v108_rows()
    if v108_rows:
        v109_err = sum(abs(float(r.get("actual_param_prune_ratio") or 0.0) - float(r.get("target_pruning_ratio") or 0.0)) for r in summary_rows) / len(summary_rows)
        v108_err = sum(abs(float(r.get("actual_param_prune_ratio") or 0.0) - float(r.get("target_pruning_ratio") or 0.0)) for r in v108_rows) / len(v108_rows)
        lines.append(f"17. vs v10.8 channel-target round8: mean abs param-budget error v10.9={v109_err:.6f}, v10.8={v108_err:.6f}.")
    else:
        lines.append("17. vs v10.8 channel-target round8: v10.8 summary not found for direct comparison.")
    candidates = [
        row
        for row in summary_rows
        if str(row.get("shape_invariant_passed")) == "True"
        and float(row.get("speedup_forward_p50_vs_baseline") or 0.0) > 1.0
        and float(row.get("mAP_drop_vs_baseline") or 1.0) <= 0.01
    ]
    best = max(candidates or summary_rows, key=lambda r: float(r.get("actual_param_prune_ratio") or 0.0))
    lines.append(f"18. recommended recovery/distillation target: {_fmt(best.get('target_pruning_ratio'), 2)}.")
    lines.append("19. TensorRT benchmark: recommended after recovery/distillation for the best AP/speed target, with adjacent targets as controls.")
    return lines


def run_v109(args: argparse.Namespace) -> int:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "run_config.json", vars(args))
    gpu_samples: list[dict[str, Any]] = []
    selected_index: int | None = None
    failure = ""
    attempted_eval: list[str] = []
    summary_rows: list[dict[str, Any]] = []
    all_latency: list[dict[str, Any]] = []
    all_ap: list[dict[str, Any]] = []
    all_reload: list[dict[str, Any]] = []
    all_budget_trace: list[dict[str, Any]] = []
    shape_summaries: list[dict[str, Any]] = []
    try:
        device, selected_index = _select_device(args, out_dir)
        if selected_index is not None:
            before_gpu = collect_gpu_state(selected_index)
            write_json(out_dir / "gpu_state_before.json", before_gpu)
            gpu_samples.append({"sample_reason": "before", **before_gpu})

        logger = setup_logger(out_dir)
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
            out_dir,
            device,
        )
        param_savings_by_unit = build_param_savings_by_unit(groups, domains)
        exact_param_ratio_predictor = build_exact_param_ratio_predictor(baseline, groups, domains, params_before)
        protected_units = _unit_rows(domains, lambda domain, unit: bool(domain.protected_reason or unit.protected_reason))
        fixed_shape_skipped_units = _unit_rows(domains, lambda domain, unit: str(unit.skipped_reason) == "fixed_shape_interface_structural_illegal")

        HEALLiDARAdapter, build_dataset, evaluate_one_model, setup_eval_logger, attempted_eval = _load_eval_helpers()
        args.evaluate_one_model = evaluate_one_model
        eval_logger = setup_eval_logger(ensure_v108_eval_dir(out_dir / "real_val500"))
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
            out_dir=out_dir,
        )

        for target in parse_targets(args.targets):
            target_dir = out_dir / target_dir_name(target)
            target_dir.mkdir(parents=True, exist_ok=True)
            try:
                selection = select_greedy_global_budget(
                    copy.deepcopy(domains),
                    target_pruning_ratio=float(target),
                    target_pruning_mode=args.target_pruning_mode,
                    predicted_total_params=float(params_before),
                    param_savings_by_unit=param_savings_by_unit,
                    param_ratio_from_plans=exact_param_ratio_predictor,
                    max_ch_sparsity=float(args.max_ch_sparsity),
                    align_channels=int(args.round_to),
                )
                selection.round_to = int(args.round_to)
                grouped_input_reports = _apply_grouped_input_legality_filter(
                    groups,
                    selection,
                    align_channels=int(args.round_to),
                )
                if grouped_input_reports:
                    selection.grouped_shape_rows.extend(grouped_input_reports)
                write_csv(target_dir / "budget_selection_trace.csv", selection.trace_rows)
                write_json(target_dir / "grouped_pergroup8_shape_report.json", selection.grouped_shape_rows)
                all_budget_trace.extend(selection.trace_rows)

                pruned = copy.deepcopy(baseline).to(device).eval()
                global_plan = _build_global_physical_plan(groups, selection, align_channels=int(args.round_to))
                write_json(target_dir / "global_physical_prune_plan.json", global_plan.to_json())
                surgery = global_plan.apply_one_shot(pruned)
                write_json(target_dir / "one_shot_surgery_report.json", surgery)
                params_after = count_parameters(pruned)
                actual_param = 1.0 - params_after / max(params_before, 1)
                selection.actual_param_prune_ratio = actual_param
                selection.param_prediction_error = actual_param - selection.predicted_param_prune_ratio
                selection.param_budget_overshoot_ratio = max(0.0, actual_param - float(target)) if args.target_pruning_mode == "param" else 0.0
                structure_after = collect_module_structure(pruned)
                shape_changes = _shape_changes(structure_before, structure_after)
                shape_report = check_model_shape_invariants(invariant_before, snapshot_model_shape_invariants(pruned))
                shape_report_path = target_dir / "shape_invariant_report.json"
                write_json(shape_report_path, shape_report["rows"])
                shape_summary = {
                    "target": float(target),
                    "shape_invariant_passed": bool(shape_report["passed"]),
                    "non_channel_shape_violation_count": int(shape_report["non_channel_shape_violation_count"]),
                    "shape_invariant_report_path": str(shape_report_path),
                }
                shape_summaries.append(shape_summary)
                if not shape_report["passed"]:
                    raise RuntimeError(f"shape_invariant_failed:{shape_report['non_channel_shape_violation_count']}")

                manifest = _manifest_for_target_v109(
                    args=args,
                    target=target,
                    selection=selection,
                    params_before=params_before,
                    params_after=params_after,
                    shape_changes=shape_changes,
                    grouped_rows=selection.grouped_shape_rows,
                    shape_report_path=shape_report_path,
                    protected_units=protected_units,
                    fixed_shape_skipped_units=fixed_shape_skipped_units,
                )
                artifacts = save_v108_model_artifacts(
                    model=pruned,
                    models_dir=target_dir / "models",
                    manifest=manifest,
                    model_config=args.model_config,
                    checkpoint_source=args.checkpoint,
                )
                write_json(target_dir / "models" / "manifest.json", manifest)
                latency, ap, reload_row = _run_target_eval_v109(
                    target=target,
                    target_dir=target_dir,
                    artifact_path=artifacts["model_object"],
                    selection=selection,
                    actual_param=actual_param,
                    dataset=dataset,
                    loader=loader,
                    device=device,
                    logger=eval_logger,
                    args=args,
                    baseline_summary=baseline_summary,
                    gpu_samples=gpu_samples,
                    selected_index=selected_index,
                )
                all_latency.append(latency)
                all_ap.append(ap)
                all_reload.append({"target": target, **reload_row})
                write_json(target_dir / "failure_report.json", {"success": True, "failure_reason": "", "traceback": ""})
                stage = _grouped_stage_summary(selection.grouped_shape_rows)
                row = {
                    "target_pruning_mode": args.target_pruning_mode,
                    "target_pruning_ratio": float(target),
                    "round_to": int(args.round_to),
                    "actual_channel_prune_ratio_on_searchable_surface": selection.actual_channel_prune_ratio_on_searchable_surface,
                    "predicted_param_prune_ratio": selection.predicted_param_prune_ratio,
                    "actual_param_prune_ratio": actual_param,
                    "param_prediction_error": actual_param - selection.predicted_param_prune_ratio,
                    "params_before": params_before,
                    "params_after": params_after,
                    "forward_p50_ms": latency["forward_latency_p50"],
                    "forward_mean_ms": latency["forward_latency_mean"],
                    "forward_p90_ms": latency["forward_latency_p90"],
                    "forward_p95_ms": latency["forward_latency_p95"],
                    "total_p50_ms": latency["total_latency_p50"],
                    "total_mean_ms": latency["total_latency_mean"],
                    "total_p90_ms": latency["total_latency_p90"],
                    "total_p95_ms": latency["total_latency_p95"],
                    "speedup_forward_p50_vs_baseline": latency["speedup_forward_p50_vs_baseline"],
                    "speedup_forward_mean_vs_baseline": latency["speedup_forward_mean_vs_baseline"],
                    "speedup_total_p50_vs_baseline": latency["speedup_total_p50_vs_baseline"],
                    "speedup_total_mean_vs_baseline": latency["speedup_total_mean_vs_baseline"],
                    "AP@0.30": ap["AP@0.30"],
                    "AP@0.50": ap["AP@0.50"],
                    "mAP": ap["mAP"],
                    "AP_drop_vs_baseline": ap["AP_drop_vs_baseline"],
                    "mAP_drop_vs_baseline": ap["mAP_drop_vs_baseline"],
                    "stage0_out_per_group_before_after": stage.get("stage0_like", {}).get("out_per_group_before_after", ""),
                    "stage0_in_per_group_before_after": stage.get("stage0_like", {}).get("in_per_group_before_after", ""),
                    "stage1_out_per_group_before_after": stage.get("stage1_like", {}).get("out_per_group_before_after", ""),
                    "stage1_in_per_group_before_after": stage.get("stage1_like", {}).get("in_per_group_before_after", ""),
                    "stage2_out_per_group_before_after": stage.get("stage2_like", {}).get("out_per_group_before_after", ""),
                    "stage2_in_per_group_before_after": stage.get("stage2_like", {}).get("in_per_group_before_after", ""),
                    "shape_invariant_passed": True,
                    "non_channel_shape_violation_count": 0,
                    "max_ch_sparsity_blocked_count": selection.max_ch_sparsity_blocked_count,
                    "skipped_because_pergroup_below_round_to_count": selection.skipped_because_pergroup_below_round_to_count,
                    "unreachable": selection.unreachable,
                    "model_artifact_path": str(artifacts["model_object"]),
                    "latency_contamination_risk": latency["latency_contamination_risk"],
                    "verdict": "invalid_latency" if latency["latency_contamination_risk"] else ("unreachable" if selection.unreachable else "ok"),
                }
                summary_rows.append(row)
            except Exception as exc:  # noqa: BLE001
                tb = traceback.format_exc()
                reason = f"{type(exc).__name__}: {exc}"
                write_json(target_dir / "failure_report.json", {"success": False, "failure_reason": reason, "traceback": tb})
                write_json(
                    target_dir / "real_val500_ap.json",
                    {
                        "target": float(target),
                        "evaluated_frames": 0,
                        "AP@0.30": None,
                        "AP@0.50": None,
                        "mAP": None,
                        "failure_reason": reason,
                        "attempted_eval_entrypoints": attempted_eval,
                        "traceback": tb,
                    },
                )
                summary_rows.append(
                    {
                        "target_pruning_mode": args.target_pruning_mode,
                        "target_pruning_ratio": float(target),
                        "round_to": int(args.round_to),
                        "actual_channel_prune_ratio_on_searchable_surface": "",
                        "predicted_param_prune_ratio": "",
                        "actual_param_prune_ratio": "",
                        "param_prediction_error": "",
                        "params_before": params_before,
                        "params_after": "",
                        "forward_p50_ms": "",
                        "forward_mean_ms": "",
                        "forward_p90_ms": "",
                        "forward_p95_ms": "",
                        "total_p50_ms": "",
                        "total_mean_ms": "",
                        "total_p90_ms": "",
                        "total_p95_ms": "",
                        "speedup_forward_p50_vs_baseline": "",
                        "speedup_forward_mean_vs_baseline": "",
                        "speedup_total_p50_vs_baseline": "",
                        "speedup_total_mean_vs_baseline": "",
                        "AP@0.30": "",
                        "AP@0.50": "",
                        "mAP": "",
                        "AP_drop_vs_baseline": "",
                        "mAP_drop_vs_baseline": "",
                        "stage0_out_per_group_before_after": "",
                        "stage0_in_per_group_before_after": "",
                        "stage1_out_per_group_before_after": "",
                        "stage1_in_per_group_before_after": "",
                        "stage2_out_per_group_before_after": "",
                        "stage2_in_per_group_before_after": "",
                        "shape_invariant_passed": False if "shape_invariant_failed" in reason else "",
                        "non_channel_shape_violation_count": "",
                        "max_ch_sparsity_blocked_count": "",
                        "skipped_because_pergroup_below_round_to_count": "",
                        "unreachable": "",
                        "verdict": f"failed:{reason}",
                    }
                )
        if selected_index is not None:
            after_gpu = collect_gpu_state(selected_index)
            write_json(out_dir / "gpu_state_after.json", after_gpu)
            gpu_samples.append({"sample_reason": "after", **after_gpu})
        write_json(out_dir / "gpu_state_during_samples.json", gpu_samples)
        write_json(out_dir / "gpu_contamination_report.json", _contamination_report(gpu_samples, selected_index))
        write_csv(out_dir / "budget_selection_trace.csv", all_budget_trace)
        write_csv(out_dir / "real_val500_latency.csv", all_latency)
        write_json(out_dir / "real_val500_ap.json", all_ap)
        write_json(out_dir / "reload_report.json", all_reload)
        write_json(out_dir / "shape_invariant_summary.json", shape_summaries)
        write_csv(out_dir / "v109_summary.csv", summary_rows)
        (out_dir / "v109_param_budget_round4_pruner_verdict.md").write_text(
            "\n".join(_verdict_lines_v109(summary_rows, failure)) + "\n",
            encoding="utf-8",
        )
        write_json(out_dir / "failure_report.json", {"success": True, "failure_reason": "", "traceback": "", "attempted_eval_entrypoints": attempted_eval})
        print(json.dumps({"success": True, "output_dir": str(out_dir)}, indent=2, ensure_ascii=False))
        return 0
    except Exception as exc:  # noqa: BLE001
        failure = f"{type(exc).__name__}: {exc}"
        tb = traceback.format_exc()
        write_json(out_dir / "failure_report.json", {"success": False, "failure_reason": failure, "traceback": tb, "attempted_eval_entrypoints": attempted_eval})
        write_csv(out_dir / "v109_summary.csv", summary_rows)
        write_json(out_dir / "shape_invariant_summary.json", shape_summaries)
        (out_dir / "v109_param_budget_round4_pruner_verdict.md").write_text(
            "\n".join(_verdict_lines_v109(summary_rows, failure)) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"success": False, "failure": failure, "output_dir": str(out_dir)}, indent=2, ensure_ascii=False))
        return 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="v10.9 param-budget round_to pruner")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--model-config", default=DEFAULT_CONFIG)
    parser.add_argument("--heal-root", default=DEFAULT_HEAL_ROOT)
    parser.add_argument("--device", default="")
    parser.add_argument("--auto-select-idle-gpu", action="store_true")
    parser.add_argument("--max-gpu-utilization", type=int, default=5)
    parser.add_argument("--max-gpu-memory-ratio", type=float, default=0.20)
    parser.add_argument("--wait-timeout-minutes", type=float, default=30.0)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--targets", default="0.10,0.20,0.30,0.50,0.60,0.70")
    parser.add_argument("--target-pruning-mode", default="param", choices=["param", "channel"])
    parser.add_argument("--importance", default="first_order_taylor", choices=["first_order_taylor"])
    parser.add_argument("--selector", default="greedy_global_ranking", choices=["greedy_global_ranking"])
    parser.add_argument("--align-channels", type=int, default=8)
    parser.add_argument("--round-to", type=int, default=None)
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
    parser.add_argument("--output-dir", default="outputs/latency_lut/v109_param_budget_round4_pruner")
    args = parser.parse_args(argv)
    effective_round_to = args.round_to if args.round_to is not None else args.align_channels
    args.round_to = validate_round_to(effective_round_to)
    args.align_channels = args.round_to
    if not args.protect_fpn_output:
        args.protect_fpn_output = True
    if not args.protect_head_output:
        args.protect_head_output = True
    args.importance_mode = args.importance
    return args


def main(argv: list[str] | None = None) -> int:
    return run_v109(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
