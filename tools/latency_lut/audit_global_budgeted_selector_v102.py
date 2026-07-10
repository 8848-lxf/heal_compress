#!/usr/bin/env python3
"""Audit v10.1/v10.2 global budgeted selector behavior.

The audit intentionally replays the original v10.1 greedy selector ordering so
that overshoot cases such as A1_010 can be explained from the sorted candidate
trace.  It does not run physical pruning or evaluation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import torch

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[2]
_UNIAD = _ROOT.parent
for _p in (_UNIAD, _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from heal_compress.pruning.full_model_surface import apply_full_model_prunable_surface  # noqa: E402
from heal_compress.pruning.propagation import GroupBuilder  # noqa: E402
from heal_compress.search.importance import compute_group_importance, compute_scope_channel_importance_map  # noqa: E402
from heal_compress.tracer.generic_tracer import trace_model  # noqa: E402
from heal_compress.tracer.op_graph import build_op_graph  # noqa: E402
from heal_compress.utils.model_utils import resolve_device  # noqa: E402
from heal_compress.pruning.model_io import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIG,
    DEFAULT_HEAL_ROOT,
    build_importance_calibration_data,
    build_protected_layers,
    configure_grouped_conv_pruning_fns,
    load_heal_model,
    move_batch_to_device,
    setup_logger as setup_prune_logger,
)
from tools.latency_lut.run_global_budgeted_alignment_eval_v101 import (  # noqa: E402
    BudgetCandidate,
    POLICY_TO_MODE,
    build_budget_candidates,
    build_estimate_actual_gap_report_v102,
    parse_csv_floats,
    parse_strategy_spec,
    ratio_tag,
    write_csv,
    write_json,
)
from tools.latency_lut.run_abcd_small_eval_v100 import build_model_args_for_strategy, compute_param_inventory, setup_v100_logger  # noqa: E402


def _old_v101_sorted(candidates: Sequence[BudgetCandidate]) -> list[BudgetCandidate]:
    return sorted(candidates, key=lambda item: (item.score, item.importance_score, -item.param_saving_if_removed, item.candidate_id))


def _candidate_row(cand: BudgetCandidate, rank: int) -> dict[str, Any]:
    meta = cand.metadata or {}
    return {
        "rank_by_score": rank,
        "unit_id": cand.candidate_id,
        "domain_id": cand.domain_id,
        "strategy_policy": cand.strategy_policy,
        "affected_modules": ";".join(cand.affected_modules),
        "importance_score": cand.importance_score,
        "param_saving_if_removed": cand.param_saving_if_removed,
        "flops_saving_if_removed": cand.flops_saving_if_removed,
        "score": cand.score,
        "legality_status": cand.legality_status,
        "alignment_status": cand.alignment_status,
        "reject_reason_if_any": cand.reject_reason_if_any,
        "contains_grouped_conv": bool(meta.get("contains_grouped_conv", False)),
        "contains_grouped_conv_output": bool(meta.get("contains_grouped_conv_output", False)),
        "contains_ordinary_conv": bool(meta.get("contains_ordinary_conv", False)),
        "contains_bn": bool(meta.get("contains_bn", False)),
        "contains_residual": bool(meta.get("contains_residual", False)),
        "contains_concat": bool(meta.get("contains_concat", False)),
    }


def replay_v101_selector(
    sorted_candidates: Sequence[BudgetCandidate],
    *,
    target_saving: float,
    baseline_total_params: int,
    target_ratio: float,
) -> tuple[list[BudgetCandidate], list[dict[str, Any]], dict[str, Any]]:
    selected: list[BudgetCandidate] = []
    trace: list[dict[str, Any]] = []
    used_domains: set[str] = set()
    current = 0.0
    rank = {cand.candidate_id: idx + 1 for idx, cand in enumerate(sorted_candidates)}
    for cand in sorted_candidates:
        if cand.legality_status != "legal" or cand.param_saving_if_removed <= 0:
            continue
        if current >= target_saving:
            break
        if cand.domain_id in used_domains:
            continue
        before = current
        after = before + float(cand.param_saving_if_removed)
        selected.append(cand)
        used_domains.add(cand.domain_id)
        current = after
        trace.append(
            {
                "select_step": len(selected),
                "rank_by_score": rank[cand.candidate_id],
                "unit_id": cand.candidate_id,
                "score": cand.score,
                "importance_score": cand.importance_score,
                "param_saving_if_removed": cand.param_saving_if_removed,
                "cumulative_estimated_param_saving_before": before,
                "cumulative_estimated_param_saving_after": after,
                "cumulative_estimated_param_prune_ratio_before": before / baseline_total_params,
                "cumulative_estimated_param_prune_ratio_after": after / baseline_total_params,
                "target_param_prune_ratio_full_model": target_ratio,
                "budget_lower_bound": target_ratio,
                "budget_upper_bound": target_ratio,
                "selected_reason": "v10.1_greedy_accept_until_estimated_target_reached",
                "would_overshoot_budget": after > target_saving,
                "accepted_despite_overshoot": after > target_saving,
                "large_atomic_unit_jump": before < target_saving and after / baseline_total_params > target_ratio + 0.03,
                "contains_grouped_conv_output": bool((cand.metadata or {}).get("contains_grouped_conv_output", False)),
            }
        )
    next_candidate = None
    for cand in sorted_candidates:
        if cand.domain_id not in used_domains and cand not in selected and cand.legality_status == "legal" and cand.param_saving_if_removed > 0:
            next_candidate = cand
            break
    estimated_ratio = current / baseline_total_params
    if estimated_ratio < target_ratio:
        status = "under_target"
    elif estimated_ratio > target_ratio + 0.03:
        status = "overshoot"
    else:
        status = "in_budget_window"
    stop = {
        "target_param_prune_ratio_full_model": target_ratio,
        "target_param_saving_abs": target_saving,
        "estimated_param_saving_selected": current,
        "estimated_param_prune_ratio_selected": estimated_ratio,
        "stop_condition": "estimated_target_reached" if current >= target_saving else "candidate_exhausted",
        "stopped_at_step": len(selected),
        "next_candidate_if_any": next_candidate.candidate_id if next_candidate else "",
        "next_candidate_param_saving": next_candidate.param_saving_if_removed if next_candidate else 0,
        "next_candidate_would_overshoot": bool(next_candidate and current + next_candidate.param_saving_if_removed > target_saving),
        "budget_status": status,
    }
    return selected, trace, stop


def _load_actual_ratio(v101_dir: Path, exp_name: str) -> tuple[float | None, int | None, int | None]:
    report = v101_dir / exp_name / "target_budget_achievement_report.json"
    if not report.exists():
        return None, None, None
    data = json.loads(report.read_text(encoding="utf-8"))
    return (
        float(data.get("actual_param_prune_ratio_full_model", 0.0) or 0.0),
        int(data.get("baseline_total_params", 0) or 0),
        int(data.get("pruned_total_params", 0) or 0),
    )


def _load_recorded_selected(v101_dir: Path, exp_name: str) -> list[dict[str, Any]]:
    report = v101_dir / exp_name / "selected_coupled_units_report.json"
    if not report.exists():
        return []
    try:
        data = json.loads(report.read_text(encoding="utf-8"))
    except Exception:
        return []
    return list(data.get("selected_candidates", []) or [])


def _trace_from_recorded(
    selected_rows: Sequence[dict[str, Any]],
    *,
    rank: dict[str, int],
    target_saving: float,
    baseline_total_params: int,
    target_ratio: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    trace: list[dict[str, Any]] = []
    current = 0.0
    for step, row in enumerate(selected_rows, start=1):
        before = current
        saving = float(row.get("param_saving_if_removed", 0.0) or 0.0)
        after = before + saving
        current = after
        trace.append(
            {
                "select_step": step,
                "rank_by_score": rank.get(str(row.get("candidate_id", "")), -1),
                "unit_id": row.get("candidate_id", ""),
                "score": row.get("score", 0.0),
                "importance_score": row.get("importance_score", 0.0),
                "param_saving_if_removed": saving,
                "cumulative_estimated_param_saving_before": before,
                "cumulative_estimated_param_saving_after": after,
                "cumulative_estimated_param_prune_ratio_before": before / baseline_total_params,
                "cumulative_estimated_param_prune_ratio_after": after / baseline_total_params,
                "target_param_prune_ratio_full_model": target_ratio,
                "budget_lower_bound": target_ratio,
                "budget_upper_bound": target_ratio,
                "selected_reason": "recorded_v10.1_selected_candidate",
                "would_overshoot_budget": after > target_saving,
                "accepted_despite_overshoot": after > target_saving,
                "large_atomic_unit_jump": before < target_saving and after / baseline_total_params > target_ratio + 0.03,
                "contains_grouped_conv_output": bool((row.get("metadata", {}) or {}).get("contains_grouped_conv_output", False)),
            }
        )
    estimated_ratio = current / baseline_total_params
    if estimated_ratio < target_ratio:
        status = "under_target"
    elif estimated_ratio > target_ratio + 0.03:
        status = "overshoot"
    else:
        status = "in_budget_window"
    return trace, {
        "target_param_prune_ratio_full_model": target_ratio,
        "target_param_saving_abs": target_saving,
        "estimated_param_saving_selected": current,
        "estimated_param_prune_ratio_selected": estimated_ratio,
        "stop_condition": "recorded_v10.1_selection_loaded",
        "stopped_at_step": len(trace),
        "next_candidate_if_any": "",
        "next_candidate_param_saving": 0,
        "next_candidate_would_overshoot": False,
        "budget_status": status,
    }


def _write_monotonicity(out: Path, strategy: str, selected_by_ratio: dict[str, set[str]]) -> None:
    tags = sorted(selected_by_ratio)
    report: dict[str, Any] = {"strategy": strategy, "ratios": tags}
    if len(tags) >= 2:
        for prev, cur in zip(tags, tags[1:]):
            missing = sorted(selected_by_ratio[prev] - selected_by_ratio[cur])
            report[f"S_{prev}_subset_S_{cur}"] = not missing
            report[f"units_in_{prev}_not_in_{cur}"] = missing
        report["monotonic"] = all(report.get(f"S_{prev}_subset_S_{cur}", False) for prev, cur in zip(tags, tags[1:]))
        report["non_monotonic_reason"] = "" if report["monotonic"] else "v10.1 greedy target-specific bundle choice; candidate set stable but selected bundle size differs by target"
    else:
        report["monotonic"] = None
        report["non_monotonic_reason"] = "single_target_only"
    write_json(out / f"{strategy}_selected_set_monotonicity_report.json", report)


def run_strategy(args: argparse.Namespace, strategy_name: str, ratios: Sequence[float], device: torch.device, root_out: Path) -> None:
    spec = parse_strategy_spec(strategy_name)
    (root_out / f"logs_{strategy_name}").mkdir(parents=True, exist_ok=True)
    (root_out / f"prune_{strategy_name}").mkdir(parents=True, exist_ok=True)
    logger = setup_v100_logger(root_out / f"logs_{strategy_name}")
    model_args = build_model_args_for_strategy(args)
    prune_logger = setup_prune_logger(root_out / f"prune_{strategy_name}")
    model, adapter = load_heal_model(model_args, device, prune_logger)
    try:
        inventory = compute_param_inventory(model)
        sample = adapter.build_synthetic_batch(model)
        trace = trace_model(model, sample, forward_fn=adapter.forward_for_task)
        protected_layers = build_protected_layers(model, adapter_protected=adapter.get_protected_layers(model), extra_prefixes=args.extra_protected_prefix or [])
        op_graph = build_op_graph(trace, model, protected_layers=protected_layers)
        grouped_mode = POLICY_TO_MODE[spec.policy_key] if spec.policy_key in {"A", "B", "D"} else "remove_groups"
        groups = GroupBuilder(op_graph, align=args.align, grouped_conv_mode=grouped_mode, protect_residual_add=False).build()
        apply_full_model_prunable_surface(groups, group_conv_policy=spec.policy_key, total_model_params=inventory["total_params"])
        configure_grouped_conv_pruning_fns(groups, argparse.Namespace(group_conv_selection_mode=POLICY_TO_MODE[spec.policy_key], allow_remove_groups=(spec.policy_key == "C")))
        for param in model.parameters():
            param.requires_grad_(True)
        calibration_data = build_importance_calibration_data(adapter, model_args, prune_logger)
        if calibration_data is not None:
            calibration_data = [move_batch_to_device(batch, device) for batch in calibration_data]
        compute_group_importance(
            model,
            groups,
            method=args.importance_mode,
            forward_fn=adapter.forward_for_task,
            calibration_data=calibration_data,
            loss_fn=adapter.compute_task_loss,
            num_calib_batches=int(args.num_calib_batches or 0),
            strict_grad=True,
        )
        scope_importance, _scope_records = compute_scope_channel_importance_map(groups, method=args.importance_mode)
        candidates, rejected, all_units, _units_by_scope, _grouped_reports = build_budget_candidates(
            groups,
            scope_importance,
            spec,
            max_pruning_ratio_per_domain=args.max_pruning_ratio_per_domain,
            min_channels=args.min_channels,
        )
        sorted_candidates = _old_v101_sorted(candidates)
        sort_violation = any(sorted_candidates[idx].score > sorted_candidates[idx + 1].score + 1e-18 for idx in range(len(sorted_candidates) - 1))
        selected_sets: dict[str, set[str]] = {}
        for ratio in ratios:
            tag = ratio_tag(ratio)
            exp_name = f"{spec.name}_{tag}"
            exp_out = root_out / exp_name
            exp_out.mkdir(parents=True, exist_ok=True)
            write_csv(exp_out / "candidate_units_sorted_by_score.csv", [_candidate_row(cand, idx + 1) for idx, cand in enumerate(sorted_candidates)])
            target_saving = float(inventory["total_params"]) * float(ratio)
            rank = {cand.candidate_id: idx + 1 for idx, cand in enumerate(sorted_candidates)}
            recorded_selected = _load_recorded_selected(Path(args.v101_output_dir), exp_name)
            if recorded_selected:
                trace_rows, stop = _trace_from_recorded(
                    recorded_selected,
                    rank=rank,
                    target_saving=target_saving,
                    baseline_total_params=int(inventory["total_params"]),
                    target_ratio=float(ratio),
                )
                selected_ids = {
                    unit
                    for row in recorded_selected
                    for unit in list(row.get("source_coupled_units", []) or [])
                }
                selected = []
            else:
                selected, trace_rows, stop = replay_v101_selector(
                    sorted_candidates,
                    target_saving=target_saving,
                    baseline_total_params=int(inventory["total_params"]),
                    target_ratio=float(ratio),
                )
                selected_ids = {unit for cand in selected for unit in cand.source_coupled_units}
            if sort_violation:
                stop["budget_status"] = "selector_sort_order_violation"
                stop["failure_reason"] = "selector_sort_order_violation"
            actual_ratio, baseline_total, pruned_total = _load_actual_ratio(Path(args.v101_output_dir), exp_name)
            if actual_ratio is not None and baseline_total is not None and pruned_total is not None:
                actual_saving = baseline_total - pruned_total
                stop["actual_param_saving_after_physical_prune"] = actual_saving
                stop["actual_param_prune_ratio_full_model"] = actual_ratio
                stop["estimate_vs_actual_error"] = actual_ratio - float(stop["estimated_param_prune_ratio_selected"])
                gap = build_estimate_actual_gap_report_v102(
                    estimated_param_saving=stop["estimated_param_saving_selected"],
                    actual_param_saving=actual_saving,
                    baseline_total_params=baseline_total,
                    per_module_estimated_saving=(
                        {cand.candidate_id: cand.param_saving_if_removed for cand in selected}
                        if selected
                        else {str(row.get("candidate_id", "")): float(row.get("param_saving_if_removed", 0.0) or 0.0) for row in recorded_selected}
                    ),
                    per_module_actual_saving={},
                )
            else:
                stop["actual_param_saving_after_physical_prune"] = None
                stop["actual_param_prune_ratio_full_model"] = None
                stop["estimate_vs_actual_error"] = None
                gap = {"status": "actual_v101_report_missing"}
            write_csv(exp_out / "selected_units_trace.csv", trace_rows)
            write_json(exp_out / "selector_stop_report.json", stop)
            write_json(exp_out / "estimate_actual_gap_report.json", gap)
            selected_sets[tag] = selected_ids
        _write_monotonicity(root_out, spec.name, selected_sets)
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit v10.1 global budgeted selector")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--model-config", default=DEFAULT_CONFIG)
    parser.add_argument("--heal-root", default=DEFAULT_HEAL_ROOT)
    parser.add_argument("--surface", default="full_model_all_safe_coupled_units")
    parser.add_argument("--selector", default="global_budgeted_coupled_unit_selector")
    parser.add_argument("--importance-mode", default="first_order_taylor")
    parser.add_argument("--strategies", default="A1")
    parser.add_argument("--target-param-prune-ratios", default="0.10,0.15,0.20")
    parser.add_argument("--device", default="cuda:5")
    parser.add_argument("--output-dir", default="outputs/latency_lut/selector_audit_v102")
    parser.add_argument("--v101-output-dir", default="outputs/latency_lut/global_budgeted_alignment_eval_v101")
    parser.add_argument("--num-calib-batches", type=int, default=1)
    parser.add_argument("--align", type=int, default=1)
    parser.add_argument("--min-channels", type=int, default=1)
    parser.add_argument("--max-pruning-ratio-per-domain", type=float, default=0.8)
    parser.add_argument("--extra-protected-prefix", action="append", default=[])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.importance_mode != "first_order_taylor":
        raise ValueError("v10.2 audit requires first_order_taylor")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(resolve_device(args.device))
    if device.type == "cuda":
        torch.cuda.set_device(device)
    strategies = [item.strip().upper() for item in str(args.strategies).split(",") if item.strip()]
    ratios = parse_csv_floats(args.target_param_prune_ratios, [0.10, 0.15, 0.20])
    write_json(out / "audit_config.json", vars(args))
    for strategy in strategies:
        run_strategy(args, strategy, ratios, device, out)
    print(json.dumps({"success": True, "output_dir": str(out), "strategies": strategies, "ratios": ratios}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
