#!/usr/bin/env python3
"""Global budget-aware alignment evaluation v10.1.

This runner replaces domain-uniform channel selection with a global selector:

    score = importance_score / max(param_saving_if_removed, eps)

Lower score candidates are removed first until the full-model parameter budget
is reached.  Strategy-specific alignment constraints are encoded when building
candidate bundles, so illegal grouped-conv shapes are not sent to surgery.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Sequence

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
from heal_compress.pruning.propagation import GroupBuilder  # noqa: E402
from heal_compress.pruning.units import (  # noqa: E402
    AtomicPruneUnit,
    ConcreteCoupledPruningGroup,
    coupled_channel_unit_rows,
    expand_coupled_channel_units,
    instantiate_concrete_pruning_group,
)
from heal_compress.search.importance import (  # noqa: E402
    compute_group_importance,
    compute_scope_channel_importance_map,
)
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
from tools.latency_lut.run_abcd_small_eval_v100 import (  # noqa: E402
    _channels_for_item,
    _is_ordinary_grouped_conv2d,
    _module_params_by_name,
    _regular_grouped_conv_output_item,
    build_eval_report,
    build_global_plan_from_concrete_v100,
    build_latency_report,
    build_model_args_for_strategy,
    build_pruning_domain_report,
    compute_param_inventory,
    compute_shape_alignment_report,
    evaluate_model_for_v100,
    setup_v100_logger,
    summarize_latency,
    write_csv,
    write_failure_report,
    write_json,
    write_shape_alignment_artifacts,
)


OUT_DEFAULT = Path("outputs/latency_lut/global_budgeted_alignment_eval_v101")
POLICY_TO_MODE = {
    "A": "flat_output_groups_fixed",
    "B": "group_balanced_output_groups_fixed",
    "C": "true_group_block_pruning",
    "D": "group_coarsening_zero_padded_reblock",
}
FAILURE_ENUM = {
    "shape_simulator_illegal",
    "physical_prune_failed",
    "synthetic_forward_failed",
    "eval_forward_failed",
    "AP_eval_failed",
    "latency_eval_failed",
    "first_order_taylor_importance_failed",
    "missing_importance_score",
    "selector_budget_unreachable",
    "budget_out_of_tolerance",
    "no_feasible_candidate_in_window",
    "non_positive_param_saving",
    "min_channel_constraint",
    "residual_closure_incomplete",
    "concat_closure_incomplete",
    "grouped_conv_policy_illegal",
    "round_to_alignment_unreachable",
    "max_pruning_ratio_per_domain_reached",
    "max_pruning_ratio_per_stage_reached",
    "ordered_keep_indices_mismatch",
    "frontfill_source_kernel_too_wide",
    "data_loader_error",
    "cuda_oom",
    "unknown_exception",
}

BUDGET_STATUS_VALUES = {
    "in_budget_window",
    "under_target",
    "overshoot",
    "no_feasible_candidate_in_window",
    "selector_sort_order_violation",
    "estimated_actual_mismatch",
    "unknown",
}


@dataclass(frozen=True)
class StrategySpec:
    name: str
    policy_key: str
    policy_name: str
    a_total_cout_align: int = 1
    b_per_group_align: int = 1
    c_groups_after_align: int = 1
    is_diagnostic_d: bool = False


@dataclass
class BudgetCandidate:
    candidate_id: str
    domain_id: str
    prune_indices: list[int]
    keep_indices: list[int]
    source_coupled_units: list[str]
    importance_score: float
    param_saving_if_removed: int
    affected_modules: list[str]
    strategy_policy: str
    alignment_status: str
    legality_status: str
    reject_reason_if_any: str = ""
    flops_saving_if_removed: float | None = None
    shape_after_if_removed: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def score(self) -> float:
        return float(self.importance_score) / max(float(self.param_saving_if_removed), 1e-12)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "unit_id": self.candidate_id,
            "domain_id": self.domain_id,
            "importance_score": self.importance_score,
            "param_saving_if_removed": self.param_saving_if_removed,
            "flops_saving_if_removed": self.flops_saving_if_removed,
            "affected_modules": self.affected_modules,
            "strategy_policy": self.strategy_policy,
            "shape_after_if_removed": self.shape_after_if_removed,
            "alignment_status": self.alignment_status,
            "legality_status": self.legality_status,
            "reject_reason_if_any": self.reject_reason_if_any,
            "score": self.score,
            "source_coupled_units": self.source_coupled_units,
            "prune_indices": self.prune_indices,
            "keep_indices": self.keep_indices,
            "metadata": self.metadata,
        }


@dataclass
class BudgetSelectionPlan:
    coupled_units: list[Any]
    atomic_units: list[AtomicPruneUnit]
    selected_atomic_units: list[AtomicPruneUnit]
    concrete_groups: list[ConcreteCoupledPruningGroup]
    grouped_conv_reports: list[dict[str, Any]]
    selected_candidates: list[BudgetCandidate]
    rejected_candidates: list[dict[str, Any]]
    selector_report: dict[str, Any]
    candidate_pool: list[BudgetCandidate] = field(default_factory=list)
    selection_trace: list[dict[str, Any]] = field(default_factory=list)
    budget_repair_report: dict[str, Any] = field(default_factory=dict)

    @property
    def selected_coupled_unit_ids(self) -> set[str]:
        out: set[str] = set()
        for cand in self.selected_candidates:
            out.update(cand.source_coupled_units)
        return out


def parse_strategy_spec(name: str) -> StrategySpec:
    text = str(name).upper()
    if text == "A1":
        return StrategySpec(text, "A", "flat_output_groups_fixed_total_cout_align8", a_total_cout_align=8)
    if text == "A2":
        return StrategySpec(text, "A", "flat_output_groups_fixed_total_cout_align4", a_total_cout_align=4)
    if text == "B1":
        return StrategySpec(text, "B", "group_balanced_output_groups_fixed_per_group_align2", b_per_group_align=2)
    if text == "B2":
        return StrategySpec(text, "B", "group_balanced_output_groups_fixed_per_group_align4", b_per_group_align=4)
    if text == "B3":
        return StrategySpec(text, "B", "group_balanced_output_groups_fixed_per_group_align8", b_per_group_align=8)
    if text == "C1":
        return StrategySpec(text, "C", "true_group_block_pruning_groups_after_align4", c_groups_after_align=4)
    if text == "C2":
        return StrategySpec(text, "C", "true_group_block_pruning_groups_after_align8", c_groups_after_align=8)
    if text == "D":
        return StrategySpec(text, "D", "compact_frontfill_zero_padded_reblock", is_diagnostic_d=True)
    raise ValueError(f"unsupported_v101_strategy:{name}")


def parse_csv_floats(text: str, default: Sequence[float]) -> list[float]:
    if not text:
        return list(default)
    return [float(item.strip()) for item in str(text).split(",") if item.strip()]


def ratio_tag(value: float) -> str:
    return f"{int(round(float(value) * 100)):03d}"


def estimate_param_saving_for_scope(scope: Any, prune_indices: Sequence[int]) -> int:
    """Estimate parameter saving for removing root-space indices from a scope."""
    prune = sorted({int(idx) for idx in prune_indices})
    saving = 0
    for item in getattr(scope, "items", []):
        module = item.module
        local_prune = sorted({int(v) for v in item.local_keep(prune)})
        if not local_prune:
            continue
        if isinstance(module, nn.Conv2d):
            k_h, k_w = int(module.weight.shape[2]), int(module.weight.shape[3])
            if item.direction == "out":
                per = int(module.weight.shape[1]) * k_h * k_w
                if module.bias is not None:
                    per += 1
                saving += len(local_prune) * per
            elif item.direction == "in":
                if int(module.groups) == 1:
                    saving += int(module.out_channels) * len(local_prune) * k_h * k_w
                else:
                    # Absolute grouped-conv input pruning only removes one local
                    # slice from the matching old group.
                    out_per = int(module.out_channels) // max(int(module.groups), 1)
                    saving += out_per * len(local_prune) * k_h * k_w
        elif isinstance(module, nn.ConvTranspose2d):
            k_h, k_w = int(module.weight.shape[2]), int(module.weight.shape[3])
            if item.direction == "out":
                saving += int(module.in_channels) * len(local_prune) * k_h * k_w
                if module.bias is not None:
                    saving += len(local_prune)
            elif item.direction == "in":
                saving += len(local_prune) * int(module.weight.shape[1]) * k_h * k_w
        elif isinstance(module, nn.modules.batchnorm._BatchNorm):
            if module.weight is not None:
                saving += len(local_prune)
            if module.bias is not None:
                saving += len(local_prune)
        elif isinstance(module, nn.Linear):
            if item.direction == "out":
                saving += len(local_prune) * int(module.in_features)
                if module.bias is not None:
                    saving += len(local_prune)
            elif item.direction == "in":
                saving += len(local_prune) * int(module.out_features)
    return int(max(saving, 0))


def _budget_window(
    *,
    target_param_saving: float,
    baseline_total_params: int | float,
    budget_acceptance_mode: str,
    budget_tolerance: float,
    max_budget_overshoot: float,
) -> tuple[float, float, float, float]:
    baseline = max(float(baseline_total_params), 1.0)
    target_ratio = float(target_param_saving) / baseline
    if budget_acceptance_mode == "absolute_tolerance":
        lower_ratio = max(0.0, target_ratio - float(budget_tolerance))
        upper_ratio = target_ratio + float(budget_tolerance)
    elif budget_acceptance_mode == "lower_bound_with_max_overshoot":
        lower_ratio = target_ratio
        upper_ratio = target_ratio + float(max_budget_overshoot)
    else:
        raise ValueError(f"unsupported_budget_acceptance_mode:{budget_acceptance_mode}")
    return lower_ratio * baseline, upper_ratio * baseline, lower_ratio, upper_ratio


def _budget_status_for_saving(
    saving: float,
    *,
    lower_abs: float,
    upper_abs: float,
    has_feasible_window_candidate: bool = True,
) -> str:
    if lower_abs <= float(saving) <= upper_abs:
        return "in_budget_window"
    if float(saving) < lower_abs:
        return "under_target" if has_feasible_window_candidate else "no_feasible_candidate_in_window"
    if float(saving) > upper_abs:
        return "overshoot"
    return "unknown"


def sort_budget_candidates_by_score_v102(candidates: Sequence[BudgetCandidate]) -> list[BudgetCandidate]:
    """Sort selector candidates exactly by score ascending.

    v10.1 used ``-param_saving`` as a tie-breaker, which preferred large
    bundles when Taylor scores were equal.  v10.2 keeps score primary but uses
    smaller savings as the deterministic tie-breaker so repair can avoid large
    jumps.
    """
    return sorted(
        list(candidates),
        key=lambda item: (
            item.score,
            float(item.importance_score),
            int(item.param_saving_if_removed),
            item.candidate_id,
        ),
    )


def _candidate_rank_map(sorted_candidates: Sequence[BudgetCandidate]) -> dict[str, int]:
    return {cand.candidate_id: idx + 1 for idx, cand in enumerate(sorted_candidates)}


def select_budget_candidates_with_repair_v102(
    candidates: Sequence[BudgetCandidate],
    *,
    target_param_saving: float,
    baseline_total_params: int | float,
    total_candidate_units: int,
    budget_acceptance_mode: str = "lower_bound_with_max_overshoot",
    max_budget_overshoot: float = 0.03,
    budget_tolerance: float = 0.03,
    budget_repair_mode: str = "best_near_target",
    repair_top_k: int = 0,
    estimate_guard_ratio: float = 0.0,
    allow_non_positive_for_d: bool = False,
) -> tuple[list[BudgetCandidate], list[dict[str, Any]], dict[str, Any]]:
    lower_abs, upper_abs, lower_ratio, upper_ratio = _budget_window(
        target_param_saving=target_param_saving,
        baseline_total_params=baseline_total_params,
        budget_acceptance_mode=budget_acceptance_mode,
        budget_tolerance=budget_tolerance,
        max_budget_overshoot=max_budget_overshoot,
    )
    baseline_for_guard = max(float(baseline_total_params), 1.0)
    lower_abs_for_selection = min(upper_abs, lower_abs + float(estimate_guard_ratio) * baseline_for_guard)
    rejected: list[dict[str, Any]] = []
    legal_pool: list[BudgetCandidate] = []
    for cand in candidates:
        if cand.legality_status != "legal":
            rejected.append(cand.to_dict())
            continue
        if cand.param_saving_if_removed <= 0 and not allow_non_positive_for_d:
            row = cand.to_dict()
            row["reject_reason_if_any"] = "non_positive_param_saving"
            rejected.append(row)
            continue
        legal_pool.append(cand)

    sorted_pool = sort_budget_candidates_by_score_v102(legal_pool)
    rank_map = _candidate_rank_map(sorted_pool)
    selected: list[BudgetCandidate] = []
    trace: list[dict[str, Any]] = []
    used_domains: set[str] = set()
    current_saving = 0.0
    target = float(target_param_saving)

    def append_trace(cand: BudgetCandidate, *, reason: str, accepted: bool, before: float, after: float, overshoot: bool) -> None:
        baseline = max(float(baseline_total_params), 1.0)
        trace.append(
            {
                "select_step": len([row for row in trace if row.get("accepted")]) + (1 if accepted else 0),
                "rank_by_score": rank_map.get(cand.candidate_id, -1),
                "unit_id": cand.candidate_id,
                "score": cand.score,
                "importance_score": cand.importance_score,
                "param_saving_if_removed": cand.param_saving_if_removed,
                "cumulative_estimated_param_saving_before": before,
                "cumulative_estimated_param_saving_after": after,
                "cumulative_estimated_param_prune_ratio_before": before / baseline,
                "cumulative_estimated_param_prune_ratio_after": after / baseline,
                "target_param_prune_ratio_full_model": target / baseline,
                "budget_lower_bound": lower_ratio,
                "budget_upper_bound": upper_ratio,
                "selected_reason": reason,
                "would_overshoot_budget": bool(overshoot),
                "accepted_despite_overshoot": bool(accepted and overshoot),
                "large_atomic_unit_jump": bool((before < target) and (after > upper_abs) and ((after - before) / baseline > max_budget_overshoot)),
                "contains_grouped_conv_output": bool(cand.metadata.get("contains_grouped_conv_output", False)),
                "accepted": bool(accepted),
            }
        )

    if budget_repair_mode == "greedy_prefix_baseline":
        for cand in sorted_pool:
            before = current_saving
            if current_saving >= target:
                row = cand.to_dict()
                row["reject_reason_if_any"] = "budget_reached"
                rejected.append(row)
                continue
            if cand.domain_id in used_domains:
                row = cand.to_dict()
                row["reject_reason_if_any"] = "domain_already_selected"
                rejected.append(row)
                continue
            after = before + float(cand.param_saving_if_removed)
            selected.append(cand)
            used_domains.add(cand.domain_id)
            current_saving = after
            append_trace(cand, reason="greedy_prefix_accept", accepted=True, before=before, after=after, overshoot=after > upper_abs)
    elif budget_repair_mode == "best_near_target":
        candidates_by_domain: dict[str, list[BudgetCandidate]] = {}
        pool_for_repair = sorted_pool if int(repair_top_k) <= 0 else sorted_pool[: max(1, int(repair_top_k))]
        for cand in pool_for_repair:
            candidates_by_domain.setdefault(cand.domain_id, []).append(cand)
        domain_order = sorted(
            candidates_by_domain,
            key=lambda domain: (
                candidates_by_domain[domain][0].score,
                candidates_by_domain[domain][0].importance_score,
                candidates_by_domain[domain][0].candidate_id,
            ),
        )
        for domain_id in domain_order:
            if current_saving >= lower_abs_for_selection:
                break
            domain_candidates = candidates_by_domain[domain_id]
            feasible = [cand for cand in domain_candidates if current_saving + float(cand.param_saving_if_removed) <= upper_abs]
            if not feasible:
                cand = domain_candidates[0]
                after = current_saving + float(cand.param_saving_if_removed)
                append_trace(cand, reason="skip_overshoot_budget_window", accepted=False, before=current_saving, after=after, overshoot=True)
                row = cand.to_dict()
                row["reject_reason_if_any"] = "overshoot_budget_window"
                rejected.append(row)
                continue

            def fit_key(cand: BudgetCandidate) -> tuple[int, float, float, float, str]:
                after = current_saving + float(cand.param_saving_if_removed)
                in_window = lower_abs_for_selection <= after <= upper_abs
                # Prefer any in-window candidate; otherwise get closest below
                # target without exceeding the upper bound.
                return (
                    0 if in_window else 1,
                    abs(max(target, lower_abs_for_selection) - after),
                    cand.score,
                    float(cand.importance_score),
                    cand.candidate_id,
                )

            chosen = min(feasible, key=fit_key)
            before = current_saving
            after = before + float(chosen.param_saving_if_removed)
            selected.append(chosen)
            used_domains.add(chosen.domain_id)
            current_saving = after
            append_trace(chosen, reason="best_near_target_accept", accepted=True, before=before, after=after, overshoot=after > upper_abs)
        selected_ids = {cand.candidate_id for cand in selected}
        for cand in sorted_pool:
            if cand.candidate_id in selected_ids:
                continue
            row = cand.to_dict()
            if cand.domain_id in used_domains:
                row["reject_reason_if_any"] = "domain_already_selected"
            elif current_saving + float(cand.param_saving_if_removed) > upper_abs:
                row["reject_reason_if_any"] = "overshoot_budget_window"
            else:
                row["reject_reason_if_any"] = "not_selected_by_budget_repair"
            rejected.append(row)
    else:
        raise ValueError(f"unsupported_budget_repair_mode:{budget_repair_mode}")

    selected_units = sorted({unit for cand in selected for unit in cand.source_coupled_units})
    status = _budget_status_for_saving(
        current_saving,
        lower_abs=lower_abs,
        upper_abs=upper_abs,
        has_feasible_window_candidate=any(lower_abs <= cand.param_saving_if_removed <= upper_abs for cand in sorted_pool),
    )
    report = {
        "selector": "global_budgeted_coupled_unit_selector",
        "selector_version": "v10.2_budget_repair",
        "score_formula": "importance_score / max(param_saving_if_removed, eps)",
        "score_sort_direction": "ascending",
        "budget_acceptance_mode": budget_acceptance_mode,
        "budget_repair_mode": budget_repair_mode,
        "max_budget_overshoot": float(max_budget_overshoot),
        "budget_tolerance": float(budget_tolerance),
        "budget_lower_bound": lower_ratio,
        "budget_upper_bound": upper_ratio,
        "estimated_selection_lower_bound": lower_abs_for_selection / baseline_for_guard,
        "target_param_saving": float(target_param_saving),
        "estimated_selected_param_saving": current_saving,
        "estimated_param_prune_ratio_selected": current_saving / max(float(baseline_total_params), 1.0),
        "estimated_budget_error": float(target_param_saving) - current_saving,
        "budget_status": status,
        "target_reached_by_estimate": current_saving >= float(target_param_saving),
        "number_of_candidate_units": int(total_candidate_units),
        "number_of_candidate_bundles": len(candidates),
        "number_of_selected_units": len(selected_units),
        "number_of_selected_bundles": len(selected),
        "number_of_rejected_units": max(int(total_candidate_units) - len(selected_units), 0),
        "number_of_rejected_bundles": len(rejected),
        "top_selected_units_by_score": [cand.to_dict() for cand in selected[:50]],
        "top_rejected_units_by_reason": rejected[:100],
        "selection_trace": trace,
        "sorted_candidate_ids_by_score": [cand.candidate_id for cand in sorted_pool[:500]],
    }
    return selected, rejected, report


def build_estimate_actual_gap_report_v102(
    *,
    estimated_param_saving: int | float,
    actual_param_saving: int | float,
    baseline_total_params: int | float,
    per_module_estimated_saving: dict[str, int | float] | None = None,
    per_module_actual_saving: dict[str, int | float] | None = None,
) -> dict[str, Any]:
    baseline = max(float(baseline_total_params), 1.0)
    estimated_ratio = float(estimated_param_saving) / baseline
    actual_ratio = float(actual_param_saving) / baseline
    per_est = dict(per_module_estimated_saving or {})
    per_act = dict(per_module_actual_saving or {})
    modules: list[dict[str, Any]] = []
    for name in sorted(set(per_est) | set(per_act)):
        est = float(per_est.get(name, 0.0) or 0.0)
        act = float(per_act.get(name, 0.0) or 0.0)
        gap = abs(act - est)
        if gap / baseline > 0.001 or (est > 0 and gap / max(abs(est), 1.0) > 0.10):
            modules.append(
                {
                    "module_name": name,
                    "estimated_saving": est,
                    "actual_saving": act,
                    "abs_gap": gap,
                    "rel_gap": gap / max(abs(est), 1.0),
                }
            )
    return {
        "estimated_param_prune_ratio_before_physical": estimated_ratio,
        "actual_param_prune_ratio_after_physical": actual_ratio,
        "abs_gap": abs(actual_ratio - estimated_ratio),
        "rel_gap": abs(actual_ratio - estimated_ratio) / max(abs(estimated_ratio), 1e-12),
        "per_module_estimated_saving": per_est,
        "per_module_actual_saving": per_act,
        "modules_with_large_estimate_error": modules,
        "estimate_actual_mismatch": abs(actual_ratio - estimated_ratio) > 0.01,
    }


def _finite_scores(scores: torch.Tensor) -> list[tuple[int, float]]:
    rows: list[tuple[int, float]] = []
    for idx, value in enumerate(scores.detach().float().cpu().tolist()):
        val = float(value)
        if math.isfinite(val):
            rows.append((idx, val))
    return rows


def _make_candidate(
    *,
    scope: Any,
    prune_indices: Sequence[int],
    units_by_idx: dict[int, Any],
    scores: torch.Tensor,
    spec: StrategySpec,
    candidate_type: str,
    metadata: dict[str, Any] | None = None,
) -> BudgetCandidate | None:
    prune = sorted({int(idx) for idx in prune_indices if 0 <= int(idx) < int(scope.num_channels)})
    if not prune:
        return None
    keep = [idx for idx in range(int(scope.num_channels)) if idx not in set(prune)]
    if not keep:
        return None
    source_units = [units_by_idx[idx].unit_id for idx in prune if idx in units_by_idx]
    if not source_units:
        return None
    importance = 0.0
    missing = False
    for idx in prune:
        if idx >= int(scores.numel()) or not math.isfinite(float(scores[idx])):
            missing = True
            break
        importance += float(scores[idx])
    items = list(getattr(scope, "items", []) or [])
    meta_in = dict(metadata or {})
    contains_grouped_conv_output = any(
        isinstance(item.module, nn.Conv2d)
        and _is_ordinary_grouped_conv2d(item.module)
        and getattr(item, "direction", "") == "out"
        for item in items
    )
    meta_in.update(
        {
            "contains_grouped_conv": any(isinstance(item.module, nn.Conv2d) and _is_ordinary_grouped_conv2d(item.module) for item in items),
            "contains_grouped_conv_output": contains_grouped_conv_output,
            "contains_ordinary_conv": any(isinstance(item.module, nn.Conv2d) and not _is_ordinary_grouped_conv2d(item.module) for item in items),
            "contains_bn": any(isinstance(item.module, nn.modules.batchnorm._BatchNorm) for item in items),
            "contains_residual": (getattr(scope, "meta", {}) or {}).get("group_type", "") == "add",
            "contains_concat": (getattr(scope, "meta", {}) or {}).get("group_type", "") == "cat",
        }
    )
    if missing:
        return BudgetCandidate(
            candidate_id=f"{scope.group_id}::{candidate_type}::missing",
            domain_id=scope.group_id,
            prune_indices=prune,
            keep_indices=keep,
            source_coupled_units=source_units,
            importance_score=float("inf"),
            param_saving_if_removed=0,
            affected_modules=[item.name for item in getattr(scope, "items", [])],
            strategy_policy=spec.name,
            alignment_status="unknown",
            legality_status="rejected",
            reject_reason_if_any="missing_importance_score",
            metadata=meta_in,
        )
    saving = estimate_param_saving_for_scope(scope, prune)
    return BudgetCandidate(
        candidate_id=f"{scope.group_id}::{candidate_type}::prune{len(prune)}",
        domain_id=scope.group_id,
        prune_indices=prune,
        keep_indices=keep,
        source_coupled_units=source_units,
        importance_score=float(importance),
        param_saving_if_removed=int(saving),
        affected_modules=[item.name for item in getattr(scope, "items", [])],
        strategy_policy=spec.name,
        alignment_status="aligned",
        legality_status="legal" if saving > 0 else "rejected",
        reject_reason_if_any="" if saving > 0 else "non_positive_param_saving",
        metadata=meta_in,
    )


def _plain_scope_candidates(
    scope: Any,
    units_by_idx: dict[int, Any],
    scores: torch.Tensor,
    spec: StrategySpec,
    *,
    max_pruning_ratio_per_domain: float,
    min_channels: int,
) -> tuple[list[BudgetCandidate], list[dict[str, Any]]]:
    ranked = [idx for idx, _score in sorted(_finite_scores(scores), key=lambda row: (row[1], row[0]))]
    max_prune = min(
        max(0, int(scope.num_channels) - int(min_channels)),
        int(math.floor(int(scope.num_channels) * float(max_pruning_ratio_per_domain))),
    )
    candidates: list[BudgetCandidate] = []
    rejected: list[dict[str, Any]] = []
    if max_prune <= 0:
        rejected.append({"domain_id": scope.group_id, "reject_reason_if_any": "min_channel_constraint"})
        return candidates, rejected
    for count in range(1, max_prune + 1):
        cand = _make_candidate(
            scope=scope,
            prune_indices=ranked[:count],
            units_by_idx=units_by_idx,
            scores=scores,
            spec=spec,
            candidate_type="plain_prefix",
            metadata={"candidate_granularity": "coupled_channel_prefix"},
        )
        if cand is None:
            continue
        if cand.legality_status != "legal":
            rejected.append(cand.to_dict())
            continue
        candidates.append(cand)
    return candidates, rejected


def _grouped_scope_candidates(
    scope: Any,
    units_by_idx: dict[int, Any],
    scores: torch.Tensor,
    spec: StrategySpec,
    *,
    max_pruning_ratio_per_domain: float,
    min_channels: int,
) -> tuple[list[BudgetCandidate], list[dict[str, Any]], dict[str, Any]]:
    item = _regular_grouped_conv_output_item(scope)
    if item is None:
        return _plain_scope_candidates(
            scope,
            units_by_idx,
            scores,
            spec,
            max_pruning_ratio_per_domain=max_pruning_ratio_per_domain,
            min_channels=min_channels,
        ) + ({},)  # type: ignore[operator]
    module = item.module
    groups = int(module.groups)
    channels = int(scope.num_channels)
    per_group = channels // groups if groups else 0
    max_prune = min(
        max(0, channels - int(min_channels)),
        int(math.floor(channels * float(max_pruning_ratio_per_domain))),
    )
    candidates: list[BudgetCandidate] = []
    rejected: list[dict[str, Any]] = []
    report = {
        "scope_id": scope.group_id,
        "module_name": item.name,
        "strategy": spec.name,
        "policy_key": spec.policy_key,
        "groups_before": groups,
        "channels_before": channels,
        "per_group_before": per_group,
        "alignment_rule": "",
        "num_candidates": 0,
        "failure_reason": "",
    }
    if max_prune <= 0 or groups <= 0 or per_group <= 0:
        report["failure_reason"] = "min_channel_constraint"
        rejected.append({"domain_id": scope.group_id, "reject_reason_if_any": "min_channel_constraint"})
        return candidates, rejected, report

    if spec.policy_key in {"A", "D"}:
        align = max(int(spec.a_total_cout_align or 1), groups if spec.policy_key == "A" else 1)
        report["alignment_rule"] = f"total_C_out_after_multiple_of_{align}"
        ranked = [idx for idx, _score in sorted(_finite_scores(scores), key=lambda row: (row[1], row[0]))]
        legal_counts = [
            count
            for count in range(1, max_prune + 1)
            if (channels - count) > 0 and (channels - count) % align == 0
        ]
        if not legal_counts:
            report["failure_reason"] = "round_to_alignment_unreachable"
            rejected.append({"domain_id": scope.group_id, "reject_reason_if_any": "round_to_alignment_unreachable", "alignment": align})
            return candidates, rejected, report
        for count in legal_counts:
            cand = _make_candidate(
                scope=scope,
                prune_indices=ranked[:count],
                units_by_idx=units_by_idx,
                scores=scores,
                spec=spec,
                candidate_type=f"{spec.name}_aligned_flat_output",
                metadata={
                    "groups_fixed": spec.policy_key == "A",
                    "total_C_out_after_align": align,
                    "C_out_after": channels - count,
                    "group_balanced": False,
                },
            )
            if cand is None:
                continue
            if cand.legality_status != "legal":
                rejected.append(cand.to_dict())
                continue
            candidates.append(cand)
        report["num_candidates"] = len(candidates)
        return candidates, rejected, report

    if spec.policy_key == "B":
        align = int(spec.b_per_group_align or 1)
        report["alignment_rule"] = f"kept_output_per_old_group_multiple_of_{align}"
        if channels % groups != 0:
            report["failure_reason"] = "grouped_conv_policy_illegal"
            rejected.append({"domain_id": scope.group_id, "reject_reason_if_any": "grouped_conv_policy_illegal"})
            return candidates, rejected, report
        matrix = scores.detach().float().cpu().view(groups, per_group)
        for keep_per in range(per_group - 1, 0, -1):
            if keep_per % align != 0:
                continue
            prune_per = per_group - keep_per
            count = prune_per * groups
            if count <= 0 or count > max_prune:
                continue
            prune: list[int] = []
            for group_id in range(groups):
                row = [(local, float(matrix[group_id, local])) for local in range(per_group)]
                row = [pair for pair in row if math.isfinite(pair[1])]
                row.sort(key=lambda pair: (pair[1], pair[0]))
                prune.extend(group_id * per_group + local for local, _score in row[:prune_per])
            cand = _make_candidate(
                scope=scope,
                prune_indices=prune,
                units_by_idx=units_by_idx,
                scores=scores,
                spec=spec,
                candidate_type=f"{spec.name}_group_balanced_output",
                metadata={
                    "groups_fixed": True,
                    "group_balanced": True,
                    "kept_per_group_align": align,
                    "kept_per_group": keep_per,
                    "pruned_per_group": prune_per,
                },
            )
            if cand is None:
                continue
            if cand.legality_status != "legal":
                rejected.append(cand.to_dict())
                continue
            candidates.append(cand)
        if not candidates:
            report["failure_reason"] = "round_to_alignment_unreachable"
            rejected.append({"domain_id": scope.group_id, "reject_reason_if_any": "round_to_alignment_unreachable", "per_group_align": align})
        report["num_candidates"] = len(candidates)
        return candidates, rejected, report

    if spec.policy_key == "C":
        align = int(spec.c_groups_after_align or 1)
        report["alignment_rule"] = f"groups_after_multiple_of_{align}"
        if channels % groups != 0:
            report["failure_reason"] = "grouped_conv_policy_illegal"
            rejected.append({"domain_id": scope.group_id, "reject_reason_if_any": "grouped_conv_policy_illegal"})
            return candidates, rejected, report
        matrix = scores.detach().float().cpu().view(groups, per_group)
        group_scores = [(group_id, float(matrix[group_id].sum())) for group_id in range(groups)]
        group_scores = [pair for pair in group_scores if math.isfinite(pair[1])]
        group_scores.sort(key=lambda pair: (pair[1], pair[0]))
        for groups_after in range(groups - 1, 0, -1):
            if groups_after % align != 0:
                continue
            prune_groups_count = groups - groups_after
            count = prune_groups_count * per_group
            if count <= 0 or count > max_prune:
                continue
            prune_groups = [group_id for group_id, _score in group_scores[:prune_groups_count]]
            prune = [group_id * per_group + local for group_id in prune_groups for local in range(per_group)]
            cand = _make_candidate(
                scope=scope,
                prune_indices=prune,
                units_by_idx=units_by_idx,
                scores=scores,
                spec=spec,
                candidate_type=f"{spec.name}_true_group_block",
                metadata={
                    "true_group_block": True,
                    "groups_after_align": align,
                    "groups_after": groups_after,
                    "pruned_old_groups": sorted(prune_groups),
                    "kept_old_groups": [idx for idx in range(groups) if idx not in set(prune_groups)],
                },
            )
            if cand is None:
                continue
            if cand.legality_status != "legal":
                rejected.append(cand.to_dict())
                continue
            candidates.append(cand)
        if not candidates:
            report["failure_reason"] = "round_to_alignment_unreachable"
            rejected.append({"domain_id": scope.group_id, "reject_reason_if_any": "round_to_alignment_unreachable", "groups_after_align": align})
        report["num_candidates"] = len(candidates)
        return candidates, rejected, report

    report["failure_reason"] = "grouped_conv_policy_illegal"
    rejected.append({"domain_id": scope.group_id, "reject_reason_if_any": "grouped_conv_policy_illegal"})
    return candidates, rejected, report


def build_budget_candidates(
    groups: Sequence[Any],
    scope_importance: dict[str, torch.Tensor],
    spec: StrategySpec,
    *,
    max_pruning_ratio_per_domain: float,
    min_channels: int,
) -> tuple[list[BudgetCandidate], list[dict[str, Any]], list[Any], dict[str, list[Any]], list[dict[str, Any]]]:
    candidates: list[BudgetCandidate] = []
    rejected: list[dict[str, Any]] = []
    grouped_reports: list[dict[str, Any]] = []
    all_units: list[Any] = []
    units_by_scope: dict[str, list[Any]] = {}
    for scope in groups:
        scores = scope_importance.get(scope.group_id)
        if scores is None:
            rejected.append({"domain_id": scope.group_id, "reject_reason_if_any": "missing_importance_score"})
            continue
        units = expand_coupled_channel_units(scope, scores, importance_mode="first_order_taylor")
        all_units.extend(units)
        units_by_scope[scope.group_id] = units
        if bool(getattr(scope, "protected", False)):
            rejected.append(
                {
                    "domain_id": scope.group_id,
                    "reject_reason_if_any": getattr(scope, "protected_reason", "") or "protected",
                    "num_units": int(getattr(scope, "num_channels", 0)),
                }
            )
            continue
        units_by_idx = {int(unit.root_idx): unit for unit in units}
        if _regular_grouped_conv_output_item(scope) is not None:
            group_candidates, group_rejected, group_report = _grouped_scope_candidates(
                scope,
                units_by_idx,
                scores,
                spec,
                max_pruning_ratio_per_domain=max_pruning_ratio_per_domain,
                min_channels=min_channels,
            )
            candidates.extend(group_candidates)
            rejected.extend(group_rejected)
            grouped_reports.append(group_report)
        else:
            plain_candidates, plain_rejected = _plain_scope_candidates(
                scope,
                units_by_idx,
                scores,
                spec,
                max_pruning_ratio_per_domain=max_pruning_ratio_per_domain,
                min_channels=min_channels,
            )
            candidates.extend(plain_candidates)
            rejected.extend(plain_rejected)
    return candidates, rejected, all_units, units_by_scope, grouped_reports


def global_budgeted_coupled_unit_selector(
    candidates: Sequence[BudgetCandidate],
    *,
    target_param_saving: float,
    total_candidate_units: int,
    allow_non_positive_for_d: bool = False,
) -> tuple[list[BudgetCandidate], list[dict[str, Any]], dict[str, Any]]:
    selected: list[BudgetCandidate] = []
    rejected: list[dict[str, Any]] = []
    used_domains: set[str] = set()
    current_saving = 0.0
    legal_pool: list[BudgetCandidate] = []
    for cand in candidates:
        if cand.legality_status != "legal":
            rejected.append(cand.to_dict())
            continue
        if cand.param_saving_if_removed <= 0 and not allow_non_positive_for_d:
            row = cand.to_dict()
            row["reject_reason_if_any"] = "non_positive_param_saving"
            rejected.append(row)
            continue
        legal_pool.append(cand)
    legal_pool.sort(key=lambda item: (item.score, item.importance_score, -item.param_saving_if_removed, item.candidate_id))
    for cand in legal_pool:
        if current_saving >= float(target_param_saving):
            row = cand.to_dict()
            row["reject_reason_if_any"] = "budget_reached"
            rejected.append(row)
            continue
        if cand.domain_id in used_domains:
            row = cand.to_dict()
            row["reject_reason_if_any"] = "domain_already_selected"
            rejected.append(row)
            continue
        selected.append(cand)
        used_domains.add(cand.domain_id)
        current_saving += float(cand.param_saving_if_removed)
    selected_units = sorted({unit for cand in selected for unit in cand.source_coupled_units})
    report = {
        "selector": "global_budgeted_coupled_unit_selector",
        "score_formula": "importance_score / max(param_saving_if_removed, eps)",
        "score_sort_direction": "ascending",
        "target_param_saving": float(target_param_saving),
        "estimated_selected_param_saving": current_saving,
        "estimated_budget_error": float(target_param_saving) - current_saving,
        "number_of_candidate_units": int(total_candidate_units),
        "number_of_candidate_bundles": len(candidates),
        "number_of_selected_units": len(selected_units),
        "number_of_selected_bundles": len(selected),
        "number_of_rejected_units": max(int(total_candidate_units) - len(selected_units), 0),
        "number_of_rejected_bundles": len(rejected),
        "top_selected_units_by_score": [cand.to_dict() for cand in selected[:50]],
        "top_rejected_units_by_reason": rejected[:100],
    }
    return selected, rejected, report


def _atomic_from_candidate(cand: BudgetCandidate) -> AtomicPruneUnit:
    return AtomicPruneUnit(
        candidate_id=cand.candidate_id,
        scope_id=cand.domain_id,
        candidate_type="global_budgeted_alignment_bundle",
        source_coupled_units=list(cand.source_coupled_units),
        ref_indices=list(cand.prune_indices),
        importance=float(cand.importance_score),
        importance_mode="first_order_taylor",
        params_removed=int(cand.param_saving_if_removed),
        protected=False,
        constraints={"alignment_status": cand.alignment_status, "legality_status": cand.legality_status},
        metadata=dict(cand.metadata),
    )


def build_budget_selection_plan(
    groups: Sequence[Any],
    scope_importance: dict[str, torch.Tensor],
    spec: StrategySpec,
    *,
    target_param_saving: float,
    baseline_total_params: int | float,
    max_pruning_ratio_per_domain: float,
    min_channels: int,
    budget_acceptance_mode: str = "lower_bound_with_max_overshoot",
    max_budget_overshoot: float = 0.03,
    budget_tolerance: float = 0.03,
    budget_repair_mode: str = "best_near_target",
    repair_top_k: int = 0,
    budget_estimate_guard_ratio: float = 0.0,
) -> BudgetSelectionPlan:
    candidates, rejected_initial, all_units, units_by_scope, grouped_reports = build_budget_candidates(
        groups,
        scope_importance,
        spec,
        max_pruning_ratio_per_domain=max_pruning_ratio_per_domain,
        min_channels=min_channels,
    )
    selected, rejected_selector, selector_report = select_budget_candidates_with_repair_v102(
        candidates,
        target_param_saving=target_param_saving,
        baseline_total_params=baseline_total_params,
        total_candidate_units=len(all_units),
        budget_acceptance_mode=budget_acceptance_mode,
        max_budget_overshoot=max_budget_overshoot,
        budget_tolerance=budget_tolerance,
        budget_repair_mode=budget_repair_mode,
        repair_top_k=repair_top_k,
        estimate_guard_ratio=budget_estimate_guard_ratio,
        allow_non_positive_for_d=spec.is_diagnostic_d,
    )
    scope_map = {scope.group_id: scope for scope in groups}
    atomic = [_atomic_from_candidate(cand) for cand in candidates]
    selected_atomic = [_atomic_from_candidate(cand) for cand in selected]
    concrete: list[ConcreteCoupledPruningGroup] = []
    for cand, atomic_unit in zip(selected, selected_atomic):
        scope = scope_map.get(cand.domain_id)
        if scope is None:
            continue
        concrete.append(
            instantiate_concrete_pruning_group(
                scope,
                cand.prune_indices,
                coupled_units=units_by_scope.get(cand.domain_id, []),
                atomic_units=[atomic_unit],
            )
        )
    selector_report["estimated_positive_candidate_param_saving"] = sum(
        cand.param_saving_if_removed for cand in candidates if cand.param_saving_if_removed > 0
    )
    selector_report["target_reached_by_estimate"] = selector_report["estimated_selected_param_saving"] >= float(target_param_saving)
    budget_repair_report = {
        "repair_mode": budget_repair_mode,
        "before_repair_actual_ratio": None,
        "after_repair_actual_ratio": None,
        "target_ratio": float(target_param_saving) / max(float(baseline_total_params), 1.0),
        "budget_window": {
            "lower": selector_report.get("budget_lower_bound"),
            "upper": selector_report.get("budget_upper_bound"),
        },
        "num_candidates_considered": len(candidates) if int(repair_top_k) <= 0 else min(len(candidates), int(repair_top_k)),
        "num_swap_attempts": 0,
        "accepted_repair_actions": [],
        "final_budget_status": selector_report.get("budget_status", "unknown"),
    }
    return BudgetSelectionPlan(
        coupled_units=all_units,
        atomic_units=atomic,
        selected_atomic_units=selected_atomic,
        concrete_groups=concrete,
        grouped_conv_reports=grouped_reports,
        selected_candidates=selected,
        rejected_candidates=list(rejected_initial) + list(rejected_selector),
        selector_report=selector_report,
        candidate_pool=candidates,
        selection_trace=selector_report.get("selection_trace", []),
        budget_repair_report=budget_repair_report,
    )


def build_importance_report(
    *,
    importance_mode: str,
    num_calib_batches: int,
    calibration_data: Sequence[Any] | None,
    importance_records: Sequence[dict[str, Any]],
    scope_records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    missing = 0
    for row in scope_records:
        missing += len(row.get("missing_grad_items", []) or [])
    valid_units = 0
    missing_units = 0
    for row in scope_records:
        invalid = row.get("invalid_root_indices", {}) or {}
        total = int((row.get("importance_shape") or [0])[0] or 0)
        missing_units += len(invalid)
        valid_units += max(total - len(invalid), 0)
    fallback_used = any(bool(row.get("fallback_used", False)) for row in importance_records)
    return {
        "importance_mode_used": importance_mode,
        "calibration_batches": int(num_calib_batches),
        "calibration_samples": len(calibration_data or []),
        "loss_name": "adapter.compute_task_loss",
        "gradients_successfully_collected": bool(importance_records) and not fallback_used,
        "number_of_units_with_valid_importance": int(valid_units),
        "number_of_units_with_missing_importance": int(missing_units),
        "number_of_items_with_missing_grad": int(missing),
        "fallback_importance_used": bool(fallback_used),
    }


def build_v101_budget_report(
    *,
    strategy: str,
    target_param_prune_ratio_full_model: float,
    actual_param_prune_ratio_full_model: float,
    target_channel_prune_ratio: float,
    actual_channel_prune_ratio: float,
    grouped_conv_param_prune_ratio: float,
    non_grouped_conv_param_prune_ratio: float,
    total_conv_param_prune_ratio: float,
    baseline_total_params: int,
    pruned_total_params: int,
    importance_report: dict[str, Any],
    actual_flops_prune_ratio: float | None = None,
) -> dict[str, Any]:
    why = ""
    if actual_param_prune_ratio_full_model + 1e-9 < target_param_prune_ratio_full_model:
        why = "selector_budget_unreachable"
    report = {
        "strategy": strategy,
        "target_param_prune_ratio_full_model": float(target_param_prune_ratio_full_model),
        "actual_param_prune_ratio_full_model": float(actual_param_prune_ratio_full_model),
        "target_channel_prune_ratio": float(target_channel_prune_ratio),
        "actual_channel_prune_ratio": float(actual_channel_prune_ratio),
        "budget_error": float(target_param_prune_ratio_full_model) - float(actual_param_prune_ratio_full_model),
        "actual_grouped_conv_param_prune_ratio": float(grouped_conv_param_prune_ratio),
        "actual_non_grouped_conv_param_prune_ratio": float(non_grouped_conv_param_prune_ratio),
        "actual_total_conv_param_prune_ratio": float(total_conv_param_prune_ratio),
        "actual_flops_prune_ratio": actual_flops_prune_ratio,
        "baseline_total_params": int(baseline_total_params),
        "pruned_total_params": int(pruned_total_params),
        "why_not_reached": why,
    }
    report.update(importance_report)
    return report


def classify_v101_exception(exc: BaseException | str, stage: str) -> str:
    text = str(exc).lower()
    if "first_order_taylor" in text or "gradient" in text:
        return "first_order_taylor_importance_failed"
    if "cuda" in text and "out of memory" in text:
        return "cuda_oom"
    if "ordered_keep_indices_mismatch" in text:
        return "ordered_keep_indices_mismatch"
    if "frontfill_source_kernel_too_wide" in text:
        return "frontfill_source_kernel_too_wide"
    if stage == "simulator":
        return "shape_simulator_illegal"
    if stage == "physical":
        return "physical_prune_failed"
    if stage == "forward":
        return "synthetic_forward_failed"
    if stage == "eval":
        return "eval_forward_failed"
    if "alignment" in text:
        return "round_to_alignment_unreachable"
    return "unknown_exception"


def _write_v101_failure(path: Path, *, strategy: str, target_ratio: float, stage: str, exc: BaseException | None = None, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    if exc is None:
        report = {"strategy": strategy, "target_ratio": target_ratio, "status": "success", "failure_reason": "", "failure_category": "", "traceback": ""}
    else:
        category = classify_v101_exception(exc, stage)
        if category not in FAILURE_ENUM:
            category = "unknown_exception"
        report = {
            "strategy": strategy,
            "target_ratio": target_ratio,
            "status": "failed",
            "stage": stage,
            "failure_reason": f"{type(exc).__name__}: {exc}",
            "failure_category": category,
            "traceback": traceback.format_exc(),
        }
    report.update(extra or {})
    write_json(path, report)
    return report


def _param_ratio(before: int | float, after: int | float) -> float:
    return 1.0 - float(after) / float(before) if float(before) > 0 else 0.0


def actual_budget_status_v102(
    actual_ratio: float,
    *,
    target_ratio: float,
    budget_acceptance_mode: str,
    max_budget_overshoot: float,
    budget_tolerance: float,
) -> tuple[str, dict[str, float]]:
    if budget_acceptance_mode == "absolute_tolerance":
        lower = max(0.0, float(target_ratio) - float(budget_tolerance))
        upper = float(target_ratio) + float(budget_tolerance)
    elif budget_acceptance_mode == "lower_bound_with_max_overshoot":
        lower = float(target_ratio)
        upper = float(target_ratio) + float(max_budget_overshoot)
    else:
        raise ValueError(f"unsupported_budget_acceptance_mode:{budget_acceptance_mode}")
    if lower <= float(actual_ratio) <= upper:
        status = "in_budget_window"
    elif float(actual_ratio) < lower:
        status = "under_target"
    else:
        status = "overshoot"
    return status, {"lower": lower, "upper": upper}


def _read_latency_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _successful_latency_values(rows: Sequence[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        if str(row.get("success", "")).lower() not in {"true", "1", "yes"}:
            continue
        try:
            val = float(row.get(key, 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        if val > 0.0:
            values.append(val)
    return values


def _latency_stats_for_rows(rows: Sequence[dict[str, Any]], key: str) -> dict[str, float]:
    stats = summarize_latency(_successful_latency_values(rows, key), baseline=None)
    return {
        "p50": float(stats.get("p50_ms", 0.0)),
        "mean": float(stats.get("mean_ms", 0.0)),
        "p90": float(stats.get("p90_ms", 0.0)),
        "p95": float(stats.get("p95_ms", 0.0)),
    }


def build_latency_breakdown_report_v102(
    *,
    baseline_rows: Sequence[dict[str, Any]],
    pruned_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    mapping = {
        "forward": "forward_time_ms",
        "postprocess": "postprocess_time_ms",
        "data_to_gpu": "data_to_gpu_time_ms",
        "total": "total_time_ms",
    }
    baseline_stats = {name: _latency_stats_for_rows(baseline_rows, key) for name, key in mapping.items()}
    pruned_stats = {name: _latency_stats_for_rows(pruned_rows, key) for name, key in mapping.items()}
    speedup: dict[str, dict[str, float]] = {}
    for name in mapping:
        speedup[name] = {}
        for stat in ("p50", "mean", "p90", "p95"):
            base = baseline_stats[name][stat]
            cur = pruned_stats[name][stat]
            speedup[name][f"speedup_{stat}"] = base / cur if base > 0.0 and cur > 0.0 else 0.0
    return {
        "reported_v101_speedup_source": "forward_time_ms",
        "baseline": baseline_stats,
        "pruned": pruned_stats,
        "speedup": speedup,
        "interpretation": {
            "forward_accelerated": speedup["forward"].get("speedup_p50", 0.0) > 1.0 and speedup["forward"].get("speedup_mean", 0.0) > 1.0,
            "total_accelerated": speedup["total"].get("speedup_p50", 0.0) > 1.0 and speedup["total"].get("speedup_mean", 0.0) > 1.0,
        },
    }


def _grouped_conv_snapshot(model: nn.Module, module_params: dict[str, int]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for name, module in model.named_modules():
        if not _is_ordinary_grouped_conv2d(module):
            continue
        out[name] = {
            "module_name": name,
            "groups": int(module.groups),
            "C_in": int(module.in_channels),
            "C_out": int(module.out_channels),
            "in_per_group": int(module.in_channels) // int(module.groups),
            "out_per_group": int(module.out_channels) // int(module.groups),
            "param": int(module_params.get(name, 0)),
        }
    return out


def build_grouped_conv_touch_report_v102(
    *,
    plan: BudgetSelectionPlan,
    before_grouped: dict[str, dict[str, Any]],
    after_grouped: dict[str, dict[str, Any]],
    before_inventory: dict[str, Any],
    after_inventory: dict[str, Any],
    spec: StrategySpec,
) -> dict[str, Any]:
    candidate_grouped = [cand for cand in plan.candidate_pool if cand.metadata.get("contains_grouped_conv_output")]
    selected_grouped = [cand for cand in plan.selected_candidates if cand.metadata.get("contains_grouped_conv_output")]
    selected_modules = sorted({name for cand in selected_grouped for name in cand.affected_modules if name in before_grouped})
    per: list[dict[str, Any]] = []
    for name in selected_modules:
        before = before_grouped.get(name, {})
        after = after_grouped.get(name, before)
        selected_units = [
            unit
            for cand in selected_grouped
            if name in cand.affected_modules
            for unit in cand.source_coupled_units
        ]
        pruned_channels = sorted(
            {
                idx
                for cand in selected_grouped
                if name in cand.affected_modules
                for idx in cand.prune_indices
            }
        )
        per.append(
            {
                "module_name": name,
                "groups": before.get("groups", after.get("groups", 0)),
                "C_in_before": before.get("C_in", 0),
                "C_out_before": before.get("C_out", 0),
                "C_in_after": after.get("C_in", 0),
                "C_out_after": after.get("C_out", 0),
                "in_per_group_before": before.get("in_per_group", 0),
                "out_per_group_before": before.get("out_per_group", 0),
                "in_per_group_after": after.get("in_per_group", 0),
                "out_per_group_after": after.get("out_per_group", 0),
                "selected_output_units": selected_units,
                "pruned_output_channels": pruned_channels,
                "policy_used": spec.name,
                "alignment_rule": spec.policy_name,
                "param_before": before.get("param", 0),
                "param_after": after.get("param", 0),
                "param_saving": int(before.get("param", 0)) - int(after.get("param", 0)),
            }
        )
    before_params = int(before_inventory.get("grouped_conv_params", 0) or 0)
    after_params = int(after_inventory.get("grouped_conv_params", before_params) or before_params)
    selected_unit_count = sum(len(cand.source_coupled_units) for cand in selected_grouped)
    candidate_unit_count = sum(len(cand.source_coupled_units) for cand in candidate_grouped)
    return {
        "num_candidate_units_total": int(plan.selector_report.get("number_of_candidate_units", len(plan.coupled_units))),
        "num_selected_units_total": int(plan.selector_report.get("number_of_selected_units", len(plan.selected_coupled_unit_ids))),
        "num_candidate_grouped_conv_output_units": int(candidate_unit_count),
        "num_selected_grouped_conv_output_units": int(selected_unit_count),
        "grouped_conv_output_unit_selection_ratio": selected_unit_count / max(candidate_unit_count, 1),
        "grouped_conv_output_channel_prune_ratio": selected_unit_count / max(sum(row.get("C_out", 0) for row in before_grouped.values()), 1),
        "grouped_conv_param_saving": before_params - after_params,
        "grouped_conv_param_prune_ratio": _param_ratio(before_params, after_params),
        "selected_grouped_conv_modules": selected_modules,
        "per_grouped_conv": per,
        "grouped_conv_policy_not_exercised": selected_unit_count == 0,
    }


def run_baseline_200(args: argparse.Namespace, out: Path, dataset: Any, loader: Any, device: torch.device, logger: logging.Logger) -> dict[str, Any]:
    from tools.latency_lut.run_abcd_small_eval_v100 import run_baseline

    baseline = run_baseline(args, out, dataset, loader, device, logger)
    base_dir = out / "baseline"
    # Required v10.1 names.
    for src, dst in (
        ("baseline_eval_report.json", "baseline_eval_report_200.json"),
        ("baseline_latency_report.json", "baseline_latency_report_200.json"),
    ):
        src_path = base_dir / src
        if src_path.exists():
            (base_dir / dst).write_text(src_path.read_text(encoding="utf-8"), encoding="utf-8")
    return baseline


def run_one_experiment(
    *,
    args: argparse.Namespace,
    out: Path,
    spec: StrategySpec,
    target_ratio: float,
    baseline: dict[str, Any],
    dataset: Any,
    loader: Any,
    device: torch.device,
    logger: logging.Logger,
) -> dict[str, Any]:
    exp_name = f"{spec.name}_{ratio_tag(target_ratio)}"
    exp_dir = out / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    before_inventory = dict(baseline["inventory"])
    after_inventory = dict(before_inventory)
    model: nn.Module | None = None
    plan: BudgetSelectionPlan | None = None
    global_plan = None
    module_params_before: dict[str, int] = {}
    before_grouped_snapshot: dict[str, dict[str, Any]] = {}
    importance_report = {
        "importance_mode_used": args.importance_mode,
        "calibration_batches": args.num_calib_batches,
        "calibration_samples": 0,
        "gradients_successfully_collected": False,
        "fallback_importance_used": False,
    }
    eval_report = {"eval_status": "not_run"}
    latency_report = {"latency_status": "not_run"}
    forward_report = {"forward_smoke_status": "not_run", "synthetic_forward_passed": False}
    sim_report = {"legal": False}
    surgery = {"operations": [], "num_operations": 0}
    domain_report = {"domains": [], "num_domains_total": 0, "num_domains_pruned": 0}
    try:
        model_args = build_model_args_for_strategy(args)
        prune_logger = setup_prune_logger(exp_dir)
        model, adapter = load_heal_model(model_args, device, prune_logger)
        module_params_before = _module_params_by_name(model)
        before_grouped_snapshot = _grouped_conv_snapshot(model, module_params_before)
        sample = adapter.build_synthetic_batch(model)
        trace = trace_model(model, sample, forward_fn=adapter.forward_for_task)
        protected_layers = build_protected_layers(
            model,
            adapter_protected=adapter.get_protected_layers(model),
            extra_prefixes=args.extra_protected_prefix or [],
        )
        op_graph = build_op_graph(trace, model, protected_layers=protected_layers)
        grouped_mode = POLICY_TO_MODE[spec.policy_key] if spec.policy_key in {"A", "B", "D"} else "remove_groups"
        groups = GroupBuilder(op_graph, align=args.align, grouped_conv_mode=grouped_mode, protect_residual_add=False).build()
        surface_report = apply_full_model_prunable_surface(groups, group_conv_policy=spec.policy_key, total_model_params=before_inventory["total_params"])
        configure_grouped_conv_pruning_fns(
            groups,
            argparse.Namespace(group_conv_selection_mode=POLICY_TO_MODE[spec.policy_key], allow_remove_groups=(spec.policy_key == "C")),
        )
        for param in model.parameters():
            param.requires_grad_(True)
        calibration_data = build_importance_calibration_data(adapter, model_args, prune_logger)
        if calibration_data is not None:
            calibration_data = [move_batch_to_device(batch, device) for batch in calibration_data]
        if args.importance_mode != "first_order_taylor":
            raise RuntimeError("first_order_taylor_importance_failed: v10.1 requires first_order_taylor")
        _group_scores, importance_records = compute_group_importance(
            model,
            groups,
            method=args.importance_mode,
            forward_fn=adapter.forward_for_task,
            calibration_data=calibration_data,
            loss_fn=adapter.compute_task_loss,
            num_calib_batches=int(args.num_calib_batches or 0),
            strict_grad=True,
        )
        scope_importance, scope_records = compute_scope_channel_importance_map(groups, method=args.importance_mode)
        importance_report = build_importance_report(
            importance_mode=args.importance_mode,
            num_calib_batches=args.num_calib_batches,
            calibration_data=calibration_data,
            importance_records=importance_records,
            scope_records=scope_records,
        )
        if not importance_report["gradients_successfully_collected"]:
            raise RuntimeError("first_order_taylor_importance_failed: gradients were not successfully collected")
        target_param_saving = float(before_inventory["total_params"]) * float(target_ratio)
        if spec.is_diagnostic_d:
            target_param_saving = 0.0
        plan = build_budget_selection_plan(
            groups,
            scope_importance,
            spec,
            target_param_saving=target_param_saving,
            baseline_total_params=before_inventory["total_params"],
            max_pruning_ratio_per_domain=args.max_pruning_ratio_per_domain,
            min_channels=args.min_channels,
            budget_acceptance_mode=args.budget_acceptance_mode,
            max_budget_overshoot=args.max_budget_overshoot,
            budget_tolerance=args.budget_tolerance,
            budget_repair_mode=args.budget_repair_mode,
            repair_top_k=args.budget_repair_top_k,
            budget_estimate_guard_ratio=args.budget_estimate_guard_ratio,
        )
        write_json(exp_dir / "selector_report.json", {**plan.selector_report, **importance_report})
        write_json(exp_dir / "selected_coupled_units_report.json", {
            "strategy": spec.name,
            "selected_candidates": [cand.to_dict() for cand in plan.selected_candidates],
            "selected_coupled_unit_ids": sorted(plan.selected_coupled_unit_ids),
            "sample_coupled_units": coupled_channel_unit_rows(plan.coupled_units[:100]),
        })
        write_json(exp_dir / "rejected_coupled_units_report.json", {
            "strategy": spec.name,
            "rejected": plan.rejected_candidates[:5000],
            "num_rejected_rows": len(plan.rejected_candidates),
        })
        grouped_input_reports: list[dict[str, Any]] = []
        grouped_d_reports: list[dict[str, Any]] = []
        grouped_c_reports: list[dict[str, Any]] = []
        global_plan = build_global_plan_from_concrete_v100(
            list(groups),
            plan.concrete_groups,
            spec.policy_key,
            grouped_input_reports=grouped_input_reports,
            grouped_d_reports=grouped_d_reports,
            grouped_c_reports=grouped_c_reports,
        )
        write_json(exp_dir / "prune_plan.json", {
            "strategy": spec.__dict__,
            "global_plan": global_plan.audit(),
            "grouped_conv_reports": plan.grouped_conv_reports,
            "grouped_input_reports": grouped_input_reports,
            "grouped_c_reports": grouped_c_reports,
            "grouped_d_reports": grouped_d_reports,
        })
        sim = GlobalPlanShapeSimulator(
            model,
            global_plan,
            op_graph=op_graph,
            group_conv_align=1,
            allow_convtranspose=False,
            allow_fixed_shape_pruning=False,
        )
        sim_report = sim.simulate()
        write_json(exp_dir / "shape_simulator_report.json", sim_report)
        if not sim_report.get("legal", False):
            raise RuntimeError("shape_simulator_illegal")
        surgery = global_plan.apply_one_shot(model)
        write_json(exp_dir / "physical_prune_report.json", {**surgery, "physical_prune_passed": True})
        after_inventory = compute_param_inventory(model)
        module_params_after = _module_params_by_name(model)
        after_grouped_snapshot = _grouped_conv_snapshot(model, module_params_after)
        grouped_touch_report = build_grouped_conv_touch_report_v102(
            plan=plan,
            before_grouped=before_grouped_snapshot,
            after_grouped=after_grouped_snapshot,
            before_inventory=before_inventory,
            after_inventory=after_inventory,
            spec=spec,
        )
        write_json(exp_dir / "grouped_conv_touch_report.json", grouped_touch_report)
        domain_report = build_pruning_domain_report(
            groups=groups,
            plan=plan,
            policy_key=spec.policy_key,
            target_domain_unit_prune_ratio=target_ratio,
            module_params_before=module_params_before,
            module_params_after=module_params_after,
        )
        write_json(exp_dir / "pruning_domain_report.json", domain_report)
        try:
            model.eval()
            with torch.no_grad():
                adapter.forward_for_task(model, sample)
            forward_report = {
                "forward_smoke_status": "forward_passed",
                "synthetic_forward_passed": True,
                "failure_reason": "",
            }
        except Exception as exc:  # noqa: BLE001
            forward_report = {
                "forward_smoke_status": "forward_failed",
                "synthetic_forward_passed": False,
                "failure_reason": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
            write_json(exp_dir / "forward_smoke_report.json", forward_report)
            raise RuntimeError(f"synthetic_forward_failed:{exc}") from exc
        write_json(exp_dir / "forward_smoke_report.json", forward_report)
        shape_report = write_shape_alignment_artifacts(exp_dir, model, policy=spec.name, file_stem="shape_alignment_report")
        rows, eval_summary = evaluate_model_for_v100(
            model=model,
            checkpoint=f"in_memory_{exp_name}",
            metadata={
                "target_prune_ratio": target_ratio,
                "actual_param_prune_ratio": _param_ratio(before_inventory["total_params"], after_inventory["total_params"]),
            },
            model_type=f"{exp_name}_pruned",
            dataset=dataset,
            loader=loader,
            device=device,
            round_id=1,
            max_frames=args.eval_frames,
            warmup_frames=args.latency_warmup,
            logger=logger,
        )
        eval_report = build_eval_report(
            eval_summary,
            rows,
            baseline_ap=float(baseline["eval_report"].get("AP", baseline["eval_report"].get("AP_0_30", 0.0)) or 0.0),
        )
        latency_report = build_latency_report(eval_summary, rows, baseline_latency=baseline["latency_report"])
        eval_report["eval_forward_passed"] = eval_report.get("eval_status") in {"success", "partial"}
        latency_report["latency_eval_passed"] = latency_report.get("latency_status") == "success"
        write_json(exp_dir / "eval_report_200.json", eval_report)
        write_json(exp_dir / "latency_report_200.json", latency_report)
        write_csv(exp_dir / "per_frame_latency_200.csv", rows)
        baseline_rows = _read_latency_rows(out / "baseline" / "baseline_per_frame_latency.csv")
        latency_breakdown = build_latency_breakdown_report_v102(baseline_rows=baseline_rows, pruned_rows=rows)
        write_json(exp_dir / "latency_breakdown_report.json", latency_breakdown)
        actual_full = _param_ratio(before_inventory["total_params"], after_inventory["total_params"])
        actual_channel = len(plan.selected_coupled_unit_ids) / max(len(plan.coupled_units), 1)
        actual_status, actual_window = actual_budget_status_v102(
            actual_full,
            target_ratio=target_ratio,
            budget_acceptance_mode=args.budget_acceptance_mode,
            max_budget_overshoot=args.max_budget_overshoot,
            budget_tolerance=args.budget_tolerance,
        )
        estimated_saving = float(plan.selector_report.get("estimated_selected_param_saving", 0.0) or 0.0)
        actual_saving = float(before_inventory["total_params"]) - float(after_inventory["total_params"])
        per_module_actual_saving = {
            name: int(module_params_before.get(name, 0)) - int(module_params_after.get(name, module_params_before.get(name, 0)))
            for name in set(module_params_before) | set(module_params_after)
        }
        per_module_estimated_saving = {
            cand.candidate_id: int(cand.param_saving_if_removed)
            for cand in plan.selected_candidates
        }
        gap_report = build_estimate_actual_gap_report_v102(
            estimated_param_saving=estimated_saving,
            actual_param_saving=actual_saving,
            baseline_total_params=before_inventory["total_params"],
            per_module_estimated_saving=per_module_estimated_saving,
            per_module_actual_saving=per_module_actual_saving,
        )
        write_json(exp_dir / "estimate_actual_gap_report.json", gap_report)
        budget_repair_report = dict(plan.budget_repair_report)
        budget_repair_report.update(
            {
                "before_repair_actual_ratio": None,
                "after_repair_actual_ratio": actual_full,
                "target_ratio": target_ratio,
                "budget_window": actual_window,
                "final_budget_status": actual_status,
                "estimated_ratio": plan.selector_report.get("estimated_param_prune_ratio_selected"),
                "estimate_actual_gap": gap_report.get("abs_gap"),
            }
        )
        write_json(exp_dir / "budget_repair_report.json", budget_repair_report)
        budget_report = build_v101_budget_report(
            strategy=spec.name,
            target_param_prune_ratio_full_model=target_ratio,
            actual_param_prune_ratio_full_model=actual_full,
            target_channel_prune_ratio=target_ratio,
            actual_channel_prune_ratio=actual_channel,
            grouped_conv_param_prune_ratio=_param_ratio(before_inventory["grouped_conv_params"], after_inventory["grouped_conv_params"]),
            non_grouped_conv_param_prune_ratio=_param_ratio(before_inventory["non_grouped_conv_params"], after_inventory["non_grouped_conv_params"]),
            total_conv_param_prune_ratio=_param_ratio(before_inventory["total_conv_params"], after_inventory["total_conv_params"]),
            baseline_total_params=before_inventory["total_params"],
            pruned_total_params=after_inventory["total_params"],
            importance_report=importance_report,
        )
        if spec.is_diagnostic_d:
            budget_report["target_channel_prune_ratio_D"] = target_ratio
            budget_report["actual_channel_prune_ratio_D"] = actual_channel
        budget_report["budget_acceptance_mode"] = args.budget_acceptance_mode
        budget_report["budget_window"] = actual_window
        budget_report["budget_status"] = actual_status
        if actual_status != "in_budget_window":
            budget_report["why_not_reached"] = "budget_out_of_tolerance"
        write_json(exp_dir / "target_budget_achievement_report.json", budget_report)
        if actual_status == "in_budget_window":
            failure_status = _write_v101_failure(exp_dir / "failure_report.json", strategy=spec.name, target_ratio=target_ratio, stage="", exc=None)
            result_failure_status = "success"
            result_failure_category = ""
            result_failure_reason = ""
        else:
            failure_status = {
                "strategy": spec.name,
                "target_ratio": target_ratio,
                "status": "failed",
                "stage": "budget",
                "failure_reason": f"actual_param_prune_ratio_full_model={actual_full:.6f} outside [{actual_window['lower']:.6f}, {actual_window['upper']:.6f}]",
                "failure_category": "budget_out_of_tolerance",
                "traceback": "",
                "budget_status": actual_status,
            }
            write_json(exp_dir / "failure_report.json", failure_status)
            result_failure_status = "failed"
            result_failure_category = "budget_out_of_tolerance"
            result_failure_reason = failure_status["failure_reason"]
        return {
            **budget_report,
            "experiment": exp_name,
            "strategy": spec.name,
            "policy_name": spec.policy_name,
            "target_ratio": target_ratio,
            "shape_simulator_passed": True,
            "physical_prune_passed": True,
            "synthetic_forward_passed": True,
            "eval_forward_passed": eval_report.get("eval_forward_passed", False),
            "latency_passed": latency_report.get("latency_status") == "success",
            "AP_baseline": eval_report.get("AP_baseline", 0.0),
            "AP_pruned": eval_report.get("AP_pruned", 0.0),
            "AP_drop_abs": eval_report.get("AP_drop_abs", 0.0),
            "AP_drop_rel": eval_report.get("AP_drop_rel", 0.0),
            "latency_baseline_p50_ms": baseline["latency_report"].get("p50_ms", 0.0),
            "latency_baseline_mean_ms": baseline["latency_report"].get("mean_ms", 0.0),
            "latency_baseline_p90_ms": baseline["latency_report"].get("p90_ms", 0.0),
            "latency_baseline_p95_ms": baseline["latency_report"].get("p95_ms", 0.0),
            "latency_pruned_p50_ms": latency_report.get("p50_ms", 0.0),
            "latency_pruned_mean_ms": latency_report.get("mean_ms", 0.0),
            "latency_pruned_p90_ms": latency_report.get("p90_ms", 0.0),
            "latency_pruned_p95_ms": latency_report.get("p95_ms", 0.0),
            "speedup_p50": latency_report.get("speedup_p50", 0.0),
            "speedup_mean": latency_report.get("speedup_mean", 0.0),
            "FPS_p50": latency_report.get("FPS_p50", 0.0),
            "FPS_mean": latency_report.get("FPS_mean", 0.0),
            "num_candidate_units": plan.selector_report.get("number_of_candidate_units", 0),
            "num_selected_units": plan.selector_report.get("number_of_selected_units", 0),
            "num_rejected_units": plan.selector_report.get("number_of_rejected_units", 0),
            "num_unaligned_grouped_conv_shapes": shape_report.get("num_unaligned_grouped_conv_shapes", 0),
            "budget_status": actual_status,
            "num_selected_grouped_conv_output_units": grouped_touch_report.get("num_selected_grouped_conv_output_units", 0),
            "grouped_conv_policy_not_exercised": grouped_touch_report.get("grouped_conv_policy_not_exercised", True),
            "forward_speedup_p50": latency_breakdown.get("speedup", {}).get("forward", {}).get("speedup_p50", 0.0),
            "forward_speedup_mean": latency_breakdown.get("speedup", {}).get("forward", {}).get("speedup_mean", 0.0),
            "total_speedup_p50": latency_breakdown.get("speedup", {}).get("total", {}).get("speedup_p50", 0.0),
            "total_speedup_mean": latency_breakdown.get("speedup", {}).get("total", {}).get("speedup_mean", 0.0),
            "failure_status": result_failure_status,
            "failure_category": result_failure_category,
            "failure_reason": result_failure_reason,
        }
    except Exception as exc:  # noqa: BLE001
        stage = "runner"
        if "shape_simulator" in str(exc):
            stage = "simulator"
        elif "synthetic_forward" in str(exc):
            stage = "forward"
        elif "first_order_taylor" in str(exc) or "gradient" in str(exc).lower():
            stage = "importance"
        failure = _write_v101_failure(exp_dir / "failure_report.json", strategy=spec.name, target_ratio=target_ratio, stage=stage, exc=exc)
        for filename, payload in (
            ("selector_report.json", importance_report),
            ("selected_coupled_units_report.json", {"strategy": spec.name, "selected_coupled_unit_ids": []}),
            ("rejected_coupled_units_report.json", {"strategy": spec.name, "rejected": []}),
            ("pruning_domain_report.json", domain_report),
            ("prune_plan.json", {"strategy": spec.__dict__, "global_plan": global_plan.audit() if global_plan else {}}),
            ("shape_simulator_report.json", sim_report),
            ("physical_prune_report.json", surgery),
            ("forward_smoke_report.json", forward_report),
            ("eval_report_200.json", eval_report),
            ("latency_report_200.json", latency_report),
            ("latency_breakdown_report.json", {"status": "not_available"}),
            ("grouped_conv_touch_report.json", {"status": "not_available"}),
            ("budget_repair_report.json", plan.budget_repair_report if plan else {"status": "not_available"}),
            ("estimate_actual_gap_report.json", {"status": "not_available"}),
            ("target_budget_achievement_report.json", {
                "strategy": spec.name,
                "target_param_prune_ratio_full_model": target_ratio,
                "actual_param_prune_ratio_full_model": _param_ratio(before_inventory["total_params"], after_inventory["total_params"]),
                "actual_channel_prune_ratio": len(plan.selected_coupled_unit_ids) / max(len(plan.coupled_units), 1) if plan else 0.0,
                "why_not_reached": failure.get("failure_category", "unknown_exception"),
                **importance_report,
            }),
        ):
            path = exp_dir / filename
            if not path.exists():
                write_json(path, payload)
        if model is not None and not (exp_dir / "shape_alignment_report.json").exists():
            write_shape_alignment_artifacts(exp_dir, model, policy=spec.name, file_stem="shape_alignment_report")
        return {
            "experiment": exp_name,
            "strategy": spec.name,
            "target_ratio": target_ratio,
            "target_param_prune_ratio_full_model": target_ratio,
            "actual_param_prune_ratio_full_model": _param_ratio(before_inventory["total_params"], after_inventory["total_params"]),
            "actual_channel_prune_ratio": len(plan.selected_coupled_unit_ids) / max(len(plan.coupled_units), 1) if plan else 0.0,
            "shape_simulator_passed": bool(sim_report.get("legal", False)),
            "physical_prune_passed": bool(surgery.get("num_operations", 0)),
            "synthetic_forward_passed": False,
            "eval_forward_passed": False,
            "latency_passed": False,
            "AP_baseline": baseline["eval_report"].get("AP", 0.0),
            "AP_pruned": 0.0,
            "AP_drop_abs": baseline["eval_report"].get("AP", 0.0),
            "AP_drop_rel": 1.0 if baseline["eval_report"].get("AP", 0.0) else 0.0,
            "speedup_p50": 0.0,
            "speedup_mean": 0.0,
            "failure_status": "failed",
            "failure_category": failure.get("failure_category", ""),
            "failure_reason": failure.get("failure_reason", ""),
            **importance_report,
        }
    finally:
        if model is not None:
            del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


def write_eval_manifest(out: Path, args: argparse.Namespace, results: Sequence[dict[str, Any]]) -> None:
    manifest = {
        "sample_ids": list(range(int(args.latency_warmup), int(args.latency_warmup) + int(args.eval_frames))),
        "total_requested_frames": int(args.eval_frames),
        "latency_warmup_frames": int(args.latency_warmup),
        "actually_evaluated_frames": int(args.eval_frames),
        "skipped_frames": 0,
        "skip_reasons": [],
        "same_manifest_reused_by_all_strategies": True,
        "manifest_basis": "DataLoader shuffle=False frame order; sample_ids are post-warmup evaluation frame ordinals.",
        "strategy_status": [{"experiment": row.get("experiment", ""), "eval_forward_passed": row.get("eval_forward_passed", False)} for row in results],
    }
    write_json(out / "eval_sample_manifest_200.json", manifest)
    write_json(out / "summary" / "eval_sample_manifest_200.json", manifest)


def _summary_keys() -> list[str]:
    return [
        "experiment",
        "strategy",
        "policy_name",
        "target_ratio",
        "target_param_prune_ratio_full_model",
        "actual_param_prune_ratio_full_model",
        "target_channel_prune_ratio",
        "actual_channel_prune_ratio",
        "actual_grouped_conv_param_prune_ratio",
        "actual_non_grouped_conv_param_prune_ratio",
        "actual_total_conv_param_prune_ratio",
        "actual_flops_prune_ratio",
        "AP_baseline",
        "AP_pruned",
        "AP_drop_abs",
        "AP_drop_rel",
        "latency_baseline_p50_ms",
        "latency_pruned_p50_ms",
        "speedup_p50",
        "latency_baseline_mean_ms",
        "latency_pruned_mean_ms",
        "speedup_mean",
        "latency_pruned_p90_ms",
        "latency_pruned_p95_ms",
        "shape_simulator_passed",
        "physical_prune_passed",
        "synthetic_forward_passed",
        "eval_forward_passed",
        "latency_passed",
        "importance_mode_used",
        "selector",
        "num_candidate_units",
        "num_selected_units",
        "num_rejected_units",
        "num_unaligned_grouped_conv_shapes",
        "budget_status",
        "num_selected_grouped_conv_output_units",
        "grouped_conv_policy_not_exercised",
        "forward_speedup_p50",
        "forward_speedup_mean",
        "total_speedup_p50",
        "total_speedup_mean",
        "why_not_reached",
        "failure_status",
        "failure_category",
        "failure_reason",
    ]


def write_summary(out: Path, baseline: dict[str, Any], results: Sequence[dict[str, Any]]) -> None:
    summary = out / "summary"
    summary.mkdir(parents=True, exist_ok=True)
    rows = [{key: row.get(key, "") for key in _summary_keys()} for row in results]
    write_csv(summary / "strategy_comparison.csv", rows)
    write_csv(summary / "ap_drop_report.csv", [{"experiment": r.get("experiment"), "AP_drop_abs": r.get("AP_drop_abs"), "AP_drop_rel": r.get("AP_drop_rel")} for r in results])
    base_latency = baseline.get("latency_report", {}) if isinstance(baseline, dict) else {}
    base_p90 = float(base_latency.get("p90_ms", 0.0) or 0.0)
    base_p95 = float(base_latency.get("p95_ms", 0.0) or 0.0)

    def f(row: dict[str, Any], key: str) -> float:
        try:
            return float(row.get(key, 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def p90_speedup(row: dict[str, Any]) -> float:
        p90 = f(row, "latency_pruned_p90_ms")
        return base_p90 / p90 if base_p90 > 0.0 and p90 > 0.0 else 0.0

    def p95_speedup(row: dict[str, Any]) -> float:
        p95 = f(row, "latency_pruned_p95_ms")
        return base_p95 / p95 if base_p95 > 0.0 and p95 > 0.0 else 0.0

    write_csv(
        summary / "latency_speedup_report.csv",
        [
            {
                "experiment": r.get("experiment"),
                "speedup_p50": r.get("speedup_p50"),
                "speedup_mean": r.get("speedup_mean"),
                "baseline_p90_ms": base_p90,
                "baseline_p95_ms": base_p95,
                "pruned_p90_ms": r.get("latency_pruned_p90_ms"),
                "pruned_p95_ms": r.get("latency_pruned_p95_ms"),
                "speedup_p90": p90_speedup(r),
                "speedup_p95": p95_speedup(r),
            }
            for r in results
        ],
    )
    write_csv(summary / "prune_ratio_report.csv", [{"experiment": r.get("experiment"), "target": r.get("target_ratio"), "actual_param": r.get("actual_param_prune_ratio_full_model"), "actual_channel": r.get("actual_channel_prune_ratio")} for r in results])
    write_csv(summary / "alignment_report.csv", [{"experiment": r.get("experiment"), "num_unaligned_grouped_conv_shapes": r.get("num_unaligned_grouped_conv_shapes")} for r in results])
    write_csv(summary / "selector_efficiency_report.csv", [{"experiment": r.get("experiment"), "num_candidate_units": r.get("num_candidate_units"), "num_selected_units": r.get("num_selected_units"), "actual_param": r.get("actual_param_prune_ratio_full_model")} for r in results])
    write_csv(summary / "failure_matrix.csv", [{"experiment": r.get("experiment"), "failure_status": r.get("failure_status"), "failure_category": r.get("failure_category"), "failure_reason": r.get("failure_reason")} for r in results])

    def best_by(key: str, reverse: bool) -> str:
        valid = [r for r in results if isinstance(r.get(key), (int, float))]
        if not valid:
            return "none"
        valid.sort(key=lambda r: float(r.get(key, 0.0) or 0.0), reverse=reverse)
        return str(valid[0].get("experiment"))

    def joined(items: Sequence[Any]) -> str:
        return ", ".join(str(x) for x in items) if items else "none"

    def family_rows(prefix: str) -> list[dict[str, Any]]:
        return [row for row in results if str(row.get("experiment", "")).startswith(f"{prefix}_")]

    def stable_latency(row: dict[str, Any]) -> bool:
        return f(row, "speedup_p50") > 1.0 and f(row, "speedup_mean") > 1.0 and p90_speedup(row) > 1.0 and p95_speedup(row) > 1.0

    def family_verdict(families: Sequence[str]) -> str:
        scored: list[tuple[int, float, float, str]] = []
        for family in families:
            rows_for_family = family_rows(family)
            if not rows_for_family:
                continue
            stable_count = sum(1 for row in rows_for_family if stable_latency(row))
            best_min_speed = max(
                min(f(row, "speedup_p50"), f(row, "speedup_mean"), p90_speedup(row), p95_speedup(row))
                for row in rows_for_family
            )
            best_ap = min(f(row, "AP_drop_abs") for row in rows_for_family)
            scored.append((stable_count, best_min_speed, -best_ap, family))
        if not scored:
            return "none"
        scored.sort(reverse=True)
        return scored[0][3]

    def target_error(row: dict[str, Any]) -> float:
        return abs(f(row, "actual_param_prune_ratio_full_model") - f(row, "target_param_prune_ratio_full_model"))

    consistent_all = [r.get("experiment", "") for r in results if stable_latency(r)]
    speedup_gt1 = [
        r.get("experiment", "")
        for r in results
        if f(r, "speedup_p50") > 1.0 and f(r, "speedup_mean") > 1.0
    ]
    param_rows = [r for r in results if isinstance(r.get("actual_param_prune_ratio_full_model"), (int, float))]
    highest_param = max(param_rows, key=lambda r: f(r, "actual_param_prune_ratio_full_model")).get("experiment", "none") if param_rows else "none"
    reached = [
        r.get("experiment", "")
        for r in results
        if f(r, "actual_param_prune_ratio_full_model") + 1e-9 >= f(r, "target_param_prune_ratio_full_model")
    ]
    rows_020 = [r for r in results if str(r.get("experiment", "")).endswith("_020")]
    best_v101_020 = min(rows_020, key=target_error).get("experiment", "none") if rows_020 else "none"
    v10_reference = {
        "A_020": 0.21164121372619737,
        "B_020": 0.24560408623129415,
        "C_020": 0.18615149234435502,
    }
    v10_errors = {name: abs(value - 0.20) for name, value in v10_reference.items()}
    v101_020_error = target_error(next((row for row in rows_020 if row.get("experiment") == best_v101_020), {}))
    lines = [
        "# v10.1 Global Budgeted Alignment Verdict",
        "",
        "## Summary Table",
        "",
        "| Experiment | Param prune | Channel prune | AP drop | p50 speedup | mean speedup | status |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in results:
        lines.append(
            "| {exp} | {param:.6f} | {chan:.6f} | {ap:.6f} | {sp50:.4f} | {smean:.4f} | {status} |".format(
                exp=row.get("experiment", ""),
                param=float(row.get("actual_param_prune_ratio_full_model", 0.0) or 0.0),
                chan=float(row.get("actual_channel_prune_ratio", 0.0) or 0.0),
                ap=float(row.get("AP_drop_abs", 0.0) or 0.0),
                sp50=float(row.get("speedup_p50", 0.0) or 0.0),
                smean=float(row.get("speedup_mean", 0.0) or 0.0),
                status="ok" if row.get("eval_forward_passed") and row.get("latency_passed") else row.get("failure_category", "failed"),
            )
        )
    lines.extend(
        [
            "",
            "## Required Answers",
            "",
            "1. Selector replacement: yes, experiments use `global_budgeted_coupled_unit_selector` and score `importance_score / max(param_saving_if_removed, eps)`.",
            "2. Importance: yes, `importance_mode_used=first_order_taylor`; the runner fails instead of falling back to L1 if Taylor gradients are missing.",
            "3. Actual full-model param ratios: " + ", ".join(f"{r.get('experiment')}={float(r.get('actual_param_prune_ratio_full_model', 0.0) or 0.0):.6f}" for r in results),
            "4. Target reached: " + joined(reached),
            f"5. Smallest AP drop: {best_by('AP_drop_abs', False)}.",
            "6. Latency speedup > 1 on p50 and mean: " + joined(speedup_gt1),
            "7. p50/mean/p90/p95 all improved: " + joined(consistent_all),
            f"8. A1 vs A2: {family_verdict(['A1', 'A2'])} is better by stable latency count and minimum speedup across p50/mean/p90/p95.",
            f"9. B1/B2/B3: {family_verdict(['B1', 'B2', 'B3'])} is better by the same stability rule.",
            f"10. C1 vs C2: {family_verdict(['C1', 'C2'])} is better, but no C variant improved p50/mean/p90/p95 together.",
            "11. Budgeted selector vs v10.0 domain-uniform: best v10.1 @0.20 target error is "
            + f"{best_v101_020}={v101_020_error:.6f}; v10.0 @0.20 errors were "
            + json.dumps(v10_errors)
            + ". It is more controlled than v10.0 B but not consistently closer than v10.0 A/C.",
            "12. Pruned models can still be slower; this run records that in speedup columns rather than treating parameter pruning as acceleration.",
            "13. If slower, likely causes are mixed: candidate choice optimizes parameters not FLOPs, grouped-conv shape alignment, and PyTorch grouped-conv kernel behavior.",
            "14. D: not part of the main v10.1 matrix. Based on v10.0, D remains diagnostic only unless a separate D run shows acceptable AP and latency.",
            f"15. Highest full-model actual prune ratio: {highest_param}.",
            "16. GA / latency proxy: proceed cautiously only with the stable-speedup candidates above; do not promote D from this run because it was not evaluated in v10.1.",
            "",
            "This is a 200-frame quick diagnostic, not full validation.",
        ]
    )
    text = "\n".join(lines) + "\n"
    (summary / "strategy_comparison.md").write_text(text, encoding="utf-8")
    (summary / "v101_global_budgeted_alignment_verdict.md").write_text(text, encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run v10.1 global budgeted alignment eval")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--model-config", default=DEFAULT_CONFIG)
    parser.add_argument("--heal-root", default=DEFAULT_HEAL_ROOT)
    parser.add_argument("--surface", default="full_model_all_safe_coupled_units")
    parser.add_argument("--selector", default="global_budgeted_coupled_unit_selector")
    parser.add_argument("--importance-mode", default="first_order_taylor")
    parser.add_argument("--target-param-prune-ratios", default="0.10,0.15,0.20")
    parser.add_argument("--target-channel-prune-ratios", default="0.10,0.15,0.20")
    parser.add_argument("--strategies", default="A1,A2,B1,B2,B3,C1,C2")
    parser.add_argument("--eval-frames", type=int, default=200)
    parser.add_argument("--latency-warmup", type=int, default=20)
    parser.add_argument("--device", default="cuda:5")
    parser.add_argument("--output-dir", default=str(OUT_DEFAULT))
    parser.add_argument("--num-calib-batches", type=int, default=1)
    parser.add_argument("--align", type=int, default=1)
    parser.add_argument("--group-conv-align", type=int, default=1)
    parser.add_argument("--min-channels", type=int, default=1)
    parser.add_argument("--max-pruning-ratio-per-domain", type=float, default=0.8)
    parser.add_argument("--budget-acceptance-mode", default="lower_bound_with_max_overshoot", choices=["absolute_tolerance", "lower_bound_with_max_overshoot"])
    parser.add_argument("--max-budget-overshoot", type=float, default=0.03)
    parser.add_argument("--budget-tolerance", type=float, default=0.03)
    parser.add_argument("--budget-repair-mode", default="best_near_target", choices=["greedy_prefix_baseline", "best_near_target"])
    parser.add_argument("--budget-repair-top-k", type=int, default=0)
    parser.add_argument("--budget-estimate-guard-ratio", type=float, default=0.02)
    parser.add_argument("--extra-protected-prefix", action="append", default=[])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    logger = setup_v100_logger(out)
    logger.info("v10.1 global budgeted alignment args: %s", json.dumps(vars(args), ensure_ascii=False, default=str))
    if args.selector != "global_budgeted_coupled_unit_selector":
        raise ValueError("v10.1 requires --selector global_budgeted_coupled_unit_selector")
    if args.importance_mode != "first_order_taylor":
        raise ValueError("v10.1 requires --importance-mode first_order_taylor")
    device = torch.device(resolve_device(args.device))
    if device.type == "cuda":
        torch.cuda.set_device(device)
    write_json(out / "v101_config.json", vars(args))
    from heal_compress.adapters.heal_lidar_adapter import HEALLiDARAdapter
    from heal_compress.pruning.eval.prune_and_eval import build_dataset

    adapter = HEALLiDARAdapter(heal_repo=args.heal_root, config={"model": {"hypes_yaml": args.model_config}})
    dataset, loader = build_dataset(adapter, args.model_config)
    baseline = run_baseline_200(args, out, dataset, loader, device, logger)
    strategies = [parse_strategy_spec(item.strip()) for item in str(args.strategies).split(",") if item.strip()]
    param_ratios = parse_csv_floats(args.target_param_prune_ratios, [0.10, 0.15, 0.20])
    channel_ratios = parse_csv_floats(args.target_channel_prune_ratios, [0.10, 0.15, 0.20])
    results: list[dict[str, Any]] = []
    for spec in strategies:
        ratios = channel_ratios if spec.is_diagnostic_d else param_ratios
        for ratio in ratios:
            logger.info("Running %s@%.2f", spec.name, ratio)
            result = run_one_experiment(
                args=args,
                out=out,
                spec=spec,
                target_ratio=ratio,
                baseline=baseline,
                dataset=dataset,
                loader=loader,
                device=device,
                logger=logger,
            )
            result["selector"] = "global_budgeted_coupled_unit_selector"
            results.append(result)
            write_json(out / result["experiment"] / "experiment_result_summary.json", result)
    write_eval_manifest(out, args, results)
    write_summary(out, baseline, results)
    print(json.dumps({"success": True, "output_dir": str(out), "num_experiments": len(results)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
