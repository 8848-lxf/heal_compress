"""TP-like greedy global budget selector for v10.8."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import torch

from .grouped_pergroup8_policy import GroupedPerGroup8Decision, grouped_per_group_aligned_independent_local_pruning


@dataclass
class RoundTo8Result:
    current_channels: int
    raw_n_pruned: int
    raw_keep: int
    aligned_keep: int
    final_n_pruned: int
    rounding_overshoot_slots: int
    rounding_overshoot_ratio: float


@dataclass
class V108RankingUnit:
    pruning_domain_id: str
    root_module_name: str
    root_dim: str
    num_root_channels: int
    coupled_unit_id: str
    root_channel_index: int
    is_grouped_conv: bool
    importance_raw: float
    importance_normalized: float
    grouped_local_unit_id: str | None = None
    group_index: int | None = None
    local_channel_index: int | None = None
    selected_for_pruning: bool = False
    protected_reason: str = ""
    skipped_reason: str = ""

    def to_report_row(self) -> dict[str, Any]:
        return {
            "pruning_domain_id": self.pruning_domain_id,
            "root_module_name": self.root_module_name,
            "root_dim": self.root_dim,
            "num_root_channels": self.num_root_channels,
            "coupled_unit_id": self.coupled_unit_id,
            "root_channel_index": self.root_channel_index,
            "is_grouped_conv": self.is_grouped_conv,
            "grouped_local_unit_id": self.grouped_local_unit_id,
            "group_index": self.group_index,
            "local_channel_index": self.local_channel_index,
            "importance_raw": self.importance_raw,
            "importance_normalized": self.importance_normalized,
            "selected_for_pruning": self.selected_for_pruning,
            "protected_reason": self.protected_reason,
            "skipped_reason": self.skipped_reason,
        }


@dataclass
class V108PruningDomain:
    pruning_domain_id: str
    root_module_name: str
    root_dim: str
    num_root_channels: int
    units: list[V108RankingUnit]
    is_grouped_conv: bool = False
    groups: int = 1
    per_group: int | None = None
    protected_reason: str = ""
    skipped_reason: str = ""


@dataclass
class V108DomainSelection:
    pruning_domain_id: str
    raw_selected_count: int = 0
    final_n_pruned: int = 0
    keep_indices: list[int] = field(default_factory=list)
    prune_indices: list[int] = field(default_factory=list)
    rounding_overshoot_slots: int = 0
    skipped_reason: str = ""
    grouped_decision: GroupedPerGroup8Decision | None = None


@dataclass
class V108GreedySelectionResult:
    target_pruning_mode: str
    target_pruning_ratio: float
    actual_channel_prune_ratio_on_searchable_surface: float
    actual_param_prune_ratio: float
    predicted_param_prune_ratio: float
    param_prediction_error: float
    param_budget_overshoot_ratio: float
    actual_flops_proxy_ratio: float | None
    rounding_overshoot_ratio: float
    max_ch_sparsity_blocked_count: int
    skipped_because_pergroup_below8_count: int
    skipped_because_pergroup_below_round_to_count: int
    unreachable: bool
    domain_plans: dict[str, V108DomainSelection]
    selected_units: list[V108RankingUnit]
    skipped_units: list[V108RankingUnit]
    trace_rows: list[dict[str, Any]]
    grouped_shape_rows: list[dict[str, Any]]


def tp_floor_keep(current_channels: int, raw_n_pruned: int, round_to: int = 8) -> RoundTo8Result:
    current = int(current_channels)
    raw_pruned = max(0, min(int(raw_n_pruned), current))
    raw_keep = current - raw_pruned
    if int(round_to) > 1:
        aligned_keep = raw_keep - (raw_keep % int(round_to))
    else:
        aligned_keep = raw_keep
    aligned_keep = max(0, min(current, aligned_keep))
    final_pruned = current - aligned_keep
    overshoot = max(0, final_pruned - raw_pruned)
    return RoundTo8Result(
        current_channels=current,
        raw_n_pruned=raw_pruned,
        raw_keep=raw_keep,
        aligned_keep=aligned_keep,
        final_n_pruned=final_pruned,
        rounding_overshoot_slots=overshoot,
        rounding_overshoot_ratio=overshoot / current if current else 0.0,
    )


def _unit_score(unit: V108RankingUnit) -> float:
    value = float(unit.importance_normalized)
    if not math.isfinite(value):
        return float("inf")
    return value


def _domain_sorted_units(domain: V108PruningDomain) -> list[V108RankingUnit]:
    return sorted(domain.units, key=lambda unit: (_unit_score(unit), int(unit.root_channel_index)))


def _unit_param_saving(
    param_savings_by_unit: Mapping[str, Mapping[int, float]] | None,
    domain_id: str,
    index: int,
) -> float:
    if not param_savings_by_unit:
        return 1.0
    return float(param_savings_by_unit.get(str(domain_id), {}).get(int(index), 0.0))


def _predicted_param_ratio(
    domain_plans: Mapping[str, V108DomainSelection],
    *,
    param_savings_by_unit: Mapping[str, Mapping[int, float]] | None,
    predicted_total_params: float,
    param_ratio_from_plans: Any | None = None,
) -> float:
    if param_ratio_from_plans is not None:
        return float(param_ratio_from_plans(domain_plans))
    total = float(predicted_total_params or 0.0)
    if total <= 0:
        total = 1.0
    saved = 0.0
    for domain_id, plan in domain_plans.items():
        for index in plan.prune_indices:
            saved += _unit_param_saving(param_savings_by_unit, str(domain_id), int(index))
    return saved / total


def _ordinary_plan_for_raw_count(domain: V108PruningDomain, raw_count: int, align: int) -> tuple[list[int], list[int], RoundTo8Result]:
    rounded = tp_floor_keep(domain.num_root_channels, raw_count, align)
    ordered = _domain_sorted_units(domain)
    prune_units = ordered[: rounded.final_n_pruned]
    prune = sorted({int(unit.root_channel_index) for unit in prune_units})
    keep = [idx for idx in range(domain.num_root_channels) if idx not in set(prune)]
    return prune, keep, rounded


def _predicted_param_ratio_with_candidate(
    domain_plans: Mapping[str, V108DomainSelection],
    *,
    domain_id: str,
    candidate_prune_indices: Sequence[int],
    param_savings_by_unit: Mapping[str, Mapping[int, float]] | None,
    predicted_total_params: float,
    param_ratio_from_plans: Any | None = None,
) -> float:
    if param_ratio_from_plans is not None:
        original = domain_plans[str(domain_id)]
        old_prune = list(original.prune_indices)
        old_keep = list(original.keep_indices)
        try:
            original.prune_indices = list(candidate_prune_indices)
            original.keep_indices = [idx for idx in original.keep_indices if idx not in set(candidate_prune_indices)]
            return float(param_ratio_from_plans(domain_plans))
        finally:
            original.prune_indices = old_prune
            original.keep_indices = old_keep
    total = float(predicted_total_params or 0.0)
    if total <= 0:
        total = 1.0
    saved = 0.0
    for existing_domain_id, plan in domain_plans.items():
        indices = list(candidate_prune_indices) if str(existing_domain_id) == str(domain_id) else list(plan.prune_indices)
        for index in indices:
            saved += _unit_param_saving(param_savings_by_unit, str(existing_domain_id), int(index))
    return saved / total


def _make_grouped_candidates(domain: V108PruningDomain, max_ch_sparsity: float, align: int) -> list[dict[str, Any]]:
    if not domain.is_grouped_conv or int(domain.groups) <= 1:
        return []
    per = int(domain.per_group or (domain.num_root_channels // int(domain.groups)))
    scores = torch.zeros(domain.num_root_channels, dtype=torch.float32)
    for unit in domain.units:
        if 0 <= int(unit.root_channel_index) < int(scores.numel()):
            scores[int(unit.root_channel_index)] = float(unit.importance_normalized)
    candidates = []
    seen: set[tuple[int, tuple[int, ...]]] = set()
    for raw_prune_per_group in range(1, per + 1):
        decision = grouped_per_group_aligned_independent_local_pruning(
            module_name=domain.root_module_name,
            scores=scores,
            groups=int(domain.groups),
            raw_prune_per_group=raw_prune_per_group,
            max_ch_sparsity=max_ch_sparsity,
            align=align,
        )
        if not decision.output_pruned:
            continue
        key = (decision.out_per_group_after, tuple(decision.global_prune_indices))
        if key in seen:
            continue
        seen.add(key)
        if decision.global_prune_indices:
            score = float(scores[decision.global_prune_indices].mean().item())
        else:
            score = float("inf")
        candidates.append(
            {
                "candidate_type": "grouped_pergroup8_block",
                "domain_id": domain.pruning_domain_id,
                "importance": score,
                "slots": len(decision.global_prune_indices),
                "decision": decision,
            }
        )
    return candidates


def _grouped_no_prune_report(domain: V108PruningDomain, reason: str, align: int) -> dict[str, Any]:
    decision = grouped_per_group_aligned_independent_local_pruning(
        module_name=domain.root_module_name,
        scores=[unit.importance_normalized for unit in domain.units],
        groups=max(1, int(domain.groups)),
        raw_prune_per_group=0,
        align=align,
    )
    decision.skipped_output_prune_reason = reason
    decision.output_pruned = False
    decision.global_keep_indices = list(range(int(domain.num_root_channels)))
    decision.global_prune_indices = []
    return decision.to_report_row()


def select_greedy_global_budget(
    domains: Sequence[V108PruningDomain],
    *,
    target_pruning_ratio: float,
    target_pruning_mode: str = "channel",
    predicted_total_params: float | None = None,
    param_savings_by_unit: Mapping[str, Mapping[int, float]] | None = None,
    param_ratio_from_plans: Any | None = None,
    max_ch_sparsity: float = 0.60,
    align_channels: int = 8,
) -> V108GreedySelectionResult:
    """Greedily select low normalized-Taylor units under local legality caps."""

    mode = str(target_pruning_mode or "channel").lower()
    if mode not in {"channel", "param"}:
        raise ValueError(f"unsupported_target_pruning_mode:{target_pruning_mode}")
    domain_map = {domain.pruning_domain_id: domain for domain in domains}
    searchable_units = [
        unit
        for domain in domains
        if not domain.protected_reason and not domain.skipped_reason
        for unit in domain.units
        if not unit.protected_reason and not unit.skipped_reason
    ]
    total_slots = len(searchable_units)
    target_slots = int(math.ceil(total_slots * float(target_pruning_ratio)))
    total_for_param_prediction = float(predicted_total_params or total_slots or 1.0)
    domain_raw_counts: dict[str, int] = {domain.pruning_domain_id: 0 for domain in domains}
    domain_plans: dict[str, V108DomainSelection] = {
        domain.pruning_domain_id: V108DomainSelection(pruning_domain_id=domain.pruning_domain_id)
        for domain in domains
    }
    grouped_selected: set[str] = set()
    candidates: list[dict[str, Any]] = []
    for domain in domains:
        if domain.protected_reason or domain.skipped_reason:
            for unit in domain.units:
                if domain.skipped_reason and not unit.skipped_reason:
                    unit.skipped_reason = domain.skipped_reason
            continue
        if domain.is_grouped_conv:
            grouped = _make_grouped_candidates(domain, max_ch_sparsity=max_ch_sparsity, align=align_channels)
            if not grouped:
                for unit in domain.units:
                    unit.skipped_reason = "per_group_after_below8" if int(align_channels) == 8 else "per_group_after_below_round_to"
            candidates.extend(grouped)
            continue
        for unit in domain.units:
            if unit.protected_reason:
                continue
            candidates.append(
                {
                    "candidate_type": "coupled_channel_unit",
                    "domain_id": domain.pruning_domain_id,
                    "importance": _unit_score(unit),
                    "slots": 1,
                    "unit": unit,
                }
            )
    candidates.sort(key=lambda row: (float(row["importance"]), str(row["domain_id"])))

    trace_rows: list[dict[str, Any]] = []
    raw_selected_slots = 0
    max_blocked = 0
    pergroup_skipped = sum(
        1
        for domain in domains
        if (
            domain.is_grouped_conv
            and not domain.protected_reason
            and not domain.skipped_reason
            and not _make_grouped_candidates(domain, max_ch_sparsity=max_ch_sparsity, align=align_channels)
        )
    )
    predicted_param_prune_ratio = 0.0
    for rank, candidate in enumerate(candidates, start=1):
        domain_id = str(candidate["domain_id"])
        domain = domain_map[domain_id]
        selected = False
        reject_reason = ""
        round_applied = False
        max_block = False
        grouped_applied = False
        per_group_before = ""
        per_group_after = ""
        budget_reached = (
            raw_selected_slots >= target_slots
            if mode == "channel"
            else predicted_param_prune_ratio >= float(target_pruning_ratio)
        )
        predicted_before = predicted_param_prune_ratio
        predicted_after = predicted_param_prune_ratio
        if budget_reached:
            reject_reason = "target_reached"
        elif candidate["candidate_type"] == "grouped_pergroup8_block":
            if domain_id in grouped_selected:
                reject_reason = "grouped_domain_already_selected"
            else:
                decision: GroupedPerGroup8Decision = candidate["decision"]
                final_ratio = len(decision.global_prune_indices) / max(domain.num_root_channels, 1)
                if final_ratio > float(max_ch_sparsity) + 1e-12:
                    reject_reason = "max_ch_sparsity_exceeded"
                    max_block = True
                    max_blocked += 1
                else:
                    predicted_after = _predicted_param_ratio_with_candidate(
                        domain_plans,
                        domain_id=domain_id,
                        candidate_prune_indices=decision.global_prune_indices,
                        param_savings_by_unit=param_savings_by_unit,
                        predicted_total_params=total_for_param_prediction,
                        param_ratio_from_plans=param_ratio_from_plans,
                    )
                    selected = True
                    grouped_selected.add(domain_id)
                    grouped_applied = True
                    domain_plans[domain_id].grouped_decision = decision
                    domain_plans[domain_id].final_n_pruned = len(decision.global_prune_indices)
                    domain_plans[domain_id].prune_indices = list(decision.global_prune_indices)
                    domain_plans[domain_id].keep_indices = list(decision.global_keep_indices)
                    raw_selected_slots += len(decision.global_prune_indices)
                    predicted_param_prune_ratio = predicted_after
                    per_group_before = decision.out_per_group_before
                    per_group_after = decision.out_per_group_after
        else:
            next_raw = domain_raw_counts[domain_id] + 1
            candidate_prune, candidate_keep, rounded = _ordinary_plan_for_raw_count(domain, next_raw, align_channels)
            round_applied = rounded.final_n_pruned != next_raw
            min_after = int(align_channels) if domain.num_root_channels >= int(align_channels) else 1
            if rounded.aligned_keep < min_after:
                reject_reason = "aligned_keep_below_min8"
            elif rounded.final_n_pruned / max(domain.num_root_channels, 1) > float(max_ch_sparsity) + 1e-12:
                reject_reason = "max_ch_sparsity_exceeded"
                max_block = True
                max_blocked += 1
            else:
                selected = True
                unit: V108RankingUnit = candidate["unit"]
                unit.selected_for_pruning = True
                domain_raw_counts[domain_id] = next_raw
                domain_plans[domain_id].raw_selected_count = next_raw
                domain_plans[domain_id].final_n_pruned = rounded.final_n_pruned
                domain_plans[domain_id].rounding_overshoot_slots = rounded.rounding_overshoot_slots
                domain_plans[domain_id].prune_indices = list(candidate_prune)
                domain_plans[domain_id].keep_indices = list(candidate_keep)
                raw_selected_slots += 1
                predicted_after = _predicted_param_ratio(
                    domain_plans,
                    param_savings_by_unit=param_savings_by_unit,
                    predicted_total_params=total_for_param_prediction,
                    param_ratio_from_plans=param_ratio_from_plans,
                )
                predicted_param_prune_ratio = predicted_after
        trace_rows.append(
            {
                "target": float(target_pruning_ratio),
                "target_pruning_mode": mode,
                "rank": rank,
                "pruning_domain_id": domain_id,
                "coupled_unit_id": getattr(candidate.get("unit"), "coupled_unit_id", ""),
                "module_names": domain.root_module_name,
                "importance_normalized": candidate["importance"],
                "candidate_type": candidate["candidate_type"],
                "current_pruned_slots_before": raw_selected_slots - (candidate["slots"] if selected and candidate["candidate_type"] == "grouped_pergroup8_block" else (1 if selected else 0)),
                "current_pruned_slots_after": raw_selected_slots,
                "predicted_param_prune_ratio_before": predicted_before,
                "predicted_param_prune_ratio_after": predicted_after,
                "selected": selected,
                "reject_reason": reject_reason,
                "round_to8_applied": round_applied,
                "round_to_applied": round_applied,
                "max_ch_sparsity_blocked": max_block,
                "grouped_pergroup_policy_applied": grouped_applied,
                "per_group_before": per_group_before,
                "per_group_after": per_group_after,
                "final_actual_channel_ratio": 0.0,
                "final_actual_param_ratio": 0.0,
            }
        )

    selected_units: list[V108RankingUnit] = []
    skipped_units: list[V108RankingUnit] = []
    total_final_pruned = 0
    total_overshoot = 0
    grouped_shape_rows: list[dict[str, Any]] = []
    for domain in domains:
        plan = domain_plans[domain.pruning_domain_id]
        if domain.is_grouped_conv:
            if plan.grouped_decision is not None:
                grouped_shape_rows.append(plan.grouped_decision.to_report_row())
                prune_set = set(plan.prune_indices)
                for unit in domain.units:
                    unit.selected_for_pruning = int(unit.root_channel_index) in prune_set
                selected_units.extend([unit for unit in domain.units if unit.selected_for_pruning])
            else:
                reason = domain.skipped_reason or domain.protected_reason or ""
                if not reason:
                    reason = next((unit.skipped_reason for unit in domain.units if unit.skipped_reason), "")
                if not reason:
                    reason = "not_selected"
                grouped_shape_rows.append(_grouped_no_prune_report(domain, reason, align_channels))
                skipped_units.extend(domain.units)
            total_final_pruned += int(plan.final_n_pruned)
            continue
        if domain.skipped_reason:
            skipped_units.extend(domain.units)
            plan.keep_indices = list(range(domain.num_root_channels))
            plan.skipped_reason = domain.skipped_reason
            continue
        raw_count = domain_raw_counts.get(domain.pruning_domain_id, 0)
        if raw_count <= 0:
            plan.keep_indices = list(range(domain.num_root_channels))
            skipped_units.extend(domain.units)
            continue
        rounded = tp_floor_keep(domain.num_root_channels, raw_count, align_channels)
        plan.raw_selected_count = raw_count
        plan.final_n_pruned = rounded.final_n_pruned
        plan.rounding_overshoot_slots = rounded.rounding_overshoot_slots
        ordered = _domain_sorted_units(domain)
        prune_units = ordered[: rounded.final_n_pruned]
        prune_set = {int(unit.root_channel_index) for unit in prune_units}
        plan.prune_indices = sorted(prune_set)
        plan.keep_indices = [idx for idx in range(domain.num_root_channels) if idx not in prune_set]
        for unit in domain.units:
            unit.selected_for_pruning = int(unit.root_channel_index) in prune_set
        selected_units.extend(prune_units)
        skipped_units.extend([unit for unit in domain.units if not unit.selected_for_pruning])
        total_final_pruned += rounded.final_n_pruned
        total_overshoot += rounded.rounding_overshoot_slots

    actual_channel_ratio = total_final_pruned / total_slots if total_slots else 0.0
    predicted_param_prune_ratio = _predicted_param_ratio(
        domain_plans,
        param_savings_by_unit=param_savings_by_unit,
        predicted_total_params=total_for_param_prediction,
        param_ratio_from_plans=param_ratio_from_plans,
    )
    param_budget_overshoot = max(0.0, predicted_param_prune_ratio - float(target_pruning_ratio)) if mode == "param" else 0.0
    unreachable = (
        predicted_param_prune_ratio + 1e-12 < float(target_pruning_ratio)
        if mode == "param"
        else actual_channel_ratio + 1e-12 < float(target_pruning_ratio)
    )
    for row in trace_rows:
        row["final_actual_channel_ratio"] = actual_channel_ratio
        row["final_actual_param_ratio"] = predicted_param_prune_ratio
        row["final_predicted_param_prune_ratio"] = predicted_param_prune_ratio
    return V108GreedySelectionResult(
        target_pruning_mode=mode,
        target_pruning_ratio=float(target_pruning_ratio),
        actual_channel_prune_ratio_on_searchable_surface=actual_channel_ratio,
        actual_param_prune_ratio=predicted_param_prune_ratio if mode == "param" else actual_channel_ratio,
        predicted_param_prune_ratio=predicted_param_prune_ratio,
        param_prediction_error=0.0,
        param_budget_overshoot_ratio=param_budget_overshoot,
        actual_flops_proxy_ratio=None,
        rounding_overshoot_ratio=total_overshoot / total_slots if total_slots else 0.0,
        max_ch_sparsity_blocked_count=max_blocked,
        skipped_because_pergroup_below8_count=pergroup_skipped,
        skipped_because_pergroup_below_round_to_count=pergroup_skipped,
        unreachable=unreachable,
        domain_plans=domain_plans,
        selected_units=selected_units,
        skipped_units=skipped_units,
        trace_rows=trace_rows,
        grouped_shape_rows=grouped_shape_rows,
    )
