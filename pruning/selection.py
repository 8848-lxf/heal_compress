"""Selection layer for TP-style coupled channel pruning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn

from ..tracer.pruning_group import PruningGroup
from .units import (
    AtomicPruneUnit,
    ConcreteCoupledPruningGroup,
    CoupledChannelUnit,
    expand_coupled_channel_units,
    grouped_conv_info,
    instantiate_concrete_pruning_group,
    item_key,
    scope_constraints,
)


@dataclass
class SelectionConfig:
    prune_ratio: float = 0.0
    selection_mode: str = "local_scope"
    group_conv_selection_mode: str = "flat_output_groups_fixed"
    align: int = 16
    group_conv_align: int = 8
    group_conv_prune_mode: str = "keep_groups"
    allow_remove_groups: bool = False
    min_groups_after_prune: int = 8
    groups_align: int = 8
    min_channels: int = 1
    importance_mode: str | None = None


@dataclass
class PruningPlan:
    coupled_units: list[CoupledChannelUnit] = field(default_factory=list)
    atomic_units: list[AtomicPruneUnit] = field(default_factory=list)
    selected_atomic_units: list[AtomicPruneUnit] = field(default_factory=list)
    concrete_groups: list[ConcreteCoupledPruningGroup] = field(default_factory=list)
    grouped_conv_reports: list[dict[str, Any]] = field(default_factory=list)

    @property
    def selected_coupled_unit_ids(self) -> set[str]:
        selected: set[str] = set()
        for candidate in self.selected_atomic_units:
            selected.update(candidate.source_coupled_units)
        return selected


def _aligned_keep_count(channels: int, prune_ratio: float, align: int, min_channels: int) -> int:
    target = int(round(channels * (1.0 - prune_ratio)))
    target = max(min_channels, min(channels, target))
    if align > 1 and target >= align and target % align != 0:
        aligned = ((target + align - 1) // align) * align
        target = min(channels, max(min_channels, aligned))
    return max(0, min(channels, target))


def _scope_scores(scope: PruningGroup, scope_importance: Mapping[str, Any]) -> torch.Tensor:
    raw = scope_importance.get(scope.group_id)
    if raw is None:
        return torch.arange(int(scope.num_channels), dtype=torch.float32)
    if torch.is_tensor(raw):
        scores = raw.detach().float().cpu()
    else:
        scores = torch.as_tensor(raw, dtype=torch.float32)
    if int(scores.numel()) != int(scope.num_channels):
        raise ValueError(
            f"scope importance shape mismatch for {scope.group_id}: "
            f"got {int(scores.numel())}, expected {scope.num_channels}"
        )
    return scores


def _unit_by_idx(units: Sequence[CoupledChannelUnit]) -> dict[int, CoupledChannelUnit]:
    return {int(unit.root_idx): unit for unit in units}


def _local_indices_by_item(scope: PruningGroup, ref_indices: Sequence[int]) -> dict[str, list[int]]:
    return {item_key(item): item.local_keep(list(ref_indices)) for item in scope.items}


def _candidate_importance(units: Sequence[CoupledChannelUnit], ref_indices: Sequence[int], reduction: str = "sum") -> float:
    idx_map = _unit_by_idx(units)
    values = [
        float(idx_map[int(idx)].importance or 0.0)
        for idx in ref_indices
        if int(idx) in idx_map and idx_map[int(idx)].importance is not None
    ]
    if not values:
        return 0.0
    if reduction == "mean":
        return float(sum(values) / len(values))
    return float(sum(values))


def _plain_candidate_type(scope: PruningGroup) -> str:
    gt = str((scope.meta or {}).get("group_type", ""))
    if gt == "add":
        return "residual_channel"
    if gt == "cat":
        return "concat_channel"
    if gt == "transformer_head_group":
        return "transformer_head"
    if "ffn" in gt:
        return "transformer_ffn_channel"
    if "hidden" in gt and scope.protected:
        return "protected_hidden"
    return "plain_channel"


def _plain_candidates(
    scope: PruningGroup,
    units: Sequence[CoupledChannelUnit],
    *,
    importance_mode: str | None,
) -> list[AtomicPruneUnit]:
    ctype = _plain_candidate_type(scope)
    candidates: list[AtomicPruneUnit] = []
    for unit in units:
        candidates.append(
            AtomicPruneUnit(
                candidate_id=f"{scope.group_id}::cand::{unit.root_idx}",
                scope_id=scope.group_id,
                candidate_type=ctype,
                source_coupled_units=[unit.unit_id],
                ref_indices=[int(unit.root_idx)],
                local_indices_by_item=dict(unit.local_indices_by_item),
                importance=unit.importance,
                importance_mode=importance_mode,
                protected=unit.protected,
                protected_reason=unit.protected_reason,
                constraints=dict(unit.constraints),
                metadata={"root_idx": int(unit.root_idx)},
            )
        )
    return candidates


def _grouped_keep_per_group(channels: int, groups: int, cfg: SelectionConfig) -> int:
    per = channels // groups
    target_total = _aligned_keep_count(channels, cfg.prune_ratio, cfg.align, cfg.min_channels)
    keep_per = max(1, target_total // groups)
    if cfg.group_conv_align > 1 and keep_per >= cfg.group_conv_align:
        if keep_per % cfg.group_conv_align != 0:
            keep_per = ((keep_per + cfg.group_conv_align - 1) // cfg.group_conv_align) * cfg.group_conv_align
    if keep_per < cfg.group_conv_align <= per:
        keep_per = cfg.group_conv_align
    return min(per, keep_per)


def _grouped_report_base(
    scope: PruningGroup,
    cfg: SelectionConfig,
    grouped: dict[str, Any],
    scores: torch.Tensor,
) -> dict[str, Any]:
    groups = int(grouped["groups"])
    per = int(grouped["per_group"])
    return {
        "scope_id": scope.group_id,
        "module_name": grouped.get("module_name", ""),
        "selection_mode": cfg.selection_mode,
        "group_conv_selection_mode": cfg.group_conv_selection_mode,
        "groups_before": groups,
        "groups_after": groups,
        "per_group_before": per,
        "per_group_after": per,
        "group_conv_align": cfg.group_conv_align,
        "groups_align": cfg.groups_align,
        "shared_local_positions": [],
        "local_position_importance": [],
        "group_keep_map": {},
        "expanded_keep_indices": list(range(int(scope.num_channels))),
        "expanded_prune_indices": [],
        "per_group_kept_count": {},
        "per_group_kept_count_align8": True,
        "scope_importance_shape": [int(scores.numel())],
        "grouped_importance_matrix_shape": [groups, per],
        "importance_source_items": ["scope_level_aggregated"],
        "structure_legal": True,
    }


def _build_shared_local_candidates(
    scope: PruningGroup,
    units: Sequence[CoupledChannelUnit],
    scores: torch.Tensor,
    cfg: SelectionConfig,
    grouped: dict[str, Any],
) -> tuple[list[AtomicPruneUnit], list[AtomicPruneUnit], dict[str, Any]]:
    groups = int(grouped["groups"])
    per = int(grouped["per_group"])
    keep_per = _grouped_keep_per_group(int(scope.num_channels), groups, cfg)
    matrix = scores.view(groups, per)
    local_imp = matrix.mean(dim=0)
    if keep_per >= per:
        local_keep = list(range(per))
    else:
        _, keep_idx = torch.topk(local_imp, keep_per, largest=True, sorted=False)
        local_keep = sorted(int(v) for v in keep_idx.tolist())
    local_keep_set = set(local_keep)
    local_prune = [idx for idx in range(per) if idx not in local_keep_set]
    expanded_keep = [g * per + p for g in range(groups) for p in local_keep]
    expanded_prune = [g * per + p for g in range(groups) for p in local_prune]

    idx_to_unit = _unit_by_idx(units)
    candidates: list[AtomicPruneUnit] = []
    for local in local_prune:
        ref = [g * per + local for g in range(groups)]
        candidates.append(
            AtomicPruneUnit(
                candidate_id=f"{scope.group_id}::shared_local::{local}",
                scope_id=scope.group_id,
                candidate_type="grouped_shared_local_position",
                source_coupled_units=[idx_to_unit[idx].unit_id for idx in ref],
                ref_indices=ref,
                local_indices_by_item=_local_indices_by_item(scope, ref),
                importance=float(local_imp[local]),
                importance_mode=cfg.importance_mode,
                protected=bool(scope.protected),
                protected_reason=scope.protected_reason or None,
                constraints=scope_constraints(scope),
                metadata={
                    "local_position": local,
                    "groups": groups,
                    "per_group_before": per,
                    "per_group_after": keep_per,
                },
            )
        )
    block_candidate = AtomicPruneUnit(
        candidate_id=f"{scope.group_id}::shared_local_block",
        scope_id=scope.group_id,
        candidate_type="grouped_shared_local_block",
        source_coupled_units=[
            idx_to_unit[idx].unit_id
            for idx in expanded_prune
        ],
        ref_indices=expanded_prune,
        local_indices_by_item=_local_indices_by_item(scope, expanded_prune),
        importance=_candidate_importance(units, expanded_prune),
        importance_mode=cfg.importance_mode,
        protected=bool(scope.protected),
        protected_reason=scope.protected_reason or None,
        constraints=scope_constraints(scope),
        metadata={
            "local_prune_positions": local_prune,
            "local_keep_positions": local_keep,
            "groups": groups,
            "per_group_before": per,
            "per_group_after": keep_per,
        },
    )
    if expanded_prune:
        candidates.append(block_candidate)

    report = _grouped_report_base(scope, cfg, grouped, scores)
    report.update(
        {
            "groups_after": groups,
            "per_group_after": keep_per,
            "shared_local_positions": local_keep,
            "local_position_importance": [float(v) for v in local_imp.tolist()],
            "expanded_keep_indices": expanded_keep,
            "expanded_prune_indices": expanded_prune,
            "per_group_kept_count": {g: keep_per for g in range(groups)},
            "per_group_kept_count_align8": bool(keep_per > 0 and keep_per % cfg.group_conv_align == 0),
            "structure_legal": bool(keep_per > 0 and keep_per % cfg.group_conv_align == 0),
        }
    )
    selected = [block_candidate] if expanded_prune else []
    return candidates, selected, report


def _build_independent_topk_candidate(
    scope: PruningGroup,
    units: Sequence[CoupledChannelUnit],
    scores: torch.Tensor,
    cfg: SelectionConfig,
    grouped: dict[str, Any],
) -> tuple[list[AtomicPruneUnit], list[AtomicPruneUnit], dict[str, Any]]:
    groups = int(grouped["groups"])
    per = int(grouped["per_group"])
    keep_per = _grouped_keep_per_group(int(scope.num_channels), groups, cfg)
    matrix = scores.view(groups, per)
    group_keep_map: dict[int, list[int]] = {}
    expanded_keep: list[int] = []
    for group_id in range(groups):
        if keep_per >= per:
            keep = list(range(per))
        else:
            _, keep_idx = torch.topk(matrix[group_id], keep_per, largest=True, sorted=False)
            keep = sorted(int(v) for v in keep_idx.tolist())
        group_keep_map[group_id] = keep
        expanded_keep.extend(group_id * per + local for local in keep)
    keep_set = set(expanded_keep)
    expanded_prune = [idx for idx in range(groups * per) if idx not in keep_set]
    idx_to_unit = _unit_by_idx(units)
    candidate = AtomicPruneUnit(
        candidate_id=f"{scope.group_id}::independent_group_topk",
        scope_id=scope.group_id,
        candidate_type="grouped_independent_topk",
        source_coupled_units=[idx_to_unit[idx].unit_id for idx in expanded_prune],
        ref_indices=expanded_prune,
        local_indices_by_item=_local_indices_by_item(scope, expanded_prune),
        importance=_candidate_importance(units, expanded_prune),
        importance_mode=cfg.importance_mode,
        protected=bool(scope.protected),
        protected_reason=scope.protected_reason or None,
        constraints=scope_constraints(scope),
        metadata={"group_keep_map": group_keep_map, "per_group_after": keep_per},
    )
    per_group_kept_count = {group_id: len(keep) for group_id, keep in group_keep_map.items()}
    aligned = all(count == keep_per and count % cfg.group_conv_align == 0 for count in per_group_kept_count.values())
    report = _grouped_report_base(scope, cfg, grouped, scores)
    report.update(
        {
            "groups_after": groups,
            "per_group_after": keep_per,
            "group_keep_map": group_keep_map,
            "expanded_keep_indices": expanded_keep,
            "expanded_prune_indices": expanded_prune,
            "per_group_kept_count": per_group_kept_count,
            "per_group_kept_count_align8": bool(aligned),
            "structure_legal": bool(aligned),
        }
    )
    selected = [] if keep_per >= per else [candidate]
    return [candidate], selected, report


def _build_remove_groups_candidate(
    scope: PruningGroup,
    units: Sequence[CoupledChannelUnit],
    scores: torch.Tensor,
    cfg: SelectionConfig,
    grouped: dict[str, Any],
) -> tuple[list[AtomicPruneUnit], list[AtomicPruneUnit], dict[str, Any]]:
    groups = int(grouped["groups"])
    per = int(grouped["per_group"])
    matrix = scores.view(groups, per)
    report = _grouped_report_base(scope, cfg, grouped, scores)
    if not cfg.allow_remove_groups:
        report.update({"structure_legal": False, "protected_reason": "remove_groups_not_allowed"})
        return [], [], report

    target_total = _aligned_keep_count(int(scope.num_channels), cfg.prune_ratio, cfg.align, cfg.min_channels)
    groups_after = max(1, min(groups, int(round(target_total / max(per, 1)))))
    if cfg.groups_align > 1 and groups_after >= cfg.groups_align:
        aligned = groups_after - (groups_after % cfg.groups_align)
        if aligned >= cfg.min_groups_after_prune:
            groups_after = aligned
    groups_after = max(cfg.min_groups_after_prune, groups_after)
    if cfg.groups_align > 1 and groups_after % cfg.groups_align != 0 and groups_after < groups:
        groups_after = min(groups, ((groups_after + cfg.groups_align - 1) // cfg.groups_align) * cfg.groups_align)
    groups_after = min(groups, groups_after)

    group_imp = matrix.sum(dim=1)
    if groups_after >= groups:
        keep_groups = list(range(groups))
    else:
        _, keep_idx = torch.topk(group_imp, groups_after, largest=True, sorted=False)
        keep_groups = sorted(int(v) for v in keep_idx.tolist())
    keep_group_set = set(keep_groups)
    prune_groups = [group_id for group_id in range(groups) if group_id not in keep_group_set]
    expanded_keep = [group_id * per + local for group_id in keep_groups for local in range(per)]
    expanded_prune = [group_id * per + local for group_id in prune_groups for local in range(per)]
    idx_to_unit = _unit_by_idx(units)
    candidate = AtomicPruneUnit(
        candidate_id=f"{scope.group_id}::remove_groups",
        scope_id=scope.group_id,
        candidate_type="grouped_whole_group",
        source_coupled_units=[idx_to_unit[idx].unit_id for idx in expanded_prune],
        ref_indices=expanded_prune,
        local_indices_by_item=_local_indices_by_item(scope, expanded_prune),
        importance=_candidate_importance(units, expanded_prune),
        importance_mode=cfg.importance_mode,
        protected=bool(scope.protected),
        protected_reason=scope.protected_reason or None,
        constraints=scope_constraints(scope),
        metadata={
            "keep_groups": keep_groups,
            "prune_groups": prune_groups,
            "group_block_importance": [float(v) for v in group_imp.tolist()],
        },
    )
    legal = bool(
        groups_after >= cfg.min_groups_after_prune
        and (cfg.groups_align <= 1 or groups_after % cfg.groups_align == 0)
        and (cfg.group_conv_align <= 1 or per % cfg.group_conv_align == 0)
    )
    report.update(
        {
            "groups_after": groups_after,
            "per_group_after": per,
            "group_keep_map": {group_id: list(range(per)) for group_id in keep_groups},
            "expanded_keep_indices": expanded_keep,
            "expanded_prune_indices": expanded_prune,
            "per_group_kept_count": {group_id: per for group_id in keep_groups},
            "per_group_kept_count_align8": bool(per % cfg.group_conv_align == 0),
            "local_position_importance": [float(v) for v in group_imp.tolist()],
            "structure_legal": legal,
        }
    )
    selected = [] if not expanded_prune else [candidate]
    return [candidate], selected, report


def _reinterpretation_metrics(c_out_before: int, c_out_after: int, groups: int, keep_idx: Sequence[int]) -> dict[str, Any]:
    if c_out_before <= 0 or c_out_after <= 0 or groups <= 0 or not keep_idx:
        return {
            "old_group_keep_count": {},
            "old_to_new_out_map": {},
            "reinterpretation_count": 0,
            "reinterpretation_ratio": 0.0,
            "grouped_keep_pattern_mismatch": False,
        }
    old_per = c_out_before // groups
    new_per = c_out_after // groups
    old_group_keep_count: dict[int, int] = {group_id: 0 for group_id in range(groups)}
    old_to_new_out_map: dict[int, int] = {}
    reinterpretation_count = 0
    for new_idx, old_idx in enumerate(sorted(int(v) for v in keep_idx)):
        old_group = old_idx // max(old_per, 1)
        new_group = new_idx // max(new_per, 1)
        old_group_keep_count[old_group] = old_group_keep_count.get(old_group, 0) + 1
        old_to_new_out_map[old_idx] = new_idx
        if old_group != new_group:
            reinterpretation_count += 1
    counts = list(old_group_keep_count.values())
    return {
        "old_group_keep_count": old_group_keep_count,
        "old_to_new_out_map": old_to_new_out_map,
        "reinterpretation_count": reinterpretation_count,
        "reinterpretation_ratio": reinterpretation_count / max(c_out_after, 1),
        "grouped_keep_pattern_mismatch": len(set(counts)) > 1,
    }


def _build_flat_output_groups_fixed_candidate(
    scope: PruningGroup,
    units: Sequence[CoupledChannelUnit],
    scores: torch.Tensor,
    cfg: SelectionConfig,
    grouped: dict[str, Any],
) -> tuple[list[AtomicPruneUnit], list[AtomicPruneUnit], dict[str, Any]]:
    groups = int(grouped["groups"])
    c_out = int(scope.num_channels)
    report = _grouped_report_base(scope, cfg, grouped, scores)
    if groups <= 1 or c_out % groups != 0:
        report.update({"structure_legal": False, "protected_reason": "invalid_regular_grouped_conv"})
        return [], [], report

    target_prune = int(round(c_out * float(cfg.prune_ratio)))
    legal_prune_counts = [count for count in range(0, c_out) if (c_out - count) > 0 and (c_out - count) % groups == 0]
    if not legal_prune_counts:
        report.update({"structure_legal": False, "protected_reason": "no_legal_cout_after"})
        return [], [], report
    prune_count = min(legal_prune_counts, key=lambda count: (abs(count - target_prune), count))
    if prune_count <= 0:
        expanded_prune: list[int] = []
        expanded_keep = list(range(c_out))
    else:
        _, prune_idx = torch.topk(scores, prune_count, largest=False, sorted=False)
        expanded_prune = sorted(int(v) for v in prune_idx.tolist())
        prune_set = set(expanded_prune)
        expanded_keep = [idx for idx in range(c_out) if idx not in prune_set]

    idx_to_unit = _unit_by_idx(units)
    candidate = AtomicPruneUnit(
        candidate_id=f"{scope.group_id}::flat_output_groups_fixed",
        scope_id=scope.group_id,
        candidate_type="grouped_flat_output_groups_fixed",
        source_coupled_units=[idx_to_unit[idx].unit_id for idx in expanded_prune if idx in idx_to_unit],
        ref_indices=expanded_prune,
        local_indices_by_item=_local_indices_by_item(scope, expanded_prune),
        importance=_candidate_importance(units, expanded_prune),
        importance_mode=cfg.importance_mode,
        protected=bool(scope.protected),
        protected_reason=scope.protected_reason or None,
        constraints=scope_constraints(scope),
        metadata={
            "groups": groups,
            "target_prune_count": target_prune,
            "adjusted_prune_count": prune_count,
            "ratio_adjusted": prune_count != target_prune,
        },
    )
    metrics = _reinterpretation_metrics(c_out, len(expanded_keep), groups, expanded_keep)
    group_keep_counts = metrics["old_group_keep_count"]
    report.update(
        {
            "groups_after": groups,
            "per_group_after": len(expanded_keep) // groups if groups else 0,
            "expanded_keep_indices": expanded_keep,
            "expanded_prune_indices": expanded_prune,
            "per_group_kept_count": group_keep_counts,
            "per_group_kept_count_align8": True,
            "structure_legal": bool(len(expanded_keep) > 0 and len(expanded_keep) % groups == 0),
            "target_prune_count": target_prune,
            "adjusted_prune_count": prune_count,
            "ratio_adjusted": prune_count != target_prune,
            "grouped_keep_pattern_mismatch": metrics["grouped_keep_pattern_mismatch"],
            "old_group_keep_count": group_keep_counts,
            "old_to_new_out_map": metrics["old_to_new_out_map"],
            "reinterpretation_count": metrics["reinterpretation_count"],
            "reinterpretation_ratio": metrics["reinterpretation_ratio"],
            "warnings": ["grouped_keep_pattern_mismatch"] if metrics["grouped_keep_pattern_mismatch"] else [],
        }
    )
    selected = [] if not expanded_prune else [candidate]
    return [candidate], selected, report


def _grouped_candidates(
    scope: PruningGroup,
    units: Sequence[CoupledChannelUnit],
    scores: torch.Tensor,
    cfg: SelectionConfig,
) -> tuple[list[AtomicPruneUnit], list[AtomicPruneUnit], dict[str, Any]]:
    grouped = grouped_conv_info(scope)
    if not grouped.get("has_grouped_conv") or not grouped.get("per_group"):
        return [], [], {}
    mode = cfg.group_conv_selection_mode
    if mode == "flat_output_groups_fixed":
        return _build_flat_output_groups_fixed_candidate(scope, units, scores, cfg, grouped)
    if mode == "group_balanced_output_groups_fixed":
        candidates, selected, report = _build_independent_topk_candidate(scope, units, scores, cfg, grouped)
        if report:
            report["group_conv_selection_mode"] = "group_balanced_output_groups_fixed"
            report["group_balance_pass"] = len(set(report.get("per_group_kept_count", {}).values())) <= 1
            report["reinterpretation_ratio"] = 0.0 if report["group_balance_pass"] else None
        for candidate in candidates:
            candidate.candidate_type = "grouped_group_balanced_output_groups_fixed"
            candidate.metadata["group_balance_pass"] = True
        return candidates, selected, report
    if mode == "group_coarsening_zero_padded_reblock":
        candidates, selected, report = _build_flat_output_groups_fixed_candidate(scope, units, scores, cfg, grouped)
        if report:
            report["group_coarsening_requested"] = True
            report["group_coarsening_resolver_status"] = "not_yet_bucket_local_zero_padded_reblock"
            report.setdefault("warnings", []).append("group_coarsening_zero_padded_reblock_not_completed")
        for candidate in candidates:
            candidate.candidate_type = "grouped_group_coarsening_zero_padded_reblock_attempt"
            candidate.metadata["group_coarsening_resolver_status"] = "not_yet_bucket_local_zero_padded_reblock"
        return candidates, selected, report
    if mode == "independent_group_topk":
        return _build_independent_topk_candidate(scope, units, scores, cfg, grouped)
    if mode in {"remove_groups", "true_group_block_pruning"}:
        return _build_remove_groups_candidate(scope, units, scores, cfg, grouped)
    return _build_shared_local_candidates(scope, units, scores, cfg, grouped)


def _select_local_scope(
    scopes: Sequence[PruningGroup],
    candidates_by_scope: Mapping[str, list[AtomicPruneUnit]],
    grouped_selected_by_scope: Mapping[str, list[AtomicPruneUnit]],
    cfg: SelectionConfig,
) -> list[AtomicPruneUnit]:
    selected: list[AtomicPruneUnit] = []
    for scope in scopes:
        if scope.protected:
            continue
        grouped_selected = list(grouped_selected_by_scope.get(scope.group_id, []))
        if grouped_selected:
            selected.extend(grouped_selected)
            continue
        candidates = [c for c in candidates_by_scope.get(scope.group_id, []) if not c.protected]
        if not candidates:
            continue
        keep = _aligned_keep_count(int(scope.num_channels), cfg.prune_ratio, cfg.align, cfg.min_channels)
        prune_count = max(0, int(scope.num_channels) - keep)
        if prune_count <= 0:
            continue
        candidates.sort(key=lambda c: float(c.importance or 0.0))
        selected.extend(candidates[:prune_count])
    return selected


def _select_global(
    candidates: Sequence[AtomicPruneUnit],
    total_units: int,
    scope_channels: Mapping[str, int],
    cfg: SelectionConfig,
) -> list[AtomicPruneUnit]:
    target = int(round(total_units * cfg.prune_ratio))
    if target <= 0:
        return []
    pool = [
        c for c in candidates
        if not c.protected and c.ref_indices and c.candidate_type != "grouped_shared_local_position"
    ]
    pool.sort(key=lambda c: float(c.importance or 0.0))
    selected: list[AtomicPruneUnit] = []
    pruned_by_scope: dict[str, int] = {}
    removed = 0
    for candidate in pool:
        if removed >= target:
            break
        scope_total = int(scope_channels.get(candidate.scope_id, 0))
        scope_pruned = int(pruned_by_scope.get(candidate.scope_id, 0))
        if scope_total > 0 and scope_total - scope_pruned - len(candidate.ref_indices) < cfg.min_channels:
            continue
        selected.append(candidate)
        pruned_by_scope[candidate.scope_id] = scope_pruned + len(candidate.ref_indices)
        removed += len(candidate.ref_indices)
    return selected


def _merge_concrete_groups(
    scopes: Sequence[PruningGroup],
    units_by_scope: Mapping[str, list[CoupledChannelUnit]],
    selected: Sequence[AtomicPruneUnit],
) -> list[ConcreteCoupledPruningGroup]:
    scope_map = {scope.group_id: scope for scope in scopes}
    selected_by_scope: dict[str, list[AtomicPruneUnit]] = {}
    for candidate in selected:
        selected_by_scope.setdefault(candidate.scope_id, []).append(candidate)

    concrete_groups: list[ConcreteCoupledPruningGroup] = []
    for scope_id, units in selected_by_scope.items():
        scope = scope_map.get(scope_id)
        if scope is None:
            continue
        prune: set[int] = set()
        for candidate in units:
            prune.update(int(idx) for idx in candidate.ref_indices)
        if not prune:
            continue
        concrete_groups.append(
            instantiate_concrete_pruning_group(
                scope,
                sorted(prune),
                coupled_units=units_by_scope.get(scope_id, []),
                atomic_units=units,
            )
        )
    return concrete_groups


def build_pruning_plan(
    scopes: Sequence[PruningGroup],
    scope_importance: Mapping[str, Any],
    cfg: SelectionConfig,
) -> PruningPlan:
    if cfg.selection_mode not in {"local_scope", "root_node_local_unit_ratio", "global_coupled_channel", "constrained_global"}:
        raise ValueError(f"Unsupported selection_mode: {cfg.selection_mode}")
    if cfg.group_conv_selection_mode not in {"shared_local_mean", "independent_group_topk", "remove_groups", "true_group_block_pruning", "flat_output_groups_fixed", "group_balanced_output_groups_fixed", "group_coarsening_zero_padded_reblock"}:
        raise ValueError(f"Unsupported group_conv_selection_mode: {cfg.group_conv_selection_mode}")

    coupled_units: list[CoupledChannelUnit] = []
    atomic_units: list[AtomicPruneUnit] = []
    grouped_reports: list[dict[str, Any]] = []
    units_by_scope: dict[str, list[CoupledChannelUnit]] = {}
    candidates_by_scope: dict[str, list[AtomicPruneUnit]] = {}
    grouped_selected_by_scope: dict[str, list[AtomicPruneUnit]] = {}

    for scope in scopes:
        scores = _scope_scores(scope, scope_importance)
        units = expand_coupled_channel_units(scope, scores, importance_mode=cfg.importance_mode)
        units_by_scope[scope.group_id] = units
        coupled_units.extend(units)
        grouped = grouped_conv_info(scope)
        if grouped.get("has_grouped_conv"):
            candidates, selected, report = _grouped_candidates(scope, units, scores, cfg)
            candidates_by_scope[scope.group_id] = candidates
            grouped_selected_by_scope[scope.group_id] = selected
            atomic_units.extend(candidates)
            if report:
                grouped_reports.append(report)
            continue
        candidates = _plain_candidates(scope, units, importance_mode=cfg.importance_mode)
        candidates_by_scope[scope.group_id] = candidates
        atomic_units.extend(candidates)

    if cfg.selection_mode in {"local_scope", "root_node_local_unit_ratio"}:
        selected = _select_local_scope(scopes, candidates_by_scope, grouped_selected_by_scope, cfg)
    else:
        # Grouped conv scopes contribute only constrained candidates here; no
        # naive per-channel candidate is emitted for those scopes.
        selected = _select_global(
            atomic_units,
            len(coupled_units),
            {scope.group_id: int(scope.num_channels) for scope in scopes},
            cfg,
        )

    concrete = _merge_concrete_groups(scopes, units_by_scope, selected)
    return PruningPlan(
        coupled_units=coupled_units,
        atomic_units=atomic_units,
        selected_atomic_units=selected,
        concrete_groups=concrete,
        grouped_conv_reports=grouped_reports,
    )
