"""Conditional Taylor shortlist repair without changing GA precision genes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ..candidate import CandidateGenotype
from ..hashing import canonical_json_hash


@dataclass(frozen=True)
class ConditionalRepairResult:
    status: str
    repaired_mask: dict[str, int]
    raw_prune_count: int
    legal_prune_count: int
    failure_reason: str = ""
    group_keep_map: dict[int, list[int]] = field(default_factory=dict)
    group_prune_map: dict[int, list[int]] = field(default_factory=dict)
    shared_position_strategy: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


def _normalized_mask(raw_mask: Mapping[str, int]) -> dict[str, int]:
    return {str(key): 1 if int(value) else 0 for key, value in raw_mask.items()}


def _ordered_by_cost(
    unit_ids: Sequence[str], conditional_costs: Mapping[str, float]
) -> list[str]:
    missing = [str(unit_id) for unit_id in unit_ids if str(unit_id) not in conditional_costs]
    if missing:
        raise RuntimeError(f"conditional_importance_missing:{','.join(sorted(missing))}")
    return sorted(
        (str(unit_id) for unit_id in unit_ids),
        key=lambda unit_id: (float(conditional_costs[unit_id]), unit_id),
    )


def conditional_dense_floor_repair(
    raw_mask: Mapping[str, int],
    *,
    conditional_costs: Mapping[str, float],
    alignment: int = 4,
    minimum_width: int = 1,
) -> ConditionalRepairResult:
    normalized = _normalized_mask(raw_mask)
    unit_ids = tuple(sorted(normalized))
    raw_prune = sum(1 for keep in normalized.values() if keep == 0)
    align = max(1, int(alignment))
    legal_prune = align * (raw_prune // align)
    if len(unit_ids) - legal_prune < int(minimum_width):
        return ConditionalRepairResult(
            "failed",
            normalized,
            raw_prune,
            legal_prune,
            failure_reason="conditional_dense_target_below_minimum_width",
        )
    ordered = _ordered_by_cost(unit_ids, conditional_costs)
    selected = set(ordered[:legal_prune])
    repaired = {unit_id: 0 if unit_id in selected else 1 for unit_id in unit_ids}
    return ConditionalRepairResult(
        "ok",
        repaired,
        raw_prune,
        legal_prune,
        metadata={
            "repair_mode": "conditional_joint_taylor_dense_prune_count_floor",
            "alignment": align,
            "prune_count_not_increased": legal_prune <= raw_prune,
        },
    )


def conditional_grouped_floor_repair(
    raw_mask: Mapping[str, int],
    *,
    physical_groups: Mapping[int, Sequence[str]],
    local_indices: Mapping[int, Mapping[str, int]],
    conditional_costs: Mapping[str, float],
    allowed_channels_per_group: Sequence[int],
) -> ConditionalRepairResult:
    normalized = _normalized_mask(raw_mask)
    allowed = tuple(sorted({int(width) for width in allowed_channels_per_group if int(width) > 0}))
    if not allowed or not physical_groups:
        return ConditionalRepairResult(
            "failed", normalized, 0, 0, failure_reason="conditional_grouped_domain_empty"
        )
    repaired = dict(normalized)
    keep_map: dict[int, list[int]] = {}
    prune_map: dict[int, list[int]] = {}
    raw_total = 0
    legal_total = 0
    targets: set[int] = set()
    for physical_group, raw_units in sorted(physical_groups.items()):
        units = tuple(str(unit_id) for unit_id in raw_units)
        raw_prune = sum(1 for unit_id in units if normalized.get(unit_id, 1) == 0)
        raw_keep = len(units) - raw_prune
        legal_keeps = [width for width in allowed if raw_keep <= width <= len(units)]
        if not legal_keeps:
            return ConditionalRepairResult(
                "failed",
                normalized,
                raw_total + raw_prune,
                legal_total,
                failure_reason=f"conditional_grouped_no_legal_keep_width:{physical_group}:{raw_keep}",
            )
        target_keep = min(legal_keeps)
        targets.add(target_keep)
        legal_prune = len(units) - target_keep
        ordered = _ordered_by_cost(units, conditional_costs)
        selected = set(ordered[:legal_prune])
        for unit_id in units:
            repaired[unit_id] = 0 if unit_id in selected else 1
        keep_map[int(physical_group)] = sorted(
            int(local_indices[physical_group][unit_id])
            for unit_id in units
            if unit_id not in selected
        )
        prune_map[int(physical_group)] = sorted(
            int(local_indices[physical_group][unit_id])
            for unit_id in units
            if unit_id in selected
        )
        raw_total += raw_prune
        legal_total += legal_prune
    if len(targets) != 1:
        return ConditionalRepairResult(
            "failed",
            normalized,
            raw_total,
            legal_total,
            failure_reason="conditional_grouped_unequal_legal_width",
            group_keep_map=keep_map,
            group_prune_map=prune_map,
        )
    return ConditionalRepairResult(
        "ok",
        repaired,
        raw_total,
        legal_total,
        group_keep_map=keep_map,
        group_prune_map=prune_map,
        shared_position_strategy=False,
        metadata={
            "repair_mode": "conditional_joint_taylor_independent_physical_group_floor",
            "target_channels_per_group": next(iter(targets)),
            "prune_count_not_increased": legal_total <= raw_total,
        },
    )


def repair_candidate_domains(
    candidate: CandidateGenotype,
    *,
    dense_domains: Mapping[str, Sequence[str]],
    grouped_domains: Mapping[str, Mapping[str, Any]],
    conditional_costs: Mapping[str, float],
    dense_alignment: int,
    minimum_width_by_domain: Mapping[str, int],
) -> tuple[CandidateGenotype, dict[str, Any]]:
    mask = dict(candidate.pruning_genes)
    domain_reports: dict[str, Any] = {}
    for domain_id, unit_ids in sorted(dense_domains.items()):
        domain_mask = {str(unit_id): mask.get(str(unit_id), 1) for unit_id in unit_ids}
        result = conditional_dense_floor_repair(
            domain_mask,
            conditional_costs=conditional_costs,
            alignment=dense_alignment,
            minimum_width=int(minimum_width_by_domain.get(domain_id, 1)),
        )
        domain_reports[domain_id] = result
        if result.status != "ok":
            raise RuntimeError(result.failure_reason)
        mask.update(result.repaired_mask)
    for domain_id, spec in sorted(grouped_domains.items()):
        all_units = [
            str(unit_id)
            for units in spec["physical_groups"].values()
            for unit_id in units
        ]
        result = conditional_grouped_floor_repair(
            {unit_id: mask.get(unit_id, 1) for unit_id in all_units},
            physical_groups=spec["physical_groups"],
            local_indices=spec["local_indices"],
            conditional_costs=conditional_costs,
            allowed_channels_per_group=spec["allowed_channels_per_group"],
        )
        domain_reports[domain_id] = result
        if result.status != "ok":
            raise RuntimeError(result.failure_reason)
        mask.update(result.repaired_mask)
    raw_precision_hash = canonical_json_hash(dict(sorted(candidate.precision_genes.items())))
    repaired = CandidateGenotype(
        pruning_genes=mask,
        precision_genes=dict(candidate.precision_genes),
        meta={**dict(candidate.meta), "repair_mode": "conditional_joint_taylor_shortlist"},
    )
    repaired_precision_hash = canonical_json_hash(dict(sorted(repaired.precision_genes.items())))
    report = {
        "status": "ok",
        "domain_reports": domain_reports,
        "raw_precision_gene_hash": raw_precision_hash,
        "repaired_precision_gene_hash": repaired_precision_hash,
        "precision_profile_modified": raw_precision_hash != repaired_precision_hash,
    }
    if report["precision_profile_modified"]:
        raise RuntimeError("conditional_repair_modified_precision_profile")
    return repaired, report

