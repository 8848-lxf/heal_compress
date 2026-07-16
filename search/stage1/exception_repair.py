"""Audited exception-only repair for legacy or corrupt external candidates."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Mapping

from ..encoding.legal_width_genotype import LegalWidthGenotype
from ..space.legal_width_inventory import LegalWidthInventory


@dataclass(frozen=True)
class LegacyWidthRepairResult:
    width_genes: dict[str, int]
    keep_widths: dict[str, int]
    group_mask: dict[str, int]
    precision_genes: dict[str, str]


class ExceptionOnlyRepairMonitor:
    def __init__(self) -> None:
        self.normal_candidate_count = 0
        self.repair_invocation_count = 0
        self.repair_reasons: Counter[str] = Counter()
        self.repair_changed_structure_count = 0
        self.repair_changed_precision_count = 0

    def observe_normal_candidate(
        self,
        candidate: LegalWidthGenotype,
        *,
        inventory: LegalWidthInventory,
        precision_actions: Mapping[str, tuple[str, ...]],
    ) -> None:
        candidate.validate(inventory, precision_actions)
        self.normal_candidate_count += 1

    def observe_repair(
        self,
        *,
        reason: str,
        structure_changed: bool,
        precision_changed: bool,
    ) -> None:
        self.repair_invocation_count += 1
        self.repair_reasons[str(reason)] += 1
        self.repair_changed_structure_count += int(bool(structure_changed))
        self.repair_changed_precision_count += int(bool(precision_changed))

    def report(self) -> dict[str, object]:
        denominator = self.normal_candidate_count + self.repair_invocation_count
        return {
            "normal_candidate_count": self.normal_candidate_count,
            "repair_invocation_count": self.repair_invocation_count,
            "repair_invocation_rate": (
                float(self.repair_invocation_count / denominator)
                if denominator
                else 0.0
            ),
            "repair_reason_histogram": dict(sorted(self.repair_reasons.items())),
            "repair_changed_structure_count": self.repair_changed_structure_count,
            "repair_changed_precision_count": self.repair_changed_precision_count,
        }


def _ordered_requested_pruned(
    unit_ids: tuple[str, ...],
    raw_mask: Mapping[str, int],
    conditional_costs: Mapping[str, float],
) -> list[str]:
    requested = [unit_id for unit_id in unit_ids if int(raw_mask.get(unit_id, 1)) == 0]
    missing = [unit_id for unit_id in requested if unit_id not in conditional_costs]
    if missing:
        raise RuntimeError(f"legacy_repair_conditional_cost_missing:{','.join(missing)}")
    return sorted(requested, key=lambda unit_id: (float(conditional_costs[unit_id]), unit_id))


def repair_legacy_mask_to_legal_width(
    raw_mask: Mapping[str, int],
    *,
    inventory: LegalWidthInventory,
    conditional_costs: Mapping[str, float],
    precision_genes: Mapping[str, str],
    monitor: ExceptionOnlyRepairMonitor,
    reason: str,
) -> LegacyWidthRepairResult:
    """Project legacy pruning to a legal conservative subset.

    Conditional costs are allowed here because this function is explicitly not
    the formal width decoder.
    """

    unknown = set(raw_mask) - set(inventory.unit_ids)
    if unknown:
        raise ValueError(f"legacy_mask_unknown_units:{sorted(unknown)}")
    repaired = {unit_id: 1 for unit_id in inventory.unit_ids}
    width_genes: dict[str, int] = {}
    keep_widths: dict[str, int] = {}
    for domain in inventory.domains:
        if domain.domain_kind == "dense":
            raw_keep = sum(int(raw_mask.get(unit_id, 1)) != 0 for unit_id in domain.unit_ids)
            legal = [width for width in domain.legal_keep_widths if width >= raw_keep]
            target_keep = min(legal) if legal else domain.original_width
            selected = _ordered_requested_pruned(
                domain.unit_ids, raw_mask, conditional_costs
            )[: domain.original_width - target_keep]
            for unit_id in selected:
                repaired[unit_id] = 0
        else:
            raw_keep_by_group = {
                group: sum(int(raw_mask.get(unit_id, 1)) != 0 for unit_id in unit_ids)
                for group, unit_ids in domain.physical_groups.items()
            }
            required_keep = max(raw_keep_by_group.values(), default=domain.per_group_original_width)
            legal = [width for width in domain.legal_keep_widths if width >= required_keep]
            target_keep = min(legal) if legal else domain.per_group_original_width
            for unit_ids in domain.physical_groups.values():
                selected = _ordered_requested_pruned(
                    unit_ids, raw_mask, conditional_costs
                )[: domain.per_group_original_width - target_keep]
                for unit_id in selected:
                    repaired[unit_id] = 0
        keep_widths[domain.domain_id] = int(target_keep)
        width_genes[domain.domain_id] = domain.legal_keep_widths.index(target_keep)
    normalized_raw = {
        unit_id: 1 if int(raw_mask.get(unit_id, 1)) else 0
        for unit_id in inventory.unit_ids
    }
    monitor.observe_repair(
        reason=reason,
        structure_changed=repaired != normalized_raw,
        precision_changed=False,
    )
    return LegacyWidthRepairResult(
        width_genes=width_genes,
        keep_widths=keep_widths,
        group_mask=repaired,
        precision_genes={str(key): str(value) for key, value in precision_genes.items()},
    )
