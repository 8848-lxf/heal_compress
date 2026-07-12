"""Mask-preserving pruning repair for Stage-1 raw keep masks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


SAFE_GROUPED_CHANNELS_PER_GROUP = (4, 8, 16, 32, 64, 128, 256, 512)


@dataclass(frozen=True)
class RepairPolicy:
    minimum_width: int = 1
    dense_alignment: int = 4
    grouped_allowed_channels_per_group: tuple[int, ...] = SAFE_GROUPED_CHANNELS_PER_GROUP


@dataclass(frozen=True)
class RepairResult:
    status: str
    repaired_mask: dict[str, int]
    target_width: int
    removed_unit_ids: tuple[str, ...] = ()
    failure_reason: str = ""
    group_keep_map: dict[int, list[int]] = field(default_factory=dict)
    group_prune_map: dict[int, list[int]] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class GroupedDomainSpec:
    groups: Mapping[int, tuple[str, ...]]
    ordered_low_to_high: Mapping[int, tuple[str, ...]]
    local_indices: Mapping[int, Mapping[str, int]]
    allowed_channels_per_group: tuple[int, ...] = SAFE_GROUPED_CHANNELS_PER_GROUP


def _normalize_mask(mask: Mapping[str, int]) -> dict[str, int]:
    return {str(key): 1 if int(value) else 0 for key, value in mask.items()}


def dense_floor_repair(
    raw_mask: Mapping[str, int],
    *,
    ordered_low_to_high: tuple[str, ...],
    alignment: int = 4,
    minimum_width: int = 1,
) -> RepairResult:
    """Floor a dense domain to an aligned width using only 1 -> 0 edits."""

    repaired = _normalize_mask(raw_mask)
    keep_ids = [unit_id for unit_id, keep in repaired.items() if keep == 1]
    raw_width = len(keep_ids)
    align = max(1, int(alignment))
    target = align * (raw_width // align)
    if raw_width > 0 and target == 0:
        target = 0
    if target < int(minimum_width):
        return RepairResult(
            status="failed",
            repaired_mask=repaired,
            target_width=target,
            failure_reason="dense_target_below_minimum_width",
        )
    remove_needed = raw_width - target
    removed: list[str] = []
    if remove_needed > 0:
        for unit_id in ordered_low_to_high:
            if remove_needed <= 0:
                break
            if repaired.get(unit_id, 0) == 1:
                repaired[unit_id] = 0
                removed.append(unit_id)
                remove_needed -= 1
    return RepairResult(
        status="ok",
        repaired_mask=repaired,
        target_width=target,
        removed_unit_ids=tuple(removed),
        metadata={"repair_mode": "dense_monotonic_floor", "raw_width": raw_width},
    )


def grouped_equal_count_floor_repair(
    raw_mask: Mapping[str, int],
    domain: GroupedDomainSpec,
    policy: RepairPolicy | None = None,
) -> RepairResult:
    """Repair grouped conv keep masks without recomputing a top-k selection."""

    cfg = policy or RepairPolicy()
    allowed = tuple(sorted({int(value) for value in domain.allowed_channels_per_group or cfg.grouped_allowed_channels_per_group}))
    repaired = _normalize_mask(raw_mask)
    raw_counts = {
        int(group): sum(1 for unit_id in unit_ids if repaired.get(unit_id, 0) == 1)
        for group, unit_ids in domain.groups.items()
    }
    if not raw_counts:
        return RepairResult("failed", repaired, 0, failure_reason="grouped_domain_empty")
    min_keep = min(raw_counts.values())
    legal = [width for width in allowed if width <= min_keep]
    if not legal:
        return RepairResult(
            "failed",
            repaired,
            0,
            failure_reason=f"grouped_no_safe_width_le_min_keep:{min_keep}",
            metadata={"raw_counts": raw_counts},
        )
    target = max(legal)
    removed: list[str] = []
    keep_map: dict[int, list[int]] = {}
    prune_map: dict[int, list[int]] = {}
    for group, unit_ids in sorted(domain.groups.items()):
        kept = [unit_id for unit_id in unit_ids if repaired.get(unit_id, 0) == 1]
        remove_needed = len(kept) - target
        for unit_id in domain.ordered_low_to_high.get(group, ()):
            if remove_needed <= 0:
                break
            if repaired.get(unit_id, 0) == 1:
                repaired[unit_id] = 0
                removed.append(unit_id)
                remove_needed -= 1
        keep_map[int(group)] = [
            int(domain.local_indices[group][unit_id])
            for unit_id in unit_ids
            if repaired.get(unit_id, 0) == 1
        ]
        prune_map[int(group)] = [
            int(domain.local_indices[group][unit_id])
            for unit_id in unit_ids
            if repaired.get(unit_id, 0) == 0
        ]
    if any(len(values) != target for values in keep_map.values()):
        return RepairResult(
            "failed",
            repaired,
            target,
            removed_unit_ids=tuple(removed),
            group_keep_map=keep_map,
            group_prune_map=prune_map,
            failure_reason="grouped_equal_count_repair_incomplete",
            metadata={"raw_counts": raw_counts},
        )
    return RepairResult(
        "ok",
        repaired,
        target,
        removed_unit_ids=tuple(removed),
        group_keep_map=keep_map,
        group_prune_map=prune_map,
        metadata={"repair_mode": "grouped_mask_preserving_equal_count_floor", "raw_counts": raw_counts},
    )
