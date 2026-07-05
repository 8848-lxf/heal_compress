from __future__ import annotations

from typing import Any, Iterable


def feasible_strict_group_keep_set(*, per_group_channels: int, align: int = 8) -> dict[str, Any]:
    if per_group_channels <= 0:
        return {"per_group_channels": per_group_channels, "align": align, "feasible_keep_per_group": [], "feasible_keep_ratios": []}
    values = [v for v in range(per_group_channels, 0, -1) if align <= 1 or v % align == 0]
    if per_group_channels not in values:
        values.insert(0, per_group_channels)
    return {
        "per_group_channels": int(per_group_channels),
        "align": int(align),
        "feasible_keep_per_group": values,
        "feasible_keep_ratios": [round(v / per_group_channels, 6) for v in values],
    }


def _counts(indices: Iterable[int], *, channels_before: int, groups: int) -> dict[str, int]:
    if groups <= 0 or channels_before <= 0 or channels_before % groups != 0:
        return {}
    per = channels_before // groups
    out = {str(g): 0 for g in range(groups)}
    for idx in sorted({int(i) for i in indices}):
        if 0 <= idx < channels_before:
            out[str(idx // per)] += 1
    return out


def audit_relaxed_group_total_align8(
    *,
    module: str,
    groups_before: int,
    groups_after: int,
    c_in_before: int,
    c_out_before: int,
    kept_out_indices: Iterable[int],
    kept_in_indices: Iterable[int],
    align: int = 8,
) -> dict[str, Any]:
    kept_out = sorted({int(i) for i in kept_out_indices})
    kept_in = sorted({int(i) for i in kept_in_indices})
    c_out_after = len(kept_out)
    c_in_after = len(kept_in)
    out_counts = _counts(kept_out, channels_before=c_out_before, groups=groups_before)
    in_counts = _counts(kept_in, channels_before=c_in_before, groups=groups_before)
    violations: list[str] = []
    if groups_before != groups_after:
        violations.append("groups_changed")
    if c_out_after > c_out_before or c_in_after > c_in_before:
        violations.append("channel_expansion")
    if align > 1 and c_out_after % align != 0:
        violations.append("total_C_out_not_align8")
    if align > 1 and c_in_after % align != 0:
        violations.append("total_C_in_not_align8")
    if groups_after <= 0 or c_out_after % groups_after != 0:
        violations.append("C_out_not_divisible_by_groups")
    if groups_after <= 0 or c_in_after % groups_after != 0:
        violations.append("C_in_not_divisible_by_groups")
    return {
        "module": module,
        "policy": "relaxed_group_total_align8",
        "groups_before": int(groups_before),
        "groups_after": int(groups_after),
        "C_out_before": int(c_out_before),
        "C_out_after": int(c_out_after),
        "C_in_before": int(c_in_before),
        "C_in_after": int(c_in_after),
        "original_group_keep_counts_out": out_counts,
        "original_group_prune_counts_out": {
            k: c_out_before // groups_before - v for k, v in out_counts.items()
        } if groups_before else {},
        "original_group_keep_count_equal_out": bool(out_counts) and len(set(out_counts.values())) == 1,
        "total_C_out_align8": align <= 1 or c_out_after % align == 0,
        "total_C_in_align8": align <= 1 or c_in_after % align == 0,
        "C_out_divisible_by_groups": groups_after > 0 and c_out_after % groups_after == 0,
        "C_in_divisible_by_groups": groups_after > 0 and c_in_after % groups_after == 0,
        "selected_keep_indices": kept_out,
        "selected_prune_indices": [i for i in range(c_out_before) if i not in set(kept_out)],
        "valid_relaxed_group_total_align8": not violations,
        "violations": violations,
    }
