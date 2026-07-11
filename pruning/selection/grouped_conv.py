"""Grouped-convolution selection with exact replay metadata."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ..config import GroupedConvConfig, GroupedConvSelectionPolicy
from ..exceptions import GroupedConvLegalityError
from ..types import GroupedConvSelectionDecision


def _top_indices(values: Sequence[float], count: int) -> list[int]:
    return sorted(
        index for index, _ in sorted(
            enumerate(float(value) for value in values),
            key=lambda row: (-row[1], row[0]),
        )[:count]
    )


def _normalize(values: Sequence[float], epsilon: float = 1e-12) -> list[float]:
    rows = [float(value) for value in values]
    mean = sum(rows) / len(rows) if rows else 0.0
    return [value / (mean + epsilon) if abs(mean) > epsilon else 0.0 for value in rows]


def select_grouped_conv_channels(
    per_group_scores: Mapping[int, Sequence[float]],
    *,
    final_channels_per_group: int,
    config: GroupedConvConfig | None = None,
) -> GroupedConvSelectionDecision:
    """Select equal counts in every group and retain exact local positions.

    The default ``independent_group_topk`` ranks each group independently.
    ``shared_local_mean`` is available only when explicitly configured.
    """

    cfg = config or GroupedConvConfig()
    scores = {int(group): [float(value) for value in values] for group, values in per_group_scores.items()}
    group_ids = sorted(scores)
    if group_ids != list(range(len(group_ids))) or not group_ids:
        raise GroupedConvLegalityError("per_group_scores must contain contiguous group ids starting at zero")
    widths = {len(scores[group]) for group in group_ids}
    if len(widths) != 1:
        raise GroupedConvLegalityError("all groups must have the same original channel count")
    channels_before = widths.pop()
    final = int(final_channels_per_group)
    if final <= 0 or final > channels_before:
        raise GroupedConvLegalityError("final_channels_per_group is outside the original width")
    if final not in cfg.allowed_channels_per_group:
        raise GroupedConvLegalityError(
            f"final channels per group {final} not in {cfg.allowed_channels_per_group}"
        )
    if cfg.selection_policy is GroupedConvSelectionPolicy.REMOVE_GROUPS:
        if not cfg.allow_remove_groups:
            raise GroupedConvLegalityError("remove_groups is not enabled")
        raise GroupedConvLegalityError("remove_groups requires explicit group-block selection, not channel top-k")

    if cfg.selection_policy is GroupedConvSelectionPolicy.SHARED_LOCAL_MEAN:
        shared = [sum(scores[group][index] for group in group_ids) / len(group_ids) for index in range(channels_before)]
        selected = _top_indices(shared, final)
        group_keep_map = {group: list(selected) for group in group_ids}
    else:
        group_keep_map = {group: _top_indices(scores[group], final) for group in group_ids}
    group_prune_map = {
        group: [index for index in range(channels_before) if index not in set(group_keep_map[group])]
        for group in group_ids
    }
    legal = all(len(values) == final for values in group_keep_map.values())
    return GroupedConvSelectionDecision(
        selection_policy=cfg.selection_policy.value,
        group_keep_map=group_keep_map,
        group_prune_map=group_prune_map,
        per_group_raw_scores=scores,
        per_group_normalized_scores={group: _normalize(scores[group]) for group in group_ids},
        selected_local_indices={group: list(values) for group, values in group_keep_map.items()},
        final_channels_per_group=final,
        alignment_repair={
            "channels_per_group_before": channels_before,
            "requested_channels_per_group": final,
            "final_channels_per_group": final,
            "repaired": False,
        },
        legality_report={
            "legal": legal,
            "groups": len(group_ids),
            "equal_channels_per_group": legal,
            "allowed_channels_per_group": list(cfg.allowed_channels_per_group),
        },
    )


__all__ = ["select_grouped_conv_channels"]
