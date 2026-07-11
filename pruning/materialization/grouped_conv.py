"""Fail-closed grouped-convolution materialization checks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..exceptions import GroupedConvLegalityError, MissingGroupKeepMapError


def validate_grouped_materialization(
    *,
    groups: int,
    channels_per_group: int,
    group_keep_map: Mapping[int, Sequence[int]] | None,
) -> dict[int, list[int]]:
    """Validate and canonicalize a saved group-local keep map.

    Replay never guesses shared local positions. A missing or incomplete map
    therefore raises :class:`MissingGroupKeepMapError`.
    """

    group_count = int(groups)
    width = int(channels_per_group)
    if group_keep_map is None or not group_keep_map:
        raise MissingGroupKeepMapError("group_keep_map is required for grouped convolution replay")
    canonical = {int(key): sorted({int(value) for value in values}) for key, values in group_keep_map.items()}
    expected = set(range(group_count))
    if set(canonical) != expected:
        raise MissingGroupKeepMapError(
            f"group_keep_map keys {sorted(canonical)} do not cover groups {sorted(expected)}"
        )
    counts = {len(values) for values in canonical.values()}
    if len(counts) != 1 or not counts or next(iter(counts)) <= 0:
        raise GroupedConvLegalityError("every group must keep the same positive channel count")
    for group_id, values in canonical.items():
        if values and (values[0] < 0 or values[-1] >= width):
            raise GroupedConvLegalityError(
                f"group {group_id} contains an out-of-range local channel for width {width}"
            )
    return canonical


def expand_group_keep_map(
    group_keep_map: Mapping[int, Sequence[int]],
    *,
    groups: int,
    channels_per_group: int,
) -> list[int]:
    canonical = validate_grouped_materialization(
        groups=groups,
        channels_per_group=channels_per_group,
        group_keep_map=group_keep_map,
    )
    return [
        group_id * int(channels_per_group) + local_index
        for group_id in range(int(groups))
        for local_index in canonical[group_id]
    ]


def validate_grouped_plan_entry(
    entry: Any,
    *,
    groups: int,
    channels_per_group: int,
) -> dict[int, list[int]]:
    """Require replay maps to exactly encode the frozen absolute indices."""

    keep_map = validate_grouped_materialization(
        groups=groups,
        channels_per_group=channels_per_group,
        group_keep_map=getattr(entry, "group_keep_map", None),
    )
    prune_raw = getattr(entry, "group_prune_map", None)
    if not prune_raw:
        raise MissingGroupKeepMapError("group_prune_map is required with group_keep_map")
    prune_map = {int(key): sorted({int(value) for value in values}) for key, values in prune_raw.items()}
    if set(prune_map) != set(range(int(groups))):
        raise MissingGroupKeepMapError("group_prune_map does not cover every group")
    for group_id in range(int(groups)):
        expected = [
            index
            for index in range(int(channels_per_group))
            if index not in set(keep_map[group_id])
        ]
        if prune_map[group_id] != expected:
            raise GroupedConvLegalityError(
                f"group_prune_map is not the complement of group_keep_map for group {group_id}"
            )
    expanded_keep = [
        group_id * int(channels_per_group) + local
        for group_id in range(int(groups))
        for local in keep_map[group_id]
    ]
    expanded_prune = [
        group_id * int(channels_per_group) + local
        for group_id in range(int(groups))
        for local in prune_map[group_id]
    ]
    if list(getattr(entry, "keep_indices", [])) != expanded_keep:
        raise GroupedConvLegalityError("expanded group_keep_map differs from frozen keep_indices")
    if list(getattr(entry, "prune_indices", [])) != expanded_prune:
        raise GroupedConvLegalityError("expanded group_prune_map differs from frozen prune_indices")
    return keep_map


__all__ = [
    "expand_group_keep_map",
    "validate_grouped_materialization",
    "validate_grouped_plan_entry",
]
