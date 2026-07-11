"""Deterministic replay of saved pruning decisions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch.nn as nn

from ..types import PhysicalPruningApplicationLedger, PhysicalPruningPlan
from .grouped_conv import expand_group_keep_map


def replay_group_keep_map(
    group_keep_map: Mapping[int, Sequence[int]] | None,
    *,
    groups: int,
    channels_per_group: int,
) -> list[int]:
    """Expand an exact saved per-group map to logical absolute indices."""

    return expand_group_keep_map(
        group_keep_map or {},
        groups=groups,
        channels_per_group=channels_per_group,
    )


def replay_pruning(
    model: nn.Module,
    plan: PhysicalPruningPlan,
    *,
    in_place: bool = False,
) -> Any:
    """Replay a previously frozen physical plan on an equivalent model."""

    from .executor import materialize_pruning

    return materialize_pruning(model, plan, in_place=in_place)


__all__ = ["replay_group_keep_map", "replay_pruning"]
