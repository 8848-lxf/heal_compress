"""Constraint filtering for global pruning candidates."""

from __future__ import annotations

from collections.abc import Sequence

from ..types import AtomicPruneUnit


def filter_selectable_units(units: Sequence[AtomicPruneUnit]) -> list[AtomicPruneUnit]:
    return [unit for unit in units if not unit.protected and unit.root_indices]


__all__ = ["filter_selectable_units"]
