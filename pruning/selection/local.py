"""Compatibility local selector; not a formal default."""

from __future__ import annotations

from collections.abc import Sequence

from ..types import AtomicPruneUnit


def rank_within_scope(units: Sequence[AtomicPruneUnit]) -> dict[str, list[AtomicPruneUnit]]:
    result: dict[str, list[AtomicPruneUnit]] = {}
    for unit in units:
        result.setdefault(unit.scope_id, []).append(unit)
    for values in result.values():
        values.sort(key=lambda unit: (unit.normalized_score, unit.stable_id))
    return result


__all__ = ["rank_within_scope"]
