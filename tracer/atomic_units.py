"""Construction of legality-aware atomic prune candidates."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from .types import (
    AtomicPruneUnit,
    ConcreteCoupledPruningGroup,
    CoupledChannelUnit,
    DependencyScope,
)


def build_atomic_prune_units_from_coupled(
    units: Sequence[CoupledChannelUnit],
) -> list[AtomicPruneUnit]:
    """Build policy-neutral atoms while retaining grouped-conv constraints.

    Grouped candidates remain individual logical channels at trace time. The
    selector must bundle them into equal per-group counts using the recorded
    ``independent_group_topk`` metadata; no shared local-position assumption is
    introduced here.
    """

    atoms: list[AtomicPruneUnit] = []
    scope_widths: dict[str, int] = defaultdict(int)
    for unit in units:
        scope_widths[unit.scope_id] = max(scope_widths[unit.scope_id], int(unit.root_channel_index) + 1)
    for unit in sorted(units, key=lambda item: item.stable_id):
        grouped = dict(unit.grouped_conv_metadata)
        constraints: dict[str, object] = {
            "original_channel_count": scope_widths[unit.scope_id],
        }
        if grouped:
            constraints.update({
                "grouped_conv": True,
                "grouped_module_path": grouped.get("module_path"),
                "groups": grouped.get("groups"),
                "channels_per_group": grouped.get("channels_per_group"),
                "selection_policy": "independent_group_topk",
                "requires_equal_prune_count_per_group": True,
                "requires_saved_group_keep_map_for_replay": True,
                "depthwise": bool(grouped.get("depthwise", False)),
            })
        atoms.append(
            AtomicPruneUnit(
                scope_id=unit.scope_id,
                root_module_path=unit.root_module_path,
                root_axis=unit.root_axis,
                root_indices=[unit.root_channel_index],
                source_coupled_unit_ids=[unit.stable_id],
                members=list(unit.members),
                constraints=constraints,
                protected=unit.protected,
                protection_reason=unit.protection_reason,
            )
        )
    return atoms


def instantiate_concrete_group(
    scope: DependencyScope,
    prune_indices: Sequence[int],
    atomic_units: Sequence[AtomicPruneUnit] = (),
) -> ConcreteCoupledPruningGroup:
    """Bind a scope to validated original-index-space prune indices."""

    selected = sorted({int(value) for value in prune_indices})
    if any(value < 0 or value >= int(scope.channel_count) for value in selected):
        raise ValueError(f"prune index outside scope {scope.stable_id}: {selected}")
    source_ids = [
        unit.stable_id
        for unit in atomic_units
        if unit.scope_id == scope.stable_id and set(unit.root_indices).intersection(selected)
    ]
    return ConcreteCoupledPruningGroup(
        scope=scope,
        prune_indices=selected,
        source_atomic_unit_ids=source_ids,
    )
