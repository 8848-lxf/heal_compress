"""Expansion of dependency scopes into minimal coupled channel units."""

from __future__ import annotations

from collections.abc import Sequence

from .types import (
    CoupledChannelUnit,
    DependencyMember,
    DependencyScope,
    ModuleInventoryEntry,
)


def _grouped_metadata(
    scope: DependencyScope,
    modules: dict[str, ModuleInventoryEntry],
) -> dict[str, object]:
    for root in scope.root_modules:
        module = modules.get(root)
        if module is None or module.groups is None or int(module.groups) <= 1:
            continue
        groups = int(module.groups)
        out_channels = int(module.out_channels or scope.channel_count)
        return {
            "module_path": root,
            "groups": groups,
            "logical_total_channels": out_channels,
            "channels_per_group": out_channels // groups if out_channels % groups == 0 else None,
            "depthwise": bool(
                module.in_channels is not None
                and int(module.in_channels) == groups == out_channels
            ),
            "selection_policy": "independent_group_topk",
        }
    return {}


def build_coupled_channel_units_from_scopes(
    scopes: Sequence[DependencyScope],
    module_inventory: Sequence[ModuleInventoryEntry] = (),
) -> list[CoupledChannelUnit]:
    """Create one stable unit for each root channel in every dependency scope."""

    modules = {module.module_path: module for module in module_inventory}
    units: list[CoupledChannelUnit] = []
    for scope in sorted(scopes, key=lambda item: item.stable_id):
        grouped = _grouped_metadata(scope, modules)
        for root_index in range(int(scope.channel_count)):
            members: list[DependencyMember] = []
            for member in scope.members:
                local = member.index_map.get(root_index)
                if local is None:
                    local = [
                        member.channel_offset + root_index
                        if member.channel_offset or root_index in member.indices
                        else root_index
                    ]
                local = [int(value) for value in local if int(value) in set(member.indices)]
                if not local:
                    continue
                members.append(
                    DependencyMember(
                        module_path=member.module_path,
                        module_type=member.module_type,
                        axis=member.axis,
                        indices=local,
                        dependency_type=member.dependency_type,
                        channel_offset=member.channel_offset,
                        index_map={root_index: local},
                        protection_reason=member.protection_reason,
                    )
                )
            units.append(
                CoupledChannelUnit(
                    scope_id=scope.stable_id,
                    root_module_path=str(grouped.get("module_path") or scope.root_module_path),
                    root_axis=scope.root_axis,
                    root_channel_index=root_index,
                    members=members,
                    dependency_types=scope.dependency_types,
                    grouped_conv_metadata=dict(grouped),
                    protected=scope.protected,
                    protection_reason=scope.protection_reason,
                )
            )
    return units
