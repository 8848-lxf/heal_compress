from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _grouped_units():
    from pruning.types import AtomicPruneUnit

    rows = []
    for index in range(16):
        group, local = divmod(index, 8)
        rows.append(
            AtomicPruneUnit(
                scope_id="scope_grouped",
                root_module_path="gconv",
                root_axis="out",
                root_indices=[index],
                source_coupled_unit_ids=[f"ccu_{index:02d}"],
                normalized_score=float(local),
                parameter_cost=1,
                channel_cost=1,
                constraints={
                    "grouped_conv": True,
                    "depthwise": False,
                    "grouped_module_path": "gconv",
                    "groups": 2,
                    "channels_per_group": 8,
                    "original_channel_count": 16,
                },
                metadata={
                    "closure_members": [
                        {"module_path": "gconv", "axis": "out", "indices": [index], "dependency_type": "root_output"}
                    ]
                },
            )
        )
    return rows


def test_grouped_conv_raw_units_are_bundled_into_legal_actions() -> None:
    from search.pruning_space.action_catalog import build_pruning_action_catalog

    catalog = build_pruning_action_catalog(_grouped_units(), grouped_conv_mode="shared_local_mean", grouped_conv_align=8)

    assert catalog.grouped_action_count > 0
    assert catalog.raw_grouped_unit_ids
    assert not any(action.action_id in catalog.raw_grouped_unit_ids for action in catalog.actions)
    action = next(action for action in catalog.actions if action.kind == "grouped_bundle")
    assert action.group_keep_map
    assert all(len(values) == 8 for values in action.group_keep_map.values())
    assert all(len(values) % 8 == 0 for values in action.group_keep_map.values())


def test_grouped_action_sampling_request_carries_group_maps() -> None:
    from search.pruning_space.action_catalog import build_pruning_action_catalog
    from search.pruning_space.grouped_bundle_adapter import request_from_pruning_actions

    catalog = build_pruning_action_catalog(_grouped_units(), grouped_conv_mode="shared_local_mean", grouped_conv_align=8)
    action = next(action for action in catalog.actions if action.kind == "grouped_bundle")

    request = request_from_pruning_actions([action])

    assert request.selected_atomic_unit_ids == [action.action_id]
    root_entries = [entry for entry in request.entries if entry.module_path == "gconv" and entry.axis == "out"]
    assert root_entries
    assert root_entries[0].group_keep_map == action.group_keep_map
    assert root_entries[0].group_prune_map == action.group_prune_map


def test_grouped_action_sampling_request_carries_group_maps_on_grouped_input_axis() -> None:
    from search.pruning_space.action_catalog import PruningSearchAction
    from search.pruning_space.grouped_bundle_adapter import request_from_pruning_actions

    action = PruningSearchAction(
        action_id="bundle_scope_shared",
        kind="grouped_bundle",
        source_atomic_unit_ids=("apu_0", "apu_1"),
        source_coupled_unit_ids=("cu_0",),
        root_module_path="block.gconv",
        root_axis="out",
        root_indices=(8, 9, 10, 11, 12, 13, 14, 15),
        scope_id="scope",
        group_keep_map={0: list(range(8)), 1: list(range(8))},
        group_prune_map={0: [], 1: []},
        closure_entries=(
            {
                "module_path": "block.gconv",
                "axis": "in",
                "indices": [8, 9, 10, 11, 12, 13, 14, 15],
                "dependency_types": ["grouped_conv_coupled_input"],
            },
            {
                "module_path": "block.gconv",
                "axis": "out",
                "indices": [8, 9, 10, 11, 12, 13, 14, 15],
                "dependency_types": ["grouped_conv:shared_local_mean"],
            },
            {
                "module_path": "block.bn",
                "axis": "channel",
                "indices": [8, 9, 10, 11, 12, 13, 14, 15],
                "dependency_types": ["conv_bn"],
            },
        ),
        constraints={
            "grouped_conv": True,
            "grouped_module_path": "block.gconv",
            "groups": 2,
            "channels_per_group_before": 8,
        },
    )

    request = request_from_pruning_actions([action])

    grouped_entries = [
        entry
        for entry in request.entries
        if entry.module_path == "block.gconv" and entry.axis in {"in", "out", "channel"}
    ]
    assert len(grouped_entries) == 2
    assert all(entry.group_keep_map == action.group_keep_map for entry in grouped_entries)
    assert all(entry.group_prune_map == action.group_prune_map for entry in grouped_entries)
    bn_entry = next(entry for entry in request.entries if entry.module_path == "block.bn")
    assert bn_entry.group_keep_map == {}


def test_grouped_scope_smaller_than_alignment_is_skipped_not_fatal() -> None:
    from pruning.types import AtomicPruneUnit
    from search.pruning_space.action_catalog import build_pruning_action_catalog

    units = [
        AtomicPruneUnit(
            scope_id="tiny_grouped",
            root_module_path="tiny",
            root_axis="out",
            root_indices=[index],
            source_coupled_unit_ids=[f"tiny_{index}"],
            normalized_score=float(index),
            constraints={
                "grouped_conv": True,
                "depthwise": False,
                "grouped_module_path": "tiny",
                "groups": 2,
                "channels_per_group": 4,
                "original_channel_count": 8,
            },
        )
        for index in range(8)
    ]

    catalog = build_pruning_action_catalog(units, grouped_conv_mode="shared_local_mean", grouped_conv_align=8)

    assert catalog.grouped_action_count == 0
    assert "tiny_grouped" in catalog.unbundleable_grouped_scopes


def test_grouped_action_slices_are_remapped_to_local_input_axis() -> None:
    import torch
    from search.proxy.parameter_slice_resolver import ParameterSlice
    from search.pruning_space.action_catalog import PruningSearchAction
    from search.pruning_space.action_codec import sanitize_action_parameter_slices

    action = PruningSearchAction(
        action_id="bundle",
        kind="grouped_bundle",
        source_atomic_unit_ids=("u",),
        source_coupled_unit_ids=("c",),
        root_module_path="gconv",
        root_axis="out",
        root_indices=(24,),
        scope_id="scope",
        constraints={"channels_per_group_before": 16},
    )
    rows = [ParameterSlice("gconv.weight", "gconv", 1, (24, 25), "prune_weight_slice")]
    params = {"gconv.weight": torch.nn.Parameter(torch.zeros(512, 16, 3, 3))}

    sanitized = sanitize_action_parameter_slices(action, rows, params)

    assert sanitized[0].indices == (8, 9)
