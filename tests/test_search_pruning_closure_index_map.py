from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _double_half_unit(*, mapped: list[int] | None = None):
    from pruning.types import AtomicPruneUnit

    return AtomicPruneUnit(
        scope_id="disco_feature_width",
        root_module_path="shrinker_m1.layers.0.double_conv.2",
        root_axis="out",
        root_indices=[3],
        source_coupled_unit_ids=["cu_3"],
        normalized_score=0.1,
        constraints={"original_channel_count": 256},
        metadata={
            "closure_members": [
                {
                    "module_path": "shrinker_m1.layers.0.double_conv.2",
                    "axis": "out",
                    "indices": [3],
                    "dependency_type": "root_output",
                    "index_map": {3: [3]},
                },
                {
                    "module_path": "fusion_net.pixel_weight_layer.conv1_1",
                    "axis": "in",
                    "indices": mapped or [3, 259],
                    "dependency_type": "disconet_neighbor_ego_concat_double_half",
                    "index_map": {3: mapped or [3, 259]},
                },
            ]
        },
    )


def test_action_catalog_preserves_non_identity_closure_index_map() -> None:
    from search.pruning_space.action_catalog import build_pruning_action_catalog

    catalog = build_pruning_action_catalog([_double_half_unit()])
    action = catalog.actions[0]
    closure = next(
        row
        for row in action.closure_entries
        if row["module_path"] == "fusion_net.pixel_weight_layer.conv1_1"
    )

    assert closure["indices"] == [3, 259]
    assert closure["closure_index_map"] == {3: [3, 259]}


def test_action_request_preserves_non_identity_closure_index_map() -> None:
    from search.pruning_space.action_catalog import build_pruning_action_catalog
    from search.pruning_space.grouped_bundle_adapter import request_from_pruning_actions

    action = build_pruning_action_catalog([_double_half_unit()]).actions[0]
    request = request_from_pruning_actions([action])
    entry = next(
        row
        for row in request.entries
        if row.module_path == "fusion_net.pixel_weight_layer.conv1_1"
    )

    assert entry.prune_indices == [3, 259]
    assert entry.metadata["closure_index_map"] == {3: [3, 259]}


def test_global_selector_preserves_non_identity_closure_index_map() -> None:
    from pruning.selection.global_ranking import select_global_units

    request = select_global_units([_double_half_unit()], channel_budget=1)
    entry = next(
        row
        for row in request.entries
        if row.module_path == "fusion_net.pixel_weight_layer.conv1_1"
    )

    assert entry.prune_indices == [3, 259]
    assert entry.metadata["closure_index_map"] == {3: [3, 259]}


def test_physical_planner_unions_compatible_partial_closure_maps() -> None:
    import torch.nn as nn

    from pruning.materialization.planner import build_physical_pruning_plan
    from pruning.types import SamplingPruningEntry, SamplingPruningRequest

    model = nn.Sequential(nn.Conv2d(8, 4, 1))
    request = SamplingPruningRequest(entries=[
        SamplingPruningEntry(
            request_id="a",
            scope_id="scope",
            module_path="0",
            axis="in",
            prune_indices=[1, 5],
            metadata={"closure_index_map": {1: [1, 5]}},
        ),
        SamplingPruningEntry(
            request_id="b",
            scope_id="scope",
            module_path="0",
            axis="in",
            prune_indices=[2, 6],
            metadata={"closure_index_map": {2: [2, 6]}},
        ),
    ])

    plan = build_physical_pruning_plan(model, request)

    assert len(plan.entries) == 1
    assert plan.entries[0].prune_indices == [1, 2, 5, 6]
    assert plan.entries[0].metadata["closure_index_map"] == {
        1: [1, 5],
        2: [2, 6],
    }


def test_grouped_global_bundle_preserves_non_identity_closure_index_map() -> None:
    from pruning.config import GroupedConvConfig
    from pruning.selection.global_ranking import _bundle_grouped_units
    from pruning.types import AtomicPruneUnit

    units = []
    for channel in range(16):
        units.append(AtomicPruneUnit(
            scope_id="grouped_disco_scope",
            root_module_path="grouped",
            root_axis="out",
            root_indices=[channel],
            source_coupled_unit_ids=[f"cu_{channel}"],
            normalized_score=float(channel),
            constraints={
                "grouped_conv": True,
                "grouped_module_path": "grouped",
                "groups": 2,
                "channels_per_group": 8,
            },
            metadata={
                "closure_members": [{
                    "module_path": "dependent",
                    "axis": "in",
                    "indices": [channel, 16 + channel],
                    "dependency_type": "double_half",
                    "index_map": {channel: [channel, 16 + channel]},
                }]
            },
        ))

    bundles = _bundle_grouped_units(
        units,
        GroupedConvConfig(allowed_channels_per_group=(4, 8)),
    )
    bundle = next(row for row in bundles if row.constraints.get("channels_per_group_after") == 4)
    closure = next(
        row for row in bundle.metadata["closure_entries"]
        if row["module_path"] == "dependent"
    )

    assert closure["closure_index_map"]
    for root_index in bundle.root_indices:
        assert closure["closure_index_map"][root_index] == [root_index, 16 + root_index]


def test_global_selector_rejects_unscored_trace_atoms_explicitly() -> None:
    from pruning.exceptions import PruningLegalityError
    from pruning.selection.global_ranking import select_global_units
    from tracer.types import AtomicPruneUnit, DependencyMember

    unit = AtomicPruneUnit(
        scope_id="trace_scope",
        root_module_path="conv",
        root_axis="out",
        root_indices=[0],
        source_coupled_unit_ids=["cu_0"],
        members=[DependencyMember("conv", "out", [0], "root")],
    )

    with pytest.raises(PruningLegalityError, match="require_importance_scoring"):
        select_global_units([unit], channel_budget=1)


def test_action_catalog_rejects_conflicting_closure_index_maps() -> None:
    from search.pruning_space.action_catalog import _closure_entries

    first = _double_half_unit()
    second = _double_half_unit(mapped=[3, 260])

    with pytest.raises(RuntimeError, match="conflicting_action_closure_index_map"):
        _closure_entries([first, second])
