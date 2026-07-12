from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_independent_group_topk_keeps_different_positions_per_group() -> None:
    from pruning.types import AtomicPruneUnit
    from search.pruning_space.action_catalog import build_pruning_action_catalog

    units = []
    scores = [100, 90, 1, 0, 0, 0, 0, 0, 0, 0, 100, 90, 1, 0, 0, 0]
    for index, score in enumerate(scores):
        units.append(
            AtomicPruneUnit(
                scope_id="g",
                root_module_path="gconv",
                root_axis="out",
                root_indices=[index],
                source_coupled_unit_ids=[f"cu{index}"],
                normalized_score=float(score),
                constraints={
                    "grouped_conv": True,
                    "depthwise": False,
                    "groups": 2,
                    "channels_per_group": 8,
                },
            )
        )

    catalog = build_pruning_action_catalog(
        units,
        grouped_conv_mode="independent_group_topk",
        grouped_conv_align=4,
        grouped_allowed_channels_per_group=(4, 8, 16, 32, 64, 128, 256, 512),
    )
    action = next(action for action in catalog.actions if action.kind == "grouped_bundle")

    assert catalog.independent_group_topk_bundle_count == 1
    assert action.group_keep_map[0] != action.group_keep_map[1]
    assert all(len(values) == 4 for values in action.group_keep_map.values())
    assert action.constraints["channels_per_group_after"] == 4
