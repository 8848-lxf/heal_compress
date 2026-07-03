from __future__ import annotations

from opencood.tools.compression.root_node_local_pruner import select_grouped_conv_units


def test_grouped_conv_shared_local_mean_preserves_groups_and_equal_keep_count():
    result = select_grouped_conv_units(
        scores_by_group=[
            [0.1, 0.9, 0.2, 0.8],
            [0.2, 1.0, 0.3, 0.7],
        ],
        keep_ratio=0.5,
        mode="shared_local_mean",
        align=1,
    )

    assert result["mode"] == "shared_local_mean"
    assert result["groups_preserved"] is True
    assert result["group_keep_map"] == {"0": [1, 3], "1": [1, 3]}
    assert result["per_group_keep_count_equal"] is True
    assert not result["violations"]


def test_grouped_conv_independent_group_topk_outputs_group_keep_map():
    result = select_grouped_conv_units(
        scores_by_group=[
            [0.1, 0.9, 0.2, 0.8],
            [1.0, 0.2, 0.7, 0.3],
        ],
        keep_ratio=0.5,
        mode="independent_group_topk",
        align=1,
    )

    assert result["mode"] == "independent_group_topk"
    assert result["group_keep_map"] == {"0": [1, 3], "1": [0, 2]}
    assert result["per_group_keep_count_equal"] is True
    assert not result["violations"]
