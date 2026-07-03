from __future__ import annotations

from tools.latency_lut.audit_grouped_conv_selection_v81 import audit_grouped_conv_record
from opencood.tools.compression.root_node_local_pruner import select_grouped_conv_units
from heal_compress.pruning.selection import SelectionConfig, _grouped_keep_per_group


def test_independent_group_topk_sorts_each_group_independently_and_outputs_map():
    result = select_grouped_conv_units(
        scores_by_group=[
            [0.1, 0.9, 0.2, 0.8],
            [1.0, 0.2, 0.7, 0.3],
        ],
        keep_ratio=0.5,
        mode="independent_group_topk",
        align=1,
    )

    assert result["group_keep_map"] == {"0": [1, 3], "1": [0, 2]}
    assert result["per_group_keep_count_equal"] is True


def test_grouped_conv_replay_missing_group_keep_map_fails():
    row = audit_grouped_conv_record(
        candidate_id="c",
        module="gconv",
        report={
            "group_conv_selection_mode": "independent_group_topk",
            "groups_before": 2,
            "groups_after": 2,
            "per_group_before": 4,
            "per_group_after": 2,
            "group_keep_map": {},
            "per_group_kept_count": {"0": 2, "1": 2},
        },
        replay_ops=[],
    )

    assert row["legal"] is False
    assert "missing_group_keep_map" in row["violations"]


def test_grouped_conv_keep_per_group_aligns_up_to_avoid_light_prune_overcut():
    cfg = SelectionConfig(
        prune_ratio=0.03,
        align=8,
        group_conv_align=8,
        group_conv_selection_mode="independent_group_topk",
    )

    assert _grouped_keep_per_group(channels=512, groups=32, cfg=cfg) == 16
