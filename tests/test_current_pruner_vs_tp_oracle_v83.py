from __future__ import annotations

from tools.latency_lut.audit_current_pruner_vs_tp_oracle_v83 import compare_ops_to_tp_oracle


def test_current_replay_extra_expansion_op_breaks_tp_equivalence():
    row = compare_ops_to_tp_oracle(
        candidate_id="c",
        domain_id="d",
        root_node="conv",
        root_module="conv",
        root_type="Conv2d",
        requested_keep_ratio=0.875,
        requested_prune_ratio=0.125,
        num_units=64,
        pruned_root_indices=[0, 1],
        tp_ops=[{"module": "conv", "op": "prune_conv_out_channels", "axis": "out", "idxs": [0, 1]}],
        current_ops=[
            {"module": "conv", "op": "prune_conv_out_channels", "axis": "out", "idxs": [0, 1]},
            {"module": "gconv", "op": "grouped_merge", "axis": "grouped_merge", "idxs": []},
        ],
        tp_group_build_success=True,
        tp_check_pruning_group_pass=True,
    )

    assert row["current_matches_tp_oracle"] is False
    assert row["extra_in_current_vs_tp"]


def test_current_replay_missing_tp_op_breaks_tp_equivalence():
    row = compare_ops_to_tp_oracle(
        candidate_id="c",
        domain_id="d",
        root_node="conv",
        root_module="conv",
        root_type="Conv2d",
        requested_keep_ratio=0.875,
        requested_prune_ratio=0.125,
        num_units=64,
        pruned_root_indices=[0, 1],
        tp_ops=[
            {"module": "conv", "op": "prune_conv_out_channels", "axis": "out", "idxs": [0, 1]},
            {"module": "next", "op": "prune_conv_in_channels", "axis": "in", "idxs": [0, 1]},
        ],
        current_ops=[{"module": "conv", "op": "prune_conv_out_channels", "axis": "out", "idxs": [0, 1]}],
        tp_group_build_success=True,
        tp_check_pruning_group_pass=True,
    )

    assert row["current_matches_tp_oracle"] is False
    assert row["missing_in_current_vs_tp"]
