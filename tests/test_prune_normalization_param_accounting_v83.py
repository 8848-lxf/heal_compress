from __future__ import annotations

from tools.latency_lut.audit_prune_normalization_param_accounting_v83 import audit_param_accounting_record


def test_param_keep_ratio_gt_one_is_invalid():
    row = audit_param_accounting_record(
        candidate_id="c",
        target_keep_ratio=0.97,
        param_keep_ratio=1.01,
        num_pruned_units=0,
        changed_layers=[],
        pre_prune_ops=[],
    )

    assert row["valid_pruned_candidate"] is False
    assert "param_keep_ratio_gt_1" in row["invalid_reasons"]
    assert "num_pruned_units_zero" in row["invalid_reasons"]


def test_channel_expansion_is_detected_and_invalid():
    row = audit_param_accounting_record(
        candidate_id="c",
        target_keep_ratio=0.97,
        param_keep_ratio=1.0,
        num_pruned_units=1,
        changed_layers=[
            {
                "layer_name": "conv",
                "baseline": {"C_in": 64, "C_out": 64},
                "candidate": {"C_in": 64, "C_out": 128},
            }
        ],
        pre_prune_ops=[{"layer": "conv", "reason": "pre_prune_group_alignment_normalization"}],
    )

    assert row["num_expansion_layers"] == 1
    assert row["pre_prune_normalization_detected"] is True
    assert row["valid_pruned_candidate"] is False
    assert "channel_expansion_detected" in row["invalid_reasons"]
