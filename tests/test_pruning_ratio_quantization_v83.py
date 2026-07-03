from __future__ import annotations

from tools.latency_lut.audit_pruning_ratio_quantization_v83 import (
    feasible_grouped_keep_counts,
    quantize_domain_keep,
)


def test_grouped_conv_per_group_16_align8_feasible_keep_ratios_are_coarse():
    counts = feasible_grouped_keep_counts(channels_per_group=16, group_conv_align=8)

    assert counts == [16, 8]


def test_target_keep_097_aligns_up_to_full_keep_for_grouped_conv():
    row = quantize_domain_keep(
        candidate_id="c",
        domain_id="d",
        root_node="gconv",
        num_units=512,
        target_keep_ratio=0.97,
        align=8,
        is_grouped_conv_domain=True,
        groups=32,
        channels_per_group=16,
        group_conv_align=8,
        actual_keep_units=512,
    )

    assert row["aligned_keep_units"] == 512
    assert row["actual_prune_units"] == 0
    assert 0.5 in row["feasible_keep_ratios"]
    assert 1.0 in row["feasible_keep_ratios"]


def test_target_keep_097_must_not_align_down_to_half_keep():
    row = quantize_domain_keep(
        candidate_id="c",
        domain_id="d",
        root_node="gconv",
        num_units=512,
        target_keep_ratio=0.97,
        align=8,
        is_grouped_conv_domain=True,
        groups=32,
        channels_per_group=16,
        group_conv_align=8,
        actual_keep_units=256,
    )

    assert row["aligned_keep_units"] == 512
    assert row["actual_keep_units"] != row["aligned_keep_units"]
    assert row["ratio_error_reason"] == "group_conv_align"
