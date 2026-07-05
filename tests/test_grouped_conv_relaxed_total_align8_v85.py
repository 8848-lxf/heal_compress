from __future__ import annotations

from pruning.grouped_conv_policy import audit_relaxed_group_total_align8, feasible_strict_group_keep_set


def test_relaxed_policy_allows_original_group_imbalance_when_total_shape_is_legal():
    kept = (
        list(range(0, 2))
        + list(range(16, 20))
        + list(range(32, 42))
        + list(range(48, 64))
    )
    row = audit_relaxed_group_total_align8(
        module="gconv",
        groups_before=4,
        groups_after=4,
        c_in_before=64,
        c_out_before=64,
        kept_out_indices=kept,
        kept_in_indices=kept,
        align=8,
    )

    assert row["original_group_keep_count_equal_out"] is False
    assert row["total_C_out_align8"] is True
    assert row["C_out_divisible_by_groups"] is True
    assert row["valid_relaxed_group_total_align8"] is True


def test_relaxed_policy_rejects_total_misalignment_group_change_and_expansion():
    row = audit_relaxed_group_total_align8(
        module="gconv",
        groups_before=4,
        groups_after=2,
        c_in_before=64,
        c_out_before=64,
        kept_out_indices=list(range(10)),
        kept_in_indices=list(range(72)),
        align=8,
    )

    assert row["valid_relaxed_group_total_align8"] is False
    assert "groups_changed" in row["violations"]
    assert "channel_expansion" in row["violations"]
    assert "total_C_out_not_align8" in row["violations"]


def test_strict_feasible_set_for_per_group16_align8():
    feasible = feasible_strict_group_keep_set(per_group_channels=16, align=8)

    assert feasible["feasible_keep_per_group"] == [16, 8]
    assert feasible["feasible_keep_ratios"] == [1.0, 0.5]

