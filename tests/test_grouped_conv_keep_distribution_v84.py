from __future__ import annotations

from tools.latency_lut.audit_grouped_conv_keep_distribution_v84 import audit_grouped_conv_distribution


def test_unequal_original_group_keep_counts_fail_even_if_final_shape_divisible():
    row = audit_grouped_conv_distribution(
        pipeline="tp_native",
        module="gconv",
        groups_before=4,
        groups_after=4,
        c_in_before=64,
        c_out_before=64,
        c_in_after=32,
        c_out_after=32,
        kept_out_indices=(
            list(range(0, 2))
            + list(range(16, 20))
            + list(range(32, 42))
            + list(range(48, 64))
        ),
        kept_in_indices=(
            list(range(0, 8))
            + list(range(16, 24))
            + list(range(32, 40))
            + list(range(48, 56))
        ),
        group_keep_map=None,
        require_group_keep_map=False,
    )

    assert row["final_shape_group_divisible"] is True
    assert row["original_group_keep_count_equal_out"] is False
    assert row["valid_for_deployment_friendly_grouped_conv"] is False
    assert "original_group_keep_count_unequal_out" in row["violations"]


def test_independent_group_topk_group_keep_map_passes_with_align8():
    group_keep_map = {str(g): list(range(8)) for g in range(4)}
    kept = [g * 16 + i for g in range(4) for i in range(8)]
    row = audit_grouped_conv_distribution(
        pipeline="current_pruner",
        module="gconv",
        groups_before=4,
        groups_after=4,
        c_in_before=64,
        c_out_before=64,
        c_in_after=32,
        c_out_after=32,
        kept_out_indices=kept,
        kept_in_indices=kept,
        group_keep_map=group_keep_map,
        require_group_keep_map=True,
    )

    assert row["original_group_keep_count_equal_out"] is True
    assert row["per_group_keep_count_align8_out"] is True
    assert row["group_keep_map_present"] is True
    assert row["group_keep_map_matches_actual"] is True
    assert row["valid_for_deployment_friendly_grouped_conv"] is True


def test_keep_count_10_is_not_align8():
    group_keep_map = {str(g): list(range(10)) for g in range(4)}
    kept = [g * 16 + i for g in range(4) for i in range(10)]
    row = audit_grouped_conv_distribution(
        pipeline="current_pruner",
        module="gconv",
        groups_before=4,
        groups_after=4,
        c_in_before=64,
        c_out_before=64,
        c_in_after=40,
        c_out_after=40,
        kept_out_indices=kept,
        kept_in_indices=kept,
        group_keep_map=group_keep_map,
        require_group_keep_map=True,
    )

    assert row["per_group_keep_count_align8_out"] is False
    assert "per_group_keep_count_not_align8_out" in row["violations"]
    assert row["valid_for_deployment_friendly_grouped_conv"] is False


def test_channel_expansion_and_groups_changed_are_invalid():
    row = audit_grouped_conv_distribution(
        pipeline="current_pruner",
        module="gconv",
        groups_before=4,
        groups_after=2,
        c_in_before=64,
        c_out_before=64,
        c_in_after=72,
        c_out_after=64,
        kept_out_indices=list(range(64)),
        kept_in_indices=list(range(72)),
        group_keep_map={str(g): list(range(16)) for g in range(4)},
        require_group_keep_map=True,
    )

    assert row["has_channel_expansion"] is True
    assert row["changed_groups_count"] is True
    assert "channel_expansion" in row["violations"]
    assert "groups_changed" in row["violations"]

