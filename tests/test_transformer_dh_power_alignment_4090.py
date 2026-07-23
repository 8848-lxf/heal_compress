from __future__ import annotations

import pytest

from search.model_families.transformer.dh_power_alignment_4090 import (
    alignment_traits,
    candidate_width_values,
    joint_candidates,
    latency_benefit_gate,
    neighbor_advantage,
    neighbor_controls,
    power_alignment_widths,
    search_candidate_gate,
    speedup_metrics,
)


def test_power_widths_for_d32_are_deduplicated_and_complete():
    assert candidate_width_values(power_alignment_widths(32, heads=8)) == (
        32,
        28,
        24,
        20,
        16,
        12,
        8,
        4,
    )


def test_power_widths_for_d64_include_ladder_and_controls():
    assert candidate_width_values(power_alignment_widths(64, heads=4)) == (
        64,
        60,
        56,
        52,
        48,
        44,
        40,
        36,
        32,
        28,
        24,
        20,
        16,
        12,
        8,
        4,
    )


def test_power_widths_for_d16_remain_bounded():
    assert candidate_width_values(power_alignment_widths(16, heads=16)) == (16, 12, 8, 4)
    assert max(candidate_width_values(power_alignment_widths(16, heads=16))) == 16


def test_four_is_power_of_two_but_not_eight_aligned():
    row = alignment_traits(4, heads=8, original_d_h=32)
    assert row["exact_power_of_two"] is True
    assert row["divisible_by_4"] is True
    assert row["divisible_by_8"] is False


def test_24_is_eight_aligned_but_not_power_of_two():
    row = alignment_traits(24, heads=8, original_d_h=32)
    assert row["exact_power_of_two"] is False
    assert row["divisible_by_8"] is True
    assert row["divisible_by_16"] is False


def test_48_is_sixteen_aligned_but_not_power_of_two():
    row = alignment_traits(48, heads=4, original_d_h=64)
    assert row["exact_power_of_two"] is False
    assert row["divisible_by_16"] is True
    assert row["divisible_by_32"] is False


def test_head_and_projection_alignment_are_separate():
    row = alignment_traits(12, heads=8, original_d_h=32)
    assert row["divisible_by_8"] is False
    assert row["projection_width"] == 96
    assert row["projection_divisible_by_32"] is True


def test_reduction_ratio_uses_original_width():
    assert alignment_traits(16, heads=8, original_d_h=32)["reduction_ratio"] == 0.5


def test_invalid_width_fails_closed():
    with pytest.raises(ValueError, match="invalid_power_alignment_width"):
        alignment_traits(36, heads=8, original_d_h=32)


@pytest.mark.parametrize(
    ("target", "original", "expected"),
    [(24, 32, (20, 28)), (16, 32, (12, 20)), (32, 32, (28,)), (4, 16, (8,))],
)
def test_neighbor_controls_match_legal_plus_minus_four(target, original, expected):
    assert neighbor_controls(target, original) == expected


def test_joint_candidates_are_explicit_not_cartesian():
    cobevt = joint_candidates("lidar_cobevt")
    v2xvit = joint_candidates("lidar_v2xvit")
    assert len(cobevt) == 9
    assert len(v2xvit) == 8
    assert cobevt[0].target_d_h_by_family == {
        "cobevt_grid_h8_d32": 32,
        "cobevt_window_h8_d32": 32,
    }
    assert v2xvit[-1].target_d_h_by_family == {
        "v2xvit_agent_relation_h8_d32": 4,
        "v2xvit_spatial_window_w16_h4_d64": 8,
        "v2xvit_spatial_window_w4_h16_d16": 4,
        "v2xvit_spatial_window_w8_h8_d32": 4,
    }


def test_speedup_metrics_keep_structure_precision_and_total_separate():
    row = speedup_metrics(
        baseline_p32_ms=10.0,
        baseline_profile_ms=5.0,
        candidate_profile_ms=4.0,
    )
    assert row == {
        "structure_speedup": 1.25,
        "precision_speedup": 2.0,
        "total_speedup": 2.5,
    }


def test_latency_gate_uses_largest_noise_threshold():
    row = latency_benefit_gate(
        baseline_p50_ms=10.0,
        candidate_p50_ms=9.7,
        baseline_repeat_cv=0.005,
        baseline_replay_drift=0.02,
    )
    assert row["required_reduction_ratio"] == pytest.approx(0.02)
    assert row["observed_reduction_ratio"] == pytest.approx(0.03)
    assert row["beneficial"] is True


def test_neighbor_advantage_requires_both_available_controls():
    assert neighbor_advantage(candidate_ms=4.0, lower_control_ms=4.1, upper_control_ms=4.2)[
        "advantage"
    ] is True
    assert neighbor_advantage(candidate_ms=4.0, lower_control_ms=3.9, upper_control_ms=4.2)[
        "advantage"
    ] is False


def test_search_candidate_gate_requires_every_evidence_gate():
    accepted = search_candidate_gate(
        fixed500_acceptable=True,
        same_profile_latency=True,
        neighbor_advantage_passed=True,
        build_repeat_stable=True,
        joint_supported=True,
    )
    assert accepted["search_space_candidate"] is True
    rejected = search_candidate_gate(
        fixed500_acceptable=True,
        same_profile_latency=True,
        neighbor_advantage_passed=False,
        build_repeat_stable=True,
        joint_supported=True,
    )
    assert rejected["search_space_candidate"] is False
    assert rejected["reasons"] == ["neighbor_control_advantage_missing"]

