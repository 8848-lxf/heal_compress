from __future__ import annotations

from tools.latency_lut.compare_grouped_conv_policy_ablation_v85 import build_policy_ablation_summary


def test_policy_ablation_summary_identifies_relaxed_matching_tp_pattern():
    summary = build_policy_ablation_summary(
        audits={
            "tp_native_50_l2": [
                {"has_unequal_original_group_keep_counts": True, "valid_for_relaxed_total_align8_policy": True}
            ],
            "current_strict_50_l2": [
                {"valid_for_strict_policy": True, "has_unequal_original_group_keep_counts": False}
            ],
            "current_relaxed_total_align8_50_l2": [
                {
                    "valid_for_relaxed_total_align8_policy": True,
                    "matches_tp_native_pattern": True,
                    "has_unequal_original_group_keep_counts": True,
                }
            ],
        },
        latency={
            "baseline": {"engine_build_success": True, "latency_p50_ms": 4.0},
            "current_relaxed_total_align8_50_l2": {"engine_build_success": True, "latency_p50_ms": 3.5},
        },
    )

    assert summary["relaxed_total_align8_matches_tp_native_pattern"] is True
    assert summary["latency_smoke"]["current_relaxed_total_align8_50_l2"]["speedup_vs_baseline"] > 1.0
