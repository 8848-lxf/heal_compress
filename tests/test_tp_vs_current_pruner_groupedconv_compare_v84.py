from __future__ import annotations

from tools.latency_lut.compare_tp_native_vs_current_pruner_v84 import summarize_comparison


def test_comparison_detects_non_equivalent_grouped_conv_policy():
    tp_rows = [
        {
            "pipeline": "tp_native",
            "valid_for_deployment_friendly_grouped_conv": False,
            "original_group_keep_count_equal_out": False,
            "per_group_keep_count_align8_out": False,
            "groups_preserved": True,
            "has_channel_expansion": False,
            "violations": ["original_group_keep_count_unequal_out"],
        }
    ]
    current_rows = [
        {
            "pipeline": "current_pruner",
            "valid_for_deployment_friendly_grouped_conv": True,
            "original_group_keep_count_equal_out": True,
            "per_group_keep_count_align8_out": True,
            "groups_preserved": True,
            "group_keep_map_present": True,
            "group_keep_map_matches_actual": True,
            "has_channel_expansion": False,
            "violations": [],
        }
    ]

    report = summarize_comparison(tp_rows, current_rows, tp_error={}, current_error={})

    assert report["direct_comparison"]["tp_native_allows_original_group_imbalance"] is True
    assert report["direct_comparison"]["current_pruner_enforces_original_group_balance"] is True
    assert report["direct_comparison"]["tp_native_equivalent_to_current_pruner_for_grouped_conv"] is False
    assert report["direct_comparison"]["current_pruner_safer_for_trt_grouped_conv"] is True

