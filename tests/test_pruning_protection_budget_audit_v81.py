from __future__ import annotations

from tools.latency_lut.audit_pruning_protection_budget_v81 import summarize_protection_budget


def test_protection_budget_audit_lists_extra_protected_prefixes():
    summary = summarize_protection_budget(
        [
            {
                "candidate_id": "c",
                "protection_rules": {
                    "extra_protected_prefixes": ["encoder_m1", "cls_head"],
                    "align": 8,
                    "group_conv_align": 8,
                },
                "domain_budget": [],
                "global_budget_summary": {"param_keep_ratio_actual": 0.9},
            }
        ]
    )

    assert summary["extra_protected_prefixes_detected"] == ["cls_head", "encoder_m1"]


def test_protection_budget_explains_identical_keep_bins():
    summary = summarize_protection_budget(
        [
            {"target_keep_ratio": 0.97, "global_budget_summary": {"param_keep_ratio_actual": 0.729}},
            {"target_keep_ratio": 0.875, "global_budget_summary": {"param_keep_ratio_actual": 0.729}},
        ]
    )

    assert "same actual param keep" in summary["why_keep0875_equals_keep097"]
