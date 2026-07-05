from __future__ import annotations

from tools.latency_lut.audit_prune_policy_v84 import audit_policy_records


def test_policy_audit_flags_channel_expansion_and_pre_prune_normalization():
    result = audit_policy_records(
        grouped_conv_records=[
            {"module": "gconv", "has_channel_expansion": True, "violations": ["channel_expansion"]}
        ],
        protection_records=[],
        normalization_records=[
            {"pre_prune_normalization_detected": True, "candidate_has_channel_expansion": True}
        ],
        fixed_width_boundaries=[],
    )

    assert result["channel_expansion_detected"] is True
    assert result["pre_prune_normalization_detected"] is True
    assert result["all_ratio_deviation_explained"] is False

