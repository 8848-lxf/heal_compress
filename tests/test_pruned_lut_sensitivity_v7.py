from __future__ import annotations

from tools.latency_lut.audit_pruned_lut_sensitivity_v7 import summarize_lut_sensitivity


def test_lut_sensitivity_distinguishes_constant_matched_keys_from_shape_changes():
    labels = [
        {"T_lut_raw": 1.0, "matched_lut_key_hash": "same", "candidate_conv_shape_hash": "shape_a"},
        {"T_lut_raw": 1.0, "matched_lut_key_hash": "same", "candidate_conv_shape_hash": "shape_b"},
    ]
    summary = summarize_lut_sensitivity(labels)
    assert summary["T_lut_raw_constant"] is True
    assert summary["matched_lut_keys_constant"] is True
    assert summary["candidate_shapes_constant"] is False
    assert summary["root_cause"] == "lut_matching_too_coarse_or_constant_components"
