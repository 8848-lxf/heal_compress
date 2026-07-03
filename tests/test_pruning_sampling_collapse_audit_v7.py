from __future__ import annotations

from tools.latency_lut.audit_pruning_sampling_collapse_v7 import summarize_collapse


def test_sampling_collapse_detects_same_structure_for_different_requested_keep():
    rows = [
        {"candidate_id": "a", "requested_target_keep_ratio": 0.97, "structure_changes_hash": "same", "actual_param_keep_ratio": 0.72},
        {"candidate_id": "b", "requested_target_keep_ratio": 0.875, "structure_changes_hash": "same", "actual_param_keep_ratio": 0.72},
    ]
    summary = summarize_collapse(rows)
    assert summary["all_requested_keep_ratios_collapsed_to_same_structure"] is True
    assert summary["all_successful_labels_from_same_actual_keep_ratio"] is True
