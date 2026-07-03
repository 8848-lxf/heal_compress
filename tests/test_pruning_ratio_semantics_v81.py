from __future__ import annotations

from tools.latency_lut.audit_pruning_ratio_semantics_v81 import audit_ratio_record, monotonic_keep_ratio_trend


def test_target_keep_ratio_097_maps_to_prune_ratio_003():
    row = audit_ratio_record(
        candidate_id="c",
        target_keep_ratio=0.97,
        export_cli_prune_ratio=0.03,
        general_pruner_args_prune_ratio=0.03,
        domain_requested_keep_ratio=0.97,
        domain_requested_prune_ratio=0.03,
        actual_param_keep_ratio=0.99,
    )

    assert row["candidate_target_prune_ratio_expected"] == 0.03
    assert row["keep_ratio_correctly_propagated"] is True
    assert row["prune_ratio_inversion_detected"] is False


def test_target_keep_ratio_0875_maps_to_prune_ratio_0125():
    row = audit_ratio_record(
        candidate_id="c",
        target_keep_ratio=0.875,
        export_cli_prune_ratio=0.125,
        general_pruner_args_prune_ratio=0.125,
        domain_requested_keep_ratio=0.875,
        domain_requested_prune_ratio=0.125,
        actual_param_keep_ratio=0.9,
    )

    assert row["candidate_target_prune_ratio_expected"] == 0.125
    assert row["keep_ratio_correctly_propagated"] is True


def test_audit_detects_keep_ratio_used_as_prune_ratio():
    row = audit_ratio_record(
        candidate_id="bad",
        target_keep_ratio=0.97,
        export_cli_prune_ratio=0.97,
        general_pruner_args_prune_ratio=0.97,
        domain_requested_keep_ratio=0.03,
        domain_requested_prune_ratio=0.97,
        actual_param_keep_ratio=0.729,
    )

    assert row["prune_ratio_inversion_detected"] is True
    assert row["keep_ratio_correctly_propagated"] is False


def test_monotonic_keep_ratio_trend_detects_error():
    assert monotonic_keep_ratio_trend({0.97: [0.8], 0.875: [0.9], 0.75: [0.7]}) is False
    assert monotonic_keep_ratio_trend({0.97: [0.95], 0.875: [0.9], 0.75: [0.8]}) is True
