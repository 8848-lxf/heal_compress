from __future__ import annotations

from tools.latency_lut.audit_prune_normalization_param_accounting_v83 import audit_param_accounting_record


def test_v82_gate_reuses_param_accounting_and_rejects_zero_pruned_units():
    row = audit_param_accounting_record(
        candidate_id="c",
        target_keep_ratio=0.5,
        param_keep_ratio=1.0,
        num_pruned_units=0,
        changed_layers=[],
        pre_prune_ops=[],
    )

    assert row["valid_pruned_candidate"] is False
    assert "num_pruned_units_zero" in row["invalid_reasons"]
