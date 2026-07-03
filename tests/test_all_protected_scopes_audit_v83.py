from __future__ import annotations

from tools.latency_lut.audit_all_protected_scopes_v83 import audit_scope_rows, summarize_protected_scopes


def test_dependency_scope_protected_reasons_are_counted():
    rows = audit_scope_rows(
        candidate_id="c",
        scopes=[
            {"scope_id": "head", "root_module": "cls_head", "protected": True, "protected_reason": "det_head_output", "num_channels": 10},
            {"scope_id": "res", "root_module": "backbone.add", "protected": True, "protected_reason": "residual_add_output_protected", "num_channels": 64},
            {"scope_id": "enc", "root_module": "encoder_m1.x", "protected": True, "protected_reason": "explicit", "num_channels": 64},
        ],
        selection_domains=[],
        protected_prefixes=["encoder_m1", "cls_head"],
    )
    summary = summarize_protected_scopes(rows)

    assert summary["protected_reasons"]["det_head_output"] == 1
    assert summary["protected_reasons"]["residual_add_output_protected"] == 1
    assert summary["num_non_head_protected_scopes"] == 2
    assert any(row["matched_protected_prefix"] == "encoder_m1" for row in rows)
