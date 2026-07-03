from __future__ import annotations

from tools.latency_lut.audit_root_node_local_pruning_semantics_v7 import evaluate_candidate_semantics


def test_root_node_semantics_rejects_module_stage_local_scope_without_root_metadata():
    result = evaluate_candidate_semantics(
        {
            "ranking_scope_declared": "local_scope",
            "ranking_scope_actual": "module_stage",
            "importance_actual": "first_order_taylor",
            "has_root_node_domain_metadata": False,
            "uses_global_ranking": False,
        }
    )
    assert result["semantic_pass"] is False
    assert "missing_root_node_domain_metadata" in result["violations"]
