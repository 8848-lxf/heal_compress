from __future__ import annotations

from tools.latency_lut.audit_root_node_local_pruner_v8 import audit_candidate_artifact


def test_audit_rejects_units_without_channel_indices(tmp_path):
    cdir = tmp_path / "cand"
    cdir.mkdir()
    (cdir / "coupled_channel_units.json").write_text(
        '[{"unit_id":"u0","root_node":"root","members":[{"module":"m","axis":"out_channels"}]}]',
        encoding="utf-8",
    )
    (cdir / "root_node_local_domains.json").write_text(
        '{"domains":[{"domain_id":"root","root_node":"root","unit_ids":["u0"],"ranking_scope":"root_node_local"}]}',
        encoding="utf-8",
    )

    row = audit_candidate_artifact(cdir)

    assert row["semantic_pass"] is False
    assert "coupled_channel_units_missing_indices" in row["violations"]


def test_audit_rejects_module_stage_scope_as_root_node_local(tmp_path):
    cdir = tmp_path / "cand"
    cdir.mkdir()
    (cdir / "coupled_channel_units.json").write_text(
        '[{"unit_id":"u0","root_node":"root","root_channel_index":0,"members":[{"module":"m","axis":"out_channels","index":0}]}]',
        encoding="utf-8",
    )
    (cdir / "root_node_local_domains.json").write_text(
        '{"domains":[{"domain_id":"stage0","root_node":"root","unit_ids":["u0"],"ranking_scope":"module_stage"}]}',
        encoding="utf-8",
    )
    (cdir / "domain_selection_summary.json").write_text(
        '{"selection_mode":"module_stage","global_ranking":false,"module_stage_based_domain":true,"domains":[]}',
        encoding="utf-8",
    )

    row = audit_candidate_artifact(cdir)

    assert row["semantic_pass"] is False
    assert "not_using_root_node_local_ranking" in row["violations"]
