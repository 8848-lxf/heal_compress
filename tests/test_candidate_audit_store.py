from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _plan_payload() -> dict:
    return {
        "schema_version": "physical-pruning-plan-v1",
        "indices_frozen_before_materialization": True,
        "conflicts": [],
        "entries": [
            {
                "module_path": "backbone_m1.blocks.1.0",
                "axis": 0,
                "original_axis_size": 128,
                "keep_indices": list(range(0, 96)),
                "prune_indices": list(range(96, 128)),
                "repaired": False,
                "repair_reason": "",
                "source_request_ids": ["request-1"],
            }
        ],
        "source_request": _request_payload(),
    }


def _request_payload() -> dict:
    return {
        "schema_version": "sampling-pruning-request-v1",
        "one_shot": True,
        "selector": "test",
        "requested_channel_cost": 32,
        "requested_parameter_cost": 4096,
        "selected_atomic_unit_ids": ["unit-3", "unit-4"],
        "entries": [
            {
                "request_id": "request-1",
                "module_path": "backbone_m1.blocks.1.0",
                "axis": 0,
                "prune_indices": list(range(96, 128)),
                "source_atomic_unit_ids": ["unit-3", "unit-4"],
            }
        ],
    }


def _write_aliases(candidate: Path) -> None:
    plan = _plan_payload()
    request = _request_payload()
    for name in (
        "physical_pruning_plan.json",
        "physical_plan.json",
        "legalized_plan.json",
    ):
        (candidate / name).write_text(json.dumps(plan), encoding="utf-8")
    for name in ("pruning_request.json", "sampling_pruning_request.json"):
        (candidate / name).write_text(json.dumps(request), encoding="utf-8")


def test_store_deduplicates_canonical_json_across_candidates(tmp_path: Path) -> None:
    from search.artifacts.candidate_audit_store import (
        CandidateAuditStore,
        resolve_artifact_reference,
    )

    store = CandidateAuditStore(tmp_path / "audit_store")
    first = store.put_json("physical_plan", {"b": 2, "a": [1, 3]})
    second = store.put_json("physical_plan", {"a": [1, 3], "b": 2})

    assert first.sha256 == second.sha256
    assert first.relative_path == second.relative_path
    blobs = list((tmp_path / "audit_store").rglob("*.json.gz"))
    assert len(blobs) == 1
    with gzip.open(blobs[0], "rt", encoding="utf-8") as handle:
        assert json.load(handle) == {"a": [1, 3], "b": 2}
    assert resolve_artifact_reference(store.root, first) == {
        "a": [1, 3],
        "b": 2,
    }


def test_completed_candidate_moves_redundant_plans_to_audit_store(
    tmp_path: Path,
) -> None:
    from search.artifacts.candidate_audit_store import (
        CandidateAuditStore,
        finalize_completed_candidate,
        resolve_artifact_reference,
    )

    generation = tmp_path / "budget_010" / "generation_001"
    candidate = generation / "stage2" / "candidate-a"
    candidate.mkdir(parents=True)
    _write_aliases(candidate)
    completion = generation / "stage2_artifact_retention.json"
    completion.write_text(json.dumps({"enabled": True}), encoding="utf-8")
    store = CandidateAuditStore(tmp_path / "audit_store")

    report = finalize_completed_candidate(
        candidate,
        store=store,
        completion_marker=completion,
    )

    assert report["status"] == "compacted"
    assert report["removed_file_count"] == 5
    assert all(
        not (candidate / name).exists()
        for name in (
            "physical_pruning_plan.json",
            "physical_plan.json",
            "legalized_plan.json",
            "pruning_request.json",
            "sampling_pruning_request.json",
        )
    )
    manifest = json.loads(
        (candidate / "candidate_audit_manifest.json").read_text(encoding="utf-8")
    )
    summary = json.loads(
        (candidate / "structure_plan_summary.json").read_text(encoding="utf-8")
    )
    assert sorted(manifest["artifacts"]) == ["physical_plan", "pruning_request"]
    assert summary["plan_entry_count"] == 1
    assert summary["request_entry_count"] == 1
    assert summary["selected_atomic_unit_count"] == 2
    assert summary["modules"] == ["backbone_m1.blocks.1.0"]
    assert "entries" not in summary
    plan_reference = manifest["artifacts"]["physical_plan"]
    assert resolve_artifact_reference(store.root, plan_reference) == _plan_payload()


def test_incomplete_candidate_refuses_content_addressed_compaction(
    tmp_path: Path,
) -> None:
    from search.artifacts.candidate_audit_store import (
        CandidateAuditStore,
        finalize_completed_candidate,
    )

    generation = tmp_path / "budget_010" / "generation_002"
    candidate = generation / "stage2" / "candidate-a"
    candidate.mkdir(parents=True)
    _write_aliases(candidate)

    with pytest.raises(RuntimeError, match="completion_marker_missing"):
        finalize_completed_candidate(
            candidate,
            store=CandidateAuditStore(tmp_path / "audit_store"),
            completion_marker=generation / "stage2_artifact_retention.json",
        )

    assert (candidate / "physical_pruning_plan.json").is_file()
    assert not (candidate / "candidate_audit_manifest.json").exists()


def test_mismatched_aliases_fail_closed_before_removal(tmp_path: Path) -> None:
    from search.artifacts.candidate_audit_store import (
        CandidateAuditStore,
        finalize_completed_candidate,
    )

    generation = tmp_path / "budget_010" / "generation_003"
    candidate = generation / "stage2" / "candidate-a"
    candidate.mkdir(parents=True)
    _write_aliases(candidate)
    (candidate / "legalized_plan.json").write_text(
        json.dumps({"schema_version": "different"}), encoding="utf-8"
    )
    completion = generation / "stage2_artifact_retention.json"
    completion.write_text("{}", encoding="utf-8")

    with pytest.raises(RuntimeError, match="artifact_alias_mismatch"):
        finalize_completed_candidate(
            candidate,
            store=CandidateAuditStore(tmp_path / "audit_store"),
            completion_marker=completion,
        )

    assert (candidate / "physical_pruning_plan.json").is_file()
    assert (candidate / "legalized_plan.json").is_file()

