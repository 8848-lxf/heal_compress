from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_completed_generation_duplicate_artifacts_become_hardlinks(
    tmp_path: Path,
) -> None:
    from search.orchestration.stage2_artifact_compaction import compact_run

    generation = tmp_path / "budget_005" / "generation_001"
    candidate = generation / "stage2" / "candidate-a"
    candidate.mkdir(parents=True)
    groups = (
        ("physical_pruning_plan.json", "physical_plan.json", "legalized_plan.json"),
        ("pruning_request.json", "sampling_pruning_request.json"),
        ("materialization_report.json", "materialization_ledger.json"),
    )
    for index, group in enumerate(groups):
        payload = (f"payload-{index}" * 100).encode()
        for name in group:
            (candidate / name).write_bytes(payload)
    (generation / "stage2_artifact_retention.json").write_text(
        json.dumps({"enabled": True}), encoding="utf-8"
    )

    result = compact_run(tmp_path)

    assert result["completed_generation_count"] == 1
    assert result["linked_alias_count"] == 4
    assert result["bytes_reclaimed"] > 0
    for group in groups:
        inodes = {(candidate / name).stat().st_ino for name in group}
        assert len(inodes) == 1
    assert (tmp_path / "stage2_hardlink_compaction.json").is_file()


def test_incomplete_generation_and_mismatched_files_are_not_modified(
    tmp_path: Path,
) -> None:
    from search.orchestration.stage2_artifact_compaction import compact_run

    complete = tmp_path / "budget_005" / "generation_001"
    candidate = complete / "stage2" / "candidate-a"
    candidate.mkdir(parents=True)
    canonical = candidate / "physical_pruning_plan.json"
    alias = candidate / "physical_plan.json"
    canonical.write_bytes(b"canonical")
    alias.write_bytes(b"different")
    (complete / "stage2_artifact_retention.json").write_text(
        "{}", encoding="utf-8"
    )
    incomplete = tmp_path / "budget_005" / "generation_002" / "stage2" / "b"
    incomplete.mkdir(parents=True)
    unfinished_a = incomplete / "pruning_request.json"
    unfinished_b = incomplete / "sampling_pruning_request.json"
    unfinished_a.write_bytes(b"same")
    unfinished_b.write_bytes(b"same")

    result = compact_run(tmp_path)

    assert result["linked_alias_count"] == 0
    assert result["content_mismatch_count"] == 1
    assert canonical.stat().st_ino != alias.stat().st_ino
    assert unfinished_a.stat().st_ino != unfinished_b.stat().st_ino
