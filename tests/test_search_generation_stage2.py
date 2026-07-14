from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _record(candidate_hash: str, rank: int):
    return SimpleNamespace(candidate_hash=candidate_hash, F1=float(rank), genotype=None, phenotype=None)


def test_generation_stage2_backfills_failures_and_deployment_duplicates(tmp_path: Path) -> None:
    from search.orchestration.generation_stage2 import deploy_generation_with_backfill

    records = [_record(name, rank) for rank, name in enumerate("abcdefg")]
    results = {
        "a": {"status": "engine_build_failure", "failure_reason": "build"},
        "b": {"status": "ok", "physical_hash": "p1", "deployment_hash": "d1", "F2": 0.5, "mAP": 0.6},
        "c": {"status": "ok", "physical_hash": "p1", "deployment_hash": "d1", "F2": 0.4, "mAP": 0.6},
        "d": {"status": "ok", "physical_hash": "p2", "deployment_hash": "d2", "F2": 0.3, "mAP": 0.6},
        "e": {"status": "ok", "physical_hash": "p3", "deployment_hash": "d3", "F2": 0.1, "mAP": 0.6},
        "f": {"status": "ok", "physical_hash": "p4", "deployment_hash": "d4", "F2": 0.2, "mAP": 0.6},
        "g": {"status": "ok", "physical_hash": "p5", "deployment_hash": "d5", "F2": 0.6, "mAP": 0.6},
    }
    calls = []

    def deploy(record, candidate_dir):
        calls.append((record.candidate_hash, Path(candidate_dir)))
        return dict(results[record.candidate_hash])

    report = deploy_generation_with_backfill(
        records,
        generation_index=0,
        output_dir=tmp_path,
        deploy_fn=deploy,
        topk=5,
    )

    assert [row["candidate_hash"] for row in report["candidates"]] == ["b", "d", "e", "f", "g"]
    assert report["winner"]["candidate_hash"] == "e"
    assert report["attempted_count"] == 7
    assert report["failure_records"][0]["candidate_hash"] == "a"
    assert report["failure_records"][1]["failure_reason"] == "duplicate_physical_deployment_hash"
    assert [name for name, _path in calls] == list("abcdefg")
    assert (tmp_path / "generation_001_top5.json").is_file()
    assert (tmp_path / "generation_001_stage2.csv").is_file()
    assert (tmp_path / "generation_001_winner.json").is_file()
    assert len(list(csv.DictReader((tmp_path / "generation_001_stage2.csv").open(encoding="utf-8")))) == 5


def test_generation_stage2_writes_failure_report_before_insufficient_top5(tmp_path: Path) -> None:
    from search.orchestration.generation_stage2 import deploy_generation_with_backfill

    records = [_record("a", 0), _record("b", 1)]

    with pytest.raises(RuntimeError, match="insufficient_unique_deployable_candidates:2<5"):
        deploy_generation_with_backfill(
            records,
            generation_index=2,
            output_dir=tmp_path,
            deploy_fn=lambda record, _candidate_dir: {
                "status": "ok",
                "physical_hash": f"p-{record.candidate_hash}",
                "deployment_hash": f"d-{record.candidate_hash}",
                "F2": 1.0,
            },
            topk=5,
        )

    payload = json.loads((tmp_path / "generation_003_top5.json").read_text(encoding="utf-8"))
    assert payload["status"] == "insufficient_unique_deployable_candidates"
    assert payload["selected_count"] == 2
    assert not (tmp_path / "generation_003_winner.json").exists()


def test_generation_stage2_parallel_batches_preserve_ranked_backfill(tmp_path: Path) -> None:
    from search.orchestration.generation_stage2 import deploy_generation_with_backfill

    records = [_record(name, rank) for rank, name in enumerate("abcdefg")]
    results = {
        "a": {"status": "engine_build_failure", "failure_reason": "build"},
        "b": {"status": "ok", "physical_hash": "p1", "deployment_hash": "d1", "F2": 0.5},
        "c": {"status": "ok", "physical_hash": "p1", "deployment_hash": "d1", "F2": 0.4},
        "d": {"status": "ok", "physical_hash": "p2", "deployment_hash": "d2", "F2": 0.3},
        "e": {"status": "ok", "physical_hash": "p3", "deployment_hash": "d3", "F2": 0.1},
        "f": {"status": "ok", "physical_hash": "p4", "deployment_hash": "d4", "F2": 0.2},
        "g": {"status": "ok", "physical_hash": "p5", "deployment_hash": "d5", "F2": 0.6},
    }
    batches = []

    def deploy_batch(items):
        batches.append([record.candidate_hash for record, _path in items])
        return [dict(results[record.candidate_hash]) for record, _path in items]

    report = deploy_generation_with_backfill(
        records,
        generation_index=0,
        output_dir=tmp_path,
        deploy_fn=None,
        deploy_batch_fn=deploy_batch,
        parallelism=3,
        topk=5,
    )

    assert batches == [list("abc"), list("def"), ["g"]]
    assert [row["candidate_hash"] for row in report["candidates"]] == ["b", "d", "e", "f", "g"]
    assert report["attempted_count"] == 7
    assert report["parallelism"] == 3


@pytest.mark.parametrize(
    ("retention", "passed"),
    [(0.2049, False), (0.205, True), (0.21, True), (0.215, True), (0.2151, False)],
)
def test_stage1_fixed_bops_admission_checks_both_interval_bounds(retention: float, passed: bool) -> None:
    from search.orchestration.generation_stage2 import fixed_bops_admission

    report = fixed_bops_admission(retention, target=0.21, tolerance=0.005)

    assert report["passed"] is passed
    assert report["legal_interval"] == pytest.approx([0.205, 0.215])
