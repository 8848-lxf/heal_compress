from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _record(candidate_hash: str, bops: float, j1: float = 1.0):
    return SimpleNamespace(
        candidate_hash=candidate_hash,
        F1=-float(j1),
        metrics={"J1": float(j1), "R_BOPS": float(bops)},
    )


def _ok_build(candidate_hash: str, bops: float = 0.20) -> dict[str, object]:
    return {
        "candidate_hash": candidate_hash,
        "status": "ok",
        "physical_hash": f"physical-{candidate_hash}",
        "deployment_hash": f"deployment-{candidate_hash}",
        "physical_BOPS_retention": float(bops),
        "BOPS_retention": float(bops),
        "engine_path": f"/{candidate_hash}.plan",
    }


def _ok_eval(row: dict[str, object], f2: float) -> dict[str, object]:
    return {
        **row,
        "status": "ok",
        "F2": float(f2),
        "evaluated": 500,
        "skipped": 0,
    }


def test_one_candidate_skips_500_and_becomes_winner(tmp_path: Path) -> None:
    from search.admission.bops_band import BopsBandPolicy
    from search.orchestration.generation_stage2 import run_generation_stage2

    eval_calls: list[dict[str, object]] = []
    report = run_generation_stage2(
        ranked_records=[_record("only", bops=0.20)],
        generation_index=0,
        output_dir=tmp_path,
        policy=BopsBandPolicy(target=0.20),
        topk=5,
        build_smoke_batch_fn=lambda rows: [_ok_build("only")],
        evaluate_500_batch_fn=lambda rows: eval_calls.extend(rows) or [],
    )

    assert report["status"] == "single_candidate_direct_winner"
    assert report["winner"]["candidate_hash"] == "only"
    assert report["winner"]["evaluation_500_skipped"] is True
    assert eval_calls == []


def test_two_candidates_both_run_500_and_max_f2_wins(tmp_path: Path) -> None:
    from search.admission.bops_band import BopsBandPolicy
    from search.orchestration.generation_stage2 import run_generation_stage2

    records = [_record("a", 0.20, 2.0), _record("b", 0.20, 1.0)]
    report = run_generation_stage2(
        ranked_records=records,
        generation_index=0,
        output_dir=tmp_path,
        policy=BopsBandPolicy(target=0.20),
        topk=5,
        build_smoke_batch_fn=lambda rows: [
            _ok_build(row.candidate_hash) for row in rows
        ],
        evaluate_500_batch_fn=lambda rows: [
            _ok_eval(row, f2=0.5 if row["candidate_hash"] == "a" else 0.6)
            for row in rows
        ],
    )

    assert report["status"] == "evaluated_generation_winner"
    assert report["winner"]["candidate_hash"] == "b"
    assert report["winner"]["evaluation_500_skipped"] is False
    assert report["evaluated_500_count"] == 2


def test_zero_candidates_records_bops_funnel_without_stage2(tmp_path: Path) -> None:
    from search.admission.bops_band import BopsBandPolicy
    from search.orchestration.generation_stage2 import run_generation_stage2

    build_calls = []

    def fail_eval(_rows):
        raise AssertionError("zero candidates must not evaluate")

    report = run_generation_stage2(
        ranked_records=[_record("low", 0.18), _record("high", 0.22)],
        generation_index=0,
        output_dir=tmp_path,
        policy=BopsBandPolicy(target=0.20),
        topk=5,
        build_smoke_batch_fn=lambda rows: build_calls.extend(rows) or [],
        evaluate_500_batch_fn=fail_eval,
    )

    assert report["status"] == "no_bops_admissible_candidates"
    assert report["selected_count"] == 0
    assert report["bops_admission"]["nearest_misses"]
    assert build_calls == []
    assert (tmp_path / "generation_001_top5.json").is_file()


def test_build_failure_does_not_trigger_engine_backfill(
    tmp_path: Path,
) -> None:
    from search.admission.bops_band import BopsBandPolicy
    from search.orchestration.generation_stage2 import run_generation_stage2

    calls: list[list[str]] = []

    def build(rows):
        calls.append([row.candidate_hash for row in rows])
        return [
            {
                "candidate_hash": row.candidate_hash,
                "status": "engine_build_failed",
                "failure_reason": "build",
            }
            if row.candidate_hash in {"a", "b"}
            else _ok_build(row.candidate_hash)
            for row in rows
        ]

    report = run_generation_stage2(
        ranked_records=[
            _record(name, 0.20, 10.0 - index)
            for index, name in enumerate("abcdefg")
        ],
        generation_index=0,
        output_dir=tmp_path,
        policy=BopsBandPolicy(target=0.20),
        topk=5,
        build_smoke_batch_fn=build,
        evaluate_500_batch_fn=lambda rows: [
            _ok_eval(row, f2=float(index)) for index, row in enumerate(rows)
        ],
    )

    assert calls == [list("abcde")]
    assert report["engine_build_attempt_count"] == 5
    assert report["build_success_count"] == 3
    assert report["evaluated_500_count"] == 3
    assert {row["candidate_hash"] for row in report["failure_records"]} == {"a", "b"}


def test_physical_preflight_backfills_without_exceeding_engine_build_cap(
    tmp_path: Path,
) -> None:
    from search.admission.bops_band import BopsBandPolicy
    from search.orchestration.generation_stage2 import run_generation_stage2

    preflight_calls: list[list[str]] = []
    build_calls: list[list[str]] = []

    def preflight(rows):
        preflight_calls.append([row.candidate_hash for row in rows])
        return [
            {
                "candidate_hash": row.candidate_hash,
                "status": "ok",
                "physical_hash": f"physical-{row.candidate_hash}",
                "physical_BOPS_retention": (
                    0.220 if row.candidate_hash in {"a", "b"} else 0.20
                ),
            }
            for row in rows
        ]

    def build(rows):
        build_calls.append([row.candidate_hash for row in rows])
        return [_ok_build(row.candidate_hash) for row in rows]

    report = run_generation_stage2(
        ranked_records=[
            _record(name, 0.20, 10.0 - index)
            for index, name in enumerate("abcdefg")
        ],
        generation_index=0,
        output_dir=tmp_path,
        policy=BopsBandPolicy(target=0.20),
        topk=5,
        physical_preflight_batch_fn=preflight,
        build_smoke_batch_fn=build,
        evaluate_500_batch_fn=lambda rows: [
            _ok_eval(row, f2=float(index)) for index, row in enumerate(rows)
        ],
    )

    assert preflight_calls == [list("abcde"), list("fg")]
    assert build_calls == [list("cdefg")]
    assert report["physical_preflight_attempt_count"] == 7
    assert report["physical_preflight_admitted_count"] == 5
    assert report["engine_build_attempt_count"] == 5
    assert all(
        row["failure_reason"] == "physical_bops_out_of_active_interval"
        for row in report["failure_records"]
    )


def test_zero_physical_preflight_candidates_skips_generation_with_reason(
    tmp_path: Path,
) -> None:
    from search.admission.bops_band import BopsBandPolicy
    from search.orchestration.generation_stage2 import run_generation_stage2

    build_calls: list[object] = []
    report = run_generation_stage2(
        ranked_records=[_record("a", 0.20), _record("b", 0.20)],
        generation_index=0,
        output_dir=tmp_path,
        policy=BopsBandPolicy(target=0.20),
        topk=5,
        physical_preflight_batch_fn=lambda rows: [
            {
                "candidate_hash": row.candidate_hash,
                "status": "ok",
                "physical_hash": f"physical-{row.candidate_hash}",
                "physical_BOPS_retention": 0.22,
            }
            for row in rows
        ],
        build_smoke_batch_fn=lambda rows: build_calls.extend(rows) or [],
        evaluate_500_batch_fn=lambda _rows: [],
    )

    assert build_calls == []
    assert report["status"] == "no_physical_bops_admissible_candidates"
    assert report["generation_skipped"] is True
    assert report["generation_skip_reason"] == "physical_bops_preflight_exhausted"
    assert report["engine_build_attempt_count"] == 0
    decision = json.loads(
        (tmp_path / "generation_001_count_decision.json").read_text(encoding="utf-8")
    )
    assert decision == {
        "build_success_count": 0,
        "engine_build_attempt_count": 0,
        "evaluated_500_count": 0,
        "evaluation_500_skipped": False,
        "failure_reason_histogram": {
            "physical_bops_out_of_active_interval": 2
        },
        "failure_stage_histogram": {"physical_bops_admission": 2},
        "generation_skip_reason": "physical_bops_preflight_exhausted",
        "generation_skipped": True,
        "physical_preflight_admitted_count": 0,
        "physical_preflight_attempt_count": 2,
        "selected_count": 0,
        "status": "no_physical_bops_admissible_candidates",
    }


def test_realized_bops_outside_active_band_is_not_admitted(tmp_path: Path) -> None:
    from search.admission.bops_band import BopsBandPolicy
    from search.orchestration.generation_stage2 import run_generation_stage2

    report = run_generation_stage2(
        ranked_records=[_record("candidate", 0.20)],
        generation_index=0,
        output_dir=tmp_path,
        policy=BopsBandPolicy(target=0.20),
        build_smoke_batch_fn=lambda _rows: [
            {
                **_ok_build("candidate", bops=0.216),
                "physical_BOPS_retention": 0.20,
            }
        ],
        evaluate_500_batch_fn=lambda _rows: [],
    )

    assert report["status"] == "no_deployable_candidates"
    assert report["selected_count"] == 0
    assert report["failure_records"][0]["status"] == "realized_bops_out_of_band"
