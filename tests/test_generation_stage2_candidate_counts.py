from __future__ import annotations

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


def test_expanded_band_runs_only_after_primary_build_supply_is_exhausted(
    tmp_path: Path,
) -> None:
    from search.admission.bops_band import BopsBandPolicy
    from search.orchestration.generation_stage2 import run_generation_stage2

    calls: list[list[str]] = []

    def build(rows):
        calls.append([row.candidate_hash for row in rows])
        return [
            (
                {
                    "candidate_hash": row.candidate_hash,
                    "status": "engine_build_failed",
                    "failure_reason": "build",
                }
                if row.candidate_hash == "primary"
                else _ok_build(row.candidate_hash, 0.207)
            )
            for row in rows
        ]

    report = run_generation_stage2(
        ranked_records=[
            _record("primary", 0.20, 2.0),
            _record("expanded", 0.207, 1.0),
        ],
        generation_index=0,
        output_dir=tmp_path,
        policy=BopsBandPolicy(
            target=0.20,
            primary_tolerance=0.005,
            expanded_tolerance=0.0075,
        ),
        build_smoke_batch_fn=build,
        evaluate_500_batch_fn=lambda _rows: [],
    )

    assert calls == [["primary"], ["expanded"]]
    assert report["status"] == "single_candidate_direct_winner"
    assert report["winner"]["candidate_hash"] == "expanded"
    assert report["active_bops_mode"] == "expanded_bops_tolerance"
    assert report["expanded_tolerance_reason"] == "primary_build_supply_exhausted"


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
