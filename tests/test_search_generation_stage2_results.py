from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_generation_with_zero_candidates_records_bops_rejection(tmp_path: Path) -> None:
    from search.stage2.generation_results import write_generation_stage2_results

    report = write_generation_stage2_results(
        tmp_path,
        round_index=2,
        generation_index=4,
        bops_target=0.2,
        selection_report={
            "selected_count": 0,
            "no_candidate_reason": "all_generation_candidates_rejected_by_BOPS_band",
        },
        candidate_rows=[],
        expected_screening_frames=500,
    )

    assert report["status"] == "no_stage2_candidate"
    assert report["winner"] is None
    failure = json.loads((tmp_path / "generation_failure.json").read_text())
    assert failure["failure_reason"] == "all_generation_candidates_rejected_by_BOPS_band"


def test_generation_with_one_candidate_skips_500_and_keeps_engine(tmp_path: Path) -> None:
    from search.stage2.generation_results import write_generation_stage2_results

    report = write_generation_stage2_results(
        tmp_path,
        round_index=0,
        generation_index=0,
        bops_target=0.3,
        selection_report={"selected_count": 1},
        candidate_rows=[
            {
                "candidate_hash": "only",
                "status": "ok",
                "engine_path": "/tmp/only.plan",
                "num_evaluated_frames": 0,
            }
        ],
        expected_screening_frames=500,
    )

    assert report["status"] == "single_candidate_direct_winner"
    assert report["winner"]["candidate_hash"] == "only"
    assert report["winner"]["evaluation_500_skipped"] is True
    assert report["evaluated_500_count"] == 0


def test_generation_multi_candidate_requires_exact_500_frames_and_minimum_f2(tmp_path: Path) -> None:
    from search.stage2.generation_results import write_generation_stage2_results

    report = write_generation_stage2_results(
        tmp_path,
        round_index=1,
        generation_index=3,
        bops_target=0.25,
        selection_report={"selected_count": 3},
        candidate_rows=[
            {
                "candidate_hash": "winner",
                "status": "ok",
                "F2": 0.2,
                "num_evaluated_frames": 500,
                "num_skipped_frames": 0,
            },
            {
                "candidate_hash": "runner-up",
                "status": "ok",
                "F2": 0.3,
                "num_evaluated_frames": 500,
                "num_skipped_frames": 0,
            },
            {
                "candidate_hash": "short-eval",
                "status": "ok",
                "F2": 0.1,
                "num_evaluated_frames": 499,
                "num_skipped_frames": 0,
            },
        ],
        expected_screening_frames=500,
    )

    assert report["status"] == "generation_winner_selected"
    assert report["winner"]["candidate_hash"] == "winner"
    assert report["evaluated_500_count"] == 2
    assert report["failures"][0]["candidate_hash"] == "short-eval"
    assert report["failures"][0]["generation_admission_rejection"] == [
        "evaluated_frames_499_expected_500"
    ]


def test_generation_result_count_must_match_selection(tmp_path: Path) -> None:
    from search.stage2.generation_results import write_generation_stage2_results

    with pytest.raises(RuntimeError, match="generation_stage2_result_count_mismatch"):
        write_generation_stage2_results(
            tmp_path,
            round_index=0,
            generation_index=0,
            bops_target=0.3,
            selection_report={"selected_count": 2},
            candidate_rows=[{"candidate_hash": "one"}],
            expected_screening_frames=500,
        )
