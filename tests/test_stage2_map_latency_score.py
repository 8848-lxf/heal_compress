from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_map_minus_latency_ratio_encodes_ten_percent_for_point_zero_one() -> None:
    from search.stage2.objective import Stage2ObjectiveConfig, compute_stage2_score

    config = Stage2ObjectiveConfig(
        score_mode="map_minus_latency_ratio",
        latency_weight=0.10,
        latency_metric="forward_p50_ms",
    )
    baseline = {"mAP": 0.73, "forward_p50_ms": 10.0}
    accurate = compute_stage2_score(
        {"status": "ok", "mAP": 0.72, "forward_p50_ms": 10.0},
        baseline=baseline,
        config=config,
    )
    faster = compute_stage2_score(
        {"status": "ok", "mAP": 0.71, "forward_p50_ms": 9.0},
        baseline=baseline,
        config=config,
    )

    assert accurate["F2"] == pytest.approx(faster["F2"])
    assert accurate["F2"] == pytest.approx(0.62)
    assert accurate["selection_direction"] == "maximize"
    assert accurate["R_latency_real"] == pytest.approx(1.0)


def test_formal_stage2_has_no_map_or_ap07_hard_gate() -> None:
    from search.stage2.objective import Stage2ObjectiveConfig, compute_stage2_score

    result = compute_stage2_score(
        {
            "status": "ok",
            "mAP": 0.01,
            "AP07": 0.0,
            "forward_p50_ms": 1.0,
        },
        baseline={"mAP": 0.73, "forward_p50_ms": 10.0},
        config=Stage2ObjectiveConfig(
            score_mode="map_minus_latency_ratio",
            latency_weight=0.10,
            latency_metric="forward_p50_ms",
            min_map=0.70,
            min_ap07=0.56,
            max_map_drop=0.01,
        ),
    )

    assert result["status"] == "ok"
    assert math.isfinite(result["F2"])
    assert result["accuracy_admission_passed"] is True


def test_formal_stage2_failure_and_nonfinite_metrics_use_negative_infinity() -> None:
    from search.stage2.objective import Stage2ObjectiveConfig, compute_stage2_score

    config = Stage2ObjectiveConfig(score_mode="map_minus_latency_ratio")
    failed = compute_stage2_score(
        {"status": "engine_build_failed"},
        baseline={"mAP": 0.73, "forward_p50_ms": 10.0},
        config=config,
    )
    nonfinite = compute_stage2_score(
        {"status": "ok", "mAP": float("nan"), "forward_p50_ms": 1.0},
        baseline={"mAP": 0.73, "forward_p50_ms": 10.0},
        config=config,
    )

    assert failed["F2"] == -float("inf")
    assert failed["selection_direction"] == "maximize"
    assert nonfinite["F2"] == -float("inf")
    assert nonfinite["status"] == "nonfinite_stage2_metric"


def test_shared_strict_fp32_reference_is_returned_without_baseline_build(tmp_path: Path) -> None:
    from search.hashing import canonical_json_hash
    from search.stage2.lidar_pyramid_real_evaluator import LidarPyramidRealEvaluator
    from search.stage2.objective import Stage2ObjectiveConfig

    unsigned = {
        "status": "ok",
        "reference_precision": "strict_fp32",
        "mAP": 0.73,
        "forward_p50_ms": 10.0,
        "engine_hash": "engine",
        "eval_hash": "evaluation",
    }
    shared = {
        **unsigned,
        "reference_hash": canonical_json_hash(unsigned),
    }
    evaluator = object.__new__(LidarPyramidRealEvaluator)
    evaluator.objective_config = Stage2ObjectiveConfig(
        score_mode="map_minus_latency_ratio",
        latency_metric="forward_p50_ms",
    )
    evaluator.shared_stage2_reference = shared
    evaluator.run_dir = tmp_path

    def forbidden(*_args, **_kwargs):
        raise AssertionError("shared reference must not rebuild a baseline")

    evaluator.evaluate_original_baseline = forbidden  # type: ignore[method-assign]

    assert evaluator._stage2_reference_baseline() == shared
