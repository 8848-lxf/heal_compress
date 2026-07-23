from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_deploy_candidate_builds_engine_without_evaluating_frames(tmp_path) -> None:
    from search.candidate import CandidatePhenotype
    from search.stage2.lidar_pyramid_real_evaluator import LidarPyramidRealEvaluator

    evaluator = object.__new__(LidarPyramidRealEvaluator)
    evaluator._deploy_only = lambda **_kwargs: {
        "status": "ok",
        "engine_path": str(tmp_path / "engine.plan"),
        "engine_hash": "engine-hash",
        "deployment_hash": "deployment-hash",
        "physical_hash": "physical-hash",
    }
    evaluator._evaluate_engine = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("deploy-only path must not evaluate frames")
    )

    result = evaluator.deploy_candidate(
        CandidatePhenotype(),
        output_dir=tmp_path,
        candidate_hash="candidate",
    )

    assert result["status"] == "ok"
    assert result["evaluation_500_skipped"] is True
    assert result["num_evaluated_frames"] == 0
    assert (tmp_path / "stage2_deployment.json").is_file()
