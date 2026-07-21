from __future__ import annotations

from pathlib import Path

import pytest


def test_cobevt_evaluator_runs_smoke_before_fixed_screening(tmp_path: Path) -> None:
    from search.stage2.lidar_cobevt_real_evaluator import LidarCobevtRealEvaluator

    calls = []

    def provider(**kwargs):
        calls.append(kwargs)
        frames = int(kwargs["num_frames"])
        return {
            "status": "ok",
            "num_evaluated_frames": frames,
            "num_skipped_frames": 0,
        }

    evaluator = LidarCobevtRealEvaluator(
        engine_evaluator=provider,
        evaluation_num_workers=8,
        ap_iou_backend="gpu",
    )
    result = evaluator.evaluate(
        engine_path=tmp_path / "engine.plan",
        output_dir=tmp_path / "evaluation",
        common_request={
            "device": "cuda:7",
            "smoke_eval_manifest_path": tmp_path / "smoke10.json",
            "screening_eval_manifest_path": tmp_path / "fixed50.json",
        },
        smoke_frames=10,
        screening_frames=50,
        warmup_frames=20,
    )

    assert result["status"] == "ok"
    assert [row["num_frames"] for row in calls] == [10, 50]
    assert [Path(row["eval_manifest_path"]).name for row in calls] == [
        "smoke10.json",
        "fixed50.json",
    ]
    assert all("smoke_eval_manifest_path" not in row for row in calls)
    assert all("screening_eval_manifest_path" not in row for row in calls)
    assert all(row["num_workers"] == 8 for row in calls)
    assert all(row["ap_iou_backend"] == "gpu" for row in calls)


def test_cobevt_evaluator_stops_after_failed_smoke(tmp_path: Path) -> None:
    from search.stage2.lidar_cobevt_real_evaluator import LidarCobevtRealEvaluator

    calls = []

    def provider(**kwargs):
        calls.append(kwargs)
        return {
            "status": "evaluation_failed",
            "num_evaluated_frames": 9,
            "num_skipped_frames": 1,
        }

    evaluator = LidarCobevtRealEvaluator(engine_evaluator=provider)
    result = evaluator.evaluate(
        engine_path=tmp_path / "engine.plan",
        output_dir=tmp_path / "evaluation",
        common_request={},
    )

    assert result["status"] == "smoke_failed"
    assert len(calls) == 1


@pytest.mark.parametrize(
    ("workers", "backend"),
    ((4, "gpu"), (8, "cpu")),
)
def test_cobevt_evaluator_requires_gpu_protocol(workers: int, backend: str) -> None:
    from search.stage2.lidar_cobevt_real_evaluator import LidarCobevtRealEvaluator

    with pytest.raises(ValueError):
        LidarCobevtRealEvaluator(
            engine_evaluator=lambda **kwargs: {},
            evaluation_num_workers=workers,
            ap_iou_backend=backend,
        )
