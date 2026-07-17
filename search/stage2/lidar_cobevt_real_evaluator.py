"""Smoke-first real engine evaluation protocol for LiDAR CoBEVT."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping


EngineEvaluator = Callable[..., Mapping[str, Any]]


class LidarCobevtRealEvaluator:
    def __init__(
        self,
        *,
        engine_evaluator: EngineEvaluator,
        evaluation_num_workers: int = 8,
        ap_iou_backend: str = "gpu",
    ) -> None:
        self.engine_evaluator = engine_evaluator
        self.evaluation_num_workers = int(evaluation_num_workers)
        self.ap_iou_backend = str(ap_iou_backend).lower()
        if self.evaluation_num_workers != 8:
            raise ValueError("cobevt_evaluation_num_workers_must_equal_8")
        if self.ap_iou_backend != "gpu":
            raise ValueError("cobevt_evaluation_gpu_ap_iou_required")

    @staticmethod
    def _complete(result: Mapping[str, Any], expected_frames: int) -> bool:
        return (
            str(result.get("status")) == "ok"
            and int(result.get("num_evaluated_frames", -1)) == int(expected_frames)
            and int(result.get("num_skipped_frames", -1)) == 0
        )

    def _run(
        self,
        *,
        engine_path: Path,
        output_dir: Path,
        common_request: Mapping[str, Any],
        num_frames: int,
        warmup_frames: int,
    ) -> dict[str, Any]:
        request = dict(common_request)
        request.update(
            {
                "ap_iou_backend": self.ap_iou_backend,
                "engine_path": engine_path,
                "num_frames": int(num_frames),
                "num_workers": self.evaluation_num_workers,
                "output_dir": output_dir,
                "strict_gpu_ap_iou": True,
                "warmup_frames": int(warmup_frames),
            }
        )
        return dict(self.engine_evaluator(**request))

    def evaluate(
        self,
        *,
        engine_path: str | Path,
        output_dir: str | Path,
        common_request: Mapping[str, Any],
        smoke_frames: int = 10,
        screening_frames: int = 50,
        warmup_frames: int = 20,
    ) -> dict[str, Any]:
        destination = Path(output_dir)
        smoke = self._run(
            engine_path=Path(engine_path),
            output_dir=destination / "smoke10",
            common_request=common_request,
            num_frames=int(smoke_frames),
            warmup_frames=int(warmup_frames),
        )
        if not self._complete(smoke, int(smoke_frames)):
            return {"status": "smoke_failed", "smoke": smoke}
        screening = self._run(
            engine_path=Path(engine_path),
            output_dir=destination / "fixed50",
            common_request=common_request,
            num_frames=int(screening_frames),
            warmup_frames=int(warmup_frames),
        )
        if not self._complete(screening, int(screening_frames)):
            return {
                "status": "screening_failed",
                "smoke": smoke,
                "screening": screening,
            }
        return {"status": "ok", "smoke": smoke, "screening": screening}


__all__ = ["LidarCobevtRealEvaluator"]
