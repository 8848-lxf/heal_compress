"""Lazy model-family dispatch that preserves the existing pyramid default."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Type

from .contracts import SearchRunner

LIDAR_PYRAMID = "lidar_pyramid"
LIDAR_COBEVT = "lidar_cobevt"


def model_family_name(config: Mapping[str, Any]) -> str:
    model = config.get("model", {})
    if not isinstance(model, Mapping):
        raise RuntimeError("model_config_must_be_mapping")
    family = str(model.get("family", LIDAR_PYRAMID)).strip().lower()
    if family not in {LIDAR_PYRAMID, LIDAR_COBEVT}:
        raise RuntimeError(f"unsupported_model_family:{family}")
    return family


def _load_pyramid_runner() -> Type[SearchRunner]:
    from ..orchestration.lidar_pyramid_search import LidarPyramidTwoStageSearch

    return LidarPyramidTwoStageSearch


def _load_cobevt_runner() -> Type[SearchRunner]:
    from ..orchestration.lidar_cobevt_smoke import LidarCobevtSmokeSearch

    return LidarCobevtSmokeSearch


def create_family_runner(
    *,
    config: dict[str, Any],
    checkpoint: str | Path,
    output_root: str | Path,
    resume: str | Path | None = None,
    pyramid_runner_cls: Type[SearchRunner] | None = None,
) -> SearchRunner:
    """Create exactly one family runner without importing the other family."""

    family = model_family_name(config)
    if family == LIDAR_PYRAMID:
        runner_cls = pyramid_runner_cls or _load_pyramid_runner()
    else:
        runner_cls = _load_cobevt_runner()
    return runner_cls(
        config=config,
        checkpoint=checkpoint,
        output_root=output_root,
        resume=resume,
    )

