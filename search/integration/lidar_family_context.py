"""Family-neutral facade over the production HEAL LiDAR context builder."""

from __future__ import annotations

from typing import Any

from .lidar_pyramid_context import (
    LidarPyramidSearchContext as LidarFamilySearchContext,
)
from .lidar_pyramid_context import build_lidar_pyramid_context as _build_lidar_context


def build_lidar_family_context(
    *, model_family: str, **kwargs: Any
) -> LidarFamilySearchContext:
    return _build_lidar_context(model_family=model_family, **kwargs)


__all__ = ["LidarFamilySearchContext", "build_lidar_family_context"]
