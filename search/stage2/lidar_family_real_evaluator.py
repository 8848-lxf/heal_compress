"""Family-aware name for the production HEAL LiDAR Stage-2 evaluator."""

from __future__ import annotations

from .lidar_pyramid_real_evaluator import LidarPyramidRealEvaluator


class LidarFamilyRealEvaluator(LidarPyramidRealEvaluator):
    """Use family hooks while preserving the validated production core."""


__all__ = ["LidarFamilyRealEvaluator"]
