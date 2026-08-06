"""Formal post-scatter TensorRT export adapter for Pyramid candidates."""

from __future__ import annotations

import torch.nn as nn

from deploy.pyramid_post_scatter import PostScatterLidarPyramid


def build_search_post_scatter_export_module(
    model: nn.Module,
    *,
    output_names: tuple[str, ...],
    modality: str = "m1",
) -> PostScatterLidarPyramid:
    """Build the fixed-BEV/dynamic-agent graph used by Stage-2 and CARLA."""

    return PostScatterLidarPyramid(model, modality, list(output_names))


__all__ = ["build_search_post_scatter_export_module"]
