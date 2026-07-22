"""V2XViT adapter for window and agent-relation d_h pruning."""

from __future__ import annotations

from torch import nn

from search.model_families.transformer.dh_physical_rewrite import (
    discover_attention_families,
    materialize_family_head_dimension,
)


def attention_families(model: nn.Module):
    return discover_attention_families("lidar_v2xvit", model)


def materialize(model: nn.Module, family, masks):
    return materialize_family_head_dimension("lidar_v2xvit", model, family, masks)


__all__ = ["attention_families", "materialize"]
