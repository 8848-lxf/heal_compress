"""CoBEVT adapter for the shared physical head-dimension rewrite."""

from __future__ import annotations

from torch import nn

from search.model_families.transformer.dh_physical_rewrite import (
    discover_attention_families,
    materialize_family_head_dimension,
)


def attention_families(model: nn.Module):
    return discover_attention_families("lidar_cobevt", model)


def materialize(model: nn.Module, family, masks):
    return materialize_family_head_dimension("lidar_cobevt", model, family, masks)


__all__ = ["attention_families", "materialize"]
