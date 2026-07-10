"""Structural skip policy helpers."""

from __future__ import annotations


FIXED_SHAPE_KEYWORDS = ("scatter", "voxel", "pillar_vfe", "pfn_layers", "canvas", "anchor")


def fixed_shape_skip_reason(module_name: str) -> str:
    low = module_name.lower()
    for key in FIXED_SHAPE_KEYWORDS:
        if key in low:
            return f"fixed_shape_structural_skip:{key}"
    return ""
