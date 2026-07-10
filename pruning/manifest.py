"""Manifest helpers for formal pruning artifacts."""

from __future__ import annotations

from typing import Any

from heal_compress.pruning.config import PruningConfig


def base_pruning_manifest(config: PruningConfig, **extra: Any) -> dict[str, Any]:
    manifest = {
        "target_pruning_mode": config.target_pruning_mode,
        "target_pruning_ratio": config.target_pruning_ratio,
        "round_to": config.round_to,
        "max_ch_sparsity": config.max_ch_sparsity,
        "stage1_min_per_group": config.stage1_min_per_group,
        "stage1_max_ch_sparsity": config.stage1_max_ch_sparsity,
        "importance": config.importance,
        "selector": config.selector,
        "protected_fpn_output": config.protect_fpn_output,
        "protected_head_output": config.protect_head_output,
        "additional_output_protection": not config.no_extra_output_protection,
        "fixed_shape_structural_skip": config.fixed_shape_structural_skip,
    }
    manifest.update(extra)
    return manifest
