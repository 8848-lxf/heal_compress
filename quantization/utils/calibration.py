from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .paths import DEFAULT_FIXED_K

INPUT_NAMES = [
    "voxel_features",
    "voxel_coords",
    "voxel_num_points",
    "pairwise_t_matrix",
    "valid_voxel_mask",
]


def load_observed_shapes(npz_dir: str | Path) -> list[dict[str, list[int]]]:
    shapes: list[dict[str, list[int]]] = []
    for path in sorted(Path(npz_dir).glob("*.npz")):
        with np.load(path) as sample:
            shapes.append({name: list(sample[name].shape) for name in INPUT_NAMES if name in sample})
    return shapes


def profile_from_observed_shapes(
    observed_shapes: list[dict[str, list[int]]],
    *,
    fixed_k: int = DEFAULT_FIXED_K,
) -> dict[str, dict[str, list[int]]]:
    if not observed_shapes:
        observed_shapes = [
            {"pairwise_t_matrix": [1, 1, 1, 4, 4]},
            {"pairwise_t_matrix": [1, 2, 2, 4, 4]},
        ]
    agent_counts = []
    for shape in observed_shapes:
        pairwise = shape.get("pairwise_t_matrix")
        if pairwise and len(pairwise) == 5:
            agent_counts.append(int(pairwise[1]))
    if not agent_counts:
        agent_counts = [1, 2]
    min_n, max_n = min(agent_counts), max(agent_counts)
    opt_n = 2 if max_n >= 2 else max_n
    return {
        "voxel_features": {"min": [fixed_k, 32, 4], "opt": [fixed_k, 32, 4], "max": [fixed_k, 32, 4]},
        "voxel_coords": {"min": [fixed_k, 4], "opt": [fixed_k, 4], "max": [fixed_k, 4]},
        "voxel_num_points": {"min": [fixed_k], "opt": [fixed_k], "max": [fixed_k]},
        "pairwise_t_matrix": {"min": [1, min_n, min_n, 4, 4], "opt": [1, opt_n, opt_n, 4, 4], "max": [1, max_n, max_n, 4, 4]},
        "valid_voxel_mask": {"min": [fixed_k], "opt": [fixed_k], "max": [fixed_k]},
    }


def calibration_manifest_summary(path: str | Path) -> dict[str, Any]:
    manifest = Path(path) / "manifest.json"
    if not manifest.is_file():
        return {"manifest_exists": False, "path": str(manifest)}
    import json

    data = json.loads(manifest.read_text(encoding="utf-8"))
    return {
        "manifest_exists": True,
        "path": str(manifest),
        "strategy": data.get("strategy"),
        "fixed_K": data.get("fixed_K"),
        "num_samples": data.get("num_samples"),
        "calibration_split": data.get("calibration_split"),
    }
