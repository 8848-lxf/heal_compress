from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


STRATEGY = "dynamic_agent_single_engine_maxK"
FIXED_K = 24064
SUPPORTED_AGENT_COUNTS = (1, 2)
INPUT_NAMES = [
    "voxel_features",
    "voxel_coords",
    "voxel_num_points",
    "pairwise_t_matrix",
    "valid_voxel_mask",
]


def single_engine_input_names() -> list[str]:
    return list(INPUT_NAMES)


def single_engine_dynamic_axes(output_names: list[str]) -> dict[str, dict[int, str]]:
    axes = {
        "pairwise_t_matrix": {1: "num_agents", 2: "num_agents"},
    }
    for name in output_names:
        axes[name] = {0: "batch"}
    return axes


def _fixedk_suffix(fixed_k: int | None = None) -> str | None:
    if fixed_k is None or int(fixed_k) == int(FIXED_K):
        return None
    return f"fixedK{int(fixed_k)}"


def onnx_path(dirs: dict[str, Path], fixed_k: int | None = None) -> Path:
    suffix = _fixedk_suffix(fixed_k)
    root = dirs["output_root"] / "artifacts" / "onnx"
    if suffix:
        root = root / suffix
    return root / STRATEGY / "lidar_pyramid_dynamic_agent_single_engine_maxK.onnx"


def engine_dir(dirs: dict[str, Path], precision: str, calibration_frames: int | None = None, fixed_k: int | None = None) -> Path:
    suffix = _fixedk_suffix(fixed_k)
    root = dirs["output_root"] / "artifacts" / "engines"
    if suffix:
        root = root / suffix
    root = root / STRATEGY
    if precision == "int8":
        if calibration_frames is None:
            raise ValueError("calibration_frames is required for INT8 single-engine paths")
        return root / f"int8_train_calib{int(calibration_frames)}"
    return root / precision


def engine_path(dirs: dict[str, Path], precision: str, calibration_frames: int | None = None, fixed_k: int | None = None) -> Path:
    suffix = precision if precision != "int8" else f"int8_train_calib{int(calibration_frames)}"
    return engine_dir(dirs, precision, calibration_frames, fixed_k=fixed_k) / f"lidar_pyramid_dynamic_agent_single_engine_maxK_{suffix}.engine"


def layerinfo_path(dirs: dict[str, Path], precision: str, calibration_frames: int | None = None, fixed_k: int | None = None) -> Path:
    suffix = precision if precision != "int8" else f"int8_train_calib{int(calibration_frames)}"
    return engine_dir(dirs, precision, calibration_frames, fixed_k=fixed_k) / f"layerinfo_dynamic_agent_single_engine_maxK_{suffix}.json"


def calibration_npz_dir(dirs: dict[str, Path], calibration_frames: int, fixed_k: int | None = None) -> Path:
    if _fixedk_suffix(fixed_k):
        return dirs["output_root"] / "artifacts" / "calibration" / f"train_calib_single_engine_maxK{int(fixed_k)}_{int(calibration_frames)}"
    return dirs["output_root"] / "artifacts" / "calibration" / f"dynamic_single_engine_maxK_train_calib{int(calibration_frames)}"


def calibration_cache_path(dirs: dict[str, Path], calibration_frames: int, fixed_k: int | None = None) -> Path:
    if _fixedk_suffix(fixed_k):
        return dirs["output_root"] / "artifacts" / "calibration" / f"lidar_pyramid_dynamic_agent_single_engine_maxK_fixedK{int(fixed_k)}_int8_train_calib{int(calibration_frames)}.cache"
    return dirs["output_root"] / "artifacts" / "calibration" / f"lidar_pyramid_dynamic_agent_single_engine_maxK_int8_train_calib{int(calibration_frames)}.cache"


def _mode_or_median(values: list[int]) -> int:
    if not values:
        return 1
    counts = Counter(int(v) for v in values)
    max_count = max(counts.values())
    modes = sorted(value for value, count in counts.items() if count == max_count)
    if len(modes) == 1:
        return int(modes[0])
    return int(np.median(np.asarray(values, dtype=np.int64)))


def profile_from_observed_shapes(observed_shapes: list[dict[str, list[int]]], *, fixed_k: int = FIXED_K) -> dict[str, dict[str, list[int]]]:
    if not observed_shapes:
        raise ValueError("observed_shapes must not be empty")
    agent_counts = []
    for shapes in observed_shapes:
        pairwise = shapes.get("pairwise_t_matrix")
        if not pairwise or len(pairwise) != 5:
            raise ValueError(f"missing or invalid pairwise_t_matrix shape: {pairwise}")
        if int(pairwise[1]) != int(pairwise[2]):
            raise ValueError(f"pairwise_t_matrix N dimensions differ: {pairwise}")
        agent_counts.append(int(pairwise[1]))
    min_n = int(min(agent_counts))
    opt_n = int(_mode_or_median(agent_counts))
    max_n = int(max(agent_counts))
    if not (min_n <= opt_n <= max_n):
        opt_n = max_n
    return {
        "voxel_features": {"min": [fixed_k, 32, 4], "opt": [fixed_k, 32, 4], "max": [fixed_k, 32, 4]},
        "voxel_coords": {"min": [fixed_k, 4], "opt": [fixed_k, 4], "max": [fixed_k, 4]},
        "voxel_num_points": {"min": [fixed_k], "opt": [fixed_k], "max": [fixed_k]},
        "pairwise_t_matrix": {"min": [1, min_n, min_n, 4, 4], "opt": [1, opt_n, opt_n, 4, 4], "max": [1, max_n, max_n, 4, 4]},
        "valid_voxel_mask": {"min": [fixed_k], "opt": [fixed_k], "max": [fixed_k]},
    }


def load_observed_shapes_from_npz(npz_dir: Path) -> list[dict[str, list[int]]]:
    shapes: list[dict[str, list[int]]] = []
    for path in sorted(Path(npz_dir).glob("*.npz")):
        with np.load(path) as sample:
            shapes.append({name: list(sample[name].shape) for name in INPUT_NAMES if name in sample})
    return shapes


def pad_pairwise_to_agent_count(pairwise: np.ndarray, target_n: int) -> np.ndarray:
    target_n = int(target_n)
    source = np.asarray(pairwise)
    if source.ndim != 5:
        raise ValueError(f"pairwise_t_matrix must be rank 5, got shape {source.shape}")
    n = int(source.shape[1])
    if n == target_n:
        return np.ascontiguousarray(source)
    if n > target_n:
        raise ValueError(f"cannot pad pairwise_t_matrix with N={n} to smaller target N={target_n}")
    out = np.zeros((1, target_n, target_n, 4, 4), dtype=source.dtype)
    out[:, :n, :n, :, :] = source
    eye = np.eye(4, dtype=source.dtype)
    for idx in range(n, target_n):
        out[0, idx, idx, :, :] = eye
    return np.ascontiguousarray(out)


def numeric_summary(values: list[int] | list[float]) -> dict[str, Any]:
    if not values:
        return {"min": None, "p50": None, "p90": None, "p95": None, "p99": None, "mean": None, "max": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "min": float(arr.min()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "mean": float(arr.mean()),
        "max": float(arr.max()),
    }
