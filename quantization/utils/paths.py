"""Path helpers retained for compatibility with earlier command entry points."""

from __future__ import annotations

import os
from pathlib import Path


DEFAULT_STRATEGY = "single_engine_maxK"
LEGACY_SINGLE_ENGINE_STRATEGY = "dynamic_agent_single_engine_maxK"
DEFAULT_FIXED_K = 29696
DEFAULT_PRECISION = "fp16"
DEFAULT_CALIBRATION_SPLIT = "train"
DEFAULT_EVAL_SPLIT = "val"
DEFAULT_OUTPUT_ROOT = Path("outputs/quantization")
DEFAULT_CONFIG: Path | None = None
DEFAULT_CHECKPOINT: Path | None = None
DEFAULT_TRT_ROOT: Path | None = None
DEFAULT_PLUGIN: Path | None = None
SUPPORTED_PRECISIONS = ("fp16", "int8")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def project_path(path: str | os.PathLike[str] | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else repo_root() / value


def validate_precision(precision: str) -> str:
    value = str(precision).lower()
    if value not in SUPPORTED_PRECISIONS:
        raise ValueError(f"unsupported precision '{precision}', expected one of {SUPPORTED_PRECISIONS}")
    return value


def normalize_strategy(strategy: str) -> str:
    value = str(strategy)
    if value in {DEFAULT_STRATEGY, LEGACY_SINGLE_ENGINE_STRATEGY, "dynamic_single_engine_maxK"}:
        return DEFAULT_STRATEGY
    raise ValueError(f"unsupported formal strategy '{strategy}'")


def legacy_strategy(strategy: str) -> str:
    normalize_strategy(strategy)
    return LEGACY_SINGLE_ENGINE_STRATEGY


def ensure_dir(path: str | os.PathLike[str] | Path) -> Path:
    output = project_path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output


def infer_output_root(path: str | os.PathLike[str] | Path) -> Path:
    value = project_path(path)
    parts = list(value.parts)
    for marker in ("artifacts", "evaluation", "benchmark", "summary", "debug", "logs", "configs"):
        if marker in parts and parts.index(marker) > 0:
            return Path(*parts[: parts.index(marker)])
    return value


def infer_output_tag(path: str | os.PathLike[str] | Path, default: str) -> str:
    value = project_path(path)
    parts = list(value.parts)
    if "evaluation" in parts and len(parts) > parts.index("evaluation") + 1:
        return parts[parts.index("evaluation") + 1]
    return default


def formal_onnx_name(fixed_k: int = DEFAULT_FIXED_K) -> str:
    return f"lidar_pyramid_dynamic_agent_single_engine_maxK_fixedK{int(fixed_k)}.onnx"


def formal_engine_name(precision: str, fixed_k: int = DEFAULT_FIXED_K, calibration_frames: int | None = None) -> str:
    value = validate_precision(precision)
    suffix = f"int8_train_calib{int(calibration_frames or 200)}" if value == "int8" else value
    return f"lidar_pyramid_dynamic_agent_single_engine_maxK_fixedK{int(fixed_k)}_{suffix}.engine"


def default_output_root_str() -> str:
    return str(DEFAULT_OUTPUT_ROOT)
