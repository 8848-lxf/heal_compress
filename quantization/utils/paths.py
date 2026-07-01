from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Any

DEFAULT_STRATEGY = "single_engine_maxK"
LEGACY_SINGLE_ENGINE_STRATEGY = "dynamic_agent_single_engine_maxK"
DEFAULT_FIXED_K = 29696
DEFAULT_PRECISION = "fp16"
DEFAULT_CALIBRATION_SPLIT = "train"
DEFAULT_EVAL_SPLIT = "val"
DEFAULT_OUTPUT_ROOT = Path("tests/quant_deploy/outputs/lidar_pyramid_agent_export_strategy_compare")
DEFAULT_CONFIG = Path(
    "/home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/"
    "LiDAROnly/lidar_pyramid/config.yaml"
)
DEFAULT_CHECKPOINT = Path(
    "/home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/"
    "LiDAROnly/lidar_pyramid/net_epoch_bestval_at17.pth"
)
DEFAULT_TRT_ROOT = Path("/home/lixingfeng/UniAD_examine/HEAL/prune_model/TensorRT-10.9_x86_cu118")
DEFAULT_PLUGIN = Path("quantization/plugins/pointpillar_scatter_trt/build/libpointpillar_scatter_trt.so")
SUPPORTED_PRECISIONS = ("fp32", "fp16", "int8")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def project_path(path: str | os.PathLike[str] | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else repo_root() / value


def add_repo_parent_to_sys_path() -> None:
    parent = repo_root().parent
    if str(parent) not in sys.path:
        sys.path.insert(0, str(parent))


def add_quant_deploy_tests_to_sys_path() -> None:
    path = repo_root() / "tests" / "quant_deploy"
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def load_quant_deploy_module(name: str) -> Any:
    add_quant_deploy_tests_to_sys_path()
    return importlib.import_module(name)


def validate_precision(precision: str) -> str:
    value = str(precision).lower()
    if value not in SUPPORTED_PRECISIONS:
        raise ValueError(f"unsupported precision '{precision}', expected one of {SUPPORTED_PRECISIONS}")
    return value


def normalize_strategy(strategy: str) -> str:
    value = str(strategy)
    if value in {DEFAULT_STRATEGY, LEGACY_SINGLE_ENGINE_STRATEGY, "dynamic_single_engine_maxK"}:
        return DEFAULT_STRATEGY
    if value in {"dynamic_bucket", "padded_agent_static"}:
        return value
    raise ValueError(f"unsupported strategy '{strategy}'")


def legacy_strategy(strategy: str) -> str:
    normalized = normalize_strategy(strategy)
    if normalized == DEFAULT_STRATEGY:
        return LEGACY_SINGLE_ENGINE_STRATEGY
    return normalized


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def assert_not_tests_outputs(path: str | os.PathLike[str] | Path) -> None:
    resolved = project_path(path)
    forbidden = repo_root() / "tests" / "outputs"
    if resolved == forbidden or _is_within(resolved, forbidden):
        raise ValueError("formal quantization outputs must not be written to tests/outputs")


def ensure_dir(path: str | os.PathLike[str] | Path) -> Path:
    out = project_path(path)
    assert_not_tests_outputs(out)
    out.mkdir(parents=True, exist_ok=True)
    return out


def infer_output_root(path: str | os.PathLike[str] | Path) -> Path:
    """Infer the quant-deploy run root from a direct artifact/report path.

    Examples:
      root/artifacts/onnx/fixedK29696/single_engine_maxK -> root
      root/evaluation/formal_tag -> root
      root -> root
    """
    value = project_path(path)
    parts = list(value.parts)
    for marker in ("artifacts", "evaluation", "benchmark", "summary", "debug", "logs", "configs"):
        if marker in parts:
            index = parts.index(marker)
            if index > 0:
                return Path(*parts[:index])
    return value


def infer_output_tag(path: str | os.PathLike[str] | Path, default: str) -> str:
    value = project_path(path)
    parts = list(value.parts)
    if "evaluation" in parts:
        index = parts.index("evaluation")
        if len(parts) > index + 1:
            return parts[index + 1]
    return default


def formal_onnx_name(fixed_k: int = DEFAULT_FIXED_K) -> str:
    return f"lidar_pyramid_dynamic_agent_single_engine_maxK_fixedK{int(fixed_k)}.onnx"


def formal_engine_name(precision: str, fixed_k: int = DEFAULT_FIXED_K, calibration_frames: int | None = None) -> str:
    precision = validate_precision(precision)
    if precision == "int8":
        suffix = f"int8_train_calib{int(calibration_frames or 200)}"
    else:
        suffix = precision
    return f"lidar_pyramid_dynamic_agent_single_engine_maxK_fixedK{int(fixed_k)}_{suffix}.engine"


def default_output_root_str() -> str:
    return str(DEFAULT_OUTPUT_ROOT)
