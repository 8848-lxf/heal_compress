#!/usr/bin/env python3
"""TensorRT EngineInspector and binding probe for one immutable engine."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def plain(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)


def probe(engine_path: Path, plugin_path: Path) -> dict[str, Any]:
    import tensorrt as trt

    expected = Path("/home/lixingfeng/miniconda3/envs/modelopt")
    prefix = Path(os.environ.get("CONDA_PREFIX", ""))
    environment = {
        "CONDA_DEFAULT_ENV": os.environ.get("CONDA_DEFAULT_ENV", ""),
        "CONDA_PREFIX": str(prefix),
        "CUDA_HOME": os.environ.get("CUDA_HOME", ""),
        "CC": os.environ.get("CC", ""),
        "CXX": os.environ.get("CXX", ""),
        "CUDACXX": os.environ.get("CUDACXX", ""),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    if environment["CONDA_DEFAULT_ENV"] != "modelopt" or prefix.resolve() != expected.resolve():
        raise RuntimeError(f"engine_probe_requires_explicit_modelopt:{environment}")
    if Path(environment["CUDA_HOME"]).resolve() != expected.resolve():
        raise RuntimeError(f"engine_probe_CUDA_HOME_not_modelopt:{environment}")
    for variable, binary in (("CC", "gcc"), ("CXX", "g++"), ("CUDACXX", "nvcc")):
        if Path(environment[variable]).resolve() != (expected / "bin" / binary).resolve():
            raise RuntimeError(f"engine_probe_{variable}_not_modelopt:{environment}")
    plugin = ctypes.CDLL(str(plugin_path.resolve()), mode=ctypes.RTLD_GLOBAL)
    logger = trt.Logger(trt.Logger.INFO)
    initialized = bool(trt.init_libnvinfer_plugins(logger, ""))
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    if engine is None:
        raise RuntimeError(f"engine_deserialize_failed:{engine_path}")
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError(f"execution_context_create_failed:{engine_path}")
    bindings = []
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        row: dict[str, Any] = {
            "index": index,
            "name": name,
            "mode": str(engine.get_tensor_mode(name)),
            "shape": list(engine.get_tensor_shape(name)),
            "dtype": str(engine.get_tensor_dtype(name)),
            "location": str(engine.get_tensor_location(name)),
            "format": str(engine.get_tensor_format(name)),
            "format_description": str(engine.get_tensor_format_desc(name)),
        }
        try:
            minimum, optimum, maximum = engine.get_tensor_profile_shape(name, 0)
            row["profile"] = {"min": list(minimum), "opt": list(optimum), "max": list(maximum)}
        except Exception as exc:  # noqa: BLE001
            row["profile_error"] = str(exc)
        bindings.append(row)
    inspector = engine.create_engine_inspector()
    inspector_json_text = inspector.get_engine_information(trt.LayerInformationFormat.JSON)
    try:
        inspector_json = json.loads(inspector_json_text)
    except json.JSONDecodeError:
        inspector_json = {"raw": inspector_json_text}
    creators = []
    registry = trt.get_plugin_registry()
    for creator in getattr(registry, "all_creators", []) or []:
        name = str(getattr(creator, "name", ""))
        if "pointpillar" in name.lower() or "scatter" in name.lower():
            creators.append(
                {
                    "name": name,
                    "version": str(getattr(creator, "plugin_version", "")),
                    "namespace": str(getattr(creator, "plugin_namespace", "")),
                }
            )
    return {
        "engine_path": str(engine_path),
        "engine_size": engine_path.stat().st_size,
        "engine_sha256": sha256_file(engine_path),
        "plugin_path": str(plugin_path),
        "plugin_sha256": sha256_file(plugin_path),
        "plugin_handle_loaded": bool(plugin),
        "trt_plugins_initialized": initialized,
        "pointpillar_plugin_creators": creators,
        "TensorRT_version": trt.__version__,
        **environment,
        "num_io_tensors": engine.num_io_tensors,
        "num_optimization_profiles": engine.num_optimization_profiles,
        "device_memory_size": int(engine.device_memory_size_v2),
        "bindings": bindings,
        "context_created": True,
        "inspector": plain(inspector_json),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--plugin", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = probe(args.engine.resolve(), args.plugin.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"engine": result["engine_path"], "bindings": result["num_io_tensors"], "status": "ok"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
