"""ModelOpt subprocess worker for TensorRT build and validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any


def _write_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _mapping_from_dict(payload: dict[str, Any]) -> Any:
    try:
        from heal_compress.quantization.types import CanonicalPrecisionEntry, CanonicalPrecisionMappingResult
    except ImportError:
        from quantization.types import CanonicalPrecisionEntry, CanonicalPrecisionMappingResult
    data = dict(payload)
    data["entries"] = [CanonicalPrecisionEntry(**dict(row)) for row in data.get("entries", [])]
    return CanonicalPrecisionMappingResult(**data)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    return parser.parse_args(argv)


def _file_sha256(path: str | Path | None) -> str:
    source = Path(path) if path else Path()
    if not source.is_file():
        return ""
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tool_version(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=30,
    )
    return {
        "command": command,
        "returncode": int(completed.returncode),
        "output": (completed.stdout or "").strip(),
    }


def _build_environment_manifest(request: dict[str, Any], build: Any) -> dict[str, Any]:
    """Fail closed when a modelopt build leaks to the system toolchain."""

    prefix = Path(os.environ.get("CONDA_PREFIX", "")).resolve()
    paths = {
        "python": str(Path(sys.executable).resolve()),
        "nvcc": str(Path(shutil.which("nvcc") or "").resolve()),
        "gcc": str(Path(shutil.which("gcc") or "").resolve()),
        "g++": str(Path(shutil.which("g++") or "").resolve()),
    }
    outside = {
        name: path
        for name, path in paths.items()
        if not path or (prefix.parent != prefix and prefix not in Path(path).parents)
    }
    if prefix.name != "modelopt" or outside:
        raise RuntimeError(
            f"modelopt_build_toolchain_not_isolated:prefix={prefix}:outside={outside}"
        )
    gpu: dict[str, Any]
    try:
        import pycuda.driver as cuda

        cuda.init()
        device = cuda.Device(0)
        major, minor = device.compute_capability()
        gpu = {
            "visible_device_index": 0,
            "name": device.name(),
            "compute_capability": f"{major}.{minor}",
        }
    except Exception as exc:  # noqa: BLE001
        gpu = {"probe_error": f"{type(exc).__name__}: {exc}"}
    try:
        import tensorrt as trt

        tensorrt_version = str(trt.__version__)
    except Exception as exc:  # noqa: BLE001
        tensorrt_version = f"unavailable:{type(exc).__name__}:{exc}"
    build_payload = build.to_dict() if hasattr(build, "to_dict") else dict(build)
    plugin_path = request.get("build_config", {}).get("plugin_path")
    manifest = {
        "schema_version": "modelopt-tensorrt-build-environment-v2",
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV", ""),
        "conda_prefix": str(prefix),
        "python_path": paths["python"],
        "nvcc_path": paths["nvcc"],
        "gcc_path": paths["gcc"],
        "gxx_path": paths["g++"],
        "nvcc_version": _tool_version([paths["nvcc"], "--version"]),
        "gcc_version": _tool_version([paths["gcc"], "--version"]),
        "gxx_version": _tool_version([paths["g++"], "--version"]),
        "CUDA_HOME": os.environ.get("CUDA_HOME", ""),
        "CC": os.environ.get("CC", ""),
        "CXX": os.environ.get("CXX", ""),
        "CUDACXX": os.environ.get("CUDACXX", ""),
        "TensorRT_root": str(request.get("tensorrt_root", "")),
        "TensorRT_version": tensorrt_version,
        "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""),
        "CMAKE_PREFIX_PATH": os.environ.get("CMAKE_PREFIX_PATH", ""),
        "gpu": gpu,
        "plugin_path": str(plugin_path or ""),
        "plugin_sha256": _file_sha256(plugin_path),
        "qdq_onnx_sha256": _file_sha256(request.get("qdq_onnx")),
        "builder_config": request.get("build_config", {}),
        "builder_command": build_payload.get("command", {}).get("command", []),
        "builder_command_hash": build_payload.get("command", {}).get("command_hash", ""),
        "system_toolchain_used": False,
        "source_repo_root": str(request.get("repo_root", "")),
        "python_package_root": str(request.get("python_package_root", "")),
        "loaded_source_modules": dict(request.get("loaded_source_modules", {})),
    }
    manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    return manifest


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    output_path = Path(request["output_path"])
    try:
        repo = Path(request.get("repo_root", Path.cwd())).resolve()
        python_package_root = Path(str(request["python_package_root"])).resolve()
        package_alias = python_package_root / "heal_compress"
        if not package_alias.is_symlink() or package_alias.resolve() != repo:
            raise RuntimeError(
                f"trt_worker_package_alias_invalid:{package_alias}:{repo}"
            )
        uniad = repo.parent
        # Insert in reverse-priority order because ``insert(0, ...)`` prepends.
        # The output-local canonical package alias must win over the formal
        # worktree that also exists below ``uniad``.
        for path in (
            str(uniad),
            str(uniad / "HEAL"),
            str(repo),
            str(python_package_root),
        ):
            if path not in sys.path:
                sys.path.insert(0, path)
        ld = request.get("ld_library_path")
        if ld:
            os.environ["LD_LIBRARY_PATH"] = str(ld) + ":" + os.environ.get("LD_LIBRARY_PATH", "")
        try:
            from heal_compress.quantization.api import build_trt_engine, validate_engine_structure, validate_precision_realization
            from heal_compress.quantization.config import TensorRTBuildConfig, TensorRTValidationConfig
        except ImportError:
            from quantization.api import build_trt_engine, validate_engine_structure, validate_precision_realization
            from quantization.config import TensorRTBuildConfig, TensorRTValidationConfig
        loaded_source_modules = {
            name: str(Path(sys.modules[value.__module__].__file__).resolve())
            for name, value in {
                "build_trt_engine": build_trt_engine,
                "validate_engine_structure": validate_engine_structure,
                "validate_precision_realization": validate_precision_realization,
            }.items()
        }
        outside = {
            name: path
            for name, path in loaded_source_modules.items()
            if repo != Path(path) and repo not in Path(path).parents
        }
        if outside:
            raise RuntimeError(
                f"trt_worker_source_worktree_leak:{outside}:expected_root={repo}"
            )
        request["loaded_source_modules"] = loaded_source_modules
        mapping = _mapping_from_dict(request["precision_mapping"])
        build_config = TensorRTBuildConfig.from_dict(request["build_config"])
        physical_snapshot = request.get("physical_snapshot")
        layer_info_path = Path(request["layer_info_path"])
        build = build_trt_engine(
            request["qdq_onnx"],
            request["engine_path"],
            mapping,
            config=build_config,
            layer_info_path=layer_info_path,
            log_path=request["log_path"],
            raise_on_failure=False,
        )
        environment_manifest = _build_environment_manifest(request, build)
        _write_json(output_path.parent / "engine_build_environment_manifest.json", environment_manifest)
        result: dict[str, Any] = {
            "status": "ok" if build.success else "engine_build_failed",
            "build": build.to_dict(),
            "build_environment_manifest": environment_manifest,
        }
        if build.success:
            structure = validate_engine_structure(
                layer_info_path,
                mapping,
                physical_snapshot=physical_snapshot,
                config=TensorRTValidationConfig(),
            )
            precision = validate_precision_realization(layer_info_path, mapping)
            result["engine_structure_validation"] = structure.to_dict()
            result["precision_realization_validation"] = precision.to_dict()
            if not structure.passed:
                result["status"] = "engine_structure_validation_failed"
            elif not precision.passed:
                result["status"] = "precision_realization_validation_failed"
    except Exception as exc:  # noqa: BLE001
        result = {
            "status": "engine_build_failed",
            "failure_reason": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
    _write_json(output_path, result)
    print(json.dumps({"status": result.get("status"), "output": str(output_path)}, sort_keys=True))
    return 0 if result.get("status") == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
