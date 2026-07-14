"""ModelOpt subprocess worker for TensorRT build and validation."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Callable


def _builder_contract_audit(build_config: Any, command: list[str]) -> dict[str, Any]:
    """Fail closed when a production command contains weak precision controls."""

    forbidden_prefixes = (
        "--fp16",
        "--int8",
        "--precisionConstraints",
        "--layerPrecisions",
        "--layerOutputTypes",
    )
    forbidden = [
        str(value)
        for value in command
        if str(value).startswith(forbidden_prefixes)
    ]
    strongly_typed = bool(getattr(build_config, "strongly_typed", False))
    production_mode = bool(getattr(build_config, "production_mode", False))
    issues = []
    if production_mode and not strongly_typed:
        issues.append("production_requires_strongly_typed")
    if strongly_typed and "--stronglyTyped" not in command:
        issues.append("strongly_typed_command_flag_missing")
    if strongly_typed and forbidden:
        issues.append("strongly_typed_command_contains_forbidden_precision_options")
    return {
        "passed": not issues,
        "strongly_typed": strongly_typed,
        "production_mode": production_mode,
        "plugin_boundary_dtype": str(
            getattr(build_config, "plugin_boundary_dtype", "")
        ).upper(),
        "forbidden_options": forbidden,
        "issues": issues,
    }


def _runtime_provenance(trt: Any, torch: Any) -> dict[str, str]:
    capability = torch.cuda.get_device_capability(0)
    return {
        "tensorrt_version": str(trt.__version__),
        "cuda_version": str(torch.version.cuda),
        "torch_version": str(torch.__version__),
        "gpu_architecture": ".".join(str(value) for value in capability),
        "gpu_name": str(torch.cuda.get_device_name(0)),
    }


def _engine_deserialization_audit(
    trt: Any,
    engine_path: str | Path,
    *,
    plugin_path: str | Path | None = None,
    plugin_loader: Callable[[str | Path], Any] | None = None,
) -> dict[str, Any]:
    """Load the production plugin before deserializing the written engine."""

    path = Path(engine_path)
    if not path.is_file():
        raise RuntimeError(f"engine_file_missing_after_build:{path}")
    resolved_plugin = Path(plugin_path) if plugin_path else None
    plugin_loaded = False
    if resolved_plugin is not None:
        if not resolved_plugin.is_file():
            raise RuntimeError(f"engine_plugin_missing:{resolved_plugin}")
        loader = plugin_loader or (
            lambda item: ctypes.CDLL(str(item), mode=ctypes.RTLD_GLOBAL)
        )
        loader(resolved_plugin)
        plugin_loaded = True
    payload = path.read_bytes()
    logger = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(payload)
    if engine is None:
        raise RuntimeError("engine_deserialize_returned_none")
    return {
        "passed": True,
        "engine_path": str(path),
        "engine_bytes": len(payload),
        "num_io_tensors": int(engine.num_io_tensors),
        "plugin_path": str(resolved_plugin) if resolved_plugin is not None else "",
        "plugin_loaded_before_deserialize": plugin_loaded,
    }


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


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    output_path = Path(request["output_path"])
    try:
        repo = Path(request.get("repo_root", Path.cwd())).resolve()
        uniad = repo.parent
        for path in (str(uniad), str(repo), str(uniad / "HEAL")):
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
        import tensorrt as trt
        import torch

        runtime_provenance = _runtime_provenance(trt, torch)
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
        builder_contract = _builder_contract_audit(
            build_config,
            list(build.command.command),
        )
        result: dict[str, Any] = {
            "status": (
                "ok"
                if build.success and builder_contract["passed"]
                else "builder_contract_validation_failed"
                if build.success
                else "engine_build_failed"
            ),
            "build": build.to_dict(),
            "builder_contract_audit": builder_contract,
            "strongly_typed": bool(builder_contract["strongly_typed"]),
            "runtime_provenance": runtime_provenance,
        }
        if build.success and builder_contract["passed"]:
            try:
                result["engine_deserialization_audit"] = _engine_deserialization_audit(
                    trt,
                    request["engine_path"],
                    plugin_path=build_config.plugin_path,
                )
            except Exception as exc:  # noqa: BLE001
                result["status"] = "engine_deserialize_failure"
                result["failure_reason"] = f"{type(exc).__name__}: {exc}"
            if result["status"] == "ok":
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
