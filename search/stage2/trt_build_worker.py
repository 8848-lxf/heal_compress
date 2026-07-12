"""ModelOpt subprocess worker for TensorRT build and validation."""

from __future__ import annotations

import argparse
import json
import os
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
        result: dict[str, Any] = {"status": "ok" if build.success else "engine_build_failed", "build": build.to_dict()}
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
