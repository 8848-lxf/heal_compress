"""Stage-2 TensorRT build and validation orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable


class TensorRTStage:
    def __init__(
        self,
        *,
        build_fn: Callable[..., Any] | None = None,
        structure_validate_fn: Callable[..., Any] | None = None,
        precision_validate_fn: Callable[..., Any] | None = None,
    ) -> None:
        if any(value is None for value in (build_fn, structure_validate_fn, precision_validate_fn)):
            try:
                from quantization.api import build_trt_engine, validate_engine_structure, validate_precision_realization
            except ImportError:
                from heal_compress.quantization.api import build_trt_engine, validate_engine_structure, validate_precision_realization
            build_fn = build_fn or build_trt_engine
            structure_validate_fn = structure_validate_fn or validate_engine_structure
            precision_validate_fn = precision_validate_fn or validate_precision_realization
        self.build_fn = build_fn
        self.structure_validate_fn = structure_validate_fn
        self.precision_validate_fn = precision_validate_fn

    def run(
        self,
        *,
        qdq_onnx: str | Path,
        engine_path: str | Path,
        precision_mapping: Any,
        build_config: Any,
        physical_snapshot: Any,
        output_dir: str | Path,
    ) -> dict[str, Any]:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        layer_info_path = destination / "layer_info.json"
        log_path = destination / "engine_build.log"
        build = self.build_fn(
            qdq_onnx,
            engine_path,
            precision_mapping,
            config=build_config,
            layer_info_path=layer_info_path,
            log_path=log_path,
            raise_on_failure=False,
        )
        if not getattr(build, "success", False):
            return {"stage2_status": "engine_build_failed", "build": build}
        structure = self.structure_validate_fn(layer_info_path, precision_mapping, physical_snapshot=physical_snapshot)
        if not getattr(structure, "passed", False):
            return {"stage2_status": "engine_structure_validation_failed", "build": build, "structure_validation": structure}
        precision = self.precision_validate_fn(layer_info_path, precision_mapping)
        if not getattr(precision, "passed", False):
            return {"stage2_status": "precision_realization_validation_failed", "build": build, "structure_validation": structure, "precision_validation": precision}
        return {"stage2_status": "ok", "build": build, "structure_validation": structure, "precision_validation": precision}
