"""Physical-pruned signal-maxK ONNX export."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from ..config import CanonicalNamingConfig, OnnxExportConfig
from ..exceptions import PhysicalStructureMismatchError
from ..types import OnnxExportResult
from .signal_maxk import export_signal_maxk_onnx
from .validation import validate_onnx_against_physical_snapshot


def export_pruned_signal_maxk_onnx(
    model: Any,
    example_inputs: Sequence[Any] | Mapping[str, Any],
    output_path: str | Path,
    physical_snapshot: Any,
    *,
    config: OnnxExportConfig | None = None,
    naming_config: CanonicalNamingConfig | None = None,
    report_path: str | Path | None = None,
) -> OnnxExportResult:
    """Export a physical model and require snapshot/initializer agreement."""

    result = export_signal_maxk_onnx(model, example_inputs, output_path, config=config, naming_config=naming_config)
    if result.origin_map is None:
        raise PhysicalStructureMismatchError("export produced no weighted origin map")
    validation = validate_onnx_against_physical_snapshot(output_path, physical_snapshot, result.origin_map)
    result.validation = validation
    if not validation.passed:
        raise PhysicalStructureMismatchError(f"pruned ONNX does not match physical snapshot: {validation.to_dict()}")
    if report_path is not None:
        from ..artifacts.io import atomic_write_json

        atomic_write_json(report_path, result.to_dict())
    return result
