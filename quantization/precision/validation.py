"""Q/DQ graph validation against physical structure truth."""

from __future__ import annotations

from pathlib import Path

from ..types import OnnxOriginMapResult, OnnxValidationResult
from ..export.validation import validate_onnx_against_physical_snapshot


def validate_qdq_against_physical_snapshot(
    qdq_onnx: str | Path,
    physical_snapshot: object,
    origin_map: OnnxOriginMapResult,
) -> OnnxValidationResult:
    """Require every canonical Q/DQ weight path to reach the physical weight."""

    return validate_onnx_against_physical_snapshot(qdq_onnx, physical_snapshot, origin_map)
