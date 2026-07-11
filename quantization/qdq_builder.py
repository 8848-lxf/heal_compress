"""Compatibility facade for the formal FP16/INT8 Q/DQ inserter."""

from __future__ import annotations

import warnings
from typing import Any, Mapping

from .config import QDQConfig
from .precision.qdq_inserter import insert_explicit_qdq
from .types import CanonicalPrecisionMappingResult, QDQInsertionResult


class QDQDeploymentBuilder:
    """Deprecated object wrapper around :func:`insert_explicit_qdq`."""

    def __init__(self, config: QDQConfig | None = None) -> None:
        self.config = config or QDQConfig()

    def insert_qdq(
        self,
        input_onnx: str,
        output_onnx: str,
        mapping: CanonicalPrecisionMappingResult,
        scales: Mapping[str, Any],
    ) -> QDQInsertionResult:
        warnings.warn(
            "QDQDeploymentBuilder is deprecated; use quantization.api.insert_explicit_qdq",
            DeprecationWarning,
            stacklevel=2,
        )
        return insert_explicit_qdq(input_onnx, output_onnx, mapping, scales=scales, config=self.config)
