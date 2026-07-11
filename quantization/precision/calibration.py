"""Calibration metadata validation (collection is caller/integration owned)."""

from __future__ import annotations

import math
from typing import Any, Mapping

from ..config import CalibrationConfig
from ..exceptions import QDQInsertionError


def validate_calibration_scales(scales: Mapping[str, Any], *, config: CalibrationConfig | None = None) -> dict[str, Any]:
    """Validate positive finite scales and record calibration provenance."""

    policy = config or CalibrationConfig()
    if policy.split != "train":
        raise QDQInsertionError("formal INT8 calibration split must be train")
    invalid = []
    for name, raw in scales.items():
        value = raw.get("scale") if isinstance(raw, Mapping) else raw
        try:
            valid = math.isfinite(float(value)) and float(value) > 0.0
        except (TypeError, ValueError):
            valid = False
        if not valid:
            invalid.append(str(name))
    if invalid:
        raise QDQInsertionError(f"invalid calibration scales: {invalid}")
    return {
        "calibration_schema_version": policy.schema_version,
        "split": policy.split,
        "frame_count": int(policy.frame_count),
        "scale_count": len(scales),
        "activation_granularity": policy.activation_granularity,
        "weight_granularity": policy.weight_granularity,
    }
