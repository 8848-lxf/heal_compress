"""Deployment legality policies for canonical precision requests."""

from __future__ import annotations

from ..config import QDQConfig
from ..types import CanonicalMappingEntry


def legalize_int8_request(entry: CanonicalMappingEntry, *, config: QDQConfig | None = None) -> tuple[str, str]:
    """Return realized request precision and an explicit fallback reason."""

    policy = config or QDQConfig()
    if entry.onnx_op_type not in {"Conv", "ConvTranspose", "Gemm", "MatMul"}:
        return "fp16", f"unsupported_int8_op:{entry.onnx_op_type}"
    if int(entry.groups or 1) > 1:
        values = [
            value
            for value in (entry.input_channels_per_group, entry.output_channels_per_group, entry.channels_per_group)
            if value is not None
        ]
        allowed = set(policy.grouped_conv_int8_allowed_channels_per_group)
        if not values or any(int(value) not in allowed for value in values):
            return "fp16", "grouped_channels_per_group_not_allowed"
    return "int8", ""
