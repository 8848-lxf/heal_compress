"""Fail-closed BF16/FP8 capability and candidate definitions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class LowPrecisionCapability:
    precision: str
    hardware_supported: bool
    tensorrt_api_supported: bool
    modelopt_supported: bool
    explicit_graph_supported: bool
    realized_auditable: bool
    reasons: tuple[str, ...]

    @property
    def deployable(self) -> bool:
        return all(
            (
                self.hardware_supported,
                self.tensorrt_api_supported,
                self.modelopt_supported,
                self.explicit_graph_supported,
                self.realized_auditable,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "deployable": self.deployable}


def capability_from_probe(precision: str, probe: Mapping[str, Any]) -> LowPrecisionCapability:
    fields = (
        "hardware_supported",
        "tensorrt_api_supported",
        "modelopt_supported",
        "explicit_graph_supported",
        "realized_auditable",
    )
    reasons = tuple(str(value) for value in probe.get("reasons", ()))
    return LowPrecisionCapability(
        precision=str(precision),
        **{field: bool(probe.get(field, False)) for field in fields},
        reasons=reasons,
    )


def require_deployable(capability: LowPrecisionCapability) -> None:
    if not capability.deployable:
        raise RuntimeError(
            f"{capability.precision.lower()}_unsupported_fail_closed:"
            + ";".join(capability.reasons or ("capability_incomplete",))
        )


FP8_PROFILE_ROLES = {
    "H5_QK_PROJECTION_FP8_DQ_FP32_QK": ("q_projection", "k_projection"),
    "H6_QKV_PROJECTION_FP8_DQ_FP32_QK": ("q_projection", "k_projection", "v_projection"),
    "H7_FFN_FP8": ("ffn1", "ffn2"),
    "H8_QK_PLUS_FFN_FP8": ("q_projection", "k_projection", "ffn1", "ffn2"),
    "H9_ALL_LINEAR_FP8_PROTECTED_CORE": (
        "q_projection",
        "k_projection",
        "v_projection",
        "output_projection",
        "ffn1",
        "ffn2",
    ),
}


__all__ = [
    "FP8_PROFILE_ROLES",
    "LowPrecisionCapability",
    "capability_from_probe",
    "require_deployable",
]
