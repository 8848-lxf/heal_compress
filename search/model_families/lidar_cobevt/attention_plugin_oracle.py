"""Contracts for the explicitly typed Attention mixed-accumulation plugin oracle."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class MixedAccumPluginContract:
    family: str
    input_precision: str
    multiplication_precision: str
    accumulator_precision: str
    output_precision: str
    scale: float

    def __post_init__(self) -> None:
        if self.family not in {"QK", "AV"}:
            raise ValueError(f"unsupported_mixed_accum_family:{self.family}")
        if self.input_precision != "FP16" or self.multiplication_precision != "FP16":
            raise ValueError("mixed_accum_plugin_requires_fp16_operands")
        if self.accumulator_precision != "FP32":
            raise ValueError("mixed_accum_plugin_requires_fp32_accumulator")
        if self.output_precision not in {"FP16", "FP32"}:
            raise ValueError(f"unsupported_mixed_accum_output:{self.output_precision}")

    @property
    def phenotype(self) -> str:
        return "F16A32"

    @property
    def evidence_level(self) -> str:
        return "A"

    @property
    def implementation(self) -> str:
        return "plugin_oracle"

    @property
    def native_tensorrt(self) -> bool:
        return False

    def to_bytes(self) -> bytes:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode("ascii")

    @classmethod
    def from_bytes(cls, payload: bytes) -> "MixedAccumPluginContract":
        return cls(**json.loads(payload.decode("ascii")))


def mixed_accum_plugin_contract(
    family: str, *, output_precision: str, scale: float
) -> MixedAccumPluginContract:
    return MixedAccumPluginContract(
        str(family), "FP16", "FP16", "FP32", str(output_precision), float(scale)
    )


def mixed_accum_output_shape(
    family: str, left: tuple[int, ...], right: tuple[int, ...]
) -> tuple[int, ...]:
    if len(left) != 4 or len(right) != 4:
        raise ValueError("mixed_accum_requires_rank4")
    if left[:2] != right[:2]:
        raise ValueError("mixed_accum_batch_shape_mismatch")
    if family == "QK":
        if left[-1] != right[-1]:
            raise ValueError("mixed_accum_reduction_shape_mismatch")
        return (*left[:2], left[-2], right[-2])
    if family == "AV":
        if left[-1] != right[-2]:
            raise ValueError("mixed_accum_reduction_shape_mismatch")
        return (*left[:2], left[-2], right[-1])
    raise ValueError(f"unsupported_mixed_accum_family:{family}")


__all__ = [
    "MixedAccumPluginContract",
    "mixed_accum_output_shape",
    "mixed_accum_plugin_contract",
]
