"""Deployment-closed V2X-ViT AV profiles and derived merge precision.

The AV operator is activation-by-activation, so its precision contract names
both operands explicitly.  Window-family merges are derived state: they never
become chromosome loci and are instead deterministically resolved from their
inputs, scale compatibility, and the audited TensorRT capability.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Sequence


PRECISION_ORDER = {"INT8": 0, "FP16": 1, "FP32": 2}


@dataclass(frozen=True)
class AVPrecisionContract:
    profile: str
    softmax_compute: str
    probability_precision: str
    value_precision: str
    compute_precision: str
    accumulation_precision: str
    output_precision: str
    probability_bits: int
    value_bits: int
    requires_calibration: bool

    def __post_init__(self) -> None:
        expected = {
            "AV32": ("FP32", "FP32", "FP32", 32, 32, False),
            "AV16": ("FP16", "FP16", "FP16", 16, 16, False),
            "AV8": ("INT8", "INT8", "INT8", 8, 8, True),
        }
        if self.profile not in expected:
            raise ValueError(f"v2xvit_av_profile_unknown:{self.profile}")
        probability, value, compute, p_bits, v_bits, calibration = expected[
            self.profile
        ]
        if self.softmax_compute != "FP32":
            raise ValueError("v2xvit_av_softmax_compute_must_be_fp32")
        if (
            self.probability_precision,
            self.value_precision,
            self.compute_precision,
            self.probability_bits,
            self.value_bits,
            self.requires_calibration,
        ) != (probability, value, compute, p_bits, v_bits, calibration):
            raise ValueError(f"v2xvit_av_profile_contract_mismatch:{self.profile}")

    @property
    def bops_factor(self) -> int:
        return int(self.probability_bits) * int(self.value_bits)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


AV_CONTRACTS = {
    "AV32": AVPrecisionContract(
        "AV32", "FP32", "FP32", "FP32", "FP32", "FP32", "FP32", 32, 32, False
    ),
    "AV16": AVPrecisionContract(
        "AV16", "FP32", "FP16", "FP16", "FP16", "TRT_REALIZED", "FP16", 16, 16, False
    ),
    "AV8": AVPrecisionContract(
        "AV8", "FP32", "INT8", "INT8", "INT8", "TRT_REALIZED", "FP16", 8, 8, True
    ),
}


def av_contract(profile: str) -> AVPrecisionContract:
    try:
        return AV_CONTRACTS[str(profile).upper()]
    except KeyError as exc:
        raise ValueError(f"v2xvit_av_profile_unknown:{profile}") from exc


def av_bops(mac: int | float, profile: str) -> float:
    value = float(mac) * av_contract(profile).bops_factor
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"v2xvit_av_bops_invalid:{mac}:{profile}")
    return value


def _normalize_precision(value: str) -> str:
    precision = str(value).upper()
    if precision not in PRECISION_ORDER:
        raise ValueError(f"v2xvit_merge_precision_invalid:{value}")
    return precision


def _scale_compatible(scales: Sequence[float | None], tolerance: float) -> bool:
    if not scales or any(value is None for value in scales):
        return False
    values = [float(value) for value in scales if value is not None]
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        return False
    reference = values[0]
    return all(abs(value - reference) <= tolerance * max(reference, value) for value in values[1:])


def derive_merge_precision(
    op_type: str,
    input_precisions: Sequence[str],
    input_scales: Sequence[float | None] = (),
    downstream_contract: str | None = None,
    trt_capability: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Derive one merge state without introducing a mutable precision gene."""

    op = str(op_type).upper()
    inputs = tuple(_normalize_precision(value) for value in input_precisions)
    if not inputs:
        raise ValueError("v2xvit_merge_inputs_empty")
    capability = dict(trt_capability or {})
    shape_ops = {"RESHAPE", "PERMUTE", "TRANSPOSE", "VIEW", "GATHER", "SCATTER"}
    requested = max(inputs, key=PRECISION_ORDER.__getitem__)
    reason = "highest_input_precision_join"
    requantize = False
    scale_compatible = False

    if op in shape_ops:
        if len(set(inputs)) != 1:
            raise ValueError(f"v2xvit_shape_merge_mixed_inputs:{inputs}")
        requested = inputs[0]
        reason = "pure_shape_inherits_input"
    elif op in {"ADD", "CONCAT"}:
        all_int8 = all(value == "INT8" for value in inputs)
        if all_int8:
            tolerance = float(capability.get("scale_relative_tolerance", 1.0e-6))
            scale_compatible = _scale_compatible(input_scales, tolerance)
            explicit_requant = bool(capability.get("explicit_requantization", False))
            int8_supported = bool(capability.get(f"{op.lower()}_int8", False))
            if int8_supported and (scale_compatible or explicit_requant):
                requested = "INT8"
                requantize = bool(not scale_compatible and explicit_requant)
                reason = "all_int8_scale_closed_and_trt_supported"
            else:
                requested = "FP16"
                reason = "all_int8_merge_promoted_fp16"
        elif "FP32" in inputs:
            requested = "FP32"
        else:
            requested = "FP16"
    else:
        raise ValueError(f"v2xvit_merge_op_unsupported:{op_type}")

    if downstream_contract is not None:
        downstream = _normalize_precision(downstream_contract)
        if PRECISION_ORDER[downstream] > PRECISION_ORDER[requested]:
            requested = downstream
            reason = f"promoted_for_downstream_{downstream.lower()}"

    return {
        "op_type": op,
        "input_precisions": list(inputs),
        "input_scales": [None if value is None else float(value) for value in input_scales],
        "derived_precision": requested,
        "scale_compatible": scale_compatible,
        "explicit_requantization": requantize,
        "independent_chromosome_gene": False,
        "reason": reason,
    }


__all__ = [
    "AV_CONTRACTS",
    "AVPrecisionContract",
    "av_bops",
    "av_contract",
    "derive_merge_precision",
]
