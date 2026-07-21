"""Precision contracts for CoBEVT Attention matrix multiplications.

The types in this module deliberately separate tensor storage, GEMM operands,
multiplication, accumulation, and output.  They are evidence records, not
requests to TensorRT.
"""

from __future__ import annotations

from dataclasses import dataclass


_PRECISIONS = {"FP32", "FP16", "BF16", "INT8", "INT32", "unknown"}
_EVIDENCE_LEVELS = {"A", "B", "C"}


@dataclass(frozen=True)
class ComputeContract:
    storage_precision: str
    left_operand_precision: str
    right_operand_precision: str
    multiplication_precision: str
    accumulator_precision: str
    output_precision: str

    def __post_init__(self) -> None:
        for field_name in (
            "storage_precision",
            "left_operand_precision",
            "right_operand_precision",
            "multiplication_precision",
            "accumulator_precision",
            "output_precision",
        ):
            value = str(getattr(self, field_name))
            if value not in _PRECISIONS:
                raise ValueError(f"unsupported_precision:{field_name}:{value}")
        if self.left_operand_precision != self.right_operand_precision:
            raise ValueError("matrix_operands_must_share_precision")
        if self.multiplication_precision == "INT8" and self.accumulator_precision != "INT32":
            raise ValueError("int8_accumulator_must_be_int32")

    @property
    def phenotype(self) -> str:
        if self.accumulator_precision == "unknown":
            return "UNKNOWN_ACCUM"
        key = (self.left_operand_precision, self.accumulator_precision)
        names = {
            ("FP32", "FP32"): "F32A32",
            ("FP16", "FP32"): "F16A32",
            ("FP16", "FP16"): "F16A16",
            ("BF16", "FP32"): "BF16A32",
            ("INT8", "INT32"): "I8A32I",
        }
        return names.get(key, "UNKNOWN_ACCUM")


@dataclass(frozen=True)
class CastMaterialization:
    phenotype: str
    materialized: bool
    is_f16a32: bool
    evidence_level: str
    reason: str


def classify_cast_materialization(
    *,
    source_precision: str,
    gemm_operand_precision: str,
    cast_execution_layer_present: bool,
    cast_tensor_written_to_memory: bool,
    cast_absorbed_by_gemm: bool,
    kernel_accumulator_precision: str = "unknown",
    kernel_contract_evidence_level: str = "C",
) -> CastMaterialization:
    level = str(kernel_contract_evidence_level)
    if level not in _EVIDENCE_LEVELS:
        raise ValueError(f"invalid_evidence_level:{level}")
    materialized = bool(cast_execution_layer_present or cast_tensor_written_to_memory)
    if source_precision == "FP16" and gemm_operand_precision == "FP32" and materialized:
        return CastMaterialization(
            "F32A32_after_materialized_cast",
            True,
            False,
            level,
            "fp16_source_was_materialized_as_fp32_before_gemm",
        )
    if (
        source_precision == "FP16"
        and gemm_operand_precision == "FP32"
        and cast_absorbed_by_gemm
        and kernel_accumulator_precision == "FP32"
        and level == "A"
    ):
        return CastMaterialization(
            "F16A32",
            False,
            True,
            level,
            "direct_kernel_contract_proves_fp16_multiplicands_fp32_accumulator",
        )
    return CastMaterialization(
        "UNKNOWN_ACCUM",
        materialized,
        False,
        level,
        "insufficient_direct_kernel_evidence",
    )


@dataclass(frozen=True)
class Int8QKRealization:
    native_int8: bool
    phenotype: str
    evidence_level: str
    reason: str


def classify_int8_qk(
    *,
    storage_precision: str,
    qk_input_precision: str,
    dequantize_before_qk: bool,
    kernel_operand_precision: str,
    kernel_accumulator_precision: str,
    evidence_level: str,
) -> Int8QKRealization:
    if evidence_level not in _EVIDENCE_LEVELS:
        raise ValueError(f"invalid_evidence_level:{evidence_level}")
    if dequantize_before_qk or qk_input_precision == "FP32":
        return Int8QKRealization(
            False,
            "F32A32_after_dequantize",
            evidence_level,
            "int8_storage_dequantized_before_fp32_qk",
        )
    native = (
        storage_precision == "INT8"
        and qk_input_precision == "INT8"
        and kernel_operand_precision == "INT8"
        and kernel_accumulator_precision == "INT32"
        and evidence_level == "A"
    )
    return Int8QKRealization(
        native,
        "I8A32I" if native else "UNKNOWN_ACCUM",
        evidence_level,
        "direct_int8_kernel_contract" if native else "native_int8_not_proven",
    )


def classify_fixed500(delta_map: float, *, finite: bool, skipped_frames: int) -> str:
    if not finite or int(skipped_frames) != 0 or float(delta_map) < -0.010:
        return "UNSAFE_FIXED500"
    if float(delta_map) >= -0.003:
        return "SAFE_FIXED500"
    return "BORDERLINE_FIXED500"


@dataclass(frozen=True)
class SearchEvidence:
    evidence_level: str
    accumulator_precision: str
    requested_realized_match: bool
    fixed500_delta_map: float
    formal_latency_gain: bool
    precision_conflicts: int
    skipped_frames: int
    finite: bool = True


@dataclass(frozen=True)
class SearchEligibility:
    eligible: bool
    reasons: tuple[str, ...]


def search_eligibility(evidence: SearchEvidence) -> SearchEligibility:
    reasons: list[str] = []
    if evidence.evidence_level != "A" or evidence.accumulator_precision == "unknown":
        reasons.append("accumulator_evidence_below_level_a")
    if not evidence.requested_realized_match:
        reasons.append("requested_realized_mismatch")
    if classify_fixed500(
        evidence.fixed500_delta_map,
        finite=evidence.finite,
        skipped_frames=evidence.skipped_frames,
    ) != "SAFE_FIXED500":
        reasons.append("fixed500_not_safe")
    if not evidence.formal_latency_gain:
        reasons.append("no_formal_latency_gain")
    if int(evidence.precision_conflicts) != 0:
        reasons.append("precision_conflict")
    if int(evidence.skipped_frames) != 0:
        reasons.append("evaluation_skips")
    return SearchEligibility(not reasons, tuple(reasons))


__all__ = [
    "CastMaterialization",
    "ComputeContract",
    "Int8QKRealization",
    "SearchEligibility",
    "SearchEvidence",
    "classify_cast_materialization",
    "classify_fixed500",
    "classify_int8_qk",
    "search_eligibility",
]
