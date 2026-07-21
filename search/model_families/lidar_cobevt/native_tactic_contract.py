"""Contracts for native TensorRT tactic evidence and cache provenance."""

from __future__ import annotations

from dataclasses import dataclass


_EVIDENCE_LEVELS = {
    "A",
    "B",
    "C",
    "LEVEL_A_DIRECT",
    "LEVEL_B_STRONG",
    "LEVEL_C_UNKNOWN",
}


@dataclass(frozen=True)
class ComputePhenotype:
    storage_precision: str
    operand_precision: str
    accumulator_precision: str
    output_precision: str
    phenotype_override: str | None = None

    @property
    def name(self) -> str:
        if self.phenotype_override:
            return self.phenotype_override
        names = {
            ("FP32", "FP32", "FP32"): "F32A32O32",
            ("FP16", "FP32", "FP16"): "F16A32O16",
            ("FP16", "FP32", "FP32"): "F16A32O32",
            ("FP16", "FP16", "FP16"): "F16A16O16",
            ("BF16", "FP32", "BF16"): "BF16A32OBF16",
            ("INT8", "INT32", "INT32"): "I8A32I",
        }
        return names.get(
            (self.operand_precision, self.accumulator_precision, self.output_precision),
            "UNKNOWN_ACCUM",
        )

    @property
    def is_native_f16a32(self) -> bool:
        return self.name in {"F16A32O16", "F16A32O32"}


def classify_native_phenotype(
    *,
    storage_precision: str,
    operand_precision: str,
    output_precision: str,
    accumulator_precision: str,
    materialized_cast: bool,
    evidence_level: str,
) -> ComputePhenotype:
    if evidence_level not in _EVIDENCE_LEVELS:
        raise ValueError(f"invalid_evidence_level:{evidence_level}")
    if materialized_cast:
        return ComputePhenotype(
            storage_precision,
            operand_precision,
            accumulator_precision,
            output_precision,
            phenotype_override="F32A32_AFTER_MATERIALIZED_CAST",
        )
    return ComputePhenotype(
        storage_precision,
        operand_precision,
        accumulator_precision,
        output_precision,
    )


def classify_evidence(
    *,
    direct_accumulator: str | None,
    tensor_core_fp16: bool,
    no_materialized_cast: bool,
    oracle_matches_f16a32: bool,
    oracle_separates_f16a16: bool,
) -> str:
    if direct_accumulator in {"FP32", "FP16", "INT32"}:
        return "LEVEL_A_DIRECT"
    if (
        tensor_core_fp16
        and no_materialized_cast
        and oracle_matches_f16a32
        and oracle_separates_f16a16
    ):
        return "LEVEL_B_STRONG"
    return "LEVEL_C_UNKNOWN"


@dataclass(frozen=True)
class CacheBinding:
    graph_signature: str
    gpu_arch: str
    tensorrt_version: str
    cuda_version: str
    shape_key: str
    cache_sha256: str


def validate_cache_binding(expected: CacheBinding, actual: CacheBinding) -> bool:
    if expected != actual:
        raise ValueError("timing_cache_binding_mismatch")
    return True


@dataclass(frozen=True)
class ShapeProfileResult:
    role: str
    status: str
    reason: str = ""


def shape_profile_gate(rows: list[dict[str, object]], *, role: str) -> ShapeProfileResult:
    if len(rows) != 6:
        return ShapeProfileResult(role, "UNKNOWN", "requires_exactly_six_shapes")
    levels = {str(row.get("evidence_level", "LEVEL_C_UNKNOWN")) for row in rows}
    stable = all(bool(row.get("pinning_stable", False)) for row in rows)
    if not stable:
        return ShapeProfileResult(role, "UNKNOWN", "pinning_not_stable")
    if levels == {"LEVEL_A_DIRECT"}:
        return ShapeProfileResult(role, "NATIVE_F16A32_PINNABLE")
    if levels <= {"LEVEL_A_DIRECT", "LEVEL_B_STRONG"} and "LEVEL_B_STRONG" in levels:
        return ShapeProfileResult(role, "NATIVE_PROBABLE_F16A32")
    return ShapeProfileResult(role, "UNKNOWN", "accumulator_evidence_incomplete")


def classify_fixed500(
    delta_map: float, *, evaluated: int, skipped: int, finite: bool
) -> str:
    if evaluated != 500 or skipped != 0 or not finite:
        return "UNSAFE"
    if float(delta_map) >= -0.003:
        return "SAFE"
    if float(delta_map) >= -0.010:
        return "BORDERLINE"
    return "UNSAFE"


@dataclass(frozen=True)
class SearchEligibility:
    allowed: bool
    reasons: tuple[str, ...]


def profile_search_eligibility(
    *,
    implementation: str,
    shape_status: str,
    full_engine_preserved: bool,
    fixed500_safety: str,
    formal_latency_gain: bool,
) -> SearchEligibility:
    reasons: list[str] = []
    if implementation != "native_tensorrt":
        reasons.append("implementation_not_native_tensorrt")
    if shape_status != "NATIVE_F16A32_PINNABLE":
        reasons.append("shape_profile_not_level_a_pinnable")
    if not full_engine_preserved:
        reasons.append("full_engine_pinning_not_preserved")
    if fixed500_safety != "SAFE":
        reasons.append("fixed500_not_safe")
    if not formal_latency_gain:
        reasons.append("no_formal_latency_gain")
    return SearchEligibility(not reasons, tuple(reasons))


__all__ = [
    "CacheBinding",
    "ComputePhenotype",
    "SearchEligibility",
    "ShapeProfileResult",
    "classify_evidence",
    "classify_fixed500",
    "classify_native_phenotype",
    "profile_search_eligibility",
    "shape_profile_gate",
    "validate_cache_binding",
]
