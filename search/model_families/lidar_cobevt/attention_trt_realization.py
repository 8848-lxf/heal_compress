"""Native TensorRT micro-engine specifications and evidence classification."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterable


@dataclass(frozen=True)
class NativeMicroSpec:
    spec_id: str
    family: str
    requested_phenotype: str
    input_precision: str
    output_precision: str
    explicit_cast_to_fp32: bool = False
    native_int8_quantize: bool = False
    optional: bool = False


def native_micro_specs() -> tuple[NativeMicroSpec, ...]:
    rows: list[NativeMicroSpec] = []
    for prefix, family in (("M", "QK"), ("N", "AV")):
        rows.extend(
            (
                NativeMicroSpec(f"{prefix}0_{family}_F32A32", family, "F32A32", "FP32", "FP32"),
                NativeMicroSpec(
                    f"{prefix}1_{family}_F16_DEFAULT",
                    family,
                    "UNKNOWN_ACCUM",
                    "FP16",
                    "FP16",
                ),
                NativeMicroSpec(
                    f"{prefix}2_{family}_F16A32_REQUEST",
                    family,
                    "F16A32",
                    "FP16",
                    "FP32",
                    explicit_cast_to_fp32=True,
                ),
                NativeMicroSpec(
                    f"{prefix}3_{family}_F16A16_REQUEST",
                    family,
                    "F16A16",
                    "FP16",
                    "FP16",
                ),
                NativeMicroSpec(
                    f"{prefix}4_{family}_I8A32I",
                    family,
                    "I8A32I",
                    "INT8",
                    "FP32",
                    native_int8_quantize=True,
                ),
                NativeMicroSpec(
                    f"{prefix}5_{family}_BF16A32",
                    family,
                    "BF16A32",
                    "BF16",
                    "FP32",
                    optional=True,
                ),
            )
        )
    return tuple(rows)


def _has_execution_layer(execution_layers: Iterable[dict[str, Any]], *tokens: str) -> bool:
    lowered = tuple(token.lower() for token in tokens)
    for layer in execution_layers:
        joined = " ".join(str(value) for value in layer.values()).lower()
        if any(token in joined for token in lowered):
            return True
    return False


def infer_accumulator_from_tactics(
    execution_layers: Iterable[dict[str, Any]],
) -> str | None:
    """Return an accumulator only when the kernel name encodes it explicitly."""

    mapping = {"f32": "FP32", "f16": "FP16", "i32": "INT32"}
    values: set[str] = set()
    for layer in execution_layers:
        tactic = str(layer.get("TacticName") or layer.get("tactic") or "").lower()
        match = re.search(r"(?:f32|f16|bf16|i8)(?:f32|f16|bf16|i8)_+(f32|f16|i32)_", tactic)
        if match:
            values.add(mapping[match.group(1)])
    return next(iter(values)) if len(values) == 1 else None


def classify_native_realization(
    *,
    requested_phenotype: str,
    build_success: bool,
    input_precisions: tuple[str, str],
    matmul_input_precisions: tuple[str, str],
    output_precision: str,
    execution_layers: Iterable[dict[str, Any]],
    direct_accumulator_metadata: str | None,
) -> dict[str, Any]:
    """Classify what a native TensorRT graph proves, without guessing tactics."""

    layers = tuple(execution_layers)
    if not build_success:
        return {
            "realized_phenotype": "unsupported_build",
            "realized_accumulator_precision": "unknown",
            "evidence_level": "C",
            "requested_realized_match": False,
            "cast_materialized": False,
            "native_int8": False,
        }

    cast_materialized = (
        input_precisions == ("FP16", "FP16")
        and matmul_input_precisions == ("FP32", "FP32")
        and _has_execution_layer(layers, "cast")
    )
    dequantized = input_precisions == ("INT8", "INT8") and matmul_input_precisions == (
        "FP32",
        "FP32",
    )

    if dequantized:
        return {
            "realized_phenotype": "F32A32_after_dequantize",
            "realized_accumulator_precision": (
                direct_accumulator_metadata or "unknown"
            ),
            "evidence_level": "A" if direct_accumulator_metadata else "C",
            "requested_realized_match": False,
            "cast_materialized": False,
            "native_int8": False,
        }

    if cast_materialized:
        return {
            "realized_phenotype": "F32A32_after_materialized_cast",
            "realized_accumulator_precision": (
                direct_accumulator_metadata or "unknown"
            ),
            "evidence_level": "A" if direct_accumulator_metadata else "C",
            "requested_realized_match": False,
            "cast_materialized": True,
            "native_int8": False,
        }

    accumulator = direct_accumulator_metadata or "unknown"
    phenotype_map = {
        (("FP32", "FP32"), "FP32"): "F32A32",
        (("FP16", "FP16"), "FP32"): "F16A32",
        (("FP16", "FP16"), "FP16"): "F16A16",
        (("BF16", "BF16"), "FP32"): "BF16A32",
        (("INT8", "INT8"), "INT32"): "I8A32I",
    }
    realized = phenotype_map.get((matmul_input_precisions, accumulator), "UNKNOWN_ACCUM")
    native_int8 = realized == "I8A32I"
    evidence_level = "A" if direct_accumulator_metadata else "C"
    return {
        "realized_phenotype": realized,
        "realized_accumulator_precision": accumulator,
        "realized_output_precision": output_precision,
        "evidence_level": evidence_level,
        "requested_realized_match": requested_phenotype == realized,
        "cast_materialized": False,
        "native_int8": native_int8,
    }


__all__ = [
    "NativeMicroSpec",
    "classify_native_realization",
    "infer_accumulator_from_tactics",
    "native_micro_specs",
]
