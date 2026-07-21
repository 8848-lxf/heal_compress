"""Requested-versus-realized TensorRT precision validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from ..types import CanonicalPrecisionMappingResult, PrecisionRealizationResult
from .layer_info import has_canonical_identity, is_weighted_compute_layer, layer_metadata, layer_name, load_layer_info, precision_name


def validate_precision_realization(
    layer_info: str | Path | Sequence[Mapping[str, Any]] | Mapping[str, Any],
    precision_mapping: CanonicalPrecisionMappingResult,
    *,
    expected_precision_overrides: Mapping[str, str] | None = None,
) -> PrecisionRealizationResult:
    """Compare exact canonical requests with realized engine compute precision."""

    rows = load_layer_info(layer_info)
    overrides = {
        str(module_path): str(precision).strip().lower()
        for module_path, precision in (expected_precision_overrides or {}).items()
    }
    invalid_overrides = {
        module_path: precision
        for module_path, precision in overrides.items()
        if precision not in {"fp32", "fp16", "bf16", "fp8", "int8"}
    }
    if invalid_overrides:
        raise ValueError(f"precision_realization_override_invalid:{invalid_overrides}")
    hidden_casts = sum("cast" in layer_metadata(row).lower() for row in rows)
    reformats = sum("reformat" in layer_metadata(row).lower() for row in rows)
    boundaries = sum(any(token in layer_metadata(row).lower() for token in ("quantize", "dequantize", "requant")) for row in rows)
    mismatches: list[dict[str, Any]] = []
    realized_int8 = 0
    realized_fp16 = 0
    realized_bf16 = 0
    realized_fp8 = 0
    unresolved = 0
    for entry in precision_mapping.entries:
        expected_precision = overrides.get(
            str(entry.module_path), str(entry.realized_request_precision)
        )
        matches = [row for row in rows if has_canonical_identity(row, entry.canonical_node_name)]
        compute_matches = [row for row in matches if is_weighted_compute_layer(row)]
        row = compute_matches[0] if len(compute_matches) == 1 else None
        if row is None:
            unresolved += 1
            mismatches.append(
                {
                    "canonical_node_name": entry.canonical_node_name,
                    "module_path": entry.module_path,
                    "requested_precision": expected_precision,
                    "expected_precision": expected_precision,
                    "realized_precision": "",
                    "reason": "canonical_layer_not_found",
                }
            )
            continue
        realized = precision_name(row)
        if realized == "int8":
            realized_int8 += 1
        elif realized == "fp16":
            realized_fp16 += 1
        elif realized == "bf16":
            realized_bf16 += 1
        elif realized == "fp8":
            realized_fp8 += 1
        if not realized or realized != expected_precision:
            mismatches.append(
                {
                    "canonical_node_name": entry.canonical_node_name,
                    "trt_layer_name": layer_name(row),
                    "module_path": entry.module_path,
                    "requested_precision": expected_precision,
                    "expected_precision": expected_precision,
                    "realized_precision": realized,
                    "fallback_reason": entry.fallback_reason,
                    "reason": "requested_precision_not_realized",
                }
            )
    return PrecisionRealizationResult(
        passed=not mismatches,
        requested_int8_count=sum(
            overrides.get(str(entry.module_path), str(entry.requested_precision))
            == "int8"
            for entry in precision_mapping.entries
        ),
        realized_int8_count=realized_int8,
        realized_fp16_count=realized_fp16,
        realized_bf16_count=realized_bf16,
        realized_fp8_count=realized_fp8,
        mismatches=mismatches,
        hidden_cast_count=int(hidden_casts),
        reformat_count=int(reformats),
        boundary_count=int(boundaries),
        unresolved_layer_count=unresolved,
    )
