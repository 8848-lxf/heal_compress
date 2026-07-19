"""Strict TensorRT provenance parsing for CoBEVT Attention capability rows."""

from __future__ import annotations

from typing import Any, Iterable, Mapping


_ROLE_TOKENS: dict[str, tuple[str, ...]] = {
    "q_projection": ("q_proj", "q_projection"),
    "k_projection": ("k_proj", "k_projection"),
    "v_projection": ("v_proj", "v_projection"),
    "qk_matmul": ("qk_matmul", "qk/matmul"),
    "softmax": ("softmax",),
    "av_matmul": ("av_matmul", "av/matmul"),
    "out_projection": ("out_proj", "out_projection"),
}


def _searchable(layer: Mapping[str, Any]) -> str:
    return " ".join(
        str(layer.get(key, ""))
        for key in ("Name", "LayerType", "TacticName", "Metadata")
    ).lower()


def _layer_roles(layer: Mapping[str, Any]) -> set[str]:
    text = _searchable(layer)
    return {
        role
        for role, tokens in _ROLE_TOKENS.items()
        if any(token in text for token in tokens)
    }


def _normalize_precision(value: str) -> str:
    lowered = str(value).lower()
    if "int8" in lowered:
        return "INT8"
    if "half" in lowered or "fp16" in lowered or "float16" in lowered:
        return "FP16"
    if "float" in lowered or "fp32" in lowered or "float32" in lowered:
        return "FP32"
    if "bf16" in lowered:
        return "BF16"
    return "unknown"


def _layer_precisions(layer: Mapping[str, Any]) -> set[str]:
    tensors = [*layer.get("Inputs", ()), *layer.get("Outputs", ())]
    values = {
        _normalize_precision(str(tensor.get("Format/Datatype", "")))
        for tensor in tensors
    }
    return {value for value in values if value != "unknown"}


def _role_precision(layers: list[Mapping[str, Any]], role: str) -> str:
    values = {
        precision
        for layer in layers
        if role in _layer_roles(layer)
        for precision in _layer_precisions(layer)
    }
    if not values:
        return "unknown"
    if len(values) == 1:
        return next(iter(values))
    return "MIXED[" + ",".join(sorted(values)) + "]"


def _explicit_accumulator_precision(
    layers: list[Mapping[str, Any]], role: str
) -> str:
    values: set[str] = set()
    for layer in layers:
        if role not in _layer_roles(layer):
            continue
        for key in (
            "AccumulatorPrecision",
            "Accumulator DataType",
            "AccumulatorDatatype",
        ):
            if key in layer:
                values.add(_normalize_precision(str(layer[key])))
    values.discard("unknown")
    return next(iter(values)) if len(values) == 1 else "unknown"


def inspect_attention_layers(
    layers: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = [dict(layer) for layer in layers]
    execution_indices = {
        index
        for index, row in enumerate(rows)
        if _layer_roles(row) & {"qk_matmul", "softmax", "av_matmul"}
    }
    complete_candidates = []
    for index, row in enumerate(rows):
        roles = _layer_roles(row)
        text = _searchable(row)
        explicit_mha = any(
            token in text
            for token in (
                "_gemm_mha",
                "fused_mha",
                "multiheadattention",
                "multi_head_attention",
            )
        )
        if (
            {"qk_matmul", "softmax", "av_matmul"} <= roles
            and explicit_mha
        ):
            complete_candidates.append(index)
    complete_fused = (
        len(complete_candidates) == 1
        and execution_indices == {complete_candidates[0]}
    )
    role_sets = [_layer_roles(row) for row in rows]
    if complete_fused:
        fusion_kind = "complete_fused_mha"
    elif complete_candidates:
        fusion_kind = "partial_mha_with_primitives"
    elif any({"softmax", "av_matmul"} <= roles for roles in role_sets):
        fusion_kind = "softmax_local_fusion"
    elif any(
        "cast" in _searchable(row)
        and bool(_layer_roles(row) & {"qk_matmul", "av_matmul"})
        for row in rows
    ):
        fusion_kind = "matmul_cast_fusion"
    elif any(
        len(roles & {"q_projection", "k_projection", "v_projection"}) >= 2
        for roles in role_sets
    ):
        fusion_kind = "projection_fusion"
    else:
        fusion_kind = "primitive"
    realized = {
        role: _role_precision(rows, role)
        for role in _ROLE_TOKENS
    }
    result: dict[str, Any] = {
        "attention_execution_layer_count": len(execution_indices),
        "av_accumulator_precision": _explicit_accumulator_precision(
            rows, "av_matmul"
        ),
        "cast_count": sum("cast" in _searchable(row) for row in rows),
        "fused_mha_detected": bool(complete_fused),
        "fusion_kind": fusion_kind,
        "layer_count": len(rows),
        "plugin_used": any(
            "plugin" in str(row.get("LayerType", "")).lower()
            or "plugin" in str(row.get("Name", "")).lower()
            for row in rows
        ),
        "qk_accumulator_precision": _explicit_accumulator_precision(
            rows, "qk_matmul"
        ),
        "realized_precision": realized,
        "reformat_count": sum(
            "reformat" in str(row.get("Name", "")).lower() for row in rows
        ),
    }
    for role, value in realized.items():
        result[f"realized_{role.removesuffix('_matmul')}_precision"] = value
    return result


def audit_requested_realized_precision(
    requested: Mapping[str, str], realized: Mapping[str, str]
) -> dict[str, Any]:
    fallback_roles = sorted(
        role
        for role, requested_value in requested.items()
        if role in _ROLE_TOKENS
        and str(realized.get(role, "unknown")).upper()
        != str(requested_value).upper()
    )
    return {
        "av_accumulator_precision": "unknown",
        "fallback_count": len(fallback_roles),
        "fallback_roles": fallback_roles,
        "precision_identity": not fallback_roles,
        "qk_accumulator_precision": "unknown",
        "requested_precision": dict(requested),
        "realized_precision": dict(realized),
    }


def classify_support(row: Mapping[str, Any]) -> str:
    if not bool(row.get("onnx_export_success", False)):
        return "unsupported_export"
    if not bool(row.get("trt_build_success", False)):
        return "unsupported_build"
    if not bool(row.get("runtime_success", False)):
        return "unsupported_build"
    if not bool(row.get("precision_identity", False)):
        return "supported_with_fallback"
    if bool(row.get("fused_mha_detected", False)):
        return "supported_fused_mha"
    return "supported_primitive"


__all__ = [
    "audit_requested_realized_precision",
    "classify_support",
    "inspect_attention_layers",
]
