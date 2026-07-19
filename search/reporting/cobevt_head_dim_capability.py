"""Strict TensorRT provenance parsing for CoBEVT Attention capability rows."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch


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


def _topk_overlap(
    reference: torch.Tensor, candidate: torch.Tensor, k: int
) -> float:
    width = int(reference.shape[-1])
    count = min(int(k), width)
    reference_indices = torch.topk(reference, count, dim=-1).indices
    candidate_indices = torch.topk(candidate, count, dim=-1).indices
    matches = (
        reference_indices.unsqueeze(-1)
        == candidate_indices.unsqueeze(-2)
    ).any(dim=-1)
    return float(matches.float().mean().item())


def _rank_correlation(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    reference_rank = torch.argsort(torch.argsort(reference, dim=-1), dim=-1).float()
    candidate_rank = torch.argsort(torch.argsort(candidate, dim=-1), dim=-1).float()
    reference_rank = reference_rank - reference_rank.mean(dim=-1, keepdim=True)
    candidate_rank = candidate_rank - candidate_rank.mean(dim=-1, keepdim=True)
    denominator = torch.sqrt(
        (reference_rank.square().sum(dim=-1))
        * (candidate_rank.square().sum(dim=-1))
    )
    numerator = (reference_rank * candidate_rank).sum(dim=-1)
    valid = denominator > 0
    if not bool(valid.any()):
        return 1.0 if torch.equal(reference, candidate) else 0.0
    return float((numerator[valid] / denominator[valid]).mean().item())


def attention_tensor_parity(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    role: str,
) -> dict[str, Any]:
    if tuple(reference.shape) != tuple(candidate.shape):
        raise ValueError("attention_parity_shape_mismatch")
    reference64 = reference.detach().double().cpu()
    candidate64 = candidate.detach().double().cpu()
    matching_negative_infinity = torch.isneginf(reference64) & torch.isneginf(
        candidate64
    )
    allowed_nonfinite = matching_negative_infinity if role == "qk_score" else torch.zeros_like(
        matching_negative_infinity
    )
    valid_values = torch.isfinite(reference64) & torch.isfinite(candidate64)
    unexpected_nonfinite = ~(valid_values | allowed_nonfinite)
    finite = not bool(unexpected_nonfinite.any())
    difference = torch.where(
        allowed_nonfinite,
        torch.zeros_like(candidate64),
        candidate64 - reference64,
    )
    reference_flat = reference64[valid_values].reshape(-1)
    candidate_flat = candidate64[valid_values].reshape(-1)
    if reference_flat.numel() == 0:
        raise ValueError("attention_parity_no_finite_elements")
    exact = torch.equal(reference64, candidate64)
    denominator = float(torch.linalg.vector_norm(reference_flat).item())
    relative_l2 = float(torch.linalg.vector_norm(difference.reshape(-1)).item()) / max(
        denominator, torch.finfo(torch.float64).eps
    )
    cosine = (
        1.0
        if exact
        else float(
            torch.nn.functional.cosine_similarity(
                reference_flat.unsqueeze(0), candidate_flat.unsqueeze(0)
            ).item()
        )
    )
    result: dict[str, Any] = {
        "candidate_dtype": str(candidate.dtype),
        "candidate_max": float(candidate_flat.max().item()),
        "candidate_mean": float(candidate_flat.mean().item()),
        "candidate_min": float(candidate_flat.min().item()),
        "candidate_std": float(candidate_flat.std(unbiased=False).item()),
        "cosine_similarity": cosine,
        "finite": finite,
        "max_absolute_error": float(difference.abs().max().item()),
        "mean_absolute_error": float(difference.abs().mean().item()),
        "nan_count": int(torch.isnan(candidate64).sum().item()),
        "inf_count": int(torch.isinf(candidate64).sum().item()),
        "matching_negative_infinity_count": int(
            matching_negative_infinity.sum().item()
        ),
        "reference_dtype": str(reference.dtype),
        "reference_max": float(reference_flat.max().item()),
        "reference_mean": float(reference_flat.mean().item()),
        "reference_min": float(reference_flat.min().item()),
        "reference_std": float(reference_flat.std(unbiased=False).item()),
        "relative_l2_error": relative_l2,
        "role": str(role),
        "shape": list(reference.shape),
    }
    if role == "qk_score":
        result.update(
            {
                "rank_correlation": _rank_correlation(
                    reference64, candidate64
                ),
                "row_max_absolute_error": float(
                    (
                        reference64.max(dim=-1).values
                        - candidate64.max(dim=-1).values
                    )
                    .abs()
                    .mean()
                    .item()
                ),
                "sign_flip_ratio": float(
                    (torch.sign(reference64) != torch.sign(candidate64))
                    .double()
                    .mean()
                    .item()
                ),
                "top1_agreement": _topk_overlap(
                    reference64, candidate64, 1
                ),
                "top4_overlap": _topk_overlap(reference64, candidate64, 4),
                "top8_overlap": _topk_overlap(reference64, candidate64, 8),
            }
        )
    elif role == "softmax":
        epsilon = torch.finfo(torch.float64).eps
        reference_probability = reference64.clamp_min(epsilon)
        candidate_probability = candidate64.clamp_min(epsilon)
        midpoint = 0.5 * (reference_probability + candidate_probability)
        kl = (
            reference_probability
            * (reference_probability.log() - candidate_probability.log())
        ).sum(dim=-1)
        js = 0.5 * (
            reference_probability
            * (reference_probability.log() - midpoint.log())
        ).sum(dim=-1) + 0.5 * (
            candidate_probability
            * (candidate_probability.log() - midpoint.log())
        ).sum(dim=-1)
        reference_entropy = -(
            reference_probability * reference_probability.log()
        ).sum(dim=-1)
        candidate_entropy = -(
            candidate_probability * candidate_probability.log()
        ).sum(dim=-1)
        result.update(
            {
                "argmax_agreement": _topk_overlap(
                    reference64, candidate64, 1
                ),
                "entropy_delta": float(
                    (candidate_entropy - reference_entropy).mean().item()
                ),
                "js_divergence": 0.0 if exact else float(js.mean().item()),
                "kl_divergence": 0.0 if exact else float(kl.mean().item()),
                "row_sum_max_error": float(
                    (candidate64.sum(dim=-1) - 1.0).abs().max().item()
                ),
                "top4_overlap": _topk_overlap(reference64, candidate64, 4),
                "top8_overlap": _topk_overlap(reference64, candidate64, 8),
                "zero_ratio": float((candidate64 == 0).double().mean().item()),
            }
        )
    elif role == "av_output":
        reference_energy = reference64.square().mean(dim=tuple(range(reference64.ndim - 1)))
        candidate_energy = candidate64.square().mean(dim=tuple(range(candidate64.ndim - 1)))
        energy_denominator = max(
            float(torch.linalg.vector_norm(reference_energy).item()),
            torch.finfo(torch.float64).eps,
        )
        result["channel_energy_relative_error"] = float(
            torch.linalg.vector_norm(candidate_energy - reference_energy).item()
            / energy_denominator
        )
    return result


def _safe_widths(
    rows: Iterable[Mapping[str, Any]],
    *,
    family: str,
    profile: str,
    fused: bool | None = None,
    require_accuracy: bool = False,
) -> list[int]:
    values = set()
    for row in rows:
        if str(row.get("structure_family")) != family:
            continue
        if str(row.get("precision_profile")) != profile:
            continue
        if not bool(row.get("runtime_success")):
            continue
        if not bool(row.get("precision_identity")):
            continue
        if not bool(row.get("numerical_safe")):
            continue
        if require_accuracy and row.get("accuracy_safe") is not True:
            continue
        if fused is not None and bool(row.get("fused_mha_detected")) != fused:
            continue
        values.add(int(row["d_qk"] if family != "v_only" else row["d_v"]))
    return sorted(values)


def derive_head_dim_search_contract(
    rows: Iterable[Mapping[str, Any]],
    *,
    hardware_scope: Mapping[str, Any],
) -> dict[str, Any]:
    evidence = [dict(row) for row in rows]
    uniform = {
        "fp32_supported_head_dims": _safe_widths(
            evidence, family="uniform", profile="P0_strict_fp32"
        ),
        "fp16_primitive_supported_head_dims": _safe_widths(
            evidence,
            family="uniform",
            profile="P1_strict_fp16_native",
            fused=False,
        ),
        "fp16_fused_supported_head_dims": _safe_widths(
            evidence,
            family="uniform",
            profile="P1_strict_fp16_native",
            fused=True,
        ),
        "f3_supported_head_dims": _safe_widths(
            evidence, family="uniform", profile="P2_f3_mixed"
        ),
        "int8_projection_supported_head_dims": _safe_widths(
            evidence,
            family="uniform",
            profile="P3_int8_projections_qk_fp32",
        ),
        "int8_primitive_supported_head_dims": _safe_widths(
            evidence,
            family="uniform",
            profile="P4_int8_native_attention",
            fused=False,
        ),
        "int8_fused_supported_head_dims": _safe_widths(
            evidence,
            family="uniform",
            profile="P4_int8_native_attention",
            fused=True,
        ),
        "unsupported_head_dims": sorted(
            {
                int(row["d_qk"])
                for row in evidence
                if row.get("structure_family") == "uniform"
                and str(row.get("support_class", "")).startswith("unsupported")
            }
        ),
    }
    qk_supported = sorted(
        set(
            _safe_widths(
                evidence, family="qk_only", profile="P0_strict_fp32"
            )
            + _safe_widths(
                evidence, family="qk_only", profile="P1_strict_fp16_native"
            )
            + _safe_widths(
                evidence, family="qk_only", profile="P2_f3_mixed"
            )
        )
    )
    v_supported = sorted(
        set(
            _safe_widths(
                evidence, family="v_only", profile="P0_strict_fp32"
            )
            + _safe_widths(
                evidence, family="v_only", profile="P1_strict_fp16_native"
            )
            + _safe_widths(
                evidence, family="v_only", profile="P2_f3_mixed"
            )
        )
    )
    legal_uniform = sorted(
        set().union(
            *(
                set(value)
                for key, value in uniform.items()
                if key != "unsupported_head_dims"
            )
        )
    )
    forbidden = []
    unresolved = []
    for row in evidence:
        record = {
            "d_qk": int(row.get("d_qk", 0)),
            "d_v": int(row.get("d_v", 0)),
            "precision_profile": str(row.get("precision_profile", "")),
        }
        if row.get("runtime_success") and not row.get("precision_identity"):
            forbidden.append({**record, "reason": "precision_fallback"})
        elif row.get("accuracy_safe") is False:
            forbidden.append({**record, "reason": "accuracy_unsafe"})
        elif row.get("accuracy_safe") in (None, "accuracy_unresolved"):
            unresolved.append(record)
    return {
        "hardware_scope": dict(hardware_scope),
        "qk_only": {
            "fused_supported_d_qk": sorted(
                {
                    int(row["d_qk"])
                    for row in evidence
                    if row.get("structure_family") == "qk_only"
                    and row.get("fused_mha_detected")
                    and row.get("runtime_success")
                    and row.get("precision_identity")
                }
            ),
            "primitive_only_d_qk": sorted(
                set(qk_supported)
                - {
                    int(row["d_qk"])
                    for row in evidence
                    if row.get("structure_family") == "qk_only"
                    and row.get("fused_mha_detected")
                }
            ),
            "supported_d_qk": qk_supported,
        },
        "search_space_recommendation": {
            "f3_accuracy_safe_widths": _safe_widths(
                evidence,
                family="uniform",
                profile="P2_f3_mixed",
                require_accuracy=True,
            ),
            "forbidden_precision_shape_pairs": sorted(
                forbidden,
                key=lambda row: (
                    row["d_qk"], row["d_v"], row["precision_profile"], row["reason"]
                ),
            ),
            "int8_fused_eligible_widths": uniform[
                "int8_fused_supported_head_dims"
            ],
            "legal_structural_widths": legal_uniform,
            "preferred_latency_widths": [],
            "unresolved_pairs": sorted(
                unresolved,
                key=lambda row: (
                    row["d_qk"], row["d_v"], row["precision_profile"]
                ),
            ),
        },
        "uniform_attention": uniform,
        "v_only": {
            "fused_supported_d_v": sorted(
                {
                    int(row["d_v"])
                    for row in evidence
                    if row.get("structure_family") == "v_only"
                    and row.get("fused_mha_detected")
                    and row.get("runtime_success")
                    and row.get("precision_identity")
                }
            ),
            "primitive_only_d_v": sorted(v_supported),
            "supported_d_v": v_supported,
        },
    }


def write_capability_matrix(
    output_dir: str | Path, rows: Iterable[Mapping[str, Any]]
) -> dict[str, Path]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    records = [dict(row) for row in rows]
    json_path = destination / "head_dim_capability_matrix.json"
    csv_path = destination / "head_dim_capability_matrix.csv"
    markdown_path = destination / "head_dim_capability_matrix.md"
    json_path.write_text(
        json.dumps(records, indent=2, sort_keys=False, default=str) + "\n",
        encoding="utf-8",
    )
    fieldnames = sorted({str(key) for row in records for key in row})
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow(
                {
                    key: (
                        json.dumps(row.get(key), sort_keys=True)
                        if isinstance(row.get(key), (dict, list, tuple))
                        else row.get(key)
                    )
                    for key in fieldnames
                }
            )
    columns = (
        "candidate_id",
        "graph_variant",
        "structure_family",
        "d_qk",
        "d_v",
        "precision_profile",
        "support_class",
        "fused_mha_detected",
        "precision_identity",
        "numerical_safe",
        "accuracy_safe",
        "p50_ms",
        "failure_reason",
    )
    lines = ["# CoBEVT Head-Dimension TensorRT Capability Matrix", ""]

    def add_table(title: str, selected: list[dict[str, Any]]) -> None:
        lines.extend(
            [
                f"## {title}",
                "",
                "| " + " | ".join(columns) + " |",
                "|" + "|".join("---" for _ in columns) + "|",
            ]
        )
        for row in selected:
            lines.append(
                "| "
                + " | ".join(str(row.get(column, "")) for column in columns)
                + " |"
            )
        lines.append("")

    profile_titles = {
        "P0_strict_fp32": "FP32",
        "P1_strict_fp16_native": "FP16",
        "P2_f3_mixed": "F3 Mixed",
        "P3_int8_projections_qk_fp32": "INT8 Projection Mixed",
        "P4_int8_native_attention": "INT8 Native or Fused",
    }
    for profile, title in profile_titles.items():
        add_table(
            title,
            [row for row in records if row.get("precision_profile") == profile],
        )
    for family, title in (
        ("uniform", "Uniform Family"),
        ("qk_only", "QK-only Family"),
        ("v_only", "V-only Family"),
    ):
        add_table(
            title,
            [row for row in records if row.get("structure_family") == family],
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"csv": csv_path, "json": json_path, "markdown": markdown_path}


__all__ = [
    "attention_tensor_parity",
    "audit_requested_realized_precision",
    "classify_support",
    "derive_head_dim_search_contract",
    "inspect_attention_layers",
    "write_capability_matrix",
]
