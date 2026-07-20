"""Auditable CoBEVT QK multiplication and accumulation diagnostics."""

from __future__ import annotations

import inspect
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch


@dataclass(frozen=True)
class AttentionAccumulationProfile:
    name: str
    family: str
    projection_dtype: str
    qk_operand_dtype: str
    requested_multiplication_dtype: str
    requested_accumulator_dtype: str
    qk_input_source: str
    trt_exact_semantics_expected: bool
    description: str


_PROFILES = {
    row.name: row
    for row in (
        AttentionAccumulationProfile(
            "R0_fp32_operands_fp32_accum",
            "FP16",
            "FP32",
            "FP32",
            "FP32",
            "FP32",
            "fp32_projection",
            True,
            "FP32 reference QK MatMul.",
        ),
        AttentionAccumulationProfile(
            "R1_fp16_operands_default_accum",
            "FP16",
            "FP16",
            "FP16",
            "FP16",
            "DEFAULT",
            "fp16_projection",
            True,
            "Native FP16 QK with TensorRT-selected accumulation.",
        ),
        AttentionAccumulationProfile(
            "R2_fp16_operands_forced_fp32_accum",
            "FP16",
            "FP16",
            "FP16",
            "FP16",
            "FP32",
            "fp16_projection",
            False,
            "Requested FP16 multiplication with separately forced FP32 accumulation.",
        ),
        AttentionAccumulationProfile(
            "R3_fp16_projection_cast_fp32_operands_fp32_accum",
            "FP16",
            "FP16",
            "FP32",
            "FP32",
            "FP32",
            "fp16_projection_cast_fp32",
            True,
            "F3: FP16 projection followed by explicit FP32 QK operands.",
        ),
        AttentionAccumulationProfile(
            "R4_fp32_projection_cast_fp16_default_accum",
            "FP16",
            "FP32",
            "FP16",
            "FP16",
            "DEFAULT",
            "fp32_projection_cast_fp16",
            True,
            "FP32 projection rounded only at the QK boundary.",
        ),
        AttentionAccumulationProfile(
            "R5_fp32_projection_fp16_operands_fp32_accum",
            "FP16",
            "FP32",
            "FP16",
            "FP16",
            "FP32",
            "fp32_projection_cast_fp16",
            False,
            "Requested FP16 multiplication and FP32 accumulation after FP32 projection.",
        ),
        AttentionAccumulationProfile(
            "I0_fp32_qk_reference",
            "INT8",
            "FP32",
            "FP32",
            "FP32",
            "FP32",
            "fp32_capture",
            True,
            "INT8 study FP32 reference.",
        ),
        AttentionAccumulationProfile(
            "I1_int8_qk_int32_accum",
            "INT8",
            "FP32",
            "INT8",
            "INT8",
            "INT32",
            "explicit_int8_qk",
            False,
            "Requested native INT8 QK multiplication and INT32 accumulation.",
        ),
        AttentionAccumulationProfile(
            "I2_int8_qk_dq_fp32_matmul",
            "INT8",
            "FP32",
            "FP32",
            "FP32",
            "FP32",
            "int8_qk_dequantized_fp32",
            True,
            "Static INT8 Q/K storage, DQ to FP32 before QK MatMul.",
        ),
        AttentionAccumulationProfile(
            "I3_static_per_tensor_int8",
            "INT8",
            "FP32",
            "INT8",
            "INT8",
            "INT32",
            "static_per_tensor_int8",
            False,
            "Static per-tensor INT8 QK request.",
        ),
        AttentionAccumulationProfile(
            "I4_static_per_block_int8_if_expressible",
            "INT8",
            "FP32",
            "INT8",
            "INT8",
            "INT32",
            "static_per_block_int8",
            False,
            "Static per-block INT8 QK when expressible by the installed TensorRT.",
        ),
    )
}

ACCUMULATION_PROFILE_NAMES = tuple(_PROFILES)


def accumulation_profile(name: str) -> AttentionAccumulationProfile:
    try:
        return _PROFILES[str(name)]
    except KeyError as exc:
        raise ValueError(f"unknown_attention_accumulation_profile:{name}") from exc


def accumulation_profile_manifest() -> dict[str, dict[str, Any]]:
    return {name: asdict(profile) for name, profile in sorted(_PROFILES.items())}


def classify_qk_tactic(tactic_name: str) -> dict[str, Any]:
    """Decode only precision fields made explicit by a TensorRT tactic name."""

    tactic = str(tactic_name).lower()
    match = re.search(
        r"gemm_(f16|f32|int8)(f16|f32|int8)_(f16|f32|int8)(f16|f32|int8)_(f16|f32|int8)",
        tactic,
    )
    if not match:
        return {
            "left_operand_precision": "unknown",
            "right_operand_precision": "unknown",
            "output_precision": "unknown",
            "compute_precision": "unknown",
            "accumulator_precision": "unknown",
            "evidence_sufficient": False,
        }
    left, right, third, fourth, fifth = match.groups()
    normalize = {"f16": "FP16", "f32": "FP32", "int8": "INT8"}
    # TensorRT tactic strings are not a stable public schema.  The all-FP32
    # pattern is nevertheless unambiguous and is used by the accepted F3 QK.
    if {left, right, third, fourth, fifth} == {"f32"}:
        return {
            "left_operand_precision": "FP32",
            "right_operand_precision": "FP32",
            "output_precision": "FP32",
            "compute_precision": "FP32",
            "accumulator_precision": "FP32",
            "evidence_sufficient": True,
        }
    return {
        "left_operand_precision": normalize[left],
        "right_operand_precision": normalize[right],
        "output_precision": normalize.get(fifth, "unknown"),
        "compute_precision": "unknown",
        "accumulator_precision": "unknown",
        "evidence_sufficient": False,
    }


def _engine_tensor_precision(layer_info: Mapping[str, Any]) -> str:
    values = []
    for key in ("Inputs", "Outputs"):
        for tensor in layer_info.get(key, ()):
            value = str(tensor.get("Format/Datatype", "")).lower()
            if "half" in value:
                values.append("FP16")
            elif "float" in value:
                values.append("FP32")
            elif "int8" in value:
                values.append("INT8")
    return next(iter(set(values))) if len(set(values)) == 1 else "unknown"


def _tactic_precision(tactic_name: str) -> str:
    tactic = str(tactic_name).lower()
    if "h16816gemm" in tactic or "f16f16" in tactic:
        return "FP16"
    if "f32f32" in tactic:
        return "FP32"
    if "int8" in tactic or "i8i8" in tactic:
        return "INT8"
    return "unknown"


def classify_attention_precision_row(
    *,
    profile_contract_id: str,
    role: str,
    layer_info: Mapping[str, Any],
    onnx_compute_precision: str = "unknown",
    explicit_onnx_boundary: bool = False,
) -> dict[str, Any]:
    """Classify one Attention role from its explicit contract and TRT evidence."""

    from search.model_families.lidar_cobevt.attention_precision_boundaries import (
        attention_boundary_profile,
    )

    normalized_role = str(role)
    profile = attention_boundary_profile(profile_contract_id)
    if normalized_role not in profile.role_dtypes:
        raise ValueError(f"attention_precision_role_unknown:{normalized_role}")
    requested = str(profile.role_dtypes[normalized_role])
    dtype_precision = _engine_tensor_precision(layer_info)
    tactic_name = str(layer_info.get("TacticName", ""))
    tactic_precision = _tactic_precision(tactic_name)
    normalized_onnx_precision = str(onnx_compute_precision).upper()
    if normalized_onnx_precision not in {"FP32", "FP16", "INT8"}:
        normalized_onnx_precision = "unknown"
    conflict = (
        dtype_precision != "unknown"
        and tactic_precision != "unknown"
        and dtype_precision != tactic_precision
    )
    if conflict:
        realized = "unknown"
        realized_source = "conflicting_engine_evidence"
        confidence = "none"
    elif dtype_precision != "unknown" and tactic_precision != "unknown":
        realized = dtype_precision
        realized_source = "engine_inspector+tactic"
        confidence = "high"
    elif dtype_precision != "unknown":
        realized = dtype_precision
        realized_source = "engine_inspector_dtype"
        confidence = "medium"
    elif tactic_precision != "unknown":
        realized = tactic_precision
        realized_source = "tactic"
        confidence = "medium"
    elif explicit_onnx_boundary and normalized_onnx_precision != "unknown":
        realized = normalized_onnx_precision
        realized_source = "onnx_explicit_boundary"
        confidence = "medium"
    else:
        realized = "unknown"
        realized_source = "insufficient_evidence"
        confidence = "none"
    decoded_qk = classify_qk_tactic(tactic_name)
    accumulator = (
        str(decoded_qk["accumulator_precision"])
        if normalized_role == "qk_matmul" and decoded_qk["evidence_sufficient"]
        else "not_applicable" if normalized_role != "qk_matmul" else "unknown"
    )
    return {
        "accumulator_precision": accumulator,
        "classification_confidence": confidence,
        "classification_conflict": conflict,
        "conflict_reason": (
            f"dtype={dtype_precision},tactic={tactic_precision}" if conflict else ""
        ),
        "precision_evidence": {
            "input_dtypes": [
                str(value.get("Format/Datatype", ""))
                for value in layer_info.get("Inputs", ())
            ],
            "output_dtypes": [
                str(value.get("Format/Datatype", ""))
                for value in layer_info.get("Outputs", ())
            ],
            "tactic": tactic_name,
            "onnx_compute_precision": normalized_onnx_precision,
            "explicit_onnx_boundary": bool(explicit_onnx_boundary),
        },
        "profile_contract_id": str(profile_contract_id),
        "requested_contract_role": normalized_role,
        "requested_precision": requested,
        "requested_precision_source": f"profile_contract.roles.{normalized_role}",
        "realized_precision": realized,
        "realized_precision_source": realized_source,
        "requested_realized_match": bool(realized != "unknown" and requested == realized),
    }


def trt_accumulator_realization(
    *,
    profile_name: str,
    strongly_typed: bool,
    layer_info: Mapping[str, Any],
    trt_version: str,
) -> dict[str, Any]:
    profile = accumulation_profile(profile_name)
    tactic = str(layer_info.get("TacticName", layer_info.get("tactic_name", "")))
    decoded = classify_qk_tactic(tactic)
    if (
        strongly_typed
        and profile.qk_operand_dtype == "FP16"
        and profile.requested_accumulator_dtype == "FP32"
        and str(trt_version).startswith("10.9")
    ):
        return {
            **decoded,
            "status": "unsupported_exact_semantics",
            "accumulator_precision": "unknown",
            "evidence_sufficient": False,
            "reason": "trt_10_9_strongly_typed_matmul_has_no_separate_accumulator_api",
        }
    return {
        **decoded,
        "status": "realized" if decoded["evidence_sufficient"] else "unknown",
    }


def classify_int8_realization(
    profile: AttentionAccumulationProfile,
    *,
    qk_input_dtype: str,
    qk_tactic: str,
) -> dict[str, Any]:
    decoded = classify_qk_tactic(qk_tactic)
    input_dtype = str(qk_input_dtype).upper()
    native = (
        profile.requested_multiplication_dtype == "INT8"
        and input_dtype == "INT8"
        and decoded["left_operand_precision"] == "INT8"
        and decoded["right_operand_precision"] == "INT8"
        and decoded["accumulator_precision"] == "INT32"
        and decoded["evidence_sufficient"]
    )
    if input_dtype == "FP32" and decoded["left_operand_precision"] == "FP32":
        multiplication = "FP32"
        accumulator = decoded["accumulator_precision"]
    else:
        multiplication = (
            "INT8" if native else decoded["compute_precision"]
        )
        accumulator = decoded["accumulator_precision"]
    return {
        "native_int8_qk": bool(native),
        "realized_multiplication_precision": multiplication,
        "realized_accumulator_precision": accumulator,
        "requested_profile": profile.name,
    }


def dynamic_quantize_api_verdict(
    *,
    python_api_present: bool,
    cpp_api_present: bool,
    allowed_output_types: Iterable[str],
    allowed_scale_types: Iterable[str],
    block_sizes: Iterable[int],
) -> dict[str, Any]:
    outputs = tuple(str(value).upper() for value in allowed_output_types)
    scales = tuple(str(value).upper() for value in allowed_scale_types)
    blocks = tuple(int(value) for value in block_sizes)
    api_present = bool(python_api_present and cpp_api_present)
    int8 = api_present and "INT8" in outputs
    return {
        "api_present": api_present,
        "allowed_output_types": list(outputs),
        "allowed_scale_types": list(scales),
        "block_sizes": list(blocks),
        "int8_dynamic_quantization_supported": int8,
        "sageattention_dynamic_int8_expressible": bool(
            int8 and any(value > 0 for value in blocks)
        ),
    }


def decompose_qk_error(
    *,
    reference_error: float,
    fp16_projection_fp32_qk_error: float,
    fp32_projection_fp16_default_error: float,
    fp32_projection_fp16_fp32_accum_error: float,
    fp16_projection_fp16_default_error: float,
) -> dict[str, float]:
    baseline = float(reference_error)
    input_rounding = float(fp16_projection_fp32_qk_error) - baseline
    multiplication = float(fp32_projection_fp16_fp32_accum_error) - baseline
    accumulation = (
        float(fp32_projection_fp16_default_error)
        - float(fp32_projection_fp16_fp32_accum_error)
    )
    interaction = (
        float(fp16_projection_fp16_default_error)
        - baseline
        - input_rounding
        - multiplication
        - accumulation
    )
    return {
        "input_rounding_contribution": input_rounding,
        "multiplication_contribution": multiplication,
        "accumulation_contribution": accumulation,
        "interaction_term": interaction,
    }


def _relative_l2(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    ref = reference.detach().double().reshape(-1)
    value = candidate.detach().double().reshape(-1)
    denominator = max(float(torch.linalg.vector_norm(ref)), 1.0e-30)
    return float(torch.linalg.vector_norm(value - ref)) / denominator


def _cosine(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    ref = reference.detach().double().reshape(-1)
    value = candidate.detach().double().reshape(-1)
    denominator = float(torch.linalg.vector_norm(ref) * torch.linalg.vector_norm(value))
    return float(torch.dot(ref, value)) / max(denominator, 1.0e-30)


def _fp16_products_fp32_accum(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    products = q.half().unsqueeze(-2) * k.half().unsqueeze(-3)
    return products.float().sum(dim=-1)


def _fp16_products_fp16_accum(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    products = q.half().unsqueeze(-2) * k.half().unsqueeze(-3)
    total = torch.zeros(products.shape[:-1], dtype=torch.float16, device=products.device)
    for index in range(products.shape[-1]):
        total = (total + products[..., index]).half()
    return total


def _qk_score(
    profile_name: str,
    *,
    q_fp32: torch.Tensor,
    k_fp32: torch.Tensor,
    q_fp16_projection: torch.Tensor,
    k_fp16_projection: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    if profile_name == "R0_fp32_operands_fp32_accum":
        return torch.matmul(q_fp32.float() * scale, k_fp32.float().transpose(-1, -2))
    if profile_name == "R1_fp16_operands_default_accum":
        return _fp16_products_fp16_accum(
            q_fp16_projection * scale, k_fp16_projection
        )
    if profile_name == "R2_fp16_operands_forced_fp32_accum":
        return _fp16_products_fp32_accum(
            q_fp16_projection * scale, k_fp16_projection
        )
    if profile_name == "R3_fp16_projection_cast_fp32_operands_fp32_accum":
        return torch.matmul(
            q_fp16_projection.float() * scale,
            k_fp16_projection.float().transpose(-1, -2),
        )
    if profile_name == "R4_fp32_projection_cast_fp16_default_accum":
        return _fp16_products_fp16_accum(q_fp32.half() * scale, k_fp32.half())
    if profile_name == "R5_fp32_projection_fp16_operands_fp32_accum":
        return _fp16_products_fp32_accum(q_fp32.half() * scale, k_fp32.half())
    raise ValueError(f"not_fp16_accumulation_profile:{profile_name}")


def run_fp16_numerical_matrix(
    capture_paths: Iterable[str | Path], *, device: torch.device
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    profile_names = tuple(name for name in ACCUMULATION_PROFILE_NAMES if name.startswith("R"))
    for path_value in capture_paths:
        path = Path(path_value).expanduser().resolve()
        capture = torch.load(path, map_location=device)
        q_fp32 = capture["q_fp32"].to(device=device, dtype=torch.float32)
        k_fp32 = capture["k_fp32"].to(device=device, dtype=torch.float32)
        q_fp16 = capture["q_fp16_projection"].to(device=device, dtype=torch.float16)
        k_fp16 = capture["k_fp16_projection"].to(device=device, dtype=torch.float16)
        scale = float(capture["scale"])
        reference = _qk_score(
            "R0_fp32_operands_fp32_accum",
            q_fp32=q_fp32,
            k_fp32=k_fp32,
            q_fp16_projection=q_fp16,
            k_fp16_projection=k_fp16,
            scale=scale,
        )
        reference_softmax = torch.softmax(reference.float(), dim=-1)
        for profile_name in profile_names:
            score = _qk_score(
                profile_name,
                q_fp32=q_fp32,
                k_fp32=k_fp32,
                q_fp16_projection=q_fp16,
                k_fp16_projection=k_fp16,
                scale=scale,
            )
            attention = torch.softmax(score.float(), dim=-1)
            finite = bool(torch.isfinite(score).all() and torch.isfinite(attention).all())
            rows.append(
                {
                    "attention_type": str(capture.get("attention_type", "")),
                    "capture_path": str(path),
                    "finite": finite,
                    "frame_id": str(capture.get("frame_id", "")),
                    "profile": profile_name,
                    "qk_cosine": _cosine(reference, score),
                    "qk_max_abs_error": float(
                        (score.float() - reference.float()).abs().max()
                    ),
                    "qk_relative_l2": _relative_l2(reference, score),
                    "softmax_cosine": _cosine(reference_softmax, attention),
                    "softmax_relative_l2": _relative_l2(
                        reference_softmax, attention
                    ),
                    "trt_exact_semantics_expected": accumulation_profile(
                        profile_name
                    ).trt_exact_semantics_expected,
                }
            )
    return rows


def write_tp_attention_contract(destination: str | Path) -> dict[str, Any]:
    import torch_pruning as tp

    source = inspect.getsource(tp.pruner.function.MultiheadAttentionPruner)
    report = {
        "torch_pruning_version": str(getattr(tp, "__version__", "unknown")),
        "source_file": str(inspect.getsourcefile(tp.pruner.function.MultiheadAttentionPruner)),
        "qkv_same_indices": all(
            marker in source
            for marker in ("q_proj_weight", "k_proj_weight", "v_proj_weight")
        ),
        "out_proj_input_and_output_pruned": (
            "linear.weight, keep_idxs, 0" in source
            and "linear.weight, keep_idxs, 1" in source
        ),
        "embed_dim_mod_num_heads_only": "layer.embed_dim - len(idxs)" in source,
        "per_original_head_balance_guaranteed": False,
        "custom_cobevt_qk_v_independent_semantics_supported": False,
    }
    path = Path(destination).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# Torch-Pruning Attention contract\n\n"
        f"- Torch-Pruning: `{report['torch_pruning_version']}`\n"
        f"- Source: `{report['source_file']}`\n"
        "- Q/K/V share the same `idxs`: true.\n"
        "- `out_proj` input and output are both pruned by the same indices: true.\n"
        "- Validation checks only total `embed_dim % num_heads`: true.\n"
        "- Equal deletion count inside every original head is guaranteed: false.\n"
        "- Independent CoBEVT QK and V/WO per-head semantics are native: false.\n\n"
        "The project therefore still requires `TransformerSemanticResolver` and its "
        "custom CoBEVT attention materializer.\n",
        encoding="utf-8",
    )
    return report


def write_profile_manifest(destination: str | Path) -> dict[str, Any]:
    payload = accumulation_profile_manifest()
    path = Path(destination).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


__all__ = [
    "ACCUMULATION_PROFILE_NAMES",
    "AttentionAccumulationProfile",
    "accumulation_profile",
    "accumulation_profile_manifest",
    "classify_int8_realization",
    "classify_attention_precision_row",
    "classify_qk_tactic",
    "decompose_qk_error",
    "dynamic_quantize_api_verdict",
    "run_fp16_numerical_matrix",
    "trt_accumulator_realization",
    "write_profile_manifest",
    "write_tp_attention_contract",
]
