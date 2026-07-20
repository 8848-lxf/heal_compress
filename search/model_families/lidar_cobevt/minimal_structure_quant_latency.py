"""Contracts for the bounded CoBEVT structure/quantization/latency study.

The module deliberately contains no model-loading or TensorRT side effects.  It
defines the immutable experiment matrix and the fail-closed admission rules used
by the resumable orchestration entry point.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from dataclasses import asdict, dataclass
from typing import Iterable, Mapping, Sequence

from .attention_precision_boundaries import (
    AttentionBoundaryProfile,
    attention_boundary_profile,
)


F3_LUT_WIDTHS = (16, 24, 32)
F3_LUT_PROFILE_ID = "F3_PRIMITIVE_V1"


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
    ).hexdigest()


@dataclass(frozen=True)
class StructureProfile:
    profile_id: str
    family: str
    d_qk: int
    d_v: int
    embed_dim: int = 256
    heads: int = 8

    def __post_init__(self) -> None:
        if self.family not in {"baseline", "uniform", "qk_only", "v_only"}:
            raise ValueError(f"invalid_structure_family:{self.family}")
        if min(self.d_qk, self.d_v, self.embed_dim, self.heads) <= 0:
            raise ValueError("nonpositive_structure_profile_field")
        if self.family == "uniform" and self.d_qk != self.d_v:
            raise ValueError("uniform_structure_requires_equal_qk_v")
        if self.family == "qk_only" and self.d_v != 32:
            raise ValueError("qk_only_structure_requires_original_v")
        if self.family == "v_only" and self.d_qk != 32:
            raise ValueError("v_only_structure_requires_original_qk")

    @property
    def structure_hash(self) -> str:
        return _canonical_hash(asdict(self))


def mandatory_structure_profiles() -> tuple[StructureProfile, ...]:
    """Return the exact user-approved S0--S4 matrix, in execution order."""

    return (
        StructureProfile("S0", "baseline", 32, 32),
        StructureProfile("S1", "uniform", 24, 24),
        StructureProfile("S2", "uniform", 16, 16),
        StructureProfile("S3", "qk_only", 24, 32),
        StructureProfile("S4", "v_only", 32, 24),
    )


def cross_precision_contracts() -> tuple[AttentionBoundaryProfile, AttentionBoundaryProfile]:
    """P0 and F3 share FP16 outside Attention and differ only inside it."""

    return (
        attention_boundary_profile("P0_rest_fp16_attention_fp32"),
        attention_boundary_profile("F3_rest_fp16_qk_fp32_minimal_island"),
    )


def structure_precision_interaction(
    *,
    s0_p0_map: float,
    s0_f3_map: float,
    candidate_p0_map: float,
    candidate_f3_map: float,
) -> dict[str, float]:
    values = tuple(
        float(value)
        for value in (s0_p0_map, s0_f3_map, candidate_p0_map, candidate_f3_map)
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("interaction_map_nonfinite")
    delta_prune = values[2] - values[0]
    delta_f3_base = values[1] - values[0]
    delta_f3_candidate = values[3] - values[2]
    return {
        "delta_prune": delta_prune,
        "delta_f3_base": delta_f3_base,
        "delta_f3_candidate": delta_f3_candidate,
        "interaction": delta_f3_candidate - delta_f3_base,
    }


@dataclass(frozen=True)
class SmoothQuantProfile:
    profile_id: str
    int8_projection_roles: tuple[str, ...]
    qk_precision: str = "FP32"
    base_profile_id: str = "F3_rest_fp16_qk_fp32_minimal_island"
    requires_successful_profile: str | None = None

    def __post_init__(self) -> None:
        legal = {
            "q_projection",
            "k_projection",
            "v_projection",
            "output_projection",
        }
        if not set(self.int8_projection_roles) <= legal:
            raise ValueError(f"smoothquant_illegal_role:{self.profile_id}")
        if self.qk_precision != "FP32":
            raise ValueError("smoothquant_native_int8_qk_forbidden")


def smoothquant_profiles() -> tuple[SmoothQuantProfile, ...]:
    return (
        SmoothQuantProfile("SQ0", ()),
        SmoothQuantProfile("SQ1", ("q_projection", "k_projection")),
        SmoothQuantProfile(
            "SQ2", ("q_projection", "k_projection", "v_projection")
        ),
        SmoothQuantProfile(
            "SQ3",
            (
                "q_projection",
                "k_projection",
                "v_projection",
                "output_projection",
            ),
            requires_successful_profile="SQ2",
        ),
    )


_SMOOTHQUANT_ROLE_MODULE_TOKEN = {
    "q_projection": "q_proj",
    "k_projection": "k_proj",
    "v_projection": "v_proj",
    "output_projection": "out_proj",
}


def selective_smoothquant_config(
    roles: Iterable[str], *, alpha: float
) -> dict[str, object]:
    """ModelOpt 0.29 config enabling only the named projection quantizers."""

    selected = tuple(str(role) for role in roles)
    unknown = sorted(set(selected) - set(_SMOOTHQUANT_ROLE_MODULE_TOKEN))
    if unknown:
        raise ValueError(f"smoothquant_unknown_projection_roles:{unknown}")
    value = float(alpha)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("smoothquant_alpha_out_of_range")
    quant_cfg: dict[str, object] = {}
    for role in selected:
        token = _SMOOTHQUANT_ROLE_MODULE_TOKEN[role]
        quant_cfg[f"*{token}*weight_quantizer"] = {
            "num_bits": 8,
            "axis": 0,
            "enable": True,
        }
        quant_cfg[f"*{token}*input_quantizer"] = {
            "num_bits": 8,
            "axis": None,
            "enable": True,
        }
    quant_cfg["default"] = {"enable": False}
    return {
        "algorithm": {"method": "smoothquant", "alpha": value},
        "quant_cfg": quant_cfg,
    }


def choose_smoothquant_alpha(
    rows: Iterable[Mapping[str, float]],
) -> dict[str, float]:
    candidates: list[dict[str, float]] = []
    for source in rows:
        row = {str(key): float(value) for key, value in source.items()}
        required = (row.get("alpha"), row.get("relative_l2"), row.get("softmax_js"))
        if any(value is None or not math.isfinite(value) for value in required):
            raise ValueError("smoothquant_metric_nonfinite")
        row["selection_score"] = float(row["relative_l2"] + row["softmax_js"])
        candidates.append(row)
    if not candidates:
        raise ValueError("smoothquant_alpha_rows_empty")
    return min(
        candidates,
        key=lambda row: (row["selection_score"], row["relative_l2"], row["alpha"]),
    )


@dataclass(frozen=True)
class AttentionDeploymentShape:
    block_id: str
    attention_kind: str
    batch: int
    num_heads: int
    sq: int
    skv: int
    external_embed_dim: int
    input_layout: str
    mask_kind: str

    def __post_init__(self) -> None:
        if self.attention_kind not in {"window", "grid"}:
            raise ValueError(f"unsupported_attention_kind:{self.attention_kind}")
        if min(
            self.batch,
            self.num_heads,
            self.sq,
            self.skv,
            self.external_embed_dim,
        ) <= 0:
            raise ValueError("nonpositive_attention_deployment_shape")


@dataclass(frozen=True)
class F3LatencyCandidate:
    block: AttentionDeploymentShape
    d_qk: int
    d_v: int
    gpu_arch: str
    gpu_uuid: str
    tensorrt_version: str
    cuda_version: str
    driver: str
    precision_profile_id: str = F3_LUT_PROFILE_ID

    @property
    def key_payload(self) -> dict[str, object]:
        return {
            "block": asdict(self.block),
            "cuda_version": self.cuda_version,
            "d_qk": self.d_qk,
            "d_v": self.d_v,
            "driver": self.driver,
            "gpu_arch": self.gpu_arch,
            "gpu_uuid": self.gpu_uuid,
            "precision_profile_id": self.precision_profile_id,
            "schema_version": "cobevt-f3-primitive-lut-key-v1",
            "tensorrt_version": self.tensorrt_version,
        }

    @property
    def key_hash(self) -> str:
        return _canonical_hash(self.key_payload)


def build_f3_lut_candidates(
    blocks: Sequence[AttentionDeploymentShape],
    *,
    gpu_arch: str,
    gpu_uuid: str,
    tensorrt_version: str,
    cuda_version: str,
    driver: str,
) -> tuple[F3LatencyCandidate, ...]:
    if len(blocks) != 6 or len({row.block_id for row in blocks}) != 6:
        raise ValueError("f3_lut_requires_six_unique_attention_blocks")
    rows = tuple(
        F3LatencyCandidate(
            block=block,
            d_qk=d_qk,
            d_v=d_v,
            gpu_arch=str(gpu_arch),
            gpu_uuid=str(gpu_uuid),
            tensorrt_version=str(tensorrt_version),
            cuda_version=str(cuda_version),
            driver=str(driver),
        )
        for block in blocks
        for d_qk in F3_LUT_WIDTHS
        for d_v in F3_LUT_WIDTHS
    )
    if len(rows) != 54 or len({row.key_hash for row in rows}) != 54:
        raise RuntimeError("f3_lut_candidate_matrix_incomplete_or_duplicate")
    return rows


def validate_f3_lut_admission(
    *,
    requested_realized_match: bool,
    fusion_kind: str,
    isolated_gpu: bool,
    formal: bool,
) -> None:
    if not requested_realized_match:
        raise ValueError("lut_requested_realized_mismatch")
    if str(fusion_kind) in {"complete_fused_mha", "fused_mha"}:
        raise ValueError("lut_fused_phenotype_forbidden")
    if formal and not isolated_gpu:
        raise ValueError("lut_formal_requires_isolated_gpu")


def latency_lut_filename(*, formal: bool) -> str:
    return (
        "f3_primitive_latency_lut.csv"
        if formal
        else "f3_primitive_latency_lut_screening.csv"
    )


def aggregate_latency_repetitions(
    rows: Sequence[Mapping[str, float]],
) -> dict[str, float | int]:
    if not rows:
        raise ValueError("latency_repetitions_empty")
    fields = ("p50_ms", "p90_ms", "p95_ms", "p99_ms")
    result: dict[str, float | int] = {"repeat_count": len(rows)}
    for field in fields:
        values = [float(row[field]) for row in rows]
        if not all(math.isfinite(value) and value >= 0 for value in values):
            raise ValueError(f"latency_metric_invalid:{field}")
        result[field] = float(statistics.median(values))
    return result


def validate_lut_full_engine_deltas(
    rows: Sequence[Mapping[str, float | str]],
) -> dict[str, object]:
    if not rows:
        raise ValueError("lut_validation_rows_empty")
    errors: list[float] = []
    for row in rows:
        predicted = float(row["predicted_delta_ms"])
        actual = float(row["actual_delta_ms"])
        if not math.isfinite(predicted) or not math.isfinite(actual):
            raise ValueError("lut_validation_delta_nonfinite")
        errors.append(abs(predicted - actual) / max(abs(actual), 1e-12))
    mean_error = float(sum(errors) / len(errors))
    status = (
        "search_ready"
        if mean_error <= 0.10
        else "screening_only"
        if mean_error <= 0.20
        else "not_additive"
    )
    return {
        "mean_relative_error": mean_error,
        "record_count": len(rows),
        "relative_errors": errors,
        "status": status,
    }


__all__ = [
    "AttentionDeploymentShape",
    "F3LatencyCandidate",
    "F3_LUT_PROFILE_ID",
    "F3_LUT_WIDTHS",
    "SmoothQuantProfile",
    "StructureProfile",
    "aggregate_latency_repetitions",
    "build_f3_lut_candidates",
    "choose_smoothquant_alpha",
    "cross_precision_contracts",
    "latency_lut_filename",
    "mandatory_structure_profiles",
    "smoothquant_profiles",
    "selective_smoothquant_config",
    "structure_precision_interaction",
    "validate_f3_lut_admission",
    "validate_lut_full_engine_deltas",
]
