"""Candidate identities for the CoBEVT TensorRT head-dimension study."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Final


UNIFORM_HEAD_DIMS: Final[tuple[int, ...]] = (
    4,
    6,
    8,
    10,
    12,
    14,
    16,
    20,
    24,
    28,
    32,
    40,
    48,
    56,
    64,
    80,
    96,
    128,
)
DECOUPLED_HEAD_DIMS: Final[tuple[int, ...]] = (
    8,
    12,
    16,
    20,
    24,
    28,
    32,
    40,
    48,
    64,
)
GRAPH_VARIANTS: Final[tuple[str, ...]] = (
    "core_attention",
    "projection_attention",
)
PRECISION_PROFILES: Final[tuple[str, ...]] = (
    "P0_strict_fp32",
    "P1_strict_fp16_native",
    "P2_f3_mixed",
    "P3_int8_projections_qk_fp32",
    "P4_int8_native_attention",
)


def _canonical_hash(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class HeadDimCandidate:
    """A complete synthetic capability candidate and cache identity."""

    graph_variant: str
    structure_family: str
    d_qk: int
    d_v: int
    precision_profile: str
    tensorrt_version: str
    gpu_architecture: str
    num_heads: int = 8
    embed_dim: int = 256
    token_length: int = 32
    window_shape: tuple[int, int] = (4, 4)
    window_groups: int = 512

    def __post_init__(self) -> None:
        if self.graph_variant not in GRAPH_VARIANTS:
            raise ValueError(f"unsupported_graph_variant:{self.graph_variant}")
        if self.structure_family not in {"uniform", "qk_only", "v_only"}:
            raise ValueError(f"unsupported_structure_family:{self.structure_family}")
        if self.precision_profile not in PRECISION_PROFILES:
            raise ValueError(f"unsupported_precision_profile:{self.precision_profile}")
        if min(
            self.d_qk,
            self.d_v,
            self.num_heads,
            self.embed_dim,
            self.token_length,
            self.window_groups,
        ) <= 0:
            raise ValueError("nonpositive_head_dim_candidate_field")
        if self.structure_family == "uniform" and self.d_qk != self.d_v:
            raise ValueError("uniform_candidate_requires_equal_qk_v")
        if self.structure_family == "qk_only" and self.d_v != 32:
            raise ValueError("qk_only_candidate_requires_original_v_width")
        if self.structure_family == "v_only" and self.d_qk != 32:
            raise ValueError("v_only_candidate_requires_original_qk_width")

    @property
    def q_projection_out(self) -> int:
        return self.num_heads * self.d_qk

    @property
    def k_projection_out(self) -> int:
        return self.num_heads * self.d_qk

    @property
    def v_projection_out(self) -> int:
        return self.num_heads * self.d_v

    @property
    def out_projection_in(self) -> int:
        return self.num_heads * self.d_v

    @property
    def out_projection_out(self) -> int:
        return self.embed_dim

    @property
    def shape_id(self) -> str:
        return (
            f"{self.graph_variant}__{self.structure_family}"
            f"__qk{self.d_qk}_v{self.d_v}"
        )

    @property
    def candidate_id(self) -> str:
        return f"{self.shape_id}__{self.precision_profile}"

    @property
    def same_shape_fp32_reference_id(self) -> str:
        return f"{self.shape_id}__P0_strict_fp32"

    @property
    def candidate_hash(self) -> str:
        return _canonical_hash(asdict(self))

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload.update(
            {
                "candidate_hash": self.candidate_hash,
                "candidate_id": self.candidate_id,
                "k_projection_out": self.k_projection_out,
                "out_projection_in": self.out_projection_in,
                "out_projection_out": self.out_projection_out,
                "q_projection_out": self.q_projection_out,
                "same_shape_fp32_reference_id": self.same_shape_fp32_reference_id,
                "shape_id": self.shape_id,
                "v_projection_out": self.v_projection_out,
            }
        )
        return payload


def _profiles_for(family: str, graph_variant: str) -> tuple[str, ...]:
    if family != "uniform":
        return PRECISION_PROFILES[:3]
    if graph_variant == "core_attention":
        return (
            "P0_strict_fp32",
            "P1_strict_fp16_native",
            "P2_f3_mixed",
            "P4_int8_native_attention",
        )
    return PRECISION_PROFILES


def build_synthetic_candidate_matrix(
    *, tensorrt_version: str, gpu_architecture: str
) -> tuple[HeadDimCandidate, ...]:
    rows: list[HeadDimCandidate] = []
    families = (
        ("uniform", ((width, width) for width in UNIFORM_HEAD_DIMS)),
        ("qk_only", ((width, 32) for width in DECOUPLED_HEAD_DIMS)),
        ("v_only", ((32, width) for width in DECOUPLED_HEAD_DIMS)),
    )
    for family, dimensions in families:
        for d_qk, d_v in dimensions:
            for graph_variant in GRAPH_VARIANTS:
                for profile in _profiles_for(family, graph_variant):
                    rows.append(
                        HeadDimCandidate(
                            graph_variant=graph_variant,
                            structure_family=family,
                            d_qk=d_qk,
                            d_v=d_v,
                            precision_profile=profile,
                            tensorrt_version=str(tensorrt_version),
                            gpu_architecture=str(gpu_architecture),
                        )
                    )
    hashes = {row.candidate_hash for row in rows}
    if len(hashes) != len(rows):
        raise RuntimeError("head_dim_candidate_hash_collision")
    return tuple(rows)


__all__ = [
    "DECOUPLED_HEAD_DIMS",
    "GRAPH_VARIANTS",
    "HeadDimCandidate",
    "PRECISION_PROFILES",
    "UNIFORM_HEAD_DIMS",
    "build_synthetic_candidate_matrix",
]
