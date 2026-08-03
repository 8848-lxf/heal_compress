"""Requested precision contracts for Transformer deployment phenotypes."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from typing import Any, Mapping


LEGAL_PRECISIONS = {"FP32", "FP16", "BF16", "INT8", "FP8_E4M3", "FP8_E5M2"}
CORE_ROLES = (
    "layernorm",
    "q_projection",
    "k_projection",
    "v_projection",
    "fused_qkv_projection",
    "qk_matmul",
    "softmax",
    "av_matmul",
    "output_projection",
    "residual_add",
    "ffn1",
    "ffn2",
)


@dataclass(frozen=True)
class PrecisionContract:
    profile_id: str
    role: str
    storage_precision: str
    operand_precision: str
    multiplication_precision: str
    accumulator_precision: str
    output_precision: str
    evidence_required: str = "requested_graph_contract"
    quantization: str = "none"

    def __post_init__(self) -> None:
        for field in (
            "storage_precision",
            "operand_precision",
            "multiplication_precision",
            "output_precision",
        ):
            value = getattr(self, field)
            if value not in LEGAL_PRECISIONS:
                raise ValueError(f"illegal_precision:{field}:{value}")
        if self.accumulator_precision not in LEGAL_PRECISIONS | {"INT32", "unknown"}:
            raise ValueError(
                f"illegal_accumulator_precision:{self.accumulator_precision}"
            )
        if self.role == "qk_matmul" and self.operand_precision == "INT8" and self.accumulator_precision != "INT32":
            raise ValueError("native_int8_qk_requires_int32_accumulator")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _contract(profile: str, role: str, precision: str, *, output: str | None = None) -> PrecisionContract:
    accumulator = "FP32" if precision == "FP32" else "unknown"
    if precision == "INT8":
        accumulator = "INT32"
    quantization = (
        "static_per_tensor_activation_symmetric_int8_per_output_channel_weight"
        if precision == "INT8"
        else "none"
    )
    return PrecisionContract(
        profile_id=profile,
        role=role,
        storage_precision=precision,
        operand_precision=precision,
        multiplication_precision=precision,
        accumulator_precision=accumulator,
        output_precision=output or precision,
        quantization=quantization,
    )


def _profile(
    profile_id: str,
    *,
    base: str,
    overrides: Mapping[str, str | tuple[str, str]],
) -> dict[str, PrecisionContract]:
    result: dict[str, PrecisionContract] = {}
    for role in CORE_ROLES:
        value = overrides.get(role, base)
        precision, output = value if isinstance(value, tuple) else (value, value)
        result[role] = _contract(profile_id, role, precision, output=output)
    return result


def precision_profiles() -> dict[str, dict[str, PrecisionContract]]:
    """Return immutable-by-construction requested profiles used by this audit."""

    profiles = {
        "B1_TRT_ATTN_FP32": _profile(
            "B1_TRT_ATTN_FP32",
            base="FP32",
            overrides={},
        ),
        "B2_TRT_STRICT_FP16": _profile(
            "B2_TRT_STRICT_FP16", base="FP16", overrides={}
        ),
        "B3_F3": _profile(
            "B3_F3",
            base="FP16",
            overrides={
                "layernorm": "FP32",
                "q_projection": ("FP16", "FP32"),
                "k_projection": ("FP16", "FP32"),
                "fused_qkv_projection": ("FP16", "FP32"),
                "qk_matmul": "FP32",
            },
        ),
    }
    single = {
        "P1_QKV_FP16": ("q_projection", "k_projection", "v_projection", "fused_qkv_projection"),
        "P2_QK_FP16": ("qk_matmul",),
        "P3_SOFTMAX_FP16": ("softmax",),
        "P4_AV_FP16": ("av_matmul",),
        "P5_OUT_FP16": ("output_projection",),
        "P6_LAYERNORM_FP16": ("layernorm",),
        "P7_RESIDUAL_FP16": ("residual_add",),
        "P8_FFN1_FP16": ("ffn1",),
        "P9_FFN2_FP16": ("ffn2",),
        "P10_ALL_FFN_FP16": ("ffn1", "ffn2"),
        "P11_ALL_PROJECTION_FP16": (
            "q_projection",
            "k_projection",
            "v_projection",
            "fused_qkv_projection",
            "output_projection",
        ),
        "P12_FULL_ATTENTION_FP16": (
            "layernorm",
            "q_projection",
            "k_projection",
            "v_projection",
            "fused_qkv_projection",
            "qk_matmul",
            "softmax",
            "av_matmul",
            "output_projection",
            "residual_add",
        ),
    }
    for name, roles in single.items():
        profiles[name] = _profile(
            name,
            base="FP32",
            overrides={role: "FP16" for role in roles},
        )
    # Phase-3 BF16 sensitivity is role-isolated against B1 Attention FP32.
    # It must not silently inherit strict-FP16 for every unrelated role.
    for name, roles in {
        "P13_QKV_BF16": ("q_projection", "k_projection", "v_projection", "fused_qkv_projection"),
        "P14_QK_BF16": ("qk_matmul",),
        "P15_FFN_BF16": ("ffn1", "ffn2"),
        "P16_FULL_ATTENTION_BF16": tuple(role for role in CORE_ROLES if role not in {"ffn1", "ffn2"}),
    }.items():
        profiles[name] = _profile(
            name,
            base="FP32",
            overrides={role: "BF16" for role in roles},
        )

    # Phase-6 H800 joint candidates are separate F3-derived contracts.  Q/K
    # output recovery and the protected QK core are explicit, so these cannot
    # be confused with the single-role P13--P16 experiments above.
    f3 = {
        "layernorm": "FP32",
        "q_projection": ("FP16", "FP32"),
        "k_projection": ("FP16", "FP32"),
        "fused_qkv_projection": ("FP16", "FP32"),
        "qk_matmul": "FP32",
    }
    profiles["H1_QKV_BF16_QK_FP32"] = _profile(
        "H1_QKV_BF16_QK_FP32",
        base="FP16",
        overrides={
            **f3,
            "q_projection": ("BF16", "FP32"),
            "k_projection": ("BF16", "FP32"),
            "fused_qkv_projection": ("BF16", "FP32"),
            "v_projection": "BF16",
        },
    )
    h2 = _profile(
        "H2_QKV_BF16_QK_BF16A32",
        base="FP16",
        overrides={
            **f3,
            "q_projection": "BF16",
            "k_projection": "BF16",
            "fused_qkv_projection": "BF16",
            "v_projection": "BF16",
            "qk_matmul": ("BF16", "FP16"),
        },
    )
    h2["qk_matmul"] = replace(
        h2["qk_matmul"],
        accumulator_precision="FP32",
        evidence_required="level_a_compute_contract",
    )
    profiles["H2_QKV_BF16_QK_BF16A32"] = h2
    profiles["H3_FFN_BF16"] = _profile(
        "H3_FFN_BF16",
        base="FP16",
        overrides={**f3, "ffn1": "BF16", "ffn2": "BF16"},
    )
    profiles["H4_FULL_TRANSFORMER_BF16_PROTECTED_QK"] = _profile(
        "H4_FULL_TRANSFORMER_BF16_PROTECTED_QK",
        base="BF16",
        overrides={
            "layernorm": "FP32",
            "q_projection": ("BF16", "FP32"),
            "k_projection": ("BF16", "FP32"),
            "fused_qkv_projection": ("BF16", "FP32"),
            "qk_matmul": "FP32",
        },
    )
    return profiles


def profile_hash(profile: Mapping[str, PrecisionContract]) -> str:
    payload = {name: row.to_dict() for name, row in sorted(profile.items())}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def qk_accumulator_variant(
    profile_id: str,
    *,
    operand: str,
    accumulator: str,
) -> PrecisionContract:
    output = "INT32" if operand == "INT8" else operand
    if output == "INT32":
        output = "FP16"
    return PrecisionContract(
        profile_id=profile_id,
        role="qk_matmul",
        storage_precision=operand,
        operand_precision=operand,
        multiplication_precision=operand,
        accumulator_precision=accumulator,
        output_precision=output,
        evidence_required="level_a_compute_contract",
        quantization="native_int8_operands" if operand == "INT8" else "none",
    )


__all__ = [
    "CORE_ROLES",
    "LEGAL_PRECISIONS",
    "PrecisionContract",
    "precision_profiles",
    "profile_hash",
    "qk_accumulator_variant",
]
