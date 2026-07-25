"""Cross-model SmoothQuant projection profiles and alpha selection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Iterable, Mapping


SMOOTHQUANT_ALPHA_GRID = (0.5, 0.6, 0.7, 0.75, 0.8)


@dataclass(frozen=True)
class SmoothQuantRoleProfile:
    profile_id: str
    int8_roles: tuple[str, ...]
    protected_roles: tuple[str, ...] = (
        "layernorm",
        "qk_matmul",
        "softmax",
        "residual_add",
    )
    qk_contract: str = "DQ_FP32_QK"

    def __post_init__(self) -> None:
        legal = {"q_projection", "k_projection", "v_projection", "output_projection", "ffn1", "ffn2"}
        if not set(self.int8_roles) <= legal:
            raise ValueError(f"smoothquant_illegal_roles:{self.profile_id}")
        if self.qk_contract != "DQ_FP32_QK":
            raise ValueError("native_int8_qk_forbidden")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def smoothquant_profiles() -> tuple[SmoothQuantRoleProfile, ...]:
    return (
        SmoothQuantRoleProfile("SQ0_F3", ()),
        SmoothQuantRoleProfile("SQ1_QK", ("q_projection", "k_projection")),
        SmoothQuantRoleProfile("SQ2_QKV", ("q_projection", "k_projection", "v_projection")),
        SmoothQuantRoleProfile("SQ3_QKVO", ("q_projection", "k_projection", "v_projection", "output_projection")),
        SmoothQuantRoleProfile("SQ4_FFN", ("ffn1", "ffn2")),
        # Build-only diagnostics used when TensorRT rejects the joint SQ4
        # graph.  They are never admitted to the final profile library unless
        # separately evaluated and explicitly promoted.
        SmoothQuantRoleProfile("SQ4A_FFN1_DIAGNOSTIC", ("ffn1",)),
        SmoothQuantRoleProfile("SQ4B_FFN2_DIAGNOSTIC", ("ffn2",)),
        SmoothQuantRoleProfile("SQ5_QK_PLUS_FFN", ("q_projection", "k_projection", "ffn1", "ffn2")),
        SmoothQuantRoleProfile(
            "SQ6_ALL_TRANSFORMER_LINEAR",
            ("q_projection", "k_projection", "v_projection", "output_projection", "ffn1", "ffn2"),
        ),
    )


def selective_smoothquant_config(
    module_paths: Iterable[str], *, alpha: float
) -> dict[str, Any]:
    """Return a ModelOpt 0.29 config enabling only exact selected Linear paths.

    Role selection is resolved against the fresh model inventory before this
    function is called.  Exact module paths prevent CoBEVT naming rules from
    leaking into V2X-ViT and keep unobserved heterogeneous branches disabled.
    """

    value = float(alpha)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("smoothquant_alpha_out_of_range")
    selected = tuple(sorted({str(path) for path in module_paths if str(path)}))
    if not selected:
        raise ValueError("smoothquant_module_paths_empty")
    quant_cfg: dict[str, Any] = {"default": {"enable": False}}
    for path in selected:
        quant_cfg[f"*{path}*weight_quantizer"] = {
            "num_bits": 8,
            "axis": 0,
            "enable": True,
        }
        quant_cfg[f"*{path}*input_quantizer"] = {
            "num_bits": 8,
            "axis": None,
            "enable": True,
        }
    return {
        "algorithm": {"method": "smoothquant", "alpha": value},
        "quant_cfg": quant_cfg,
    }


def choose_alpha(rows: Iterable[Mapping[str, float]]) -> dict[str, float]:
    """Minimize equally normalized numerical damage metrics, fail closed."""

    values = [{str(key): float(value) for key, value in row.items()} for row in rows]
    if not values:
        raise ValueError("smoothquant_alpha_rows_empty")
    required = (
        "projection_relative_l2",
        "qk_relative_l2",
        "softmax_js",
        "ffn_output_relative_l2",
        "residual_update_relative_l2",
        "saturation_ratio",
        "scale_stability_delta",
    )
    if any(
        key not in row or not math.isfinite(row[key])
        for row in values
        for key in ("alpha", *required)
    ):
        raise ValueError("smoothquant_alpha_metric_missing_or_nonfinite")
    for key in required:
        series = [row[key] for row in values]
        lo, hi = min(series), max(series)
        span = hi - lo
        for row in values:
            row[f"normalized_{key}"] = 0.0 if span == 0 else (row[key] - lo) / span
    for row in values:
        row["selection_score"] = sum(row[f"normalized_{key}"] for key in required)
    return min(values, key=lambda row: (row["selection_score"], row["qk_relative_l2"], row["alpha"]))


__all__ = [
    "SMOOTHQUANT_ALPHA_GRID",
    "SmoothQuantRoleProfile",
    "choose_alpha",
    "selective_smoothquant_config",
    "smoothquant_profiles",
]
