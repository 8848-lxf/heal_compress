"""Canonical roles shared by CoBEVT and V2X-ViT without name aliasing."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any


CANONICAL_TRANSFORMER_ROLES = (
    "input_embedding_projection",
    "q_projection",
    "k_projection",
    "v_projection",
    "fused_qkv_projection",
    "output_projection",
    "qk_matmul",
    "qk_scale",
    "mask_relation_add",
    "softmax",
    "av_matmul",
    "layernorm",
    "residual_add",
    "ffn1",
    "activation",
    "ffn2",
    "positional_relative_embedding",
    "agent_attention",
    "temporal_attention",
    "spatial_window_grid_attention",
    "split_attention_gate",
    "detection_head",
    "pointpillar_cnn_encoder",
    "communication_fusion",
    "missing_role_mapping",
)


@dataclass(frozen=True)
class CanonicalRole:
    model_family: str
    canonical_role: str
    module_path: str = ""
    onnx_node: str = ""
    onnx_op_type: str = ""
    attention_kind: str = ""
    block: str = ""
    confidence: str = "high"
    mapping_reason: str = ""

    def __post_init__(self) -> None:
        if self.canonical_role not in CANONICAL_TRANSFORMER_ROLES:
            raise ValueError(f"unknown_canonical_transformer_role:{self.canonical_role}")
        if self.model_family not in {"lidar_cobevt", "lidar_v2xvit"}:
            raise ValueError(f"unsupported_transformer_model_family:{self.model_family}")
        if self.confidence not in {"high", "medium", "low", "missing"}:
            raise ValueError(f"invalid_role_mapping_confidence:{self.confidence}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _block_from_module(path: str) -> str:
    patterns = (
        r"^(fusion_net\.layers\.\d+\.(?:window|grid)_(?:attention|ffd))",
        r"^(fusion_net\.fusion_net\.encoder\.layers\.\d+\.0\.layers\.\d+\.[01])",
        r"^(fusion_net\.fusion_net\.encoder\.layers\.\d+\.[01])",
    )
    for pattern in patterns:
        match = re.match(pattern, path)
        if match:
            return match.group(1)
    return ""


def _attention_kind(path: str) -> str:
    lower = path.lower()
    if "window_attention" in lower:
        return "window"
    if "grid_attention" in lower:
        return "grid"
    if ".pwmsa." in lower:
        return "spatial_window"
    if any(token in lower for token in ("q_linears", "k_linears", "v_linears", "a_linears")):
        return "agent_relation"
    if "split_attn" in lower:
        return "split_attention"
    return ""


def classify_weighted_module(model_family: str, module_path: str) -> CanonicalRole:
    """Classify one real weighted module; unknown Transformer paths fail closed."""

    family = str(model_family)
    path = str(module_path)
    lower = path.lower()
    kind = _attention_kind(path)
    block = _block_from_module(path)
    role = ""
    reason = ""

    if path in {"cls_head", "reg_head", "dir_head"} or lower.endswith(
        (".cls_head", ".reg_head", ".dir_head")
    ):
        role, reason = "detection_head", "detection output head"
    elif "pillar_vfe" in lower or "prior_feed" in lower:
        role, reason = "input_embedding_projection", "input/pillar embedding projection"
    elif lower.endswith((".q_proj",)) or ".q_linears." in lower:
        role, reason = "q_projection", "explicit Q projection module"
    elif lower.endswith((".k_proj",)) or ".k_linears." in lower:
        role, reason = "k_projection", "explicit K projection module"
    elif lower.endswith((".v_proj",)) or ".v_linears." in lower:
        role, reason = "v_projection", "explicit V projection module"
    elif lower.endswith(".to_qkv"):
        role, reason = "fused_qkv_projection", "fused QKV projection module"
    elif lower.endswith(".out_proj") or lower.endswith(".to_out.0") or ".a_linears." in lower:
        role, reason = "output_projection", "attention output projection module"
    elif "split_attn.fc1" in lower or "split_attn.fc2" in lower:
        role, reason = "split_attention_gate", "multi-window split-attention gate"
    elif family == "lidar_cobevt" and re.search(r"_(?:ffd)\.fn\.net\.0$", lower):
        role, reason = "ffn1", "CoBEVT FFN expansion"
    elif family == "lidar_cobevt" and re.search(r"_(?:ffd)\.fn\.net\.3$", lower):
        role, reason = "ffn2", "CoBEVT FFN projection"
    elif family == "lidar_v2xvit" and re.search(r"encoder\.layers\.\d+\.1\.fn\.net\.0$", lower):
        role, reason = "ffn1", "V2X-ViT encoder FFN expansion"
    elif family == "lidar_v2xvit" and re.search(r"encoder\.layers\.\d+\.1\.fn\.net\.3$", lower):
        role, reason = "ffn2", "V2X-ViT encoder FFN projection"
    elif any(token in lower for token in ("fusion_net", "compressor", "aligner")):
        role, reason = "communication_fusion", "weighted communication/fusion operator"
    elif any(token in lower for token in ("encoder_", "backbone", "blocks", "deblocks", "shrinker", "shrink_conv")):
        role, reason = "pointpillar_cnn_encoder", "non-Transformer LiDAR encoder/backbone"
    else:
        role, reason = "pointpillar_cnn_encoder", "weighted non-Transformer model operator"

    if (
        role == "pointpillar_cnn_encoder"
        and "fusion_net" in lower
        and any(token in lower for token in ("attention", "attn", "encoder.layers"))
    ):
        role = "missing_role_mapping"
        reason = "Transformer-like weighted path has no safe role rule"
    confidence = "missing" if role == "missing_role_mapping" else "high"
    return CanonicalRole(
        model_family=family,
        canonical_role=role,
        module_path=path,
        attention_kind=kind,
        block=block,
        confidence=confidence,
        mapping_reason=reason,
    )


def classify_onnx_primitive(
    model_family: str,
    *,
    node_name: str,
    op_type: str,
    equation: str = "",
) -> CanonicalRole | None:
    """Classify parameter-free Transformer primitives from topology-rich names."""

    family = str(model_family)
    name = str(node_name)
    lower = name.lower()
    op = str(op_type)
    role = ""
    kind = ""
    if "split_attn" in lower:
        kind = "split_attention"
    elif "pwmsa" in lower or "window_attention" in lower:
        kind = "spatial_window" if "pwmsa" in lower else "window"
    elif "grid_attention" in lower:
        kind = "grid"
    elif "/layers." in lower and "/fn/" in lower:
        kind = "agent_relation" if family == "lidar_v2xvit" else ""

    if op == "LayerNormalization":
        role = "layernorm"
    elif op == "Softmax":
        role = "split_attention_gate" if "split_attn" in lower else "softmax"
    elif op == "Einsum":
        eq = str(equation).replace(" ", "")
        if "->" in eq and (eq.endswith("ij") or name.endswith("/Einsum")):
            role = "qk_matmul"
        elif name.endswith(("/Einsum_1", "/Einsum_2")):
            role = "av_matmul"
    elif op == "MatMul" and any(token in lower for token in ("qk", "attention_score")):
        role = "qk_matmul"
    elif op in {"Mul", "Div"} and any(token in lower for token in ("attention", "pwmsa")):
        role = "qk_scale"
    elif op in {"Where", "Add"} and any(token in lower for token in ("attention", "pwmsa", "/fn/")):
        role = "mask_relation_add"
    elif op == "Add" and re.search(r"(?:/layers\.\d+(?:\.\d+)?/add(?:_\d+)?|/add_\d+)$", lower):
        role = "residual_add"
    elif op == "Concat" and any(token in lower for token in ("split_attn", "fusion")):
        role = "communication_fusion"
    if not role:
        return None
    return CanonicalRole(
        model_family=family,
        canonical_role=role,
        onnx_node=name,
        onnx_op_type=op,
        attention_kind=kind,
        confidence="high" if role not in {"qk_scale", "mask_relation_add"} else "medium",
        mapping_reason="ONNX operator/topology role rule",
    )


__all__ = [
    "CANONICAL_TRANSFORMER_ROLES",
    "CanonicalRole",
    "classify_onnx_primitive",
    "classify_weighted_module",
]
