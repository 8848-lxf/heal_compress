"""Capability-only CoBEVT Attention resizing, including zero-padded expansion."""

from __future__ import annotations

import math
import hashlib
import json
from typing import Any

import torch
from torch import nn

from .attention_dim_pruning import (
    AttentionBottleneckPruneReport,
    PrunableCobevtAttention,
)


def resize_stock_attention_for_capability(
    module: nn.Module, *, d_qk: int, d_v: int
) -> PrunableCobevtAttention:
    """Resize head-internal axes while preserving the external embedding width.

    Widths above the checkpoint width are initialized by zero padding. Q/K
    surviving rows are rescaled so the changed ``1/sqrt(d_qk)`` factor leaves
    the original logits unchanged. This is an integration capability control,
    not a learned expansion recipe.
    """

    heads = int(module.heads)
    embed_dim = int(module.to_qkv.in_features)
    fused_width = int(module.to_qkv.out_features)
    if fused_width % (3 * heads):
        raise ValueError("stock_cobevt_qkv_shape_invalid")
    original = fused_width // 3 // heads
    if min(int(d_qk), int(d_v)) <= 0:
        raise ValueError("capability_head_dimension_must_be_positive")
    source_out = module.to_out[0]
    dropout = float(getattr(module.to_out[1], "p", 0.0))
    converted = PrunableCobevtAttention(
        embed_dim=embed_dim,
        heads=heads,
        d_qk=int(d_qk),
        d_v=int(d_v),
        window_size=module.window_size,
        relative_position_rows=module.relative_position_bias_table.num_embeddings,
        dropout=dropout,
        bias=module.to_qkv.bias is not None,
    )
    source_weight = module.to_qkv.weight
    converted.to(device=source_weight.device, dtype=source_weight.dtype)
    projection_width = heads * original
    q_source = source_weight[:projection_width].reshape(
        heads, original, embed_dim
    )
    k_source = source_weight[
        projection_width : 2 * projection_width
    ].reshape(heads, original, embed_dim)
    v_source = source_weight[2 * projection_width :].reshape(
        heads, original, embed_dim
    )
    q_target = converted.q_proj.weight.reshape(heads, int(d_qk), embed_dim)
    k_target = converted.k_proj.weight.reshape(heads, int(d_qk), embed_dim)
    v_target = converted.v_proj.weight.reshape(heads, int(d_v), embed_dim)
    out_source = source_out.weight.reshape(embed_dim, heads, original)
    out_target = converted.out_proj.weight.reshape(embed_dim, heads, int(d_v))
    qk_copy = min(original, int(d_qk))
    v_copy = min(original, int(d_v))
    qk_expansion_scale = (
        math.pow(float(d_qk) / float(original), 0.25)
        if int(d_qk) > original
        else 1.0
    )
    with torch.no_grad():
        q_target.zero_()
        k_target.zero_()
        v_target.zero_()
        out_target.zero_()
        q_target[:, :qk_copy].copy_(
            q_source[:, :qk_copy] * qk_expansion_scale
        )
        k_target[:, :qk_copy].copy_(
            k_source[:, :qk_copy] * qk_expansion_scale
        )
        v_target[:, :v_copy].copy_(v_source[:, :v_copy])
        out_target[:, :, :v_copy].copy_(out_source[:, :, :v_copy])
        if module.to_qkv.bias is not None:
            source_bias = module.to_qkv.bias.reshape(3, heads, original)
            q_bias = converted.q_proj.bias.reshape(heads, int(d_qk))
            k_bias = converted.k_proj.bias.reshape(heads, int(d_qk))
            v_bias = converted.v_proj.bias.reshape(heads, int(d_v))
            q_bias.zero_()
            k_bias.zero_()
            v_bias.zero_()
            q_bias[:, :qk_copy].copy_(
                source_bias[0, :, :qk_copy] * qk_expansion_scale
            )
            k_bias[:, :qk_copy].copy_(
                source_bias[1, :, :qk_copy] * qk_expansion_scale
            )
            v_bias[:, :v_copy].copy_(source_bias[2, :, :v_copy])
        if source_out.bias is not None:
            converted.out_proj.bias.copy_(source_out.bias)
        converted.relative_position_bias_table.weight.copy_(
            module.relative_position_bias_table.weight
        )
    converted.relative_position_index = (
        module.relative_position_index.detach().clone()
    )
    converted.train(module.training)
    return converted


def materialize_model_attention_for_capability(
    model: nn.Module, *, d_qk: int, d_v: int
) -> AttentionBottleneckPruneReport:
    """Replace every stock CoBEVT fusion Attention with an explicit resize."""

    source_rows = [
        (name, module)
        for name, module in model.named_modules()
        if name.startswith("fusion_net")
        and module.__class__.__name__ == "Attention"
        and hasattr(module, "to_qkv")
    ]
    if not source_rows:
        raise RuntimeError("cobevt_stock_attention_modules_missing")
    before = sum(parameter.numel() for parameter in model.parameters())
    operations = []
    for name, source in source_rows:
        replacement = resize_stock_attention_for_capability(
            source, d_qk=int(d_qk), d_v=int(d_v)
        )
        parent_path, leaf = name.rsplit(".", 1)
        parent = model.get_submodule(parent_path)
        if leaf.isdigit() and isinstance(parent, (nn.ModuleList, nn.Sequential)):
            parent[int(leaf)] = replacement
        else:
            setattr(parent, leaf, replacement)
        operations.append(
            {
                "module_path": name,
                "d_qk": int(d_qk),
                "d_v": int(d_v),
                "embed_dim": int(replacement.embed_dim),
                "q_projection_out": int(replacement.inner_dim_qk),
                "v_projection_out": int(replacement.inner_dim_v),
                "out_projection_in": int(replacement.inner_dim_v),
                "out_projection_out": int(replacement.embed_dim),
            }
        )
    after = sum(parameter.numel() for parameter in model.parameters())
    structure_payload = {
        "d_qk": int(d_qk),
        "d_v": int(d_v),
        "modules": operations,
        "recipe": "cobevt-head-dim-capability-resize-v1",
    }
    structure_hash = hashlib.sha256(
        json.dumps(
            structure_payload, sort_keys=True, separators=(",", ":")
        ).encode("ascii")
    ).hexdigest()
    issues = []
    if len(operations) != 6:
        issues.append(
            {
                "reason": "attention_module_count_mismatch",
                "actual": len(operations),
                "expected": 6,
            }
        )
    if not all(
        row["embed_dim"] == 256 and row["out_projection_out"] == 256
        for row in operations
    ):
        issues.append({"reason": "external_embedding_changed"})
    return AttentionBottleneckPruneReport(
        passed=not issues,
        attention_module_count=len(operations),
        original_parameter_count=int(before),
        predicted_parameter_count=int(after),
        physical_parameter_count=int(after),
        structure_hash=structure_hash,
        operations=tuple(operations),
        issues=tuple(issues),
    )


__all__ = [
    "materialize_model_attention_for_capability",
    "resize_stock_attention_for_capability",
]
