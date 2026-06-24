"""Legality checks for Transformer pruning groups and pruned modules."""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from ..tracer.pruning_group import PruningGroup


def check_transformer_group(
    group: PruningGroup,
    group_keep: list[int],
    *,
    min_heads: int = 1,
    head_align: int = 1,
    ffn_align: int = 8,
) -> dict[str, Any]:
    """Validate a Transformer pruning group before physical surgery."""
    issues: list[dict[str, Any]] = []
    gt = group.meta.get("group_type", "")
    if group.protected:
        return {"legal": False, "issues": [{"issue": "protected", "reason": group.protected_reason}]}
    if gt == "transformer_head_group":
        num_heads_after = int(group.meta.get("num_heads_after", 0))
        inner_dim_after = int(group.meta.get("inner_dim_after", len(group_keep)))
        head_dim = int(group.meta.get("head_dim", 0))
        if num_heads_after < min_heads:
            issues.append({"issue": "violates_min_heads", "num_heads_after": num_heads_after})
        if head_align > 1 and num_heads_after >= head_align and num_heads_after % head_align != 0:
            issues.append({"issue": "violates_head_align", "head_align": head_align, "num_heads_after": num_heads_after})
        if inner_dim_after != num_heads_after * head_dim:
            issues.append({"issue": "invalid_inner_dim_after", "inner_dim_after": inner_dim_after})
        if inner_dim_after % 8 != 0:
            issues.append({"issue": "violates_inner_dim_align8", "inner_dim_after": inner_dim_after})
    elif gt in ("transformer_ffn_group", "transformer_gated_ffn_group"):
        after = len(group_keep)
        if after <= 0:
            issues.append({"issue": "empty_ffn_keep"})
        if ffn_align > 1 and after >= ffn_align and after % ffn_align != 0:
            issues.append({"issue": "violates_ffn_align", "ffn_align": ffn_align, "ffn_dim_after": after})
    elif gt in ("native_mha_protected_group", "transformer_hidden_group_protected"):
        issues.append({"issue": "protected_transformer_group", "group_type": gt})
    return {"legal": not issues, "issues": issues}


def check_transformer_model_legality(
    model: nn.Module,
    *,
    protect_hidden: bool = True,
) -> dict[str, Any]:
    """Post-surgery Transformer structural scan."""
    issues: list[dict[str, Any]] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if module.weight.shape != (module.out_features, module.in_features):
                issues.append({
                    "layer": name,
                    "issue": "linear_weight_shape_mismatch",
                    "weight": list(module.weight.shape),
                    "in_features": module.in_features,
                    "out_features": module.out_features,
                })
        if isinstance(module, nn.LayerNorm) and protect_hidden:
            shape = module.normalized_shape
            if isinstance(shape, int):
                normalized = shape
            elif len(shape) == 1:
                normalized = int(shape[0])
            else:
                normalized = None
            if normalized is None or normalized <= 0:
                issues.append({"layer": name, "issue": "invalid_layernorm_shape", "normalized_shape": str(shape)})
        if isinstance(module, nn.MultiheadAttention):
            if module.embed_dim % module.num_heads != 0:
                issues.append({
                    "layer": name,
                    "issue": "native_mha_invalid_heads",
                    "embed_dim": module.embed_dim,
                    "num_heads": module.num_heads,
                })
    return {"legal": not issues, "issues": issues}

