"""Build Transformer-aware pruning groups from analyzer results."""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from .pruning_group import PruningGroup
from .transformer_analyzer import AttentionPattern, FFNPattern, TransformerAnalysis, TransformerAnalyzer
from ..pruning.transformer_pruning_fns import (
    make_set_attention_metadata_fn,
    prune_transformer_linear_in,
    prune_transformer_linear_out,
)


TRANSFORMER_GROUP_TYPES = (
    "transformer_head_group",
    "transformer_qkv_group",
    "transformer_ffn_group",
    "transformer_gated_ffn_group",
    "transformer_hidden_group_protected",
    "native_mha_protected_group",
)


def _range(n: int) -> list[int]:
    return list(range(int(n)))


class TransformerGroupBuilder:
    """Create ``PruningGroup`` objects for recognized Transformer structures."""

    def __init__(
        self,
        model: nn.Module,
        *,
        enable_native_mha_pruning: bool = False,
        min_heads: int = 1,
        head_align: int = 1,
        ffn_align: int = 8,
        protect_hidden: bool = True,
    ):
        self.model = model
        self.modules = dict(model.named_modules())
        self.enable_native_mha_pruning = bool(enable_native_mha_pruning)
        self.min_heads = int(min_heads)
        self.head_align = int(head_align)
        self.ffn_align = int(ffn_align)
        self.protect_hidden = bool(protect_hidden)
        self.analysis: TransformerAnalysis | None = None

    def build(self, analysis: TransformerAnalysis | None = None) -> list[PruningGroup]:
        if analysis is None:
            analysis = TransformerAnalyzer(
                self.model,
                min_heads=self.min_heads,
                enable_native_mha_pruning=self.enable_native_mha_pruning,
                protect_hidden=self.protect_hidden,
            ).analyze()
        self.analysis = analysis
        groups: list[PruningGroup] = []
        for pat in analysis.attention_patterns:
            groups.extend(self._build_attention_groups(pat))
        for pat in analysis.ffn_patterns:
            groups.append(self._build_ffn_group(pat))
        for pat in analysis.native_mha:
            groups.append(self._build_native_mha_group(pat))
        if self.protect_hidden:
            for i, meta in enumerate(analysis.hidden_protected):
                groups.append(self._build_hidden_group(i, meta))
        return groups

    def _attention_parent(self, pat: AttentionPattern) -> nn.Module:
        module = self.modules.get(pat.module_name)
        return module if module is not None else self.model

    def _build_attention_groups(self, pat: AttentionPattern) -> list[PruningGroup]:
        groups: list[PruningGroup] = []
        if pat.num_heads <= 0 or pat.head_dim <= 0:
            g = PruningGroup(f"transformer::{pat.attention_name}::unsupported", num_channels=0)
            g.meta.update(self._attention_meta(pat, group_type="transformer_qkv_group"))
            g.protect("qkv_pattern_not_confident")
            return [g]
        for head_id in range(pat.num_heads):
            start = head_id * pat.head_dim
            stop = (head_id + 1) * pat.head_dim
            head_idxs = list(range(start, stop))
            kept_heads = pat.num_heads - 1
            inner_after = kept_heads * pat.head_dim
            g = PruningGroup(
                group_id=f"transformer::{pat.attention_name}::head{head_id}",
                num_channels=pat.inner_dim,
                meta=self._attention_meta(
                    pat,
                    group_type="transformer_head_group",
                    head_id=head_id,
                    num_heads_after=kept_heads,
                    inner_dim_after=inner_after,
                ),
            )
            if pat.protected:
                g.protect(pat.protected_reason or "unsupported_transformer_pattern")
                groups.append(g)
                continue
            reason = self._check_head_legality(pat, kept_heads, inner_after)
            if reason:
                g.protect(reason)
                groups.append(g)
                continue
            keep_inner = [i for i in range(pat.inner_dim) if i not in set(head_idxs)]
            if pat.qkv_type == "split_qkv":
                for name in (pat.q_proj_name, pat.k_proj_name, pat.v_proj_name):
                    g.add_dep(name, self.modules[name], prune_transformer_linear_out, "out",
                              idxs=_range(pat.inner_dim), reason="transformer_head_qkv_out")
                g.add_dep(pat.out_proj_name, self.modules[pat.out_proj_name], prune_transformer_linear_in, "in",
                          idxs=_range(pat.inner_dim), reason="transformer_head_out_proj_in")
            elif pat.qkv_type == "fused_qkv":
                g.add_dep(
                    pat.qkv_proj_name,
                    self.modules[pat.qkv_proj_name],
                    prune_transformer_linear_out,
                    "out",
                    idxs=_range(pat.inner_dim * 3),
                    idx_transform=lambda keep, inner=pat.inner_dim: (
                        list(keep)
                        + [inner + i for i in keep]
                        + [2 * inner + i for i in keep]
                    ),
                    reason="transformer_fused_qkv_out",
                )
                g.add_dep(
                    pat.out_proj_name,
                    self.modules[pat.out_proj_name],
                    prune_transformer_linear_in,
                    "in",
                    idxs=_range(pat.inner_dim),
                    reason="transformer_fused_out_proj_in",
                )
            parent = self._attention_parent(pat)
            g.add_dep(
                pat.module_name or pat.attention_name,
                parent,
                make_set_attention_metadata_fn(
                    num_heads_after=kept_heads,
                    inner_dim_after=inner_after,
                    head_dim=pat.head_dim,
                ),
                "meta",
                idxs=[],
                idx_transform=lambda _keep: [0],
                reason="transformer_attention_metadata",
            )
            g.meta["keep_indices"] = keep_inner
            groups.append(g)
        return groups

    def _check_head_legality(self, pat: AttentionPattern, kept_heads: int, inner_after: int) -> str:
        if kept_heads < self.min_heads:
            return "violates_min_heads"
        if self.head_align > 1 and kept_heads >= self.head_align and kept_heads % self.head_align != 0:
            return "violates_head_align"
        if inner_after != kept_heads * pat.head_dim:
            return "invalid_inner_dim_after"
        if inner_after % 8 != 0:
            return "violates_inner_dim_align8"
        return ""

    def _build_ffn_group(self, pat: FFNPattern) -> PruningGroup:
        group_type = "transformer_gated_ffn_group" if pat.ffn_type == "gated" else "transformer_ffn_group"
        g = PruningGroup(
            group_id=f"transformer::{pat.ffn_name}::{pat.ffn_type}_ffn",
            num_channels=pat.ffn_dim,
            meta={
                "group_type": group_type,
                "transformer_block_name": pat.block_name,
                "ffn_name": pat.ffn_name,
                "ffn_type": pat.ffn_type,
                "hidden_dim": pat.hidden_dim,
                "ffn_dim_before": pat.ffn_dim,
                "ffn_dim_after": pat.ffn_dim,
                "ffn_align": self.ffn_align,
            },
        )
        if pat.ffn_dim <= self.ffn_align:
            g.protect("ffn_dim_too_small")
            return g
        if pat.ffn_type == "standard":
            g.add_dep(pat.fc1_name, self.modules[pat.fc1_name], prune_transformer_linear_out, "out",
                      idxs=_range(pat.ffn_dim), reason="transformer_ffn_fc1_out")
            g.add_dep(pat.fc2_name, self.modules[pat.fc2_name], prune_transformer_linear_in, "in",
                      idxs=_range(pat.ffn_dim), reason="transformer_ffn_fc2_in")
        else:
            g.add_dep(pat.gate_proj_name, self.modules[pat.gate_proj_name], prune_transformer_linear_out, "out",
                      idxs=_range(pat.ffn_dim), reason="transformer_gated_gate_out")
            g.add_dep(pat.up_proj_name, self.modules[pat.up_proj_name], prune_transformer_linear_out, "out",
                      idxs=_range(pat.ffn_dim), reason="transformer_gated_up_out")
            g.add_dep(pat.down_proj_name, self.modules[pat.down_proj_name], prune_transformer_linear_in, "in",
                      idxs=_range(pat.ffn_dim), reason="transformer_gated_down_in")
        return g

    def _build_native_mha_group(self, pat: AttentionPattern) -> PruningGroup:
        g = PruningGroup(
            group_id=f"transformer::{pat.attention_name}::native_mha",
            num_channels=pat.inner_dim,
            meta=self._attention_meta(pat, group_type="native_mha_protected_group"),
        )
        if pat.module_name in self.modules:
            g.add_dep(pat.module_name, self.modules[pat.module_name], lambda m, keep: {"axis": "native_mha_noop"}, "meta")
        g.protect(pat.protected_reason or "native_mha_default_protected")
        return g

    def _build_hidden_group(self, idx: int, meta: dict[str, Any]) -> PruningGroup:
        g = PruningGroup(
            group_id=f"transformer::hidden::{idx}",
            num_channels=int(meta.get("hidden_dim", 0)),
            meta={
                "group_type": "transformer_hidden_group_protected",
                **meta,
            },
        )
        name = meta.get("module_name")
        if name in self.modules:
            g.add_dep(name, self.modules[name], lambda m, keep: {"axis": "hidden_noop"}, "out")
        g.protect(meta.get("protected_reason", "hidden_dim_global_dependency"))
        return g

    def _attention_meta(
        self,
        pat: AttentionPattern,
        *,
        group_type: str,
        head_id: int | None = None,
        num_heads_after: int | None = None,
        inner_dim_after: int | None = None,
    ) -> dict[str, Any]:
        return {
            "group_type": group_type,
            "transformer_block_name": pat.block_name,
            "attention_name": pat.attention_name,
            "attention_type": "native_mha" if pat.qkv_type == "native_mha" else "custom_attention",
            "qkv_type": pat.qkv_type,
            "num_heads_before": pat.num_heads,
            "num_heads_after": pat.num_heads if num_heads_after is None else num_heads_after,
            "head_dim": pat.head_dim,
            "head_id": "" if head_id is None else head_id,
            "inner_dim_before": pat.inner_dim,
            "inner_dim_after": pat.inner_dim if inner_dim_after is None else inner_dim_after,
            "hidden_dim": pat.hidden_dim,
            "q_proj_name": pat.q_proj_name,
            "k_proj_name": pat.k_proj_name,
            "v_proj_name": pat.v_proj_name,
            "qkv_proj_name": pat.qkv_proj_name,
            "out_proj_name": pat.out_proj_name,
            "module_name": pat.module_name,
        }


def build_transformer_pruning_groups(
    model: nn.Module,
    **kwargs: Any,
) -> tuple[list[PruningGroup], TransformerAnalysis]:
    builder = TransformerGroupBuilder(model, **kwargs)
    analysis = TransformerAnalyzer(
        model,
        min_heads=kwargs.get("min_heads", 1),
        enable_native_mha_pruning=kwargs.get("enable_native_mha_pruning", False),
        protect_hidden=kwargs.get("protect_hidden", True),
    ).analyze()
    return builder.build(analysis), analysis
