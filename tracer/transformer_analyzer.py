"""Best-effort Transformer structure analyzer.

The analyzer recognizes common custom Transformer blocks without depending on
model-specific classes:

* split Q/K/V projections + output projection
* fused QKV projection + output projection
* standard FFN (fc1/linear1 -> fc2/linear2)
* gated FFN (gate/up -> down)
* native ``nn.MultiheadAttention`` protected modules

Uncertain patterns are reported as protected rather than pruned.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch.nn as nn


SPLIT_Q_NAMES = ("q_proj", "query", "q")
SPLIT_K_NAMES = ("k_proj", "key", "k")
SPLIT_V_NAMES = ("v_proj", "value", "v")
OUT_NAMES = ("out_proj", "o_proj", "proj", "o")
FUSED_QKV_NAMES = ("qkv", "in_proj", "qkv_proj")
FC1_NAMES = ("fc1", "linear1", "mlp.fc1")
FC2_NAMES = ("fc2", "linear2", "mlp.fc2")
GATE_NAMES = ("gate_proj", "gate", "w1")
UP_NAMES = ("up_proj", "up", "w3")
DOWN_NAMES = ("down_proj", "down", "w2")


def _leaf(name: str) -> str:
    return name.rsplit(".", 1)[-1].lower()


def _ends_with_any(name: str, candidates: tuple[str, ...]) -> bool:
    low = name.lower()
    leaf = _leaf(low)
    return leaf in candidates or any(low.endswith("." + c) for c in candidates)


def _parent_name(name: str) -> str:
    return name.rsplit(".", 1)[0] if "." in name else ""


def _ancestor_names(name: str) -> list[str]:
    parts = name.split(".")
    return [".".join(parts[:i]) for i in range(1, len(parts))]


def _common_parent(names: list[str]) -> str:
    if not names:
        return ""
    split = [n.split(".")[:-1] for n in names]
    common: list[str] = []
    for vals in zip(*split):
        if len(set(vals)) == 1:
            common.append(vals[0])
        else:
            break
    return ".".join(common)


@dataclass
class AttentionPattern:
    block_name: str
    attention_name: str
    qkv_type: str
    q_proj_name: str = ""
    k_proj_name: str = ""
    v_proj_name: str = ""
    qkv_proj_name: str = ""
    out_proj_name: str = ""
    num_heads: int = 0
    head_dim: int = 0
    inner_dim: int = 0
    hidden_dim: int = 0
    protected: bool = False
    protected_reason: str = ""
    module_name: str = ""


@dataclass
class FFNPattern:
    block_name: str
    ffn_name: str
    ffn_type: str
    fc1_name: str = ""
    fc2_name: str = ""
    gate_proj_name: str = ""
    up_proj_name: str = ""
    down_proj_name: str = ""
    hidden_dim: int = 0
    ffn_dim: int = 0
    protected: bool = False
    protected_reason: str = ""


@dataclass
class TransformerAnalysis:
    attention_patterns: list[AttentionPattern] = field(default_factory=list)
    ffn_patterns: list[FFNPattern] = field(default_factory=list)
    native_mha: list[AttentionPattern] = field(default_factory=list)
    hidden_protected: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "num_attention_patterns": len(self.attention_patterns),
            "num_ffn_patterns": len(self.ffn_patterns),
            "num_native_mha": len(self.native_mha),
            "num_hidden_protected": len(self.hidden_protected),
            "warnings": self.warnings,
        }


class TransformerAnalyzer:
    """Recognize safe Transformer pruning patterns in a live model."""

    def __init__(
        self,
        model: nn.Module,
        *,
        min_heads: int = 1,
        enable_native_mha_pruning: bool = False,
        protect_hidden: bool = True,
    ):
        self.model = model
        self.modules = dict(model.named_modules())
        self.linears = {n: m for n, m in self.modules.items() if isinstance(m, nn.Linear)}
        self.min_heads = int(min_heads)
        self.enable_native_mha_pruning = bool(enable_native_mha_pruning)
        self.protect_hidden = bool(protect_hidden)

    def analyze(self) -> TransformerAnalysis:
        result = TransformerAnalysis()
        result.native_mha.extend(self._find_native_mha())
        result.attention_patterns.extend(self._find_split_attention())
        result.attention_patterns.extend(self._find_fused_attention())
        result.ffn_patterns.extend(self._find_standard_ffn())
        result.ffn_patterns.extend(self._find_gated_ffn())
        if self.protect_hidden:
            result.hidden_protected.extend(self._find_hidden_protected())
        return result

    def _infer_heads(self, module_name: str, inner_dim: int) -> tuple[int, int]:
        candidates = [module_name, *_ancestor_names(module_name)]
        for name in reversed(candidates):
            module = self.modules.get(name)
            if module is None:
                continue
            num_heads = getattr(module, "num_heads", getattr(module, "n_heads", getattr(module, "heads", None)))
            head_dim = getattr(module, "head_dim", None)
            if num_heads:
                num_heads = int(num_heads)
                if inner_dim % num_heads == 0:
                    return num_heads, inner_dim // num_heads
            if head_dim:
                head_dim = int(head_dim)
                if head_dim > 0 and inner_dim % head_dim == 0:
                    return inner_dim // head_dim, head_dim
        # Conservative fallback used by toy/custom modules with no metadata.
        for h in (16, 12, 8, 6, 4, 3, 2, 1):
            if inner_dim % h == 0:
                return h, inner_dim // h
        return 1, inner_dim

    def _find_native_mha(self) -> list[AttentionPattern]:
        out: list[AttentionPattern] = []
        for name, module in self.modules.items():
            if not isinstance(module, nn.MultiheadAttention):
                continue
            head_dim = int(module.embed_dim // module.num_heads)
            pat = AttentionPattern(
                block_name=_parent_name(name),
                attention_name=name,
                module_name=name,
                qkv_type="native_mha",
                num_heads=int(module.num_heads),
                head_dim=head_dim,
                inner_dim=int(module.embed_dim),
                hidden_dim=int(module.embed_dim),
                protected=not self.enable_native_mha_pruning,
                protected_reason="" if self.enable_native_mha_pruning else "native_mha_default_protected",
            )
            out.append(pat)
        return out

    def _find_split_attention(self) -> list[AttentionPattern]:
        q_names = [n for n in self.linears if _ends_with_any(n, SPLIT_Q_NAMES)]
        out: list[AttentionPattern] = []
        for qn in q_names:
            parent = _parent_name(qn)
            siblings = {n: m for n, m in self.linears.items() if _parent_name(n) == parent}
            kn = next((n for n in siblings if _ends_with_any(n, SPLIT_K_NAMES)), "")
            vn = next((n for n in siblings if _ends_with_any(n, SPLIT_V_NAMES)), "")
            on = next((n for n in siblings if _ends_with_any(n, OUT_NAMES)), "")
            if not (kn and vn and on):
                continue
            q, k, v, o = self.linears[qn], self.linears[kn], self.linears[vn], self.linears[on]
            if not (q.out_features == k.out_features == v.out_features == o.in_features):
                continue
            if not (q.in_features == k.in_features == v.in_features == o.out_features):
                reason = "attention_forward_shape_not_safe"
            else:
                reason = ""
            inner_dim = int(q.out_features)
            num_heads, head_dim = self._infer_heads(parent, inner_dim)
            protected = bool(reason or num_heads <= self.min_heads or inner_dim % num_heads != 0)
            out.append(AttentionPattern(
                block_name=_parent_name(parent),
                attention_name=parent,
                qkv_type="split_qkv",
                q_proj_name=qn,
                k_proj_name=kn,
                v_proj_name=vn,
                out_proj_name=on,
                num_heads=num_heads,
                head_dim=head_dim,
                inner_dim=inner_dim,
                hidden_dim=int(o.out_features),
                protected=protected,
                protected_reason=reason if reason else ("min_heads" if protected else ""),
                module_name=parent,
            ))
        return out

    def _find_fused_attention(self) -> list[AttentionPattern]:
        out: list[AttentionPattern] = []
        qkv_names = [n for n in self.linears if _ends_with_any(n, FUSED_QKV_NAMES)]
        for qn in qkv_names:
            parent = _parent_name(qn)
            siblings = {n: m for n, m in self.linears.items() if _parent_name(n) == parent}
            on = next((n for n in siblings if n != qn and _ends_with_any(n, OUT_NAMES)), "")
            if not on:
                continue
            qkv, proj = self.linears[qn], self.linears[on]
            if qkv.out_features % 3 != 0:
                continue
            inner_dim = qkv.out_features // 3
            if proj.in_features != inner_dim:
                continue
            num_heads, head_dim = self._infer_heads(parent, inner_dim)
            protected = bool(num_heads <= self.min_heads or inner_dim % num_heads != 0)
            out.append(AttentionPattern(
                block_name=_parent_name(parent),
                attention_name=parent,
                qkv_type="fused_qkv",
                qkv_proj_name=qn,
                out_proj_name=on,
                num_heads=num_heads,
                head_dim=head_dim,
                inner_dim=int(inner_dim),
                hidden_dim=int(proj.out_features),
                protected=protected,
                protected_reason="min_heads" if protected else "",
                module_name=parent,
            ))
        return out

    def _find_standard_ffn(self) -> list[FFNPattern]:
        out: list[FFNPattern] = []
        for n, fc1 in self.linears.items():
            if not _ends_with_any(n, FC1_NAMES):
                continue
            parent = _parent_name(n)
            fc2n = next((m for m in self.linears if _parent_name(m) == parent and _ends_with_any(m, FC2_NAMES)), "")
            if not fc2n:
                continue
            fc2 = self.linears[fc2n]
            if fc1.out_features != fc2.in_features:
                continue
            out.append(FFNPattern(
                block_name=_parent_name(parent),
                ffn_name=parent,
                ffn_type="standard",
                fc1_name=n,
                fc2_name=fc2n,
                hidden_dim=int(fc2.out_features),
                ffn_dim=int(fc1.out_features),
            ))
        return out

    def _find_gated_ffn(self) -> list[FFNPattern]:
        out: list[FFNPattern] = []
        for gate_n, gate in self.linears.items():
            if not _ends_with_any(gate_n, GATE_NAMES):
                continue
            parent = _parent_name(gate_n)
            up_n = next((m for m in self.linears if _parent_name(m) == parent and _ends_with_any(m, UP_NAMES)), "")
            down_n = next((m for m in self.linears if _parent_name(m) == parent and _ends_with_any(m, DOWN_NAMES)), "")
            if not (up_n and down_n):
                continue
            up, down = self.linears[up_n], self.linears[down_n]
            if not (gate.out_features == up.out_features == down.in_features):
                continue
            out.append(FFNPattern(
                block_name=_parent_name(parent),
                ffn_name=parent,
                ffn_type="gated",
                gate_proj_name=gate_n,
                up_proj_name=up_n,
                down_proj_name=down_n,
                hidden_dim=int(down.out_features),
                ffn_dim=int(gate.out_features),
            ))
        return out

    def _find_hidden_protected(self) -> list[dict[str, Any]]:
        protected: list[dict[str, Any]] = []
        for name, module in self.modules.items():
            if isinstance(module, nn.LayerNorm):
                shape = module.normalized_shape
                hidden_dim = int(shape[0]) if isinstance(shape, (tuple, list)) else int(shape)
                protected.append({
                    "module_name": name,
                    "module_type": "LayerNorm",
                    "group_type": "transformer_hidden_group_protected",
                    "hidden_dim": hidden_dim,
                    "protected_reason": "hidden_dim_global_dependency",
                })
        return protected

