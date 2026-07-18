"""Physical head-internal dimension pruning for CoBEVT Attention."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import torch
import torch.nn as nn
from einops import rearrange


def _as_keep_rows(values: Iterable[Iterable[int]]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(value) for value in row) for row in values)


def _validate_keep_rows(
    rows: tuple[tuple[int, ...], ...], *, label: str, original_width: int
) -> None:
    if not rows:
        raise ValueError(f"{label}_heads_empty")
    counts = {len(row) for row in rows}
    if len(counts) != 1:
        raise ValueError(f"{label}_keep_count_mismatch")
    if not next(iter(counts)):
        raise ValueError(f"{label}_keep_count_zero")
    for row in rows:
        if tuple(sorted(set(row))) != row:
            raise ValueError(f"{label}_indices_must_be_sorted_unique")
        if row[0] < 0 or row[-1] >= int(original_width):
            raise ValueError(f"{label}_index_out_of_range")


@dataclass(frozen=True)
class AttentionDimMask:
    """Independent QK and VO local-coordinate masks for every physical head."""

    qk_keep_by_head: tuple[tuple[int, ...], ...]
    vo_keep_by_head: tuple[tuple[int, ...], ...]
    original_d_qk: int = 32
    original_d_v: int = 32

    def __post_init__(self) -> None:
        qk = _as_keep_rows(self.qk_keep_by_head)
        vo = _as_keep_rows(self.vo_keep_by_head)
        object.__setattr__(self, "qk_keep_by_head", qk)
        object.__setattr__(self, "vo_keep_by_head", vo)
        if len(qk) != len(vo):
            raise ValueError("qk_vo_head_count_mismatch")
        _validate_keep_rows(qk, label="qk", original_width=self.original_d_qk)
        _validate_keep_rows(vo, label="vo", original_width=self.original_d_v)

    @property
    def heads(self) -> int:
        return len(self.qk_keep_by_head)

    @property
    def d_qk(self) -> int:
        return len(self.qk_keep_by_head[0])

    @property
    def d_v(self) -> int:
        return len(self.vo_keep_by_head[0])

    @property
    def q_keep_by_head(self) -> tuple[tuple[int, ...], ...]:
        return self.qk_keep_by_head

    @property
    def k_keep_by_head(self) -> tuple[tuple[int, ...], ...]:
        return self.qk_keep_by_head

    @property
    def v_keep_by_head(self) -> tuple[tuple[int, ...], ...]:
        return self.vo_keep_by_head

    @property
    def out_input_keep_by_head(self) -> tuple[tuple[int, ...], ...]:
        return self.vo_keep_by_head

    def to_dict(self) -> dict[str, Any]:
        return {
            "d_qk": self.d_qk,
            "d_v": self.d_v,
            "heads": self.heads,
            "original_d_qk": int(self.original_d_qk),
            "original_d_v": int(self.original_d_v),
            "qk_keep_by_head": [list(row) for row in self.qk_keep_by_head],
            "vo_keep_by_head": [list(row) for row in self.vo_keep_by_head],
        }


def _flatten_head_indices(
    rows: tuple[tuple[int, ...], ...], stride: int
) -> tuple[int, ...]:
    return tuple(
        head * int(stride) + local
        for head, values in enumerate(rows)
        for local in values
    )


class PrunableCobevtAttention(nn.Module):
    """CoBEVT Attention with explicit, independently sized Q/K/V projections."""

    def __init__(
        self,
        *,
        embed_dim: int,
        heads: int,
        d_qk: int,
        d_v: int,
        window_size: Iterable[int],
        relative_position_rows: int,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if min(embed_dim, heads, d_qk, d_v, relative_position_rows) <= 0:
            raise ValueError("invalid_cobevt_prunable_attention_dimensions")
        self.embed_dim = int(embed_dim)
        self.heads = int(heads)
        self.d_qk = int(d_qk)
        self.d_v = int(d_v)
        self.inner_dim_qk = self.heads * self.d_qk
        self.inner_dim_v = self.heads * self.d_v
        self.scale = self.d_qk**-0.5
        self.window_size = [int(value) for value in window_size]
        self.q_proj = nn.Linear(self.embed_dim, self.inner_dim_qk, bias=bias)
        self.k_proj = nn.Linear(self.embed_dim, self.inner_dim_qk, bias=bias)
        self.v_proj = nn.Linear(self.embed_dim, self.inner_dim_v, bias=bias)
        self.out_proj = nn.Linear(self.inner_dim_v, self.embed_dim, bias=bias)
        self.output_dropout = nn.Dropout(float(dropout))
        self.attend = nn.Softmax(dim=-1)
        self.relative_position_bias_table = nn.Embedding(
            int(relative_position_rows), self.heads
        )
        self.register_buffer(
            "relative_position_index", torch.empty(0, dtype=torch.long)
        )

    @classmethod
    def from_stock_attention(
        cls,
        module: nn.Module,
        *,
        qk_keep_by_head: Iterable[Iterable[int]],
        vo_keep_by_head: Iterable[Iterable[int]],
        input_keep_indices: Iterable[int] | None = None,
        output_keep_indices: Iterable[int] | None = None,
    ) -> "PrunableCobevtAttention":
        heads = int(module.heads)
        input_dim = int(module.to_qkv.in_features)
        fused_width = int(module.to_qkv.out_features)
        if fused_width % 3 or (fused_width // 3) % heads:
            raise ValueError("stock_cobevt_qkv_shape_invalid")
        projection_width = fused_width // 3
        original_dim = projection_width // heads
        mask = AttentionDimMask(
            _as_keep_rows(qk_keep_by_head),
            _as_keep_rows(vo_keep_by_head),
            original_d_qk=original_dim,
            original_d_v=original_dim,
        )
        if mask.heads != heads:
            raise ValueError("attention_mask_head_count_mismatch")
        input_keep = tuple(
            range(input_dim)
            if input_keep_indices is None
            else (int(value) for value in input_keep_indices)
        )
        output_keep = tuple(
            input_keep
            if output_keep_indices is None
            else (int(value) for value in output_keep_indices)
        )
        if len(input_keep) != len(output_keep):
            raise ValueError("attention_residual_input_output_width_mismatch")
        stock_out = module.to_out[0]
        dropout = float(getattr(module.to_out[1], "p", 0.0))
        converted = cls(
            embed_dim=len(input_keep),
            heads=heads,
            d_qk=mask.d_qk,
            d_v=mask.d_v,
            window_size=module.window_size,
            relative_position_rows=module.relative_position_bias_table.num_embeddings,
            dropout=dropout,
            bias=module.to_qkv.bias is not None,
        )
        source_weight = module.to_qkv.weight
        converted.to(device=source_weight.device, dtype=source_weight.dtype)
        input_index = torch.as_tensor(
            input_keep, dtype=torch.long, device=source_weight.device
        )
        output_index = torch.as_tensor(
            output_keep, dtype=torch.long, device=stock_out.weight.device
        )
        qk_rows = _flatten_head_indices(mask.qk_keep_by_head, original_dim)
        vo_rows = _flatten_head_indices(mask.vo_keep_by_head, original_dim)
        q_index = torch.as_tensor(qk_rows, dtype=torch.long, device=source_weight.device)
        k_index = q_index + projection_width
        v_index = torch.as_tensor(vo_rows, dtype=torch.long, device=source_weight.device)
        v_index = v_index + 2 * projection_width
        out_input_index = torch.as_tensor(
            vo_rows, dtype=torch.long, device=stock_out.weight.device
        )
        with torch.no_grad():
            converted.q_proj.weight.copy_(
                source_weight.index_select(0, q_index).index_select(1, input_index)
            )
            converted.k_proj.weight.copy_(
                source_weight.index_select(0, k_index).index_select(1, input_index)
            )
            converted.v_proj.weight.copy_(
                source_weight.index_select(0, v_index).index_select(1, input_index)
            )
            converted.out_proj.weight.copy_(
                stock_out.weight.index_select(0, output_index).index_select(
                    1, out_input_index
                )
            )
            if module.to_qkv.bias is not None:
                converted.q_proj.bias.copy_(module.to_qkv.bias.index_select(0, q_index))
                converted.k_proj.bias.copy_(module.to_qkv.bias.index_select(0, k_index))
                converted.v_proj.bias.copy_(module.to_qkv.bias.index_select(0, v_index))
            if stock_out.bias is not None:
                converted.out_proj.bias.copy_(stock_out.bias.index_select(0, output_index))
            converted.relative_position_bias_table.weight.copy_(
                module.relative_position_bias_table.weight
            )
        converted.relative_position_index = module.relative_position_index.detach().clone()
        converted.train(module.training)
        return converted

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, agent_size, height, width, window_height, window_width, _ = x.shape
        x = rearrange(x, "b l x y w1 w2 d -> (b x y) (l w1 w2) d")
        q = rearrange(
            self.q_proj(x), "b n (h d) -> b h n d", h=self.heads, d=self.d_qk
        )
        k = rearrange(
            self.k_proj(x), "b n (h d) -> b h n d", h=self.heads, d=self.d_qk
        )
        v = rearrange(
            self.v_proj(x), "b n (h d) -> b h n d", h=self.heads, d=self.d_v
        )
        sim = torch.einsum("b h i d, b h j d -> b h i j", q * self.scale, k)
        bias = self.relative_position_bias_table(self.relative_position_index)
        sim = sim + rearrange(bias, "i j h -> h i j")
        if mask is not None:
            mask = rearrange(mask, "b x y w1 w2 e l -> (b x y) e (l w1 w2)")
            sim = sim.masked_fill(mask.unsqueeze(1) == 0, -float("inf"))
        attn = self.attend(sim)
        out = torch.einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(
            out,
            "b h (l w1 w2) d -> b l w1 w2 (h d)",
            l=agent_size,
            w1=window_height,
            w2=window_width,
        )
        out = self.output_dropout(self.out_proj(out))
        return rearrange(
            out,
            "(b x y) l w1 w2 d -> b l x y w1 w2 d",
            b=batch,
            x=height,
            y=width,
        )


def _stock_attention_modules(model: nn.Module) -> list[tuple[str, nn.Module]]:
    return [
        (name, module)
        for name, module in model.named_modules()
        if name.startswith("fusion_net")
        and module.__class__.__name__ == "Attention"
        and hasattr(module, "to_qkv")
    ]


def _set_submodule(model: nn.Module, path: str, replacement: nn.Module) -> None:
    parent_path, leaf = path.rsplit(".", 1)
    parent = model.get_submodule(parent_path)
    if leaf.isdigit() and isinstance(parent, (nn.ModuleList, nn.Sequential)):
        parent[int(leaf)] = replacement
    else:
        setattr(parent, leaf, replacement)


def uniform_attention_masks(
    model: nn.Module, *, d_qk: int, d_v: int
) -> dict[str, AttentionDimMask]:
    rows = _stock_attention_modules(model)
    if not rows:
        raise RuntimeError("cobevt_stock_attention_modules_missing")
    result: dict[str, AttentionDimMask] = {}
    for name, module in rows:
        heads = int(module.heads)
        projection = int(module.to_qkv.out_features) // 3
        original = projection // heads
        result[name] = AttentionDimMask(
            tuple(tuple(range(int(d_qk))) for _ in range(heads)),
            tuple(tuple(range(int(d_v))) for _ in range(heads)),
            original_d_qk=original,
            original_d_v=original,
        )
    return result


def attention_masks_structure_hash(masks: Mapping[str, AttentionDimMask]) -> str:
    payload = {
        "modules": {name: mask.to_dict() for name, mask in sorted(masks.items())},
        "recipe": "cobevt-attention-dim-pruning-v1",
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class AttentionBottleneckPruneReport:
    passed: bool
    attention_module_count: int
    original_parameter_count: int
    predicted_parameter_count: int
    physical_parameter_count: int
    structure_hash: str
    operations: tuple[dict[str, Any], ...]
    issues: tuple[dict[str, Any], ...]


def materialize_attention_bottleneck(
    model: nn.Module, masks: Mapping[str, AttentionDimMask]
) -> AttentionBottleneckPruneReport:
    rows = _stock_attention_modules(model)
    names = {name for name, _ in rows}
    if names != set(masks):
        missing = sorted(names - set(masks))
        extra = sorted(set(masks) - names)
        raise ValueError(f"attention_mask_inventory_mismatch:missing={missing}:extra={extra}")
    original_count = sum(int(parameter.numel()) for parameter in model.parameters())
    operations: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    for name, module in rows:
        replacement = PrunableCobevtAttention.from_stock_attention(
            module,
            qk_keep_by_head=masks[name].qk_keep_by_head,
            vo_keep_by_head=masks[name].vo_keep_by_head,
        )
        _set_submodule(model, name, replacement)
        operations.append(
            {
                "module": name,
                "before": {
                    "d_qk": int(module.to_qkv.out_features // 3 // module.heads),
                    "d_v": int(module.to_qkv.out_features // 3 // module.heads),
                    "embed_dim": int(module.to_qkv.in_features),
                },
                "after": {
                    "d_qk": replacement.d_qk,
                    "d_v": replacement.d_v,
                    "embed_dim": replacement.embed_dim,
                },
            }
        )
    physical_count = sum(int(parameter.numel()) for parameter in model.parameters())
    for name, module in model.named_modules():
        if isinstance(module, PrunableCobevtAttention):
            if module.q_proj.out_features != module.heads * module.d_qk:
                issues.append({"module": name, "reason": "q_projection_shape_mismatch"})
            if module.v_proj.out_features != module.heads * module.d_v:
                issues.append({"module": name, "reason": "v_projection_shape_mismatch"})
            if module.out_proj.in_features != module.heads * module.d_v:
                issues.append({"module": name, "reason": "out_projection_shape_mismatch"})
            if not math.isclose(module.scale, module.d_qk**-0.5):
                issues.append({"module": name, "reason": "attention_scale_mismatch"})
    return AttentionBottleneckPruneReport(
        passed=not issues and len(rows) == 6,
        attention_module_count=len(rows),
        original_parameter_count=original_count,
        predicted_parameter_count=physical_count,
        physical_parameter_count=physical_count,
        structure_hash=attention_masks_structure_hash(masks),
        operations=tuple(operations),
        issues=tuple(issues),
    )


__all__ = [
    "AttentionBottleneckPruneReport",
    "AttentionDimMask",
    "PrunableCobevtAttention",
    "attention_masks_structure_hash",
    "materialize_attention_bottleneck",
    "uniform_attention_masks",
]
