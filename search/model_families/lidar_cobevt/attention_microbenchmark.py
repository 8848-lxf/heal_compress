"""Real-shape CoBEVT Attention FP16 and explicit-QDQ microbenchmarks."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional

from .attention_dim_pruning import PrunableCobevtAttention


@dataclass(frozen=True)
class AttentionMicrobenchmarkSpec:
    variant: str
    d_qk: int
    d_v: int

    @property
    def spec_id(self) -> str:
        return f"{self.variant}_qk{self.d_qk}_v{self.d_v}"


def attention_microbenchmark_specs() -> tuple[AttentionMicrobenchmarkSpec, ...]:
    uniform = tuple(
        AttentionMicrobenchmarkSpec("uniform", width, width)
        for width in (8, 12, 16, 20, 24, 28, 32)
    )
    asymmetric = (
        AttentionMicrobenchmarkSpec("qk_only", 24, 32),
        AttentionMicrobenchmarkSpec("qk_only", 16, 32),
        AttentionMicrobenchmarkSpec("v_only_control", 32, 16),
    )
    return uniform + asymmetric


def _scale(value: torch.Tensor) -> torch.Tensor:
    maximum = value.detach().abs().amax().float()
    return torch.clamp(maximum / 127.0, min=torch.finfo(torch.float32).eps)


def _per_output_scales(weight: torch.Tensor) -> torch.Tensor:
    flat = weight.detach().float().abs().reshape(weight.shape[0], -1)
    return torch.clamp(
        flat.amax(dim=1) / 127.0, min=torch.finfo(torch.float32).eps
    )


def _fake_quant_tensor(value: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return torch.fake_quantize_per_tensor_affine(
        value, scale, 0, -128, 127
    )


class ExplicitQDQLinear(nn.Module):
    def __init__(self, source: nn.Linear) -> None:
        super().__init__()
        self.weight = nn.Parameter(source.weight.detach().clone())
        self.bias = (
            nn.Parameter(source.bias.detach().clone())
            if source.bias is not None
            else None
        )
        self.register_buffer("input_scale", torch.ones((), dtype=torch.float32))
        self.register_buffer("output_scale", torch.ones((), dtype=torch.float32))
        self.register_buffer(
            "weight_scale", _per_output_scales(self.weight).to(torch.float32)
        )
        self.register_buffer(
            "weight_zero_point",
            torch.zeros(self.weight.shape[0], dtype=torch.int32),
        )
        self.quant_enabled = False

    def calibrate(self, value: torch.Tensor) -> torch.Tensor:
        output = functional.linear(value, self.weight, self.bias)
        self.input_scale.copy_(_scale(value))
        self.output_scale.copy_(_scale(output))
        self.weight_scale.copy_(_per_output_scales(self.weight))
        return output

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if not self.quant_enabled:
            return functional.linear(value, self.weight, self.bias)
        activation = _fake_quant_tensor(value, self.input_scale)
        weight = torch.fake_quantize_per_channel_affine(
            self.weight,
            self.weight_scale,
            self.weight_zero_point,
            0,
            -128,
            127,
        )
        output = functional.linear(activation, weight, self.bias)
        return _fake_quant_tensor(output, self.output_scale)


class _QKMatMul(nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = float(scale)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        return torch.matmul(q * self.scale, k.transpose(-1, -2))


class _AVMatMul(nn.Module):
    def forward(self, attention: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        return torch.matmul(attention, value)


class ExplicitQDQAttentionGraph(nn.Module):
    """Attention graph whose linear and AV inputs carry explicit Q/DQ."""

    def __init__(
        self,
        source: PrunableCobevtAttention,
        *,
        use_mask_rpe: bool,
    ) -> None:
        super().__init__()
        self.embed_dim = int(source.embed_dim)
        self.heads = int(source.heads)
        self.d_qk = int(source.d_qk)
        self.d_v = int(source.d_v)
        self.scale = float(source.scale)
        self.use_mask_rpe = bool(use_mask_rpe)
        self.q_proj = ExplicitQDQLinear(source.q_proj)
        self.k_proj = ExplicitQDQLinear(source.k_proj)
        self.v_proj = ExplicitQDQLinear(source.v_proj)
        self.out_proj = ExplicitQDQLinear(source.out_proj)
        self.qk_matmul = _QKMatMul(self.scale)
        self.av_matmul = _AVMatMul()
        self.softmax = nn.Softmax(dim=-1)
        self.relative_position_bias_table = nn.Embedding(
            source.relative_position_bias_table.num_embeddings,
            source.relative_position_bias_table.embedding_dim,
        )
        with torch.no_grad():
            self.relative_position_bias_table.weight.copy_(
                source.relative_position_bias_table.weight
            )
        self.register_buffer(
            "relative_position_index",
            source.relative_position_index.detach().clone(),
        )
        self.register_buffer("attention_scale", torch.ones((), dtype=torch.float32))
        self.quant_enabled = False

    @classmethod
    def from_attention(
        cls,
        source: PrunableCobevtAttention,
        *,
        use_mask_rpe: bool,
    ) -> "ExplicitQDQAttentionGraph":
        return cls(source, use_mask_rpe=use_mask_rpe)

    def _project(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = rearrange(
            self.q_proj(x), "b n (h d) -> b h n d", h=self.heads, d=self.d_qk
        )
        k = rearrange(
            self.k_proj(x), "b n (h d) -> b h n d", h=self.heads, d=self.d_qk
        )
        v = rearrange(
            self.v_proj(x), "b n (h d) -> b h n d", h=self.heads, d=self.d_v
        )
        return q, k, v

    def _attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        sim = self.qk_matmul(q, k)
        if self.use_mask_rpe:
            bias = self.relative_position_bias_table(self.relative_position_index)
            sim = sim + rearrange(bias, "i j h -> h i j")
            flat_mask = rearrange(
                mask, "b x y w1 w2 e l -> (b x y) e (l w1 w2)"
            )
            sim = sim.masked_fill(flat_mask.unsqueeze(1) == 0, -float("inf"))
        attention = self.softmax(sim)
        if self.quant_enabled:
            attention = _fake_quant_tensor(attention, self.attention_scale)
        return self.av_matmul(attention, v)

    def calibrate(self, x: torch.Tensor, mask: torch.Tensor) -> None:
        batch, agents, height, width, window_h, window_w, _ = x.shape
        flat = rearrange(x, "b l x y w1 w2 d -> (b x y) (l w1 w2) d")
        q_raw = self.q_proj.calibrate(flat)
        k_raw = self.k_proj.calibrate(flat)
        v_raw = self.v_proj.calibrate(flat)
        q = rearrange(q_raw, "b n (h d) -> b h n d", h=self.heads, d=self.d_qk)
        k = rearrange(k_raw, "b n (h d) -> b h n d", h=self.heads, d=self.d_qk)
        v = rearrange(v_raw, "b n (h d) -> b h n d", h=self.heads, d=self.d_v)
        sim = self.qk_matmul(q, k)
        if self.use_mask_rpe:
            bias = self.relative_position_bias_table(self.relative_position_index)
            sim = sim + rearrange(bias, "i j h -> h i j")
            flat_mask = rearrange(
                mask, "b x y w1 w2 e l -> (b x y) e (l w1 w2)"
            )
            sim = sim.masked_fill(flat_mask.unsqueeze(1) == 0, -float("inf"))
        attention = self.softmax(sim)
        self.attention_scale.copy_(_scale(attention))
        out = self.av_matmul(attention, v)
        out = rearrange(
            out,
            "b h (l w1 w2) d -> b l w1 w2 (h d)",
            l=agents,
            w1=window_h,
            w2=window_w,
        )
        self.out_proj.calibrate(out)
        self.quant_enabled = True
        self.q_proj.quant_enabled = True
        self.k_proj.quant_enabled = True
        self.v_proj.quant_enabled = True
        self.out_proj.quant_enabled = True

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, agents, height, width, window_h, window_w, _ = x.shape
        flat = rearrange(x, "b l x y w1 w2 d -> (b x y) (l w1 w2) d")
        q, k, v = self._project(flat)
        out = self._attention(q, k, v, mask)
        out = rearrange(
            out,
            "b h (l w1 w2) d -> b l w1 w2 (h d)",
            l=agents,
            w1=window_h,
            w2=window_w,
        )
        out = self.out_proj(out)
        return rearrange(
            out,
            "(b x y) l w1 w2 d -> b l x y w1 w2 d",
            b=batch,
            x=height,
            y=width,
        )


__all__ = [
    "AttentionMicrobenchmarkSpec",
    "ExplicitQDQAttentionGraph",
    "attention_microbenchmark_specs",
]
