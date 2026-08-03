"""Physical Q/K/V/O and FFN width rewrite acceptance tests."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from search.pruning_space.transformer_domains import build_transformer_pruning_domains
from search.pruning_space.transformer_physical_pruner import materialize_transformer_widths
from search.pruning_space.unified_physical_pruner import (
    materialize_unified_widths,
)
from search.proxy.transformer_parameter_slices import build_transformer_unit_parameter_slices


class FusedAttention(nn.Module):
    def __init__(self, d_model: int = 32, heads: int = 4, d_h: int = 8) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = d_h
        self.inner_dim = heads * d_h
        self.scale = d_h**-0.5
        self.to_qkv = nn.Linear(d_model, 3 * self.inner_dim, bias=True)
        self.to_out = nn.Sequential(nn.Linear(self.inner_dim, d_model, bias=True), nn.Dropout(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = x.shape
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q = q.view(batch, tokens, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, tokens, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, tokens, self.heads, self.head_dim).transpose(1, 2)
        score = (q @ k.transpose(-2, -1)) * self.scale
        out = (score.softmax(dim=-1) @ v).transpose(1, 2).reshape(batch, tokens, -1)
        return self.to_out(out)


class SeparateAttention(nn.Module):
    def __init__(self, d_model: int = 32, heads: int = 4, d_h: int = 8) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = d_h
        self.inner_dim = heads * d_h
        self.scale = d_h**-0.5
        self.q_proj = nn.Linear(d_model, self.inner_dim, bias=True)
        self.k_proj = nn.Linear(d_model, self.inner_dim, bias=True)
        self.v_proj = nn.Linear(d_model, self.inner_dim, bias=True)
        self.out_proj = nn.Linear(self.inner_dim, d_model, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = x.shape
        shape = (batch, tokens, self.heads, self.head_dim)
        q = self.q_proj(x).view(shape).transpose(1, 2)
        k = self.k_proj(x).view(shape).transpose(1, 2)
        v = self.v_proj(x).view(shape).transpose(1, 2)
        score = (q @ k.transpose(-2, -1)) * self.scale
        out = (score.softmax(dim=-1) @ v).transpose(1, 2).reshape(batch, tokens, -1)
        return self.out_proj(out)


class StandardFFN(nn.Module):
    def __init__(self, d_model: int = 32, d_ff: int = 512) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff, bias=True)
        self.fc2 = nn.Linear(d_ff, d_model, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.nn.functional.gelu(self.fc1(x)))


class GatedFFN(nn.Module):
    def __init__(self, d_model: int = 32, d_ff: int = 256) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=True)
        self.up_proj = nn.Linear(d_model, d_ff, bias=True)
        self.down_proj = nn.Linear(d_ff, d_model, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class Toy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fused = FusedAttention()
        self.separate = SeparateAttention()
        self.ffn = StandardFFN()
        self.gated = GatedFFN()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.fused(x) + self.separate(x)
        return x + self.ffn(x) + self.gated(x)


def _space(model: nn.Module):
    attention = {
        "fused": {
            "qk": (
                tuple(range(8)), tuple(reversed(range(8))), tuple(range(8)), tuple(range(8))
            ),
            "vo": tuple(tuple(reversed(range(8))) for _ in range(4)),
        },
        "separate": {
            "qk": tuple(tuple(range(8)) for _ in range(4)),
            "vo": tuple(tuple(range(2, 8)) + tuple(range(2)) for _ in range(4)),
        },
    }
    ffn = {"ffn": tuple(range(512)), "gated": tuple(reversed(range(256)))}
    return build_transformer_pruning_domains(
        model,
        model_name="toy",
        attention_rankings=attention,
        ffn_rankings=ffn,
    )[0]


def test_fused_and_separate_qkv_are_physically_rewritten_with_bias_and_scale() -> None:
    torch.manual_seed(9)
    model = Toy().eval()
    domains = _space(model)
    original_fused = model.fused.to_qkv.weight.detach().clone()
    original_parameter_count = sum(value.numel() for value in model.parameters())
    widths = {row.domain_id: row.original_width for row in domains}
    widths["attention_dh::fused"] = 4
    widths["attention_dh::separate"] = 4
    report = materialize_transformer_widths(model, domains, widths, model_name="toy")

    assert report.passed, report.issues
    assert report.physical_parameter_count < original_parameter_count
    assert model.fused.to_qkv.weight.shape == (3 * 4 * 4, 32)
    assert model.fused.to_qkv.bias.shape == (3 * 4 * 4,)
    assert model.fused.to_out[0].weight.shape == (32, 4 * 4)
    assert model.separate.q_proj.weight.shape == (4 * 4, 32)
    assert model.separate.k_proj.weight.shape == (4 * 4, 32)
    assert model.separate.v_proj.weight.shape == (4 * 4, 32)
    assert model.separate.out_proj.weight.shape == (32, 4 * 4)
    assert model.fused.heads == model.separate.heads == 4
    assert model.fused.head_dim == model.separate.head_dim == 4
    assert model.fused.scale == model.separate.scale == 1.0 / math.sqrt(4.0)
    # Head 0 Q keeps local 4..7 and is compacted to the first four rows.
    torch.testing.assert_close(model.fused.to_qkv.weight[:4], original_fused[4:8])
    output = model(torch.randn(2, 5, 32))
    assert output.shape == (2, 5, 32)
    assert torch.isfinite(output).all()
    assert not report.mask_only and not report.hidden_padding


def test_standard_and_gated_ffn_coupling_is_physical_and_keeps_d_model() -> None:
    model = Toy().eval()
    domains = _space(model)
    widths = {row.domain_id: row.original_width for row in domains}
    widths["ffn_hidden::ffn"] = 256
    widths["ffn_hidden::gated"] = 128
    report = materialize_transformer_widths(model, domains, widths, model_name="toy")

    assert report.passed, report.issues
    assert model.ffn.fc1.in_features == model.ffn.fc2.out_features == 32
    assert model.ffn.fc1.out_features == model.ffn.fc2.in_features == 256
    assert model.gated.gate_proj.out_features == 128
    assert model.gated.up_proj.out_features == 128
    assert model.gated.down_proj.in_features == 128
    assert model.gated.down_proj.out_features == 32
    assert torch.isfinite(model(torch.randn(1, 3, 32))).all()


def test_requested_original_width_is_not_silently_rewritten_or_repaired() -> None:
    model = Toy().eval()
    domains = _space(model)
    widths = {row.domain_id: row.original_width for row in domains}
    report = materialize_transformer_widths(model, domains, widths, model_name="toy")
    assert report.passed
    assert not report.operations
    assert report.requested_widths == report.realized_widths


def test_unified_materializer_physically_decodes_transformer_domains_without_mutating_source() -> None:
    source = Toy().eval()
    domains = _space(source)
    widths = {row.domain_id: row.original_width for row in domains}
    widths["attention_dh::fused"] = 4
    widths["ffn_hidden::ffn"] = 256
    result = materialize_unified_widths(
        source,
        (),
        domains,
        widths,
        model_name="toy",
    )
    assert result.report.passed, result.report.issues
    assert result.report.requested_widths == result.report.realized_widths
    assert result.model.fused.head_dim == 4
    assert result.model.ffn.fc1.out_features == 256
    assert source.fused.head_dim == 8
    assert source.ffn.fc1.out_features == 512
    assert result.report.physical_parameter_count < result.report.original_parameter_count
    assert torch.isfinite(result.model(torch.randn(1, 3, 32))).all()


def test_transformer_atoms_resolve_to_qk_vo_and_gated_parameter_slices() -> None:
    model = Toy().eval()
    domains = _space(model)
    mapping = build_transformer_unit_parameter_slices(model, domains)

    qk = mapping["attention_dh::fused::head0::qk::0"]
    assert any(row.parameter_name == "fused.to_qkv.weight" and row.indices == (0, 32) for row in qk)
    assert any(row.parameter_name == "fused.to_qkv.bias" and row.indices == (0, 32) for row in qk)
    vo = mapping["attention_dh::fused::head0::vo::0"]
    assert any(row.parameter_name == "fused.to_qkv.weight" and row.indices == (64,) for row in vo)
    assert any(row.parameter_name == "fused.to_out.0.weight" and row.axis == 1 and row.indices == (0,) for row in vo)

    gated = mapping["ffn_hidden::gated::neuron::7"]
    assert {row.parameter_name for row in gated} >= {
        "gated.gate_proj.weight",
        "gated.gate_proj.bias",
        "gated.up_proj.weight",
        "gated.up_proj.bias",
        "gated.down_proj.weight",
    }
