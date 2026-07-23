"""Unified instance-local Attention/FFN width-domain contracts."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from heal_compress.search.pruning_space.transformer_domains import (
    SharedTransformerParameterError,
    build_transformer_pruning_domains,
    fixed_transformer_rankings_from_unit_scores,
    legal_attention_widths,
    legal_ffn_widths,
)


class FusedAttention(nn.Module):
    def __init__(self, *, heads: int = 4, d_h: int = 16, d_model: int = 64) -> None:
        super().__init__()
        self.heads = heads
        self.to_qkv = nn.Linear(d_model, 3 * heads * d_h, bias=True)
        self.to_out = nn.Sequential(nn.Linear(heads * d_h, d_model), nn.Dropout(0.0))
        self.attend = nn.Softmax(dim=-1)
        self.scale = d_h**-0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = x.shape
        qkv = self.to_qkv(x).view(batch, tokens, 3, self.heads, -1)
        q, k, v = qkv.unbind(dim=2)
        score = torch.einsum("bthd,bshd->bhts", q, k) * self.scale
        out = torch.einsum("bhts,bshd->bthd", self.attend(score), v)
        return self.to_out(out.reshape(batch, tokens, -1))


class StandardFFN(nn.Module):
    def __init__(self, d_model: int = 64, d_ff: int = 1024) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class GatedFFN(nn.Module):
    def __init__(self, d_model: int = 64, d_ff: int = 512) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff)
        self.up_proj = nn.Linear(d_model, d_ff)
        self.down_proj = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = FusedAttention()
        self.ffn = StandardFFN()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        return x + self.ffn(x)


class TwoBlocks(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([Block(), Block()])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


def _rankings(model: nn.Module):
    attention = {
        f"layers.{index}.attn": {
            "qk": tuple(tuple(range(16)) for _ in range(4)),
            "vo": tuple(tuple(reversed(range(16))) for _ in range(4)),
        }
        for index in range(2)
    }
    # Make head 1 different to prove that local positions are not shared.
    attention["layers.0.attn"]["qk"] = (
        tuple(range(16)),
        tuple(reversed(range(16))),
        tuple(range(16)),
        tuple(range(16)),
    )
    ffn = {f"layers.{index}.ffn": tuple(range(1024)) for index in range(2)}
    return attention, ffn


def test_each_attention_instance_is_independent_and_family_does_not_merge() -> None:
    model = TwoBlocks().eval()
    attention_ranking, ffn_ranking = _rankings(model)
    domains, attention, ffn = build_transformer_pruning_domains(
        model,
        model_name="toy",
        attention_rankings=attention_ranking,
        ffn_rankings=ffn_ranking,
    )
    attention_domains = [row for row in domains if row.domain_type == "attention_dh"]
    assert len(attention) == len(attention_domains) == 2
    assert len(ffn) == 2
    assert len({row.domain_id for row in attention_domains}) == 2
    assert len({row.family for row in attention_domains}) == 1
    assert all(row.constraints["residual_does_not_tie_d_h"] for row in attention_domains)


def test_attention_decoder_couples_qk_and_vo_but_allows_independent_positions() -> None:
    model = TwoBlocks().eval()
    attention_ranking, ffn_ranking = _rankings(model)
    domains, _, _ = build_transformer_pruning_domains(
        model,
        model_name="toy",
        attention_rankings=attention_ranking,
        ffn_rankings=ffn_ranking,
    )
    domain = next(row for row in domains if row.domain_id == "attention_dh::layers.0.attn")
    decoded = domain.decode_width(8)
    assert len(decoded["qk_keep_by_head"]) == len(decoded["vo_keep_by_head"]) == 4
    assert all(len(row) == 8 for row in decoded["qk_keep_by_head"])
    assert all(len(row) == 8 for row in decoded["vo_keep_by_head"])
    assert decoded["qk_keep_by_head"][0] != decoded["qk_keep_by_head"][1]
    assert decoded["qk_keep_by_head"][0] != decoded["vo_keep_by_head"][0]
    assert not decoded["shared_qkvo_index"]


def test_shared_parameter_instances_fail_closed_instead_of_becoming_false_genes() -> None:
    class Shared(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            attention = FusedAttention()
            self.left = attention
            self.right = attention

    with pytest.raises(SharedTransformerParameterError, match="requires_explicit_tied_domain"):
        build_transformer_pruning_domains(
            Shared(),
            model_name="toy",
            allow_identity_ranking=True,
        )


def test_attention_and_ffn_legal_width_ladders_include_original() -> None:
    assert legal_attention_widths(32) == (4, 8, 16, 32)
    assert legal_attention_widths(24) == (4, 8, 16, 24)
    assert legal_ffn_widths(1024) == (4, 8, 16, 32, 64, 128, 256, 512, 1024)
    assert legal_ffn_widths(768)[-2:] == (512, 768)
    assert 512 in legal_ffn_widths(768)


def test_standard_and_gated_ffn_domains_keep_d_model_fixed() -> None:
    class FFNs(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.standard = StandardFFN()
            self.gated = GatedFFN()

    domains, attention, ffn = build_transformer_pruning_domains(
        FFNs(),
        model_name="toy",
        allow_identity_ranking=True,
    )
    assert not attention
    assert len(ffn) == 2
    assert {row.constraints["ffn_type"] for row in domains} == {"standard", "gated"}
    assert all(row.constraints["d_model"] == 64 for row in domains)
    assert all(row.constraints["d_model_fixed"] for row in domains)
    assert max(next(row for row in domains if row.module_path == "standard").legal_widths) == 1024


def test_raw_common_loss_scores_build_fixed_per_head_and_ffn_rankings() -> None:
    model = TwoBlocks().eval()
    diagnostic, attention, ffn = build_transformer_pruning_domains(
        model,
        model_name="toy",
        allow_identity_ranking=True,
    )
    scores = {
        unit_id: float(len(domain.ordered_unit_ids) - position)
        for domain in diagnostic
        for position, unit_id in enumerate(domain.ordered_unit_ids)
    }
    attention_rankings, ffn_rankings, manifest = fixed_transformer_rankings_from_unit_scores(
        attention, ffn, scores
    )
    formal, _, _ = build_transformer_pruning_domains(
        model,
        model_name="toy",
        attention_rankings=attention_rankings,
        ffn_rankings=ffn_rankings,
    )
    assert manifest["normalization_applied"] is False
    assert manifest["type_calibration"] == "identity"
    assert all(not row.metadata["diagnostic_identity_ranking"] for row in formal)
    assert all("second_order_task_loss" in row.ranking_method for row in formal)


def test_transformer_model_adapter_detects_all_four_real_config_contracts() -> None:
    from heal_compress.search.adapters.transformer_models import (
        detect_transformer_model_adapter,
    )

    configurations = {
        "v2xvit": ("heter_model_baseline", "v2xvit"),
        "cobevt": ("heter_model_baseline", "cobevt"),
        "attfusion": ("heter_model_baseline", "att"),
        "coalign": ("heter_model_baseline_ms", "att"),
    }
    for expected, (core, fusion) in configurations.items():
        config = {
            "model": {
                "core_method": core,
                "args": {"fusion_method": fusion},
            }
        }
        assert detect_transformer_model_adapter(config).model_key == expected
