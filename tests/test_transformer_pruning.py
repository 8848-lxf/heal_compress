"""Unit tests for Transformer pruning support."""

from __future__ import annotations

import torch
import torch.nn as nn

from heal_compress.pruning.transformer_checker import check_transformer_group, check_transformer_model_legality
from heal_compress.tracer.transformer_groups import build_transformer_pruning_groups


def _group(groups, group_type):
    return next(g for g in groups if g.meta.get("group_type") == group_type)


class StandardFFN(nn.Module):
    def __init__(self, hidden_dim=64, ffn_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, ffn_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(ffn_dim, hidden_dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class GatedFFN(nn.Module):
    def __init__(self, hidden_dim=64, ffn_dim=128):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, ffn_dim)
        self.up_proj = nn.Linear(hidden_dim, ffn_dim)
        self.down_proj = nn.Linear(ffn_dim, hidden_dim)

    def forward(self, x):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class SplitAttention(nn.Module):
    def __init__(self, hidden_dim=64, num_heads=4, head_dim=16):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim
        self.q_proj = nn.Linear(hidden_dim, self.inner_dim)
        self.k_proj = nn.Linear(hidden_dim, self.inner_dim)
        self.v_proj = nn.Linear(hidden_dim, self.inner_dim)
        self.out_proj = nn.Linear(self.inner_dim, hidden_dim)

    def forward(self, x):
        b, n, _ = x.shape
        q = self.q_proj(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        out = (attn.softmax(dim=-1) @ v).transpose(1, 2).reshape(b, n, self.inner_dim)
        return self.out_proj(out)


class FusedAttention(nn.Module):
    def __init__(self, hidden_dim=64, num_heads=4, head_dim=16):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim
        self.qkv = nn.Linear(hidden_dim, 3 * self.inner_dim)
        self.proj = nn.Linear(self.inner_dim, hidden_dim)

    def forward(self, x):
        b, n, _ = x.shape
        qkv = self.qkv(x).view(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        out = (attn.softmax(dim=-1) @ v).transpose(1, 2).reshape(b, n, self.inner_dim)
        return self.proj(out)


class BlockWithNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(64)
        self.attn = SplitAttention()
        self.norm2 = nn.LayerNorm(64)
        self.mlp = StandardFFN()

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


def test_standard_ffn_pruning():
    model = StandardFFN().eval()
    groups, _ = build_transformer_pruning_groups(model, ffn_align=8)
    group = _group(groups, "transformer_ffn_group")
    keep = list(range(96))
    assert check_transformer_group(group, keep, ffn_align=8)["legal"]
    group.prune(keep)
    assert model.fc1.out_features == 96
    assert model.fc2.in_features == 96
    assert model.fc2.out_features == 64
    assert model.fc1.out_features % 8 == 0
    assert model(torch.randn(2, 5, 64)).shape == (2, 5, 64)


def test_gated_ffn_pruning():
    model = GatedFFN().eval()
    groups, _ = build_transformer_pruning_groups(model, ffn_align=8)
    group = _group(groups, "transformer_gated_ffn_group")
    keep = list(range(96))
    assert check_transformer_group(group, keep, ffn_align=8)["legal"]
    group.prune(keep)
    assert model.gate_proj.out_features == 96
    assert model.up_proj.out_features == 96
    assert model.down_proj.in_features == 96
    assert model.down_proj.out_features == 64
    assert model(torch.randn(2, 5, 64)).shape == (2, 5, 64)


def test_split_qkv_whole_head_pruning():
    model = SplitAttention().eval()
    groups, _ = build_transformer_pruning_groups(model, min_heads=1, head_align=1)
    group = next(g for g in groups if g.meta.get("group_type") == "transformer_head_group" and g.meta.get("head_id") == 0)
    keep = group.meta["keep_indices"]
    assert check_transformer_group(group, keep, min_heads=1, head_align=1)["legal"]
    group.prune(keep)
    assert model.num_heads == 3
    assert model.inner_dim == 48
    assert model.head_dim == 16
    assert model.q_proj.out_features == 48
    assert model.k_proj.out_features == 48
    assert model.v_proj.out_features == 48
    assert model.out_proj.in_features == 48
    assert model.out_proj.out_features == 64
    assert model(torch.randn(2, 5, 64)).shape == (2, 5, 64)


def test_fused_qkv_whole_head_pruning():
    model = FusedAttention().eval()
    groups, _ = build_transformer_pruning_groups(model, min_heads=1, head_align=1)
    group = next(g for g in groups if g.meta.get("group_type") == "transformer_head_group" and g.meta.get("head_id") == 0)
    keep = group.meta["keep_indices"]
    assert check_transformer_group(group, keep, min_heads=1, head_align=1)["legal"]
    group.prune(keep)
    assert model.qkv.out_features == 3 * 48
    assert model.proj.in_features == 48
    assert model.proj.out_features == 64
    assert model.num_heads == 3
    assert model.head_dim == 16
    assert model(torch.randn(2, 5, 64)).shape == (2, 5, 64)


def test_native_mha_default_protected():
    class Native(nn.Module):
        def __init__(self):
            super().__init__()
            self.mha = nn.MultiheadAttention(64, 4, batch_first=True)

        def forward(self, x):
            return self.mha(x, x, x, need_weights=False)[0]

    model = Native().eval()
    groups, _ = build_transformer_pruning_groups(model, enable_native_mha_pruning=False)
    group = _group(groups, "native_mha_protected_group")
    assert group.protected
    assert group.protected_reason == "native_mha_default_protected"
    assert not group.prune(list(range(48)))["applied"]
    assert model(torch.randn(2, 5, 64)).shape == (2, 5, 64)


def test_hidden_size_default_protected():
    model = BlockWithNorm().eval()
    groups, _ = build_transformer_pruning_groups(model, protect_hidden=True)
    hidden = [g for g in groups if g.meta.get("group_type") == "transformer_hidden_group_protected"]
    assert hidden
    assert all(g.protected for g in hidden)
    before = tuple(model.norm1.normalized_shape)
    assert check_transformer_model_legality(model, protect_hidden=True)["legal"]
    assert model(torch.randn(2, 5, 64)).shape == (2, 5, 64)
    assert tuple(model.norm1.normalized_shape) == before

