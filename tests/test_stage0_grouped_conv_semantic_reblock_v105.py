from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.latency_lut.stage0_grouped_conv_reblock_v105 import (  # noqa: E402
    reblock_grouped_conv_semantic_preserving,
)


def _stage0_conv() -> nn.Conv2d:
    torch.manual_seed(1234)
    conv = nn.Conv2d(128, 128, kernel_size=3, padding=1, groups=32, bias=True)
    with torch.no_grad():
        conv.weight.normal_(mean=0.0, std=0.2)
        conv.bias.normal_(mean=0.0, std=0.1)
    return conv.eval()


def _assert_equivalent(groups_new: int) -> None:
    conv = _stage0_conv()
    new_conv = reblock_grouped_conv_semantic_preserving(conv, groups_new=groups_new).eval()
    x = torch.randn(2, 128, 50, 176)
    with torch.no_grad():
        old_y = conv(x)
        new_y = new_conv(x)
    diff = (old_y - new_y).abs()
    assert float(diff.max()) < 1e-5
    assert float(diff.mean()) < 1e-6


def test_toy_equivalence_groups32_to_groups16() -> None:
    _assert_equivalent(groups_new=16)


def test_toy_equivalence_groups32_to_groups8() -> None:
    _assert_equivalent(groups_new=8)


def test_weight_block_diagonal_correctness_groups16() -> None:
    conv = _stage0_conv()
    with torch.no_grad():
        conv.weight.copy_(torch.arange(conv.weight.numel(), dtype=conv.weight.dtype).view_as(conv.weight))
    old_weight = conv.weight.detach().clone()

    new_conv = reblock_grouped_conv_semantic_preserving(conv, groups_new=16)

    assert tuple(new_conv.weight.shape) == (128, 8, 3, 3)
    assert torch.equal(new_conv.weight[0, 0:4], old_weight[0])
    assert torch.equal(new_conv.weight[0, 4:8], torch.zeros_like(new_conv.weight[0, 4:8]))
    assert torch.equal(new_conv.weight[4, 4:8], old_weight[4])
    assert torch.equal(new_conv.weight[4, 0:4], torch.zeros_like(new_conv.weight[4, 0:4]))


def test_weight_block_diagonal_correctness_groups8() -> None:
    conv = _stage0_conv()
    with torch.no_grad():
        conv.weight.copy_(torch.arange(conv.weight.numel(), dtype=conv.weight.dtype).view_as(conv.weight))
    old_weight = conv.weight.detach().clone()

    new_conv = reblock_grouped_conv_semantic_preserving(conv, groups_new=8)

    assert tuple(new_conv.weight.shape) == (128, 16, 3, 3)
    for old_group in range(4):
        oc = old_group * 4
        start = old_group * 4
        assert torch.equal(new_conv.weight[oc, start : start + 4], old_weight[oc])
        before = new_conv.weight[oc, :start]
        after = new_conv.weight[oc, start + 4 :]
        assert torch.count_nonzero(before).item() == 0
        assert torch.count_nonzero(after).item() == 0


def test_no_channel_shape_change_only_groups_changed() -> None:
    conv = _stage0_conv()
    new_conv = reblock_grouped_conv_semantic_preserving(conv, groups_new=16)

    assert new_conv.in_channels == conv.in_channels == 128
    assert new_conv.out_channels == conv.out_channels == 128
    assert conv.groups == 32
    assert new_conv.groups == 16
    assert new_conv.kernel_size == conv.kernel_size
    assert new_conv.stride == conv.stride
    assert new_conv.padding == conv.padding
    assert new_conv.dilation == conv.dilation
    assert new_conv.bias is not None
    assert torch.equal(new_conv.bias, conv.bias)
