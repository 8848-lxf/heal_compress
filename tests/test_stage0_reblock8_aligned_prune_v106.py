from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.latency_lut.stage0_reblock8_aligned_prune_v106 import (  # noqa: E402
    build_groups16_no_prune_report,
    build_reblock8_keep_indices,
    reblock8_then_prune_stage0_block_hidden_width,
)


class ToyBottleneck(nn.Module):
    def __init__(self, *, residual: bool = False) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(128, 128, 1, bias=True)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 128, 3, padding=1, groups=32, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 128, 1, bias=True)
        self.residual = residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.bn1(self.conv1(x))
        out = self.bn2(self.conv2(out))
        out = self.conv3(out)
        if self.residual:
            out = out + x
        return out


def test_toy_bottleneck_shape_legality_after_reblock8_prune() -> None:
    block = ToyBottleneck().eval()

    report = reblock8_then_prune_stage0_block_hidden_width(block)

    assert report["hidden_width_before"] == 128
    assert report["hidden_width_after"] == 64
    assert block.conv1.out_channels == 64
    assert block.bn1.num_features == 64
    assert block.conv2.in_channels == 64
    assert block.conv2.out_channels == 64
    assert block.conv2.groups == 8
    assert tuple(block.conv2.weight.shape) == (64, 8, 3, 3)
    assert block.bn2.num_features == 64
    assert block.conv3.in_channels == 64
    assert block.conv3.out_channels == 128

    with torch.no_grad():
        y = block(torch.randn(1, 128, 17, 19))
    assert tuple(y.shape) == (1, 128, 17, 19)


def test_groups16_no_prune_rule_reports_no_candidate() -> None:
    report = build_groups16_no_prune_report()

    assert report["groups16_aligned_pruning_available"] is False
    assert report["available_prune_targets"] == []
    assert report["reason"] == "per_group already equals minimum 8-aligned floor"


def test_keep_index_correctness_for_default_keep_first2() -> None:
    keep = build_reblock8_keep_indices([0, 1])

    assert len(keep) == 64
    for group_id in range(8):
        expected = list(range(group_id * 16, group_id * 16 + 8))
        assert keep[group_id * 8 : (group_id + 1) * 8] == expected

    block = ToyBottleneck().eval()
    report = reblock8_then_prune_stage0_block_hidden_width(block, keep_subgroups_per_new_group=[0, 1])

    assert report["hidden_keep_indices"] == keep
    assert block.conv2.in_channels // block.conv2.groups == 8
    assert block.conv2.out_channels // block.conv2.groups == 8


def test_residual_output_shape_is_unchanged() -> None:
    block = ToyBottleneck(residual=True).eval()
    x = torch.randn(2, 128, 11, 13)

    reblock8_then_prune_stage0_block_hidden_width(block)

    with torch.no_grad():
        y = block(x)
    assert tuple(y.shape) == tuple(x.shape)
