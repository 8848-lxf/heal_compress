from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.latency_lut.stage512_grouped_pergroup8_prune_v107 import (  # noqa: E402
    build_stage512_keep_indices,
    prune_stage512_block_hidden_width,
)


class Stage512ToyBlock(nn.Module):
    def __init__(self, *, residual: bool = False) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(512, 512, 1)
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = nn.Conv2d(512, 512, 3, padding=1, groups=32)
        self.bn2 = nn.BatchNorm2d(512)
        self.conv3 = nn.Conv2d(512, 512, 1)
        self.residual = residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.bn1(self.conv1(x))
        out = self.bn2(self.conv2(out))
        out = self.conv3(out)
        if self.residual:
            out = out + x
        return out


def test_stage512_keep_indices_keep_first8_per_group() -> None:
    keep = build_stage512_keep_indices()

    assert len(keep) == 256
    for group_id in range(32):
        expected = list(range(group_id * 16, group_id * 16 + 8))
        assert keep[group_id * 8 : (group_id + 1) * 8] == expected


def test_stage512_toy_block_shape_legality_after_prune() -> None:
    block = Stage512ToyBlock().eval()

    report = prune_stage512_block_hidden_width(block)

    assert report["hidden_width_before"] == 512
    assert report["hidden_width_after"] == 256
    assert block.conv1.out_channels == 256
    assert block.bn1.num_features == 256
    assert block.conv2.in_channels == 256
    assert block.conv2.out_channels == 256
    assert block.conv2.groups == 32
    assert tuple(block.conv2.weight.shape) == (256, 8, 3, 3)
    assert block.bn2.num_features == 256
    assert block.conv3.in_channels == 256
    assert block.conv3.out_channels == 512

    with torch.no_grad():
        y = block(torch.randn(1, 512, 9, 11))
    assert tuple(y.shape) == (1, 512, 9, 11)


def test_stage512_residual_output_shape_unchanged() -> None:
    block = Stage512ToyBlock(residual=True).eval()
    x = torch.randn(2, 512, 7, 9)

    prune_stage512_block_hidden_width(block)

    with torch.no_grad():
        y = block(x)
    assert tuple(y.shape) == tuple(x.shape)
