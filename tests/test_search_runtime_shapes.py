from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class _Toy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.head = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        pooled = y.mean(dim=(2, 3))
        return self.head(pooled)


def test_runtime_shape_profiler_records_real_hw_not_default_one() -> None:
    from search.proxy.runtime_shape_profiler import profile_runtime_layer_shapes

    model = _Toy().eval()
    profile = profile_runtime_layer_shapes(model, torch.randn(1, 3, 12, 10))

    conv = profile.by_module["conv"][0]
    assert conv.h_out == 12
    assert conv.w_out == 10
    assert conv.call_index == 0
    assert conv.groups == 1
    assert "head" in profile.by_module
