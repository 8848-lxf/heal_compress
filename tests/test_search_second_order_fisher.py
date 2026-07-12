from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_second_order_fisher_uses_gradient_and_empirical_fisher_diag() -> None:
    import pytest
    import torch

    from pruning.api import score_pruning_units
    from pruning.config import ImportanceConfig, ImportanceMode, ImportanceNormalizationConfig, ImportanceNormalizationStrategy
    from tracer.types import CoupledChannelUnit, DependencyMember

    model = torch.nn.Sequential()
    model.add_module("conv", torch.nn.Conv2d(1, 2, kernel_size=1, bias=True))
    with torch.no_grad():
        model.conv.weight[:] = torch.tensor([[[[2.0]]], [[[3.0]]]])
        model.conv.bias[:] = torch.tensor([5.0, 7.0])
    model.conv.weight._importance_grad = torch.tensor([[[[0.1]]], [[[0.2]]]])
    model.conv.bias._importance_grad = torch.tensor([0.3, 0.4])
    model.conv.weight._importance_fisher_diag = torch.tensor([[[[0.01]]], [[[0.04]]]])
    model.conv.bias._importance_fisher_diag = torch.tensor([0.09, 0.16])
    units = [
        CoupledChannelUnit(
            scope_id="scope",
            root_module_path="conv",
            root_axis="out",
            root_channel_index=0,
            members=[DependencyMember("conv", "out", [0], "root_output")],
            stable_id="u0",
        ),
        CoupledChannelUnit(
            scope_id="scope",
            root_module_path="conv",
            root_axis="out",
            root_channel_index=1,
            members=[DependencyMember("conv", "out", [1], "root_output")],
            stable_id="u1",
        ),
    ]

    result = score_pruning_units(
        model,
        units,
        config=ImportanceConfig(
            mode=ImportanceMode.SECOND_ORDER_FISHER,
            normalization=ImportanceNormalizationConfig(strategy=ImportanceNormalizationStrategy.NONE),
        ),
        calibration_batches=3,
        task_loss="dummy",
    )

    expected_u0 = abs(0.1 * 2.0) + 0.5 * 0.01 * 2.0**2 + abs(0.3 * 5.0) + 0.5 * 0.09 * 5.0**2
    expected_u1 = abs(0.2 * 3.0) + 0.5 * 0.04 * 3.0**2 + abs(0.4 * 7.0) + 0.5 * 0.16 * 7.0**2
    assert result.raw_scores["u0"] == pytest.approx(expected_u0)
    assert result.raw_scores["u1"] == pytest.approx(expected_u1)
    assert result.calibration_batches == 3
    assert result.task_loss == "dummy"
    assert result.implementation_version.startswith("second-order-fisher")
