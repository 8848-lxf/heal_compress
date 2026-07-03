from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from heal_compress.search.importance import compute_group_importance


class TinyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DummyItem:
    def __init__(self, module: nn.Module) -> None:
        self.module = module
        self.name = "conv"
        self.direction = "out"
        self.idxs = [0, 1, 2, 3]
        self.idx_transform = None

    def local_keep(self, reference):
        return list(reference)


class DummyGroup:
    def __init__(self, module: nn.Module) -> None:
        self.group_id = "group::conv"
        self.num_channels = 4
        self.items = [DummyItem(module)]
        self.protected = False
        self.meta = {"group_type": "plain"}


def test_group_importance_records_score_source_and_normalized_fields():
    model = TinyNet()
    group = DummyGroup(model.conv)

    _scores, records = compute_group_importance(model, [group], method="l1_norm")

    record = records[0]
    assert record["score_source"] == "l1_norm"
    assert record["fallback_used"] is False
    assert record["raw_sum_score"] == record["importance_score"]
    assert record["normalized_score"] > 0
    assert record["num_weights_in_group"] == model.conv.weight.numel()
    assert record["num_layers_in_group"] == 1


@pytest.mark.parametrize(
    ("method", "needle"),
    [
        ("first_order_taylor", "first_order_taylor_gradient_missing"),
        ("second_order_fisher", "second_order_fisher_gradient_missing"),
    ],
)
def test_taylor_fisher_refuse_l1_fallback_without_calibration_data(method: str, needle: str):
    model = TinyNet()
    group = DummyGroup(model.conv)

    with pytest.raises(RuntimeError, match=needle):
        compute_group_importance(model, [group], method=method, strict_grad=True)


@pytest.mark.parametrize("method", ["first_order_taylor", "second_order_fisher"])
def test_taylor_fisher_use_real_backward_gradients_without_fallback(method: str):
    model = TinyNet()
    group = DummyGroup(model.conv)
    batch = torch.randn(2, 3, 4, 4)

    _scores, records = compute_group_importance(
        model,
        [group],
        method=method,
        forward_fn=lambda m, x: m(x),
        calibration_data=[batch],
        loss_fn=lambda output, _batch: output.pow(2).mean(),
        num_calib_batches=1,
        strict_grad=True,
    )

    record = records[0]
    assert record["score_source"] == method
    assert record["fallback_used"] is False
    assert record["importance_score"] > 0
