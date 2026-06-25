from __future__ import annotations

import torch
import torch.nn as nn

from heal_compress.pruning.pruning_fns import prune_bn, prune_conv_in, prune_conv_out
from heal_compress.pruning.units import expand_coupled_channel_units
from heal_compress.search.importance import compute_scope_channel_importance
from heal_compress.tracer.pruning_group import PruningGroup


def test_coupled_channel_unit_expands_single_root_idx_through_scope_items():
    conv1 = nn.Conv2d(3, 8, 3, padding=1, bias=False)
    bn1 = nn.BatchNorm2d(8)
    conv2 = nn.Conv2d(8, 4, 1, bias=False)

    with torch.no_grad():
        for idx in range(8):
            conv1.weight[idx].fill_(idx + 1)
            conv2.weight[:, idx].fill_(10 + idx)

    scope = PruningGroup(
        group_id="toy.conv1_hidden",
        num_channels=8,
        meta={"group_type": "plain"},
    )
    scope.add_dep("conv1", conv1, prune_conv_out, "out", reason="root_out")
    scope.add_dep("bn1", bn1, prune_bn, "out", reason="bn")
    scope.add_dep("conv2", conv2, prune_conv_in, "in", reason="downstream_in")

    scope_imp, record = compute_scope_channel_importance(scope, method="l1_norm")
    units = expand_coupled_channel_units(scope, scope_imp, importance_mode="l1_norm")

    assert tuple(scope_imp.shape) == (8,)
    assert record["importance_source_items"] == ["conv1:out", "bn1:out", "conv2:in"]
    assert len(units) == 8

    unit = units[3]
    assert unit.scope_id == "toy.conv1_hidden"
    assert unit.root_idx == 3
    assert unit.local_indices_by_item == {
        "conv1:out": [3],
        "bn1:out": [3],
        "conv2:in": [3],
    }
    assert unit.item_modules == ["conv1", "bn1", "conv2"]
    assert unit.item_directions == ["out", "out", "in"]
    assert unit.importance == float(scope_imp[3])
    assert unit.group_id_in_grouped_conv is None
    assert unit.local_idx_in_group is None
