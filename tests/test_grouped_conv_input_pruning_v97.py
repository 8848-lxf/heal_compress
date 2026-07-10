from __future__ import annotations

import torch
import torch.nn as nn

from heal_compress.pruning.grouped_conv import (
    grouped_conv_pruning_fn,
    prune_grouped_conv_input_balanced,
    resolve_grouped_conv_input_keep,
)
from heal_compress.pruning.physical_prune_plan import GlobalPhysicalPrunePlan, ModuleAxisPruneRequest
from heal_compress.pruning.pruning_fns import prune_conv_out
from heal_compress.pruning.propagation import GroupBuilder
from heal_compress.search.importance import compute_group_importance
from heal_compress.tracer.pruning_group import PruningGroup
from heal_compress.tracer.generic_tracer import trace_model
from heal_compress.tracer.op_graph import build_op_graph


class GroupedInputToy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(64, 128, 1, bias=False)
        self.conv2 = nn.Conv2d(128, 128, 3, padding=1, groups=32, bias=False)
        self.conv3 = nn.Conv2d(128, 256, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv3(self.conv2(self.conv1(x)))


def test_grouped_conv_input_balanced_surgery_keeps_output_and_downstream_width() -> None:
    model = GroupedInputToy().eval()
    prune = [group_id * 4 for group_id in range(32)]

    plan = GlobalPhysicalPrunePlan()
    plan.add_request(ModuleAxisPruneRequest("conv1", "out", prune, source_recipe_id="root"))
    plan.add_request(
        ModuleAxisPruneRequest("conv2", "grouped_input_balanced", prune, source_recipe_id="root")
    )
    report = plan.apply_one_shot(model)

    assert model.conv1.out_channels == 96
    assert model.conv2.in_channels == 96
    assert model.conv2.groups == 32
    assert tuple(model.conv2.weight.shape) == (128, 3, 3, 3)
    assert model.conv2.out_channels == 128
    assert model.conv3.in_channels == 128
    assert any(op["axis"] == "grouped_input_balanced" for op in report["operations"])
    assert model(torch.randn(2, 64, 8, 8)).shape == (2, 256, 8, 8)


def test_grouped_conv_input_flat_uneven_keep_is_rejected_or_repaired() -> None:
    conv = nn.Conv2d(128, 128, 3, padding=1, groups=32, bias=False)
    uneven_keep = list(range(96))

    rejected = resolve_grouped_conv_input_keep(conv, uneven_keep, allow_repair=False)

    assert rejected["legal"] is False
    assert rejected["reason"] == "grouped_input_balance_violation"

    repaired = resolve_grouped_conv_input_keep(conv, uneven_keep, allow_repair=True)

    assert repaired["legal"] is True
    assert repaired["repaired"] is True
    assert repaired["reason"] == "grouped_input_balance_violation"
    assert len(repaired["keep_indices"]) % conv.groups == 0
    per_group_counts = repaired["per_group_kept_count"]
    assert len(set(per_group_counts.values())) == 1


def test_grouped_conv_output_policy_a_group_does_not_include_upstream_or_grouped_input() -> None:
    model = GroupedInputToy().eval()
    sample = torch.randn(1, 64, 8, 8)
    trace = trace_model(model, sample)
    op_graph = build_op_graph(trace, model)
    groups = GroupBuilder(
        op_graph,
        align=1,
        grouped_conv_mode="flat_output_groups_fixed",
        protect_residual_add=False,
    ).build()
    conv2_group = next(g for g in groups if g.group_id == "group::conv2")
    item_keys = {(item.name, item.direction, getattr(item.pruning_fn, "__name__", "")) for item in conv2_group.items}

    assert ("conv2", "out", grouped_conv_pruning_fn("flat_output_groups_fixed").__name__) in item_keys
    assert ("conv1", "out", prune_conv_out.__name__) not in item_keys
    assert ("conv2", "in", prune_grouped_conv_input_balanced.__name__) not in item_keys


def test_grouped_conv_input_importance_uses_absolute_input_space_without_oob() -> None:
    model = GroupedInputToy().eval()
    scope = PruningGroup("group::conv1", num_channels=128)
    scope.add_dep("conv1", model.conv1, prune_conv_out, "out", idxs=list(range(128)), reason="root_out")
    scope.add_dep(
        "conv2",
        model.conv2,
        prune_grouped_conv_input_balanced,
        "in",
        idxs=list(range(128)),
        reason="grouped_consumer_in_balanced",
    )

    scores, records = compute_group_importance(model, [scope], method="l1_norm")

    assert scores[scope.group_id] < float("inf")
    assert records[0]["importance_risk_reasons"] == ""
    assert scope.protected is False
