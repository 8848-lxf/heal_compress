from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


class GroupedBlockToy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(16, 128, 1)
        self.conv2 = nn.Conv2d(128, 128, 3, padding=1, groups=32)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv3(self.bn2(self.conv2(self.conv1(x))))


def _block_indices(groups: list[int], per_group: int) -> list[int]:
    out: list[int] = []
    for group_id in groups:
        out.extend(range(group_id * per_group, (group_id + 1) * per_group))
    return out


def _c_plan(pruned_groups: list[int]) -> object:
    from heal_compress.pruning.physical_prune_plan import GlobalPhysicalPrunePlan, ModuleAxisPruneRequest

    channel_prune = _block_indices(pruned_groups, 4)
    plan = GlobalPhysicalPrunePlan()
    plan.add_request(ModuleAxisPruneRequest("conv1", "out", channel_prune, source_recipe_id="c_toy"))
    plan.add_request(
        ModuleAxisPruneRequest(
            "conv2",
            "grouped_true_group_block",
            pruned_groups,
            source_recipe_id="c_toy",
            metadata={
                "old_groups": 32,
                "pruned_old_groups": pruned_groups,
                "in_per_group_before": 4,
                "out_per_group_before": 4,
            },
        )
    )
    plan.add_request(ModuleAxisPruneRequest("bn2", "out", channel_prune, source_recipe_id="c_toy"))
    plan.add_request(ModuleAxisPruneRequest("conv3", "in", channel_prune, source_recipe_id="c_toy"))
    return plan


def test_true_group_block_one_shot_pruning_preserves_per_group_width_and_forward() -> None:
    from heal_compress.pruning.global_plan_shape_simulator import GlobalPlanShapeSimulator

    model = GroupedBlockToy().eval()
    plan = _c_plan([0, 3, 7, 31])

    sim = GlobalPlanShapeSimulator(model, plan, group_conv_align=4).simulate()
    assert sim["legal"] is True
    assert sim["shapes"]["conv2"]["groups_after"] == 28
    assert sim["shapes"]["conv2"]["in_channels_after"] == 112
    assert sim["shapes"]["conv2"]["out_channels_after"] == 112
    assert sim["shapes"]["conv2"]["weight_shape_after"] == [112, 4, 3, 3]

    report = plan.apply_one_shot(model)

    assert model.conv1.out_channels == 112
    assert model.conv2.in_channels == 112
    assert model.conv2.out_channels == 112
    assert model.conv2.groups == 28
    assert tuple(model.conv2.weight.shape) == (112, 4, 3, 3)
    assert model.bn2.num_features == 112
    assert model.conv3.in_channels == 112
    assert model(torch.randn(2, 16, 8, 8)).shape == (2, 256, 8, 8)
    conv2_op = next(op for op in report["operations"] if op["module_name"] == "conv2")
    assert conv2_op["physical_axis"] == "grouped_true_group_block"
    assert conv2_op["groups_after"] == 28


def test_true_group_block_resolver_rejects_partial_old_group_pruning() -> None:
    from heal_compress.pruning.grouped_conv import resolve_grouped_conv_true_group_block_keep

    module = GroupedBlockToy().conv2
    resolved = resolve_grouped_conv_true_group_block_keep(module, prune_indices=[1, 2])

    assert resolved["legal"] is False
    assert resolved["reason"] == "c_strategy_requires_complete_old_group_block"


def test_true_group_block_strategy_does_not_change_default_a_strategy() -> None:
    from heal_compress.pruning.grouped_conv import grouped_conv_pruning_fn

    assert grouped_conv_pruning_fn("flat_output_groups_fixed").__name__ == "prune_grouped_flat_output_groups_fixed"
    assert grouped_conv_pruning_fn("A").__name__ == "prune_grouped_flat_output_groups_fixed"


def test_true_group_block_report_writer_creates_stable_artifacts(tmp_path: Path) -> None:
    from tools.latency_lut.grouped_conv_c_true_group_block_v991 import run_toy_c_strategy_reports

    report = run_toy_c_strategy_reports(tmp_path)

    assert report["toy_forward_passed"] is True
    assert (tmp_path / "c_true_group_block_toy_report.json").exists()
    assert (tmp_path / "c_true_group_block_global_plan_report.json").exists()
    assert (tmp_path / "c_true_group_block_simulator_report.json").exists()
    assert (tmp_path / "c_true_group_block_failure_cases.jsonl").exists()
    assert json.loads((tmp_path / "c_true_group_block_toy_report.json").read_text())["after"]["conv2_groups"] == 28


def test_true_group_block_plan_builder_can_sync_upstream_bn_before_grouped_input() -> None:
    from tools.latency_lut.grouped_conv_c_true_group_block_v991 import build_true_group_block_plan

    class BottleneckLikeToy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(16, 128, 1)
            self.bn1 = nn.BatchNorm2d(128)
            self.conv2 = nn.Conv2d(128, 128, 3, padding=1, groups=32)
            self.bn2 = nn.BatchNorm2d(128)
            self.conv3 = nn.Conv2d(128, 256, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.bn1(self.conv1(x))
            x = self.bn2(self.conv2(x))
            return self.conv3(x)

    model = BottleneckLikeToy().eval()
    plan = build_true_group_block_plan(
        grouped_module_name="conv2",
        upstream_module_name="conv1",
        upstream_bn_module_name="bn1",
        bn_module_name="bn2",
        downstream_module_name="conv3",
        old_groups=32,
        in_per_group=4,
        out_per_group=4,
        pruned_old_groups=[0],
        source_recipe_id="bn_closure",
    )
    plan.apply_one_shot(model)

    assert model.conv1.out_channels == 124
    assert model.bn1.num_features == 124
    assert model.conv2.in_channels == 124
    assert model(torch.randn(1, 16, 8, 8)).shape == (1, 256, 8, 8)
