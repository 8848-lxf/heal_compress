from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


class DGroupedToy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(16, 32, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1, groups=8, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 24, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv3(self.bn2(self.conv2(self.conv1(x))))


def _d_plan(old_output_keep: list[int]) -> object:
    from heal_compress.pruning.physical_prune_plan import GlobalPhysicalPrunePlan, ModuleAxisPruneRequest

    output_prune = [idx for idx in range(32) if idx not in set(old_output_keep)]
    plan = GlobalPhysicalPrunePlan()
    plan.add_request(
        ModuleAxisPruneRequest(
            "conv2",
            "grouped_d_compact_frontfill_reblock",
            output_prune,
            source_recipe_id="d_toy",
            metadata={
                "groups_new": 4,
                "old_input_keep_indices": list(range(32)),
                "old_output_keep_indices": old_output_keep,
            },
        )
    )
    plan.add_request(
        ModuleAxisPruneRequest(
            "bn2",
            "out",
            output_prune,
            source_recipe_id="d_toy",
            metadata={"ordered_keep_indices": old_output_keep},
        )
    )
    plan.add_request(
        ModuleAxisPruneRequest(
            "conv3",
            "in",
            output_prune,
            source_recipe_id="d_toy",
            metadata={"ordered_keep_indices": old_output_keep},
        )
    )
    return plan


def test_d_frontfill_weight_copy_and_zero_pad() -> None:
    from heal_compress.pruning.grouped_conv import prune_grouped_conv_d_compact_frontfill_reblock

    conv = nn.Conv2d(32, 32, 3, padding=1, groups=8, bias=False)
    with torch.no_grad():
        conv.weight.copy_(torch.arange(conv.weight.numel(), dtype=conv.weight.dtype).view_as(conv.weight))
    old_weight = conv.weight.detach().clone()
    old_output_keep = [12, 0, 1, 2] + [4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16]

    report = prune_grouped_conv_d_compact_frontfill_reblock(
        conv,
        old_output_keep_indices=old_output_keep,
        old_input_keep_indices=list(range(32)),
        groups_new=4,
    )

    new_out = old_output_keep.index(12)
    assert conv.groups == 4
    assert conv.in_channels == 32
    assert conv.out_channels == 16
    assert tuple(conv.weight.shape) == (16, 8, 3, 3)
    assert torch.equal(conv.weight[new_out, 0:4], old_weight[12, 0:4])
    assert torch.equal(conv.weight[new_out, 4:8], torch.zeros_like(conv.weight[new_out, 4:8]))
    assert report["old_output_to_new_output_map"]["12"] == new_out
    assert report["semantic_preserved"] is False
    assert report["compact_first"] is True
    assert report["frontfill_weight_transplant"] is True
    assert report["weight_values_retained"] is True
    assert report["new_connections_zero_initialized"] is True
    assert report["requires_recovery_finetune"] is True


def test_d_does_not_reject_semantically_misaligned_old_filter() -> None:
    from heal_compress.pruning.grouped_conv import resolve_grouped_conv_d_compact_frontfill_reblock

    conv = nn.Conv2d(32, 32, 3, padding=1, groups=8, bias=False)
    resolved = resolve_grouped_conv_d_compact_frontfill_reblock(
        conv,
        old_output_keep_indices=[12] + list(range(0, 12)) + [13, 14, 15],
        old_input_keep_indices=list(range(32)),
        groups_new=4,
    )

    assert resolved["legal"] is True
    assert resolved["reason"] != "invalid_reblock_assignment"
    assert resolved["old_output_to_new_output_map"]["12"] == 0
    assert resolved["semantic_preserved"] is False


def test_d_rejects_source_kernel_too_wide() -> None:
    from heal_compress.pruning.grouped_conv import resolve_grouped_conv_d_compact_frontfill_reblock

    conv = nn.Conv2d(32, 32, 3, padding=1, groups=8, bias=False)
    resolved = resolve_grouped_conv_d_compact_frontfill_reblock(
        conv,
        old_output_keep_indices=list(range(16)),
        old_input_keep_indices=list(range(16)),
        groups_new=8,
    )

    assert resolved["legal"] is False
    assert resolved["reason"] == "frontfill_source_kernel_too_wide"


def test_d_one_shot_forward_smoke() -> None:
    model = DGroupedToy().eval()
    old_output_keep = [12, 0, 1, 2] + [4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16]
    plan = _d_plan(old_output_keep)

    report = plan.apply_one_shot(model)

    assert model.conv2.groups == 4
    assert model.conv2.in_channels == 32
    assert model.conv2.out_channels == 16
    assert tuple(model.conv2.weight.shape) == (16, 8, 3, 3)
    assert model.bn2.num_features == 16
    assert model.conv3.in_channels == 16
    assert model(torch.randn(2, 16, 8, 8)).shape == (2, 24, 8, 8)
    conv2_op = next(op for op in report["operations"] if op["module_name"] == "conv2")
    assert conv2_op["physical_axis"] == "grouped_d_compact_frontfill_reblock"
    assert conv2_op["semantic_preserved"] is False
    assert conv2_op["frontfill_weight_transplant"] is True


def test_d_downstream_bn_and_conv_input_follow_compact_output_order() -> None:
    model = DGroupedToy().eval()
    old_output_keep = [12, 0, 1, 2] + [4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16]
    with torch.no_grad():
        model.bn2.weight.copy_(torch.arange(32, dtype=model.bn2.weight.dtype))
        model.bn2.bias.copy_(torch.arange(100, 132, dtype=model.bn2.bias.dtype))
        model.bn2.running_mean.copy_(torch.arange(200, 232, dtype=model.bn2.running_mean.dtype))
        model.bn2.running_var.copy_(torch.arange(300, 332, dtype=model.bn2.running_var.dtype))
        model.conv3.weight.copy_(
            torch.arange(model.conv3.weight.numel(), dtype=model.conv3.weight.dtype).view_as(model.conv3.weight)
        )
    old_bn_weight = model.bn2.weight.detach().clone()
    old_bn_bias = model.bn2.bias.detach().clone()
    old_bn_mean = model.bn2.running_mean.detach().clone()
    old_bn_var = model.bn2.running_var.detach().clone()
    old_conv3_weight = model.conv3.weight.detach().clone()

    plan = _d_plan(old_output_keep)
    plan.apply_one_shot(model)

    assert torch.equal(model.bn2.weight, old_bn_weight[old_output_keep])
    assert torch.equal(model.bn2.bias, old_bn_bias[old_output_keep])
    assert torch.equal(model.bn2.running_mean, old_bn_mean[old_output_keep])
    assert torch.equal(model.bn2.running_var, old_bn_var[old_output_keep])
    assert torch.equal(model.conv3.weight[:, 0], old_conv3_weight[:, 12])
    assert torch.equal(model.conv3.weight[:, 1], old_conv3_weight[:, 0])


def test_d_shape_simulator_records_frontfill_semantics() -> None:
    from heal_compress.pruning.global_plan_shape_simulator import GlobalPlanShapeSimulator

    model = DGroupedToy().eval()
    plan = _d_plan([12, 0, 1, 2] + [4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16])

    sim = GlobalPlanShapeSimulator(model, plan, group_conv_align=4).simulate()

    assert sim["legal"] is True
    op = next(op for op in sim["operations"] if op["module_name"] == "conv2")
    assert op["shape_after"]["groups_after"] == 4
    assert op["shape_after"]["weight_shape_after"] == [16, 8, 3, 3]
    assert op["semantic_preserved"] is False
    assert op["frontfill_weight_transplant"] is True
    assert op["zero_initialized_new_connections"] is True


def test_d_report_writer_creates_outputs(tmp_path: Path) -> None:
    from tools.latency_lut.grouped_conv_d_compact_frontfill_reblock_v992 import run_toy_d_strategy_reports

    summary = run_toy_d_strategy_reports(tmp_path)

    assert summary["toy_forward_passed"] is True
    assert (tmp_path / "d_compact_frontfill_toy_report.json").exists()
    assert (tmp_path / "d_compact_frontfill_weight_mapping_report.json").exists()
    assert (tmp_path / "d_compact_frontfill_simulator_report.json").exists()
    assert (tmp_path / "d_compact_frontfill_failure_cases.jsonl").exists()
    report = json.loads((tmp_path / "d_compact_frontfill_toy_report.json").read_text())
    assert report["semantic_preserved"] is False
