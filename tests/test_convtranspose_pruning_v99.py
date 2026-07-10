from __future__ import annotations

import torch
import torch.nn as nn

from heal_compress.pruning.global_plan_shape_simulator import GlobalPlanShapeSimulator
from heal_compress.pruning.physical_prune_plan import GlobalPhysicalPrunePlan, ModuleAxisPruneRequest
from heal_compress.tracer.generic_tracer import trace_model
from heal_compress.tracer.op_graph import build_op_graph


class ConvTransposeToy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(16, 32, 1)
        self.deconv = nn.ConvTranspose2d(32, 24, 2, stride=2)
        self.conv2 = nn.Conv2d(24, 16, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.deconv(self.conv1(x)))


def _graph(model: nn.Module) -> object:
    sample = torch.randn(2, 16, 8, 8)
    return build_op_graph(trace_model(model, sample), model)


def test_convtranspose2d_input_pruning_syncs_upstream_only() -> None:
    model = ConvTransposeToy().eval()
    prune = [0, 3, 7, 11]
    plan = GlobalPhysicalPrunePlan()
    plan.add_request(ModuleAxisPruneRequest("conv1", "out", prune, source_recipe_id="root"))
    plan.add_request(ModuleAxisPruneRequest("deconv", "in", prune, source_recipe_id="root"))

    sim = GlobalPlanShapeSimulator(model, plan, op_graph=_graph(model), allow_convtranspose=True).simulate()
    assert sim["legal"] is True

    report = plan.apply_one_shot(model)

    assert model.conv1.out_channels == 28
    assert model.deconv.in_channels == 28
    assert tuple(model.deconv.weight.shape) == (28, 24, 2, 2)
    assert model.deconv.out_channels == 24
    assert model.conv2.in_channels == 24
    assert any(op["module_name"] == "deconv" and op["physical_axis"] == "in" for op in report["operations"])
    assert model(torch.randn(2, 16, 8, 8)).shape == (2, 16, 16, 16)


def test_convtranspose2d_output_pruning_syncs_downstream_input() -> None:
    model = ConvTransposeToy().eval()
    prune = [1, 5, 9, 13, 17, 21]
    plan = GlobalPhysicalPrunePlan()
    plan.add_request(ModuleAxisPruneRequest("deconv", "out", prune, source_recipe_id="root"))
    plan.add_request(ModuleAxisPruneRequest("conv2", "in", prune, source_recipe_id="root"))

    sim = GlobalPlanShapeSimulator(model, plan, op_graph=_graph(model), allow_convtranspose=True).simulate()
    assert sim["legal"] is True

    report = plan.apply_one_shot(model)

    assert model.deconv.out_channels == 18
    assert tuple(model.deconv.weight.shape) == (32, 18, 2, 2)
    assert model.deconv.bias is not None and tuple(model.deconv.bias.shape) == (18,)
    assert model.conv2.in_channels == 18
    assert any(op["module_name"] == "deconv" and op["physical_axis"] == "out" for op in report["operations"])
    assert model(torch.randn(2, 16, 8, 8)).shape == (2, 16, 16, 16)


def test_grouped_convtranspose2d_is_rejected_by_simulator_for_now() -> None:
    model = nn.Sequential()
    model.add_module("deconv", nn.ConvTranspose2d(16, 16, 2, stride=2, groups=4))
    plan = GlobalPhysicalPrunePlan()
    plan.add_request(ModuleAxisPruneRequest("deconv", "out", [0, 1, 2, 3], source_recipe_id="root"))

    sim = GlobalPlanShapeSimulator(model, plan, allow_convtranspose=True).simulate()

    assert sim["legal"] is False
    assert any(issue["issue"] == "unsupported_grouped_convtranspose" for issue in sim["issues"])
