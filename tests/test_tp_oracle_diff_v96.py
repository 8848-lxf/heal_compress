from __future__ import annotations

import torch.nn as nn


def test_tp_oracle_diff_reports_unavailable_without_failing() -> None:
    from heal_compress.pruning.physical_prune_plan import GlobalPhysicalPrunePlan, ModuleAxisPruneRequest
    from heal_compress.pruning.tp_oracle_diff import build_tp_oracle_diff

    model = nn.Sequential()
    model.add_module("conv1", nn.Conv2d(3, 8, 1, bias=False))
    model.add_module("conv2", nn.Conv2d(8, 8, 1, bias=False))

    plan = GlobalPhysicalPrunePlan()
    plan.add_request(ModuleAxisPruneRequest("conv1", "out", [0, 1], source_recipe_id="root"))

    diff = build_tp_oracle_diff(model, plan, root_module_name="conv1", root_axis="out", root_indices=[0, 1])

    assert "available" in diff
    assert "project_plan_members" in diff
    assert "missing_dependencies_in_project_plan" in diff
    assert "extra_project_plan_members" in diff
    if diff["available"] is False:
        assert diff["status"] in {"torch_pruning_not_installed", "oracle_build_failed"}


def test_tp_oracle_diff_normalizes_project_specific_grouped_input_axis() -> None:
    from heal_compress.pruning.tp_oracle_diff import _project_member_keys

    rows = [
        {"module_name": "conv2", "axis": "grouped_input_balanced"},
        {"module_name": "gconv", "axis": "grouped_flat_output"},
    ]

    assert ("conv2", "in") in _project_member_keys(rows)
    assert ("gconv", "out") in _project_member_keys(rows)
