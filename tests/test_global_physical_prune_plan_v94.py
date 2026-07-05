from __future__ import annotations

import torch
import torch.nn as nn


def test_global_plan_unions_same_module_axis_requests():
    from heal_compress.pruning.physical_prune_plan import GlobalPhysicalPrunePlan, ModuleAxisPruneRequest

    plan = GlobalPhysicalPrunePlan()
    plan.add_request(ModuleAxisPruneRequest("conv3", "in", [2, 5], source_recipe_id="r1"))
    plan.add_request(ModuleAxisPruneRequest("conv3", "in", [5, 9], source_recipe_id="r2"))

    request = plan.get_request("conv3", "in")
    assert request.prune_indices == [2, 5, 9]
    assert request.source_recipe_ids == ["r1", "r2"]


def test_concat_offset_transform():
    from heal_compress.pruning.physical_prune_plan import concat_offset_transform

    assert concat_offset_transform([1, 3], offset=16) == [17, 19]


def test_one_shot_surgery_no_index_shift():
    from heal_compress.pruning.physical_prune_plan import GlobalPhysicalPrunePlan, ModuleAxisPruneRequest

    model = nn.Sequential()
    model.add_module("conv", nn.Conv2d(8, 8, 1, bias=False))
    with torch.no_grad():
        model.conv.weight.copy_(torch.arange(64, dtype=torch.float32).view(8, 8, 1, 1))

    plan = GlobalPhysicalPrunePlan()
    plan.add_request(ModuleAxisPruneRequest("conv", "out", [1, 3], source_recipe_id="r1"))
    plan.add_request(ModuleAxisPruneRequest("conv", "out", [3, 5], source_recipe_id="r2"))
    report = plan.apply_one_shot(model)

    assert model.conv.out_channels == 5
    assert report["operations"][0]["applied_once"] is True
    assert report["operations"][0]["prune_indices"] == [1, 3, 5]
    assert report["operations"][0]["keep_indices"] == [0, 2, 4, 6, 7]
    assert report["num_duplicate_module_axis_requests"] == 1


def test_grouped_coarsen_zero_padded_reblock_surgery():
    from heal_compress.pruning.physical_prune_plan import GlobalPhysicalPrunePlan, ModuleAxisPruneRequest

    model = nn.Sequential()
    model.add_module("gconv", nn.Conv2d(16, 16, 1, groups=4, bias=False))
    with torch.no_grad():
        model.gconv.weight.copy_(torch.arange(16 * 4, dtype=torch.float32).view(16, 4, 1, 1))

    plan = GlobalPhysicalPrunePlan()
    plan.add_request(
        ModuleAxisPruneRequest(
            "gconv",
            "grouped_coarsen_out",
            [2, 3, 6, 7, 10, 11, 14, 15],
            source_recipe_id="d1",
            metadata={"groups_new": 2, "old_groups": 4},
        )
    )
    report = plan.apply_one_shot(model)

    assert model.gconv.groups == 2
    assert model.gconv.in_channels == 16
    assert model.gconv.out_channels == 8
    assert tuple(model.gconv.weight.shape) == (8, 8, 1, 1)
    assert report["operations"][0]["axis"] == "grouped_coarsen_out"
    assert report["operations"][0]["weight_truncation_count"] == 0
    # Old filter 4 belongs to old group 1 and is copied into the second half of
    # merged new group 0's local input slice.
    assert torch.equal(model.gconv.weight[2, 4:8], torch.arange(16, 20, dtype=torch.float32).view(4, 1, 1))
    assert torch.count_nonzero(model.gconv.weight[2, 0:4]) == 0
