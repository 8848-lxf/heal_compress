from __future__ import annotations

import torch.nn as nn


def test_simulator_rejects_unsynced_downstream_conv_input() -> None:
    from heal_compress.pruning.global_plan_shape_simulator import GlobalPlanShapeSimulator
    from heal_compress.pruning.physical_prune_plan import GlobalPhysicalPrunePlan, ModuleAxisPruneRequest
    from heal_compress.tracer.op_graph import OP_CONV, OpGraph, OpNode

    model = nn.Sequential()
    model.add_module("conv1", nn.Conv2d(8, 16, 1, bias=False))
    model.add_module("conv2", nn.Conv2d(16, 24, 1, bias=False))

    graph = OpGraph(model)
    graph.add_node(OpNode("conv1", OP_CONV, "Conv2d", in_channels=8, out_channels=16, groups=1))
    graph.add_node(OpNode("conv2", OP_CONV, "Conv2d", in_channels=16, out_channels=24, groups=1))
    graph.add_edge("conv1", "conv2", 0, "flow")
    graph.finalize()

    plan = GlobalPhysicalPrunePlan()
    plan.add_request(ModuleAxisPruneRequest("conv1", "out", [0, 1, 2, 3], source_recipe_id="r1"))

    report = GlobalPlanShapeSimulator(model, plan, op_graph=graph).simulate()

    assert report["legal"] is False
    issues = {item["issue"] for item in report["issues"]}
    assert "downstream_input_channel_mismatch" in issues


def test_simulator_rejects_grouped_conv_inner_channel_misalignment() -> None:
    from heal_compress.pruning.global_plan_shape_simulator import GlobalPlanShapeSimulator
    from heal_compress.pruning.physical_prune_plan import GlobalPhysicalPrunePlan, ModuleAxisPruneRequest

    model = nn.Sequential()
    model.add_module("gconv", nn.Conv2d(64, 64, 1, groups=4, bias=False))

    plan = GlobalPhysicalPrunePlan()
    plan.add_request(ModuleAxisPruneRequest("gconv", "out", list(range(48, 64)), source_recipe_id="r1"))

    report = GlobalPlanShapeSimulator(model, plan, group_conv_align=8).simulate()

    assert report["legal"] is False
    issues = {item["issue"] for item in report["issues"]}
    assert "grouped_conv_inner_channel_alignment" in issues


def test_simulator_rejects_fixed_shape_contract_pruning() -> None:
    from heal_compress.pruning.global_plan_shape_simulator import GlobalPlanShapeSimulator
    from heal_compress.pruning.physical_prune_plan import GlobalPhysicalPrunePlan, ModuleAxisPruneRequest

    class Toy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder_m1 = nn.Module()
            self.encoder_m1.pillar_vfe = nn.Module()
            self.encoder_m1.pillar_vfe.pfn_layers = nn.ModuleList([nn.Linear(10, 64)])

    model = Toy()
    plan = GlobalPhysicalPrunePlan()
    plan.add_request(
        ModuleAxisPruneRequest(
            "encoder_m1.pillar_vfe.pfn_layers.0",
            "out",
            list(range(52, 64)),
            source_recipe_id="pfn",
        )
    )

    report = GlobalPlanShapeSimulator(model, plan).simulate()

    assert report["legal"] is False
    issues = {item["issue"] for item in report["issues"]}
    assert "fixed_shape_contract_pruned" in issues


def test_grouped_flat_selection_respects_per_group_alignment() -> None:
    import torch

    from heal_compress.pruning.selection import SelectionConfig, build_pruning_plan
    from heal_compress.pruning.pruning_fns import prune_conv_out
    from heal_compress.tracer.pruning_group import PruningGroup

    gconv = nn.Conv2d(128, 128, 3, padding=1, groups=16, bias=False)
    scope = PruningGroup("group::gconv", num_channels=128)
    scope.add_dep(
        "gconv",
        gconv,
        prune_conv_out,
        "out",
        idxs=list(range(128)),
        reason="grouped_conv:flat_output_groups_fixed",
    )
    scores = {"group::gconv": torch.arange(128, dtype=torch.float32)}

    plan = build_pruning_plan(
        [scope],
        scores,
        SelectionConfig(
            prune_ratio=0.20,
            selection_mode="root_node_local_unit_ratio",
            group_conv_selection_mode="flat_output_groups_fixed",
            group_conv_align=8,
            align=4,
        ),
    )

    assert plan.concrete_groups == []
    assert plan.grouped_conv_reports[0]["structure_legal"] is True
    assert plan.grouped_conv_reports[0]["expanded_prune_indices"] == []
    assert plan.grouped_conv_reports[0]["per_group_after"] == 8


def test_build_protected_layers_keeps_fixed_shape_pfn_and_scatter() -> None:
    import sys
    from pathlib import Path

    tests_dir = Path(__file__).resolve().parent
    if str(tests_dir) not in sys.path:
        sys.path.insert(0, str(tests_dir))
    from test_general_pruner import build_protected_layers

    class Toy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder_m1 = nn.Module()
            self.encoder_m1.pillar_vfe = nn.Module()
            self.encoder_m1.pillar_vfe.pfn_layers = nn.ModuleList([nn.Linear(10, 64)])
            self.encoder_m1.scatter = nn.Identity()
            self.cls_head = nn.Conv2d(64, 2, 1)

    model = Toy()
    protected = build_protected_layers(
        model,
        adapter_protected=[
            "encoder_m1.pillar_vfe",
            "encoder_m1.pillar_vfe.pfn_layers.0",
            "encoder_m1.scatter",
            "cls_head",
        ],
        extra_prefixes=[],
    )

    assert "encoder_m1.pillar_vfe" in protected
    assert "encoder_m1.pillar_vfe.pfn_layers.0" in protected
    assert "encoder_m1.scatter" in protected
    assert "cls_head" in protected


def test_full_model_surface_protects_fixed_shape_interface_group() -> None:
    from heal_compress.pruning.full_model_surface import apply_full_model_prunable_surface
    from heal_compress.pruning.pruning_fns import prune_linear_out
    from heal_compress.tracer.pruning_group import PruningGroup

    pfn = nn.Linear(10, 64)
    scope = PruningGroup("group::encoder_m1.pillar_vfe.pfn_layers.0", num_channels=64)
    scope.add_dep(
        "encoder_m1.pillar_vfe.pfn_layers.0",
        pfn,
        prune_linear_out,
        "out",
        idxs=list(range(64)),
        reason="root_out",
    )

    apply_full_model_prunable_surface([scope], group_conv_policy="A", total_model_params=1)

    assert scope.protected is True
    assert scope.protected_reason == "protected_fixed_shape_pfn_scatter_voxel_contract"


def test_full_model_surface_protects_pyramid_single_head_contract() -> None:
    from heal_compress.pruning.full_model_surface import apply_full_model_prunable_surface
    from heal_compress.pruning.pruning_fns import prune_conv_out
    from heal_compress.tracer.pruning_group import PruningGroup

    conv = nn.Conv2d(128, 64, 1)
    scope = PruningGroup("group::pyramid_backbone.single_head_0", num_channels=64)
    scope.add_dep(
        "pyramid_backbone.single_head_0",
        conv,
        prune_conv_out,
        "out",
        idxs=list(range(64)),
        reason="root_out",
    )

    apply_full_model_prunable_surface([scope], group_conv_policy="A", total_model_params=1)

    assert scope.protected is True
    assert scope.protected_reason == "protected_convtranspose_deblock_or_fpn_output_contract"
