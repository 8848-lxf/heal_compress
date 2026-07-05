from __future__ import annotations


def test_residual_closure():
    from heal_compress.pruning.physical_prune_plan import GlobalPhysicalPrunePlan, ModuleAxisPruneRequest

    plan = GlobalPhysicalPrunePlan()
    plan.add_request(ModuleAxisPruneRequest("main", "out", [2], source_recipe_id="main"))
    plan.add_request(ModuleAxisPruneRequest("shortcut", "out", [5], source_recipe_id="shortcut"))
    closure = plan.apply_residual_closure(
        component_id="add0",
        module_axes=[("main", "out"), ("shortcut", "out"), ("post", "in")],
    )

    assert closure["closed_indices"] == [2, 5]
    assert plan.get_request("main", "out").prune_indices == [2, 5]
    assert plan.get_request("shortcut", "out").prune_indices == [2, 5]
    assert plan.get_request("post", "in").prune_indices == [2, 5]


def test_coupled_channel_unit_has_member_level_index_proof():
    from heal_compress.pruning.strategy_aware_recipes import build_coupled_channel_units_for_conv_bn_next

    units = build_coupled_channel_units_for_conv_bn_next(
        root_module="conv",
        bn_module="bn",
        next_module="next",
        channels=3,
    )

    assert len(units) == 3
    unit = units[2]
    assert unit.root_channel_index == 2
    assert [(m.module_name, m.axis, m.local_index) for m in unit.members] == [
        ("conv", "out", 2),
        ("bn", "out", 2),
        ("next", "in", 2),
    ]
