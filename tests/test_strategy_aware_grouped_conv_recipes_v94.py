from __future__ import annotations


def _names():
    return {
        "conv1": "block.conv1",
        "conv2": "block.conv2",
        "bn2": "block.bn2",
        "conv3": "block.conv3",
    }


def test_A_recipe_does_not_include_upstream_or_current_input():
    from heal_compress.pruning.strategy_aware_recipes import build_grouped_bottleneck_recipe

    recipe = build_grouped_bottleneck_recipe(
        policy="A",
        module_names=_names(),
        channels=32,
        groups=4,
        prune_indices=[0, 1, 2, 3],
    )

    reqs = {(r.module_name, r.axis) for r in recipe.requests}
    assert ("block.conv2", "out") in reqs
    assert ("block.bn2", "out") in reqs
    assert ("block.conv3", "in") in reqs
    assert ("block.conv1", "out") not in reqs
    assert ("block.conv2", "in") not in reqs
    assert recipe.metadata["groups_after"] == 4
    assert recipe.metadata["groups_changed"] is False


def test_B_recipe_group_balanced():
    from heal_compress.pruning.strategy_aware_recipes import build_grouped_bottleneck_recipe

    recipe = build_grouped_bottleneck_recipe(
        policy="B",
        module_names=_names(),
        channels=32,
        groups=4,
        prune_indices=[0, 1, 8, 9, 16, 17, 24, 25],
    )

    assert recipe.metadata["group_balance_pass"] is True
    assert recipe.metadata["old_group_keep_count"] == {0: 6, 1: 6, 2: 6, 3: 6}
    assert recipe.metadata["reinterpretation_ratio"] == 0.0


def test_C_recipe_includes_true_group_block():
    from heal_compress.pruning.strategy_aware_recipes import build_grouped_bottleneck_recipe

    recipe = build_grouped_bottleneck_recipe(
        policy="C",
        module_names=_names(),
        channels=32,
        groups=4,
        prune_group_ids=[1, 3],
        in_per_group=8,
        out_per_group=8,
    )

    reqs = {(r.module_name, r.axis) for r in recipe.requests}
    assert ("block.conv1", "out") in reqs
    assert ("block.conv2", "in") in reqs
    assert ("block.conv2", "out") in reqs
    assert ("block.bn2", "out") in reqs
    assert ("block.conv3", "in") in reqs
    assert recipe.metadata["groups_after"] == 2
    assert recipe.metadata["in_per_group_after"] == 8
    assert recipe.metadata["out_per_group_after"] == 8
    assert recipe.metadata["reinterpretation_ratio"] == 0.0


def test_D_recipe_bucket_local_zero_pad():
    from heal_compress.pruning.strategy_aware_recipes import build_grouped_bottleneck_recipe

    recipe = build_grouped_bottleneck_recipe(
        policy="D",
        module_names=_names(),
        channels=32,
        groups=4,
        groups_new=2,
        keep_indices=[0, 1, 8, 9, 16, 17, 24, 25],
        in_per_group=8,
        out_per_group=8,
    )

    assert recipe.metadata["groups_new"] == 2
    assert recipe.metadata["merge_factor"] == 2
    assert recipe.metadata["weight_truncation_count"] == 0
    assert recipe.metadata["semantic_mismatch_count"] == 0
    copies = recipe.metadata["zero_pad_copy_plan"]
    assert copies[0]["old_group"] == 0
    assert copies[0]["target_new_group"] == 0
    assert copies[0]["copied_slice_range"] == [0, 8]
    assert copies[2]["old_group"] == 1
    assert copies[2]["target_new_group"] == 0
    assert copies[2]["copied_slice_range"] == [8, 16]
