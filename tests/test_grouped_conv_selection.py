from __future__ import annotations

import torch
import torch.nn as nn

from heal_compress.pruning.group_checker import check_pruning_group
from heal_compress.pruning.grouped_conv import grouped_conv_pruning_fn
from heal_compress.pruning.pruning_fns import prune_bn, prune_conv_in, prune_conv_out
from heal_compress.pruning.selection import SelectionConfig, build_pruning_plan
from heal_compress.tracer.pruning_group import PruningGroup


class GroupedToyNet(nn.Module):
    def __init__(self, groups: int, per_group: int):
        super().__init__()
        channels = groups * per_group
        self.stem = nn.Conv2d(3, channels, 1, bias=False)
        self.grouped = nn.Conv2d(channels, channels, 3, padding=1, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.head = nn.Conv2d(channels, 2, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.grouped(x)
        x = self.bn(x)
        return self.head(x)


def _grouped_scope(model: GroupedToyNet, grouped_mode: str) -> PruningGroup:
    channels = model.grouped.out_channels
    scope = PruningGroup(
        group_id="toy.grouped_hidden",
        num_channels=channels,
        meta={"group_type": "plain"},
    )
    scope.add_dep("stem", model.stem, prune_conv_out, "out", reason="root_out")
    scope.add_dep(
        "grouped",
        model.grouped,
        grouped_conv_pruning_fn(grouped_mode),
        "out",
        reason=f"grouped_conv:{grouped_mode}",
    )
    scope.add_dep("bn", model.bn, prune_bn, "out", reason="bn")
    scope.add_dep("head", model.head, prune_conv_in, "in", reason="downstream_in")
    return scope


def _shared_low_local_importance(groups: int, per_group: int) -> torch.Tensor:
    base = torch.cat([
        torch.arange(0, per_group // 2, dtype=torch.float32),
        torch.arange(100, 100 + per_group // 2, dtype=torch.float32),
    ])
    return torch.cat([base + group_id for group_id in range(groups)])


def test_shared_local_mean_uses_scope_importance_and_repeats_same_local_positions():
    model = GroupedToyNet(groups=4, per_group=16).eval()
    scope = _grouped_scope(model, "keep_groups")
    scope_imp = _shared_low_local_importance(groups=4, per_group=16)

    plan = build_pruning_plan(
        [scope],
        {scope.group_id: scope_imp},
        SelectionConfig(
            prune_ratio=0.5,
            selection_mode="local_scope",
            group_conv_selection_mode="shared_local_mean",
            align=8,
            group_conv_align=8,
            min_channels=8,
        ),
    )

    concrete = plan.concrete_groups[0]
    report = plan.grouped_conv_reports[0]
    expected_prune = [group_id * 16 + local for group_id in range(4) for local in range(8)]

    assert concrete.prune_indices == expected_prune
    assert report["scope_importance_shape"] == [64]
    assert report["grouped_importance_matrix_shape"] == [4, 16]
    assert report["importance_source_items"] == ["scope_level_aggregated"]
    assert report["shared_local_positions"] == list(range(8, 16))
    assert report["expanded_prune_indices"] == expected_prune
    assert report["per_group_after"] == 8
    assert report["per_group_kept_count_align8"] is True


def test_independent_group_topk_allows_different_local_keep_maps_and_forward_passes():
    model = GroupedToyNet(groups=4, per_group=16).eval()
    scope = _grouped_scope(model, "independent_group_topk")
    imp = torch.zeros(64, dtype=torch.float32)
    desired_keep = {
        0: list(range(0, 8)),
        1: list(range(8, 16)),
        2: list(range(4, 12)),
        3: [0, 2, 4, 6, 8, 10, 12, 14],
    }
    for group_id, locals_ in desired_keep.items():
        for rank, local in enumerate(locals_):
            imp[group_id * 16 + local] = 100.0 + rank

    plan = build_pruning_plan(
        [scope],
        {scope.group_id: imp},
        SelectionConfig(
            prune_ratio=0.5,
            selection_mode="constrained_global",
            group_conv_selection_mode="independent_group_topk",
            align=8,
            group_conv_align=8,
            min_channels=8,
        ),
    )

    concrete = plan.concrete_groups[0]
    report = plan.grouped_conv_reports[0]

    assert report["group_keep_map"] == desired_keep
    assert report["per_group_kept_count"] == {0: 8, 1: 8, 2: 8, 3: 8}
    assert report["per_group_kept_count_align8"] is True
    assert report["groups_after"] == 4
    assert concrete.keep_indices == report["expanded_keep_indices"]
    assert {u.candidate_type for u in plan.selected_atomic_units} == {"grouped_independent_topk"}

    check = check_pruning_group(scope, concrete.keep_indices, group_conv_align=8)
    assert check["legal"], check["issues"]
    result = scope.prune(concrete.keep_indices)
    assert result["applied"]
    assert model.grouped.groups == 4
    assert model.grouped.in_channels == model.grouped.out_channels == 32
    assert model.grouped.in_channels // model.grouped.groups == 8
    assert model(torch.randn(2, 3, 8, 8)).shape == (2, 2, 8, 8)


def test_remove_groups_deletes_complete_group_blocks_and_updates_group_count():
    model = GroupedToyNet(groups=16, per_group=8).eval()
    scope = _grouped_scope(model, "remove_groups")
    imp = torch.cat([
        torch.full((8,), float(group_id))
        for group_id in range(16)
    ])

    plan = build_pruning_plan(
        [scope],
        {scope.group_id: imp},
        SelectionConfig(
            prune_ratio=0.5,
            selection_mode="constrained_global",
            group_conv_selection_mode="remove_groups",
            allow_remove_groups=True,
            align=8,
            group_conv_align=8,
            groups_align=8,
            min_groups_after_prune=8,
            min_channels=8,
        ),
    )

    concrete = plan.concrete_groups[0]
    report = plan.grouped_conv_reports[0]

    assert report["groups_after"] == 8
    assert report["groups_after"] % 8 == 0
    assert report["expanded_prune_indices"] == list(range(0, 64))
    assert concrete.keep_indices == list(range(64, 128))

    check = check_pruning_group(scope, concrete.keep_indices, group_conv_align=8)
    assert check["legal"], check["issues"]
    result = scope.prune(concrete.keep_indices)
    assert result["applied"]
    assert model.grouped.groups == 8
    assert model.grouped.in_channels == model.grouped.out_channels == 64
    assert model(torch.randn(2, 3, 8, 8)).shape == (2, 2, 8, 8)


def test_constrained_global_grouped_conv_avoids_naive_channel_topk():
    model = GroupedToyNet(groups=4, per_group=16).eval()
    scope = _grouped_scope(model, "keep_groups")
    imp = torch.arange(64, dtype=torch.float32)

    plan = build_pruning_plan(
        [scope],
        {scope.group_id: imp},
        SelectionConfig(
            prune_ratio=0.5,
            selection_mode="constrained_global",
            group_conv_selection_mode="shared_local_mean",
            align=8,
            group_conv_align=8,
            min_channels=8,
        ),
    )

    selected_types = {u.candidate_type for u in plan.selected_atomic_units}
    assert selected_types == {"grouped_shared_local_block"}
    assert all(idx % 16 < 8 for idx in plan.concrete_groups[0].prune_indices)
