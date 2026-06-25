from __future__ import annotations

import torch
import torch.nn as nn

from heal_compress.pruning.pruning_fns import prune_conv_out
from heal_compress.pruning.selection import SelectionConfig, build_pruning_plan
from heal_compress.tracer.pruning_group import PruningGroup


def _plain_scope(scope_id: str, scores: torch.Tensor) -> tuple[PruningGroup, torch.Tensor]:
    conv = nn.Conv2d(3, int(scores.numel()), 1, bias=False)
    scope = PruningGroup(
        group_id=scope_id,
        num_channels=int(scores.numel()),
        meta={"group_type": "plain"},
    )
    scope.add_dep(scope_id.replace(".", "_"), conv, prune_conv_out, "out", reason="root_out")
    return scope, scores.float()


def test_local_scope_prunes_each_dependency_scope_independently():
    scope_a, imp_a = _plain_scope("scope.A", torch.tensor([100.0, 101.0, 102.0, 103.0]))
    scope_b, imp_b = _plain_scope("scope.B", torch.tensor([1.0, 2.0, 3.0, 4.0]))

    plan = build_pruning_plan(
        [scope_a, scope_b],
        {"scope.A": imp_a, "scope.B": imp_b},
        SelectionConfig(prune_ratio=0.5, selection_mode="local_scope", align=1, min_channels=1),
    )

    by_scope = {g.scope_id: g.prune_indices for g in plan.concrete_groups}
    assert by_scope["scope.A"] == [0, 1]
    assert by_scope["scope.B"] == [0, 1]


def test_global_coupled_channel_sorts_units_globally_not_whole_scopes():
    scope_a, imp_a = _plain_scope("scope.A", torch.tensor([100.0, 101.0, 102.0, 103.0]))
    scope_b, imp_b = _plain_scope("scope.B", torch.tensor([1.0, 2.0, 3.0, 4.0]))

    plan = build_pruning_plan(
        [scope_a, scope_b],
        {"scope.A": imp_a, "scope.B": imp_b},
        SelectionConfig(
            prune_ratio=0.5,
            selection_mode="global_coupled_channel",
            align=1,
            min_channels=1,
        ),
    )

    by_scope = {g.scope_id: g.prune_indices for g in plan.concrete_groups}
    assert by_scope["scope.A"] == [0]
    assert by_scope["scope.B"] == [0, 1, 2]
    assert {u.candidate_type for u in plan.selected_atomic_units} == {"plain_channel"}
