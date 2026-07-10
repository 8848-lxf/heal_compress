from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
UNIAD = ROOT.parent
for path in (UNIAD, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from heal_compress.pruning.artifacts import (  # noqa: E402
    load_v108_model_object,
    save_v108_model_artifacts,
    smoke_v108_model,
)
from heal_compress.pruning.greedy_budget_selector import (  # noqa: E402
    V108PruningDomain,
    V108RankingUnit,
    select_greedy_global_budget,
    tp_floor_keep,
)
from heal_compress.pruning.grouped_pergroup8_policy import (  # noqa: E402
    grouped_per_group_aligned_independent_local_pruning,
)
from heal_compress.pruning.protection_policy import apply_v108_default_protection  # noqa: E402
from heal_compress.pruning.pruning_fns import prune_bn, prune_conv_in, prune_conv_out  # noqa: E402
from heal_compress.pruning.taylor_importance import compute_taylor_importance_for_scope  # noqa: E402
from heal_compress.tracer.pruning_group import PruningGroup  # noqa: E402
from tools.latency_lut.run_v108_complete_taylor_greedy_pruner import ensure_v108_eval_dir  # noqa: E402


def _plain_domain(num_channels: int) -> V108PruningDomain:
    units = [
        V108RankingUnit(
            pruning_domain_id="domain::plain",
            root_module_name="conv",
            root_dim="out",
            num_root_channels=num_channels,
            coupled_unit_id=f"domain::plain::idx{i}",
            root_channel_index=i,
            is_grouped_conv=False,
            importance_raw=float(i + 1),
            importance_normalized=float(i + 1),
        )
        for i in range(num_channels)
    ]
    return V108PruningDomain(
        pruning_domain_id="domain::plain",
        root_module_name="conv",
        root_dim="out",
        num_root_channels=num_channels,
        units=units,
    )


def test_coupled_unit_taylor_importance_uses_dependency_mean_and_domain_normalizer() -> None:
    conv1 = nn.Conv2d(1, 2, 1, bias=False)
    bn = nn.BatchNorm2d(2)
    conv2 = nn.Conv2d(2, 1, 1, bias=False)
    with torch.no_grad():
        conv1.weight.copy_(torch.tensor([[[[1.0]]], [[[2.0]]]]))
        bn.weight.copy_(torch.tensor([2.0, 4.0]))
        bn.bias.zero_()
        conv2.weight.copy_(torch.tensor([[[[3.0]], [[5.0]]]]))
    conv1.weight.grad = torch.tensor([[[[0.5]]], [[[1.5]]]])
    bn.weight.grad = torch.tensor([1.0, 1.0])
    bn.bias.grad = torch.zeros(2)
    conv2.weight.grad = torch.tensor([[[[2.0]], [[4.0]]]])

    scope = PruningGroup(group_id="group::conv1", num_channels=2)
    scope.add_dep("conv1", conv1, prune_conv_out, "out", reason="root_out")
    scope.add_dep("bn", bn, prune_bn, "out", reason="conv_bn")
    scope.add_dep("conv2", conv2, prune_conv_in, "in", reason="downstream_in")

    result = compute_taylor_importance_for_scope(scope)

    raw0 = (0.5 + 2.0 + 6.0) / 3.0
    raw1 = (3.0 + 4.0 + 20.0) / 3.0
    domain_mean = (raw0 + raw1) / 2.0
    assert result.raw_scores.tolist() == pytest.approx([raw0, raw1])
    assert result.normalized_scores.tolist() == pytest.approx([raw0 / domain_mean, raw1 / domain_mean])
    assert result.unit_rows[0]["importance_raw"] == pytest.approx(raw0)
    assert result.unit_rows[1]["importance_normalized"] == pytest.approx(raw1 / domain_mean)


def test_tp_style_round_to8_floors_keep_and_records_overpruning() -> None:
    rounded = tp_floor_keep(current_channels=34, raw_n_pruned=3, round_to=8)

    assert rounded.raw_keep == 31
    assert rounded.aligned_keep == 24
    assert rounded.final_n_pruned == 10
    assert rounded.rounding_overshoot_slots == 7
    assert rounded.rounding_overshoot_ratio == pytest.approx(7 / 34)


def test_max_ch_sparsity_blocks_domain_above_sixty_percent() -> None:
    domain = _plain_domain(64)

    result = select_greedy_global_budget(
        [domain],
        target_pruning_ratio=0.70,
        max_ch_sparsity=0.60,
        align_channels=8,
    )

    assert result.actual_channel_prune_ratio_on_searchable_surface <= 0.60
    assert result.max_ch_sparsity_blocked_count > 0
    assert result.unreachable is True
    assert result.domain_plans["domain::plain"].final_n_pruned == 32


def test_structural_skipped_domain_is_not_protected_but_not_selected() -> None:
    domain = _plain_domain(64)
    domain.skipped_reason = "fixed_shape_interface_structural_illegal"
    for unit in domain.units:
        unit.skipped_reason = domain.skipped_reason

    result = select_greedy_global_budget(
        [domain],
        target_pruning_ratio=0.10,
        max_ch_sparsity=0.60,
        align_channels=8,
    )

    assert domain.protected_reason == ""
    assert result.actual_channel_prune_ratio_on_searchable_surface == 0.0
    assert result.unreachable is True
    assert result.skipped_units[0].skipped_reason == "fixed_shape_interface_structural_illegal"


@pytest.mark.parametrize(
    ("channels", "groups", "raw_prune_per_group", "reason"),
    [
        (128, 32, 1, "per_group_after_below8"),
        (256, 32, 1, "per_group_after_below8"),
    ],
)
def test_grouped_conv_stage0_and_stage1_output_prune_skipped_below8(
    channels: int,
    groups: int,
    raw_prune_per_group: int,
    reason: str,
) -> None:
    scores = torch.arange(channels, dtype=torch.float32)

    decision = grouped_per_group_aligned_independent_local_pruning(
        module_name="pyramid.grouped",
        scores=scores,
        groups=groups,
        raw_prune_per_group=raw_prune_per_group,
    )

    assert decision.output_pruned is False
    assert decision.out_per_group_after == channels // groups
    assert decision.skipped_output_prune_reason == reason
    assert decision.legality_passed is True


def test_grouped_conv_stage2_can_prune_per_group16_to8() -> None:
    scores = torch.arange(512, dtype=torch.float32)

    decision = grouped_per_group_aligned_independent_local_pruning(
        module_name="pyramid.stage2.grouped",
        scores=scores,
        groups=32,
        raw_prune_per_group=4,
    )

    assert decision.output_pruned is True
    assert decision.out_per_group_before == 16
    assert decision.out_per_group_after == 8
    assert len(decision.global_keep_indices) == 256
    assert all(len(v) == 8 for v in decision.per_group_output_keep_indices.values())


def test_grouped_conv_independent_local_ranking_keeps_same_count_without_cross_group_reorder() -> None:
    scores = torch.tensor(
        list(range(16)) + list(reversed(range(16))),
        dtype=torch.float32,
    )

    decision = grouped_per_group_aligned_independent_local_pruning(
        module_name="toy.grouped",
        scores=scores,
        groups=2,
        raw_prune_per_group=4,
    )

    assert decision.per_group_output_keep_indices[0] == list(range(8, 16))
    assert decision.per_group_output_keep_indices[1] == list(range(0, 8))
    assert len(decision.per_group_output_keep_indices[0]) == len(decision.per_group_output_keep_indices[1])
    assert decision.global_keep_indices == list(range(8, 16)) + list(range(16, 24))


def test_fpn_and_head_outputs_are_protected_but_other_outputs_are_not() -> None:
    conv = nn.Conv2d(8, 16, 1)
    grouped = nn.Conv2d(128, 128, 3, padding=1, groups=32)
    groups: list[PruningGroup] = []
    for name, module in [
        ("neck.fpn_out", conv),
        ("cls_head", conv),
        ("pyramid_backbone.blocks.0.conv2", grouped),
        ("backbone.conv", conv),
    ]:
        scope = PruningGroup(group_id=f"group::{name}", num_channels=int(module.out_channels))
        scope.add_dep(name, module, prune_conv_out, "out", reason="root_out")
        groups.append(scope)

    report = apply_v108_default_protection(groups)

    by_id = {group.group_id: group for group in groups}
    assert by_id["group::neck.fpn_out"].protected is True
    assert by_id["group::cls_head"].protected is True
    assert by_id["group::pyramid_backbone.blocks.0.conv2"].protected is False
    assert by_id["group::backbone.conv"].protected is False
    assert "pyramid_backbone.blocks.0.conv2" in report["unprotected_grouped_conv_outputs"]
    assert report["total_protected_units"] == 32


def test_v108_model_artifact_save_reload_and_smoke(tmp_path: Path) -> None:
    model = nn.Sequential(nn.Conv2d(3, 8, 1), nn.ReLU(), nn.Conv2d(8, 2, 1)).eval()
    manifest = {"target_pruning_ratio": 0.1, "requires_architecture_patch": True}

    paths = save_v108_model_artifacts(
        model=model,
        models_dir=tmp_path,
        manifest=manifest,
        model_config="toy.yaml",
        checkpoint_source="toy.pth",
    )
    loaded = load_v108_model_object(paths["model_object"], device=torch.device("cpu"))
    smoke = smoke_v108_model(loaded, torch.randn(1, 3, 4, 4))

    assert paths["model_object"].name == "pruned_model_object.pth"
    assert paths["state_dict_manifest"].name == "pruned_state_dict_with_manifest.pth"
    assert paths["model_object"].is_file()
    assert paths["state_dict_manifest"].is_file()
    assert smoke["reload_forward_smoke_passed"] is True


def test_v108_eval_dir_helper_creates_logger_parent(tmp_path: Path) -> None:
    eval_dir = tmp_path / "nested" / "real_val500"

    resolved = ensure_v108_eval_dir(eval_dir)

    assert resolved == eval_dir
    assert eval_dir.is_dir()
