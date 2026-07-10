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

from heal_compress.pruning.artifacts import save_v108_model_artifacts  # noqa: E402
from heal_compress.pruning.greedy_budget_selector import (  # noqa: E402
    V108PruningDomain,
    V108RankingUnit,
    select_greedy_global_budget,
    tp_floor_keep,
)
from heal_compress.pruning.grouped_pergroup8_policy import (  # noqa: E402
    grouped_per_group_aligned_independent_local_pruning,
)
from heal_compress.pruning.shape_invariants import (  # noqa: E402
    check_model_shape_invariants,
    snapshot_model_shape_invariants,
)
from tools.latency_lut.run_v109_param_budget_round4_pruner import (  # noqa: E402
    parse_args,
    validate_round_to,
)


def _plain_domain(num_channels: int, *, domain_id: str = "domain::plain") -> V108PruningDomain:
    units = [
        V108RankingUnit(
            pruning_domain_id=domain_id,
            root_module_name="conv",
            root_dim="out",
            num_root_channels=num_channels,
            coupled_unit_id=f"{domain_id}::idx{i}",
            root_channel_index=i,
            is_grouped_conv=False,
            importance_raw=float(i + 1),
            importance_normalized=float(i + 1),
        )
        for i in range(num_channels)
    ]
    return V108PruningDomain(
        pruning_domain_id=domain_id,
        root_module_name="conv",
        root_dim="out",
        num_root_channels=num_channels,
        units=units,
    )


def _grouped_domain(channels: int, groups: int, *, domain_id: str = "domain::grouped", base_score: float = 100.0) -> V108PruningDomain:
    per = channels // groups
    units = [
        V108RankingUnit(
            pruning_domain_id=domain_id,
            root_module_name="pyramid.stage1.grouped",
            root_dim="out",
            num_root_channels=channels,
            coupled_unit_id=f"{domain_id}::idx{i}",
            root_channel_index=i,
            is_grouped_conv=True,
            group_index=i // per,
            local_channel_index=i % per,
            grouped_local_unit_id=f"{domain_id}::g{i // per}::l{i % per}",
            importance_raw=base_score + float(i),
            importance_normalized=base_score + float(i),
        )
        for i in range(channels)
    ]
    return V108PruningDomain(
        pruning_domain_id=domain_id,
        root_module_name="pyramid.stage1.grouped",
        root_dim="out",
        num_root_channels=channels,
        units=units,
        is_grouped_conv=True,
        groups=groups,
        per_group=per,
    )


def test_parse_args_defaults_to_param_mode_and_round_to8() -> None:
    args = parse_args([])

    assert args.target_pruning_mode == "param"
    assert args.round_to == 8


def test_channel_mode_remains_v108_channel_slot_budget() -> None:
    domain = _plain_domain(16)

    result = select_greedy_global_budget(
        [domain],
        target_pruning_ratio=0.25,
        target_pruning_mode="channel",
        max_ch_sparsity=0.60,
        align_channels=4,
    )

    assert result.target_pruning_mode == "channel"
    assert result.actual_channel_prune_ratio_on_searchable_surface == pytest.approx(0.25)
    assert result.domain_plans["domain::plain"].final_n_pruned == 4


def test_param_mode_stops_when_predicted_param_budget_is_reached() -> None:
    domain = _plain_domain(16)
    savings = {"domain::plain": {idx: 10.0 for idx in range(16)}}

    result = select_greedy_global_budget(
        [domain],
        target_pruning_ratio=0.25,
        target_pruning_mode="param",
        predicted_total_params=160.0,
        param_savings_by_unit=savings,
        max_ch_sparsity=0.60,
        align_channels=4,
    )

    assert result.target_pruning_mode == "param"
    assert result.predicted_param_prune_ratio == pytest.approx(0.25)
    assert result.param_budget_overshoot_ratio == pytest.approx(0.0)
    assert result.domain_plans["domain::plain"].final_n_pruned == 4


def test_param_mode_can_use_exact_plan_level_param_predictor() -> None:
    domain = _plain_domain(16)

    def exact_ratio(plans: dict[str, object]) -> float:
        plan = plans["domain::plain"]
        return len(plan.prune_indices) / 8.0

    result = select_greedy_global_budget(
        [domain],
        target_pruning_ratio=0.50,
        target_pruning_mode="param",
        param_ratio_from_plans=exact_ratio,
        max_ch_sparsity=0.60,
        align_channels=4,
    )

    assert result.predicted_param_prune_ratio == pytest.approx(0.50)
    assert result.domain_plans["domain::plain"].final_n_pruned == 4


@pytest.mark.parametrize("round_to", [4, 8, 16, 32, 64, 128])
def test_round_to_accepts_supported_powers_of_two(round_to: int) -> None:
    assert validate_round_to(round_to) == round_to


@pytest.mark.parametrize("round_to", [0, 3, 6, 12, 256])
def test_round_to_rejects_invalid_values(round_to: int) -> None:
    with pytest.raises(ValueError):
        validate_round_to(round_to)


def test_align_channels_is_compatibility_fallback_but_round_to_wins() -> None:
    fallback = parse_args(["--align-channels", "4"])
    explicit = parse_args(["--align-channels", "4", "--round-to", "16"])

    assert fallback.round_to == 4
    assert explicit.round_to == 16


def test_ordinary_conv_round_to4_floor_keep_records_overshoot() -> None:
    rounded = tp_floor_keep(current_channels=18, raw_n_pruned=3, round_to=4)

    assert rounded.raw_keep == 15
    assert rounded.aligned_keep == 12
    assert rounded.final_n_pruned == 6
    assert rounded.rounding_overshoot_slots == 3


def test_grouped_round_to4_stage0_output_prune_is_skipped() -> None:
    decision = grouped_per_group_aligned_independent_local_pruning(
        module_name="pyramid.stage0.grouped",
        scores=torch.arange(128, dtype=torch.float32),
        groups=32,
        raw_prune_per_group=1,
        align=4,
    )

    assert decision.output_pruned is False
    assert decision.out_per_group_before == 4
    assert decision.out_per_group_after == 4
    assert decision.skipped_output_prune_reason == "per_group_after_below_round_to"


def test_grouped_round_to4_stage1_can_prune_per_group8_to4() -> None:
    decision = grouped_per_group_aligned_independent_local_pruning(
        module_name="pyramid.stage1.grouped",
        scores=torch.arange(256, dtype=torch.float32),
        groups=32,
        raw_prune_per_group=1,
        align=4,
    )

    assert decision.output_pruned is True
    assert decision.out_per_group_before == 8
    assert decision.out_per_group_after == 4
    assert all(len(v) == 4 for v in decision.per_group_output_keep_indices.values())


def test_unselected_grouped_domain_shape_report_does_not_claim_pruning() -> None:
    plain = _plain_domain(16, domain_id="domain::plain")
    grouped = _grouped_domain(256, 32, domain_id="domain::grouped", base_score=1000.0)

    result = select_greedy_global_budget(
        [plain, grouped],
        target_pruning_ratio=0.01,
        target_pruning_mode="channel",
        max_ch_sparsity=0.60,
        align_channels=4,
    )

    grouped_rows = [row for row in result.grouped_shape_rows if row["module_name"] == "pyramid.stage1.grouped"]
    assert grouped_rows
    assert grouped_rows[0]["output_pruned"] is False
    assert grouped_rows[0]["out_per_group_before"] == 8
    assert grouped_rows[0]["out_per_group_after"] == 8
    assert grouped_rows[0]["skipped_output_prune_reason"] == "not_selected"


def test_grouped_round_to4_stage2_can_prune_to8_but_not4_under_sixty_percent_cap() -> None:
    to_8 = grouped_per_group_aligned_independent_local_pruning(
        module_name="pyramid.stage2.grouped",
        scores=torch.arange(512, dtype=torch.float32),
        groups=32,
        raw_prune_per_group=5,
        align=4,
        max_ch_sparsity=0.60,
    )
    to_4 = grouped_per_group_aligned_independent_local_pruning(
        module_name="pyramid.stage2.grouped",
        scores=torch.arange(512, dtype=torch.float32),
        groups=32,
        raw_prune_per_group=9,
        align=4,
        max_ch_sparsity=0.60,
    )

    assert to_8.output_pruned is True
    assert to_8.out_per_group_after == 8
    assert to_4.output_pruned is False
    assert to_4.skipped_output_prune_reason == "max_ch_sparsity_exceeded"


def test_shape_invariant_allows_channel_changes_but_keeps_conv2d_spatial_config() -> None:
    before_model = nn.Sequential(nn.Conv2d(3, 8, 3, stride=2, padding=1, dilation=1, groups=1, bias=True))
    after_model = nn.Sequential(nn.Conv2d(3, 4, 3, stride=2, padding=1, dilation=1, groups=1, bias=True))

    report = check_model_shape_invariants(
        snapshot_model_shape_invariants(before_model),
        snapshot_model_shape_invariants(after_model),
    )

    assert report["passed"] is True
    assert report["non_channel_shape_violation_count"] == 0
    assert report["rows"][0]["allowed_channel_change"] is True


def test_shape_invariant_rejects_conv2d_kernel_size_changed_to_channel_count() -> None:
    before_model = nn.Sequential(nn.Conv2d(3, 8, 3, stride=2, padding=1, dilation=1, groups=1))
    after_model = nn.Sequential(nn.Conv2d(3, 4, 4, stride=2, padding=1, dilation=1, groups=1))

    report = check_model_shape_invariants(
        snapshot_model_shape_invariants(before_model),
        snapshot_model_shape_invariants(after_model),
    )

    assert report["passed"] is False
    assert report["non_channel_shape_violation_count"] == 1
    assert "kernel_size" in report["rows"][0]["violation_reason"]


def test_artifact_manifest_contains_v109_budget_and_shape_fields(tmp_path: Path) -> None:
    model = nn.Sequential(nn.Conv2d(3, 4, 1)).eval()
    manifest = {
        "target_pruning_mode": "param",
        "target_pruning_ratio": 0.3,
        "round_to": 4,
        "shape_invariant_report_path": "target_0.30/shape_invariant_report.json",
    }

    paths = save_v108_model_artifacts(
        model=model,
        models_dir=tmp_path,
        manifest=manifest,
        model_config="toy.yaml",
        checkpoint_source="toy.pth",
    )
    payload = torch.load(paths["state_dict_manifest"], map_location="cpu", weights_only=False)

    saved = payload["architecture_manifest"]
    assert saved["target_pruning_mode"] == "param"
    assert saved["round_to"] == 4
    assert saved["shape_invariant_report_path"].endswith("shape_invariant_report.json")
