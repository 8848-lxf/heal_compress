from __future__ import annotations

from pathlib import Path

import pytest

from pruning.config import GroupedConvConfig, GroupedConvSelectionPolicy, PruningConfig
from pruning.selection.grouped_conv import select_grouped_conv_channels
from pruning.types import SamplingPruningEntry, SamplingPruningRequest
from tools.experiments.run_lidar_pyramid_formal_pruner_validation import (
    ExperimentContractError,
    assert_independent_model_origins,
    build_frame_manifest,
    compare_shape_snapshots,
    compute_speedup,
    execute_build_if_preflight,
    physical_param_prune_ratio,
    scan_runtime_dependencies,
    select_target_grouped_conv_layers,
    validate_active_roots,
    validate_directional_contract,
    validate_engine_stage_order,
    validate_formal_defaults,
    validate_latency_scope,
    validate_one_shot_request,
    validate_strict_precision_rows,
    width_feasibility,
)


def _grouped_rows(count: int = 3) -> list[dict[str, object]]:
    return [
        {
            "module_path": f"pyramid_backbone.resnet.layer1.{index}.conv2",
            "module_type": "Conv2d",
            "groups": 32,
            "in_channels": 128,
            "out_channels": 128,
            "root_pruning_allowed": True,
            "dependency_input_pruning_allowed": True,
            "fixed_output_contract": False,
            "scope_id": f"scope::{index}",
        }
        for index in range(count)
    ]


def test_selects_exactly_three_real_pyramid_grouped_convs() -> None:
    selected = select_target_grouped_conv_layers(_grouped_rows())
    assert [row["module_path"] for row in selected] == [
        "pyramid_backbone.resnet.layer1.0.conv2",
        "pyramid_backbone.resnet.layer1.1.conv2",
        "pyramid_backbone.resnet.layer1.2.conv2",
    ]


def test_grouped_target_count_not_three_fails_closed() -> None:
    with pytest.raises(ExperimentContractError, match="exactly 3"):
        select_target_grouped_conv_layers(_grouped_rows(2))


def test_grouped_experiment_only_allows_target_active_roots() -> None:
    request = SamplingPruningRequest(
        entries=[SamplingPruningEntry("r", "s", "pyramid_backbone.resnet.layer1.0.conv2", "out", [0])]
    )
    validate_active_roots(request, {"pyramid_backbone.resnet.layer1.0.conv2"})
    request.entries.append(SamplingPruningEntry("bad", "x", "backbone_m1.conv", "out", [0]))
    with pytest.raises(ExperimentContractError, match="non-target active root"):
        validate_active_roots(request, {"pyramid_backbone.resnet.layer1.0.conv2"})


def test_taylor_independent_can_keep_different_local_positions() -> None:
    decision = select_grouped_conv_channels(
        {0: [9, 1, 8, 2, 7, 3, 6, 4], 1: [1, 9, 2, 8, 3, 7, 4, 6]},
        final_channels_per_group=4,
    )
    assert decision.group_keep_map[0] != decision.group_keep_map[1]


def test_l2_shared_position_uses_identical_positions() -> None:
    config = GroupedConvConfig(selection_policy=GroupedConvSelectionPolicy.SHARED_LOCAL_MEAN)
    decision = select_grouped_conv_channels(
        {0: [9, 1, 8, 2, 7, 3, 6, 4], 1: [1, 9, 2, 8, 3, 7, 4, 6]},
        final_channels_per_group=4,
        config=config,
    )
    assert decision.group_keep_map[0] == decision.group_keep_map[1]


def test_controlled_strategies_have_equal_final_shapes() -> None:
    left = {"modules": {"g": {"attrs": {"in_channels": 64, "out_channels": 64, "groups": 8}, "parameter_shapes": {"weight": [64, 8, 3, 3]}}}}
    assert compare_shape_snapshots(left, left)["shape_equivalent"] is True


def test_shape_mismatch_blocks_ap_comparison() -> None:
    left = {"modules": {"g": {"attrs": {"out_channels": 64}}}}
    right = {"modules": {"g": {"attrs": {"out_channels": 32}}}}
    with pytest.raises(ExperimentContractError, match="shape mismatch"):
        compare_shape_snapshots(left, right, require_equivalent=True)


def test_keep_width_infeasible_is_explicit_skip() -> None:
    assert width_feasibility(4, 8)["status"] == "skipped_infeasible"
    assert width_feasibility(16, 8)["status"] == "feasible"


def test_500_frame_manifest_hash_is_deterministic() -> None:
    ids = [f"{index:06d}" for index in range(600)]
    first = build_frame_manifest(ids, frame_count=500, dataset_config_hash="cfg", split="validation", seed=0)
    second = build_frame_manifest(ids, frame_count=500, dataset_config_hash="cfg", split="validation", seed=0)
    assert first["frame_count"] == 500
    assert first["frame_list_hash"] == second["frame_list_hash"]


def test_each_ratio_originates_from_original_checkpoint() -> None:
    rows = [{"model_id": f"prune_{ratio}", "source_checkpoint_hash": "original"} for ratio in range(1, 8)]
    assert_independent_model_origins(rows, "original")
    rows[-1]["source_checkpoint_hash"] = "prune_06"
    with pytest.raises(ExperimentContractError, match="original checkpoint"):
        assert_independent_model_origins(rows, "original")


def test_actual_parameter_ratio_comes_from_physical_counts() -> None:
    assert physical_param_prune_ratio({"parameter_count": 1000}, {"parameter_count": 700}) == pytest.approx(0.3)


def test_request_is_one_shot_and_frozen_before_execution() -> None:
    validate_one_shot_request(SamplingPruningRequest(one_shot=True))
    with pytest.raises(ExperimentContractError, match="one-shot"):
        validate_one_shot_request(SamplingPruningRequest(one_shot=False))


def test_formal_defaults_include_dense_alignment_four() -> None:
    report = validate_formal_defaults(PruningConfig())
    assert report["dense_conv_channel_alignment"] == 4


def test_formal_defaults_include_grouped_legal_set() -> None:
    report = validate_formal_defaults(PruningConfig())
    assert report["allowed_channels_per_group"] == [4, 8, 16, 32, 64, 128, 256, 512]


def test_directional_contract_protects_output_but_allows_input() -> None:
    validate_directional_contract(
        {"fixed_output_contract": True, "root_pruning_allowed": False, "input_dependency_pruning_allowed": True}
    )


def test_directional_contract_rejects_frozen_consumer_input() -> None:
    with pytest.raises(ExperimentContractError, match="dependency input"):
        validate_directional_contract(
            {"fixed_output_contract": True, "root_pruning_allowed": False, "input_dependency_pruning_allowed": False}
        )


def test_forward_latency_scope_excludes_loader_and_postprocess() -> None:
    validate_latency_scope({"includes": ["prepared_input", "model_forward", "cuda_synchronize"]})
    with pytest.raises(ExperimentContractError, match="forward-only"):
        validate_latency_scope({"includes": ["dataloader", "model_forward"]})


def test_speedup_uses_original_forward_p50() -> None:
    assert compute_speedup(original_p50_ms=10.0, candidate_p50_ms=4.0) == pytest.approx(2.5)


def test_preflight_failure_never_calls_engine_builder() -> None:
    calls: list[str] = []
    result = execute_build_if_preflight(False, lambda: calls.append("called"))
    assert result["build_called"] is False
    assert calls == []


@pytest.mark.parametrize("mode", ["strict_fp32", "strict_fp16", "strict_int8"])
def test_strict_precision_mismatch_fails(mode: str) -> None:
    requested = mode.removeprefix("strict_")
    rows = [{"canonical_node_name": "__canonical__x", "requested": requested, "realized": "fp16"}]
    if requested == "fp16":
        assert validate_strict_precision_rows(mode, rows)["passed"] is True
    else:
        assert validate_strict_precision_rows(mode, rows)["passed"] is False


def test_engine_checker_only_runs_after_successful_build() -> None:
    validate_engine_stage_order(["preflight", "build", "structure", "precision", "provenance", "smoke"])
    with pytest.raises(ExperimentContractError, match="stage order"):
        validate_engine_stage_order(["structure", "build"])


def test_experiment_runtime_has_no_test_or_legacy_algorithm_imports() -> None:
    path = Path("tools/experiments/run_lidar_pyramid_formal_pruner_validation.py")
    report = scan_runtime_dependencies(path)
    assert report["forbidden_imports"] == []

