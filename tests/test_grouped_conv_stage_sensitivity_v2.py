from __future__ import annotations

from copy import deepcopy

import pytest
import torch
from torch import nn

from tools.experiments.grouped_conv_stage_sensitivity.contracts import (
    EXPECTED_STAGE_COUNTS,
    STAGE_PREFIXES,
    build_candidate_matrix,
    build_stage_inventory,
    compute_pruning_strength_plan,
    filter_active_root_units,
    select_independent_group_positions,
    select_tp_shared_positions,
)
from tools.experiments.grouped_conv_stage_sensitivity.analysis import (
    build_sensitivity_rankings,
    build_strategy_comparisons,
    cumulative_interaction,
    physical_shape_hash,
    stage1_attribution,
    validate_pairwise_structure,
)
from tools.experiments.grouped_conv_stage_sensitivity.torch_pruning_strategy import (
    torch_pruning_l2_channel_scores,
)
from tools.experiments.grouped_conv_stage_sensitivity.pytorch_evaluator import (
    apply_baseline_metrics,
    frame_list_hash,
    validate_manifest_binding,
)
from tools.experiments.grouped_conv_stage_sensitivity.runtime import (
    build_controlled_strategy_request,
    build_frame_manifest,
    build_group_score_records,
    parameter_reduction_breakdown,
    build_tp_importance_result,
    tensor_output_contract,
    state_dict_content_hash,
    validate_core_freeze,
)
from tools.experiments.run_grouped_conv_stage_sensitivity import (
    build_generation_summary,
    build_parser,
    scan_forbidden_runtime_dependencies,
)
from pruning.types import ImportanceResult
from tracer.types import AtomicPruneUnit, DependencyMember


def _inventory_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for stage, count, width in (("stage0", 3, 4), ("stage1", 5, 8), ("stage2", 8, 16)):
        prefix = STAGE_PREFIXES[stage]
        for index in range(count):
            rows.append(
                {
                    "module_path": f"{prefix}{index}.conv2",
                    "module_type": "Conv2d",
                    "in_channels": 32 * width,
                    "out_channels": 32 * width,
                    "groups": 32,
                    "scope_id": f"scope::{stage}::{index}",
                    "root_pruning_allowed": True,
                    "dependency_input_pruning_allowed": True,
                    "root_axis": "out",
                    "protected_reason": "",
                    "trace_source": "formal_trace",
                }
            )
    return rows


def test_inventory_classifies_all_sixteen_modules_by_explicit_prefix() -> None:
    report = build_stage_inventory(_inventory_rows())

    assert report["observed_stage_counts"] == EXPECTED_STAGE_COUNTS
    assert report["total_count"] == 16
    assert report["inventory_count_mismatch"] is False
    assert report["unclassified_grouped_modules"] == []
    assert all(row["in_channels_divisible_by_groups"] for row in report["rows"])
    assert all(row["out_channels_divisible_by_groups"] for row in report["rows"])


def test_inventory_never_treats_three_stages_as_exactly_three_layers() -> None:
    report = build_stage_inventory(_inventory_rows()[:3])

    assert report["inventory_count_mismatch"] is True
    assert report["total_count"] == 3
    assert report["observed_stage_counts"] == {"stage0": 3, "stage1": 0, "stage2": 0}
    assert report["missing_module_paths"]


def test_unmatched_grouped_module_is_reported_not_silently_assigned() -> None:
    rows = _inventory_rows()
    rows.append(
        {
            **rows[0],
            "module_path": "pyramid_backbone.resnet.unknown.0.conv2",
            "scope_id": "scope::unknown",
        }
    )
    report = build_stage_inventory(rows)

    assert report["unclassified_grouped_modules"] == [
        "pyramid_backbone.resnet.unknown.0.conv2"
    ]


def test_strength_plan_marks_stage0_infeasible_and_uses_common_legal_widths() -> None:
    plan = compute_pruning_strength_plan(build_stage_inventory(_inventory_rows())["rows"])

    assert plan["stages"]["stage0"]["status"] == "stage_output_pruning_infeasible"
    assert plan["stages"]["stage0"]["mild"] is None
    assert plan["stages"]["stage1"]["mild"] == 4
    assert plan["stages"]["stage1"]["aggressive"] is None
    assert plan["stages"]["stage1"]["aggressive_status"] == "aggressive_unavailable"
    assert plan["stages"]["stage2"]["mild"] == 8
    assert plan["stages"]["stage2"]["aggressive"] == 4


def test_candidate_matrix_records_infeasible_rows_without_noop_substitution() -> None:
    strength = compute_pruning_strength_plan(build_stage_inventory(_inventory_rows())["rows"])
    rows = build_candidate_matrix(strength)

    assert len(rows) == 16
    feasible = [row for row in rows if row["candidate_status"] == "planned"]
    infeasible = [row for row in rows if row["candidate_status"] == "infeasible"]
    assert len(feasible) == 6
    assert len(infeasible) == 10
    assert all(row["target_widths_by_stage"] for row in feasible)
    assert all(row["infeasible_reason"] for row in infeasible)
    assert not any(
        row["stage_scope"] == "all_grouped_stages" and row["candidate_status"] == "planned"
        for row in rows
    )


def test_taylor_independent_ranking_can_keep_different_local_positions() -> None:
    result = select_independent_group_positions(
        {
            0: [9.0, 1.0, 8.0, 2.0, 7.0, 3.0, 6.0, 4.0],
            1: [1.0, 9.0, 2.0, 8.0, 3.0, 7.0, 4.0, 6.0],
        },
        keep_width=4,
    )

    assert result["group_keep_map"] == {0: [0, 2, 4, 6], 1: [1, 3, 5, 7]}
    assert result["group_prune_map"] == {0: [1, 3, 5, 7], 1: [0, 2, 4, 6]}
    assert result["selection"] == "independent_group_topk"


def test_tp_shared_position_ranking_repeats_identical_positions_per_group() -> None:
    result = select_tp_shared_positions(
        {
            0: [9.0, 1.0, 8.0, 2.0, 7.0, 3.0, 6.0, 4.0],
            1: [1.0, 9.0, 2.0, 8.0, 3.0, 7.0, 4.0, 6.0],
        },
        keep_width=4,
    )

    assert result["shared_local_mean_scores"] == pytest.approx([5.0] * 8)
    assert result["group_keep_map"] == {0: [0, 1, 2, 3], 1: [0, 1, 2, 3]}
    assert result["group_prune_map"] == {0: [4, 5, 6, 7], 1: [4, 5, 6, 7]}
    assert result["selection"] == "shared_position_topk"


def test_torch_pruning_importance_api_returns_real_per_channel_scores() -> None:
    module = nn.Conv2d(8, 8, kernel_size=3, groups=2, bias=False)
    result = torch_pruning_l2_channel_scores(module, module_path="grouped")
    expected = module.weight.detach().flatten(1).abs().pow(2).sum(1).cpu()

    assert result["torch_pruning_api_used"] is True
    assert result["importance_class"] == "MagnitudeImportance"
    assert result["constructor_args"] == {
        "p": 2,
        "group_reduction": "mean",
        "normalizer": None,
        "bias": False,
    }
    assert torch.tensor(result["scores"]).shape == (8,)
    assert torch.allclose(torch.tensor(result["scores"]), expected)


def test_root_whitelist_filters_only_active_grouped_roots() -> None:
    units = [
        {"stable_id": "a", "root_module_path": "pyramid_backbone.resnet.layer1.0.conv2"},
        {"stable_id": "b", "root_module_path": "pyramid_backbone.resnet.layer2.0.conv2"},
        {"stable_id": "c", "root_module_path": "cls_head"},
    ]

    selected = filter_active_root_units(
        units, {"pyramid_backbone.resnet.layer1.0.conv2"}
    )

    assert [row["stable_id"] for row in selected] == ["a"]


def test_physical_shape_hash_excludes_weight_content() -> None:
    left = {
        "parameter_count": 12,
        "modules": [
            {
                "canonical_module_name": "g",
                "module_type": "Conv2d",
                "weight_shape": [4, 2, 1, 1],
                "bias_shape": [],
                "in_channels": 4,
                "out_channels": 4,
                "groups": 2,
                "kernel_size": [1, 1],
                "stride": [1, 1],
                "padding": [0, 0],
                "dilation": [1, 1],
                "weight_content": [1.0],
            }
        ],
    }
    right = deepcopy(left)
    right["modules"][0]["weight_content"] = [999.0]

    assert physical_shape_hash(left) == physical_shape_hash(right)
    validation = validate_pairwise_structure(left, right)
    assert validation["comparison_valid"] is True
    assert validation["physical_shape_hash_match"] is True


def test_pairwise_structure_rejects_parameter_count_or_shape_mismatch() -> None:
    left = {
        "parameter_count": 12,
        "modules": [{"canonical_module_name": "g", "module_type": "Conv2d", "weight_shape": [4, 2]}],
    }
    right = deepcopy(left)
    right["modules"][0]["weight_shape"] = [2, 2]
    right["parameter_count"] = 8

    validation = validate_pairwise_structure(left, right)
    assert validation["comparison_valid"] is False
    assert validation["failure_reason"] == "structure_mismatch"


def test_cumulative_interaction_uses_declared_descriptive_formula() -> None:
    result = cumulative_interaction(
        all_stage_drop=0.30,
        single_stage_drops={"stage0": 0.05, "stage1": 0.10, "stage2": 0.08},
    )

    assert result["sum_single_stage_delta_map"] == pytest.approx(0.23)
    assert result["cumulative_interaction"] == pytest.approx(0.07)
    assert result["interaction_class"] == "superadditive"


def _trace_atoms_for_one_grouped_root() -> tuple[list[AtomicPruneUnit], ImportanceResult]:
    root = "pyramid_backbone.resnet.layer1.0.conv2"
    atoms: list[AtomicPruneUnit] = []
    raw_scores: dict[str, float] = {}
    normalized_scores: dict[str, float] = {}
    for absolute_index in range(16):
        coupled_id = f"cu::{absolute_index}"
        atom = AtomicPruneUnit(
            scope_id="scope::grouped",
            root_module_path=root,
            root_axis="out",
            root_indices=[absolute_index],
            source_coupled_unit_ids=[coupled_id],
            members=[
                DependencyMember(
                    module_path=root,
                    module_type="Conv2d",
                    axis="out",
                    indices=[absolute_index],
                    dependency_type="root_output",
                ),
                DependencyMember(
                    module_path="pyramid_backbone.resnet.layer1.0.bn2",
                    module_type="BatchNorm2d",
                    axis="channel",
                    indices=[absolute_index],
                    dependency_type="batchnorm_output",
                ),
            ],
            constraints={
                "grouped_conv": True,
                "grouped_module_path": root,
                "groups": 2,
                "channels_per_group": 8,
                "original_channel_count": 16,
                "depthwise": False,
            },
        )
        atoms.append(atom)
        group, local = divmod(absolute_index, 8)
        score = float(local if group == 0 else 7 - local)
        raw_scores[coupled_id] = score
        normalized_scores[coupled_id] = score
    importance = ImportanceResult(
        mode="controlled",
        normalization="none",
        aggregation="root_channel",
        raw_scores=raw_scores,
        normalized_scores=normalized_scores,
        unit_parameter_costs={key: 1 for key in raw_scores},
    )
    return atoms, importance


def test_controlled_request_uses_formal_closure_and_exact_independent_width() -> None:
    atoms, importance = _trace_atoms_for_one_grouped_root()
    root = "pyramid_backbone.resnet.layer1.0.conv2"

    result = build_controlled_strategy_request(
        atoms,
        importance,
        allowed_roots={root},
        target_width=4,
        strategy="taylor_independent_group_ranking",
    )

    request = result["request"]
    root_entry = next(row for row in request.entries if row.module_path == root)
    bn_entry = next(row for row in request.entries if row.module_path.endswith("bn2"))
    assert request.requested_channel_cost == 8
    assert root_entry.group_keep_map == {0: [4, 5, 6, 7], 1: [0, 1, 2, 3]}
    assert len(root_entry.group_keep_map[0]) == 4
    assert bn_entry.metadata["dependency_driven"] is True
    assert result["selected_active_roots"] == [root]


def test_controlled_request_tp_shared_uses_identical_local_positions() -> None:
    atoms, importance = _trace_atoms_for_one_grouped_root()
    root = "pyramid_backbone.resnet.layer1.0.conv2"

    result = build_controlled_strategy_request(
        atoms,
        importance,
        allowed_roots={root},
        target_width=4,
        strategy="torch_pruning_l2_shared_position",
    )

    root_entry = next(row for row in result["request"].entries if row.module_path == root)
    assert root_entry.group_keep_map[0] == root_entry.group_keep_map[1]
    assert root_entry.group_keep_map == {0: [0, 1, 2, 3], 1: [0, 1, 2, 3]}


def test_controlled_request_rejects_missing_whitelisted_root() -> None:
    atoms, importance = _trace_atoms_for_one_grouped_root()
    with pytest.raises(ValueError, match="missing grouped root units"):
        build_controlled_strategy_request(
            atoms,
            importance,
            allowed_roots={"pyramid_backbone.resnet.layer2.0.conv2"},
            target_width=4,
            strategy="taylor_independent_group_ranking",
        )


def test_500_frame_manifest_binds_config_checkpoint_and_evaluator_hashes() -> None:
    frame_ids = [f"{index:06d}" for index in range(600)]
    first = build_frame_manifest(
        frame_ids,
        frame_count=500,
        dataset_config_hash="config",
        checkpoint_hash="checkpoint",
        evaluation_code_hash="evaluator",
    )
    second = build_frame_manifest(
        frame_ids,
        frame_count=500,
        dataset_config_hash="config",
        checkpoint_hash="checkpoint",
        evaluation_code_hash="evaluator",
    )

    assert first == second
    assert first["frame_count"] == 500
    assert first["frame_ids"] == frame_ids[:500]
    assert first["dataset_config_hash"] == "config"
    assert first["checkpoint_hash"] == "checkpoint"
    assert first["evaluation_code_hash"] == "evaluator"


def test_state_dict_content_hash_changes_with_weights_not_mapping_order() -> None:
    first = {"a": torch.tensor([1.0, 2.0]), "b": torch.tensor([3], dtype=torch.int64)}
    reordered = {"b": first["b"].clone(), "a": first["a"].clone()}
    changed = {"a": torch.tensor([1.0, 9.0]), "b": first["b"].clone()}

    assert state_dict_content_hash(first) == state_dict_content_hash(reordered)
    assert state_dict_content_hash(first) != state_dict_content_hash(changed)


def test_state_dict_content_hash_supports_scalar_integer_buffers() -> None:
    state = {
        "batch_counter": torch.tensor(0, dtype=torch.int64),
        "weight": torch.tensor([1.0]),
    }

    assert len(state_dict_content_hash(state)) == 64


def test_core_freeze_validation_detects_any_hash_change(tmp_path) -> None:
    frozen = tmp_path / "core.py"
    frozen.write_text("before\n", encoding="utf-8")
    before = {
        "frozen_files": [
            {
                "path": "core.py",
                "exists": True,
                "sha256": __import__("hashlib").sha256(b"before\n").hexdigest(),
            }
        ]
    }

    passed = validate_core_freeze(before, repository_root=tmp_path)
    frozen.write_text("after\n", encoding="utf-8")
    failed = validate_core_freeze(before, repository_root=tmp_path)

    assert passed["core_files_unchanged"] is True
    assert failed["core_files_unchanged"] is False
    assert failed["status"] == "invalid_due_to_core_file_modification"


def test_tp_scores_map_to_formal_importance_result_and_atomic_ids() -> None:
    atoms, _ = _trace_atoms_for_one_grouped_root()
    root = "pyramid_backbone.resnet.layer1.0.conv2"
    model = nn.Module()
    model.add_module("pyramid_backbone", nn.Module())
    model.pyramid_backbone.add_module("resnet", nn.Module())
    model.pyramid_backbone.resnet.add_module("layer1", nn.Module())
    model.pyramid_backbone.resnet.layer1.add_module("0", nn.Module())
    model.pyramid_backbone.resnet.layer1._modules["0"].add_module(
        "conv2", nn.Conv2d(16, 16, kernel_size=1, groups=2, bias=False)
    )

    result = build_tp_importance_result(model, atoms, allowed_roots={root})

    assert result["importance"].mode == "torch_pruning_l2"
    assert set(result["importance"].raw_scores) == {f"cu::{index}" for index in range(16)}
    assert result["torch_pruning_api_used"] is True
    assert result["modules"][0]["returned_per_channel_importance"] is True


def test_runner_exposes_only_pytorch_experiment_stages() -> None:
    choices = build_parser()._actions[1].choices
    assert set(choices) == {"prepare", "dry_run", "generate", "evaluate", "analyze", "finalize"}


def test_runner_has_no_quantization_onnx_tensorrt_or_trtexec_dependency() -> None:
    report = scan_forbidden_runtime_dependencies(
        "tools/experiments/run_grouped_conv_stage_sensitivity.py"
    )
    assert report["forbidden_imports"] == []
    assert report["forbidden_calls"] == []


def test_group_score_records_capture_independent_rank_and_keep_decision() -> None:
    atoms, importance = _trace_atoms_for_one_grouped_root()
    root = "pyramid_backbone.resnet.layer1.0.conv2"
    request_result = build_controlled_strategy_request(
        atoms,
        importance,
        allowed_roots={root},
        target_width=4,
        strategy="taylor_independent_group_ranking",
    )

    rows = build_group_score_records(
        atoms,
        importance,
        request_result["request"],
        strategy="taylor_independent_group_ranking",
    )

    assert len(rows) == 16
    group0_local7 = next(row for row in rows if row["group_id"] == 0 and row["local_position"] == 7)
    group1_local0 = next(row for row in rows if row["group_id"] == 1 and row["local_position"] == 0)
    assert group0_local7["rank_within_group"] == 1
    assert group1_local0["rank_within_group"] == 1
    assert group0_local7["kept"] is True
    assert group1_local0["kept"] is True


def test_group_score_records_capture_tp_shared_rank() -> None:
    atoms, importance = _trace_atoms_for_one_grouped_root()
    root = "pyramid_backbone.resnet.layer1.0.conv2"
    request_result = build_controlled_strategy_request(
        atoms,
        importance,
        allowed_roots={root},
        target_width=4,
        strategy="torch_pruning_l2_shared_position",
    )

    rows = build_group_score_records(
        atoms,
        importance,
        request_result["request"],
        strategy="torch_pruning_l2_shared_position",
    )

    assert {row["shared_local_mean_score"] for row in rows} == {3.5}
    assert all(row["shared_rank"] == row["local_position"] + 1 for row in rows)
    assert all(row["torch_pruning_api_used"] is True for row in rows)


def test_tensor_output_contract_checks_shapes_and_finiteness() -> None:
    passed = tensor_output_contract(
        {"cls": torch.ones(1, 2), "nested": [torch.zeros(3)]}
    )
    failed = tensor_output_contract({"cls": torch.tensor([float("nan")])})

    assert passed["all_finite"] is True
    assert passed["tensor_shapes"] == {"cls": [1, 2], "nested.0": [3]}
    assert failed["all_finite"] is False
    assert failed["non_finite_tensor_paths"] == ["cls"]


def test_parameter_breakdown_separates_active_roots_from_dependency_closure() -> None:
    original = {
        "modules": [
            {"canonical_module_name": "root", "parameter_count": 100},
            {"canonical_module_name": "dep", "parameter_count": 80},
            {"canonical_module_name": "untouched", "parameter_count": 50},
        ]
    }
    candidate = {
        "modules": [
            {"canonical_module_name": "root", "parameter_count": 60},
            {"canonical_module_name": "dep", "parameter_count": 50},
            {"canonical_module_name": "untouched", "parameter_count": 50},
        ]
    }

    result = parameter_reduction_breakdown(
        original,
        candidate,
        active_roots={"root"},
        closure_modules={"root", "dep"},
    )

    assert result["active_root_parameter_reduction"] == 40
    assert result["dependency_driven_parameter_reduction"] == 30
    assert result["total_parameter_reduction_in_closure"] == 70


def test_generation_summary_keeps_infeasible_distinct_from_invalid() -> None:
    matrix = [
        {"candidate_id": "valid", "candidate_status": "planned"},
        {"candidate_id": "invalid", "candidate_status": "planned"},
        {
            "candidate_id": "infeasible",
            "candidate_status": "infeasible",
            "infeasible_reason": "stage_output_pruning_infeasible",
        },
    ]
    generated = [
        {"candidate_id": "valid", "candidate_status": "valid"},
        {
            "candidate_id": "invalid",
            "candidate_status": "invalid",
            "failure_stage": "forward",
            "failure_reason": "shape",
        },
    ]

    summary = build_generation_summary(matrix, generated)

    assert summary["valid_candidate_count"] == 1
    assert summary["invalid_candidate_count"] == 1
    assert summary["infeasible_candidate_count"] == 1
    assert summary["rows"][2]["candidate_status"] == "infeasible"


def test_manifest_binding_requires_exact_order_and_count() -> None:
    manifest = {
        "frame_ids": ["a", "b"],
        "frame_count": 2,
        "frame_list_hash": frame_list_hash(["a", "b"]),
    }
    bound = validate_manifest_binding(manifest, ["a", "b", "c"])
    assert bound["frame_ids"] == ["a", "b"]
    with pytest.raises(ValueError, match="order"):
        validate_manifest_binding(manifest, ["b", "a", "c"])


def test_baseline_metrics_include_ap_drops_retention_and_parameter_normalization() -> None:
    baseline = {
        "ap_0.03": 0.8,
        "ap_0.30": 0.7,
        "ap_0.50": 0.6,
        "ap_0.70": 0.5,
        "mAP": 0.65,
        "candidate_parameter_count": 5_000_000,
    }
    candidate = {
        "ap_0.03": 0.7,
        "ap_0.30": 0.6,
        "ap_0.50": 0.5,
        "ap_0.70": 0.4,
        "mAP": 0.55,
        "original_parameter_count": 5_000_000,
        "candidate_parameter_count": 4_000_000,
    }

    result = apply_baseline_metrics(candidate, baseline)

    assert result["ap_0.03_absolute_drop"] == pytest.approx(0.1)
    assert result["mAP_absolute_drop"] == pytest.approx(0.1)
    assert result["mAP_retention"] == pytest.approx(0.55 / 0.65)
    assert result["delta_map_per_million_pruned_parameters"] == pytest.approx(0.1)


def _analysis_models() -> list[dict[str, object]]:
    baseline = {
        "model_id": "original",
        "stage_scope": "original",
        "strategy": "original_unpruned",
        "strength": "none",
        "ap_0.03": 0.8,
        "ap_0.30": 0.7,
        "ap_0.50": 0.6,
        "ap_0.70": 0.5,
        "mAP": 0.65,
        "mAP_retention": 1.0,
        "mAP_absolute_drop": 0.0,
        "delta_map_per_million_pruned_parameters": None,
        "actual_parameter_reduction": 0,
    }
    rows = [baseline]
    values = {
        ("stage1_only", "taylor_independent_group_ranking"): 0.50,
        ("stage1_only", "torch_pruning_l2_shared_position"): 0.10,
        ("stage2_only", "taylor_independent_group_ranking"): 0.64,
        ("stage2_only", "torch_pruning_l2_shared_position"): 0.60,
    }
    for (scope, strategy), value in values.items():
        rows.append(
            {
                "model_id": f"{scope}__mild__{strategy}",
                "stage_scope": scope,
                "strategy": strategy,
                "strength": "mild",
                "ap_0.03": value,
                "ap_0.30": value,
                "ap_0.50": value,
                "ap_0.70": value,
                "mAP": value,
                "mAP_retention": value / 0.65,
                "mAP_absolute_drop": 0.65 - value,
                "delta_map_per_million_pruned_parameters": (0.65 - value) / 0.2,
                "actual_parameter_reduction": 200_000,
            }
        )
    return rows


def test_strategy_comparison_uses_only_valid_structure_pairs() -> None:
    models = _analysis_models()
    pairs = [
        {"stage_scope": "stage1_only", "strength": "mild", "comparison_valid": True},
        {"stage_scope": "stage2_only", "strength": "mild", "comparison_valid": False},
    ]

    rows = build_strategy_comparisons(models, pairs)

    assert len(rows) == 2
    assert rows[0]["comparison_valid"] is True
    assert rows[0]["mAP_difference_independent_minus_tp"] == pytest.approx(0.4)
    assert rows[0]["strategy_verdict"] == "independent_better"
    assert rows[1]["comparison_valid"] is False
    assert rows[1]["mAP_difference_independent_minus_tp"] is None


def test_sensitivity_rankings_report_absolute_and_parameter_normalized_order() -> None:
    rankings = build_sensitivity_rankings(_analysis_models())
    independent = [
        row
        for row in rankings
        if row["strategy"] == "taylor_independent_group_ranking"
        and row["strength"] == "mild"
    ]

    assert [row["stage"] for row in sorted(independent, key=lambda row: row["absolute_sensitivity_rank"])] == [
        "stage1",
        "stage2",
    ]
    assert [
        row["stage"] for row in sorted(independent, key=lambda row: row["parameter_normalized_sensitivity_rank"])
    ] == ["stage1", "stage2"]


def test_stage1_attribution_can_support_both_intrinsic_sensitivity_and_tp_exacerbation() -> None:
    result = stage1_attribution(_analysis_models(), material_difference=0.005)

    assert result["stage1_intrinsic_sensitivity"] == "supported"
    assert result["shared_position_exacerbation"] == "supported"
    assert result["combined_attribution"] == "supported"
    assert result["all_stage_cumulative_effect"] == "inconclusive"
