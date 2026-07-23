from __future__ import annotations

import pytest

from search.model_families.transformer.dh_power_alignment_4090 import (
    alignment_traits,
    candidate_width_values,
    joint_candidates,
    latency_benefit_gate,
    neighbor_advantage,
    neighbor_controls,
    power_alignment_widths,
    search_candidate_gate,
    speedup_metrics,
)
from search.orchestration.lidar_transformer_dh_power_alignment_4090 import (
    Runtime4090Paths,
    assign_queue_owners,
    calibration_identity,
    formal_branch_guard,
    fresh_build_contract,
    single_family_candidate_manifest,
    validate_4090_runtime,
)


def test_power_widths_for_d32_are_deduplicated_and_complete():
    assert candidate_width_values(power_alignment_widths(32, heads=8)) == (
        32,
        28,
        24,
        20,
        16,
        12,
        8,
        4,
    )


def test_power_widths_for_d64_include_ladder_and_controls():
    assert candidate_width_values(power_alignment_widths(64, heads=4)) == (
        64,
        60,
        56,
        52,
        48,
        44,
        40,
        36,
        32,
        28,
        24,
        20,
        16,
        12,
        8,
        4,
    )


def test_power_widths_for_d16_remain_bounded():
    assert candidate_width_values(power_alignment_widths(16, heads=16)) == (16, 12, 8, 4)
    assert max(candidate_width_values(power_alignment_widths(16, heads=16))) == 16


def test_four_is_power_of_two_but_not_eight_aligned():
    row = alignment_traits(4, heads=8, original_d_h=32)
    assert row["exact_power_of_two"] is True
    assert row["divisible_by_4"] is True
    assert row["divisible_by_8"] is False


def test_24_is_eight_aligned_but_not_power_of_two():
    row = alignment_traits(24, heads=8, original_d_h=32)
    assert row["exact_power_of_two"] is False
    assert row["divisible_by_8"] is True
    assert row["divisible_by_16"] is False


def test_48_is_sixteen_aligned_but_not_power_of_two():
    row = alignment_traits(48, heads=4, original_d_h=64)
    assert row["exact_power_of_two"] is False
    assert row["divisible_by_16"] is True
    assert row["divisible_by_32"] is False


def test_head_and_projection_alignment_are_separate():
    row = alignment_traits(12, heads=8, original_d_h=32)
    assert row["divisible_by_8"] is False
    assert row["projection_width"] == 96
    assert row["projection_divisible_by_32"] is True


def test_reduction_ratio_uses_original_width():
    assert alignment_traits(16, heads=8, original_d_h=32)["reduction_ratio"] == 0.5


def test_invalid_width_fails_closed():
    with pytest.raises(ValueError, match="invalid_power_alignment_width"):
        alignment_traits(36, heads=8, original_d_h=32)


@pytest.mark.parametrize(
    ("target", "original", "expected"),
    [(24, 32, (20, 28)), (16, 32, (12, 20)), (32, 32, (28,)), (4, 16, (8,))],
)
def test_neighbor_controls_match_legal_plus_minus_four(target, original, expected):
    assert neighbor_controls(target, original) == expected


def test_joint_candidates_are_explicit_not_cartesian():
    cobevt = joint_candidates("lidar_cobevt")
    v2xvit = joint_candidates("lidar_v2xvit")
    assert len(cobevt) == 9
    assert len(v2xvit) == 8
    assert cobevt[0].target_d_h_by_family == {
        "cobevt_grid_h8_d32": 32,
        "cobevt_window_h8_d32": 32,
    }
    assert v2xvit[-1].target_d_h_by_family == {
        "v2xvit_agent_relation_h8_d32": 4,
        "v2xvit_spatial_window_w16_h4_d64": 8,
        "v2xvit_spatial_window_w4_h16_d16": 4,
        "v2xvit_spatial_window_w8_h8_d32": 4,
    }


def test_speedup_metrics_keep_structure_precision_and_total_separate():
    row = speedup_metrics(
        baseline_p32_ms=10.0,
        baseline_profile_ms=5.0,
        candidate_profile_ms=4.0,
    )
    assert row == {
        "structure_speedup": 1.25,
        "precision_speedup": 2.0,
        "total_speedup": 2.5,
    }


def test_latency_gate_uses_largest_noise_threshold():
    row = latency_benefit_gate(
        baseline_p50_ms=10.0,
        candidate_p50_ms=9.7,
        baseline_repeat_cv=0.005,
        baseline_replay_drift=0.02,
    )
    assert row["required_reduction_ratio"] == pytest.approx(0.02)
    assert row["observed_reduction_ratio"] == pytest.approx(0.03)
    assert row["beneficial"] is True


def test_neighbor_advantage_requires_both_available_controls():
    assert neighbor_advantage(candidate_ms=4.0, lower_control_ms=4.1, upper_control_ms=4.2)[
        "advantage"
    ] is True
    assert neighbor_advantage(candidate_ms=4.0, lower_control_ms=3.9, upper_control_ms=4.2)[
        "advantage"
    ] is False


def test_search_candidate_gate_requires_every_evidence_gate():
    accepted = search_candidate_gate(
        fixed500_acceptable=True,
        same_profile_latency=True,
        neighbor_advantage_passed=True,
        build_repeat_stable=True,
        joint_supported=True,
    )
    assert accepted["search_space_candidate"] is True
    rejected = search_candidate_gate(
        fixed500_acceptable=True,
        same_profile_latency=True,
        neighbor_advantage_passed=False,
        build_repeat_stable=True,
        joint_supported=True,
    )
    assert rejected["search_space_candidate"] is False
    assert rejected["reasons"] == ["neighbor_control_advantage_missing"]


def _fake_runtime_tree(tmp_path):
    prefix = tmp_path / "anaconda3" / "envs" / "modelopt"
    trt = tmp_path / "TensorRT-10.9_x86_cu118"
    plugin = tmp_path / "libheal_trt_plugins.so"
    for name in ("python", "nvcc", "g++"):
        path = prefix / "bin" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
    trtexec = trt / "targets" / "x86_64-linux-gnu" / "bin" / "trtexec"
    trtexec.parent.mkdir(parents=True, exist_ok=True)
    trtexec.write_text("trtexec")
    (trt / "targets" / "x86_64-linux-gnu" / "lib").mkdir(parents=True)
    plugin.write_text("plugin")
    return Runtime4090Paths(prefix, trt, plugin)


def test_4090_runtime_accepts_conda_nvcc_and_sm89(tmp_path):
    paths = _fake_runtime_tree(tmp_path)
    result = validate_4090_runtime(paths, nvcc_archs=("compute_89", "sm_89"))
    assert result["platform"] == "RTX4090_SM89"
    assert result["nvcc_inside_conda"] is True


def test_4090_runtime_rejects_system_nvcc(tmp_path):
    paths = _fake_runtime_tree(tmp_path)
    paths = Runtime4090Paths(paths.modelopt_prefix, paths.tensorrt_root, paths.plugin_path, tmp_path / "usr/bin/nvcc")
    paths.nvcc_path.parent.mkdir(parents=True, exist_ok=True)
    paths.nvcc_path.write_text("system")
    with pytest.raises(RuntimeError, match="nvcc_outside_modelopt_prefix"):
        validate_4090_runtime(paths, nvcc_archs=("sm_89",))


def test_4090_runtime_rejects_missing_sm89(tmp_path):
    with pytest.raises(RuntimeError, match="sm89_not_supported"):
        validate_4090_runtime(_fake_runtime_tree(tmp_path), nvcc_archs=("sm_80", "sm_90"))


def test_formal_branch_guard_fails_on_remote_head_change():
    with pytest.raises(RuntimeError, match="formal_search_branch_changed"):
        formal_branch_guard("3293f4e", "different")
    assert formal_branch_guard("3293f4e", "3293f4e") is True


def test_calibration_identity_is_bound_to_structure():
    first = calibration_identity("lidar_cobevt", "structure-a", "manifest", "P8")
    second = calibration_identity("lidar_cobevt", "structure-b", "manifest", "P8")
    assert first != second
    assert calibration_identity("lidar_cobevt", "structure-a", "manifest", "P16") == "not_int8"


def test_fresh_build_contract_forbids_timing_cache_reuse():
    contract = fresh_build_contract()
    assert contract["timing_cache_reused"] is False
    assert contract["engine_reused"] is False
    assert contract["onnx_reused_across_structures"] is False


def test_single_family_manifest_has_expected_unique_structure_count():
    rows = single_family_candidate_manifest()
    by_model = {}
    for row in rows:
        by_model.setdefault(row["model"], set()).add(row["structure_signature"])
    assert len(by_model["lidar_cobevt"]) == 15
    assert len(by_model["lidar_v2xvit"]) == 33


def test_queue_assignment_uses_only_requested_gpus_and_is_deterministic():
    rows = [{"candidate_id": f"candidate-{index}"} for index in range(9)]
    assigned = assign_queue_owners(rows, gpu_ids=(4, 5, 6, 7))
    assert [row["physical_gpu"] for row in assigned] == [4, 5, 6, 7, 4, 5, 6, 7, 4]
    assert {row["physical_gpu"] for row in assigned} == {4, 5, 6, 7}
