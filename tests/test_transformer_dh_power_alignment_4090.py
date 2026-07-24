from __future__ import annotations

import json
from pathlib import Path

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
    modelopt_source_root_for_runtime,
    single_family_candidate_manifest,
    validate_4090_runtime,
)
from search.orchestration.lidar_transformer_dh_power_alignment_matrix_4090 import (
    all_matrix_formal_latency_candidates,
    configured_build_repeat_specs,
    candidate_alias_map,
    execute_fresh_build_plan,
    fresh_build_latency_candidates,
    formal_latency_evidence_ready,
    formal_latency_candidate_batches,
    formal_latency_evidence_index,
    formal_latency_batch_result_reusable,
    fresh_build_repeat_plan,
    priority_rows_through_tier,
    priority_execution_queue,
    priority_aligned_single_family_queue,
    remaining_structure_queue,
    unique_structure_queue,
    worker_queue,
)
from search.reporting.transformer_dh_power_alignment_4090 import (
    accuracy_class,
    apply_search_admission,
    build_repeat_stability,
    compact_evidence_record,
    precision_interaction,
    result_cardinality,
    summarize_build_repeat_evidence,
    write_power_alignment_reports,
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


def test_4090_runtime_resolves_vendored_modelopt_sibling(tmp_path, monkeypatch):
    paths = _fake_runtime_tree(tmp_path)
    source = paths.tensorrt_root.parent / "Model-Optimizer-0.29.0"
    package = source / "modelopt" / "__init__.py"
    package.parent.mkdir(parents=True)
    package.write_text("__version__ = '0.29.0'\n")
    monkeypatch.delenv("MODELOPT_SOURCE_ROOT", raising=False)
    assert modelopt_source_root_for_runtime(paths) == source.resolve()


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


def test_fresh_build_repeat_plan_creates_independent_pairs(tmp_path):
    baseline = tmp_path / "source" / "baseline" / "P16"
    candidate = tmp_path / "source" / "candidate" / "P16"
    rows = fresh_build_repeat_plan(
        baseline_directory=baseline,
        candidate_directory=candidate,
        output_directory=tmp_path / "repeats",
        profile="P16",
        repeats=3,
    )
    assert len(rows) == 6
    assert [row["repeat_index"] for row in rows] == [1, 1, 2, 2, 3, 3]
    assert [row["role"] for row in rows] == ["baseline", "candidate"] * 3
    assert len({row["output_directory"] for row in rows}) == 6
    assert all(row["timing_cache_reused"] is False for row in rows)


def test_fresh_build_repeat_plan_rejects_profile_mismatch(tmp_path):
    with pytest.raises(ValueError, match="fresh_build_source_profile_mismatch"):
        fresh_build_repeat_plan(
            baseline_directory=tmp_path / "baseline" / "P32",
            candidate_directory=tmp_path / "candidate" / "P16",
            output_directory=tmp_path / "repeats",
            profile="P16",
            repeats=3,
        )


def test_execute_fresh_build_plan_writes_new_engines(tmp_path):
    sources = {}
    for role in ("baseline", "candidate"):
        source = tmp_path / "source" / role / "P16"
        (source / "engine_build").mkdir(parents=True)
        typed = source / (
            "strongly_typed.onnx"
            if role == "baseline"
            else "strongly_typed_explicit_qdq.onnx"
        )
        typed.write_bytes(role.encode())
        (source / "engine_build" / "trt_build_request.json").write_text(
            json.dumps(
                {
                    "qdq_onnx": str(typed),
                    "precision_mapping": {"role": role},
                    "build_config": {"strongly_typed": True},
                    "physical_snapshot": {"model_family": "lidar_cobevt"},
                    "tensorrt_root": str(tmp_path / "TensorRT-10.9"),
                }
            ),
            encoding="utf-8",
        )
        sources[role] = source
    plan = fresh_build_repeat_plan(
        baseline_directory=sources["baseline"],
        candidate_directory=sources["candidate"],
        output_directory=tmp_path / "repeats",
        profile="P16",
        repeats=2,
    )
    calls = []

    def fake_builder(**kwargs):
        calls.append(kwargs)
        kwargs["engine_path"].parent.mkdir(parents=True, exist_ok=True)
        kwargs["engine_path"].write_bytes(str(kwargs["engine_path"]).encode())
        kwargs["output_dir"].mkdir(parents=True, exist_ok=True)
        (kwargs["output_dir"] / "engine_layer_info.json").write_text("[]")
        return {"status": "ok"}

    rows = execute_fresh_build_plan(
        plan,
        physical_gpu=4,
        build_engine_fn=fake_builder,
    )
    assert len(rows) == 4
    assert len(calls) == 4
    assert all(call["gpu_id"] == 4 for call in calls)
    assert all(call["conda_env"] == "modelopt" for call in calls)
    assert len({row["engine_sha256"] for row in rows}) == 4
    assert all(Path(row["engine_directory"]).joinpath("engine.plan").is_file() for row in rows)


def test_fresh_build_latency_candidates_bind_one_repeat_pair(tmp_path):
    repeat = tmp_path / "repeat_2"
    for role in ("baseline", "candidate"):
        directory = repeat / role
        directory.mkdir(parents=True)
        (directory / "engine.plan").write_bytes(role.encode())
    rows = fresh_build_latency_candidates(repeat, profile="P8", repeat_index=2)
    assert [row["candidate_id"] for row in rows] == ["baseline", "C1"]
    assert all(row["profile"] == "P8" for row in rows)
    assert all(row["repeat_index"] == 2 for row in rows)


def test_remaining_structure_queue_deduplicates_aliases_and_completed():
    rows = [
        {"candidate_id": "B0", "structure_signature": "base", "model": "m"},
        {"candidate_id": "C0", "structure_signature": "base", "model": "m"},
        {"candidate_id": "C1", "structure_signature": "one", "model": "m"},
        {"candidate_id": "C2", "structure_signature": "two", "model": "m"},
    ]
    queue = remaining_structure_queue(
        rows,
        completed_signatures={"base", "one"},
        gpu_ids=(4, 5),
    )
    assert [row["candidate_id"] for row in queue] == ["C2"]
    assert queue[0]["physical_gpu"] == 4


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


@pytest.mark.parametrize(
    ("delta", "expected"),
    [(0.0, "SAFE"), (-0.003, "SAFE"), (-0.0031, "BORDERLINE"), (-0.01, "BORDERLINE"), (-0.0101, "UNSAFE")],
)
def test_fixed500_accuracy_class_is_fail_closed(delta, expected):
    assert accuracy_class(delta, evaluated=500, skipped=0, finite=True) == expected
    assert accuracy_class(delta, evaluated=499, skipped=0, finite=True) == "INVALID_EVALUATION"


def test_precision_interaction_separates_structure_and_precision_losses():
    result = precision_interaction(
        baseline_p32_map=0.70,
        baseline_profile_map=0.69,
        candidate_p32_map=0.68,
        candidate_profile_map=0.665,
    )
    assert result["delta_structure"] == pytest.approx(-0.025)
    assert result["delta_structure_p32"] == pytest.approx(-0.02)
    assert result["delta_precision_base"] == pytest.approx(-0.01)
    assert result["delta_precision_candidate"] == pytest.approx(-0.015)
    assert result["interaction"] == pytest.approx(-0.005)


def test_build_repeat_stability_requires_two_passes_and_no_slowdown():
    stable = build_repeat_stability((0.02, 0.015, 0.0), required_reduction=0.01)
    assert stable["build_repeat_stable"] is True
    assert stable["passing_builds"] == 2
    unstable = build_repeat_stability((0.02, 0.015, -0.001), required_reduction=0.01)
    assert unstable["build_repeat_stable"] is False


def test_result_cardinality_separates_aliases_structures_and_phenotypes():
    rows = [
        {"structure_signature": "baseline", "profile": "P32", "engine_sha256": "a"},
        {"structure_signature": "baseline", "profile": "P32", "engine_sha256": "a"},
        {"structure_signature": "baseline", "profile": "P16", "engine_sha256": "b"},
        {"structure_signature": "candidate", "profile": "P32", "engine_sha256": "c"},
    ]
    assert result_cardinality(rows) == {
        "result_alias_rows": 4,
        "unique_physical_structures": 2,
        "unique_phenotypes": 3,
    }


def test_build_repeat_evidence_requires_three_unique_fresh_engines():
    rows = [
        {
            "candidate_id": "C1",
            "profile": "P16",
            "baseline_replay": False,
            "formal": True,
            "latency_reduction": reduction,
            "required_reduction": 0.01,
            "engine_sha256": f"engine-{index}",
            "p50_ms": 4.0 - index * 0.01,
        }
        for index, reduction in enumerate((0.02, 0.015, 0.0), start=1)
    ]
    summary = summarize_build_repeat_evidence(
        rows,
        candidate_id="C1",
        profile="P16",
    )
    assert summary["build_repeat_stable"] is True
    assert summary["independent_engine_count"] == 3
    assert summary["engine_sha256s"] == ["engine-1", "engine-2", "engine-3"]

    with pytest.raises(RuntimeError, match="three_unique_fresh_engines_required"):
        summarize_build_repeat_evidence(
            [{**row, "engine_sha256": "same"} for row in rows],
            candidate_id="C1",
            profile="P16",
        )


def test_build_repeat_specs_preserve_historical_agent24_role_alias(tmp_path):
    specs = configured_build_repeat_specs(tmp_path)
    agent_specs = [spec for spec in specs if "v2xvit_agent24" in str(spec[0])]
    assert len(agent_specs) == 3
    assert {spec[2] for spec in agent_specs} == {"P32", "P16", "P8"}
    assert {spec[3] for spec in agent_specs} == {"agent24"}


def test_search_admission_requires_repeat_stability_and_joint_support():
    single = {
        "candidate_id": "single16",
        "model": "lidar_cobevt",
        "family": "window",
        "d_h": 16,
        "profile": "P16",
        "structure_kind": "single_family",
        "accuracy_class": "SAFE",
        "latency_beneficial": True,
        "neighbor_control_advantage": True,
    }
    joint = {
        "candidate_id": "joint",
        "model": "lidar_cobevt",
        "profile": "P16",
        "structure_kind": "joint",
        "target_d_h_by_family": {"window": 16},
        "accuracy_class": "BORDERLINE",
        "latency_beneficial": True,
    }
    repeats = [
        {
            "candidate_id": "single16",
            "profile": "P16",
            "build_repeat_stable": True,
        }
    ]
    rows = apply_search_admission([single, joint], repeats)
    assert rows[0]["joint_supported"] is True
    assert rows[0]["build_repeat_stable"] is True
    assert rows[0]["search_space_candidate"] is True
    assert rows[1]["search_space_candidate"] is False


def test_compact_evidence_record_requires_hashes_and_realized_precision():
    row = {
        "candidate_id": "C1",
        "structure_hash": "structure",
        "structure_signature": "signature",
        "onnx_sha256": "onnx",
        "engine_sha256": "engine",
        "requested_realized_conflict_count": 0,
        "evaluated": 500,
        "skipped": 0,
    }
    compact = compact_evidence_record(row)
    assert compact["candidate_id"] == "C1"
    assert compact["structure_signature"] == "signature"
    with pytest.raises(RuntimeError, match="compact_evidence_missing"):
        compact_evidence_record({**row, "engine_sha256": ""})


def test_report_writer_emits_compact_contract_and_root_conclusion(tmp_path):
    row = {
        "model": "lidar_cobevt",
        "family": "baseline",
        "candidate_id": "B0",
        "structure_kind": "baseline",
        "structure_hash": "structure",
        "structure_signature": "signature",
        "engine_sha256": "engine",
        "profile": "P32",
        "evaluated": 500,
        "skipped": 0,
        "mAP": 0.7,
        "p50_ms": 5.0,
        "accuracy_class": "SAFE",
        "latency_class": "NO_BENEFIT",
        "search_space_candidate": False,
    }
    result = write_power_alignment_reports(
        tmp_path,
        candidate_rows=[{"candidate_id": "B0", "model": "lidar_cobevt"}],
        single_family_rows=[row],
        joint_rows=[],
    )
    assert result["summary"]["single_family_fixed500_rows"] == 1
    assert result["summary"]["unique_physical_structures"] == 1
    assert result["summary"]["unique_phenotypes"] == 1
    for name in (
        "power_alignment_candidate_manifest.csv",
        "power_alignment_single_family_fixed500.csv",
        "power_alignment_contract.json",
        "root_conclusion_power_alignment.md",
        "reports/report_summary.json",
    ):
        assert (tmp_path / name).is_file(), name
    conclusion = (tmp_path / "root_conclusion_power_alignment.md").read_text()
    assert "## Allowed search candidates" in conclusion
    assert "No candidate passed all five admission gates." in conclusion


def test_single_and_joint_manifests_deduplicate_to_63_physical_structures():
    rows = [*single_family_candidate_manifest()]
    from search.orchestration.lidar_transformer_dh_power_alignment_4090 import joint_candidate_manifest

    rows.extend(joint_candidate_manifest())
    unique = unique_structure_queue(rows)
    assert len(unique) == 63
    assert len({row["structure_signature"] for row in unique}) == 63


def test_baseline_aliases_share_one_physical_structure():
    rows = [*single_family_candidate_manifest()]
    from search.orchestration.lidar_transformer_dh_power_alignment_4090 import joint_candidate_manifest

    rows.extend(joint_candidate_manifest())
    aliases = candidate_alias_map(rows)
    assert sorted(aliases["lidar_cobevt__B0"]) == ["C0", "lidar_cobevt__B0"]
    assert sorted(aliases["lidar_v2xvit__B0"]) == ["V0", "lidar_v2xvit__B0"]


def test_priority_queue_contains_baselines_and_all_multiple_of_eight_single_family_widths():
    rows = priority_aligned_single_family_queue(single_family_candidate_manifest())
    assert len(rows) == 22
    assert {row["candidate_id"] for row in rows[:2]} == {
        "lidar_cobevt__B0",
        "lidar_v2xvit__B0",
    }
    assert all(
        row["structure_kind"] == "baseline" or int(row["d_h"]) % 8 == 0
        for row in rows
    )
    assert not any(row.get("d_h") in {12, 20, 28} for row in rows)


def test_priority_queue_runs_exact_8_16_32_before_other_aligned_widths():
    rows = priority_aligned_single_family_queue(single_family_candidate_manifest())
    tiers = [int(row["priority_tier"]) for row in rows]
    assert tiers == sorted(tiers)
    assert {int(row["d_h"]) for row in rows if row["priority_tier"] == 1} == {8, 16, 32}
    assert {int(row["d_h"]) for row in rows if row["priority_tier"] == 2} == {
        24,
        40,
        48,
        56,
    }


def test_priority_execution_queue_is_deterministically_partitioned_across_four_gpus():
    rows = priority_execution_queue(single_family_candidate_manifest(), gpu_ids=(4, 5, 6, 7))
    assert len(rows) == 22
    assert [row["physical_gpu"] for row in rows[:8]] == [4, 5, 6, 7, 4, 5, 6, 7]
    assert [len(worker_queue(rows, gpu)) for gpu in (4, 5, 6, 7)] == [6, 6, 5, 5]


def test_worker_queue_rejects_gpu_not_present_in_manifest():
    rows = priority_execution_queue(single_family_candidate_manifest(), gpu_ids=(4, 5))
    with pytest.raises(ValueError, match="gpu_not_in_priority_queue"):
        worker_queue(rows, 7)


def test_formal_latency_requires_exact_build_and_complete_fixed500():
    build = {"status": "ok", "requested_realized_conflict_count": 0}
    fixed = {"status": "ok", "evaluated": 500, "skipped": 0}
    assert formal_latency_evidence_ready(build, fixed) is True
    assert formal_latency_evidence_ready({**build, "requested_realized_conflict_count": 1}, fixed) is False
    assert formal_latency_evidence_ready(build, {**fixed, "evaluated": 499}) is False
    assert formal_latency_evidence_ready(build, {**fixed, "skipped": 1}) is False


def test_formal_latency_profile_must_match_build_and_fixed500():
    build = {
        "status": "ok",
        "profile": "P8",
        "requested_realized_conflict_count": 0,
    }
    fixed = {
        "status": "ok",
        "profile": "P8",
        "evaluated": 500,
        "skipped": 0,
    }
    assert formal_latency_evidence_ready(build, fixed, profile="P8") is True
    assert formal_latency_evidence_ready({**build, "profile": "P16"}, fixed, profile="P8") is False
    assert formal_latency_evidence_ready(build, {**fixed, "profile": "P16"}, profile="P8") is False


def test_all_matrix_formal_latency_candidates_group_every_unique_structure(tmp_path):
    rows = [
        {
            "candidate_id": "model__B0",
            "model": "model",
            "structure_signature": "baseline",
            "target_d_h_by_family": {"family": 32},
        },
        {
            "candidate_id": "model__family__dh_016",
            "model": "model",
            "structure_signature": "candidate",
            "target_d_h_by_family": {"family": 16},
        },
        {
            "candidate_id": "alias",
            "model": "model",
            "structure_signature": "candidate",
            "target_d_h_by_family": {"family": 16},
        },
    ]

    def evidence_directory(row, profile):
        directory = tmp_path / str(row["candidate_id"]) / profile
        directory.mkdir(parents=True)
        (directory / "engine.plan").write_bytes(b"engine")
        return directory

    def evidence_loader(directory, profile):
        return (
            {
                "status": "ok",
                "profile": profile,
                "requested_realized_conflict_count": 0,
                "engine_sha256": "engine-hash",
            },
            {
                "status": "ok",
                "profile": profile,
                "evaluated": 500,
                "skipped": 0,
                "mAP": 0.6,
                "structure_hash": "structure-hash",
            },
        )

    grouped = all_matrix_formal_latency_candidates(
        rows,
        profiles=("P32", "P16", "P8"),
        evidence_directory=evidence_directory,
        evidence_loader=evidence_loader,
    )
    assert set(grouped) == {("model", "P32"), ("model", "P16"), ("model", "P8")}
    assert all(len(group) == 2 for group in grouped.values())
    assert all(group[0]["candidate_id"] == "model__B0" for group in grouped.values())


def test_all_matrix_formal_latency_candidates_fail_closed_on_incomplete_evidence(tmp_path):
    rows = [
        {
            "candidate_id": "model__B0",
            "model": "model",
            "structure_signature": "baseline",
            "target_d_h_by_family": {"family": 32},
        }
    ]

    with pytest.raises(RuntimeError, match="all_matrix_formal_latency_evidence_incomplete"):
        all_matrix_formal_latency_candidates(
            rows,
            profiles=("P16",),
            evidence_directory=lambda row, profile: tmp_path,
            evidence_loader=lambda directory, profile: ({}, {}),
        )


def test_all_matrix_formal_latency_candidates_can_screen_completed_prefix(tmp_path):
    rows = [
        {
            "candidate_id": "model__B0",
            "model": "model",
            "structure_signature": "baseline",
            "target_d_h_by_family": {"family": 32},
        },
        {
            "candidate_id": "missing",
            "model": "model",
            "structure_signature": "missing",
            "target_d_h_by_family": {"family": 16},
        },
    ]
    baseline = tmp_path / "model__B0" / "P16"
    baseline.mkdir(parents=True)
    (baseline / "engine.plan").write_bytes(b"engine")

    def load(directory, profile):
        if directory == baseline:
            return (
                {
                    "status": "ok",
                    "profile": profile,
                    "requested_realized_conflict_count": 0,
                    "engine_sha256": "engine",
                },
                {
                    "status": "ok",
                    "profile": profile,
                    "evaluated": 500,
                    "skipped": 0,
                    "mAP": 0.6,
                    "structure_hash": "baseline",
                },
            )
        return {}, {}

    grouped = all_matrix_formal_latency_candidates(
        rows,
        profiles=("P16",),
        evidence_directory=lambda row, profile: tmp_path / row["candidate_id"] / profile,
        evidence_loader=load,
        fail_on_missing=False,
    )
    assert [row["candidate_id"] for row in grouped[("model", "P16")]] == ["model__B0"]


def test_formal_latency_batches_repeat_baseline_and_cover_candidates_once():
    rows = [{"candidate_id": "model__B0"}] + [
        {"candidate_id": f"candidate-{index}"} for index in range(17)
    ]
    batches = formal_latency_candidate_batches(
        rows, baseline_id="model__B0", maximum_candidates=8
    )
    assert len(batches) == 3
    assert all(batch[0]["candidate_id"] == "model__B0" for batch in batches)
    candidates = [row["candidate_id"] for batch in batches for row in batch[1:]]
    assert candidates == [f"candidate-{index}" for index in range(17)]


def test_formal_latency_evidence_index_aggregates_baseline_replays():
    rows = [
        {
            "candidate_id": "model__B0",
            "model": "model",
            "profile": "P16",
            "structure_signature": "baseline",
            "baseline_replay": True,
            "p50_ms": 10.0,
            "p90_ms": 11.0,
            "p95_ms": 12.0,
            "p99_ms": 13.0,
        },
        {
            "candidate_id": "candidate",
            "model": "model",
            "profile": "P16",
            "structure_signature": "candidate",
            "baseline_replay": False,
            "p50_ms": 8.0,
            "p90_ms": 9.0,
            "p95_ms": 10.0,
            "p99_ms": 11.0,
        },
        {
            "candidate_id": "model__B0",
            "model": "model",
            "profile": "P16",
            "structure_signature": "baseline",
            "baseline_replay": True,
            "p50_ms": 11.0,
            "p90_ms": 12.0,
            "p95_ms": 13.0,
            "p99_ms": 14.0,
        },
    ]
    index = formal_latency_evidence_index(rows)
    assert index[("baseline", "P16")]["p50_ms"] == 10.5
    assert index[("baseline", "P16")]["baseline_measurement_count"] == 2
    assert index[("candidate", "P16")]["p50_ms"] == 8.0


def test_formal_latency_batch_reuse_requires_identity_profile_and_engine_hash():
    expected = [
        {"candidate_id": "model__B0", "profile": "P16", "engine_sha256": "base"},
        {"candidate_id": "candidate", "profile": "P16", "engine_sha256": "candidate"},
    ]
    actual = [
        {
            "candidate_id": "model__B0",
            "profile": "P16",
            "engine_sha256": "base",
            "baseline_replay": True,
            "formal": True,
        },
        {
            "candidate_id": "candidate",
            "profile": "P16",
            "engine_sha256": "candidate",
            "baseline_replay": False,
            "formal": True,
        },
        {
            "candidate_id": "model__B0",
            "profile": "P16",
            "engine_sha256": "base",
            "baseline_replay": True,
            "formal": True,
        },
    ]
    assert formal_latency_batch_result_reusable(actual, expected) is True
    assert formal_latency_batch_result_reusable(
        [{**row, "engine_sha256": "stale"} if row["candidate_id"] == "candidate" else row for row in actual],
        expected,
    ) is False


def test_priority_tier_one_contains_only_baseline_and_exact_8_16_32_rows():
    queue = priority_execution_queue(single_family_candidate_manifest())
    rows = priority_rows_through_tier(queue, 1)
    assert len(rows) == 14
    assert all(int(row["priority_tier"]) <= 1 for row in rows)
    assert {int(row["d_h"]) for row in rows if row["d_h"] is not None} == {8, 16, 32}
