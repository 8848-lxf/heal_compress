from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_fisher_loss_uses_pruned_importance_ratio_and_keep_mask_semantics() -> None:
    import pytest
    import torch

    from search.candidate import CandidateGenotype
    from search.canonicalization import SearchSpaceSpec, canonicalize_candidate
    from search.proxy.fisher_proxy import FisherStatistics, FisherTaylorProxy
    from search.proxy.parameter_slice_resolver import ParameterSlice

    model = torch.nn.Sequential()
    model.add_module("conv", torch.nn.Conv2d(1, 2, kernel_size=1, bias=False))
    with torch.no_grad():
        model.conv.weight[:] = torch.tensor([[[[1.0]]], [[[10.0]]]])
    stats = FisherStatistics(
        gradients={"conv.weight": torch.ones_like(model.conv.weight)},
        fisher_diag={"conv.weight": torch.zeros_like(model.conv.weight)},
    )
    slices = {
        "u_low": [ParameterSlice("conv.weight", "conv", 0, (0,), "prune_weight_slice")],
        "u_high": [ParameterSlice("conv.weight", "conv", 0, (1,), "prune_weight_slice")],
    }
    space = SearchSpaceSpec(pruning_unit_ids=["u_low", "u_high"], precision_layer_ids=["conv"])
    proxy = FisherTaylorProxy(model, statistics=stats, unit_to_parameter_names=slices, normalize_by_total=True)

    all_keep = canonicalize_candidate(CandidateGenotype({"u_low": 1, "u_high": 1}, {"conv": "FP32"}), space)
    prune_low = canonicalize_candidate(CandidateGenotype({"u_low": 0, "u_high": 1}, {"conv": "FP32"}), space)
    prune_high = canonicalize_candidate(CandidateGenotype({"u_low": 1, "u_high": 0}, {"conv": "FP32"}), space)

    assert proxy.evaluate(all_keep) == pytest.approx(0.0)
    assert proxy.evaluate(prune_low) == pytest.approx(1.0 / 11.0)
    assert proxy.evaluate(prune_low) < proxy.evaluate(prune_high)


def test_proxy_objective_uses_fp32_reference_and_soft_bops_penalty_only() -> None:
    import pytest

    from search.candidate import CandidatePhenotype, PrecisionDecision
    from search.proxy.objective import ProxyObjective, ProxyObjectiveConfig

    class ConstantProxy:
        def __init__(self, value):
            self.value = value

        def evaluate(self, _phenotype):
            return self.value

    class Size:
        def evaluate_breakdown(self, _phenotype):
            return {"R_size_vs_fp32": 0.50, "R_size_vs_fp16_deploy": 1.00}

    class Bops:
        def evaluate_breakdown(self, _phenotype):
            return {"R_bops_vs_fp32": 0.30, "R_bops_vs_fp16_deploy": 1.20, "bops_fp32_baseline": 100.0}

    objective = ProxyObjective(
        fisher=ConstantProxy(0.10),
        sqnr=ConstantProxy(0.20),
        size=Size(),
        bops=Bops(),
        config=ProxyObjectiveConfig(
            alpha_fisher=0.55,
            beta_sqnr=0.25,
            gamma_size=0.05,
            delta_bops=0.15,
            bops_threshold=0.25,
            bops_penalty_formula="squared_relative_excess",
        ),
    )

    metrics = objective.evaluate(CandidatePhenotype(precision_profile={"m": PrecisionDecision("FP16", "FP16")}))

    expected_penalty = max(0.0, 0.30 / 0.25 - 1.0) ** 2
    expected = 0.55 * 0.10 + 0.25 * 0.20 + 0.05 * 0.50 + 0.15 * expected_penalty
    assert metrics["R_size"] == pytest.approx(0.50)
    assert metrics["R_bops"] == pytest.approx(0.30)
    assert metrics["P_bops"] == pytest.approx(expected_penalty)
    assert metrics["F1"] == pytest.approx(expected)


def test_final_runner_uses_direct_unormalized_fisher_and_sqnr_terms(tmp_path) -> None:
    import json

    from search.orchestration.lidar_pyramid_search import LidarPyramidTwoStageSearch

    class MustNotEvaluate:
        def evaluate(self, _phenotype):
            raise AssertionError("identity normalization must not sample proxy candidates")

    runner = LidarPyramidTwoStageSearch(
        config={"proxy": {"term_normalization": "none"}},
        checkpoint=tmp_path / "model.pth",
        output_root=tmp_path,
    )
    stats = runner._build_normalization(object(), MustNotEvaluate(), tmp_path)

    assert stats.medians == {}
    assert stats.normalize("L_fisher", 0.25) == 0.25
    assert stats.normalize("L_sqnr", 0.50) == 0.50
    manifest = json.loads((tmp_path / "archives" / "proxy_normalization.json").read_text(encoding="utf-8"))
    assert manifest == {"medians": {}, "strategy": "none", "version": "none-v1"}


def test_outer_round_bops_target_schedule_is_025_to_018() -> None:
    import pytest

    from search.proxy.objective import bops_target_for_outer_round

    assert bops_target_for_outer_round(0, 4, {"start_target": 0.25, "end_target": 0.18}) == pytest.approx(0.25)
    assert bops_target_for_outer_round(1, 4, {"start_target": 0.25, "end_target": 0.18}) == pytest.approx(0.2266666667)
    assert bops_target_for_outer_round(2, 4, {"start_target": 0.25, "end_target": 0.18}) == pytest.approx(0.2033333333)
    assert bops_target_for_outer_round(3, 4, {"start_target": 0.25, "end_target": 0.18}) == pytest.approx(0.18)
    assert bops_target_for_outer_round(0, 1, {"start_target": 0.25, "end_target": 0.18}) == pytest.approx(0.25)


def test_ga_final_validation_records_reference_and_candidate_resource_phases(tmp_path) -> None:
    import json
    from contextlib import contextmanager
    from types import SimpleNamespace

    from search.candidate import CandidatePhenotype
    from search.orchestration.lidar_pyramid_search import LidarPyramidTwoStageSearch

    source = tmp_path / "source"
    source.mkdir()
    (source / "phenotype.json").write_text(
        json.dumps(CandidatePhenotype().to_dict()),
        encoding="utf-8",
    )
    round_dir = tmp_path / "round_000"
    round_dir.mkdir()
    (round_dir / "round_best_candidate.json").write_text(
        json.dumps({"candidate_hash": "candidate", "artifact_dir": str(source)}),
        encoding="utf-8",
    )

    class FakeEvaluator:
        def _stage2_reference_baseline(self):
            return {"mAP": 1.0}

        def reevaluate_existing_candidate_engine(self, *_args, **kwargs):
            return {"status": "ok", "F2": 0.25, "R_latency_real": 0.5}

    class FakeRecorder:
        def __init__(self):
            self.phases = []

        @contextmanager
        def phase(self, name, gpu_ids, **metadata):
            self.phases.append((name, list(gpu_ids), dict(metadata)))
            yield

    runner = LidarPyramidTwoStageSearch(
        config={},
        checkpoint=tmp_path / "model.pth",
        output_root=tmp_path,
    )
    recorder = FakeRecorder()
    runner._resource_recorder = recorder
    runner._full_validation_evaluator = lambda _context, _run_dir: FakeEvaluator()

    rows = runner._full_validate_ga_round_winners(
        SimpleNamespace(physical_gpu_id=5),
        tmp_path,
    )

    assert rows[0]["status"] == "ok"
    assert [row[0] for row in recorder.phases] == [
        "stage2.ga_final_reference",
        "stage2.ga_final_full_validation",
    ]
    assert all(row[1] == [5] for row in recorder.phases)
    assert recorder.phases[1][2]["candidate_hash"] == "candidate"
    assert (tmp_path / "final_full_validation_results.json").is_file()


def test_ga_generation_winners_are_full_validated_and_selected_per_budget(
    tmp_path, monkeypatch
) -> None:
    import json
    from contextlib import contextmanager
    from types import SimpleNamespace

    import torch

    from search.candidate import CandidatePhenotype
    from search.orchestration.lidar_pyramid_search import LidarPyramidTwoStageSearch

    source = tmp_path / "engine_source"
    source.mkdir()
    phenotype = CandidatePhenotype()
    generation_dir = (
        tmp_path / "round_000" / "generations" / "generation_003"
    )
    generation_dir.mkdir(parents=True)
    (generation_dir / "generation_winner.json").write_text(
        json.dumps(
            {
                "candidate_hash": "generation-winner",
                "artifact_dir": str(source),
                "phenotype": phenotype.to_dict(),
                "F2": 0.4,
                "stage1_metrics": {"F1": 0.1, "BOPS_target": 0.3},
            }
        ),
        encoding="utf-8",
    )

    class FakeEvaluator:
        def _stage2_reference_baseline(self):
            return {"mAP": 1.0}

        def reevaluate_existing_candidate_engine(self, *_args, **kwargs):
            return {
                "candidate_hash": "generation-winner",
                "status": "ok",
                "F2": 0.2,
                "mAP": 0.8,
                "R_latency_real": 0.5,
                "artifact_dir": str(kwargs["output_dir"]),
            }

    class FakeRecorder:
        def __init__(self):
            self.phases = []

        @contextmanager
        def phase(self, name, gpu_ids, **metadata):
            self.phases.append((name, list(gpu_ids), dict(metadata)))
            yield

    evaluator = FakeEvaluator()
    runner = LidarPyramidTwoStageSearch(
        config={},
        checkpoint=tmp_path / "model.pth",
        output_root=tmp_path,
    )
    recorder = FakeRecorder()
    runner._resource_recorder = recorder
    runner._full_validation_evaluator = lambda _context, _run_dir: evaluator
    runner._ga_stage2_evaluator_pool = (
        lambda _context, _evaluator, _run_dir, pool_kind: [(5, evaluator)]
    )
    monkeypatch.setattr(torch.cuda, "set_device", lambda _device: None)

    rows = runner._full_validate_ga_generation_winners(
        SimpleNamespace(physical_gpu_id=5),
        tmp_path,
    )

    assert rows[0]["status"] == "ok"
    winner = json.loads(
        (tmp_path / "round_000" / "round_best_candidate.json").read_text()
    )
    assert winner["candidate_hash"] == "generation-winner"
    assert winner["winning_generations"] == ["generation_003"]
    assert winner["winner_selection_reason"] == (
        "minimum_full_validation_weighted_AP_latency_F2"
    )
    final = json.loads(
        (tmp_path / "final_full_validation_results.json").read_text()
    )
    assert final["successful_budget_winner_count"] == 1
    assert final["candidate_engine_rebuild_count"] == 0
    assert [row[0] for row in recorder.phases] == [
        "stage2.ga_budget_final_reference",
        "stage2.ga_generation_winner_full_validation",
    ]


def test_generation_protocol_resume_skips_completed_stage1_and_screening(
    tmp_path,
) -> None:
    import json
    from types import SimpleNamespace

    from search.orchestration.lidar_pyramid_search import (
        LidarPyramidTwoStageSearch,
    )

    round_dir = tmp_path / "round_000"
    generation_dir = round_dir / "generations/generation_000"
    generation_dir.mkdir(parents=True)
    (round_dir / "generation_stage2_summary.json").write_text(
        json.dumps({"actual_generation_count": 1}),
        encoding="utf-8",
    )
    (round_dir / "round_state.json").write_text(
        json.dumps({"phase": "round_complete"}),
        encoding="utf-8",
    )
    (generation_dir / "generation_winner.json").write_text(
        json.dumps({"candidate_hash": "winner"}),
        encoding="utf-8",
    )
    runner = LidarPyramidTwoStageSearch(
        config={},
        checkpoint=tmp_path / "model.pth",
        output_root=tmp_path,
        resume=tmp_path,
    )
    calls = []
    runner._full_validate_ga_generation_winners = (
        lambda _context, _run_dir: calls.append("finalized")
        or [{"candidate_hash": "winner", "status": "ok"}]
    )

    rows = runner._run_ga(
        SimpleNamespace(),
        None,
        None,
        tmp_path,
        {
            "stage2_selection_scope": "per_generation_topk",
            "bops_targets": [0.3],
        },
        stage1_only=False,
    )

    assert calls == ["finalized"]
    assert rows == [{"candidate_hash": "winner", "status": "ok"}]
    report = json.loads(
        (tmp_path / "generation_winner_finalization_resume.json").read_text()
    )
    assert report["status"] == "complete"
    assert report["skipped_stage1_and_generation_screening"] is True


def test_dense_and_grouped_repair_are_monotonic_mask_preserving() -> None:
    from search.pruning_space.mask_repair import (
        GroupedDomainSpec,
        RepairPolicy,
        dense_floor_repair,
        grouped_equal_count_floor_repair,
    )

    dense = dense_floor_repair(
        {"u0": 1, "u1": 1, "u2": 1, "u3": 1, "u4": 1, "u5": 1},
        ordered_low_to_high=("u0", "u1", "u2", "u3", "u4", "u5"),
        alignment=4,
        minimum_width=1,
    )
    assert sum(dense.repaired_mask.values()) == 4
    assert dense.repaired_mask["u0"] == 0
    assert dense.repaired_mask["u1"] == 0
    assert all(dense.repaired_mask[key] <= {"u0": 1, "u1": 1, "u2": 1, "u3": 1, "u4": 1, "u5": 1}[key] for key in dense.repaired_mask)

    grouped = grouped_equal_count_floor_repair(
        {
            "g0_0": 1,
            "g0_1": 1,
            "g0_2": 1,
            "g0_3": 1,
            "g0_4": 1,
            "g1_0": 1,
            "g1_1": 1,
            "g1_2": 1,
            "g1_3": 1,
            "g1_4": 1,
            "g1_5": 1,
        },
        GroupedDomainSpec(
            groups={
                0: ("g0_0", "g0_1", "g0_2", "g0_3", "g0_4"),
                1: ("g1_0", "g1_1", "g1_2", "g1_3", "g1_4", "g1_5"),
            },
            ordered_low_to_high={
                0: ("g0_0", "g0_1", "g0_2", "g0_3", "g0_4"),
                1: ("g1_0", "g1_1", "g1_2", "g1_3", "g1_4", "g1_5"),
            },
            local_indices={
                0: {"g0_0": 0, "g0_1": 1, "g0_2": 2, "g0_3": 3, "g0_4": 4},
                1: {"g1_0": 0, "g1_1": 1, "g1_2": 2, "g1_3": 3, "g1_4": 4, "g1_5": 5},
            },
            allowed_channels_per_group=(4, 8, 16, 32, 64, 128, 256, 512),
        ),
        RepairPolicy(),
    )

    assert grouped.status == "ok"
    assert grouped.target_width == 4
    assert {group: len(values) for group, values in grouped.group_keep_map.items()} == {0: 4, 1: 4}
    assert grouped.group_keep_map[0] != grouped.group_keep_map[1]
    assert grouped.repaired_mask["g0_0"] == 0
    assert grouped.repaired_mask["g1_0"] == 0


def test_repaired_topk_rescores_and_returns_unique_phenotypes() -> None:
    from search.candidate import CandidateGenotype
    from search.canonicalization import SearchSpaceSpec
    from search.stage1.repair_selection import select_repaired_stage2_topk

    space = SearchSpaceSpec(pruning_unit_ids=["a", "b"], precision_layer_ids=["m"])
    raw = [
        (CandidateGenotype({"a": 1, "b": 1}, {"m": "FP16"}), 10.0, {"F1": 10.0}),
        (CandidateGenotype({"a": 1, "b": 1}, {"m": "FP16"}), 9.0, {"F1": 9.0}),
        (CandidateGenotype({"a": 0, "b": 1}, {"m": "INT8"}), 8.0, {"F1": 8.0}),
    ]

    def repair(genotype):
        return genotype, {"status": "ok"}

    def rescore(phenotype):
        return {"F1": float(len(phenotype.pruned_unit_ids)), "L_fisher": float(len(phenotype.pruned_unit_ids))}

    selected, report = select_repaired_stage2_topk(raw, space=space, repair_fn=repair, rescore_fn=rescore, topk=5)

    assert len(selected) == 2
    assert selected[0].metrics["F1"] == 0.0
    assert selected[1].metrics["F1"] == 1.0
    assert report["duplicate_repaired_phenotype_count"] == 1


def test_stage2_score_uses_tau_ap_and_fp16_latency_reference() -> None:
    import pytest

    from search.stage2.objective import Stage2ObjectiveConfig, compute_stage2_score

    result = compute_stage2_score(
        {"mAP": 0.49, "forward_p50_ms": 6.0},
        baseline={"mAP": 0.50, "forward_p50_ms": 4.0},
        config=Stage2ObjectiveConfig(eta_map=0.80, eta_latency=0.20, latency_metric="forward_p50_ms", tau_ap=0.02),
    )

    assert result["L_map_real"] == pytest.approx(0.5)
    assert result["R_latency_real"] == pytest.approx(1.5)
    assert result["F2"] == pytest.approx(0.8 * 0.5 + 0.2 * 1.5)


def test_deployment_hash_changes_with_qdq_topology_and_merge_contract() -> None:
    from search.hashing import canonical_json_hash, deployment_hash

    common = {
        "physical_hash_value": "physical",
        "realized_precision_profile": {"conv": "INT8"},
        "calibration_scale_hash": "scales",
        "onnx_export_config_hash": "onnx",
        "tensorrt_version": "10.9",
        "gpu_compute_capability": "9.0",
        "builder_flags": {"fp16": True, "int8": True},
        "optimization_profiles": {"fixed_k": 29696},
        "plugin_hashes": {"scatter": "plugin"},
    }
    fp16_merge = canonical_json_hash(
        {"qdq_topology_hash": "topology-a", "merge_policy": "A_fp16_merge"}
    )
    int8_merge = canonical_json_hash(
        {"qdq_topology_hash": "topology-a", "merge_policy": "B_int8_common_scale"}
    )
    different_topology = canonical_json_hash(
        {"qdq_topology_hash": "topology-b", "merge_policy": "A_fp16_merge"}
    )

    hashes = {
        deployment_hash(**common, quantization_contract_hash=fp16_merge),
        deployment_hash(**common, quantization_contract_hash=int8_merge),
        deployment_hash(**common, quantization_contract_hash=different_topology),
    }
    assert len(hashes) == 3


def test_raw_grouped_input_parameter_slices_use_local_group_coordinates() -> None:
    import torch
    from types import SimpleNamespace

    from tracer.types import DependencyMember
    from search.proxy.parameter_slice_resolver import build_unit_parameter_slices

    model = torch.nn.Sequential()
    model.add_module("gconv", torch.nn.Conv2d(16, 16, kernel_size=1, groups=2, bias=False))
    unit = SimpleNamespace(
        stable_id="u_abs_12",
        scope_id="scope",
        root_module_path="gconv",
        root_axis="out",
        root_indices=[12],
        members=[DependencyMember("gconv", "in", [12], "grouped_conv_coupled_input")],
        constraints={"grouped_conv": True, "groups": 2, "channels_per_group": 8},
    )

    slices = build_unit_parameter_slices(model, [unit])

    assert slices["u_abs_12"][0].parameter_name == "gconv.weight"
    assert slices["u_abs_12"][0].axis == 1
    assert slices["u_abs_12"][0].indices == (4,)


def test_random_immigrant_keeps_grouped_domains_repairable_without_repairing_zero_to_one() -> None:
    import random

    from search.canonicalization import SearchSpaceSpec
    from search.ga.immigrants import random_immigrant

    metadata = {}
    unit_ids = []
    for absolute in range(16):
        unit_id = f"g{absolute}"
        unit_ids.append(unit_id)
        metadata[unit_id] = {
            "scope_id": "scope",
            "constraints": {"grouped_conv": True, "groups": 2, "channels_per_group": 8},
            "root_indices": [absolute],
        }
    space = SearchSpaceSpec(
        pruning_unit_ids=unit_ids,
        precision_layer_ids=["m"],
        pruning_unit_metadata=metadata,
    )

    genotype = random_immigrant(space, random.Random(7), keep_probability=0.10)

    counts = [0, 0]
    for unit_id, keep in genotype.pruning_genes.items():
        absolute = int(unit_id[1:])
        counts[absolute // 8] += int(keep)
    assert counts[0] >= 4
    assert counts[1] >= 4


def test_mutation_keeps_grouped_domains_repairable() -> None:
    import random

    from search.candidate import CandidateGenotype
    from search.canonicalization import SearchSpaceSpec
    from search.ga.mutation import mutate_candidate

    metadata = {}
    unit_ids = []
    for absolute in range(16):
        unit_id = f"g{absolute}"
        unit_ids.append(unit_id)
        metadata[unit_id] = {
            "scope_id": "scope",
            "constraints": {"grouped_conv": True, "groups": 2, "channels_per_group": 8},
            "root_indices": [absolute],
        }
    space = SearchSpaceSpec(pruning_unit_ids=unit_ids, precision_layer_ids=["m"], pruning_unit_metadata=metadata)
    candidate = CandidateGenotype({unit_id: 1 for unit_id in unit_ids}, {"m": "FP16"})

    mutated = mutate_candidate(candidate, space, random.Random(3), prune_mutation_rate=1.0, precision_mutation_rate=0.0)

    counts = [0, 0]
    for unit_id, keep in mutated.pruning_genes.items():
        absolute = int(unit_id[1:])
        counts[absolute // 8] += int(keep)
    assert counts[0] >= 4
    assert counts[1] >= 4


def test_stage2_request_uses_repaired_group_maps_for_raw_grouped_units() -> None:
    from types import SimpleNamespace

    from search.adapters.pruning_adapter import FormalPruningAdapter
    from search.candidate import CandidatePhenotype

    unit = SimpleNamespace(
        stable_id="g8",
        protected=False,
        scope_id="scope",
        root_module_path="gconv",
        root_axis="out",
        root_indices=[8],
        source_coupled_unit_ids=["cu8"],
        channel_cost=1,
        parameter_cost=1,
        constraints={"grouped_conv": True, "depthwise": False, "grouped_module_path": "gconv"},
        members=[],
    )
    phenotype = CandidatePhenotype(
        pruned_unit_ids=["g8"],
        metadata={
            "group_keep_map_by_scope": {"scope": {0: [0, 1, 2, 3], 1: [0, 2, 4, 6]}},
            "group_prune_map_by_scope": {"scope": {0: [4, 5, 6, 7], 1: [1, 3, 5, 7]}},
        },
    )

    request = FormalPruningAdapter(
        build_plan_fn=lambda *a, **k: None,
        legalize_plan_fn=lambda *a, **k: None,
        materialize_fn=lambda *a, **k: None,
        snapshot_fn=lambda *a, **k: None,
        hash_fn=lambda *a, **k: None,
        validate_fn=lambda *a, **k: None,
    ).request_from_phenotype(phenotype, [unit])

    entry = next(row for row in request.entries if row.module_path == "gconv")
    assert entry.group_keep_map == {0: [0, 1, 2, 3], 1: [0, 2, 4, 6]}
    assert entry.group_prune_map == {0: [4, 5, 6, 7], 1: [1, 3, 5, 7]}
