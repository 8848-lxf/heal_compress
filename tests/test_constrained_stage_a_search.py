from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _policy():
    from search.constrained.policy import ConstrainedStageAPolicy

    return ConstrainedStageAPolicy(
        r_mac_floor=0.95,
        int8_mac_share_min=0.14,
        int8_mac_share_max=0.22,
        bops_target=0.21,
        bops_tolerance=0.005,
        min_map=0.705088,
        min_ap07=0.564003,
    )


def test_constrained_resource_gate_enforces_all_three_closed_intervals() -> None:
    from search.constrained.policy import constrained_resource_admission

    policy = _policy()
    accepted = constrained_resource_admission(
        {"R_MAC": 0.95, "int8_macs_share_full": 0.14, "R_bops_vs_fp32": 0.205},
        policy,
    )
    rejected = constrained_resource_admission(
        {"R_MAC": 0.9499, "int8_macs_share_full": 0.2201, "R_bops_vs_fp32": 0.2151},
        policy,
    )

    assert accepted["passed"] is True
    assert accepted["failure_reasons"] == []
    assert rejected["passed"] is False
    assert rejected["failure_reasons"] == [
        "R_MAC_below_floor",
        "INT8_MAC_share_out_of_range",
        "legalized_BOPS_out_of_budget",
    ]


def test_full_canonical_int8_share_is_mac_weighted_not_layer_count() -> None:
    from search.constrained.policy import canonical_int8_mac_share

    share = canonical_int8_mac_share(
        {"large": "FP16", "small_a": "INT8", "small_b": "INT8"},
        {"large": 80.0, "small_a": 15.0, "small_b": 5.0},
    )

    assert share == pytest.approx(0.20)
    assert share != pytest.approx(2.0 / 3.0)


def test_fp32_and_non_allowlisted_int8_genes_fail_closed() -> None:
    from search.constrained.policy import validate_precision_genes

    policy = _policy()
    assert validate_precision_genes(
        {"pg_c": "INT8", "pg_late": "FP16"},
        int8_allowlist={"pg_c"},
        policy=policy,
    )["passed"] is True

    fp32 = validate_precision_genes(
        {"pg_c": "FP32", "pg_late": "FP16"},
        int8_allowlist={"pg_c"},
        policy=policy,
    )
    illegal_int8 = validate_precision_genes(
        {"pg_c": "INT8", "pg_late": "INT8"},
        int8_allowlist={"pg_c"},
        policy=policy,
    )

    assert fp32["failure_reasons"] == ["FP32_precision_gene_forbidden:pg_c"]
    assert illegal_int8["failure_reasons"] == ["INT8_group_not_allowlisted:pg_late"]


def test_pruning_repair_must_preserve_precision_gene_hash() -> None:
    from search.candidate import CandidateGenotype
    from search.constrained.policy import precision_repair_identity

    raw = CandidateGenotype({"u0": 0, "u1": 1}, {"pg_c": "INT8", "pg_late": "FP16"})
    repaired = CandidateGenotype({"u0": 1, "u1": 1}, dict(raw.precision_genes))
    changed = CandidateGenotype({"u0": 1, "u1": 1}, {"pg_c": "FP16", "pg_late": "FP16"})

    accepted = precision_repair_identity(raw, repaired)
    rejected = precision_repair_identity(raw, changed)

    assert accepted["passed"] is True
    assert accepted["raw_precision_gene_hash"] == accepted["repaired_precision_gene_hash"]
    assert rejected["passed"] is False
    assert rejected["failure_reason"] == "pruning_repair_modified_precision_genes"


def test_stage_a_unlock_requires_five_unique_fully_admitted_candidates() -> None:
    from search.constrained.policy import constrained_smoke_unlock

    rows = [
        {
            "candidate_hash": f"g{index}",
            "physical_hash": f"p{index}",
            "deployment_hash": f"d{index}",
            "status": "ok",
            "evaluated": 200,
            "skipped": 0,
            "resource_admission_passed": True,
            "accuracy_admission_passed": True,
            "precision_identity_passed": True,
            "deployment_audits_passed": True,
        }
        for index in range(5)
    ]

    accepted = constrained_smoke_unlock(rows, workers_stopped=True, residual_gpu_processes=[])
    duplicate = [dict(row) for row in rows]
    duplicate[-1]["physical_hash"] = duplicate[0]["physical_hash"]
    rejected = constrained_smoke_unlock(duplicate, workers_stopped=True, residual_gpu_processes=[])

    assert accepted["MULTIGPU_TOP5_SMOKE_PASS"] is True
    assert accepted["STAGE_A_ALLOWED"] is True
    assert accepted["STAGE_A_STARTED"] is False
    assert accepted["STAGE_B_ALLOWED"] is False
    assert rejected["MULTIGPU_TOP5_SMOKE_PASS"] is False
    assert rejected["STAGE_A_ALLOWED"] is False
    assert "physical_hash_not_unique" in rejected["failure_reasons"]


def test_realized_resource_metrics_use_full_canonical_mac_denominator() -> None:
    from types import SimpleNamespace

    from search.stage2.realized_bops import compute_realized_bops

    physical_shapes = [
        SimpleNamespace(
            module_path="large",
            call_index=0,
            module_type="Conv2d",
            c_in=8,
            c_out=8,
            groups=1,
            macs=60.0,
        ),
        SimpleNamespace(
            module_path="small",
            call_index=0,
            module_type="Conv2d",
            c_in=4,
            c_out=4,
            groups=1,
            macs=20.0,
        ),
    ]
    baseline_shapes = [
        SimpleNamespace(module_path="large", call_index=0, macs=80.0),
        SimpleNamespace(module_path="small", call_index=0, macs=20.0),
    ]
    physical_snapshot = {
        "snapshot_schema_version": "physical-structure-snapshot-v2",
        "parameter_count": 80,
        "modules": [
            {"module_path": "large", "in_channels": 8, "out_channels": 8, "groups": 1, "parameter_count": 60},
            {"module_path": "small", "in_channels": 4, "out_channels": 4, "groups": 1, "parameter_count": 20},
        ],
    }
    baseline_snapshot = {
        "snapshot_schema_version": "physical-structure-snapshot-v2",
        "parameter_count": 100,
        "modules": [
            {"module_path": "large", "in_channels": 10, "out_channels": 10, "groups": 1, "parameter_count": 80},
            {"module_path": "small", "in_channels": 4, "out_channels": 4, "groups": 1, "parameter_count": 20},
        ],
    }

    report = compute_realized_bops(
        physical_runtime_shapes=physical_shapes,
        baseline_runtime_shapes=baseline_shapes,
        realized_precision_profile={"large": "FP16", "small": "INT8"},
        physical_snapshot=physical_snapshot,
        baseline_snapshot=baseline_snapshot,
        target_retention=None,
        tolerance=0.005,
    )

    assert report["R_MAC"] == pytest.approx(0.80)
    assert report["int8_macs_share_full"] == pytest.approx(0.20)
    assert report["int8_macs_share_physical"] == pytest.approx(0.25)


def test_smoke10_does_not_apply_formal_ap_thresholds() -> None:
    from search.constrained.policy import smoke10_admission

    report = smoke10_admission(
        {
            "status": "ok",
            "num_evaluated_frames": 10,
            "num_skipped_frames": 0,
            "mAP": 0.001,
            "AP@0.7": 0.0,
            "forward_p50_ms": 4.0,
        }
    )

    assert report["passed"] is True
    assert report["formal_ap_gate_applied"] is False
    collapsed = smoke10_admission(
        {
            "status": "ok",
            "num_evaluated_frames": 10,
            "num_skipped_frames": 0,
            "mAP": 0.0,
            "AP@0.7": 0.0,
            "forward_p50_ms": 4.0,
        }
    )
    assert collapsed["failure_reasons"] == [
        "smoke_detection_complete_collapse"
    ]


def test_200_frame_gate_requires_absolute_map_and_ap07() -> None:
    from search.stage2.objective import Stage2ObjectiveConfig, compute_stage2_score

    config = Stage2ObjectiveConfig(
        min_map=0.705088,
        min_ap07=0.564003,
        required_evaluated_frames=200,
        required_skipped_frames=0,
    )
    baseline = {"mAP": 0.725, "forward_p50_ms": 5.7}
    accepted = compute_stage2_score(
        {
            "status": "ok",
            "mAP": 0.706,
            "AP@0.7": 0.565,
            "forward_p50_ms": 4.0,
            "num_evaluated_frames": 200,
            "num_skipped_frames": 0,
        },
        baseline=baseline,
        config=config,
    )
    rejected = compute_stage2_score(
        {
            "status": "ok",
            "mAP": 0.704,
            "AP@0.7": 0.563,
            "forward_p50_ms": 3.0,
            "num_evaluated_frames": 200,
            "num_skipped_frames": 0,
        },
        baseline=baseline,
        config=config,
    )

    assert accepted["status"] == "ok"
    assert accepted["accuracy_admission_passed"] is True
    assert rejected["status"] == "accuracy_hard_gate_failed"
    assert rejected["accuracy_admission_passed"] is False
    assert rejected["failure_reasons"] == ["mAP_below_minimum", "AP07_below_minimum"]


def test_repaired_resource_infeasible_candidate_never_enters_stage2() -> None:
    from search.candidate import CandidateGenotype
    from search.canonicalization import SearchSpaceSpec
    from search.stage1.repair_selection import select_repaired_stage2_topk

    space = SearchSpaceSpec(pruning_unit_ids=["u"], precision_layer_ids=["m"])
    raw = [
        (CandidateGenotype({"u": 1}, {"m": "INT8"}), 0.1, {"F1": 0.1}),
        (CandidateGenotype({"u": 0}, {"m": "INT8"}), 0.2, {"F1": 0.2}),
    ]

    def rescore(phenotype):
        eligible = not phenotype.pruned_unit_ids
        return {
            "F1": 0.1 if eligible else 0.2,
            "bops_feasible": True,
            "hard_constraints_feasible": eligible,
            "hard_constraint_failure_reasons": [] if eligible else ["R_MAC_below_floor"],
        }

    selected, report = select_repaired_stage2_topk(
        raw,
        space=space,
        repair_fn=lambda genotype: (genotype, {"status": "ok"}),
        rescore_fn=rescore,
        topk=5,
        hard_gate_fields=("bops_feasible", "hard_constraints_feasible"),
    )

    assert len(selected) == 1
    assert report["hard_gate_ineligible_count"] == 1
    assert report["hard_gate_failure_reasons"] == {"R_MAC_below_floor": 1}


def _population_space():
    from search.canonicalization import SearchSpaceSpec
    from search.quantization_space.types import QuantizationSearchGroup

    macs = {
        "pg_c": 197.0,
        "pg_late_1": 1.0,
        "pg_late_2": 2.0,
        "pg_late_3": 3.0,
        "pg_late_4": 4.0,
        "pg_late_5": 5.0,
        "pg_fixed": 788.0,
    }
    groups = tuple(
        QuantizationSearchGroup(
            group_id=group_id,
            module_paths=(group_id,),
            canonical_node_ids=(),
            allowed_precisions=("FP16", "INT8") if group_id != "pg_fixed" else ("FP16",),
            protected=group_id == "pg_fixed",
            protection_reason="fixed" if group_id == "pg_fixed" else "",
            ordering=index,
            parameter_count=1,
            baseline_macs=macs[group_id],
        )
        for index, group_id in enumerate(macs)
    )
    return SearchSpaceSpec(
        pruning_unit_ids=[f"u{index:02d}" for index in range(12)],
        precision_layer_ids=list(macs),
        quantization_groups=groups,
        default_precision="FP16",
    ), macs


def test_seed_family_allocation_matches_35_35_20_10() -> None:
    from search.constrained.population import allocate_seed_family_counts

    counts = allocate_seed_family_counts(1024)

    assert sum(counts.values()) == 1024
    assert counts["c_neighborhood"] / 1024 == pytest.approx(0.35, abs=0.001)
    assert counts["hybrid"] / 1024 == pytest.approx(0.35, abs=0.001)
    assert counts["b_derived"] / 1024 == pytest.approx(0.20, abs=0.001)
    assert counts["constrained_fresh"] / 1024 == pytest.approx(0.10, abs=0.001)


def test_anchor_b_order_is_restored_until_rmac_floor() -> None:
    from search.constrained.population import restore_anchor_b_mask

    order = [f"u{index}" for index in range(12)]
    restored = restore_anchor_b_mask(
        order,
        r_mac_by_pruned_count={count: 1.0 - 0.01 * count for count in range(13)},
        r_mac_floor=0.95,
        alignment=2,
    )

    assert restored["pruned_unit_ids"] == order[:4]
    assert restored["restored_unit_count"] == 8
    assert restored["R_MAC"] == pytest.approx(0.96)


def test_constrained_population_has_exact_family_counts_and_one_anchor_c() -> None:
    from search.constrained.population import ConstrainedSeedFactory

    space, macs = _population_space()
    policy = _policy()
    allowlist = ["pg_c", "pg_late_1", "pg_late_2", "pg_late_3", "pg_late_4", "pg_late_5"]

    def metrics_fn(candidates):
        rows = []
        total = sum(macs.values())
        for candidate in candidates:
            pruned = sum(int(value) == 0 for value in candidate.pruning_genes.values())
            r_mac = 1.0 - 0.005 * pruned
            int8_share = sum(
                macs[group_id]
                for group_id, precision in candidate.precision_genes.items()
                if precision == "INT8"
            ) / total
            rows.append(
                {
                    "R_MAC": r_mac,
                    "int8_macs_share_full": int8_share,
                    "R_bops_vs_fp32": 0.25 * r_mac - 0.1875 * int8_share,
                }
            )
        return rows

    factory = ConstrainedSeedFactory(
        space=space,
        policy=policy,
        int8_allowlist=allowlist,
        anchor_c_group_id="pg_c",
        local_fisher_order=space.pruning_unit_ids,
        anchor_b_fisher_order=space.pruning_unit_ids,
        repair_fn=lambda candidate: (candidate, {"status": "ok"}),
        metrics_fn=metrics_fn,
        alignment=1,
        random_seed=42,
    )
    population, report = factory.build(20)

    assert len(population) == 20
    assert report["family_accepted_counts"] == {
        "b_derived": 4,
        "c_neighborhood": 7,
        "constrained_fresh": 2,
        "hybrid": 7,
    }
    assert report["exact_anchor_c_count"] == 1
    assert report["generated_unique_genotype_count"] >= 20
    assert report["repair_legal_count"] >= 20
    assert report["R_MAC_eligible_count"] >= 20
    assert report["INT8_MAC_share_eligible_count"] >= 20
    assert report["legalized_BOPS_eligible_count"] >= 20
    assert len({candidate.meta["seed_family"] for candidate in population}) == 4
    assert all("FP32" not in candidate.precision_genes.values() for candidate in population)
    assert len({str(candidate.to_dict()) for candidate in population}) == 20


def test_constrained_population_fails_closed_when_discrete_space_is_too_small() -> None:
    from search.constrained.population import ConstrainedPopulationSupplyError, ConstrainedSeedFactory

    space, _macs = _population_space()
    factory = ConstrainedSeedFactory(
        space=space,
        policy=_policy(),
        int8_allowlist=["pg_c"],
        anchor_c_group_id="pg_c",
        local_fisher_order=[],
        anchor_b_fisher_order=[],
        repair_fn=lambda candidate: (candidate, {"status": "ok"}),
        metrics_fn=lambda candidates: [
            {"R_MAC": 1.0, "int8_macs_share_full": 0.197, "R_bops_vs_fp32": 0.213}
            for _candidate in candidates
        ],
        alignment=1,
        random_seed=42,
        proposal_multiplier=2,
    )

    with pytest.raises(ConstrainedPopulationSupplyError) as error:
        factory.build(20)

    assert error.value.report["status"] == "constrained_population_supply_exhausted"
    assert error.value.report["requested_seed_count"] == 20


def test_hybrid_pruning_variants_stay_aligned_in_local_low_sensitivity_band() -> None:
    from search.constrained.population import ConstrainedSeedFactory

    space, _macs = _population_space()
    factory = ConstrainedSeedFactory(
        space=space,
        policy=_policy(),
        int8_allowlist=["pg_c"],
        anchor_c_group_id="pg_c",
        local_fisher_order=space.pruning_unit_ids,
        anchor_b_fisher_order=space.pruning_unit_ids,
        repair_fn=lambda candidate: (candidate, {"status": "ok"}),
        metrics_fn=lambda _candidates: [],
        alignment=2,
    )

    exact = factory._pruned_prefix("hybrid", 1)
    variant = factory._pruned_prefix("hybrid", 7)

    assert len(exact) == len(variant) == 4
    assert exact != variant
    assert set(variant) <= set(space.pruning_unit_ids[:20])
    assert factory._pruned_prefix("b_derived", 1) == space.pruning_unit_ids[:10]


def test_precision_allowlist_keeps_pg0141_and_rejects_sensitive_domains() -> None:
    from search.constrained.context import PrecisionSensitivityRecord, select_int8_allowlist

    rows = [
        PrecisionSensitivityRecord("pg_0141", ("pyramid_backbone.deblocks.2.0",), 200.0, 0.01, 0.02, 1.0, True, ""),
        PrecisionSensitivityRecord("pg_late", ("pyramid_backbone.resnet.layer2.3.conv1",), 20.0, 0.02, 0.02, 1.0, True, ""),
        PrecisionSensitivityRecord("pg_early", ("backbone_m1.resnet.layer0.0.conv1",), 300.0, 0.001, 0.001, 1.0, True, ""),
        PrecisionSensitivityRecord("pg_shrink", ("shrink_conv.layers.0.double_conv.2",), 400.0, 0.001, 0.001, 1.0, True, ""),
        PrecisionSensitivityRecord("pg_head", ("reg_head",), 5.0, 0.001, 0.001, 1.0, True, ""),
        PrecisionSensitivityRecord("pg_illegal", ("pyramid_backbone.resnet.layer2.4.conv1",), 20.0, 0.01, 0.01, 1.0, False, "INT8_not_legal"),
    ]

    report = select_int8_allowlist(
        rows,
        priority_group_id="pg_0141",
        allowed_module_prefixes=("pyramid_backbone.deblocks.2.0", "pyramid_backbone.resnet.layer2."),
        max_groups=8,
    )

    assert report["selected_group_ids"] == ["pg_0141", "pg_late"]
    by_id = {row["group_id"]: row for row in report["groups"]}
    assert by_id["pg_0141"]["legal_INT8_status"] is True
    assert by_id["pg_early"]["rejection_reason"] == "module_not_in_late_low_sensitivity_allowlist"
    assert by_id["pg_shrink"]["rejection_reason"] == "module_not_in_late_low_sensitivity_allowlist"
    assert by_id["pg_head"]["rejection_reason"] == "module_not_in_late_low_sensitivity_allowlist"
    assert by_id["pg_illegal"]["rejection_reason"] == "INT8_not_legal"
    assert all(
        {"canonical_MAC", "MAC_share", "taylor_fisher_perturbation", "SQNR_loss", "sensitivity_prior", "ranking_score", "legal_INT8_status", "rejection_reason"}
        <= set(row)
        for row in report["groups"]
    )


def test_precision_sensitivity_prefers_runtime_canonical_macs_over_weight_size() -> None:
    from types import SimpleNamespace

    import torch.nn as nn

    from search.canonicalization import SearchSpaceSpec
    from search.constrained.context import measure_precision_sensitivity
    from search.quantization_space.types import QuantizationSearchGroup

    model = nn.Sequential()
    model.add_module("late", nn.Conv2d(1, 1, 3, bias=False))
    group = QuantizationSearchGroup(
        group_id="pg_late",
        module_paths=("late",),
        canonical_node_ids=("late_conv",),
        allowed_precisions=("FP16", "INT8"),
        protected=False,
        protection_reason="",
        ordering=0,
        parameter_count=9,
        baseline_macs=9.0,
    )
    context = SimpleNamespace(
        model=model,
        search_space=SearchSpaceSpec(
            pruning_unit_ids=[],
            precision_layer_ids=["late"],
            quantization_groups=(group,),
        ),
    )

    rows = measure_precision_sensitivity(
        context,
        fisher_statistics=SimpleNamespace(gradients={}, fisher_diag={}),
        baseline_runtime_shapes=[
            SimpleNamespace(module_path="late", call_index=0, macs=100.0)
        ],
    )

    assert rows[0].canonical_macs == pytest.approx(100.0)


def test_constrained_context_filters_existing_trace_without_modifying_it() -> None:
    from types import SimpleNamespace

    from search.canonicalization import SearchSpaceSpec
    from search.constrained.context import apply_constrained_pruning_context

    def unit(stable_id, root):
        return SimpleNamespace(
            stable_id=stable_id,
            root_module_path=root,
            root_axis="out",
            root_indices=[int(stable_id[1:])],
            scope_id=f"scope-{root}",
            constraints={},
            protected=False,
            source_coupled_unit_ids=[],
            members=[],
            normalized_score=0.0,
        )

    all_units = [
        unit("u0", "shrink_conv.layers.0.double_conv.0"),
        unit("u1", "shrink_conv.layers.0.double_conv.0"),
        unit("u2", "shrink_conv.layers.0.double_conv.2"),
        unit("u3", "backbone_m1.resnet.layer0.0.conv1"),
    ]
    context = SimpleNamespace(
        trace_result=SimpleNamespace(atomic_prune_units=all_units),
        search_space=SearchSpaceSpec(pruning_unit_ids=["u3"], precision_layer_ids=["m"]),
        atomic_prune_units=[all_units[-1]],
        pruning_action_catalog=None,
    )

    constrained, report = apply_constrained_pruning_context(
        context,
        allowed_root_patterns=("shrink_conv.layers.0.double_conv.0",),
        grouped_conv_mode="independent_group_topk",
        grouped_conv_align=4,
        grouped_allowed_channels_per_group=(4, 8, 16),
    )

    assert [unit.stable_id for unit in constrained.atomic_prune_units] == ["u0", "u1"]
    assert constrained.search_space.pruning_unit_ids == ["u0", "u1"]
    assert report["source"] == "existing_formal_trace_atomic_prune_units"
    assert report["tracer_modified"] is False
    assert len(context.trace_result.atomic_prune_units) == 4


def test_ga_uses_explicit_constrained_generation_zero_without_breeding() -> None:
    from search.candidate import CandidateGenotype
    from search.canonicalization import SearchSpaceSpec
    from search.ga.engine import GAConfig, GeneticSearchEngine

    space = SearchSpaceSpec(
        pruning_unit_ids=["u0", "u1"],
        precision_layer_ids=["pg0", "pg1"],
        default_precision="FP16",
    )
    seeds = [
        CandidateGenotype(
            {"u0": index % 2, "u1": (index // 2) % 2},
            {"pg0": "INT8" if index & 4 else "FP16", "pg1": "FP16"},
            {"seed_index": index},
        )
        for index in range(8)
    ]
    seen = []
    engine = GeneticSearchEngine(
        space,
        GAConfig(
            initial_population_size=8,
            population_size=4,
            offspring_size=4,
            num_generations=1,
        ),
    )

    scored = engine.run(
        evaluator=lambda candidate, generation: seen.append(candidate.meta["seed_index"])
        or {"F1": float(candidate.meta["seed_index"]), "generation": generation},
        initial_population=seeds,
    )

    assert seen == list(range(8))
    assert len(scored) == 8
    assert all(row[0].meta.get("seed_index") is not None for row in scored)


def test_constrained_smoke_config_is_independent_and_fail_closed() -> None:
    import yaml

    root = Path(__file__).resolve().parents[1]
    old_path = root / "search/configs/lidar_pyramid_4090_ga_stage2_multigpu_smoke.yaml"
    new_path = root / "search/configs/lidar_pyramid_4090_ga_stage2_constrained_smoke.yaml"
    old = yaml.safe_load(old_path.read_text(encoding="utf-8"))
    config = yaml.safe_load(new_path.read_text(encoding="utf-8"))

    assert old["precision"]["candidates"] == ["FP32", "FP16", "INT8"]
    assert config["precision"]["candidates"] == ["FP16", "INT8"]
    assert config["runtime"]["gpu_id"] == "5"
    assert config["runtime"]["plugin_boundary_dtype"] == "fp32"
    assert config["search"]["initial_population_size"] == 1024
    assert config["search"]["population_size"] == 512
    assert config["search"]["offspring_size"] == 512
    assert config["search"]["generations_per_round"] == 1
    assert config["search"]["topk_stage2"] == 5
    constrained = config["constrained_search"]
    assert constrained["enabled"] is True
    assert constrained["r_mac_floor"] == pytest.approx(0.95)
    assert constrained["int8_mac_share"] == [0.14, 0.22]
    assert constrained["seed_family_fractions"] == {
        "c_neighborhood": 0.35,
        "hybrid": 0.35,
        "b_derived": 0.20,
        "constrained_fresh": 0.10,
    }
    assert constrained["auto_start_stage_a"] is False
    assert constrained["max_light_pruned_units"] == 20
    assert config["stage2"]["smoke_frames"] == 10
    assert config["stage2"]["num_frames"] == 200
    assert config["stage2"]["warmup_frames"] == 20
    assert config["stage2"]["min_map"] == pytest.approx(0.705088)
    assert config["stage2"]["min_ap07"] == pytest.approx(0.564003)
    assert config["stage2_parallel"]["gpu_ids"] == [4, 5, 6, 7]
    assert config["cache"] == {
        "fresh_run": True,
        "reuse_external_cache": False,
        "resume": False,
    }


def test_constrained_top5_requires_individually_unique_physical_and_deployment_hashes(
    tmp_path: Path,
) -> None:
    from types import SimpleNamespace

    from search.orchestration.generation_stage2 import (
        deploy_generation_with_backfill,
    )

    records = [
        SimpleNamespace(candidate_hash=f"c{index}", F1=float(index))
        for index in range(7)
    ]

    def deploy(record, _candidate_dir):
        index = int(record.candidate_hash[1:])
        return {
            "status": "ok",
            "physical_hash": "p0" if index == 1 else f"p{index}",
            "deployment_hash": "d0" if index == 2 else f"d{index}",
            "F2": float(index),
        }

    report = deploy_generation_with_backfill(
        records,
        generation_index=0,
        output_dir=tmp_path,
        deploy_fn=deploy,
        topk=5,
        require_individual_hash_uniqueness=True,
    )

    assert [row["candidate_hash"] for row in report["candidates"]] == [
        "c0",
        "c3",
        "c4",
        "c5",
        "c6",
    ]
    assert {row["failure_reason"] for row in report["failure_records"]} == {
        "duplicate_physical_hash",
        "duplicate_deployment_hash",
    }


def test_two_level_stage2_builds_once_and_applies_ap_gate_only_to_full200(
    tmp_path: Path,
) -> None:
    import json

    from search.candidate import CandidatePhenotype, PrecisionDecision
    from search.stage2.lidar_pyramid_real_evaluator import (
        LidarPyramidRealEvaluator,
    )
    from search.stage2.objective import Stage2ObjectiveConfig

    evaluator = object.__new__(LidarPyramidRealEvaluator)
    evaluator.num_frames = 200
    evaluator.warmup_frames = 20
    evaluator.latency_rounds = 1
    evaluator.run_dir = tmp_path
    evaluator.objective_config = Stage2ObjectiveConfig(
        latency_metric="forward_p50_ms",
        min_map=0.705088,
        min_ap07=0.564003,
        required_evaluated_frames=200,
        required_skipped_frames=0,
        r_mac_floor=0.95,
        int8_mac_share_min=0.14,
        int8_mac_share_max=0.22,
    )
    evaluator._stage2_reference_baseline = lambda: {
        "mAP": 0.725088,
        "forward_p50_ms": 5.6,
    }
    evaluator._stage1_manifest_record = lambda _candidate_hash: {}
    calls = {"deploy": 0, "full": 0}

    def deploy(**kwargs):
        calls["deploy"] += 1
        destination = Path(kwargs["output_dir"])
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "realized_bops_audit.json").write_text(
            json.dumps(
                {
                    "passed": True,
                    "realized_bops": 21.0,
                    "bops_retention": 0.21,
                    "R_MAC": 0.97,
                    "int8_macs_share_full": 0.20,
                    "physical_params": 90,
                    "parameter_retention": 0.90,
                    "weight_storage_retention": 0.45,
                    "realized_precision_counts": {"FP16": 1},
                }
            ),
            encoding="utf-8",
        )
        (destination / "engine_realized_precision_profile.json").write_text(
            json.dumps(
                {
                    "passed": True,
                    "realized_precision_profile": {"layer": "FP16"},
                }
            ),
            encoding="utf-8",
        )
        for name in (
            "physical_validation.json",
            "pruning_quantization_group_audit.json",
            "production_qdq_boundary_audit.json",
            "engine_structure_validation.json",
            "precision_realization_validation.json",
            "typed_graph_report.json",
        ):
            (destination / name).write_text(
                json.dumps({"passed": True, "status": "passed", "issues": []}),
                encoding="utf-8",
            )
        (destination / "merge_precision_realization.json").write_text(
            json.dumps(
                {
                    "passed": True,
                    "status": "ok",
                    "issues": [],
                    "merges": [{"merge_op_name": "/Concat_9"}],
                }
            ),
            encoding="utf-8",
        )
        return {
            "status": "ok",
            "evaluation": {
                "status": "ok",
                "mAP": 0.001,
                "AP@0.7": 0.0,
                "forward_p50_ms": 4.0,
                "num_evaluated_frames": 10,
                "num_skipped_frames": 0,
            },
            "physical_hash": "physical",
            "deployment_hash": "deployment",
            "engine_hash": "engine",
            "engine_path": str(destination / "engine.plan"),
        }

    def evaluate_full(_engine_path, _output_dir):
        calls["full"] += 1
        assert evaluator.num_frames == 200
        assert evaluator.warmup_frames == 20
        return {
            "status": "ok",
            "mAP": 0.710,
            "AP@0.7": 0.570,
            "forward_p50_ms": 4.0,
            "num_evaluated_frames": 200,
            "num_skipped_frames": 0,
        }

    evaluator._deploy_and_evaluate = deploy
    evaluator._evaluate_engine = evaluate_full
    phenotype = CandidatePhenotype(
        pruned_unit_ids=["late_unit"],
        precision_profile={"layer": PrecisionDecision("FP16", "FP16", "")},
    )

    result = evaluator.evaluate_candidate_two_level(
        phenotype,
        output_dir=tmp_path / "candidate",
        candidate_hash="candidate",
        smoke_frames=10,
        smoke_warmup_frames=10,
    )

    assert calls == {"deploy": 1, "full": 1}
    assert result["status"] == "ok"
    assert result["smoke10_admission"]["formal_ap_gate_applied"] is False
    assert result["accuracy_admission_passed"] is True
    assert result["evaluated"] == 200
    assert result["skipped"] == 0
    assert result["requested_precision_profile_hash"] == result[
        "realized_precision_profile_hash"
    ]
