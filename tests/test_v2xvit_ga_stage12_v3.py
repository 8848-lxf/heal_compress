from __future__ import annotations

import random

import pytest


def _domain(index: int):
    from search.pruning_space.local_domains import LocalPruningDomain

    domain_id = f"domain_{index}"
    return LocalPruningDomain(
        domain_id=domain_id,
        root_module_path=domain_id,
        root_axis="out",
        scope_id=domain_id,
        kind="dense",
        original_width=8,
        total_original_width=8,
        ordered_unit_ids=(f"{domain_id}:0",),
        legal_widths=(4, 8),
        width_to_pruned_unit_ids={4: (f"{domain_id}:0",), 8: ()},
        unit_root_indices={f"{domain_id}:0": (0,)},
        domain_type="cnn_channel",
    )


def _group(group_id: str, *, protected: bool = False):
    from search.quantization_space.types import QuantizationSearchGroup

    return QuantizationSearchGroup(
        group_id=group_id,
        module_paths=(f"module.{group_id}",),
        canonical_node_ids=(group_id,),
        allowed_precisions=("FP32",) if protected else ("FP32", "FP16", "INT8"),
        protected=protected,
        protection_reason="fixed" if protected else "",
        ordering=0,
        parameter_count=1,
        baseline_macs=1.0,
        metadata={"default_precision": "FP32"},
    )


def _space(domain_count: int = 2):
    from search.canonicalization import SearchSpaceSpec

    return SearchSpaceSpec(
        pruning_unit_ids=[],
        precision_layer_ids=[],
        pruning_domains=tuple(_domain(index) for index in range(domain_count)),
        quantization_groups=(_group("projection"), _group("qk", protected=True)),
        default_precision="FP32",
    )


def _candidate(space, mask: int = 0, precision: str = "FP32"):
    from search.candidate import CandidateGenotype

    widths = {
        locus: 4 if mask & (1 << index) else 8
        for index, locus in enumerate(space.pruning_gene_ids)
    }
    return CandidateGenotype(
        pruning_width_genes=widths,
        precision_genes={"projection": precision},
    )


def test_formal_configuration_is_exactly_ten_generations() -> None:
    from search.ga.stage12_v3 import StrictGAConfig
    from scripts.run_v2xvit_formal_ga_gen10 import FORMAL_SEEDS, LABELS, TARGETS

    config = StrictGAConfig(target_bops_retention=0.30)
    assert config.generations == 10
    assert FORMAL_SEEDS == (0,)
    assert LABELS == ("010",)
    assert TARGETS == {"010": 0.10}
    with pytest.raises(ValueError, match="generations_must_equal_10"):
        StrictGAConfig(target_bops_retention=0.30, generations=11)


def test_fixed_precision_locus_is_absent_and_fails_if_injected() -> None:
    from search.candidate import CandidateGenotype
    from search.ga.stage12_v3 import validate_genotype_schema

    space = _space()
    assert space.precision_gene_ids == ["projection"]
    invalid = CandidateGenotype(
        pruning_width_genes={locus: 8 for locus in space.pruning_gene_ids},
        precision_genes={"projection": "FP32", "qk": "INT8"},
    )
    with pytest.raises(ValueError, match="precision_schema_mismatch"):
        validate_genotype_schema(invalid, space)


def test_adjacent_mutation_and_same_locus_crossover_need_no_repair() -> None:
    from search.ga.stage12_v3 import (
        adjacent_mutation,
        same_locus_crossover,
        validate_genotype_schema,
    )

    space = _space()
    left = _candidate(space, 0, "FP32")
    right = _candidate(space, 3, "INT8")
    child = same_locus_crossover(left, right, space, random.Random(4))
    validate_genotype_schema(child, space)
    mutated = adjacent_mutation(child, space, random.Random(8))
    validate_genotype_schema(mutated, space)
    differences = sum(
        left_value != right_value
        for left_value, right_value in zip(
            [
                *(child.pruning_width_genes[key] for key in space.pruning_gene_ids),
                child.precision_genes["projection"],
            ],
            [
                *(mutated.pruning_width_genes[key] for key in space.pruning_gene_ids),
                mutated.precision_genes["projection"],
            ],
        )
    )
    assert differences == 1
    if child.precision_genes != mutated.precision_genes:
        assert (child.precision_genes["projection"], mutated.precision_genes["projection"]) in {
            ("FP32", "FP16"), ("FP16", "FP32"), ("FP16", "INT8"), ("INT8", "FP16")
        }


def test_stage1_rank_does_not_reward_more_pruning_or_smaller_weight_size() -> None:
    from search.ga.stage12_v3 import rank_stage1

    conservative = {
        "J_total": 1.0,
        "bops_feasible": True,
        "bops_deviation": 0.001,
        "R_parameter_retention": 0.9,
        "mixed_weight_retention": 0.8,
        "complete_phenotype_hash": "b",
    }
    aggressive = {
        **conservative,
        "R_parameter_retention": 0.2,
        "mixed_weight_retention": 0.1,
        "complete_phenotype_hash": "a",
    }
    assert rank_stage1([aggressive, conservative])[0] == conservative


def test_stage1_bops_hard_gate_runs_before_taylor_and_size() -> None:
    from search.ga.stage12_v3 import UnifiedTaylorStage1Evaluator

    class Forbidden:
        def __getattr__(self, name):
            raise AssertionError(f"must_not_evaluate_after_failed_bops_gate:{name}")

    space = _space()
    baseline = _candidate(space)
    evaluator = UnifiedTaylorStage1Evaluator(
        space,
        baseline=baseline,
        structure_proxy=Forbidden(),
        weight_proxy=Forbidden(),
        activation_cache=Forbidden(),
        bops_evaluator=lambda _phenotype: {"R_bops_vs_fp32": 0.50},
        size_evaluator=lambda _phenotype: (_ for _ in ()).throw(
            AssertionError("size_must_not_run_after_failed_bops_gate")
        ),
        target=0.30,
    )
    first = evaluator(baseline)
    second = evaluator(baseline)
    assert first["bops_hard_gate_passed"] is False
    assert first["taylor_evaluated_after_bops_hard_gate"] is False
    assert second["stage1_cache_hit"] is True


def test_stage2_gate_formula_dominance_and_no_eligible_behavior() -> None:
    from search.ga.stage12_v3 import Stage2Result, score_stage2, select_generation_winner

    space = _space()
    genotype = _candidate(space)
    good = Stage2Result("good", genotype, "ok", 0.60, 8.0, True, 50, 0)
    fast_low = Stage2Result("fast", genotype, "ok", 0.596, 7.0, True, 50, 0)
    bad = Stage2Result("bad", genotype, "ok", 0.58, 6.0, True, 50, 0)
    assert score_stage2(bad, greedy_map=0.60, greedy_p50_ms=10.0)["eligible"] is False
    stage1 = {
        key: {
            "R_parameter_retention": 0.8,
            "mixed_weight_retention": 0.7,
            "bops_deviation": 0.0,
        }
        for key in ("good", "fast", "bad")
    }
    winner, _ = select_generation_winner(
        [good, fast_low], greedy_map=0.60, greedy_p50_ms=10.0, stage1_by_hash=stage1
    )
    assert winner is not None
    assert winner.complete_phenotype_hash == "good"
    winner, _ = select_generation_winner(
        [bad], greedy_map=0.60, greedy_p50_ms=10.0, stage1_by_hash=stage1
    )
    assert winner is None


def test_historical_real_elite_does_not_consume_new_stage2_quota() -> None:
    from search.ga.stage12_v3 import select_new_stage2_candidates

    rows = [
        {
            "complete_phenotype_hash": str(index),
            "physical_structure_hash": f"p{index}",
            "precision_map_hash": f"q{index}",
        }
        for index in range(8)
    ]
    selected = select_new_stage2_candidates(
        rows, evaluated_hashes={"0", "1"}, quota=5
    )
    assert [row["complete_phenotype_hash"] for row in selected] == ["2", "3", "4", "5", "6"]


def test_runner_generation_zero_is_initialization_and_evolution_is_one_to_ten() -> None:
    from search.ga.stage12_v3 import (
        Stage2Result,
        StrictGAConfig,
        StrictStage12V3Runner,
        phenotype_identity,
    )

    space = _space(domain_count=8)
    population = [_candidate(space, mask, ("FP32", "FP16", "INT8")[mask % 3]) for mask in range(64)]

    def stage1(genotype):
        identity = phenotype_identity(genotype, space)
        mask_cost = sum(8 - value for value in genotype.pruning_width_genes.values())
        precision_cost = {"FP32": 0.0, "FP16": 1.0, "INT8": 2.0}[genotype.precision_genes["projection"]]
        return {
            **identity,
            "J_total": mask_cost + precision_cost,
            "J_struct_gate": mask_cost,
            "J_WQ": precision_cost,
            "J_AQ": 0.0,
            "bops_feasible": True,
            "bops_deviation": 0.0,
            "R_parameter_retention": 0.8,
            "mixed_weight_retention": 0.7,
            "genotype": genotype,
        }

    greedy = population[0]
    greedy_hash = phenotype_identity(greedy, space)["complete_phenotype_hash"]
    anchor = Stage2Result(greedy_hash, greedy, "ok", 0.60, 10.0, True, 50, 0)

    def stage2(genotype, generation):
        identity = phenotype_identity(genotype, space)["complete_phenotype_hash"]
        return Stage2Result(identity, genotype, "ok", 0.60, 9.5, True, 50, 0, {"generation": generation})

    result = StrictStage12V3Runner(
        space,
        StrictGAConfig(target_bops_retention=0.30, random_seed=17),
        stage1_evaluator=stage1,
        stage2_evaluator=stage2,
    ).run(population, greedy_anchor=anchor)
    assert result["history"][0]["generation"] == 0
    assert result["history"][0]["counted_as_evolution_generation"] is False
    assert [row["generation"] for row in result["history"][1:]] == list(range(1, 11))
    assert result["completed_evolution_generations"] == 10
    assert all(row["stage2_new_candidate_count"] <= 5 for row in result["history"])
    assert all(row.get("greedy_anchor_retained", True) for row in result["history"])
    assert all(row["offspring_generated"] == 64 for row in result["history"][1:])
    assert all(row["offspring_unique_count"] == 64 for row in result["history"][1:])
    assert all(row["survivor_size"] == 64 for row in result["history"][1:])
    assert all(row["survivor_unique_count"] == 64 for row in result["history"][1:])


def test_runner_rejects_duplicate_initial_physical_phenotype() -> None:
    from search.ga.stage12_v3 import (
        Stage2Result,
        StrictGAConfig,
        StrictStage12V3Runner,
        phenotype_identity,
    )

    space = _space(domain_count=8)
    population = [
        _candidate(space, mask, ("FP32", "FP16", "INT8")[mask % 3])
        for mask in range(63)
    ]
    population.append(population[0])

    def stage1(genotype):
        return {
            **phenotype_identity(genotype, space),
            "J_total": 0.0,
            "bops_feasible": True,
            "bops_deviation": 0.0,
            "R_parameter_retention": 1.0,
            "mixed_weight_retention": 1.0,
            "genotype": genotype,
        }

    greedy = population[0]
    anchor = Stage2Result(
        phenotype_identity(greedy, space)["complete_phenotype_hash"],
        greedy,
        "ok",
        0.6,
        10.0,
        True,
        50,
        0,
    )
    runner = StrictStage12V3Runner(
        space,
        StrictGAConfig(target_bops_retention=0.10),
        stage1_evaluator=stage1,
        stage2_evaluator=lambda *_args: anchor,
    )
    with pytest.raises(ValueError, match="duplicate_phenotype"):
        runner.run(population, greedy_anchor=anchor)


def test_stage2_realization_requires_functional_and_attention_audits(tmp_path) -> None:
    import json

    from scripts.run_v2xvit_formal_ga_gen10 import precision_realized_exact

    (tmp_path / "engine_build_acceptance.json").write_text(json.dumps({
        "status": "ok",
        "precision_realization_validation": {
            "passed": True,
            "mismatches": [],
            "unresolved_layer_count": 0,
        },
    }))
    exact, reason = precision_realized_exact(tmp_path)
    assert exact is False
    assert reason["failure"] == "functional_attention_or_av_precision_audit_missing"
    (tmp_path / "functional_precision_trt_audit.json").write_text(json.dumps({
        "passed": True,
        "conflict_count": 0,
        "unmapped_count": 0,
        "fallback_count": 0,
    }))
    (tmp_path / "trt_attention_fp32_audit.json").write_text(json.dumps({
        "passed": True,
    }))
    (tmp_path / "av_profile_trt_audit.json").write_text(json.dumps({
        "passed": True,
        "conflict_count": 0,
        "unmapped_count": 0,
        "fallback_count": 0,
    }))
    exact, _ = precision_realized_exact(tmp_path)
    assert exact is True


def test_runner_parallel_stage2_batch_preserves_stage1_order() -> None:
    from search.ga.stage12_v3 import (
        Stage2Result,
        StrictGAConfig,
        StrictStage12V3Runner,
        phenotype_identity,
    )

    space = _space(domain_count=8)
    population = [
        _candidate(space, mask, ("FP32", "FP16", "INT8")[mask % 3])
        for mask in range(64)
    ]

    def stage1(genotype):
        identity = phenotype_identity(genotype, space)
        return {
            **identity,
            "J_total": float(sum(genotype.pruning_width_genes.values())),
            "J_struct_gate": 0.0,
            "J_WQ": 0.0,
            "J_AQ": 0.0,
            "bops_feasible": True,
            "bops_deviation": 0.0,
            "R_parameter_retention": 0.8,
            "mixed_weight_retention": 0.7,
            "genotype": genotype,
        }

    greedy = population[0]
    greedy_hash = phenotype_identity(greedy, space)["complete_phenotype_hash"]
    anchor = Stage2Result(greedy_hash, greedy, "ok", 0.60, 10.0, True, 50, 0)

    class BatchOnlyEvaluator:
        def __init__(self):
            self.calls = []

        def __call__(self, _genotype, _generation):
            raise AssertionError("scalar_stage2_must_not_run_when_batch_api_exists")

        def evaluate_many(self, genotypes, generation):
            hashes = [
                phenotype_identity(genotype, space)["complete_phenotype_hash"]
                for genotype in genotypes
            ]
            self.calls.append((generation, hashes))
            return [
                Stage2Result(identity, genotype, "ok", 0.60, 9.5, True, 50, 0)
                for identity, genotype in zip(hashes, genotypes)
            ]

    evaluator = BatchOnlyEvaluator()
    result = StrictStage12V3Runner(
        space,
        StrictGAConfig(target_bops_retention=0.30, random_seed=23),
        stage1_evaluator=stage1,
        stage2_evaluator=evaluator,
    ).run(population, greedy_anchor=anchor)
    assert result["completed_evolution_generations"] == 10
    assert [generation for generation, _ in evaluator.calls] == list(range(1, 11))
    assert all(1 <= len(hashes) <= 5 for _, hashes in evaluator.calls)
    for record, (_, hashes) in zip(result["history"][1:], evaluator.calls):
        assert record["stage2_new_candidate_hashes"] == hashes


def test_parallel_worker_domain_payload_roundtrip_preserves_physical_ranking() -> None:
    from scripts.run_v2xvit_ga_stage2_worker import domain_from_payload

    domain = _domain(0)
    restored = domain_from_payload(domain.to_dict())
    assert restored.domain_id == domain.domain_id
    assert restored.legal_widths == domain.legal_widths
    assert restored.ordered_unit_ids == domain.ordered_unit_ids
    assert restored.width_to_pruned_unit_ids == domain.width_to_pruned_unit_ids
    assert restored.decode_width(4) == domain.decode_width(4)


def test_ga_final_report_precision_counts_only_mutable_loci() -> None:
    from scripts.finalize_v2xvit_ga_gen10_reports import precision_counts

    genotype = {
        "precision_genes": {
            "mutable_a": "FP32",
            "mutable_b": "FP16",
            "mutable_c": "INT8",
            "mutable_d": "INT8",
        }
    }
    assert precision_counts(genotype) == {"FP32": 1, "FP16": 1, "INT8": 2}


def test_anchor_constrained_space_restores_serialized_nested_mask_and_hash() -> None:
    from search.ga.anchor_constrained_space import constrain_domains_to_frozen_anchors
    from search.pruning_space.local_domains import LocalPruningDomain

    domain = LocalPruningDomain(
        domain_id="cnn",
        root_module_path="cnn",
        root_axis="out",
        scope_id="cnn",
        kind="dense",
        original_width=4,
        total_original_width=4,
        ordered_unit_ids=("u3", "u2", "u1", "u0"),
        legal_widths=(2, 4),
        width_to_pruned_unit_ids={2: ("u3", "u2"), 4: ()},
        unit_root_indices={f"u{i}": (i,) for i in range(4)},
        ranking_hash="replayed-rank",
        domain_type="cnn_channel",
    )
    all_keep = {
        "metadata": {
            "domains": {
                "cnn": {
                    "retained_width": 4,
                    "pruned_unit_ids": [],
                    "ranking_hash": "frozen-rank",
                    "decoded_width_state": {
                        "domain_id": "cnn",
                        "domain_type": "cnn_channel",
                        "retained_width": 4,
                        "pruned_unit_ids": [],
                    },
                }
            }
        }
    }
    pruned = {
        "metadata": {
            "domains": {
                "cnn": {
                    "retained_width": 2,
                    "pruned_unit_ids": ["u0", "u1"],
                    "ranking_hash": "frozen-rank",
                    "decoded_width_state": {
                        "domain_id": "cnn",
                        "domain_type": "cnn_channel",
                        "retained_width": 2,
                        "pruned_unit_ids": ["u0", "u1"],
                    },
                }
            }
        }
    }
    constrained = constrain_domains_to_frozen_anchors(
        (domain,), (all_keep, pruned)
    )[0]
    assert constrained.ranking_hash == "frozen-rank"
    assert constrained.pruned_unit_ids_for_width(2) == ("u0", "u1")
    assert constrained.decode_width(2)["pruned_unit_ids"] == ["u0", "u1"]


def test_frozen_domain_manifest_roundtrip_preserves_every_width_mask(tmp_path) -> None:
    from search.ga.frozen_domain_manifest import (
        build_manifest,
        domain_table_hash,
        load_manifest,
        write_manifest,
    )
    from search.pruning_space.local_domains import LocalPruningDomain

    domain = LocalPruningDomain(
        domain_id="cnn",
        root_module_path="cnn",
        root_axis="out",
        scope_id="cnn",
        kind="dense",
        original_width=4,
        total_original_width=4,
        ordered_unit_ids=("u0", "u1", "u2", "u3"),
        legal_widths=(2, 3, 4),
        width_to_pruned_unit_ids={2: ("u0", "u1"), 3: ("u0",), 4: ()},
        unit_root_indices={f"u{i}": (i,) for i in range(4)},
        ranking_hash="frozen",
        domain_type="cnn_channel",
        dependency_members=({"module_path": "next", "axis": "input"},),
    )
    manifest = build_manifest(
        (domain,), anchor_phenotypes=(), source_root="/source",
        calibration_hash="calibration", trace_hash="trace",
    )
    path = tmp_path / "domains.json"
    write_manifest(path, manifest)
    payload, restored = load_manifest(path)
    assert payload["domain_table_hash"] == domain_table_hash((domain,))
    assert restored[0].to_dict() == domain.to_dict()
    assert restored[0].width_to_pruned_unit_ids == domain.width_to_pruned_unit_ids


def test_frozen_domain_manifest_tamper_fails_closed(tmp_path) -> None:
    import json

    import pytest

    from search.ga.frozen_domain_manifest import build_manifest, load_manifest, write_manifest
    from search.pruning_space.local_domains import LocalPruningDomain

    domain = LocalPruningDomain(
        domain_id="cnn", root_module_path="cnn", root_axis="out",
        scope_id="cnn", kind="dense", original_width=2,
        total_original_width=2, ordered_unit_ids=("u0", "u1"),
        legal_widths=(1, 2),
        width_to_pruned_unit_ids={1: ("u0",), 2: ()},
        unit_root_indices={"u0": (0,), "u1": (1,)},
        domain_type="cnn_channel",
    )
    path = tmp_path / "domains.json"
    write_manifest(path, build_manifest(
        (domain,), anchor_phenotypes=(), source_root="/source",
        calibration_hash="calibration",
    ))
    payload = json.loads(path.read_text())
    payload["domains"][0]["width_to_pruned_unit_ids"]["1"] = ["u1"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="manifest_hash_mismatch"):
        load_manifest(path)


def test_frozen_domain_manifest_rejects_tracer_schema_drift() -> None:
    import pytest

    from search.ga.frozen_domain_manifest import validate_against_replay
    from search.pruning_space.local_domains import LocalPruningDomain

    common = dict(
        domain_id="cnn", root_module_path="cnn", root_axis="out",
        scope_id="cnn", kind="dense", original_width=2,
        total_original_width=2, ordered_unit_ids=("u0", "u1"),
        legal_widths=(1, 2), width_to_pruned_unit_ids={1: ("u0",), 2: ()},
        unit_root_indices={"u0": (0,), "u1": (1,)},
        domain_type="cnn_channel",
    )
    frozen = LocalPruningDomain(**common)
    replay = LocalPruningDomain(**{**common, "root_module_path": "different"})
    with pytest.raises(RuntimeError, match="schema_drift:cnn:root_module_path"):
        validate_against_replay((frozen,), (replay,))
