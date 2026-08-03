from __future__ import annotations

import random

import pytest

from search.candidate import CandidateGenotype
from search.canonicalization import SearchSpaceSpec
from search.ga.stage12_v3 import (
    RealAnchorArchive,
    Stage1Policy,
    adjacent_legal_mutation,
    evaluate_stage1_population,
    run_stage2_pipeline,
    same_locus_crossover,
    score_stage2_candidates,
    select_stage2_new_candidates,
)
from search.pruning_space.local_domains import LocalPruningDomain
from search.quantization_space.types import QuantizationSearchGroup


def _space() -> SearchSpaceSpec:
    domain = LocalPruningDomain(
        domain_id="cnn::a",
        root_module_path="a",
        root_axis="out",
        scope_id="a",
        kind="dense",
        original_width=8,
        total_original_width=8,
        ordered_unit_ids=("u0", "u1"),
        legal_widths=(4, 8),
        width_to_pruned_unit_ids={4: ("u0",), 8: ()},
        unit_root_indices={"u0": (0,), "u1": (1,)},
    )
    mutable = QuantizationSearchGroup(
        group_id="precision::a",
        module_paths=("a",),
        canonical_node_ids=("n0",),
        allowed_precisions=("FP32", "FP16", "INT8"),
        protected=False,
        protection_reason="",
        ordering=0,
        parameter_count=8,
        baseline_macs=8,
    )
    fixed = QuantizationSearchGroup(
        group_id="precision::qk",
        module_paths=("qk",),
        canonical_node_ids=("n1",),
        allowed_precisions=("FP32",),
        protected=True,
        protection_reason="qk_fp32",
        ordering=1,
        parameter_count=0,
        baseline_macs=8,
    )
    return SearchSpaceSpec(
        pruning_unit_ids=["u0", "u1"],
        precision_layer_ids=["a", "qk"],
        quantization_groups=(mutable, fixed),
        pruning_domains=(domain,),
        default_precision="FP32",
    )


def _candidate(width: int, precision: str) -> CandidateGenotype:
    return CandidateGenotype(
        pruning_width_genes={"cnn::a": width},
        precision_genes={"precision::a": precision},
    )


def _resources(phenotype):
    width = 4 if "u0" in phenotype.pruned_unit_ids else 8
    precision = phenotype.realized_precision_profile["a"]
    retention = {("8", "FP32"): 0.50, ("4", "FP32"): 0.30, ("8", "FP16"): 0.30, ("4", "FP16"): 0.20}.get(
        (str(width), precision), 0.10
    )
    parameter_retention = width / 8
    mixed = parameter_retention * {"FP32": 1.0, "FP16": 0.5, "INT8": 0.25}[precision]
    return {
        "BOPS": retention * 100,
        "R_BOPS": retention,
        "params": width,
        "parameter_retention": parameter_retention,
        "mixed_weight_size": mixed * 32,
        "mixed_weight_retention": mixed,
    }


def _proxy(phenotype):
    width = 4 if "u0" in phenotype.pruned_unit_ids else 8
    precision = phenotype.realized_precision_profile["a"]
    return {
        "J_struct": 0.2 if width == 4 else 0.0,
        "J_WQ": 0.05 if precision != "FP32" else 0.0,
        "J_AQ": 0.03 if precision != "FP32" else 0.0,
    }


def test_stage1_hard_gate_dedup_and_proxy_sum() -> None:
    result = evaluate_stage1_population(
        [_candidate(4, "FP32"), _candidate(4, "FP32"), _candidate(8, "FP32")],
        _space(),
        policy=Stage1Policy(0.30),
        resource_evaluator=_resources,
        proxy_evaluator=_proxy,
    )
    assert result["eligible_count"] == 1
    row = result["eligible"][0]
    assert row["J_total"] == pytest.approx(row["J_struct"] + row["J_WQ"] + row["J_AQ"])
    assert result["records"][1]["dedup_status"] == "duplicate_physical_hash"
    assert result["records"][2]["hard_gate_status"] == "rejected_outside_bops_band"
    assert result["repair_count"] == 0


def test_fixed_precision_locus_is_not_a_chromosome_gene() -> None:
    assert _space().precision_gene_ids == ["precision::a"]
    invalid = CandidateGenotype(
        pruning_width_genes={"cnn::a": 4},
        precision_genes={"precision::a": "FP16", "precision::qk": "FP32"},
    )
    result = evaluate_stage1_population(
        [invalid], _space(), policy=Stage1Policy(0.30), resource_evaluator=_resources, proxy_evaluator=_proxy
    )
    assert result["eligible_count"] == 0
    assert "fixed_locus" in result["records"][0]["failure_reason"]


def test_mutation_is_adjacent_and_crossover_is_homologous() -> None:
    space = _space()
    source = _candidate(8, "FP32")
    precision_child = adjacent_legal_mutation(source, space, random.Random(2), locus_kind="precision")
    assert precision_child.precision_genes["precision::a"] == "FP16"
    assert precision_child.pruning_width_genes == source.pruning_width_genes
    structure_child = adjacent_legal_mutation(source, space, random.Random(3), locus_kind="structure")
    assert structure_child.pruning_width_genes["cnn::a"] == 4
    child = same_locus_crossover(source, _candidate(4, "FP16"), space, random.Random(4))
    assert child.pruning_width_genes["cnn::a"] in {4, 8}
    assert child.precision_genes["precision::a"] in {"FP32", "FP16"}
    assert child.meta["repair_count"] == 0


def test_stage2_pipeline_stops_at_first_failure_without_fallback() -> None:
    called = []

    def runner(step, _candidate, _state):
        called.append(step)
        return {"status": "failed" if step == "tensorrt_engine_build" else "ok"}

    result = run_stage2_pipeline({"physical_hash": "x"}, step_runner=runner)
    assert result["failed_step"] == "tensorrt_engine_build"
    assert result["precision_fallback_allowed"] is False
    assert "requested_realized_precision" not in called


def _stage2_row(name, m_ap, latency, **extra):
    return {
        "physical_hash": name,
        "status": "ok",
        "requested_realized_exact": True,
        "mAP": m_ap,
        "p50_ms": latency,
        "parameter_retention": extra.get("parameter_retention", 0.8),
        "mixed_weight_retention": extra.get("mixed_weight_retention", 0.5),
        "BOPS_deviation": extra.get("BOPS_deviation", 0.001),
    }


def test_stage2_accuracy_gate_score_and_dominant_winner() -> None:
    anchor = _stage2_row("g", 0.60, 10.0)
    result = score_stage2_candidates(
        [_stage2_row("dominant", 0.601, 8.0), _stage2_row("unsafe", 0.59, 7.0)],
        greedy_anchor=anchor,
    )
    assert result["winner"]["physical_hash"] == "dominant"
    assert result["winner_reason"] == "dominant_highest_map_and_lowest_latency"
    assert result["rows"][1]["stage2_eligible"] is False
    assert result["rows"][0]["F_S2"] == pytest.approx(0.2 * -0.2 + 0.8 * 0.8)


def test_no_eligible_candidate_does_not_fabricate_winner() -> None:
    result = score_stage2_candidates(
        [_stage2_row("x", 0.50, 8.0)], greedy_anchor=_stage2_row("g", 0.60, 10.0)
    )
    assert result["winner"] is None
    assert result["new_generation_winner_created"] is False


def test_historical_elites_do_not_consume_new_engine_quota() -> None:
    rows = [{"physical_hash": f"n{index}"} for index in range(7)]
    result = select_stage2_new_candidates(
        rows,
        evaluated_hashes={"n0"},
        historical_real_elites=[{"physical_hash": "old"}, {"physical_hash": "old"}],
        quota=5,
    )
    assert len(result["historical_real_elites"]) == 1
    assert len(result["new_candidates"]) == 5
    assert result["historical_elites_consume_quota"] is False


def test_v1_v2_v3_anchor_update_and_dedup() -> None:
    greedy = {**_stage2_row("g", 0.60, 10.0), "F_S2": 0.8, "stage2_eligible": True}
    archive = RealAnchorArchive(greedy)
    rows = [
        {**_stage2_row("a", 0.602, 9.0), "F_S2": 0.70, "stage2_eligible": True},
        {**_stage2_row("b", 0.601, 8.0), "F_S2": 0.72, "stage2_eligible": True},
        {**_stage2_row("bad", 0.4, 5.0), "F_S2": 0.1, "stage2_eligible": False},
    ]
    archive.update(rows)
    result = archive.anchors()
    assert result["greedy_anchor_preserved"] is True
    assert result["black_box_surrogate"] is False
    assert {row["physical_hash"] for row in result["anchors"]} == {"g", "a", "b"}
    assert "highest_global_mAP" in result["roles_by_physical_hash"]["a"]
    assert "lowest_global_p50" in result["roles_by_physical_hash"]["b"]


def test_deterministic_same_seed() -> None:
    left = adjacent_legal_mutation(_candidate(8, "FP32"), _space(), random.Random(17))
    right = adjacent_legal_mutation(_candidate(8, "FP32"), _space(), random.Random(17))
    assert left == right


def test_multi_budget_greedy_runs_past_first_band_and_captures_all() -> None:
    from search.greedy.conservative_joint import run_conservative_joint_greedy_multi_budget

    class Structure:
        @staticmethod
        def pruning_action_breakdown(current, successor):
            return {"delta_J_struct": 0.2}

    class Weight:
        @staticmethod
        def weight_quantization_action_breakdown(current, successor):
            return {"delta_J_WQ": 0.04, "first_order_abs_sum": 0.03, "second_order_abs_sum": 0.01}

    class Activation:
        @staticmethod
        def quantization_action_breakdown(current, successor, *, changed_gene_id):
            return {"delta_J_AQ": 0.04, "first_order_abs_sum": 0.03, "second_order_abs_sum": 0.01}

    def bops(phenotype):
        width = 4 if "u0" in phenotype.pruned_unit_ids else 8
        precision = phenotype.realized_precision_profile["a"]
        retention = {
            (8, "FP32"): 0.50,
            (4, "FP32"): 0.30,
            (8, "FP16"): 0.30,
            (4, "FP16"): 0.20,
            (8, "INT8"): 0.20,
            (4, "INT8"): 0.10,
        }[(width, precision)]
        return {"bops_total": retention * 100.0, "R_bops_vs_fp32": retention}

    def size(phenotype):
        width = 4 if "u0" in phenotype.pruned_unit_ids else 8
        precision = phenotype.realized_precision_profile["a"]
        retention = width / 8
        bits = {"FP32": 32, "FP16": 16, "INT8": 8}[precision]
        return {
            "R_parameter_retention": retention,
            "parameter_count_after": width,
            "size_bits_total": width * bits,
            "R_size_vs_fp32": retention * bits / 32,
        }

    result = run_conservative_joint_greedy_multi_budget(
        _space(),
        structural_proxy=Structure(),
        weight_proxy=Weight(),
        activation_proxy=Activation(),
        bops_evaluator=bops,
        size_evaluator=size,
        targets=(0.30, 0.20, 0.10),
        tolerance_abs=1.0e-9,
    )
    assert result["budget_reached"] == {0.30: True, 0.20: True, 0.10: True}
    assert all(result["budget_band_candidate_counts"][target] > 0 for target in result["targets"])
    assert result["selected_step_count"] >= 3
    assert result["repair_count"] == 0
