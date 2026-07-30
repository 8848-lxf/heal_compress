from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def _domain():
    from search.pruning_space.local_domains import LocalPruningDomain

    return LocalPruningDomain(
        domain_id="backbone::out",
        root_module_path="backbone",
        root_axis="out",
        scope_id="backbone",
        kind="dense",
        original_width=12,
        total_original_width=12,
        ordered_unit_ids=("u0", "u1"),
        legal_widths=(4, 8, 12),
        width_to_pruned_unit_ids={4: ("u0", "u1"), 8: ("u0",), 12: ()},
        unit_root_indices={"u0": (0, 1, 2, 3), "u1": (4, 5, 6, 7)},
        domain_type="cnn_channel",
    )


def _group(group_id: str, protected: bool = False):
    from search.quantization_space.types import QuantizationSearchGroup

    return QuantizationSearchGroup(
        group_id=group_id,
        module_paths=(group_id,),
        canonical_node_ids=(group_id,),
        allowed_precisions=("FP32",) if protected else ("FP32", "FP16", "INT8"),
        protected=protected,
        protection_reason="fixed" if protected else "",
        ordering=0 if not protected else 1,
        parameter_count=4,
        baseline_macs=4.0,
    )


def _space():
    from search.canonicalization import SearchSpaceSpec

    return SearchSpaceSpec(
        pruning_unit_ids=["u0", "u1"],
        precision_layer_ids=["conv", "fixed"],
        pruning_domains=(_domain(),),
        quantization_groups=(_group("conv"), _group("fixed", True)),
        default_precision="FP32",
    )


def test_explicit_gen5_contract_does_not_weaken_default_gen10() -> None:
    from search.ga.stage12_v3 import StrictGAConfig

    assert StrictGAConfig(0.1).generations == 10
    configured = StrictGAConfig(
        0.1, generations=5, generation_contract="formal_gen5"
    )
    assert configured.generations == 5
    with pytest.raises(ValueError, match="generations_contract_mismatch"):
        StrictGAConfig(0.1, generations=4, generation_contract="formal_gen5")


def test_cnn_baseline_contains_only_mutable_loci() -> None:
    from search.ga.cnn_stage12_v3 import baseline_genotype
    from search.ga.stage12_v3 import validate_genotype_schema

    space = _space()
    candidate = baseline_genotype(space)
    assert candidate.pruning_width_genes == {"backbone::out": 12}
    assert candidate.precision_genes == {"conv": "FP32"}
    assert "fixed" not in candidate.precision_genes
    validate_genotype_schema(candidate, space)


def test_cnn_greedy_neighbors_are_decreasing_and_adjacent() -> None:
    from search.ga.cnn_stage12_v3 import baseline_genotype, decreasing_neighbors

    space = _space()
    candidate = baseline_genotype(space)
    rows = decreasing_neighbors(candidate, space)
    by_type = {kind: child for kind, _locus, child in rows}
    assert by_type["structure"].pruning_width_genes["backbone::out"] == 8
    assert by_type["precision"].precision_genes["conv"] == "FP16"


def test_cnn_formal_entrypoint_freezes_five_generations_and_one_seed() -> None:
    source = (__import__("pathlib").Path(__file__).parents[1]
              / "scripts/run_cnn_formal_ga_gen5.py").read_text()
    assert "requires_exactly_5_generations" in source
    assert "single_seed_zero_required" in source
    assert '"StrictStage12V3Runner"' in source
    assert "full1789_executed" in source


def test_size_proxy_canonical_field_is_exposed_to_selector_without_recompute() -> None:
    from search.ga.cnn_stage12_v3 import size_metrics_with_alias

    class Proxy:
        calls = 0

        def evaluate_breakdown(self, _phenotype):
            self.calls += 1
            return {"R_size_vs_fp32": 0.375, "R_parameter_retention": 0.5}

    proxy = Proxy()
    result = size_metrics_with_alias(proxy, object())
    assert proxy.calls == 1
    assert result["mixed_weight_retention"] == 0.375
    assert result["R_size_vs_fp32"] == 0.375


class _AuditMetricProxy:
    def __init__(self, reductions, risks=None):
        self.reductions = dict(reductions)
        self.risks = dict(risks or {})

    @staticmethod
    def _profile(phenotype):
        return dict(phenotype.metadata["stage1_legalized_group_profile"])

    def evaluate_breakdown(self, phenotype):
        profile = self._profile(phenotype)
        selected = {
            locus for locus, precision in profile.items() if precision == "FP16"
        }
        retention = 1.0 - sum(
            self.reductions.get(locus, 0.0) for locus in selected
        )
        return {
            "R_bops_vs_fp32": retention,
            "R_parameter_retention": 1.0,
            "R_size_vs_fp32": retention,
        }

    def weight_quantization_action_breakdown(self, before, after):
        before_profile = self._profile(before)
        after_profile = self._profile(after)
        changed = [
            locus
            for locus in after_profile
            if before_profile[locus] != after_profile[locus]
        ]
        assert len(changed) == 1
        return {"delta_J_WQ": self.risks[changed[0]]}


class _ZeroActionProxy:
    def action_breakdown(self, _before, _after):
        return {"delta_J_AQ": 0.0}

    def pruning_action_breakdown(self, _before, _after):
        return {"delta_J_prune": 0.0}


def _frontier_recovery_prepared():
    from search.candidate import CandidateGenotype
    from search.canonicalization import SearchSpaceSpec, canonicalize_candidate
    from search.quantization_space.types import QuantizationSearchGroup

    reductions = {"pg::a": 0.70, "pg::b": 0.25, "pg::c": 0.20}
    risks = {"pg::a": 0.001, "pg::b": 0.10, "pg::c": 0.10}
    groups = tuple(
        QuantizationSearchGroup(
            group_id=locus,
            module_paths=(locus,),
            canonical_node_ids=(locus,),
            allowed_precisions=("FP32", "FP16"),
            protected=False,
            protection_reason="",
            ordering=index,
            parameter_count=1,
            baseline_macs=1.0,
        )
        for index, locus in enumerate(reductions)
    )
    space = SearchSpaceSpec(
        pruning_unit_ids=[],
        precision_layer_ids=list(reductions),
        pruning_domains=(),
        quantization_groups=groups,
        default_precision="FP32",
    )
    baseline = CandidateGenotype(
        precision_genes={locus: "FP32" for locus in reductions},
        meta={"created_by": "test", "repair_count": 0},
    )
    bops = _AuditMetricProxy(reductions)
    weight = _AuditMetricProxy(reductions, risks)

    def evaluator(*, target, enforce_bops_hard_gate):
        def evaluate(candidate):
            phenotype = canonicalize_candidate(candidate, space)
            metrics = bops.evaluate_breakdown(phenotype)
            deviation = abs(metrics["R_bops_vs_fp32"] - target)
            return {
                **metrics,
                "target": target,
                "bops_deviation": deviation,
                "bops_feasible": deviation <= 0.005,
                "bops_hard_gate_passed": deviation <= 0.005,
                "J_total": 0.0,
                "structural_repair_count": 0,
                "precision_repair_count": 0,
                "budget_repair_count": 0,
            }
        return evaluate

    return SimpleNamespace(
        space=space,
        baseline=baseline,
        bops=bops,
        size=bops,
        structure=_ZeroActionProxy(),
        weight=weight,
        activation=_ZeroActionProxy(),
        evaluator=evaluator,
    )


def test_cnn_greedy_restores_legal_frontier_and_finite_beam(
    tmp_path: Path,
) -> None:
    from search.ga.cnn_stage12_v3 import greedy_anchors

    anchors = greedy_anchors(
        _frontier_recovery_prepared(),
        targets=(0.75, 0.55),
        output_root=tmp_path,
        recovery_beam_width=2,
        recovery_seed_pool_size=4,
        recovery_max_depth=4,
    )

    assert set(anchors) == {0.75, 0.55}
    assert anchors[0.75].precision_genes["pg::b"] == "FP16"
    assert anchors[0.55].precision_genes["pg::b"] == "FP16"
    assert anchors[0.55].precision_genes["pg::c"] == "FP16"
    winners = json.loads(
        (tmp_path / "reports/greedy_exact_winners.json").read_text()
    )
    assert "primary_evaluated_neighbor_frontier" in winners["0.75"][
        "capture"
    ]["capture_sources"]
    assert winners["0.75"]["capture"]["selected_primary_path"] is False
    assert winners["0.55"]["capture"]["capture_source"].startswith(
        "target_directed_beam_recovery"
    )
    assert winners["0.75"]["metrics"]["target"] == 0.75
    assert winners["0.55"]["metrics"]["target"] == 0.55
    assert winners["0.75"]["metrics"]["bops_hard_gate_passed"] is True
    assert winners["0.55"]["metrics"]["bops_hard_gate_passed"] is True
    audit = json.loads(
        (tmp_path / "reports/greedy_search_audit.json").read_text()
    )["activation_taylor_enabled"]
    assert audit["primary_selected_step_count"] == 1
    assert audit["recovery_iteration_count"] == 1
    assert audit["total_search_iteration_count"] == 2
    assert audit["structure_repair_count"] == 0
    assert audit["precision_repair_count"] == 0
    assert audit["budget_repair_count"] == 0


def test_cobevt_fixed500_reuses_hash_bound_static_precision_acceptance(
    tmp_path: Path,
) -> None:
    from search.candidate import CandidateGenotype
    from search.ga.cnn_stage12_v3 import (
        GENERATION_WINNER_VALIDATION_SCHEMA,
        validate_generation_winner,
    )
    from search.ga.stage12_v3 import Stage2Result

    candidate = CandidateGenotype(
        pruning_width_genes={"backbone::out": 12},
        precision_genes={"conv": "FP32"},
        meta={"created_by": "test", "repair_count": 0},
    )
    complete_hash = "a" * 64
    source = tmp_path / "stage2"
    engine = source / "deployment/candidate.plan"
    engine.parent.mkdir(parents=True)
    engine.write_bytes(b"cobevt-engine")
    engine_hash = hashlib.sha256(engine.read_bytes()).hexdigest()
    (source / "candidate_stage2_result.json").write_text(json.dumps({
        "candidate_hash": complete_hash,
        "status": "ok",
        "physical_acceptance": True,
        "engine_acceptance": True,
        "qdq_acceptance": True,
        "precision_acceptance": True,
        "merge_acceptance": True,
        "transformer_attention_fp32_acceptance": True,
        "transformer_functional_precision_acceptance": True,
        "requested_int8_count": 1,
        "realized_int8_count": 1,
        "engine_path": str(engine),
        "engine_sha256": engine_hash,
    }))
    (source / "strict_stage2_result.json").write_text(json.dumps({
        "complete_phenotype_hash": complete_hash,
        "status": "ok",
        "requested_realized_exact": True,
    }))
    (source / "deployment/functional_precision_trt_audit.json").write_text(
        json.dumps({
            "passed": True,
            "conflict_count": 0,
            "unmapped_count": 0,
            "fallback_count": 0,
        })
    )
    screening = Stage2Result(
        complete_phenotype_hash=complete_hash,
        genotype=candidate,
        status="ok",
        map=0.64,
        p50_ms=5.2,
        requested_realized_exact=True,
        evaluated=300,
        skipped=0,
        metadata={
            "artifact_dir": str(source),
            "engine_hash": engine_hash,
            "precision_fallback": False,
        },
    )

    class MetricsOnlyEvaluator:
        def reevaluate_existing_candidate_engine(self, *_args, **_kwargs):
            # This is the real CoBEVT fixed500 payload shape: evaluation
            # metrics only, without the immutable Stage-2 precision fields.
            return {
                "status": "ok",
                "mAP": 0.645,
                "forward_p50_ms": 5.0,
                "num_evaluated_frames": 500,
                "num_skipped_frames": 0,
            }

    prepared = SimpleNamespace(
        space=_space(),
        spec=SimpleNamespace(model_id="cobevt"),
    )
    result = validate_generation_winner(
        prepared,
        screening_result=screening,
        output_root=tmp_path / "run",
        budget_label="030",
        generation=0,
        validation_evaluator=MetricsOnlyEvaluator(),
    )

    assert result.deployable
    assert result.requested_realized_exact is True
    assert result.map == 0.645
    assert result.evaluated == 500
    assert result.skipped == 0
    assert result.metadata["validation_contract_schema"] == (
        GENERATION_WINNER_VALIDATION_SCHEMA
    )
    assert result.metadata["static_deployment_acceptance"] == {
        "passed": True,
        "issues": [],
        "candidate_hash": complete_hash,
        "engine_path": str(engine),
        "engine_sha256": engine_hash,
        "requested_int8_count": 1,
        "realized_int8_count": 1,
        "conflict_count": 0,
        "unmapped_count": 0,
        "fallback_count": 0,
    }


def test_cobevt_fixed500_fails_closed_on_engine_hash_mismatch(
    tmp_path: Path,
) -> None:
    from search.ga.cnn_stage12_v3 import _cobevt_static_stage2_acceptance
    from search.ga.stage12_v3 import Stage2Result
    from search.candidate import CandidateGenotype

    candidate = CandidateGenotype(
        pruning_width_genes={"backbone::out": 12},
        precision_genes={"conv": "FP32"},
    )
    complete_hash = "b" * 64
    source = tmp_path / "stage2"
    engine = source / "deployment/candidate.plan"
    engine.parent.mkdir(parents=True)
    engine.write_bytes(b"actual-engine")
    bad_hash = "0" * 64
    (source / "candidate_stage2_result.json").write_text(json.dumps({
        "candidate_hash": complete_hash,
        "status": "ok",
        "physical_acceptance": True,
        "engine_acceptance": True,
        "qdq_acceptance": True,
        "precision_acceptance": True,
        "merge_acceptance": True,
        "transformer_attention_fp32_acceptance": True,
        "transformer_functional_precision_acceptance": True,
        "requested_int8_count": 0,
        "realized_int8_count": 0,
        "engine_path": str(engine),
        "engine_sha256": bad_hash,
    }))
    (source / "strict_stage2_result.json").write_text(json.dumps({
        "complete_phenotype_hash": complete_hash,
    }))
    (source / "deployment/functional_precision_trt_audit.json").write_text(
        json.dumps({
            "passed": True,
            "conflict_count": 0,
            "unmapped_count": 0,
            "fallback_count": 0,
        })
    )
    screening = Stage2Result(
        complete_hash,
        candidate,
        "ok",
        0.6,
        5.0,
        True,
        300,
        0,
        {"artifact_dir": str(source), "engine_hash": bad_hash},
    )

    audit = _cobevt_static_stage2_acceptance(
        screening_result=screening,
        source=source,
    )
    assert audit["passed"] is False
    assert "engine_sha256_mismatch" in audit["issues"]


def test_cnn_greedy_frontier_and_recovery_are_deterministic(
    tmp_path: Path,
) -> None:
    from search.ga.cnn_stage12_v3 import greedy_anchors

    first = tmp_path / "first"
    second = tmp_path / "second"
    greedy_anchors(
        _frontier_recovery_prepared(),
        targets=(0.75, 0.55),
        output_root=first,
        recovery_beam_width=2,
        recovery_seed_pool_size=4,
    )
    greedy_anchors(
        _frontier_recovery_prepared(),
        targets=(0.75, 0.55),
        output_root=second,
        recovery_beam_width=2,
        recovery_seed_pool_size=4,
    )
    assert (first / "reports/greedy_exact_winners.json").read_text() == (
        second / "reports/greedy_exact_winners.json"
    ).read_text()
