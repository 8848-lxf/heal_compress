from __future__ import annotations

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
    from search.ga.strict_stage12_v3 import StrictGAConfig

    assert StrictGAConfig(0.1).generations == 10
    configured = StrictGAConfig(
        0.1, generations=5, generation_contract="formal_gen5"
    )
    assert configured.generations == 5
    with pytest.raises(ValueError, match="generations_contract_mismatch"):
        StrictGAConfig(0.1, generations=4, generation_contract="formal_gen5")


def test_cnn_baseline_contains_only_mutable_loci() -> None:
    from search.ga.cnn_stage12_v3 import baseline_genotype
    from search.ga.strict_stage12_v3 import validate_genotype_schema

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


def test_cnn_formal_entrypoint_supports_fresh_ten_and_pyramid_replay() -> None:
    source = (__import__("pathlib").Path(__file__).parents[1]
              / "scripts/run_cnn_formal_ga_gen5.py").read_text()
    assert "requires_5_or_10_generations" in source
    assert "continuation_mode = bool(generations == 10 and args.resume)" in source
    assert "cnn_gen5_replay_continuation_is_pyramid_only" in source
    assert "freeze_gen5_continuation_state" in source
    assert "verify_gen5_replay_prefix" in source
    assert "single_seed_zero_required" in source
    assert '"StrictStage12V3Runner"' in source
    assert "full1789_executed" in source
    assert "all_requested_greedy_exact_anchors_in_band" in source
    assert "greedy_only" in source
    assert "formal_ga_start_authorized_by_gate" in source
    import scripts.run_cnn_formal_ga_gen5 as runner
    assert callable(runner.stage2_payload)
    adapter_source = (__import__("pathlib").Path(__file__).parents[1]
                      / "search/ga/cnn_stage12_v3.py").read_text()
    assert "cnn-formal-presearch-proxy-cache-v1" in adapter_source
    assert "physical_ranking_frozen_across_resume" in adapter_source


def test_pyramid_gen5_snapshot_and_replay_verification_are_fail_closed(
    tmp_path: Path,
) -> None:
    from scripts.run_cnn_formal_ga_gen5 import (
        freeze_gen5_continuation_state,
        verify_gen5_replay_prefix,
    )

    labels = ("030", "025", "020", "015", "010", "005")
    (tmp_path / "reports").mkdir()
    (tmp_path / "provenance").mkdir()
    (tmp_path / "provenance/start.json").write_text("{}")
    completed = {
        label: {"completed_evolution_generations": 5} for label in labels
    }
    (tmp_path / "reports/formal_ga_results.json").write_text(
        json.dumps({"formal_generations": 5, "results": completed})
    )
    (tmp_path / "reports/formal_ga_budget_summary.csv").write_text("budget\n")
    (tmp_path / "reports/final_acceptance.json").write_text("{}")
    for label in labels:
        seed = tmp_path / f"ga/budget_{label}/seed_0"
        seed.mkdir(parents=True)
        (seed / "budget_summary.json").write_text("{}")
        for generation in range(6):
            destination = seed / f"generation_{generation:02d}"
            destination.mkdir()
            (destination / "generation_summary.json").write_text(
                json.dumps({"generation": generation, "budget": label})
            )

    snapshot = freeze_gen5_continuation_state(tmp_path)
    assert snapshot["source_formal_generations"] == 5
    assert snapshot["new_generations"] == [6, 7, 8, 9, 10]
    audit = verify_gen5_replay_prefix(tmp_path, labels)
    assert audit["all_generation_00_to_05_summaries_exact"] is True

    changed = tmp_path / "ga/budget_030/seed_0/generation_05/generation_summary.json"
    changed.write_text('{"generation": 5, "drift": true}')
    with pytest.raises(RuntimeError, match="replay_prefix_mismatch"):
        verify_gen5_replay_prefix(tmp_path, labels)


def test_cnn_two_tier_real_evaluation_protocol_is_frozen() -> None:
    from search.ga.cnn_stage12_v3 import (
        EVALUATION_MANIFEST_FRAMES,
        EVALUATION_MANIFEST_WARMUP_FRAMES,
        GENERATION_WINNER_FRAMES,
        GENERATION_WINNER_PROTOCOL,
        GENERATION_WINNER_WARMUP_FRAMES,
        STAGE2_SCREENING_FRAMES,
        STAGE2_SCREENING_PROTOCOL,
        STAGE2_SCREENING_WARMUP_FRAMES,
    )

    assert STAGE2_SCREENING_FRAMES == 300
    assert STAGE2_SCREENING_WARMUP_FRAMES == 100
    assert STAGE2_SCREENING_PROTOCOL == "top5_fixed300_warmup100_screening"
    assert GENERATION_WINNER_FRAMES == 500
    assert GENERATION_WINNER_WARMUP_FRAMES == 200
    assert GENERATION_WINNER_PROTOCOL == "generation_winner_fixed500_warmup200"
    assert EVALUATION_MANIFEST_FRAMES == 500
    assert EVALUATION_MANIFEST_WARMUP_FRAMES == 200


def test_physical_gpu_is_mapped_to_process_local_cuda_ordinal(monkeypatch) -> None:
    from search.ga.cnn_stage12_v3 import logical_cuda_device_index

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    assert logical_cuda_device_index(3) == 0
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    assert logical_cuda_device_index(3) == 1
    with pytest.raises(RuntimeError, match="physical_gpu_not_visible"):
        logical_cuda_device_index(4)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    assert logical_cuda_device_index(3) == 3


def test_context_gpu_selection_recovers_isolated_physical_gpu(monkeypatch) -> None:
    from search.integration import runtime_environment

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "6")
    monkeypatch.setattr(
        runtime_environment,
        "query_gpus",
        lambda: [
            {
                "index": index,
                "memory_total_mib": 81920,
                "memory_used_mib": 0,
                "memory_free_mib": 81920,
                "utilization_gpu_pct": 0,
            }
            for index in range(8)
        ],
    )

    selection = runtime_environment.select_gpu("0", exclude_gpu_ids=[])

    assert selection.runtime_device == "cuda:0"
    assert selection.physical_gpu_id == 6


def test_context_gpu_selection_maps_physical_gpu_to_visible_ordinal(monkeypatch) -> None:
    from search.integration import runtime_environment

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,6")
    monkeypatch.setattr(
        runtime_environment,
        "query_gpus",
        lambda: [
            {
                "index": index,
                "memory_total_mib": 81920,
                "memory_used_mib": 0,
                "memory_free_mib": 81920,
                "utilization_gpu_pct": 0,
            }
            for index in range(8)
        ],
    )

    selection = runtime_environment.select_gpu("6", exclude_gpu_ids=[])

    assert selection.runtime_device == "cuda:1"
    assert selection.physical_gpu_id == 6


def test_size_proxy_field_is_mapped_to_strict_stage1_contract() -> None:
    from search.ga.cnn_stage12_v3 import canonical_size_metrics

    row = canonical_size_metrics({
        "R_size_vs_fp32": 0.375,
        "R_parameter_retention": 0.75,
    })
    assert row["mixed_weight_retention"] == 0.375
    assert row["R_parameter_retention"] == 0.75


def test_grouped_gate_rerank_rebuilds_physical_keep_map() -> None:
    from search.proxy.conservative_gate_activation_taylor import (
        GateDomainScores,
        rerank_domains_by_gate_scores,
    )
    from search.pruning_space.local_domains import LocalPruningDomain

    domain = LocalPruningDomain(
        domain_id="grouped::out",
        root_module_path="grouped",
        root_axis="out",
        scope_id="grouped-scope",
        kind="regular_grouped",
        original_width=4,
        total_original_width=8,
        ordered_unit_ids=("g0u0", "g0u1", "g0u2", "g0u3", "g1u0", "g1u1", "g1u2", "g1u3"),
        ordered_unit_ids_by_group={
            0: ("g0u0", "g0u1", "g0u2", "g0u3"),
            1: ("g1u0", "g1u1", "g1u2", "g1u3"),
        },
        group_local_indices={
            0: {f"g0u{i}": i for i in range(4)},
            1: {f"g1u{i}": i for i in range(4)},
        },
        unit_root_indices={
            **{f"g0u{i}": (i,) for i in range(4)},
            **{f"g1u{i}": (4 + i,) for i in range(4)},
        },
        legal_widths=(2, 4),
        width_to_pruned_unit_ids={2: ("g0u0", "g0u1", "g1u0", "g1u1"), 4: ()},
        group_keep_maps={2: {0: [2, 3], 1: [2, 3]}, 4: {0: [0, 1, 2, 3], 1: [0, 1, 2, 3]}},
        group_prune_maps={2: {0: [0, 1], 1: [0, 1]}, 4: {0: [], 1: []}},
        groups=2,
    )
    scores = GateDomainScores(
        domain_id=domain.domain_id,
        # The lowest gate coordinates differ between groups.  A flat sort
        # would remove four coordinates from group 1 and none from group 0.
        unit_scores={
            "g0u0": 10.0, "g0u1": 11.0, "g0u2": 1.0, "g0u3": 2.0,
            "g1u0": 3.0, "g1u1": 4.0, "g1u2": 20.0, "g1u3": 21.0,
        },
        semantic_root_tensor="grouped",
        gate_tensor="grouped",
        physical_dependencies=("grouped.weight",),
        family="grouped_conv",
    )

    reranked = rerank_domains_by_gate_scores((domain,), {domain.domain_id: scores})[0]
    assert reranked.width_to_pruned_unit_ids[2] == ("g0u2", "g0u3", "g1u0", "g1u1")
    assert reranked.group_prune_maps[2] == {0: [2, 3], 1: [0, 1]}
    assert reranked.group_keep_maps[2] == {0: [0, 1], 1: [2, 3]}
    expanded_keep = [
        group * domain.original_width + local
        for group in range(domain.groups)
        for local in reranked.group_keep_maps[2][group]
    ]
    selected = set(reranked.width_to_pruned_unit_ids[2])
    frozen_keep = sorted(
        index
        for unit_id, indices in reranked.unit_root_indices.items()
        if unit_id not in selected
        for index in indices
    )
    assert expanded_keep == [0, 1, 6, 7]
    assert frozen_keep == expanded_keep


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
        retention = 1.0 - sum(self.reductions.get(locus, 0.0) for locus in selected)
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
            return {
                **metrics,
                "target": target,
                "bops_feasible": (
                    abs(metrics["R_bops_vs_fp32"] - target) <= 0.005
                    or not enforce_bops_hard_gate
                ),
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


def test_cnn_greedy_restores_legal_frontier_and_finite_beam(tmp_path: Path) -> None:
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
        "capture_sources"
    ]
    assert winners["0.75"]["selected_primary_path"] is False
    assert winners["0.55"]["capture_source"].startswith(
        "target_directed_beam_recovery"
    )
    audit = json.loads((tmp_path / "reports/greedy_search_audit.json").read_text())
    assert audit["primary_selected_step_count"] == 1
    assert audit["recovery_iteration_count"] == 1
    assert audit["total_search_iteration_count"] == 2
    assert audit["structure_repair_count"] == 0
    assert audit["precision_repair_count"] == 0
    assert audit["budget_repair_count"] == 0


def test_cnn_greedy_frontier_and_recovery_are_deterministic(tmp_path: Path) -> None:
    from search.ga.cnn_stage12_v3 import greedy_anchors

    first = tmp_path / "first"
    second = tmp_path / "second"
    greedy_anchors(
        _frontier_recovery_prepared(), targets=(0.75, 0.55), output_root=first,
        recovery_beam_width=2, recovery_seed_pool_size=4,
    )
    greedy_anchors(
        _frontier_recovery_prepared(), targets=(0.75, 0.55), output_root=second,
        recovery_beam_width=2, recovery_seed_pool_size=4,
    )
    assert (first / "reports/greedy_exact_winners.json").read_text() == (
        second / "reports/greedy_exact_winners.json"
    ).read_text()


def test_completed_exact_greedy_anchors_resume_with_strict_validation(
    tmp_path: Path,
) -> None:
    from search.ga.cnn_stage12_v3 import greedy_anchors, load_greedy_anchors

    prepared = _frontier_recovery_prepared()
    original = greedy_anchors(
        prepared, targets=(0.75, 0.55), output_root=tmp_path,
        recovery_beam_width=2, recovery_seed_pool_size=4,
    )
    resumed = load_greedy_anchors(
        prepared, targets=(0.75, 0.55), output_root=tmp_path,
    )
    assert {
        target: candidate.to_dict() for target, candidate in resumed.items()
    } == {
        target: candidate.to_dict() for target, candidate in original.items()
    }
