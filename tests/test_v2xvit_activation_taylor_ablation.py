from __future__ import annotations


def _group(group_id: str):
    from search.quantization_space.types import QuantizationSearchGroup

    return QuantizationSearchGroup(
        group_id=group_id,
        module_paths=(f"module.{group_id}",),
        canonical_node_ids=(group_id,),
        allowed_precisions=("FP32", "FP16"),
        protected=False,
        protection_reason="",
        ordering=0,
        parameter_count=1,
        baseline_macs=1.0,
        metadata={"default_precision": "FP32"},
    )


def _space(*group_ids: str):
    from search.canonicalization import SearchSpaceSpec
    from search.pruning_space.local_domains import LocalPruningDomain

    fixed_domain = LocalPruningDomain(
        domain_id="fixed",
        root_module_path="fixed",
        root_axis="out",
        scope_id="fixed",
        kind="dense",
        original_width=4,
        total_original_width=4,
        ordered_unit_ids=("fixed:0",),
        legal_widths=(4,),
        width_to_pruned_unit_ids={4: ()},
        unit_root_indices={"fixed:0": (0,)},
        domain_type="cnn_channel",
    )

    return SearchSpaceSpec(
        pruning_unit_ids=[],
        precision_layer_ids=[],
        pruning_domains=(fixed_domain,),
        quantization_groups=tuple(_group(group_id) for group_id in group_ids),
        default_precision="FP32",
    )


class _StructureProxy:
    def pruning_action_breakdown(self, _before, _after):
        return {
            "delta_J_prune": 0.0,
            "first_order_abs_sum": 0.0,
            "second_order_abs_sum": 0.0,
            "newly_pruned_parameter_count": 0,
        }


class _WeightProxy:
    def weight_quantization_action_breakdown(self, _before, _after):
        return {
            "delta_J_prune": 0.0,
            "delta_J_WQ": 1.0,
            "first_order_abs_sum": 1.0,
            "second_order_abs_sum": 0.0,
        }


class _ActivationCache:
    def action_breakdown(self, _before, _after):
        return {"delta_J_AQ": 100.0}


def _bops(phenotype):
    lowered = sum(
        decision.realized_precision == "FP16"
        for decision in phenotype.precision_profile.values()
    )
    total = 100.0 - 10.0 * lowered
    return {"bops_total": total, "R_bops_vs_fp32": total / 100.0}


def _size(_phenotype):
    return {
        "size_bits_total": 320.0,
        "R_parameter_retention": 1.0,
        "R_size_vs_fp32": 1.0,
    }


def test_stage1_jaq_zero_preserves_raw_diagnostic_but_excludes_fitness() -> None:
    from search.candidate import CandidateGenotype
    from search.ga.stage12_v3 import UnifiedTaylorStage1Evaluator

    space = _space("projection")
    baseline = CandidateGenotype(precision_genes={"projection": "FP32"})
    candidate = CandidateGenotype(precision_genes={"projection": "FP16"})
    common = dict(
        baseline=baseline,
        structure_proxy=_StructureProxy(),
        weight_proxy=_WeightProxy(),
        activation_cache=_ActivationCache(),
        bops_evaluator=_bops,
        size_evaluator=_size,
        target=0.90,
    )

    joint = UnifiedTaylorStage1Evaluator(space, **common)(candidate)
    ablated = UnifiedTaylorStage1Evaluator(
        space, **common, activation_taylor_fitness_weight=0.0
    )(candidate)

    assert joint["J_WQ"] == 1.0
    assert joint["J_AQ"] == 100.0
    assert joint["J_total"] == 101.0
    assert ablated["J_AQ"] == 100.0
    assert ablated["J_AQ_fitness_contribution"] == 0.0
    assert ablated["J_total"] == 1.0
    assert ablated["activation_taylor_used_for_fitness"] is False


def test_greedy_jaq_zero_never_leaks_prior_raw_aq_into_cumulative_score() -> None:
    from search.greedy.weight_only_abs import run_weight_only_abs_greedy

    result = run_weight_only_abs_greedy(
        _space("a", "b"),
        weight_proxy=_WeightProxy(),
        structure_proxy=_StructureProxy(),
        activation_cache=_ActivationCache(),
        activation_taylor_weight=0.0,
        bops_evaluator=_bops,
        size_evaluator=_size,
        target=0.80,
        tolerance_abs=1.0e-12,
        maximum_steps=None,
    )
    selected = [row for row in result["trace"] if row["selected"]]

    assert len(selected) == 2
    assert selected[-1]["cumulative_proxy"] == 2.0
    assert selected[-1]["cumulative_activation_taylor"] == 200.0
    assert (
        selected[-1]["cumulative_activation_taylor_fitness_contribution"]
        == 0.0
    )
    assert result["activation_taylor_used_for_fitness"] is False


def test_negative_activation_taylor_weight_fails_closed() -> None:
    import pytest

    from search.candidate import CandidateGenotype
    from search.ga.stage12_v3 import UnifiedTaylorStage1Evaluator

    space = _space("projection")
    with pytest.raises(ValueError, match="activation_taylor_fitness_weight_invalid"):
        UnifiedTaylorStage1Evaluator(
            space,
            baseline=CandidateGenotype(
                precision_genes={"projection": "FP32"}
            ),
            structure_proxy=_StructureProxy(),
            weight_proxy=_WeightProxy(),
            activation_cache=_ActivationCache(),
            bops_evaluator=_bops,
            size_evaluator=_size,
            target=0.90,
            activation_taylor_fitness_weight=-1.0,
        )
