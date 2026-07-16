from __future__ import annotations


def _unit(index: int):
    from pruning.types import AtomicPruneUnit

    return AtomicPruneUnit(
        "scope",
        "backbone.conv",
        "out",
        [index],
        [f"coupled_{index}"],
        float(index),
    )


def _space():
    from search.canonicalization import SearchSpaceSpec
    from search.pruning_space.local_domains import build_local_pruning_domains
    from search.quantization_space.types import QuantizationSearchGroup

    units = [_unit(index) for index in range(8)]
    domain = build_local_pruning_domains(
        units,
        importance_scores={unit.stable_id: float(index) for index, unit in enumerate(units)},
    )[0]
    group = QuantizationSearchGroup(
        group_id="pg::conv",
        module_paths=("backbone.conv",),
        canonical_node_ids=("conv",),
        allowed_precisions=("FP32", "FP16", "INT8"),
        protected=False,
        protection_reason="",
        ordering=0,
        parameter_count=16,
        baseline_macs=100.0,
    )
    return SearchSpaceSpec(
        pruning_unit_ids=[unit.stable_id for unit in units],
        precision_layer_ids=["backbone.conv"],
        quantization_groups=(group,),
        pruning_domains=(domain,),
        default_precision="FP32",
    )


def test_greedy_search_recomputes_neighbors_and_captures_each_budget_once() -> None:
    from search.greedy import GreedyBudgetSearch, GreedySearchConfig

    space = _space()
    calls: list[tuple[int, int]] = []

    def evaluate(candidates, step):
        calls.append((step, len(candidates)))
        rows = []
        for candidate in candidates:
            width = next(iter(candidate.pruning_width_genes.values()))
            precision = next(iter(candidate.precision_genes.values()))
            bits = {"FP32": 32, "FP16": 16, "INT8": 8}[precision]
            r_bops = (width / 8.0) * (bits / 32.0) ** 2
            prune_loss = (8 - width) * 0.04
            quant_loss = {"FP32": 0.0, "FP16": 0.01, "INT8": 0.08}[precision]
            rows.append(
                {
                    "R_bops_vs_fp32": r_bops,
                    "R_size_vs_fp32": (width / 8.0) * (bits / 32.0),
                    "L_joint_weight_taylor": prune_loss + quant_loss,
                }
            )
        return rows

    result = GreedyBudgetSearch(
        space,
        config=GreedySearchConfig(bops_targets=(0.50, 0.25, 0.125)),
    ).run(evaluate)

    assert result.unreachable_targets == ()
    assert set(result.budget_candidates) == {0.50, 0.25, 0.125}
    assert all(
        result.budget_metrics[target]["R_bops_vs_fp32"] <= target
        for target in result.budget_candidates
    )
    assert result.termination_reason == "minimum_target_reached"
    assert result.evaluated_neighbor_count > len(result.steps)
    assert len({
        (
            tuple(sorted(candidate.pruning_width_genes.items())),
            tuple(sorted(candidate.precision_genes.items())),
        )
        for candidate in result.budget_candidates.values()
    }) <= len(result.budget_candidates)
    assert any(count > 1 for _step, count in calls)


def test_greedy_search_respects_protected_precision_group() -> None:
    from dataclasses import replace

    from search.greedy import GreedyBudgetSearch, GreedySearchConfig
    from search.quantization_space.types import QuantizationSearchGroup

    space = _space()
    protected = QuantizationSearchGroup(
        group_id="pg::conv",
        module_paths=("backbone.conv",),
        canonical_node_ids=("conv",),
        allowed_precisions=("FP16",),
        protected=True,
        protection_reason="test",
        ordering=0,
        parameter_count=16,
        baseline_macs=100.0,
    )
    space = replace(space, quantization_groups=(protected,))

    def evaluate(candidates, _step):
        return [
            {
                "R_bops_vs_fp32": next(iter(row.pruning_width_genes.values())) / 32.0,
                "R_size_vs_fp32": 1.0,
                "L_joint_weight_taylor": 0.1,
            }
            for row in candidates
        ]

    result = GreedyBudgetSearch(
        space,
        config=GreedySearchConfig(bops_targets=(0.125,)),
    ).run(evaluate)

    assert result.initial_candidate.precision_genes == {"pg::conv": "FP16"}
    assert all(step.action_kind != "precision" for step in result.steps)
