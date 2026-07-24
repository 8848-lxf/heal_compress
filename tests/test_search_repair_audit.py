from __future__ import annotations

import random

import pytest


def _domain(domain_id: str, domain_type: str, original: int, legal: tuple[int, ...]):
    from search.pruning_space.local_domains import LocalPruningDomain

    rankings = (
        {
            "qk_low_to_high_by_head": (tuple(range(original)),),
            "vo_low_to_high_by_head": (tuple(range(original)),),
        }
        if domain_type == "attention_dh"
        else (
            {"ffn_low_to_high": tuple(range(original))}
            if domain_type == "ffn_hidden"
            else {}
        )
    )
    return LocalPruningDomain(
        domain_id=domain_id,
        root_module_path=domain_id,
        root_axis="out",
        scope_id=domain_id,
        kind="transformer" if domain_type.startswith(("attention", "ffn")) else "dense",
        domain_type=domain_type,
        original_width=original,
        total_original_width=original,
        ordered_unit_ids=(),
        legal_widths=legal,
        width_to_pruned_unit_ids={width: () for width in legal},
        ranking_groups=rankings,
        constraints={"heads": 1} if domain_type == "attention_dh" else {},
    )


def _precision_group(
    group_id: str,
    *,
    protected: bool = False,
    allowed: tuple[str, ...] = ("FP32", "FP16", "INT8"),
    default: str = "FP32",
):
    from search.quantization_space.types import QuantizationSearchGroup

    return QuantizationSearchGroup(
        group_id=group_id,
        module_paths=(f"module.{group_id}",),
        canonical_node_ids=(group_id,),
        allowed_precisions=allowed,
        protected=protected,
        protection_reason="fixed_contract" if protected else "",
        ordering=0,
        parameter_count=1,
        baseline_macs=1.0,
        metadata={"default_precision": default},
    )


def _space(*, protected_qk: bool = False):
    from search.canonicalization import SearchSpaceSpec

    groups = [_precision_group("projection")]
    if protected_qk:
        groups.append(
            _precision_group(
                "qk_matmul",
                protected=True,
                allowed=("FP32",),
                default="FP32",
            )
        )
    return SearchSpaceSpec(
        pruning_unit_ids=[],
        precision_layer_ids=[],
        quantization_groups=tuple(groups),
        pruning_domains=(
            _domain("attention", "attention_dh", 8, (4, 8)),
            _domain("ffn", "ffn_hidden", 12, (6, 12)),
        ),
        default_precision="FP32",
    )


def test_canonicalization_is_expression_only_and_not_repair() -> None:
    from search.candidate import CandidateGenotype
    from search.repair_audit import canonicalize_genotype_with_audit

    canonical, audit = canonicalize_genotype_with_audit(
        CandidateGenotype(), _space(protected_qk=True)
    )
    assert canonical.pruning_width_genes == {"attention": 8, "ffn": 12}
    assert canonical.precision_genes == {"projection": "FP32"}
    assert canonical.meta["constant_precision_group_profile"] == {
        "qk_matmul": "FP32"
    }
    assert audit.canonicalization_count == 4
    assert audit.structural_repair_count == 0
    assert audit.precision_repair_count == 0
    assert audit.phenotype_changed_by_repair is False


@pytest.mark.parametrize(
    ("field", "value"), (("attention", 6), ("ffn", 8))
)
def test_arbitrary_transformer_width_cannot_pass_strict_genotype_validation(
    field: str, value: int
) -> None:
    from search.candidate import CandidateGenotype
    from search.canonicalization import repair_genotype

    with pytest.raises(ValueError, match="illegal_domain_width_gene"):
        repair_genotype(
            CandidateGenotype(pruning_width_genes={field: value}), _space()
        )


def test_qk_is_not_a_variable_precision_locus_and_explicit_gene_fails() -> None:
    from search.candidate import CandidateGenotype
    from search.canonicalization import repair_genotype

    space = _space(protected_qk=True)
    assert space.precision_gene_ids == ["projection"]
    with pytest.raises(
        ValueError, match="non_variable_precision_genes_must_not_enter_genotype"
    ):
        repair_genotype(
            CandidateGenotype(precision_genes={"qk_matmul": "INT8"}), space
        )


def test_crossover_copies_same_locus_never_absolute_width_across_domains() -> None:
    from search.candidate import CandidateGenotype
    from search.ga.crossover import block_crossover

    space = _space()
    left = CandidateGenotype(
        pruning_width_genes={"attention": 8, "ffn": 12},
        precision_genes={"projection": "FP32"},
    )
    right = CandidateGenotype(
        pruning_width_genes={"attention": 4, "ffn": 6},
        precision_genes={"projection": "INT8"},
    )
    for seed in range(32):
        child = block_crossover(left, right, space, random.Random(seed))
        assert child.pruning_width_genes["attention"] in {4, 8}
        assert child.pruning_width_genes["ffn"] in {6, 12}
        assert child.pruning_width_genes["attention"] != 6
        assert child.pruning_width_genes["ffn"] != 4


def test_greedy_runs_to_exhaustion_without_any_repair_or_budget_projection() -> None:
    from search.canonicalization import SearchSpaceSpec
    from search.greedy import GreedyBudgetSearch, GreedySearchConfig

    space = SearchSpaceSpec(
        pruning_unit_ids=[],
        precision_layer_ids=[],
        quantization_groups=(_precision_group("projection"),),
        pruning_domains=(_domain("cnn", "cnn_channel", 8, (2, 4, 8)),),
        default_precision="FP32",
    )

    def evaluate(candidates, _step):
        rows = []
        for candidate in candidates:
            width = candidate.pruning_width_genes["cnn"]
            precision = candidate.precision_genes["projection"]
            bits = {"FP32": 32, "FP16": 16, "INT8": 8}[precision]
            rows.append(
                {
                    "R_bops_vs_fp32": width / 8.0 * (bits / 32.0) ** 2,
                    "L_joint_weight_activation_taylor": (8 - width) + (32 - bits),
                    "R_parameter_retention": width / 8.0,
                    "mixed_weight_size_bytes": width * bits,
                }
            )
        return rows

    result = GreedyBudgetSearch(
        space,
        config=GreedySearchConfig(
            bops_targets=(0.05,),
            run_to_exhaustion=True,
            enable_budget_recovery=False,
        ),
    ).run(evaluate)
    assert result.termination_reason == "no_remaining_legal_action"
    assert len(result.steps) == 4
    assert result.budget_recovery_evaluated_neighbor_count == 0
    semantics = result.to_dict()["search_semantics"]
    assert semantics["actual_repair"] == {
        "structural": 0,
        "precision": 0,
        "budget_projection": 0,
        "canonicalization_is_not_repair": True,
    }


def test_stage2_repair_path_never_reuses_old_fitness_or_hash() -> None:
    from search.candidate import CandidateGenotype
    from search.canonicalization import canonicalize_candidate
    from search.hashing import candidate_hash
    from search.stage1.repair_selection import select_repaired_stage2_topk

    space = _space()
    raw = CandidateGenotype(
        pruning_width_genes={"attention": 8, "ffn": 12},
        precision_genes={"projection": "FP32"},
    )
    changed = CandidateGenotype(
        pruning_width_genes={"attention": 4, "ffn": 12},
        precision_genes={"projection": "FP32"},
    )
    expected_hash = candidate_hash(canonicalize_candidate(changed, space), space)
    rescore_calls = []

    selected, _report = select_repaired_stage2_topk(
        [(raw, 1.0, {"F1": 1.0, "bops_feasible": True})],
        space=space,
        repair_fn=lambda _candidate: (
            changed,
            {"status": "ok", "actual_structural_repair_count": 1},
        ),
        rescore_fn=lambda phenotype: rescore_calls.append(phenotype)
        or {"F1": 7.0, "bops_feasible": True},
        topk=1,
    )
    assert len(rescore_calls) == 1
    assert selected[0].F1 == 7.0
    assert selected[0].metrics["raw_F1"] == 1.0
    assert selected[0].candidate_hash == expected_hash


def test_hard_gate_and_dedup_filter_without_modifying_candidate() -> None:
    from search.candidate import CandidateGenotype
    from search.stage1.repair_selection import select_repaired_stage2_topk

    space = _space()
    candidate = CandidateGenotype(
        pruning_width_genes={"attention": 8, "ffn": 12},
        precision_genes={"projection": "FP32"},
    )
    before = candidate.to_dict()
    selected, report = select_repaired_stage2_topk(
        [
            (candidate, 1.0, {"F1": 1.0, "bops_feasible": True}),
            (candidate, 2.0, {"F1": 2.0, "bops_feasible": True}),
        ],
        space=space,
        repair_fn=lambda row: (row, {"status": "ok"}),
        rescore_fn=lambda _phenotype: {"F1": 3.0, "bops_feasible": False},
        eligibility_fn=lambda metrics: bool(metrics["bops_feasible"]),
        topk=1,
    )
    assert candidate.to_dict() == before
    assert selected == []
    assert report["duplicate_repaired_phenotype_count"] == 1
    assert report["rejected_after_repaired_rescore"] == 1
    assert report["repair_failed_count"] == 0
