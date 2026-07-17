from __future__ import annotations

import random

import pytest


def _unit(scope: str, root: str, index: int, score: float, *, grouped: bool = False):
    from pruning.types import AtomicPruneUnit

    constraints = (
        {
            "grouped_conv": True,
            "depthwise": False,
            "groups": 2,
            "channels_per_group": 8,
        }
        if grouped
        else {}
    )
    return AtomicPruneUnit(
        scope,
        root,
        "out",
        [index],
        [f"coupled_{index}"],
        score,
        constraints=constraints,
    )


def test_dense_domain_width_expands_fixed_low_taylor_prefix_without_repair() -> None:
    from search.candidate import CandidateGenotype
    from search.canonicalization import SearchSpaceSpec, canonicalize_candidate
    from search.pruning_space.local_domains import build_local_pruning_domains

    units = [_unit("scope", "backbone.conv", index, float(8 - index)) for index in range(8)]
    scores = {unit.stable_id: float(index) for index, unit in enumerate(units)}
    domains = build_local_pruning_domains(units, importance_scores=scores)
    assert len(domains) == 1
    domain = domains[0]
    assert domain.legal_widths == (4, 8)
    assert domain.ordered_unit_ids == tuple(unit.stable_id for unit in units)

    space = SearchSpaceSpec(
        pruning_unit_ids=[unit.stable_id for unit in units],
        precision_layer_ids=[],
        pruning_domains=tuple(domains),
        pruning_policy_version="legal-domain-width-fixed-ranking-v1",
    )
    phenotype = canonicalize_candidate(
        CandidateGenotype(
            pruning_width_genes={domain.domain_id: 4},
        ),
        space,
    )

    assert phenotype.pruned_unit_ids == sorted(unit.stable_id for unit in units[:4])
    assert phenotype.metadata["domain_width_profile"] == {domain.domain_id: 4}
    assert phenotype.metadata["domains"][domain.domain_id]["alignment_repair_applied"] is False
    assert phenotype.metadata["repair_version"] == "domain-width-legal-by-construction-v1"


def test_regular_grouped_width_gene_prunes_equal_fixed_prefix_per_group() -> None:
    from search.pruning_space.local_domains import build_local_pruning_domains, expand_domain_width_genes

    units = [_unit("grouped_scope", "backbone.grouped", index, float(index), grouped=True) for index in range(16)]
    scores = {
        unit.stable_id: float(index % 8) + (0.01 if index >= 8 else 0.0)
        for index, unit in enumerate(units)
    }
    domain = build_local_pruning_domains(units, importance_scores=scores)[0]

    assert domain.kind == "regular_grouped"
    assert domain.legal_widths == (4, 8)
    pruned, metadata = expand_domain_width_genes({domain.domain_id: 4}, [domain])
    assert set(pruned) == {unit.stable_id for unit in [*units[:4], *units[8:12]]}
    assert metadata["group_keep_map_by_scope"]["grouped_scope"] == {
        0: [4, 5, 6, 7],
        1: [4, 5, 6, 7],
    }
    assert metadata["group_prune_map_by_scope"]["grouped_scope"] == {
        0: [0, 1, 2, 3],
        1: [0, 1, 2, 3],
    }


def test_illegal_domain_width_fails_instead_of_post_alignment_repair() -> None:
    from search.candidate import CandidateGenotype
    from search.canonicalization import SearchSpaceSpec, canonicalize_candidate
    from search.pruning_space.local_domains import build_local_pruning_domains

    units = [_unit("scope", "backbone.conv", index, float(index)) for index in range(8)]
    domains = build_local_pruning_domains(units)
    space = SearchSpaceSpec(
        pruning_unit_ids=[unit.stable_id for unit in units],
        precision_layer_ids=[],
        pruning_domains=tuple(domains),
    )

    with pytest.raises(ValueError, match="illegal_domain_width_gene"):
        canonicalize_candidate(
            CandidateGenotype(pruning_width_genes={domains[0].domain_id: 6}),
            space,
        )


def test_domain_width_ga_operators_preserve_legal_widths() -> None:
    from search.candidate import CandidateGenotype
    from search.canonicalization import SearchSpaceSpec, repair_genotype
    from search.ga.immigrants import random_immigrant
    from search.ga.mutation import mutate_candidate
    from search.pruning_space.local_domains import build_local_pruning_domains

    units = [_unit("scope", "backbone.conv", index, float(index)) for index in range(16)]
    domains = build_local_pruning_domains(units)
    space = SearchSpaceSpec(
        pruning_unit_ids=[unit.stable_id for unit in units],
        precision_layer_ids=["layer"],
        pruning_domains=tuple(domains),
    )
    rng = random.Random(7)
    candidate = random_immigrant(space, rng)
    mutated = mutate_candidate(
        repair_genotype(candidate, space),
        space,
        rng,
        prune_mutation_rate=1.0,
        precision_mutation_rate=0.0,
    )

    assert set(mutated.pruning_width_genes) == {domains[0].domain_id}
    assert mutated.pruning_width_genes[domains[0].domain_id] in domains[0].legal_widths
    assert mutated.pruning_genes == {}


def test_compact_domain_width_genotype_preserves_expanded_atomic_mask() -> None:
    from search.candidate import CandidateGenotype
    from search.canonicalization import (
        SearchSpaceSpec,
        canonicalize_candidate,
        repair_genotype,
    )
    from search.pruning_space.local_domains import build_local_pruning_domains

    units = [_unit("scope", "backbone.conv", index, float(index)) for index in range(16)]
    domain = build_local_pruning_domains(units)[0]
    space = SearchSpaceSpec(
        pruning_unit_ids=[unit.stable_id for unit in units],
        precision_layer_ids=[],
        pruning_domains=(domain,),
    )
    width_gene = {domain.domain_id: domain.legal_widths[0]}
    legacy_redundant = CandidateGenotype(
        pruning_genes={unit.stable_id: 1 for unit in units},
        pruning_width_genes=width_gene,
    )
    compact = CandidateGenotype(pruning_width_genes=width_gene)

    assert repair_genotype(legacy_redundant, space).pruning_genes == {}
    assert canonicalize_candidate(legacy_redundant, space).to_dict() == canonicalize_candidate(
        compact, space
    ).to_dict()


def test_candidate_width_genes_round_trip() -> None:
    from search.candidate import CandidateGenotype

    candidate = CandidateGenotype(
        pruning_genes={"atomic": 1},
        precision_genes={"pg": "INT8"},
        pruning_width_genes={"root::out": 12},
    )

    restored = CandidateGenotype.from_dict(candidate.to_dict())
    assert restored == candidate


def test_joint_weight_taylor_scores_combined_prune_and_per_channel_quantization_once() -> None:
    import torch

    from search.candidate import CandidatePhenotype, PrecisionDecision
    from search.proxy.fisher_proxy import FisherStatistics
    from search.proxy.joint_weight_taylor import JointWeightTaylorProxy
    from search.proxy.parameter_slice_resolver import ParameterSlice

    model = torch.nn.Sequential()
    model.add_module("linear", torch.nn.Linear(4, 3, bias=False))
    with torch.no_grad():
        model.linear.weight.copy_(
            torch.tensor(
                [
                    [0.11, -0.27, 0.39, -0.51],
                    [0.07, -0.19, 0.43, -0.62],
                    [0.13, -0.31, 0.47, -0.73],
                ]
            )
        )
    weight = model.linear.weight.detach()
    statistics = FisherStatistics(
        gradients={"linear.weight": torch.full_like(weight, 0.2)},
        fisher_diag={"linear.weight": torch.full_like(weight, 0.04)},
        manifest_hash="toy",
    )
    slices = {
        "u0": [ParameterSlice("linear.weight", "linear", 0, (0,), "prune_weight_slice")]
    }
    proxy = JointWeightTaylorProxy(
        model,
        statistics=statistics,
        unit_to_parameter_slices=slices,
    )
    fp32 = CandidatePhenotype(
        precision_profile={"linear": PrecisionDecision("FP32", "FP32")}
    )
    quantized = CandidatePhenotype(
        precision_profile={"linear": PrecisionDecision("INT8", "INT8")}
    )
    combined = CandidatePhenotype(
        pruned_unit_ids=["u0"],
        precision_profile={"linear": PrecisionDecision("INT8", "INT8")},
    )

    fp32_metrics = proxy.evaluate_breakdown(fp32)
    quant_metrics = proxy.evaluate_breakdown(quantized)
    combined_metrics = proxy.evaluate_breakdown(combined)

    assert fp32_metrics["L_joint_weight_taylor"] == pytest.approx(0.0)
    assert quant_metrics["L_joint_weight_taylor"] > 0.0
    assert combined_metrics["L_joint_weight_taylor"] > quant_metrics["L_joint_weight_taylor"]
    assert combined_metrics["activation_taylor_included"] is False
    assert combined_metrics["L_joint_weight_taylor"] == pytest.approx(
        combined_metrics["L_joint_weight_taylor_raw"]
        / combined_metrics["joint_weight_taylor_denominator"]
    )


def test_joint_objective_uses_hard_bops_and_parameter_retention_is_report_only() -> None:
    from search.candidate import CandidatePhenotype
    from search.proxy.objective import ProxyObjective, ProxyObjectiveConfig

    class Joint:
        def evaluate_breakdown(self, _phenotype):
            return {
                "L_joint_weight_taylor": 0.25,
                "L_joint_weight_taylor_raw": 2.5,
                "joint_weight_taylor_denominator": 10.0,
            }

    class Breakdown:
        def __init__(self, payload):
            self.payload = payload

        def evaluate_breakdown(self, _phenotype):
            return dict(self.payload)

    objective = ProxyObjective(
        joint_weight_taylor=Joint(),
        size=Breakdown({"R_size_vs_fp32": 0.2, "R_size_vs_fp16_deploy": 0.4}),
        bops=Breakdown({"R_bops_vs_fp32": 0.11, "R_bops_vs_fp16_deploy": 0.44}),
        config=ProxyObjectiveConfig(
            objective_mode="joint_weight_taylor_hard_bops",
            bops_threshold=0.10,
            bops_constraint_mode="hard_feasibility",
            parameter_retention_tiebreak_epsilon=0.0,
        ),
    )

    metrics = objective.evaluate(CandidatePhenotype())

    assert metrics["proxy_score_raw"] == pytest.approx(0.25)
    assert metrics["F1"] > 1.0e6
    assert metrics["bops_violation"] == pytest.approx(0.01)
    assert metrics["parameter_retention_role"] == "report_only"


@pytest.mark.parametrize(
    ("r_bops", "feasible"),
    [(0.0951, True), (0.1049, True), (0.0948, False), (0.1052, False)],
)
def test_joint_objective_uses_two_sided_absolute_bops_band(
    r_bops: float, feasible: bool
) -> None:
    from search.candidate import CandidatePhenotype
    from search.proxy.objective import ProxyObjective, ProxyObjectiveConfig

    class Joint:
        def evaluate_breakdown(self, _phenotype):
            return {"L_joint_weight_taylor": 0.25}

    class Breakdown:
        def __init__(self, payload):
            self.payload = payload

        def evaluate_breakdown(self, _phenotype):
            return dict(self.payload)

    objective = ProxyObjective(
        joint_weight_taylor=Joint(),
        size=Breakdown({"R_parameter_retention": 0.8}),
        bops=Breakdown({"R_bops_vs_fp32": r_bops}),
        config=ProxyObjectiveConfig(
            objective_mode="joint_weight_taylor_hard_bops",
            bops_threshold=0.10,
            bops_constraint_mode="hard_band_feasibility",
            bops_tolerance_abs=0.005,
        ),
    )

    metrics = objective.evaluate(CandidatePhenotype())

    assert metrics["bops_feasible"] is feasible
    assert (metrics["bops_violation"] == 0.0) is feasible
    assert metrics["F1"] == pytest.approx(0.25)


def test_domain_width_torch_batch_matches_scalar_joint_taylor() -> None:
    import torch

    from search.candidate import CandidateGenotype
    from search.canonicalization import SearchSpaceSpec, canonicalize_candidate
    from search.proxy.bops_proxy import BOPSProxy
    from search.proxy.fisher_proxy import FisherStatistics
    from search.proxy.gpu_batch_proxy import TorchBatchedProxyScorer
    from search.proxy.joint_weight_taylor import JointWeightTaylorProxy
    from search.proxy.normalization import NormalizationStats
    from search.proxy.objective import ProxyObjective, ProxyObjectiveConfig
    from search.proxy.parameter_slice_resolver import ParameterSlice
    from search.proxy.runtime_shape_profiler import RuntimeLayerShape
    from search.proxy.size_proxy import SizeProxy
    from search.pruning_space.local_domains import build_local_pruning_domains

    model = torch.nn.Sequential()
    model.add_module("conv", torch.nn.Conv2d(2, 8, kernel_size=1, bias=False))
    model.register_parameter(
        "untracked_scale",
        torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0])),
    )
    with torch.no_grad():
        model.conv.weight.copy_(
            torch.arange(1, 17, dtype=torch.float32).reshape(8, 2, 1, 1) / 19.0
        )
    statistics = FisherStatistics(
        gradients={"conv.weight": torch.full_like(model.conv.weight, 0.2)},
        fisher_diag={"conv.weight": torch.full_like(model.conv.weight, 0.04)},
        manifest_hash="toy-domain",
    )
    units = [_unit("scope", "conv", index, float(index)) for index in range(8)]
    slices = {
        unit.stable_id: [
            ParameterSlice(
                "conv.weight",
                "conv",
                0,
                (index,),
                "prune_weight_slice",
            )
        ]
        for index, unit in enumerate(units)
    }
    scores = {unit.stable_id: float(index) for index, unit in enumerate(units)}
    domain = build_local_pruning_domains(units, importance_scores=scores)[0]
    space = SearchSpaceSpec(
        pruning_unit_ids=[unit.stable_id for unit in units],
        precision_layer_ids=["conv"],
        pruning_domains=(domain,),
        default_precision="FP16",
    )
    shapes = [
        RuntimeLayerShape(
            module_path="conv",
            call_index=0,
            module_type="Conv2d",
            input_shape=(1, 2, 4, 4),
            output_shape=(1, 8, 4, 4),
            c_in=2,
            c_out=8,
            h_out=4,
            w_out=4,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            dilation=(1, 1),
            groups=1,
            weight_shape=(8, 2, 1, 1),
            precision_group_id="conv",
            macs=256.0,
        )
    ]
    config = ProxyObjectiveConfig(
        objective_mode="joint_weight_taylor_hard_bops",
        bops_threshold=0.10,
        bops_constraint_mode="hard_band_feasibility",
        bops_tolerance_abs=0.005,
    )
    scalar = ProxyObjective(
        joint_weight_taylor=JointWeightTaylorProxy(
            model,
            statistics=statistics,
            unit_to_parameter_slices=slices,
        ),
        size=SizeProxy(model, unit_to_parameter_slices=slices),
        bops=BOPSProxy(
            model,
            unit_to_parameter_slices=slices,
            runtime_shapes=shapes,
        ),
        config=config,
    )
    batched = TorchBatchedProxyScorer.from_components(
        model=model,
        space=space,
        unit_to_parameter_slices=slices,
        fisher_statistics=statistics,
        runtime_shapes=shapes,
        normalization=NormalizationStats(),
        config=config,
        device="cpu",
        batch_size=8,
    )
    phenotypes = [
        canonicalize_candidate(
            CandidateGenotype(
                precision_genes={"conv": precision},
                pruning_width_genes={domain.domain_id: width},
            ),
            space,
        )
        for width, precision in ((8, "FP32"), (8, "INT8"), (4, "FP16"), (4, "INT8"))
    ]

    batch_rows = batched.evaluate_batch(
        phenotypes,
        generation=0,
        outer_round=0,
    ).metrics

    assert batched.uses_explicit_candidate_masks is True
    assert batched.channel_resolver.action_out_mask is None
    assert batched.channel_resolver.constant_base_params.item() == pytest.approx(3.0)
    for phenotype, batch_row in zip(phenotypes, batch_rows):
        scalar_row = scalar.evaluate(phenotype)
        for key in (
            "L_joint_weight_taylor",
            "L_pruning_only_taylor",
            "L_retained_weight_quant_taylor",
            "R_size_vs_fp32",
            "R_parameter_retention",
            "R_bops_vs_fp32",
            "bops_violation",
            "bops_abs_delta",
            "F1",
        ):
            assert batch_row[key] == pytest.approx(
                scalar_row[key], rel=1.0e-5, abs=1.0e-7
            )
        assert batch_row["bops_feasible"] is scalar_row["bops_feasible"]
        assert batch_row["parameter_count_base"] == pytest.approx(19.0)
        assert batch_row["constant_untracked_parameter_count"] == pytest.approx(3.0)
        assert scalar_row["constant_untracked_parameter_count"] == pytest.approx(3.0)
