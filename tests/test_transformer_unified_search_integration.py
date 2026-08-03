from __future__ import annotations

import random

import pytest
import torch.nn as nn


def _attention_domain():
    from search.pruning_space.local_domains import LocalPruningDomain

    rankings = tuple(tuple(range(16)) for _ in range(2))
    return LocalPruningDomain(
        domain_id="attention_dh::block.attn",
        root_module_path="block.attn",
        root_axis="per_head",
        scope_id="block.attn",
        kind="transformer_attention",
        domain_type="attention_dh",
        model="toy",
        module_path="block.attn",
        family="toy_window",
        original_width=16,
        total_original_width=32,
        groups=2,
        ordered_unit_ids=(),
        legal_widths=(4, 8, 16),
        width_to_pruned_unit_ids={4: (), 8: (), 16: ()},
        ranking_groups={
            "qk_low_to_high_by_head": rankings,
            "vo_low_to_high_by_head": rankings,
        },
        constraints={"heads": 2, "shared_qkvo_index": False},
    )


def _ffn_domain():
    from search.pruning_space.local_domains import LocalPruningDomain

    return LocalPruningDomain(
        domain_id="ffn_hidden::block.ffn",
        root_module_path="block.ffn",
        root_axis="hidden",
        scope_id="block.ffn",
        kind="transformer_ffn",
        domain_type="ffn_hidden",
        model="toy",
        module_path="block.ffn",
        family="toy_ffn",
        original_width=256,
        total_original_width=256,
        ordered_unit_ids=(),
        legal_widths=(64, 128, 256),
        width_to_pruned_unit_ids={64: (), 128: (), 256: ()},
        ranking_groups={"ffn_low_to_high": tuple(range(256))},
        constraints={"ffn_type": "standard"},
    )


def _space(*, with_precision: bool = False):
    from search.canonicalization import SearchSpaceSpec
    from search.quantization_space.types import QuantizationSearchGroup

    groups = ()
    layers: list[str] = []
    if with_precision:
        groups = (
            QuantizationSearchGroup(
                group_id="precision::projection",
                module_paths=("block.attn.qkv",),
                canonical_node_ids=("qkv",),
                allowed_precisions=("FP32", "FP16", "INT8"),
                protected=False,
                protection_reason="",
                ordering=0,
                parameter_count=1,
                baseline_macs=1.0,
            ),
        )
        layers = ["block.attn.qkv"]
    return SearchSpaceSpec(
        pruning_unit_ids=[],
        precision_layer_ids=layers,
        quantization_groups=groups,
        pruning_domains=(_attention_domain(), _ffn_domain()),
        default_precision="FP32",
    )


def test_greedy_enumerates_attention_and_ffn_actions_and_audits_type() -> None:
    from search.greedy import GreedyBudgetSearch, GreedySearchConfig

    search = GreedyBudgetSearch(
        _space(),
        config=GreedySearchConfig(bops_targets=(0.75,), maximum_steps=1),
    )
    initial = search._initial_candidate()
    actions = [action for _candidate, action in search._neighbors(initial)]
    assert {row["domain_type"] for row in actions} == {"attention_dh", "ffn_hidden"}

    def evaluate(candidates, _step):
        rows = []
        for candidate in candidates:
            attention = candidate.pruning_width_genes["attention_dh::block.attn"]
            ffn = candidate.pruning_width_genes["ffn_hidden::block.ffn"]
            changed_attention = attention < 16
            changed_ffn = ffn < 256
            rows.append(
                {
                    "R_bops_vs_fp32": 0.5 * attention / 16.0 + 0.5 * ffn / 256.0,
                    "L_joint_weight_activation_taylor": 0.1 * (
                        int(changed_attention) + int(changed_ffn)
                    ),
                    # Equal marginal score; deployment latency resolves the tie.
                    "latency_proxy_ms": 2.0 if changed_attention else 3.0,
                    "R_parameter_retention": 0.8,
                    "mixed_weight_size_bytes": 100.0,
                }
            )
        return rows

    result = search.run(evaluate)
    assert result.steps[0].action_gene_id == "attention_dh::block.attn"
    assert result.steps[0].action_domain_type == "attention_dh"
    assert result.steps[0].action_family == "toy_window"
    assert result.to_dict()["search_semantics"]["activation_taylor_included"] is True


def test_ga_transformer_mutation_crossover_and_codec_remain_legal() -> None:
    from search.candidate import CandidateGenotype
    from search.canonicalization import canonicalize_candidate, repair_genotype
    from search.ga.crossover import block_crossover
    from search.ga.mutation import mutate_candidate

    space = _space(with_precision=True)
    left = repair_genotype(
        CandidateGenotype(
            pruning_width_genes={
                "attention_dh::block.attn": 16,
                "ffn_hidden::block.ffn": 256,
            },
            precision_genes={"precision::projection": "FP32"},
        ),
        space,
    )
    right = repair_genotype(
        CandidateGenotype(
            pruning_width_genes={
                "attention_dh::block.attn": 4,
                "ffn_hidden::block.ffn": 64,
            },
            precision_genes={"precision::projection": "INT8"},
        ),
        space,
    )
    rng = random.Random(17)
    for _ in range(20):
        child = block_crossover(left, right, space, rng)
        mutated = mutate_candidate(
            child,
            space,
            rng,
            prune_mutation_rate=1.0,
            precision_mutation_rate=1.0,
            action_count=2,
            adjacent_precision=True,
        )
        for domain in space.pruning_domains:
            assert mutated.pruning_width_genes[domain.domain_id] in domain.legal_widths
        restored = CandidateGenotype.from_dict(mutated.to_dict())
        assert restored == mutated
        phenotype = canonicalize_candidate(restored, space)
        assert set(phenotype.metadata["domain_width_profile"]) == {
            "attention_dh::block.attn",
            "ffn_hidden::block.ffn",
        }


def test_qk_fp32_is_constant_contract_not_mutable_or_repaired_gene() -> None:
    from search.candidate import CandidateGenotype
    from search.canonicalization import SearchSpaceSpec, repair_genotype
    from search.quantization_space.transformer_precision import (
        TransformerPrecisionUnit,
        validate_external_precision_profile,
    )

    unit = TransformerPrecisionUnit(
        unit_id="precision::qk",
        model="toy",
        family="toy_attention",
        module_paths=("block.attn::__qk_matmul__",),
        role="qk_matmul",
        allowed_states=("A32",),
        default_state="A32",
        activation_only=True,
        protected=True,
        protection_reason="qk_fp32",
    )
    space = SearchSpaceSpec(
        pruning_unit_ids=[],
        precision_layer_ids=list(unit.module_paths),
        quantization_groups=(unit.to_search_group(),),
        default_precision="FP32",
    )
    assert space.precision_gene_ids == []
    repaired = repair_genotype(CandidateGenotype(), space)
    assert repaired.precision_genes == {}
    assert repaired.meta["constant_precision_group_profile"] == {
        "precision::qk": "FP32"
    }
    with pytest.raises(
        ValueError, match="non_variable_precision_genes_must_not_enter_genotype"
    ):
        repair_genotype(
            CandidateGenotype(precision_genes={"precision::qk": "INT8"}),
            space,
        )
    with pytest.raises(ValueError, match="transformer_precision_request_illegal"):
        validate_external_precision_profile({"precision::qk": "A8"}, (unit,))


def test_active_projection_keeps_softmax_and_ffn_activation_contracts() -> None:
    from search.adapters.transformer_models import build_transformer_search_components
    from search.proxy.joint_weight_activation_taylor import (
        taylor_units_from_transformer_precision,
    )

    class Attention(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.heads = 4
            self.to_qkv = nn.Linear(32, 3 * 4 * 8, bias=False)
            self.to_out = nn.Sequential(nn.Linear(32, 32, bias=False))
            self.attend = nn.Softmax(dim=-1)

    class FeedForward(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(32, 128), nn.GELU(), nn.Dropout(0.0), nn.Linear(128, 32)
            )

    class PreNorm(nn.Module):
        def __init__(self, fn: nn.Module) -> None:
            super().__init__()
            self.fn = fn

    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.grid_attention = PreNorm(Attention())
            self.window_attention = PreNorm(Attention())
            self.grid_ffd = PreNorm(FeedForward())
            self.window_ffd = PreNorm(FeedForward())

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList((Block(),))

    model = Model()
    active_paths = (
        "layers.0.grid_attention.fn.to_qkv",
        "layers.0.grid_attention.fn.to_out.0",
        "layers.0.window_attention.fn.to_qkv",
        "layers.0.window_attention.fn.to_out.0",
        "layers.0.grid_ffd.fn.net.0",
        "layers.0.grid_ffd.fn.net.3",
        "layers.0.window_ffd.fn.net.0",
        "layers.0.window_ffd.fn.net.3",
    )
    components = build_transformer_search_components(
        model,
        {
            "model": {
                "core_method": "heter_model_baseline",
                "args": {"fusion_method": "cobevt"},
            }
        },
        allow_identity_ranking=True,
        active_module_paths=active_paths,
    )
    by_owner = {
        owner: {
            unit.role
            for unit in components.precision_units
            if unit.unit_id.startswith(f"transformer_precision::{owner}::")
        }
        for owner in (
            "layers.0.grid_attention.fn",
            "layers.0.window_attention.fn",
            "layers.0.grid_ffd.fn",
            "layers.0.window_ffd.fn",
        )
    }
    assert "softmax" in by_owner["layers.0.grid_attention.fn"]
    assert "softmax" in by_owner["layers.0.window_attention.fn"]
    assert "ffn_activation" in by_owner["layers.0.grid_ffd.fn"]
    assert "ffn_activation" in by_owner["layers.0.window_ffd.fn"]
    taylor_roles = {
        unit.unit_type
        for unit in taylor_units_from_transformer_precision(
            model, components.precision_units, active_module_paths=active_paths
        )
    }
    assert "softmax" in taylor_roles
    assert "ffn_activation" in taylor_roles


def test_ga_near_taylor_tie_prefers_latency_before_parameter_retention() -> None:
    from search.candidate import CandidateGenotype
    from search.ga.ranking import rank_constraint_first

    fast = CandidateGenotype(pruning_width_genes={"d": 8})
    compact = CandidateGenotype(pruning_width_genes={"d": 4})
    rows = [
        (
            compact,
            1.0,
            {
                "L_joint_weight_activation_taylor": 1.0,
                "latency_proxy_ms": 2.0,
                "R_parameter_retention": 0.5,
                "bops_feasible": True,
            },
        ),
        (
            fast,
            1.01,
            {
                "L_joint_weight_activation_taylor": 1.01,
                "latency_proxy_ms": 1.0,
                "R_parameter_retention": 0.9,
                "bops_feasible": True,
            },
        ),
    ]
    ranked = rank_constraint_first(rows, taylor_relative_epsilon=0.05)
    assert ranked[0][0] == fast


def test_stage2_topk_records_required_ranks_reason_and_configs() -> None:
    from search.candidate import CandidateGenotype
    from search.canonicalization import canonicalize_candidate
    from search.stage1.repair_selection import select_repaired_stage2_topk

    space = _space(with_precision=True)
    candidates = []
    widths = ((16, 256), (8, 256), (16, 128), (8, 128), (4, 64))
    for index, (attention, ffn) in enumerate(widths):
        genotype = CandidateGenotype(
            pruning_width_genes={
                "attention_dh::block.attn": attention,
                "ffn_hidden::block.ffn": ffn,
            },
            precision_genes={"precision::projection": "FP16"},
        )
        candidates.append((genotype, float(index), {"F1": float(index)}))

    def repair(genotype):
        return genotype, {"status": "ok"}

    def rescore(phenotype):
        attention = phenotype.metadata["domain_width_profile"][
            "attention_dh::block.attn"
        ]
        index = [value[0] for value in widths].index(attention) if attention == 4 else 0
        return {
            "F1": float(index),
            "L_joint_weight_activation_taylor": float(16 - attention) / 16.0,
            "latency_proxy_ms": float(attention),
            "R_parameter_retention": float(attention) / 16.0,
            "mixed_weight_size_bytes": float(attention * 100),
            "R_bops_vs_fp32": 0.3,
            "BOPS_target": 0.3,
            "bops_feasible": True,
        }

    selected, report = select_repaired_stage2_topk(
        candidates,
        space=space,
        repair_fn=repair,
        rescore_fn=rescore,
        topk=3,
    )
    assert len(selected) == 3
    required = {
        "candidate_hash",
        "taylor_rank",
        "latency_proxy_rank",
        "parameter_rank",
        "bops_deviation",
        "selection_reason",
        "width_configuration",
        "precision_configuration",
    }
    assert all(required <= set(row.metrics) for row in selected)
    assert all(required <= set(row) for row in report["selected_audit"])
    assert all(row.metrics["bops_deviation"] == 0.0 for row in selected)


def test_stage2_bounded_repair_pool_places_hard_gate_before_taylor() -> None:
    from search.candidate import CandidateGenotype
    from search.stage1.repair_selection import select_repaired_stage2_topk

    space = _space()
    rows = []
    for index, width in enumerate((4, 8, 16)):
        candidate = CandidateGenotype(
            pruning_width_genes={
                "attention_dh::block.attn": width,
                "ffn_hidden::block.ffn": 256,
            }
        )
        feasible = width == 16
        rows.append(
            (
                candidate,
                float(index),
                {
                    "F1": float(index),
                    "bops_feasible": feasible,
                    "R_bops_vs_fp32": 0.30 if feasible else 0.10,
                },
            )
        )

    def rescore(phenotype):
        width = phenotype.metadata["domain_width_profile"][
            "attention_dh::block.attn"
        ]
        feasible = width == 16
        return {
            "F1": float(16 - width),
            "L_joint_weight_activation_taylor": float(16 - width),
            "bops_feasible": feasible,
            "R_bops_vs_fp32": 0.30 if feasible else 0.10,
            "BOPS_target": 0.30,
        }

    selected, report = select_repaired_stage2_topk(
        rows,
        space=space,
        repair_fn=lambda genotype: (genotype, {"status": "ok"}),
        rescore_fn=rescore,
        topk=1,
        repair_pool_size=1,
        eligibility_fn=lambda metrics: bool(metrics["bops_feasible"]),
    )
    assert len(selected) == 1
    assert selected[0].genotype.pruning_width_genes[
        "attention_dh::block.attn"
    ] == 16
    assert report["raw_hard_gate_ordering_applied"] is True
