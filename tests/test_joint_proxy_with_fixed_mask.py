from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pruning.types import AtomicPruneUnit


def test_canonical_prune_ranking_is_precision_independent_and_audited() -> None:
    from search.decoding.fixed_taylor_width_decoder import (
        build_canonical_prune_ranking,
    )
    from search.proxy.fisher_proxy import FisherStatistics
    from search.proxy.parameter_slice_resolver import ParameterSlice
    from search.space.legal_width_inventory import build_legal_width_inventory

    model = torch.nn.Sequential()
    model.add_module("conv", torch.nn.Conv2d(2, 2, 1, bias=False))
    with torch.no_grad():
        model.conv.weight.copy_(
            torch.tensor([[[[1.0]], [[2.0]]], [[[3.0]], [[4.0]]]])
        )
    units = [
        AtomicPruneUnit(
            "scope", "conv", "out", [index], [f"c{index}"], 0.0,
            _stable_id=f"u{index}",
        )
        for index in range(2)
    ]
    inventory = build_legal_width_inventory(
        units,
        dense_alignment=1,
        minimum_retained_channels=1,
        per_domain_max_prune_rate=0.5,
    )
    slices = {
        f"u{index}": [
            ParameterSlice("conv.weight", "conv", 0, (index,), "root_out")
        ]
        for index in range(2)
    }
    statistics = FisherStatistics(
        gradients={"conv.weight": torch.full_like(model.conv.weight, 0.5)},
        fisher_diag={"conv.weight": torch.full_like(model.conv.weight, 0.25)},
    )

    ranking = build_canonical_prune_ranking(
        model,
        statistics=statistics,
        unit_to_parameter_slices=slices,
        inventory=inventory,
        checkpoint_hash="checkpoint",
        fisher_manifest_hash="fisher",
    )

    assert ranking.ranking_mode == "prune_only_second_order_fisher"
    assert ranking.ranking_hash
    assert ranking.manifest["ranking_depends_on_precision"] is False
    assert sum(row.parameter_element_count for row in ranking.rows) == 4
    assert all(row.duplicate_parameter_element_count == 0 for row in ranking.rows)
    assert [row.atomic_unit_id for row in ranking.rows] == ["u0", "u1"]
    assert ranking.rows[0].first_order_score == pytest.approx(1.5)
    assert ranking.rows[0].second_order_score == pytest.approx(2.125)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_canonical_ranking_accepts_cpu_fisher_for_cuda_model() -> None:
    from search.decoding.fixed_taylor_width_decoder import (
        build_canonical_prune_ranking,
    )
    from search.proxy.fisher_proxy import FisherStatistics
    from search.proxy.parameter_slice_resolver import ParameterSlice
    from search.space.legal_width_inventory import build_legal_width_inventory

    model = torch.nn.Sequential()
    model.add_module("conv", torch.nn.Conv2d(2, 2, 1, bias=False))
    model = model.cuda()
    units = [
        AtomicPruneUnit(
            "scope", "conv", "out", [index], [f"c{index}"], 0.0,
            _stable_id=f"u{index}",
        )
        for index in range(2)
    ]
    inventory = build_legal_width_inventory(
        units, dense_alignment=1, per_domain_max_prune_rate=0.5
    )
    slices = {
        f"u{index}": [
            ParameterSlice("conv.weight", "conv", 0, (index,), "root_out")
        ]
        for index in range(2)
    }
    statistics = FisherStatistics(
        gradients={"conv.weight": torch.ones_like(model.conv.weight.cpu())},
        fisher_diag={"conv.weight": torch.ones_like(model.conv.weight.cpu())},
    )

    ranking = build_canonical_prune_ranking(
        model,
        statistics=statistics,
        unit_to_parameter_slices=slices,
        inventory=inventory,
        checkpoint_hash="checkpoint",
        fisher_manifest_hash="fisher",
    )

    assert len(ranking.rows) == 2
    assert all(row.parameter_element_count == 2 for row in ranking.rows)


def test_flat_slice_union_indexes_only_covered_elements() -> None:
    from search.decoding.fixed_taylor_width_decoder import (
        _slice_union_flat_indices,
    )
    from search.proxy.parameter_slice_resolver import ParameterSlice

    indices = _slice_union_flat_indices(
        (2, 3, 2),
        [
            ParameterSlice("weight", "conv", 0, (1,), "out"),
            ParameterSlice("weight", "conv", 1, (1,), "in"),
            ParameterSlice("weight", "conv", 1, (1,), "duplicate"),
        ],
    )

    assert indices.device.type == "cpu"
    assert indices.tolist() == [2, 3, 6, 7, 8, 9, 10, 11]


def test_same_fixed_mask_has_precision_dependent_joint_score() -> None:
    from search.candidate import CandidatePhenotype, PrecisionDecision
    from search.decoding.fixed_taylor_width_decoder import FixedTaylorWidthDecoder
    from search.proxy.fisher_proxy import FisherStatistics
    from search.proxy.joint_taylor import JointTaylorProxy
    from search.proxy.parameter_slice_resolver import ParameterSlice
    from search.space.legal_width_inventory import build_legal_width_inventory

    model = torch.nn.Sequential()
    model.add_module("conv", torch.nn.Conv2d(2, 2, 1, bias=False))
    with torch.no_grad():
        model.conv.weight.copy_(
            torch.tensor([[[[0.11]], [[0.23]]], [[[0.37]], [[0.41]]]])
        )
    units = [
        AtomicPruneUnit(
            "scope", "conv", "out", [index], [f"c{index}"], 0.0,
            _stable_id=f"u{index}",
        )
        for index in range(2)
    ]
    inventory = build_legal_width_inventory(
        units, dense_alignment=1, per_domain_max_prune_rate=0.5
    )
    domain_id = inventory.domain_ids[0]
    decoder = FixedTaylorWidthDecoder(
        inventory,
        [
            {
                "domain_id": domain_id,
                "physical_group_id": 0,
                "atomic_unit_id": f"u{index}",
                "first_order_score": float(index),
                "second_order_score": float(index),
            }
            for index in range(2)
        ],
    )
    decoded = decoder.decode({domain_id: 0})
    slices = {
        f"u{index}": [
            ParameterSlice("conv.weight", "conv", 0, (index,), "root_out")
        ]
        for index in range(2)
    }
    statistics = FisherStatistics(
        gradients={"conv.weight": torch.full_like(model.conv.weight, 0.1)},
        fisher_diag={"conv.weight": torch.full_like(model.conv.weight, 0.02)},
    )
    proxy = JointTaylorProxy(
        model,
        statistics=statistics,
        unit_to_parameter_slices=slices,
        mode="joint_taylor_second_order_fisher_diag",
    )
    fp16 = CandidatePhenotype(
        list(decoded.pruned_unit_ids),
        {"conv": PrecisionDecision("FP16", "FP16")},
    )
    int8 = CandidatePhenotype(
        list(decoded.pruned_unit_ids),
        {"conv": PrecisionDecision("INT8", "INT8")},
    )

    assert fp16.pruned_unit_ids == int8.pruned_unit_ids
    assert proxy.evaluate(fp16).total_importance != pytest.approx(
        proxy.evaluate(int8).total_importance
    )


def test_scalar_and_batched_joint_score_align_for_fixed_mask() -> None:
    from search.candidate import CandidateGenotype
    from search.canonicalization import SearchSpaceSpec, canonicalize_candidate
    from search.proxy.bops_proxy import BOPSProxy
    from search.proxy.fisher_proxy import FisherStatistics
    from search.proxy.gpu_batch_proxy import TorchBatchedProxyScorer
    from search.proxy.joint_taylor import JointTaylorProxy
    from search.proxy.normalization import NormalizationStats
    from search.proxy.objective import ProxyObjective, ProxyObjectiveConfig
    from search.proxy.parameter_slice_resolver import ParameterSlice
    from search.proxy.runtime_shape_profiler import RuntimeLayerShape
    from search.proxy.size_proxy import SizeProxy

    model = torch.nn.Sequential()
    model.add_module("conv0", torch.nn.Conv2d(2, 2, 1, bias=True))
    model.add_module("conv1", torch.nn.Conv2d(2, 2, 1, bias=False))
    with torch.no_grad():
        model.conv0.weight.copy_(
            torch.tensor([[[[0.11]], [[0.23]]], [[[0.37]], [[0.41]]]])
        )
        model.conv0.bias.copy_(torch.tensor([0.17, 0.29]))
        model.conv1.weight.copy_(
            torch.tensor([[[[0.13]], [[0.31]]], [[[0.43]], [[0.47]]]])
        )
    statistics = FisherStatistics(
        gradients={
            name: torch.full_like(value, 0.1)
            for name, value in model.named_parameters()
        },
        fisher_diag={
            name: torch.full_like(value, 0.02)
            for name, value in model.named_parameters()
        },
    )
    slices = {
        "coupled": [
            ParameterSlice("conv0.weight", "conv0", 0, (0,), "root_out"),
            ParameterSlice("conv0.bias", "conv0", 0, (0,), "root_bias"),
            ParameterSlice("conv1.weight", "conv1", 1, (0,), "successor_in"),
        ]
    }
    shapes = [
        RuntimeLayerShape(
            module_path=name,
            call_index=0,
            module_type="Conv2d",
            input_shape=(1, 2, 4, 4),
            output_shape=(1, 2, 4, 4),
            c_in=2,
            c_out=2,
            h_out=4,
            w_out=4,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            dilation=(1, 1),
            groups=1,
            weight_shape=tuple(getattr(model, name).weight.shape),
            precision_group_id=name,
            macs=64.0,
        )
        for name in ("conv0", "conv1")
    ]
    space = SearchSpaceSpec(
        pruning_unit_ids=["coupled"],
        precision_layer_ids=["conv0", "conv1"],
        default_precision="FP16",
    )
    config = ProxyObjectiveConfig(
        bops_threshold=None,
        proxy_mode="joint_taylor_second_order_fisher_diag",
        task_score_mapping="linear_fixed_scale",
        joint_loss_scale=10.0,
        task_weight=0.8,
        prune_weight=0.2,
    )
    scalar = ProxyObjective(
        joint=JointTaylorProxy(
            model,
            statistics=statistics,
            unit_to_parameter_slices=slices,
            mode=config.proxy_mode,
        ),
        size=SizeProxy(model, unit_to_parameter_slices=slices),
        bops=BOPSProxy(model, unit_to_parameter_slices=slices, runtime_shapes=shapes),
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
                {"coupled": keep},
                {"conv0": "INT8", "conv1": "FP16"},
            ),
            space,
        )
        for keep in (1, 0)
    ]

    rows = batched.evaluate_batch(phenotypes, generation=0, outer_round=0).metrics
    for phenotype, row in zip(phenotypes, rows):
        expected = scalar.evaluate(phenotype)
        for key in (
            "L_joint_raw",
            "L_joint_first_order",
            "L_joint_second_order",
            "normalized_joint_loss",
            "L_scale",
            "candidate_params",
            "R_prune",
            "J1",
            "F1",
        ):
            assert row[key] == pytest.approx(expected[key], rel=1e-5, abs=1e-7)
        assert row["sqnr_main_objective_contribution"] == 0.0
        assert row["task_score_mapping"] == "linear_fixed_scale"
        for obsolete in (
            "tau",
            "exponent_value",
            "S_task",
            "task_score_saturated",
        ):
            assert obsolete not in row
            assert obsolete not in expected


def test_stage1_evaluator_decodes_legal_width_candidate_without_repair() -> None:
    from search.canonicalization import SearchSpaceSpec
    from search.decoding.fixed_taylor_width_decoder import FixedTaylorWidthDecoder
    from search.encoding.legal_width_genotype import LegalWidthGenotype
    from search.space.legal_width_inventory import build_legal_width_inventory
    from search.stage1.proxy_evaluator import Stage1ProxyEvaluator

    units = [
        AtomicPruneUnit(
            "scope", "conv", "out", [index], [f"c{index}"], float(index),
            _stable_id=f"u{index}",
        )
        for index in range(4)
    ]
    inventory = build_legal_width_inventory(
        units, dense_alignment=1, per_domain_max_prune_rate=0.5
    )
    domain_id = inventory.domain_ids[0]
    decoder = FixedTaylorWidthDecoder(
        inventory,
        [
            {
                "domain_id": domain_id,
                "physical_group_id": 0,
                "atomic_unit_id": f"u{index}",
                "first_order_score": float(index),
                "second_order_score": float(index),
            }
            for index in range(4)
        ],
    )
    space = SearchSpaceSpec(
        pruning_unit_ids=list(inventory.unit_ids),
        precision_layer_ids=["conv"],
        structure_gene_type="legal_keep_width",
        legal_width_inventory=inventory,
        fixed_width_decoder=decoder,
        precision_action_space={"conv": ("FP16", "INT8")},
    )

    class Objective:
        @staticmethod
        def evaluate(phenotype):
            return {
                "F1": float(len(phenotype.pruned_unit_ids)),
                "normal_candidate_repair_invoked": phenotype.metadata[
                    "normal_candidate_repair_invoked"
                ],
            }

    candidate = LegalWidthGenotype({domain_id: 0}, {"conv": "FP16"})
    result = Stage1ProxyEvaluator(space, objective=Objective()).evaluate(candidate)

    assert result["F1"] == 2.0
    assert result["normal_candidate_repair_invoked"] is False
    assert result["phenotype"]["metadata"]["structure_hash"]
