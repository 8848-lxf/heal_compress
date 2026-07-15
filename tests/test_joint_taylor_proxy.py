from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from search.candidate import CandidatePhenotype, PrecisionDecision
from search.proxy.fisher_proxy import FisherStatistics
from search.proxy.parameter_slice_resolver import ParameterSlice


class TwoLayerNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.left = nn.Linear(2, 2, bias=False)
        self.right = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.left.weight.copy_(torch.tensor([[1.125, -2.25], [3.5, -4.75]]))
            self.right.weight.copy_(torch.tensor([[5.125, -6.25], [7.5, -8.75]]))


def _phenotype(*, pruned: list[str], left: str, right: str) -> CandidatePhenotype:
    return CandidatePhenotype(
        pruned_unit_ids=pruned,
        precision_profile={
            "left": PrecisionDecision(left, left),
            "right": PrecisionDecision(right, right),
        },
    )


def _stats(model: nn.Module) -> FisherStatistics:
    return FisherStatistics(
        gradients={name: torch.full_like(param, 0.25) for name, param in model.named_parameters()},
        fisher_diag={name: torch.full_like(param, 0.5) for name, param in model.named_parameters()},
    )


def _slices() -> dict[str, list[ParameterSlice]]:
    return {
        "coupled": [
            ParameterSlice("left.weight", "left", 0, (0,), "prune_weight_slice"),
            ParameterSlice("right.weight", "right", 1, (1,), "prune_weight_slice"),
        ]
    }


def test_effective_delta_prunes_removed_values_and_quantizes_retained_values() -> None:
    from search.proxy.joint_taylor import effective_parameter_delta

    weight = torch.tensor([[1.125, -2.25], [3.5, -4.75]])
    row = ParameterSlice("left.weight", "left", 0, (0,), "prune_weight_slice")

    delta = effective_parameter_delta(weight, precision="FP16", pruned_slices=[row])

    assert torch.equal(delta[0], -weight[0])
    expected_kept = weight[1].to(torch.float16).to(weight.dtype) - weight[1]
    assert torch.equal(delta[1], expected_kept)


def test_joint_proxy_accumulates_cross_layer_group_with_each_layer_precision() -> None:
    from search.proxy.joint_taylor import JointTaylorProxy, effective_parameter_delta

    model = TwoLayerNet()
    proxy = JointTaylorProxy(
        model,
        statistics=_stats(model),
        unit_to_parameter_slices=_slices(),
        mode="joint_taylor_second_order_fisher_diag",
    )
    phenotype = _phenotype(pruned=["coupled"], left="FP16", right="INT8")

    result = proxy.evaluate(phenotype)

    left_delta = effective_parameter_delta(
        model.left.weight.detach(),
        precision="FP16",
        pruned_slices=_slices()["coupled"][:1],
    )
    right_delta = effective_parameter_delta(
        model.right.weight.detach(),
        precision="INT8",
        pruned_slices=_slices()["coupled"][1:],
    )
    expected_first = (
        (torch.full_like(left_delta, 0.25) * left_delta).abs().sum()
        + (torch.full_like(right_delta, 0.25) * right_delta).abs().sum()
    )
    expected_second = 0.5 * (
        (torch.full_like(left_delta, 0.5) * left_delta.square()).sum()
        + (torch.full_like(right_delta, 0.5) * right_delta.square()).sum()
    )

    assert result.first_order_sum == pytest.approx(float(expected_first), rel=1e-6)
    assert result.second_order_fisher_sum == pytest.approx(float(expected_second), rel=1e-6)
    assert result.total_importance == pytest.approx(float(expected_first + expected_second), rel=1e-6)


def test_overlapping_dependency_slices_are_unioned_without_double_counting() -> None:
    from search.proxy.joint_taylor import JointTaylorProxy

    model = TwoLayerNet()
    repeated = ParameterSlice("left.weight", "left", 0, (0,), "prune_weight_slice")
    proxy = JointTaylorProxy(
        model,
        statistics=_stats(model),
        unit_to_parameter_slices={"u0": [repeated], "u1": [repeated]},
        mode="joint_taylor_first_order",
    )

    one = proxy.evaluate(_phenotype(pruned=["u0"], left="FP32", right="FP32"))
    two = proxy.evaluate(_phenotype(pruned=["u0", "u1"], left="FP32", right="FP32"))

    assert two.total_importance == pytest.approx(one.total_importance)
    assert two.unique_parameter_count == one.unique_parameter_count
    assert two.duplicate_slice_count == 0


def test_first_order_and_second_order_modes_are_deterministic_and_not_normalized() -> None:
    from search.proxy.joint_taylor import JointTaylorProxy

    model = TwoLayerNet()
    phenotype = _phenotype(pruned=["coupled"], left="FP32", right="FP32")
    first = JointTaylorProxy(
        model,
        statistics=_stats(model),
        unit_to_parameter_slices=_slices(),
        mode="joint_taylor_first_order",
    ).evaluate(phenotype)
    second_proxy = JointTaylorProxy(
        model,
        statistics=_stats(model),
        unit_to_parameter_slices=_slices(),
        mode="joint_taylor_second_order_fisher_diag",
    )
    second_a = second_proxy.evaluate(phenotype)
    second_b = second_proxy.evaluate(phenotype)

    assert first.total_importance == pytest.approx(first.first_order_sum)
    assert second_a.total_importance == pytest.approx(
        second_a.first_order_sum + second_a.second_order_fisher_sum
    )
    assert second_a == second_b
    assert second_a.total_importance > first.total_importance
    assert second_a.normalization == "none"
    assert second_a.sqnr_main_objective_contribution == 0.0


def test_empirical_fisher_accumulator_uses_mean_gradient_square() -> None:
    from search.proxy.fisher_statistics import accumulate_gradient_samples

    samples = [
        {"weight": torch.tensor([1.0, -1.0])},
        {"weight": torch.tensor([-1.0, 1.0])},
    ]

    statistics, audit = accumulate_gradient_samples(samples)

    assert torch.equal(statistics.gradients["weight"], torch.zeros(2))
    assert torch.equal(statistics.fisher_diag["weight"], torch.ones(2))
    assert not torch.equal(
        statistics.fisher_diag["weight"], statistics.gradients["weight"].square()
    )
    assert audit["sample_count"] == 2
    assert audit["mean_gradient_accumulation_method"] == "mean(g)"
    assert audit["fisher_accumulation_method"] == "mean(g^2)"


def test_fisher_statistics_manifest_contains_required_lineage() -> None:
    from search.proxy.fisher_statistics import build_fisher_statistics_manifest

    manifest = build_fisher_statistics_manifest(
        checkpoint_hash="checkpoint",
        model_config_hash="config",
        calibration_manifest_hash="calibration",
        sample_count=8,
        micro_batch_size=1,
        parameter_count=12,
        finite_gradient_count=12,
        nonfinite_count=0,
        code_commit="commit",
        creation_timestamp="2026-07-15T00:00:00+08:00",
    )

    assert manifest == {
        "checkpoint_hash": "checkpoint",
        "model_config_hash": "config",
        "calibration_manifest_hash": "calibration",
        "sample_count": 8,
        "micro_batch_size": 1,
        "loss_components": ["L_cls", "L_reg", "L_dir", "L_obj"],
        "parameter_count": 12,
        "finite_gradient_count": 12,
        "nonfinite_count": 0,
        "mean_gradient_accumulation_method": "mean(g)",
        "fisher_accumulation_method": "mean(g^2)",
        "dtype": "float32",
        "code_commit": "commit",
        "creation_timestamp": "2026-07-15T00:00:00+08:00",
        "hessian_claim": "empirical_fisher_diagonal_approximation_not_full_hessian",
    }


def test_conditional_group_cost_subtracts_keep_quantization_cost() -> None:
    from search.proxy.joint_taylor import JointTaylorProxy

    model = TwoLayerNet()
    proxy = JointTaylorProxy(
        model,
        statistics=_stats(model),
        unit_to_parameter_slices=_slices(),
        mode="conditional_joint_taylor_second_order_fisher_diag",
    )
    fp16 = proxy.conditional_group_costs(
        _phenotype(pruned=[], left="FP16", right="FP16")
    )["coupled"]
    int8 = proxy.conditional_group_costs(
        _phenotype(pruned=[], left="INT8", right="INT8")
    )["coupled"]

    assert fp16.finite is True
    assert fp16.duplicate_slice_count == 0
    assert fp16.involved_precision_groups == ("left", "right")
    assert fp16.total_importance != pytest.approx(int8.total_importance)
