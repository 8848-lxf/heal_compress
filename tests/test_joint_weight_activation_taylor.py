"""Joint structure/weight/activation output-Taylor proxy contracts."""

from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from search.candidate import CandidatePhenotype, PrecisionDecision
from search.proxy.candidate_perturbation import pseudo_quantize_activation
from search.proxy.joint_weight_activation_taylor import (
    JointOutputTaylorStatistics,
    JointWeightActivationTaylorProxy,
    ModelCandidateOutputProvider,
    TaylorDeploymentUnit,
    coalesce_deployment_units,
    collect_joint_output_taylor_statistics,
    score_joint_output_perturbation,
)


class SoftmaxNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(5, 4, bias=True)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.softmax(self.linear(x))


def _units() -> tuple[TaylorDeploymentUnit, ...]:
    return (
        TaylorDeploymentUnit(
            unit_id="linear_output",
            module_path="linear",
            unit_type="projection",
            boundary="module_output",
            precision_owner="linear",
            quantizer_id="q::linear_output",
        ),
        TaylorDeploymentUnit(
            unit_id="softmax_output",
            module_path="softmax",
            unit_type="softmax",
            boundary="module_output",
            precision_owner="softmax",
            quantizer_id="q::softmax_output",
            has_weight=False,
        ),
    )


def _phenotype(linear: str, softmax: str) -> CandidatePhenotype:
    return CandidatePhenotype(
        precision_profile={
            "linear": PrecisionDecision(linear, linear),
            "softmax": PrecisionDecision(softmax, softmax),
        }
    )


def _proxy():
    torch.manual_seed(23)
    model = SoftmaxNet().eval()
    batches = (
        (torch.randn(3, 5), torch.randn(3, 4)),
        (torch.randn(2, 5), torch.randn(2, 4)),
    )

    def forward(current: nn.Module, batch):
        return current(batch[0])

    def loss(output: torch.Tensor, batch):
        return (output * batch[1]).mean() + output.square().mean()

    statistics = collect_joint_output_taylor_statistics(
        model,
        batches,
        forward_fn=forward,
        loss_fn=loss,
        units=_units(),
        calibration_manifest_hash="manifest::fixed-two-samples",
    )
    provider = ModelCandidateOutputProvider(
        model,
        batches,
        forward_fn=forward,
        units=_units(),
        unit_to_parameter_slices={},
        calibration_manifest_hash="manifest::fixed-two-samples",
    )
    return JointWeightActivationTaylorProxy(
        statistics=statistics,
        output_provider=provider,
        units=_units(),
    ), statistics


def test_w32a32_has_zero_quantization_perturbation_without_pruning() -> None:
    proxy, _ = _proxy()
    metrics = proxy.evaluate_breakdown(_phenotype("FP32", "FP32"))
    assert metrics["L_joint_weight_activation_taylor"] == 0.0
    assert metrics["L_joint_first_order"] == 0.0
    assert metrics["L_joint_second_order"] == 0.0
    assert metrics["normalization_applied"] is False


def test_w16a16_and_w8a8_activation_perturbations_are_nonzero() -> None:
    proxy, _ = _proxy()
    fp16 = proxy.evaluate_breakdown(_phenotype("FP16", "FP16"))
    int8 = proxy.evaluate_breakdown(_phenotype("INT8", "INT8"))
    assert fp16["L_joint_weight_activation_taylor"] > 0.0
    assert int8["L_joint_weight_activation_taylor"] > 0.0
    assert fp16["activation_taylor_included"]
    assert int8["structural_weight_activation_cross_terms_preserved"]


def test_softmax_a8_uses_output_activation_delta_and_is_not_zero() -> None:
    proxy, _ = _proxy()
    metrics = proxy.evaluate_breakdown(_phenotype("FP32", "INT8"))
    softmax = next(row for row in metrics["unit_breakdown"] if row["unit_id"] == "softmax_output")
    assert softmax["mean_delta_l1"] > 0.0
    assert softmax["joint_loss_increment"] > 0.0
    assert metrics["softmax_activation_taylor_supported"]


def test_joint_output_formula_uses_raw_common_task_loss_units() -> None:
    statistics = JointOutputTaylorStatistics(
        baseline_outputs={"u": (torch.tensor([1.0, 2.0]),)},
        gradients={"u": (torch.tensor([0.5, -0.25]),)},
        calibration_manifest_hash="m",
        unit_manifest_hash="u",
        sample_count=1,
    )
    metrics = score_joint_output_perturbation(
        statistics,
        {"u": (torch.tensor([1.2, 1.6]),)},
    )
    delta = torch.tensor([0.2, -0.4])
    gradient = torch.tensor([0.5, -0.25])
    expected = (gradient * delta).abs().sum() + 0.5 * (gradient.square() * delta.square()).sum()
    assert metrics["L_joint_weight_activation_taylor"] == pytest.approx(expected.item())
    assert metrics["normalization_applied"] is False
    assert metrics["type_calibration"] == "identity"


def test_weight_activation_product_cross_term_is_preserved_in_real_output_delta() -> None:
    activation = torch.tensor([[0.314159, 0.173205]], dtype=torch.float32)
    weight = torch.tensor([[0.271828], [0.141421]], dtype=torch.float32)
    qa = pseudo_quantize_activation(activation, "INT8")
    qw = pseudo_quantize_activation(weight, "INT8")
    baseline = activation @ weight
    joint = qa @ qw
    weight_only = activation @ qw
    activation_only = qa @ weight
    joint_delta = joint - baseline
    additive_without_cross = (weight_only - baseline) + (activation_only - baseline)
    assert not torch.equal(joint_delta, additive_without_cross)


def test_quantizer_id_is_coalesced_and_conflicting_reuse_fails() -> None:
    first = _units()[0]
    duplicate = TaylorDeploymentUnit(
        unit_id="alias_linear_output",
        module_path="linear",
        unit_type="projection_alias",
        boundary="module_output",
        precision_owner="linear",
        quantizer_id=first.quantizer_id,
    )
    selected, aliases = coalesce_deployment_units((first, duplicate))
    assert len(selected) == 1
    assert aliases[first.quantizer_id] == ["linear_output", "alias_linear_output"]

    conflict = TaylorDeploymentUnit(
        unit_id="wrong_boundary",
        module_path="linear",
        unit_type="projection",
        boundary="module_input",
        precision_owner="linear",
        quantizer_id=first.quantizer_id,
    )
    try:
        coalesce_deployment_units((first, conflict))
    except ValueError as exc:
        assert "quantizer_id_boundary_conflict" in str(exc)
    else:
        raise AssertionError("conflicting quantizer_id reuse must fail")


def test_functional_softmax_boundary_is_captured_and_quantized() -> None:
    class FunctionalAttention(nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return value.softmax(dim=-1)

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attn = FunctionalAttention()

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return self.attn(value)

    unit = TaylorDeploymentUnit(
        unit_id="functional_softmax",
        module_path="attn",
        unit_type="softmax",
        boundary="functional_output",
        functional_op="softmax",
        precision_owner="attn",
        quantizer_id="q::functional_softmax",
        has_weight=False,
    )
    model = Model().eval()
    batches = ((torch.randn(2, 7, requires_grad=True), torch.randn(2, 7)),)
    statistics = collect_joint_output_taylor_statistics(
        model,
        batches,
        forward_fn=lambda current, batch: current(batch[0]),
        loss_fn=lambda output, batch: (output * batch[1]).sum(),
        units=(unit,),
        calibration_manifest_hash="functional-softmax-manifest",
    )
    provider = ModelCandidateOutputProvider(
        model,
        batches,
        forward_fn=lambda current, batch: current(batch[0]),
        units=(unit,),
        unit_to_parameter_slices={},
        calibration_manifest_hash="functional-softmax-manifest",
    )
    proxy = JointWeightActivationTaylorProxy(
        statistics=statistics,
        output_provider=provider,
        units=(unit,),
    )
    metrics = proxy.evaluate_breakdown(
        CandidatePhenotype(precision_profile={"attn": PrecisionDecision("INT8", "INT8")})
    )
    assert metrics["L_joint_weight_activation_taylor"] > 0.0


def test_calibration_statistics_are_reproducible_for_same_manifest_and_seed() -> None:
    _, first = _proxy()
    _, second = _proxy()
    assert first.calibration_manifest_hash == second.calibration_manifest_hash
    assert first.unit_manifest_hash == second.unit_manifest_hash
    assert first.task_losses == second.task_losses
    for unit_id in first.baseline_outputs:
        for left, right in zip(first.baseline_outputs[unit_id], second.baseline_outputs[unit_id]):
            torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)


def test_functional_einsum_av_boundary_is_captured_and_quantized() -> None:
    class AliasAttention(nn.Module):
        def forward(self, attention: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
            # Local import alias exercises the same pattern used by CoBEVT.
            from torch import einsum

            return einsum("bij,bjd->bid", attention, value)

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attn = AliasAttention()

        def forward(self, attention: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
            return self.attn(attention, value)

    unit = TaylorDeploymentUnit(
        unit_id="functional_av",
        module_path="attn",
        unit_type="av_matmul",
        boundary="functional_output",
        functional_op="einsum",
        precision_owner="attn::__av_matmul__",
        quantizer_id="q::functional_av",
        call_index=0,
        has_weight=False,
    )
    model = Model().eval()
    attention = torch.softmax(torch.randn(2, 3, 3), dim=-1).detach().requires_grad_(True)
    value = torch.randn(2, 3, 5, requires_grad=True)
    target = torch.randn(2, 3, 5)
    batches = ((attention, value, target),)
    statistics = collect_joint_output_taylor_statistics(
        model,
        batches,
        forward_fn=lambda current, batch: current(batch[0], batch[1]),
        loss_fn=lambda output, batch: (output * batch[2]).sum(),
        units=(unit,),
        calibration_manifest_hash="functional-av-manifest",
    )
    provider = ModelCandidateOutputProvider(
        model,
        batches,
        forward_fn=lambda current, batch: current(batch[0], batch[1]),
        units=(unit,),
        unit_to_parameter_slices={},
        calibration_manifest_hash="functional-av-manifest",
    )
    proxy = JointWeightActivationTaylorProxy(
        statistics=statistics,
        output_provider=provider,
        units=(unit,),
    )
    metrics = proxy.evaluate_breakdown(
        CandidatePhenotype(
            precision_profile={
                "attn::__av_matmul__": PrecisionDecision("INT8", "INT8")
            }
        )
    )
    assert metrics["L_joint_weight_activation_taylor"] > 0.0
    assert metrics["unit_breakdown"][0]["mean_delta_l1"] > 0.0
