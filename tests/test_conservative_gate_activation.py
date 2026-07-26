import torch
import torch.nn as nn

from search.proxy.conservative_gate_activation_taylor import (
    ActivationTaylorCache,
    FunctionalGateTaylorProxy,
    GateDomainScores,
    _score,
    collect_activation_taylor_cache,
)
from search.proxy.joint_weight_activation_taylor import TaylorDeploymentUnit


def test_gate_score_is_elementwise_absolute_before_sum():
    value = torch.tensor([1.0, -2.0])
    grad = torch.tensor([2.0, 3.0])
    score = _score(value, grad)
    # |g*u| + .5|g^2*u^2| = [4, 24] (no signed cancellation).
    assert torch.equal(score, torch.tensor([4.0, 24.0]))


def test_activation_cache_requires_exact_transition_and_is_nonnegative():
    cache = ActivationTaylorCache(
        {"g": ("u",)},
        {("u", "FP32", "FP16"): 1.5, ("u", "FP16", "INT8"): 2.0},
        (),
    )
    class P:
        def __init__(self, value): self.realized_precision_profile = {"u": value}
    value = cache.action_breakdown(P("FP32"), P("FP16"))
    assert value["delta_J_AQ"] == 1.5


def test_gate_breakdown_reports_first_and_second_order_without_changing_total():
    scores = GateDomainScores(
        domain_id="d",
        unit_scores={"u": 3.0},
        semantic_root_tensor="root",
        gate_tensor="gate",
        physical_dependencies=(),
        family="cnn",
        unit_first_scores={"u": 1.0},
        unit_second_scores={"u": 2.0},
    )
    proxy = FunctionalGateTaylorProxy({"d": scores})

    class P:
        def __init__(self, pruned):
            self.pruned_unit_ids = tuple(pruned)

    row = proxy.pruning_action_breakdown(P(()), P(("u",)))
    assert row["delta_J_prune"] == 3.0
    assert row["first_order_abs_sum"] == 1.0
    assert row["second_order_abs_sum"] == 2.0
    assert row["component_breakdown_available"] is True


def test_activation_taylor_uses_fp64_accumulation_for_finite_fp32_inputs():
    class Scale(nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return value

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale = Scale()

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return self.scale(value)

    unit = TaylorDeploymentUnit(
        unit_id="activation::scale",
        module_path="scale",
        unit_type="activation",
        boundary="module_input",
        precision_owner="scale",
        quantizer_id="activation::scale",
        metadata={"precision_group_id": "scale"},
    )
    value = torch.tensor([10000.0, 7501.0], dtype=torch.float32)
    cache = collect_activation_taylor_cache(
        Model().eval(),
        (unit,),
        {"scale": (unit.unit_id,)},
        forward_fn=lambda model, batch: model(batch),
        loss_fn=lambda output, batch: (output * 1.0e20).sum(),
        batch=value,
    )
    score = cache.transitions[(unit.unit_id, "FP16", "INT8")]
    assert score >= 0.0
    assert torch.isfinite(torch.tensor(score, dtype=torch.float64))
