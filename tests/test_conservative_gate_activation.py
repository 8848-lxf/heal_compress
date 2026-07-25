import torch

from search.proxy.conservative_gate_activation_taylor import (
    ActivationTaylorCache,
    GateDomainScores,
    _score,
)


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
