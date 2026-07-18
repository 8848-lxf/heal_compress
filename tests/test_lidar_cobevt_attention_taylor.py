from __future__ import annotations

import torch
import torch.nn as nn


class _TinyExplicitAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.heads = 2
        self.d_qk = 2
        self.d_v = 2
        self.q_proj = nn.Linear(3, 4, bias=True)
        self.k_proj = nn.Linear(3, 4, bias=True)
        self.v_proj = nn.Linear(3, 4, bias=True)
        self.out_proj = nn.Linear(4, 3, bias=False)


def test_first_order_qk_importance_couples_matching_q_and_k_rows():
    from search.model_families.lidar_cobevt.attention_taylor import (
        first_order_attention_scores,
    )

    module = _TinyExplicitAttention()
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.fill_(1.0)
    gradients = {name: torch.ones_like(value) for name, value in module.named_parameters()}
    scores = first_order_attention_scores(module, gradients)

    # Per local QK dimension: two 3-element rows plus two scalar biases.
    assert scores.qk_by_head == ((8.0, 8.0), (8.0, 8.0))
    # Per local VO dimension: one 3-element V row + bias + one 3-element WO col.
    assert scores.vo_by_head == ((7.0, 7.0), (7.0, 7.0))


def test_first_order_ranking_is_per_head_stable_and_independent_for_qk_and_vo():
    from search.model_families.lidar_cobevt.attention_taylor import (
        first_order_attention_scores,
        keep_indices_from_scores,
    )

    module = _TinyExplicitAttention()
    gradients = {name: torch.ones_like(value) for name, value in module.named_parameters()}
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.fill_(1.0)
        module.q_proj.weight[1].fill_(0.01)
        module.k_proj.weight[1].fill_(0.01)
        module.v_proj.weight[0].fill_(0.01)
        module.out_proj.weight[:, 0].fill_(0.01)
    scores = first_order_attention_scores(module, gradients)
    qk_keep = keep_indices_from_scores(scores.qk_by_head, keep_width=1)
    vo_keep = keep_indices_from_scores(scores.vo_by_head, keep_width=1)

    assert qk_keep[0] == (0,)
    assert vo_keep[0] == (1,)
    assert qk_keep != vo_keep


def test_taylor_scoring_does_not_double_count_or_average_elements():
    from search.model_families.lidar_cobevt.attention_taylor import (
        first_order_attention_scores,
    )

    module = _TinyExplicitAttention()
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.fill_(2.0)
    gradients = {name: torch.full_like(value, 3.0) for name, value in module.named_parameters()}
    scores = first_order_attention_scores(module, gradients)

    assert scores.qk_by_head[0][0] == 48.0
    assert scores.vo_by_head[0][0] == 42.0


def test_model_taylor_masks_are_module_local_and_deterministic():
    from search.model_families.lidar_cobevt.attention_dim_pruning import (
        PrunableCobevtAttention,
    )
    from search.model_families.lidar_cobevt.attention_taylor import (
        attention_masks_from_mean_gradients,
    )

    class Pair(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.first = _TinyExplicitAttention()
            self.second = _TinyExplicitAttention()

    model = Pair()
    assert not any(isinstance(module, PrunableCobevtAttention) for module in model.modules())
    gradients = {
        name: torch.ones_like(parameter) for name, parameter in model.named_parameters()
    }
    with torch.no_grad():
        model.first.q_proj.weight[1].fill_(0.001)
        model.first.k_proj.weight[1].fill_(0.001)
        model.second.q_proj.weight[0].fill_(0.001)
        model.second.k_proj.weight[0].fill_(0.001)

    masks, audit = attention_masks_from_mean_gradients(
        model,
        gradients,
        d_qk=1,
        d_v=1,
        module_type=_TinyExplicitAttention,
    )

    assert masks["first"].qk_keep_by_head[0] == (0,)
    assert masks["second"].qk_keep_by_head[0] == (1,)
    assert audit["first"]["qk_ranking_by_head"][0][0] == 1
    assert audit["second"]["qk_ranking_by_head"][0][0] == 0
