"""First-order task-loss Taylor rankings for CoBEVT Attention dimensions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import torch
import torch.nn as nn


@dataclass(frozen=True)
class AttentionTaylorScores:
    qk_by_head: tuple[tuple[float, ...], ...]
    vo_by_head: tuple[tuple[float, ...], ...]

    def to_dict(self) -> dict[str, list[list[float]]]:
        return {
            "qk_by_head": [list(row) for row in self.qk_by_head],
            "vo_by_head": [list(row) for row in self.vo_by_head],
        }


def _required_gradient(
    gradients: Mapping[str, torch.Tensor], name: str, parameter: torch.Tensor
) -> torch.Tensor:
    if name not in gradients:
        raise ValueError(f"attention_taylor_gradient_missing:{name}")
    gradient = gradients[name]
    if tuple(gradient.shape) != tuple(parameter.shape):
        raise ValueError(f"attention_taylor_gradient_shape_mismatch:{name}")
    if not bool(torch.isfinite(gradient).all()):
        raise ValueError(f"attention_taylor_gradient_nonfinite:{name}")
    return gradient


def _element_score(parameter: torch.Tensor, gradient: torch.Tensor) -> torch.Tensor:
    return (parameter.detach().to(torch.float64) * gradient.detach().to(torch.float64)).abs()


def first_order_attention_scores(
    module: nn.Module, gradients: Mapping[str, torch.Tensor]
) -> AttentionTaylorScores:
    heads = int(module.heads)
    d_qk = int(module.d_qk)
    d_v = int(module.d_v)
    q = _element_score(
        module.q_proj.weight,
        _required_gradient(gradients, "q_proj.weight", module.q_proj.weight),
    ).sum(dim=1)
    k = _element_score(
        module.k_proj.weight,
        _required_gradient(gradients, "k_proj.weight", module.k_proj.weight),
    ).sum(dim=1)
    v = _element_score(
        module.v_proj.weight,
        _required_gradient(gradients, "v_proj.weight", module.v_proj.weight),
    ).sum(dim=1)
    out = _element_score(
        module.out_proj.weight,
        _required_gradient(gradients, "out_proj.weight", module.out_proj.weight),
    ).sum(dim=0)
    if module.q_proj.bias is not None:
        q = q + _element_score(
            module.q_proj.bias,
            _required_gradient(gradients, "q_proj.bias", module.q_proj.bias),
        )
        k = k + _element_score(
            module.k_proj.bias,
            _required_gradient(gradients, "k_proj.bias", module.k_proj.bias),
        )
        v = v + _element_score(
            module.v_proj.bias,
            _required_gradient(gradients, "v_proj.bias", module.v_proj.bias),
        )
    qk = (q + k).reshape(heads, d_qk)
    vo = (v + out).reshape(heads, d_v)
    if not bool(torch.isfinite(qk).all()) or not bool(torch.isfinite(vo).all()):
        raise ValueError("attention_taylor_score_nonfinite")
    return AttentionTaylorScores(
        qk_by_head=tuple(tuple(float(value) for value in row) for row in qk.tolist()),
        vo_by_head=tuple(tuple(float(value) for value in row) for row in vo.tolist()),
    )


def keep_indices_from_scores(
    scores_by_head: Iterable[Iterable[float]], *, keep_width: int
) -> tuple[tuple[int, ...], ...]:
    rows = tuple(tuple(float(value) for value in row) for row in scores_by_head)
    if not rows or keep_width <= 0:
        raise ValueError("attention_taylor_keep_width_invalid")
    result = []
    for row in rows:
        if keep_width > len(row):
            raise ValueError("attention_taylor_keep_width_exceeds_original")
        ranking = sorted(range(len(row)), key=lambda index: (row[index], index))
        pruned = set(ranking[: len(row) - int(keep_width)])
        result.append(tuple(index for index in range(len(row)) if index not in pruned))
    return tuple(result)


def attention_masks_from_mean_gradients(
    model: nn.Module,
    gradients: Mapping[str, torch.Tensor],
    *,
    d_qk: int,
    d_v: int,
    module_type: type[nn.Module] | None = None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    from .attention_dim_pruning import AttentionDimMask, PrunableCobevtAttention

    expected_type = PrunableCobevtAttention if module_type is None else module_type
    masks: dict[str, AttentionDimMask] = {}
    audit: dict[str, dict[str, Any]] = {}
    for name, module in model.named_modules():
        if not isinstance(module, expected_type):
            continue
        prefix = f"{name}." if name else ""
        local = {
            parameter_name[len(prefix) :]: value
            for parameter_name, value in gradients.items()
            if parameter_name.startswith(prefix)
        }
        scores = first_order_attention_scores(module, local)
        qk_keep = keep_indices_from_scores(scores.qk_by_head, keep_width=int(d_qk))
        vo_keep = keep_indices_from_scores(scores.vo_by_head, keep_width=int(d_v))
        masks[name] = AttentionDimMask(
            qk_keep,
            vo_keep,
            original_d_qk=int(module.d_qk),
            original_d_v=int(module.d_v),
        )
        qk_rankings = tuple(
            tuple(sorted(range(len(row)), key=lambda index: (row[index], index)))
            for row in scores.qk_by_head
        )
        vo_rankings = tuple(
            tuple(sorted(range(len(row)), key=lambda index: (row[index], index)))
            for row in scores.vo_by_head
        )
        audit[name] = {
            **scores.to_dict(),
            "d_qk": int(d_qk),
            "d_v": int(d_v),
            "qk_keep_by_head": [list(row) for row in qk_keep],
            "vo_keep_by_head": [list(row) for row in vo_keep],
            "qk_ranking_by_head": [list(row) for row in qk_rankings],
            "vo_ranking_by_head": [list(row) for row in vo_rankings],
        }
    if not masks:
        raise RuntimeError("attention_taylor_modules_missing")
    return masks, audit


__all__ = [
    "AttentionTaylorScores",
    "attention_masks_from_mean_gradients",
    "first_order_attention_scores",
    "keep_indices_from_scores",
]
