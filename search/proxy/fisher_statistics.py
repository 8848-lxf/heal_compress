"""Empirical-Fisher accumulation helpers.

The empirical diagonal is ``mean(g**2)`` over samples or explicitly declared
micro-batches. It is an approximation and is never represented as a full
Hessian.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import torch

from .fisher_proxy import FisherStatistics


def build_fisher_statistics_manifest(
    *,
    checkpoint_hash: str,
    model_config_hash: str,
    calibration_manifest_hash: str,
    sample_count: int,
    micro_batch_size: int,
    parameter_count: int,
    finite_gradient_count: int,
    nonfinite_count: int,
    code_commit: str,
    creation_timestamp: str,
) -> dict[str, Any]:
    return {
        "checkpoint_hash": str(checkpoint_hash),
        "model_config_hash": str(model_config_hash),
        "calibration_manifest_hash": str(calibration_manifest_hash),
        "sample_count": int(sample_count),
        "micro_batch_size": int(micro_batch_size),
        "loss_components": ["L_cls", "L_reg", "L_dir", "L_obj"],
        "parameter_count": int(parameter_count),
        "finite_gradient_count": int(finite_gradient_count),
        "nonfinite_count": int(nonfinite_count),
        "mean_gradient_accumulation_method": "mean(g)",
        "fisher_accumulation_method": "mean(g^2)",
        "dtype": "float32",
        "code_commit": str(code_commit),
        "creation_timestamp": str(creation_timestamp),
        "hessian_claim": "empirical_fisher_diagonal_approximation_not_full_hessian",
    }


def accumulate_gradient_samples(
    samples: Iterable[Mapping[str, torch.Tensor]],
) -> tuple[FisherStatistics, dict[str, Any]]:
    gradient_sum: dict[str, torch.Tensor] = {}
    gradient_square_sum: dict[str, torch.Tensor] = {}
    count = 0
    for sample in samples:
        count += 1
        for name, raw_gradient in sample.items():
            gradient = raw_gradient.detach().to(dtype=torch.float64, device="cpu")
            if name not in gradient_sum:
                gradient_sum[name] = torch.zeros_like(gradient)
                gradient_square_sum[name] = torch.zeros_like(gradient)
            if gradient_sum[name].shape != gradient.shape:
                raise RuntimeError(f"fisher_gradient_shape_mismatch:{name}")
            gradient_sum[name] += gradient
            gradient_square_sum[name] += gradient.square()
    if count <= 0:
        raise RuntimeError("fisher_statistics_missing:no_gradient_samples")
    gradients = {
        name: (value / float(count)).to(dtype=torch.float32)
        for name, value in gradient_sum.items()
    }
    fisher_diag = {
        name: (gradient_square_sum[name] / float(count)).to(dtype=torch.float32)
        for name in gradient_sum
    }
    nonfinite = sum(
        int((~torch.isfinite(value)).sum().item())
        for values in (gradients, fisher_diag)
        for value in values.values()
    )
    parameter_count = sum(int(value.numel()) for value in gradients.values())
    audit = {
        "sample_count": int(count),
        "parameter_count": int(parameter_count),
        "finite_gradient_count": int(parameter_count - nonfinite),
        "nonfinite_count": int(nonfinite),
        "mean_gradient_accumulation_method": "mean(g)",
        "fisher_accumulation_method": "mean(g^2)",
        "hessian_claim": "empirical_fisher_diagonal_approximation_not_full_hessian",
        "dtype": "float32",
    }
    if nonfinite:
        raise RuntimeError(f"fisher_statistics_nonfinite:{nonfinite}")
    return (
        FisherStatistics(
            gradients=gradients,
            fisher_diag=fisher_diag,
            statistics_version="empirical-fisher-diagonal-mean-g2-v2",
        ),
        audit,
    )
