"""Transformer tensor-op coverage for the model-agnostic runtime tracer."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import einsum as imported_einsum

from heal_compress.tracer.generic_tracer import trace_model


class ImportedEinsumAttention(nn.Module):
    """Mimic CoBEVT's module-level ``from torch import einsum`` usage."""

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        return imported_einsum("bhid,bhjd->bhij", q, k)


class OperatorMatmulAttention(nn.Module):
    def forward(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        return q @ k.transpose(-2, -1)


def _ops(trace: dict) -> list[str]:
    return [
        str(node.get("op"))
        for node in trace["nodes"].values()
        if node.get("type") == "TensorOp"
    ]


def test_runtime_tracer_captures_imported_einsum_alias_and_restores_it() -> None:
    before = imported_einsum
    sample = (torch.randn(1, 2, 3, 4), torch.randn(1, 2, 5, 4))
    trace = trace_model(
        ImportedEinsumAttention().eval(),
        sample,
        forward_fn=lambda model, values: model(*values),
    )
    assert "torch.einsum" in _ops(trace)
    assert imported_einsum is before


def test_runtime_tracer_captures_matmul_operator() -> None:
    sample = (torch.randn(1, 2, 3, 4), torch.randn(1, 2, 5, 4))
    trace = trace_model(
        OperatorMatmulAttention().eval(),
        sample,
        forward_fn=lambda model, values: model(*values),
    )
    assert "Tensor.__matmul__" in _ops(trace)
