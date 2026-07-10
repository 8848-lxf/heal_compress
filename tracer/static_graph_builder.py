"""Compatibility wrapper for formal static graph construction."""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from heal_compress.tracer.dependency_tracer import build_dependency_graph


def build_static_graph(model: nn.Module, sample_batch: Any | None = None) -> dict[str, Any]:
    return build_dependency_graph(model, sample_batch)
