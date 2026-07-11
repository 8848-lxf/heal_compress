"""HEAL LiDAR-pyramid tracing adapter without repository or data paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import torch.nn as nn

from ..api import trace_model
from ..config import TraceConfig
from ..types import TraceResult


@dataclass(frozen=True)
class HEALLiDARPyramidTraceAdapter:
    """Bind a caller-provided HEAL forward recipe to the formal tracer.

    Model construction, checkpoint loading and representative inputs stay with
    the caller. This adapter contributes no hard-coded repository paths and
    does not claim coverage for branches absent from ``example_inputs``.
    """

    forward_fn: Callable[[nn.Module, Any], Any] | None = None

    def trace(
        self,
        model: nn.Module,
        example_inputs: Any,
        *,
        config: TraceConfig | Mapping[str, Any] | None = None,
    ) -> TraceResult:
        """Trace one representative LiDAR-pyramid forward."""

        return trace_model(
            model,
            example_inputs,
            config=config,
            forward_fn=self.forward_fn,
        )


def trace_heal_lidar_pyramid(
    model: nn.Module,
    example_inputs: Any,
    *,
    config: TraceConfig | Mapping[str, Any] | None = None,
    forward_fn: Callable[[nn.Module, Any], Any] | None = None,
) -> TraceResult:
    """Trace a caller-constructed HEAL LiDAR-pyramid model."""

    return HEALLiDARPyramidTraceAdapter(forward_fn=forward_fn).trace(
        model,
        example_inputs,
        config=config,
    )

