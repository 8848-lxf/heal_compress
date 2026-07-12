"""Read-only adapter over the formal tracer public API."""

from __future__ import annotations

from typing import Any, Callable


class FormalTracerAdapter:
    """Small injectable wrapper around `tracer.api.trace_model`."""

    def __init__(self, trace_model_fn: Callable[..., Any] | None = None) -> None:
        if trace_model_fn is None:
            try:
                from tracer.api import trace_model as trace_model_fn
            except ImportError:
                from heal_compress.tracer.api import trace_model as trace_model_fn
        self.trace_model_fn = trace_model_fn

    def trace(self, model: Any, example_inputs: Any, **kwargs: Any) -> Any:
        return self.trace_model_fn(model, example_inputs, **kwargs)
