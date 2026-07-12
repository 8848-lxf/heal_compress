"""Evaluation adapter for formal engine evaluation and HEAL-specific hooks."""

from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping


class FormalEvaluationAdapter:
    """Thin wrapper around `quantization.api.evaluate_engine`."""

    def __init__(self, evaluate_engine_fn: Callable[..., Any] | None = None) -> None:
        if evaluate_engine_fn is None:
            try:
                from quantization.api import evaluate_engine as evaluate_engine_fn
            except ImportError:
                from heal_compress.quantization.api import evaluate_engine as evaluate_engine_fn
        self.evaluate_engine_fn = evaluate_engine_fn

    def evaluate(
        self,
        runner: Any,
        prepared_inputs: Iterable[Mapping[str, Any]],
        **kwargs: Any,
    ) -> Any:
        return self.evaluate_engine_fn(runner, prepared_inputs, **kwargs)
