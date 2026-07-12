"""Stage-2 real evaluation adapter."""

from __future__ import annotations

from typing import Any, Callable


class RealEvaluator:
    def __init__(self, evaluate_fn: Callable[..., dict[str, Any]] | None = None) -> None:
        self.evaluate_fn = evaluate_fn

    def evaluate(self, **kwargs: Any) -> dict[str, Any]:
        if self.evaluate_fn is None:
            return {
                "status": "evaluation_failed",
                "failure_reason": "missing_public_interface: provide prepared engine runner, inputs, postprocess, and metric adapter",
            }
        return dict(self.evaluate_fn(**kwargs))
