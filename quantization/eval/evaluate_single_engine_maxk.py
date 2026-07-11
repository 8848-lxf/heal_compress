"""Deprecated entry forwarding to the generic typed evaluator."""

from __future__ import annotations

import warnings
from typing import Any

from ..evaluation.evaluator import evaluate_engine


def evaluate(*args: Any, **kwargs: Any) -> Any:
    warnings.warn(
        "evaluate_single_engine_maxk is deprecated; use quantization.api.evaluate_engine",
        DeprecationWarning,
        stacklevel=2,
    )
    return evaluate_engine(*args, **kwargs)


def main() -> int:
    raise SystemExit("Use the typed evaluator with an explicit runner and prepared inputs.")


if __name__ == "__main__":
    raise SystemExit(main())
