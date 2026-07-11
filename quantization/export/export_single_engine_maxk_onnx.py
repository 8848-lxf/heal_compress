"""Deprecated name for :func:`export_signal_maxk_onnx`."""

from __future__ import annotations

import warnings
from typing import Any

from .signal_maxk import export_signal_maxk_onnx


def export_single_engine_maxk_onnx(*args: Any, **kwargs: Any) -> Any:
    """Forward to the typed formal exporter."""

    warnings.warn(
        "export_single_engine_maxk_onnx is deprecated; use quantization.api.export_signal_maxk_onnx",
        DeprecationWarning,
        stacklevel=2,
    )
    return export_signal_maxk_onnx(*args, **kwargs)


def main() -> int:
    raise SystemExit("Use the programmatic typed API with an explicit model and example inputs.")


if __name__ == "__main__":
    raise SystemExit(main())
