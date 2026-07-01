"""Formal quantization and TensorRT deployment tools.

This package now contains two surfaces:

* formal deployment CLIs under ``quantization.export/build/calibrate/eval``;
* the earlier pseudo-quantization helpers, loaded lazily for compatibility.

The lazy exports keep ``python -m quantization...`` working from this repository
root, where the package is imported as ``quantization`` rather than
``heal_compress.quantization``.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "PseudoQuantManager",
    "QDQDeploymentBuilder",
    "pseudo_quantize_weight",
]


def __getattr__(name: str) -> Any:
    if name in {"PseudoQuantManager", "pseudo_quantize_weight"}:
        module = import_module(".pseudo_quant", __name__)
        return getattr(module, name)
    if name == "QDQDeploymentBuilder":
        module = import_module(".qdq_builder", __name__)
        return getattr(module, name)
    raise AttributeError(name)
