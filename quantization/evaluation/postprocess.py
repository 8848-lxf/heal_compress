"""Output adapters for caller-owned detection decoding."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Mapping, Sequence


def adapt_engine_outputs(outputs: Mapping[str, Any], output_names: Sequence[str] = ("cls_preds", "reg_preds", "dir_preds")) -> OrderedDict:
    """Wrap canonical detection tensors for HEAL-style post-processing."""

    missing = [name for name in output_names if name not in outputs]
    if missing:
        raise KeyError(f"engine outputs are missing detection tensors: {missing}")
    return OrderedDict(ego={name: outputs[name] for name in output_names})
