"""Optional caller-controlled forward invariant checking."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn


def run_forward_invariant(model: nn.Module, example_inputs: Any) -> Any:
    """Run a no-gradient representative forward supplied by the caller."""

    training = model.training
    model.eval()
    try:
        with torch.no_grad():
            if isinstance(example_inputs, Mapping):
                return model(**dict(example_inputs))
            if isinstance(example_inputs, tuple):
                return model(*example_inputs)
            return model(example_inputs)
    finally:
        model.train(training)


__all__ = ["run_forward_invariant"]
