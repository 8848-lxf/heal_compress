"""Search-time weight-only pseudo-quantization.

Implements symmetric uniform quantization for INT8/INT4 and FP16 truncation.
Activations are NOT quantized (kept at original FP32/FP16 precision).
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Iterator

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def pseudo_quantize_weight(weight: torch.Tensor, bit_label: str) -> torch.Tensor:
    """Apply pseudo-quantization to a weight tensor.

    Symmetric uniform quantization:
        s = max(|W|) / (2^(bits-1) - 1)
        W_q = clip(round(W/s), -2^(bits-1), 2^(bits-1)-1)
        W_dq = W_q * s

    FP16 pseudo-quantization: W.half().float()

    Args:
        weight: Weight tensor (any shape).
        bit_label: Bit-width label ('FP16', 'INT8', or 'INT4').

    Returns:
        Pseudo-quantized weight tensor (same dtype as input).
    """
    if bit_label == "FP16":
        return weight.half().float()

    bits = _label_to_bits(bit_label)
    if bits >= 16:
        return weight

    qmax = float(2 ** (bits - 1) - 1)
    scale = weight.detach().abs().amax()
    if scale <= 0:
        return weight
    scale = scale / qmax
    quantized = torch.clamp(torch.round(weight / scale), -qmax, qmax)
    return quantized * scale


def pseudo_quantize_weight_per_channel(
    weight: torch.Tensor,
    bit_label: str,
    axis: int = 0,
) -> torch.Tensor:
    """Apply per-channel pseudo-quantization.

    Computes a separate scale per output channel (along axis).

    Args:
        weight: Weight tensor.
        bit_label: Bit-width label.
        axis: Channel axis (default 0 for output channels).

    Returns:
        Per-channel pseudo-quantized weight tensor.
    """
    if bit_label == "FP16":
        return weight.half().float()

    bits = _label_to_bits(bit_label)
    if bits >= 16:
        return weight

    qmax = float(2 ** (bits - 1) - 1)

    # Compute per-channel scale
    reduce_dims = [i for i in range(weight.dim()) if i != axis]
    if reduce_dims:
        amax = weight.detach().abs().amax(dim=reduce_dims, keepdim=True)
    else:
        amax = weight.detach().abs().amax()
    scale = amax / qmax
    scale = torch.clamp(scale, min=1e-12)

    quantized = torch.clamp(torch.round(weight / scale), -qmax, qmax)
    return quantized * scale


def _label_to_bits(label: str) -> int:
    """Convert bit-width label to integer bits.

    Args:
        label: 'FP16', 'INT8', or 'INT4'.

    Returns:
        Integer bit count (16, 8, or 4).
    """
    mapping = {"FP16": 16, "INT8": 8, "INT4": 4}
    return mapping.get(label, 16)


class PseudoQuantManager:
    """Context manager for applying weight-only pseudo-quantization during search.

    Temporarily replaces model weights with their pseudo-quantized versions.
    On exit, restores original weights.

    Args:
        model: The HEAL model.
    """

    def __init__(self, model: nn.Module):
        self.model = model

    @contextmanager
    def apply(
        self,
        bitwidth_vars: dict[str, str],
        granularity: str = "per_tensor",
    ) -> Iterator[None]:
        """Apply pseudo-quantization to model weights.

        Args:
            bitwidth_vars: Map of layer_name -> bit-width label.
            granularity: 'per_tensor' or 'per_channel'.

        Yields:
            None. Model weights are temporarily quantized.
        """
        modules = dict(self.model.named_modules())
        backups: dict[str, torch.Tensor] = {}

        try:
            for layer_name, bit_label in bitwidth_vars.items():
                module = modules.get(layer_name)
                if module is None or not hasattr(module, "weight"):
                    continue
                backups[layer_name] = module.weight.detach().clone()
                if granularity == "per_channel":
                    qw = pseudo_quantize_weight_per_channel(
                        module.weight.data, bit_label
                    )
                else:
                    qw = pseudo_quantize_weight(module.weight.data, bit_label)
                module.weight.data.copy_(qw)
            yield
        finally:
            for layer_name, weight in backups.items():
                module = modules.get(layer_name)
                if module is not None:
                    module.weight.data.copy_(weight)
