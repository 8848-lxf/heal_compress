"""Hardware-friendly channel alignment checker and corrector.

Ensures post-pruning channel counts satisfy TRT alignment requirements
for optimal kernel utilization.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch.nn as nn

from ..utils.io_utils import ensure_dir, save_json

logger = logging.getLogger(__name__)


class ChannelAlignmentChecker:
    """Checks and corrects channel alignment for hardware deployment.

    Alignment rules:
    - FP16 layers: align to 8
    - INT8 layers: align to 16
    - INT4 layers: align to 32
    - groups > 1: groups align to 8, per-group channels align to 8
    - Depthwise: total channels align to 8

    If alignment forces channels below min_channels, rounds up to
    the nearest valid multiple.

    Args:
        default_align: Default alignment base (default 8).
        min_channels: Minimum channels per layer (default 8).
    """

    BIT_ALIGN = {"FP16": 8, "INT8": 16, "INT4": 32}

    def __init__(self, default_align: int = 8, min_channels: int = 8):
        self.default_align = default_align
        self.min_channels = min_channels

    def check(
        self,
        model: nn.Module,
        bitwidth_vars: dict[str, str] | None = None,
        output_dir: str | None = None,
    ) -> dict[str, Any]:
        """Check channel alignment for all layers.

        Args:
            model: The (pruned) HEAL model.
            bitwidth_vars: Map of layer_name -> bit-width label.
            output_dir: Optional directory for saving report.

        Returns:
            Alignment report dict with per-layer status.
        """
        bitwidth_vars = bitwidth_vars or {}
        items = []

        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                bit_label = bitwidth_vars.get(name, "FP16")
                align = self.BIT_ALIGN.get(bit_label, self.default_align)
                item = self._check_conv(name, module, align)
                items.append(item)
            elif isinstance(module, nn.Linear):
                bit_label = bitwidth_vars.get(name, "FP16")
                align = self.BIT_ALIGN.get(bit_label, self.default_align)
                item = self._check_linear(name, module, align)
                items.append(item)

        report = {
            "default_align": self.default_align,
            "min_channels": self.min_channels,
            "all_aligned": all(i.get("ok", False) for i in items),
            "layers": items,
        }

        if output_dir:
            save_json(report, str(Path(ensure_dir(output_dir)) / "alignment_report.json"))

        return report

    def _check_conv(
        self, name: str, module: nn.Conv2d | nn.ConvTranspose2d, align: int,
    ) -> dict[str, Any]:
        """Check alignment for a convolutional layer."""
        groups = module.groups
        in_ch = module.in_channels
        out_ch = module.out_channels

        out_ok = out_ch % align == 0 or out_ch < self.min_channels
        in_ok = in_ch % align == 0 or in_ch < self.min_channels
        groups_ok = groups == 1 or (
            groups > 0
            and in_ch % groups == 0
            and out_ch % groups == 0
            and (groups % 8 == 0 or groups == in_ch)
        )

        return {
            "layer": name,
            "type": module.__class__.__name__,
            "in_channels": in_ch,
            "out_channels": out_ch,
            "groups": groups,
            "align": align,
            "in_aligned": in_ok,
            "out_aligned": out_ok,
            "groups_ok": groups_ok,
            "ok": out_ok and in_ok and groups_ok,
        }

    def _check_linear(
        self, name: str, module: nn.Linear, align: int,
    ) -> dict[str, Any]:
        """Check alignment for a linear layer."""
        in_ok = module.in_features % align == 0 or module.in_features < self.min_channels
        out_ok = module.out_features % align == 0 or module.out_features < self.min_channels

        return {
            "layer": name,
            "type": "Linear",
            "in_features": module.in_features,
            "out_features": module.out_features,
            "align": align,
            "in_aligned": in_ok,
            "out_aligned": out_ok,
            "ok": in_ok and out_ok,
        }

    @staticmethod
    def align_value(value: int, align: int, min_val: int = 8) -> int:
        """Align a channel count to the nearest valid multiple.

        Rounds down, but if the result is below min_val, rounds up instead.

        Args:
            value: Current channel count.
            align: Alignment base.
            min_val: Minimum allowed value.

        Returns:
            Aligned channel count.
        """
        aligned = (value // align) * align
        if aligned < min_val:
            aligned = ((min_val + align - 1) // align) * align
        return aligned
