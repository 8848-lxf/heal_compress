"""Realized BOPS accounting from physical module shapes and precision realization."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import torch.nn as nn

from ..proxy.size_proxy import BIT_WIDTHS


def _module_channels(module: nn.Module, attr: str, fallback: int | None) -> int:
    return int(getattr(module, attr, fallback or 1) or fallback or 1)


def _kernel_size(module: nn.Module, fallback: Any) -> tuple[int, int]:
    value = getattr(module, "kernel_size", fallback or (1, 1))
    if isinstance(value, tuple):
        if len(value) == 1:
            return int(value[0]), int(value[0])
        return int(value[0]), int(value[1])
    return int(value), int(value)


def _precision_bits(precision: str) -> tuple[int, int]:
    bits = BIT_WIDTHS.get(str(precision).upper(), 16)
    return int(bits), int(bits)


def compute_realized_bops(
    model: nn.Module,
    *,
    runtime_shapes: Iterable[Any],
    realized_precision_profile: Mapping[str, Any],
) -> dict[str, Any]:
    """Compute BOPS ratio using actual physical module widths and realized precision."""

    modules = dict(model.named_modules())
    profile = {str(key): str(value).upper() for key, value in dict(realized_precision_profile).items()}
    total_bops = 0.0
    total_macs = 0.0
    fp32_baseline = 0.0
    fp16_baseline = 0.0
    int8_layers = 0
    int8_macs = 0.0
    rows: list[dict[str, Any]] = []
    counted: set[tuple[str, int]] = set()
    for shape in runtime_shapes:
        module_path = str(getattr(shape, "module_path", ""))
        call_index = int(getattr(shape, "call_index", 0) or 0)
        key = (module_path, call_index)
        if key in counted:
            continue
        counted.add(key)
        module = modules.get(module_path)
        if module is None:
            continue
        precision = profile.get(module_path, "FP16")
        weight_bits, activation_bits = _precision_bits(precision)
        base_macs = float(getattr(shape, "macs", 0.0) or 0.0)
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            kh, kw = _kernel_size(module, getattr(shape, "kernel_size", (1, 1)))
            c_in = _module_channels(module, "in_channels", getattr(shape, "c_in", 1))
            c_out = _module_channels(module, "out_channels", getattr(shape, "c_out", 1))
            groups = _module_channels(module, "groups", getattr(shape, "groups", 1))
            h_out = int(getattr(shape, "h_out", 1) or 1)
            w_out = int(getattr(shape, "w_out", 1) or 1)
            macs = float(h_out * w_out * kh * kw * c_in * c_out / max(groups, 1))
        elif isinstance(module, nn.Linear):
            c_in = _module_channels(module, "in_features", getattr(shape, "c_in", 1))
            c_out = _module_channels(module, "out_features", getattr(shape, "c_out", 1))
            groups = 1
            h_out = 1
            w_out = 1
            macs = float(c_in * c_out)
        else:
            continue
        bops = macs * weight_bits * activation_bits
        total_macs += macs
        total_bops += bops
        fp32_baseline += base_macs * 32.0 * 32.0
        fp16_baseline += base_macs * 16.0 * 16.0
        if precision == "INT8":
            int8_layers += 1
            int8_macs += macs
        rows.append(
            {
                "module_path": module_path,
                "call_index": call_index,
                "module_type": type(module).__name__,
                "realized_precision": precision,
                "C_in_after": c_in,
                "C_out_after": c_out,
                "groups_after": groups,
                "H_out": h_out,
                "W_out": w_out,
                "MACs_realized": macs,
                "MACs_fp32_baseline": base_macs,
                "weight_bits": weight_bits,
                "activation_bits": activation_bits,
                "BOPS_realized": bops,
            }
        )
    fp32 = fp32_baseline or 1.0
    fp16 = fp16_baseline or 1.0
    return {
        "schema_version": "realized-bops-v1",
        "weighted_layer_count": len(rows),
        "realized_int8_layer_count": int8_layers,
        "realized_int8_macs_ratio": float(int8_macs / max(total_macs, 1.0)),
        "BOPS_realized": float(total_bops),
        "BOPS_fp32_baseline": float(fp32),
        "BOPS_fp16_baseline": float(fp16),
        "R_BOPS_realized": float(total_bops / fp32),
        "R_BOPS_realized_vs_fp16": float(total_bops / fp16),
        "layers": rows,
    }
