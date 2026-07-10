"""Shape invariant checks for physical channel pruning."""

from __future__ import annotations

from typing import Any, Mapping

import torch.nn as nn


def _shape(value: Any) -> list[int]:
    return [int(v) for v in tuple(value.shape)] if value is not None else []


def _tuple(value: Any) -> list[int] | str | bool | float | None:
    if isinstance(value, tuple):
        return [int(v) if isinstance(v, int) else v for v in value]
    return value


def snapshot_model_shape_invariants(model: nn.Module) -> dict[str, dict[str, Any]]:
    """Capture module attributes that must not change except channel axes."""

    out: dict[str, dict[str, Any]] = {}
    for name, module in model.named_modules():
        if name == "":
            continue
        if isinstance(module, nn.Conv2d):
            out[name] = {
                "module_name": name,
                "module_type": "Conv2d",
                "weight_shape": _shape(module.weight),
                "bias_shape": _shape(module.bias),
                "kernel_size": _tuple(module.kernel_size),
                "stride": _tuple(module.stride),
                "padding": _tuple(module.padding),
                "dilation": _tuple(module.dilation),
                "output_padding": None,
                "groups": int(module.groups),
                "in_channels": int(module.in_channels),
                "out_channels": int(module.out_channels),
            }
        elif isinstance(module, nn.ConvTranspose2d):
            out[name] = {
                "module_name": name,
                "module_type": "ConvTranspose2d",
                "weight_shape": _shape(module.weight),
                "bias_shape": _shape(module.bias),
                "kernel_size": _tuple(module.kernel_size),
                "stride": _tuple(module.stride),
                "padding": _tuple(module.padding),
                "dilation": _tuple(module.dilation),
                "output_padding": _tuple(module.output_padding),
                "groups": int(module.groups),
                "in_channels": int(module.in_channels),
                "out_channels": int(module.out_channels),
            }
        elif isinstance(module, nn.modules.batchnorm._BatchNorm):
            out[name] = {
                "module_name": name,
                "module_type": module.__class__.__name__,
                "weight_shape": _shape(module.weight),
                "bias_shape": _shape(module.bias),
                "running_mean_shape": _shape(module.running_mean),
                "running_var_shape": _shape(module.running_var),
                "num_features": int(module.num_features),
                "eps": float(module.eps),
                "momentum": module.momentum,
                "affine": bool(module.affine),
                "track_running_stats": bool(module.track_running_stats),
            }
        elif isinstance(module, nn.GroupNorm):
            out[name] = {
                "module_name": name,
                "module_type": "GroupNorm",
                "weight_shape": _shape(module.weight),
                "bias_shape": _shape(module.bias),
                "num_channels": int(module.num_channels),
                "num_groups": int(module.num_groups),
                "eps": float(module.eps),
                "affine": bool(module.affine),
            }
        elif isinstance(module, nn.LayerNorm):
            out[name] = {
                "module_name": name,
                "module_type": "LayerNorm",
                "weight_shape": _shape(module.weight),
                "bias_shape": _shape(module.bias),
                "normalized_shape": list(module.normalized_shape),
                "eps": float(module.eps),
                "elementwise_affine": bool(module.elementwise_affine),
            }
        elif isinstance(module, nn.Linear):
            out[name] = {
                "module_name": name,
                "module_type": "Linear",
                "weight_shape": _shape(module.weight),
                "bias_shape": _shape(module.bias),
                "in_features": int(module.in_features),
                "out_features": int(module.out_features),
            }
    return out


def _same(before: Mapping[str, Any], after: Mapping[str, Any], key: str, reasons: list[str]) -> None:
    if before.get(key) != after.get(key):
        reasons.append(f"{key}_changed:{before.get(key)}->{after.get(key)}")


def _conv_row(name: str, before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    for key in ("kernel_size", "stride", "padding", "dilation", "output_padding", "groups"):
        _same(before, after, key, reasons)
    b_w = list(before.get("weight_shape") or [])
    a_w = list(after.get("weight_shape") or [])
    if len(b_w) != len(a_w):
        reasons.append(f"weight_rank_changed:{b_w}->{a_w}")
    elif len(b_w) >= 4 and b_w[2:] != a_w[2:]:
        reasons.append(f"weight_spatial_changed:{b_w[2:]}->{a_w[2:]}")
    allowed_channel_change = (
        before.get("in_channels") != after.get("in_channels")
        or before.get("out_channels") != after.get("out_channels")
        or before.get("weight_shape") != after.get("weight_shape")
        or before.get("bias_shape") != after.get("bias_shape")
    )
    return {
        "module_name": name,
        "module_type": before.get("module_type", after.get("module_type", "")),
        "before_weight_shape": before.get("weight_shape", []),
        "after_weight_shape": after.get("weight_shape", []),
        "before_kernel_size": before.get("kernel_size"),
        "after_kernel_size": after.get("kernel_size"),
        "before_stride": before.get("stride"),
        "after_stride": after.get("stride"),
        "before_padding": before.get("padding"),
        "after_padding": after.get("padding"),
        "before_dilation": before.get("dilation"),
        "after_dilation": after.get("dilation"),
        "before_output_padding": before.get("output_padding"),
        "after_output_padding": after.get("output_padding"),
        "before_groups": before.get("groups"),
        "after_groups": after.get("groups"),
        "allowed_channel_change": bool(allowed_channel_change),
        "non_channel_shape_changed": bool(reasons),
        "violation_reason": ";".join(reasons),
        "passed": not reasons,
    }


def _norm_or_linear_row(name: str, before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    module_type = str(before.get("module_type", after.get("module_type", "")))
    if module_type in {"BatchNorm1d", "BatchNorm2d", "BatchNorm3d", "SyncBatchNorm"}:
        for key in ("eps", "momentum", "affine", "track_running_stats"):
            _same(before, after, key, reasons)
    elif module_type == "GroupNorm":
        for key in ("num_groups", "eps", "affine"):
            _same(before, after, key, reasons)
    elif module_type == "LayerNorm":
        for key in ("eps", "elementwise_affine"):
            _same(before, after, key, reasons)
    allowed_channel_change = before != after and not reasons
    return {
        "module_name": name,
        "module_type": module_type,
        "before_weight_shape": before.get("weight_shape", []),
        "after_weight_shape": after.get("weight_shape", []),
        "before_kernel_size": None,
        "after_kernel_size": None,
        "before_stride": None,
        "after_stride": None,
        "before_padding": None,
        "after_padding": None,
        "before_dilation": None,
        "after_dilation": None,
        "before_output_padding": None,
        "after_output_padding": None,
        "before_groups": before.get("num_groups"),
        "after_groups": after.get("num_groups"),
        "allowed_channel_change": bool(allowed_channel_change),
        "non_channel_shape_changed": bool(reasons),
        "violation_reason": ";".join(reasons),
        "passed": not reasons,
    }


def check_model_shape_invariants(
    before: Mapping[str, Mapping[str, Any]],
    after: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare snapshots and report non-channel shape/config violations."""

    rows: list[dict[str, Any]] = []
    for name in sorted(before):
        b = before[name]
        a = after.get(name)
        if a is None:
            rows.append(
                {
                    "module_name": name,
                    "module_type": b.get("module_type", ""),
                    "before_weight_shape": b.get("weight_shape", []),
                    "after_weight_shape": [],
                    "allowed_channel_change": False,
                    "non_channel_shape_changed": True,
                    "violation_reason": "module_missing_after_pruning",
                    "passed": False,
                }
            )
            continue
        if b.get("module_type") != a.get("module_type"):
            rows.append(
                {
                    "module_name": name,
                    "module_type": b.get("module_type", ""),
                    "before_weight_shape": b.get("weight_shape", []),
                    "after_weight_shape": a.get("weight_shape", []),
                    "allowed_channel_change": False,
                    "non_channel_shape_changed": True,
                    "violation_reason": f"module_type_changed:{b.get('module_type')}->{a.get('module_type')}",
                    "passed": False,
                }
            )
            continue
        if b.get("module_type") in {"Conv2d", "ConvTranspose2d"}:
            rows.append(_conv_row(name, b, a))
        else:
            rows.append(_norm_or_linear_row(name, b, a))
    violation_count = sum(1 for row in rows if row.get("non_channel_shape_changed"))
    return {
        "passed": violation_count == 0,
        "non_channel_shape_violation_count": violation_count,
        "num_modules_checked": len(rows),
        "rows": rows,
    }
