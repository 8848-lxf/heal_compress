#!/usr/bin/env python3
"""v10.6 Stage0 reblock8 + 8-aligned hidden-width pruning speed test."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[2]
_UNIAD = _ROOT.parent
for _p in (_UNIAD, _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tools.latency_lut.select_idle_gpu_for_latency import (  # noqa: E402
    collect_gpu_state,
    snapshot_to_dict,
    wait_for_idle_gpu,
)
from tools.latency_lut.stage0_grouped_conv_reblock_v105 import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIG,
    DEFAULT_HEAL_ROOT,
    conv2d_flops,
    latency_stats,
    reblock_grouped_conv_semantic_preserving,
)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[dict[str, Any]], *, fields: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        seen: set[str] = set()
        out_fields: list[str] = []
        for row in rows:
            for key in row:
                if key not in seen:
                    out_fields.append(key)
                    seen.add(key)
        fields = out_fields or ["empty"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        if rows:
            writer.writerows(rows)


def percentile(values: Sequence[float], pct: float) -> float:
    vals = sorted(float(v) for v in values)
    if not vals:
        return 0.0
    if len(vals) == 1:
        return vals[0]
    rank = (len(vals) - 1) * float(pct)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - rank) + vals[hi] * (rank - lo)


def _time_callable(fn, *, warmup: int, repeat: int, device: torch.device) -> dict[str, float]:
    with torch.no_grad():
        for _ in range(max(0, int(warmup))):
            fn()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            starters: list[torch.cuda.Event] = []
            enders: list[torch.cuda.Event] = []
            for _ in range(max(1, int(repeat))):
                starter = torch.cuda.Event(enable_timing=True)
                ender = torch.cuda.Event(enable_timing=True)
                starters.append(starter)
                enders.append(ender)
                starter.record()
                fn()
                ender.record()
            torch.cuda.synchronize(device)
            times = [float(s.elapsed_time(e)) for s, e in zip(starters, enders)]
        else:
            times = []
            for _ in range(max(1, int(repeat))):
                t0 = time.perf_counter()
                fn()
                times.append((time.perf_counter() - t0) * 1000.0)
    return latency_stats(times)


def build_groups16_no_prune_report() -> dict[str, Any]:
    return {
        "groups16_aligned_pruning_available": False,
        "available_prune_targets": [],
        "reason": "per_group already equals minimum 8-aligned floor",
        "groups": 16,
        "C_in": 128,
        "C_out": 128,
        "per_group": 8,
    }


def build_reblock8_keep_indices(keep_subgroups_per_new_group: Sequence[int] | None = None) -> list[int]:
    keep_subgroups = [0, 1] if keep_subgroups_per_new_group is None else [int(v) for v in keep_subgroups_per_new_group]
    if len(keep_subgroups) != 2:
        raise ValueError("v10.6 requires exactly two kept old 4-channel subgroups per new group")
    if sorted(set(keep_subgroups)) != sorted(keep_subgroups):
        raise ValueError("kept old subgroups must be unique")
    if any(v < 0 or v > 3 for v in keep_subgroups):
        raise ValueError("kept old subgroups must be in [0, 3]")
    keep: list[int] = []
    for new_group in range(8):
        base = new_group * 16
        for subgroup in keep_subgroups:
            start = base + subgroup * 4
            keep.extend(range(start, start + 4))
    return keep


def _new_conv_like(
    conv: nn.Conv2d,
    *,
    in_channels: int,
    out_channels: int,
    groups: int,
    bias: bool,
) -> nn.Conv2d:
    new_conv = nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=groups,
        bias=bias,
        padding_mode=conv.padding_mode,
        device=conv.weight.device,
        dtype=conv.weight.dtype,
    )
    new_conv.train(conv.training)
    return new_conv


def _prune_conv_out(conv: nn.Conv2d, keep_out: Sequence[int]) -> nn.Conv2d:
    keep = torch.as_tensor(list(keep_out), dtype=torch.long, device=conv.weight.device)
    new_conv = _new_conv_like(
        conv,
        in_channels=int(conv.in_channels),
        out_channels=int(keep.numel()),
        groups=int(conv.groups),
        bias=conv.bias is not None,
    )
    with torch.no_grad():
        new_conv.weight.copy_(conv.weight.index_select(0, keep))
        if conv.bias is not None and new_conv.bias is not None:
            new_conv.bias.copy_(conv.bias.index_select(0, keep))
    return new_conv


def _prune_conv_in(conv: nn.Conv2d, keep_in: Sequence[int]) -> nn.Conv2d:
    keep = torch.as_tensor(list(keep_in), dtype=torch.long, device=conv.weight.device)
    new_conv = _new_conv_like(
        conv,
        in_channels=int(keep.numel()),
        out_channels=int(conv.out_channels),
        groups=int(conv.groups),
        bias=conv.bias is not None,
    )
    with torch.no_grad():
        new_conv.weight.copy_(conv.weight.index_select(1, keep))
        if conv.bias is not None and new_conv.bias is not None:
            new_conv.bias.copy_(conv.bias)
    return new_conv


def _prune_batchnorm(bn: nn.modules.batchnorm._BatchNorm, keep_features: Sequence[int]) -> nn.Module:
    keep = torch.as_tensor(list(keep_features), dtype=torch.long, device=bn.weight.device if bn.affine else bn.running_mean.device)
    new_bn = nn.BatchNorm2d(
        int(keep.numel()),
        eps=bn.eps,
        momentum=bn.momentum,
        affine=bn.affine,
        track_running_stats=bn.track_running_stats,
        device=keep.device,
        dtype=(bn.weight.dtype if bn.affine else bn.running_mean.dtype),
    )
    with torch.no_grad():
        if bn.affine:
            new_bn.weight.copy_(bn.weight.index_select(0, keep))
            new_bn.bias.copy_(bn.bias.index_select(0, keep))
        if bn.track_running_stats:
            new_bn.running_mean.copy_(bn.running_mean.index_select(0, keep))
            new_bn.running_var.copy_(bn.running_var.index_select(0, keep))
            new_bn.num_batches_tracked.copy_(bn.num_batches_tracked)
    new_bn.train(bn.training)
    return new_bn


def _prune_reblocked8_conv2(conv: nn.Conv2d, hidden_keep_indices: Sequence[int]) -> nn.Conv2d:
    reblocked = reblock_grouped_conv_semantic_preserving(conv, groups_new=8)
    keep = [int(v) for v in hidden_keep_indices]
    if len(keep) != 64:
        raise ValueError("hidden_keep_indices must contain 64 channels")
    if any(v < 0 or v >= 128 for v in keep):
        raise ValueError("hidden_keep_indices out of range")

    grouped_keep: list[list[int]] = []
    for group_id in range(8):
        group_keep = [idx for idx in keep if group_id * 16 <= idx < (group_id + 1) * 16]
        if len(group_keep) != 8:
            raise ValueError("each reblocked group must keep exactly 8 channels")
        grouped_keep.append(group_keep)

    new_conv = _new_conv_like(reblocked, in_channels=64, out_channels=64, groups=8, bias=reblocked.bias is not None)
    with torch.no_grad():
        new_conv.weight.zero_()
        for group_id, group_keep in enumerate(grouped_keep):
            local_in = torch.as_tensor([idx - group_id * 16 for idx in group_keep], dtype=torch.long, device=reblocked.weight.device)
            for local_out, old_oc in enumerate(group_keep):
                new_oc = group_id * 8 + local_out
                new_conv.weight[new_oc].copy_(reblocked.weight[old_oc].index_select(0, local_in))
                if reblocked.bias is not None and new_conv.bias is not None:
                    new_conv.bias[new_oc].copy_(reblocked.bias[old_oc])
    return new_conv


def reblock8_then_prune_stage0_block_hidden_width(
    block: nn.Module,
    keep_subgroups_per_new_group: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Reblock a Stage0 bottleneck conv2 to groups=8, then prune hidden width 128->64."""
    required = ("conv1", "bn1", "conv2", "bn2", "conv3")
    missing = [name for name in required if not hasattr(block, name)]
    if missing:
        raise ValueError(f"block is missing required modules: {missing}")
    conv1 = block.conv1
    bn1 = block.bn1
    conv2 = block.conv2
    bn2 = block.bn2
    conv3 = block.conv3
    if not isinstance(conv1, nn.Conv2d) or not isinstance(conv2, nn.Conv2d) or not isinstance(conv3, nn.Conv2d):
        raise TypeError("conv1/conv2/conv3 must be nn.Conv2d")
    if not isinstance(bn1, nn.modules.batchnorm._BatchNorm) or not isinstance(bn2, nn.modules.batchnorm._BatchNorm):
        raise TypeError("bn1/bn2 must be BatchNorm modules")
    if int(conv2.groups) != 32 or int(conv2.in_channels) != 128 or int(conv2.out_channels) != 128:
        raise ValueError("conv2 must be Stage0 grouped conv: C_in=C_out=128, groups=32")
    if int(conv1.out_channels) != 128 or int(bn1.num_features) != 128 or int(bn2.num_features) != 128 or int(conv3.in_channels) != 128:
        raise ValueError("block hidden width must be 128 before pruning")

    hidden_keep = build_reblock8_keep_indices(keep_subgroups_per_new_group)
    old_conv3_out = int(conv3.out_channels)
    block.conv1 = _prune_conv_out(conv1, hidden_keep)
    block.bn1 = _prune_batchnorm(bn1, hidden_keep)
    block.conv2 = _prune_reblocked8_conv2(conv2, hidden_keep)
    block.bn2 = _prune_batchnorm(bn2, hidden_keep)
    block.conv3 = _prune_conv_in(conv3, hidden_keep)

    return {
        "hidden_width_before": 128,
        "hidden_width_after": 64,
        "groups_before": 32,
        "groups_after": 8,
        "per_group_before": 4,
        "per_group_after": 8,
        "hidden_keep_indices": hidden_keep,
        "keep_subgroups_per_new_group": [0, 1] if keep_subgroups_per_new_group is None else [int(v) for v in keep_subgroups_per_new_group],
        "conv3_out_channels_unchanged": int(block.conv3.out_channels) == old_conv3_out,
    }


def _conv_shape(conv: nn.Conv2d) -> dict[str, Any]:
    return {
        "in_channels": int(conv.in_channels),
        "out_channels": int(conv.out_channels),
        "groups": int(conv.groups),
        "kernel_size": list(conv.kernel_size),
        "stride": list(conv.stride),
        "padding": list(conv.padding),
        "dilation": list(conv.dilation),
        "bias": conv.bias is not None,
        "weight_shape": [int(v) for v in tuple(conv.weight.shape)],
    }


def _bn_shape(bn: nn.modules.batchnorm._BatchNorm) -> dict[str, Any]:
    return {
        "num_features": int(bn.num_features),
        "affine": bool(bn.affine),
        "track_running_stats": bool(bn.track_running_stats),
    }


def find_stage0_blocks(model: nn.Module) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, module in model.named_modules():
        if not name:
            continue
        if not all(hasattr(module, attr) for attr in ("conv1", "bn1", "conv2", "bn2", "conv3")):
            continue
        row: dict[str, Any] = {"block_name": name, "eligible": False, "reject_reason": ""}
        try:
            conv1, bn1, conv2, bn2, conv3 = module.conv1, module.bn1, module.conv2, module.bn2, module.conv3
            if not isinstance(conv1, nn.Conv2d) or not isinstance(conv2, nn.Conv2d) or not isinstance(conv3, nn.Conv2d):
                raise ValueError("conv1/conv2/conv3 are not Conv2d")
            if not isinstance(bn1, nn.modules.batchnorm._BatchNorm) or not isinstance(bn2, nn.modules.batchnorm._BatchNorm):
                raise ValueError("bn1/bn2 are not BatchNorm")
            row.update(
                {
                    "conv1": _conv_shape(conv1),
                    "bn1": _bn_shape(bn1),
                    "conv2": _conv_shape(conv2),
                    "bn2": _bn_shape(bn2),
                    "conv3": _conv_shape(conv3),
                }
            )
            checks = [
                (int(conv2.groups) == 32, "conv2 groups != 32"),
                (int(conv2.in_channels) == 128 and int(conv2.out_channels) == 128, "conv2 C_in/C_out != 128"),
                (int(conv1.out_channels) == 128, "conv1 out_channels != 128"),
                (int(bn1.num_features) == 128, "bn1 num_features != 128"),
                (int(bn2.num_features) == 128, "bn2 num_features != 128"),
                (int(conv3.in_channels) == 128, "conv3 in_channels != 128"),
            ]
            failures = [reason for ok, reason in checks if not ok]
            if failures:
                row["reject_reason"] = "; ".join(failures)
            else:
                row["eligible"] = True
        except Exception as exc:  # noqa: BLE001
            row["reject_reason"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    return rows


def _get_parent_and_leaf(model: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    parent_name, _, leaf = module_name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    return parent, leaf


def _reblock_stage0_conv2_modules(model: nn.Module, block_rows: Sequence[Mapping[str, Any]], *, groups_new: int) -> list[str]:
    changed: list[str] = []
    for row in block_rows:
        if not row.get("eligible"):
            continue
        conv_name = f"{row['block_name']}.conv2"
        conv = model.get_submodule(conv_name)
        if not isinstance(conv, nn.Conv2d):
            continue
        parent, leaf = _get_parent_and_leaf(model, conv_name)
        setattr(parent, leaf, reblock_grouped_conv_semantic_preserving(conv, groups_new=groups_new))
        changed.append(conv_name)
    return changed


def _prune_stage0_blocks(model: nn.Module, block_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for row in block_rows:
        if not row.get("eligible"):
            continue
        block_name = str(row["block_name"])
        block = model.get_submodule(block_name)
        report = reblock8_then_prune_stage0_block_hidden_width(block)
        reports.append({"block_name": block_name, **report})
    return reports


def _flatten_tensors(value: Any) -> list[torch.Tensor]:
    out: list[torch.Tensor] = []
    if torch.is_tensor(value):
        out.append(value.detach())
    elif isinstance(value, Mapping):
        for key in sorted(value):
            out.extend(_flatten_tensors(value[key]))
    elif isinstance(value, (list, tuple)):
        for item in value:
            out.extend(_flatten_tensors(item))
    return out


def _output_shapes(value: Any) -> list[list[int]]:
    return [[int(v) for v in tuple(t.shape)] for t in _flatten_tensors(value)]


def _outputs_finite(value: Any) -> bool:
    tensors = _flatten_tensors(value)
    return bool(tensors) and all(bool(torch.isfinite(t).all().item()) for t in tensors if t.is_floating_point())


def _variant_meta(variant: str) -> dict[str, Any]:
    if variant == "stage0_reblock16_no_prune":
        return {
            "hidden_width": 128,
            "groups": 16,
            "per_group": 8,
            "stage0_conv2_param_ratio": 2.0,
            "stage0_conv2_flops_ratio": 2.0,
            "stage0_block_hidden_width_ratio": 1.0,
            "notes": "groups16 no-prune control; no 8-aligned pruning space",
        }
    if variant == "stage0_reblock8_no_prune":
        return {
            "hidden_width": 128,
            "groups": 8,
            "per_group": 16,
            "stage0_conv2_param_ratio": 4.0,
            "stage0_conv2_flops_ratio": 4.0,
            "stage0_block_hidden_width_ratio": 1.0,
            "notes": "groups8 no-prune v10.5 control",
        }
    if variant == "stage0_reblock8_prune_pergroup8_all_blocks":
        return {
            "hidden_width": 64,
            "groups": 8,
            "per_group": 8,
            "stage0_conv2_param_ratio": 1.0,
            "stage0_conv2_flops_ratio": 1.0,
            "stage0_block_hidden_width_ratio": 0.5,
            "notes": "groups8 semantic reblock then keep first two old subgroups per new group",
        }
    return {
        "hidden_width": 128,
        "groups": 32,
        "per_group": 4,
        "stage0_conv2_param_ratio": 1.0,
        "stage0_conv2_flops_ratio": 1.0,
        "stage0_block_hidden_width_ratio": 1.0,
        "notes": "baseline",
    }


class MicroBottleneck(nn.Module):
    def __init__(self, *, hidden_width: int, groups: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(128, hidden_width, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_width)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(hidden_width, hidden_width, 3, padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_width)
        self.conv3 = nn.Conv2d(hidden_width, 128, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu(out + x)


def _module_to_env(module: nn.Module, x: torch.Tensor, *, layout: str) -> tuple[nn.Module, torch.Tensor]:
    if layout == "channels_last":
        module = module.to(memory_format=torch.channels_last)
        x = x.contiguous(memory_format=torch.channels_last)
    return module.eval(), x


def _block_flops(h: int, w: int, hidden_width: int, groups: int) -> float:
    return (
        conv2d_flops(1, h, w, 128, hidden_width, 1, 1)
        + conv2d_flops(1, h, w, hidden_width, hidden_width, 3, groups)
        + conv2d_flops(1, h, w, hidden_width, 128, 1, 1)
    )


def run_microbenchmarks(args: argparse.Namespace, out_dir: Path, device: torch.device) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    hws = [(100, 352), (50, 176)]
    envs = [("fp32", "nchw"), ("fp16", "channels_last")] if device.type == "cuda" else [("fp32", "nchw")]
    conv_variants = [
        ("baseline", 128, 32),
        ("stage0_reblock16_no_prune", 128, 16),
        ("stage0_reblock8_no_prune", 128, 8),
        ("stage0_reblock8_prune_pergroup8", 64, 8),
    ]
    block_variants = [
        ("baseline", 128, 32),
        ("stage0_reblock8_prune_pergroup8", 64, 8),
    ]
    baselines: dict[tuple[str, str, str, int, int], float] = {}
    for dtype_name, layout in envs:
        dtype = torch.float16 if dtype_name == "fp16" else torch.float32
        for h, w in hws:
            for variant, c, groups in conv_variants:
                conv = nn.Conv2d(c, c, 3, padding=1, groups=groups, bias=False).to(device=device, dtype=dtype).eval()
                x = torch.randn(1, c, h, w, device=device, dtype=dtype)
                conv, x = _module_to_env(conv, x, layout=layout)
                stats = _time_callable(lambda m=conv, inp=x: m(inp), warmup=args.latency_warmup, repeat=args.latency_repeat, device=device)
                key = ("conv2_only", dtype_name, layout, h, w)
                if variant == "baseline":
                    baselines[key] = stats["p50"]
                base = baselines.get(key, stats["p50"])
                flops = conv2d_flops(1, h, w, c, c, 3, groups)
                gflops = flops / 1e9
                rows.append(
                    {
                        "benchmark_type": "conv2_only",
                        "variant": variant,
                        "dtype": dtype_name,
                        "layout": layout,
                        "H": h,
                        "W": w,
                        "hidden_width": c,
                        "groups": groups,
                        "per_group": c // groups,
                        "latency_p50": stats["p50"],
                        "latency_mean": stats["mean"],
                        "latency_p90": stats["p90"],
                        "latency_p95": stats["p95"],
                        "speedup_vs_baseline": base / stats["p50"] if stats["p50"] > 0 else 0.0,
                        "throughput_GFLOPs_s": gflops / (stats["mean"] / 1000.0) if stats["mean"] > 0 else 0.0,
                        "latency_per_GFLOP": stats["mean"] / gflops if gflops > 0 else 0.0,
                    }
                )
            for variant, hidden_width, groups in block_variants:
                block = MicroBottleneck(hidden_width=hidden_width, groups=groups).to(device=device, dtype=dtype).eval()
                x = torch.randn(1, 128, h, w, device=device, dtype=dtype)
                block, x = _module_to_env(block, x, layout=layout)
                stats = _time_callable(lambda m=block, inp=x: m(inp), warmup=args.latency_warmup, repeat=args.latency_repeat, device=device)
                key = ("bottleneck_block", dtype_name, layout, h, w)
                if variant == "baseline":
                    baselines[key] = stats["p50"]
                base = baselines.get(key, stats["p50"])
                flops = _block_flops(h, w, hidden_width, groups)
                gflops = flops / 1e9
                rows.append(
                    {
                        "benchmark_type": "bottleneck_block",
                        "variant": variant,
                        "dtype": dtype_name,
                        "layout": layout,
                        "H": h,
                        "W": w,
                        "hidden_width": hidden_width,
                        "groups": groups,
                        "per_group": hidden_width // groups,
                        "latency_p50": stats["p50"],
                        "latency_mean": stats["mean"],
                        "latency_p90": stats["p90"],
                        "latency_p95": stats["p95"],
                        "speedup_vs_baseline": base / stats["p50"] if stats["p50"] > 0 else 0.0,
                        "throughput_GFLOPs_s": gflops / (stats["mean"] / 1000.0) if stats["mean"] > 0 else 0.0,
                        "latency_per_GFLOP": stats["mean"] / gflops if gflops > 0 else 0.0,
                    }
                )
    write_csv(out_dir / "microbench_stage0_reblock8_aligned_prune.csv", rows)
    return rows


def _build_variants(baseline: nn.Module, block_rows: Sequence[Mapping[str, Any]], device: torch.device) -> tuple[dict[str, nn.Module], dict[str, Any]]:
    variants: dict[str, nn.Module] = {"baseline": baseline}
    transform_report: dict[str, Any] = {"groups16_no_prune_rule": build_groups16_no_prune_report(), "variant_transforms": {}}

    reblock16 = copy.deepcopy(baseline).to(device).eval()
    changed16 = _reblock_stage0_conv2_modules(reblock16, block_rows, groups_new=16)
    variants["stage0_reblock16_no_prune"] = reblock16
    transform_report["variant_transforms"]["stage0_reblock16_no_prune"] = {"modules_reblocked": changed16}

    reblock8 = copy.deepcopy(baseline).to(device).eval()
    changed8 = _reblock_stage0_conv2_modules(reblock8, block_rows, groups_new=8)
    variants["stage0_reblock8_no_prune"] = reblock8
    transform_report["variant_transforms"]["stage0_reblock8_no_prune"] = {"modules_reblocked": changed8}

    pruned = copy.deepcopy(baseline).to(device).eval()
    prune_reports = _prune_stage0_blocks(pruned, block_rows)
    variants["stage0_reblock8_prune_pergroup8_all_blocks"] = pruned
    transform_report["variant_transforms"]["stage0_reblock8_prune_pergroup8_all_blocks"] = {"blocks_pruned": prune_reports}

    return variants, transform_report


def _latency_row(
    variant: str,
    stats: Mapping[str, float],
    baseline_stats: Mapping[str, float],
    reblock8_stats: Mapping[str, float] | None,
    num_blocks: int,
) -> dict[str, Any]:
    meta = _variant_meta(variant)
    row = {
        "variant": variant,
        "hidden_width": meta["hidden_width"],
        "groups": meta["groups"],
        "per_group": meta["per_group"],
        "num_stage0_blocks_modified": 0 if variant == "baseline" else num_blocks,
        "stage0_conv2_param_ratio": meta["stage0_conv2_param_ratio"],
        "stage0_conv2_flops_ratio": meta["stage0_conv2_flops_ratio"],
        "stage0_block_hidden_width_ratio": meta["stage0_block_hidden_width_ratio"],
        "forward_latency_p50": stats["p50"],
        "forward_latency_mean": stats["mean"],
        "forward_latency_p90": stats["p90"],
        "forward_latency_p95": stats["p95"],
        "speedup_p50_vs_baseline": baseline_stats["p50"] / stats["p50"] if stats["p50"] > 0 else 0.0,
        "speedup_mean_vs_baseline": baseline_stats["mean"] / stats["mean"] if stats["mean"] > 0 else 0.0,
        "speedup_p90_vs_baseline": baseline_stats["p90"] / stats["p90"] if stats["p90"] > 0 else 0.0,
        "speedup_p95_vs_baseline": baseline_stats["p95"] / stats["p95"] if stats["p95"] > 0 else 0.0,
        "speedup_p50_vs_reblock8_no_prune": "",
        "notes": meta["notes"],
    }
    if reblock8_stats is not None:
        row["speedup_p50_vs_reblock8_no_prune"] = reblock8_stats["p50"] / stats["p50"] if stats["p50"] > 0 else 0.0
    return row


def run_real_model_experiment(args: argparse.Namespace, out_dir: Path, device: torch.device, gpu_samples: list[dict[str, Any]], selected_index: int | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    from heal_compress.utils.model_utils import resolve_device
    from heal_compress.pruning.model_io import load_heal_model, setup_logger

    logger = setup_logger(out_dir)
    args.device = str(device)
    device = torch.device(resolve_device(args.device))
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model, adapter = load_heal_model(args, device, logger)
    model.eval()

    block_rows = find_stage0_blocks(model)
    eligible_blocks = [row for row in block_rows if row.get("eligible")]
    write_json(out_dir / "stage0_blocks_report.json", block_rows)
    if not eligible_blocks:
        raise RuntimeError("stage0_blocks_not_found")

    variants, transform_report = _build_variants(model, eligible_blocks, device)
    write_json(out_dir / "stage0_reblock8_aligned_prune_transform_report.json", transform_report)

    sample = adapter.build_synthetic_batch(model)
    smoke_rows: list[dict[str, Any]] = []
    for variant, variant_model in variants.items():
        try:
            with torch.no_grad():
                output = adapter.forward_for_task(variant_model, sample)
            smoke_rows.append(
                {
                    "variant": variant,
                    "passed": True,
                    "output_shapes": _output_shapes(output),
                    "output_finite": _outputs_finite(output),
                    "failure_reason": "",
                }
            )
        except Exception as exc:  # noqa: BLE001
            smoke_rows.append(
                {
                    "variant": variant,
                    "passed": False,
                    "output_shapes": [],
                    "output_finite": False,
                    "failure_reason": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            )
    write_json(out_dir / "forward_smoke_report.json", smoke_rows)
    if any(not row["passed"] for row in smoke_rows):
        raise RuntimeError("forward_smoke_failed")

    sanity_rows: list[dict[str, Any]] = []
    for variant, variant_model in variants.items():
        all_passed = True
        all_finite = True
        failure = ""
        for _ in range(max(1, int(args.sanity_samples))):
            try:
                sanity_sample = adapter.build_synthetic_batch(model)
                with torch.no_grad():
                    output = adapter.forward_for_task(variant_model, sanity_sample)
                all_finite = all_finite and _outputs_finite(output)
            except Exception as exc:  # noqa: BLE001
                all_passed = False
                all_finite = False
                failure = f"{type(exc).__name__}: {exc}"
                break
        sanity_rows.append(
            {
                "variant": variant,
                "num_samples": int(args.sanity_samples),
                "forward_all_passed": all_passed,
                "output_finite": all_finite,
                "AP@0.30": None,
                "notes": failure or "synthetic finite-output sanity; no AP recovery eval in v10.6",
            }
        )
    write_json(out_dir / "stage0_reblock8_aligned_prune_sanity.json", sanity_rows)

    raw_stats: dict[str, dict[str, float]] = {}
    for idx, (variant, variant_model) in enumerate(variants.items()):
        raw_stats[variant] = _time_callable(
            lambda m=variant_model: adapter.forward_for_task(m, sample),
            warmup=args.latency_warmup,
            repeat=args.latency_repeat,
            device=device,
        )
        if selected_index is not None:
            gpu_samples.append({"sample_reason": f"after_latency_{variant}", **collect_gpu_state(selected_index)})

    baseline_stats = raw_stats["baseline"]
    reblock8_stats = raw_stats.get("stage0_reblock8_no_prune")
    latency_rows = [
        _latency_row(variant, stats, baseline_stats, reblock8_stats, len(eligible_blocks))
        for variant, stats in raw_stats.items()
    ]
    write_csv(out_dir / "stage0_reblock8_aligned_prune_latency.csv", latency_rows)
    return latency_rows, smoke_rows, sanity_rows, {"block_rows": block_rows, "transform_report": transform_report}


def _median_speed(rows: Sequence[Mapping[str, Any]], *, benchmark_type: str, variant: str) -> float:
    vals = [
        float(row.get("speedup_vs_baseline") or 0.0)
        for row in rows
        if row.get("benchmark_type") == benchmark_type and row.get("variant") == variant
    ]
    return statistics.median(vals) if vals else 0.0


def _row_for(rows: Sequence[Mapping[str, Any]], variant: str) -> Mapping[str, Any]:
    return next((row for row in rows if row.get("variant") == variant), {})


def write_verdict(
    out_dir: Path,
    *,
    latency_rows: Sequence[dict[str, Any]],
    micro_rows: Sequence[dict[str, Any]],
    smoke_rows: Sequence[dict[str, Any]],
    sanity_rows: Sequence[dict[str, Any]],
    failure: str,
    contamination_risk: bool,
) -> None:
    pruned_latency = _row_for(latency_rows, "stage0_reblock8_prune_pergroup8_all_blocks")
    reblock8_latency = _row_for(latency_rows, "stage0_reblock8_no_prune")
    reblock16_latency = _row_for(latency_rows, "stage0_reblock16_no_prune")
    pruned_smoke = _row_for(smoke_rows, "stage0_reblock8_prune_pergroup8_all_blocks")
    pruned_sanity = _row_for(sanity_rows, "stage0_reblock8_prune_pergroup8_all_blocks")
    conv2_speed = _median_speed(micro_rows, benchmark_type="conv2_only", variant="stage0_reblock8_prune_pergroup8")
    block_speed = _median_speed(micro_rows, benchmark_type="bottleneck_block", variant="stage0_reblock8_prune_pergroup8")
    whole_speed = float(pruned_latency.get("speedup_p50_vs_baseline") or 0.0)
    vs_reblock8 = pruned_latency.get("speedup_p50_vs_reblock8_no_prune", "")
    lines = [
        "# Stage0 Reblock8 8-Aligned Pruning v10.6 Verdict",
        "",
        f"failure: {failure or 'none'}",
        f"latency_contamination_risk: {str(bool(contamination_risk)).lower()}",
        "",
        "1. groups16 reblock 后是否还有 8-aligned per-group pruning 空间？没有。groups16 already has per_group=8, so groups16_aligned_pruning_available=false and available_prune_targets=[].",
        "2. groups8 reblock 后剪到 per_group=8 是否结构合法？是，脚本将 hidden width 128 -> 64, groups=8, per_group=8, and keeps block external output unchanged.",
        f"3. stage0_reblock8_prune_pergroup8_all_blocks forward smoke: {bool(pruned_smoke.get('passed', False))}.",
        f"4. conv2-only microbenchmark median speedup vs baseline groups32/per_group4: {conv2_speed:.6f}.",
        f"5. bottleneck block microbenchmark median speedup vs baseline: {block_speed:.6f}.",
        f"6. whole-model forward latency speedup_p50 vs baseline: {whole_speed:.6f}.",
        f"7. aligned pruning speedup_p50 vs stage0_reblock8_no_prune: {vs_reblock8}.",
        "8. If speedup is >1, group merge + aligned per-group pruning is a Stage0 latency candidate; if not, local shape repair does not pay back in PyTorch/cuDNN.",
        "9. If no whole-model speedup, likely causes are Stage0 conv2 not being the model-level bottleneck, hidden-width savings too local, or cuDNN still not favoring this local structure enough.",
        "10. Decoder recommendation: do not add this to precision-sensitive formal decoder as a hard rule; at most keep it as a latency candidate requiring accuracy recovery checks.",
        "",
        "Controls:",
        f"- stage0_reblock16_no_prune speedup_p50_vs_baseline: {reblock16_latency.get('speedup_p50_vs_baseline', '')}",
        f"- stage0_reblock8_no_prune speedup_p50_vs_baseline: {reblock8_latency.get('speedup_p50_vs_baseline', '')}",
        f"- sanity finite output for pruned variant: {pruned_sanity.get('output_finite', '')}",
    ]
    (out_dir / "stage0_reblock8_aligned_prune_verdict.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="v10.6 Stage0 reblock8 aligned pruning speed test")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--model-config", default=DEFAULT_CONFIG)
    parser.add_argument("--heal-root", default=DEFAULT_HEAL_ROOT)
    parser.add_argument("--device", default="")
    parser.add_argument("--auto-select-idle-gpu", action="store_true")
    parser.add_argument("--max-gpu-utilization", type=int, default=5)
    parser.add_argument("--max-gpu-memory-ratio", type=float, default=0.20)
    parser.add_argument("--wait-timeout-minutes", type=float, default=30.0)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--latency-warmup", type=int, default=50)
    parser.add_argument("--latency-repeat", type=int, default=300)
    parser.add_argument("--sanity-samples", type=int, default=50)
    parser.add_argument("--output-dir", default="outputs/latency_lut/stage0_reblock8_aligned_prune_v106")
    return parser.parse_args(argv)


def _select_device(args: argparse.Namespace, out_dir: Path) -> tuple[torch.device, int | None, list[dict[str, Any]]]:
    if args.auto_select_idle_gpu:
        selected, reason, attempts = wait_for_idle_gpu(
            max_utilization=args.max_gpu_utilization,
            max_memory_ratio=args.max_gpu_memory_ratio,
            wait_timeout_minutes=args.wait_timeout_minutes,
            poll_seconds=args.poll_seconds,
        )
        write_json(
            out_dir / "selected_gpu.json",
            {
                "success": True,
                "selected_gpu_index": selected.index,
                "selected_gpu": snapshot_to_dict(selected),
                "selected_gpu_reason": reason,
                "attempts": attempts,
            },
        )
        device = torch.device(f"cuda:{selected.index}" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            torch.cuda.set_device(device)
        return device, int(selected.index), attempts
    device = torch.device(args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.cuda.set_device(device)
        return device, int(device.index or 0), []
    return device, None, []


def _contamination_report(gpu_samples: Sequence[Mapping[str, Any]], selected_index: int | None) -> dict[str, Any]:
    current_pid = os.getpid()
    external_processes: list[dict[str, Any]] = []
    for sample in gpu_samples:
        for process in sample.get("running_processes", []) or []:
            try:
                pid = int(process.get("pid"))
            except Exception:
                pid = -1
            if pid != current_pid:
                external_processes.append(dict(process))
    return {
        "selected_gpu_index": selected_index,
        "current_pid": current_pid,
        "latency_contamination_risk": bool(external_processes),
        "external_processes_seen": external_processes,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "run_config.json", vars(args))

    latency_rows: list[dict[str, Any]] = []
    micro_rows: list[dict[str, Any]] = []
    smoke_rows: list[dict[str, Any]] = []
    sanity_rows: list[dict[str, Any]] = []
    gpu_samples: list[dict[str, Any]] = []
    selected_index: int | None = None
    failure = ""
    contamination_risk = False
    try:
        device, selected_index, _ = _select_device(args, out_dir)
        if selected_index is not None:
            before = collect_gpu_state(selected_index)
            write_json(out_dir / "gpu_state_before.json", before)
            gpu_samples.append({"sample_reason": "before", **before})
        else:
            write_json(out_dir / "gpu_state_before.json", {})

        micro_rows = run_microbenchmarks(args, out_dir, device)
        if selected_index is not None:
            gpu_samples.append({"sample_reason": "after_microbench", **collect_gpu_state(selected_index)})
        latency_rows, smoke_rows, sanity_rows, _ = run_real_model_experiment(args, out_dir, device, gpu_samples, selected_index)

        if selected_index is not None:
            after = collect_gpu_state(selected_index)
            write_json(out_dir / "gpu_state_after.json", after)
            gpu_samples.append({"sample_reason": "after", **after})
        else:
            write_json(out_dir / "gpu_state_after.json", {})
        write_json(out_dir / "gpu_state_during_samples.json", gpu_samples)
        contamination = _contamination_report(gpu_samples, selected_index)
        contamination_risk = bool(contamination["latency_contamination_risk"])
        write_json(out_dir / "gpu_contamination_report.json", contamination)
        write_json(out_dir / "failure_report.json", {"success": True, "failure_reason": "", "traceback": ""})
    except Exception as exc:  # noqa: BLE001
        failure = f"{type(exc).__name__}: {exc}"
        write_json(out_dir / "failure_report.json", {"success": False, "failure_reason": failure, "traceback": traceback.format_exc()})
        if selected_index is not None:
            try:
                after = collect_gpu_state(selected_index)
                write_json(out_dir / "gpu_state_after.json", after)
                gpu_samples.append({"sample_reason": "after_failure", **after})
                write_json(out_dir / "gpu_state_during_samples.json", gpu_samples)
                contamination = _contamination_report(gpu_samples, selected_index)
                contamination_risk = bool(contamination["latency_contamination_risk"])
                write_json(out_dir / "gpu_contamination_report.json", contamination)
            except Exception:
                pass
    write_verdict(
        out_dir,
        latency_rows=latency_rows,
        micro_rows=micro_rows,
        smoke_rows=smoke_rows,
        sanity_rows=sanity_rows,
        failure=failure,
        contamination_risk=contamination_risk,
    )
    print(json.dumps({"success": not failure, "failure": failure, "output_dir": str(out_dir)}, indent=2, ensure_ascii=False))
    return 0 if not failure else 1


if __name__ == "__main__":
    raise SystemExit(main())
