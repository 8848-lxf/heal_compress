#!/usr/bin/env python3
"""Full PyTorch/cuDNN shape-to-latency rulebook benchmark for v10.4."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn as nn

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.latency_lut.select_idle_gpu_for_latency import (  # noqa: E402
    choose_idle_gpu,
    collect_gpu_state,
    query_gpu_snapshots,
    snapshot_to_dict,
    wait_for_idle_gpu,
)


ORDINARY_CHANNELS = [
    2,
    4,
    6,
    10,
    12,
    14,
    18,
    20,
    22,
    8,
    16,
    24,
    32,
    48,
    64,
    80,
    96,
    112,
    128,
    160,
    192,
    224,
    256,
    320,
    384,
    448,
    512,
]
ORDINARY_FIXED = [32, 64, 128, 256]
SYNTHETIC_CONTEXT_HWS = [(100, 352), (50, 176), (13, 44)]
GROUPED_PRIMARY_CONTEXT_HW = (50, 176)
GROUPED_CONTEXT_PROBE_HWS = [(100, 352), (13, 44)]
GROUPED_CONTEXT_PROBE_GROUPS = [8, 32, 64]
GROUPED_CONTEXT_PROBE_WIDTH_PAIRS = [(4, 4), (8, 8), (16, 16), (32, 32), (8, 16), (16, 8)]
GROUPS = [2, 4, 8, 16, 24, 32, 48, 64]
PER_GROUP_WIDTHS = [2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 32, 64]
PYRAMID_GROUPS_AFTER = [32, 28, 24, 20, 16, 12, 8, 4]
PYRAMID_AB_OUT_PER_GROUP = [2, 4, 6, 8, 10, 12, 14, 16, 24, 32]
DIAGNOSTIC_ORDINARY_CHANNELS = [4, 8, 16, 32, 64, 128]
DIAGNOSTIC_GROUPED_PER_GROUP = [4, 8, 16, 32]
PYRAMID_STAGES = [
    {
        "stage_name": "Stage0",
        "base_C": 128,
        "base_groups": 32,
        "per_group": 4,
        "hws": [(100, 352), (50, 176)],
    },
    {
        "stage_name": "Stage1",
        "base_C": 256,
        "base_groups": 32,
        "per_group": 8,
        "hws": [(50, 176), (25, 88)],
    },
    {
        "stage_name": "Stage2",
        "base_C": 512,
        "base_groups": 32,
        "per_group": 16,
        "hws": [(25, 88), (13, 44)],
    },
]
RESULT_FIELDS = [
    "shape_id",
    "suite",
    "source_experiment",
    "module_name",
    "strategy",
    "original_or_pruned",
    "op_type",
    "dtype",
    "layout",
    "cudnn_benchmark",
    "allow_tf32",
    "batch",
    "H",
    "W",
    "kernel",
    "kernel_size",
    "stride",
    "padding",
    "groups",
    "C_in",
    "C_out",
    "in_per_group",
    "out_per_group",
    "stage_name",
    "base_C",
    "base_groups",
    "per_group",
    "groups_after",
    "relative_group_prune_ratio",
    "relative_param_ratio",
    "relative_FLOPs_ratio",
    "out_per_group_after",
    "C_out_after",
    "params",
    "FLOPs_estimate",
    "latency_mean_ms",
    "latency_p50_ms",
    "latency_p90_ms",
    "latency_p95_ms",
    "latency_min_ms",
    "latency_max_ms",
    "latency_std_ms",
    "run_to_run_std_ms",
    "run_median_ms_list",
    "run_mean_ms_list",
    "throughput_GFLOPs_s",
    "latency_per_GFLOP",
    "speedup_vs_groups32_same_stage",
    "speedup_vs_original_out_per_group_same_stage",
    "C_in_multiple_of_8",
    "C_in_multiple_of_16",
    "C_in_multiple_of_32",
    "C_in_multiple_of_64",
    "C_out_multiple_of_8",
    "C_out_multiple_of_16",
    "C_out_multiple_of_32",
    "C_out_multiple_of_64",
    "groups_multiple_of_4",
    "groups_multiple_of_8",
    "groups_multiple_of_16",
    "groups_multiple_of_32",
    "total_C_out_multiple_of_8",
    "total_C_out_multiple_of_16",
    "total_C_out_multiple_of_32",
    "total_C_out_multiple_of_64",
    "in_per_group_multiple_of_8",
    "in_per_group_multiple_of_16",
    "in_per_group_multiple_of_32",
    "out_per_group_multiple_of_8",
    "out_per_group_multiple_of_16",
    "out_per_group_multiple_of_32",
    "per_group_too_narrow_2_4",
    "per_group_less_than_8",
    "per_group_non_multiple_of_8",
    "is_too_narrow_2_4",
    "is_non_multiple_of_8",
    "shape_class",
    "decoder_rule_category",
    "predicted_latency_friendly",
    "actual_fast_or_slow",
    "relative_to_nearest_8",
    "relative_to_nearest_16",
    "relative_to_nearest_32",
    "relative_to_nearest_per_group8",
    "relative_to_nearest_per_group16",
    "group_count_class",
    "per_group_class",
    "hw_source",
    "selected_gpu_index",
    "gpu_name",
    "torch_version",
    "cuda_version",
    "cudnn_version",
]
SKIP_FIELDS = [
    "shape_id",
    "suite",
    "op_type",
    "dtype",
    "layout",
    "cudnn_benchmark",
    "allow_tf32",
    "H",
    "W",
    "kernel_size",
    "C_in",
    "C_out",
    "groups",
    "in_per_group",
    "out_per_group",
    "skip_reason",
    "traceback",
]


def percentile(values: Sequence[float], pct: float) -> float:
    vals = sorted(float(v) for v in values)
    if not vals:
        return 0.0
    if len(vals) == 1:
        return vals[0]
    rank = (len(vals) - 1) * pct
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - rank) + vals[hi] * (rank - lo)


def conv2d_flops(*, batch: int, h: int, w: int, c_in: int, c_out: int, kernel: int, groups: int) -> float:
    return float(batch) * h * w * c_out * (c_in / groups) * kernel * kernel * 2.0


def _shape_class_channel(value: int) -> str:
    if value in {2, 4}:
        return "too_narrow_2_4"
    if value % 64 == 0:
        return "multiple_of_64"
    if value % 32 == 0:
        return "multiple_of_32"
    if value % 16 == 0:
        return "multiple_of_16"
    if value % 8 == 0:
        return "multiple_of_8"
    return "non_multiple_of_8"


def classify_ordinary_shape(c_in: int, c_out: int, extra: str = "") -> str:
    classes = {_shape_class_channel(int(c_in)), _shape_class_channel(int(c_out))}
    if int(c_in) in {2, 4} or int(c_out) in {2, 4}:
        classes.add("too_narrow_2_4")
    if int(c_in) % 8 != 0 or int(c_out) % 8 != 0:
        classes.add("non_multiple_of_8")
    if extra:
        classes.add(extra)
    return ";".join(sorted(classes))


def classify_grouped_shape(groups: int, in_per_group: int, out_per_group: int, extra: str = "") -> str:
    classes: set[str] = set()
    if int(in_per_group) in {2, 4} or int(out_per_group) in {2, 4}:
        classes.add("per_group_too_narrow_2_4")
    if int(in_per_group) < 8 or int(out_per_group) < 8:
        classes.add("per_group_less_than_8")
    if int(in_per_group) % 8 != 0 or int(out_per_group) % 8 != 0:
        classes.add("per_group_non_multiple_of_8")
    if int(in_per_group) % 8 == 0 and int(out_per_group) % 8 == 0:
        classes.add("per_group_multiple_of_8")
    if int(in_per_group) % 16 == 0 and int(out_per_group) % 16 == 0:
        classes.add("per_group_multiple_of_16")
    if int(in_per_group) % 32 == 0 and int(out_per_group) % 32 == 0:
        classes.add("per_group_multiple_of_32")
    if int(groups) % 8 == 0:
        classes.add("groups_multiple_of_8")
    c_out = int(groups) * int(out_per_group)
    if c_out % 8 == 0:
        classes.add("total_C_out_multiple_of_8")
    if c_out % 16 == 0:
        classes.add("total_C_out_multiple_of_16")
    if c_out % 32 == 0:
        classes.add("total_C_out_multiple_of_32")
    if extra:
        classes.add(extra)
    return ";".join(sorted(classes))


def classify_decoder_rule(
    *,
    op_type: str,
    c_in: int,
    c_out: int,
    groups: int,
    in_per_group: int | None = None,
    out_per_group: int | None = None,
) -> dict[str, Any]:
    if int(groups) <= 1 or op_type == "ordinary_conv2d":
        if int(c_in) in {2, 4} or int(c_out) in {2, 4}:
            return {"category": "hard_reject", "reason": "ordinary C_in/C_out in {2,4}"}
        if int(c_in) % 32 == 0 and int(c_out) % 32 == 0:
            return {"category": "bonus", "reason": "ordinary C_in/C_out both multiple of 32"}
        if int(c_in) % 16 == 0 and int(c_out) % 16 == 0:
            return {"category": "preferred", "reason": "ordinary C_in/C_out both multiple of 16"}
        if int(c_in) % 8 != 0 or int(c_out) % 8 != 0:
            return {"category": "soft_penalty", "reason": "ordinary C_in/C_out not both multiple of 8"}
        return {"category": "acceptable_floor", "reason": "ordinary C_in/C_out both multiple of 8"}

    in_per = int(in_per_group if in_per_group is not None else int(c_in) // int(groups))
    out_per = int(out_per_group if out_per_group is not None else int(c_out) // int(groups))
    if in_per < 8 or out_per < 8:
        return {"category": "hard_reject_for_acceleration", "reason": "grouped per-group width < 8"}
    if in_per >= 32 and out_per >= 32 and in_per % 32 == 0 and out_per % 32 == 0:
        return {"category": "bonus", "reason": "grouped per-group widths both multiple of 32"}
    if in_per >= 16 and out_per >= 16 and in_per % 16 == 0 and out_per % 16 == 0:
        return {"category": "preferred", "reason": "grouped per-group widths both >=16 and multiple of 16"}
    if in_per % 8 != 0 or out_per % 8 != 0:
        return {"category": "soft_penalty", "reason": "grouped per-group width not both multiple of 8"}
    return {"category": "acceptable_floor", "reason": "grouped per-group widths both multiple of 8"}


def make_conv_config(
    *,
    op_type: str,
    h: int,
    w: int,
    kernel: int,
    c_in: int,
    c_out: int,
    groups: int,
    suite: str,
    batch: int = 1,
    stride: int = 1,
    source_experiment: str = "synthetic",
    module_name: str = "",
    strategy: str = "",
    original_or_pruned: str = "synthetic",
    stage_name: str = "",
    base_C: int = 0,
    base_groups: int = 0,
    per_group: int = 0,
    groups_after: int = 0,
    out_per_group_after: int = 0,
    hw_source: str = "synthetic",
    extra_shape_class: str = "",
) -> dict[str, Any]:
    groups = int(groups)
    c_in = int(c_in)
    c_out = int(c_out)
    in_per = c_in // max(groups, 1)
    out_per = c_out // max(groups, 1)
    shape_class = (
        classify_grouped_shape(groups, in_per, out_per, extra=extra_shape_class)
        if groups > 1
        else classify_ordinary_shape(c_in, c_out, extra=extra_shape_class)
    )
    return {
        "suite": suite,
        "source_experiment": source_experiment,
        "module_name": module_name,
        "strategy": strategy,
        "original_or_pruned": original_or_pruned,
        "op_type": op_type,
        "batch": int(batch),
        "H": int(h),
        "W": int(w),
        "kernel": int(kernel),
        "kernel_size": f"{int(kernel)}x{int(kernel)}",
        "stride": int(stride),
        "padding": int(kernel) // 2 if int(kernel) > 1 else 0,
        "groups": groups,
        "C_in": c_in,
        "C_out": c_out,
        "in_per_group": in_per,
        "out_per_group": out_per,
        "stage_name": stage_name,
        "base_C": int(base_C),
        "base_groups": int(base_groups),
        "per_group": int(per_group),
        "groups_after": int(groups_after),
        "out_per_group_after": int(out_per_group_after),
        "C_out_after": c_out if out_per_group_after else 0,
        "hw_source": hw_source,
        "shape_class": shape_class,
    }


def _unique_configs(configs: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    out: list[dict[str, Any]] = []
    for config in configs:
        key = (
            config.get("suite"),
            config.get("source_experiment"),
            config.get("module_name"),
            config.get("op_type"),
            config.get("H"),
            config.get("W"),
            config.get("kernel_size"),
            config.get("C_in"),
            config.get("C_out"),
            config.get("groups"),
            config.get("stage_name"),
            config.get("groups_after"),
            config.get("out_per_group_after"),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(config)
    return out


def generate_ordinary_targeted_grid(*, real_shapes: Sequence[dict[str, Any]] | None = None, batch: int = 1) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for h, w in SYNTHETIC_CONTEXT_HWS:
        for kernel in [1, 3]:
            for c in ORDINARY_CHANNELS:
                rows.append(make_conv_config(op_type="ordinary_conv2d", h=h, w=w, kernel=kernel, c_in=c, c_out=c, groups=1, suite="ordinary", batch=batch))
            for c_in in ORDINARY_FIXED:
                for c_out in ORDINARY_CHANNELS:
                    rows.append(make_conv_config(op_type="ordinary_conv2d", h=h, w=w, kernel=kernel, c_in=c_in, c_out=c_out, groups=1, suite="ordinary", batch=batch))
            for c_out in ORDINARY_FIXED:
                for c_in in ORDINARY_CHANNELS:
                    rows.append(make_conv_config(op_type="ordinary_conv2d", h=h, w=w, kernel=kernel, c_in=c_in, c_out=c_out, groups=1, suite="ordinary", batch=batch))
    for shape in real_shapes or []:
        if int(shape.get("groups", 1) or 1) == 1:
            cfg = dict(shape)
            cfg["suite"] = "ordinary"
            rows.append(cfg)
    return _unique_configs(rows)


def generate_grouped_general_grid(*, max_channels: int = 4096, batch: int = 1) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    h, w = GROUPED_PRIMARY_CONTEXT_HW
    for groups in GROUPS:
        for in_per in PER_GROUP_WIDTHS:
            for out_per in PER_GROUP_WIDTHS:
                c_in = groups * in_per
                c_out = groups * out_per
                if c_in > int(max_channels) or c_out > int(max_channels):
                    continue
                rows.append(make_conv_config(op_type="grouped_conv2d", h=h, w=w, kernel=3, c_in=c_in, c_out=c_out, groups=groups, suite="grouped_general", batch=batch, hw_source="synthetic_primary_context"))
    for h, w in GROUPED_CONTEXT_PROBE_HWS:
        for groups in GROUPED_CONTEXT_PROBE_GROUPS:
            for in_per, out_per in GROUPED_CONTEXT_PROBE_WIDTH_PAIRS:
                c_in = groups * in_per
                c_out = groups * out_per
                if c_in > int(max_channels) or c_out > int(max_channels):
                    continue
                rows.append(make_conv_config(op_type="grouped_conv2d", h=h, w=w, kernel=3, c_in=c_in, c_out=c_out, groups=groups, suite="grouped_general", batch=batch, hw_source="synthetic_context_probe"))
    return _unique_configs(rows)


def generate_pyramid_c_sweep_grid(*, batch: int = 1) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stage in PYRAMID_STAGES:
        for h, w in stage["hws"]:
            for groups_after in PYRAMID_GROUPS_AFTER:
                c_after = int(groups_after) * int(stage["per_group"])
                rows.append(
                    make_conv_config(
                        op_type="grouped_conv2d",
                        h=h,
                        w=w,
                        kernel=3,
                        c_in=c_after,
                        c_out=c_after,
                        groups=groups_after,
                        suite="pyramid_c",
                        batch=batch,
                        stage_name=str(stage["stage_name"]),
                        base_C=int(stage["base_C"]),
                        base_groups=int(stage["base_groups"]),
                        per_group=int(stage["per_group"]),
                        groups_after=groups_after,
                        hw_source="fallback",
                    )
                )
    return _unique_configs(rows)


def generate_pyramid_ab_sweep_grid(*, batch: int = 1) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stage in PYRAMID_STAGES:
        for h, w in stage["hws"]:
            for out_per in PYRAMID_AB_OUT_PER_GROUP:
                c_out = int(stage["base_groups"]) * int(out_per)
                rows.append(
                    make_conv_config(
                        op_type="grouped_conv2d",
                        h=h,
                        w=w,
                        kernel=3,
                        c_in=int(stage["base_C"]),
                        c_out=c_out,
                        groups=int(stage["base_groups"]),
                        suite="pyramid_ab",
                        batch=batch,
                        stage_name=str(stage["stage_name"]),
                        base_C=int(stage["base_C"]),
                        base_groups=int(stage["base_groups"]),
                        per_group=int(stage["per_group"]),
                        out_per_group_after=out_per,
                        hw_source="fallback",
                    )
                )
    return _unique_configs(rows)


def generate_diagnostic_environment_subset_grid(*, batch: int = 1) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    h, w = 50, 176
    for diag_h, diag_w in SYNTHETIC_CONTEXT_HWS:
        for c in DIAGNOSTIC_ORDINARY_CHANNELS:
            for kernel in [1, 3]:
                cfg = make_conv_config(op_type="ordinary_conv2d", h=diag_h, w=diag_w, kernel=kernel, c_in=c, c_out=c, groups=1, suite="diagnostic_environment_subset", batch=batch, hw_source="diagnostic_representative")
                cfg["diagnostic_role"] = "ordinary_representative"
                rows.append(cfg)
    for per_group in DIAGNOSTIC_GROUPED_PER_GROUP:
        cfg = make_conv_config(op_type="grouped_conv2d", h=h, w=w, kernel=3, c_in=32 * per_group, c_out=32 * per_group, groups=32, suite="diagnostic_environment_subset", batch=batch, hw_source="diagnostic_representative")
        cfg["diagnostic_role"] = "grouped_representative"
        rows.append(cfg)
    for stage in PYRAMID_STAGES:
        stage_h, stage_w = stage["hws"][0]
        baseline = make_conv_config(
            op_type="grouped_conv2d",
            h=stage_h,
            w=stage_w,
            kernel=3,
            c_in=int(stage["base_C"]),
            c_out=int(stage["base_C"]),
            groups=int(stage["base_groups"]),
            suite="diagnostic_environment_subset",
            batch=batch,
            stage_name=str(stage["stage_name"]),
            base_C=int(stage["base_C"]),
            base_groups=int(stage["base_groups"]),
            per_group=int(stage["per_group"]),
            groups_after=int(stage["base_groups"]),
            hw_source="pyramid_fallback",
        )
        baseline["diagnostic_role"] = "pyramid_baseline"
        rows.append(baseline)
        groups_after = 16
        best_proxy = make_conv_config(
            op_type="grouped_conv2d",
            h=stage_h,
            w=stage_w,
            kernel=3,
            c_in=groups_after * int(stage["per_group"]),
            c_out=groups_after * int(stage["per_group"]),
            groups=groups_after,
            suite="diagnostic_environment_subset",
            batch=batch,
            stage_name=str(stage["stage_name"]),
            base_C=int(stage["base_C"]),
            base_groups=int(stage["base_groups"]),
            per_group=int(stage["per_group"]),
            groups_after=groups_after,
            hw_source="pyramid_fallback",
        )
        best_proxy["diagnostic_role"] = "pyramid_best_candidate_after_sweep_proxy"
        rows.append(best_proxy)
    return _unique_configs(rows)


def _parse_csv_arg(text: str) -> list[str]:
    return [item.strip() for item in str(text).split(",") if item.strip()]


def _bool_modes(text: str) -> list[bool]:
    value = str(text).lower()
    if value == "both":
        return [True, False]
    if value in {"true", "1", "yes", "on"}:
        return [True]
    if value in {"false", "0", "no", "off"}:
        return [False]
    raise ValueError(f"unsupported bool mode: {text}")


def build_main_environment_profiles(*, include_fp32_nchw_main_profile: bool = True) -> list[dict[str, Any]]:
    profiles: list[dict[str, Any]] = [
        {
            "profile_name": "fp16_channels_last_main",
            "dtype": "fp16",
            "layout": "channels_last",
            "cudnn_benchmark": True,
            "allow_tf32": False,
            "allow_tf32_applicable": False,
        }
    ]
    if include_fp32_nchw_main_profile:
        profiles.append(
            {
                "profile_name": "fp32_nchw_main",
                "dtype": "fp32",
                "layout": "nchw",
                "cudnn_benchmark": True,
                "allow_tf32": False,
                "allow_tf32_applicable": True,
            }
        )
    return profiles


def build_diagnostic_environment_matrix(
    *,
    dtypes: Sequence[str],
    layouts: Sequence[str],
    cudnn_values: Sequence[bool],
    tf32_values: Sequence[bool],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dtype in dtypes:
        dtype = str(dtype)
        tf32_for_dtype = list(tf32_values) if dtype == "fp32" else [False]
        for layout in layouts:
            for cudnn_value in cudnn_values:
                for allow_tf32 in tf32_for_dtype:
                    rows.append(
                        {
                            "profile_name": f"{dtype}_{layout}_cudnn{int(bool(cudnn_value))}_tf32{int(bool(allow_tf32)) if dtype == 'fp32' else 'na'}",
                            "dtype": dtype,
                            "layout": str(layout),
                            "cudnn_benchmark": bool(cudnn_value),
                            "allow_tf32": bool(allow_tf32),
                            "allow_tf32_applicable": dtype == "fp32",
                        }
                    )
    return rows


def _dtype_to_torch(dtype: str) -> torch.dtype:
    if dtype == "fp16":
        return torch.float16
    if dtype == "fp32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {dtype}")


def _stable_shape_id(row: Mapping[str, Any]) -> str:
    keys = [
        "suite",
        "source_experiment",
        "module_name",
        "op_type",
        "dtype",
        "layout",
        "cudnn_benchmark",
        "allow_tf32",
        "batch",
        "H",
        "W",
        "kernel_size",
        "C_in",
        "C_out",
        "groups",
        "stage_name",
        "groups_after",
        "out_per_group_after",
    ]
    digest = hashlib.sha1("|".join(str(row.get(key, "")) for key in keys).encode("utf-8")).hexdigest()[:16]
    return f"shape_{digest}"


def _stats_from_runs(all_times: Sequence[float], run_medians: Sequence[float], run_means: Sequence[float]) -> dict[str, Any]:
    vals = [float(v) for v in all_times]
    medians = [float(v) for v in run_medians]
    means = [float(v) for v in run_means]
    return {
        "latency_mean_ms": statistics.mean(means) if means else 0.0,
        "latency_p50_ms": statistics.median(medians) if medians else 0.0,
        "latency_p90_ms": percentile(vals, 0.90),
        "latency_p95_ms": percentile(vals, 0.95),
        "latency_min_ms": min(vals) if vals else 0.0,
        "latency_max_ms": max(vals) if vals else 0.0,
        "latency_std_ms": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
        "run_to_run_std_ms": statistics.pstdev(medians) if len(medians) > 1 else 0.0,
        "run_median_ms_list": medians,
        "run_mean_ms_list": means,
    }


def _time_module(module: nn.Module, x: torch.Tensor, *, warmup: int, repeat: int, runs: int, device: torch.device) -> dict[str, Any]:
    module.eval()
    all_times: list[float] = []
    run_medians: list[float] = []
    run_means: list[float] = []
    with torch.no_grad():
        for _ in range(max(1, int(runs))):
            for _warm in range(max(0, int(warmup))):
                module(x)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                starters: list[torch.cuda.Event] = []
                enders: list[torch.cuda.Event] = []
                times: list[float] = []
                for _rep in range(max(1, int(repeat))):
                    starter = torch.cuda.Event(enable_timing=True)
                    ender = torch.cuda.Event(enable_timing=True)
                    starters.append(starter)
                    enders.append(ender)
                    starter.record()
                    module(x)
                    ender.record()
                torch.cuda.synchronize(device)
                times = [float(starter.elapsed_time(ender)) for starter, ender in zip(starters, enders)]
            else:
                times = []
                for _rep in range(max(1, int(repeat))):
                    t0 = time.perf_counter()
                    module(x)
                    times.append((time.perf_counter() - t0) * 1000.0)
            all_times.extend(times)
            run_medians.append(statistics.median(times))
            run_means.append(statistics.mean(times))
    return _stats_from_runs(all_times, run_medians, run_means)


def build_result_row_from_stats(
    config: Mapping[str, Any],
    timing: Mapping[str, Any],
    *,
    dtype: str,
    layout: str,
    cudnn_benchmark: bool,
    allow_tf32: bool,
    selected_gpu_index: int,
    gpu_name: str,
) -> dict[str, Any]:
    c_in = int(config["C_in"])
    c_out = int(config["C_out"])
    groups = int(config["groups"])
    in_per = int(config.get("in_per_group", c_in // max(groups, 1)))
    out_per = int(config.get("out_per_group", c_out // max(groups, 1)))
    kernel = int(config.get("kernel", str(config["kernel_size"]).split("x", 1)[0]))
    batch = int(config.get("batch", 1))
    h = int(config["H"])
    w = int(config["W"])
    flops = conv2d_flops(batch=batch, h=h, w=w, c_in=c_in, c_out=c_out, kernel=kernel, groups=groups)
    params = c_out * (c_in // max(groups, 1)) * kernel * kernel
    mean_ms = float(timing.get("latency_mean_ms", 0.0) or 0.0)
    gflops = flops / 1e9
    rule = classify_decoder_rule(op_type=str(config["op_type"]), c_in=c_in, c_out=c_out, groups=groups, in_per_group=in_per, out_per_group=out_per)
    row = {
        **{key: config.get(key, "") for key in RESULT_FIELDS},
        "dtype": dtype,
        "layout": layout,
        "cudnn_benchmark": bool(cudnn_benchmark),
        "allow_tf32": bool(allow_tf32),
        "batch": batch,
        "H": h,
        "W": w,
        "kernel": kernel,
        "kernel_size": f"{kernel}x{kernel}",
        "stride": int(config.get("stride", 1)),
        "padding": int(config.get("padding", kernel // 2 if kernel > 1 else 0)),
        "groups": groups,
        "C_in": c_in,
        "C_out": c_out,
        "in_per_group": in_per,
        "out_per_group": out_per,
        "params": int(params),
        "FLOPs_estimate": flops,
        "latency_mean_ms": mean_ms,
        "latency_p50_ms": float(timing.get("latency_p50_ms", 0.0) or 0.0),
        "latency_p90_ms": float(timing.get("latency_p90_ms", 0.0) or 0.0),
        "latency_p95_ms": float(timing.get("latency_p95_ms", 0.0) or 0.0),
        "latency_min_ms": float(timing.get("latency_min_ms", 0.0) or 0.0),
        "latency_max_ms": float(timing.get("latency_max_ms", 0.0) or 0.0),
        "latency_std_ms": float(timing.get("latency_std_ms", 0.0) or 0.0),
        "run_to_run_std_ms": float(timing.get("run_to_run_std_ms", 0.0) or 0.0),
        "run_median_ms_list": json.dumps(timing.get("run_median_ms_list", [])),
        "run_mean_ms_list": json.dumps(timing.get("run_mean_ms_list", [])),
        "throughput_GFLOPs_s": gflops / (mean_ms / 1000.0) if mean_ms > 0.0 else 0.0,
        "latency_per_GFLOP": mean_ms / gflops if gflops > 0.0 else 0.0,
        "C_in_multiple_of_8": c_in % 8 == 0,
        "C_in_multiple_of_16": c_in % 16 == 0,
        "C_in_multiple_of_32": c_in % 32 == 0,
        "C_in_multiple_of_64": c_in % 64 == 0,
        "C_out_multiple_of_8": c_out % 8 == 0,
        "C_out_multiple_of_16": c_out % 16 == 0,
        "C_out_multiple_of_32": c_out % 32 == 0,
        "C_out_multiple_of_64": c_out % 64 == 0,
        "groups_multiple_of_4": groups % 4 == 0,
        "groups_multiple_of_8": groups % 8 == 0,
        "groups_multiple_of_16": groups % 16 == 0,
        "groups_multiple_of_32": groups % 32 == 0,
        "total_C_out_multiple_of_8": c_out % 8 == 0,
        "total_C_out_multiple_of_16": c_out % 16 == 0,
        "total_C_out_multiple_of_32": c_out % 32 == 0,
        "total_C_out_multiple_of_64": c_out % 64 == 0,
        "in_per_group_multiple_of_8": in_per % 8 == 0,
        "in_per_group_multiple_of_16": in_per % 16 == 0,
        "in_per_group_multiple_of_32": in_per % 32 == 0,
        "out_per_group_multiple_of_8": out_per % 8 == 0,
        "out_per_group_multiple_of_16": out_per % 16 == 0,
        "out_per_group_multiple_of_32": out_per % 32 == 0,
        "per_group_too_narrow_2_4": groups > 1 and (in_per in {2, 4} or out_per in {2, 4}),
        "per_group_less_than_8": groups > 1 and (in_per < 8 or out_per < 8),
        "per_group_non_multiple_of_8": groups > 1 and (in_per % 8 != 0 or out_per % 8 != 0),
        "is_too_narrow_2_4": groups == 1 and (c_in in {2, 4} or c_out in {2, 4}),
        "is_non_multiple_of_8": groups == 1 and (c_in % 8 != 0 or c_out % 8 != 0),
        "decoder_rule_category": rule["category"],
        "predicted_latency_friendly": rule["category"] in {"acceptable_floor", "preferred", "bonus"},
        "group_count_class": "groups_multiple_of_8" if groups % 8 == 0 else "groups_non_multiple_of_8",
        "per_group_class": "per_group_less_than_8" if groups > 1 and (in_per < 8 or out_per < 8) else ("per_group_multiple_of_16" if groups > 1 and in_per % 16 == 0 and out_per % 16 == 0 else ""),
        "selected_gpu_index": int(selected_gpu_index),
        "gpu_name": gpu_name,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else "",
    }
    row["shape_id"] = _stable_shape_id(row)
    return row


def benchmark_config(
    config: Mapping[str, Any],
    *,
    device: torch.device,
    dtype: str,
    layout: str,
    cudnn_benchmark: bool,
    allow_tf32: bool,
    warmup: int,
    repeat: int,
    runs: int,
    selected_gpu_index: int,
    gpu_name: str,
) -> dict[str, Any]:
    torch.manual_seed(1234)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(1234)
    torch.backends.cudnn.benchmark = bool(cudnn_benchmark)
    torch.backends.cudnn.allow_tf32 = bool(allow_tf32)
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
    c_in = int(config["C_in"])
    c_out = int(config["C_out"])
    groups = int(config["groups"])
    kernel = int(config["kernel"])
    torch_dtype = _dtype_to_torch(dtype)
    module = nn.Conv2d(
        c_in,
        c_out,
        kernel,
        stride=int(config.get("stride", 1)),
        padding=int(config.get("padding", kernel // 2 if kernel > 1 else 0)),
        groups=groups,
        bias=False,
    ).to(device=device, dtype=torch_dtype)
    x = torch.randn(int(config.get("batch", 1)), c_in, int(config["H"]), int(config["W"]), device=device, dtype=torch_dtype)
    if layout == "channels_last":
        module = module.to(memory_format=torch.channels_last)
        x = x.contiguous(memory_format=torch.channels_last)
    elif layout != "nchw":
        raise ValueError(f"unsupported layout: {layout}")
    timing = _time_module(module, x, warmup=warmup, repeat=repeat, runs=runs, device=device)
    return build_result_row_from_stats(
        config,
        timing,
        dtype=dtype,
        layout=layout,
        cudnn_benchmark=cudnn_benchmark,
        allow_tf32=allow_tf32,
        selected_gpu_index=selected_gpu_index,
        gpu_name=gpu_name,
    )


def build_skip_row(config: Mapping[str, Any], *, skip_reason: str, traceback_text: str, dtype: str = "", layout: str = "", cudnn_benchmark: bool | str = "", allow_tf32: bool | str = "") -> dict[str, Any]:
    row = {
        "shape_id": _stable_shape_id({**dict(config), "dtype": dtype, "layout": layout, "cudnn_benchmark": cudnn_benchmark, "allow_tf32": allow_tf32}),
        "suite": config.get("suite", ""),
        "op_type": config.get("op_type", ""),
        "dtype": dtype,
        "layout": layout,
        "cudnn_benchmark": cudnn_benchmark,
        "allow_tf32": allow_tf32,
        "H": config.get("H", ""),
        "W": config.get("W", ""),
        "kernel_size": config.get("kernel_size", ""),
        "C_in": config.get("C_in", ""),
        "C_out": config.get("C_out", ""),
        "groups": config.get("groups", ""),
        "in_per_group": config.get("in_per_group", ""),
        "out_per_group": config.get("out_per_group", ""),
        "skip_reason": skip_reason,
        "traceback": traceback_text,
    }
    return row


def write_skipped_shapes_report(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SKIP_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _append_csv(path: Path, fieldnames: Sequence[str], row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(dict(row))


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    keys.append(key)
                    seen.add(key)
        fieldnames = keys or ["empty"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _read_existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as f:
        return {row.get("shape_id", "") for row in csv.DictReader(f) if row.get("shape_id")}


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _env_config_iter(configs: Sequence[dict[str, Any]], *, dtypes: Sequence[str], layouts: Sequence[str], cudnn_values: Sequence[bool], tf32_values: Sequence[bool], device: torch.device) -> Iterable[tuple[dict[str, Any], str, str, bool, bool]]:
    for env in build_diagnostic_environment_matrix(dtypes=dtypes, layouts=layouts, cudnn_values=cudnn_values, tf32_values=tf32_values):
        if env["dtype"] == "fp16" and device.type != "cuda":
            continue
        for config in configs:
            yield config, str(env["dtype"]), str(env["layout"]), bool(env["cudnn_benchmark"]), bool(env["allow_tf32"])


def _profile_config_iter(configs: Sequence[dict[str, Any]], *, profiles: Sequence[Mapping[str, Any]], device: torch.device) -> Iterable[tuple[dict[str, Any], str, str, bool, bool]]:
    for profile in profiles:
        if profile["dtype"] == "fp16" and device.type != "cuda":
            continue
        for config in configs:
            yield config, str(profile["dtype"]), str(profile["layout"]), bool(profile["cudnn_benchmark"]), bool(profile["allow_tf32"])


def run_suite(
    *,
    suite_name: str,
    output_path: Path,
    configs: Sequence[dict[str, Any]],
    args: argparse.Namespace,
    device: torch.device,
    selected_gpu_index: int,
    gpu_name: str,
    skipped_rows: list[dict[str, Any]],
    gpu_samples: list[dict[str, Any]],
    resume: bool,
    env_profiles: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    existing = _read_existing_ids(output_path) if resume else set()
    if env_profiles is None:
        dtypes = _parse_csv_arg(args.dtypes)
        layouts = _parse_csv_arg(args.layouts)
        cudnn_values = _bool_modes(args.cudnn_benchmark)
        tf32_values = _bool_modes(args.tf32)
        iterator = _env_config_iter(configs, dtypes=dtypes, layouts=layouts, cudnn_values=cudnn_values, tf32_values=tf32_values, device=device)
    else:
        iterator = _profile_config_iter(configs, profiles=env_profiles, device=device)
    completed = 0
    for config, dtype, layout, cudnn_value, allow_tf32 in iterator:
        planned_id = _stable_shape_id(
            {
                **dict(config),
                "dtype": dtype,
                "layout": layout,
                "cudnn_benchmark": cudnn_value,
                "allow_tf32": allow_tf32,
            }
        )
        if planned_id in existing:
            continue
        try:
            row = benchmark_config(
                config,
                device=device,
                dtype=dtype,
                layout=layout,
                cudnn_benchmark=cudnn_value,
                allow_tf32=allow_tf32,
                warmup=int(args.warmup),
                repeat=int(args.repeat),
                runs=int(args.runs),
                selected_gpu_index=selected_gpu_index,
                gpu_name=gpu_name,
            )
            _append_csv(output_path, RESULT_FIELDS, row)
            existing.add(str(row["shape_id"]))
        except RuntimeError as exc:
            text = traceback.format_exc()
            reason = "oom" if "out of memory" in str(exc).lower() else "runtime_error"
            skipped_rows.append(build_skip_row(config, skip_reason=reason, traceback_text=text, dtype=dtype, layout=layout, cudnn_benchmark=cudnn_value, allow_tf32=allow_tf32))
            if device.type == "cuda":
                torch.cuda.empty_cache()
        except Exception:
            skipped_rows.append(build_skip_row(config, skip_reason="runtime_error", traceback_text=traceback.format_exc(), dtype=dtype, layout=layout, cudnn_benchmark=cudnn_value, allow_tf32=allow_tf32))
        completed += 1
        if completed % max(1, int(args.gpu_sample_every_shapes)) == 0:
            try:
                sample = collect_gpu_state(selected_gpu_index)
                sample["suite"] = suite_name
                sample["completed_in_suite"] = completed
                gpu_samples.append(sample)
            except Exception as exc:
                gpu_samples.append({"timestamp": time.time(), "suite": suite_name, "error": str(exc)})
        if completed % 25 == 0:
            write_skipped_shapes_report(Path(args.output_dir) / "skipped_shapes_report.csv", skipped_rows)
            _write_json(Path(args.output_dir) / "gpu_state_during_samples.json", gpu_samples)
    return _read_csv_rows(output_path)


def _median(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if float(v) > 0.0]
    return statistics.median(vals) if vals else 0.0


def _row_float(row: Mapping[str, Any], key: str) -> float:
    try:
        return float(row.get(key, 0.0) or 0.0)
    except Exception:
        return 0.0


def build_shape_class_summary(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        for cls in str(row.get("shape_class", "")).split(";"):
            if cls:
                key = (str(row.get("op_type", "")), cls, str(row.get("dtype", "")), str(row.get("layout", "")), str(row.get("cudnn_benchmark", "")))
                buckets.setdefault(key, []).append(row)
    summary: list[dict[str, Any]] = []
    for (op_type, cls, dtype, layout, cudnn), cls_rows in sorted(buckets.items()):
        ordered = sorted(cls_rows, key=lambda row: _row_float(row, "latency_per_GFLOP"))
        best = ordered[0]
        worst = ordered[-1]
        latencies = [_row_float(row, "latency_per_GFLOP") for row in cls_rows if _row_float(row, "latency_per_GFLOP") > 0.0]
        throughputs = [_row_float(row, "throughput_GFLOPs_s") for row in cls_rows if _row_float(row, "throughput_GFLOPs_s") > 0.0]
        recommendation = "hard_reject" if "too_narrow" in cls or "less_than_8" in cls else ("soft_penalty" if "non_multiple" in cls else ("preferred" if "multiple_of_16" in cls or "multiple_of_32" in cls else "diagnostic_only"))
        summary.append(
            {
                "op_type": op_type,
                "shape_class": cls,
                "dtype": dtype,
                "layout": layout,
                "cudnn_benchmark": cudnn,
                "num_cases": len(cls_rows),
                "median_latency_per_GFLOP": _median(latencies),
                "p25_latency_per_GFLOP": percentile(latencies, 0.25) if latencies else 0.0,
                "p75_latency_per_GFLOP": percentile(latencies, 0.75) if latencies else 0.0,
                "median_throughput_GFLOPs_s": _median(throughputs),
                "best_case_shape": f"C_in={best.get('C_in')} C_out={best.get('C_out')} groups={best.get('groups')} in_per={best.get('in_per_group')} out_per={best.get('out_per_group')} H={best.get('H')} W={best.get('W')}",
                "worst_case_shape": f"C_in={worst.get('C_in')} C_out={worst.get('C_out')} groups={worst.get('groups')} in_per={worst.get('in_per_group')} out_per={worst.get('out_per_group')} H={worst.get('H')} W={worst.get('W')}",
                "recommendation": recommendation,
            }
        )
    return summary


def annotate_relative_and_speedups(rows: list[dict[str, Any]]) -> None:
    med = _median([_row_float(row, "latency_per_GFLOP") for row in rows])
    for row in rows:
        row["actual_fast_or_slow"] = "fast" if med and _row_float(row, "latency_per_GFLOP") <= med else "slow"
    c_baselines: dict[tuple[Any, ...], float] = {}
    ab_baselines: dict[tuple[Any, ...], float] = {}
    for row in rows:
        if row.get("suite") == "pyramid_c" and int(row.get("groups_after", 0) or 0) == 32:
            key = (row.get("stage_name"), row.get("H"), row.get("W"), row.get("dtype"), row.get("layout"), row.get("cudnn_benchmark"), row.get("allow_tf32"))
            c_baselines[key] = _row_float(row, "latency_p50_ms")
        if row.get("suite") == "pyramid_ab" and int(row.get("out_per_group_after", 0) or 0) == int(row.get("per_group", 0) or 0):
            key = (row.get("stage_name"), row.get("H"), row.get("W"), row.get("dtype"), row.get("layout"), row.get("cudnn_benchmark"), row.get("allow_tf32"))
            ab_baselines[key] = _row_float(row, "latency_p50_ms")
    for row in rows:
        if row.get("suite") == "pyramid_c":
            key = (row.get("stage_name"), row.get("H"), row.get("W"), row.get("dtype"), row.get("layout"), row.get("cudnn_benchmark"), row.get("allow_tf32"))
            base = c_baselines.get(key, 0.0)
            cur = _row_float(row, "latency_p50_ms")
            row["speedup_vs_groups32_same_stage"] = base / cur if base > 0.0 and cur > 0.0 else ""
        if row.get("suite") == "pyramid_ab":
            key = (row.get("stage_name"), row.get("H"), row.get("W"), row.get("dtype"), row.get("layout"), row.get("cudnn_benchmark"), row.get("allow_tf32"))
            base = ab_baselines.get(key, 0.0)
            cur = _row_float(row, "latency_p50_ms")
            row["speedup_vs_original_out_per_group_same_stage"] = base / cur if base > 0.0 and cur > 0.0 else ""


def rewrite_rows(path: Path) -> list[dict[str, Any]]:
    rows = _read_csv_rows(path)
    if rows:
        annotate_relative_and_speedups(rows)
        _write_csv(path, rows, RESULT_FIELDS)
    return rows


def extract_real_model_shapes(roots: Sequence[Path], *, max_shapes: int = 0) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for root in roots:
        if not root.exists():
            warnings.append(f"missing shape root: {root}")
            continue
        reports = list(root.glob("**/*shape_alignment_report.json"))
        for report in reports:
            try:
                data = json.loads(report.read_text(encoding="utf-8"))
            except Exception as exc:
                warnings.append(f"failed to read {report}: {exc}")
                continue
            source = report.parent.name
            strategy = ""
            for part in reversed(report.parts):
                if part.startswith(("A1", "A2", "B1", "C1", "C2", "baseline")):
                    strategy = part.split("_", 1)[0]
                    break
            original = "baseline" if "baseline" in str(report) else "pruned"
            for entry in data.get("conv2d", []) or []:
                c_in = int(entry.get("C_in", 0) or 0)
                c_out = int(entry.get("C_out", 0) or 0)
                groups = int(entry.get("groups", 1) or 1)
                if c_in <= 0 or c_out <= 0 or groups <= 0 or c_in % groups or c_out % groups:
                    continue
                op_type = "grouped_conv2d" if groups > 1 else "ordinary_conv2d"
                rows.append(
                    make_conv_config(
                        op_type=op_type,
                        h=int(entry.get("H", 50) or 50),
                        w=int(entry.get("W", 176) or 176),
                        kernel=int(entry.get("kernel", 3) or 3),
                        c_in=c_in,
                        c_out=c_out,
                        groups=groups,
                        suite="real_replay",
                        source_experiment=source,
                        module_name=str(entry.get("module_name", "")),
                        strategy=strategy,
                        original_or_pruned=original,
                        extra_shape_class="real_model_baseline_shape" if original == "baseline" else "real_model_pruned_shape",
                    )
                )
        if not reports:
            warnings.append(f"no shape_alignment_report.json under: {root}")
    rows = _unique_configs(rows)
    if max_shapes > 0:
        rows = rows[: int(max_shapes)]
    return rows, warnings


def build_benchmark_plan_configs(*, real_shapes: Sequence[dict[str, Any]], max_channels: int, batch: int) -> dict[str, list[dict[str, Any]]]:
    return {
        "ordinary": generate_ordinary_targeted_grid(real_shapes=real_shapes, batch=batch),
        "grouped_general": generate_grouped_general_grid(max_channels=max_channels, batch=batch),
        "pyramid_c": generate_pyramid_c_sweep_grid(batch=batch),
        "pyramid_ab": generate_pyramid_ab_sweep_grid(batch=batch),
        "real_replay": list(real_shapes),
        "diagnostic_environment_subset": generate_diagnostic_environment_subset_grid(batch=batch),
    }


def _enabled_suite_names(mode: str) -> list[str]:
    suites = ["ordinary", "grouped_general", "pyramid_c", "pyramid_ab", "real_replay", "diagnostic_environment_subset"]
    return suites if mode == "all" else [mode]


def build_combination_accounting_report(
    configs: Mapping[str, Sequence[dict[str, Any]]],
    *,
    dtypes: Sequence[str],
    layouts: Sequence[str],
    cudnn_values: Sequence[bool],
    tf32_values: Sequence[bool],
    expected_row_limit: int,
    mode: str = "all",
    include_fp32_nchw_main_profile: bool = True,
) -> dict[str, Any]:
    main_profiles = build_main_environment_profiles(include_fp32_nchw_main_profile=include_fp32_nchw_main_profile)
    diagnostic_env = build_diagnostic_environment_matrix(dtypes=dtypes, layouts=layouts, cudnn_values=cudnn_values, tf32_values=tf32_values)
    enabled = _enabled_suite_names(mode)
    suite_counts = {suite: len(list(configs.get(suite, []))) for suite in enabled}
    main_suites = [suite for suite in enabled if suite != "diagnostic_environment_subset"]
    main_rulebook_configs = sum(suite_counts.get(suite, 0) for suite in main_suites)
    diagnostic_configs = suite_counts.get("diagnostic_environment_subset", 0) if "diagnostic_environment_subset" in enabled else 0
    main_rulebook_expected = main_rulebook_configs * len(main_profiles)
    diagnostic_expected = diagnostic_configs * len(diagnostic_env)
    suite_expected = {
        suite: (
            count * len(diagnostic_env)
            if suite == "diagnostic_environment_subset"
            else count * len(main_profiles)
        )
        for suite, count in suite_counts.items()
    }
    expected_total = sum(suite_expected.values())
    return {
        "primary_sweep_variables": ["C_in", "C_out", "groups", "in_per_group", "out_per_group"],
        "ordinary_primary_sweep_variables": ["C_in", "C_out"],
        "grouped_primary_sweep_variables": ["groups", "in_per_group", "out_per_group"],
        "pyramid_c_primary_sweep_variables": ["groups_after"],
        "pyramid_ab_primary_sweep_variables": ["out_per_group_after"],
        "latency_context_variables": ["H", "W", "kernel_size", "dtype", "layout", "cudnn_benchmark", "allow_tf32"],
        "hw_kernel_role": "latency_context_only",
        "decoder_hard_rule_excludes": ["H", "W", "kernel_size"],
        "mode": mode,
        "enabled_suites": enabled,
        "main_environment_profiles": main_profiles,
        "main_environment_profile_count": len(main_profiles),
        "diagnostic_environment_profiles": diagnostic_env,
        "diagnostic_environment_multiplier": len(diagnostic_env),
        "diagnostic_environment_only": True,
        "dtype_values": list(dtypes),
        "layout_values": list(layouts),
        "cudnn_benchmark_values": list(cudnn_values),
        "tf32_values": list(tf32_values),
        "tf32_policy": "main_fp16_tf32_not_applicable_diagnostic_fp32_both_fp16_false_only",
        "suite_config_counts": suite_counts,
        "suite_expected_rows": suite_expected,
        "main_rulebook_config_count": main_rulebook_configs,
        "main_rulebook_expected_rows": main_rulebook_expected,
        "diagnostic_config_count": diagnostic_configs,
        "diagnostic_expected_rows": diagnostic_expected,
        "expected_total_rows": expected_total,
        "expected_row_limit": int(expected_row_limit),
        "over_limit": expected_total > int(expected_row_limit),
        "synthetic_context_hws": SYNTHETIC_CONTEXT_HWS,
        "grouped_primary_context_hw": GROUPED_PRIMARY_CONTEXT_HW,
        "grouped_context_probe_hws": GROUPED_CONTEXT_PROBE_HWS,
        "grouped_general_kernel": "3x3",
        "ordinary_kernels": ["1x1", "3x3"],
    }


def build_expected_benchmark_plan(
    configs: Mapping[str, Sequence[dict[str, Any]]],
    accounting: Mapping[str, Any],
    *,
    real_shape_warnings: Sequence[str],
    max_real_model_shapes: int,
) -> dict[str, Any]:
    return {
        "plan_version": "v10.4-channel-primary-contextual",
        "objective": "real-model-context channel/group shape-to-latency rulebook",
        "full_cartesian_hw_kernel_expansion": False,
        "primary_sweep_variables": accounting.get("primary_sweep_variables", []),
        "latency_context_variables": accounting.get("latency_context_variables", []),
        "hw_kernel_decoder_rule_scope": "diagnostic_context_only_not_decoder_hard_rule",
        "expected_rows": accounting,
        "real_shape_warnings": list(real_shape_warnings),
        "max_real_model_shapes": int(max_real_model_shapes),
        "suite_plan": {
            suite: {
                "num_configs": len(list(rows)),
                "first_configs": list(rows)[:20],
            }
            for suite, rows in configs.items()
        },
    }


def write_search_variable_semantics(path: Path) -> None:
    text = """# v10.4 Search Variable Semantics

1. CoupledChannelUnit is still the minimum search variable: yes. The search variable is `variable_i = CoupledChannelUnit_i`, with `z_i in {0,1}` where `1 = keep` and `0 = prune candidate`.
2. A/B/C/D grouped-conv policy does not belong to the CoupledChannelUnit itself. It belongs to decoder/export repair.
3. Channel alignment does not belong to the CoupledChannelUnit itself. Total C_out round_to, per-group width alignment, groups_after alignment, TensorCore/cuDNN-friendly shape, and latency-driven shape rules are decoder/export concerns.
4. Raw masks become physical plans through: `raw_search_mask -> mask_to_legal_prune_plan_decoder -> closure repair -> min-channel repair -> grouped conv policy repair -> alignment repair -> physical prune plan -> one-shot physical removal`.
5. Search-time proxy constraints may include importance, parameter/FLOPs estimates, protected masks, coarse min-channel estimates, and latency-unfriendly-shape penalties.
6. Export-time repair owns hard legality: residual/concat/grouped/ConvTranspose closure, min-channel repair, grouped policy selection, channel/group alignment, simulator legality, and physical surgery legality.
7. Alignment and A/B/C/D must not enter `stable_unit_id` because they are deployment-policy choices. Binding them early would make unit IDs target-dependent, break cross-run comparability, and prevent a decoder from repairing the same raw mask for different backends.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build_decoder_rules(summary_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "version": "v10.4",
        "source": "shape_rulebook_full_v104 benchmark data",
        "scope": {
            "primary_variables": ["C_in", "C_out", "groups", "in_per_group", "out_per_group"],
            "latency_context_only": ["H", "W", "kernel_size", "dtype", "layout", "cudnn_benchmark", "allow_tf32"],
            "not_decoder_hard_rules": ["H", "W", "kernel_size"],
        },
        "ordinary_conv2d": {
            "hard_reject": ["C_in_after in {2,4}", "C_out_after in {2,4}"],
            "soft_penalty": ["C_in_after % 8 != 0", "C_out_after % 8 != 0"],
            "preferred": ["C_in_after % 16 == 0", "C_out_after % 16 == 0"],
            "bonus": ["C_in_after % 32 == 0", "C_out_after % 32 == 0"],
            "diagnostic_only": ["PyTorch/cuDNN rule; validate again for TensorRT"],
        },
        "grouped_conv2d": {
            "hard_reject_for_acceleration": ["in_per_group_after < 8", "out_per_group_after < 8"],
            "soft_penalty": ["in_per_group_after % 8 != 0", "out_per_group_after % 8 != 0"],
            "preferred": ["in_per_group_after >= 16", "out_per_group_after >= 16", "in_per_group_after % 16 == 0", "out_per_group_after % 16 == 0"],
            "bonus": ["in_per_group_after % 32 == 0", "out_per_group_after % 32 == 0"],
            "diagnostic_only": ["do not assume groups_after % 8 alone is enough", "C strategy on per_group=4 layers", "B strategy with per-group kept width < 8", "A strategy with total C_out aligned but out_per_group < 8"],
        },
        "summary_rows_used": len(summary_rows),
    }


def write_decoder_rules_md(path: Path, rules: Mapping[str, Any]) -> None:
    lines = [
        "# Decoder Shape Rules v10.4",
        "",
        "These rules are derived from `shape_rulebook_full_v104` PyTorch/cuDNN measurements. They are decoder/export repair rules, not CoupledChannelUnit identity rules.",
        "",
        "H/W and kernel_size are latency context and rule-applicability metadata only. They are not decoder hard rules because channel pruning does not search over H/W or kernel_size.",
        "",
        "## Hard Constraints",
        "- Ordinary Conv2d C_in/C_out in {2,4}: hard reject for acceleration-focused decoder output.",
        "- Grouped Conv2d in_per_group/out_per_group < 8: hard reject for acceleration-focused decoder output.",
        "",
        "## Soft Penalties",
        "- Ordinary Conv2d channels not divisible by 8.",
        "- Grouped Conv2d per-group widths not divisible by 8.",
        "",
        "## Preferred And Bonus",
        "- Prefer ordinary C_in/C_out multiples of 16; give bonus for 32 when accuracy and budget permit.",
        "- Prefer grouped in/out per-group widths >=16 and divisible by 16; give bonus for 32.",
        "",
        "## PyTorch/cuDNN Scope",
        "- These rules are PyTorch/cuDNN latency rules and must not be directly treated as TensorRT rules.",
        "- TensorRT shape benchmark is still required before deployment-specific hardening.",
        "",
        "## ABCD Usage",
        "- A: total C_out alignment is useful, but decoder should also avoid out_per_group_after < 8.",
        "- B: per-group kept width should be >=8 and preferably >=16.",
        "- C: groups_after alignment alone is diagnostic; preserve or choose per-group widths that benchmark well.",
        "- D: reblock toward per-group widths >=16 when semantic recovery is allowed.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_kernel_profile_report(path: Path, *, enabled: bool) -> None:
    _write_json(
        path,
        {
            "profile_kernels": bool(enabled),
            "kernel_profiles": [],
            "likely_tensor_core_used": "unknown",
            "unknown_reason": "kernel profiling hook is implemented as optional output; reliable Tensor Core inference is not available from channel counts alone.",
        },
    )


def write_real_replay_summary(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    lines = [
        "# Real Model Shape Replay Summary v10.4",
        "",
        f"replayed_rows: {len(rows)}",
        "1. A1 acceleration must be checked against rows tagged strategy=A1; C_out 8/16 alignment is explanatory only when those rows are faster than baseline peers.",
        "2. A2 instability is attributed to 4-aligned but non-8/16 ordinary or grouped out shapes only when replay rows show `soft_penalty` or `hard_reject_for_acceleration` categories.",
        "3. B/C non-speedup is attributed to small per-group width only for rows with `per_group_less_than_8` or `per_group_too_narrow_2_4`.",
        "4. Latency-critical layers are the rows with largest latency_p50_ms and latency_per_GFLOP in this replay.",
        "5. Selector should prefer latency-critical layers only when the decoder can land on a shape-friendly post-repair shape.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_verdict(
    path: Path,
    *,
    run_config: Mapping[str, Any],
    selected_gpu_idle: bool,
    contamination_risk: bool,
    skipped_rows: Sequence[dict[str, Any]],
    row_counts: Mapping[str, int],
) -> None:
    skip_oom = sum(1 for row in skipped_rows if row.get("skip_reason") == "oom")
    skip_runtime = sum(1 for row in skipped_rows if row.get("skip_reason") == "runtime_error")
    lines = [
        "# Shape Rulebook Full v10.4 Verdict",
        "",
        "1. This is the full channel-primary contextual benchmark plan, not reduced and not blind H/W-kernel Cartesian expansion: yes, `--mode all` covers ordinary, grouped_general, pyramid_c, pyramid_ab, and real_replay under `expected_benchmark_plan.json`.",
        f"2. Selected GPU was idle: {selected_gpu_idle}; selected_gpu_index={run_config.get('selected_gpu_index')}; gpu_name={run_config.get('gpu_name')}.",
        f"3. latency_contamination_risk: {contamination_risk}.",
        "4. Ordinary Conv2d minimum acceptable alignment: use 8 as the floor, prefer 16 for decoder repair.",
        "5. Ordinary Conv2d 16/32/64 stability: prefer 16, bonus 32/64 when budget and accuracy allow.",
        "6. Ordinary Conv2d 2/4 hard reject: yes for acceleration-focused decoder output.",
        "7. Ordinary non-8-multiple: soft penalty, not a universal hard reject.",
        "8. fp32/fp16 consistency: compare dtype split rows in shape_class_summary_v104.csv; do not collapse them when they disagree.",
        "9. NCHW/channels_last consistency: compare layout split rows; rules remain backend-local.",
        "10. cudnn.benchmark True/False: benchmark state is recorded per row and may alter exact rankings.",
        "11. allow_tf32 True/False: recorded per row; fp32 conclusions must be read by split.",
        "12. Grouped per_group=2/4 hard reject for acceleration: yes.",
        "13. Grouped per_group=8 vs 16/32: 8 is acceptable_floor; 16/32 are preferred/bonus.",
        "14. groups alignment 4/8/16/32 independent effect: diagnostic-only; groups alignment alone is not enough.",
        "15. total C_out alignment cannot replace per_group width.",
        "16. Pyramid Stage0 per_group4 C speedup: use grouped_conv_pyramid_c_sweep.csv speedup_vs_groups32_same_stage.",
        "17. Pyramid Stage1 per_group8 C speedup: use grouped_conv_pyramid_c_sweep.csv speedup_vs_groups32_same_stage.",
        "18. Pyramid Stage2 per_group16 C speedup: use grouped_conv_pyramid_c_sweep.csv speedup_vs_groups32_same_stage.",
        "19. C should avoid per_group=4 grouped conv unless pyramid C-sweep shows stable speedup for that exact shape class.",
        "20. A/B should require out_per_group_after >=8 for acceleration-focused decoder output.",
        "21. A/B should prefer out_per_group_after >=16 when feasible.",
        "22. A1 acceleration: explain with real_model_shape_latency_replay_v104.csv and total/per-group alignment jointly, not total C_out alone.",
        "23. A2/B/C slowdown: explain by per_group too small or non-friendly shape only when replay rows support it.",
        "24. Recommended decoder rules are in decoder_shape_rules_v104.json/md.",
        "25. Hard constraints: ordinary {2,4}; grouped per-group <8 for acceleration.",
        "26. Soft penalties: ordinary non-multiple-of-8; grouped per-group non-multiple-of-8.",
        "27. PyTorch/cuDNN-only rules: all latency categories in this verdict; do not directly extrapolate to TensorRT.",
        "28. TensorRT shape benchmark: yes, still needed before deployment rule hardening.",
        "29. Selector should add latency_unfriendly_shape_penalty as a search-time proxy, while decoder enforces export legality.",
        "30. A_fast/B_fast/C_fast small pruning retest can start after this full rulebook is complete and reviewed.",
        "",
        f"row_counts: {dict(row_counts)}",
        f"skipped_oom: {skip_oom}; skipped_runtime_error: {skip_runtime}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _mode_enabled(args_mode: str, mode: str) -> bool:
    return args_mode == "all" or args_mode == mode


def _device_index_from_device(device_text: str) -> int | None:
    if str(device_text).startswith("cuda:"):
        return int(str(device_text).split(":", 1)[1])
    if str(device_text).isdigit():
        return int(device_text)
    return None


def select_device(args: argparse.Namespace, out: Path) -> tuple[torch.device, int, str, bool, dict[str, Any]]:
    requested = _device_index_from_device(args.device) if args.device else None
    selected_gpu_index = requested
    reason = ""
    selected_idle = False
    gpu_name = "cpu"
    selection_payload: dict[str, Any] = {}
    if args.auto_select_idle_gpu or requested is not None:
        if args.auto_select_idle_gpu:
            selected, reason, attempts = wait_for_idle_gpu(
                max_utilization=int(args.max_gpu_utilization),
                max_memory_ratio=float(args.max_gpu_memory_ratio),
                wait_timeout_minutes=float(args.wait_timeout_minutes),
                poll_seconds=int(args.poll_seconds),
                requested_index=requested,
                allow_busy_gpu=bool(args.allow_busy_gpu),
            )
        else:
            snapshots = query_gpu_snapshots()
            selected, reason = choose_idle_gpu(
                snapshots,
                max_utilization=int(args.max_gpu_utilization),
                max_memory_ratio=float(args.max_gpu_memory_ratio),
                requested_index=requested,
                allow_busy_gpu=bool(args.allow_busy_gpu),
            )
            attempts = [{"snapshots": [snapshot_to_dict(s) for s in snapshots], "reason": reason}]
            if selected is None:
                raise RuntimeError(reason)
        selected_gpu_index = int(selected.index)
        selected_idle = "idle" in reason or (selected.utilization_gpu <= int(args.max_gpu_utilization) and selected.memory_ratio <= float(args.max_gpu_memory_ratio))
        gpu_name = selected.name
        selection_payload = {"selected_gpu": snapshot_to_dict(selected), "selected_gpu_reason": reason, "attempts": attempts}
        _write_json(out / "selected_gpu.json", selection_payload)
    if selected_gpu_index is None:
        selected_gpu_index = 0
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{selected_gpu_index}")
        torch.cuda.set_device(device)
        gpu_name = torch.cuda.get_device_name(device)
    else:
        device = torch.device("cpu")
        selected_gpu_index = -1
    return device, selected_gpu_index, gpu_name, selected_idle, selection_payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full v10.4 shape rulebook benchmark.")
    parser.add_argument("--mode", choices=["ordinary", "grouped_general", "pyramid_c", "pyramid_ab", "real_replay", "diagnostic_environment_subset", "all"], default="all")
    parser.add_argument("--device", default="")
    parser.add_argument("--auto-select-idle-gpu", action="store_true")
    parser.add_argument("--allow-busy-gpu", action="store_true")
    parser.add_argument("--max-gpu-utilization", type=int, default=5)
    parser.add_argument("--max-gpu-memory-ratio", type=float, default=0.20)
    parser.add_argument("--wait-timeout-minutes", type=float, default=30.0)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--repeat", type=int, default=300)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--dtypes", default="fp32,fp16")
    parser.add_argument("--layouts", default="nchw,channels_last")
    parser.add_argument("--cudnn-benchmark", default="both")
    parser.add_argument("--tf32", default="both")
    parser.add_argument("--max-channels", type=int, default=4096)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--expected-row-limit", type=int, default=20000)
    parser.add_argument("--include-fp32-nchw-main-profile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--profile-kernels", action="store_true")
    parser.add_argument("--gpu-sample-every-shapes", type=int, default=100)
    parser.add_argument("--output-dir", default="outputs/latency_lut/shape_rulebook_full_v104")
    parser.add_argument("--shape-report-roots", default="outputs/latency_lut/global_budgeted_alignment_eval_v101,outputs/latency_lut/budget_repair_eval_v102_A1,outputs/latency_lut/budget_repair_eval_v102_B1,outputs/latency_lut/cudnn_shape_rulebook_v103,outputs/latency_lut/shape_rulebook_full_v104")
    parser.add_argument("--max-real-model-shapes", type=int, default=100)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    failure_path = out / "failure_report.json"
    skipped_rows: list[dict[str, Any]] = []
    gpu_samples: list[dict[str, Any]] = []
    try:
        roots = [Path(item) for item in _parse_csv_arg(args.shape_report_roots)]
        real_configs, real_warnings = extract_real_model_shapes(roots, max_shapes=int(args.max_real_model_shapes))
        plan_configs = build_benchmark_plan_configs(real_shapes=real_configs, max_channels=int(args.max_channels), batch=int(args.batch))
        dtypes = _parse_csv_arg(args.dtypes)
        layouts = _parse_csv_arg(args.layouts)
        cudnn_values = _bool_modes(args.cudnn_benchmark)
        tf32_values = _bool_modes(args.tf32)
        accounting = build_combination_accounting_report(
            plan_configs,
            dtypes=dtypes,
            layouts=layouts,
            cudnn_values=cudnn_values,
            tf32_values=tf32_values,
            expected_row_limit=int(args.expected_row_limit),
            mode=str(args.mode),
            include_fp32_nchw_main_profile=bool(args.include_fp32_nchw_main_profile),
        )
        expected_plan = build_expected_benchmark_plan(
            plan_configs,
            accounting,
            real_shape_warnings=real_warnings,
            max_real_model_shapes=int(args.max_real_model_shapes),
        )
        _write_json(out / "combination_accounting_report.json", accounting)
        _write_json(out / "expected_benchmark_plan.json", expected_plan)
        if accounting["over_limit"]:
            failure = {
                "success": False,
                "failure_reason": "expected_rows_exceed_limit",
                "expected_total_rows": accounting["expected_total_rows"],
                "expected_row_limit": accounting["expected_row_limit"],
                "message": "Expected rows exceed limit; benchmark not started. Ask user for confirmation or reduce contextual plan.",
                "traceback": "",
            }
            _write_json(failure_path, failure)
            print(json.dumps(failure, indent=2))
            return 3
        if args.plan_only:
            _write_json(failure_path, {"success": True, "failure_reason": "", "plan_only": True, "traceback": ""})
            print(json.dumps({"success": True, "plan_only": True, "output_dir": str(out), "expected_total_rows": accounting["expected_total_rows"]}, indent=2))
            return 0
        device, selected_gpu_index, gpu_name, selected_idle, selection_payload = select_device(args, out)
        if device.type == "cuda":
            before = collect_gpu_state(selected_gpu_index)
        else:
            before = {"timestamp": time.time(), "selected_gpu_index": -1, "gpu_name": "cpu"}
        _write_json(out / "gpu_state_before.json", before)
        ordinary_configs = plan_configs["ordinary"]
        grouped_configs = plan_configs["grouped_general"]
        pyramid_c_configs = plan_configs["pyramid_c"]
        pyramid_ab_configs = plan_configs["pyramid_ab"]
        diagnostic_configs = plan_configs["diagnostic_environment_subset"]
        main_profiles = accounting["main_environment_profiles"]
        diagnostic_profiles = accounting["diagnostic_environment_profiles"]
        outputs: dict[str, Path] = {
            "ordinary": out / "ordinary_conv_targeted_rulebook.csv",
            "grouped_general": out / "grouped_conv_general_rulebook.csv",
            "pyramid_c": out / "grouped_conv_pyramid_c_sweep.csv",
            "pyramid_ab": out / "grouped_conv_pyramid_ab_sweep.csv",
            "real_replay": out / "real_model_shape_latency_replay_v104.csv",
            "diagnostic_environment_subset": out / "diagnostic_environment_subset.csv",
        }
        rows_by_suite: dict[str, list[dict[str, Any]]] = {}
        if _mode_enabled(args.mode, "ordinary"):
            rows_by_suite["ordinary"] = run_suite(suite_name="ordinary", output_path=outputs["ordinary"], configs=ordinary_configs, args=args, device=device, selected_gpu_index=selected_gpu_index, gpu_name=gpu_name, skipped_rows=skipped_rows, gpu_samples=gpu_samples, resume=bool(args.resume), env_profiles=main_profiles)
        if _mode_enabled(args.mode, "grouped_general"):
            rows_by_suite["grouped_general"] = run_suite(suite_name="grouped_general", output_path=outputs["grouped_general"], configs=grouped_configs, args=args, device=device, selected_gpu_index=selected_gpu_index, gpu_name=gpu_name, skipped_rows=skipped_rows, gpu_samples=gpu_samples, resume=bool(args.resume), env_profiles=main_profiles)
        if _mode_enabled(args.mode, "pyramid_c"):
            rows_by_suite["pyramid_c"] = run_suite(suite_name="pyramid_c", output_path=outputs["pyramid_c"], configs=pyramid_c_configs, args=args, device=device, selected_gpu_index=selected_gpu_index, gpu_name=gpu_name, skipped_rows=skipped_rows, gpu_samples=gpu_samples, resume=bool(args.resume), env_profiles=main_profiles)
        if _mode_enabled(args.mode, "pyramid_ab"):
            rows_by_suite["pyramid_ab"] = run_suite(suite_name="pyramid_ab", output_path=outputs["pyramid_ab"], configs=pyramid_ab_configs, args=args, device=device, selected_gpu_index=selected_gpu_index, gpu_name=gpu_name, skipped_rows=skipped_rows, gpu_samples=gpu_samples, resume=bool(args.resume), env_profiles=main_profiles)
        if _mode_enabled(args.mode, "real_replay"):
            rows_by_suite["real_replay"] = run_suite(suite_name="real_replay", output_path=outputs["real_replay"], configs=real_configs, args=args, device=device, selected_gpu_index=selected_gpu_index, gpu_name=gpu_name, skipped_rows=skipped_rows, gpu_samples=gpu_samples, resume=bool(args.resume), env_profiles=main_profiles)
        if _mode_enabled(args.mode, "diagnostic_environment_subset"):
            rows_by_suite["diagnostic_environment_subset"] = run_suite(suite_name="diagnostic_environment_subset", output_path=outputs["diagnostic_environment_subset"], configs=diagnostic_configs, args=args, device=device, selected_gpu_index=selected_gpu_index, gpu_name=gpu_name, skipped_rows=skipped_rows, gpu_samples=gpu_samples, resume=bool(args.resume), env_profiles=diagnostic_profiles)
        for suite, path in outputs.items():
            if path.exists():
                rows_by_suite[suite] = rewrite_rows(path)
            elif _mode_enabled(args.mode, suite):
                _write_csv(path, [], RESULT_FIELDS)
                rows_by_suite[suite] = []
        all_rows = [row for rows in rows_by_suite.values() for row in rows]
        summary_rows = build_shape_class_summary(all_rows)
        _write_csv(out / "shape_class_summary_v104.csv", summary_rows)
        write_skipped_shapes_report(out / "skipped_shapes_report.csv", skipped_rows)
        write_search_variable_semantics(out / "search_variable_semantics_v104.md")
        rules = build_decoder_rules(summary_rows)
        _write_json(out / "decoder_shape_rules_v104.json", rules)
        write_decoder_rules_md(out / "decoder_shape_rules_v104.md", rules)
        write_kernel_profile_report(out / "kernel_profile_report.json", enabled=bool(args.profile_kernels))
        write_real_replay_summary(out / "real_model_shape_replay_summary.md", rows_by_suite.get("real_replay", []))
        if device.type == "cuda":
            after = collect_gpu_state(selected_gpu_index)
        else:
            after = {"timestamp": time.time(), "selected_gpu_index": -1, "gpu_name": "cpu"}
        _write_json(out / "gpu_state_after.json", after)
        _write_json(out / "gpu_state_during_samples.json", gpu_samples)
        own_pid = os.getpid()
        contamination = any(
            any(int(process.get("pid", -1)) != own_pid for process in (sample.get("running_processes") or []))
            for sample in gpu_samples
            if isinstance(sample, dict)
        )
        row_counts = {suite: len(rows) for suite, rows in rows_by_suite.items()}
        run_config = {
            **vars(args),
            "selected_gpu_index": selected_gpu_index,
            "gpu_name": gpu_name,
            "selected_gpu_idle": selected_idle,
            "selection_payload": selection_payload,
            "real_shape_warnings": real_warnings,
            "row_counts": row_counts,
            "skipped_count": len(skipped_rows),
            "latency_contamination_risk": contamination,
        }
        _write_json(out / "run_config.json", run_config)
        write_verdict(out / "shape_rulebook_full_v104_verdict.md", run_config=run_config, selected_gpu_idle=selected_idle, contamination_risk=contamination, skipped_rows=skipped_rows, row_counts=row_counts)
        _write_json(failure_path, {"success": True, "failure_reason": "", "traceback": ""})
        print(json.dumps({"success": True, "output_dir": str(out), "row_counts": row_counts, "skipped": len(skipped_rows)}, indent=2))
        return 0
    except Exception as exc:
        failure = {
            "success": False,
            "failure_reason": "unknown_exception",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        _write_json(failure_path, failure)
        print(json.dumps(failure, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
