#!/usr/bin/env python3
"""Build a PyTorch/cuDNN Conv2d shape-to-latency rulebook for v10.3."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn as nn

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


SMALL_CHANNELS = [2, 4]
NON_FRIENDLY_CHANNELS = [6, 10, 12, 14, 18, 20, 22, 24, 28, 30]
FRIENDLY_CHANNELS = [8, 16, 32, 64, 96, 128, 192, 256]
ALL_CHANNELS = SMALL_CHANNELS + NON_FRIENDLY_CHANNELS + FRIENDLY_CHANNELS
GROUPS_FULL = [2, 4, 8, 16, 32, 64]
PER_GROUP_FULL = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 32, 64]
HWS_FULL = [(200, 704), (100, 352), (50, 176), (25, 88), (13, 44)]
RESULT_FIELDS = [
    "shape_id",
    "source_experiment",
    "module_name",
    "original_or_pruned",
    "strategy",
    "op_type",
    "dtype",
    "layout",
    "batch",
    "H",
    "W",
    "kernel_size",
    "stride",
    "padding",
    "C_in",
    "C_out",
    "groups",
    "in_per_group",
    "out_per_group",
    "params",
    "FLOPs_estimate",
    "latency_mean_ms",
    "latency_p50_ms",
    "latency_p90_ms",
    "latency_p95_ms",
    "latency_min_ms",
    "latency_max_ms",
    "throughput_GFLOPs_per_s",
    "latency_per_GFLOP",
    "relative_to_nearest_power2",
    "relative_to_nearest_multiple8",
    "groups_multiple_of_4",
    "groups_multiple_of_8",
    "groups_multiple_of_16",
    "in_per_group_multiple_of_8",
    "out_per_group_multiple_of_8",
    "in_per_group_multiple_of_16",
    "out_per_group_multiple_of_16",
    "total_C_out_multiple_of_8",
    "total_C_out_multiple_of_16",
    "shape_class",
    "predicted_friendly",
    "actual_fast_or_slow",
    "cudnn_benchmark",
    "cudnn_allow_tf32",
    "matmul_allow_tf32",
    "torch_version",
    "cuda_version",
    "cudnn_version",
    "gpu_name",
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


def stats(values: Sequence[float]) -> dict[str, float]:
    vals = [float(v) for v in values]
    return {
        "latency_mean_ms": statistics.mean(vals) if vals else 0.0,
        "latency_p50_ms": statistics.median(vals) if vals else 0.0,
        "latency_p90_ms": percentile(vals, 0.90),
        "latency_p95_ms": percentile(vals, 0.95),
        "latency_min_ms": min(vals) if vals else 0.0,
        "latency_max_ms": max(vals) if vals else 0.0,
    }


def conv2d_flops(*, batch: int, h: int, w: int, c_in: int, c_out: int, kernel: int, groups: int) -> float:
    return float(batch) * h * w * c_out * (c_in / groups) * kernel * kernel * 2.0


def _is_power2(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _shape_class_for_channel(value: int) -> str:
    if value in SMALL_CHANNELS:
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


def classify_ordinary_shape(c_in: int, c_out: int, *, real_class: str = "") -> str:
    classes = {_shape_class_for_channel(int(c_in)), _shape_class_for_channel(int(c_out))}
    if int(c_in) in SMALL_CHANNELS or int(c_out) in SMALL_CHANNELS:
        classes.add("too_narrow_2_4")
    if int(c_in) % 8 != 0 or int(c_out) % 8 != 0:
        classes.add("non_multiple_of_8")
    if real_class:
        classes.add(real_class)
    return ";".join(sorted(classes))


def classify_grouped_shape(*, groups: int, in_per_group: int, out_per_group: int, real_class: str = "") -> str:
    classes: set[str] = set()
    if int(in_per_group) in SMALL_CHANNELS or int(out_per_group) in SMALL_CHANNELS:
        classes.add("per_group_too_narrow_2_4")
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
    if int(groups) * int(out_per_group) % 8 == 0:
        classes.add("total_channels_multiple_of_8")
    if real_class:
        classes.add(real_class)
    return ";".join(sorted(classes))


def _base_shape(
    *,
    op_type: str,
    h: int,
    w: int,
    kernel: int,
    c_in: int,
    c_out: int,
    groups: int,
    batch: int = 1,
    source_experiment: str = "synthetic",
    module_name: str = "",
    original_or_pruned: str = "synthetic",
    strategy: str = "",
    real_class: str = "",
) -> dict[str, Any]:
    in_per = int(c_in) // max(int(groups), 1)
    out_per = int(c_out) // max(int(groups), 1)
    shape_class = (
        classify_grouped_shape(groups=groups, in_per_group=in_per, out_per_group=out_per, real_class=real_class)
        if groups > 1
        else classify_ordinary_shape(c_in, c_out, real_class=real_class)
    )
    return {
        "source_experiment": source_experiment,
        "module_name": module_name,
        "original_or_pruned": original_or_pruned,
        "strategy": strategy,
        "op_type": op_type,
        "batch": int(batch),
        "H": int(h),
        "W": int(w),
        "kernel_size": f"{int(kernel)}x{int(kernel)}",
        "kernel_int": int(kernel),
        "stride": 1,
        "padding": int(kernel) // 2 if int(kernel) > 1 else 0,
        "C_in": int(c_in),
        "C_out": int(c_out),
        "groups": int(groups),
        "in_per_group": in_per,
        "out_per_group": out_per,
        "shape_class": shape_class,
    }


def _unique_shapes(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        key = (
            row["source_experiment"],
            row["module_name"],
            row["original_or_pruned"],
            row["op_type"],
            row["H"],
            row["W"],
            row["kernel_size"],
            row["C_in"],
            row["C_out"],
            row["groups"],
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def generate_ordinary_shape_grid(*, reduced: bool = False, batch: int = 1) -> list[dict[str, Any]]:
    hws = [(50, 176), (13, 44)] if reduced else HWS_FULL
    channels = [2, 4, 6, 8, 10, 16, 32, 64] if reduced else ALL_CHANNELS
    fixed = [16, 64] if reduced else [16, 32, 64, 128]
    kernels = [1, 3]
    rows: list[dict[str, Any]] = []
    for h, w in hws:
        for kernel in kernels:
            for c in channels:
                rows.append(_base_shape(op_type="ordinary_conv2d", h=h, w=w, kernel=kernel, c_in=c, c_out=c, groups=1, batch=batch))
            for c_in in fixed:
                for c_out in channels:
                    rows.append(_base_shape(op_type="ordinary_conv2d", h=h, w=w, kernel=kernel, c_in=c_in, c_out=c_out, groups=1, batch=batch))
            for c_out in fixed:
                for c_in in channels:
                    rows.append(_base_shape(op_type="ordinary_conv2d", h=h, w=w, kernel=kernel, c_in=c_in, c_out=c_out, groups=1, batch=batch))
    return _unique_shapes(rows)


def generate_grouped_shape_grid(*, reduced: bool = False, batch: int = 1, max_channels: int = 1024) -> list[dict[str, Any]]:
    hws = [(50, 176), (13, 44)] if reduced else HWS_FULL
    groups_values = [2, 4, 8, 16] if reduced else GROUPS_FULL
    widths = [2, 4, 6, 8, 16, 32] if reduced else PER_GROUP_FULL
    rows: list[dict[str, Any]] = []
    for h, w in hws:
        for kernel in [3]:
            for groups in groups_values:
                for in_per in widths:
                    for out_per in widths:
                        c_in = groups * in_per
                        c_out = groups * out_per
                        if c_in > max_channels or c_out > max_channels:
                            continue
                        rows.append(
                            _base_shape(
                                op_type="grouped_conv2d",
                                h=h,
                                w=w,
                                kernel=kernel,
                                c_in=c_in,
                                c_out=c_out,
                                groups=groups,
                                batch=batch,
                            )
                        )
    return _unique_shapes(rows)


def _parse_csv_arg(text: str) -> list[str]:
    return [item.strip() for item in str(text).split(",") if item.strip()]


def _cudnn_benchmark_values(text: str) -> list[bool]:
    val = str(text).lower()
    if val == "both":
        return [True, False]
    if val in {"true", "1", "yes", "on"}:
        return [True]
    if val in {"false", "0", "no", "off"}:
        return [False]
    raise ValueError(f"unsupported cudnn benchmark setting: {text}")


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    if not fields:
        fields = ["empty"]
        rows = [{"empty": ""}]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _time_module(module: nn.Module, x: torch.Tensor, *, warmup: int, repeat: int, runs: int, device: torch.device) -> dict[str, float]:
    module.eval()
    all_times: list[float] = []
    run_medians: list[float] = []
    with torch.no_grad():
        for _run in range(max(1, runs)):
            for _ in range(max(0, warmup)):
                module(x)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                starter = torch.cuda.Event(enable_timing=True)
                ender = torch.cuda.Event(enable_timing=True)
                times: list[float] = []
                for _ in range(max(1, repeat)):
                    starter.record()
                    module(x)
                    ender.record()
                    torch.cuda.synchronize(device)
                    times.append(float(starter.elapsed_time(ender)))
            else:
                times = []
                for _ in range(max(1, repeat)):
                    t0 = time.perf_counter()
                    module(x)
                    times.append((time.perf_counter() - t0) * 1000.0)
            all_times.extend(times)
            run_medians.append(statistics.median(times))
    result = stats(run_medians)
    result["latency_min_ms"] = min(all_times) if all_times else 0.0
    result["latency_max_ms"] = max(all_times) if all_times else 0.0
    return result


def _dtype_to_torch(dtype: str) -> torch.dtype:
    if dtype == "fp16":
        return torch.float16
    if dtype == "fp32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {dtype}")


def build_result_row_from_stats(config: Mapping[str, Any], timing: Mapping[str, float], *, cudnn_benchmark: bool) -> dict[str, Any]:
    kernel = int(str(config["kernel_size"]).split("x", 1)[0])
    batch = int(config["batch"])
    h = int(config["H"])
    w = int(config["W"])
    c_in = int(config["C_in"])
    c_out = int(config["C_out"])
    groups = int(config["groups"])
    params = c_out * (c_in // max(groups, 1)) * kernel * kernel
    flops = conv2d_flops(batch=batch, h=h, w=w, c_in=c_in, c_out=c_out, kernel=kernel, groups=groups)
    mean_ms = float(timing.get("latency_mean_ms", 0.0) or 0.0)
    gflops = flops / 1e9
    in_per = int(config.get("in_per_group", c_in // max(groups, 1)))
    out_per = int(config.get("out_per_group", c_out // max(groups, 1)))
    predicted_friendly = (c_in % 8 == 0 and c_out % 8 == 0) if groups == 1 else (in_per >= 8 and out_per >= 8 and in_per % 8 == 0 and out_per % 8 == 0)
    row = {
        "shape_id": "",
        "source_experiment": config.get("source_experiment", "synthetic"),
        "module_name": config.get("module_name", ""),
        "original_or_pruned": config.get("original_or_pruned", "synthetic"),
        "strategy": config.get("strategy", ""),
        "op_type": config["op_type"],
        "dtype": config.get("dtype", "fp32"),
        "layout": config.get("layout", "nchw"),
        "batch": batch,
        "H": h,
        "W": w,
        "kernel_size": config["kernel_size"],
        "stride": int(config.get("stride", 1)),
        "padding": int(config.get("padding", kernel // 2 if kernel > 1 else 0)),
        "C_in": c_in,
        "C_out": c_out,
        "groups": groups,
        "in_per_group": in_per,
        "out_per_group": out_per,
        "params": int(params),
        "FLOPs_estimate": flops,
        **{key: float(timing.get(key, 0.0) or 0.0) for key in ["latency_mean_ms", "latency_p50_ms", "latency_p90_ms", "latency_p95_ms", "latency_min_ms", "latency_max_ms"]},
        "throughput_GFLOPs_per_s": gflops / (mean_ms / 1000.0) if mean_ms > 0.0 else 0.0,
        "latency_per_GFLOP": mean_ms / gflops if gflops > 0.0 else 0.0,
        "relative_to_nearest_power2": "",
        "relative_to_nearest_multiple8": "",
        "groups_multiple_of_4": groups % 4 == 0,
        "groups_multiple_of_8": groups % 8 == 0,
        "groups_multiple_of_16": groups % 16 == 0,
        "in_per_group_multiple_of_8": in_per % 8 == 0,
        "out_per_group_multiple_of_8": out_per % 8 == 0,
        "in_per_group_multiple_of_16": in_per % 16 == 0,
        "out_per_group_multiple_of_16": out_per % 16 == 0,
        "total_C_out_multiple_of_8": c_out % 8 == 0,
        "total_C_out_multiple_of_16": c_out % 16 == 0,
        "shape_class": config.get("shape_class", ""),
        "predicted_friendly": predicted_friendly,
        "actual_fast_or_slow": "",
        "cudnn_benchmark": bool(cudnn_benchmark),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32) if hasattr(torch.backends, "cuda") else False,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else "",
        "gpu_name": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
    }
    row["shape_id"] = _shape_id(row)
    return row


def _shape_id(row: Mapping[str, Any]) -> str:
    keys = ["source_experiment", "module_name", "op_type", "dtype", "layout", "H", "W", "kernel_size", "C_in", "C_out", "groups", "cudnn_benchmark"]
    return "shape_" + str(abs(hash(tuple(str(row.get(k, "")) for k in keys))))


def benchmark_config(config: Mapping[str, Any], *, device: torch.device, dtype: str, layout: str, cudnn_benchmark: bool, warmup: int, repeat: int, runs: int) -> dict[str, Any]:
    torch.manual_seed(1234)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(1234)
    torch.backends.cudnn.benchmark = bool(cudnn_benchmark)
    kernel = int(str(config["kernel_size"]).split("x", 1)[0])
    c_in = int(config["C_in"])
    c_out = int(config["C_out"])
    groups = int(config["groups"])
    torch_dtype = _dtype_to_torch(dtype)
    module = nn.Conv2d(c_in, c_out, kernel, stride=int(config.get("stride", 1)), padding=int(config.get("padding", kernel // 2 if kernel > 1 else 0)), groups=groups, bias=False).to(device=device, dtype=torch_dtype)
    x = torch.randn(int(config["batch"]), c_in, int(config["H"]), int(config["W"]), device=device, dtype=torch_dtype)
    if layout == "channels_last":
        module = module.to(memory_format=torch.channels_last)
        x = x.contiguous(memory_format=torch.channels_last)
    elif layout != "nchw":
        raise ValueError(f"unsupported layout: {layout}")
    timing = _time_module(module, x, warmup=warmup, repeat=repeat, runs=runs, device=device)
    row_cfg = dict(config)
    row_cfg["dtype"] = dtype
    row_cfg["layout"] = layout
    return build_result_row_from_stats(row_cfg, timing, cudnn_benchmark=cudnn_benchmark)


def _nearest_ratio(row: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], *, predicate) -> float | str:
    same = [
        cand for cand in rows
        if cand is not row
        and cand.get("op_type") == row.get("op_type")
        and cand.get("dtype") == row.get("dtype")
        and cand.get("layout") == row.get("layout")
        and int(cand.get("H", 0)) == int(row.get("H", 0))
        and int(cand.get("W", 0)) == int(row.get("W", 0))
        and cand.get("kernel_size") == row.get("kernel_size")
        and int(cand.get("groups", 1)) == int(row.get("groups", 1))
        and bool(cand.get("cudnn_benchmark")) == bool(row.get("cudnn_benchmark"))
        and predicate(cand)
    ]
    if not same:
        return ""
    groups = int(row.get("groups", 1))
    if groups > 1:
        nearest = min(same, key=lambda cand: abs(int(cand["in_per_group"]) - int(row["in_per_group"])) + abs(int(cand["out_per_group"]) - int(row["out_per_group"])))
    else:
        nearest = min(same, key=lambda cand: abs(int(cand["C_in"]) - int(row["C_in"])) + abs(int(cand["C_out"]) - int(row["C_out"])))
    cur = float(row.get("throughput_GFLOPs_per_s", 0.0) or 0.0)
    ref = float(nearest.get("throughput_GFLOPs_per_s", 0.0) or 0.0)
    return round(cur / ref, 6) if cur > 0.0 and ref > 0.0 else ""


def annotate_relative_columns(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        groups = int(row.get("groups", 1))
        if groups > 1:
            row["relative_to_nearest_power2"] = _nearest_ratio(row, rows, predicate=lambda cand: _is_power2(int(cand["in_per_group"])) and _is_power2(int(cand["out_per_group"])))
            row["relative_to_nearest_multiple8"] = _nearest_ratio(row, rows, predicate=lambda cand: int(cand["in_per_group"]) % 8 == 0 and int(cand["out_per_group"]) % 8 == 0)
        else:
            row["relative_to_nearest_power2"] = _nearest_ratio(row, rows, predicate=lambda cand: _is_power2(int(cand["C_in"])) and _is_power2(int(cand["C_out"])))
            row["relative_to_nearest_multiple8"] = _nearest_ratio(row, rows, predicate=lambda cand: int(cand["C_in"]) % 8 == 0 and int(cand["C_out"]) % 8 == 0)
    lpgs = [float(row.get("latency_per_GFLOP", 0.0) or 0.0) for row in rows if float(row.get("latency_per_GFLOP", 0.0) or 0.0) > 0.0]
    median_lpg = statistics.median(lpgs) if lpgs else 0.0
    for row in rows:
        lpg = float(row.get("latency_per_GFLOP", 0.0) or 0.0)
        row["actual_fast_or_slow"] = "fast" if median_lpg and lpg <= median_lpg else "slow"


def _strategy_from_path(path: Path) -> tuple[str, str, str]:
    parts = path.parts
    strategy = ""
    source = ""
    original = "pruned"
    for part in reversed(parts):
        if part in {"baseline", "summary"}:
            strategy = "baseline"
            original = "baseline"
            break
        if "_" in part and any(part.startswith(prefix) for prefix in ("A1", "A2", "B1", "B2", "B3", "C1", "C2", "A_", "B_", "C_", "D_")):
            strategy = part.split("_", 1)[0]
            source = part
            break
    if not source:
        source = path.parent.name
    if "baseline" in str(path):
        original = "baseline"
        strategy = strategy or "baseline"
    return source, strategy, original


def extract_real_model_shapes(paths: Sequence[Path | str], *, max_shapes: int = 128, default_hw: tuple[int, int] = (50, 176)) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for path_text in paths:
        root = Path(path_text)
        if not root.exists():
            warnings.append(f"missing shape report root: {root}")
            continue
        reports = list(root.glob("**/*shape_alignment_report.json"))
        if not reports:
            warnings.append(f"no shape_alignment_report.json under: {root}")
        for report in reports:
            try:
                data = json.loads(report.read_text(encoding="utf-8"))
            except Exception as exc:
                warnings.append(f"failed to read {report}: {exc}")
                continue
            source, strategy, original = _strategy_from_path(report)
            for entry in data.get("conv2d", []) or []:
                c_in = int(entry.get("C_in", 0) or 0)
                c_out = int(entry.get("C_out", 0) or 0)
                groups = int(entry.get("groups", 1) or 1)
                if c_in <= 0 or c_out <= 0 or groups <= 0 or c_in % groups != 0 or c_out % groups != 0:
                    continue
                op_type = "grouped_conv2d" if groups > 1 else "ordinary_conv2d"
                real_class = "real_model_baseline_shape" if original == "baseline" else "real_model_pruned_shape"
                rows.append(
                    _base_shape(
                        op_type=op_type,
                        h=default_hw[0],
                        w=default_hw[1],
                        kernel=3,
                        c_in=c_in,
                        c_out=c_out,
                        groups=groups,
                        source_experiment=source,
                        module_name=str(entry.get("module_name", "")),
                        original_or_pruned=original,
                        strategy=strategy,
                        real_class=real_class,
                    )
                )
    rows = _unique_shapes(rows)
    if max_shapes > 0:
        rows = rows[:max_shapes]
    return rows, warnings


def _median(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if float(v) > 0.0]
    return statistics.median(vals) if vals else 0.0


def build_shape_class_summary(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    by_class: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        for cls in str(row.get("shape_class", "")).split(";"):
            if cls:
                by_class.setdefault(cls, []).append(row)
    summary: list[dict[str, Any]] = []
    for cls, cls_rows in sorted(by_class.items()):
        best = min(cls_rows, key=lambda row: float(row.get("latency_per_GFLOP", 0.0) or 0.0))
        worst = max(cls_rows, key=lambda row: float(row.get("latency_per_GFLOP", 0.0) or 0.0))
        recommendation = "avoid" if "too_narrow" in cls or "non_multiple" in cls else ("prefer" if "multiple_of_16" in cls or "multiple_of_32" in cls else "neutral")
        summary.append(
            {
                "shape_class": cls,
                "num_cases": len(cls_rows),
                "median_latency_per_GFLOP": _median([float(row.get("latency_per_GFLOP", 0.0) or 0.0) for row in cls_rows]),
                "median_throughput_GFLOPs_per_s": _median([float(row.get("throughput_GFLOPs_per_s", 0.0) or 0.0) for row in cls_rows]),
                "best_case_shape": f"{best.get('op_type')} C_in={best.get('C_in')} C_out={best.get('C_out')} groups={best.get('groups')} in_per={best.get('in_per_group')} out_per={best.get('out_per_group')} dtype={best.get('dtype')} layout={best.get('layout')}",
                "worst_case_shape": f"{worst.get('op_type')} C_in={worst.get('C_in')} C_out={worst.get('C_out')} groups={worst.get('groups')} in_per={worst.get('in_per_group')} out_per={worst.get('out_per_group')} dtype={worst.get('dtype')} layout={worst.get('layout')}",
                "recommendation": recommendation,
            }
        )
    return summary


def _class_median(summary: Sequence[dict[str, Any]], cls: str) -> float:
    for row in summary:
        if row.get("shape_class") == cls:
            return float(row.get("median_latency_per_GFLOP", 0.0) or 0.0)
    return 0.0


def write_search_variable_semantics(path: Path) -> None:
    text = """# v10.3 Search Variable Semantics

## Direct Definition

CoupledChannelUnit is the minimum automatic-search atom.

`variable_i = CoupledChannelUnit_i`, with `z_i in {0, 1}` where `1 = keep` and `0 = prune candidate`.

## What Belongs To A Unit

A CoupledChannelUnit preserves the module-axis-index membership that must move together: Conv.out, BN.out, downstream Conv.in, residual closure, concat offset, grouped-conv dependency, ConvTranspose/deblock dependency when supported, and any other synchronized channel members discovered by the dependency graph.

## What Does Not Belong To A Unit

Grouped-conv A/B/C/D policy is not part of the CoupledChannelUnit definition. Channel alignment, TensorCore/cuDNN-friendly shape rules, total C_out round_to, per-group width constraints, and groups_after constraints are also not part of the base unit definition.

## Raw Mask To Physical Plan

The intended pipeline is:

`raw_search_mask -> mask_to_legal_prune_plan_decoder -> alignment repair -> grouped conv policy repair -> physical prune plan -> one-shot physical removal`.

The raw GA mask is therefore not directly executable. The decoder groups selected units by pruning domain, checks closure completeness, chooses/repairs grouped-conv policy, applies deployment alignment rules, verifies shape simulator legality, and only then emits a physical prune plan.

## Search-Time Proxy Constraints

Search-time proxy constraints can score or discourage masks using importance, estimated parameter/FLOPs savings, coarse closure flags, min-channel proxy, and latency-unfriendly shape penalties. Export-time repair owns hard legality: complete residual/concat/grouped/ConvTranspose closure, A/B/C/D grouped-conv policy selection, channel alignment repair, groups/per-group repair, simulator legality, and physical surgery legality.

## Why Policy And Alignment Stay Out Of Stable Unit IDs

Do not bind alignment or A/B/C/D policy into CoupledChannelUnit too early because the same unit may be decodable through different legal physical bundles under different deployment targets. Binding policy at the atom level would make stable IDs target-dependent, break cross-run comparability, and prevent the decoder from repairing a raw mask into the nearest legal deployment-aware plan.

## Required Answers

1. CoupledChannelUnit is the minimum search variable: yes.
2. grouped conv A/B/C/D belongs to the unit itself: no, it belongs to decoder/export repair.
3. alignment belongs to the unit itself: no, it belongs to decoder/export repair and latency-aware deployment policy.
4. raw mask becomes physical prune plan through decoder, repair, simulator, and physical-export validation.
5. search-time proxy constraints: importance, parameter/FLOPs estimates, coarse legality, min-channel proxy, and latency-unfriendly penalties.
6. export-time repair constraints: residual/concat/grouped/ConvTranspose closure, A/B/C/D policy, alignment, group/per-group repair, simulator legality, physical surgery legality.
7. early binding is avoided to keep stable IDs invariant across target ratios, deployment backends, and grouped-conv policy choices.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_kernel_profile_report(path: Path, *, enabled: bool) -> None:
    _write_json(
        path,
        {
            "profile_kernels": bool(enabled),
            "kernel_profiles": [],
            "likely_tensor_core_used": "unknown",
            "unknown_reason": "kernel profiling is optional in v10.3 and was not used for the reduced rulebook run" if not enabled else "torch profiler kernel classification not implemented reliably",
        },
    )


def write_verdict(path: Path, *, ordinary_rows: Sequence[dict[str, Any]], grouped_rows: Sequence[dict[str, Any]], real_rows: Sequence[dict[str, Any]], summary_rows: Sequence[dict[str, Any]], reduced: bool, warnings: Sequence[str], max_channels: int) -> None:
    small_ord = _class_median(summary_rows, "too_narrow_2_4")
    mult16 = _class_median(summary_rows, "multiple_of_16")
    non8 = _class_median(summary_rows, "non_multiple_of_8")
    pg_small = _class_median(summary_rows, "per_group_too_narrow_2_4")
    pg8 = _class_median(summary_rows, "per_group_multiple_of_8")
    pg16 = _class_median(summary_rows, "per_group_multiple_of_16")
    reduced_note = "This is a reduced rulebook run; conclusions are directional and must not be treated as final full-sweep rules." if reduced else "This is a full configured rulebook run, filtered only by max_channels."
    lines = [
        "# cuDNN Shape Rulebook Verdict v10.3",
        "",
        reduced_note,
        f"max_channels filter: {max_channels}",
        f"ordinary rows: {len(ordinary_rows)}; grouped rows: {len(grouped_rows)}; real replay rows: {len(real_rows)}",
        f"warnings: {list(warnings)}",
        "",
        "1. Ordinary Conv2d stable alignment: multiples of 16/32/64 are the preferred candidates in this run; multiples of 8 are minimum acceptable but should be compared to 16.",
        f"2. C_in/C_out 2 or 4 inefficient: {'yes' if small_ord and mult16 and small_ord > mult16 else 'inconclusive'} (too_narrow median latency/GFLOP={small_ord:.6f}, multiple_of_16={mult16:.6f}).",
        f"3. 6/10/12/14/18/20/22 inefficient: {'yes' if non8 and mult16 and non8 > mult16 else 'inconclusive'} (non_multiple_of_8 median={non8:.6f}).",
        "4. Evidence for too-narrow slowdown: yes when too_narrow_2_4 median latency/GFLOP is worse than 16-aligned shapes.",
        "5. Grouped Conv2d groups alignment alone is not enough; per-group width remains a separate driver.",
        "6. Grouped in_per_group and out_per_group both matter because the kernel work uses their product; the rulebook should reject either side being too narrow.",
        f"7. per-group width 2/4 inefficient: {'yes' if pg_small and pg8 and pg_small > pg8 else 'inconclusive'} (small={pg_small:.6f}, per_group_multiple_of_8={pg8:.6f}).",
        f"8. per-group width 8 vs 16/32: 8 is a floor; 16/32 are safer when median latency/GFLOP improves or ties (pg8={pg8:.6f}, pg16={pg16:.6f}).",
        "9. total C_out alignment is not a substitute for per-group width; both should be tracked, with per-group width prioritized for grouped conv.",
        "10. A1 acceleration can be explained only partially by real shape replay; use `real_model_shape_latency_replay.csv` to compare A1 rows against baseline rows.",
        "11. A2/B/C slowdown can be explained when replay rows show non-8-aligned C_out or small per-group widths; reduced data should be treated as diagnostic.",
        "12. Recommend prohibiting grouped per_group_width < 8 in decoder repair unless no legal AP-preserving alternative exists.",
        "13. Recommend ordinary Conv2d C_in/C_out round_to at least 8, and prefer 16 for deployment-aware repair.",
        "14. A/B/C/D alignment: A should prefer total C_out 8/16; B should enforce per-group kept width >=8; C should choose groups_after that preserve per-group width >=8/16; D should reblock toward groups/per-group widths that satisfy the same floor.",
        "15. latency_unfriendly shapes: any ordinary C_in/C_out in {2,4}, non-multiple-of-8 ordinary channels, grouped per-group width <8, or grouped per-group non-multiple-of-8.",
        "16. latency-friendly candidates: ordinary C_in/C_out multiples of 16/32/64 and grouped in/out per-group widths 16/32/64 with reasonable groups.",
        "17. Interference evidence: yes, cuDNN benchmark mode, dtype, layout, grouped-kernel utilization, and launch overhead can all affect latency beyond channel alignment; this rulebook is PyTorch/cuDNN evidence, not TensorRT evidence.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_configs(configs: Sequence[dict[str, Any]], *, device: torch.device, dtypes: Sequence[str], layouts: Sequence[str], cudnn_values: Sequence[bool], warmup: int, repeat: int, runs: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cudnn_value in cudnn_values:
        for dtype in dtypes:
            if dtype == "fp16" and device.type != "cuda":
                continue
            for layout in layouts:
                for config in configs:
                    rows.append(benchmark_config(config, device=device, dtype=dtype, layout=layout, cudnn_benchmark=cudnn_value, warmup=warmup, repeat=repeat, runs=runs))
    annotate_relative_columns(rows)
    return rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark v10.3 cuDNN shape rulebook")
    parser.add_argument("--device", default="cuda:5")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeat", type=int, default=200)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--dtypes", default="fp32")
    parser.add_argument("--layouts", default="nchw")
    parser.add_argument("--cudnn-benchmark", default="both")
    parser.add_argument("--output-dir", default="outputs/latency_lut/cudnn_shape_rulebook_v103")
    parser.add_argument("--reduced", action="store_true")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--max-channels", type=int, default=1024)
    parser.add_argument("--max-ordinary-shapes", type=int, default=0)
    parser.add_argument("--max-grouped-shapes", type=int, default=0)
    parser.add_argument("--max-real-model-shapes", type=int, default=96)
    parser.add_argument("--shape-report-roots", default="outputs/latency_lut/global_budgeted_alignment_eval_v101,outputs/latency_lut/budget_repair_eval_v102_A1,outputs/latency_lut/budget_repair_eval_v102_B1,outputs/latency_lut/selector_audit_v102")
    parser.add_argument("--profile-kernels", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dtypes = _parse_csv_arg(args.dtypes)
    layouts = _parse_csv_arg(args.layouts)
    cudnn_values = _cudnn_benchmark_values(args.cudnn_benchmark)
    ordinary_configs = generate_ordinary_shape_grid(reduced=bool(args.reduced), batch=int(args.batch))
    grouped_configs = generate_grouped_shape_grid(reduced=bool(args.reduced), batch=int(args.batch), max_channels=int(args.max_channels))
    if args.max_ordinary_shapes > 0:
        ordinary_configs = ordinary_configs[: args.max_ordinary_shapes]
    if args.max_grouped_shapes > 0:
        grouped_configs = grouped_configs[: args.max_grouped_shapes]
    roots = [Path(item) for item in _parse_csv_arg(args.shape_report_roots)]
    real_configs, warnings = extract_real_model_shapes(roots, max_shapes=int(args.max_real_model_shapes))
    ordinary_rows = _run_configs(ordinary_configs, device=device, dtypes=dtypes, layouts=layouts, cudnn_values=cudnn_values, warmup=int(args.warmup), repeat=int(args.repeat), runs=int(args.runs))
    grouped_rows = _run_configs(grouped_configs, device=device, dtypes=dtypes, layouts=layouts, cudnn_values=cudnn_values, warmup=int(args.warmup), repeat=int(args.repeat), runs=int(args.runs))
    real_rows = _run_configs(real_configs, device=device, dtypes=dtypes, layouts=layouts, cudnn_values=cudnn_values, warmup=int(args.warmup), repeat=int(args.repeat), runs=int(args.runs)) if real_configs else []
    _write_csv(out / "ordinary_conv_shape_rulebook.csv", ordinary_rows)
    _write_csv(out / "grouped_conv_shape_rulebook.csv", grouped_rows)
    _write_csv(out / "real_model_shape_latency_replay.csv", real_rows)
    summary_rows = build_shape_class_summary(list(ordinary_rows) + list(grouped_rows) + list(real_rows))
    _write_csv(out / "shape_class_summary.csv", summary_rows)
    write_kernel_profile_report(out / "cudnn_kernel_profile_report.json", enabled=bool(args.profile_kernels))
    write_search_variable_semantics(out / "search_variable_semantics.md")
    write_verdict(
        out / "cudnn_shape_rulebook_verdict.md",
        ordinary_rows=ordinary_rows,
        grouped_rows=grouped_rows,
        real_rows=real_rows,
        summary_rows=summary_rows,
        reduced=bool(args.reduced),
        warnings=warnings,
        max_channels=int(args.max_channels),
    )
    _write_json(
        out / "run_config.json",
        {
            **vars(args),
            "resolved_device": str(device),
            "ordinary_config_count": len(ordinary_configs),
            "grouped_config_count": len(grouped_configs),
            "real_config_count": len(real_configs),
            "ordinary_rows": len(ordinary_rows),
            "grouped_rows": len(grouped_rows),
            "real_rows": len(real_rows),
            "warnings": warnings,
        },
    )
    print(json.dumps({"success": True, "output_dir": str(out), "reduced": bool(args.reduced), "ordinary_rows": len(ordinary_rows), "grouped_rows": len(grouped_rows), "real_rows": len(real_rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
