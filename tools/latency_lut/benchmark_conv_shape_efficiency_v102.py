#!/usr/bin/env python3
"""Benchmark Conv2d/grouped Conv2d shape efficiency on PyTorch/cuDNN."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


FRIENDLY = {8, 16, 32, 64, 128, 256}
SMALL = {2, 4}
ATYPICAL = {6, 10, 12, 14, 18, 20, 22, 24, 28, 30}


def _percentile(values: Sequence[float], pct: float) -> float:
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


def _stats(values: Sequence[float]) -> dict[str, float]:
    vals = list(values)
    return {
        "latency_mean_ms": statistics.mean(vals) if vals else 0.0,
        "latency_p50_ms": statistics.median(vals) if vals else 0.0,
        "latency_p90_ms": _percentile(vals, 0.90),
        "latency_p95_ms": _percentile(vals, 0.95),
    }


def _conv_flops(batch: int, h: int, w: int, c_in: int, c_out: int, kernel: int, groups: int) -> float:
    return float(batch) * h * w * c_out * (c_in / groups) * kernel * kernel * 2.0


def _time_module(module: nn.Module, x: torch.Tensor, *, warmup: int, repeat: int, runs: int) -> dict[str, float]:
    module.eval()
    run_medians: list[float] = []
    all_times: list[float] = []
    with torch.no_grad():
        for _ in range(max(1, runs)):
            for _warm in range(warmup):
                module(x)
            torch.cuda.synchronize()
            times: list[float] = []
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            for _rep in range(repeat):
                starter.record()
                module(x)
                ender.record()
                torch.cuda.synchronize()
                times.append(float(starter.elapsed_time(ender)))
            all_times.extend(times)
            run_medians.append(statistics.median(times))
    stats = _stats(run_medians)
    stats["latency_min_ms"] = min(all_times) if all_times else 0.0
    stats["latency_max_ms"] = max(all_times) if all_times else 0.0
    return stats


def _benchmark_shape(
    *,
    op_type: str,
    batch: int,
    h: int,
    w: int,
    kernel: int,
    c_in: int,
    c_out: int,
    groups: int,
    device: torch.device,
    warmup: int,
    repeat: int,
    runs: int,
    cudnn_benchmark: bool,
) -> dict[str, Any]:
    torch.backends.cudnn.benchmark = bool(cudnn_benchmark)
    padding = kernel // 2 if kernel > 1 else 0
    module = nn.Conv2d(c_in, c_out, kernel, padding=padding, groups=groups, bias=False).to(device)
    x = torch.randn(batch, c_in, h, w, device=device)
    stats = _time_module(module, x, warmup=warmup, repeat=repeat, runs=runs)
    params = sum(p.numel() for p in module.parameters())
    flops = _conv_flops(batch, h, w, c_in, c_out, kernel, groups)
    mean_ms = stats["latency_mean_ms"]
    gflops = flops / 1e9
    return {
        "op_type": op_type,
        "batch": batch,
        "H": h,
        "W": w,
        "kernel": f"{kernel}x{kernel}",
        "C_in": c_in,
        "C_out": c_out,
        "groups": groups,
        "in_per_group": c_in // groups,
        "out_per_group": c_out // groups,
        "params": params,
        "FLOPs": flops,
        **stats,
        "throughput_GFLOPs_per_s": gflops / (mean_ms / 1000.0) if mean_ms > 0.0 else 0.0,
        "latency_per_GFLOP": mean_ms / gflops if gflops > 0.0 else 0.0,
        "speed_relative_to_nearest_friendly_shape": "",
        "cudnn_benchmark": bool(cudnn_benchmark),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
    }


def _ordinary_shapes() -> list[tuple[int, int, int, int, int]]:
    hws = [(200, 704), (100, 352), (50, 176), (25, 88), (13, 44)]
    channels = sorted(FRIENDLY | SMALL | ATYPICAL)
    shapes: list[tuple[int, int, int, int, int]] = []
    for h, w in hws:
        for kernel in (1, 3):
            for c in channels:
                if h * w > 40000 and c > 64:
                    continue
                shapes.append((h, w, kernel, c, c))
    for kernel in (1, 3):
        for c in channels:
            shapes.append((50, 176, kernel, 64, c))
            shapes.append((50, 176, kernel, c, 64))
    return sorted(set(shapes))


def _grouped_shapes() -> list[tuple[int, int, int, int, int, int]]:
    widths = [2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 32]
    groups_values = [1, 2, 4, 8, 16, 32, 64]
    shapes: list[tuple[int, int, int, int, int, int]] = []
    for groups in groups_values:
        for in_per in widths:
            for out_per in widths:
                c_in = groups * in_per
                c_out = groups * out_per
                if c_in > 512 or c_out > 512:
                    continue
                shapes.append((50, 176, 3, c_in, c_out, groups))
    return shapes


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _latency_per_gflop(row: dict[str, Any]) -> float:
    try:
        return float(row.get("latency_per_GFLOP", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _throughput(row: dict[str, Any]) -> float:
    try:
        return float(row.get("throughput_GFLOPs_per_s", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _is_friendly_shape(row: dict[str, Any]) -> bool:
    groups = int(row.get("groups", 1) or 1)
    if groups > 1:
        return int(row.get("in_per_group", 0) or 0) in FRIENDLY and int(row.get("out_per_group", 0) or 0) in FRIENDLY
    return int(row.get("C_in", 0) or 0) in FRIENDLY and int(row.get("C_out", 0) or 0) in FRIENDLY


def _shape_bucket_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("op_type"),
        row.get("H"),
        row.get("W"),
        row.get("kernel"),
        row.get("groups"),
        row.get("cudnn_benchmark"),
    )


def _friendly_distance(row: dict[str, Any], candidate: dict[str, Any]) -> int:
    groups = int(row.get("groups", 1) or 1)
    if groups > 1:
        return abs(int(row.get("in_per_group", 0) or 0) - int(candidate.get("in_per_group", 0) or 0)) + abs(
            int(row.get("out_per_group", 0) or 0) - int(candidate.get("out_per_group", 0) or 0)
        )
    return abs(int(row.get("C_in", 0) or 0) - int(candidate.get("C_in", 0) or 0)) + abs(
        int(row.get("C_out", 0) or 0) - int(candidate.get("C_out", 0) or 0)
    )


def _annotate_relative_speed_to_friendly(rows: Sequence[dict[str, Any]]) -> None:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        buckets.setdefault(_shape_bucket_key(row), []).append(row)

    for bucket_rows in buckets.values():
        friendly = [row for row in bucket_rows if _is_friendly_shape(row)]
        if not friendly:
            continue
        for row in bucket_rows:
            nearest = min(friendly, key=lambda cand: (_friendly_distance(row, cand), _latency_per_gflop(cand)))
            row_throughput = _throughput(row)
            nearest_throughput = _throughput(nearest)
            if row_throughput > 0.0 and nearest_throughput > 0.0:
                rel = row_throughput / nearest_throughput
            else:
                row_lpg = _latency_per_gflop(row)
                nearest_lpg = _latency_per_gflop(nearest)
                rel = nearest_lpg / row_lpg if row_lpg > 0.0 and nearest_lpg > 0.0 else 0.0
            row["speed_relative_to_nearest_friendly_shape"] = round(float(rel), 6)
            row["nearest_friendly_shape"] = (
                f"C_in={nearest.get('C_in')},C_out={nearest.get('C_out')},groups={nearest.get('groups')}"
            )


def _summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    def group_values(predicate) -> list[float]:
        return [float(row["latency_per_GFLOP"]) for row in rows if predicate(row) and float(row.get("latency_per_GFLOP", 0.0) or 0.0) > 0.0]

    def med(vals: Sequence[float]) -> float:
        return statistics.median(vals) if vals else 0.0

    return {
        "small_width_latency_per_gflop": med(group_values(lambda r: int(r["in_per_group"]) in SMALL or int(r["out_per_group"]) in SMALL)),
        "friendly_width_latency_per_gflop": med(group_values(lambda r: int(r["in_per_group"]) in FRIENDLY and int(r["out_per_group"]) in FRIENDLY)),
        "atypical_width_latency_per_gflop": med(group_values(lambda r: int(r["in_per_group"]) in ATYPICAL or int(r["out_per_group"]) in ATYPICAL)),
        "c_out_small_latency_per_gflop": med(group_values(lambda r: int(r.get("C_out", 0) or 0) in SMALL)),
        "c_out_friendly_latency_per_gflop": med(group_values(lambda r: int(r.get("C_out", 0) or 0) in FRIENDLY)),
        "aligned_groups_latency_per_gflop": med(group_values(lambda r: int(r.get("groups", 1) or 1) in {4, 8, 16, 32})),
    }


def _best_row(rows: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    valid = [row for row in rows if _latency_per_gflop(row) > 0.0]
    return min(valid, key=_latency_per_gflop) if valid else None


def _row_label(row: dict[str, Any] | None) -> str:
    if not row:
        return "n/a"
    return (
        f"C_in={row.get('C_in')} C_out={row.get('C_out')} groups={row.get('groups')} "
        f"in_per_group={row.get('in_per_group')} out_per_group={row.get('out_per_group')} "
        f"kernel={row.get('kernel')} benchmark={row.get('cudnn_benchmark')} "
        f"latency_per_GFLOP={_latency_per_gflop(row):.6f}"
    )


def _yes_no(condition: bool) -> str:
    return "yes" if condition else "no"


def _write_summary(
    path: Path,
    ordinary_rows: Sequence[dict[str, Any]],
    grouped_rows: Sequence[dict[str, Any]],
    *,
    real_rows: Sequence[dict[str, Any]] | None = None,
) -> None:
    ordinary = _summarize(ordinary_rows)
    grouped = _summarize(grouped_rows)
    real_rows = list(real_rows or [])
    small_bad = grouped["small_width_latency_per_gflop"] > grouped["friendly_width_latency_per_gflop"]
    atypical_bad = grouped["atypical_width_latency_per_gflop"] > grouped["friendly_width_latency_per_gflop"]
    ordinary_small_bad = ordinary["c_out_small_latency_per_gflop"] > ordinary["c_out_friendly_latency_per_gflop"]
    ordinary_atypical_bad = ordinary["atypical_width_latency_per_gflop"] > ordinary["friendly_width_latency_per_gflop"]
    grouped_alignment_alone = grouped["aligned_groups_latency_per_gflop"] <= grouped["friendly_width_latency_per_gflop"]
    best_ordinary = _best_row(ordinary_rows)
    best_grouped = _best_row(grouped_rows)
    lines = [
        "# v10.2 Conv Shape Efficiency Summary",
        "",
        f"GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu'}",
        f"PyTorch: {torch.__version__}; CUDA: {torch.version.cuda}",
        "",
        "## Direct Answers",
        "",
        "1. Ordinary Conv2d C_in/C_out alignment: friendly C values have median latency/GFLOP "
        + f"{ordinary['friendly_width_latency_per_gflop']:.6f}; small values have "
        + f"{ordinary['small_width_latency_per_gflop']:.6f}; atypical values have "
        + f"{ordinary['atypical_width_latency_per_gflop']:.6f}. Best observed ordinary shape: {_row_label(best_ordinary)}.",
        "2. C_out=2/4 is inefficient versus friendly C_out in this run: "
        + _yes_no(ordinary_small_bad)
        + f" (small C_out median {ordinary['c_out_small_latency_per_gflop']:.6f}, friendly C_out median {ordinary['c_out_friendly_latency_per_gflop']:.6f}).",
        "3. Atypical C values 6/10/12/14/18/20/22 are inefficient versus friendly values in this run: "
        + _yes_no(ordinary_atypical_bad)
        + f" (ordinary atypical median {ordinary['atypical_width_latency_per_gflop']:.6f}).",
        "4. Grouped per-group width 2/4: "
        + ("slower than friendly widths in this run." if small_bad else "not slower than friendly widths by median latency/GFLOP in this run.")
        + f" Small median {grouped['small_width_latency_per_gflop']:.6f}.",
        "5. Grouped per-group width 8/16/32: median latency/GFLOP is "
        + f"{grouped['friendly_width_latency_per_gflop']:.6f}. Best observed grouped shape: {_row_label(best_grouped)}.",
        "6. Groups alignment alone is sufficient: "
        + _yes_no(grouped_alignment_alone)
        + f" (aligned-groups median {grouped['aligned_groups_latency_per_gflop']:.6f}); per-group width should still be audited directly.",
        "7. A1 C_out%8 explanation: use `real_model_shape_benchmark.csv` and `speed_relative_to_nearest_friendly_shape`; values below 1.0 mean the produced shape is slower than its nearest friendly counterpart.",
        "8. B/C slowdown explanation: supported when grouped rows show small or atypical per-group widths with worse latency/GFLOP. In this run small_bad="
        + _yes_no(small_bad)
        + ", atypical_bad="
        + _yes_no(atypical_bad)
        + ".",
        "9. Next alignment rule recommendation: prefer ordinary C_in/C_out round_to 8 or 16, A total C_out round_to 8+, B per-group kept width >=8 when legal, and C groups_after choices that keep per-group widths >=8/16. Avoid per-group width <8 unless AP requires it.",
        "",
        "## File Notes",
        "",
        f"- ordinary rows: {len(ordinary_rows)}",
        f"- grouped rows: {len(grouped_rows)}",
        f"- real model rows: {len(real_rows)}",
        "- `speed_relative_to_nearest_friendly_shape`: throughput ratio to the nearest friendly shape within the same op/H/W/kernel/groups/cuDNN bucket. Values below 1.0 are slower than friendly.",
        "",
        "This is a PyTorch/cuDNN microbenchmark, not TensorRT evidence.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _real_model_shapes(v101_output_dir: Path) -> list[tuple[str, int, int, int, int, int, int]]:
    shapes: set[tuple[str, int, int, int, int, int, int]] = set()
    for report in v101_output_dir.glob("**/*shape_alignment_report.json"):
        try:
            data = json.loads(report.read_text(encoding="utf-8"))
        except Exception:
            continue
        for row in data.get("conv2d", []) or []:
            c_in = int(row.get("C_in", 0) or 0)
            c_out = int(row.get("C_out", 0) or 0)
            groups = int(row.get("groups", 1) or 1)
            if c_in <= 0 or c_out <= 0 or groups <= 0:
                continue
            op_type = "grouped_conv2d" if groups > 1 else "conv2d"
            shapes.add((op_type, 50, 176, 3, c_in, c_out, groups))
    return sorted(shapes)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark conv shape efficiency")
    parser.add_argument("--device", default="cuda:5")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeat", type=int, default=200)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--output-dir", default="outputs/latency_lut/conv_shape_efficiency_v102")
    parser.add_argument("--v101-output-dir", default="outputs/latency_lut/global_budgeted_alignment_eval_v101")
    parser.add_argument("--max-ordinary-shapes", type=int, default=0)
    parser.add_argument("--max-grouped-shapes", type=int, default=0)
    parser.add_argument("--max-real-model-shapes", type=int, default=64)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ordinary_shapes = _ordinary_shapes()
    grouped_shapes = _grouped_shapes()
    if args.max_ordinary_shapes > 0:
        ordinary_shapes = ordinary_shapes[: args.max_ordinary_shapes]
    if args.max_grouped_shapes > 0:
        grouped_shapes = grouped_shapes[: args.max_grouped_shapes]
    ordinary_rows: list[dict[str, Any]] = []
    grouped_rows: list[dict[str, Any]] = []
    for cudnn_benchmark in (True, False):
        for h, w, kernel, c_in, c_out in ordinary_shapes:
            ordinary_rows.append(
                _benchmark_shape(
                    op_type="conv2d",
                    batch=args.batch,
                    h=h,
                    w=w,
                    kernel=kernel,
                    c_in=c_in,
                    c_out=c_out,
                    groups=1,
                    device=device,
                    warmup=args.warmup,
                    repeat=args.repeat,
                    runs=args.runs,
                    cudnn_benchmark=cudnn_benchmark,
                )
            )
        for h, w, kernel, c_in, c_out, groups in grouped_shapes:
            grouped_rows.append(
                _benchmark_shape(
                    op_type="grouped_conv2d" if groups > 1 else "conv2d_groups1_control",
                    batch=args.batch,
                    h=h,
                    w=w,
                    kernel=kernel,
                    c_in=c_in,
                    c_out=c_out,
                    groups=groups,
                    device=device,
                    warmup=args.warmup,
                    repeat=args.repeat,
                    runs=args.runs,
                    cudnn_benchmark=cudnn_benchmark,
                )
            )
    _annotate_relative_speed_to_friendly(ordinary_rows)
    _annotate_relative_speed_to_friendly(grouped_rows)
    _write_csv(out / "ordinary_conv_shape_benchmark.csv", ordinary_rows)
    _write_csv(out / "grouped_conv_shape_benchmark.csv", grouped_rows)
    real_shapes = _real_model_shapes(Path(args.v101_output_dir))
    if args.max_real_model_shapes > 0:
        real_shapes = real_shapes[: args.max_real_model_shapes]
    real_rows: list[dict[str, Any]] = []
    for cudnn_benchmark in (True, False):
        for op_type, h, w, kernel, c_in, c_out, groups in real_shapes:
            real_rows.append(
                _benchmark_shape(
                    op_type=f"real_model_{op_type}",
                    batch=args.batch,
                    h=h,
                    w=w,
                    kernel=kernel,
                    c_in=c_in,
                    c_out=c_out,
                    groups=groups,
                    device=device,
                    warmup=args.warmup,
                    repeat=args.repeat,
                    runs=args.runs,
                    cudnn_benchmark=cudnn_benchmark,
                )
            )
    _annotate_relative_speed_to_friendly(real_rows)
    _write_csv(out / "real_model_shape_benchmark.csv", real_rows)
    heatmap_rows = [
        {
            "op_type": row["op_type"],
            "groups": row["groups"],
            "in_per_group": row["in_per_group"],
            "out_per_group": row["out_per_group"],
            "kernel": row["kernel"],
            "cudnn_benchmark": row["cudnn_benchmark"],
            "latency_per_GFLOP": row["latency_per_GFLOP"],
        }
        for row in grouped_rows
    ]
    _write_csv(out / "shape_efficiency_heatmap_data.csv", heatmap_rows)
    _write_summary(out / "shape_efficiency_summary.md", ordinary_rows, grouped_rows, real_rows=real_rows)
    print({"output_dir": str(out), "ordinary_shapes": len(ordinary_rows), "grouped_shapes": len(grouped_rows)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
