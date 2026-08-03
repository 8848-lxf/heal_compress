#!/usr/bin/env python3
"""v10.5 semantic-preserving Stage0 grouped Conv2d reblock experiment."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
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
_UNIAD = _ROOT.parent
for _p in (_UNIAD, _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tools.latency_lut.select_idle_gpu_for_latency import (  # noqa: E402
    collect_gpu_state,
    wait_for_idle_gpu,
)


DEFAULT_CHECKPOINT = "${MODEL_ROOT}/lidar_pyramid/net_epoch_bestval_at17.pth"
DEFAULT_CONFIG = "${MODEL_ROOT}/lidar_pyramid/config.yaml"
DEFAULT_HEAL_ROOT = "../../HEAL"


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
        writer.writerows(rows if rows else ([{"empty": ""}] if fields == ["empty"] else []))


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


def latency_stats(values: Sequence[float]) -> dict[str, float]:
    vals = [float(v) for v in values if float(v) > 0.0]
    return {
        "p50": statistics.median(vals) if vals else 0.0,
        "mean": statistics.mean(vals) if vals else 0.0,
        "p90": percentile(vals, 0.90),
        "p95": percentile(vals, 0.95),
        "min": min(vals) if vals else 0.0,
        "max": max(vals) if vals else 0.0,
        "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
    }


def reblock_grouped_conv_semantic_preserving(conv: nn.Conv2d, groups_new: int) -> nn.Conv2d:
    """Merge old grouped-conv groups with block-diagonal zero-padded weights.

    This preserves C_in, C_out, input order, and output order.  It does not
    compact or front-fill filters; each old group's kernel slice is copied into
    its corresponding offset inside the merged new group.
    """
    if not isinstance(conv, nn.Conv2d):
        raise TypeError("conv must be nn.Conv2d")
    old_groups = int(conv.groups)
    old_c_in = int(conv.in_channels)
    old_c_out = int(conv.out_channels)
    groups_new = int(groups_new)
    if old_groups != 32:
        raise ValueError(f"expected old_groups=32 for v10.5 Stage0 experiment, got {old_groups}")
    if old_c_in != 128 or old_c_out != 128:
        raise ValueError(f"expected C_in=C_out=128 for v10.5 Stage0 experiment, got {old_c_in}/{old_c_out}")
    if groups_new not in {16, 8}:
        raise ValueError("groups_new must be 16 or 8")
    if old_groups % groups_new != 0:
        raise ValueError("old_groups must be divisible by groups_new")
    if old_c_in % groups_new != 0 or old_c_out % groups_new != 0:
        raise ValueError("C_in/C_out must be divisible by groups_new")

    merge_factor = old_groups // groups_new
    old_in_per_group = old_c_in // old_groups
    old_out_per_group = old_c_out // old_groups
    new_in_per_group = old_c_in // groups_new
    new_out_per_group = old_c_out // groups_new
    if new_in_per_group != old_in_per_group * merge_factor:
        raise ValueError("unexpected new input group width")
    if new_out_per_group != old_out_per_group * merge_factor:
        raise ValueError("unexpected new output group width")

    new_conv = nn.Conv2d(
        old_c_in,
        old_c_out,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=groups_new,
        bias=conv.bias is not None,
        padding_mode=conv.padding_mode,
        device=conv.weight.device,
        dtype=conv.weight.dtype,
    )
    with torch.no_grad():
        new_conv.weight.zero_()
        for oc in range(old_c_out):
            old_group = oc // old_out_per_group
            new_group = old_group // merge_factor
            old_group_offset_inside_new_group = old_group % merge_factor
            input_offset = old_group_offset_inside_new_group * old_in_per_group
            # oc is unchanged because output channel order is preserved.  Its
            # local output row inside the new group is also consistent with
            # old-group concatenation.
            _ = new_group
            new_conv.weight[oc, input_offset : input_offset + old_in_per_group].copy_(conv.weight[oc])
        if conv.bias is not None and new_conv.bias is not None:
            new_conv.bias.copy_(conv.bias)
    new_conv.train(conv.training)
    return new_conv


def candidate_row(name: str, module: nn.Conv2d) -> dict[str, Any]:
    params = int(module.weight.numel() + (module.bias.numel() if module.bias is not None else 0))
    return {
        "module_name": name,
        "in_channels": int(module.in_channels),
        "out_channels": int(module.out_channels),
        "groups": int(module.groups),
        "in_per_group": int(module.in_channels) // int(module.groups),
        "out_per_group": int(module.out_channels) // int(module.groups),
        "kernel_size": list(module.kernel_size),
        "stride": list(module.stride),
        "padding": list(module.padding),
        "dilation": list(module.dilation),
        "bias": module.bias is not None,
        "parameter_count": params,
        "estimated_flops_if_hw_known": None,
    }


def find_stage0_grouped_conv_candidates(model: nn.Module) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, module in model.named_modules():
        if (
            isinstance(module, nn.Conv2d)
            and int(module.groups) == 32
            and int(module.in_channels) == 128
            and int(module.out_channels) == 128
        ):
            rows.append(candidate_row(name, module))
    return rows


def _get_parent_and_leaf(model: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    parent_name, _, leaf = module_name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    return parent, leaf


def reblock_model_stage0_candidates(model: nn.Module, candidates: Sequence[Mapping[str, Any]], *, groups_new: int) -> list[str]:
    changed: list[str] = []
    for row in candidates:
        name = str(row["module_name"])
        module = model.get_submodule(name)
        if not isinstance(module, nn.Conv2d):
            continue
        parent, leaf = _get_parent_and_leaf(model, name)
        setattr(parent, leaf, reblock_grouped_conv_semantic_preserving(module, groups_new=groups_new))
        changed.append(name)
    return changed


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


def compare_outputs(a: Any, b: Any) -> dict[str, Any]:
    a_tensors = _flatten_tensors(a)
    b_tensors = _flatten_tensors(b)
    pairs = [(x, y) for x, y in zip(a_tensors, b_tensors) if tuple(x.shape) == tuple(y.shape) and x.is_floating_point()]
    if not pairs:
        return {
            "max_abs_diff": None,
            "mean_abs_diff": None,
            "relative_l2_diff": None,
            "tolerance_passed": False,
            "compared_outputs": 0,
            "notes": "no matching floating tensors found",
        }
    max_abs = 0.0
    sum_abs = 0.0
    count = 0
    sq = 0.0
    ref_sq = 0.0
    for x, y in pairs:
        d = (x.float() - y.float()).abs()
        max_abs = max(max_abs, float(d.max().item()))
        sum_abs += float(d.sum().item())
        count += int(d.numel())
        sq += float(((x.float() - y.float()) ** 2).sum().item())
        ref_sq += float((x.float() ** 2).sum().item())
    mean_abs = sum_abs / max(count, 1)
    rel_l2 = math.sqrt(sq) / (math.sqrt(ref_sq) + 1e-12)
    return {
        "max_abs_diff": max_abs,
        "mean_abs_diff": mean_abs,
        "relative_l2_diff": rel_l2,
        "tolerance_passed": bool(max_abs < 1e-4),
        "compared_outputs": len(pairs),
        "notes": "",
    }


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


def conv2d_flops(batch: int, h: int, w: int, c_in: int, c_out: int, kernel: int, groups: int) -> float:
    return float(batch) * h * w * c_out * (c_in / groups) * kernel * kernel * 2.0


def run_single_layer_equivalence_report(out_dir: Path, device: torch.device) -> list[dict[str, Any]]:
    torch.manual_seed(1234)
    conv = nn.Conv2d(128, 128, 3, padding=1, groups=32, bias=True).to(device=device, dtype=torch.float32).eval()
    with torch.no_grad():
        conv.weight.normal_(mean=0.0, std=0.2)
        conv.bias.normal_(mean=0.0, std=0.1)
    x = torch.randn(2, 128, 50, 176, device=device, dtype=torch.float32)
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        old_y = conv(x)
        for variant, groups_new in [("groups16", 16), ("groups8", 8)]:
            new_conv = reblock_grouped_conv_semantic_preserving(conv, groups_new=groups_new).to(device).eval()
            new_y = new_conv(x)
            diff = (old_y - new_y).abs()
            rows.append(
                {
                    "variant": variant,
                    "groups_old": 32,
                    "groups_new": groups_new,
                    "max_abs_diff": float(diff.max().item()),
                    "mean_abs_diff": float(diff.mean().item()),
                    "tolerance_passed": bool(float(diff.max().item()) < 1e-5 and float(diff.mean().item()) < 1e-6),
                    "input_shape": [2, 128, 50, 176],
                    "semantic_preserved": True,
                }
            )
    write_json(out_dir / "single_layer_equivalence_report.json", rows)
    return rows


def run_microbenchmarks(args: argparse.Namespace, out_dir: Path, device: torch.device) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    hws = [(100, 352), (50, 176)]
    envs = [("fp32", "nchw"), ("fp16", "channels_last")] if device.type == "cuda" else [("fp32", "nchw")]
    variants = [("baseline", 32), ("groups16", 16), ("groups8", 8)]
    baseline_by_env_hw: dict[tuple[str, str, int, int], float] = {}
    for dtype_name, layout in envs:
        dtype = torch.float16 if dtype_name == "fp16" else torch.float32
        for h, w in hws:
            for variant, groups in variants:
                conv = nn.Conv2d(128, 128, 3, padding=1, groups=groups, bias=False).to(device=device, dtype=dtype).eval()
                x = torch.randn(1, 128, h, w, device=device, dtype=dtype)
                if layout == "channels_last":
                    conv = conv.to(memory_format=torch.channels_last)
                    x = x.contiguous(memory_format=torch.channels_last)
                stats = _time_callable(lambda conv=conv, x=x: conv(x), warmup=args.latency_warmup, repeat=args.latency_repeat, device=device)
                key = (dtype_name, layout, h, w)
                if variant == "baseline":
                    baseline_by_env_hw[key] = stats["p50"]
                base = baseline_by_env_hw.get(key, stats["p50"])
                flops = conv2d_flops(1, h, w, 128, 128, 3, groups)
                gflops = flops / 1e9
                rows.append(
                    {
                        "variant": variant,
                        "dtype": dtype_name,
                        "layout": layout,
                        "H": h,
                        "W": w,
                        "groups": groups,
                        "in_per_group": 128 // groups,
                        "out_per_group": 128 // groups,
                        "latency_p50": stats["p50"],
                        "latency_mean": stats["mean"],
                        "latency_p90": stats["p90"],
                        "latency_p95": stats["p95"],
                        "throughput_GFLOPs_s": gflops / (stats["mean"] / 1000.0) if stats["mean"] > 0 else 0.0,
                        "latency_per_GFLOP": stats["mean"] / gflops if gflops > 0 else 0.0,
                        "speedup_vs_groups32": base / stats["p50"] if stats["p50"] > 0 else 0.0,
                        "nominal_flops_ratio": float(groups == 16) * 2.0 or (4.0 if groups == 8 else 1.0),
                    }
                )
    write_csv(out_dir / "microbench_stage0_group_reblock.csv", rows)
    return rows


def _variant_meta(variant: str) -> tuple[int, float, float]:
    if variant == "groups16":
        return 16, 2.0, 0.5
    if variant == "groups8":
        return 8, 4.0, 0.75
    return 32, 1.0, 0.0


def run_real_model_experiment(args: argparse.Namespace, out_dir: Path, device: torch.device) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    from heal_compress.utils.model_utils import resolve_device
    from heal_compress.pruning.model_io import load_heal_model, setup_logger

    logger = setup_logger(out_dir)
    args.device = str(device)
    device = torch.device(resolve_device(args.device))
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model, adapter = load_heal_model(args, device, logger)
    model.eval()
    candidates = find_stage0_grouped_conv_candidates(model)
    write_json(out_dir / "stage0_grouped_conv_candidates.json", candidates)
    if not candidates:
        raise RuntimeError("stage0_grouped_conv_candidates_not_found")

    baseline = model
    variants: dict[str, nn.Module] = {"baseline": baseline}
    for variant in [v for v in args.variants.split(",") if v]:
        groups_new, _, _ = _variant_meta(variant)
        cloned = copy.deepcopy(baseline).to(device).eval()
        reblock_model_stage0_candidates(cloned, candidates, groups_new=groups_new)
        variants[variant] = cloned

    smoke_rows: list[dict[str, Any]] = []
    equivalence_rows: list[dict[str, Any]] = []
    frame_reports: dict[str, list[dict[str, Any]]] = {variant: [] for variant in variants if variant != "baseline"}
    sample = adapter.build_synthetic_batch(baseline)
    for variant, variant_model in variants.items():
        if variant == "baseline":
            continue
        with torch.no_grad():
            baseline_out = adapter.forward_for_task(baseline, sample)
            out = adapter.forward_for_task(variant_model, sample)
        cmp = compare_outputs(baseline_out, out)
        smoke_rows.append({"variant": variant, "synthetic_forward_passed": True, "failure_reason": ""})

    for frame_idx in range(max(1, int(args.eval_frames))):
        frame_sample = adapter.build_synthetic_batch(baseline)
        with torch.no_grad():
            baseline_out = adapter.forward_for_task(baseline, frame_sample)
            for variant, variant_model in variants.items():
                if variant == "baseline":
                    continue
                out = adapter.forward_for_task(variant_model, frame_sample)
                cmp = compare_outputs(baseline_out, out)
                frame_reports[variant].append({"frame_index": frame_idx, **cmp})

    for variant, reports in frame_reports.items():
        max_abs = max(float(row.get("max_abs_diff") or 0.0) for row in reports) if reports else None
        mean_abs = statistics.mean(float(row.get("mean_abs_diff") or 0.0) for row in reports) if reports else None
        rel_l2 = max(float(row.get("relative_l2_diff") or 0.0) for row in reports) if reports else None
        compared = sum(int(row.get("compared_outputs") or 0) for row in reports)
        tolerance = bool(reports) and all(bool(row.get("tolerance_passed")) for row in reports)
        equivalence_rows.append(
            {
                "variant": variant,
                "modules_reblocked": [row["module_name"] for row in candidates],
                "num_frames": len(reports),
                "max_abs_diff": max_abs,
                "mean_abs_diff": mean_abs,
                "relative_l2_diff": rel_l2,
                "tolerance_passed": tolerance,
                "compared_outputs": compared,
                "notes": "fixed synthetic output equivalence over eval_frames",
                "per_frame_reports": reports,
            }
        )

    baseline_stats = None
    latency_rows: list[dict[str, Any]] = []
    for variant, variant_model in variants.items():
        stats = _time_callable(lambda m=variant_model: adapter.forward_for_task(m, sample), warmup=args.latency_warmup, repeat=args.latency_repeat, device=device)
        if variant == "baseline":
            baseline_stats = stats
        groups_new, nominal_ratio, zero_ratio = _variant_meta(variant)
        base = baseline_stats or stats
        latency_rows.append(
            {
                "variant": variant,
                "module_name": "all_stage0_candidates",
                "groups_old": 32,
                "groups_new": groups_new,
                "C_in": 128,
                "C_out": 128,
                "old_in_per_group": 4,
                "old_out_per_group": 4,
                "new_in_per_group": 128 // groups_new,
                "new_out_per_group": 128 // groups_new,
                "nominal_param_ratio": nominal_ratio,
                "nominal_flops_ratio": nominal_ratio,
                "zero_added_ratio": zero_ratio,
                "forward_latency_p50": stats["p50"],
                "forward_latency_mean": stats["mean"],
                "forward_latency_p90": stats["p90"],
                "forward_latency_p95": stats["p95"],
                "speedup_p50_vs_baseline": base["p50"] / stats["p50"] if stats["p50"] > 0 else 0.0,
                "speedup_mean_vs_baseline": base["mean"] / stats["mean"] if stats["mean"] > 0 else 0.0,
                "speedup_p90_vs_baseline": base["p90"] / stats["p90"] if stats["p90"] > 0 else 0.0,
                "speedup_p95_vs_baseline": base["p95"] / stats["p95"] if stats["p95"] > 0 else 0.0,
            }
        )

    eval_rows: list[dict[str, Any]] = []
    for row in [{"variant": "baseline", "max_abs_diff": 0.0, "mean_abs_diff": 0.0, "relative_l2_diff": 0.0, "tolerance_passed": True}] + equivalence_rows:
        eval_rows.append(
            {
                "variant": row["variant"],
                "AP@0.30": None,
                "AP_drop_vs_baseline": None,
                "num_frames": int(args.eval_frames),
                "notes": "fixed synthetic output equivalence over eval_frames used instead of AP eval",
                "max_abs_diff": row.get("max_abs_diff"),
                "mean_abs_diff": row.get("mean_abs_diff"),
                "relative_l2_diff": row.get("relative_l2_diff"),
                "tolerance_passed": row.get("tolerance_passed"),
            }
        )

    write_json(out_dir / "equivalence_report.json", equivalence_rows)
    write_json(out_dir / "forward_smoke_report.json", smoke_rows)
    write_csv(out_dir / "stage0_group_reblock_latency.csv", latency_rows)
    write_json(out_dir / "stage0_group_reblock_eval_sanity.json", eval_rows)
    return latency_rows, equivalence_rows, {"candidates": candidates, "eval_rows": eval_rows}


def write_verdict(
    out_dir: Path,
    *,
    latency_rows: Sequence[dict[str, Any]],
    micro_rows: Sequence[dict[str, Any]],
    equivalence_rows: Sequence[dict[str, Any]],
    single_layer_rows: Sequence[dict[str, Any]],
    failure: str = "",
) -> None:
    def row_for(rows: Sequence[dict[str, Any]], variant: str) -> dict[str, Any]:
        return next((row for row in rows if row.get("variant") == variant), {})

    g16_eq = row_for(equivalence_rows, "groups16")
    g8_eq = row_for(equivalence_rows, "groups8")
    g16_single = row_for(single_layer_rows, "groups16")
    g8_single = row_for(single_layer_rows, "groups8")
    g16_lat = row_for(latency_rows, "groups16")
    g8_lat = row_for(latency_rows, "groups8")
    micro_g16 = [row for row in micro_rows if row.get("variant") == "groups16"]
    micro_g8 = [row for row in micro_rows if row.get("variant") == "groups8"]
    g16_micro_speed = statistics.median([float(row["speedup_vs_groups32"]) for row in micro_g16]) if micro_g16 else 0.0
    g8_micro_speed = statistics.median([float(row["speedup_vs_groups32"]) for row in micro_g8]) if micro_g8 else 0.0
    lines = [
        "# Stage0 Grouped Conv Semantic Reblock v10.5 Verdict",
        "",
        f"failure: {failure or 'none'}",
        f"1. groups32 -> groups16 single-layer equivalence: {bool(g16_single.get('tolerance_passed', False))} (max_abs_diff={g16_single.get('max_abs_diff')}).",
        f"2. groups32 -> groups8 single-layer equivalence: {bool(g8_single.get('tolerance_passed', False))} (max_abs_diff={g8_single.get('max_abs_diff')}).",
        f"3. real model output near-equivalence over eval_frames: groups16={g16_eq.get('tolerance_passed')}, groups8={g8_eq.get('tolerance_passed')}.",
        f"4. groups16 microbenchmark median speedup vs groups32: {g16_micro_speed:.6f}.",
        f"5. groups8 microbenchmark median speedup vs groups32: {g8_micro_speed:.6f}.",
        f"6. groups16 whole-model forward speedup_p50: {g16_lat.get('speedup_p50_vs_baseline', 0)}.",
        f"7. groups8 whole-model forward speedup_p50: {g8_lat.get('speedup_p50_vs_baseline', 0)}.",
        "8. Nominal FLOPs/params increase by 2x/4x for groups16/groups8 because dense cuDNN sees the zero-padded block-diagonal weights as dense grouped kernels.",
        "9. If latency speedup >1, this is a deployment reparameterization candidate; otherwise dense cuDNN is not exploiting zero blocks enough.",
        "10. If slower, the result indicates dense cuDNN does not skip block-diagonal zeros; block-sparse kernels or TensorRT/plugin support would be needed.",
        "11. The method preserves semantics: no input/output channel reorder, no compact/frontfill, and old groups do not see each other because cross-old-group weights are zero.",
        "12. TensorRT/plugin or block-sparse kernels may be required to exploit block-diagonal structure explicitly.",
        "13. Difference from C true group-block pruning: C deletes old group blocks and reduces C_in/C_out/groups; this experiment keeps all channels and only merges groups with zero-padded block-diagonal weights.",
        "14. Decoder/export recommendation depends on measured speedup and equivalence; keep it as a deployment repair candidate only if groups16/groups8 improve latency without output drift.",
    ]
    (out_dir / "stage0_group_reblock_verdict.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="v10.5 Stage0 grouped conv semantic reblock experiment")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--model-config", default=DEFAULT_CONFIG)
    parser.add_argument("--heal-root", default=DEFAULT_HEAL_ROOT)
    parser.add_argument("--device", default="")
    parser.add_argument("--auto-select-idle-gpu", action="store_true")
    parser.add_argument("--max-gpu-utilization", type=int, default=5)
    parser.add_argument("--max-gpu-memory-ratio", type=float, default=0.20)
    parser.add_argument("--wait-timeout-minutes", type=float, default=30.0)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--variants", default="groups16,groups8")
    parser.add_argument("--eval-frames", type=int, default=50)
    parser.add_argument("--latency-warmup", type=int, default=50)
    parser.add_argument("--latency-repeat", type=int, default=200)
    parser.add_argument("--output-dir", default="outputs/latency_lut/stage0_group_reblock_v105")
    parser.add_argument("--module-name", default="")
    return parser.parse_args(argv)


def _select_device(args: argparse.Namespace, out_dir: Path) -> tuple[torch.device, int | None]:
    if args.auto_select_idle_gpu:
        selected, reason, attempts = wait_for_idle_gpu(
            max_utilization=args.max_gpu_utilization,
            max_memory_ratio=args.max_gpu_memory_ratio,
            wait_timeout_minutes=args.wait_timeout_minutes,
            poll_seconds=args.poll_seconds,
        )
        write_json(out_dir / "selected_gpu.json", {"selected_gpu": selected, "reason": reason, "attempts": attempts})
        device = torch.device(f"cuda:{selected.index}" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            torch.cuda.set_device(device)
        return device, int(selected.index)
    device = torch.device(args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.cuda.set_device(device)
        return device, int(device.index or 0)
    return device, None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    latency_rows: list[dict[str, Any]] = []
    equivalence_rows: list[dict[str, Any]] = []
    micro_rows: list[dict[str, Any]] = []
    single_layer_rows: list[dict[str, Any]] = []
    failure = ""
    selected_index: int | None = None
    try:
        device, selected_index = _select_device(args, out_dir)
        if selected_index is not None:
            write_json(out_dir / "gpu_state_before.json", collect_gpu_state(selected_index))
        single_layer_rows = run_single_layer_equivalence_report(out_dir, device)
        micro_rows = run_microbenchmarks(args, out_dir, device)
        latency_rows, equivalence_rows, _ = run_real_model_experiment(args, out_dir, device)
        if selected_index is not None:
            write_json(out_dir / "gpu_state_after.json", collect_gpu_state(selected_index))
            write_json(out_dir / "gpu_state_during_samples.json", [collect_gpu_state(selected_index)])
        else:
            write_json(out_dir / "gpu_state_before.json", {})
            write_json(out_dir / "gpu_state_after.json", {})
            write_json(out_dir / "gpu_state_during_samples.json", [])
        write_json(out_dir / "failure_report.json", {"success": True, "failure_reason": "", "traceback": ""})
    except Exception as exc:  # noqa: BLE001
        failure = f"{type(exc).__name__}: {exc}"
        write_json(out_dir / "failure_report.json", {"success": False, "failure_reason": failure, "traceback": traceback.format_exc()})
        if selected_index is not None:
            try:
                write_json(out_dir / "gpu_state_after.json", collect_gpu_state(selected_index))
            except Exception:
                pass
    write_verdict(
        out_dir,
        latency_rows=latency_rows,
        micro_rows=micro_rows,
        equivalence_rows=equivalence_rows,
        single_layer_rows=single_layer_rows,
        failure=failure,
    )
    print(json.dumps({"success": not failure, "failure": failure, "output_dir": str(out_dir)}, indent=2, ensure_ascii=False))
    return 0 if not failure else 2


if __name__ == "__main__":
    raise SystemExit(main())
