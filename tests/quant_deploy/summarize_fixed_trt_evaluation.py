from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from quant_deploy_utils import ensure_quant_deploy_run_dirs, read_json, save_json, write_summary_files


INVALID_REASON = "pairwise_t_matrix dtype binding mismatch"


def _ap_from_baseline(baseline: dict[str, Any], key: str) -> float | None:
    value = baseline.get(key)
    return float(value) if value is not None else None


def _ap_from_eval(eval_report: dict[str, Any], key: str) -> float | None:
    aliases = {"AP@0.30": "ap_0_3", "AP@0.50": "ap_0_5", "AP@0.70": "ap_0_7"}
    value = eval_report.get(aliases[key])
    return float(value) if value is not None else None


def _mean_ap(values: list[float | None]) -> float | None:
    valid = [float(v) for v in values if v is not None]
    return round(sum(valid) / len(valid), 4) if valid else None


def _speedup(numerator_ms: float | None, denominator_ms: float | None) -> float | None:
    if numerator_ms is None or denominator_ms in (None, 0):
        return None
    return float(numerator_ms) / float(denominator_ms)


def _round_optional(value: float | None, digits: int = 6) -> float | None:
    return round(float(value), digits) if value is not None else None


def _engine_summary(
    label: str,
    baseline: dict[str, Any],
    eval_report: dict[str, Any],
    bench_report: dict[str, Any],
    *,
    pytorch_forward_p50_ms: float | None,
    trt_fp32_forward_p50_ms: float | None,
) -> dict[str, Any]:
    ap_keys = ["AP@0.30", "AP@0.50", "AP@0.70"]
    ap = {key: _ap_from_eval(eval_report, key) for key in ap_keys}
    baseline_ap = {key: _ap_from_baseline(baseline, key) for key in ap_keys}
    ap_drop = {
        key: _round_optional((baseline_ap[key] or 0.0) - (ap[key] or 0.0), 4)
        if baseline_ap[key] is not None and ap[key] is not None
        else None
        for key in ap_keys
    }
    forward_p50 = bench_report.get("forward_p50_ms", eval_report.get("forward_p50_ms"))
    forward_p90 = bench_report.get("forward_p90_ms", eval_report.get("forward_p90_ms"))
    forward_p95 = bench_report.get("forward_p95_ms", eval_report.get("forward_p95_ms"))
    return {
        "engine": label,
        "success": bool(eval_report.get("success", False)),
        "actual_frames": eval_report.get("actual_frames"),
        "AP@0.30": ap["AP@0.30"],
        "AP@0.50": ap["AP@0.50"],
        "AP@0.70": ap["AP@0.70"],
        "map": _mean_ap([ap[key] for key in ap_keys]),
        "forward_p50_ms": forward_p50,
        "forward_p90_ms": forward_p90,
        "forward_p95_ms": forward_p95,
        "fps": bench_report.get("fps") or (1000.0 / float(forward_p50) if forward_p50 else None),
        "speedup_vs_pytorch": _speedup(pytorch_forward_p50_ms, forward_p50),
        "speedup_vs_trt_fp32": _speedup(trt_fp32_forward_p50_ms, forward_p50),
        "ap_drop_vs_pytorch": ap_drop,
        "mean_ap_drop_vs_pytorch": _mean_ap([ap_drop[key] for key in ap_keys]),
        "eval_report": eval_report,
        "benchmark_report": bench_report,
    }


def _baseline_summary(baseline: dict[str, Any]) -> dict[str, Any]:
    ap_keys = ["AP@0.30", "AP@0.50", "AP@0.70"]
    ap = {key: _ap_from_baseline(baseline, key) for key in ap_keys}
    forward_p50 = baseline.get("forward_time_p50_ms")
    forward_p90 = baseline.get("forward_time_p90_ms")
    forward_p95 = baseline.get("forward_time_p95_ms")
    return {
        "engine": "PyTorch",
        "success": True,
        "actual_frames": baseline.get("actual_frames"),
        "AP@0.30": ap["AP@0.30"],
        "AP@0.50": ap["AP@0.50"],
        "AP@0.70": ap["AP@0.70"],
        "map": _mean_ap([ap[key] for key in ap_keys]),
        "forward_p50_ms": forward_p50,
        "forward_p90_ms": forward_p90,
        "forward_p95_ms": forward_p95,
        "fps": 1000.0 / float(forward_p50) if forward_p50 else None,
        "speedup_vs_pytorch": 1.0,
        "speedup_vs_trt_fp32": None,
        "ap_drop_vs_pytorch": {key: 0.0 for key in ap_keys},
        "mean_ap_drop_vs_pytorch": 0.0,
        "eval_report": baseline,
    }


def build_fixed_trt_summary_fields(
    baseline: dict[str, Any],
    fp32_eval: dict[str, Any],
    fp16_eval: dict[str, Any],
    fp32_bench: dict[str, Any],
    fp16_bench: dict[str, Any],
    output_errors: dict[str, Any],
) -> dict[str, Any]:
    pytorch_forward = baseline.get("forward_time_p50_ms")
    fp32_forward = fp32_bench.get("forward_p50_ms", fp32_eval.get("forward_p50_ms"))
    baseline_item = _baseline_summary(baseline)
    fp32_item = _engine_summary(
        "TensorRT FP32",
        baseline,
        fp32_eval,
        fp32_bench,
        pytorch_forward_p50_ms=pytorch_forward,
        trt_fp32_forward_p50_ms=fp32_forward,
    )
    fp16_item = _engine_summary(
        "TensorRT FP16",
        baseline,
        fp16_eval,
        fp16_bench,
        pytorch_forward_p50_ms=pytorch_forward,
        trt_fp32_forward_p50_ms=fp32_forward,
    )
    return {
        "previous_trt_ap_zero_invalidated": True,
        "invalid_reason": INVALID_REASON,
        "fixed_trt_eval_summary": {
            "pytorch": baseline_item,
            "fp32": fp32_item,
            "fp16": fp16_item,
        },
        "trt_fp32_output_error_after_dtype_fix": output_errors.get("trt_fp32_vs_pytorch_error"),
        "trt_fp16_output_error_after_dtype_fix": output_errors.get("trt_fp16_vs_pytorch_error"),
    }


def mark_old_ap_reports_invalid(dirs: dict[str, Path]) -> None:
    for precision in ("fp32", "fp16", "int8"):
        path = dirs[f"evaluation_{precision}"] / f"eval_metrics_{precision}.json"
        old = read_json(path, default=None)
        if not old:
            continue
        if old.get("ap_0_3") == 0.0 and old.get("ap_0_5") == 0.0 and old.get("ap_0_7") == 0.0:
            invalid_path = path.with_name(f"eval_metrics_{precision}_invalid_dtype_binding.json")
            old["previous_trt_ap_zero_invalidated"] = True
            old["invalid_reason"] = INVALID_REASON
            save_json(old, invalid_path)


def write_fixed_reports(args: argparse.Namespace) -> dict[str, Any]:
    dirs = ensure_quant_deploy_run_dirs(args.output_root)
    mark_old_ap_reports_invalid(dirs)
    baseline = read_json(dirs["evaluation"] / "pytorch_baseline_50" / "baseline_eval_summary.json", default={}) or {}
    fp32_eval = read_json(dirs["evaluation_fp32"] / "eval_metrics_fp32.json", default={}) or {}
    fp16_eval = read_json(dirs["evaluation_fp16"] / "eval_metrics_fp16.json", default={}) or {}
    fp32_bench = read_json(dirs["benchmark_fp32"] / "benchmark_fp32.json", default={}) or {}
    fp16_bench = read_json(dirs["benchmark_fp16"] / "benchmark_fp16.json", default={}) or {}
    summary = read_json(dirs["summary"] / "summary_all.json", default={}) or {}
    fields = build_fixed_trt_summary_fields(baseline, fp32_eval, fp16_eval, fp32_bench, fp16_bench, summary)
    fp32_report = fields["fixed_trt_eval_summary"]["fp32"]
    fp16_report = fields["fixed_trt_eval_summary"]["fp16"]
    save_json(fp32_report, dirs["evaluation"] / "trt_fp32_ap_report.json")
    save_json(fp16_report, dirs["evaluation"] / "trt_fp16_ap_report.json")
    summary.update(fields)
    precisions = dict(summary.get("precisions") or {})
    fixed_eval_by_precision = {"fp32": fp32_report, "fp16": fp16_report}
    for precision, bench in (("fp32", fp32_bench), ("fp16", fp16_bench)):
        fixed_eval = fixed_eval_by_precision[precision]
        existing = dict(precisions.get(precision) or {})
        existing.update(
            {
                "implemented": True,
                "build_success": True,
                "benchmark_success": bool(bench.get("success")),
                "engine_path": bench.get("engine_path"),
                "engine_size_MB": bench.get("engine_size_MB"),
                "actual_frames": fixed_eval.get("actual_frames"),
                "ap_0_3": fixed_eval.get("AP@0.30"),
                "ap_0_5": fixed_eval.get("AP@0.50"),
                "ap_0_7": fixed_eval.get("AP@0.70"),
                "map": fixed_eval.get("map"),
                "ap_drop_vs_pytorch": fixed_eval.get("ap_drop_vs_pytorch"),
                "mean_ap_drop_vs_pytorch": fixed_eval.get("mean_ap_drop_vs_pytorch"),
                "speedup_vs_pytorch": fixed_eval.get("speedup_vs_pytorch"),
                "speedup_vs_trt_fp32": fixed_eval.get("speedup_vs_trt_fp32"),
                "forward_p50_ms": bench.get("forward_p50_ms"),
                "forward_p90_ms": bench.get("forward_p90_ms"),
                "forward_p95_ms": bench.get("forward_p95_ms"),
                "fps": bench.get("fps"),
                "error": bench.get("error"),
                "reevaluated_after_dtype_fix": True,
                "previous_trt_ap_zero_invalidated": True,
                "invalid_reason": INVALID_REASON,
            }
        )
        precisions[precision] = existing
    fp32_p50 = precisions.get("fp32", {}).get("forward_p50_ms")
    for precision in ("fp32", "fp16"):
        p50 = precisions.get(precision, {}).get("forward_p50_ms")
        precisions[precision]["speedup_vs_fp32"] = (float(fp32_p50) / float(p50)) if fp32_p50 and p50 else None
    if "int8" in precisions:
        precisions["int8"]["reevaluated_after_dtype_fix"] = False
        precisions["int8"]["ap_invalidated"] = True
        precisions["int8"]["invalid_reason"] = INVALID_REASON
        precisions["int8"]["error"] = "not reevaluated in this run; previous AP=0 invalidated by dtype binding bug"
    summary.update(
        {
            "precisions": precisions,
            "plugin_needed": False,
            "need_bevwarp_plugin": False,
            "need_pointpillar_scatter_plugin": False,
            "need_bevpool_plugin": False,
            "need_inverse_plugin": False,
            "trt_dtype_binding_mismatch_found": True,
            "trt_dtype_binding_mismatched_inputs": ["pairwise_t_matrix"],
        }
    )
    write_summary_files(summary, dirs)
    return fields


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize fixed TensorRT FP32/FP16 AP and latency after dtype binding fix.")
    parser.add_argument("--output_root", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    fields = write_fixed_reports(parse_args(argv))
    print(json.dumps(fields["fixed_trt_eval_summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
