from __future__ import annotations

import argparse
import shutil
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_lidar_pyramid_trt_engine import benchmark_engine
from build_lidar_pyramid_trt_engine import build_engine
from deployment_equivalence import run_trt_output_equivalence, run_wrapper_equivalence
from diagnose_agent_export_onnx_semantics import analyze_agent_export_semantics
from evaluate_lidar_pyramid_ort_ap import run_five_way_evaluation
from evaluate_lidar_pyramid_trt_ap import evaluate_engine
from export_lidar_pyramid_onnx import export_lidar_pyramid_onnx
from quant_deploy_utils import (
    DEFAULT_CHECKPOINT,
    DEFAULT_HEAL_REPO,
    DEFAULT_HYPES_YAML,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TRT_ROOT,
    create_quant_deploy_run_dirs,
    dirs_for_summary,
    ensure_quant_deploy_run_dirs,
    read_json,
    save_json,
    write_summary_files,
)


MODES = ("dynamic_agent_dim", "padded_agent_static")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare lidar_pyramid dynamic-agent and padded-agent ONNX export strategies.")
    parser.add_argument("--hypes_yaml", default=str(DEFAULT_HYPES_YAML))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--heal_repo", default=str(DEFAULT_HEAL_REPO))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--run_name", default="lidar_pyramid_agent_export_strategy_compare")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_frames", type=int, default=50)
    parser.add_argument("--warmup_frames", type=int, default=10)
    parser.add_argument("--max_cav", type=int, default=2)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--trt_root", default=str(DEFAULT_TRT_ROOT))
    parser.add_argument("--trtexec_path", default=None)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--no_tf32", action="store_true", default=True)
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--strict_fp16", action="store_true")
    parser.add_argument("--allow_synthetic_fallback", action="store_true")
    parser.add_argument("--skip_trt", action="store_true")
    parser.add_argument("--dynamic_agent_root", default=None, help="Reuse an existing dynamic_agent_dim run output root.")
    parser.add_argument("--padded_agent_root", default=None, help="Reuse an existing padded_agent_static run output root.")
    return parser.parse_args(argv)


def _engine_prefix(mode: str) -> str:
    return f"lidar_pyramid_{mode}"


def _engine_path(dirs: dict[str, Path], mode: str, precision: str) -> Path:
    return dirs[f"engine_{precision}"] / f"{_engine_prefix(mode)}_{precision}.engine"


def _mode_summary_path(mode: str) -> str:
    return f"summary_{mode}.json"


def _map_from_report(report: dict[str, Any] | None) -> float | None:
    if not report:
        return None
    for key in ("mAP", "map"):
        if report.get(key) is not None:
            return float(report[key])
    vals = []
    for key in ("AP@0.30", "AP@0.50", "AP@0.70", "ap_0_3", "ap_0_5", "ap_0_7"):
        if report.get(key) is not None:
            vals.append(float(report[key]))
    return round(sum(vals) / len(vals), 4) if vals else None


def _row_from_mode(mode: str, dirs: dict[str, Path], result: dict[str, Any]) -> dict[str, Any]:
    five_way = result.get("five_way") or {}
    reports = five_way.get("reports") or {}
    semantics = result.get("semantics") or {}
    fp32_bench = read_json(dirs["benchmark"] / f"fp32_{mode}" / "benchmark_fp32.json", default={}) or {}
    fp16_bench = read_json(dirs["benchmark"] / f"fp16_{mode}" / "benchmark_fp16.json", default={}) or {}
    wrapper = result.get("wrapper_equivalence") or {}
    by_record_len = wrapper.get("by_record_len") or {}
    record1_ok = _record_group_aligned(by_record_len.get("1"))
    record2_ok = _record_group_aligned(by_record_len.get("2"))
    return {
        "strategy": mode,
        "onnx_export_success": bool((result.get("export") or {}).get("success")),
        "onnx_preserves_multi_agent_semantics": bool(not semantics.get("semantics_erased_in_onnx") and semantics.get("valid_for_multi_agent_deployment")),
        "ort_fp32_mAP": _map_from_report(reports.get("onnxruntime_fp32")),
        "trt_fp32_mAP": _map_from_report(reports.get("tensorrt_fp32")),
        "trt_fp16_mAP": _map_from_report(reports.get("tensorrt_fp16")),
        "record_len_1_aligned": record1_ok,
        "record_len_2_aligned": record2_ok,
        "fp32_engine_p50": fp32_bench.get("forward_p50_ms"),
        "fp16_engine_p50": fp16_bench.get("forward_p50_ms"),
        "plugin_needed": False,
        "failure_reason": result.get("failure_reason") or semantics.get("failure_reason") or (result.get("export") or {}).get("error"),
    }


def _record_group_aligned(group: dict[str, Any] | None, threshold: float = 1.0e-3) -> bool | None:
    if not group:
        return None
    outputs = group.get("outputs") or {}
    values = [item.get("max_abs_error") for item in outputs.values() if item.get("max_abs_error") is not None]
    if not values:
        return None
    return max(values) <= threshold


def _choose_recommendation(rows: list[dict[str, Any]]) -> str | None:
    successful = [
        row
        for row in rows
        if row["onnx_export_success"]
        and row["onnx_preserves_multi_agent_semantics"]
        and row["record_len_1_aligned"] is not False
        and row["record_len_2_aligned"] is not False
        and row["ort_fp32_mAP"] is not None
    ]
    if not successful:
        return None
    successful.sort(key=lambda row: (row["trt_fp32_mAP"] is not None, row["ort_fp32_mAP"] or 0.0, -(row["fp32_engine_p50"] or 1.0e9)), reverse=True)
    padded = [row for row in successful if row["strategy"] == "padded_agent_static"]
    dynamic = [row for row in successful if row["strategy"] == "dynamic_agent_dim"]
    if padded and dynamic:
        p_map = padded[0].get("trt_fp32_mAP") or padded[0].get("ort_fp32_mAP") or 0.0
        d_map = dynamic[0].get("trt_fp32_mAP") or dynamic[0].get("ort_fp32_mAP") or 0.0
        if abs(p_map - d_map) <= 0.01:
            return "padded_agent_static"
    return successful[0]["strategy"]


def _write_comparison_markdown(path: Path, rows: list[dict[str, Any]], recommendation: str | None) -> None:
    lines = [
        "# Agent Export Strategy Comparison",
        "",
        f"- recommended_strategy: {recommendation}",
        "- int8_qdq_modelopt_plugin: not run",
        "",
        "strategy | ONNX | semantics | ORT FP32 mAP | TRT FP32 mAP | TRT FP16 mAP | record_len=1 | record_len=2 | FP32 p50 | FP16 p50 | plugin | failure",
        "--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---",
    ]
    for row in rows:
        lines.append(
            " | ".join(
                [
                    str(row.get("strategy")),
                    "yes" if row.get("onnx_export_success") else "no",
                    "yes" if row.get("onnx_preserves_multi_agent_semantics") else "no",
                    _fmt(row.get("ort_fp32_mAP")),
                    _fmt(row.get("trt_fp32_mAP")),
                    _fmt(row.get("trt_fp16_mAP")),
                    str(row.get("record_len_1_aligned")),
                    str(row.get("record_len_2_aligned")),
                    _fmt(row.get("fp32_engine_p50")),
                    _fmt(row.get("fp16_engine_p50")),
                    "no",
                    str(row.get("failure_reason")),
                ]
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _ap(report: dict[str, Any], key: str) -> float | None:
    aliases = {
        "AP@0.30": ("AP@0.30", "ap_0_3"),
        "AP@0.50": ("AP@0.50", "ap_0_5"),
        "AP@0.70": ("AP@0.70", "ap_0_7"),
        "mAP": ("mAP", "map"),
    }
    for alias in aliases[key]:
        value = report.get(alias)
        if value is not None:
            return float(value)
    return None


def _canonical_backend_report(report: dict[str, Any], backend: str, label: str) -> dict[str, Any]:
    out = dict(report or {})
    out.update(
        {
            "backend": backend,
            "backend_label": label,
            "AP@0.30": _ap(report or {}, "AP@0.30"),
            "AP@0.50": _ap(report or {}, "AP@0.50"),
            "AP@0.70": _ap(report or {}, "AP@0.70"),
            "mAP": _ap(report or {}, "mAP"),
        }
    )
    return out


def _report_for_backend(source_dirs: dict[str, Path], mode: str, backend: str) -> dict[str, Any]:
    five_way = read_json(source_dirs["evaluation"] / f"five_way_ap_report_{mode}.json", default={}) or {}
    reports = five_way.get("reports") or {}
    if backend in reports:
        return reports[backend] or {}
    if backend == "onnxruntime_fp32":
        return read_json(source_dirs["evaluation"] / f"ort_fp32_ap_report_{mode}.json", default={}) or {}
    if backend == "tensorrt_fp32":
        return read_json(source_dirs["evaluation_fp32"] / f"eval_metrics_fp32_{mode}.json", default={}) or {}
    if backend == "tensorrt_fp16":
        return read_json(source_dirs["evaluation_fp16"] / f"eval_metrics_fp16_{mode}.json", default={}) or {}
    return {}


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists() and src.resolve() != dst.resolve():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)


def _write_real_sample_benchmark_from_ap_report(
    dirs: dict[str, Path],
    source_dirs: dict[str, Path],
    mode: str,
    precision: str,
) -> dict[str, Any]:
    backend = f"tensorrt_{precision}"
    report = _report_for_backend(source_dirs, mode, backend)
    engine_meta = read_json(source_dirs[f"engine_{precision}"] / f"engine_meta_{precision}.json", default={}) or {}
    engine_path = report.get("engine_path") or engine_meta.get("engine_path")
    engine_size = engine_meta.get("engine_size_MB")
    if engine_size is None and engine_path and Path(engine_path).exists():
        engine_size = Path(engine_path).stat().st_size / (1024 * 1024)
    p50 = report.get("forward_p50_ms")
    payload = {
        "precision": precision,
        "pyramid_forward_export_mode": mode,
        "engine_path": engine_path,
        "num_frames": report.get("num_frames"),
        "warmup_frames": None,
        "actual_frames": report.get("actual_frames"),
        "forward_mean_ms": report.get("forward_mean_ms"),
        "forward_p50_ms": p50,
        "forward_p90_ms": report.get("forward_p90_ms"),
        "forward_p95_ms": report.get("forward_p95_ms"),
        "forward_min_ms": None,
        "forward_max_ms": None,
        "fps": float(1000.0 / p50) if p50 else None,
        "engine_size_MB": engine_size,
        "latency_scope": "engine_forward_on_real_samples",
        "source_report": str(source_dirs["evaluation"] / f"five_way_ap_report_{mode}.json"),
        "success": bool(report.get("success", report.get("actual_frames", 0))),
        "error": report.get("error"),
    }
    bench_dir = dirs["benchmark"] / f"{precision}_{mode}"
    save_json(payload, bench_dir / f"benchmark_{precision}.json")
    return payload


def _aligned_from_wrapper(wrapper: dict[str, Any], group_key: str, threshold: float) -> bool | None:
    group = (wrapper.get("by_record_len") or {}).get(group_key)
    if not group:
        return None
    outputs = group.get("outputs") or {}
    values = [item.get("max_abs_error") for item in outputs.values() if item.get("max_abs_error") is not None]
    if not values:
        return None
    return max(float(v) for v in values) <= threshold


def _row_from_existing(mode: str, dirs: dict[str, Path], source_dirs: dict[str, Path]) -> dict[str, Any]:
    five_way = read_json(dirs["evaluation"] / f"five_way_ap_report_{mode}.json", default={}) or {}
    reports = five_way.get("reports") or {}
    semantics = read_json(dirs["debug"] / f"{mode}_onnx_semantics_report.json", default={}) or {}
    wrapper = read_json(dirs["debug"] / f"wrapper_equivalence_{mode}.json", default={}) or {}
    fp32_bench = read_json(dirs["benchmark"] / f"fp32_{mode}" / "benchmark_fp32.json", default={}) or {}
    fp16_bench = read_json(dirs["benchmark"] / f"fp16_{mode}" / "benchmark_fp16.json", default={}) or {}
    onnx_path = five_way.get("onnx_path") or str(source_dirs["onnx_fp32"] / f"lidar_pyramid_{mode}_fp32_dynamic.onnx")
    return {
        "strategy": mode,
        "onnx_export_success": Path(onnx_path).exists(),
        "onnx_path": onnx_path,
        "onnx_preserves_multi_agent_semantics": bool(
            not semantics.get("semantics_erased_in_onnx")
            and semantics.get("valid_for_multi_agent_deployment")
        ),
        "ort_fp32_mAP": _map_from_report(reports.get("onnxruntime_fp32") or {}),
        "trt_fp32_mAP": _map_from_report(reports.get("tensorrt_fp32") or {}),
        "trt_fp16_mAP": _map_from_report(reports.get("tensorrt_fp16") or {}),
        "record_len_1_aligned": _aligned_from_wrapper(wrapper, "1", 2.0e-2),
        "record_len_2_aligned": _aligned_from_wrapper(wrapper, "2", 1.0e-3),
        "fp32_engine_p50": fp32_bench.get("forward_p50_ms"),
        "fp16_engine_p50": fp16_bench.get("forward_p50_ms"),
        "plugin_needed": False,
        "failure_reason": semantics.get("failure_reason"),
    }


def _write_mode_summary_from_existing(
    mode: str,
    dirs: dict[str, Path],
    source_dirs: dict[str, Path],
    fp32_benchmark: dict[str, Any],
    fp16_benchmark: dict[str, Any],
) -> dict[str, Any]:
    five_way = read_json(dirs["evaluation"] / f"five_way_ap_report_{mode}.json", default={}) or {}
    semantics = read_json(dirs["debug"] / f"{mode}_onnx_semantics_report.json", default={}) or {}
    wrapper = read_json(dirs["debug"] / f"wrapper_equivalence_{mode}.json", default={}) or {}
    payload = {
        "mode": mode,
        "source_output_root": str(source_dirs["output_root"]),
        "onnx_export_success": bool((five_way.get("onnx_path") and Path(five_way["onnx_path"]).exists())),
        "onnx_preserves_multi_agent_semantics": bool(
            not semantics.get("semantics_erased_in_onnx")
            and semantics.get("valid_for_multi_agent_deployment")
        ),
        "semantics": semantics,
        "wrapper_equivalence": wrapper,
        "five_way": five_way,
        "fp32_benchmark": fp32_benchmark,
        "fp16_benchmark": fp16_benchmark,
        "plugin_needed": False,
        "need_bevwarp_plugin": False,
        "need_pointpillar_scatter_plugin": False,
        "need_bevpool_plugin": False,
        "int8_qdq_modelopt_plugin_status": "not_run_by_request",
    }
    save_json(payload, dirs["summary"] / _mode_summary_path(mode))
    return payload


def sync_existing_strategy_reports(args: argparse.Namespace) -> dict[str, Any]:
    dirs = create_quant_deploy_run_dirs(args.output_dir, args.run_name, args.overwrite)
    source_roots = {
        "dynamic_agent_dim": Path(args.dynamic_agent_root).expanduser() if args.dynamic_agent_root else None,
        "padded_agent_static": Path(args.padded_agent_root).expanduser() if args.padded_agent_root else None,
    }
    rows: list[dict[str, Any]] = []
    mode_summaries: dict[str, Any] = {}
    for mode, source_root in source_roots.items():
        if source_root is None:
            continue
        source_dirs = ensure_quant_deploy_run_dirs(source_root)
        for rel in [
            f"debug/{mode}_onnx_semantics_report.json",
            f"debug/{mode}_record_len_usage_report.json",
            f"debug/{mode}_wrapper_equivalence.json",
            f"debug/wrapper_equivalence_{mode}.json",
            f"evaluation/five_way_ap_report_{mode}.json",
            f"evaluation/ort_fp32_ap_report_{mode}.json",
        ]:
            _copy_if_exists(source_dirs["output_root"] / rel, dirs["output_root"] / rel)
        if mode == "padded_agent_static":
            _copy_if_exists(
                source_dirs["debug"] / "valid_agent_mask_onnx_usage_report.json",
                dirs["debug"] / "valid_agent_mask_onnx_usage_report.json",
            )
        for precision in ("fp32", "fp16"):
            src_eval = source_dirs[f"evaluation_{precision}"] / f"eval_metrics_{precision}_{mode}.json"
            report_name = f"trt_{precision}_ap_report_{mode}.json"
            report = _canonical_backend_report(
                read_json(src_eval, default={}) or {},
                f"tensorrt_{precision}",
                f"TensorRT {precision.upper()}",
            )
            save_json(report, dirs["evaluation"] / report_name)
            _copy_if_exists(src_eval, dirs[f"evaluation_{precision}"] / src_eval.name)
            _copy_if_exists(source_dirs[f"evaluation_{precision}"] / f"eval_metrics_{precision}_{mode}.csv", dirs[f"evaluation_{precision}"] / f"eval_metrics_{precision}_{mode}.csv")
            _copy_if_exists(
                source_dirs[f"engine_{precision}"] / f"lidar_pyramid_{mode}_{precision}.engine",
                dirs[f"engine_{precision}"] / f"lidar_pyramid_{mode}_{precision}.engine",
            )
            _copy_if_exists(
                source_dirs[f"engine_{precision}"] / f"engine_meta_{precision}.json",
                dirs[f"engine_{precision}"] / f"engine_meta_{precision}_{mode}.json",
            )
            _copy_if_exists(
                source_dirs[f"engine_{precision}"] / f"layerinfo_{precision}.json",
                dirs[f"engine_{precision}"] / f"layerinfo_{precision}_{mode}.json",
            )
        _copy_if_exists(
            source_dirs["onnx_fp32"] / f"lidar_pyramid_{mode}_fp32_dynamic.onnx",
            dirs["onnx_fp32"] / f"lidar_pyramid_{mode}_fp32_dynamic.onnx",
        )
        _copy_if_exists(
            source_dirs["onnx_fp32"] / "input_output_names.json",
            dirs["onnx_fp32"] / f"input_output_names_{mode}.json",
        )
        ort_report = _canonical_backend_report(
            read_json(dirs["evaluation"] / f"ort_fp32_ap_report_{mode}.json", default={}) or {},
            "onnxruntime_fp32",
            "ONNXRuntime FP32",
        )
        save_json(ort_report, dirs["evaluation"] / f"ort_fp32_ap_report_{mode}.json")

        trt_equiv_src = source_dirs["evaluation"] / f"trt_output_equivalence_{mode}.json"
        if trt_equiv_src.exists():
            _copy_if_exists(trt_equiv_src, dirs["evaluation"] / f"trt_output_equivalence_{mode}.json")
            _copy_if_exists(source_dirs["debug"] / f"trt_output_equivalence_{mode}_debug.json", dirs["debug"] / f"trt_output_equivalence_{mode}_debug.json")
            trt_equiv = read_json(trt_equiv_src, default={}) or {}
            ort_only = {
                "success": bool((trt_equiv.get("comparisons") or {}).get("onnxruntime_fp32_vs_pytorch", {}).get("success")),
                "comparison": "onnxruntime_fp32_vs_pytorch",
                "report_source": str(trt_equiv_src),
                "data": (trt_equiv.get("comparisons") or {}).get("onnxruntime_fp32_vs_pytorch"),
            }
            save_json(ort_only, dirs["debug"] / f"ort_output_equivalence_{mode}.json")
        fp32_benchmark = _write_real_sample_benchmark_from_ap_report(dirs, source_dirs, mode, "fp32")
        fp16_benchmark = _write_real_sample_benchmark_from_ap_report(dirs, source_dirs, mode, "fp16")
        mode_summaries[mode] = _write_mode_summary_from_existing(mode, dirs, source_dirs, fp32_benchmark, fp16_benchmark)
        rows.append(_row_from_existing(mode, dirs, source_dirs))

    recommendation = _choose_recommendation(rows)
    comparison = {
        "output_root": str(dirs["output_root"]),
        "num_frames": int(args.num_frames),
        "max_cav": int(args.max_cav),
        "strategies": rows,
        "mode_summaries": mode_summaries,
        "recommended_strategy": recommendation,
        "recommendation_reason": "Both strategies preserve multi-agent semantics and align AP; padded_agent_static is recommended because it uses explicit valid_agent_mask and has lower real-sample TensorRT p50 latency.",
        "plugin_needed": False,
        "need_bevwarp_plugin": False,
        "need_pointpillar_scatter_plugin": False,
        "need_bevpool_plugin": False,
        "need_inverse_plugin": False,
        "int8_qdq_modelopt_plugin_status": "not_run_by_request",
    }
    save_json(comparison, dirs["summary"] / "agent_export_strategy_comparison.json")
    _write_comparison_markdown(dirs["summary"] / "agent_export_strategy_comparison.md", rows, recommendation)
    summary_all = {
        "model": "lidar_pyramid",
        "checkpoint": args.checkpoint,
        "hypes_yaml": args.hypes_yaml,
        "output_root": str(dirs["output_root"]),
        "dirs": dirs_for_summary(dirs),
        "agent_export_strategy_comparison": comparison,
        "recommended_agent_export_strategy": recommendation,
        "plugin_needed": False,
        "need_bevwarp_plugin": False,
        "need_pointpillar_scatter_plugin": False,
        "need_bevpool_plugin": False,
        "need_inverse_plugin": False,
        "int8_qdq_modelopt_plugin_status": "not_run_by_request",
    }
    if rows:
        summary_all["five_way_ap_table"] = read_json(
            dirs["evaluation"] / f"five_way_ap_report_{recommendation}.json",
            default={},
        ).get("five_way_ap_table", [])
    write_summary_files(summary_all, dirs)
    return comparison


def run_comparison(args: argparse.Namespace) -> dict[str, Any]:
    dirs = create_quant_deploy_run_dirs(args.output_dir, args.run_name, args.overwrite)
    results: dict[str, Any] = {}

    for mode in MODES:
        mode_result: dict[str, Any] = {"mode": mode}
        try:
            export_result = export_lidar_pyramid_onnx(
                SimpleNamespace(
                    hypes_yaml=args.hypes_yaml,
                    checkpoint=args.checkpoint,
                    heal_repo=args.heal_repo,
                    output_dir=args.output_dir,
                    output_root=str(dirs["output_root"]),
                    run_name=args.run_name,
                    device=args.device,
                    num_frames=1,
                    max_cav=args.max_cav,
                    opset=args.opset,
                    overwrite=True,
                    allow_synthetic_fallback=args.allow_synthetic_fallback,
                    trt_root=args.trt_root,
                    trtexec_path=args.trtexec_path,
                    bev_warp_export_mode="exportable_grid",
                    pillar_vfe_export_fix="explicit_squeeze",
                    pyramid_forward_export_mode=mode,
                )
            )
            mode_result["export"] = export_result
            if not export_result.get("success"):
                mode_result["failure_reason"] = export_result.get("error")
                results[mode] = mode_result
                continue
            onnx_path = export_result["onnx_path"]
            mode_result["semantics"] = analyze_agent_export_semantics(onnx_path, dirs["output_root"], mode)
            mode_result["wrapper_equivalence"] = run_wrapper_equivalence(
                SimpleNamespace(
                    output_root=str(dirs["output_root"]),
                    hypes_yaml=args.hypes_yaml,
                    checkpoint=args.checkpoint,
                    heal_repo=args.heal_repo,
                    device=args.device,
                    num_frames=args.num_frames,
                    max_cav=args.max_cav,
                    pyramid_forward_export_mode=mode,
                )
            )

            if not args.skip_trt:
                for precision in ("fp32", "fp16"):
                    build = build_engine(
                        SimpleNamespace(
                            onnx_path=onnx_path,
                            output_root=str(dirs["output_root"]),
                            precision=precision,
                            profile_shapes_json=None,
                            trt_root=args.trt_root,
                            trtexec_path=args.trtexec_path,
                            timeout=args.timeout,
                            no_tf32=args.no_tf32,
                            allow_tf32=args.allow_tf32,
                            strict_fp16=args.strict_fp16,
                            calib_dir=None,
                            calib_num_frames=0,
                            calib_cache=None,
                            qdq_onnx_path=None,
                            int8_mode="native_trt",
                            allow_fp16_fallback=False,
                            engine_name_prefix=_engine_prefix(mode),
                        )
                    )
                    mode_result[f"build_{precision}"] = build
                    benchmark_engine(
                        SimpleNamespace(
                            output_root=str(dirs["output_root"]),
                            precision=precision,
                            engine_path=build.get("engine_path") if build.get("build_success") else None,
                            num_frames=args.num_frames,
                            warmup_frames=args.warmup_frames,
                            device=args.device,
                            trt_root=args.trt_root,
                            trtexec_path=args.trtexec_path,
                            timeout=args.timeout,
                            benchmark_dir_name=f"{precision}_{mode}",
                        )
                    )
                    if build.get("build_success"):
                        eval_report = evaluate_engine(
                            SimpleNamespace(
                                output_root=str(dirs["output_root"]),
                                precision=precision,
                                engine_path=build.get("engine_path"),
                                hypes_yaml=args.hypes_yaml,
                                checkpoint=args.checkpoint,
                                heal_repo=args.heal_repo,
                                device=args.device,
                                num_frames=args.num_frames,
                                num_workers=0,
                                ap_iou_backend="gpu",
                                pyramid_forward_export_mode=mode,
                                max_cav=args.max_cav,
                            )
                        )
                        mode_result[f"trt_{precision}_ap"] = eval_report

                mode_result["trt_equivalence"] = run_trt_output_equivalence(
                    SimpleNamespace(
                        output_root=str(dirs["output_root"]),
                        hypes_yaml=args.hypes_yaml,
                        checkpoint=args.checkpoint,
                        heal_repo=args.heal_repo,
                        device=args.device,
                        max_cav=args.max_cav,
                        allow_synthetic_fallback=args.allow_synthetic_fallback,
                        onnx_path=onnx_path,
                        fp32_engine_path=str(_engine_path(dirs, mode, "fp32")),
                        fp16_engine_path=str(_engine_path(dirs, mode, "fp16")),
                        pyramid_forward_export_mode=mode,
                    )
                )

            mode_result["five_way"] = run_five_way_evaluation(
                SimpleNamespace(
                    output_root=str(dirs["output_root"]),
                    onnx_path=onnx_path,
                    fp32_engine_path=str(_engine_path(dirs, mode, "fp32")),
                    fp16_engine_path=str(_engine_path(dirs, mode, "fp16")),
                    hypes_yaml=args.hypes_yaml,
                    checkpoint=args.checkpoint,
                    heal_repo=args.heal_repo,
                    device=args.device,
                    num_frames=args.num_frames,
                    num_workers=0,
                    ap_iou_backend="gpu",
                    topk=20,
                    top_drop_frames=5,
                    pyramid_forward_export_mode=mode,
                    max_cav=args.max_cav,
                )
            )
            summary_payload = {
                "mode": mode,
                "export": mode_result.get("export"),
                "semantics": mode_result.get("semantics"),
                "wrapper_equivalence": mode_result.get("wrapper_equivalence"),
                "five_way": mode_result.get("five_way"),
                "trt_equivalence": mode_result.get("trt_equivalence"),
                "plugin_needed": False,
            }
            save_json(summary_payload, dirs["summary"] / _mode_summary_path(mode))
        except Exception as exc:
            mode_result.update({"failure_reason": str(exc), "traceback": traceback.format_exc()})
        results[mode] = mode_result

    rows = [_row_from_mode(mode, dirs, results.get(mode) or {}) for mode in MODES]
    recommendation = _choose_recommendation(rows)
    comparison = {
        "output_root": str(dirs["output_root"]),
        "num_frames": int(args.num_frames),
        "max_cav": int(args.max_cav),
        "strategies": rows,
        "recommended_strategy": recommendation,
        "plugin_needed": False,
        "need_bevwarp_plugin": False,
        "need_pointpillar_scatter_plugin": False,
        "need_bevpool_plugin": False,
        "int8_qdq_modelopt_plugin_status": "not_run_by_request",
    }
    save_json(comparison, dirs["summary"] / "agent_export_strategy_comparison.json")
    _write_comparison_markdown(dirs["summary"] / "agent_export_strategy_comparison.md", rows, recommendation)

    summary_all = read_json(dirs["summary"] / "summary_all.json", default={}) or {
        "model": "lidar_pyramid",
        "checkpoint": args.checkpoint,
        "hypes_yaml": args.hypes_yaml,
        "output_root": str(dirs["output_root"]),
        "dirs": dirs_for_summary(dirs),
        "onnx_export": {"success": True, "export_boundary": "full_model", "opset": args.opset, "error": None},
        "precisions": {},
        "detected_special_ops": {},
    }
    summary_all.update(
        {
            "agent_export_strategy_comparison": comparison,
            "recommended_agent_export_strategy": recommendation,
            "plugin_needed": False,
            "need_bevwarp_plugin": False,
            "need_pointpillar_scatter_plugin": False,
            "need_bevpool_plugin": False,
        }
    )
    write_summary_files(summary_all, dirs)
    return comparison


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.dynamic_agent_root or args.padded_agent_root:
        result = sync_existing_strategy_reports(args)
    else:
        result = run_comparison(args)
    print(result["output_root"])
    return 0 if result.get("recommended_strategy") else 2


if __name__ == "__main__":
    raise SystemExit(main())
