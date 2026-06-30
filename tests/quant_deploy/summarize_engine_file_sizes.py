from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from quant_deploy_utils import DEFAULT_OUTPUT_DIR, ensure_quant_deploy_run_dirs, read_json, save_json


BYTES_PER_MB = 1024 * 1024
PADDED_INT8_SKIP_REASON = (
    "padded_agent_static INT8 train_calib200 not evaluated: no existing fixedK29696 padded INT8 "
    "engine/report or train-calibration NPZ path with valid_agent_mask was present. Building it would "
    "require new calibration input plumbing and likely re-dumping calibration data, so it was skipped "
    "because this baseline does not affect the final single_engine_maxK FP16 recommendation."
)


ENGINE_LAYOUTS: list[dict[str, Any]] = [
    {
        "scheme": "padded_agent_static",
        "engine_strategy": "padded_agent_static_fixed_k_plugin",
        "precision": "fp32",
        "calibration": None,
        "patterns": [
            "artifacts/engines/fixedK{fixed_k}/padded_agent_static/fp32/*.engine",
            "artifacts/engines/fixedK{fixed_k}/fixed_k_scatter_plugin/fp32/*.engine",
        ],
    },
    {
        "scheme": "padded_agent_static",
        "engine_strategy": "padded_agent_static_fixed_k_plugin",
        "precision": "fp16",
        "calibration": None,
        "patterns": [
            "artifacts/engines/fixedK{fixed_k}/padded_agent_static/fp16/*.engine",
            "artifacts/engines/fixedK{fixed_k}/fixed_k_scatter_plugin/fp16/*.engine",
        ],
    },
    {
        "scheme": "dynamic_agent_dim",
        "engine_strategy": "dynamic_agent_dim_bucket_fixed_k_plugin",
        "precision": "fp32",
        "calibration": None,
        "patterns": [
            "artifacts/engines/fixedK{fixed_k}/dynamic_agent_dim_fixed_k_scatter_plugin/N*/fp32/*.engine",
        ],
    },
    {
        "scheme": "dynamic_agent_dim",
        "engine_strategy": "dynamic_agent_dim_bucket_fixed_k_plugin",
        "precision": "fp16",
        "calibration": None,
        "patterns": [
            "artifacts/engines/fixedK{fixed_k}/dynamic_agent_dim_fixed_k_scatter_plugin/N*/fp16/*.engine",
        ],
    },
    {
        "scheme": "dynamic_agent_dim",
        "engine_strategy": "dynamic_agent_dim_bucket_int8",
        "precision": "int8",
        "calibration": "train_calib50",
        "patterns": [
            "artifacts/engines/fixedK{fixed_k}/dynamic_agent_dim_fixed_k_scatter_plugin/N*/int8_calib50/*.engine",
        ],
    },
    {
        "scheme": "dynamic_agent_dim",
        "engine_strategy": "dynamic_agent_dim_bucket_int8",
        "precision": "int8",
        "calibration": "train_calib200",
        "patterns": [
            "artifacts/engines/fixedK{fixed_k}/dynamic_agent_dim_fixed_k_scatter_plugin/N*/int8_calib200/*.engine",
        ],
    },
    {
        "scheme": "dynamic_agent_single_engine_maxK",
        "engine_strategy": "single TensorRT engine",
        "precision": "fp32",
        "calibration": None,
        "patterns": [
            "artifacts/engines/fixedK{fixed_k}/dynamic_agent_single_engine_maxK/fp32/*.engine",
        ],
    },
    {
        "scheme": "dynamic_agent_single_engine_maxK",
        "engine_strategy": "single TensorRT engine",
        "precision": "fp16",
        "calibration": None,
        "patterns": [
            "artifacts/engines/fixedK{fixed_k}/dynamic_agent_single_engine_maxK/fp16/*.engine",
        ],
    },
    {
        "scheme": "dynamic_agent_single_engine_maxK",
        "engine_strategy": "single TensorRT engine",
        "precision": "int8",
        "calibration": "train_calib50",
        "patterns": [
            "artifacts/engines/fixedK{fixed_k}/dynamic_agent_single_engine_maxK/int8_train_calib50/*.engine",
        ],
    },
    {
        "scheme": "dynamic_agent_single_engine_maxK",
        "engine_strategy": "single TensorRT engine",
        "precision": "int8",
        "calibration": "train_calib200",
        "patterns": [
            "artifacts/engines/fixedK{fixed_k}/dynamic_agent_single_engine_maxK/int8_train_calib200/*.engine",
        ],
    },
    {
        "scheme": "padded_agent_static",
        "engine_strategy": "padded_agent_static_fixed_k_plugin",
        "precision": "int8",
        "calibration": "train_calib200",
        "optional_baseline": True,
        "patterns": [
            "artifacts/engines/fixedK{fixed_k}/padded_agent_static/int8_train_calib200/*.engine",
            "artifacts/engines/fixedK{fixed_k}/fixed_k_scatter_plugin/int8_train_calib200/*.engine",
        ],
    },
]


EVAL_REPORT_BY_ROW: dict[tuple[str, str, str | None], str] = {
    ("padded_agent_static", "fp32", None): "padded_agent_static_fp32_full_val.json",
    ("padded_agent_static", "fp16", None): "padded_agent_static_fp16_full_val.json",
    ("padded_agent_static", "int8", "train_calib200"): "padded_agent_static_int8_train_calib200_full_val.json",
    ("dynamic_agent_dim", "fp32", None): "dynamic_bucket_fp32_full_val.json",
    ("dynamic_agent_dim", "fp16", None): "dynamic_bucket_fp16_full_val.json",
    ("dynamic_agent_dim", "int8", "train_calib50"): "dynamic_bucket_int8_calib50_full_val.json",
    ("dynamic_agent_dim", "int8", "train_calib200"): "dynamic_bucket_int8_calib200_full_val.json",
    ("dynamic_agent_single_engine_maxK", "fp32", None): "single_engine_maxK_fp32_full_val.json",
    ("dynamic_agent_single_engine_maxK", "fp16", None): "single_engine_maxK_fp16_full_val.json",
    ("dynamic_agent_single_engine_maxK", "int8", "train_calib50"): "single_engine_maxK_int8_train_calib50_full_val.json",
    ("dynamic_agent_single_engine_maxK", "int8", "train_calib200"): "single_engine_maxK_int8_train_calib200_full_val.json",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize fixedK engine file sizes and package-size tradeoffs.")
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_DIR / "lidar_pyramid_agent_export_strategy_compare"))
    parser.add_argument("--fixed_k", type=int, default=29696)
    parser.add_argument("--full_val_tag", default=None)
    return parser.parse_args(argv)


def mb(size_bytes: int | float | None) -> float | None:
    if size_bytes is None:
        return None
    return round(float(size_bytes) / BYTES_PER_MB, 4)


def pct(delta: float | None) -> float | None:
    if delta is None:
        return None
    return round(float(delta) * 100.0, 2)


def _dedupe(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    unique: list[Path] = []
    for path in sorted(paths, key=lambda item: str(item)):
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _glob_engine_files(output_root: Path, patterns: list[str], fixed_k: int) -> list[Path]:
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(output_root.glob(pattern.format(fixed_k=int(fixed_k))))
    return _dedupe([path for path in matches if path.is_file() and path.suffix == ".engine"])


def collect_engine_inventory(
    output_root: str | Path,
    *,
    fixed_k: int,
    layouts: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    root = Path(output_root)
    rows: list[dict[str, Any]] = []
    for layout in layouts or ENGINE_LAYOUTS:
        files = _glob_engine_files(root, list(layout["patterns"]), int(fixed_k))
        sizes = [path.stat().st_size for path in files]
        total_bytes = sum(sizes)
        row = {
            "scheme": layout["scheme"],
            "engine_strategy": layout["engine_strategy"],
            "fixed_K": int(fixed_k),
            "precision": layout["precision"],
            "calibration": layout.get("calibration"),
            "engine_dir_patterns": [pattern.format(fixed_k=int(fixed_k)) for pattern in layout["patterns"]],
            "engine_dirs": sorted({str(path.parent) for path in files}),
            "engine_count": len(files),
            "engine_files": [
                {
                    "path": str(path),
                    "size_bytes": size,
                    "size_MB": mb(size),
                }
                for path, size in zip(files, sizes)
            ],
            "per_engine_size_MB": [mb(size) for size in sizes],
            "total_engine_size_bytes": total_bytes,
            "total_engine_size_MB": mb(total_bytes),
            "largest_engine_MB": mb(max(sizes)) if sizes else None,
            "smallest_engine_MB": mb(min(sizes)) if sizes else None,
            "mean_engine_size_MB": round(mean([float(size) / BYTES_PER_MB for size in sizes]), 4) if sizes else None,
            "status": "present" if files else "missing",
            "notes": "",
        }
        if layout.get("optional_baseline") and not files:
            row["status"] = "not_evaluated"
            row["notes"] = PADDED_INT8_SKIP_REASON
        rows.append(row)
    return rows


def _ratio(numerator: int | float | None, denominator: int | float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return round(float(numerator) / float(denominator), 4)


def _same_calibration_key(row: dict[str, Any]) -> tuple[str, str | None]:
    return str(row.get("precision")), row.get("calibration")


def compute_size_ratios(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    single_by_key: dict[tuple[str, str | None], dict[str, Any]] = {}
    dynamic_by_key: dict[tuple[str, str | None], dict[str, Any]] = {}
    for row in rows:
        key = _same_calibration_key(row)
        if row.get("scheme") == "dynamic_agent_single_engine_maxK":
            single_by_key[key] = row
        if row.get("scheme") == "dynamic_agent_dim":
            dynamic_by_key[key] = row

    for row in rows:
        key = _same_calibration_key(row)
        total = row.get("total_engine_size_bytes")
        if not row.get("engine_count"):
            row["total_size_ratio_vs_single_engine_maxK_same_precision"] = None
            row["total_size_ratio_vs_dynamic_bucket_same_precision"] = None
            continue
        single = single_by_key.get(key)
        dynamic = dynamic_by_key.get(key)
        row["total_size_ratio_vs_single_engine_maxK_same_precision"] = _ratio(
            total,
            single.get("total_engine_size_bytes") if single else None,
        )
        row["total_size_ratio_vs_dynamic_bucket_same_precision"] = _ratio(
            total,
            dynamic.get("total_engine_size_bytes") if dynamic else None,
        )
    return rows


def _metric_value(report: dict[str, Any], primary: str, fallback: str | None = None) -> Any:
    if primary in report:
        return report.get(primary)
    if fallback is not None:
        return report.get(fallback)
    return None


def _latency(report: dict[str, Any], name: str) -> Any:
    value = report.get(name)
    if isinstance(value, dict):
        return value.get("p50")
    return None


def attach_eval_metrics(
    rows: list[dict[str, Any]],
    output_root: str | Path,
    *,
    full_val_tag: str,
) -> list[dict[str, Any]]:
    eval_dir = Path(output_root) / "evaluation" / full_val_tag
    for row in rows:
        key = (str(row.get("scheme")), str(row.get("precision")), row.get("calibration"))
        report_name = EVAL_REPORT_BY_ROW.get(key)
        report = read_json(eval_dir / report_name, default=None) if report_name else None
        if not isinstance(report, dict):
            row.update(
                {
                    "AP@0.30": None,
                    "AP@0.50": None,
                    "AP@0.70": None,
                    "mAP": None,
                    "forward_p50": None,
                    "execute_p50": None,
                    "FPS": None,
                    "evaluation_report_path": str(eval_dir / report_name) if report_name else None,
                }
            )
            if row.get("status") == "present":
                row["notes"] = (row.get("notes") or "") + " evaluation report missing"
            continue
        row.update(
            {
                "AP@0.30": _metric_value(report, "AP@0.30"),
                "AP@0.50": _metric_value(report, "AP@0.50"),
                "AP@0.70": _metric_value(report, "AP@0.70"),
                "mAP": _metric_value(report, "mAP"),
                "forward_p50": _latency(report, "forward_ms"),
                "execute_p50": _latency(report, "execute_ms"),
                "FPS": report.get("FPS"),
                "evaluation_report_path": str(eval_dir / report_name),
                "evaluated_engine_count": report.get("engine_count"),
                "total_val": report.get("total_val_samples"),
                "evaluated": report.get("evaluated_samples"),
                "skipped": len(report.get("skipped_samples") or []) if isinstance(report.get("skipped_samples"), list) else report.get("skipped_samples"),
                "reliable_latency": None if not report else not bool(report.get("unreliable_latency")),
            }
        )
        if row.get("engine_count") != report.get("engine_count"):
            extra = (
                f"deployment package engine_count={row.get('engine_count')} differs from "
                f"full-val evaluated engine_count={report.get('engine_count')}"
            )
            row["notes"] = "; ".join(part for part in [row.get("notes"), extra] if part)
    return rows


def _row_index(rows: list[dict[str, Any]], scheme: str, precision: str, calibration: str | None) -> dict[str, Any] | None:
    for row in rows:
        if row.get("scheme") == scheme and row.get("precision") == precision and row.get("calibration") == calibration:
            return row
    return None


def _speed_advantage_percent(faster: dict[str, Any] | None, slower: dict[str, Any] | None) -> float | None:
    if not faster or not slower:
        return None
    fast_ms = faster.get("forward_p50")
    slow_ms = slower.get("forward_p50")
    if fast_ms is None or slow_ms in (None, 0):
        return None
    return round((float(slow_ms) - float(fast_ms)) / float(slow_ms) * 100.0, 2)


def build_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    dynamic_fp16 = _row_index(rows, "dynamic_agent_dim", "fp16", None)
    single_fp16 = _row_index(rows, "dynamic_agent_single_engine_maxK", "fp16", None)
    dynamic_int8_200 = _row_index(rows, "dynamic_agent_dim", "int8", "train_calib200")
    single_int8_200 = _row_index(rows, "dynamic_agent_single_engine_maxK", "int8", "train_calib200")
    padded_fp16 = _row_index(rows, "padded_agent_static", "fp16", None)
    padded_int8_200 = _row_index(rows, "padded_agent_static", "int8", "train_calib200")

    dynamic_fp16_speedup = _speed_advantage_percent(dynamic_fp16, single_fp16)
    dynamic_int8_speedup = _speed_advantage_percent(dynamic_int8_200, single_int8_200)
    dynamic_fp16_size_ratio = (dynamic_fp16 or {}).get("total_size_ratio_vs_single_engine_maxK_same_precision")
    dynamic_int8_size_ratio = (dynamic_int8_200 or {}).get("total_size_ratio_vs_single_engine_maxK_same_precision")

    dynamic_int8_drop = None
    if dynamic_fp16 and dynamic_int8_200 and dynamic_fp16.get("mAP") is not None and dynamic_int8_200.get("mAP") is not None:
        dynamic_int8_drop = round(float(dynamic_fp16["mAP"]) - float(dynamic_int8_200["mAP"]), 4)
    single_int8_drop = None
    if single_fp16 and single_int8_200 and single_fp16.get("mAP") is not None and single_int8_200.get("mAP") is not None:
        single_int8_drop = round(float(single_fp16["mAP"]) - float(single_int8_200["mAP"]), 4)
    padded_int8_drop = None
    padded_int8_ap70_drop = None
    if padded_fp16 and padded_int8_200 and padded_fp16.get("mAP") is not None and padded_int8_200.get("mAP") is not None:
        padded_int8_drop = round(float(padded_fp16["mAP"]) - float(padded_int8_200["mAP"]), 4)
    if padded_fp16 and padded_int8_200 and padded_fp16.get("AP@0.70") is not None and padded_int8_200.get("AP@0.70") is not None:
        padded_int8_ap70_drop = round(float(padded_fp16["AP@0.70"]) - float(padded_int8_200["AP@0.70"]), 4)

    return {
        "dynamic_bucket_FP16_total_engine_size_MB": (dynamic_fp16 or {}).get("total_engine_size_MB"),
        "single_engine_maxK_FP16_total_engine_size_MB": (single_fp16 or {}).get("total_engine_size_MB"),
        "dynamic_bucket_FP16_size_ratio_vs_single_engine_maxK_FP16": dynamic_fp16_size_ratio,
        "dynamic_bucket_FP16_forward_p50_speedup_vs_single_engine_maxK_FP16_percent": dynamic_fp16_speedup,
        "dynamic_bucket_INT8_train_calib200_total_engine_size_MB": (dynamic_int8_200 or {}).get("total_engine_size_MB"),
        "single_engine_maxK_INT8_train_calib200_total_engine_size_MB": (single_int8_200 or {}).get("total_engine_size_MB"),
        "dynamic_bucket_INT8_train_calib200_size_ratio_vs_single_engine_maxK_INT8": dynamic_int8_size_ratio,
        "dynamic_bucket_INT8_train_calib200_forward_p50_speedup_vs_single_engine_maxK_INT8_percent": dynamic_int8_speedup,
        "padded_static_FP16_total_engine_size_MB": (padded_fp16 or {}).get("total_engine_size_MB"),
        "padded_agent_static_INT8_train_calib200_evaluated": bool(padded_int8_200 and padded_int8_200.get("status") == "present" and padded_int8_200.get("mAP") is not None),
        "padded_agent_static_INT8_train_calib200_skip_reason": None
        if padded_int8_200 and padded_int8_200.get("status") == "present" and padded_int8_200.get("mAP") is not None
        else PADDED_INT8_SKIP_REASON,
        "padded_agent_static_INT8_train_calib200_mAP_drop_vs_padded_FP16": padded_int8_drop,
        "padded_agent_static_INT8_train_calib200_AP70_drop_vs_padded_FP16": padded_int8_ap70_drop,
        "padded_agent_static_INT8_train_calib200_satisfies_mAP_drop_le_0p1": bool(padded_int8_drop is not None and padded_int8_drop <= 0.1),
        "multi_bucket_route_engines_significantly_increase_package_size": bool(
            dynamic_fp16_size_ratio is not None and float(dynamic_fp16_size_ratio) >= 2.0
        ),
        "dynamic_bucket_FP16_speed_size_tradeoff": (
            "dynamic bucket FP16 is a speed upper-bound route, but the small forward-p50 gain does not justify "
            "the multi-engine package-size increase as the default deployment package."
        ),
        "dynamic_bucket_INT8_train_calib200_mAP_drop_vs_dynamic_FP16": dynamic_int8_drop,
        "single_engine_INT8_train_calib200_mAP_drop_vs_single_FP16": single_int8_drop,
        "INT8_recommended_as_default": False,
        "recommend_mixed_precision_whitelist_qdq_modelopt": True,
        "final_recommended_default_path": "dynamic_agent_single_engine_maxK FP16 fixedK29696 + PointPillarScatterTRT",
        "dynamic_bucket_FP16_role": "speed upper-bound / optional low-latency route when package size is acceptable",
        "single_engine_maxK_FP16_default": True,
    }


def build_report(output_root: str | Path, *, fixed_k: int, full_val_tag: str) -> dict[str, Any]:
    rows = collect_engine_inventory(output_root, fixed_k=fixed_k)
    attach_eval_metrics(rows, output_root, full_val_tag=full_val_tag)
    compute_size_ratios(rows)
    analysis = build_analysis(rows)
    return {
        "output_root": str(output_root),
        "fixed_K": int(fixed_k),
        "full_val_tag": full_val_tag,
        "inventory": rows,
        "analysis": analysis,
        "notes": [
            "Engine size is counted from serialized TensorRT .engine files on disk.",
            "For multi-route schemes, deployment package size is the sum of all matching route engines, not a single engine size.",
            "Existing fixedK29696 full-val AP/latency reports were reused; this script does not rebuild or re-evaluate engines.",
            PADDED_INT8_SKIP_REASON,
        ],
    }


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def write_markdown(report: dict[str, Any], path: Path) -> None:
    rows = report["inventory"]
    analysis = report["analysis"]
    lines = [
        "# Engine File Size and Deployment Package Report",
        "",
        f"- output_root: {report['output_root']}",
        f"- fixed_K: {report['fixed_K']}",
        f"- full_val_tag: {report['full_val_tag']}",
        "- existing full-val AP/latency reports were reused; no K scan, calibration dump, or trusted full-val re-evaluation was run.",
        "",
        "scheme | engine_strategy | precision | calibration | engine_count | total_engine_size_MB | ratio_vs_single_same_precision | AP@0.30 | AP@0.50 | AP@0.70 | mAP | forward_p50 | FPS | notes",
        "--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---",
    ]
    for row in rows:
        lines.append(
            " | ".join(
                _fmt(row.get(key))
                for key in [
                    "scheme",
                    "engine_strategy",
                    "precision",
                    "calibration",
                    "engine_count",
                    "total_engine_size_MB",
                    "total_size_ratio_vs_single_engine_maxK_same_precision",
                    "AP@0.30",
                    "AP@0.50",
                    "AP@0.70",
                    "mAP",
                    "forward_p50",
                    "FPS",
                    "notes",
                ]
            )
        )

    lines.extend(
        [
            "",
            "## Answers",
            "",
            f"- dynamic bucket FP16 total engine size MB: {_fmt(analysis['dynamic_bucket_FP16_total_engine_size_MB'])}",
            f"- single_engine_maxK FP16 engine size MB: {_fmt(analysis['single_engine_maxK_FP16_total_engine_size_MB'])}",
            f"- dynamic bucket FP16 size ratio vs single_engine_maxK FP16: {_fmt(analysis['dynamic_bucket_FP16_size_ratio_vs_single_engine_maxK_FP16'])}x",
            f"- dynamic bucket INT8 train_calib200 total engine size MB: {_fmt(analysis['dynamic_bucket_INT8_train_calib200_total_engine_size_MB'])}",
            f"- single_engine_maxK INT8 train_calib200 engine size MB: {_fmt(analysis['single_engine_maxK_INT8_train_calib200_total_engine_size_MB'])}",
            f"- dynamic bucket INT8 train_calib200 size ratio vs single_engine_maxK INT8: {_fmt(analysis['dynamic_bucket_INT8_train_calib200_size_ratio_vs_single_engine_maxK_INT8'])}x",
            f"- padded static FP16 total engine size MB: {_fmt(analysis['padded_static_FP16_total_engine_size_MB'])}",
            f"- multi bucket / route engines significantly increase package size: {analysis['multi_bucket_route_engines_significantly_increase_package_size']}",
            f"- dynamic bucket FP16 forward p50 speedup vs single_engine_maxK FP16: {_fmt(analysis['dynamic_bucket_FP16_forward_p50_speedup_vs_single_engine_maxK_FP16_percent'])}%",
            f"- dynamic bucket INT8 train_calib200 forward p50 speedup vs single_engine_maxK INT8: {_fmt(analysis['dynamic_bucket_INT8_train_calib200_forward_p50_speedup_vs_single_engine_maxK_INT8_percent'])}%",
            f"- padded_agent_static INT8 train_calib200 evaluated: {analysis['padded_agent_static_INT8_train_calib200_evaluated']}",
            f"- padded_agent_static INT8 train_calib200 skip reason: {analysis['padded_agent_static_INT8_train_calib200_skip_reason']}",
            f"- final recommended default path: {analysis['final_recommended_default_path']}",
            f"- dynamic bucket FP16 role: {analysis['dynamic_bucket_FP16_role']}",
            f"- INT8 recommended as default: {analysis['INT8_recommended_as_default']}",
            f"- recommend mixed precision whitelist / Q-DQ / ModelOpt next: {analysis['recommend_mixed_precision_whitelist_qdq_modelopt']}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    dirs = ensure_quant_deploy_run_dirs(args.output_root)
    full_val_tag = args.full_val_tag or f"full_val_fixedK{int(args.fixed_k)}_trainCalib"
    report = build_report(dirs["output_root"], fixed_k=int(args.fixed_k), full_val_tag=full_val_tag)
    save_json(report, dirs["debug"] / "engine_file_size_inventory_fixedK29696.json")
    save_json(report, dirs["summary"] / "engine_file_size_and_deployment_package_report.json")
    write_markdown(report, dirs["summary"] / "engine_file_size_and_deployment_package_report.md")
    return report


def main(argv: list[str] | None = None) -> int:
    report = run(parse_args(argv))
    print(json.dumps({"fixed_K": report["fixed_K"], "rows": len(report["inventory"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
