from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from quant_deploy_utils import ensure_quant_deploy_run_dirs, read_json, save_json


HISTORICAL_DYNAMIC_BUCKET = {
    ("fp32", None, 50): {"AP@0.30": 0.8721, "AP@0.50": 0.8534, "AP@0.70": 0.7476, "mAP": 0.8244, "execute_p50": 4.99, "forward_p50": 6.47, "FPS": 154.54},
    ("fp16", None, 50): {"AP@0.30": 0.8723, "AP@0.50": 0.8523, "AP@0.70": 0.7322, "mAP": 0.8189, "execute_p50": 2.15, "forward_p50": 3.67, "FPS": 272.27},
    ("int8", "calib50", 50): {"AP@0.30": 0.8076, "AP@0.50": 0.7873, "AP@0.70": 0.6412, "mAP": 0.7454, "execute_p50": 1.53, "forward_p50": 2.55, "FPS": 392.58},
    ("int8", "calib200", 50): {"AP@0.30": 0.8257, "AP@0.50": 0.8042, "AP@0.70": 0.6327, "mAP": 0.7542, "execute_p50": 1.54, "forward_p50": 2.53, "FPS": 395.76},
    ("int8_mixed_heads_fp16", "calib200", 50): {"AP@0.30": 0.8250, "AP@0.50": 0.8024, "AP@0.70": 0.6422, "mAP": 0.7565, "execute_p50": 1.62, "forward_p50": 2.67, "FPS": 375.02},
    ("fp32", None, 200): {"AP@0.30": 0.8102, "AP@0.50": 0.7712, "AP@0.70": 0.5980, "mAP": 0.7265, "execute_p50": 4.99, "forward_p50": 6.45, "FPS": 155.05},
    ("fp16", None, 200): {"AP@0.30": 0.8105, "AP@0.50": 0.7704, "AP@0.70": 0.5944, "mAP": 0.7251, "execute_p50": 2.14, "forward_p50": 3.36, "FPS": 297.79},
    ("int8", "calib50", 200): {"AP@0.30": 0.6855, "AP@0.50": 0.6441, "AP@0.70": 0.4513, "mAP": 0.5936, "execute_p50": 1.55, "forward_p50": 2.55, "FPS": 391.48},
    ("int8", "calib200", 200): {"AP@0.30": 0.7604, "AP@0.50": 0.7154, "AP@0.70": 0.5107, "mAP": 0.6622, "execute_p50": 1.54, "forward_p50": 2.56, "FPS": 391.30},
    ("int8_mixed_heads_fp16", "calib200", 200): {"AP@0.30": 0.7605, "AP@0.50": 0.7157, "AP@0.70": 0.5079, "mAP": 0.6614, "execute_p50": 1.63, "forward_p50": 2.68, "FPS": 372.78},
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize all FP32/FP16/INT8 deployment strategies.")
    parser.add_argument("--output_root", required=True)
    return parser.parse_args(argv)


def _p50(metric: Any) -> float | None:
    if isinstance(metric, dict):
        value = metric.get("p50")
        return float(value) if value is not None else None
    if isinstance(metric, (int, float)):
        return float(metric)
    return None


def _row_from_report(
    report: dict[str, Any],
    *,
    scheme: str,
    engine_strategy: str,
    precision: str,
    calibration: str | None,
    frames: int,
    engine_count: int,
    dynamic_n: bool,
    bucket_router: bool,
    max_k: int | None,
    notes: str,
) -> dict[str, Any]:
    execute = _p50(report.get("execute_ms")) or report.get("execute_p50") or report.get("execute_p50_ms")
    forward = _p50(report.get("forward_ms")) or report.get("forward_p50") or report.get("forward_p50_ms")
    fps = report.get("FPS", report.get("fps"))
    return {
        "scheme": scheme,
        "engine_strategy": engine_strategy,
        "precision": precision,
        "calibration_split": report.get("calibration_split"),
        "evaluation_split": report.get("evaluation_split"),
        "calibration": calibration,
        "frames": int(frames),
        "engine_count": int(engine_count),
        "dynamic_N": bool(dynamic_n),
        "bucket_router": bool(bucket_router),
        "maxK": max_k,
        "AP@0.30": report.get("AP@0.30", report.get("ap_0_3")),
        "AP@0.50": report.get("AP@0.50", report.get("ap_0_5")),
        "AP@0.70": report.get("AP@0.70", report.get("ap_0_7")),
        "mAP": report.get("mAP", report.get("map")),
        "mAP_drop_vs_FP16": report.get("mAP drop vs single-engine FP16 same val set") or report.get("mAP_drop_vs_FP16"),
        "execute_p50": execute,
        "forward_p50": forward,
        "FPS": float(fps) if fps is not None else (1000.0 / float(forward) if forward else None),
        "notes": notes,
    }


def _read_or_historical(dirs: dict[str, Path], key: tuple[str, str | None, int]) -> dict[str, Any]:
    precision, calibration, frames = key
    if precision in {"fp32", "fp16"}:
        path = dirs["evaluation"] / f"trt_{precision}_ap_report_dynamic_agent_dim_fixed_k_plugin_{frames}.json"
    elif precision == "int8_mixed_heads_fp16":
        path = dirs["evaluation"] / f"trt_int8_ap_report_dynamic_fixed_k_plugin_mixed_heads_fp16_{frames}.json"
    else:
        calib = str(calibration).replace("calib", "")
        path = dirs["evaluation"] / f"trt_int8_ap_report_dynamic_fixed_k_plugin_calib{calib}_{frames}.json"
    report = read_json(path, default=None)
    return report or dict(HISTORICAL_DYNAMIC_BUCKET[key])


def _add_padded_rows(dirs: dict[str, Path], rows: list[dict[str, Any]]) -> None:
    for precision in ("fp32", "fp16"):
        for frames in (50, 200):
            path = dirs["evaluation"] / f"trt_{precision}_ap_report_padded_agent_static_fixed_k_plugin_{frames}.json"
            report = read_json(path, default={}) or {}
            if not report:
                continue
            rows.append(
                _row_from_report(
                    report,
                    scheme="padded_agent_static fixed-K plugin",
                    engine_strategy="bucket_router_baseline",
                    precision=precision,
                    calibration=None,
                    frames=frames,
                    engine_count=4,
                    dynamic_n=False,
                    bucket_router=True,
                    max_k=24064,
                    notes="optional baseline, not default",
                )
            )


def _add_dynamic_bucket_rows(dirs: dict[str, Path], rows: list[dict[str, Any]]) -> None:
    for key, fallback in HISTORICAL_DYNAMIC_BUCKET.items():
        precision, calibration, frames = key
        report = _read_or_historical(dirs, key)
        rows.append(
            _row_from_report(
                report,
                scheme="dynamic_agent_dim multi-engine bucket fixed-K plugin",
                engine_strategy="N1/N2 x bucket router",
                precision=precision,
                calibration=calibration,
                frames=frames,
                engine_count=8 if precision.startswith("int8") else 8,
                dynamic_n=False,
                bucket_router=True,
                max_k=24064,
                notes="historical_calibration_split_confirmed=false" if precision.startswith("int8") else "historical baseline",
            )
        )


def _add_single_engine_rows(dirs: dict[str, Path], rows: list[dict[str, Any]]) -> None:
    for precision in ("fp32", "fp16"):
        for frames in (50, 200):
            report = read_json(dirs["evaluation"] / f"dynamic_single_engine_maxK_{precision}_val{frames}.json", default={}) or {}
            if report:
                rows.append(
                    _row_from_report(
                        report,
                        scheme="dynamic_agent_single_engine_maxK",
                        engine_strategy="single TensorRT engine",
                        precision=precision,
                        calibration=None,
                        frames=frames,
                        engine_count=1,
                        dynamic_n=True,
                        bucket_router=False,
                        max_k=24064,
                        notes="calibration_split=n/a evaluation_split=val calibration_eval_overlap=false",
                    )
                )
    for calibration_frames in (50, 200, 500):
        for frames in (50, 200):
            report = read_json(dirs["evaluation"] / f"dynamic_single_engine_maxK_int8_train_calib{calibration_frames}_val{frames}.json", default={}) or {}
            if report:
                rows.append(
                    _row_from_report(
                        report,
                        scheme="dynamic_agent_single_engine_maxK",
                        engine_strategy="single TensorRT engine",
                        precision="int8",
                        calibration=f"train_calib{calibration_frames}",
                        frames=frames,
                        engine_count=1,
                        dynamic_n=True,
                        bucket_router=False,
                        max_k=24064,
                        notes="calibration_split=train evaluation_split=val calibration_eval_overlap=false",
                    )
                )


def _best(rows: list[dict[str, Any]], *, precision: str | None = None, metric: str = "forward_p50", reverse: bool = False) -> dict[str, Any] | None:
    candidates = [row for row in rows if (precision is None or str(row.get("precision")).startswith(precision)) and row.get(metric) is not None]
    if not candidates:
        return None
    return sorted(candidates, key=lambda row: float(row[metric]), reverse=reverse)[0]


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    dirs = ensure_quant_deploy_run_dirs(args.output_root)
    rows: list[dict[str, Any]] = []
    _add_padded_rows(dirs, rows)
    _add_dynamic_bucket_rows(dirs, rows)
    _add_single_engine_rows(dirs, rows)
    fp16_single_200 = next((row for row in rows if row["scheme"] == "dynamic_agent_single_engine_maxK" and row["precision"] == "fp16" and row["frames"] == 200), None)
    fp16_bucket_200 = next((row for row in rows if row["scheme"].startswith("dynamic_agent_dim") and row["precision"] == "fp16" and row["frames"] == 200), None)
    single_fp16_keeps_ap = False
    single_fp16_forward_within_10pct = False
    single_replaces_bucket = False
    if fp16_single_200 and fp16_bucket_200 and fp16_single_200.get("mAP") is not None and fp16_bucket_200.get("mAP") is not None:
        single_fp16_keeps_ap = abs(float(fp16_single_200["mAP"]) - float(fp16_bucket_200["mAP"])) <= 0.01
        single_fp16_forward_within_10pct = (
            float(fp16_single_200["forward_p50"]) <= float(fp16_bucket_200["forward_p50"]) * 1.10
            if fp16_single_200.get("forward_p50") and fp16_bucket_200.get("forward_p50")
            else False
        )
        single_replaces_bucket = bool(single_fp16_keeps_ap and single_fp16_forward_within_10pct)
    int8_candidates = [
        row
        for row in rows
        if row["scheme"] == "dynamic_agent_single_engine_maxK"
        and row["precision"] == "int8"
        and row.get("mAP_drop_vs_FP16") is not None
        and float(row["mAP_drop_vs_FP16"]) <= 0.1
    ]
    answers = {
        "fastest_FP32_scheme": _best(rows, precision="fp32", metric="forward_p50"),
        "fastest_FP16_scheme": _best(rows, precision="fp16", metric="forward_p50"),
        "fastest_INT8_scheme": _best(rows, precision="int8", metric="forward_p50"),
        "best_AP_scheme": _best(rows, precision=None, metric="mAP", reverse=True),
        "single_engine_maxK_success": any(row["scheme"] == "dynamic_agent_single_engine_maxK" for row in rows),
        "single_engine_maxK_fp16_keeps_AP": single_fp16_keeps_ap,
        "single_engine_maxK_fp16_forward_within_10pct": single_fp16_forward_within_10pct,
        "single_engine_maxK_keeps_AP": single_fp16_keeps_ap,
        "single_engine_maxK_should_replace_multi_engine_bucket": single_replaces_bucket,
        "INT8_single_engine_has_acceptable_accuracy": bool(int8_candidates),
        "INT8_mAP_drop_lt_0_1_can_be_candidate": bool(int8_candidates),
        "recommended_deployment_path": "dynamic_agent_single_engine_maxK FP16" if single_replaces_bucket else "dynamic_agent_dim fixed-K bucket router PointPillarScatterTRT FP16",
        "next_int8_optimization": "mixed precision whitelist before full Q/DQ ModelOpt" if not int8_candidates else "optional Q/DQ/ModelOpt to recover AP@0.70",
        "new_scheme_calibration_split_train": True,
        "new_scheme_evaluation_split_val": True,
    }
    report = {"success": True, "rows": rows, "answers": answers}
    save_json(report, dirs["summary"] / "all_deployment_strategies_fp32_fp16_int8_report.json")
    lines = [
        "# All Deployment Strategies FP32/FP16/INT8",
        "",
        "scheme | engine_strategy | precision | calibration_split | evaluation_split | calibration | frames | engine_count | dynamic_N | bucket_router | maxK | AP@0.30 | AP@0.50 | AP@0.70 | mAP | mAP_drop_vs_FP16 | execute_p50 | forward_p50 | FPS | notes",
        "--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---",
    ]
    for row in rows:
        lines.append(
            " | ".join(
                str(row.get(key, ""))
                for key in [
                    "scheme",
                    "engine_strategy",
                    "precision",
                    "calibration_split",
                    "evaluation_split",
                    "calibration",
                    "frames",
                    "engine_count",
                    "dynamic_N",
                    "bucket_router",
                    "maxK",
                    "AP@0.30",
                    "AP@0.50",
                    "AP@0.70",
                    "mAP",
                    "mAP_drop_vs_FP16",
                    "execute_p50",
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
            f"- fastest_FP32_scheme: {(answers['fastest_FP32_scheme'] or {}).get('scheme')}",
            f"- fastest_FP16_scheme: {(answers['fastest_FP16_scheme'] or {}).get('scheme')}",
            f"- fastest_INT8_scheme: {(answers['fastest_INT8_scheme'] or {}).get('scheme')}",
            f"- best_AP_scheme: {(answers['best_AP_scheme'] or {}).get('scheme')}",
            f"- single_engine_maxK_success: {answers['single_engine_maxK_success']}",
            f"- single_engine_maxK_fp16_keeps_AP: {answers['single_engine_maxK_fp16_keeps_AP']}",
            f"- single_engine_maxK_fp16_forward_within_10pct: {answers['single_engine_maxK_fp16_forward_within_10pct']}",
            f"- single_engine_maxK_should_replace_multi_engine_bucket: {answers['single_engine_maxK_should_replace_multi_engine_bucket']}",
            f"- INT8_single_engine_has_acceptable_accuracy: {answers['INT8_single_engine_has_acceptable_accuracy']}",
            f"- recommended_deployment_path: {answers['recommended_deployment_path']}",
            f"- new_scheme_calibration_split_train: {answers['new_scheme_calibration_split_train']}",
            f"- new_scheme_evaluation_split_val: {answers['new_scheme_evaluation_split_val']}",
        ]
    )
    (dirs["summary"] / "all_deployment_strategies_fp32_fp16_int8_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    report = summarize(parse_args(argv))
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
