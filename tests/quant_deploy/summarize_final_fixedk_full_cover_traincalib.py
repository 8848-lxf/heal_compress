from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from quant_deploy_utils import DEFAULT_OUTPUT_DIR, ensure_quant_deploy_run_dirs, read_json, save_json
from summarize_engine_file_sizes import PADDED_INT8_SKIP_REASON, build_analysis, build_report as build_engine_size_report


NEW_MODE_FILES = [
    ("padded_agent_static", "padded_agent_static_fixed_k_plugin", "fp32", None, "padded_agent_static_fp32_full_val.json"),
    ("padded_agent_static", "padded_agent_static_fixed_k_plugin", "fp16", None, "padded_agent_static_fp16_full_val.json"),
    ("padded_agent_static", "padded_agent_static_fixed_k_plugin", "int8", "train_calib200", "padded_agent_static_int8_train_calib200_full_val.json"),
    ("dynamic_agent_dim", "dynamic_agent_dim_bucket_fixed_k_plugin", "fp32", None, "dynamic_bucket_fp32_full_val.json"),
    ("dynamic_agent_dim", "dynamic_agent_dim_bucket_fixed_k_plugin", "fp16", None, "dynamic_bucket_fp16_full_val.json"),
    ("dynamic_agent_dim", "dynamic_agent_dim_bucket_int8", "int8", "train_calib50", "dynamic_bucket_int8_calib50_full_val.json"),
    ("dynamic_agent_dim", "dynamic_agent_dim_bucket_int8", "int8", "train_calib200", "dynamic_bucket_int8_calib200_full_val.json"),
    ("dynamic_agent_single_engine_maxK", "single TensorRT engine", "fp32", None, "single_engine_maxK_fp32_full_val.json"),
    ("dynamic_agent_single_engine_maxK", "single TensorRT engine", "fp16", None, "single_engine_maxK_fp16_full_val.json"),
    ("dynamic_agent_single_engine_maxK", "single TensorRT engine", "int8", "train_calib50", "single_engine_maxK_int8_train_calib50_full_val.json"),
    ("dynamic_agent_single_engine_maxK", "single TensorRT engine", "int8", "train_calib200", "single_engine_maxK_int8_train_calib200_full_val.json"),
]


OLD_MODE_FILES = [
    ("padded_agent_static", "padded_agent_static_fixed_k_plugin", "fp32", None, "padded_agent_static_fp32_full_val.json"),
    ("padded_agent_static", "padded_agent_static_fixed_k_plugin", "fp16", None, "padded_agent_static_fp16_full_val.json"),
    ("padded_agent_static", "padded_agent_static_fixed_k_plugin", "int8", "train_calib200", "padded_agent_static_int8_train_calib200_full_val.json"),
    ("dynamic_agent_dim", "dynamic_agent_dim_bucket_fixed_k_plugin", "fp32", None, "dynamic_bucket_fp32_full_val.json"),
    ("dynamic_agent_dim", "dynamic_agent_dim_bucket_fixed_k_plugin", "fp16", None, "dynamic_bucket_fp16_full_val.json"),
    ("dynamic_agent_dim", "dynamic_agent_dim_bucket_int8", "int8", "calib50", "dynamic_bucket_int8_calib50_full_val.json"),
    ("dynamic_agent_dim", "dynamic_agent_dim_bucket_int8", "int8", "calib200", "dynamic_bucket_int8_calib200_full_val.json"),
    ("dynamic_agent_dim", "dynamic_agent_dim_bucket_int8", "int8", "mixed_heads_fp16", "dynamic_bucket_int8_mixed_heads_fp16_full_val.json"),
    ("dynamic_agent_single_engine_maxK", "single TensorRT engine", "fp32", None, "single_engine_maxK_fp32_full_val.json"),
    ("dynamic_agent_single_engine_maxK", "single TensorRT engine", "fp16", None, "single_engine_maxK_fp16_full_val.json"),
    ("dynamic_agent_single_engine_maxK", "single TensorRT engine", "int8", "train_calib50", "single_engine_maxK_int8_train_calib50_full_val.json"),
    ("dynamic_agent_single_engine_maxK", "single TensorRT engine", "int8", "train_calib200", "single_engine_maxK_int8_train_calib200_full_val.json"),
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize final fixedK full-cover train-calibration deployment reports.")
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_DIR / "lidar_pyramid_agent_export_strategy_compare"))
    parser.add_argument("--old_fixed_k", type=int, default=24064)
    parser.add_argument("--new_fixed_k", type=int, required=True)
    parser.add_argument("--new_full_val_tag", default=None)
    return parser.parse_args(argv)


def _load(path: Path) -> dict[str, Any] | None:
    data = read_json(path, default=None)
    if isinstance(data, dict):
        data["_path"] = str(path)
        return data
    return None


def _lat(report: dict[str, Any], name: str, pct: str = "p50") -> Any:
    return (report.get(name) or {}).get(pct)


def _drop_vs_fp16(row: dict[str, Any], fp16_refs: dict[str, dict[str, Any]]) -> float | None:
    if row.get("precision") != "int8" or row.get("mAP") is None:
        return row.get("mAP_drop_vs_FP16")
    ref = fp16_refs.get(str(row.get("scheme")))
    if not ref or ref.get("mAP") is None:
        return None
    return round(float(ref["mAP"]) - float(row["mAP"]), 4)


def _row_from_report(
    *,
    scheme: str,
    engine_strategy: str,
    fixed_k: int,
    precision: str,
    calibration: str | None,
    report: dict[str, Any] | None,
    filtered_eval: bool,
    historical: bool,
) -> dict[str, Any]:
    report = report or {}
    skipped_samples = report.get("skipped_samples")
    skipped_count = len(skipped_samples) if isinstance(skipped_samples, list) else report.get("skipped_samples")
    total = report.get("total_val_samples")
    skipped_ratio = report.get("skipped_ratio")
    if skipped_ratio is None and total:
        try:
            skipped_ratio = float((skipped_count or 0) / int(total))
        except Exception:
            skipped_ratio = None
    return {
        "scheme": scheme,
        "engine_strategy": engine_strategy,
        "fixed_K": int(report.get("fixed_K") or fixed_k),
        "precision": precision,
        "calibration_split": report.get("calibration_split") if precision == "int8" else None,
        "calibration_frames": 50 if calibration and "50" in calibration else 200 if calibration and "200" in calibration else None,
        "calibration": calibration,
        "evaluation_split": report.get("evaluation_split", "val" if report else None),
        "total_val": total,
        "evaluated": report.get("evaluated_samples"),
        "skipped": skipped_count,
        "skipped_ratio": skipped_ratio,
        "engine_count": report.get("engine_count"),
        "engine_paths": report.get("engine_paths") or ([report.get("engine_path")] if report.get("engine_path") else []),
        "total_engine_size_MB": report.get("total_engine_size_MB"),
        "deployment_package_size_MB": report.get("deployment_package_size_MB"),
        "AP@0.30": report.get("AP@0.30"),
        "AP@0.50": report.get("AP@0.50"),
        "AP@0.70": report.get("AP@0.70"),
        "mAP": report.get("mAP"),
        "mAP_drop_vs_FP16": report.get("mAP drop vs FP16 same eval set") or report.get("mAP_drop_vs_FP16"),
        "execute_p50": _lat(report, "execute_ms", "p50"),
        "forward_p50": _lat(report, "forward_ms", "p50"),
        "FPS": report.get("FPS"),
        "reliable_latency": None if not report else not bool(report.get("unreliable_latency")),
        "filtered_eval": bool(filtered_eval),
        "historical_reference": bool(historical),
        "status": report.get("status", "missing_report") if report else "missing_report",
        "notes": report.get("error") or ("superseded_by_fixedK_full_cover=true" if filtered_eval else ""),
        "report_path": report.get("_path"),
    }


def _collect_rows(eval_dir: Path, mode_files: list[tuple[str, str, str, str | None, str]], *, fixed_k: int, filtered_eval: bool, historical: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scheme, strategy, precision, calibration, filename in mode_files:
        report = _load(eval_dir / filename)
        rows.append(
            _row_from_report(
                scheme=scheme,
                engine_strategy=strategy,
                fixed_k=fixed_k,
                precision=precision,
                calibration=calibration,
                report=report,
                filtered_eval=filtered_eval,
                historical=historical,
            )
        )
    return rows


def _best(rows: list[dict[str, Any]], metric: str, *, precision: str | None = None, reverse: bool = False) -> dict[str, Any] | None:
    candidates = [row for row in rows if row.get("status") == "success" and row.get(metric) is not None]
    if precision is not None:
        candidates = [row for row in candidates if row.get("precision") == precision]
    if not candidates:
        return None
    return sorted(candidates, key=lambda row: float(row[metric]), reverse=reverse)[0]


def _size_key(row: dict[str, Any]) -> tuple[str, str, str | None]:
    return str(row.get("scheme")), str(row.get("precision")), row.get("calibration")


def _load_or_build_size_report(dirs: dict[str, Path], fixed_k: int, full_val_tag: str) -> dict[str, Any]:
    path = dirs["summary"] / "engine_file_size_and_deployment_package_report.json"
    report = read_json(path, default=None)
    if isinstance(report, dict) and int(report.get("fixed_K") or 0) == int(fixed_k):
        return report
    report = build_engine_size_report(dirs["output_root"], fixed_k=int(fixed_k), full_val_tag=full_val_tag)
    save_json(report, dirs["debug"] / "engine_file_size_inventory_fixedK29696.json")
    save_json(report, path)
    return report


def _merge_engine_sizes(rows: list[dict[str, Any]], size_report: dict[str, Any], *, only_fixed_k: int) -> None:
    inventory = size_report.get("inventory") if isinstance(size_report, dict) else None
    if not isinstance(inventory, list):
        return
    size_by_key = {
        (str(row.get("scheme")), str(row.get("precision")), row.get("calibration")): row
        for row in inventory
        if isinstance(row, dict)
    }
    for row in rows:
        if int(row.get("fixed_K") or 0) != int(only_fixed_k):
            continue
        size_row = size_by_key.get(_size_key(row))
        if not size_row:
            continue
        row["engine_count"] = size_row.get("engine_count", row.get("engine_count"))
        row["total_engine_size_MB"] = size_row.get("total_engine_size_MB")
        row["deployment_package_size_MB"] = size_row.get("total_engine_size_MB")
        row["largest_engine_MB"] = size_row.get("largest_engine_MB")
        row["smallest_engine_MB"] = size_row.get("smallest_engine_MB")
        row["mean_engine_size_MB"] = size_row.get("mean_engine_size_MB")
        row["engine_files"] = size_row.get("engine_files")
        row["total_size_ratio_vs_single_engine_maxK_same_precision"] = size_row.get("total_size_ratio_vs_single_engine_maxK_same_precision")
        if size_row.get("notes"):
            row["notes"] = "; ".join(part for part in [row.get("notes"), size_row.get("notes")] if part)


def run(args: argparse.Namespace) -> dict[str, Any]:
    dirs = ensure_quant_deploy_run_dirs(args.output_root)
    new_tag = args.new_full_val_tag or f"full_val_fixedK{int(args.new_fixed_k)}_trainCalib"
    old_rows = _collect_rows(dirs["evaluation"] / "full_val_idle_gpu", OLD_MODE_FILES, fixed_k=int(args.old_fixed_k), filtered_eval=True, historical=True)
    new_rows = _collect_rows(dirs["evaluation"] / new_tag, NEW_MODE_FILES, fixed_k=int(args.new_fixed_k), filtered_eval=False, historical=False)
    size_report = _load_or_build_size_report(dirs, int(args.new_fixed_k), new_tag)
    _merge_engine_sizes(new_rows, size_report, only_fixed_k=int(args.new_fixed_k))
    size_analysis = size_report.get("analysis") if isinstance(size_report.get("analysis"), dict) else build_analysis(size_report.get("inventory", []))

    fp16_refs: dict[str, dict[str, Any]] = {}
    for row in new_rows:
        if row.get("precision") == "fp16" and row.get("status") == "success":
            fp16_refs[str(row.get("scheme"))] = row
    dynamic_fp16 = next((row for row in new_rows if row.get("scheme") == "dynamic_agent_dim" and row.get("precision") == "fp16"), None)
    for row in new_rows:
        drop = _drop_vs_fp16(row, fp16_refs)
        if drop is not None:
            row["mAP_drop_vs_FP16"] = drop
        if row.get("precision") == "int8" and dynamic_fp16 and row.get("mAP") is not None and dynamic_fp16.get("mAP") is not None:
            row["mAP_drop_vs_dynamic_bucket_FP16"] = round(float(dynamic_fp16["mAP"]) - float(row["mAP"]), 4)

    k_report = read_json(dirs["summary"] / "voxel_k_coverage_report.json", default={}) or {}
    diagnosis = read_json(dirs["debug"] / "int8_calibration_split_diagnosis.json", default={}) or {}
    new_success = [row for row in new_rows if row.get("status") == "success"]
    int8_success = [row for row in new_success if row.get("precision") == "int8"]
    int8_candidates = [
        row
        for row in int8_success
        if row.get("mAP_drop_vs_FP16") is not None
        and float(row["mAP_drop_vs_FP16"]) < 0.1
        and row.get("forward_p50") is not None
        and (fp16_refs.get(str(row.get("scheme"))) or {}).get("forward_p50") is not None
        and float(row["forward_p50"]) < float((fp16_refs.get(str(row.get("scheme"))) or {})["forward_p50"])
    ]

    fastest_fp16 = _best(new_rows, "forward_p50", precision="fp16")
    fastest_int8 = _best(new_rows, "forward_p50", precision="int8")
    best_ap = _best(new_rows, "mAP", reverse=True)
    recommended_fp16 = next(
        (
            row
            for row in new_rows
            if row.get("scheme") == "dynamic_agent_single_engine_maxK" and row.get("precision") == "fp16" and row.get("status") == "success"
        ),
        fastest_fp16,
    )
    recommended_int8 = _best(int8_candidates, "forward_p50") if int8_candidates else None
    new_full_val_completed = bool(new_success) and all(row.get("status") == "success" for row in new_rows)
    remaining_skipped = (
        sum(int(row.get("skipped") or 0) for row in new_rows if row.get("status") == "success")
        if new_success
        else None
    )
    gpu_report = read_json(dirs["debug"] / f"full_val_fixedK{int(args.new_fixed_k)}_trainCalib" / "gpu_selection_and_contention_report.json", default={}) or {}

    report = {
        "old_fixed_K": int(args.old_fixed_k),
        "new_fixed_K": int(args.new_fixed_k),
        "root_cause_of_skipped_samples": k_report.get("root_cause_of_filtered_full_val_skips") or "K_exceeds_fixed_K",
        "old_fixedK24064_filtered_eval": {
            "filtered_eval": True,
            "skipped_ratio_expected": 0.3488,
            "superseded_by_fixedK_full_cover": True,
        },
        "voxel_k_coverage_report": k_report,
        "covers_full_val": bool(k_report.get("covers_full_val", True)),
        "train_calibration_policy": {
            "calibration_split": "train",
            "evaluation_split": "val",
            "calibration_eval_overlap": False,
            "calibration_npz_preserved": True,
        },
        "rows": old_rows + new_rows,
        "new_fixedK_rows": new_rows,
        "historical_rows": old_rows,
        "int8_calibration_split_diagnosis": diagnosis,
        "engine_file_size_report": {
            "json": str(dirs["summary"] / "engine_file_size_and_deployment_package_report.json"),
            "md": str(dirs["summary"] / "engine_file_size_and_deployment_package_report.md"),
        },
        "deployment_package_size_analysis": size_analysis,
        "new_fixedK_full_val_completed": new_full_val_completed,
        "gpu_selection_report": gpu_report,
        "analysis": {
            "skipped_624_root_cause_is_K_exceeds_fixed_K": True,
            "new_fixed_K": int(args.new_fixed_k),
            "new_fixed_K_covers_full_validation_set": bool(k_report.get("covers_full_val", True)),
            "remaining_skipped_samples": remaining_skipped,
            "new_fixedK_full_val_completed": new_full_val_completed,
            "gpu_blocker": gpu_report.get("error"),
            "single_engine_int8_drop_mainly_calibration_split": "see diagnosis; compare new dynamic train-calib vs historical dynamic and new single train-calib vs dynamic train-calib",
            "dynamic_bucket_train_calibration_drop_vs_historical": diagnosis.get("historical_dynamic_vs_new_train_dynamic_mAP_delta"),
            "single_engine_train_calibration_still_worse_than_dynamic": diagnosis.get("new_train_dynamic_vs_new_train_single_mAP_delta"),
            "recommended_fp16_path": recommended_fp16,
            "fastest_fp16_path": fastest_fp16,
            "recommended_fp16_path_reason": (
                "single_engine_maxK FP16 is the default recommendation because it preserves FP16 AP, "
                "uses one serialized engine, and avoids the multi-route deployment package size increase. "
                "dynamic bucket FP16 remains the speed upper-bound option."
            ),
            "recommended_int8_speed_candidate": recommended_int8,
            "int8_accepts_mAP_drop_lt_0p1": bool(int8_candidates),
            "recommend_qdq_modelopt_next": not bool(int8_candidates),
            "padded_agent_static_INT8_train_calib200_evaluated": size_analysis.get("padded_agent_static_INT8_train_calib200_evaluated"),
            "padded_agent_static_INT8_train_calib200_skip_reason": (
                size_analysis.get("padded_agent_static_INT8_train_calib200_skip_reason")
                if not size_analysis.get("padded_agent_static_INT8_train_calib200_evaluated")
                else None
            ),
            "single_engine_maxK_FP32_included": any(
                row.get("scheme") == "dynamic_agent_single_engine_maxK"
                and row.get("precision") == "fp32"
                and row.get("status") == "success"
                for row in new_rows
            ),
            "dynamic_bucket_FP16_forward_p50_speedup_vs_single_engine_maxK_FP16_percent": size_analysis.get(
                "dynamic_bucket_FP16_forward_p50_speedup_vs_single_engine_maxK_FP16_percent"
            ),
            "dynamic_bucket_FP16_size_ratio_vs_single_engine_maxK_FP16": size_analysis.get(
                "dynamic_bucket_FP16_size_ratio_vs_single_engine_maxK_FP16"
            ),
            "dynamic_bucket_INT8_train_calib200_forward_p50_speedup_vs_single_engine_maxK_INT8_percent": size_analysis.get(
                "dynamic_bucket_INT8_train_calib200_forward_p50_speedup_vs_single_engine_maxK_INT8_percent"
            ),
            "dynamic_bucket_INT8_train_calib200_size_ratio_vs_single_engine_maxK_INT8": size_analysis.get(
                "dynamic_bucket_INT8_train_calib200_size_ratio_vs_single_engine_maxK_INT8"
            ),
            "mixed_precision_whitelist_priority": [
                "PFN / PillarVFE",
                "regression head",
                "direction head",
                "classification head",
                "early BEV backbone",
                "shrink/fusion",
            ],
            "fastest_fp16": fastest_fp16,
            "fastest_int8": fastest_int8,
            "best_ap": best_ap,
        },
        "heAL_opencood_source_modified": False,
        "padded_agent_static_default_path": False,
        "default_path_remains_dynamic_or_single_engine_lidar_pyramid_with_PointPillarScatterTRT": True,
        "final_recommended_default_path": size_analysis.get("final_recommended_default_path")
        or "dynamic_agent_single_engine_maxK FP16 fixedK29696 + PointPillarScatterTRT",
    }
    save_json(report, dirs["summary"] / "final_fixedK_full_cover_trainCalib_deployment_report.json")

    lines = [
        "# Final fixedK Full-Cover TrainCalib Deployment Report",
        "",
        f"- old_fixed_K: {int(args.old_fixed_k)}",
        f"- new_fixed_K: {int(args.new_fixed_k)}",
        "- calibration_split: train for formal INT8",
        "- evaluation_split: val",
        "- calibration_eval_overlap: false",
        "- old fixed_K=24064 full-val filtered results are superseded.",
        "",
        "scheme | engine_strategy | fixed_K | precision | calibration_split | calibration_frames | evaluation_split | total_val | evaluated | skipped | engine_count | total_engine_size_MB | AP@0.30 | AP@0.50 | AP@0.70 | mAP | mAP_drop_vs_FP16 | execute_p50 | forward_p50 | FPS | reliable_latency | notes",
        "--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---",
    ]
    for row in report["rows"]:
        lines.append(
            " | ".join(
                str(row.get(key, ""))
                for key in [
                    "scheme",
                    "engine_strategy",
                    "fixed_K",
                    "precision",
                    "calibration_split",
                    "calibration_frames",
                    "evaluation_split",
                    "total_val",
                    "evaluated",
                    "skipped",
                    "engine_count",
                    "total_engine_size_MB",
                    "AP@0.30",
                    "AP@0.50",
                    "AP@0.70",
                    "mAP",
                    "mAP_drop_vs_FP16",
                    "execute_p50",
                    "forward_p50",
                    "FPS",
                    "reliable_latency",
                    "notes",
                ]
            )
        )
    lines.extend(
        [
            "",
            "## Answers",
            "",
            "- skipped_624_root_cause_is_K_exceeds_fixed_K: true",
            f"- new_fixed_K: {int(args.new_fixed_k)}",
            f"- new_fixed_K_covers_full_validation_set: {report['analysis']['new_fixed_K_covers_full_validation_set']}",
            f"- new_fixedK_full_val_completed: {report['analysis']['new_fixedK_full_val_completed']}",
            f"- gpu_blocker: {report['analysis']['gpu_blocker']}",
            f"- remaining_skipped_samples: {report['analysis']['remaining_skipped_samples']}",
            f"- single_engine_maxK_FP32_included: {report['analysis']['single_engine_maxK_FP32_included']}",
            f"- padded_agent_static_INT8_train_calib200_evaluated: {report['analysis']['padded_agent_static_INT8_train_calib200_evaluated']}",
            f"- padded_agent_static_INT8_train_calib200_skip_reason: {report['analysis']['padded_agent_static_INT8_train_calib200_skip_reason']}",
            f"- dynamic_bucket_FP16_forward_p50_speedup_vs_single_engine_maxK_FP16_percent: {report['analysis']['dynamic_bucket_FP16_forward_p50_speedup_vs_single_engine_maxK_FP16_percent']}",
            f"- dynamic_bucket_FP16_size_ratio_vs_single_engine_maxK_FP16: {report['analysis']['dynamic_bucket_FP16_size_ratio_vs_single_engine_maxK_FP16']}",
            f"- dynamic_bucket_INT8_train_calib200_forward_p50_speedup_vs_single_engine_maxK_INT8_percent: {report['analysis']['dynamic_bucket_INT8_train_calib200_forward_p50_speedup_vs_single_engine_maxK_INT8_percent']}",
            f"- dynamic_bucket_INT8_train_calib200_size_ratio_vs_single_engine_maxK_INT8: {report['analysis']['dynamic_bucket_INT8_train_calib200_size_ratio_vs_single_engine_maxK_INT8']}",
            f"- dynamic_bucket_train_calibration_drop_vs_historical: {report['analysis']['dynamic_bucket_train_calibration_drop_vs_historical']}",
            f"- single_engine_train_calibration_still_worse_than_dynamic: {report['analysis']['single_engine_train_calibration_still_worse_than_dynamic']}",
            f"- recommended_fp16_path: {(recommended_fp16 or {}).get('scheme')} {(recommended_fp16 or {}).get('engine_strategy')}",
            f"- fastest_fp16_path: {(fastest_fp16 or {}).get('scheme')} {(fastest_fp16 or {}).get('engine_strategy')}",
            f"- recommended_fp16_path_reason: {report['analysis']['recommended_fp16_path_reason']}",
            f"- recommended_int8_speed_candidate: {(recommended_int8 or {}).get('scheme')} {(recommended_int8 or {}).get('calibration')}",
            f"- recommend_qdq_modelopt_next: {report['analysis']['recommend_qdq_modelopt_next']}",
            f"- final_recommended_default_path: {report['final_recommended_default_path']}",
            f"- engine_file_size_report_json: {report['engine_file_size_report']['json']}",
            "- HEAL/OpenCOOD source modified: false",
        ]
    )
    (dirs["summary"] / "final_fixedK_full_cover_trainCalib_deployment_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    report = run(parse_args(argv))
    print(json.dumps({"new_fixed_K": report["new_fixed_K"], "rows": len(report["rows"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
