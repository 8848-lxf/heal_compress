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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose INT8 calibration split and AP drop for fixedK deployment experiments.")
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_DIR / "lidar_pyramid_agent_export_strategy_compare"))
    parser.add_argument("--fixed_k", type=int, required=True)
    parser.add_argument("--full_val_tag", default=None)
    return parser.parse_args(argv)


def _load(path: Path) -> dict[str, Any] | None:
    data = read_json(path, default=None)
    return data if isinstance(data, dict) else None


def _row(label: str, report: dict[str, Any] | None, *, calibration_split: str | None, historical: bool = False) -> dict[str, Any]:
    report = report or {}
    return {
        "label": label,
        "status": report.get("status", "missing_report") if report else "missing_report",
        "historical_reference": bool(historical),
        "calibration_split": calibration_split if calibration_split is not None else report.get("calibration_split"),
        "calibration_split_confirmed": bool(calibration_split == "train") if not historical else False,
        "possible_val_calibration_leakage": "unknown" if historical else False,
        "evaluation_split": report.get("evaluation_split"),
        "total_val_samples": report.get("total_val_samples"),
        "evaluated_samples": report.get("evaluated_samples"),
        "skipped_samples": len(report.get("skipped_samples") or []) if report else None,
        "AP@0.30": report.get("AP@0.30"),
        "AP@0.50": report.get("AP@0.50"),
        "AP@0.70": report.get("AP@0.70"),
        "mAP": report.get("mAP"),
        "forward_p50": (report.get("forward_ms") or {}).get("p50"),
        "reliable_latency": not bool(report.get("unreliable_latency")) if report else None,
        "report_path": report.get("_path"),
    }


def _with_path(path: Path) -> dict[str, Any] | None:
    data = _load(path)
    if data is not None:
        data["_path"] = str(path)
    return data


def _delta(a: dict[str, Any] | None, b: dict[str, Any] | None, key: str = "mAP") -> float | None:
    if not a or not b or a.get(key) is None or b.get(key) is None:
        return None
    return round(float(a[key]) - float(b[key]), 4)


def run(args: argparse.Namespace) -> dict[str, Any]:
    dirs = ensure_quant_deploy_run_dirs(args.output_root)
    tag = args.full_val_tag or f"full_val_fixedK{int(args.fixed_k)}_trainCalib"
    eval_dir = dirs["evaluation"] / tag
    old_eval_dir = dirs["evaluation"] / "full_val_idle_gpu"

    old_dynamic_calib200 = _with_path(old_eval_dir / "dynamic_bucket_int8_calib200_full_val.json")
    new_dynamic_calib200 = _with_path(eval_dir / "dynamic_bucket_int8_calib200_full_val.json")
    new_single_calib200 = _with_path(eval_dir / "single_engine_maxK_int8_train_calib200_full_val.json")
    dynamic_fp16 = _with_path(eval_dir / "dynamic_bucket_fp16_full_val.json")
    single_fp16 = _with_path(eval_dir / "single_engine_maxK_fp16_full_val.json")

    comparison_rows = [
        _row("historical_dynamic_bucket_int8_calib200", old_dynamic_calib200, calibration_split=None, historical=True),
        _row("new_dynamic_bucket_int8_train_calib200", new_dynamic_calib200, calibration_split="train"),
        _row("new_single_engine_maxK_int8_train_calib200", new_single_calib200, calibration_split="train"),
        _row("new_dynamic_bucket_fp16_reference", dynamic_fp16, calibration_split=None),
        _row("new_single_engine_fp16_reference", single_fp16, calibration_split=None),
    ]

    dynamic_train_drop_vs_fp16 = None
    single_train_drop_vs_fp16 = None
    if dynamic_fp16 and new_dynamic_calib200:
        dynamic_train_drop_vs_fp16 = _delta(dynamic_fp16, new_dynamic_calib200)
    if single_fp16 and new_single_calib200:
        single_train_drop_vs_fp16 = _delta(single_fp16, new_single_calib200)

    old_vs_new_dynamic = _delta(old_dynamic_calib200, new_dynamic_calib200)
    single_vs_dynamic_train = _delta(new_dynamic_calib200, new_single_calib200)

    suspected_root_causes: list[dict[str, Any]] = []
    if old_vs_new_dynamic is not None and old_vs_new_dynamic > 0.02:
        suspected_root_causes.append(
            {
                "rank": len(suspected_root_causes) + 1,
                "cause": "historical calibration split or sample-selection mismatch",
                "evidence": f"historical dynamic bucket INT8 mAP exceeds new train-calib dynamic bucket by {old_vs_new_dynamic}",
            }
        )
    if single_vs_dynamic_train is not None and single_vs_dynamic_train > 0.02:
        suspected_root_causes.append(
            {
                "rank": len(suspected_root_causes) + 1,
                "cause": "single-cache wider N/K/padding distribution",
                "evidence": f"new train-calib dynamic bucket mAP exceeds new train-calib single engine by {single_vs_dynamic_train}",
            }
        )
    if single_train_drop_vs_fp16 is not None and single_train_drop_vs_fp16 > 0.1:
        suspected_root_causes.append(
            {
                "rank": len(suspected_root_causes) + 1,
                "cause": "activation quantization too aggressive for single-engine maxK",
                "evidence": f"single-engine train-calib INT8 mAP drop vs FP16 is {single_train_drop_vs_fp16}",
            }
        )
    if not suspected_root_causes:
        suspected_root_causes.append(
            {
                "rank": 1,
                "cause": "insufficient completed full-val reports",
                "evidence": "Run fixedK train-calib full-val evaluation to complete diagnosis.",
            }
        )

    report = {
        "fixed_K": int(args.fixed_k),
        "old_dynamic_bucket_calibration_split_confirmed": False,
        "new_dynamic_bucket_calibration_split": "train",
        "new_single_engine_calibration_split": "train",
        "val_calib_diagnostic_executed": False,
        "val_calib_diagnostic_note": "Not executed; formal metrics must use train calibration only.",
        "historical_calibration_split_confirmed": False,
        "possible_val_calibration_leakage": "unknown",
        "AP_mAP_comparison_table": comparison_rows,
        "dynamic_bucket_train_calib200_mAP_drop_vs_fp16": dynamic_train_drop_vs_fp16,
        "single_engine_train_calib200_mAP_drop_vs_fp16": single_train_drop_vs_fp16,
        "historical_dynamic_vs_new_train_dynamic_mAP_delta": old_vs_new_dynamic,
        "new_train_dynamic_vs_new_train_single_mAP_delta": single_vs_dynamic_train,
        "activation_layer_precision_summary": "Use dynamic/single INT8 layer precision audit reports if available; this diagnosis is report-level.",
        "suspected_root_causes_ranked": suspected_root_causes,
        "recommended_fix": [
            "Treat old INT8 results as historical only unless calibration_split is proven train.",
            "Prefer train-calib fixedK full-val rows for formal INT8 comparison.",
            "If single-engine INT8 drop remains >0.1, try mixed precision white-listing before full Q/DQ ModelOpt.",
            "Protect PFN/PillarVFE, regression head, direction head, classification head, early BEV backbone, then shrink/fusion in that order.",
        ],
    }
    save_json(report, dirs["debug"] / "int8_calibration_split_diagnosis.json")
    lines = [
        "# INT8 Calibration Split Diagnosis",
        "",
        f"- fixed_K: {int(args.fixed_k)}",
        "- old_dynamic_bucket_calibration_split_confirmed: false",
        "- new_dynamic_bucket_calibration_split: train",
        "- new_single_engine_calibration_split: train",
        "- val_calib_diagnostic_executed: false",
        "",
        "label | calibration_split | status | AP@0.70 | mAP | forward_p50 | note",
        "--- | --- | --- | --- | --- | --- | ---",
    ]
    for row in comparison_rows:
        note = "historical only; split unconfirmed" if row["historical_reference"] else ""
        lines.append(
            f"{row['label']} | {row.get('calibration_split')} | {row.get('status')} | "
            f"{row.get('AP@0.70')} | {row.get('mAP')} | {row.get('forward_p50')} | {note}"
        )
    lines.extend(["", "## Suspected Root Causes", ""])
    for cause in suspected_root_causes:
        lines.append(f"- {cause['rank']}. {cause['cause']}: {cause['evidence']}")
    (dirs["summary"] / "int8_calibration_split_diagnosis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    report = run(parse_args(argv))
    print(json.dumps({"fixed_K": report["fixed_K"], "diagnosis_rows": len(report["AP_mAP_comparison_table"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
