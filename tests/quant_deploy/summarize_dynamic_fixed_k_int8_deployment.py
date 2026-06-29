from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from quant_deploy_utils import ensure_quant_deploy_run_dirs, read_json, save_json


MODE = "dynamic_agent_dim_fixed_k_plugin"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize dynamic fixed-K INT8 deployment results.")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--calibration_frames", type=int, nargs="+", default=[50, 200])
    return parser.parse_args(argv)


def _load_eval(dirs: dict[str, Path], precision: str, frames: int, calib: int | None = None) -> dict[str, Any]:
    if precision == "int8":
        return read_json(dirs["evaluation"] / f"trt_int8_ap_report_dynamic_fixed_k_plugin_calib{calib}_{frames}.json", default={}) or {}
    if precision == "int8_mixed_heads_fp16":
        return read_json(dirs["evaluation"] / f"trt_int8_ap_report_dynamic_fixed_k_plugin_mixed_heads_fp16_{frames}.json", default={}) or {}
    return read_json(dirs["evaluation"] / f"trt_{precision}_ap_report_{MODE}_{frames}.json", default={}) or {}


def _p50(value: Any) -> Any:
    return value.get("p50") if isinstance(value, dict) else None


def _row(report: dict[str, Any], *, mode: str, strategy: str, calib: Any, frames: int, fp16: dict[str, Any] | None = None) -> dict[str, Any]:
    fp16_map = float((fp16 or {}).get("mAP", (fp16 or {}).get("map", 0.0))) if fp16 else None
    this_map = float(report.get("mAP", report.get("map", 0.0))) if report else None
    forward = report.get("forward_ms") if isinstance(report.get("forward_ms"), dict) else {}
    execute = report.get("execute_ms") if isinstance(report.get("execute_ms"), dict) else {}
    return {
        "mode": mode,
        "precision strategy": strategy,
        "calibration frames": calib,
        "eval frames": frames,
        "AP@0.30": report.get("AP@0.30", report.get("ap_0_3")),
        "AP@0.50": report.get("AP@0.50", report.get("ap_0_5")),
        "AP@0.70": report.get("AP@0.70", report.get("ap_0_7")),
        "mAP": this_map,
        "mAP drop vs FP16": round(fp16_map - this_map, 4) if fp16_map is not None and this_map is not None else None,
        "execute p50": execute.get("p50") or report.get("execute_p50_ms"),
        "forward p50": forward.get("p50") or report.get("forward_p50_ms"),
        "FPS": report.get("FPS") or report.get("fps"),
        "notes": report.get("error") or ("per-N fixed-K bucket router" if report else "missing"),
        "success": report.get("success") if report else False,
    }


def _best_row(rows: list[dict[str, Any]], *, strategy: str, frames: int) -> dict[str, Any]:
    candidates = [
        row
        for row in rows
        if row.get("precision strategy") == strategy
        and int(row.get("eval frames") or -1) == int(frames)
        and row.get("success")
        and row.get("mAP") is not None
    ]
    return max(candidates, key=lambda row: float(row["mAP"])) if candidates else {}


def _largest_ap_drop(fp16_row: dict[str, Any], int8_row: dict[str, Any]) -> str | None:
    drops: dict[str, float] = {}
    for key in ("AP@0.30", "AP@0.50", "AP@0.70"):
        if fp16_row.get(key) is not None and int8_row.get(key) is not None:
            drops[key] = float(fp16_row[key]) - float(int8_row[key])
    if not drops:
        return None
    key, value = max(drops.items(), key=lambda item: item[1])
    return f"{key} drop={value:.4f}"


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    dirs = ensure_quant_deploy_run_dirs(args.output_root)
    rows = []
    for frames in (50, 200):
        fp16 = _load_eval(dirs, "fp16", frames)
        rows.append(_row(_load_eval(dirs, "fp32", frames), mode="dynamic fixed-K plugin", strategy="fp32", calib="-", frames=frames, fp16=fp16))
        rows.append(_row(fp16, mode="dynamic fixed-K plugin", strategy="fp16", calib="-", frames=frames, fp16=fp16))
        for calib in [int(v) for v in args.calibration_frames]:
            rows.append(_row(_load_eval(dirs, "int8", frames, calib), mode="dynamic fixed-K plugin", strategy="native int8", calib=calib, frames=frames, fp16=fp16))
        mixed = _load_eval(dirs, "int8_mixed_heads_fp16", frames)
        if mixed:
            rows.append(_row(mixed, mode="dynamic fixed-K plugin", strategy="int8 mixed heads-FP16", calib=mixed.get("calibration_frames"), frames=frames, fp16=fp16))

    build_reports = [read_json(dirs["benchmark"] / f"dynamic_fixed_k_int8_engine_build_calib{int(v)}.json", default={}) or {} for v in args.calibration_frames]
    mixed_build_report = read_json(dirs["benchmark"] / "dynamic_fixed_k_int8_mixed_precision_build_report.json", default={}) or {}
    precision_reports = [read_json(dirs["debug"] / f"int8_layer_precision_profile_calib{int(v)}.json", default={}) or {} for v in args.calibration_frames]
    mixed_precision_report = read_json(dirs["debug"] / "int8_layer_precision_profile_calib200_mixed_heads_fp16.json", default={}) or {}
    audit = read_json(dirs["debug"] / "pointpillar_scatter_int8_support_audit.json", default={}) or {}
    any_int8_success = any(row.get("success") for row in rows if row["precision strategy"] == "native int8")
    fp16_200 = next((row for row in rows if row["precision strategy"] == "fp16" and row["eval frames"] == 200), {})
    best_int8_200 = _best_row(rows, strategy="native int8", frames=200)
    mixed_200 = next((row for row in rows if row["precision strategy"] == "int8 mixed heads-FP16" and row["eval frames"] == 200 and row.get("success")), {})
    int8_latency_better = bool(best_int8_200 and fp16_200 and best_int8_200.get("forward p50") and fp16_200.get("forward p50") and float(best_int8_200["forward p50"]) < float(fp16_200["forward p50"]))
    int8_map_drop = best_int8_200.get("mAP drop vs FP16")
    native_ok = bool(any_int8_success and int8_latency_better and int8_map_drop is not None and float(int8_map_drop) <= 0.01)
    mixed_better = (
        bool(mixed_200 and best_int8_200)
        and mixed_200.get("mAP") is not None
        and best_int8_200.get("mAP") is not None
        and float(mixed_200["mAP"]) > float(best_int8_200["mAP"])
    )
    answers = {
        "INT8 engine 是否成功构建": any(report.get("any_build_success") for report in build_reports),
        "INT8 engine 是否真的运行了 INT8 层": any(((report.get("answers") or {}).get("int8_engine_really_uses_int8")) for report in precision_reports),
        "PointPillarScatterTRT 在 INT8 engine 中以什么精度执行": [report.get("PointPillarScatterTRT precision") for report in precision_reports if report],
        "INT8 相比 FP16 是否有 latency 收益": int8_latency_better,
        "INT8 相比 FP16 的 AP 损失是多少": int8_map_drop,
        "AP 损失主要发生在哪个 IoU": _largest_ap_drop(fp16_200, best_int8_200),
        "native INT8 是否可接受": native_ok,
        "mixed precision INT8 是否优于 native INT8": mixed_better if mixed_200 else ("not_run" if not any_int8_success else "not_evaluated"),
        "是否建议进入 Q/DQ / ModelOpt 进一步优化": bool(any_int8_success and not native_ok),
        "是否建议当前就把 INT8 纳入默认部署路径": native_ok,
        "当前默认部署路径是否仍为 dynamic_agent_dim + fixed-K bucket router + PointPillarScatterTRT": True,
        "padded_agent_static 是否仍只作为 baseline": True,
    }
    report = {
        "rows": rows,
        "answers": answers,
        "plugin_int8_support_audit": audit,
        "build_reports": build_reports,
        "mixed_build_report": mixed_build_report,
        "precision_reports": precision_reports,
        "mixed_precision_report": mixed_precision_report,
        "heal_opencood_source_modified": False,
        "default_path": "dynamic_agent_dim + fixed-K bucket router + PointPillarScatterTRT",
        "padded_agent_static_role": "optional baseline only",
    }
    save_json(report, dirs["summary"] / "dynamic_fixed_k_int8_deployment_report.json")
    headers = ["mode", "precision strategy", "calibration frames", "eval frames", "AP@0.30", "AP@0.50", "AP@0.70", "mAP", "mAP drop vs FP16", "execute p50", "forward p50", "FPS", "notes"]
    lines = [
        "# Dynamic Fixed-K INT8 Deployment Report",
        "",
        " | ".join(headers),
        " | ".join(["---"] * len(headers)),
    ]
    for row in rows:
        lines.append(" | ".join(str(row.get(h)) for h in headers))
    lines.extend(["", "## Answers", ""])
    for key, value in answers.items():
        lines.append(f"- {key}: {value}")
    (dirs["summary"] / "dynamic_fixed_k_int8_deployment_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    report = summarize(parse_args(argv))
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
