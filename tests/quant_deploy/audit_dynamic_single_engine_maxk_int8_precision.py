from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent_mode_fixed_k_plugin_ablation import _layer_name, _layer_time_ms, _layer_type, _load_profile_layers
from dynamic_single_engine_maxk_common import layerinfo_path
from quant_deploy_utils import ensure_quant_deploy_run_dirs, save_json


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit dynamic single-engine maxK INT8 layer precision.")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--calibration_frames", type=int, nargs="+", default=[50, 200])
    return parser.parse_args(argv)


def _precision_of(row: dict[str, Any]) -> str:
    text = json.dumps(row, ensure_ascii=False).lower()
    if "int8" in text:
        return "int8"
    if "fp16" in text or "half" in text:
        return "fp16"
    if "fp32" in text or "float" in text:
        return "fp32"
    return "unknown"


def _category(name: str, layer_type: str) -> str:
    low = f"{name} {layer_type}".lower()
    if any(token in low for token in ("reformat", "cast", "quantize", "dequantize")):
        return "reformat_cast_qdq"
    if layer_type.lower() == "pluginv2" or "pointpillar" in low:
        return "plugin"
    if "pfn" in low or "pillar_vfe" in low:
        return "PillarVFE/PFN"
    if "shrink" in low:
        return "shrink_conv"
    if "cls" in low or "reg" in low or "dir" in low or "head" in low:
        return "detection_head"
    if "matmul" in low or "gemm" in low:
        return "gemm_matmul"
    if "conv" in layer_type.lower() or "convolution" in low or "backbone" in low:
        return "BEV_backbone"
    return "other"


def _tensor_dtypes(row: dict[str, Any], key: str) -> list[str]:
    values = []
    raw = row.get("raw")
    if not isinstance(raw, dict):
        return values
    for tensor in raw.get(key, []) or []:
        if isinstance(tensor, dict):
            dtype = str(tensor.get("Format/Datatype", "")).strip()
            if dtype:
                values.append(dtype)
    return values


def _count_layers(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {
        "INT8 Conv count": 0,
        "FP16 Conv count": 0,
        "FP32 Conv count": 0,
        "INT8 MatMul/GEMM count": 0,
        "FP16/FP32 fallback layers": 0,
        "Reformat/Cast/Q/DQ count": 0,
        "category_precision": {},
    }
    for row in rows:
        name = str(row.get("name", ""))
        layer_type = str(row.get("layer_type", ""))
        precision = str(row.get("precision", "unknown"))
        category = str(row.get("category", "other"))
        counts["category_precision"].setdefault(category, {})
        counts["category_precision"][category][precision] = counts["category_precision"][category].get(precision, 0) + 1
        low = f"{name} {layer_type}".lower()
        if "conv" in low or "convolution" in low:
            if precision == "int8":
                counts["INT8 Conv count"] += 1
            elif precision == "fp16":
                counts["FP16 Conv count"] += 1
            elif precision == "fp32":
                counts["FP32 Conv count"] += 1
        if ("matmul" in low or "gemm" in low) and precision == "int8":
            counts["INT8 MatMul/GEMM count"] += 1
        if precision in {"fp16", "fp32"}:
            counts["FP16/FP32 fallback layers"] += 1
        if any(token in low for token in ("reformat", "cast", "quantize", "dequantize")):
            counts["Reformat/Cast/Q/DQ count"] += 1
    return counts


def audit_one(dirs: dict[str, Path], calibration_frames: int) -> dict[str, Any]:
    path = layerinfo_path(dirs, "int8", int(calibration_frames))
    layers = _load_profile_layers(path)
    rows = []
    for item in layers:
        name = _layer_name(item)
        layer_type = _layer_type(item)
        rows.append(
            {
                "name": name,
                "layer_type": layer_type,
                "precision": _precision_of(item),
                "category": _category(name, layer_type),
                "latency_ms": _layer_time_ms(item),
                "raw": item,
            }
        )
    counts = _count_layers(rows)
    top20 = sorted(rows, key=lambda row: float(row.get("latency_ms") or 0.0), reverse=True)[:20]
    plugin_rows = [row for row in rows if row["category"] == "plugin"]
    report = {
        "strategy": "dynamic_agent_single_engine_maxK",
        "calibration_split": "train",
        "calibration_frames": int(calibration_frames),
        "layerinfo_path": str(path),
        "single_engine": True,
        "engine_count": 1,
        "counts": counts,
        "PointPillarScatterTRT precision": sorted({row["precision"] for row in plugin_rows}) or ["unknown"],
        "PointPillarScatterTRT io_dtypes": [
            {
                "name": row["name"],
                "input_dtypes": _tensor_dtypes(row, "Inputs"),
                "output_dtypes": _tensor_dtypes(row, "Outputs"),
            }
            for row in plugin_rows
        ],
        "PillarVFE/PFN precision": counts["category_precision"].get("PillarVFE/PFN", {}),
        "BEV backbone precision": counts["category_precision"].get("BEV_backbone", {}),
        "shrink_conv precision": counts["category_precision"].get("shrink_conv", {}),
        "detection head precision": counts["category_precision"].get("detection_head", {}),
        "plugin precision": counts["category_precision"].get("plugin", {}),
        "top 20 latency layers": top20,
        "answers": {
            "just_renamed_fp16_engine": counts["INT8 Conv count"] == 0 and counts["INT8 MatMul/GEMM count"] == 0,
            "really_runs_many_int8_layers": counts["INT8 Conv count"] > 0 or counts["INT8 MatMul/GEMM count"] > 0,
            "PointPillarScatterTRT_causes_large_fallback": "unknown" if not plugin_rows else counts["FP16/FP32 fallback layers"] > counts["INT8 Conv count"],
            "int8_speedup_space": counts["INT8 Conv count"] > 0,
        },
        "layers": rows,
    }
    save_json(report, dirs["debug"] / f"dynamic_single_engine_maxK_int8_layer_precision_train_calib{int(calibration_frames)}.json")
    return report


def audit_all(args: argparse.Namespace) -> dict[str, Any]:
    dirs = ensure_quant_deploy_run_dirs(args.output_root)
    reports = [audit_one(dirs, int(frames)) for frames in args.calibration_frames]
    lines = [
        "# Dynamic Single Engine maxK INT8 Layer Precision",
        "",
        "calibration | INT8 conv | FP16 conv | FP32 conv | plugin precision | renamed FP16 | engine",
        "--- | --- | --- | --- | --- | --- | ---",
    ]
    for report in reports:
        counts = report["counts"]
        lines.append(
            f"{report['calibration_frames']} | {counts['INT8 Conv count']} | {counts['FP16 Conv count']} | "
            f"{counts['FP32 Conv count']} | {report['PointPillarScatterTRT precision']} | "
            f"{report['answers']['just_renamed_fp16_engine']} | {report['layerinfo_path']}"
        )
    (dirs["summary"] / "dynamic_single_engine_maxK_int8_layer_precision_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"success": True, "reports": reports}


def main(argv: list[str] | None = None) -> int:
    report = audit_all(parse_args(argv))
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
