from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent_mode_fixed_k_plugin_ablation import FIXED_K_BUCKETS, _dynamic_layerinfo_path, _layer_name, _layer_time_ms, _layer_type, _load_profile_layers
from quant_deploy_utils import ensure_quant_deploy_run_dirs, save_json


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit layer precision in dynamic fixed-K INT8 engines.")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--calibration_frames", type=int, required=True)
    parser.add_argument("--mixed_heads_fp16", action="store_true")
    return parser.parse_args(argv)


def _precision_tag(calibration_frames: int, mixed_heads_fp16: bool = False) -> str:
    suffix = "_mixed_heads_fp16" if mixed_heads_fp16 else ""
    return f"int8_calib{int(calibration_frames)}{suffix}"


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
    if layer_type.lower() == "pluginv2" or name == "/PointPillarScatterTRT":
        return "plugin"
    if "pfn" in low or "pillar_vfe" in low:
        return "PillarVFE/PFN"
    if "shrink" in low:
        return "shrink_conv"
    if "cls" in low or "reg" in low or "dir" in low or "head" in low:
        return "detection_head"
    if "conv" in layer_type.lower() or "convolution" in low or "backbone" in low:
        return "BEV_backbone"
    if "pointpillar" in low:
        return "reformat_cast_qdq"
    if "matmul" in low or "gemm" in low:
        return "gemm_matmul"
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
        "INT8 Conv layers count": 0,
        "FP16 Conv layers count": 0,
        "FP32 Conv layers count": 0,
        "INT8 GEMM / MatMul layers count": 0,
        "FP16/FP32 fallback layers": 0,
        "Reformat / Cast / Quantize / Dequantize layers count": 0,
    }
    category_precision: dict[str, dict[str, int]] = {}
    for row in rows:
        name = str(row.get("name", ""))
        layer_type = str(row.get("layer_type", ""))
        precision = str(row.get("precision", "unknown"))
        cat = str(row.get("category", "other"))
        category_precision.setdefault(cat, {})
        category_precision[cat][precision] = category_precision[cat].get(precision, 0) + 1
        low = f"{name} {layer_type}".lower()
        if "conv" in low or "convolution" in low:
            if precision == "int8":
                counts["INT8 Conv layers count"] += 1
            elif precision == "fp16":
                counts["FP16 Conv layers count"] += 1
            elif precision == "fp32":
                counts["FP32 Conv layers count"] += 1
        if ("gemm" in low or "matmul" in low) and precision == "int8":
            counts["INT8 GEMM / MatMul layers count"] += 1
        if precision in {"fp16", "fp32"}:
            counts["FP16/FP32 fallback layers"] += 1
        if any(token in low for token in ("reformat", "cast", "quantize", "dequantize")):
            counts["Reformat / Cast / Quantize / Dequantize layers count"] += 1
    counts["category_precision"] = category_precision
    return counts


def audit_precision(args: argparse.Namespace) -> dict[str, Any]:
    dirs = ensure_quant_deploy_run_dirs(args.output_root)
    precision_tag = _precision_tag(int(args.calibration_frames), bool(args.mixed_heads_fp16))
    all_rows: list[dict[str, Any]] = []
    per_engine = []
    for fixed_n in (1, 2):
        for bucket in FIXED_K_BUCKETS:
            path = _dynamic_layerinfo_path(dirs, precision_tag, fixed_n, int(bucket["bucket_id"]))
            layers = _load_profile_layers(path)
            rows = []
            for item in layers:
                name = _layer_name(item)
                layer_type = _layer_type(item)
                row = {
                    "fixed_N": fixed_n,
                    "bucket_id": int(bucket["bucket_id"]),
                    "name": name,
                    "layer_type": layer_type,
                    "precision": _precision_of(item),
                    "category": _category(name, layer_type),
                    "latency_ms": _layer_time_ms(item),
                    "raw": item,
                }
                rows.append(row)
                all_rows.append(row)
            per_engine.append({"fixed_N": fixed_n, "bucket_id": int(bucket["bucket_id"]), "layerinfo_path": str(path), "num_layers": len(rows), "counts": _count_layers(rows)})
    counts = _count_layers(all_rows)
    top20 = sorted(all_rows, key=lambda row: float(row.get("latency_ms") or 0.0), reverse=True)[:20]
    plugin_rows = [row for row in all_rows if row["category"] == "plugin"]
    plugin_io = [
        {
            "fixed_N": row["fixed_N"],
            "bucket_id": row["bucket_id"],
            "name": row["name"],
            "input_dtypes": _tensor_dtypes(row, "Inputs"),
            "output_dtypes": _tensor_dtypes(row, "Outputs"),
        }
        for row in plugin_rows
    ]
    report = {
        "calibration_frames": int(args.calibration_frames),
        "engine_precision_tag": precision_tag,
        "precision_strategy": "mixed_heads_fp16" if args.mixed_heads_fp16 else "native",
        "per_engine": per_engine,
        "counts": counts,
        "PointPillarScatterTRT precision": sorted({row["precision"] for row in plugin_rows}) or ["unknown"],
        "PointPillarScatterTRT io_dtypes": plugin_io,
        "PillarVFE / PFN precision": counts["category_precision"].get("PillarVFE/PFN", {}),
        "BEV backbone precision": counts["category_precision"].get("BEV_backbone", {}),
        "shrink_conv precision": counts["category_precision"].get("shrink_conv", {}),
        "detection head precision": counts["category_precision"].get("detection_head", {}),
        "plugin precision": counts["category_precision"].get("plugin", {}),
        "top 20 latency layers": top20,
        "answers": {
            "int8_engine_really_uses_int8": counts["INT8 Conv layers count"] > 0 or counts["INT8 GEMM / MatMul layers count"] > 0,
            "just_renamed_fp16_engine": counts["INT8 Conv layers count"] == 0 and counts["INT8 GEMM / MatMul layers count"] == 0,
            "plugin_causes_large_fallback": "unknown" if not plugin_rows else counts["FP16/FP32 fallback layers"] > counts["INT8 Conv layers count"],
            "main_latency_layers": [row["category"] for row in top20[:5]],
            "int8_speedup_space": counts["INT8 Conv layers count"] > 0,
        },
    }
    suffix = f"calib{int(args.calibration_frames)}" + ("_mixed_heads_fp16" if args.mixed_heads_fp16 else "")
    save_json(report, dirs["debug"] / f"int8_layer_precision_profile_{suffix}.json")
    lines = [
        "# INT8 Layer Precision Profile",
        "",
        f"- INT8 Conv layers count: {counts['INT8 Conv layers count']}",
        f"- FP16 Conv layers count: {counts['FP16 Conv layers count']}",
        f"- FP32 Conv layers count: {counts['FP32 Conv layers count']}",
        f"- PointPillarScatterTRT precision: {report['PointPillarScatterTRT precision']}",
        f"- Reformat/Cast/Q/DQ layers: {counts['Reformat / Cast / Quantize / Dequantize layers count']}",
        "",
        "rank | latency_ms | precision | category | layer_type | layer",
        "--- | --- | --- | --- | --- | ---",
    ]
    for idx, row in enumerate(top20, start=1):
        lines.append(f"{idx} | {row.get('latency_ms')} | {row.get('precision')} | {row.get('category')} | {row.get('layer_type')} | {str(row.get('name')).replace('|', '/')}")
    report_name = "int8_layer_precision_profile_mixed_heads_fp16_report.md" if args.mixed_heads_fp16 else "int8_layer_precision_profile_report.md"
    (dirs["summary"] / report_name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    report = audit_precision(parse_args(argv))
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
