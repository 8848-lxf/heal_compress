from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from opencood.tools.compression.latency_lut.schema import read_record_jsonl


PRECISIONS = ["TRT_FP32", "TRT_FP16", "TRT_INT8_QDQ"]


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_units(path: str | Path) -> list[dict[str, Any]]:
    data = _load_json(path)
    return list(data.get("units") or data)


def _coarse_lut_family(unit: dict[str, Any]) -> tuple[str, str, str] | None:
    unit_type = str(unit.get("unit_type"))
    module_path = str(unit.get("module_path") or "")
    unit_id = str(unit.get("unit_id") or "")
    if unit_type == "plugin" or "scatter" in module_path:
        return ("scatter", "scatter", "plugin")
    if module_path.startswith("pfn") or unit_id.startswith("pillar_vfe") or "pfn_layers" in unit_id:
        return ("pfn", "pfn", "pfn_block")
    if module_path == "backbone.stage1" or unit_id.startswith("layer0."):
        return ("backbone", "stage1", "conv_block")
    if module_path == "backbone.stage2" or unit_id.startswith("layer1."):
        return ("backbone", "stage2", "residual_block")
    if module_path == "backbone.stage3" or unit_id.startswith("layer2."):
        return ("backbone", "stage3", "residual_block")
    if module_path.startswith("shrink") or unit_id.startswith("shrink"):
        return ("shrink", "compression", "compression_1x1")
    if module_path.startswith("pyramid_fusion") or "fusion" in unit_id:
        return ("pyramid_fusion", "scale1", "fusion_block")
    if module_path.startswith("detection_head") or any(head in unit_id for head in ("cls_head", "reg_head", "dir_head")):
        return ("detection_head", "cls", "head_branch")
    return None


def audit(args: argparse.Namespace) -> dict[str, Any]:
    units = _load_units(args.inventory)
    scale_cache = _load_json(args.scale_cache) if Path(args.scale_cache).is_file() else {}
    scale_units = set((scale_cache.get("units") or {}).keys())
    records = [record for record in read_record_jsonl(args.lut) if record.status in {"success", "ok"}]
    family_precision = {
        (record.key.module_name, record.key.block_name, record.key.block_type, record.key.precision_profile)
        for record in records
        if record.key.block_type != "precision_boundary"
    }

    rows: list[dict[str, Any]] = []
    missing_keys: list[dict[str, Any]] = []
    for unit in units:
        unit_id = str(unit.get("unit_id"))
        family = _coarse_lut_family(unit)
        has = {}
        for precision in PRECISIONS:
            has[precision] = bool(family and (*family, precision) in family_precision)
            if not has[precision]:
                missing_keys.append(
                    {
                        "unit_id": unit_id,
                        "module_path": unit.get("module_path"),
                        "unit_type": unit.get("unit_type"),
                        "mapped_family": list(family) if family else None,
                        "precision_profile": precision,
                    }
                )
        has_scale = unit_id in scale_units
        int8_supported = bool(unit.get("int8_supported"))
        int8_available = bool(int8_supported and has["TRT_INT8_QDQ"] and has_scale)
        fp_available = bool(has["TRT_FP16"] and has["TRT_FP32"])
        rows.append(
            {
                "unit_id": unit_id,
                "module_path": unit.get("module_path"),
                "unit_type": unit.get("unit_type"),
                "mapped_lut_family": list(family) if family else None,
                "has_fp32_lut": has["TRT_FP32"],
                "has_fp16_lut": has["TRT_FP16"],
                "has_int8_qdq_lut": has["TRT_INT8_QDQ"],
                "has_activation_scale": has_scale,
                "int8_supported_by_inventory": int8_supported,
                "can_search_fp32_fp16": fp_available,
                "can_search_int8": int8_available,
                "can_search_precision": ["FP16", "FP32"] + (["INT8"] if int8_available else []),
                "int8_unavailable_reason": None
                if int8_available
                else (
                    "int8_unit_not_supported"
                    if not int8_supported
                    else ("int8_activation_scale_missing" if not has_scale else "int8_qdq_lut_missing")
                ),
            }
        )

    units_with_fp32 = [row["unit_id"] for row in rows if row["has_fp32_lut"]]
    units_with_fp16 = [row["unit_id"] for row in rows if row["has_fp16_lut"]]
    units_with_int8 = [row["unit_id"] for row in rows if row["has_int8_qdq_lut"]]
    units_with_scale = [row["unit_id"] for row in rows if row["has_activation_scale"]]
    allowed = [row["unit_id"] for row in rows if row["can_search_fp32_fp16"]]
    unavailable_int8 = [row for row in rows if not row["can_search_int8"]]
    report = {
        "total_atomic_units": len(rows),
        "units_with_fp32_lut": len(units_with_fp32),
        "units_with_fp16_lut": len(units_with_fp16),
        "units_with_int8_qdq_lut": len(units_with_int8),
        "units_with_activation_scale": len(units_with_scale),
        "units_missing_fp32_lut": [row["unit_id"] for row in rows if not row["has_fp32_lut"]],
        "units_missing_fp16_lut": [row["unit_id"] for row in rows if not row["has_fp16_lut"]],
        "units_missing_int8_lut": [row["unit_id"] for row in rows if not row["has_int8_qdq_lut"]],
        "units_missing_activation_scale": [row["unit_id"] for row in rows if not row["has_activation_scale"]],
        "units_allowed_for_ga_precision_search": allowed,
        "units_unavailable_for_int8_search": unavailable_int8,
        "exact_missing_lut_keys": missing_keys,
        "unit_coverage": rows,
        "note": "Coverage is audited through the current coarse LUT family mapping; exact per-atomic TensorRT LUT expansion is still needed for true atomic latency additivity.",
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md = [
        "# LUT Coverage Audit",
        "",
        f"- total_atomic_units: {report['total_atomic_units']}",
        f"- units_with_fp32_lut: {report['units_with_fp32_lut']}",
        f"- units_with_fp16_lut: {report['units_with_fp16_lut']}",
        f"- units_with_int8_qdq_lut: {report['units_with_int8_qdq_lut']}",
        f"- units_with_activation_scale: {report['units_with_activation_scale']}",
        f"- units_allowed_for_ga_precision_search: {len(allowed)}",
        f"- units_unavailable_for_int8_search: {len(unavailable_int8)}",
        "",
        "FP32/FP16 missing LUT records must not receive default latency. INT8 missing QDQ LUT or missing activation scale is unavailable and must not fallback to FP16.",
        "",
        "## Major Gaps",
        f"- Missing FP32 LUT units: {len(report['units_missing_fp32_lut'])}",
        f"- Missing FP16 LUT units: {len(report['units_missing_fp16_lut'])}",
        f"- Missing INT8 LUT units: {len(report['units_missing_int8_lut'])}",
        f"- Missing activation scale units: {len(report['units_missing_activation_scale'])}",
        "",
        report["note"],
    ]
    Path(args.markdown).write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ["total_atomic_units", "units_with_fp32_lut", "units_with_fp16_lut", "units_with_int8_qdq_lut", "units_with_activation_scale"]}, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", default="outputs/latency_lut/atomic_deployment_unit_inventory.json")
    parser.add_argument("--lut", default="outputs/latency_lut/lut_records.jsonl")
    parser.add_argument("--scale-cache", default="outputs/latency_lut/activation_scale_cache_calib200.json")
    parser.add_argument("--output", default="outputs/latency_lut/lut_coverage_audit.json")
    parser.add_argument("--markdown", default="outputs/latency_lut/lut_coverage_audit.md")
    return parser.parse_args()


def main() -> int:
    audit(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
