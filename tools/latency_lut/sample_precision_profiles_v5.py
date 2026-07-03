from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.latency_lut.v5_pruned_mixed_common import has_all_three, has_int8, load_json, safe_id, write_json


def _exported_candidates(export_dir: Path, requested: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {str(c["candidate_id"]): c for c in requested}
    out: list[dict[str, Any]] = []
    for report_path in sorted(export_dir.glob("*/pruned_model_export_report.json")):
        report = load_json(report_path)
        if not report.get("success"):
            continue
        cid = str(report.get("candidate_id") or report_path.parent.name)
        cand = dict(by_id.get(cid) or load_json(report_path.parent / "candidate.json") or {})
        cand["structure_audit_path"] = str(report_path.parent / "structure_audit.json")
        cand["width_changed_onnx"] = str(report_path.parent / "width_changed.onnx")
        out.append(cand)
    return out


def _load_units(inventory_path: Path, scale_cache_path: Path, lut_path: Path) -> tuple[list[str], list[str], list[str]]:
    inventory = load_json(inventory_path, {"units": []})
    units = list(inventory.get("units") or [])
    scale_cache = load_json(scale_cache_path, {"units": {}})
    scale_units = set((scale_cache.get("units") or {}).keys())
    lut_units: set[str] = set()
    if lut_path.is_file():
        for line in lut_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("valid", row.get("status") == "success"):
                lut_units.add(str(row.get("unit_id") or row.get("metadata", {}).get("atomic_unit_id") or ""))
    fp_units: list[str] = []
    int8_units: list[str] = []
    head_units: list[str] = []
    for unit in units:
        uid = str(unit.get("unit_id"))
        if unit.get("unit_type") not in {"conv_bn_act", "gemm"}:
            continue
        fp_units.append(uid)
        if "head" in uid or "cls_head" in uid or "reg_head" in uid or "dir_head" in uid:
            head_units.append(uid)
        supported = set(unit.get("supported_precision") or [])
        has_scale = uid in scale_units or any(uid in key or key in uid for key in scale_units)
        has_lut = uid in lut_units or any(uid in key or key in uid for key in lut_units)
        if "INT8" in supported and has_scale and has_lut:
            int8_units.append(uid)
    return fp_units, int8_units, head_units


def _profile(cid: str, pid: str, overrides: dict[str, str]) -> dict[str, Any]:
    cfg = {"default": "FP16", "overrides": overrides}
    vals = ["FP16"] + list(overrides.values())
    return {
        "candidate_id": cid,
        "precision_profile_id": pid,
        "precision_profile": cfg,
        "has_fp32": "FP32" in vals,
        "has_fp16": "FP16" in vals,
        "has_int8_qdq": "INT8" in vals or "INT8_QDQ" in vals,
        "num_fp32_units": sum(1 for v in overrides.values() if v == "FP32"),
        "num_fp16_units": 1 + sum(1 for v in overrides.values() if v == "FP16"),
        "num_int8_qdq_units": sum(1 for v in overrides.values() if v in {"INT8", "INT8_QDQ"}),
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    requested = list(load_json(args.candidates, {"candidates": []}).get("candidates") or [])
    exported = _exported_candidates(Path(args.export_dir), requested)
    if not exported:
        exported = requested
    fp_units, int8_units, head_units = _load_units(Path(args.inventory), Path(args.scale_cache), Path(args.lut))
    profiles: list[dict[str, Any]] = []
    for idx, cand in enumerate(exported):
        cid = str(cand["candidate_id"])
        profiles.append(_profile(cid, f"{cid}_fp16", {}))
        head = head_units[idx % len(head_units)] if head_units else "detection_head"
        profiles.append(_profile(cid, f"{cid}_head_fp32", {head: "FP32"}))
        if int8_units:
            int8 = int8_units[idx % len(int8_units)]
            profiles.append(_profile(cid, f"{cid}_int8_{safe_id(int8)[:40]}", {int8: "INT8"}))
            profiles.append(_profile(cid, f"{cid}_fp32_int8_{safe_id(head)[:24]}_{safe_id(int8)[:24]}", {head: "FP32", int8: "INT8"}))
        if len(profiles) >= int(args.min_profiles):
            break
    payload = {
        "schema_version": 5,
        "profiles": profiles,
        "num_profiles": len(profiles),
        "num_with_int8_qdq": sum(1 for p in profiles if has_int8(p["precision_profile"])),
        "num_with_fp32_fp16_int8_qdq": sum(1 for p in profiles if has_all_three(p["precision_profile"])),
        "int8_units_available": int8_units,
    }
    write_json(args.output, payload)
    lines = [
        "# Pruned Precision Profiles v5",
        "",
        f"- profiles: {len(profiles)}",
        f"- profiles_with_int8_qdq: {payload['num_with_int8_qdq']}",
        f"- profiles_with_fp32_fp16_int8_qdq: {payload['num_with_fp32_fp16_int8_qdq']}",
        f"- int8_units_available: {len(int8_units)}",
        "",
        "INT8_QDQ is only assigned to units with activation scale and measured LUT coverage according to the current inventory/cache/LUT files.",
    ]
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k != "profiles"}, indent=2))
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="outputs/latency_lut/pruned_width_changed_candidates_v5.json")
    parser.add_argument("--export-dir", "--export_dir", dest="export_dir", default="outputs/latency_lut/pruned_width_changed_onnx_v5")
    parser.add_argument("--inventory", default="outputs/latency_lut/atomic_deployment_unit_inventory_v2.json")
    parser.add_argument("--scale-cache", "--scale_cache", dest="scale_cache", default="outputs/latency_lut/activation_scale_cache_calib200.json")
    parser.add_argument("--lut", default="outputs/latency_lut/layer_width_precision_lut_measurements_v4.jsonl")
    parser.add_argument("--output", default="outputs/latency_lut/pruned_precision_profiles_v5.json")
    parser.add_argument("--report", default="outputs/latency_lut/pruned_precision_profiles_v5_report.md")
    parser.add_argument("--min-profiles", "--min_profiles", dest="min_profiles", type=int, default=30)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    build(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
