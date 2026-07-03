from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


WIDTHS = [8, 16, 32, 64, 128, 192, 256, 384, 512]
PRECISIONS = ["FP32", "FP16", "INT8_QDQ"]


def _load_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _stable_hash(row: dict[str, Any]) -> str:
    payload = json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _first_positive(*values: Any) -> int | None:
    for value in values:
        try:
            ivalue = int(value)
        except Exception:
            continue
        if ivalue > 0:
            return ivalue
    return None


def _scale_entry(unit_id: str, cache: dict[str, Any]) -> dict[str, Any] | None:
    units = cache.get("units") or {}
    if unit_id in units:
        return units[unit_id]
    for key, value in units.items():
        if key == unit_id or key.startswith(unit_id + ".") or unit_id in key:
            return value
    return None


def _channels_from_scale(entry: dict[str, Any] | None) -> tuple[int | None, int | None]:
    if not entry:
        return None, None
    c_out = None
    for value in (entry.get("weight_tensors") or {}).values():
        scale = value.get("scale") if isinstance(value, dict) else None
        if isinstance(scale, list) and scale:
            c_out = len(scale)
            break
    c_in = None
    return c_in, c_out


def _unit_kind(unit: dict[str, Any]) -> str | None:
    unit_type = str(unit.get("unit_type") or "")
    stage = str(unit.get("parent_stage") or "")
    uid = str(unit.get("unit_id") or "")
    if unit_type == "gemm":
        return "gemm"
    if unit_type != "conv_bn_act":
        return None
    if "shrink" in stage or "shrink" in uid:
        return "shrink"
    if "fusion" in stage or "fusion" in uid:
        return "fusion"
    if "head" in stage or "cls_head" in uid or "reg_head" in uid or "dir_head" in uid:
        return "head"
    return "conv_bn_act"


def build(args: argparse.Namespace) -> dict[str, Any]:
    inventory = _load_json(args.inventory)
    cache = _load_json(args.scale_cache)
    units = list(inventory.get("units") or [])
    keys: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    for unit in units:
        kind = _unit_kind(unit)
        if not kind:
            continue
        unit_id = str(unit.get("unit_id"))
        shape = dict(unit.get("shape_signature") or {})
        entry = _scale_entry(unit_id, cache)
        scale_cin, scale_cout = _channels_from_scale(entry)
        c_out_orig = _first_positive(shape.get("C_out"), scale_cout, 64)
        c_in_orig = _first_positive(shape.get("C_in"), scale_cin, c_out_orig)
        h = _first_positive(shape.get("H"), 16)
        w = _first_positive(shape.get("W"), 16)
        kernel = _first_positive(shape.get("kernel_size"), 1)
        stride = _first_positive(shape.get("stride"), 1)
        groups = _first_positive(shape.get("groups"), 1)
        supported = set(unit.get("supported_precision") or ["FP32", "FP16"])
        has_scale = bool(entry and entry.get("scale_valid", True) and (entry.get("input_activation_tensors") or entry.get("output_activation_tensors")) and entry.get("weight_tensors"))
        for width in WIDTHS:
            synthetic_width = width > int(c_out_orig or 0)
            if synthetic_width and not args.include_synthetic_widths:
                continue
            c_out = int(width)
            c_in = min(int(width), int(c_in_orig or width))
            if groups and (c_in % int(groups) != 0 or c_out % int(groups) != 0):
                invalid.append({"unit_id": unit_id, "width": width, "reason": "group_divisibility_failed"})
                continue
            for precision in PRECISIONS:
                activation_required = precision == "INT8_QDQ"
                activation_available = bool(has_scale)
                illegal_reason = ""
                if precision == "INT8_QDQ" and "INT8" not in supported:
                    illegal_reason = "unsupported_int8_op"
                elif precision == "INT8_QDQ" and not activation_available:
                    illegal_reason = "missing_activation_scale_before_autofill"
                row = {
                    "unit_id": unit_id,
                    "module_path": str(unit.get("module_path") or unit_id),
                    "onnx_node": (unit.get("covered_onnx_nodes") or [""])[0],
                    "unit_type": kind,
                    "parent_stage": str(unit.get("parent_stage") or ""),
                    "parent_block": str(unit.get("parent_block") or ""),
                    "H": h,
                    "W": w,
                    "kernel_size": kernel,
                    "stride": stride,
                    "groups": groups,
                    "C_in_original": c_in_orig,
                    "C_out_original": c_out_orig,
                    "sampled_width": width,
                    "C_in_aligned8": c_in,
                    "C_out_aligned8": c_out,
                    "precision": precision,
                    "activation_scale_required": activation_required,
                    "activation_scale_available": activation_available,
                    "is_shape_legal": not bool(illegal_reason),
                    "illegal_reason": illegal_reason,
                    "synthetic_width": synthetic_width,
                }
                row["lut_key"] = _stable_hash(row)
                keys.append(row)
    residual_stages = sorted({str(unit.get("parent_stage")) for unit in units if str(unit.get("parent_stage", "")).startswith("backbone.stage")})
    for stage in residual_stages:
        for width in WIDTHS:
            for precision in PRECISIONS:
                row = {
                    "unit_id": f"{stage}.residual_block_width_{width}",
                    "module_path": stage,
                    "onnx_node": "",
                    "unit_type": "residual_block",
                    "parent_stage": stage,
                    "parent_block": "residual_block",
                    "H": 16,
                    "W": 16,
                    "kernel_size": 3,
                    "stride": 1,
                    "groups": 1,
                    "C_in_original": width,
                    "C_out_original": width,
                    "sampled_width": width,
                    "C_in_aligned8": width,
                    "C_out_aligned8": width,
                    "precision": precision,
                    "activation_scale_required": precision == "INT8_QDQ",
                    "activation_scale_available": precision != "INT8_QDQ",
                    "is_shape_legal": precision != "INT8_QDQ",
                    "illegal_reason": "int8_residual_merge_not_supported" if precision == "INT8_QDQ" else "",
                    "synthetic_width": False,
                }
                row["lut_key"] = _stable_hash(row)
                keys.append(row)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"keys": keys, "invalid": invalid}, ensure_ascii=False, indent=2), encoding="utf-8")
    int8 = [row for row in keys if row["precision"] == "INT8_QDQ"]
    stats = {
        "total_keys": len(keys),
        "fp32_keys": sum(1 for row in keys if row["precision"] == "FP32"),
        "fp16_keys": sum(1 for row in keys if row["precision"] == "FP16"),
        "int8_qdq_keys": len(int8),
        "int8_keys_with_scale": sum(1 for row in int8 if row["activation_scale_available"]),
        "int8_keys_missing_scale_before_autofill": sum(1 for row in int8 if not row["activation_scale_available"]),
        "int8_keys_missing_scale_after_autofill": sum(1 for row in int8 if not row["activation_scale_available"]),
        "invalid_int8_reason_counts": dict(Counter(row["illegal_reason"] for row in int8 if row["illegal_reason"])),
        "unit_type_counts": dict(Counter(row["unit_type"] for row in keys)),
    }
    Path(args.report).write_text(
        "# Layer Width Precision Key Space v4 Report\n\n"
        + "\n".join(f"- {key}: {value}" for key, value in stats.items())
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return stats


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", default="outputs/latency_lut/atomic_deployment_unit_inventory_v2.json")
    parser.add_argument("--scale-cache", "--scale_cache", dest="scale_cache", default="outputs/latency_lut/activation_scale_cache_calib200.json")
    parser.add_argument("--output", default="outputs/latency_lut/layer_width_precision_key_space_v4.json")
    parser.add_argument("--report", default="outputs/latency_lut/layer_width_precision_key_space_v4_report.md")
    parser.add_argument("--include-synthetic-widths", "--include_synthetic_widths", dest="include_synthetic_widths", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    build(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
