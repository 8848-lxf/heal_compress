from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from opencood.tools.compression.latency_lut.schema import DEPLOY_MODE, FIXED_K, LatencyLUTKey, read_record_jsonl


KEEP_RATIOS = (1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25)
PRECISIONS = ("FP32", "FP16", "INT8_QDQ")
PROFILE_BY_PRECISION = {"FP32": "TRT_FP32", "FP16": "TRT_FP16", "INT8_QDQ": "TRT_INT8_QDQ"}
WEIGHT_BY_PRECISION = {"FP32": "FP32", "FP16": "FP16", "INT8_QDQ": "INT8"}


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _align8(value: int | None) -> int | None:
    if value is None:
        return None
    return int(math.ceil(max(1, int(value)) / 8.0) * 8)


def _first_int(*values: Any) -> int | None:
    for value in values:
        if value is None:
            continue
        try:
            ivalue = int(value)
        except Exception:
            continue
        if ivalue > 0:
            return ivalue
    return None


def _channels_from_scale(unit_id: str, scale_cache: dict[str, Any]) -> tuple[int | None, int | None]:
    units = dict(scale_cache.get("units") or {})
    entry = units.get(unit_id)
    if entry is None:
        for key, value in units.items():
            if key == unit_id or key in unit_id or unit_id in key:
                entry = value
                break
    if not isinstance(entry, dict):
        return None, None
    weight_tensors = entry.get("weight_tensors") or {}
    c_out = None
    for value in weight_tensors.values():
        scale = value.get("scale") if isinstance(value, dict) else None
        if isinstance(scale, list) and scale:
            c_out = len(scale)
            break
    input_tensors = entry.get("input_activation_tensors") or {}
    c_in = None
    for value in input_tensors.values():
        shape = value.get("shape") if isinstance(value, dict) else None
        if isinstance(shape, list) and len(shape) >= 2:
            c_in = _first_int(shape[1])
            break
    return c_in, c_out


def _scale_entry(unit_id: str, scale_cache: dict[str, Any]) -> dict[str, Any] | None:
    units = dict(scale_cache.get("units") or {})
    entry = units.get(unit_id)
    if isinstance(entry, dict):
        return entry
    for key, value in units.items():
        if key == unit_id or key in unit_id or unit_id in key:
            return value if isinstance(value, dict) else None
    return None


def _first_scale(entry: dict[str, Any] | None) -> tuple[float | None, float | None, str | None]:
    if not isinstance(entry, dict):
        return None, None, None
    activation_scale = None
    for tensors_name in ("input_activation_tensors", "output_activation_tensors"):
        tensors = entry.get(tensors_name) or {}
        for value in tensors.values():
            if isinstance(value, dict) and value.get("scale") is not None:
                activation_scale = float(value["scale"])
                break
        if activation_scale is not None:
            break
    weight_scale = None
    for value in (entry.get("weight_tensors") or {}).values():
        scale = value.get("scale") if isinstance(value, dict) else None
        if isinstance(scale, list) and scale:
            weight_scale = float(max(float(item) for item in scale))
            break
        if scale is not None:
            weight_scale = float(scale)
            break
    return activation_scale, weight_scale, str(entry.get("scale_source") or "train_calib200")


def _record_index(records_path: str | Path) -> dict[str, list[Any]]:
    index: dict[str, list[Any]] = defaultdict(list)
    for record in read_record_jsonl(records_path):
        if record.status not in {"success", "ok"}:
            continue
        key = record.key
        for token in {key.block_name, key.module_name, f"{key.module_name}.{key.block_name}"}:
            if token:
                index[str(token).lower()].append(record)
    return index


def _coarse_shape_from_records(unit: dict[str, Any], record_index: dict[str, list[Any]]) -> dict[str, Any]:
    probes = [
        str(unit.get("unit_id", "")).lower(),
        str(unit.get("parent_stage", "")).lower(),
        str(unit.get("parent_block", "")).lower(),
    ]
    options = []
    for probe in probes:
        options.extend(record_index.get(probe, []))
    if not options:
        # fall back from full-engine atomic stage names to coarse module records
        stage = str(unit.get("parent_stage", "")).lower()
        if "backbone" in stage:
            options.extend(record_index.get("backbone", []))
        elif "shrink" in stage:
            options.extend(record_index.get("shrink", []))
        elif "fusion" in stage:
            options.extend(record_index.get("pyramid_fusion", []))
        elif "head" in stage or "detection" in stage:
            options.extend(record_index.get("detection_head", []))
    if not options:
        return {}
    best = sorted(
        options,
        key=lambda record: (
            0 if record.key.precision_profile == "TRT_FP16" else 1,
            -(record.key.C_out or record.key.C_mid or record.key.C_in or 0),
        ),
    )[0]
    key = best.key
    return {
        "H": key.H,
        "W": key.W,
        "C_in": key.C_in,
        "C_mid": key.C_mid,
        "C_out": key.C_out,
        "kernel_size": key.kernel_size,
        "stride": key.stride,
        "groups": key.groups,
        "coarse_lut_key": key.stable_hash(),
    }


def _unit_type_to_block_type(unit: dict[str, Any]) -> tuple[str, str]:
    unit_type = str(unit.get("unit_type") or "")
    stage = str(unit.get("parent_stage") or "")
    uid = str(unit.get("unit_id") or "")
    if unit_type == "plugin":
        return "scatter", "plugin"
    if unit_type == "gemm":
        return "pfn", "pfn_block"
    if "shrink" in stage or "shrink" in uid:
        return "shrink", "compression_1x1"
    if "fusion" in stage or "fusion" in uid:
        return "pyramid_fusion", "fusion_block"
    if "head" in stage or "cls_head" in uid or "reg_head" in uid or "dir_head" in uid:
        return "detection_head", "head_branch"
    if unit_type == "conv_bn_act":
        return "backbone", "conv_block"
    if unit_type in {"merge", "concat"}:
        return "elementwise_merge", "merge"
    if unit_type == "grid_sample":
        return "grid_sample", "grid_sample"
    return unit_type or "unknown", unit_type or "unknown"


def _shape_signature(unit: dict[str, Any], shape: dict[str, Any], c_in: int | None, c_out: int | None) -> str:
    return (
        f"{unit.get('unit_id')}|H={shape.get('H')}|W={shape.get('W')}|"
        f"Cin={c_in}|Cout={c_out}|K={shape.get('kernel_size')}|S={shape.get('stride')}"
    )


def _lut_key_hash(row: dict[str, Any]) -> str:
    module_name, block_type = _unit_type_to_block_type(row)
    profile = PROFILE_BY_PRECISION[str(row["precision"])]
    weight = WEIGHT_BY_PRECISION[str(row["precision"])]
    metadata = {
        "atomic_unit_id": row["unit_id"],
        "parent_stage": row.get("parent_stage"),
        "parent_block": row.get("parent_block"),
        "shape_signature": row.get("shape_signature"),
    }
    if row.get("scale_source") is not None:
        metadata["scale_source"] = row.get("scale_source")
    if row.get("activation_scale") is not None:
        metadata["activation_scale"] = row.get("activation_scale")
    if row.get("weight_scale") is not None:
        metadata["weight_scale"] = row.get("weight_scale")
    key = LatencyLUTKey(
        deploy_mode=DEPLOY_MODE,
        fixed_K=FIXED_K,
        module_name=module_name,
        block_name=str(row["unit_id"]),
        block_type=block_type,
        H=row.get("H"),
        W=row.get("W"),
        C_in=row.get("C_in_aligned8") or row.get("C_in"),
        C_mid=row.get("C_mid_aligned8") or row.get("C_mid"),
        C_out=row.get("C_out_aligned8") or row.get("C_out"),
        kernel_size=row.get("K"),
        stride=row.get("stride"),
        padding=(int(row.get("K") or 1) // 2 if int(row.get("K") or 1) > 1 else 0),
        groups=row.get("groups"),
        precision_profile=profile,
        weight_precision=weight,
        activation_precision=weight if weight != "INT8" else "INT8",
        compute_precision=weight,
        plugin_flag=block_type == "plugin",
        plugin_name="PointPillarScatterTRT" if block_type == "plugin" else None,
        metadata=metadata,
    )
    return key.stable_hash()


def build_key_space(args: argparse.Namespace) -> dict[str, Any]:
    inventory = _load_json(args.inventory)
    scale_cache = _load_json(args.scale_cache) if Path(args.scale_cache).is_file() else {}
    record_index = _record_index(args.lut)
    units = list(inventory.get("units") or [])
    rows: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    for unit in units:
        unit_type = str(unit.get("unit_type") or "")
        stage = str(unit.get("parent_stage") or "")
        if unit_type in {"fixed_overhead_bucket", "memory_reformat_bucket"}:
            continue
        supported = set(unit.get("supported_precision") or unit.get("ga_enabled_precision") or ["FP16", "FP32"])
        shape = dict(unit.get("shape_signature") or {})
        coarse = _coarse_shape_from_records(unit, record_index)
        for name, value in coarse.items():
            shape.setdefault(name, value)
        scale_cin, scale_cout = _channels_from_scale(str(unit.get("unit_id")), scale_cache)
        scale_entry = _scale_entry(str(unit.get("unit_id")), scale_cache)
        activation_scale, weight_scale, scale_source = _first_scale(scale_entry)
        base_c_in = _first_int(shape.get("C_in"), scale_cin, shape.get("C_out"), scale_cout)
        base_c_out = _first_int(shape.get("C_out"), scale_cout, shape.get("C_in"), scale_cin)
        h = _first_int(shape.get("H"))
        w = _first_int(shape.get("W"))
        kernel = _first_int(shape.get("kernel_size"), 1)
        stride = _first_int(shape.get("stride"), 1)
        groups = _first_int(shape.get("groups"), 1)
        measurable_compute = unit_type in {"conv_bn_act", "gemm"} or "shrink" in stage or "fusion" in stage or "head" in stage
        if unit_type in {"plugin", "grid_sample", "merge", "concat", "cast_boundary", "qdq_boundary"}:
            measurable_compute = False
        if measurable_compute and (base_c_in is None or base_c_out is None):
            invalid.append({"unit_id": unit.get("unit_id"), "reason": "missing_channel_shape"})
            continue
        if measurable_compute and (h is None or w is None):
            # These are still benchmarkable as coarse surrogate subgraphs.
            h = 16
            w = 16
        ratios = KEEP_RATIOS if measurable_compute else (1.0,)
        for ratio in ratios:
            c_in = _align8(round(float(base_c_in or 1) * ratio)) if measurable_compute else base_c_in
            c_out = _align8(round(float(base_c_out or base_c_in or 1) * ratio)) if measurable_compute else base_c_out
            for precision in PRECISIONS:
                if precision == "INT8_QDQ":
                    if "INT8" not in supported:
                        invalid.append({"unit_id": unit.get("unit_id"), "precision": precision, "reason": "unsupported_precision"})
                        continue
                    if not scale_entry or not scale_entry.get("scale_valid", True) or activation_scale is None or weight_scale is None:
                        invalid.append({"unit_id": unit.get("unit_id"), "precision": precision, "reason": "missing_activation_scale"})
                        continue
                elif precision not in supported and f"TRT_{precision}" not in supported:
                    invalid.append({"unit_id": unit.get("unit_id"), "precision": precision, "reason": "unsupported_precision"})
                    continue
                expected_source = "subgraph_engine" if measurable_compute else "measured_bucket"
                if unit_type in {"grid_sample", "merge", "concat"}:
                    expected_source = "coarse_bucket"
                row = {
                    "unit_id": str(unit.get("unit_id")),
                    "unit_type": unit_type,
                    "parent_stage": stage,
                    "parent_block": str(unit.get("parent_block") or ""),
                    "layer_index": str(unit.get("layer_index") or ""),
                    "H": h,
                    "W": w,
                    "K": kernel,
                    "stride": stride,
                    "groups": groups,
                    "C_in": base_c_in,
                    "C_mid": None,
                    "C_out": base_c_out,
                    "C_in_aligned8": c_in,
                    "C_mid_aligned8": None,
                    "C_out_aligned8": c_out,
                    "channel_keep_ratio": float(ratio),
                    "precision": precision,
                    "shape_signature": _shape_signature(unit, shape, c_in, c_out),
                    "lut_key": "",
                    "expected_source": expected_source,
                    "required_for_latency_proxy": bool(unit.get("can_enter_latency_proxy", False) or measurable_compute),
                    "scale_source": scale_source if precision == "INT8_QDQ" else None,
                    "activation_scale": activation_scale if precision == "INT8_QDQ" else None,
                    "weight_scale": weight_scale if precision == "INT8_QDQ" else None,
                }
                row["lut_key"] = _lut_key_hash(row) if expected_source == "subgraph_engine" else f"{unit_type}:{precision}:{row['shape_signature']}"
                rows.append(row)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"keys": rows, "invalid": invalid}, ensure_ascii=False, indent=2), encoding="utf-8")
    stats = {
        "total_units": len(units),
        "total_keys": len(rows),
        "invalid_key_requests": len(invalid),
        "by_unit_type": dict(Counter(row["unit_type"] for row in rows)),
        "by_precision": dict(Counter(row["precision"] for row in rows)),
        "by_expected_source": dict(Counter(row["expected_source"] for row in rows)),
        "invalid_by_reason": dict(Counter(row["reason"] for row in invalid)),
    }
    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# LUT Key Space v3 Report",
        "",
        f"- total atomic units: {stats['total_units']}",
        f"- total LUT key requests: {stats['total_keys']}",
        f"- invalid/skipped key requests: {stats['invalid_key_requests']}",
        f"- by unit type: `{stats['by_unit_type']}`",
        f"- by precision: `{stats['by_precision']}`",
        f"- by expected source: `{stats['by_expected_source']}`",
        f"- invalid by reason: `{stats['invalid_by_reason']}`",
        "",
        "Channel samples use keep ratios `1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25` and align sampled channel widths upward to multiples of 8.",
        "Rows with missing activation scale for INT8 are excluded from measured INT8 benchmark execution and must not fall back to FP16.",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    plan = {
        "goal": "expand measured LUT coverage before calibrator retraining",
        "granularity": ["stage-level", "block-level", "atomic layer-level", "ConvBNAct deployment unit", "8-aligned channel width"],
        "unit_types": [
            "backbone ConvBNAct",
            "shrink/compression",
            "fusion Conv",
            "detection head Conv",
            "GEMM/Linear",
            "plugin/scatter",
            "GridSample/geometry",
            "Add/residual merge",
            "Concat/fusion merge",
            "Cast/reformat boundary",
            "Q/DQ boundary",
        ],
        "target_coverage": {
            "FP32_FP16_measured_lut": ">=90%",
            "INT8_QDQ_measured_lut_initial": ">=70%",
            "non_compute_measured_bucket": ">=70%",
        },
        "key_space_stats": stats,
    }
    Path(args.plan_json).write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.plan_md).write_text(
        "# LUT Sample Expansion Plan\n\n"
        "This stage expands measurement data only. It does not train Ridge/MLP and does not connect GA.\n\n"
        f"- key requests generated: {stats['total_keys']}\n"
        f"- invalid/skipped requests: {stats['invalid_key_requests']}\n"
        f"- expected source breakdown: `{stats['by_expected_source']}`\n\n"
        "Priority order: measured subgraph engines for ConvBNAct/shrink/fusion/head/GEMM, then measured plugin and boundary buckets, then coarse buckets for GridSample/merge when exact microbenchmarks are unavailable.\n",
        encoding="utf-8",
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return stats


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", default="outputs/latency_lut/atomic_deployment_unit_inventory_v2.json")
    parser.add_argument("--scale-cache", "--scale_cache", dest="scale_cache", default="outputs/latency_lut/activation_scale_cache_calib200.json")
    parser.add_argument("--lut", default="outputs/latency_lut/lut_records.jsonl")
    parser.add_argument("--output", default="outputs/latency_lut/lut_key_space_v3.json")
    parser.add_argument("--report", default="outputs/latency_lut/lut_key_space_v3_report.md")
    parser.add_argument("--plan-json", "--plan_json", dest="plan_json", default="outputs/latency_lut/lut_sample_expansion_plan.json")
    parser.add_argument("--plan-md", "--plan_md", dest="plan_md", default="outputs/latency_lut/lut_sample_expansion_plan.md")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    build_key_space(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
