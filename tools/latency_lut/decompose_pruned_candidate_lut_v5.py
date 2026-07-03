from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.latency_lut.v5_pruned_mixed_common import (
    audit_width_changed_onnx,
    has_int8,
    load_json,
    load_lut_rows,
    write_json,
)


def _norm(text: str) -> str:
    return "".join(ch.lower() for ch in text if ch.isalnum())


def _precision_label(value: str) -> str:
    v = str(value).upper()
    if v in {"INT8", "TRT_INT8_QDQ"}:
        return "INT8_QDQ"
    if v in {"FP32", "TRT_FP32"}:
        return "FP32"
    return "FP16"


def _align8(value: Any) -> int:
    try:
        ivalue = int(value)
    except Exception:
        return 0
    return int(math.ceil(max(ivalue, 1) / 8.0) * 8)


def _profile_for_layer(layer_name: str, profile: dict[str, Any]) -> str:
    cfg = dict(profile or {})
    default = _precision_label(str(cfg.get("default", "FP16")))
    overrides = {str(k): _precision_label(str(v)) for k, v in dict(cfg.get("overrides") or {}).items()}
    norm_layer = _norm(layer_name)
    best = default
    best_len = -1
    for key, value in overrides.items():
        nk = _norm(key)
        if nk and (nk in norm_layer or norm_layer in nk) and len(nk) > best_len:
            best = value
            best_len = len(nk)
    return best


def _match_lut(layer_name: str, shape: dict[str, Any], precision: str, rows: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, str]:
    norm_layer = _norm(layer_name)
    width = _align8(shape.get("C_out"))
    cin = _align8(shape.get("C_in"))
    candidates = [
        row
        for row in rows
        if row.get("valid", row.get("status") == "success")
        and _precision_label(str(row.get("precision") or row.get("precision_profile") or "")) == precision
    ]
    name_matches = [
        row for row in candidates
        if _norm(str(row.get("unit_id") or row.get("block_name") or "")) in norm_layer
        or norm_layer in _norm(str(row.get("unit_id") or row.get("block_name") or ""))
    ]
    exact = [
        row for row in name_matches
        if int(row.get("C_out_aligned8") or row.get("C_out") or row.get("sampled_width") or 0) == width
        and int(row.get("C_in_aligned8") or row.get("C_in") or cin) == cin
    ]
    if exact:
        return exact[0], "exact"
    same_width = [
        row for row in name_matches
        if int(row.get("C_out_aligned8") or row.get("C_out") or row.get("sampled_width") or 0) == width
    ]
    if same_width:
        return same_width[0], "coarse_name_width"
    if name_matches:
        return min(name_matches, key=lambda row: abs(int(row.get("C_out_aligned8") or row.get("C_out") or row.get("sampled_width") or 0) - width)), "coarse_name_nearest_width"
    by_width = [
        row for row in candidates
        if int(row.get("C_out_aligned8") or row.get("C_out") or row.get("sampled_width") or 0) == width
    ]
    if by_width:
        return by_width[0], "coarse_width_only"
    return None, "missing"


def _bucket_sum(path: str | Path) -> dict[str, float]:
    data = load_json(path, [])
    if isinstance(data, dict):
        rows = list(data.get("buckets") or data.get("records") or data.get("measurements") or [])
    else:
        rows = list(data or [])
    out = {
        "T_boundary_cast": 0.0,
        "T_boundary_qdq": 0.0,
        "T_plugin_or_scatter": 0.0,
        "T_grid_sample_or_geometry": 0.0,
        "T_elementwise_merge": 0.0,
        "T_memory_reformat": 0.0,
        "T_fixed_overhead": 0.0,
    }
    for row in rows:
        if not row.get("valid", True):
            continue
        latency = float(row.get("latency_p50_ms") or row.get("p50_ms") or row.get("latency_ms") or 0.0)
        comp = str(row.get("component_type") or row.get("bucket_id") or "")
        if "cast" in comp:
            out["T_boundary_cast"] += latency
        elif "qdq" in comp.lower() or "quant" in comp.lower():
            out["T_boundary_qdq"] += latency
        elif "scatter" in comp or "plugin" in comp:
            out["T_plugin_or_scatter"] += latency
        elif "grid" in comp.lower() or "geometry" in comp.lower() or "warp" in comp.lower():
            out["T_grid_sample_or_geometry"] += latency
        elif "add" in comp.lower() or "concat" in comp.lower() or "merge" in comp.lower():
            out["T_elementwise_merge"] += latency
        elif "reformat" in comp.lower() or "memory" in comp.lower():
            out["T_memory_reformat"] += latency
        elif "fixed" in comp.lower() or "enqueue" in comp.lower() or "overhead" in comp.lower():
            out["T_fixed_overhead"] += latency
    return out


def decompose_one(candidate: dict[str, Any], profile: dict[str, Any], rows: list[dict[str, Any]], buckets: dict[str, float], out_dir: Path) -> dict[str, Any]:
    cid = str(candidate["candidate_id"])
    pid = str(profile["precision_profile_id"])
    audit_path = Path(str(candidate.get("structure_audit_path") or ""))
    audit = load_json(audit_path) if audit_path.is_file() else audit_width_changed_onnx(cid, candidate.get("width_changed_onnx", ""), candidate=candidate)
    conv_shapes = dict(audit.get("candidate_conv_shapes") or {})
    if not conv_shapes:
        # Recompute and keep the audit small if older export reports did not
        # include the full shape dictionaries.
        audit = audit_width_changed_onnx(cid, candidate.get("width_changed_onnx", ""), candidate=candidate)
        conv_shapes = dict(audit.get("candidate_conv_shapes") or {})
    matched: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    coarse: list[dict[str, Any]] = []
    by_precision = Counter()
    compute = 0.0
    profile_cfg = dict(profile.get("precision_profile") or {})
    for layer_name, shape in sorted(conv_shapes.items()):
        precision = _profile_for_layer(layer_name, profile_cfg)
        row, match_type = _match_lut(layer_name, shape, precision, rows)
        requested_key = {
            "layer_name": layer_name,
            "precision": precision,
            "C_in": shape.get("C_in"),
            "C_out": shape.get("C_out"),
            "C_in_aligned8": _align8(shape.get("C_in")),
            "C_out_aligned8": _align8(shape.get("C_out")),
            "kernel": shape.get("kernel"),
            "groups": shape.get("groups"),
        }
        if row is None:
            missing.append(requested_key)
            continue
        latency = float(row.get("latency_p50_ms") or row.get("latency_mean_ms") or 0.0)
        compute += latency
        by_precision[precision] += latency
        item = {
            **requested_key,
            "matched_lut_key": row.get("lut_key") or row.get("key_hash"),
            "match_type": match_type,
            "latency_p50_ms": latency,
        }
        matched.append(item)
        if not match_type.startswith("exact"):
            coarse.append(item)
    components = dict(buckets)
    components["T_compute_covered"] = compute
    t_raw = sum(float(v) for v in components.values()) if not missing else None
    payload = {
        "candidate_id": cid,
        "precision_profile_id": pid,
        "T_lut_raw": t_raw,
        **components,
        "matched_lut_keys": matched,
        "missing_keys": missing,
        "unavailable_keys": [],
        "coarse_keys": coarse,
        "uncertainty": 0.05 * len(coarse) + 0.5 * len(missing),
        "T_lut_by_precision": dict(by_precision),
        "is_width_changed_subnet": bool(audit.get("is_width_changed_subnet")),
        "is_pruned": bool(audit.get("is_pruned")),
        "param_keep_ratio": audit.get("param_keep_ratio"),
        "num_changed_conv_layers": audit.get("num_changed_conv_layers"),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / f"{cid}_{pid}.json", payload)
    return payload


def run(args: argparse.Namespace) -> dict[str, Any]:
    candidates_payload = load_json(args.candidates, {"candidates": []})
    profiles_payload = load_json(args.profiles, {"profiles": []})
    candidates_by_id = {str(c["candidate_id"]): c for c in candidates_payload.get("candidates", [])}
    # Prefer exported candidate files because they contain width_changed_onnx
    # and structure_audit_path.
    export_dir = Path(args.export_dir)
    for path in export_dir.glob("*/candidate.json"):
        data = load_json(path)
        if data.get("candidate_id"):
            candidates_by_id[str(data["candidate_id"])] = data
    rows = load_lut_rows(args.lut_v4, args.lut_v5)
    buckets = _bucket_sum(args.structural_buckets)
    out_dir = Path(args.output_dir)
    decomps: list[dict[str, Any]] = []
    for profile in profiles_payload.get("profiles", []):
        cid = str(profile["candidate_id"])
        candidate = candidates_by_id.get(cid)
        if not candidate:
            continue
        decomps.append(decompose_one(candidate, profile, rows, buckets, out_dir))
    stats = {
        "num_decompositions": len(decomps),
        "with_missing_keys": sum(1 for d in decomps if d.get("missing_keys")),
        "missing_key_count": sum(len(d.get("missing_keys") or []) for d in decomps),
        "with_int8_profile": sum(1 for p in profiles_payload.get("profiles", []) if has_int8(p.get("precision_profile") or {})),
    }
    lines = ["# Pruned Candidate LUT Decomposition v5", "", *[f"- {k}: {v}" for k, v in stats.items()]]
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2))
    return stats


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="outputs/latency_lut/pruned_width_changed_candidates_v5.json")
    parser.add_argument("--export-dir", "--export_dir", dest="export_dir", default="outputs/latency_lut/pruned_width_changed_onnx_v5")
    parser.add_argument("--profiles", default="outputs/latency_lut/pruned_precision_profiles_v5.json")
    parser.add_argument("--lut-v4", "--lut_v4", dest="lut_v4", default="outputs/latency_lut/layer_width_precision_lut_measurements_v4.jsonl")
    parser.add_argument("--lut-v5", "--lut_v5", dest="lut_v5", default="outputs/latency_lut/layer_width_precision_lut_measurements_v5.jsonl")
    parser.add_argument("--structural-buckets", "--structural_buckets", dest="structural_buckets", default="outputs/latency_lut/structural_op_latency_buckets_v4.json")
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", default="outputs/latency_lut/pruned_candidate_lut_decomposition_v5")
    parser.add_argument("--report", default="outputs/latency_lut/pruned_candidate_lut_decomposition_v5_report.md")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
