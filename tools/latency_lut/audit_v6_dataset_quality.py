from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REGIONS = [
    "pillar_or_voxel_encoder",
    "backbone",
    "shrink_or_neck",
    "pyramid_fusion",
    "detection_head",
    "geometry_or_plugin_or_scatter",
    "other",
]
PRECISIONS = ["FP32", "FP16", "INT8_QDQ"]


def _load_json(path: str | Path, default: Any = None) -> Any:
    p = Path(path)
    if not p.is_file():
        return {} if default is None else default
    return json.loads(p.read_text(encoding="utf-8"))


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.is_file():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: str | Path, payload: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        p.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with p.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _csv_value(row.get(k)) for k in fields})


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def _mean(values: list[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def _std(values: list[float]) -> float:
    return float(statistics.pstdev(values)) if len(values) > 1 else 0.0


def _corr(x: list[float], y: list[float]) -> float:
    if len(x) < 2 or len(y) < 2 or len(set(x)) < 2 or len(set(y)) < 2:
        return 0.0
    mx, my = _mean(x), _mean(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    denx = math.sqrt(sum((a - mx) ** 2 for a in x))
    deny = math.sqrt(sum((b - my) ** 2 for b in y))
    return float(num / (denx * deny)) if denx and deny else 0.0


def _rank(values: list[float]) -> list[float]:
    order = sorted((v, i) for i, v in enumerate(values))
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and order[j][0] == order[i][0]:
            j += 1
        rank = (i + j - 1) / 2.0
        for _, idx in order[i:j]:
            ranks[idx] = rank
        i = j
    return ranks


def _spearman(x: list[float], y: list[float]) -> float:
    return _corr(_rank(x), _rank(y))


def _region(name: str) -> str:
    low = str(name).lower()
    if any(k in low for k in ("cls_head", "reg_head", "dir_head", "head", "single_head")):
        return "detection_head"
    if any(k in low for k in ("scatter", "warp", "grid", "geometry", "bev_pool", "plugin")):
        return "geometry_or_plugin_or_scatter"
    if any(k in low for k in ("shrink", "neck", "downsample", "reduce", "compression")):
        return "shrink_or_neck"
    if any(k in low for k in ("fusion", "pyramid", "fuse", "multiscale")):
        return "pyramid_fusion"
    if any(k in low for k in ("pillar", "pfn", "voxel", "encoder_m1", "pillar_vfe")):
        return "pillar_or_voxel_encoder"
    if any(k in low for k in ("backbone", "resnet", "layer0", "layer1", "layer2", "layer3")):
        return "backbone"
    return "other"


def _norm_precision(value: Any) -> str:
    text = str(value).upper()
    if text in {"INT8", "INT8_QDQ", "TRT_INT8_QDQ"}:
        return "INT8_QDQ"
    if text in {"FP32", "TRT_FP32"}:
        return "FP32"
    return "FP16"


def _empty_region_counts(observed: bool = False) -> dict[str, dict[str, int]]:
    keys = ["FP32", "FP16", "INT8"] if observed else PRECISIONS
    return {region: {precision: 0 for precision in keys} for region in REGIONS}


def _profile_values(profile: dict[str, Any]) -> list[str]:
    values = [_norm_precision(profile.get("default", "FP16"))]
    values.extend(_norm_precision(v) for v in dict(profile.get("overrides") or {}).values())
    return values


def _stage_prefix(layer_name: str) -> str:
    name = str(layer_name).strip("/")
    parts = name.split("/")
    if parts and parts[0]:
        return parts[0]
    dot = name.split(".")
    return dot[0] if dot else "unknown"


def _structure_audit(row: dict[str, Any]) -> dict[str, Any]:
    audit = _load_json(row.get("structure_audit_path") or "", {})
    if audit:
        return audit
    return {
        "param_keep_ratio": row.get("param_keep_ratio"),
        "num_changed_conv_layers": row.get("num_changed_conv_layers"),
        "changed_conv_layers": [],
    }


def _result_for(row: dict[str, Any], results_dir: Path) -> dict[str, Any]:
    result_path = row.get("result_path")
    if result_path and Path(result_path).is_file():
        return _load_json(result_path, {})
    cid = str(row.get("candidate_id"))
    matches = sorted(results_dir.glob(f"{cid}.result.json"))
    return _load_json(matches[0], {}) if matches else {}


def _requested_by_region(profile: dict[str, Any]) -> dict[str, dict[str, int]]:
    counts = _empty_region_counts(False)
    # Record default FP16 as one global default bucket in other; explicit
    # overrides carry the actionable mixed precision coverage.
    counts["other"][_norm_precision(profile.get("default", "FP16"))] += 1
    for unit, precision in dict(profile.get("overrides") or {}).items():
        counts[_region(unit)][_norm_precision(precision)] += 1
    return counts


def _observed_by_region(result: dict[str, Any], row: dict[str, Any]) -> dict[str, dict[str, int]]:
    counts = _empty_region_counts(True)
    verification = result.get("precision_verification") or {}
    for unit, observations in dict(verification.get("observed_by_unit") or {}).items():
        region = _region(unit)
        for obs in observations or []:
            precision = str(obs.get("precision", "")).upper()
            if precision in {"FP32", "FP16", "INT8"}:
                counts[region][precision] += 1
    if not any(sum(v.values()) for v in counts.values()):
        # Fall back to aggregate counts when inspector per-unit rows are not in
        # the top-level result. This keeps the audit conservative.
        counts["other"]["FP32"] += int(row.get("observed_fp32_layers") or 0)
        counts["other"]["FP16"] += int(row.get("observed_fp16_layers") or 0)
        counts["other"]["INT8"] += int(row.get("observed_int8_layers") or 0)
    return counts


def audit_dataset(
    *,
    dataset: str | Path,
    results_dir: str | Path,
    decomposition_dir: str | Path,
    profiles: str | Path,
) -> dict[str, Any]:
    rows = _read_jsonl(dataset)
    results_root = Path(results_dir)
    failures: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    residual_rows: list[dict[str, Any]] = []
    precision_rows: list[dict[str, Any]] = []
    region_rows: list[dict[str, Any]] = []
    structure_rows: list[dict[str, Any]] = []
    basic = {
        "num_labels": len(rows),
        "num_width_changed": 0,
        "num_pruned": 0,
        "num_pruned_mixed": 0,
        "num_dtype_closure_valid": 0,
        "num_with_int8_qdq": 0,
        "num_with_fp32_fp16_int8_qdq": 0,
        "num_missing_lut_keys": 0,
        "num_unavailable_lut_keys": 0,
        "num_precision_verification_failures": 0,
        "num_eval_failures": 0,
    }
    latencies_p50: list[float] = []
    latencies_p90: list[float] = []
    maps: list[float] = []
    lut_raws: list[float] = []
    real_p50s: list[float] = []
    residuals: list[float] = []
    param_keep: list[float] = []
    changed_counts: list[int] = []
    structures: set[str] = set()
    profile_patterns: set[str] = set()
    prefix_counter: Counter[str] = Counter()
    precision_region_coverage = {region: {"labels_with_fp32": 0, "labels_with_fp16": 0, "labels_with_int8_qdq": 0} for region in REGIONS}
    int8_unit_coverage: Counter[str] = Counter()
    fp32_unit_coverage: Counter[str] = Counter()
    total_dtype_fixes = 0
    total_casts = 0
    labels_with_zero_dtype_fixes = 0
    closure_errors = 0
    max_casts = 0
    coarse_ratios: list[float] = []
    uncertainties: list[float] = []
    high_uncertainty = 0
    per_label_latency: list[dict[str, Any]] = []
    for row in rows:
        cid = str(row.get("candidate_id", ""))
        pid = str(row.get("precision_profile_id", ""))
        profile = dict(row.get("precision_profile") or {})
        structures.add(str(row.get("base_structure_candidate_id") or cid.split("__", 1)[0]))
        profile_patterns.add(json.dumps(profile, sort_keys=True))
        checks = {
            "is_width_changed_subnet": row.get("is_width_changed_subnet") is True,
            "is_pruned": row.get("is_pruned") is True,
            "is_pruned_mixed": row.get("is_pruned_mixed") is True,
            "route2_explicit_precision": row.get("route2_explicit_precision") is True,
            "dtype_closure_valid": row.get("dtype_closure_valid") is True,
            "missing_keys": not row.get("missing_keys"),
            "unavailable_keys": not row.get("unavailable_keys"),
            "precision_verification_failures": not row.get("precision_verification_failures"),
            "T_lut_raw": float(row.get("T_lut_raw") or 0.0) > 0,
            "T_real_p50": float(row.get("T_real_p50") or 0.0) > 0,
        }
        for check, ok in checks.items():
            if not ok:
                failures.append({"candidate_id": cid, "precision_profile_id": pid, "failed_check": check, "reason": f"{check} check failed"})
        basic["num_width_changed"] += int(row.get("is_width_changed_subnet") is True)
        basic["num_pruned"] += int(row.get("is_pruned") is True)
        basic["num_pruned_mixed"] += int(row.get("is_pruned_mixed") is True)
        basic["num_dtype_closure_valid"] += int(row.get("dtype_closure_valid") is True)
        basic["num_with_int8_qdq"] += int(bool(row.get("has_int8_qdq")))
        basic["num_with_fp32_fp16_int8_qdq"] += int(bool(row.get("has_fp32") and row.get("has_fp16") and row.get("has_int8_qdq")))
        basic["num_missing_lut_keys"] += len(row.get("missing_keys") or [])
        basic["num_unavailable_lut_keys"] += len(row.get("unavailable_keys") or [])
        basic["num_precision_verification_failures"] += len(row.get("precision_verification_failures") or [])
        if not row.get("T_real_p50"):
            basic["num_eval_failures"] += 1
        audit = _structure_audit(row)
        changed = list(audit.get("changed_conv_layers") or [])
        for item in changed:
            prefix_counter[_stage_prefix(item.get("layer_name") or "")] += 1
        candidate_shapes = audit.get("candidate_conv_shapes") or {}
        c_out_values = sorted({int(v.get("C_out")) for v in candidate_shapes.values() if isinstance(v, dict) and v.get("C_out") is not None})
        pk = float(audit.get("param_keep_ratio") or row.get("param_keep_ratio") or 0.0)
        nchanged = int(audit.get("num_changed_conv_layers") or row.get("num_changed_conv_layers") or 0)
        param_keep.append(pk)
        changed_counts.append(nchanged)
        structure_rows.append(
            {
                "candidate_id": cid,
                "precision_profile_id": pid,
                "structure_candidate_id": row.get("base_structure_candidate_id"),
                "param_keep_ratio": pk,
                "num_changed_conv_layers": nchanged,
                "changed_conv_layers": [x.get("layer_name") for x in changed],
                "changed_layer_count_by_stage_or_prefix": dict(Counter(_stage_prefix(x.get("layer_name") or "") for x in changed)),
                "min_C_out_new": min(c_out_values) if c_out_values else 0,
                "max_C_out_new": max(c_out_values) if c_out_values else 0,
                "unique_C_out_new": c_out_values,
                "num_root_node_domains_changed": 0,
                "root_node_domains_changed": [],
            }
        )
        p50 = float(row.get("T_real_p50") or 0.0)
        p90 = float(row.get("T_real_p90") or 0.0)
        mean = float(row.get("T_real_mean") or 0.0)
        map_value = float(row.get("mAP") or 0.0)
        t_lut = float(row.get("T_lut_raw") or 0.0)
        residual = p50 - t_lut
        latencies_p50.append(p50)
        latencies_p90.append(p90)
        maps.append(map_value)
        lut_raws.append(t_lut)
        real_p50s.append(p50)
        residuals.append(residual)
        ratio = float(p90 / p50) if p50 else 0.0
        per_label_latency.append(
            {
                "candidate_id": cid,
                "precision_profile_id": pid,
                "T_real_p50": p50,
                "T_real_p90": p90,
                "T_real_mean": mean,
                "mAP": map_value,
                "eval_frames": int(row.get("num_val_frames") or 50),
                "latency_outlier": False,
                "map_outlier": map_value < 0.1,
                "p90_p50_ratio": ratio,
            }
        )
        decomp = _load_json(row.get("lut_decomposition_path") or "", {})
        matched = len(decomp.get("matched_lut_keys") or [])
        coarse = len(decomp.get("coarse_keys") or [])
        coarse_ratio = float(coarse / matched) if matched else 0.0
        coarse_ratios.append(coarse_ratio)
        uncertainty = float(decomp.get("uncertainty") or row.get("uncertainty") or 0.0)
        uncertainties.append(uncertainty)
        high_uncertainty += int(uncertainty > 1.0)
        residual_row = {
            "candidate_id": cid,
            "precision_profile_id": pid,
            "T_lut_raw": t_lut,
            "T_real_p50": p50,
            "T_real_p90": p90,
            "residual_p50": residual,
            "abs_error": abs(residual),
            "ape": abs(residual) / p50 if p50 else None,
        }
        residual_rows.append(residual_row)
        req_region = _requested_by_region(profile)
        result = _result_for(row, results_root)
        obs_region = _observed_by_region(result, row)
        for unit, precision in dict(profile.get("overrides") or {}).items():
            norm = _norm_precision(precision)
            if norm == "INT8_QDQ":
                int8_unit_coverage[str(unit)] += 1
            elif norm == "FP32":
                fp32_unit_coverage[str(unit)] += 1
        for region, counts in req_region.items():
            if counts["FP32"]:
                precision_region_coverage[region]["labels_with_fp32"] += 1
            if counts["FP16"]:
                precision_region_coverage[region]["labels_with_fp16"] += 1
            if counts["INT8_QDQ"]:
                precision_region_coverage[region]["labels_with_int8_qdq"] += 1
        precision_rows.append(
            {
                "candidate_id": cid,
                "precision_profile_id": pid,
                "num_requested_fp32_units": sum(1 for v in dict(profile.get("overrides") or {}).values() if _norm_precision(v) == "FP32"),
                "num_requested_fp16_units": 1 + sum(1 for v in dict(profile.get("overrides") or {}).values() if _norm_precision(v) == "FP16"),
                "num_requested_int8_qdq_units": sum(1 for v in dict(profile.get("overrides") or {}).values() if _norm_precision(v) == "INT8_QDQ"),
                "num_observed_fp32_layers": int(row.get("observed_fp32_layers") or 0),
                "num_observed_fp16_layers": int(row.get("observed_fp16_layers") or 0),
                "num_observed_int8_layers": int(row.get("observed_int8_layers") or 0),
                "requested_vs_observed_match": not row.get("precision_verification_failures"),
                "mismatched_units": [],
                "precision_verification_failures": row.get("precision_verification_failures") or [],
            }
        )
        region_rows.append(
            {
                "candidate_id": cid,
                "precision_profile_id": pid,
                "requested_precision_by_region": req_region,
                "observed_precision_by_region": obs_region,
                "requested_vs_observed_region_match": not row.get("precision_verification_failures"),
                "region_mismatches": [],
            }
        )
        closure = _load_json(row.get("dtype_closure_report_path") or "", {})
        validation = _load_json(row.get("dtype_closure_validation_report_path") or "", {})
        fixes = int(closure.get("num_dtype_mismatches_fixed") or row.get("num_dtype_mismatches_fixed") or 0)
        casts = int(closure.get("num_cast_inserted") or row.get("num_cast_inserted_by_dtype_closure") or 0)
        total_dtype_fixes += fixes
        total_casts += casts
        max_casts = max(max_casts, casts)
        labels_with_zero_dtype_fixes += int(fixes == 0)
        closure_errors += int(validation.get("num_errors") or 0)
        label_rows.append(
            {
                "candidate_id": cid,
                "precision_profile_id": pid,
                "structure_candidate_id": row.get("base_structure_candidate_id"),
                "param_keep_ratio": pk,
                "num_changed_conv_layers": nchanged,
                "has_fp32": row.get("has_fp32"),
                "has_fp16": row.get("has_fp16"),
                "has_int8_qdq": row.get("has_int8_qdq"),
                "observed_fp32_layers": row.get("observed_fp32_layers"),
                "observed_fp16_layers": row.get("observed_fp16_layers"),
                "observed_int8_layers": row.get("observed_int8_layers"),
                "dtype_closure_valid": row.get("dtype_closure_valid"),
                "T_lut_raw": t_lut,
                "T_real_p50": p50,
                "T_real_p90": p90,
                "mAP": map_value,
                "num_missing_keys": len(row.get("missing_keys") or []),
                "num_coarse_keys": coarse,
                "coarse_key_ratio": coarse_ratio,
                "uncertainty": uncertainty,
                "residual_p50": residual,
                "requested_precision_by_region": req_region,
                "observed_precision_by_region": obs_region,
            }
        )
    p50_mean = _mean(latencies_p50)
    p50_std = _std(latencies_p50)
    map_mean = _mean(maps)
    map_std = _std(maps)
    for item in per_label_latency:
        item["latency_outlier"] = abs(float(item["T_real_p50"]) - p50_mean) > 2 * p50_std if p50_std else False
        item["latency_unstable"] = float(item["p90_p50_ratio"]) > 1.2
        item["map_outlier"] = abs(float(item["mAP"]) - map_mean) > 2 * map_std if map_std else item["map_outlier"]
    latency_stats = {
        "T_real_p50_min": min(latencies_p50) if latencies_p50 else 0.0,
        "T_real_p50_max": max(latencies_p50) if latencies_p50 else 0.0,
        "T_real_p50_mean": p50_mean,
        "T_real_p50_std": p50_std,
        "T_real_p90_min": min(latencies_p90) if latencies_p90 else 0.0,
        "T_real_p90_max": max(latencies_p90) if latencies_p90 else 0.0,
        "T_real_p90_mean": _mean(latencies_p90),
        "T_real_p90_std": _std(latencies_p90),
        "mAP_min": min(maps) if maps else 0.0,
        "mAP_max": max(maps) if maps else 0.0,
        "mAP_mean": map_mean,
        "mAP_std": map_std,
        "num_latency_outliers": sum(1 for x in per_label_latency if x["latency_outlier"]),
        "num_map_outliers": sum(1 for x in per_label_latency if x["map_outlier"]),
        "num_latency_unstable": sum(1 for x in per_label_latency if x["latency_unstable"]),
    }
    structure_diversity_pass = len(structures) >= 3 and len({round(x, 4) for x in param_keep}) >= 3 and _mean(changed_counts) > 2 and all(x < 1.0 for x in param_keep) and all(x > 0 for x in changed_counts)
    structure_stats = {
        "param_keep_ratio_min": min(param_keep) if param_keep else 0.0,
        "param_keep_ratio_max": max(param_keep) if param_keep else 0.0,
        "param_keep_ratio_mean": _mean(param_keep),
        "num_changed_conv_layers_min": min(changed_counts) if changed_counts else 0,
        "num_changed_conv_layers_max": max(changed_counts) if changed_counts else 0,
        "num_changed_conv_layers_mean": _mean([float(x) for x in changed_counts]),
        "unique_structure_candidates": len(structures),
        "unique_param_keep_ratios": sorted({round(x, 6) for x in param_keep}),
        "changed_layer_prefix_coverage": dict(prefix_counter),
        "structure_diversity_pass": structure_diversity_pass,
        "structure_diversity_reason": "pass" if structure_diversity_pass else "structure candidates or param keep ratios are not diverse enough",
    }
    regions_never_int8 = [r for r, c in precision_region_coverage.items() if c["labels_with_int8_qdq"] == 0]
    regions_never_fp32 = [r for r, c in precision_region_coverage.items() if c["labels_with_fp32"] == 0]
    regions_only_fp16 = [r for r, c in precision_region_coverage.items() if c["labels_with_fp16"] and not c["labels_with_fp32"] and not c["labels_with_int8_qdq"]]
    precision_region_pass = (
        precision_region_coverage["backbone"]["labels_with_int8_qdq"] > 0
        and precision_region_coverage["shrink_or_neck"]["labels_with_int8_qdq"] > 0
        and precision_region_coverage["detection_head"]["labels_with_fp32"] > 0
        and len(int8_unit_coverage) >= 2
    )
    precision_region_summary = {
        "precision_region_coverage": precision_region_coverage,
        "regions_never_int8": regions_never_int8,
        "regions_never_fp32": regions_never_fp32,
        "regions_only_fp16": regions_only_fp16,
        "precision_region_coverage_pass": precision_region_pass,
        "precision_region_coverage_reason": "pass" if precision_region_pass else "INT8/FP32 requested coverage is too concentrated by region",
    }
    precision_summary = {
        "labels_with_fp32": sum(1 for r in rows if r.get("has_fp32")),
        "labels_with_fp16": sum(1 for r in rows if r.get("has_fp16")),
        "labels_with_int8_qdq": sum(1 for r in rows if r.get("has_int8_qdq")),
        "labels_with_all_three": sum(1 for r in rows if r.get("has_fp32") and r.get("has_fp16") and r.get("has_int8_qdq")),
        "observed_fp32_layers_total": sum(int(r.get("observed_fp32_layers") or 0) for r in rows),
        "observed_fp16_layers_total": sum(int(r.get("observed_fp16_layers") or 0) for r in rows),
        "observed_int8_layers_total": sum(int(r.get("observed_int8_layers") or 0) for r in rows),
        "int8_unit_coverage": dict(int8_unit_coverage),
        "fp32_unit_coverage": dict(fp32_unit_coverage),
        "precision_coverage_pass": sum(1 for r in rows if r.get("has_int8_qdq")) >= 5 and sum(1 for r in rows if r.get("has_fp32") and r.get("has_fp16") and r.get("has_int8_qdq")) >= 5 and len(int8_unit_coverage) >= 2,
        "precision_coverage_reason": "pass" if len(int8_unit_coverage) >= 2 else "INT8 profiles are concentrated in too few units",
    }
    dtype_summary = {
        "dtype_closure_valid_labels": basic["num_dtype_closure_valid"],
        "total_dtype_mismatches_fixed": total_dtype_fixes,
        "total_cast_inserted_by_dtype_closure": total_casts,
        "max_cast_inserted_single_label": max_casts,
        "labels_with_zero_dtype_fixes": labels_with_zero_dtype_fixes,
        "labels_with_dtype_closure_errors": closure_errors,
        "dtype_closure_quality_pass": basic["num_dtype_closure_valid"] == len(rows) and closure_errors == 0 and total_dtype_fixes > 0,
        "dtype_closure_quality_reason": "pass" if basic["num_dtype_closure_valid"] == len(rows) and closure_errors == 0 and total_dtype_fixes > 0 else "closure reports missing or errors present",
    }
    lut_summary = {
        "T_lut_raw_min": min(lut_raws) if lut_raws else 0.0,
        "T_lut_raw_max": max(lut_raws) if lut_raws else 0.0,
        "T_lut_raw_mean": _mean(lut_raws),
        "coarse_key_ratio_mean": _mean(coarse_ratios),
        "max_uncertainty": max(uncertainties) if uncertainties else 0.0,
        "labels_with_missing_keys": sum(1 for r in rows if r.get("missing_keys")),
        "labels_with_unavailable_keys": sum(1 for r in rows if r.get("unavailable_keys")),
        "labels_with_high_uncertainty": high_uncertainty,
        "lut_decomposition_quality_pass": all(x > 0 for x in lut_raws) and basic["num_missing_lut_keys"] == 0 and basic["num_unavailable_lut_keys"] == 0 and _mean(coarse_ratios) <= 0.3,
        "lut_decomposition_quality_reason": "pass" if _mean(coarse_ratios) <= 0.3 else "coarse key ratio is too high",
    }
    alignment = {
        "pearson_corr_lut_vs_real_p50": _corr(lut_raws, real_p50s),
        "spearman_corr_lut_vs_real_p50": _spearman(lut_raws, real_p50s),
        "raw_lut_mae_ms": round(_mean([abs(x) for x in residuals]), 12),
        "raw_lut_mape": _mean([abs(res) / real for res, real in zip(residuals, real_p50s) if real]),
        "residual_min": min(residuals) if residuals else 0.0,
        "residual_max": max(residuals) if residuals else 0.0,
        "residual_mean": _mean(residuals),
        "residual_std": _std(residuals),
        "worst_overestimate_labels": sorted(residual_rows, key=lambda x: x["residual_p50"])[:3],
        "worst_underestimate_labels": sorted(residual_rows, key=lambda x: x["residual_p50"], reverse=True)[:3],
    }
    alignment["lut_real_alignment_pass"] = alignment["spearman_corr_lut_vs_real_p50"] >= 0.3 and (max(real_p50s) - min(real_p50s) if real_p50s else 0.0) >= 0.05
    alignment["lut_real_alignment_reason"] = "pass" if alignment["lut_real_alignment_pass"] else "raw LUT has weak rank alignment or narrow latency range"
    failure_rows, failure_summary = _failure_analysis(results_root)
    minimum_checks = [
        len(rows) >= 15,
        basic["num_width_changed"] == len(rows),
        basic["num_pruned"] == len(rows),
        basic["num_pruned_mixed"] == len(rows),
        basic["num_dtype_closure_valid"] == len(rows),
        basic["num_with_int8_qdq"] >= 5,
        basic["num_with_fp32_fp16_int8_qdq"] >= 5,
        basic["num_missing_lut_keys"] == 0,
        basic["num_unavailable_lut_keys"] == 0,
        basic["num_precision_verification_failures"] == 0,
        all(x > 0 for x in lut_raws),
        all(x > 0 for x in real_p50s),
        len(structures) >= 3,
        len(profile_patterns) >= 3,
        precision_region_summary["precision_region_coverage_pass"],
        dtype_summary["dtype_closure_quality_pass"],
        lut_summary["lut_decomposition_quality_pass"],
        precision_region_coverage["backbone"]["labels_with_int8_qdq"] > 0,
        precision_region_coverage["shrink_or_neck"]["labels_with_int8_qdq"] > 0,
        precision_region_coverage["detection_head"]["labels_with_fp32"] > 0,
    ]
    warnings: list[str] = []
    if not alignment["lut_real_alignment_pass"]:
        warnings.append("raw LUT vs T_real_p50 alignment is weak; calibration should start with interpretable residual analysis, not GA use")
    if not structure_diversity_pass:
        warnings.append("structure diversity is limited")
    if failure_summary.get("failed_samples_concentrated"):
        warnings.append("failed samples are concentrated and should be considered before broadening sampling")
    decision_pass = all(minimum_checks)
    decision = {
        "dataset_quality_pass": decision_pass,
        "can_train_latency_proxy_next": decision_pass,
        "recommended_next_step": "calibrate latency proxy with v6 pruned+mixed full-engine labels, starting from interpretable affine/residual correction before Ridge/MLP" if decision_pass else "expand or repair v6 dataset before latency proxy calibration",
        "blocking_issues": [] if decision_pass else [name for name, ok in zip(
            [
                "num_labels", "width_changed", "pruned", "pruned_mixed", "dtype_closure_valid", "int8_count", "all_three_count",
                "missing_lut", "unavailable_lut", "precision_failures", "T_lut_raw", "T_real_p50", "structure_candidates",
                "precision_patterns", "precision_region_coverage", "dtype_closure_quality", "lut_decomposition_quality",
                "backbone_int8", "shrink_or_neck_int8", "detection_head_fp32",
            ],
            minimum_checks,
        ) if not ok],
        "warnings": warnings,
    }
    return {
        "basic": basic,
        "failed_label_checks": failures,
        "structure": structure_stats,
        "per_label_structure": structure_rows,
        "latency_map_distribution": latency_stats,
        "per_label_latency": per_label_latency,
        "precision_region": precision_region_summary,
        "per_label_precision_region": region_rows,
        "precision": precision_summary,
        "per_label_precision": precision_rows,
        "dtype_closure": dtype_summary,
        "lut_decomposition": lut_summary,
        "latency_alignment": alignment,
        "failures": failure_summary,
        "failed_samples": failure_rows,
        "tables": {
            "labels": label_rows,
            "residuals": residual_rows,
            "precision": precision_rows,
            "precision_region": region_rows,
            "structure": structure_rows,
            "failure": failure_rows,
        },
        "final_decision": decision,
    }


def _failure_analysis(results_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(results_dir.glob("*.result.json")):
        data = _load_json(path, {})
        if data.get("success"):
            continue
        cid = str(data.get("candidate_id") or path.stem.replace(".result", ""))
        prep = data.get("route2_prepare") or {}
        validation = prep.get("validation") or prep.get("dtype_closure_validation") or {}
        errors = validation.get("errors") or []
        first_error = errors[0] if errors else {}
        status = str(data.get("status") or "")
        build = data.get("build_report") or {}
        attempts = build.get("attempts") or []
        precision = "INT8_QDQ" if "int8" in cid.lower() else "mixed"
        row = {
            "candidate_id": cid,
            "precision_profile_id": cid.split("__", 1)[1] if "__" in cid else "",
            "structure_candidate_id": cid.split("__", 1)[0],
            "failed_stage": data.get("failed_stage"),
            "error_type": status,
            "error_message_tail": str(data.get("error") or build.get("error") or "")[-500:],
            "op_type_involved": first_error.get("op_type", ""),
            "onnx_node_name": first_error.get("node", ""),
            "trt_layer_name": "",
            "region": _region(first_error.get("node", cid)),
            "precision_involved": precision,
            "param_keep_ratio": None,
            "num_changed_conv_layers": None,
            "root_node_domains_changed": [],
            "dtype_closure_valid": bool(validation.get("valid")),
            "num_dtype_closure_errors": int(validation.get("num_errors") or 0),
            "num_missing_lut_keys": 0,
        }
        if attempts:
            ver = attempts[0].get("precision_verification") or {}
            failures = ver.get("failures") or []
            if failures:
                unit = failures[0].get("unit", "")
                row["region"] = _region(unit)
                row["trt_layer_name"] = json.dumps(failures[0].get("bad_layers") or failures[0].get("unknown_layers") or [])[:300]
        rows.append(row)
    by_stage = Counter(str(r["failed_stage"]) for r in rows)
    by_op = Counter(str(r["op_type_involved"]) for r in rows if r["op_type_involved"])
    by_region = Counter(str(r["region"]) for r in rows)
    by_precision = Counter(str(r["precision_involved"]) for r in rows)
    by_structure = Counter(str(r["structure_candidate_id"]) for r in rows)
    dominant_stage, stage_count = by_stage.most_common(1)[0] if by_stage else ("", 0)
    dominant_region, region_count = by_region.most_common(1)[0] if by_region else ("", 0)
    concentrated = bool(rows) and (stage_count / len(rows) >= 0.7 or region_count / len(rows) >= 0.7)
    summary = {
        "num_failed_samples": len(rows),
        "failures_by_stage": {
            "pruned_qdq_dtype_closure_failed": by_stage.get("pruned_qdq_dtype_closure_failed", 0),
            "engine_build_failed": by_stage.get("engine_build", 0),
            "engine_inspector_verification_failed": by_stage.get("precision_verification", 0),
            "runner_eval_failed": by_stage.get("runner_eval", 0),
            "lut_missing_key_failed": by_stage.get("lut_decomposition", 0),
            "int8_scale_collection_failed": by_stage.get("activation_scale_autofill", 0),
            "onnx_export_failed": by_stage.get("onnx_export", 0),
        },
        "failures_by_op_type": dict(by_op),
        "failures_by_region": dict(by_region),
        "failures_by_precision_type": dict(by_precision),
        "failures_by_structure_candidate": dict(by_structure),
        "failures_by_root_node_domain": {},
        "dominant_failure_stage": dominant_stage,
        "dominant_failure_op_type": by_op.most_common(1)[0][0] if by_op else "",
        "dominant_failure_region": dominant_region,
        "dominant_failure_structure": by_structure.most_common(1)[0][0] if by_structure else "",
        "failed_samples_concentrated": concentrated,
        "concentration_type": "region" if concentrated else "none",
        "main_concentration": dominant_region if concentrated else "",
        "recommended_fix": "inspect EngineInspector precision mapping and remaining dtype closure failures before expanding unsupported INT8 regions" if concentrated else "continue targeted expansion",
        "failure_concentration_pass": not concentrated,
        "failure_concentration_findings": ["failures concentrated by stage or region"] if concentrated else [],
    }
    return rows, summary


def _write_outputs(report: dict[str, Any], output_json: str | Path, output_md: str | Path) -> None:
    _write_json(output_json, {k: v for k, v in report.items() if k != "tables"})
    out_dir = Path(output_json).parent
    _write_csv(out_dir / "v6_dataset_labels_table.csv", report["tables"]["labels"])
    _write_csv(out_dir / "v6_lut_vs_real_residuals.csv", report["tables"]["residuals"])
    _write_csv(out_dir / "v6_precision_coverage_table.csv", report["tables"]["precision"])
    _write_csv(out_dir / "v6_precision_region_coverage_table.csv", report["tables"]["precision_region"])
    _write_csv(out_dir / "v6_structure_coverage_table.csv", report["tables"]["structure"])
    _write_csv(out_dir / "v6_failure_concentration_table.csv", report["tables"]["failure"])
    md = [
        "# V6 Dataset Quality Audit",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(report["final_decision"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## Basic",
        "",
        "```json",
        json.dumps(report["basic"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## Structure Diversity",
        "",
        "```json",
        json.dumps(report["structure"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## LUT vs Real",
        "",
        "```json",
        json.dumps(report["latency_alignment"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## Failure Concentration",
        "",
        "```json",
        json.dumps(report["failures"], ensure_ascii=False, indent=2),
        "```",
    ]
    Path(output_md).parent.mkdir(parents=True, exist_ok=True)
    Path(output_md).write_text("\n".join(md) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="outputs/latency_lut/full_engine_calibration_dataset_v6.jsonl")
    parser.add_argument("--results-dir", "--results_dir", dest="results_dir", default="outputs/latency_lut/pruned_route2_results_v6")
    parser.add_argument("--decomposition-dir", "--decomposition_dir", dest="decomposition_dir", default="outputs/latency_lut/pruned_candidate_lut_decomposition_v6")
    parser.add_argument("--profiles", default="outputs/latency_lut/pruned_precision_profiles_v6.json")
    parser.add_argument("--output-json", "--output_json", dest="output_json", default="outputs/latency_lut/v6_dataset_quality_audit.json")
    parser.add_argument("--output-md", "--output_md", dest="output_md", default="outputs/latency_lut/v6_dataset_quality_audit.md")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = audit_dataset(dataset=args.dataset, results_dir=args.results_dir, decomposition_dir=args.decomposition_dir, profiles=args.profiles)
    _write_outputs(report, args.output_json, args.output_md)
    print(json.dumps(report["final_decision"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
