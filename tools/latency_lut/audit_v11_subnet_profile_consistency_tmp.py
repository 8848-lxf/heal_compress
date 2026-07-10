#!/usr/bin/env python3
"""Audit v11 mixed-precision LUT subnet/profile artifacts.

This script is intentionally read-only with respect to subnet/profile artifacts.
It writes only audit outputs under the dataset root.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


DEFAULT_ROOT = Path("outputs/latency_lut/v11_mixed_precision_lut_dataset_trt_full")
PROFILE_IDS = [f"profile_{i:03d}" for i in range(4)]
SUBNET_IDS = [f"subnet_{i:03d}" for i in range(50)]
AP_KEYS = ["AP@0.03", "AP@0.30", "AP@0.50", "AP@0.70", "mAP"]
LATENCY_KEYS = [
    "forward_mean_ms",
    "forward_p50_ms",
    "data_to_gpu_mean_ms",
    "data_to_gpu_p50_ms",
    "postprocess_mean_ms",
    "postprocess_p50_ms",
    "total_mean_ms",
    "total_p50_ms",
]


def read_json(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:  # pragma: no cover - audit records errors
        return {"__json_read_error__": repr(exc)}


def json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_hash(obj: Any, n: int = 16) -> str:
    return hashlib.sha256(json_dumps(obj).encode("utf-8")).hexdigest()[:n]


def file_sha256(path: Path) -> Optional[str]:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_size(path: Path) -> Optional[int]:
    return path.stat().st_size if path.exists() and path.is_file() else None


def file_mtime(path: Path) -> Optional[float]:
    return path.stat().st_mtime if path.exists() and path.is_file() else None


def short_hash(value: Optional[str], n: int = 16) -> str:
    return value[:n] if value else ""


def get_nested(obj: Any, path: Iterable[str], default: Any = None) -> Any:
    cur = obj
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def truthy_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    if isinstance(value, (int, float)):
        return bool(value)
    return False


def flatten_key_values(obj: Any, keys: Iterable[str]) -> List[str]:
    wanted = set(keys)
    out: List[str] = []

    def rec(x: Any) -> None:
        if isinstance(x, dict):
            for k, v in x.items():
                if k in wanted and isinstance(v, (str, int, float, bool)):
                    out.append(str(v))
                rec(v)
        elif isinstance(x, list):
            for item in x:
                rec(item)

    rec(obj)
    return out


def find_strings_containing(obj: Any, needle: str) -> List[str]:
    out: List[str] = []

    def rec(x: Any) -> None:
        if isinstance(x, dict):
            for v in x.values():
                rec(v)
        elif isinstance(x, list):
            for item in x:
                rec(item)
        elif isinstance(x, str) and needle in x:
            out.append(x)

    rec(obj)
    return out


def onnx_summary(path: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "onnx_parse_success": False,
        "onnx_parse_error": "",
        "onnx_conv_node_count": None,
        "onnx_gemm_node_count": None,
        "onnx_matmul_node_count": None,
        "onnx_initializer_count": None,
        "onnx_initializer_shape_hash": "",
    }
    if not path.exists():
        return out
    try:
        import onnx  # type: ignore

        model = onnx.load(str(path), load_external_data=False)
        counts = Counter(node.op_type for node in model.graph.node)
        init_summary = [
            {
                "name": init.name,
                "dims": [int(d) for d in init.dims],
                "data_type": int(init.data_type),
            }
            for init in model.graph.initializer
        ]
        out.update(
            {
                "onnx_parse_success": True,
                "onnx_conv_node_count": counts.get("Conv", 0),
                "onnx_gemm_node_count": counts.get("Gemm", 0),
                "onnx_matmul_node_count": counts.get("MatMul", 0),
                "onnx_initializer_count": len(init_summary),
                "onnx_initializer_shape_hash": stable_hash(init_summary),
            }
        )
    except Exception as exc:  # pragma: no cover - audit records errors
        out["onnx_parse_error"] = repr(exc)
    return out


def shape_payload_from_manifest(manifest: Any, before_after_file: Any) -> Tuple[Any, str]:
    if isinstance(manifest, dict):
        for key in ("before_after_shapes", "module_shapes", "module_channel_before_after"):
            value = manifest.get(key)
            if value:
                return value, f"pruning_manifest.{key}"
    if before_after_file:
        return before_after_file, "module_channel_before_after.json"
    return None, ""


def audit_subnet(root: Path, subnet_id: str) -> Dict[str, Any]:
    subnet_dir = root / "subnets" / subnet_id
    manifest_path = subnet_dir / "pruning_manifest.json"
    manifest = read_json(manifest_path)
    before_after_file = read_json(subnet_dir / "module_channel_before_after.json")
    shape_payload, shape_source = shape_payload_from_manifest(manifest, before_after_file)
    onnx_path = subnet_dir / "onnx" / "model_signal_maxk.onnx"
    export_report = read_json(subnet_dir / "onnx" / "real_onnx_export_report.json") or {}
    if not isinstance(export_report, dict):
        export_report = {}

    other_model_files: Dict[str, Dict[str, Any]] = {}
    for name in ("pruned_model.pth", "model.pth", "pruned_state_dict_with_manifest.pth"):
        p = subnet_dir / name
        if p.exists():
            other_model_files[name] = {"size": file_size(p), "sha256": file_sha256(p)}

    pmo = subnet_dir / "pruned_model_object.pth"
    d: Dict[str, Any] = {
        "subnet_id": subnet_id,
        "subnet_dir_exists": subnet_dir.exists(),
        "pruning_manifest_exists": manifest_path.exists(),
        "structure_hash": get_nested(manifest, ["structure_hash"], ""),
        "actual_param_prune_ratio": get_nested(manifest, ["actual_param_prune_ratio"], None),
        "actual_param_prune_ratio_on_searchable_surface": get_nested(
            manifest, ["actual_param_prune_ratio_on_searchable_surface"], None
        ),
        "actual_channel_prune_ratio": get_nested(manifest, ["actual_channel_prune_ratio"], None),
        "actual_channel_prune_ratio_on_searchable_surface": get_nested(
            manifest, ["actual_channel_prune_ratio_on_searchable_surface"], None
        ),
        "shape_hash": stable_hash(shape_payload) if shape_payload is not None else "",
        "shape_hash_source": shape_source,
        "pruned_model_object_exists": pmo.exists(),
        "pruned_model_object_size": file_size(pmo),
        "pruned_model_object_sha256": file_sha256(pmo),
        "other_pruned_model_files": other_model_files,
        "model_signal_maxk_onnx_exists": onnx_path.exists(),
        "model_signal_maxk_onnx_size": file_size(onnx_path),
        "model_signal_maxk_onnx_sha256": file_sha256(onnx_path),
        "real_onnx_export_report_exists": (subnet_dir / "onnx" / "real_onnx_export_report.json").exists(),
        "real_onnx_export_input_names": export_report.get("input_names"),
        "real_onnx_export_output_names": export_report.get("output_names"),
        "real_onnx_export_source_exporter": export_report.get("source_exporter"),
        "real_onnx_export_success": export_report.get("export_success"),
    }
    d.update(onnx_summary(onnx_path))
    return d


def count_inserted_qdq(qdq: Any) -> Optional[int]:
    if not isinstance(qdq, dict):
        return None
    nodes = qdq.get("inserted_qdq_nodes")
    if isinstance(nodes, list):
        return len(nodes)
    for key in ("qdq_node_count", "quantize_linear_count"):
        if isinstance(qdq.get(key), int):
            return int(qdq[key])
    return None


def first_mismatch_layers(precision_report: Any, limit: int = 10) -> List[Any]:
    layers = get_nested(precision_report, ["mismatch_layers"], [])
    if isinstance(layers, list):
        return layers[:limit]
    return []


def get_latency(eval_report: Any, latency_report: Any) -> Dict[str, Any]:
    src = get_nested(eval_report, ["latency_summary"], None)
    if not isinstance(src, dict):
        src = latency_report if isinstance(latency_report, dict) else {}
    return {k: src.get(k) for k in LATENCY_KEYS}


def get_ap(eval_report: Any) -> Dict[str, Any]:
    ap = get_nested(eval_report, ["ap"], {})
    if not isinstance(ap, dict):
        ap = {}
    return {k: ap.get(k) for k in AP_KEYS}


def profile_gate_complete(row: Dict[str, Any]) -> bool:
    return bool(
        row.get("engine_exists")
        and row.get("eval_success") is True
        and row.get("evaluated_frames")
        and row.get("synthetic_used") is False
        and row.get("validation_dataloader_used") is True
        and row.get("engine_structure_check_passed") is not False
        and row.get("engine_precision_realization_passed") is not False
        and row.get("build_success") is not False
    )


def audit_profile(root: Path, subnet_id: str, profile_id: str, subnet: Dict[str, Any]) -> Dict[str, Any]:
    profile_dir = root / "subnets" / subnet_id / profile_id
    profile = read_json(profile_dir / "mixed_precision_profile.json") or {}
    qdq = read_json(profile_dir / "qdq_insert_report.json") or {}
    structure = read_json(profile_dir / "engine_structure_check_report.json") or {}
    precision = read_json(profile_dir / "engine_precision_realization_report.json") or {}
    smoke = read_json(profile_dir / "trt_smoke_report.json") or {}
    eval_report = read_json(profile_dir / "eval_report.json") or {}
    latency_report = read_json(profile_dir / "eval_latency_summary.json") or {}
    worker = read_json(profile_dir / "worker_result.json") or {}
    failure = read_json(profile_dir / "profile_failure_report.json") or {}
    build = read_json(profile_dir / "build_report.json") or {}

    if not isinstance(profile, dict):
        profile = {}
    if not isinstance(qdq, dict):
        qdq = {}
    if not isinstance(structure, dict):
        structure = {}
    if not isinstance(precision, dict):
        precision = {}
    if not isinstance(smoke, dict):
        smoke = {}
    if not isinstance(eval_report, dict):
        eval_report = {}
    if not isinstance(latency_report, dict):
        latency_report = {}
    if not isinstance(worker, dict):
        worker = {}
    if not isinstance(failure, dict):
        failure = {}
    if not isinstance(build, dict):
        build = {}

    engine = profile_dir / "engine.plan"
    ap = get_ap(eval_report)
    latency = get_latency(eval_report, latency_report)
    eval_engine_paths = flatten_key_values(eval_report, ["engine_path", "engine", "engine_plan"])
    eval_profile_dirs = flatten_key_values(eval_report, ["profile_dir", "output_dir", "profile_path"])
    current_rel = f"subnets/{subnet_id}/{profile_id}"
    eval_contains_current = bool(find_strings_containing(eval_report, current_rel))
    worker_engine_paths = flatten_key_values(worker, ["engine_path", "engine", "engine_plan"])
    worker_contains_current = bool(find_strings_containing(worker, current_rel))
    build_engine_paths = flatten_key_values(build, ["engine_path", "engine", "engine_plan"])

    row: Dict[str, Any] = {
        **subnet,
        "profile_id": profile_id,
        "profile_dir_exists": profile_dir.exists(),
        "mixed_precision_profile_exists": (profile_dir / "mixed_precision_profile.json").exists(),
        "profile_template_id": profile.get("profile_template_id"),
        "int8_group_count": profile.get("int8_group_count"),
        "precision_assignment_hash": profile.get("precision_assignment_hash")
        or stable_hash(profile.get("precision_group_assignments") or profile.get("layer_precision_assignment")),
        "qdq_insert_report_exists": (profile_dir / "qdq_insert_report.json").exists(),
        "qdq_success": qdq.get("success"),
        "inserted_qdq_nodes_count": count_inserted_qdq(qdq),
        "unmatched_int8_precision_groups": qdq.get("unmatched_int8_precision_groups"),
        "unmatched_int8_precision_group_count": len(qdq.get("unmatched_int8_precision_groups") or [])
        if isinstance(qdq.get("unmatched_int8_precision_groups"), list)
        else None,
        "engine_exists": engine.exists(),
        "engine_size": file_size(engine),
        "engine_sha256": file_sha256(engine),
        "build_report_exists": (profile_dir / "build_report.json").exists(),
        "build_success": build.get("build_success", build.get("success")),
        "build_failure_reason": build.get("failure_reason"),
        "engine_structure_check_report_exists": (profile_dir / "engine_structure_check_report.json").exists(),
        "engine_structure_check_passed": structure.get("structure_check_passed"),
        "engine_structure_failure_reason": structure.get("failure_reason"),
        "engine_precision_realization_report_exists": (profile_dir / "engine_precision_realization_report.json").exists(),
        "engine_precision_realization_passed": precision.get("precision_realization_passed"),
        "engine_precision_mismatch_count": precision.get(
            "mismatch_count", precision.get("precision_realization_mismatch_count")
        ),
        "engine_precision_first_10_mismatch_layers": first_mismatch_layers(precision, 10),
        "trt_smoke_report_exists": (profile_dir / "trt_smoke_report.json").exists(),
        "trt_smoke_status": smoke.get("status"),
        "trt_smoke_success": smoke.get("success", smoke.get("smoke_success")),
        "trt_smoke_failure_reason": smoke.get("failure_reason"),
        "eval_report_exists": (profile_dir / "eval_report.json").exists(),
        "eval_success": eval_report.get("eval_success", eval_report.get("success")),
        "evaluated_frames": eval_report.get("evaluated_frames"),
        "synthetic_used": eval_report.get("synthetic_used"),
        "validation_dataloader_used": eval_report.get("validation_dataloader_used"),
        "eval_failure_reason": eval_report.get("failure_reason"),
        "AP@0.03": ap.get("AP@0.03"),
        "AP@0.30": ap.get("AP@0.30"),
        "AP@0.50": ap.get("AP@0.50"),
        "AP@0.70": ap.get("AP@0.70"),
        "mAP": ap.get("mAP"),
        "worker_result_exists": (profile_dir / "worker_result.json").exists(),
        "worker_status": worker.get("status"),
        "worker_failure_reason": worker.get("failure_reason")
        or get_nested(worker, ["failure", "failure_reason"], None),
        "profile_failure_report_exists": (profile_dir / "profile_failure_report.json").exists(),
        "profile_failure_stage_failed": failure.get("stage_failed"),
        "profile_failure_reason": failure.get("failure_reason"),
        "eval_engine_path_in_eval_report": eval_engine_paths[0] if eval_engine_paths else "",
        "eval_profile_dir_in_eval_report": eval_profile_dirs[0] if eval_profile_dirs else "",
        "eval_report_contains_current_subnet_profile_path": eval_contains_current,
        "worker_engine_path": worker_engine_paths[0] if worker_engine_paths else "",
        "worker_result_contains_current_subnet_profile_path": worker_contains_current,
        "build_engine_path": build_engine_paths[0] if build_engine_paths else "",
    }
    row.update(latency)
    row["profile_gate_complete"] = profile_gate_complete(row)
    return row


def csv_safe(v: Any) -> Any:
    if isinstance(v, (dict, list, tuple)):
        return json.dumps(v, ensure_ascii=False, sort_keys=True)
    return v


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    keys: List[str] = []
    for row in rows:
        for k in row.keys():
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow({k: csv_safe(row.get(k)) for k in keys})


def ap_tuple(row: Dict[str, Any], rounded: Optional[int] = None) -> Optional[Tuple[Any, ...]]:
    vals = [row.get(k) for k in AP_KEYS]
    if any(v is None or v == "" for v in vals):
        return None
    if rounded is not None:
        return tuple(round(float(v), rounded) for v in vals)
    return tuple(vals)


def unique_count(rows: Iterable[Dict[str, Any]], key: str, only_existing: bool = True) -> int:
    vals = []
    for r in rows:
        v = r.get(key)
        if only_existing and (v is None or v == ""):
            continue
        vals.append(v)
    return len(set(vals))


def summarize_profiles(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for profile_id in PROFILE_IDS:
        prs = [r for r in rows if r["profile_id"] == profile_id]
        completed = [r for r in prs if r.get("eval_success") is True]
        ap_groups: Dict[str, List[str]] = defaultdict(list)
        ap_groups_round6: Dict[str, List[str]] = defaultdict(list)
        for r in completed:
            t = ap_tuple(r)
            t6 = ap_tuple(r, 6)
            if t is not None:
                ap_groups[json_dumps(t)].append(r["subnet_id"])
            if t6 is not None:
                ap_groups_round6[json_dumps(t6)].append(r["subnet_id"])
        repeated = {
            k: v for k, v in sorted(ap_groups.items(), key=lambda kv: (-len(kv[1]), kv[0])) if len(v) > 3
        }
        out[profile_id] = {
            "completed_eval_count": len(completed),
            "unique_ap_tuple_count": len(ap_groups),
            "unique_ap_tuple_round6_count": len(ap_groups_round6),
            "ap_tuple_subnets": dict(sorted(ap_groups.items())),
            "ap_tuple_repeated_more_than_3": repeated,
            "unique_forward_p50_count": unique_count(completed, "forward_p50_ms"),
            "unique_engine_hash_count": unique_count(prs, "engine_sha256"),
            "unique_onnx_hash_count": unique_count(prs, "model_signal_maxk_onnx_sha256"),
        }
    return out


def summarize_ap_rounding(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for profile_id in PROFILE_IDS:
        completed = [r for r in rows if r["profile_id"] == profile_id and r.get("eval_success") is True]
        out[profile_id] = {}
        for digits in (6, 5, 4, 3, 2):
            groups: Dict[str, List[str]] = defaultdict(list)
            for r in completed:
                t = ap_tuple(r, digits)
                if t is not None:
                    groups[json_dumps(t)].append(r["subnet_id"])
            repeated = {
                k: v
                for k, v in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))
                if len(v) > 1
            }
            repeated_gt3 = {k: v for k, v in repeated.items() if len(v) > 3}
            out[profile_id][f"round_{digits}"] = {
                "unique_ap_tuple_count": len(groups),
                "repeat_group_count": len(repeated),
                "max_repeat_count": max([len(v) for v in groups.values()] or [0]),
                "repeated_more_than_3": repeated_gt3,
            }
    return out


def find_hash_anomalies(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    subnet_rows = []
    seen = set()
    for r in rows:
        sid = r["subnet_id"]
        if sid not in seen:
            subnet_rows.append(r)
            seen.add(sid)

    by_pmo: Dict[str, set] = defaultdict(set)
    by_onnx: Dict[str, set] = defaultdict(set)
    by_structure: Dict[str, set] = defaultdict(set)
    for r in subnet_rows:
        if r.get("pruned_model_object_sha256"):
            by_onnx[r.get("model_signal_maxk_onnx_sha256")].add(r.get("pruned_model_object_sha256"))
            by_pmo[r.get("pruned_model_object_sha256")].add(r.get("model_signal_maxk_onnx_sha256"))
        if r.get("structure_hash"):
            by_structure[r.get("structure_hash")].add(r.get("shape_hash"))

    pmo_diff_onnx_same = []
    for onnx_hash, pmos in by_onnx.items():
        if onnx_hash and len(pmos) > 1:
            pmo_diff_onnx_same.append({"onnx_hash": onnx_hash, "pruned_model_hashes": sorted(pmos)})

    structure_diff_shape_same = []
    shape_to_structures: Dict[str, set] = defaultdict(set)
    for r in subnet_rows:
        if r.get("shape_hash"):
            shape_to_structures[r.get("shape_hash")].add(r.get("structure_hash"))
    for shape_hash, structures in shape_to_structures.items():
        if len([s for s in structures if s]) > 1:
            structure_diff_shape_same.append({"shape_hash": shape_hash, "structure_hashes": sorted(structures)})

    onnx_diff_engine_same = []
    for profile_id in PROFILE_IDS:
        engine_to_onnx: Dict[str, set] = defaultdict(set)
        for r in rows:
            if r["profile_id"] == profile_id and r.get("engine_sha256"):
                engine_to_onnx[r.get("engine_sha256")].add(r.get("model_signal_maxk_onnx_sha256"))
        for engine_hash, onnx_hashes in engine_to_onnx.items():
            if len([x for x in onnx_hashes if x]) > 1:
                onnx_diff_engine_same.append(
                    {"profile_id": profile_id, "engine_hash": engine_hash, "onnx_hashes": sorted(onnx_hashes)}
                )

    different_hashes_same_ap = []
    for profile_id in PROFILE_IDS:
        ap_to_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in rows:
            if r["profile_id"] == profile_id and r.get("eval_success") is True:
                t = ap_tuple(r)
                if t is not None:
                    ap_to_rows[json_dumps(t)].append(r)
        for k, group in ap_to_rows.items():
            if len(group) <= 1:
                continue
            if (
                unique_count(group, "structure_hash") > 1
                and unique_count(group, "model_signal_maxk_onnx_sha256") > 1
                and unique_count(group, "engine_sha256") > 1
            ):
                different_hashes_same_ap.append(
                    {
                        "profile_id": profile_id,
                        "ap_tuple": k,
                        "subnets": [g["subnet_id"] for g in group],
                        "unique_structure_hash": unique_count(group, "structure_hash"),
                        "unique_onnx_hash": unique_count(group, "model_signal_maxk_onnx_sha256"),
                        "unique_engine_hash": unique_count(group, "engine_sha256"),
                    }
                )

    return {
        "pruned_model_object_hash_different_but_onnx_hash_same": pmo_diff_onnx_same,
        "onnx_hash_different_but_engine_hash_same": onnx_diff_engine_same,
        "structure_hash_different_but_shape_hash_same": structure_diff_shape_same,
        "structure_onnx_engine_different_but_ap_identical": different_hashes_same_ap,
    }


def profile003_failure_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    scope = [r for r in rows if r["profile_id"] == "profile_003" and "subnet_014" <= r["subnet_id"] <= "subnet_049"]
    summary = {
        "scope_count": len(scope),
        "missing_profile_dir": 0,
        "qdq_failed": 0,
        "engine_build_failed": 0,
        "engine_structure_mismatch": 0,
        "engine_precision_mismatch": 0,
        "trt_smoke_failed": 0,
        "eval_failed": 0,
        "complete": 0,
        "by_subnet": {},
    }
    for r in scope:
        sid = r["subnet_id"]
        missing = not r.get("profile_dir_exists")
        qdq_failed = r.get("qdq_success") is False or r.get("profile_failure_stage_failed") == "qdq_failed"
        build_failed = (
            r.get("build_success") is False
            or r.get("worker_status") == "engine_build_failed"
            or r.get("profile_failure_stage_failed") == "engine_build_failed"
        )
        structure_failed = (
            r.get("engine_structure_check_passed") is False
            or r.get("profile_failure_stage_failed") == "engine_structure_mismatch"
        )
        precision_failed = (
            r.get("engine_precision_realization_passed") is False
            or (r.get("engine_precision_mismatch_count") or 0) > 0
            or r.get("profile_failure_stage_failed") == "engine_precision_mismatch"
        )
        smoke_failed = (
            r.get("trt_smoke_success") is False
            or r.get("profile_failure_stage_failed") == "trt_smoke_failed"
        )
        eval_failed = (
            r.get("eval_report_exists")
            and r.get("eval_success") is not True
            or r.get("profile_failure_stage_failed") == "eval_failed"
        )
        complete = bool(r.get("profile_gate_complete"))
        for key, flag in [
            ("missing_profile_dir", missing),
            ("qdq_failed", qdq_failed),
            ("engine_build_failed", build_failed),
            ("engine_structure_mismatch", structure_failed),
            ("engine_precision_mismatch", precision_failed),
            ("trt_smoke_failed", smoke_failed),
            ("eval_failed", eval_failed),
            ("complete", complete),
        ]:
            if flag:
                summary[key] += 1
        summary["by_subnet"][sid] = {
            "status": r.get("worker_status") or r.get("profile_failure_stage_failed") or ("complete" if complete else ""),
            "failure_reason": r.get("profile_failure_reason") or r.get("build_failure_reason") or r.get("worker_failure_reason"),
            "engine_exists": r.get("engine_exists"),
            "build_success": r.get("build_success"),
            "qdq_success": r.get("qdq_success"),
            "eval_success": r.get("eval_success"),
        }
    return summary


def mtime_clusters(profile_dir: Path, gap_seconds: int = 1800) -> Dict[str, Any]:
    files = [p for p in profile_dir.rglob("*") if p.is_file()]
    mtimes = sorted((p.stat().st_mtime, str(p.relative_to(profile_dir))) for p in files)
    clusters: List[List[Tuple[float, str]]] = []
    for mtime, rel in mtimes:
        if not clusters or mtime - clusters[-1][-1][0] > gap_seconds:
            clusters.append([])
        clusters[-1].append((mtime, rel))

    def fmt(ts: float) -> str:
        return _dt.datetime.fromtimestamp(ts).isoformat(timespec="seconds")

    return {
        "file_count": len(files),
        "cluster_count_gap_30m": len(clusters),
        "clusters": [
            {
                "start": fmt(c[0][0]),
                "end": fmt(c[-1][0]),
                "file_count": len(c),
                "sample_files": [x[1] for x in c[:8]],
            }
            for c in clusters
        ],
    }


def audit_subnet002(root: Path, row_by_pair: Dict[Tuple[str, str], Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for pid in ("profile_000", "profile_001"):
        profile_dir = root / "subnets" / "subnet_002" / pid
        row = row_by_pair.get(("subnet_002", pid), {})
        out[pid] = {
            "profile_dir_exists": profile_dir.exists(),
            "worker_result_exists": (profile_dir / "worker_result.json").exists(),
            "eval_report_exists": (profile_dir / "eval_report.json").exists(),
            "eval_success": row.get("eval_success"),
            "evaluated_frames": row.get("evaluated_frames"),
            "profile_gate_complete": row.get("profile_gate_complete"),
            "mtime_clusters": mtime_clusters(profile_dir) if profile_dir.exists() else {},
        }
    return out


def partial_artifact_profiles(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for r in rows:
        if r.get("profile_gate_complete") or not r.get("profile_dir_exists"):
            continue
        has_artifacts = any(
            [
                r.get("mixed_precision_profile_exists"),
                r.get("build_report_exists"),
                r.get("engine_exists"),
                r.get("trt_smoke_report_exists"),
                r.get("eval_report_exists"),
                r.get("worker_result_exists"),
            ]
        )
        if not has_artifacts:
            continue
        out.append(
            {
                "subnet_id": r["subnet_id"],
                "profile_id": r["profile_id"],
                "engine_exists": r.get("engine_exists"),
                "build_success": r.get("build_success"),
                "trt_smoke_success": r.get("trt_smoke_success"),
                "eval_success": r.get("eval_success"),
                "evaluated_frames": r.get("evaluated_frames"),
                "failure_stage": r.get("profile_failure_stage_failed"),
                "worker_status": r.get("worker_status"),
                "reason": r.get("profile_failure_reason")
                or r.get("build_failure_reason")
                or r.get("eval_failure_reason")
                or r.get("worker_failure_reason"),
            }
        )
    return out


def stale_failure_reports(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for r in rows:
        if r.get("eval_success") is True and r.get("profile_failure_report_exists"):
            out.append(
                {
                    "subnet_id": r["subnet_id"],
                    "profile_id": r["profile_id"],
                    "evaluated_frames": r.get("evaluated_frames"),
                    "failure_stage": r.get("profile_failure_stage_failed"),
                    "failure_reason": r.get("profile_failure_reason"),
                    "worker_result_exists": r.get("worker_result_exists"),
                }
            )
    return out


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    try:
        with path.open("r", newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except Exception as exc:  # pragma: no cover - audit records errors
        return [{"__csv_read_error__": repr(exc)}]


def read_jsonl_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if isinstance(row, dict):
                    row["__line_no__"] = line_no
                    rows.append(row)
            except Exception as exc:
                rows.append({"__line_no__": line_no, "__jsonl_read_error__": repr(exc)})
    return rows


def row_pair(row: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    sid = row.get("subnet_id") or row.get("subnet")
    pid = row.get("profile_id") or row.get("profile")
    if sid is None or pid is None:
        return None
    return str(sid), str(pid)


def audit_top_level_indices(root: Path, row_by_pair: Dict[Tuple[str, str], Dict[str, Any]]) -> Dict[str, Any]:
    files = [
        "mixed_precision_profile_index.csv",
        "engine_eval_summary.csv",
        "full_engine_training_samples.jsonl",
        "component_lut_samples.csv",
        "failure_summary.csv",
    ]
    out: Dict[str, Any] = {}
    all_label_true: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for name in files:
        path = root / name
        rows: List[Dict[str, Any]]
        if name.endswith(".jsonl"):
            rows = read_jsonl_rows(path)
        else:
            rows = read_csv_rows(path)
        pair_counts = Counter(p for p in (row_pair(r) for r in rows) if p)
        duplicates = {f"{p[0]}/{p[1]}": c for p, c in pair_counts.items() if c > 1}
        label_true_pairs: List[str] = []
        for r in rows:
            p = row_pair(r)
            if not p:
                continue
            if truthy_bool(r.get("label_available")):
                label_true_pairs.append(f"{p[0]}/{p[1]}")
                all_label_true[p].append(name)
        out[name] = {
            "exists": path.exists(),
            "row_count": len(rows),
            "duplicate_subnet_profile_rows": duplicates,
            "label_available_true_count": len(label_true_pairs),
            "same_pair_multiple_label_available_true": {
                k: c for k, c in Counter(label_true_pairs).items() if c > 1
            },
        }

    incomplete_label_true = []
    for p, names in sorted(all_label_true.items()):
        row = row_by_pair.get(p)
        if not row or not row.get("profile_gate_complete"):
            incomplete_label_true.append(
                {
                    "subnet_id": p[0],
                    "profile_id": p[1],
                    "files": sorted(set(names)),
                    "gate_complete": bool(row and row.get("profile_gate_complete")),
                    "engine_exists": row.get("engine_exists") if row else None,
                    "eval_success": row.get("eval_success") if row else None,
                    "evaluated_frames": row.get("evaluated_frames") if row else None,
                    "failure_stage": row.get("profile_failure_stage_failed") if row else None,
                }
            )
    out["_aggregate"] = {
        "label_available_true_but_profile_gate_incomplete": incomplete_label_true,
        "label_available_true_but_profile_gate_incomplete_count": len(incomplete_label_true),
    }

    for meta in ("manifest.json", "progress_state.json"):
        p = root / meta
        data = read_json(p)
        out[meta] = {
            "exists": p.exists(),
            "keys": sorted(data.keys()) if isinstance(data, dict) else [],
            "data": data,
        }
    return out


def profile014_detail(root: Path, row_by_pair: Dict[Tuple[str, str], Dict[str, Any]]) -> Dict[str, Any]:
    sid, pid = "subnet_014", "profile_003"
    profile_dir = root / "subnets" / sid / pid
    profile = read_json(profile_dir / "mixed_precision_profile.json") or {}
    qdq = read_json(profile_dir / "qdq_insert_report.json") or {}
    precision = read_json(profile_dir / "engine_precision_realization_report.json") or {}
    build_log = profile_dir / "build_log.txt"
    tail = ""
    if build_log.exists():
        lines = build_log.read_text(encoding="utf-8", errors="replace").splitlines()
        tail = "\n".join(lines[-200:])
    return {
        "row": row_by_pair.get((sid, pid), {}),
        "mixed_precision_profile": {
            "profile_template_id": get_nested(profile, ["profile_template_id"], None),
            "int8_group_count": get_nested(profile, ["int8_group_count"], None),
            "precision_assignment_hash": get_nested(profile, ["precision_assignment_hash"], None),
        },
        "qdq_insert_report": {
            "exists": (profile_dir / "qdq_insert_report.json").exists(),
            "success": get_nested(qdq, ["success"], None),
            "inserted_qdq_nodes_count": count_inserted_qdq(qdq),
            "unmatched_int8_precision_groups": get_nested(qdq, ["unmatched_int8_precision_groups"], None),
        },
        "engine_plan": {
            "exists": (profile_dir / "engine.plan").exists(),
            "size": file_size(profile_dir / "engine.plan"),
            "sha256": file_sha256(profile_dir / "engine.plan"),
        },
        "build_log_tail_200": tail,
        "trt_layer_info_exists": (profile_dir / "trt_layer_info.json").exists(),
        "engine_structure_check_report": read_json(profile_dir / "engine_structure_check_report.json"),
        "engine_precision_realization_report": {
            "exists": (profile_dir / "engine_precision_realization_report.json").exists(),
            "mismatch_count": get_nested(precision, ["mismatch_count"], get_nested(precision, ["precision_realization_mismatch_count"], None)),
            "first_20_mismatch_layers": first_mismatch_layers(precision, 20),
            "raw": precision if precision else None,
        },
        "trt_smoke_report": read_json(profile_dir / "trt_smoke_report.json"),
        "eval_report": read_json(profile_dir / "eval_report.json"),
        "profile_failure_report": read_json(profile_dir / "profile_failure_report.json"),
        "worker_result": read_json(profile_dir / "worker_result.json"),
        "build_report": read_json(profile_dir / "build_report.json"),
    }


def first20_table(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    table = []
    for r in rows:
        sid = r["subnet_id"]
        if sid in seen:
            continue
        seen.add(sid)
        table.append(
            {
                "subnet_id": sid,
                "structure_hash": r.get("structure_hash"),
                "shape_hash": r.get("shape_hash"),
                "pruned_model_hash": short_hash(r.get("pruned_model_object_sha256")),
                "onnx_hash": short_hash(r.get("model_signal_maxk_onnx_sha256")),
                "param_ratio": r.get("actual_param_prune_ratio")
                if r.get("actual_param_prune_ratio") is not None
                else r.get("actual_param_prune_ratio_on_searchable_surface"),
                "channel_ratio": r.get("actual_channel_prune_ratio")
                if r.get("actual_channel_prune_ratio") is not None
                else r.get("actual_channel_prune_ratio_on_searchable_surface"),
            }
        )
        if len(table) >= 20:
            break
    return table


def format_md_table(rows: List[Dict[str, Any]], columns: List[str], max_rows: Optional[int] = None) -> str:
    if max_rows is not None:
        rows = rows[:max_rows]
    lines = ["|" + "|".join(columns) + "|", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in rows:
        vals = []
        for c in columns:
            v = row.get(c, "")
            if isinstance(v, float):
                vals.append(f"{v:.8g}")
            else:
                vals.append(str(v).replace("|", "\\|"))
        lines.append("|" + "|".join(vals) + "|")
    return "\n".join(lines)


def infer_profile003_cause(summary: Dict[str, Any], detail: Dict[str, Any]) -> Dict[str, str]:
    row = detail.get("row") or {}
    build_log = detail.get("build_log_tail_200") or ""
    qdq = detail.get("qdq_insert_report") or {}
    if row.get("build_success") is False and "Could not find any implementation" in build_log:
        return {
            "choice": "C/H",
            "reason": (
                "QDQ/profile legality passed and TensorRT reached engine build, then failed with "
                "'Could not find any implementation' for a quantized Conv fusion. Evidence supports "
                "TensorRT build failure caused by high_int8 plus pruned shape/tactic constraints."
            ),
        }
    if qdq.get("success") is False:
        return {"choice": "B", "reason": "QDQ insertion report failed before engine build."}
    if row.get("engine_structure_check_passed") is False:
        return {"choice": "D", "reason": "Engine exists but structure check failed."}
    if row.get("engine_precision_realization_passed") is False:
        return {"choice": "E", "reason": "Engine exists but precision realization failed."}
    if row.get("trt_smoke_success") is False or row.get("eval_success") is False:
        return {"choice": "F", "reason": "Build passed but smoke/eval failed."}
    return {"choice": "undetermined", "reason": "Available reports do not isolate a single gate."}


def write_report(root: Path, summary: Dict[str, Any]) -> None:
    lines: List[str] = []
    diversity = summary["diversity"]
    profile_stats = summary["profile_stats"]
    anomalies = summary["hash_anomalies"]
    failure003 = summary["profile003_failure_summary"]
    cause = summary["profile003_inferred_cause"]
    top = summary["top_level_indices"]["_aggregate"]
    blind = summary["eval_engine_path_blind_spots"]
    stale = summary["stale_failure_reports_on_completed_profiles"]
    partial = summary["partial_artifact_profiles"]

    label_trust = "not trustworthy for training labels until fixes/reruns"
    continue_run = "do not continue 50x4 generation"
    rerun = summary["recommended_reruns"]

    lines.append("# v11 LUT Artifact Gate-Sensitive Audit")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append(f"- Current LUT labels: {label_trust}.")
    lines.append(f"- Recommendation: {continue_run}; freeze generation and repair audit gaps first.")
    lines.append(f"- Profiles recommended for cleanup/rerun after confirmation: {', '.join(rerun) if rerun else 'none from current evidence'}.")
    lines.append(f"- Label rows with label_available=true but incomplete profile gate: {top['label_available_true_but_profile_gate_incomplete_count']}.")
    lines.append(f"- Completed profiles with stale profile_failure_report.json: {len(stale)}.")
    lines.append(f"- Partial artifact profiles without complete labels: {len(partial)}.")
    lines.append("")

    lines.append("## Subnet Diversity")
    lines.append(f"- unique structure_hash: {diversity['unique_structure_hash']}")
    lines.append(f"- unique shape_hash: {diversity['unique_shape_hash']}")
    lines.append(f"- unique pruned_model_object hash: {diversity['unique_pruned_model_object_hash']}")
    lines.append(f"- unique model_signal_maxk.onnx hash: {diversity['unique_model_signal_maxk_onnx_hash']}")
    for pid, count in diversity["unique_engine_hash_by_profile"].items():
        lines.append(f"- unique {pid} engine hash: {count}")
    lines.append("")
    lines.append("First 20 subnet comparison:")
    lines.append(format_md_table(summary["first20_subnet_table"], ["subnet_id", "structure_hash", "shape_hash", "pruned_model_hash", "onnx_hash", "param_ratio", "channel_ratio"]))
    lines.append("")
    lines.append("Hash anomaly checks:")
    for key, value in anomalies.items():
        lines.append(f"- {key}: {len(value)}")
    lines.append("")

    lines.append("## AP Repetition")
    for pid in PROFILE_IDS:
        ps = profile_stats[pid]
        rounded = summary["ap_rounding_stats"][pid]
        lines.append(
            f"- {pid}: completed_eval={ps['completed_eval_count']}, unique_AP_tuple={ps['unique_ap_tuple_count']}, "
            f"unique_forward_p50={ps['unique_forward_p50_count']}, unique_engine_hash={ps['unique_engine_hash_count']}, "
            f"unique_ONNX_hash={ps['unique_onnx_hash_count']}"
        )
        lines.append(
            f"  - rounded AP uniqueness: round6={rounded['round_6']['unique_ap_tuple_count']}, "
            f"round4={rounded['round_4']['unique_ap_tuple_count']}, "
            f"round3={rounded['round_3']['unique_ap_tuple_count']} max_repeat={rounded['round_3']['max_repeat_count']}, "
            f"round2={rounded['round_2']['unique_ap_tuple_count']} max_repeat={rounded['round_2']['max_repeat_count']}"
        )
        repeated = ps["ap_tuple_repeated_more_than_3"]
        if repeated:
            for ap_key, subnets in repeated.items():
                lines.append(f"  - repeated AP {ap_key}: {', '.join(subnets)}")
        rounded_repeated = rounded["round_3"]["repeated_more_than_3"]
        if rounded_repeated:
            for ap_key, subnets in rounded_repeated.items():
                lines.append(f"  - repeated AP at 3 decimals {ap_key}: {', '.join(subnets)}")
    lines.append("")
    lines.append("Evidence table for identical AP with different hashes:")
    ap_diff = anomalies["structure_onnx_engine_different_but_ap_identical"]
    if ap_diff:
        lines.append(format_md_table(ap_diff, ["profile_id", "ap_tuple", "subnets", "unique_structure_hash", "unique_onnx_hash", "unique_engine_hash"], max_rows=20))
    else:
        lines.append("- none")
    lines.append("")
    lines.append(
        "Interpretation: identical AP tuples across different structure/ONNX/engine hashes are abnormal. "
        "The stored eval reports do not contain explicit engine_path/profile_dir, so evaluator path binding is an audit blind spot."
    )
    lines.append("")

    lines.append("## Evaluator Engine Path Evidence")
    lines.append(f"- completed profiles missing engine_path in eval_report: {blind['completed_missing_eval_engine_path_count']}")
    lines.append("First 20 completed profiles:")
    lines.append(format_md_table(summary["first20_completed_eval_path_table"], ["subnet_id", "profile_id", "engine_path_in_eval_report", "profile_dir_in_eval_report", "output_report_contains_current_path", "path_issue"]))
    lines.append("")

    lines.append("## profile_003 Failure After subnet_014")
    lines.append(f"- missing profile dir: {failure003['missing_profile_dir']}")
    lines.append(f"- qdq failed: {failure003['qdq_failed']}")
    lines.append(f"- engine_build_failed: {failure003['engine_build_failed']}")
    lines.append(f"- engine_structure_mismatch: {failure003['engine_structure_mismatch']}")
    lines.append(f"- engine_precision_mismatch: {failure003['engine_precision_mismatch']}")
    lines.append(f"- trt_smoke_failed: {failure003['trt_smoke_failed']}")
    lines.append(f"- eval_failed: {failure003['eval_failed']}")
    lines.append(f"- complete: {failure003['complete']}")
    lines.append(f"- inferred cause: {cause['choice']} - {cause['reason']}")
    lines.append("")
    lines.append("subnet_014/profile_003 detail:")
    d = summary["subnet014_profile003_detail"]
    lines.append(f"- mixed_precision_profile: {json.dumps(d['mixed_precision_profile'], ensure_ascii=False, sort_keys=True)}")
    lines.append(f"- qdq_insert_report: {json.dumps(d['qdq_insert_report'], ensure_ascii=False, sort_keys=True)}")
    lines.append(f"- engine.plan: {json.dumps(d['engine_plan'], ensure_ascii=False, sort_keys=True)}")
    lines.append(f"- trt_layer_info exists: {d['trt_layer_info_exists']}")
    lines.append(f"- engine_structure_check_report: {json.dumps(d['engine_structure_check_report'], ensure_ascii=False, sort_keys=True)}")
    lines.append(
        "- engine_precision_realization_report: "
        f"{json.dumps({k: d['engine_precision_realization_report'].get(k) for k in ['exists', 'mismatch_count', 'first_20_mismatch_layers']}, ensure_ascii=False, sort_keys=True)}"
    )
    lines.append(f"- trt_smoke_report: {json.dumps(d['trt_smoke_report'], ensure_ascii=False, sort_keys=True)}")
    lines.append(f"- eval_report: {json.dumps(d['eval_report'], ensure_ascii=False, sort_keys=True)}")
    lines.append(f"- profile_failure_report: {json.dumps(d['profile_failure_report'], ensure_ascii=False, sort_keys=True)}")
    worker = d.get("worker_result") if isinstance(d.get("worker_result"), dict) else {}
    lines.append(
        "- worker_result: "
        f"{json.dumps({'status': worker.get('status'), 'build_failure_reason': get_nested(worker, ['build', 'failure_reason'], None), 'engine_path': get_nested(worker, ['build', 'engine_path'], None)}, ensure_ascii=False, sort_keys=True)}"
    )
    build_report = d.get("build_report") if isinstance(d.get("build_report"), dict) else {}
    lines.append(
        "- build_report: "
        f"{json.dumps({'success': build_report.get('success'), 'build_success': build_report.get('build_success'), 'failure_reason': build_report.get('failure_reason'), 'engine_path': build_report.get('engine_path')}, ensure_ascii=False, sort_keys=True)}"
    )
    lines.append("")
    lines.append("<details><summary>subnet_014/profile_003 build_log tail 200</summary>")
    lines.append("")
    lines.append("```text")
    lines.append(d.get("build_log_tail_200") or "")
    lines.append("```")
    lines.append("")
    lines.append("</details>")
    lines.append("")

    lines.append("## subnet_002/profile_000 and profile_001 Double-Master Check")
    for pid, info in summary["subnet002_pollution_check"].items():
        lines.append(
            f"- subnet_002/{pid}: worker_result_exists={info['worker_result_exists']}, "
            f"eval_success={info['eval_success']}, evaluated_frames={info['evaluated_frames']}, "
            f"gate_complete={info['profile_gate_complete']}, mtime_clusters_30m={info['mtime_clusters'].get('cluster_count_gap_30m')}"
        )
    lines.append("")
    if stale:
        lines.append("Completed profiles with stale failure reports:")
        lines.append(format_md_table(stale, ["subnet_id", "profile_id", "evaluated_frames", "failure_stage", "failure_reason", "worker_result_exists"]))
        lines.append("")
    if partial:
        lines.append("Partial artifact profiles without complete labels:")
        lines.append(format_md_table(partial, ["subnet_id", "profile_id", "engine_exists", "build_success", "trt_smoke_success", "eval_success", "evaluated_frames", "failure_stage", "worker_status", "reason"]))
        lines.append("")
    lines.append("Top-level index duplicate/incomplete-label summary:")
    for name, meta in summary["top_level_indices"].items():
        if name.startswith("_") or name in {"manifest.json", "progress_state.json"}:
            continue
        note = ""
        if name == "component_lut_samples.csv":
            note = " (same subnet/profile repeats are expected component-level rows)"
        lines.append(
            f"- {name}: rows={meta['row_count']}, duplicate_pairs={len(meta['duplicate_subnet_profile_rows'])}, "
            f"multi_label_true_pairs={len(meta['same_pair_multiple_label_available_true'])}{note}"
        )
    lines.append("")

    lines.append("## Required Fixes")
    lines.append("- Add engine_path, profile_dir, subnet_id, profile_id, and report output_dir to eval_report.json.")
    lines.append("- Treat repeated AP with different structure/ONNX/engine hashes as a hard gate failure until evaluator path binding is proven.")
    lines.append("- For profile_003 high_int8, lower the sampled INT8 region or add shape/tactic constraints for the failing quantized Conv family before rerun.")
    lines.append("- Remove stale profile_failure_report.json only after confirming the corresponding profile gate evidence, or rerun those profiles for clean auditable directories.")
    lines.append("- Clean/rerun only after confirmation: failed profile_003 entries from subnet_014 onward, and any label_available=true rows whose profile gate is incomplete.")
    lines.append("- Do not delete profile directories from this audit; quarantine recommendations are recorded for manual confirmation.")
    lines.append("")

    (root / "audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    root = args.root

    subnet_audits = {sid: audit_subnet(root, sid) for sid in SUBNET_IDS}
    rows: List[Dict[str, Any]] = []
    for sid in SUBNET_IDS:
        for pid in PROFILE_IDS:
            rows.append(audit_profile(root, sid, pid, subnet_audits[sid]))

    csv_path = root / "audit_subnet_profile_consistency.csv"
    json_path = root / "audit_subnet_profile_consistency.json"
    write_csv(csv_path, rows)

    row_by_pair = {(r["subnet_id"], r["profile_id"]): r for r in rows}
    diversity = {
        "unique_structure_hash": unique_count(list(subnet_audits.values()), "structure_hash"),
        "unique_shape_hash": unique_count(list(subnet_audits.values()), "shape_hash"),
        "unique_pruned_model_object_hash": unique_count(list(subnet_audits.values()), "pruned_model_object_sha256"),
        "unique_model_signal_maxk_onnx_hash": unique_count(list(subnet_audits.values()), "model_signal_maxk_onnx_sha256"),
        "unique_engine_hash_by_profile": {
            pid: unique_count([r for r in rows if r["profile_id"] == pid], "engine_sha256") for pid in PROFILE_IDS
        },
    }
    eval_path_table = []
    for r in [x for x in rows if x.get("eval_success") is True][:20]:
        issue = ""
        if not r.get("eval_engine_path_in_eval_report"):
            issue = "missing_engine_path"
        elif f"subnets/{r['subnet_id']}/{r['profile_id']}" not in r.get("eval_engine_path_in_eval_report", ""):
            issue = "points_elsewhere"
        eval_path_table.append(
            {
                "subnet_id": r["subnet_id"],
                "profile_id": r["profile_id"],
                "engine_path_in_eval_report": r.get("eval_engine_path_in_eval_report"),
                "profile_dir_in_eval_report": r.get("eval_profile_dir_in_eval_report"),
                "output_report_contains_current_path": r.get("eval_report_contains_current_subnet_profile_path"),
                "path_issue": issue,
            }
        )
    completed_missing_eval_engine = [
        r for r in rows if r.get("eval_success") is True and not r.get("eval_engine_path_in_eval_report")
    ]

    summary: Dict[str, Any] = {
        "root": str(root),
        "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "subnet_count": len(SUBNET_IDS),
        "profile_count": len(rows),
        "subnets": subnet_audits,
        "rows": rows,
        "diversity": diversity,
        "first20_subnet_table": first20_table(rows),
        "profile_stats": summarize_profiles(rows),
        "ap_rounding_stats": summarize_ap_rounding(rows),
        "hash_anomalies": find_hash_anomalies(rows),
        "profile003_failure_summary": profile003_failure_summary(rows),
        "subnet014_profile003_detail": profile014_detail(root, row_by_pair),
        "subnet002_pollution_check": audit_subnet002(root, row_by_pair),
        "top_level_indices": audit_top_level_indices(root, row_by_pair),
        "first20_completed_eval_path_table": eval_path_table,
        "partial_artifact_profiles": partial_artifact_profiles(rows),
        "stale_failure_reports_on_completed_profiles": stale_failure_reports(rows),
        "eval_engine_path_blind_spots": {
            "completed_missing_eval_engine_path_count": len(completed_missing_eval_engine),
            "completed_missing_eval_engine_path_pairs": [
                {"subnet_id": r["subnet_id"], "profile_id": r["profile_id"]} for r in completed_missing_eval_engine
            ],
        },
    }
    cause = infer_profile003_cause(summary["profile003_failure_summary"], summary["subnet014_profile003_detail"])
    summary["profile003_inferred_cause"] = cause

    recommended = []
    for sid, meta in summary["profile003_failure_summary"]["by_subnet"].items():
        if meta.get("status") and meta.get("status") != "complete":
            recommended.append(f"{sid}/profile_003")
    incomplete = summary["top_level_indices"]["_aggregate"]["label_available_true_but_profile_gate_incomplete"]
    for item in incomplete:
        pair = f"{item['subnet_id']}/{item['profile_id']}"
        if pair not in recommended:
            recommended.append(pair)
    for item in summary["partial_artifact_profiles"]:
        pair = f"{item['subnet_id']}/{item['profile_id']}"
        if pair not in recommended:
            recommended.append(pair)
    for item in summary["stale_failure_reports_on_completed_profiles"]:
        pair = f"{item['subnet_id']}/{item['profile_id']}"
        if pair not in recommended:
            recommended.append(pair)
    summary["recommended_reruns"] = recommended

    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    write_report(root, summary)

    print("AUDIT_ROOT", root)
    print("CSV", csv_path)
    print("JSON", json_path)
    print("MD", root / "audit_report.md")
    print("UNIQUE_STRUCTURE_HASH", diversity["unique_structure_hash"])
    print("UNIQUE_SHAPE_HASH", diversity["unique_shape_hash"])
    print("UNIQUE_PRUNED_MODEL_HASH", diversity["unique_pruned_model_object_hash"])
    print("UNIQUE_ONNX_HASH", diversity["unique_model_signal_maxk_onnx_hash"])
    print("UNIQUE_ENGINE_HASH_BY_PROFILE", json.dumps(diversity["unique_engine_hash_by_profile"], sort_keys=True))
    for pid, ps in summary["profile_stats"].items():
        print(
            "PROFILE_STATS",
            pid,
            "completed_eval",
            ps["completed_eval_count"],
            "unique_ap_tuple",
            ps["unique_ap_tuple_count"],
            "unique_forward_p50",
            ps["unique_forward_p50_count"],
            "unique_engine_hash",
            ps["unique_engine_hash_count"],
            "unique_onnx_hash",
            ps["unique_onnx_hash_count"],
        )
    print("PROFILE003_FAILURE_SUMMARY", json.dumps(summary["profile003_failure_summary"], ensure_ascii=False, sort_keys=True)[:4000])
    print("PROFILE003_INFERRED_CAUSE", json.dumps(cause, ensure_ascii=False, sort_keys=True))
    print(
        "INCOMPLETE_LABEL_TRUE_COUNT",
        summary["top_level_indices"]["_aggregate"]["label_available_true_but_profile_gate_incomplete_count"],
    )
    print("EVAL_ENGINE_PATH_BLIND_SPOTS", len(completed_missing_eval_engine))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
