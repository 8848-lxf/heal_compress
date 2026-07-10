#!/usr/bin/env python3
"""Resumable multi-GPU v12 LUT engine build + 300-frame eval workers."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[2]
_UNIAD = _ROOT.parent
for _p in (_UNIAD, _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tools.latency_lut import run_v11_mixed_precision_lut_dataset_builder as builder  # noqa: E402
from tools.latency_lut.prepare_v12_combined_lut_dataset import (  # noqa: E402
    DATASET_VERSION,
    summarize_surface_contract_from_manifest,
)
from tools.latency_lut.physical_structure_v2 import (  # noqa: E402
    HASH_SCHEMA_VERSION,
    atomic_write_json,
    compute_deployment_profile_hash_v2,
)


GATE_FAILURE_STAGES = {
    "profile_legality_failed",
    "real_onnx_export_failed",
    "onnx_or_qdq_failed",
    "engine_build_failed",
    "engine_structure_mismatch",
    "engine_precision_mismatch",
    "trt_smoke_failed",
    "eval_failed",
    "label_unavailable",
    "worker_crashed",
    "physical_structure_preflight_failed",
}


@dataclass(frozen=True)
class ProfileJob:
    subnet_dir: Path
    subnet_id: str
    subnet_index: int
    structure_hash: str
    profile_id: str
    profile_index: int
    profile_dir: Path
    source_subset: str


@dataclass
class RunningWorker:
    job: ProfileJob
    slot_id: str
    gpu: str
    process: subprocess.Popen
    log_handle: Any
    log_path: Path
    result_path: Path


@dataclass
class IndexState:
    output_dir: Path
    args: argparse.Namespace
    subnet_rows: list[dict[str, Any]]
    profile_rows: list[dict[str, Any]] = field(default_factory=list)
    eval_rows: list[dict[str, Any]] = field(default_factory=list)
    component_rows: list[dict[str, Any]] = field(default_factory=list)
    training_rows: list[dict[str, Any]] = field(default_factory=list)
    label_rows: list[dict[str, Any]] = field(default_factory=list)
    failure_rows: list[dict[str, Any]] = field(default_factory=list)
    progress_state: dict[str, Any] = field(default_factory=dict)
    consecutive_failures: int = 0
    total_gate_failures: int = 0


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def read_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if str(key) not in seen:
                seen.add(str(key))
                fields.append(str(key))
    if not fields:
        fields = ["empty"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({str(k): json.dumps(v, sort_keys=True) if isinstance(v, (dict, list, tuple)) else v for k, v in row.items()})


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, default=str) + "\n")


def sha256_file(path: Path) -> str:
    if not path.is_file():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _requested_int8_modules(profile: Mapping[str, Any]) -> list[str]:
    return sorted(
        str(module)
        for module, precision_value in (profile.get("layer_precision_assignment") or {}).items()
        if str(precision_value).lower() == "int8"
    )


def _current_deployment_profile_hash(subnet_dir: Path, profile: Mapping[str, Any]) -> str:
    physical_hash = read_json(subnet_dir / "physical_hash_v2.json", {})
    return compute_deployment_profile_hash_v2(
        shape_hash_v2=str(physical_hash.get("shape_hash_v2", "")),
        profile=profile,
        requested_int8_modules=_requested_int8_modules(profile),
    )


def _build_command_onnx_paths(build_report: Mapping[str, Any]) -> list[str]:
    paths = []
    for token in build_report.get("command", []) or []:
        token = str(token)
        if token.startswith("--onnx="):
            paths.append(token.split("=", 1)[1])
    return paths


def capture_engine_provenance_v2(
    *,
    subnet_dir: Path,
    profile_dir: Path,
    profile: Mapping[str, Any],
    capture_mode: str,
) -> dict[str, Any]:
    engine_path = profile_dir / "engine.plan"
    qdq_path = profile_dir / "onnx/model_mixed_qdq.onnx"
    build_report_path = profile_dir / "build_report.json"
    mapping_path = profile_dir / "canonical_precision_mapping.json"
    layer_info_path = profile_dir / "trt_layer_info.json"
    build_report = read_json(build_report_path, {})
    command_paths = _build_command_onnx_paths(build_report)
    command_matches = any(Path(value).resolve() == qdq_path.resolve() for value in command_paths)
    qdq_older_than_engine = qdq_path.is_file() and engine_path.is_file() and qdq_path.stat().st_mtime <= engine_path.stat().st_mtime
    layer_info_current = layer_info_path.is_file() and engine_path.is_file() and layer_info_path.stat().st_mtime >= engine_path.stat().st_mtime
    payload = {
        "provenance_schema_version": "engine-provenance-v2",
        "capture_mode": capture_mode,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "engine_path": str(engine_path.resolve()),
        "engine_sha256": sha256_file(engine_path),
        "engine_size_bytes": engine_path.stat().st_size if engine_path.is_file() else 0,
        "mixed_qdq_onnx_path": str(qdq_path.resolve()),
        "mixed_qdq_onnx_sha256": sha256_file(qdq_path),
        "canonical_mapping_sha256": sha256_file(mapping_path),
        "trt_layer_info_sha256": sha256_file(layer_info_path),
        "build_report_sha256": sha256_file(build_report_path),
        "build_report_success": bool(build_report.get("build_success")),
        "build_report_returncode": build_report.get("returncode"),
        "build_command_onnx_paths": command_paths,
        "build_command_matches_current_onnx": command_matches,
        "qdq_onnx_not_newer_than_engine": qdq_older_than_engine,
        "trt_layer_info_not_older_than_engine": layer_info_current,
        "physical_hash_v2": read_json(subnet_dir / "physical_hash_v2.json", {}),
        "deployment_profile_hash_v2": _current_deployment_profile_hash(subnet_dir, profile),
    }
    payload["provenance_valid"] = bool(
        payload["engine_sha256"]
        and payload["engine_size_bytes"] > 0
        and payload["mixed_qdq_onnx_sha256"]
        and payload["canonical_mapping_sha256"]
        and payload["build_report_success"]
        and command_matches
        and qdq_older_than_engine
        and payload["trt_layer_info_sha256"]
        and layer_info_current
    )
    atomic_write_json(profile_dir / "engine_provenance_v2.json", payload)
    return payload


def validate_existing_engine_for_reuse(
    *, subnet_dir: Path, profile_dir: Path, profile: Mapping[str, Any]
) -> dict[str, Any]:
    engine_path = profile_dir / "engine.plan"
    existing = read_json(profile_dir / "engine_provenance_v2.json", {})
    current_engine_hash = sha256_file(engine_path)
    current_qdq_hash = sha256_file(profile_dir / "onnx/model_mixed_qdq.onnx")
    current_mapping_hash = sha256_file(profile_dir / "canonical_precision_mapping.json")
    current_layer_info_hash = sha256_file(profile_dir / "trt_layer_info.json")
    current_profile_hash = _current_deployment_profile_hash(subnet_dir, profile)
    if existing:
        valid = bool(
            existing.get("provenance_valid")
            and current_engine_hash
            and current_engine_hash == existing.get("engine_sha256")
            and current_qdq_hash == existing.get("mixed_qdq_onnx_sha256")
            and current_mapping_hash == existing.get("canonical_mapping_sha256")
            and current_layer_info_hash == existing.get("trt_layer_info_sha256")
            and current_profile_hash == existing.get("deployment_profile_hash_v2")
        )
        return {"engine_reuse_valid": valid, "validation_source": "engine_provenance_v2", "provenance": existing}
    captured = capture_engine_provenance_v2(
        subnet_dir=subnet_dir,
        profile_dir=profile_dir,
        profile=profile,
        capture_mode="legacy_reconstructed_build_command_path_mtime_and_hash",
    )
    return {"engine_reuse_valid": bool(captured.get("provenance_valid")), "validation_source": "legacy_reconstructed", "provenance": captured}


def _truthy(value: Any) -> bool:
    return builder.truthy(value)


def _row_key(row: Mapping[str, Any]) -> tuple[str, str]:
    return str(row.get("subnet_id", "")), str(row.get("profile_id", ""))


def _replace_row(rows: Sequence[Mapping[str, Any]], row: Mapping[str, Any]) -> list[dict[str, Any]]:
    key = _row_key(row)
    kept = [dict(item) for item in rows if _row_key(item) != key]
    kept.append(dict(row))
    return sorted(kept, key=lambda item: (str(item.get("subnet_id", "")), str(item.get("profile_id", ""))))


def _remove_row(rows: Sequence[Mapping[str, Any]], subnet_id: str, profile_id: str) -> list[dict[str, Any]]:
    return [dict(row) for row in rows if _row_key(row) != (subnet_id, profile_id)]


def _latency(latency: Mapping[str, Any], key: str) -> Any:
    if key in latency:
        return latency.get(key)
    legacy = key.replace("forward_latency_", "forward_")
    return latency.get(legacy)


def _grouped_unsupported_count(subnet_dir: Path) -> int:
    rows = read_json(subnet_dir / "grouped_conv_int8_eligibility_report.json", [])
    return sum(1 for row in rows if isinstance(row, Mapping) and not row.get("int8_shape_supported"))


def build_lut_sample_label(
    *,
    subnet_dir: Path,
    profile_dir: Path,
    subnet_id: str,
    profile_id: str,
    source_subset: str,
    required_eval_frames: int,
    require_physical_metadata_v2: bool = False,
    failure_stage: str = "",
    failure_reason: str = "",
) -> dict[str, Any]:
    manifest = read_json(subnet_dir / "pruning_manifest.json", {})
    physical_hash = read_json(subnet_dir / "physical_hash_v2.json", {})
    profile = read_json(profile_dir / "mixed_precision_profile.json", {})
    qdq = read_json(profile_dir / "qdq_insert_report.json", {})
    build = read_json(profile_dir / "build_report.json", {})
    structure = read_json(profile_dir / "engine_structure_check_report.json", {})
    precision = read_json(profile_dir / "engine_precision_realization_report.json", {})
    smoke = read_json(profile_dir / "trt_smoke_report.json", {})
    eval_report = read_json(profile_dir / "eval_report.json", {})
    latency = eval_report.get("latency_summary") if isinstance(eval_report.get("latency_summary"), Mapping) else {}
    ap = eval_report.get("ap") if isinstance(eval_report.get("ap"), Mapping) else {}
    contract = summarize_surface_contract_from_manifest(manifest, source_subset=source_subset)
    grouped_unsupported = _grouped_unsupported_count(subnet_dir)
    true_mismatch = int(precision.get("precision_realization_mismatch_count", precision.get("mismatch_count", 0)) or 0)
    requested_int8_count = sum(1 for value in (profile.get("layer_precision_assignment") or {}).values() if str(value).lower() == "int8")
    requested_int8_modules = sorted(
        str(module)
        for module, precision_value in (profile.get("layer_precision_assignment") or {}).items()
        if str(precision_value).lower() == "int8"
    )
    deployment_profile_hash_v2 = compute_deployment_profile_hash_v2(
        shape_hash_v2=str(physical_hash.get("shape_hash_v2", "")),
        profile=profile,
        requested_int8_modules=requested_int8_modules,
    )
    qdq_nodes = len(qdq.get("inserted_qdq_nodes") or [])
    label_available = (
        bool(build.get("build_success", (profile_dir / "engine.plan").is_file()))
        and bool(structure.get("structure_check_passed"))
        and bool(precision.get("precision_realization_passed"))
        and true_mismatch == 0
        and bool(smoke.get("success"))
        and bool(eval_report.get("eval_success"))
        and int(eval_report.get("evaluated_frames") or 0) == int(required_eval_frames)
        and not bool(eval_report.get("synthetic_used"))
        and bool(eval_report.get("validation_dataloader_used"))
        and bool(qdq.get("success", requested_int8_count == 0))
        and (requested_int8_count == 0 or qdq_nodes > 0)
        and grouped_unsupported == 0
    )
    physical_metadata_v2_valid = bool(
        physical_hash.get("hash_schema_version") == HASH_SCHEMA_VERSION
        and physical_hash.get("structure_hash_v2")
        and physical_hash.get("shape_hash_v2")
        and read_json(profile_dir / "physical_structure_preflight_report.json", {}).get("preflight_passed")
    )
    if require_physical_metadata_v2:
        label_available = bool(label_available and physical_metadata_v2_valid)
    if not label_available and not failure_reason:
        failure_reason = str(
            eval_report.get("failure_reason")
            or smoke.get("failure_reason")
            or precision.get("failure_reason")
            or structure.get("failure_reason")
            or build.get("failure_reason")
            or "label_gate_not_satisfied"
        )
    if not label_available and not failure_stage:
        failure_stage = str(eval_report.get("status") or smoke.get("status") or precision.get("status") or structure.get("status") or build.get("status") or "label_unavailable")
    return {
        "dataset_version": DATASET_VERSION,
        "source_subset": source_subset,
        "subnet_id": subnet_id,
        "profile_id": profile_id,
        "structure_hash": str(manifest.get("structure_hash", "")),
        "shape_hash": str(manifest.get("shape_hash", "")),
        "hash_schema_version": physical_hash.get("hash_schema_version", ""),
        "structure_hash_v2": physical_hash.get("structure_hash_v2", ""),
        "shape_hash_v2": physical_hash.get("shape_hash_v2", ""),
        "deployment_profile_hash_v2": deployment_profile_hash_v2,
        "physical_metadata_v2_valid": physical_metadata_v2_valid,
        "onnx_sha256": sha256_file(subnet_dir / "onnx" / "model_signal_maxk.onnx"),
        "engine_sha256": sha256_file(profile_dir / "engine.plan"),
        "uses_taylor_ranking": bool(manifest.get("uses_taylor_ranking", False)),
        "random_sampling_method": manifest.get("random_sampling_method", ""),
        "actual_param_prune_ratio": manifest.get("actual_param_prune_ratio", manifest.get("achieved_global_param_prune_ratio")),
        "actual_channel_prune_ratio": manifest.get("actual_channel_prune_ratio", manifest.get("achieved_global_channel_prune_ratio")),
        "actual_weight_size_prune_ratio": manifest.get("actual_weight_size_prune_ratio", manifest.get("actual_param_prune_ratio", manifest.get("achieved_global_param_prune_ratio"))),
        "actual_bops_or_mac_prune_ratio": manifest.get("actual_bops_or_mac_prune_ratio", manifest.get("achieved_global_bops_prune_ratio")),
        **contract,
        "requested_int8_layer_count": requested_int8_count,
        "requested_int8_group_ratio": profile.get("requested_int8_group_ratio"),
        "inserted_qdq_nodes_count": qdq_nodes,
        "actual_int8_realized_count": int(precision.get("int8_realized_layer_count", 0) or 0),
        "fused_int8_with_fp16_boundary_count": int(precision.get("int8_compute_fp16_boundary_count", 0) or 0),
        "true_precision_mismatch_count": true_mismatch,
        "grouped_conv_unsupported_shape_count": grouped_unsupported,
        "engine_build_success": bool(build.get("build_success", (profile_dir / "engine.plan").is_file())),
        "engine_structure_check_passed": bool(structure.get("structure_check_passed")),
        "engine_precision_realization_check_passed": bool(precision.get("precision_realization_passed")),
        "trt_smoke_success": bool(smoke.get("success")),
        "eval_success": bool(eval_report.get("eval_success")),
        "evaluated_frames": int(eval_report.get("evaluated_frames") or 0),
        "synthetic_used": bool(eval_report.get("synthetic_used")),
        "validation_dataloader_used": bool(eval_report.get("validation_dataloader_used")),
        "skipped_frames": int(eval_report.get("skipped_frames", 0) or 0),
        "AP@0.03": ap.get("AP@0.03"),
        "AP@0.30": ap.get("AP@0.30"),
        "AP@0.50": ap.get("AP@0.50"),
        "AP@0.70": ap.get("AP@0.70"),
        "mAP": ap.get("mAP"),
        "forward_latency_mean_ms": _latency(latency, "forward_mean_ms"),
        "forward_latency_p50_ms": _latency(latency, "forward_p50_ms"),
        "forward_latency_p90_ms": _latency(latency, "forward_p90_ms"),
        "label_available": bool(label_available),
        "failure_stage": "" if label_available else failure_stage,
        "failure_reason": "" if label_available else failure_reason,
    }


def _label_success_complete(profile_dir: Path, required_eval_frames: int) -> bool:
    label = read_json(profile_dir / "lut_sample_label.json", {})
    return bool(label.get("label_available")) and int(label.get("evaluated_frames") or 0) == int(required_eval_frames)


def _retry_count(profile_dir: Path) -> int:
    failed = read_json(profile_dir / ".failed", {})
    try:
        return int(failed.get("retry_count", 0) or 0)
    except Exception:
        return 0


def _subnet_dirs(dataset_dir: Path, max_subnets: int = 0) -> list[Path]:
    dirs = sorted(path for path in (dataset_dir / "subnets").glob("subnet_*") if (path / "pruning_manifest.json").is_file())
    return dirs[: int(max_subnets)] if int(max_subnets) > 0 else dirs


def _subnet_index_from_id(subnet_id: str, fallback: int) -> int:
    try:
        return int(str(subnet_id).split("_")[-1])
    except Exception:
        return fallback


def _subnet_row(subnet_dir: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    physical_hash = read_json(subnet_dir / "physical_hash_v2.json", {})
    return {
        "subnet_id": str(manifest.get("subnet_id", subnet_dir.name)),
        "source_subset": manifest.get("source_subset", ""),
        "structure_hash": manifest.get("structure_hash", ""),
        "shape_hash": manifest.get("shape_hash", ""),
        "hash_schema_version": physical_hash.get("hash_schema_version", ""),
        "structure_hash_v2": physical_hash.get("structure_hash_v2", ""),
        "shape_hash_v2": physical_hash.get("shape_hash_v2", ""),
        "actual_param_prune_ratio": manifest.get("actual_param_prune_ratio", manifest.get("achieved_global_param_prune_ratio", "")),
        "actual_channel_prune_ratio": manifest.get("actual_channel_prune_ratio", manifest.get("achieved_global_channel_prune_ratio", "")),
        "pruning_manifest_path": str(subnet_dir / "pruning_manifest.json"),
    }


def _load_profile_allowlist(path_value: str | Path) -> set[tuple[str, str]]:
    path = Path(path_value) if str(path_value) else Path()
    if not str(path_value) or not path.is_file():
        return set()
    if path.suffix.lower() == ".csv":
        rows: Any = read_csv(path)
    else:
        payload = read_json(path, {})
        rows = payload.get("profiles", []) if isinstance(payload, Mapping) else payload
    return {
        (str(row.get("subnet_id", "")), str(row.get("profile_id", "")))
        for row in rows or []
        if isinstance(row, Mapping) and row.get("subnet_id") and row.get("profile_id")
    }


def _failure_matches_reasons(profile_dir: Path, reasons: set[str]) -> tuple[bool, str]:
    report = read_json(profile_dir / "profile_failure_report.json", {})
    if not report:
        return False, "profile_failure_report_missing"
    stage = str(report.get("stage_failed", ""))
    reason = str(report.get("failure_reason", ""))
    combined = f"{stage}:{reason}"
    if reasons and not any(value == stage or value == reason or value in combined for value in reasons):
        return False, f"failure_reason_not_selected:{combined}"
    return True, f"selected_failure:{combined}"


def _physical_metadata_v2_available(subnet_dir: Path) -> bool:
    snapshot = read_json(subnet_dir / "physical_structure_snapshot_v2.json", {})
    physical_hash = read_json(subnet_dir / "physical_hash_v2.json", {})
    return (
        (subnet_dir / "physical_structure_snapshot_v2.json").is_file()
        and isinstance(snapshot.get("modules"), list)
        and (subnet_dir / "physical_hash_v2.json").is_file()
        and bool(physical_hash.get("shape_hash_v2"))
    )


def build_pending_jobs(args: argparse.Namespace) -> tuple[list[ProfileJob], list[dict[str, Any]]]:
    dataset_dir = Path(args.dataset_dir)
    jobs: list[ProfileJob] = []
    subnet_rows: list[dict[str, Any]] = []
    failed_only = bool(getattr(args, "failed_only", False))
    allowlist_path = str(getattr(args, "profile_allowlist", "") or "")
    allowlist = _load_profile_allowlist(allowlist_path)
    failure_reasons = {value.strip() for value in str(getattr(args, "failure_reasons", "") or "").split(",") if value.strip()}
    for fallback_index, subnet_dir in enumerate(_subnet_dirs(dataset_dir, int(getattr(args, "max_subnets", 0) or 0))):
        manifest = read_json(subnet_dir / "pruning_manifest.json", {})
        subnet_id = str(manifest.get("subnet_id", subnet_dir.name))
        subnet_index = _subnet_index_from_id(subnet_id, fallback_index)
        structure_hash = str(manifest.get("structure_hash", ""))
        source_subset = str(manifest.get("source_subset", ""))
        subnet_rows.append(_subnet_row(subnet_dir, manifest))
        if bool(getattr(args, "require_physical_metadata_v2", False)) and not _physical_metadata_v2_available(subnet_dir):
            continue
        for profile_index in range(int(args.precision_profiles_per_subnet)):
            profile_id = f"profile_{profile_index:03d}"
            profile_dir = subnet_dir / profile_id
            key = (subnet_id, profile_id)
            if profile_dir.joinpath(".running.lock").exists():
                continue
            if allowlist_path and key not in allowlist:
                continue
            if failed_only:
                existing_label = read_json(profile_dir / "lut_sample_label.json", {})
                if bool(existing_label.get("label_available")):
                    continue
                selected, selection_reason = _failure_matches_reasons(profile_dir, failure_reasons)
                if not selected:
                    continue
            if bool(getattr(args, "skip_existing_success", False)) and _label_success_complete(profile_dir, int(args.eval_frame_count)):
                continue
            if _retry_count(profile_dir) > int(getattr(args, "max_retries", 1)):
                continue
            jobs.append(
                ProfileJob(
                    subnet_dir=subnet_dir,
                    subnet_id=subnet_id,
                    subnet_index=subnet_index,
                    structure_hash=structure_hash,
                    profile_id=profile_id,
                    profile_index=profile_index,
                    profile_dir=profile_dir,
                    source_subset=source_subset,
                )
            )
            if failed_only:
                print(json.dumps({"selected_profile": f"{subnet_id}/{profile_id}", "selection_reason": selection_reason}, sort_keys=True), flush=True)
    return jobs, subnet_rows


def _load_or_generate_profile(args: argparse.Namespace, job: ProfileJob) -> tuple[dict[str, Any], list[Any], set[str]]:
    groups = builder._precision_groups_from_json(job.subnet_dir / "precision_coupling_groups.json")
    normalized: dict[int, dict[str, Any]] = {}
    for profile_path in builder._profile_paths(job.subnet_dir):
        try:
            idx = int(profile_path.parent.name.split("_")[-1])
        except ValueError:
            continue
        normalized[idx] = builder._normalize_existing_profile(
            builder._read_profile(profile_path),
            profile_index=idx,
            subnet_id=job.subnet_id,
            structure_hash=job.structure_hash,
        )
    other_hashes = {str(profile.get("precision_assignment_hash", "")) for idx, profile in normalized.items() if idx != job.profile_index}
    if job.profile_index in normalized and not bool(args.overwrite_profiles) and builder._profile_matches_per_engine_template(normalized[job.profile_index], job.profile_index):
        profile = normalized[job.profile_index]
    else:
        profile = builder._generate_missing_profile(
            args=args,
            subnet_id=job.subnet_id,
            subnet_index=job.subnet_index,
            structure_hash=job.structure_hash,
            groups=groups,
            profile_index=job.profile_index,
            existing_hashes=other_hashes,
        )
        if profile is None:
            profile = builder.sample_stratified_mixed_precision_profile(
                subnet_id=job.subnet_id,
                structure_hash=job.structure_hash,
                groups=groups,
                profile_index=job.profile_index,
                profile_seed=int(args.profile_seed) + 99991,
                subnet_index=job.subnet_index,
                precision_modes=[p.strip().lower() for p in str(args.precision_modes).split(",") if p.strip()],
            )
    profile["profile_id"] = job.profile_id
    profile["profile_index"] = job.profile_index
    profile["dataset_version"] = DATASET_VERSION
    profile["source_subset"] = job.source_subset
    return profile, groups, other_hashes


def _adapt_builder_args(args: argparse.Namespace) -> argparse.Namespace:
    args.source_dir = str(args.dataset_dir)
    args.output_dir = str(args.dataset_dir)
    args.mode = "v12-lut-engine-eval-worker"
    args.build_engines = bool(args.build_engine)
    args.eval_engines = bool(args.run_eval)
    args.eval_frames = int(args.eval_frame_count)
    args.smoke_frames = int(args.smoke_frames)
    args.warmup_frames = int(args.warmup_frames)
    args.plugin = str(args.plugin_path)
    args.require_onnx_origin_map = True
    args.allow_precision_mismatch_eval = False
    args.enable_synthetic_trt_eval = False
    args.cuda_visible_devices = str(getattr(args, "worker_gpu", "") or "")
    args.device = str(getattr(args, "device", "") or ("cuda:0" if os.environ.get("CUDA_VISIBLE_DEVICES") else ""))
    return args


def _failure_report(job: ProfileJob, stage: str, reason: str, traceback_text: str = "") -> dict[str, Any]:
    return {
        "subnet_id": job.subnet_id,
        "profile_id": job.profile_id,
        "stage_failed": stage,
        "failure_reason": reason,
        "traceback": traceback_text,
        "recovery_action": "inspect_profile_artifacts_and_resume",
    }


def _minimal_failure_result(job: ProfileJob, *, status: str, reason: str, traceback_text: str = "") -> dict[str, Any]:
    failure = _failure_report(job, status, reason, traceback_text)
    return {
        "status": status,
        "subnet_id": job.subnet_id,
        "profile_id": job.profile_id,
        "index_row": {"subnet_id": job.subnet_id, "profile_id": job.profile_id, "structure_hash": job.structure_hash, "status": status, "build_success": False, "eval_success": False, "failure_reason": reason},
        "eval_row": {"subnet_id": job.subnet_id, "profile_id": job.profile_id, "structure_hash": job.structure_hash, "status": status, "build_success": False, "eval_success": False, "synthetic_used": False, "validation_dataloader_used": False, "evaluated_frames": 0, "failure_reason": reason},
        "component_rows": [],
        "training_row": {"subnet_id": job.subnet_id, "profile_id": job.profile_id, "structure_hash": job.structure_hash, "label_available": False, "failure_reason": reason},
        "lut_sample_label": {"dataset_version": DATASET_VERSION, "subnet_id": job.subnet_id, "profile_id": job.profile_id, "label_available": False, "failure_stage": status, "failure_reason": reason},
        "failure": failure,
    }


def _acquire_lock(profile_dir: Path, worker_id: str) -> bool:
    profile_dir.mkdir(parents=True, exist_ok=True)
    lock = profile_dir / ".running.lock"
    try:
        fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(json.dumps({"worker_id": worker_id, "pid": os.getpid(), "started_at": time.time()}) + "\n")
    return True


def _release_lock(profile_dir: Path) -> None:
    lock = profile_dir / ".running.lock"
    if lock.exists():
        lock.unlink()


def _mark_done(profile_dir: Path) -> None:
    (profile_dir / ".done").write_text(json.dumps({"done_at": time.time()}) + "\n", encoding="utf-8")
    failed = profile_dir / ".failed"
    if failed.exists():
        failed.unlink()


def _mark_failed(profile_dir: Path, result: Mapping[str, Any]) -> None:
    retries = _retry_count(profile_dir) + 1
    payload = {"retry_count": retries, "failed_at": time.time(), "status": result.get("status"), "failure": result.get("failure", {})}
    write_json(profile_dir / ".failed", payload)


def _preserve_before_recovery(profile_dir: Path) -> None:
    preserved_dir = profile_dir / "recovery_preserved_artifacts"
    for relative in (
        "profile_failure_report.json",
        "lut_sample_label.json",
        "engine_structure_check_report.json",
        "engine_precision_realization_report.json",
        "trt_smoke_report.json",
        "eval_report.json",
    ):
        source = profile_dir / relative
        destination = preserved_dir / relative
        if source.is_file() and not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)


def _preserve_engine_build_artifacts(profile_dir: Path) -> None:
    engine_hash = sha256_file(profile_dir / "engine.plan")[:16] or "missing"
    preserved_dir = profile_dir / "recovery_preserved_artifacts" / f"engine_build_{engine_hash}"
    for relative in ("engine.plan", "build_report.json", "build_log.txt", "trt_layer_info.json"):
        source = profile_dir / relative
        destination = preserved_dir / relative
        if source.is_file() and not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)


def _resolve_recovered_failure_report(profile_dir: Path) -> None:
    path = profile_dir / "profile_failure_report.json"
    report = read_json(path, {})
    if not report:
        return
    report.update(
        {
            "resolved": True,
            "resolution_status": "canonical_initializer_false_positive_recovered_with_physical_structure_v2",
            "resolved_at": datetime.now(timezone.utc).isoformat(),
            "resolved_by": "v12_failed_only_recovery",
        }
    )
    atomic_write_json(path, report)


def run_failed_profile_recovery_pipeline(ctx: dict[str, Any]) -> dict[str, Any]:
    """Recover an existing gated profile without regenerating ONNX or Q/DQ."""
    args: argparse.Namespace = ctx["args"]
    subnet_dir = Path(ctx["subnet_dir"])
    profile_dir = Path(ctx["profile_dir"])
    subnet_id = str(ctx["subnet_id"])
    profile_id = str(ctx["profile_id"])
    profile = ctx["profile"]
    groups = ctx["groups"]
    reports: dict[str, Any] = {"status": "recovery_started"}
    ctx["output_onnx"] = str(profile_dir / "onnx/model_mixed_qdq.onnx")
    try:
        qdq = read_json(profile_dir / "qdq_insert_report.json", {})
        mapping = read_json(profile_dir / "canonical_precision_mapping.json", {})
        if not (profile_dir / "onnx/model_mixed_qdq.onnx").is_file() or not mapping.get("entries") or not bool(qdq.get("success", True)):
            reason = "existing_onnx_qdq_or_canonical_mapping_not_reusable"
            reports["status"] = "onnx_or_qdq_failed"
            reports["failure"] = _failure_report(ProfileJob(subnet_dir, subnet_id, int(ctx.get("subnet_index", 0)), str(ctx.get("structure_hash", "")), profile_id, int(ctx.get("profile_index", 0)), profile_dir, ""), reports["status"], reason)
            return reports

        preflight = builder.run_physical_structure_preflight_stage(ctx)
        reports["preflight"] = preflight
        if not preflight.get("preflight_passed"):
            reports["status"] = "physical_structure_preflight_failed"
            reports["failure"] = _failure_report(ProfileJob(subnet_dir, subnet_id, int(ctx.get("subnet_index", 0)), str(ctx.get("structure_hash", "")), profile_id, int(ctx.get("profile_index", 0)), profile_dir, ""), reports["status"], str(preflight.get("failure_reason", "")))
            return reports

        reuse = {"engine_reuse_valid": False, "validation_source": "reuse_not_requested"}
        if bool(getattr(args, "reuse_existing_engine_if_valid", False)):
            reuse = validate_existing_engine_for_reuse(subnet_dir=subnet_dir, profile_dir=profile_dir, profile=profile)
        reports["engine_reuse"] = reuse
        if reuse.get("engine_reuse_valid"):
            build_report = read_json(profile_dir / "build_report.json", {})
            build_report = {**build_report, "engine_reused": True, "engine_reuse_validation_source": reuse.get("validation_source")}
        else:
            if not bool(getattr(args, "build_engine", False)):
                reports["status"] = "engine_build_failed"
                reports["failure"] = _failure_report(ProfileJob(subnet_dir, subnet_id, int(ctx.get("subnet_index", 0)), str(ctx.get("structure_hash", "")), profile_id, int(ctx.get("profile_index", 0)), profile_dir, ""), reports["status"], "existing_engine_not_reusable_and_build_engine_disabled")
                return reports
            _preserve_engine_build_artifacts(profile_dir)
            build_report = builder.run_engine_build_stage(ctx)
            if build_report.get("build_success"):
                capture_engine_provenance_v2(subnet_dir=subnet_dir, profile_dir=profile_dir, profile=profile, capture_mode="direct_post_build_hash_capture")
        reports["build"] = build_report
        ctx["engine_path"] = build_report.get("engine_path") or str(profile_dir / "engine.plan")
        if not build_report.get("build_success"):
            reports["status"] = "engine_build_failed"
            return reports

        structure = builder.run_engine_structure_stage(ctx)
        reports["structure"] = structure
        if not structure.get("structure_check_passed"):
            reports["status"] = "engine_structure_mismatch"
            return reports

        precision = builder.run_engine_precision_stage(ctx)
        reports["precision"] = precision
        if not precision.get("precision_realization_passed"):
            reports["status"] = "engine_precision_mismatch"
            return reports

        smoke = builder.run_trt_smoke_stage(ctx)
        reports["smoke"] = smoke
        if not smoke.get("success"):
            reports["status"] = "trt_smoke_failed"
            return reports

        eval_report = builder.run_real_eval_stage(ctx)
        reports["eval"] = eval_report
        builder._ensure_eval_artifacts(profile_dir, eval_report)
        if not eval_report.get("eval_success"):
            reports["status"] = "eval_failed"
            return reports
        reports["status"] = "eval_success"
        return reports
    finally:
        index_row, eval_row, component_rows, training_row = builder._profile_output_rows(
            subnet_id=subnet_id,
            profile_id=profile_id,
            structure_hash=str(ctx.get("structure_hash", "")),
            profile=profile,
            profile_dir=profile_dir,
            status=str(reports.get("status", "failed")),
            build_report=reports.get("build"),
            structure_report=reports.get("structure"),
            precision_report=reports.get("precision"),
            smoke_report=reports.get("smoke"),
            eval_report=reports.get("eval"),
            groups=groups,
        )
        reports["index_row"] = index_row
        reports["eval_row"] = eval_row
        reports["component_rows"] = component_rows
        reports["training_row"] = training_row


def run_worker(args: argparse.Namespace) -> int:
    args = _adapt_builder_args(args)
    if str(getattr(args, "worker_gpu", "")):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.worker_gpu)
        args.cuda_visible_devices = str(args.worker_gpu)
        args.device = "cuda:0"
    subnet_dir = Path(args.worker_subnet_dir)
    manifest = read_json(subnet_dir / "pruning_manifest.json", {})
    subnet_id = str(manifest.get("subnet_id", subnet_dir.name))
    job = ProfileJob(
        subnet_dir=subnet_dir,
        subnet_id=subnet_id,
        subnet_index=_subnet_index_from_id(subnet_id, 0),
        structure_hash=str(manifest.get("structure_hash", "")),
        profile_id=f"profile_{int(args.worker_profile_index):03d}",
        profile_index=int(args.worker_profile_index),
        profile_dir=subnet_dir / f"profile_{int(args.worker_profile_index):03d}",
        source_subset=str(manifest.get("source_subset", "")),
    )
    result_path = Path(args.worker_result_path) if args.worker_result_path else job.profile_dir / "worker_result.json"
    worker_id = f"gpu{getattr(args, 'worker_gpu', '')}:{os.getpid()}"
    if not _acquire_lock(job.profile_dir, worker_id):
        result = _minimal_failure_result(job, status="profile_locked", reason="running_lock_exists")
        write_json(result_path, result)
        return 1
    try:
        if bool(getattr(args, "failed_only", False)):
            _preserve_before_recovery(job.profile_dir)
        profile, groups, other_hashes = _load_or_generate_profile(args, job)
        ctx = {
            "args": args,
            "subnet_dir": job.subnet_dir,
            "subnet_id": job.subnet_id,
            "subnet_index": job.subnet_index,
            "structure_hash": job.structure_hash,
            "groups": groups,
            "profile": profile,
            "profile_id": job.profile_id,
            "profile_index": job.profile_index,
            "profile_dir": job.profile_dir,
            "existing_hashes": other_hashes,
        }
        result = run_failed_profile_recovery_pipeline(ctx) if bool(getattr(args, "failed_only", False)) else builder.run_one_profile_pipeline(ctx)
        label = build_lut_sample_label(
            subnet_dir=job.subnet_dir,
            profile_dir=job.profile_dir,
            subnet_id=job.subnet_id,
            profile_id=job.profile_id,
            source_subset=job.source_subset,
            required_eval_frames=int(args.eval_frame_count),
            require_physical_metadata_v2=bool(getattr(args, "require_physical_metadata_v2", False)),
        )
        write_json(job.profile_dir / "lut_sample_label.json", label)
        if label["label_available"]:
            result["status"] = "eval_success"
            if bool(getattr(args, "failed_only", False)):
                _resolve_recovered_failure_report(job.profile_dir)
            _mark_done(job.profile_dir)
        else:
            result["status"] = str(result.get("status") or "label_unavailable")
            failure = _failure_report(job, str(label.get("failure_stage") or result["status"]), str(label.get("failure_reason") or "label_unavailable"))
            result["failure"] = failure
            write_json(job.profile_dir / "profile_failure_report.json", failure)
            _mark_failed(job.profile_dir, result)
        result["lut_sample_label"] = label
        result["worker_gpu"] = str(getattr(args, "worker_gpu", ""))
        write_json(result_path, result)
        return 0 if label["label_available"] else 1
    except Exception as exc:  # noqa: BLE001
        result = _minimal_failure_result(job, status="worker_crashed", reason=f"{type(exc).__name__}: {exc}", traceback_text=traceback.format_exc())
        write_json(job.profile_dir / "profile_failure_report.json", result["failure"])
        write_json(job.profile_dir / "lut_sample_label.json", result["lut_sample_label"])
        write_json(result_path, result)
        _mark_failed(job.profile_dir, result)
        return 1
    finally:
        _release_lock(job.profile_dir)


def _load_index_state(output_dir: Path, args: argparse.Namespace, subnet_rows: list[dict[str, Any]]) -> IndexState:
    progress = {
        "mode": "v12_lut_engine_eval_workers",
        "dataset_version": DATASET_VERSION,
        "eval_frame_count": int(args.eval_frame_count),
        "profiles_started": 0,
        "profiles_completed": 0,
        "profiles_failed": 0,
        "engines_built": 0,
        "engines_structure_checked": 0,
        "engines_precision_checked": 0,
        "engines_smoke_passed": 0,
        "engines_eval_success": 0,
        "current_profile": "",
        "last_completed_profile": "",
        "last_failure": None,
        "failure_stage_counts": {},
        "stopped_early": False,
        "stop_reason": "",
    }
    return IndexState(
        output_dir=output_dir,
        args=args,
        subnet_rows=subnet_rows,
        profile_rows=read_csv(output_dir / "mixed_precision_profile_index.csv") if bool(args.resume) else [],
        eval_rows=read_csv(output_dir / "engine_eval_summary.csv") if bool(args.resume) else [],
        component_rows=read_csv(output_dir / "component_lut_samples.csv") if bool(args.resume) else [],
        training_rows=[json.loads(line) for line in (output_dir / "full_engine_training_samples.jsonl").read_text(encoding="utf-8").splitlines()] if bool(args.resume) and (output_dir / "full_engine_training_samples.jsonl").is_file() else [],
        label_rows=read_csv(output_dir / "lut_sample_labels.csv") if bool(args.resume) else [],
        failure_rows=read_csv(output_dir / "failure_summary.csv") if bool(args.resume) else [],
        progress_state=progress,
    )


def _write_state(state: IndexState) -> None:
    out = state.output_dir
    write_csv(out / "subnet_index.csv", state.subnet_rows)
    write_csv(out / "mixed_precision_profile_index.csv", state.profile_rows)
    write_csv(out / "engine_eval_summary.csv", state.eval_rows)
    write_csv(out / "component_lut_samples.csv", state.component_rows)
    write_jsonl(out / "full_engine_training_samples.jsonl", state.training_rows)
    write_csv(out / "lut_sample_labels.csv", state.label_rows)
    write_csv(out / "failure_summary.csv", state.failure_rows)
    write_json(out / "progress_state.json", state.progress_state)
    manifest = {
        "dataset_version": DATASET_VERSION,
        "mode": "v12_lut_engine_eval_workers",
        "eval_frame_count": int(state.args.eval_frame_count),
        "successful_subnet_count": len(state.subnet_rows),
        "successful_label_count": sum(_truthy(row.get("label_available")) for row in state.label_rows),
        "engine_eval_success_count": sum(_truthy(row.get("eval_success")) and int(row.get("evaluated_frames") or 0) == int(state.args.eval_frame_count) for row in state.label_rows),
        "failure_summary": dict(state.progress_state.get("failure_stage_counts", {})),
        "stopped_early": bool(state.progress_state.get("stopped_early")),
        "stop_reason": state.progress_state.get("stop_reason", ""),
    }
    write_json(out / "manifest.json", manifest)


def _apply_result(state: IndexState, result: Mapping[str, Any]) -> bool:
    status = str(result.get("status", "worker_failed"))
    index_row = dict(result.get("index_row") or {})
    eval_row = dict(result.get("eval_row") or {})
    label = dict(result.get("lut_sample_label") or {})
    subnet_id = str(label.get("subnet_id") or index_row.get("subnet_id") or result.get("subnet_id", ""))
    profile_id = str(label.get("profile_id") or index_row.get("profile_id") or result.get("profile_id", ""))
    if index_row:
        state.profile_rows = _replace_row(state.profile_rows, index_row)
    if eval_row:
        state.eval_rows = _replace_row(state.eval_rows, eval_row)
    if result.get("component_rows"):
        state.component_rows = _remove_row(state.component_rows, subnet_id, profile_id) + [dict(row) for row in result.get("component_rows", [])]
    if result.get("training_row"):
        training = dict(result["training_row"])
        training["pilot_label_only"] = False
        state.training_rows = _remove_row(state.training_rows, subnet_id, profile_id) + [training]
    if label:
        state.label_rows = _replace_row(state.label_rows, label)
    if label.get("label_available"):
        state.consecutive_failures = 0
        state.failure_rows = _remove_row(state.failure_rows, subnet_id, profile_id)
    else:
        state.consecutive_failures += 1
        failure = dict(result.get("failure") or _failure_report(
            ProfileJob(Path(), subnet_id, 0, str(label.get("structure_hash", "")), profile_id, 0, Path(), str(label.get("source_subset", ""))),
            str(label.get("failure_stage") or status),
            str(label.get("failure_reason") or status),
        ))
        if str(failure.get("stage_failed", status)) in GATE_FAILURE_STAGES:
            state.total_gate_failures += 1
        state.failure_rows = _replace_row(state.failure_rows, failure)
        state.progress_state["last_failure"] = failure
        counts = dict(state.progress_state.get("failure_stage_counts", {}))
        stage = str(failure.get("stage_failed") or status)
        counts[stage] = counts.get(stage, 0) + 1
        state.progress_state["failure_stage_counts"] = counts
        state.progress_state["profiles_failed"] = int(state.progress_state.get("profiles_failed", 0)) + 1
    state.progress_state["profiles_completed"] = int(state.progress_state.get("profiles_completed", 0)) + 1
    state.progress_state["last_completed_profile"] = f"{subnet_id}/{profile_id}"
    state.progress_state["current_profile"] = ""
    state.progress_state["engines_built"] = sum(_truthy(row.get("engine_build_success")) for row in state.label_rows)
    state.progress_state["engines_structure_checked"] = sum(_truthy(row.get("engine_structure_check_passed")) for row in state.label_rows)
    state.progress_state["engines_precision_checked"] = sum(_truthy(row.get("engine_precision_realization_check_passed")) for row in state.label_rows)
    state.progress_state["engines_smoke_passed"] = sum(_truthy(row.get("trt_smoke_success")) for row in state.label_rows)
    state.progress_state["engines_eval_success"] = sum(_truthy(row.get("label_available")) for row in state.label_rows)
    stopped = state.consecutive_failures >= int(state.args.max_consecutive_failures)
    if stopped:
        state.progress_state["stopped_early"] = True
        state.progress_state["stop_reason"] = f"failure_threshold:consecutive={state.consecutive_failures}"
    _write_state(state)
    return stopped


def _worker_slots(args: argparse.Namespace) -> list[tuple[str, str]]:
    gpus = [gpu.strip() for gpu in str(args.gpus).split(",") if gpu.strip()]
    slots: list[tuple[str, str]] = []
    for gpu in gpus:
        for idx in range(int(args.workers_per_gpu)):
            slots.append((f"{gpu}:{idx}", gpu))
    return slots


def build_worker_command(args: argparse.Namespace, job: ProfileJob, *, slot_id: str, gpu: str, result_path: Path) -> tuple[list[str], dict[str, str], Path]:
    log_dir = Path(args.log_dir) if str(args.log_dir) else Path(args.dataset_dir) / "logs" / "v12_worker_profiles"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{job.subnet_id}_{job.profile_id}_gpu{gpu}_slot{slot_id.replace(':', '_')}.log"
    cmd = [
        sys.executable,
        str(_THIS),
        "--worker",
        "--dataset-dir",
        str(args.dataset_dir),
        "--precision-profiles-per-subnet",
        str(args.precision_profiles_per_subnet),
        "--profile-seed",
        str(args.profile_seed),
        "--precision-modes",
        str(args.precision_modes),
        "--eval-frame-count",
        str(args.eval_frame_count),
        "--warmup-frames",
        str(args.warmup_frames),
        "--smoke-frames",
        str(args.smoke_frames),
        "--calib-train-frames",
        str(args.calib_train_frames),
        "--trt-root",
        str(args.trt_root),
        "--trt-build-timeout-seconds",
        str(args.trt_build_timeout_seconds),
        "--fixed-k",
        str(args.fixed_k),
        "--plugin-path",
        str(args.plugin_path),
        "--heal-root",
        str(args.heal_root),
        "--model-config",
        str(args.model_config),
        "--checkpoint",
        str(args.checkpoint),
        "--num-workers",
        str(args.num_workers),
        "--worker-subnet-dir",
        str(job.subnet_dir),
        "--worker-profile-index",
        str(job.profile_index),
        "--worker-gpu",
        str(gpu),
        "--worker-result-path",
        str(result_path),
    ]
    for flag, enabled in (
        ("--build-engine", args.build_engine),
        ("--run-structure-check", args.run_structure_check),
        ("--run-precision-check", args.run_precision_check),
        ("--run-smoke", args.run_smoke),
        ("--run-eval", args.run_eval),
        ("--require-engine-structure-check", args.require_engine_structure_check),
        ("--require-precision-realization-check", args.require_precision_realization_check),
        ("--require-validation-dataloader", args.require_validation_dataloader),
        ("--forbid-synthetic-eval", args.forbid_synthetic_eval),
        ("--reuse-existing-engine-if-valid", getattr(args, "reuse_existing_engine_if_valid", False)),
        ("--require-physical-metadata-v2", getattr(args, "require_physical_metadata_v2", False)),
        ("--require-preflight-pass", getattr(args, "require_preflight_pass", False)),
        ("--failed-only", getattr(args, "failed_only", False)),
    ):
        if bool(enabled):
            cmd.append(flag)
    if args.trtexec:
        cmd.extend(["--trtexec", str(args.trtexec)])
    if bool(args.resume):
        cmd.append("--resume")
    if bool(args.skip_existing_success):
        cmd.append("--skip-existing-success")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    trt_root = Path(str(args.trt_root))
    ld_parts = [
        "/home/lixingfeng/anaconda3/envs/modelopt/lib",
        str(trt_root / "lib"),
        str(trt_root / "targets/x86_64-linux-gnu/lib"),
    ]
    if env.get("LD_LIBRARY_PATH"):
        ld_parts.append(env["LD_LIBRARY_PATH"])
    env["LD_LIBRARY_PATH"] = ":".join(ld_parts)
    pp = [str(_UNIAD), str(_ROOT), str(Path(args.heal_root))]
    if env.get("PYTHONPATH"):
        pp.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = ":".join(pp)
    env["CUDA_HOME"] = "/home/lixingfeng/anaconda3/envs/modelopt"
    return cmd, env, log_path


def _read_worker_result(path: Path, job: ProfileJob, returncode: int) -> dict[str, Any]:
    if path.is_file():
        result = read_json(path, {})
        if returncode != 0 and str(result.get("status", "")) == "eval_success":
            result["status"] = "worker_returncode_failed"
            result["failure_reason"] = f"worker_returncode={returncode}"
        return result
    return _minimal_failure_result(job, status="worker_result_missing", reason=f"worker_returncode={returncode}, result missing: {path}")


def terminate_active_workers(active: Mapping[str, RunningWorker]) -> None:
    for worker in active.values():
        if worker.process.poll() is None:
            worker.process.terminate()
    deadline = time.time() + 30
    for worker in active.values():
        if worker.process.poll() is not None:
            continue
        try:
            worker.process.wait(timeout=max(0.0, deadline - time.time()))
        except subprocess.TimeoutExpired:
            worker.process.kill()


def run_coordinator(args: argparse.Namespace) -> int:
    args = _adapt_builder_args(args)
    output_dir = Path(args.dataset_dir)
    jobs, subnet_rows = build_pending_jobs(args)
    state = _load_index_state(output_dir, args, subnet_rows)
    state.progress_state["queued_profiles"] = len(jobs)
    _write_state(state)
    slots = _worker_slots(args)
    if not slots:
        raise ValueError("--gpus must contain at least one GPU id")
    pending = list(jobs)
    active: dict[str, RunningWorker] = {}
    stopped = False
    while pending or active:
        while pending and len(active) < len(slots):
            slot_id, gpu = next((slot for slot in slots if slot[0] not in active), ("", ""))
            if not slot_id:
                break
            job = pending.pop(0)
            result_path = job.profile_dir / "worker_result.json"
            if result_path.exists():
                result_path.unlink()
            cmd, env, log_path = build_worker_command(args, job, slot_id=slot_id, gpu=gpu, result_path=result_path)
            log_handle = log_path.open("w", encoding="utf-8")
            state.progress_state["profiles_started"] = int(state.progress_state.get("profiles_started", 0)) + 1
            state.progress_state["current_profile"] = f"{job.subnet_id}/{job.profile_id}@gpu{gpu}"
            _write_state(state)
            process = subprocess.Popen(cmd, cwd=str(_ROOT), env=env, stdout=log_handle, stderr=subprocess.STDOUT)
            active[slot_id] = RunningWorker(job=job, slot_id=slot_id, gpu=gpu, process=process, log_handle=log_handle, log_path=log_path, result_path=result_path)
        completed: list[str] = []
        for slot_id, worker in list(active.items()):
            rc = worker.process.poll()
            if rc is None:
                continue
            worker.log_handle.close()
            result = _read_worker_result(worker.result_path, worker.job, int(rc))
            stopped = _apply_result(state, result)
            completed.append(slot_id)
            if stopped:
                break
        for slot_id in completed:
            active.pop(slot_id, None)
        if stopped:
            terminate_active_workers(active)
            for worker in active.values():
                worker.log_handle.close()
            _write_state(state)
            print(json.dumps({"success": False, "stopped_early": True, "stop_reason": state.progress_state.get("stop_reason", ""), "output_dir": str(output_dir)}, indent=2))
            return 2
        if active:
            time.sleep(float(args.poll_seconds))
    state.progress_state["current_profile"] = ""
    _write_state(state)
    print(json.dumps({"success": True, "output_dir": str(output_dir), "profiles_completed": state.progress_state.get("profiles_completed", 0)}, indent=2))
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default="outputs/latency_lut/v12_combined_lut_dataset_300frames")
    parser.add_argument("--max-subnets", type=int, default=0)
    parser.add_argument("--precision-profiles-per-subnet", type=int, default=4)
    parser.add_argument("--gpus", default="1,2,6,7")
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--log-dir", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-existing-success", action="store_true")
    parser.add_argument("--failed-only", action="store_true")
    parser.add_argument("--failure-reasons", default="")
    parser.add_argument("--profile-allowlist", default="")
    parser.add_argument("--reuse-existing-engine-if-valid", action="store_true")
    parser.add_argument("--require-physical-metadata-v2", action="store_true")
    parser.add_argument("--require-preflight-pass", action="store_true")
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--max-consecutive-failures", type=int, default=999999)
    parser.add_argument("--eval-frame-count", type=int, default=300)
    parser.add_argument("--build-engine", action="store_true")
    parser.add_argument("--run-structure-check", action="store_true")
    parser.add_argument("--run-precision-check", action="store_true")
    parser.add_argument("--run-smoke", action="store_true")
    parser.add_argument("--run-eval", action="store_true")
    parser.add_argument("--require-engine-structure-check", action="store_true")
    parser.add_argument("--require-precision-realization-check", action="store_true")
    parser.add_argument("--require-validation-dataloader", action="store_true")
    parser.add_argument("--forbid-synthetic-eval", action="store_true")
    parser.add_argument("--profile-seed", type=int, default=20260708)
    parser.add_argument("--precision-modes", default="fp32,fp16,int8")
    parser.add_argument("--overwrite-profiles", action="store_true")
    parser.add_argument("--calib-train-frames", type=int, default=200)
    parser.add_argument("--warmup-frames", type=int, default=100)
    parser.add_argument("--smoke-frames", type=int, default=5)
    parser.add_argument("--ap-thresholds", default="0.03,0.30,0.50,0.70")
    parser.add_argument("--trt-root", default="/home/lixingfeng/UniAD_examine/HEAL/prune_model/TensorRT-10.9_x86_cu118")
    parser.add_argument("--trtexec", default="")
    parser.add_argument("--trt-build-timeout-seconds", type=int, default=180)
    parser.add_argument("--fixed-k", "--fixed_k", dest="fixed_k", type=int, default=29696)
    parser.add_argument("--plugin-path", default="quantization/plugins/pointpillar_scatter_trt/build/libpointpillar_scatter_trt.so")
    parser.add_argument("--heal-root", default="/home/lixingfeng/UniAD_examine/HEAL")
    parser.add_argument("--model-config", default="/home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_pyramid/config.yaml")
    parser.add_argument("--checkpoint", default="/home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_pyramid/net_epoch_bestval_at17.pth")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-subnet-dir", default="", help=argparse.SUPPRESS)
    parser.add_argument("--worker-profile-index", type=int, default=-1, help=argparse.SUPPRESS)
    parser.add_argument("--worker-gpu", default="", help=argparse.SUPPRESS)
    parser.add_argument("--worker-result-path", default="", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.worker:
        return run_worker(args)
    return run_coordinator(args)


if __name__ == "__main__":
    raise SystemExit(main())
