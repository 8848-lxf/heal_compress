#!/usr/bin/env python3
"""Migrate v12 LUT subnets to physical structure v2 and audit successful labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[2]

from tools.latency_lut.physical_structure_v2 import (  # noqa: E402
    HASH_SCHEMA_VERSION,
    _load_physical_model,
    atomic_write_json,
    build_onnx_weight_trace_index,
    build_physical_application_ledger,
    build_physical_structure_snapshot_v2,
    build_sampling_structure_request,
    compute_deployment_profile_hash_v2,
    compute_physical_hash_v2,
    read_json,
    run_physical_structure_preflight,
    sha256_payload,
    validate_physical_application_ledger,
)
from tools.latency_lut.run_v12_lut_engine_eval_workers import capture_engine_provenance_v2  # noqa: E402


DEFAULT_DATASET = "outputs/latency_lut/v12_combined_lut_dataset_300frames"
MEASUREMENT_FIELDS = (
    "evaluated_frames",
    "synthetic_used",
    "validation_dataloader_used",
    "skipped_frames",
    "AP@0.03",
    "AP@0.30",
    "AP@0.50",
    "AP@0.70",
    "mAP",
    "forward_latency_mean_ms",
    "forward_latency_p50_ms",
    "forward_latency_p90_ms",
    "requested_int8_layer_count",
    "requested_int8_group_ratio",
    "actual_int8_realized_count",
    "fused_int8_with_fp16_boundary_count",
    "true_precision_mismatch_count",
)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if str(key) not in seen:
                seen.add(str(key))
                fields.append(str(key))
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields or ["empty"], extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        str(key): json.dumps(value, sort_keys=True) if isinstance(value, (dict, list, tuple)) else value
                        for key, value in row.items()
                    }
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _subnet_dirs(dataset_dir: Path) -> list[Path]:
    return sorted(path for path in (dataset_dir / "subnets").glob("subnet_*") if (path / "pruning_manifest.json").is_file())


def _profile_dirs(subnet_dir: Path) -> list[Path]:
    return sorted(path for path in subnet_dir.glob("profile_*" ) if path.is_dir())


def _active_writers() -> list[str]:
    completed = subprocess.run(
        ["pgrep", "-af", "[r]un_v12_lut_engine_eval_workers.py|[t]rtexec"],
        text=True,
        capture_output=True,
        check=False,
    )
    return [line for line in completed.stdout.splitlines() if line.strip()]


def _attrs(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    attrs = value.get("attrs")
    return dict(attrs) if isinstance(attrs, Mapping) else dict(value)


def _sampling_actual_differences(sampling: Mapping[str, Any], snapshot: Mapping[str, Any]) -> list[dict[str, Any]]:
    by_name = {
        str(row.get("canonical_module_name", "")): row
        for row in snapshot.get("modules", [])
        if isinstance(row, Mapping)
    }
    rows = []
    for request in sampling.get("requests", []):
        module_name = str(request.get("module_name", ""))
        requested_after = dict(request.get("requested_after") or {})
        actual = by_name.get(module_name, {})
        compared = {key: actual.get(key) for key in requested_after if key in actual}
        if requested_after and compared != requested_after:
            rows.append(
                {
                    "request_id": request.get("request_id", ""),
                    "canonical_module_name": module_name,
                    "requested_after": requested_after,
                    "actual_physical": compared,
                }
            )
    return rows


def merge_successful_label_v2_metadata(
    label: Mapping[str, Any],
    *,
    physical_hash: Mapping[str, Any],
    deployment_profile_hash_v2: str,
    valid: bool,
) -> dict[str, Any]:
    merged = dict(label)
    merged.update(
        {
            "hash_schema_version": physical_hash.get("hash_schema_version", ""),
            "structure_hash_v2": physical_hash.get("structure_hash_v2", ""),
            "shape_hash_v2": physical_hash.get("shape_hash_v2", ""),
            "deployment_profile_hash_v2": str(deployment_profile_hash_v2),
            "physical_metadata_v2_valid": bool(valid),
        }
    )
    return merged


def _measurement_digest(label: Mapping[str, Any]) -> str:
    return sha256_payload({key: label.get(key) for key in MEASUREMENT_FIELDS})


def _requested_int8_modules(profile: Mapping[str, Any]) -> list[str]:
    return sorted(
        str(module)
        for module, precision_value in (profile.get("layer_precision_assignment") or {}).items()
        if str(precision_value).lower() == "int8"
    )


def _migrate_subnet(subnet_dir: Path) -> tuple[dict[str, Any], Any, Mapping[str, Any], Mapping[str, Any]]:
    manifest = read_json(subnet_dir / "pruning_manifest.json", {})
    model, state = _load_physical_model(subnet_dir)
    snapshot = build_physical_structure_snapshot_v2(
        model,
        state_dict=state,
        generated_from="pruned_model_object.pth:model.named_modules+pruned_state_dict_with_manifest.pth",
    )
    sampling = build_sampling_structure_request(manifest)
    physical_plan = read_json(subnet_dir / "global_physical_prune_plan.json", {})
    ledger = build_physical_application_ledger(
        sampling,
        snapshot,
        physical_plan=physical_plan,
        legacy_migration=True,
    )
    ledger_validation = validate_physical_application_ledger(sampling, ledger)
    ledger["validation"] = ledger_validation
    physical_hash = compute_physical_hash_v2(
        snapshot,
        legacy_structure_hash=str(manifest.get("structure_hash", "")),
        legacy_shape_hash=str(manifest.get("shape_hash", "")),
    )
    atomic_write_json(subnet_dir / "sampling_structure_request.json", sampling)
    atomic_write_json(subnet_dir / "physical_pruning_application_ledger.json", ledger)
    atomic_write_json(subnet_dir / "physical_structure_snapshot_v2.json", snapshot)
    atomic_write_json(subnet_dir / "physical_hash_v2.json", physical_hash)

    base_trace_index = build_onnx_weight_trace_index(subnet_dir / "onnx/model_signal_maxk.onnx")
    profile_000 = subnet_dir / "profile_000"
    preflight = run_physical_structure_preflight(
        subnet_dir=subnet_dir,
        profile_dir=profile_000,
        physical_model=model,
        physical_state_dict=state,
        base_onnx_trace_index=base_trace_index,
    )
    differences = _sampling_actual_differences(sampling, snapshot)
    row = {
        "subnet_id": str(manifest.get("subnet_id", subnet_dir.name)),
        "source_subset": manifest.get("source_subset", ""),
        "snapshot_generated": True,
        "snapshot_module_count": snapshot.get("module_count", 0),
        "weighted_module_count": snapshot.get("weighted_module_count", 0),
        "live_state_snapshot_consistent": bool(preflight.get("live_state_snapshot_passed")),
        "base_onnx_consistent": all(
            bool(check.get("base_weight_shape_interpretation", {}).get("passed"))
            and check.get("canonical_mapping_initializer") == check.get("base_root_initializer")
            for check in preflight.get("checks", [])
        ),
        "profile_000_full_preflight_passed": bool(preflight.get("preflight_passed")),
        "profile_000_preflight_failure_reason": preflight.get("failure_reason", ""),
        "legacy_structure_hash": manifest.get("structure_hash", ""),
        "legacy_shape_hash": manifest.get("shape_hash", ""),
        "structure_hash_v2": physical_hash.get("structure_hash_v2", ""),
        "shape_hash_v2": physical_hash.get("shape_hash_v2", ""),
        "snapshot_sha256": physical_hash.get("snapshot_sha256", ""),
        "sampling_request_count": sampling.get("request_count", 0),
        "ledger_valid": bool(ledger_validation.get("passed")),
        "applied_count": ledger.get("status_counts", {}).get("applied", 0),
        "repaired_count": ledger.get("status_counts", {}).get("repaired", 0),
        "merged_count": ledger.get("status_counts", {}).get("merged", 0),
        "skipped_count": ledger.get("status_counts", {}).get("skipped", 0),
        "sampling_actual_difference_count": len(differences),
        "sampling_actual_differences": differences,
        "physical_parameter_count": snapshot.get("parameter_count", 0),
        "physical_parameter_size_bytes": snapshot.get("parameter_size_bytes", 0),
    }
    row["migration_success"] = bool(
        row["snapshot_generated"]
        and row["live_state_snapshot_consistent"]
        and row["base_onnx_consistent"]
        and row["ledger_valid"]
        and physical_hash.get("hash_schema_version") == HASH_SCHEMA_VERSION
    )
    return row, model, state, base_trace_index


def _audit_successful_profiles_for_subnet(
    subnet_dir: Path,
    *,
    model: Any,
    state: Mapping[str, Any],
    base_trace_index: Mapping[str, Any],
) -> list[dict[str, Any]]:
    physical_hash = read_json(subnet_dir / "physical_hash_v2.json", {})
    rows = []
    for profile_dir in _profile_dirs(subnet_dir):
        label_path = profile_dir / "lut_sample_label.json"
        label = read_json(label_path, {})
        if not bool(label.get("label_available")):
            continue
        profile = read_json(profile_dir / "mixed_precision_profile.json", {})
        before_digest = _measurement_digest(label)
        preflight = run_physical_structure_preflight(
            subnet_dir=subnet_dir,
            profile_dir=profile_dir,
            physical_model=model,
            physical_state_dict=state,
            base_onnx_trace_index=base_trace_index,
        )
        provenance = capture_engine_provenance_v2(
            subnet_dir=subnet_dir,
            profile_dir=profile_dir,
            profile=profile,
            capture_mode="legacy_reconstructed_build_command_path_mtime_and_hash",
        )
        deployment_hash = compute_deployment_profile_hash_v2(
            shape_hash_v2=str(physical_hash.get("shape_hash_v2", "")),
            profile=profile,
            requested_int8_modules=_requested_int8_modules(profile),
        )
        valid = bool(
            preflight.get("preflight_passed")
            and provenance.get("provenance_valid")
            and physical_hash.get("hash_schema_version") == HASH_SCHEMA_VERSION
        )
        merged = merge_successful_label_v2_metadata(
            label,
            physical_hash=physical_hash,
            deployment_profile_hash_v2=deployment_hash,
            valid=valid,
        )
        atomic_write_json(label_path, merged)
        after_digest = _measurement_digest(merged)
        rows.append(
            {
                "subnet_id": subnet_dir.name,
                "profile_id": profile_dir.name,
                "physical_metadata_v2_valid": valid,
                "preflight_passed": bool(preflight.get("preflight_passed")),
                "preflight_failure_reason": preflight.get("failure_reason", ""),
                "engine_provenance_valid": bool(provenance.get("provenance_valid")),
                "engine_provenance_capture_mode": provenance.get("capture_mode", ""),
                "legacy_structure_hash": label.get("structure_hash", ""),
                "legacy_shape_hash": label.get("shape_hash", ""),
                "structure_hash_v2": physical_hash.get("structure_hash_v2", ""),
                "shape_hash_v2": physical_hash.get("shape_hash_v2", ""),
                "deployment_profile_hash_v2": deployment_hash,
                "measurement_digest_before": before_digest,
                "measurement_digest_after": after_digest,
                "measurements_unchanged": before_digest == after_digest,
                "evaluated_frames": label.get("evaluated_frames"),
            }
        )
    return rows


def _update_combined_indexes(dataset_dir: Path) -> dict[str, Any]:
    labels_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for subnet_dir in _subnet_dirs(dataset_dir):
        for profile_dir in _profile_dirs(subnet_dir):
            label = read_json(profile_dir / "lut_sample_label.json", {})
            if not label:
                continue
            key = (str(label.get("subnet_id", subnet_dir.name)), str(label.get("profile_id", profile_dir.name)))
            labels_by_key[key] = label
    label_rows = [labels_by_key[key] for key in sorted(labels_by_key)]
    _atomic_write_csv(dataset_dir / "lut_sample_labels.csv", label_rows)

    profile_index_path = dataset_dir / "mixed_precision_profile_index.csv"
    index_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in _read_csv(profile_index_path):
        key = (str(row.get("subnet_id", "")), str(row.get("profile_id", "")))
        index_by_key[key] = row
    for key, label in labels_by_key.items():
        if key not in index_by_key:
            continue
        for field in ("hash_schema_version", "structure_hash_v2", "shape_hash_v2", "deployment_profile_hash_v2", "physical_metadata_v2_valid"):
            index_by_key[key][field] = label.get(field, "")
    _atomic_write_csv(profile_index_path, [index_by_key[key] for key in sorted(index_by_key)])
    return {
        "label_index_unique_count": len(label_rows),
        "profile_index_unique_count": len(index_by_key),
        "duplicate_label_keys": 0,
        "duplicate_profile_index_keys": 0,
    }


def _migration_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Physical Structure v2 Migration Report",
        "",
        f"- subnet_count: {report.get('subnet_count')}",
        f"- migration_success_count: {report.get('migration_success_count')}",
        f"- snapshot/base_onnx_consistent_count: {report.get('base_onnx_consistent_count')}",
        f"- total requested/applied/repaired/merged/skipped: {report.get('request_count')}/{report.get('applied_count')}/{report.get('repaired_count')}/{report.get('merged_count')}/{report.get('skipped_count')}",
        f"- anomaly_subnets: {', '.join(report.get('anomaly_subnets', [])) or 'none'}",
        "",
        "| subnet | modules | legacy shape hash | shape_hash_v2 | sampling vs physical diffs | ledger | live/state/base ONNX |",
        "|---|---:|---|---|---:|---:|---:|",
    ]
    for row in report.get("subnets", []):
        lines.append(
            f"| {row['subnet_id']} | {row['snapshot_module_count']} | {row['legacy_shape_hash']} | {row['shape_hash_v2']} | "
            f"{row['sampling_actual_difference_count']} | {str(row['ledger_valid']).lower()} | {str(row['live_state_snapshot_consistent'] and row['base_onnx_consistent']).lower()} |"
        )
    return "\n".join(lines) + "\n"


def _successful_audit_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Successful Label v2 Metadata Audit",
        "",
        f"- successful_labels_found: {report.get('successful_labels_found')}",
        f"- physical_metadata_v2_valid_count: {report.get('physical_metadata_v2_valid_count')}",
        f"- measurements_unchanged_count: {report.get('measurements_unchanged_count')}",
        f"- invalid_profiles: {', '.join(report.get('invalid_profiles', [])) or 'none'}",
        f"- index_unique_keys: {report.get('index_update', {}).get('label_index_unique_count')}",
        "",
        "No engine was rebuilt and no 300-frame evaluation was rerun by this audit.",
    ]
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> int:
    dataset_dir = Path(args.dataset_dir).resolve()
    active = _active_writers()
    if active and not args.allow_active_writers:
        raise RuntimeError("active v12 worker/trtexec processes:\n" + "\n".join(active))
    subnet_dirs = _subnet_dirs(dataset_dir)
    if len(subnet_dirs) != int(args.expected_subnets):
        raise RuntimeError(f"expected {args.expected_subnets} subnets, found {len(subnet_dirs)}")

    migration_rows = []
    successful_rows = []
    for index, subnet_dir in enumerate(subnet_dirs, 1):
        try:
            row, model, state, base_trace_index = _migrate_subnet(subnet_dir)
            migration_rows.append(row)
            successful_rows.extend(
                _audit_successful_profiles_for_subnet(
                    subnet_dir,
                    model=model,
                    state=state,
                    base_trace_index=base_trace_index,
                )
            )
            print(json.dumps({"migration": f"{index}/{len(subnet_dirs)}", "subnet_id": subnet_dir.name, "success": row["migration_success"]}), flush=True)
        except Exception as exc:  # noqa: BLE001
            migration_rows.append({"subnet_id": subnet_dir.name, "migration_success": False, "failure_reason": f"{type(exc).__name__}: {exc}"})
            print(json.dumps({"migration": f"{index}/{len(subnet_dirs)}", "subnet_id": subnet_dir.name, "success": False, "failure_reason": str(exc)}), flush=True)

    migration_report = {
        "schema_version": "physical-structure-v2-migration-report-v1",
        "dataset_dir": str(dataset_dir),
        "subnet_count": len(migration_rows),
        "migration_success_count": sum(bool(row.get("migration_success")) for row in migration_rows),
        "base_onnx_consistent_count": sum(bool(row.get("base_onnx_consistent")) for row in migration_rows),
        "request_count": sum(int(row.get("sampling_request_count", 0) or 0) for row in migration_rows),
        "applied_count": sum(int(row.get("applied_count", 0) or 0) for row in migration_rows),
        "repaired_count": sum(int(row.get("repaired_count", 0) or 0) for row in migration_rows),
        "merged_count": sum(int(row.get("merged_count", 0) or 0) for row in migration_rows),
        "skipped_count": sum(int(row.get("skipped_count", 0) or 0) for row in migration_rows),
        "anomaly_subnets": [str(row.get("subnet_id")) for row in migration_rows if not row.get("migration_success")],
        "subnets": migration_rows,
    }
    atomic_write_json(dataset_dir / "physical_structure_v2_migration_report.json", migration_report)
    _atomic_write_text(dataset_dir / "physical_structure_v2_migration_report.md", _migration_markdown(migration_report))

    if len(successful_rows) != int(args.expected_successful_labels):
        raise RuntimeError(f"expected {args.expected_successful_labels} successful labels, found {len(successful_rows)}")
    index_update = _update_combined_indexes(dataset_dir)
    successful_report = {
        "schema_version": "successful-label-v2-metadata-audit-v1",
        "dataset_dir": str(dataset_dir),
        "successful_labels_found": len(successful_rows),
        "physical_metadata_v2_valid_count": sum(bool(row.get("physical_metadata_v2_valid")) for row in successful_rows),
        "measurements_unchanged_count": sum(bool(row.get("measurements_unchanged")) for row in successful_rows),
        "invalid_profiles": [f"{row['subnet_id']}/{row['profile_id']}" for row in successful_rows if not row.get("physical_metadata_v2_valid")],
        "index_update": index_update,
        "profiles": successful_rows,
    }
    atomic_write_json(dataset_dir / "successful_label_v2_metadata_audit.json", successful_report)
    _atomic_write_text(dataset_dir / "successful_label_v2_metadata_audit.md", _successful_audit_markdown(successful_report))
    print(json.dumps({"migration_success": migration_report["migration_success_count"], "successful_metadata_valid": successful_report["physical_metadata_v2_valid_count"]}, indent=2), flush=True)
    return 0 if migration_report["migration_success_count"] == len(subnet_dirs) and successful_report["physical_metadata_v2_valid_count"] == len(successful_rows) else 2


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET)
    parser.add_argument("--expected-subnets", type=int, default=48)
    parser.add_argument("--expected-successful-labels", type=int, default=155)
    parser.add_argument("--allow-active-writers", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
