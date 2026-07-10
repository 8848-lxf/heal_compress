#!/usr/bin/env python3
"""Preflight and allowlist only the 35 v12 canonical-shape false positives."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from tools.latency_lut.migrate_v12_physical_structure_v2 import _atomic_write_csv, _atomic_write_text
from tools.latency_lut.physical_structure_v2 import (
    _load_physical_model,
    atomic_write_json,
    build_onnx_weight_trace_index,
    read_json,
    run_physical_structure_preflight,
)
from tools.latency_lut.run_v12_lut_engine_eval_workers import validate_existing_engine_for_reuse


DEFAULT_DATASET = "outputs/latency_lut/v12_combined_lut_dataset_300frames"
CANONICAL_MISMATCH_RECOVERY_TARGETS = tuple(
    (subnet_id, f"profile_{profile_index:03d}")
    for subnet_id, profile_indices in (
        ("subnet_003", (0, 2, 3)),
        ("subnet_004", (0, 1, 2, 3)),
        ("subnet_019", (0, 1, 2, 3)),
        ("subnet_021", (0, 1, 2, 3)),
        ("subnet_022", (0, 1, 2, 3)),
        ("subnet_023", (0, 1, 2, 3)),
        ("subnet_028", (0, 1, 2, 3)),
        ("subnet_029", (0, 1, 2, 3)),
        ("subnet_031", (0, 1, 2, 3)),
    )
    for profile_index in profile_indices
)
EXCLUDED_TIMEOUT_PROFILES = (("subnet_003", "profile_001"), ("subnet_043", "profile_002"))


def _active_writers() -> list[str]:
    completed = subprocess.run(
        ["pgrep", "-af", "[r]un_v12_lut_engine_eval_workers.py|[t]rtexec"],
        text=True,
        capture_output=True,
        check=False,
    )
    return [line for line in completed.stdout.splitlines() if line.strip()]


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Canonical Mismatch Recovery Preflight",
        "",
        f"- requested_targets: {report['requested_target_count']}",
        f"- selected_profiles: {report['selected_profile_count']}",
        f"- preflight_passed: {report['preflight_passed_count']}",
        f"- existing_engines_reusable: {report['existing_engine_reusable_count']}",
        f"- excluded_timeout_profiles: {', '.join(report['excluded_timeout_profiles'])}",
        f"- rejected_profiles: {', '.join(report['rejected_profiles']) or 'none'}",
        "",
        "| subnet/profile | preflight | engine reusable | old failure | selection reason |",
        "|---|---:|---:|---|---|",
    ]
    for row in report["profiles"]:
        lines.append(
            f"| {row['subnet_id']}/{row['profile_id']} | {str(row['preflight_passed']).lower()} | "
            f"{str(row['engine_reuse_valid']).lower()} | {row['original_failure_reason']} | {row['selection_reason']} |"
        )
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> int:
    dataset_dir = Path(args.dataset_dir).resolve()
    active = _active_writers()
    if active:
        raise RuntimeError("active v12 worker/trtexec processes:\n" + "\n".join(active))
    targets_by_subnet: dict[str, list[str]] = defaultdict(list)
    for subnet_id, profile_id in CANONICAL_MISMATCH_RECOVERY_TARGETS:
        targets_by_subnet[subnet_id].append(profile_id)

    rows = []
    selected = []
    for subnet_id in sorted(targets_by_subnet):
        subnet_dir = dataset_dir / "subnets" / subnet_id
        model, state = _load_physical_model(subnet_dir)
        base_index = build_onnx_weight_trace_index(subnet_dir / "onnx/model_signal_maxk.onnx")
        for profile_id in sorted(targets_by_subnet[subnet_id]):
            profile_dir = subnet_dir / profile_id
            failure = read_json(profile_dir / "profile_failure_report.json", {})
            label = read_json(profile_dir / "lut_sample_label.json", {})
            profile = read_json(profile_dir / "mixed_precision_profile.json", {})
            preflight = run_physical_structure_preflight(
                subnet_dir=subnet_dir,
                profile_dir=profile_dir,
                physical_model=model,
                physical_state_dict=state,
                base_onnx_trace_index=base_index,
            )
            reuse = validate_existing_engine_for_reuse(subnet_dir=subnet_dir, profile_dir=profile_dir, profile=profile)
            original_reason = str(failure.get("failure_reason", ""))
            original_selected_failure = "canonical_onnx_initializer_shape_mismatch" in original_reason
            grouped_rows = read_json(subnet_dir / "grouped_conv_int8_eligibility_report.json", [])
            unsupported_grouped = sum(1 for item in grouped_rows if isinstance(item, dict) and not item.get("int8_shape_supported"))
            selected_for_recovery = bool(
                original_selected_failure
                and not bool(label.get("label_available"))
                and preflight.get("preflight_passed")
                and unsupported_grouped == 0
                and (subnet_dir / "physical_hash_v2.json").is_file()
            )
            reasons = []
            if original_selected_failure:
                reasons.append("original_failure_reason_matches")
            if preflight.get("preflight_passed"):
                reasons.append("physical_structure_v2_preflight_passed")
            if reuse.get("engine_reuse_valid"):
                reasons.append("existing_engine_provenance_valid")
            elif selected_for_recovery:
                reasons.append("engine_rebuild_required")
            row = {
                "subnet_id": subnet_id,
                "profile_id": profile_id,
                "selected": selected_for_recovery,
                "selection_reason": ";".join(reasons),
                "original_failure_stage": failure.get("stage_failed", ""),
                "original_failure_reason": original_reason,
                "label_available_before": bool(label.get("label_available")),
                "preflight_passed": bool(preflight.get("preflight_passed")),
                "preflight_failure_reason": preflight.get("failure_reason", ""),
                "engine_reuse_valid": bool(reuse.get("engine_reuse_valid")),
                "engine_reuse_validation_source": reuse.get("validation_source", ""),
                "grouped_conv_unsupported_shape_count": unsupported_grouped,
            }
            rows.append(row)
            if selected_for_recovery:
                selected.append(row)
            print(json.dumps({"profile": f"{subnet_id}/{profile_id}", "selected": selected_for_recovery, "reason": row["selection_reason"]}, sort_keys=True), flush=True)

    if len(selected) != 35:
        raise RuntimeError(f"recovery allowlist must contain exactly 35 profiles, got {len(selected)}")
    allowlist_payload = {
        "schema_version": "v12-canonical-mismatch-recovery-allowlist-v1",
        "failure_reason_filter": "canonical_onnx_initializer_shape_mismatch",
        "profile_count": len(selected),
        "excluded_timeout_profiles": [f"{subnet}/{profile}" for subnet, profile in EXCLUDED_TIMEOUT_PROFILES],
        "profiles": selected,
    }
    allowlist_json = dataset_dir / "canonical_mismatch_recovery_allowlist.json"
    allowlist_csv = dataset_dir / "canonical_mismatch_recovery_allowlist.csv"
    atomic_write_json(allowlist_json, allowlist_payload)
    _atomic_write_csv(allowlist_csv, selected)
    report = {
        "schema_version": "v12-canonical-mismatch-recovery-preflight-v1",
        "requested_target_count": len(CANONICAL_MISMATCH_RECOVERY_TARGETS),
        "selected_profile_count": len(selected),
        "preflight_passed_count": sum(bool(row["preflight_passed"]) for row in rows),
        "existing_engine_reusable_count": sum(bool(row["engine_reuse_valid"]) for row in selected),
        "engine_rebuild_required_count": sum(not bool(row["engine_reuse_valid"]) for row in selected),
        "excluded_timeout_profiles": [f"{subnet}/{profile}" for subnet, profile in EXCLUDED_TIMEOUT_PROFILES],
        "rejected_profiles": [f"{row['subnet_id']}/{row['profile_id']}" for row in rows if not row["selected"]],
        "allowlist_json": str(allowlist_json),
        "allowlist_csv": str(allowlist_csv),
        "profiles": rows,
    }
    atomic_write_json(dataset_dir / "canonical_mismatch_recovery_preflight_report.json", report)
    _atomic_write_text(dataset_dir / "canonical_mismatch_recovery_preflight_report.md", _markdown(report))
    print(json.dumps({"selected_profiles": len(selected), "preflight_passed": report["preflight_passed_count"], "engine_reusable": report["existing_engine_reusable_count"]}, indent=2), flush=True)
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
