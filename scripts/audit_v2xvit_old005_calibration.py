#!/usr/bin/env python3
"""Fail-closed audit of the historical V2X-ViT 0.05 train200 calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED = {
    "algorithm",
    "requested_frames",
    "processed_frames",
    "skipped_frames",
    "manifest_hash",
    "checkpoint_hash",
    "physical_hash",
    "state_dict_shape_hash",
    "precision_map_hash",
    "onnx_hash",
    "cache_hash",
    "scale_hash",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    root = args.old_root.expanduser()
    candidates = sorted(root.rglob("*calibration*.json")) if root.is_dir() else []
    valid = []
    evidence = []
    for path in candidates:
        try:
            payload: Any = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            evidence.append({"path": str(path), "valid": False, "reason": f"parse:{exc}"})
            continue
        metadata = dict(payload.get("metadata") or payload) if isinstance(payload, dict) else {}
        missing = sorted(REQUIRED - set(metadata))
        row = {
            "path": str(path),
            "valid": not missing,
            "missing_fields": missing,
            "processed_frames": metadata.get("processed_frames"),
            "skipped_frames": metadata.get("skipped_frames"),
        }
        evidence.append(row)
        if row["valid"] and metadata.get("processed_frames") == 200 and metadata.get("skipped_frames") == 0:
            valid.append(row)
    report = {
        "schema_version": "v2xvit-old005-calibration-audit-v1",
        "old_root": str(root),
        "old_root_exists": root.is_dir(),
        "candidate_calibration_files": len(candidates),
        "fully_bound_train200_records": len(valid),
        "old_train200_calibration_verified": bool(valid),
        "required_fields": sorted(REQUIRED),
        "evidence": evidence,
        "conclusion": (
            "verified"
            if valid
            else "not_verified_missing_or_incomplete_historical_provenance"
        ),
    }
    for path in (args.output_json, args.output_md):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise RuntimeError(f"refusing_to_overwrite:{path}")
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.write_text(
        "# Historical 0.05 train200 calibration audit\n\n"
        f"- old root exists: `{report['old_root_exists']}`\n"
        f"- calibration JSON files: `{len(candidates)}`\n"
        f"- fully bound records: `{len(valid)}`\n"
        f"- old_train200_calibration_verified: `{report['old_train200_calibration_verified']}`\n\n"
        "A JMIX-FRESH directory name is not calibration evidence. Every cache-invalidating hash must be present.\n",
        encoding="utf-8",
    )
    print(json.dumps({"old_train200_calibration_verified": bool(valid)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
