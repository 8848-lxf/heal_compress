#!/usr/bin/env python3
"""Fail AV16 closed after a legal pruned shape disproves global closure."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def run(root: Path) -> int:
    reports = root / "reports"
    failed = root / "engines/greedy_exact_winners/budget_010/JMIX-FRESH"
    audit = read(failed / "av_profile_trt_audit.json")
    calibration = read(failed / "calibration_manifest.json")
    if (
        int(calibration.get("processed_frames", -1)) != 200
        or int(calibration.get("skipped_frames", -1)) != 0
        or audit.get("profile_counts") != {"AV32": 0, "AV16": 12, "AV8": 0}
        or int(audit.get("conflict_count", 0)) <= 0
        or int(audit.get("fallback_count", 0)) <= 0
    ):
        raise RuntimeError("v2xvit_av16_shape_closure_counterexample_invalid")

    acceptance_path = reports / "av_profile_acceptance.json"
    acceptance = read(acceptance_path)
    for row in acceptance["profiles"]:
        if row["profile"] == "AV16":
            row["original_structure_legal"] = bool(row.get("legal_for_search"))
            row["legal_for_search"] = False
            row["pruned_shape_deployment_closed"] = False
            row["pruned_shape_conflict_count"] = int(audit["conflict_count"])
            row["pruned_shape_fallback_count"] = int(audit["fallback_count"])
            row["exclusion_reason"] = (
                "requested AV16 was exact on the original structure, but six of "
                "twelve instances had no exact FP16 AV compute layer on a legal "
                "R=0.10 pruned phenotype"
            )
    acceptance["legal_av_profiles"] = ["AV32"]
    acceptance["shape_closure_counterexample"] = str(
        failed / "av_profile_trt_audit.json"
    )
    acceptance["formal_search_policy"] = "AV32_only"
    write(acceptance_path, acceptance)
    write(reports / "av_profile_shape_closure_addendum.json", {
        "schema_version": "v2xvit-av-pruned-shape-closure-addendum-v1",
        "original_structure_av16_valid": True,
        "pruned_shape_av16_deployment_closed": False,
        "requested_av16_count": 12,
        "conflict_count": int(audit["conflict_count"]),
        "fallback_count": int(audit["fallback_count"]),
        "unmapped_count": int(audit["unmapped_count"]),
        "failing_nodes": [
            row["canonical_node"] for row in audit["rows"] if row["fallback"]
        ],
        "legal_av_profiles": ["AV32"],
        "old_search_space_invalidated": True,
        "old_taylor_cache_invalidated": True,
        "old_greedy_invalidated": True,
    })
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True, type=Path)
    return run(parser.parse_args().output_root.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
