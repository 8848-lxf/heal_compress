#!/usr/bin/env python3
"""Create a provenance-locked formal prefix aggregate from extra repeats.

This utility never mutates the source run. It is intended for a protocol
revision where a completed five-repeat run contains the exact first three
serial repeats required by the current formal contract.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from search.integration.heal_lidar_family_fair_evaluation import (
    sha256_file,
    write_csv,
    write_json,
)


def _load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    for row in rows:
        row["repeat_index"] = int(row["repeat_index"])
        if row.get("budget") not in (None, ""):
            row["budget"] = float(row["budget"])
        else:
            row["budget"] = None
    return rows


def _aggregate_for_kind(
    kind: str,
) -> Callable[[list[dict[str, Any]], int], list[dict[str, Any]]]:
    if kind == "cnn-joint":
        from scripts.run_cnn_formal_joint_full1789_repeat5 import _aggregate

        return _aggregate
    if kind == "pyramid-joint":
        from scripts.run_pyramid_greedy_ga_full1789_repeat5 import _aggregate

        return _aggregate
    if kind == "pyramid-pq":
        from scripts.run_pyramid_latest_pq_decomposition_repeat5 import _aggregate

        return _aggregate
    raise ValueError(f"formal_repeat_prefix_kind_unknown:{kind}")


def run(args: argparse.Namespace) -> int:
    if int(args.repeat_count) < 2:
        raise ValueError("formal_repeat_prefix_count_must_be_at_least_two")
    source_root = args.source_root.resolve()
    source_csv = source_root / "reports/repeat_results.csv"
    source_report = source_root / "reports/final_report.json"
    if not source_csv.is_file() or not source_report.is_file():
        raise RuntimeError(
            f"formal_repeat_prefix_source_incomplete:{source_csv}:{source_report}"
        )
    original = json.loads(source_report.read_text(encoding="utf-8"))
    if not bool(original.get("passed")):
        raise RuntimeError("formal_repeat_prefix_source_failed")
    if int(original.get("repeat_count", 0)) < int(args.repeat_count):
        raise RuntimeError("formal_repeat_prefix_source_too_short")

    rows = _load_rows(source_csv)
    selected = [
        row
        for row in rows
        if int(row["repeat_index"]) < int(args.repeat_count)
    ]
    if not selected:
        raise RuntimeError("formal_repeat_prefix_empty")
    observed = sorted({int(row["repeat_index"]) for row in selected})
    expected = list(range(int(args.repeat_count)))
    if observed != expected:
        raise RuntimeError(f"formal_repeat_prefix_missing:{observed}:{expected}")
    if not all(
        int(row["num_evaluated_frames"]) == 1789
        and int(row["num_skipped_frames"]) == 0
        and math.isfinite(float(row["mAP"]))
        for row in selected
    ):
        raise RuntimeError("formal_repeat_prefix_evaluation_invalid")

    aggregate = _aggregate_for_kind(args.kind)(selected, int(args.repeat_count))
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    (output_root / "reports").mkdir()
    write_csv(output_root / "reports/repeat_results.csv", selected)
    write_csv(
        output_root / f"reports/repeat{int(args.repeat_count)}_mean_std.csv",
        aggregate,
    )
    provenance = {
        "schema_version": "formal-repeat-prefix-reaggregation-v1",
        "kind": args.kind,
        "source_root": str(source_root),
        "source_repeat_results_sha256": sha256_file(source_csv),
        "source_final_report_sha256": sha256_file(source_report),
        "source_repeat_count": int(original["repeat_count"]),
        "formal_repeat_count": int(args.repeat_count),
        "selected_repeat_indices": expected,
        "source_artifacts_modified": False,
        "evaluation_rerun": False,
    }
    write_json(output_root / "provenance.json", provenance)
    write_json(output_root / "reports/final_report.json", {
        "passed": True,
        "formal_protocol": True,
        "repeat_count": int(args.repeat_count),
        "evaluation_frames": 1789,
        "result_count": len(selected),
        "aggregate": aggregate,
        "provenance": provenance,
    })
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kind", choices=("cnn-joint", "pyramid-joint", "pyramid-pq"),
        required=True,
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repeat-count", type=int, default=3)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
