#!/usr/bin/env python3
"""Derive deterministic, naturally in-band GA populations from the Greedy trace."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


BUDGETS = ("010",)
TARGETS = {label: int(label) / 100.0 for label in BUDGETS}


def state_hash(state: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(state, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f"refusing_to_overwrite:{path}")
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def run(root: Path) -> None:
    pools: dict[str, dict[str, dict[str, Any]]] = {label: {} for label in BUDGETS}
    baseline_widths: dict[str, int] | None = None
    trace = root / "greedy_trace.csv"
    with trace.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            retention = float(row["current_retention"])
            labels = [
                label for label, target in TARGETS.items()
                if abs(retention - target) <= 0.005
            ]
            if not labels:
                continue
            state = json.loads(row["state_after"])
            if baseline_widths is None:
                # The original widths are the maximum seen per locus.  Exact
                # winner payloads below supply a complete fallback.
                baseline_widths = {key: int(value) for key, value in state["widths"].items()}
            identity = state_hash(state)
            precision_counts = {
                name: sum(value == name for value in state["precision"].values())
                for name in ("FP32", "FP16", "INT8")
            }
            payload = {
                "state_hash": identity,
                "state": state,
                "R_bops": retention,
                "bops_deviation": min(abs(retention - TARGETS[label]) for label in labels),
                "parameter_retention": float(row["R_parameter_retention"]),
                "precision_counts": precision_counts,
                "source_step": int(row["step"]),
                "source_candidate_hash": row["candidate_hash"],
                "source": (
                    "greedy_selected_budget_band_candidate"
                    if str(row.get("selected", "")).lower() == "true"
                    else "greedy_unselected_legal_neighbor_candidate"
                ),
                "repair_count": 0,
            }
            for label in labels:
                pools[label].setdefault(identity, payload)

    output: dict[str, Any] = {}
    for label in BUDGETS:
        exact = json.loads((root / f"greedy/budget_{label}/exact_winner.json").read_text())
        exact_state = {
            "widths": exact["genotype"]["pruning_width_genes"],
            "precision": exact["genotype"]["precision_genes"],
        }
        exact_hash = state_hash(exact_state)
        rows = list(pools[label].values())
        # Preserve deliberately different structure/precision tradeoffs.  No
        # proxy, AP, latency, or repair is used to create these strata.
        structure_first = sorted(
            rows, key=lambda row: (-float(row["parameter_retention"]), row["state_hash"])
        )
        precision_first = sorted(
            rows,
            key=lambda row: (
                -int(row["precision_counts"]["FP16"] + row["precision_counts"]["INT8"]),
                row["state_hash"],
            ),
        )
        mixed = sorted(
            rows,
            key=lambda row: (
                abs(float(row["parameter_retention"]) - 0.5 * (
                    min(float(item["parameter_retention"]) for item in rows)
                    + max(float(item["parameter_retention"]) for item in rows)
                )),
                row["state_hash"],
            ),
        )
        chosen: dict[str, dict[str, Any]] = {
            exact_hash: {
                "state_hash": exact_hash,
                "state": exact_state,
                "R_bops": float(exact["bops"]["R_bops_vs_fp32"]),
                "source": "exact_greedy_anchor",
                "repair_count": 0,
            }
        }
        strata = (
            ("structure_biased", structure_first),
            ("precision_biased", precision_first),
            ("mixed", mixed),
        )
        for stratum, sequence in strata:
            added = 0
            for row in sequence:
                if added >= 16:
                    break
                item = dict(row)
                item["initialization_stratum"] = stratum
                before = len(chosen)
                chosen.setdefault(str(item["state_hash"]), item)
                added += len(chosen) - before
        for row in sorted(rows, key=lambda row: row["state_hash"]):
            if len(chosen) >= 64:
                break
            item = dict(row)
            item["initialization_stratum"] = "deterministic_global"
            chosen.setdefault(str(item["state_hash"]), item)
        if len(chosen) != 64:
            raise RuntimeError(f"ga_initial_pool_not_64:budget_{label}:{len(chosen)}")
        output[label] = {
            "budget": TARGETS[label],
            "tolerance_abs": 0.005,
            "population_size": 64,
            "exact_greedy_anchor_hash": exact["candidate_hash"],
            "candidate_count_available": len(rows),
            "candidates": list(chosen.values()),
            "repair_count": 0,
        }
    write(root / "ga/initial_populations.json", output)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    run(args.output_root.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
