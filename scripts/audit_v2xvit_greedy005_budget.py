#!/usr/bin/env python3
"""Replay the immutable 0.05 Greedy budget-band selector and distribution."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median
from typing import Any


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _precision_counts(profile: dict[str, str]) -> dict[str, int]:
    return {state: sum(value == state for value in profile.values()) for state in ("INT8", "FP16", "FP32")}


def _widths(state: dict[str, Any], prefix: str) -> list[int]:
    widths = state.get("widths", {})
    return [int(value) for key, value in sorted(widths.items()) if key.startswith(prefix)]


def _row_key_taylor(row: dict[str, Any]) -> tuple[Any, ...]:
    return (row["cumulative_total_taylor"], row["candidate_hash"])


def _summary(row: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "candidate_hash", "step", "R_BOPS", "bops_deviation", "cumulative_total_taylor",
        "cumulative_structural_taylor", "cumulative_weight_quant_taylor", "parameter_count",
        "parameter_retention", "mixed_weight_size_bytes", "shrinker_width", "attention_dh",
        "ffn_dff", "INT8_count", "FP16_count", "FP32_count",
    )
    return {key: row[key] for key in keys}


def run(old_root: Path, output_root: Path) -> int:
    trace_path = old_root / "greedy_trace.csv"
    winner_path = old_root / "winner/v2xvit_greedy005_winner.json"
    winner = json.loads(winner_path.read_text(encoding="utf-8"))
    base_parameters = float(winner["size"]["parameter_count_base"])
    original_hash = str(winner["candidate_hash"])
    band: dict[str, dict[str, Any]] = {}
    with trace_path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            retention = float(raw["current_retention"])
            if abs(retention - 0.05) > 0.005:
                continue
            state = json.loads(raw["state_after"])
            profile = state.get("precision", {})
            counts = _precision_counts(profile)
            parameter_retention = float(raw["R_parameter_retention"])
            mixed_size = float(raw["mixed_weight_size_bytes"])
            widths = state.get("widths", {})
            shrinker = int(widths["shrinker_m1.layers.0.double_conv.0::out"])
            row = {
                "candidate_hash": raw["candidate_hash"],
                "step": int(raw["step"]),
                "R_BOPS": retention,
                "bops_deviation": abs(retention - 0.05),
                "cumulative_total_taylor": float(raw["cumulative_proxy"]),
                "cumulative_structural_taylor": float(raw["cumulative_pruning_taylor"]),
                "cumulative_weight_quant_taylor": float(raw["cumulative_weight_quantization_taylor"]),
                "parameter_count": parameter_retention * base_parameters,
                "parameter_retention": parameter_retention,
                "parameter_prune_ratio": 1.0 - parameter_retention,
                "mixed_weight_size_bytes": mixed_size,
                "mixed_weight_compression_ratio": base_parameters * 4.0 / mixed_size,
                "structure_state_json": json.dumps(widths, sort_keys=True, separators=(",", ":")),
                "precision_map_json": json.dumps(profile, sort_keys=True, separators=(",", ":")),
                "shrinker_width": shrinker,
                "attention_dh": json.dumps(_widths(state, "attention_dh::"), separators=(",", ":")),
                "ffn_dff": json.dumps(_widths(state, "ffn_hidden::"), separators=(",", ":")),
                **{f"{key}_count": value for key, value in counts.items()},
            }
            incumbent = band.get(row["candidate_hash"])
            old_key = (row["cumulative_total_taylor"], row["bops_deviation"], row["mixed_weight_size_bytes"], row["candidate_hash"])
            incumbent_key = None if incumbent is None else (incumbent["cumulative_total_taylor"], incumbent["bops_deviation"], incumbent["mixed_weight_size_bytes"], incumbent["candidate_hash"])
            if incumbent is None or old_key < incumbent_key:
                band[row["candidate_hash"]] = row
    rows = list(band.values())
    if len(rows) != 1457:
        raise RuntimeError(f"old005_budget_candidate_count_mismatch:{len(rows)}!=1457")
    by_taylor = sorted(rows, key=_row_key_taylor)
    for rank, row in enumerate(by_taylor, 1):
        row["pure_taylor_rank"] = rank
    by_params = sorted(rows, key=lambda row: (-row["parameter_retention"], row["candidate_hash"]))
    for rank, row in enumerate(by_params, 1):
        row["parameter_retention_rank"] = rank
    by_size = sorted(rows, key=lambda row: (row["mixed_weight_size_bytes"], row["candidate_hash"]))
    for rank, row in enumerate(by_size, 1):
        row["mixed_size_rank"] = rank
    original_selector = min(rows, key=lambda row: (row["cumulative_total_taylor"], row["bops_deviation"], row["mixed_weight_size_bytes"], row["candidate_hash"]))
    taylor_only = min(rows, key=lambda row: (row["cumulative_total_taylor"], row["candidate_hash"]))
    taylor_deviation = min(rows, key=lambda row: (row["cumulative_total_taylor"], row["bops_deviation"], row["candidate_hash"]))
    best = by_taylor[0]["cumulative_total_taylor"]
    equivalent = [row for row in rows if abs(row["cumulative_total_taylor"] - best) / max(abs(best), 1e-30) < 1e-8]
    equivalence_no_compression_tiebreak = min(equivalent, key=lambda row: (row["bops_deviation"], row["candidate_hash"]))
    values = [row["cumulative_total_taylor"] for row in rows]
    second = by_taylor[1]["cumulative_total_taylor"]
    winner_row = next(row for row in rows if row["candidate_hash"] == original_hash)
    distribution = {
        "candidate_count": len(rows),
        "taylor": {
            "min": min(values), "second_best": second, "p1": _percentile(values, 0.01),
            "p5": _percentile(values, 0.05), "p50": median(values), "p95": _percentile(values, 0.95),
            "max": max(values), "top2_relative_difference": (second - best) / max(abs(best), 1e-30),
            "within_best_0.1_percent": sum(value <= best * 1.001 for value in values),
            "within_best_0.5_percent": sum(value <= best * 1.005 for value in values),
            "within_best_1_percent": sum(value <= best * 1.01 for value in values),
        },
        "winner": {**_summary(winner_row), "pure_taylor_rank": winner_row["pure_taylor_rank"], "parameter_retention_rank": winner_row["parameter_retention_rank"], "mixed_size_rank": winner_row["mixed_size_rank"]},
    }
    replay = {
        "source_winner_hash": original_hash,
        "A_original_selector": _summary(original_selector),
        "B_taylor_only": _summary(taylor_only),
        "C_taylor_then_bops_deviation": _summary(taylor_deviation),
        "D_equivalence_1e_minus_8_without_parameter_or_size_tiebreak": _summary(equivalence_no_compression_tiebreak),
        "equivalence_candidate_count": len(equivalent),
        "selector_replay_exact": original_selector["candidate_hash"] == original_hash,
        "parameter_prune_tiebreak_present_in_source": False,
        "mixed_weight_size_tiebreak_present_in_source": True,
        "winner_changed_without_mixed_size_tiebreak": original_selector["candidate_hash"] != taylor_deviation["candidate_hash"],
        "structural_collapse_attribution": "greedy_action_trajectory" if original_selector["candidate_hash"] == taylor_only["candidate_hash"] else "budget_selector_tiebreak_contributed",
    }
    output_root.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with (output_root / "budget_candidate_distribution.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(sorted(rows, key=lambda row: (row["cumulative_total_taylor"], row["candidate_hash"])))
    (output_root / "taylor_distribution.json").write_text(json.dumps(distribution, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_root / "winner_selector_replay.json").write_text(json.dumps(replay, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Old 0.05 winner selector replay", "",
        f"- Budget-band candidates: {len(rows)}", f"- Selector replay exact: `{replay['selector_replay_exact']}`",
        f"- Pure Taylor winner: `{taylor_only['candidate_hash']}`", f"- Original winner: `{original_hash}`",
        f"- Parameter-prune tie-break exists: `{replay['parameter_prune_tiebreak_present_in_source']}`",
        f"- Mixed-size tie-break exists: `{replay['mixed_weight_size_tiebreak_present_in_source']}`",
        f"- Winner changes when mixed-size tie-break is removed: `{replay['winner_changed_without_mixed_size_tiebreak']}`",
        f"- Collapse attribution: `{replay['structural_collapse_attribution']}`", "",
    ]
    (output_root / "winner_selector_replay.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"candidate_count": len(rows), "winner": original_hash, "pure_taylor": taylor_only["candidate_hash"], "selector_replay_exact": replay["selector_replay_exact"]}, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    return run(args.old_root.resolve(), args.output_root.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
