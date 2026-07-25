#!/usr/bin/env python3
"""Audit a regenerated V2X-ViT 0.05 Stage-1 budget band and selectors."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


TARGET = 0.05
TOLERANCE = 0.005


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f"refusing_to_overwrite:{path}")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f"refusing_to_overwrite:{path}")
    fields = list(rows[0]) if rows else ["candidate_hash"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _float(row: dict[str, str], key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise RuntimeError(f"old005_nonfinite:{key}:{value}")
    return value


def _selector_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: row[key]
        for key in (
            "candidate_hash",
            "step",
            "R_BOPS",
            "bops_deviation",
            "cumulative_total_taylor",
            "parameter_retention",
            "parameter_count",
            "mixed_weight_size_bytes",
            "precision_counts",
            "shrinker_width",
            "attention_widths",
            "ffn_widths",
        )
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    source = args.regenerated_root.resolve()
    trace_path = source / "greedy_trace.csv"
    winner_path = source / "winner/v2xvit_greedy005_winner.json"
    if not trace_path.is_file() or not winner_path.is_file():
        raise RuntimeError(f"old005_regenerated_artifacts_missing:{source}")
    winner_payload = json.loads(winner_path.read_text(encoding="utf-8"))
    base_count = float(winner_payload["size"]["parameter_count_base"])
    base_fp32_bytes = base_count * 4.0
    by_hash: dict[str, dict[str, Any]] = {}
    with trace_path.open(encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            retention = _float(raw, "current_retention")
            if abs(retention - TARGET) > TOLERANCE + 1.0e-12:
                continue
            state = json.loads(raw["state_after"])
            widths = {str(key): int(value) for key, value in state["widths"].items()}
            precision = {str(key): str(value) for key, value in state["precision"].items()}
            parameter_retention = _float(raw, "R_parameter_retention")
            mixed_size = _float(raw, "mixed_weight_size_bytes")
            precision_counts = {
                value: sum(item == value for item in precision.values())
                for value in ("FP32", "FP16", "INT8")
            }
            attention = {
                key: value for key, value in widths.items() if key.startswith("attention_dh::")
            }
            ffn = {key: value for key, value in widths.items() if key.startswith("ffn_hidden::")}
            shrinker = {
                key: value for key, value in widths.items() if "shrink" in key.lower()
            }
            row = {
                "candidate_hash": raw["candidate_hash"],
                "step": int(raw["step"]),
                "R_BOPS": retention,
                "bops_deviation": retention - TARGET,
                "cumulative_total_taylor": _float(raw, "cumulative_proxy"),
                "cumulative_structural_taylor": _float(raw, "cumulative_pruning_taylor"),
                "cumulative_weight_quant_taylor": _float(
                    raw, "cumulative_weight_quantization_taylor"
                ),
                "parameter_count": int(round(base_count * parameter_retention)),
                "parameter_retention": parameter_retention,
                "parameter_prune_ratio": 1.0 - parameter_retention,
                "mixed_weight_size_bytes": mixed_size,
                "mixed_weight_compression_ratio": base_fp32_bytes / mixed_size,
                "precision_counts": precision_counts,
                "shrinker_width": shrinker,
                "attention_widths": attention,
                "ffn_widths": ffn,
                "structure_state": widths,
                "precision_map": precision,
            }
            incumbent = by_hash.get(row["candidate_hash"])
            if incumbent is None or (
                row["cumulative_total_taylor"],
                abs(row["bops_deviation"]),
            ) < (
                incumbent["cumulative_total_taylor"],
                abs(incumbent["bops_deviation"]),
            ):
                by_hash[row["candidate_hash"]] = row
    rows = list(by_hash.values())
    if not rows:
        raise RuntimeError("old005_budget_band_empty")
    taylor = np.asarray([row["cumulative_total_taylor"] for row in rows], dtype=np.float64)
    ordered_taylor = sorted(rows, key=lambda row: (row["cumulative_total_taylor"], row["candidate_hash"]))
    minimum = float(taylor.min())
    equivalence = [
        row
        for row in rows
        if abs(float(row["cumulative_total_taylor"]) - minimum)
        / max(abs(minimum), 1.0e-30)
        < 1.0e-8
    ]
    selector_a = min(
        rows,
        key=lambda row: (
            row["cumulative_total_taylor"],
            abs(row["bops_deviation"]),
            row["mixed_weight_size_bytes"],
            row["candidate_hash"],
        ),
    )
    selector_b = ordered_taylor[0]
    selector_c = min(
        rows,
        key=lambda row: (
            row["cumulative_total_taylor"],
            abs(row["bops_deviation"]),
            row["candidate_hash"],
        ),
    )
    selector_d = min(
        equivalence,
        key=lambda row: (abs(row["bops_deviation"]), row["candidate_hash"]),
    )
    actual_hash = str(winner_payload["candidate_hash"])
    winner = by_hash.get(actual_hash)
    if winner is None:
        raise RuntimeError(f"old005_winner_not_in_regenerated_band:{actual_hash}")
    parameter_rank = 1 + sum(
        row["parameter_retention"] > winner["parameter_retention"] for row in rows
    )
    mixed_rank = 1 + sum(
        row["mixed_weight_size_bytes"] < winner["mixed_weight_size_bytes"] for row in rows
    )
    pure_taylor_rank = 1 + sum(
        row["cumulative_total_taylor"] < winner["cumulative_total_taylor"] for row in rows
    )
    second = ordered_taylor[1]["cumulative_total_taylor"] if len(rows) > 1 else minimum
    distribution = {
        "candidate_count": len(rows),
        "taylor_min": minimum,
        "taylor_second_best": second,
        "taylor_p1": float(np.percentile(taylor, 1)),
        "taylor_p5": float(np.percentile(taylor, 5)),
        "taylor_p50": float(np.percentile(taylor, 50)),
        "taylor_p95": float(np.percentile(taylor, 95)),
        "taylor_max": float(taylor.max()),
        "top2_relative_difference": float((second - minimum) / max(abs(minimum), 1.0e-30)),
        "within_best_0_1_percent": int(np.sum(taylor <= minimum * 1.001)),
        "within_best_0_5_percent": int(np.sum(taylor <= minimum * 1.005)),
        "within_best_1_percent": int(np.sum(taylor <= minimum * 1.01)),
        "winner_pure_taylor_rank": pure_taylor_rank,
        "winner_parameter_retention_rank_descending": parameter_rank,
        "winner_mixed_size_rank_ascending": mixed_rank,
        "taylor_equivalence_band_relative": 1.0e-8,
        "taylor_equivalence_band_count": len(equivalence),
    }
    replay = {
        "actual_regenerated_winner": _selector_payload(winner),
        "A_original_selector": _selector_payload(selector_a),
        "B_pure_taylor_minimum": _selector_payload(selector_b),
        "C_taylor_then_bops_deviation": _selector_payload(selector_c),
        "D_equivalence_band_without_parameter_or_mixed_size": _selector_payload(selector_d),
        "actual_matches_original_replay": actual_hash == selector_a["candidate_hash"],
        "parameter_prune_used_by_actual_selector": False,
        "mixed_weight_size_used_by_actual_selector": True,
        "parameter_prune_changed_winner": False,
        "mixed_size_changed_winner": selector_a["candidate_hash"] != selector_c["candidate_hash"],
        "trajectory_or_final_tiebreak": (
            "final_tiebreak_changed_winner"
            if selector_a["candidate_hash"] != selector_c["candidate_hash"]
            else "greedy_action_trajectory_dominated"
        ),
    }
    csv_rows = []
    for row in sorted(rows, key=lambda value: (value["cumulative_total_taylor"], value["candidate_hash"])):
        csv_rows.append(
            {
                **{key: value for key, value in row.items() if not isinstance(value, dict)},
                "precision_counts": json.dumps(row["precision_counts"], sort_keys=True),
                "shrinker_width": json.dumps(row["shrinker_width"], sort_keys=True),
                "attention_widths": json.dumps(row["attention_widths"], sort_keys=True),
                "ffn_widths": json.dumps(row["ffn_widths"], sort_keys=True),
                "structure_state": json.dumps(row["structure_state"], sort_keys=True),
                "precision_map": json.dumps(row["precision_map"], sort_keys=True),
            }
        )
    _write_csv(args.output_dir / "budget_candidate_distribution.csv", csv_rows)
    _write_json(args.output_dir / "taylor_distribution.json", distribution)
    _write_json(args.output_dir / "winner_selector_replay.json", replay)
    markdown = "\n".join(
        [
            "# Regenerated V2X-ViT R_BOPS=0.05 selector audit",
            "",
            f"- Budget-band candidates: {len(rows)}",
            f"- Taylor min / p50 / max: {minimum:.12g} / {distribution['taylor_p50']:.12g} / {distribution['taylor_max']:.12g}",
            f"- Top-2 relative difference: {distribution['top2_relative_difference']:.12g}",
            f"- Candidates within 0.1% / 0.5% / 1%: {distribution['within_best_0_1_percent']} / {distribution['within_best_0_5_percent']} / {distribution['within_best_1_percent']}",
            f"- Actual selector uses parameter pruning as a tie-break: false.",
            f"- Actual selector uses mixed weight size after Taylor and BOPS deviation: true.",
            f"- Final classification: `{replay['trajectory_or_final_tiebreak']}`.",
            "",
        ]
    )
    md_path = args.output_dir / "winner_selector_replay.md"
    if md_path.exists():
        raise RuntimeError(f"refusing_to_overwrite:{md_path}")
    md_path.write_text(markdown, encoding="utf-8")
    return {"distribution": distribution, "replay": replay}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regenerated-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
