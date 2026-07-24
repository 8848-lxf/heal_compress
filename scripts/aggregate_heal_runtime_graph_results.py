#!/usr/bin/env python3
"""Aggregate final F-Cooper/Disco Greedy and GA runtime-graph results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "docs/codex_handoffs"
    / "H800_feature-heal-unified-search-h800_fcooper_disco_full_results.csv"
)
EXPECTED_BUDGETS = {0.05, 0.10, 0.15, 0.20, 0.25, 0.30}

RUNS = (
    (
        "F-Cooper",
        "Greedy",
        REPO_ROOT
        / "outputs/h800_heal_lidar_fcooper_runtime_graph_joint_greedy_20260722_135143",
    ),
    (
        "F-Cooper",
        "GA",
        REPO_ROOT
        / "outputs/h800_heal_lidar_fcooper_runtime_graph_joint_ga_20260723_043415",
    ),
    (
        "Disco",
        "Greedy",
        REPO_ROOT
        / "outputs/h800_heal_lidar_disco_runtime_graph_joint_greedy_20260723_035226",
    ),
    (
        "Disco",
        "GA",
        REPO_ROOT
        / "outputs/h800_heal_lidar_disco_runtime_graph_joint_ga_20260723_075044",
    ),
)

FIELDS = (
    "model",
    "search_method",
    "bops_budget_target",
    "actual_bops_retention",
    "bops_compression_x",
    "fp32_parameter_count",
    "candidate_parameter_count",
    "parameter_pruning_rate",
    "parameter_compression_x",
    "mixed_weight_retention_vs_fp32",
    "mixed_weight_compression_x",
    "fp32_AP30",
    "fp32_AP50",
    "fp32_AP70",
    "fp32_mAP",
    "candidate_AP30",
    "candidate_AP50",
    "candidate_AP70",
    "candidate_mAP",
    "delta_AP30",
    "delta_AP50",
    "delta_AP70",
    "delta_mAP",
    "fp32_forward_mean_ms",
    "candidate_forward_mean_ms",
    "speedup_mean_x",
    "fp32_forward_p50_ms",
    "candidate_forward_p50_ms",
    "speedup_p50_x",
    "fp32_forward_p90_ms",
    "candidate_forward_p90_ms",
    "speedup_p90_x",
    "evaluated_frames",
    "skipped_frames",
    "candidate_hash",
    "source_run",
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def require_number(mapping: dict[str, Any], key: str) -> float:
    value = mapping.get(key)
    if not isinstance(value, (int, float)):
        raise ValueError(f"missing_numeric_field:{key}")
    return float(value)


def validate_evaluation(row: dict[str, Any], label: str) -> None:
    if row.get("status") != "ok":
        raise ValueError(f"non_ok_evaluation:{label}:{row.get('status')}")
    if row.get("evaluation_acceptance") is False:
        raise ValueError(f"evaluation_rejected:{label}")
    if int(row.get("num_evaluated_frames", -1)) != 1789:
        raise ValueError(f"wrong_frame_count:{label}")
    if int(row.get("num_skipped_frames", -1)) != 0:
        raise ValueError(f"skipped_frames:{label}")


def deployment_checks_pass(row: dict[str, Any]) -> bool:
    checks = (
        "evaluation_acceptance",
        "physical_acceptance",
        "qdq_acceptance",
        "engine_acceptance",
        "precision_acceptance",
        "merge_acceptance",
    )
    return row.get("status") == "ok" and all(row.get(key) is True for key in checks)


def greedy_candidates(run_dir: Path) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    payload = read_json(run_dir / "greedy/full_validation_results.json")
    rows = []
    for candidate in payload["candidates"]:
        if not deployment_checks_pass(candidate):
            raise ValueError(f"greedy_deployment_rejected:{candidate.get('candidate_hash')}")
        rows.append((candidate, candidate))
    return rows


def ga_candidates(run_dir: Path) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    payload = read_json(run_dir / "final_full_validation_results.json")
    if payload.get("successful_budget_winner_count") != 6:
        raise ValueError(f"incomplete_ga_budget_winners:{run_dir}")
    if payload.get("missing_budget_rounds"):
        raise ValueError(f"missing_ga_budget_rounds:{run_dir}")
    rows = []
    for candidate in read_json(run_dir / "final_budget_winners.json"):
        generation_dir = Path(candidate["source_artifact_dir"]).parents[1]
        generation_winner = read_json(generation_dir / "generation_winner.json")
        if generation_winner.get("candidate_hash") != candidate.get("candidate_hash"):
            raise ValueError(f"ga_winner_hash_mismatch:{generation_dir}")
        if not deployment_checks_pass(generation_winner):
            raise ValueError(f"ga_deployment_rejected:{candidate.get('candidate_hash')}")
        rows.append((candidate, generation_winner))
    return rows


def build_row(
    model: str,
    method: str,
    run_dir: Path,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    source: dict[str, Any],
) -> dict[str, Any]:
    validate_evaluation(candidate, f"{model}:{method}:{candidate.get('candidate_hash')}")
    if candidate.get("eval_manifest_hash") != baseline.get("eval_manifest_hash"):
        raise ValueError(f"eval_manifest_mismatch:{model}:{method}")

    if method == "Greedy":
        budget = float(candidate["budgets"][0])
        bops_compression = require_number(candidate, "BOPS_compression_x")
        actual_bops = 1.0 / bops_compression
        parameter_base = require_number(candidate, "physical_parameter_count_before")
        parameter_after = require_number(candidate, "physical_parameter_count_after")
        parameter_pruning = require_number(candidate, "physical_parameter_pruning_ratio")
        mixed_compression = require_number(candidate, "mixed_weight_compression_x")
    else:
        stage1 = source["stage1_metrics"]
        budget = require_number(stage1, "BOPS_target")
        actual_bops = require_number(stage1, "R_bops_vs_fp32")
        bops_compression = 1.0 / actual_bops
        parameter_base = require_number(source, "physical_parameter_count_before")
        parameter_after = require_number(source, "physical_parameter_count_after")
        parameter_pruning = require_number(source, "physical_parameter_pruning_ratio")
        mixed_compression = 1.0 / require_number(stage1, "R_size_vs_fp32")

    result: dict[str, Any] = {
        "model": model,
        "search_method": method,
        "bops_budget_target": budget,
        "actual_bops_retention": actual_bops,
        "bops_compression_x": bops_compression,
        "fp32_parameter_count": int(parameter_base),
        "candidate_parameter_count": int(parameter_after),
        "parameter_pruning_rate": parameter_pruning,
        "parameter_compression_x": parameter_base / parameter_after,
        "mixed_weight_retention_vs_fp32": 1.0 / mixed_compression,
        "mixed_weight_compression_x": mixed_compression,
        "evaluated_frames": int(candidate["num_evaluated_frames"]),
        "skipped_frames": int(candidate["num_skipped_frames"]),
        "candidate_hash": candidate["candidate_hash"],
        "source_run": str(run_dir.relative_to(REPO_ROOT)),
    }
    for source_key, label in (
        ("AP@0.3", "AP30"),
        ("AP@0.5", "AP50"),
        ("AP@0.7", "AP70"),
        ("mAP", "mAP"),
    ):
        baseline_value = require_number(baseline, source_key)
        candidate_value = require_number(candidate, source_key)
        result[f"fp32_{label}"] = baseline_value
        result[f"candidate_{label}"] = candidate_value
        result[f"delta_{label}"] = candidate_value - baseline_value
    for statistic in ("mean", "p50", "p90"):
        key = f"forward_{statistic}_ms"
        baseline_value = require_number(baseline, key)
        candidate_value = require_number(candidate, key)
        result[f"fp32_{key}"] = baseline_value
        result[f"candidate_{key}"] = candidate_value
        result[f"speedup_{statistic}_x"] = baseline_value / candidate_value
    return result


def aggregate() -> list[dict[str, Any]]:
    all_rows = []
    for model, method, run_dir in RUNS:
        baseline = read_json(run_dir / "full_validation/reference/evaluation.json")
        validate_evaluation(baseline, f"{model}:{method}:FP32")
        candidate_rows = (
            greedy_candidates(run_dir) if method == "Greedy" else ga_candidates(run_dir)
        )
        rows = [
            build_row(model, method, run_dir, baseline, candidate, source)
            for candidate, source in candidate_rows
        ]
        budgets = {round(float(row["bops_budget_target"]), 2) for row in rows}
        if budgets != EXPECTED_BUDGETS or len(rows) != 6:
            raise ValueError(f"budget_coverage_error:{model}:{method}:{sorted(budgets)}")
        all_rows.extend(rows)
    return sorted(
        all_rows,
        key=lambda row: (
            0 if row["model"] == "F-Cooper" else 1,
            0 if row["search_method"] == "Greedy" else 1,
            row["bops_budget_target"],
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    rows = aggregate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
