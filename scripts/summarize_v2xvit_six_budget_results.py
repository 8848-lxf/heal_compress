#!/usr/bin/env python3
"""Create an auditable partial/final fixed500 summary for six V2X-ViT budgets."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.watch_and_run_v2xvit_six_budget_latency import LABELS, completion_state


AP_KEYS = ("AP@0.3", "AP@0.5", "AP@0.7", "mAP")


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_summary(root: Path) -> dict[str, Any]:
    baseline_path = root / "baseline_fixed500/B0/evaluation.json"
    baseline = _load(baseline_path) if baseline_path.is_file() else None
    readiness = completion_state(root)
    result: dict[str, Any] = {
        "status": "complete" if readiness["ready"] else "partial",
        "fixed500_manifest_hash": baseline.get("eval_manifest_hash") if baseline else None,
        "baseline": {key: baseline.get(key) for key in AP_KEYS} if baseline else None,
        "baseline_evaluated": baseline.get("num_evaluated_frames") if baseline else None,
        "baseline_skipped": baseline.get("num_skipped_frames") if baseline else None,
        "budgets": {},
        "readiness": readiness,
    }
    latency_path = root / "reports/v2xvit_six_budget_final_latency.json"
    latency = _load(latency_path).get("budgets", {}) if latency_path.is_file() else {}
    greedy_resources: dict[str, dict[str, str]] = {}
    resource_path = root / "reports/six_budget_greedy_summary.csv"
    if resource_path.is_file():
        with resource_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                greedy_resources[f"{round(float(row['budget']) * 100):03d}"] = row
    for label in LABELS:
        summary_path = root / f"ga/budget_{label}/seed_0/budget_summary.json"
        if not summary_path.is_file():
            generations = sorted((root / f"ga/budget_{label}/seed_0").glob(
                "generation_*/generation_summary.json"
            ))
            result["budgets"][label] = {
                "status": "running" if generations else "pending",
                "latest_completed_generation": (
                    int(generations[-1].parent.name.rsplit("_", 1)[1]) if generations else None
                ),
                "greedy_resources": greedy_resources.get(label),
            }
            continue
        formal = _load(summary_path)
        row: dict[str, Any] = {
            "status": "formal_ga_complete",
            "target_bops_retention": formal["budget"],
            "completed_evolution_generations": formal["completed_evolution_generations"],
            "stage2_real_evaluation_count": formal["stage2_real_evaluation_count"],
            "ga_improved_greedy_at_fixed50": formal["ga_improved_greedy"],
            "greedy_resources": greedy_resources.get(label),
            "controls": {},
        }
        for control, key in (("greedy", "greedy_anchor"), ("ga", "final_winner")):
            candidate = formal[key]
            candidate_hash = candidate["complete_phenotype_hash"]
            evaluation_path = (
                root / f"{control}_final_fixed500_partial/budget_{label}/"
                f"{candidate_hash}/evaluation.json"
            )
            metric = _load(evaluation_path) if evaluation_path.is_file() else None
            control_row: dict[str, Any] = {
                "candidate_hash": candidate_hash,
                "fixed50_mAP": candidate["mAP"],
                "fixed50_screening_p50_ms": candidate["p50_ms"],
                "requested_realized_exact": candidate["requested_realized_exact"],
                "fixed500_status": "complete" if metric else "pending",
            }
            if metric:
                control_row.update({key: metric[key] for key in AP_KEYS})
                control_row.update({
                    "evaluated": metric["num_evaluated_frames"],
                    "skipped": metric["num_skipped_frames"],
                    "manifest_hash": metric["eval_manifest_hash"],
                })
                if baseline:
                    control_row["mAP_drop_vs_B0"] = baseline["mAP"] - metric["mAP"]
                    control_row["mAP_retention_vs_B0"] = metric["mAP"] / baseline["mAP"]
            if label in latency and latency[label].get("status") == "ok":
                latency_control = "Greedy" if control == "greedy" else "GA-final"
                control_row["formal_latency"] = latency[label]["controls"][latency_control]
            row["controls"][control] = control_row
        if all(item["fixed500_status"] == "complete" for item in row["controls"].values()):
            row["status"] = "fixed500_complete"
            row["ga_minus_greedy_fixed500_mAP"] = (
                row["controls"]["ga"]["mAP"] - row["controls"]["greedy"]["mAP"]
            )
        result["budgets"][label] = row
    return result


def write_summary(root: Path) -> dict[str, Any]:
    result = build_summary(root)
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    json_path = reports / "v2xvit_six_budget_final_fixed500_summary.json"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rows = []
    baseline = result.get("baseline") or {}
    rows.append({"budget": "B0", "control": "B0", **baseline,
                 "evaluated": result.get("baseline_evaluated"),
                 "skipped": result.get("baseline_skipped")})
    for label, budget in result["budgets"].items():
        for control, metric in budget.get("controls", {}).items():
            rows.append({
                "budget": float(label) / 100.0, "control": control,
                "candidate_hash": metric["candidate_hash"],
                **{key: metric.get(key) for key in AP_KEYS},
                "mAP_drop_vs_B0": metric.get("mAP_drop_vs_B0"),
                "mAP_retention_vs_B0": metric.get("mAP_retention_vs_B0"),
                "evaluated": metric.get("evaluated"), "skipped": metric.get("skipped"),
            })
    if rows:
        fields = sorted({key for row in rows for key in row})
        with (reports / "v2xvit_six_budget_final_fixed500_summary.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    result = write_summary(args.output_root.resolve())
    return 0 if result["status"] == "complete" or not args.require_complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
