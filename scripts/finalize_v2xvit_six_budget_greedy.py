#!/usr/bin/env python3
"""Finalize six-budget exact-Greedy validation and strict GA admission gates."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


LABELS = ("030", "025", "020", "015", "010", "005")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def collapse(drop: float, retention: float) -> str:
    if retention < 0.50:
        return "CATASTROPHIC"
    if drop > 0.10 or retention < 0.80:
        return "SEVERE_COLLAPSE"
    if drop > 0.03:
        return "SIGNIFICANT_DROP"
    if drop > 0.01:
        return "MILD_DROP"
    return "SAFE"


def run(root: Path) -> None:
    winners = json.loads((root / "reports/six_budget_greedy_winners.json").read_text())
    fixed = json.loads((root / "reports/six_budget_fixed500.json").read_text())
    latency = json.loads((root / "reports/six_budget_latency.json").read_text())
    convergence = json.loads((root / "reports/taylor_sample_convergence.json").read_text())
    controls = fixed["controls"]
    b0 = controls["B0"]
    b0_map = float(b0["mAP"])
    fixed_rows: list[dict[str, Any]] = []
    latency_rows: list[dict[str, Any]] = []
    compression_rows: list[dict[str, Any]] = []
    collapse_rows: dict[str, Any] = {}
    admission_rows: dict[str, Any] = {}
    admissible: list[float] = []
    calibration_rows: dict[str, Any] = {}
    for label in LABELS:
        budget = int(label) / 100.0
        winner = winners[f"{budget:.2f}"]
        s32 = controls[f"budget_{label}/S32"]
        jmix = controls[f"budget_{label}/JMIX-FRESH"]
        s32_map = float(s32["mAP"])
        jmix_map = float(jmix["mAP"])
        s32_drop = b0_map - s32_map
        jmix_drop = b0_map - jmix_map
        s32_ret = s32_map / b0_map
        jmix_ret = jmix_map / b0_map
        latency_row = latency["budgets"][label]
        latency_valid = latency_row.get("status") == "ok"
        speedup = (
            float(latency_row["controls"]["JMIX-FRESH"]["speedup_p50_vs_B0"])
            if latency_valid else None
        )
        acceptance = json.loads((root / (
            f"engines/greedy_exact_winners/budget_{label}/JMIX-FRESH/"
            "engine_build_acceptance.json"
        )).read_text())
        precision = dict(acceptance.get("precision_realization_validation") or {})
        exact = bool(
            acceptance.get("status") == "ok" and precision.get("passed")
            and not precision.get("mismatches")
            and int(precision.get("unresolved_layer_count", 0)) == 0
        )
        precision_counts = {
            state: sum(value == state for value in winner["genotype"]["precision_genes"].values())
            for state in ("FP32", "FP16", "INT8")
        }
        has_int8 = precision_counts["INT8"] > 0
        calibration_path = root / (
            f"engines/greedy_exact_winners/budget_{label}/JMIX-FRESH/calibration_manifest.json"
        )
        calibration = json.loads(calibration_path.read_text())
        calibration_ok = bool(
            (not has_int8 and int(calibration.get("requested_frames", 0)) == 0)
            or (
                has_int8
                and int(calibration.get("requested_frames", -1)) == 200
                and int(calibration.get("processed_frames", -1)) == 200
                and int(calibration.get("skipped_frames", -1)) == 0
                and all(calibration.get(key) for key in (
                    "checkpoint_hash", "physical_hash", "state_dict_shape_hash",
                    "precision_map_hash", "onnx_hash", "manifest_hash",
                    "calibration_algorithm_config_hash", "scale_hash", "cache_hash",
                ))
            )
        )
        calibration_rows[label] = {
            "activation_calibration_required": has_int8,
            "train200_verified": calibration_ok if has_int8 else None,
            "manifest": calibration,
        }
        build_ok = acceptance.get("status") == "ok"
        reached = bool(winner.get("budget_reached"))
        ga_ok = bool(
            reached and s32_drop <= 0.01 and jmix_drop <= 0.01
            and build_ok and exact and latency_valid and speedup is not None and speedup > 1.0
            and convergence.get("passed") and calibration_ok
        )
        if ga_ok:
            admission = "GA_ADMISSIBLE"
            admissible.append(budget)
        elif not build_ok or not exact or not calibration_ok:
            admission = "GA_DEPLOYMENT_INVALID"
        elif not convergence.get("passed"):
            admission = "GA_PROXY_UNRELIABLE"
        else:
            admission = "GA_UNSAFE"
        fixed_rows.extend([
            {"budget": budget, "control": "B0", "AP30": b0["AP@0.3"],
             "AP50": b0["AP@0.5"], "AP70": b0["AP@0.7"], "mAP": b0_map,
             "evaluated": b0["num_evaluated_frames"], "skipped": b0["num_skipped_frames"]},
            {"budget": budget, "control": "S32", "AP30": s32["AP@0.3"],
             "AP50": s32["AP@0.5"], "AP70": s32["AP@0.7"], "mAP": s32_map,
             "drop": s32_drop, "retention": s32_ret,
             "evaluated": s32["num_evaluated_frames"], "skipped": s32["num_skipped_frames"]},
            {"budget": budget, "control": "JMIX-FRESH", "AP30": jmix["AP@0.3"],
             "AP50": jmix["AP@0.5"], "AP70": jmix["AP@0.7"], "mAP": jmix_map,
             "drop": jmix_drop, "retention": jmix_ret,
             "evaluated": jmix["num_evaluated_frames"], "skipped": jmix["num_skipped_frames"]},
        ])
        if latency_valid:
            for control in ("B0", "S32", "JMIX-FRESH"):
                item = latency_row["controls"][control]
                latency_rows.append({
                    "budget": budget, "control": control,
                    "p50_ms": item["forward_p50_ms"], "fps": item["fps"],
                    "speedup": item.get("speedup_p50_vs_B0", 1.0),
                    "baseline_replay_valid": True,
                })
        size = winner["size"]
        bops = winner["bops"]
        compression_rows.append({
            "budget": budget, "R_BOPS": bops["R_bops_vs_fp32"],
            "BOPS_compression": 1.0 / float(bops["R_bops_vs_fp32"]),
            "parameter_retention": size["R_parameter_retention"],
            "parameter_compression": 1.0 / float(size["R_parameter_retention"]),
            "mixed_weight_retention": size["R_size_vs_fp32"],
            "mixed_weight_compression": 1.0 / float(size["R_size_vs_fp32"]),
        })
        collapse_rows[label] = {
            "budget": budget, "S32_drop": s32_drop, "S32_retention": s32_ret,
            "S32_class": collapse(s32_drop, s32_ret),
            "JMIX_drop": jmix_drop, "JMIX_retention": jmix_ret,
            "JMIX_class": collapse(jmix_drop, jmix_ret),
        }
        admission_rows[label] = {
            "budget": budget, "status": admission, "budget_reached": reached,
            "S32_drop_gate": s32_drop <= 0.01, "JMIX_drop_gate": jmix_drop <= 0.01,
            "engine_build_success": build_ok, "requested_realized_exact": exact,
            "precision_fallback_count": len(precision.get("mismatches") or []),
            "latency_batch_valid": latency_valid, "JMIX_speedup": speedup,
            "taylor_convergence_pass": bool(convergence.get("passed")),
            "train200_contract_pass": calibration_ok,
        }
    write_csv(root / "reports/six_budget_fixed500.csv", fixed_rows)
    write_csv(root / "reports/six_budget_latency.csv", latency_rows)
    write_csv(root / "reports/six_budget_compression.csv", compression_rows)
    write_json(root / "reports/six_budget_ap_collapse.json", collapse_rows)
    write_json(root / "reports/greedy_train200_audit.json", calibration_rows)
    write_json(root / "reports/ga_budget_admission.json", {
        "ga_admissible_budgets": admissible,
        "budget_results": admission_rows,
        "formal_ga_may_run_only_for_listed_budgets": True,
        "full1789_allowed": False,
    })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    run(args.output_root.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
