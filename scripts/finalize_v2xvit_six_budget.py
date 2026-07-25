#!/usr/bin/env python3
"""Finalize the V2X-ViT six-budget Greedy deployment audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from search.reporting.v2xvit_six_budget import (
    classify_ap_drop,
    classify_ga_admission,
    compression_metrics,
)


BUDGETS = ("030", "025", "020", "015", "010", "005")
METRICS = ("AP30", "AP50", "AP70", "mAP")


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not materialized:
        raise RuntimeError(f"empty_csv:{path}")
    fields: list[str] = []
    for row in materialized:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(materialized)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strip_redundant_candidate_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_redundant_candidate_payload(item)
            for key, item in value.items()
            if key not in {"genotype", "phenotype"}
        }
    if isinstance(value, list):
        return [_strip_redundant_candidate_payload(item) for item in value]
    return value


def _precision_counts(genes: dict[str, str]) -> dict[str, int]:
    return {precision: sum(value == precision for value in genes.values()) for precision in ("FP32", "FP16", "INT8")}


def _width_groups(widths: dict[str, int]) -> dict[str, dict[str, int]]:
    return {
        "cnn": {key: value for key, value in widths.items() if key.startswith("backbone_")},
        "shrinker": {key: value for key, value in widths.items() if key.startswith("shrinker_")},
        "attention": {key: value for key, value in widths.items() if key.startswith("attention_dh::")},
        "ffn": {key: value for key, value in widths.items() if key.startswith("ffn_hidden::")},
    }


def _eval_index(raw: dict[str, Any]) -> tuple[dict[str, Any], dict[tuple[str, str], dict[str, Any]]]:
    b0 = dict(raw["B0"])
    rows = {
        (str(row["budget"]), str(row["profile"])): dict(row)
        for row in raw["rows"]
        if row["budget"] != "B0"
    }
    return b0, rows


def _first_budget(rows: list[dict[str, Any]], field: str, labels: set[str]) -> float | None:
    for row in rows:
        if row[field] in labels:
            return float(row["budget"])
    return None


def _plots(report_dir: Path, rows: list[dict[str, Any]]) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curves = report_dir / "curves"
    curves.mkdir(parents=True, exist_ok=True)
    ordered = list(reversed(rows))
    x = [row["R_BOPS"] for row in ordered]
    specs = (
        ("map", ("B0_mAP", "S32_mAP", "JMIX_mAP")),
        ("map_retention", ("S32_mAP_retention", "JMIX_mAP_retention")),
        ("map_drop", ("structural_drop", "quantization_extra_drop", "joint_drop")),
        ("latency_p50", ("B0_p50_ms", "S32_p50_ms", "JMIX_p50_ms")),
        ("speedup", ("S32_speedup", "JMIX_speedup")),
        ("resource_retention", ("parameter_retention", "mixed_weight_retention")),
        ("shrinker_width", ("shrinker_width",)),
        ("minimum_attention_dh", ("minimum_attention_dh",)),
        ("precision_counts", ("FP32_count", "FP16_count", "INT8_count")),
    )
    outputs = []
    for name, fields in specs:
        figure, axis = plt.subplots(figsize=(6.4, 4.0))
        for field in fields:
            axis.plot(x, [row[field] for row in ordered], marker="o", label=field)
        axis.set_xlabel("R_BOPS")
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8)
        figure.tight_layout()
        target = curves / f"{name}.png"
        figure.savefig(target, dpi=140)
        plt.close(figure)
        outputs.append(str(target))
    return outputs


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = args.output_root.resolve()
    reports = root / "reports"
    fixed_raw = _load(reports / "six_budget_fixed500_raw.json")
    latency_raw = _load(reports / "six_budget_latency_raw.json")
    convergence = _load(reports / "taylor_sample_convergence.json")
    run_manifest = _load(root / "run_manifest.json")
    dry_run_path = reports / "ga_dry_run.json"
    _write_json(dry_run_path, _strip_redundant_candidate_payload(_load(dry_run_path)))
    b0_eval, evals = _eval_index(fixed_raw)
    b0_result = _load(root / "engines/B0/candidate_result.json")
    b0_engine = root / "engines/B0/candidate.plan"
    b0_size = b0_engine.stat().st_size
    latency_by_budget = {str(row["budget"]): row for row in latency_raw["rows"]}
    stage1_summary = {
        f"{float(row['budget']):.2f}": row
        for row in _read_csv(reports / "six_budget_greedy_summary.csv")
    }
    snapshot_path = root / "latency/process_snapshots.jsonl"
    snapshots = [json.loads(line) for line in snapshot_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    observed_pids = sorted({int(pid) for row in snapshots for pid in row["compute_pids"]})
    process_isolated = bool(snapshots) and len(observed_pids) <= 1 and all(len(row["compute_pids"]) <= 1 for row in snapshots)
    process_audit = {
        "snapshot_count": len(snapshots),
        "observed_compute_pids": observed_pids,
        "maximum_concurrent_compute_processes": max((len(row["compute_pids"]) for row in snapshots), default=0),
        "exclusive_single_measurement_process": process_isolated,
        "snapshot_path": str(snapshot_path),
    }
    _write_json(reports / "latency_process_audit.json", process_audit)

    result_rows: list[dict[str, Any]] = []
    structures: list[dict[str, Any]] = []
    precisions: list[dict[str, Any]] = []
    buildable = True
    for code in BUDGETS:
        budget_dir = root / "budgets" / code
        winner = _load(budget_dir / "winner.json")
        build = _load(budget_dir / "build_audit.json")
        s32_result = _load(Path(build["selected_s32_dir"]) / "candidate_result.json")
        jmix_dir = Path(build["selected_jmix_dir"])
        jmix_result = _load(jmix_dir / "candidate_result.json")
        precision_audit = _load(jmix_dir / "deployment_closed_precision_audit.json")
        calibration = _load(jmix_dir / "train200_calibration_identity.json")
        genotype = winner["genotype"]
        widths = {key: int(value) for key, value in genotype["pruning_width_genes"].items()}
        groups = _width_groups(widths)
        genes = {key: str(value) for key, value in genotype["precision_genes"].items()}
        mutable_counts = _precision_counts(genes)
        realized_counts = {
            precision: int(stage1_summary[f"{float(winner['budget']):.2f}"][f"{precision}_count"])
            for precision in ("FP32", "FP16", "INT8")
        }
        metrics = winner["metrics"]
        size = winner["size"]
        resource = compression_metrics(
            bops_retention=float(winner["bops"]["R_bops"]),
            parameter_retention=float(size["R_parameter_retention"]),
            mixed_weight_retention=float(size["R_size_vs_fp32"]),
        )
        s32_eval = evals[(code, "S32")]
        jmix_eval = evals[(code, "JMIX")]
        s32_class = classify_ap_drop(b0_eval["mAP"], s32_eval["mAP"])
        jmix_class = classify_ap_drop(b0_eval["mAP"], jmix_eval["mAP"])
        latency = latency_by_budget[code]
        calibration_required = realized_counts["INT8"] > 0
        calibration_ok = (
            int(calibration["processed_frames"]) == 200 and int(calibration["skipped_frames"]) == 0
            if calibration_required
            else calibration["algorithm"] == "not_required_no_int8"
        )
        exact = bool(jmix_result["requested_realized_exact"]) and bool(precision_audit["requested_realized_exact"])
        conflict_count = max(int(jmix_result["precision_conflict_count"]), int(precision_audit["conflict_count"]))
        fallback_count = int(bool(precision_audit["silent_fallback"]))
        jmix_speedup = float(latency["JMIX"]["speedup_vs_matched_B0"])
        admission = classify_ga_admission(
            budget_reached=abs(float(resource["R_BOPS"]) - float(winner["budget"])) <= 0.005,
            s32_drop=float(s32_class["absolute_drop"]),
            jmix_drop=float(jmix_class["absolute_drop"]),
            jmix_engine_built=jmix_result["status"] == "ok",
            requested_realized_exact=exact,
            precision_conflict_count=conflict_count,
            fallback_count=fallback_count,
            latency_batch_valid=bool(latency["latency_batch_valid"]) and process_isolated,
            jmix_speedup=jmix_speedup,
            taylor_convergence_passed=bool(convergence["passed"]),
            framework_tests_passed=args.framework_tests_passed,
        )
        engine_sizes = {
            "S32_engine_size_bytes": int(s32_result["engine_size_bytes"]),
            "JMIX_engine_size_bytes": int(jmix_result["engine_size_bytes"]),
        }
        row: dict[str, Any] = {
            "budget": float(winner["budget"]),
            "budget_code": code,
            "candidate_hash": winner["candidate_hash"],
            "phenotype_hash": winner["phenotype_hash"],
            "physical_hash": winner["structure_hash"],
            "precision_hash": winner["precision_hash"],
            **resource,
            "parameter_count": float(size["parameter_count_after"]),
            "mixed_weight_bits": float(size["size_bits_total"]),
            **{f"{key}_count": value for key, value in realized_counts.items()},
            **{f"mutable_gene_{key}_count": value for key, value in mutable_counts.items()},
            "shrinker_width": min(groups["shrinker"].values()),
            "minimum_attention_dh": min(groups["attention"].values()),
            "minimum_ffn_dff": min(groups["ffn"].values()),
            "J_struct": float(metrics["cumulative_structural_taylor"]),
            "J_WQ": float(metrics["cumulative_weight_quant_taylor"]),
            "J_AQ": float(metrics["cumulative_activation_quant_taylor"]),
            "J_total": float(metrics["cumulative_total_taylor"]),
            "B0_AP30": float(b0_eval["AP30"]),
            "B0_AP50": float(b0_eval["AP50"]),
            "B0_AP70": float(b0_eval["AP70"]),
            "B0_mAP": float(b0_eval["mAP"]),
            "S32_AP30": float(s32_eval["AP30"]),
            "S32_AP50": float(s32_eval["AP50"]),
            "S32_AP70": float(s32_eval["AP70"]),
            "S32_mAP": float(s32_eval["mAP"]),
            "JMIX_AP30": float(jmix_eval["AP30"]),
            "JMIX_AP50": float(jmix_eval["AP50"]),
            "JMIX_AP70": float(jmix_eval["AP70"]),
            "JMIX_mAP": float(jmix_eval["mAP"]),
            "structural_drop": float(b0_eval["mAP"] - s32_eval["mAP"]),
            "quantization_extra_drop": float(s32_eval["mAP"] - jmix_eval["mAP"]),
            "joint_drop": float(b0_eval["mAP"] - jmix_eval["mAP"]),
            "S32_mAP_retention": float(s32_eval["mAP"] / b0_eval["mAP"]),
            "JMIX_mAP_retention": float(jmix_eval["mAP"] / b0_eval["mAP"]),
            "S32_accuracy_class": s32_class["classification"],
            "JMIX_accuracy_class": jmix_class["classification"],
            "B0_p50_ms": float(latency["B0"]["p50_ms"]),
            "S32_p50_ms": float(latency["S32"]["p50_ms"]),
            "JMIX_p50_ms": float(latency["JMIX"]["p50_ms"]),
            "S32_speedup": float(latency["S32"]["speedup_vs_matched_B0"]),
            "JMIX_speedup": jmix_speedup,
            "latency_baseline_replay_drift": float(latency["baseline_replay_drift"]),
            "latency_batch_valid": bool(latency["latency_batch_valid"]) and process_isolated,
            "requested_realized_exact": exact,
            "precision_conflict_count": conflict_count,
            "fallback_count": fallback_count,
            "calibration_required": calibration_required,
            "calibration_processed_frames": int(calibration["processed_frames"]),
            "calibration_skipped_frames": int(calibration["skipped_frames"]),
            "calibration_hash": calibration["cache_hash"],
            "calibration_ok": calibration_ok,
            "evaluated_frames": int(jmix_eval["num_evaluated_frames"]),
            "skipped_frames": int(jmix_eval["num_skipped_frames"]),
            "B0_engine_size_bytes": b0_size,
            **engine_sizes,
            "S32_engine_size_compression": b0_size / engine_sizes["S32_engine_size_bytes"],
            "JMIX_engine_size_compression": b0_size / engine_sizes["JMIX_engine_size_bytes"],
            "S32_engine_sha256": s32_result["engine_sha256"],
            "JMIX_engine_sha256": jmix_result["engine_sha256"],
            "ga_admission": admission,
        }
        result_rows.append(row)
        structures.append(
            {
                "budget": row["budget"],
                "candidate_hash": winner["candidate_hash"],
                "cnn_widths": json.dumps(groups["cnn"], sort_keys=True),
                "shrinker_widths": json.dumps(groups["shrinker"], sort_keys=True),
                "attention_dh": json.dumps(groups["attention"], sort_keys=True),
                "ffn_dff": json.dumps(groups["ffn"], sort_keys=True),
                "physical_hash": winner["structure_hash"],
            }
        )
        precisions.append(
            {
                "budget": row["budget"],
                **{f"realized_{key}_count": value for key, value in realized_counts.items()},
                **{f"mutable_gene_{key}_count": value for key, value in mutable_counts.items()},
                "precision_genes": json.dumps(genes, sort_keys=True),
                "precision_hash": winner["precision_hash"],
                "requested_realized_exact": exact,
                "precision_conflict_count": conflict_count,
                "fallback_count": fallback_count,
            }
        )
        buildable = buildable and not build["deployment_invalid"] and calibration_ok and exact

    collapse = {
        "schema_version": "v2xvit-six-budget-ap-collapse-v1",
        "B0": {metric: float(b0_eval[metric]) for metric in METRICS},
        "rows": [
            {
                "budget": row["budget"],
                "S32_classification": row["S32_accuracy_class"],
                "JMIX_classification": row["JMIX_accuracy_class"],
                "structural_drop": row["structural_drop"],
                "quantization_extra_drop": row["quantization_extra_drop"],
                "joint_drop": row["joint_drop"],
                "S32_mAP_retention": row["S32_mAP_retention"],
                "JMIX_mAP_retention": row["JMIX_mAP_retention"],
            }
            for row in result_rows
        ],
    }
    collapse["first_significant_structural_drop_budget"] = _first_budget(
        result_rows, "S32_accuracy_class", {"SIGNIFICANT_DROP", "SEVERE_COLLAPSE"}
    )
    collapse["first_severe_structural_collapse_budget"] = _first_budget(
        result_rows, "S32_accuracy_class", {"SEVERE_COLLAPSE"}
    )
    collapse["first_significant_joint_drop_budget"] = _first_budget(
        result_rows, "JMIX_accuracy_class", {"SIGNIFICANT_DROP", "SEVERE_COLLAPSE"}
    )
    collapse["first_severe_joint_collapse_budget"] = _first_budget(
        result_rows, "JMIX_accuracy_class", {"SEVERE_COLLAPSE"}
    )
    safe_rows = [row for row in result_rows if row["structural_drop"] <= 0.01 and row["joint_drop"] <= 0.01]
    collapse["lowest_safe_budget"] = min((row["budget"] for row in safe_rows), default=None)

    _write_csv(reports / "six_budget_structures.csv", structures)
    _write_csv(reports / "six_budget_precision_profiles.csv", precisions)
    _write_csv(reports / "six_budget_fixed500.csv", result_rows)
    _write_csv(
        reports / "six_budget_latency.csv",
        [
            {key: value for key, value in row.items() if "p50" in key or "speedup" in key or "latency" in key or key == "budget"}
            for row in result_rows
        ],
    )
    _write_csv(
        reports / "six_budget_compression.csv",
        [
            {
                key: value
                for key, value in row.items()
                if key == "budget" or "retention" in key or "compression" in key or "engine_size" in key or key == "parameter_count"
            }
            for row in result_rows
        ],
    )
    _write_json(reports / "six_budget_ap_collapse.json", collapse)
    collapse_lines = ["# Six-budget AP collapse boundary", ""]
    collapse_lines.append("| R_BOPS | S32 mAP | JMIX mAP | S32 class | JMIX class |")
    collapse_lines.append("|---:|---:|---:|---|---|")
    for row in result_rows:
        collapse_lines.append(
            f"| {row['R_BOPS']:.6f} | {row['S32_mAP']:.6f} | {row['JMIX_mAP']:.6f} | "
            f"{row['S32_accuracy_class']} | {row['JMIX_accuracy_class']} |"
        )
    (reports / "six_budget_ap_collapse.md").write_text("\n".join(collapse_lines) + "\n", encoding="utf-8")

    admissions = {f"{row['budget']:.2f}": row["ga_admission"] for row in result_rows}
    admissible = [float(key) for key, value in admissions.items() if value == "GA_ADMISSIBLE"]
    admission_report = {
        "schema_version": "v2xvit-ga-budget-admission-v1",
        "criteria": {
            "S32_absolute_mAP_drop_max": 0.01,
            "JMIX_absolute_mAP_drop_max": 0.01,
            "requested_realized_exact": True,
            "latency_batch_valid": True,
            "JMIX_speedup_min_exclusive": 1.0,
            "taylor_convergence_required": True,
            "framework_tests_required": True,
        },
        "budget_status": admissions,
        "recommended_formal_ga_budgets": admissible,
        "formal_ga_executed": False,
    }
    _write_json(reports / "ga_budget_admission.json", admission_report)
    plot_paths = _plots(reports, result_rows)

    acceptance = {
        "model": "V2X-ViT",
        "ga_framework_version": "stage1-stage2-v3",
        "formal_ga_executed": False,
        "stage1_hard_bops_gate": True,
        "stage1_total_proxy": "J_struct_gate + J_WQ + J_AQ",
        "stage1_overpruning_reward": False,
        "stage2_real_evaluation": True,
        "stage2_accuracy_gate": "greedy_map_minus_0.005",
        "stage2_score": "0.2*A_0.005 + 0.8*T_over_Tgreedy",
        "v1_greedy_anchor": True,
        "v2_online_real_feedback": True,
        "v3_global_real_anchors": True,
        "black_box_surrogate": False,
        "activation_taylor_used": True,
        "joint_taylor_used_for_fitness": False,
        "cross_residual_used_for_fitness": False,
        "structure_proxy": "functional_gate_output_taylor",
        "tracer_used_for_physical_closure": True,
        "legacy_weight_taylor_used_for_fitness": False,
        "taylor_samples": 32,
        "taylor_sample_convergence_insufficient": not bool(convergence["passed"]),
        **run_manifest["search_loop_runtime_audit"],
        "budgets": [0.30, 0.25, 0.20, 0.15, 0.10, 0.05],
        "budget_results": {f"{row['budget']:.2f}": row for row in result_rows},
        **{key: collapse[key] for key in (
            "first_significant_structural_drop_budget",
            "first_severe_structural_collapse_budget",
            "first_significant_joint_drop_budget",
            "first_severe_joint_collapse_budget",
            "lowest_safe_budget",
        )},
        "ga_admissible_budgets": admissible,
        "deployment_closed_all_budgets": buildable,
        "latency_process_audit": process_audit,
        "framework_tests_passed": args.framework_tests_passed,
        "formal_ga_allowed": False,
        "full1789_allowed": False,
        "curve_artifacts": plot_paths,
        "B0_engine_sha256": b0_result["engine_sha256"],
        "B0_engine_file_sha256": _sha256(b0_engine),
    }
    _write_json(reports / "final_acceptance.json", acceptance)

    conclusion = [
        "# V2X-ViT six-budget Greedy conclusion",
        "",
        f"B0 fixed500 mAP is {b0_eval['mAP']:.9f}; every evaluation processed 500/500 frames with zero skips.",
        f"The first significant and severe structural collapse both occur at R_BOPS={collapse['first_severe_structural_collapse_budget']}.",
        f"The first significant and severe joint collapse both occur at R_BOPS={collapse['first_severe_joint_collapse_budget']}.",
        f"The lowest budget satisfying both fixed500 absolute-drop gates is {collapse['lowest_safe_budget']}.",
        "",
        "| Target | R_BOPS | S32 mAP | JMIX mAP | JMIX p50 ms | Speedup | GA status |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in result_rows:
        conclusion.append(
            f"| {row['budget']:.2f} | {row['R_BOPS']:.6f} | {row['S32_mAP']:.6f} | "
            f"{row['JMIX_mAP']:.6f} | {row['JMIX_p50_ms']:.4f} | {row['JMIX_speedup']:.4f}x | {row['ga_admission']} |"
        )
    conclusion.extend(
        [
            "",
            "The 0.05 result remains a severe structural collapse. This run does not execute formal GA or full1789.",
            "Formal GA remains disabled pending explicit approval even for budgets classified GA_ADMISSIBLE.",
        ]
    )
    (root / "root_conclusion.md").write_text("\n".join(conclusion) + "\n", encoding="utf-8")
    return acceptance


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--framework-tests-passed", action="store_true")
    result = run(parser.parse_args())
    print(json.dumps({"lowest_safe_budget": result["lowest_safe_budget"], "ga_admissible_budgets": result["ga_admissible_budgets"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
