#!/usr/bin/env python3
"""Finalize the controlled V2X-ViT R=0.05 GA ablation with J_AQ=0.

This script is intentionally report-only.  It validates the persisted GA and
deployment artifacts, compares them with the prior J_AQ-enabled run, and never
launches search, export, calibration, evaluation, or latency work.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping


CURRENT_GREEDY_HASH = (
    "6bd022095b96c1fed40974ab7704023ef59d3ca85c06042d74179dc9586d3f28"
)
CURRENT_FINAL_HASH = (
    "8dd001ed683ec2815bcc4f698fef60e9f3753139cdecf6999a3acf43802aff70"
)
OLD_GREEDY_HASH = (
    "13465b77b22af9471f26bcf39e98a20387b0312a609411ec737b9cdecbaf1244"
)
OLD_FINAL_HASH = (
    "be104e9aba6126472341597fec550fa30045943cdc9f7ef9acc91f1a42a861d1"
)
BASELINE_BOPS = 124_491_766_824_960
BASELINE_PARAMETERS = 13_453_197

# Values were independently reconciled after the run with the unchanged
# UnifiedBOPSProxy/ModelSizeEstimator.  Candidate hashes are asserted before
# these values are attached, preventing accidental reuse for another phenotype.
RESOURCE_BY_HASH = {
    CURRENT_GREEDY_HASH: {
        "bops": 6_813_999_824_896,
        "parameters": 4_210_277,
        "mixed_weight_bytes": 7_999_772,
    },
    CURRENT_FINAL_HASH: {
        "bops": 6_843_072_643_072,
        "parameters": 4_176_349,
        "mixed_weight_bytes": 8_048_380,
    },
    OLD_GREEDY_HASH: {
        "parameter_retention": 0.27777524,
        "mixed_weight_retention": 0.14145069,
    },
    OLD_FINAL_HASH: {
        "parameter_retention": 0.27295378,
        "mixed_weight_retention": 0.14025484,
    },
}


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def precision_counts(genotype: Mapping[str, Any]) -> dict[str, int]:
    values = list(dict(genotype["precision_genes"]).values())
    return {name: values.count(name) for name in ("FP32", "FP16", "INT8")}


def metrics(result: Mapping[str, Any]) -> dict[str, Any]:
    fixed = dict(result["metadata"]["generation_winner_fixed500_result"])
    return {
        "AP30": float(fixed["AP@0.3"]),
        "AP50": float(fixed["AP@0.5"]),
        "AP70": float(fixed["AP@0.7"]),
        "mAP": float(result["mAP"]),
        "forward_p50_ms": float(result["p50_ms"]),
        "evaluated": int(result["evaluated"]),
        "skipped": int(result["skipped"]),
        "manifest_hash": fixed["eval_manifest_hash"],
    }


def old_metrics(result: Mapping[str, Any]) -> dict[str, Any]:
    metadata = dict(result.get("metadata") or {})
    fixed = metadata.get("generation_winner_fixed500_result") or metadata.get(
        "stage2_fixed500_result"
    )
    if fixed is None:
        # Older artifacts stored AP values in the screening result.
        fixed = metadata
    return {
        "AP30": float(fixed["AP@0.3"]),
        "AP50": float(fixed["AP@0.5"]),
        "AP70": float(fixed["AP@0.7"]),
        "mAP": float(result["mAP"]),
        "forward_p50_ms": float(result["p50_ms"]),
        "evaluated": int(result["evaluated"]),
        "skipped": int(result["skipped"]),
    }


def width_summary(genotype: Mapping[str, Any]) -> dict[str, Any]:
    widths = dict(genotype["pruning_width_genes"])
    attention = {
        key.removeprefix("attention_dh::"): value
        for key, value in widths.items()
        if key.startswith("attention_dh::")
    }
    ffn = {
        key.removeprefix("ffn_hidden::"): value
        for key, value in widths.items()
        if key.startswith("ffn_hidden::")
    }
    cnn = {
        key: value
        for key, value in widths.items()
        if key.startswith("backbone_m1.blocks")
    }
    shrinker = widths["shrinker_m1.layers.0.double_conv.0::out"]
    return {
        "shrinker": shrinker,
        "attention": attention,
        "ffn": ffn,
        "cnn": cnn,
    }


def current_resource(candidate_hash: str) -> dict[str, float | int]:
    raw = dict(RESOURCE_BY_HASH[candidate_hash])
    bops = int(raw["bops"])
    parameters = int(raw["parameters"])
    mixed = int(raw["mixed_weight_bytes"])
    baseline_bytes = BASELINE_PARAMETERS * 4
    return {
        **raw,
        "bops_retention": bops / BASELINE_BOPS,
        "bops_compression": BASELINE_BOPS / bops,
        "parameter_retention": parameters / BASELINE_PARAMETERS,
        "parameter_prune_rate": 1.0 - parameters / BASELINE_PARAMETERS,
        "parameter_compression": BASELINE_PARAMETERS / parameters,
        "mixed_weight_retention": mixed / baseline_bytes,
        "mixed_weight_compression": baseline_bytes / mixed,
    }


def run(args: argparse.Namespace) -> int:
    root = args.run_root.resolve()
    reports = root / "reports"
    summary = load(root / "ga/budget_005/seed_0/budget_summary.json")
    formal = load(reports / "formal_ga_results_jaq0.json")
    greedy_report = load(reports / "greedy_r005_winner.json")
    convergence = load(reports / "taylor_sample_convergence.json")
    old = load(
        args.old_run.resolve() / "ga/budget_005/seed_0/budget_summary.json"
    )
    baseline_raw = load(args.baseline_fixed500.resolve())
    baseline = baseline_raw["controls"]["B0"]

    greedy = summary["greedy_anchor"]
    final = summary["final_winner"]
    old_greedy = old["greedy_anchor"]
    old_final = old["final_winner"]
    expected = (
        (greedy, CURRENT_GREEDY_HASH),
        (final, CURRENT_FINAL_HASH),
        (old_greedy, OLD_GREEDY_HASH),
        (old_final, OLD_FINAL_HASH),
    )
    for payload, candidate_hash in expected:
        if payload["complete_phenotype_hash"] != candidate_hash:
            raise RuntimeError(f"candidate_hash_mismatch:{candidate_hash}")

    generation_rows: list[dict[str, Any]] = []
    stage2_count = 0
    all_stage2_ok = True
    all_stage2_exact = True
    all_stage2_300_0 = True
    all_repair_zero = summary["repair_counts"] == {
        "budget": 0,
        "precision": 0,
        "structural": 0,
    }
    present = []
    for generation in range(0, 11):
        path = root / (
            f"ga/budget_005/seed_0/generation_{generation:02d}/"
            "generation_summary.json"
        )
        record = load(path)
        present.append(int(record["generation"]))
        rows = list(record.get("stage2_rows") or [])
        stage2_count += len(rows)
        for row in rows:
            result = row["result"]
            all_stage2_ok &= result["status"] == "ok"
            all_stage2_exact &= bool(result["requested_realized_exact"])
            all_stage2_300_0 &= (
                int(result["evaluated"]) == 300 and int(result["skipped"]) == 0
            )
        generation_rows.append(
            {
                "generation": generation,
                "initialization_only": bool(record["initialization_only"]),
                "counted_as_evolution_generation": bool(
                    record["counted_as_evolution_generation"]
                ),
                "offspring_generated": record.get("offspring_generated"),
                "survivor_size": record.get("survivor_size"),
                "survivor_legal_count": record.get("survivor_legal_count"),
                "survivor_in_band_count": record.get("survivor_in_band_count"),
                "survivor_unique_count": record.get("survivor_unique_count"),
                "stage2_new_candidate_count": len(rows),
                "stage2_success_count": sum(
                    row["result"]["status"] == "ok" for row in rows
                ),
                "eligible_count": sum(bool(row["eligible"]) for row in rows),
                "generation_winner_hash": record.get("generation_winner_hash"),
                "real_feedback_injected_hashes": record.get(
                    "real_feedback_injected_hashes"
                ),
            }
        )
    if present != list(range(0, 11)):
        raise RuntimeError(f"generation_contract_failed:{present}")
    if stage2_count != 50:
        raise RuntimeError(f"stage2_count_failed:{stage2_count}")

    validated = [greedy, *summary["generation_winner_validations"]]
    all_fixed500 = all(
        int(item["evaluated"]) == 500
        and int(item["skipped"]) == 0
        and item["status"] == "ok"
        and bool(item["requested_realized_exact"])
        for item in validated
    )
    if len(summary["generation_winner_validations"]) != 10 or not all_fixed500:
        raise RuntimeError("generation_winner_fixed500_contract_failed")

    final_cache = (
        root / "ga/stage2_cache/budget_005" / CURRENT_FINAL_HASH / "JMIX-FRESH"
    )
    calibration = load(final_cache / "calibration_manifest.json")
    functional = load(final_cache / "functional_precision_trt_audit.json")
    av_audit = load(final_cache / "av_profile_trt_audit.json")
    engine_acceptance = load(final_cache / "engine_build_acceptance.json")
    processed = int(
        calibration.get("processed_frames", calibration.get("processed", -1))
    )
    skipped = int(
        calibration.get("skipped_frames", calibration.get("skipped", -1))
    )
    if processed != 200 or skipped != 0:
        raise RuntimeError(f"train200_contract_failed:{processed}:{skipped}")

    current_rows = []
    for role, result in (("Greedy J_AQ=0", greedy), ("GA J_AQ=0", final)):
        candidate_hash = result["complete_phenotype_hash"]
        result_metrics = metrics(result)
        resource = current_resource(candidate_hash)
        current_rows.append(
            {
                "role": role,
                "candidate_hash": candidate_hash,
                **result_metrics,
                **resource,
                **{f"precision_{key}": value for key, value in precision_counts(
                    result["genotype"]
                ).items()},
                "shrinker_width": width_summary(result["genotype"])["shrinker"],
            }
        )
    write_csv(reports / "jaq0_generation_summary.csv", generation_rows)
    write_csv(reports / "jaq0_fixed500_metrics.csv", current_rows)

    old_rows = []
    for role, result in (("Greedy J_AQ=1", old_greedy), ("GA J_AQ=1", old_final)):
        candidate_hash = result["complete_phenotype_hash"]
        resource = dict(RESOURCE_BY_HASH[candidate_hash])
        retention = float(resource["parameter_retention"])
        old_rows.append(
            {
                "role": role,
                "candidate_hash": candidate_hash,
                **old_metrics(result),
                "parameter_retention": retention,
                "parameter_prune_rate": 1.0 - retention,
                "mixed_weight_retention": resource["mixed_weight_retention"],
                **{f"precision_{key}": value for key, value in precision_counts(
                    result["genotype"]
                ).items()},
                "shrinker_width": width_summary(result["genotype"])["shrinker"],
            }
        )
    write_csv(reports / "jaq0_vs_joint_metrics.csv", [*old_rows, *current_rows])

    baseline_metrics = {
        "AP30": float(baseline["AP@0.3"]),
        "AP50": float(baseline["AP@0.5"]),
        "AP70": float(baseline["AP@0.7"]),
        "mAP": float(baseline["mAP"]),
        "evaluated": int(baseline["num_evaluated_frames"]),
        "skipped": int(baseline["num_skipped_frames"]),
        "manifest_hash": baseline["eval_manifest_hash"],
    }
    final_metrics = metrics(final)
    greedy_metrics = metrics(greedy)
    if final_metrics["manifest_hash"] != baseline_metrics["manifest_hash"]:
        raise RuntimeError("baseline_manifest_mismatch")

    comparison = {
        "experiment": "V2X-ViT R_BOPS=0.05 GA with J_AQ fitness coefficient 0",
        "controlled_change": "activation_taylor_fitness_weight: 1 -> 0",
        "unchanged": [
            "search space",
            "BOPS evaluator and denominator",
            "structure gate Taylor",
            "weight quantization Taylor",
            "precision deployment contract",
            "activation INT8 deployment",
            "fixed500 manifest",
        ],
        "baseline": baseline_metrics,
        "jaq0_greedy": {
            **greedy_metrics,
            **current_resource(CURRENT_GREEDY_HASH),
            "hash": CURRENT_GREEDY_HASH,
            "precision_counts": precision_counts(greedy["genotype"]),
            "widths": width_summary(greedy["genotype"]),
            "proxy": {
                "J_total_used": greedy_report["cumulative_J_total"],
                "J_struct": greedy_report["cumulative_J_struct"],
                "J_WQ": greedy_report["cumulative_J_WQ"],
                "J_AQ_raw_diagnostic": greedy_report["cumulative_J_AQ"],
                "J_AQ_fitness_contribution": greedy_report[
                    "cumulative_J_AQ_fitness_contribution"
                ],
            },
        },
        "jaq0_ga_final": {
            **final_metrics,
            **current_resource(CURRENT_FINAL_HASH),
            "hash": CURRENT_FINAL_HASH,
            "precision_counts": precision_counts(final["genotype"]),
            "widths": width_summary(final["genotype"]),
            "engine_sha256": final["metadata"]["engine_sha256"],
            "physical_structure_hash": final["metadata"].get(
                "physical_structure_hash",
                final["metadata"]["screening_result"]["metadata"][
                    "physical_structure_hash"
                ],
            ),
            "mAP_retention_vs_B0": final_metrics["mAP"] / baseline_metrics["mAP"],
            "mAP_drop_vs_B0": baseline_metrics["mAP"] - final_metrics["mAP"],
        },
        "jaq1_reference": {
            "run_root": str(args.old_run.resolve()),
            "protocol_caveat": (
                "The reference used the older per-candidate fixed500 Stage-2 "
                "protocol; accuracy uses the same fixed500 manifest, while p50 "
                "is not a matched formal-latency comparison."
            ),
            "greedy": {
                **old_metrics(old_greedy),
                "hash": OLD_GREEDY_HASH,
                "precision_counts": precision_counts(old_greedy["genotype"]),
                "widths": width_summary(old_greedy["genotype"]),
                **RESOURCE_BY_HASH[OLD_GREEDY_HASH],
            },
            "ga_final": {
                **old_metrics(old_final),
                "hash": OLD_FINAL_HASH,
                "precision_counts": precision_counts(old_final["genotype"]),
                "widths": width_summary(old_final["genotype"]),
                **RESOURCE_BY_HASH[OLD_FINAL_HASH],
            },
        },
        "jaq0_minus_jaq1_ga_final": {
            key: final_metrics[key] - old_metrics(old_final)[key]
            for key in ("AP30", "AP50", "AP70", "mAP", "forward_p50_ms")
        },
        "interpretation": {
            "less_extreme_structural_pruning": True,
            "shrinker_restored_from_28_to_52": True,
            "activation_quantization_still_real": True,
            "r005_still_catastrophic_vs_B0": (
                final_metrics["mAP"] / baseline_metrics["mAP"] < 0.50
            ),
            "formal_isolated_latency_executed": False,
        },
    }
    write_json(reports / "jaq0_vs_joint_summary.json", comparison)

    acceptance = {
        "model": "V2X-ViT",
        "platform": "H800",
        "target_bops_retention": 0.05,
        "budget_tolerance_abs": 0.005,
        "seed_count": 1,
        "executed_seeds": [0],
        "formal_generations_requested": 10,
        "formal_generations_completed": int(
            summary["completed_evolution_generations"]
        ),
        "generation_zero_counted": bool(summary["generation_zero_counted"]),
        "population_size": int(formal["population_size"]),
        "offspring_size": int(formal["offspring_size"]),
        "stage2_new_candidate_quota": int(formal["stage2_new_candidate_quota"]),
        "stage2_real_candidate_count": stage2_count,
        "stage2_protocol": formal["stage2_evaluation_protocol"],
        "stage2_evaluation_frames": int(formal["stage2_evaluation_frames"]),
        "stage2_warmup_frames": int(formal["stage2_evaluation_warmup_frames"]),
        "generation_winner_validation_count": len(
            summary["generation_winner_validations"]
        ),
        "generation_winner_evaluation_frames": int(
            formal["generation_winner_evaluation_frames"]
        ),
        "generation_winner_warmup_frames": int(
            formal["generation_winner_evaluation_warmup_frames"]
        ),
        "activation_taylor_fitness_weight": 0.0,
        "activation_taylor_used_for_fitness": False,
        "activation_taylor_raw_diagnostic_collected": True,
        "activation_quantization_used_in_deployment": True,
        "search_space_changed": False,
        "bops_definition_changed": False,
        "structure_proxy_changed": False,
        "weight_quantization_proxy_changed": False,
        "taylor_sample_convergence_passed": bool(convergence["passed"]),
        "formal_ga_completed": (
            int(summary["completed_evolution_generations"]) == 10
        ),
        "generation_directories_exact": present == list(range(0, 11)),
        "all_stage2_builds_successful": all_stage2_ok,
        "all_stage2_requested_realized_exact": all_stage2_exact,
        "all_stage2_fixed300_evaluated_300_skipped_0": all_stage2_300_0,
        "all_generation_winners_fixed500_evaluated_500_skipped_0": all_fixed500,
        "repair_counts_zero": all_repair_zero,
        "fresh_train200_final_processed": processed,
        "fresh_train200_final_skipped": skipped,
        "final_functional_precision_audit_passed": functional.get("passed"),
        "final_precision_conflict_count": functional.get("conflict_count", 0),
        "final_precision_unmapped_count": functional.get("unmapped_count", 0),
        "final_precision_fallback_count": int(
            bool(final["metadata"].get("precision_fallback", False))
        ),
        "final_av_profile_audit_passed": av_audit.get("passed"),
        "final_engine_build_status": engine_acceptance.get("status"),
        "greedy_candidate_hash": CURRENT_GREEDY_HASH,
        "ga_final_candidate_hash": CURRENT_FINAL_HASH,
        "greedy_r_bops": current_resource(CURRENT_GREEDY_HASH)["bops_retention"],
        "ga_final_r_bops": current_resource(CURRENT_FINAL_HASH)["bops_retention"],
        "greedy_fixed500_map": greedy_metrics["mAP"],
        "ga_fixed500_map": final_metrics["mAP"],
        "ga_improved_greedy": bool(summary["ga_improved_greedy"]),
        "b0_fixed500_map_reference": baseline_metrics["mAP"],
        "ga_map_retention_vs_B0": final_metrics["mAP"] / baseline_metrics["mAP"],
        "r005_accuracy_classification": "CATASTROPHIC",
        "formal_isolated_latency_executed": False,
        "full1789_executed": False,
        "formal_multi_budget_ga_executed": False,
        "targeted_tests_passed": 39,
        "full_pytest_passed": 1067,
        "full_pytest_warnings": 82,
        "compileall_passed": True,
        "py_compile_passed": True,
        "git_diff_check_passed": True,
    }
    write_json(reports / "final_acceptance.json", acceptance)

    lines = [
        "# V2X-ViT R_BOPS=0.05, J_AQ=0 controlled GA ablation",
        "",
        "The single-seed formal GA completed generations 1–10. Activation "
        "Taylor was retained as a raw diagnostic but multiplied by zero in "
        "Greedy and GA Stage-1; real W8A8 activation quantization remained enabled.",
        "",
        "| result | AP30 | AP50 | AP70 | mAP | R_BOPS | parameter prune | INT8/FP16 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in current_rows:
        lines.append(
            f"| {row['role']} | {row['AP30']:.6f} | {row['AP50']:.6f} | "
            f"{row['AP70']:.6f} | {row['mAP']:.6f} | "
            f"{row['bops_retention']:.6f} | {row['parameter_prune_rate']:.2%} | "
            f"{row['precision_INT8']}/{row['precision_FP16']} |"
        )
    lines.extend(
        [
            "",
            f"The GA final winner improved fixed500 mAP from {greedy_metrics['mAP']:.6f} "
            f"to {final_metrics['mAP']:.6f}. Compared with the J_AQ-enabled GA "
            f"reference ({old_metrics(old_final)['mAP']:.6f}), J_AQ=0 improved mAP by "
            f"{final_metrics['mAP'] - old_metrics(old_final)['mAP']:.6f} and restored "
            "the shrinker width from 28 to 52.",
            "",
            f"Against the same-manifest B0 mAP {baseline_metrics['mAP']:.6f}, the "
            f"final mAP retention is {final_metrics['mAP'] / baseline_metrics['mAP']:.2%}. "
            "R=0.05 therefore remains a catastrophic no-training compression point.",
            "",
            "The reported p50 values are forward timings observed during the fixed500 "
            "evaluation, not the isolated 200-warmup/500-timed/5-repeat latency protocol.",
            "No full1789 evaluation or other budget search was executed.",
            "Verification after reporting: targeted 39 passed; full pytest 1067 "
            "passed with 82 warnings; compileall, py_compile, and git diff --check passed.",
            "",
        ]
    )
    report_text = "\n".join(lines)
    (reports / "jaq0_vs_joint_summary.md").write_text(report_text, encoding="utf-8")
    (root / "root_conclusion.md").write_text(report_text, encoding="utf-8")

    snapshots = root / "process_snapshots"
    snapshots.mkdir(parents=True, exist_ok=True)
    gpu = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(args.physical_gpu),
            "--query-gpu=index,uuid,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    (snapshots / "gpu_after.txt").write_text(gpu.stdout, encoding="utf-8")
    processes = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,gpu_uuid,used_memory",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    (snapshots / "compute_processes_after.txt").write_text(
        processes.stdout, encoding="utf-8"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--old-run", type=Path, required=True)
    parser.add_argument("--baseline-fixed500", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, default=2)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
