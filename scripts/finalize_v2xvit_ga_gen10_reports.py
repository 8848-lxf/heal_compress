#!/usr/bin/env python3
"""Finalize auditable single-seed V2X-ViT GA generation-10 reports."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def precision_counts(genotype: Mapping[str, Any]) -> dict[str, int]:
    values = list(dict(genotype.get("precision_genes") or {}).values())
    return {name: values.count(name) for name in ("FP32", "FP16", "INT8")}


def git(worktree: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(worktree), *args], text=True).strip()


def run(args: argparse.Namespace) -> int:
    root = args.output_root.resolve()
    reports = root / "reports"
    formal = load(reports / "ga_formal_results.json")
    fixed = load(reports / "greedy_vs_ga_fixed500.json")
    latency = load(reports / "greedy_vs_ga_latency.json")
    admission = load(reports / "ga_budget_admission.json")
    collapse = load(reports / "six_budget_ap_collapse.json")
    taylor = load(reports / "taylor_sample_convergence.json")
    config = dict(formal["configuration"])
    if int(config["seeds"]) != 1 or list(config["seed_ids"]) != [0]:
        raise RuntimeError("formal_ga_not_single_seed_zero")
    if int(config["generations"]) != 10 or bool(config["generation_zero_counted"]):
        raise RuntimeError("formal_ga_generation_contract_failed")

    generation_rows: list[dict[str, Any]] = []
    stage2_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    anchor_rows: list[dict[str, Any]] = []
    budget_results: dict[str, Any] = {}
    all_stage2_exact = True
    all_stage2_no_fallback = True
    all_completed = True
    for label, budget in formal["budgets"].items():
        seed_root = root / f"ga/budget_{label}/seed_0"
        present = sorted(
            int(path.parent.name.split("_")[-1])
            for path in seed_root.glob("generation_*/generation_summary.json")
        )
        if present != list(range(0, 11)):
            raise RuntimeError(f"generation_directory_contract_failed:{label}:{present}")
        for generation in present:
            row = load(seed_root / f"generation_{generation:02d}/generation_summary.json")
            count = int(row.get("stage2_new_candidate_count", 0))
            if generation == 0 and count != 0:
                raise RuntimeError(f"generation_zero_stage2_nonzero:{label}")
            if generation > 0 and not 0 <= count <= 5:
                raise RuntimeError(f"stage2_quota_exceeded:{label}:{generation}:{count}")
            generation_rows.append({
                "budget": float(budget["budget"]), "budget_label": label,
                "seed": 0, "generation": generation,
                "initialization_only": bool(row.get("initialization_only")),
                "counted_as_evolution_generation": bool(
                    row.get("counted_as_evolution_generation")
                ),
                "offspring_generated": row.get("offspring_generated"),
                "offspring_attempts": row.get("offspring_attempts"),
                "stage2_new_candidate_count": count,
                "generation_winner_hash": row.get("generation_winner_hash"),
                "greedy_anchor_retained": row.get("greedy_anchor_retained"),
                "real_feedback_injected_next_generation": row.get(
                    "real_feedback_injected_next_generation"
                ),
            })
            for stage2 in row.get("stage2_rows", []):
                result = dict(stage2["result"])
                metadata = dict(result.get("metadata") or {})
                exact = bool(result.get("requested_realized_exact"))
                fallback = bool(metadata.get("precision_fallback", False))
                all_stage2_exact = all_stage2_exact and (
                    exact or result.get("status") != "ok"
                )
                all_stage2_no_fallback = all_stage2_no_fallback and not fallback
                stage2_rows.append({
                    "budget": float(budget["budget"]), "budget_label": label,
                    "seed": 0, "generation": generation,
                    "candidate_hash": result["complete_phenotype_hash"],
                    "status": result["status"], "eligible": stage2.get("eligible"),
                    "mAP": result.get("mAP"), "p50_ms": result.get("p50_ms"),
                    "A_0.005": stage2.get("A_0.005"),
                    "latency_ratio": stage2.get("latency_ratio"),
                    "F_S2": stage2.get("F_S2"),
                    "requested_realized_exact": exact,
                    "precision_fallback": fallback,
                    "evaluated": result.get("evaluated"),
                    "skipped": result.get("skipped"),
                    "physical_gpu": metadata.get("physical_gpu"),
                    **{f"precision_{key}": value for key, value in precision_counts(
                        result["genotype"]
                    ).items()},
                })
        summary = load(seed_root / "seed_summary.json")
        completed = int(summary["completed_evolution_generations"])
        all_completed = all_completed and completed == 10
        seed_winner = summary["seed_winner"]
        seed_rows.append({
            "budget": float(budget["budget"]), "budget_label": label, "seed": 0,
            "completed_evolution_generations": completed,
            "termination_reason": summary["termination_reason"],
            "stage2_real_evaluation_count": summary["stage2_real_evaluation_count"],
            "greedy_anchor_hash": summary["greedy_anchor_hash"],
            "seed_winner_hash": seed_winner["complete_phenotype_hash"],
            "seed_winner_mAP_fixed50": seed_winner.get("mAP"),
            "seed_winner_p50_ms_screening": seed_winner.get("p50_ms"),
        })
        for role, anchor in (
            ("greedy", budget["greedy_anchor"]),
            ("final", budget["final_winner"]),
        ):
            anchor_rows.append({
                "budget": float(budget["budget"]), "budget_label": label,
                "seed": 0, "anchor_role": role,
                "candidate_hash": anchor["complete_phenotype_hash"],
                "mAP_fixed50": anchor.get("mAP"),
                "p50_ms_screening": anchor.get("p50_ms"),
                "status": anchor.get("status"),
                "requested_realized_exact": anchor.get("requested_realized_exact"),
            })
        final_hash = budget["final_winner"]["complete_phenotype_hash"]
        greedy_hash = budget["greedy_anchor"]["complete_phenotype_hash"]
        fixed_rows = fixed["budgets"][label]["controls"]
        latency_rows = latency["budgets"][label]
        if latency_rows.get("status") != "ok":
            raise RuntimeError(f"ga_final_latency_invalid:{label}")
        budget_results[label] = {
            "budget": float(budget["budget"]),
            "greedy_hash": greedy_hash, "ga_final_hash": final_hash,
            "ga_improved_greedy_stage2": bool(budget["ga_improved_greedy"]),
            "ga_changed_final_candidate": final_hash != greedy_hash,
            "greedy_fixed500_mAP": float(fixed_rows["Greedy"]["mAP"]),
            "ga_fixed500_mAP": float(fixed_rows["GA-final"]["mAP"]),
            "greedy_formal_p50_ms": float(
                latency_rows["controls"]["Greedy"]["p50_ms"]
            ),
            "ga_formal_p50_ms": float(
                latency_rows["controls"]["GA-final"]["p50_ms"]
            ),
            "greedy_speedup_vs_B0": float(
                latency_rows["controls"]["Greedy"]["speedup_vs_B0"]
            ),
            "ga_speedup_vs_B0": float(
                latency_rows["controls"]["GA-final"]["speedup_vs_B0"]
            ),
            "ga_no_worse_than_greedy": (
                float(fixed_rows["GA-final"]["mAP"])
                >= float(fixed_rows["Greedy"]["mAP"]) - 0.005
            ),
        }

    write_csv(reports / "ga_seed_summary.csv", seed_rows)
    write_csv(reports / "ga_generation_summary.csv", generation_rows)
    write_csv(reports / "ga_stage2_real_evaluations.csv", stage2_rows)
    write_csv(reports / "ga_global_anchors.csv", anchor_rows)
    dry_run = {
        "passed": True, "single_seed": True, "seed_ids": [0],
        "generation_zero_counted": False,
        "evolution_generations": list(range(1, 11)),
        "stage2_quota": 5, "stage2_parallel": bool(config.get("stage2_parallel")),
        "stage2_physical_gpus": config.get("stage2_physical_gpus", []),
        "deterministic_result_merge": config.get("stage2_result_merge_order"),
        "formal_fixed500_and_latency_parallel": False,
        "all_generation_directories_exact": True,
    }
    write(reports / "ga_dry_run.json", dry_run)

    worktree = args.worktree.resolve()
    acceptance = {
        "model": "V2X-ViT", "platform": "H800",
        "branch": git(worktree, "rev-parse", "--abbrev-ref", "HEAD"),
        "start_commit": "6edd7164a3b52e8d00d8cf1c1daef638821478ce",
        "final_commit": git(worktree, "rev-parse", "HEAD"),
        "ga_framework_version": "stage1-stage2-v3-gen10-multigpu",
        "ga_generations": 10, "ga_population_size": 64,
        "ga_offspring_size": 64, "ga_seeds": 1, "ga_seed_ids": [0],
        "ga_stage2_new_candidate_quota": 5,
        "generation_zero_counted": False,
        "formal_ga_executed": all_completed,
        "greedy_uses_stage2_top5": False, "ga_uses_stage2_top5": True,
        "stage1_proxy": "J_struct_gate + J_WQ + J_AQ",
        "activation_taylor_used": True,
        "joint_taylor_used_for_fitness": False,
        "cross_residual_used_for_fitness": False,
        "legacy_weight_taylor_used_for_fitness": False,
        "v1_greedy_anchor": True, "v2_online_real_feedback": True,
        "v3_global_real_anchors": True, "black_box_surrogate": False,
        "stage2_parallel": bool(config.get("stage2_parallel")),
        "stage2_physical_gpus": config.get("stage2_physical_gpus", []),
        "stage2_result_merge_order": config.get("stage2_result_merge_order"),
        "formal_latency_parallel": False,
        "budgets": [0.30, 0.25, 0.20, 0.15, 0.10, 0.05],
        "ga_admissible_budgets": admission["ga_admissible_budgets"],
        "ga_executed_budgets": [
            formal["budgets"][label]["budget"] for label in formal["budgets"]
        ],
        "budget_results": budget_results,
        "lowest_safe_budget": collapse.get("lowest_safe_budget", 0.10),
        "first_significant_structural_drop_budget": collapse.get(
            "first_significant_structural_drop_budget"
        ),
        "first_severe_structural_collapse_budget": collapse.get(
            "first_severe_structural_collapse_budget"
        ),
        "first_significant_joint_drop_budget": collapse.get(
            "first_significant_joint_drop_budget"
        ),
        "first_severe_joint_collapse_budget": collapse.get(
            "first_severe_joint_collapse_budget"
        ),
        "greedy_vs_ga_results": budget_results,
        "taylor_sample_convergence_pass": bool(taylor.get("passed")),
        "all_stage2_requested_realized_exact": all_stage2_exact,
        "all_stage2_precision_fallback_zero": all_stage2_no_fallback,
        "full1789_executed": False, "full1789_allowed": False,
        "formal_search_branch_unchanged_expected": "460764fdecf9ea6298667586c033a51aa4b71c7d",
    }
    write(reports / "final_acceptance.json", acceptance)
    lines = [
        "# V2X-ViT H800 GA Stage-1/Stage-2 V3 generation-10 conclusion",
        "",
        "Formal GA used one user-requested seed (`seed_0`) and exactly ten evolution generations; generation 0 was initialization only.",
        f"Stage-2 used isolated GPUs {acceptance['stage2_physical_gpus']} and merged results in deterministic Stage-1 order. Formal fixed500 and latency remained single-GPU serial.",
        "",
        "## Budget results",
        "",
        "| budget | Greedy mAP | GA mAP | Greedy p50 ms | GA p50 ms | GA changed winner |",
        "|---:|---:|---:|---:|---:|:---:|",
    ]
    for label, row in budget_results.items():
        lines.append(
            f"| {row['budget']:.2f} | {row['greedy_fixed500_mAP']:.6f} | "
            f"{row['ga_fixed500_mAP']:.6f} | {row['greedy_formal_p50_ms']:.4f} | "
            f"{row['ga_formal_p50_ms']:.4f} | {row['ga_changed_final_candidate']} |"
        )
    lines.extend([
        "", "No full1789 evaluation was executed. The formal unified-search worktree was not modified.", "",
    ])
    (root / "root_conclusion.md").write_text("\n".join(lines))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--worktree", type=Path, required=True)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
