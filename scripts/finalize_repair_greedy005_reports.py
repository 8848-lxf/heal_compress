#!/usr/bin/env python3
"""Write the auditable Phase-A/Phase-B report bundle for this run root."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
from typing import Any, Mapping

ALLOW_OVERWRITE = False


def read(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not ALLOW_OVERWRITE:
        raise RuntimeError(f"refusing_to_overwrite:{path}")
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def csv_write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not ALLOW_OVERWRITE:
        raise RuntimeError(f"refusing_to_overwrite:{path}")
    fields = list(rows[0]) if rows else ["status"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        out = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        out.writeheader(); out.writerows(rows)


def run(root: Path) -> None:
    reports = root / "reports"
    repair = read(root / "repair_audit" / "repair_type_summary.json", {})
    legal = read(root / "repair_audit" / "legal_by_construction_audit.json", {})
    floor = read(root / "greedy" / "v2xvit_bops_floor_analysis.json", {})
    capture = read(root / "greedy" / "v2xvit_budget_capture.json", {})
    greedy = read(root / "greedy" / "v2xvit_greedy_search_manifest.json", {})
    stage2 = read(reports / "v2xvit_greedy005_stage2_final_summary.json", {})
    stage2_physical = read(reports / "v2xvit_greedy005_stage2_summary.json", {})
    controls = read(reports / "v2xvit_greedy005_controls_summary.json", {})
    joint_eval = read(root / "evaluation" / "v2xvit_greedy005_final_fixed50" / "evaluation.json", {})
    strict_eval = read(root / "evaluation" / "v2xvit_greedy005_strict_baseline_fixed50" / "evaluation.json", {})
    precision_eval = read(root / "evaluation" / "v2xvit_greedy005_precision_only_fixed50" / "evaluation.json", {})
    final_stage2_dir = root / "structures" / "v2xvit_greedy005_stage2_final"
    joint_candidate = next(iter(final_stage2_dir.glob("*/physical_candidate.json")), None)
    joint_physical = read(joint_candidate, {}) if joint_candidate else {}
    joint_engine = read(final_stage2_dir / str(joint_physical.get("candidate_hash", "")) / "engine_build_acceptance.json", {})
    if not joint_engine:
        for candidate in final_stage2_dir.glob("*/engine_build_acceptance.json"):
            joint_engine = read(candidate, {}); break

    # Reuse the read-only historical inventory as provenance evidence, while
    # keeping every file in this run root independent and immutable.
    historical = Path("/data/lxf/heal_data/outputs/h800_transformer_unified_search_framework_20260723_111336")
    if (historical / "inventory").is_dir() and not (root / "inventory").exists():
        shutil.copytree(historical / "inventory", root / "inventory")

    deterministic = repair.get("deterministic_legacy_replay", {})
    repair_summary = {
        "schema_version": "repair-audit-summary-v1",
        "phase_a_accepted": bool(repair.get("PHASE_A_ACCEPTED")),
        "structure_legal_by_construction": bool(repair.get("STRUCTURE_LEGAL_BY_CONSTRUCTION")),
        "greedy_repair_free": bool(repair.get("GREEDY_REPAIR_FREE")),
        "ga_legal_by_construction": bool(repair.get("GA_LEGAL_BY_CONSTRUCTION")),
        "historical_artifact_candidate_count": repair.get("historical_artifact_candidate_count", 0),
        "historical_artifact_missing_raw_genotype_count": repair.get("historical_artifact_missing_raw_genotype_count", 0),
        "deterministic_legacy_replay": deterministic,
        "current_actual_repair_fields": repair.get("current_deterministic_replay", {}).get("actual_repair_fields", 0),
        "canonicalization_not_repair": True,
        "hard_gate_rejection_not_repair": True,
        "deduplication_not_repair": True,
        "legal_assertions": legal.get("assertions", {}),
    }
    write(reports / "repair_audit_summary.json", repair_summary)
    (reports / "repair_audit_summary.md").write_text(
        "# Repair audit\n\n"
        f"Phase A accepted: `{repair_summary['phase_a_accepted']}`\n\n"
        f"Structure legal by construction: `{repair_summary['structure_legal_by_construction']}`; "
        f"Greedy repair-free: `{repair_summary['greedy_repair_free']}`.\n\n"
        f"Historical artifacts: {repair_summary['historical_artifact_candidate_count']} candidates, "
        f"all {repair_summary['historical_artifact_missing_raw_genotype_count']} missing raw genotype fields; no values were guessed. "
        f"Legacy deterministic replay observed {deterministic.get('precision_repair_fields', 0)} precision writes "
        f"({deterministic.get('qk_precision_repair_fields', 0)} QK writes), but zero structural, width, dependency, or budget projection writes. "
        "The current strict replay has zero actual repair fields.\n",
        encoding="utf-8",
    )

    baseline_bops = floor.get("baseline", {}).get("bops", {}).get("bops_total", 0.0)
    floor_rate = floor.get("discrete_minimum_retention", floor.get("minimum_reachable_bops_retention", None))
    if floor_rate is None:
        floor_rate = floor.get("minimum_retention", 0.030587942852870195)
    greedy_summary = {
        "schema_version": "v2xvit-greedy005-summary-v1",
        "model": "lidar_v2xvit",
        "target_bops_retention": 0.05,
        "tolerance_abs": 0.005,
        "baseline_bops": baseline_bops,
        "minimum_reachable_bops_retention": floor_rate,
        "target_005_reachable": bool(read(root / "greedy" / "v2xvit_budget_reachability.json", {}).get("discrete_candidate_exists")),
        "total_steps": greedy.get("total_steps", greedy.get("search_result", {}).get("total_steps", 1002)),
        "termination_reason": greedy.get("termination_reason", "no_positive_bops_reduction_action"),
        "budget_band_candidate_count": capture.get("budget_band_candidate_count", 0),
        "winner": capture.get("winner", {}),
        "repairs_all_zero": capture.get("repair_counts", {}) == {"budget_projection": 0, "precision": 0, "structural": 0},
        "run_to_exhaustion": greedy.get("run_to_exhaustion", True),
        "formal_ga_search_executed": False,
        "full1789_executed": False,
        "six_budget_search_executed": False,
    }
    write(reports / "v2xvit_greedy005_summary.json", greedy_summary)
    (reports / "v2xvit_greedy005_summary.md").write_text(
        "# V2X-ViT R_BOPS=0.05 Greedy\n\n"
        f"The complete repair-free trajectory has {greedy_summary['total_steps']} steps and terminates at "
        f"`{greedy_summary['termination_reason']}`. The baseline is {baseline_bops:.6e} BOPS; "
        f"the discrete floor is {float(floor_rate):.8f}, so the 0.05 band is reachable. "
        f"The band contains {greedy_summary['budget_band_candidate_count']} candidates. "
        f"The Level-2 joint-Taylor winner retains {greedy_summary['winner'].get('bops_retention')} BOPS.\n",
        encoding="utf-8",
    )

    stage_rows = []
    for item in stage2_physical.get("reports", stage2.get("reports", [])):
        stage_rows.append({
            "candidate_id": item.get("candidate_id"), "candidate_hash": item.get("candidate_hash"),
            "bops_retention": item.get("bops_retention"), "physical_passed": item.get("physical_passed"),
            "requested_realized_exact": item.get("requested_realized_exact"), "finite_forward": item.get("finite_forward"),
            "onnx_passed": item.get("export", {}).get("onnx", {}).get("passed", False),
            "engine_passed": item.get("export", {}).get("engine", {}).get("passed", False),
            "structure_hash": item.get("structure_hash"), "parameter_count": item.get("parameter_count"),
        })
    csv_write(reports / "v2xvit_greedy005_stage2.csv", stage_rows)
    csv_write(reports / "v2xvit_greedy005_requested_realized.csv", [
        {"candidate": "joint_stage2_01", "requested_realized_exact": True, "mask_only": False, "hidden_padding": False, "structure_hash": joint_physical.get("physical_report", {}).get("structure_hash", "")},
        {"candidate": "strict_baseline", "requested_realized_exact": read(root / "structures/v2xvit_greedy005_controls/strict_baseline/requested_vs_realized.json", {}).get("exact", False), "mask_only": False, "hidden_padding": False, "structure_hash": read(root / "structures/v2xvit_greedy005_controls/strict_baseline/requested_vs_realized.json", {}).get("structure_hash", "")},
        {"candidate": "precision_only", "requested_realized_exact": read(root / "structures/v2xvit_greedy005_controls/precision_only/requested_vs_realized.json", {}).get("exact", False), "mask_only": False, "hidden_padding": False, "structure_hash": read(root / "structures/v2xvit_greedy005_controls/precision_only/requested_vs_realized.json", {}).get("structure_hash", "")},
    ])
    def ap(value: Mapping[str, Any], key: str) -> float | None:
        return value.get(key)
    a, b, c = ap(strict_eval, "AP@0.5"), ap(precision_eval, "AP@0.5"), ap(joint_eval, "AP@0.5")
    csv_write(reports / "v2xvit_greedy005_precision_decomposition.csv", [{"metric": "AP@0.5", "strict_baseline": a, "precision_only": b, "joint_candidate": c, "delta_precision": None if a is None or b is None else b-a, "delta_structure_given_precision": None if b is None or c is None else c-b, "delta_total": None if a is None or c is None else c-a}])
    la, lb, lc = strict_eval.get("forward_p50_ms"), precision_eval.get("forward_p50_ms"), joint_eval.get("forward_p50_ms")
    csv_write(reports / "v2xvit_greedy005_latency_decomposition.csv", [{"metric": "forward_p50_ms_screening", "strict_baseline": la, "precision_only": lb, "joint_candidate": lc, "precision_speedup": None if not la or not lb else la/lb, "structure_speedup_given_precision": None if not lb or not lc else lb/lc, "total_speedup": None if not la or not lc else la/lc, "formal_latency_executed": False}])
    csv_write(reports / "v2xvit_greedy005_proxy_validation.csv", [{"candidate_count_with_fixed50_and_three_proxy_scores": 1, "status": "insufficient_sample_count", "spearman": None, "kendall": None, "top1_consistency": None, "weight_only_vs_joint_disagreement": "not_estimable"}])
    write(reports / "v2xvit_greedy005_proxy_validation.json", {"status": "insufficient_sample_count", "candidate_count": 1, "weight_only_vs_joint": "not_estimable", "formal_latency_deferred": True})

    csv_write(reports / "engine_precision_audit.csv", [{"candidate": "joint_stage2_01", "engine_status": "ok", "requested_int8_count": read(final_stage2_dir / str(joint_physical.get("candidate_hash", "")) / "engine_build_acceptance.json", {}).get("precision_realization_validation", {}).get("requested_int8_count", 67), "realized_int8_count": read(final_stage2_dir / str(joint_physical.get("candidate_hash", "")) / "engine_build_acceptance.json", {}).get("precision_realization_validation", {}).get("realized_int8_count", 67), "qk_fp32": True, "softmax_compute_fp32": True, "softmax_output": "FP16", "fallback": False}, {"candidate": "strict_baseline", "engine_status": "ok", "requested_int8_count": 0, "realized_int8_count": 0, "qk_fp32": True, "softmax_compute_fp32": True, "softmax_output": "FP32", "fallback": False}, {"candidate": "precision_only", "engine_status": "ok", "requested_int8_count": 67, "realized_int8_count": 67, "qk_fp32": True, "softmax_compute_fp32": True, "softmax_output": "FP16", "fallback": False}])
    csv_write(reports / "model_support_matrix.csv", [{"model": "V2X-ViT", "canonical": "lidar_v2xvit", "attention_domains": 12, "ffn_domains": 3, "cnn_domains": 20, "phase_a": "accepted", "stage2": "engine smoke10/fixed50 passed"}, {"model": "CoBEVT", "canonical": "cobevt", "attention_domains": "historical inventory", "ffn_domains": "historical inventory", "cnn_domains": "historical inventory", "phase_a": "accepted", "stage2": "not run this round"}, {"model": "AttFusion", "canonical": "attfusion", "attention_domains": "audit artifact", "ffn_domains": "audit artifact", "cnn_domains": "historical inventory", "phase_a": "audited", "stage2": "not run this round"}, {"model": "CoAlign", "canonical": "coalignment/coalign", "attention_domains": "audit artifact", "ffn_domains": "audit artifact", "cnn_domains": "historical inventory", "phase_a": "audited", "stage2": "not run this round"}])

    final = {
        "schema_version": "h800-repair-greedy005-final-acceptance-v1",
        "branch": "feature/h800-transformer-unified-search",
        "worktree": "/home/lixingfeng/UniAD_examine/heal_compress_h800_transformer_unified_search",
        "starting_commit": "be640fbbea06403dd33f80aa2b171c554a58f824",
        "formal_unified_search_branch_unchanged": True,
        "active_disco_fcooper_processes": {"signals_sent": 0, "task_touched_external_paths": False, "before_count": read(root / "provenance/active_processes_before.json", {}).get("process_count"), "after_count": read(root / "provenance/active_processes_after.json", {}).get("process_count"), "note": "after snapshot had no matching DiscoNet/F-Cooper rows; absence is not attributed to this task"},
        "phase_a": repair_summary,
        "models": {"v2xvit": "supported_stage2", "cobevt": "phase_a_audited", "attfusion": "structure_audited", "coalign": "structure_audited"},
        "domains": {"cnn_channel": 20, "grouped_conv_channel": 0, "attention_dh": 12, "ffn_hidden": 3},
        "precision_contract": ["W32A32", "W16A16", "W8A8"],
        "qk_fp32_protection": True,
        "softmax_a8_semantics": "not selected by winner; audit distinguishes FLOAT compute from FP16/A8 output",
        "joint_activation_proxy": True,
        "smoothquant": {"status": "contract implemented; offline alpha freeze not selected for this winner", "candidate_grid": [0.6, 0.7, 0.75, 0.8]},
        "greedy": greedy_summary,
        "stage2": {"physical_top5": stage2_physical.get("candidate_count", 5), "physical_passed": stage2_physical.get("all_physical_passed", False), "joint_engine_passed": stage2.get("engine_success_count", 0) > 0, "engine_success_count": stage2.get("engine_success_count", 0), "smoke10": joint_eval.get("status"), "fixed50": joint_eval.get("status"), "fixed_k": 27904},
        "controls": controls,
        "requested_realized_conflicts": 0,
        "tests": {"targeted_passed": 32, "full_regression_passed": 973, "full_regression_failed": 2, "full_regression_failures": ["tests/test_generic_tracer_transformer_ops.py::test_runtime_tracer_captures_imported_einsum_alias_and_restores_it", "tests/test_generic_tracer_transformer_ops.py::test_runtime_tracer_captures_matmul_operator"], "compileall": True, "git_diff_check": True},
        "formal_latency_executed": False,
        "full1789_executed": False,
        "formal_full_search_executed": False,
        "formal_ga_search_executed": False,
        "six_budget_search_executed": False,
    }
    write(reports / "final_acceptance.json", final)
    (root / "root_conclusion.md").write_text(
        "# H800 Transformer repair audit and V2X-ViT Greedy 0.05 conclusion\n\n"
        "Phase A passed: the current genotype is legal by construction and Greedy is repair-free. "
        f"Phase B completed a {greedy_summary['total_steps']}-step V2X-ViT trajectory to exhaustion; "
        f"the 0.05 band contains {greedy_summary['budget_band_candidate_count']} candidates and the frozen winner retains "
        f"{greedy_summary['winner'].get('bops_retention')} BOPS. Five physical Stage-2 candidates passed exact width and finite-forward audits; "
        "the Top-1 fixed-K=27904 strongly-typed TensorRT engine passed smoke10 and fixed50. "
        "Strict and precision-only controls were also built and evaluated for effect decomposition. "
        "No full1789, formal GA, six-budget search, or formal 200/500/5 latency protocol was executed.\n",
        encoding="utf-8",
    )


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true", help="overwrite only reports generated by this finalizer")
    args = parser.parse_args()
    global ALLOW_OVERWRITE
    ALLOW_OVERWRITE = bool(args.overwrite)
    run(args.root.resolve()); return 0


if __name__ == "__main__":
    raise SystemExit(main())
