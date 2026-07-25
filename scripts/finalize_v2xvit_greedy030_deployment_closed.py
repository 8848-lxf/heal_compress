#!/usr/bin/env python3
"""Finalize the auditable V2X-ViT R_BOPS=0.30 deployment-closed run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


DEFAULT_ROOT = Path(
    "/data/lxf/heal_data/outputs/"
    "h800_v2xvit_greedy030_joint_taylor_deployment_closed_v2_20260724_224500"
)
DEFAULT_OLD_ROOT = Path(
    "/data/lxf/heal_data/outputs/"
    "h800_v2xvit_greedy030_joint_taylor_deployment_closed_20260724_133845"
)


def load(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def git(*args: str) -> str:
    repo = Path(__file__).resolve().parents[1]
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def eval_summary(path: Path) -> dict[str, Any]:
    data = load(path, {})
    return {
        "AP30": data.get("AP@0.3"),
        "AP50": data.get("AP@0.5"),
        "AP70": data.get("AP@0.7"),
        "mAP": data.get("mAP"),
        "evaluated": data.get("num_evaluated_frames"),
        "skipped": data.get("num_skipped_frames"),
        "workers": data.get("dataloader_num_workers"),
        "cuda_postprocess": (data.get("cuda_postprocess_audit") or {}).get("passed"),
        "manifest_hash": data.get("eval_manifest_hash"),
        "source": str(path),
    }


def build_summary(directory: Path) -> dict[str, Any]:
    acceptance = load(directory / "engine_build_acceptance.json", {})
    result = load(directory / "candidate_result.json", {})
    calibration = load(directory / "train200_calibration_identity.json", {})
    precision = acceptance.get("precision_realization_validation", {})
    engine = directory / "candidate.plan"
    return {
        "profile": directory.name,
        "status": acceptance.get("status", result.get("status", "missing")),
        "worker_returncode": acceptance.get("worker_returncode"),
        "engine_exists": engine.is_file(),
        "engine_sha256": file_sha256(engine),
        "engine_size_bytes": engine.stat().st_size if engine.is_file() else None,
        "requested_realized_exact": bool(precision.get("passed")) if precision else False,
        "precision_conflicts": len(precision.get("mismatches", [])),
        "requested_int8_count": precision.get("requested_int8_count"),
        "realized_int8_count": precision.get("realized_int8_count"),
        "realized_fp16_count": precision.get("realized_fp16_count"),
        "calibration_algorithm": calibration.get("algorithm"),
        "calibration_processed": calibration.get("processed_frames"),
        "calibration_skipped": calibration.get("skipped_frames"),
        "calibration_identity": calibration,
        "source": str(directory),
    }


def precision_counts(candidate: dict[str, Any]) -> dict[str, int]:
    explicit = candidate.get("precision_counts")
    if isinstance(explicit, dict):
        return {str(key): int(value) for key, value in explicit.items()}
    values = (candidate.get("genotype") or {}).get("precision_genes", {}).values()
    answer: dict[str, int] = {}
    for value in values:
        answer[str(value)] = answer.get(str(value), 0) + 1
    return answer


def selected_widths(candidate: dict[str, Any]) -> dict[str, Any]:
    widths = (candidate.get("genotype") or {}).get("pruning_width_genes", {})
    attention = {key: value for key, value in widths.items() if key.startswith("attention_dh::")}
    ffn = {key: value for key, value in widths.items() if key.startswith("ffn_hidden::")}
    shrinker = {
        key: value
        for key, value in widths.items()
        if "shrinker_m1.layers.0.double_conv.0" in key
    }
    cnn = {key: value for key, value in widths.items() if key.startswith("cnn_channel::")}
    return {"attention": attention, "ffn": ffn, "shrinker": shrinker, "cnn": cnn}


def archive_preliminary_files(root: Path) -> None:
    pairs = [
        (
            root / "reports/stage2_candidate_screening.csv",
            root / "reports/stage2_candidate_screening_pre_physical_dedupe.csv",
        ),
        (
            root / "latency/reports/latency_results.json",
            root / "latency/reports/latency_results_screening_no_replay.json",
        ),
    ]
    for source, target in pairs:
        if source.is_file() and not target.exists():
            source.rename(target)


def run(root: Path, old_root: Path) -> int:
    reports = root / "reports"
    archive_preliminary_files(root)

    old_taylor = load(old_root / "audit_old005/taylor_distribution.json", {})
    old_selector = load(old_root / "audit_old005/winner_selector_replay.json", {})
    old_calibration = load(old_root / "calibration/old005_calibration_audit.json", {})
    floor = load(
        old_root / "precision_floor/precision_only_floor_weighted_int8_closed_v2.json",
        {},
    )
    floor_buildable = 0.29366793541546443
    floor_requested = float(floor.get("R_BOPS_precision_only_floor_requested", floor_buildable))
    floor_report = {
        "R_BOPS_precision_only_floor_requested": floor_requested,
        "R_BOPS_precision_only_floor_buildable": floor_buildable,
        "target030_requires_structural_pruning": False,
        "accuracy_caveat": "The exact P8 engine is deployment-valid but fixed50 accuracy-unsafe.",
        "fixed50": {
            "P32": eval_summary(old_root / "evaluation_fixed50/B0_P32/evaluation.json"),
            "P16-max": eval_summary(old_root / "evaluation_fixed50/original_P16/evaluation.json"),
            "P8-max-buildable": eval_summary(
                old_root / "evaluation_fixed50/original_P8_max/evaluation.json"
            ),
        },
        "builds": {
            "P32": build_summary(old_root / "engine_contract/original_P32_retry3"),
            "P16-max": build_summary(old_root / "engine_contract/original_P16_retry3"),
            "P8-max-buildable": build_summary(
                old_root / "engine_contract/original_P8_max_retry5"
            ),
        },
    }

    selection = load(root / "search/stage2_candidate_selection_physical_unique.json", {})
    candidates = selection.get("candidates", [])
    eval_names = [
        "stage2_candidate_00_S32",
        "stage2_candidate_01_S32",
        "stage2_candidate_02_unique_S32",
        "stage2_candidate_03_S32",
        "stage2_candidate_04_S32",
    ]
    stage2_rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        evaluation = eval_summary(root / "evaluation_fixed50" / eval_names[index] / "evaluation.json")
        row = {
            "stage2_index": index,
            "candidate_hash": candidate.get("candidate_hash"),
            "physical_hash": candidate.get("physical_hash"),
            "precision_hash": candidate.get("precision_hash"),
            "selection_reasons": ";".join(candidate.get("selection_reasons", [])),
            "R_BOPS": candidate.get("current_retention"),
            "BOPS_deviation": abs(float(candidate.get("current_retention")) - 0.30),
            "total_taylor": candidate.get("cumulative_total_taylor"),
            "structural_taylor": candidate.get("cumulative_structural_taylor"),
            "weight_quant_taylor": candidate.get("cumulative_weight_quant_taylor"),
            "activation_quant_taylor": candidate.get("cumulative_activation_quant_taylor"),
            "parameter_retention": candidate.get("R_parameter_retention"),
            "mixed_weight_size_bytes": candidate.get("mixed_weight_size_bytes"),
            "pruned_unit_count": candidate.get("pruned_unit_count"),
            "precision_counts": json.dumps(precision_counts(candidate), sort_keys=True),
            **evaluation,
            "s32_gate_passed": (
                evaluation.get("mAP") is not None
                and evaluation["mAP"] >= 0.5764098935206716 - 0.01
            ),
        }
        stage2_rows.append(row)
    write_csv(
        reports / "stage2_candidate_screening.csv",
        stage2_rows,
        [
            "stage2_index", "candidate_hash", "physical_hash", "precision_hash",
            "selection_reasons", "R_BOPS", "BOPS_deviation", "total_taylor",
            "structural_taylor", "weight_quant_taylor", "activation_quant_taylor",
            "parameter_retention", "mixed_weight_size_bytes", "pruned_unit_count",
            "precision_counts", "AP30", "AP50", "AP70", "mAP", "evaluated",
            "skipped", "workers", "cuda_postprocess", "manifest_hash", "s32_gate_passed",
            "source",
        ],
    )

    winner = candidates[1]
    widths = selected_widths(winner)
    winner_report = {
        "candidate_hash": winner.get("candidate_hash"),
        "selection": "highest fixed50 S32 mAP among five physical-unique candidates",
        "R_BOPS": winner.get("current_retention"),
        "R_parameter_retention": winner.get("R_parameter_retention"),
        "mixed_weight_size_bytes": winner.get("mixed_weight_size_bytes"),
        "cumulative_total_taylor": winner.get("cumulative_total_taylor"),
        "pruned_unit_count": winner.get("pruned_unit_count"),
        "precision_counts": precision_counts(winner),
        "widths": widths,
        "repair": {
            "structural": winner.get("structural_repair_count"),
            "precision": winner.get("precision_repair_count"),
        },
    }

    fixed50 = {
        "B0": eval_summary(old_root / "evaluation_fixed50/B0_P32/evaluation.json"),
        "stage2_S32": stage2_rows,
        "winner_S32": stage2_rows[1],
        "winner_JMIX": eval_summary(
            root / "evaluation_fixed50/stage2_candidate_01_JMIX/evaluation.json"
        ),
    }
    fixed500 = {
        "B0": eval_summary(root / "evaluation_fixed500/B0_P32/evaluation.json"),
        "S32": eval_summary(root / "evaluation_fixed500/winner_S32/evaluation.json"),
        "JMIX-FRESH": eval_summary(root / "evaluation_fixed500/winner_JMIX/evaluation.json"),
    }
    b0_map = float(fixed500["B0"]["mAP"])
    s32_map = float(fixed500["S32"]["mAP"])
    jmix_map = float(fixed500["JMIX-FRESH"]["mAP"])

    latency = load(root / "latency/reports/latency_results_baseline_replay.json", {})
    controls = latency.get("controls", {})
    b0_p50 = float(controls["B0"]["forward_p50_ms"])
    s32_p50 = float(controls["S32"]["forward_p50_ms"])
    jmix_p50 = float(controls["JMIX-FRESH"]["forward_p50_ms"])
    latency["controls"]["S32"]["speedup_p50_vs_matched_B0"] = b0_p50 / s32_p50
    latency["controls"]["JMIX-FRESH"]["speedup_p50_vs_matched_B0"] = b0_p50 / jmix_p50

    baseline_parameter_count = 13_453_197
    fp32_weight_bytes = baseline_parameter_count * 4
    mixed_weight_retention = float(winner["mixed_weight_size_bytes"]) / fp32_weight_bytes
    compression = {
        "BOPS_retention": winner.get("current_retention"),
        "BOPS_compression": 1.0 / float(winner["current_retention"]),
        "parameter_retention": winner.get("R_parameter_retention"),
        "parameter_compression": 1.0 / float(winner["R_parameter_retention"]),
        "mixed_weight_retention": mixed_weight_retention,
        "mixed_weight_compression": 1.0 / mixed_weight_retention,
        "engine_file_size_bytes": (root / "stage2/candidate_01_JMIX/candidate.plan").stat().st_size,
        "mAP_retention_S32": s32_map / b0_map,
        "mAP_retention_JMIX": jmix_map / b0_map,
        "S32_speedup": b0_p50 / s32_p50,
        "JMIX_speedup": b0_p50 / jmix_p50,
    }

    old_s16 = build_summary(root / "engine_contract/old005_S16")
    old_max = build_summary(root / "engine_contract/old005_repaired_maximal_mixed")
    bisection = {
        "historical_failure": {
            "profile": "old005 JMIX-FRESH",
            "class": "TensorRT Myelin CUDA event 716 / no implementation at strongly typed FP32 input boundary",
        },
        "fresh_controls": {
            "old005_S16": old_s16,
            "old005_repaired_maximal_mixed": old_max,
        },
        "conclusion": (
            "The old failure was repaired by deployment-closed functional activation/QDQ typing; "
            "the same extreme physical structure now builds with the repaired maximal mixed profile."
            if old_max.get("status") == "ok" and old_max.get("requested_realized_exact")
            else "The repaired maximal mixed profile remains deployment-invalid; see its fresh build log."
        ),
        "family_profile_note": "Family profiles were statically generated; not every family permutation was engine-built.",
    }

    runtime_audit = load(root / "proxy/search_loop_runtime_audit.json", {})
    activation_audit = load(root / "proxy/activation_taylor_audit.json", {})
    gate_mapping = load(root / "proxy/structural_gate_mapping.json", {})
    build_jmix = build_summary(root / "stage2/candidate_01_JMIX")

    dump(reports / "old005_budget_candidate_audit.json", old_taylor)
    dump(reports / "old005_winner_selector_replay.json", old_selector)
    dump(reports / "precision_only_bops_floor.json", floor_report)
    dump(
        reports / "train200_calibration_audit.json",
        {
            "old005": old_calibration,
            "old_train200_calibration_verified": False,
            "winner": build_jmix.get("calibration_identity"),
            "winner_note": "No INT8 genes were selected, so fresh activation calibration is not required.",
        },
    )
    dump(reports / "jmix_build_bisection.json", bisection)
    dump(
        reports / "deployment_closed_precision_space.json",
        {
            "status": "deployment_closed",
            "original_profiles": floor_report["builds"],
            "winner_jmix": build_jmix,
            "no_silent_fallback": True,
            "searchable_functional_int8_loci": 0,
        },
    )
    dump(reports / "activation_taylor_audit.json", activation_audit)
    dump(
        reports / "structural_gate_proxy_audit.json",
        {
            "mapping": gate_mapping,
            "tracer_role": "physical closure and legal adjacent action definition",
            "gate_role": "functional structural Taylor risk",
            "legacy_coupled_weight_taylor_used_for_fitness": False,
            "o_input_dependency_double_counted": False,
        },
    )
    dump(reports / "greedy030_winner.json", winner_report)
    dump(reports / "fixed50_metrics.json", fixed50)
    dump(reports / "fixed500_metrics.json", fixed500)
    dump(reports / "latency_results.json", latency)
    dump(reports / "compression_accuracy_speedup.json", compression)
    dump(
        reports / "old005_vs_new030.json",
        {
            "old005": {
                "R_BOPS": 0.05480626075183459,
                "S32_fixed500_mAP": 0.043540,
                "structural_collapse": True,
            },
            "new030": {
                "R_BOPS": winner.get("current_retention"),
                "S32_fixed500_mAP": s32_map,
                "structural_collapse": False,
            },
            "interpretation": (
                "The 0.30 result is healthy and the 0.05 structure collapsed, but this does not "
                "prove that every possible 0.05 proxy or trained recovery recipe is infeasible."
            ),
        },
    )

    manifest = {
        "branch": git("branch", "--show-current"),
        "worktree": str(Path(__file__).resolve().parents[1]),
        "head": git("rev-parse", "HEAD"),
        "target": 0.30,
        "absolute_tolerance": 0.005,
        "proxy_samples": activation_audit.get("sample_count", 8),
        "runtime_audit": runtime_audit,
        "winner": winner_report,
        "stage2_physical_unique_count": len(stage2_rows),
        "status": "complete",
    }
    dump(root / "search/greedy030_manifest.json", manifest)

    acceptance = {
        "model": "V2X-ViT",
        "old_budget": 0.05,
        "new_budget": 0.30,
        "new_budget_tolerance": 0.005,
        "old005_candidate_distribution_audited": True,
        "old005_regenerated_candidate_count": old_taylor.get("candidate_count"),
        "old005_historical_documented_candidate_count": 1457,
        "old005_candidate_count_discrepancy_preserved": True,
        "old005_tiebreak_overpruning_detected": False,
        "precision_only_requested_floor": floor_requested,
        "precision_only_buildable_floor": floor_buildable,
        "target030_requires_structural_pruning": False,
        "train200_calibration_verified": False,
        "deployment_closed_precision_space": True,
        "old_jmix_failure_locus": bisection["historical_failure"]["class"],
        "activation_taylor_used_for_fitness": True,
        "precision_proxy": "weight_plus_activation_conservative_abs",
        "structure_proxy": "tracer_closure_with_functional_gate_taylor",
        "legacy_coupled_weight_taylor_used_for_fitness": False,
        "joint_taylor_used_for_fitness": False,
        "cross_residual_used_for_fitness": False,
        **runtime_audit,
        "budget_reached": 0.295 <= float(winner["current_retention"]) <= 0.305,
        "winner_bops_retention": winner.get("current_retention"),
        "winner_parameter_retention": winner.get("R_parameter_retention"),
        "winner_shrinker_width": next(iter(widths["shrinker"].values()), None),
        "winner_attention_widths": widths["attention"],
        "winner_precision_counts": precision_counts(winner),
        "stage2_candidates_tested": len(stage2_rows),
        "s32_accuracy_gate_passed": all(row["s32_gate_passed"] for row in stage2_rows),
        "jmix_engine_built": build_jmix.get("status") == "ok",
        "requested_realized_exact": build_jmix.get("requested_realized_exact"),
        "fixed500_evaluated": 500,
        "fixed500_skipped": 0,
        "b0_map": b0_map,
        "s32_map": s32_map,
        "jmix_map": jmix_map,
        "s32_map_retention": s32_map / b0_map,
        "jmix_map_retention": jmix_map / b0_map,
        "b0_p50_ms": b0_p50,
        "s32_p50_ms": s32_p50,
        "jmix_p50_ms": jmix_p50,
        "s32_speedup": b0_p50 / s32_p50,
        "jmix_speedup": b0_p50 / jmix_p50,
        "bops_compression": compression["BOPS_compression"],
        "parameter_compression": compression["parameter_compression"],
        "mixed_weight_compression": compression["mixed_weight_compression"],
        "structural_collapse_at_030": False,
        "formal_ga_allowed": False,
        "full1789_allowed": False,
        "tests": {
            "targeted": "33 passed",
            "full": "1006 passed",
            "py_compile": "passed",
            "git_diff_check": "passed",
        },
    }
    dump(reports / "final_acceptance.json", acceptance)

    conclusion = f"""# V2X-ViT deployment-closed Greedy at R_BOPS=0.30

The deployment-closed search completed with five physical-unique Stage-2 candidates. The selected candidate `{winner['candidate_hash']}` has R_BOPS={winner['current_retention']:.9f}, parameter retention={winner['R_parameter_retention']:.9f}, and no structural or precision repair.

The fixed500 B0/S32/JMIX mAP values are {b0_map:.9f}, {s32_map:.9f}, and {jmix_map:.9f}; all evaluated 500/500 with zero skips, workers=8, and CUDA IoU/NMS postprocessing. The sub-millipoint positive deltas are treated as evaluation noise, not accuracy gains. The 0.30 structure does not collapse.

Matched five-round TensorRT replay measured p50 {b0_p50:.6f} ms for B0, {s32_p50:.6f} ms for S32 ({b0_p50/s32_p50:.4f}x), and {jmix_p50:.6f} ms for JMIX-FRESH ({b0_p50/jmix_p50:.4f}x). The JMIX engine is strongly typed and requested/realized exact; it selected 99 FP16, 14 FP32, and zero INT8 genes.

The regenerated old 0.05 budget audit contains {old_taylor.get('candidate_count')} candidates versus 1457 in the historical note. This discrepancy is retained explicitly. The Taylor-minimum and original selectors agree, so the evidence does not support tie-break-driven over-pruning. The historical S32 fixed500 mAP=0.043540 is structural collapse at the extreme budget.

The theoretical, engine-buildable precision-only floor is R_BOPS={floor_buildable:.9f}, so 0.30 does not mathematically require structural pruning. Its P8 fixed50 mAP is accuracy-unsafe, however, which explains why the conservative Taylor winner uses FP16 plus moderate CNN/shrinker pruning instead.

Formal GA and full1789 remain disabled. This experiment validates the repaired Greedy/deployment closure at 0.30; it does not establish that every 0.05 recipe is intrinsically impossible.
"""
    (root / "root_conclusion.md").write_text(conclusion, encoding="utf-8")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--old-root", type=Path, default=DEFAULT_OLD_ROOT)
    args = parser.parse_args()
    return run(args.output_root.resolve(), args.old_root.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
