#!/usr/bin/env python3
"""Reconcile repeat-3 AP/latency with exact V2X-ViT resource proxies."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.audit_heal_transformer_search_models import _load
from scripts.run_v2xvit_greedy005_full import _baseline_candidate, _build_full_space
from scripts.smoke_transformer_unified_search import _multi_agent_validation_batch
from search.candidate import CandidateGenotype
from search.canonicalization import canonicalize_candidate


LABELS = ("030", "025", "020", "015", "010", "005")


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _source_root(label: str, main: Path, frozen005: Path) -> Path:
    return frozen005 if label == "005" else main


def _phenotype(
    genotype: Mapping[str, Any], space: Any
) -> Any:
    return canonicalize_candidate(CandidateGenotype.from_dict(dict(genotype)), space)


def _resource(
    phenotype: Any, built: Mapping[str, Any]
) -> dict[str, float]:
    bops = built["bops"].evaluate_breakdown(phenotype)
    size = built["size"].evaluate_breakdown(phenotype)
    return {
        "R_BOPS": float(bops["R_bops_vs_fp32"]),
        "BOPS_compression": 1.0 / float(bops["R_bops_vs_fp32"]),
        "proxy_parameter_count": float(size["parameter_count_after"]),
        "proxy_parameter_retention": float(size["R_parameter_retention"]),
        "proxy_parameter_prune_rate": float(size["parameter_pruning_rate"]),
        "proxy_parameter_compression": 1.0 / float(size["R_parameter_retention"]),
        "mixed_weight_retention": float(size["R_size_vs_fp32"]),
        "mixed_weight_compression": 1.0 / float(size["R_size_vs_fp32"]),
        "mixed_weight_size_bytes": float(size["size_bits_total"]) / 8.0,
    }


def _precision_counts(genotype: Mapping[str, Any]) -> dict[str, int]:
    values = tuple(dict(genotype.get("precision_genes") or {}).values())
    return {
        state: sum(str(value) == state for value in values)
        for state in ("FP32", "FP16", "INT8")
    }


def _physical_count(root: Path, label: str, candidate_hash: str) -> int:
    report = _read(
        root
        / f"ga/stage2_cache/budget_{label}/{candidate_hash}/physical_report.json"
    )
    return int(report["physical_parameter_count"])


def _ap_fields(summary: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, float]:
    result = {
        "AP30_mean": float(summary["AP@0.3_mean"]),
        "AP30_std": float(summary["AP@0.3_std"]),
        "AP50_mean": float(summary["AP@0.5_mean"]),
        "AP50_std": float(summary["AP@0.5_std"]),
        "AP70_mean": float(summary["AP@0.7_mean"]),
        "AP70_std": float(summary["AP@0.7_std"]),
        "mAP_mean": float(summary["mAP_mean"]),
        "mAP_std": float(summary["mAP_std"]),
        "forward_p50_ms_mean": float(summary["forward_p50_ms_mean"]),
        "forward_p50_ms_std": float(summary["forward_p50_ms_std"]),
    }
    for label, key in (("AP30", "AP@0.3"), ("AP50", "AP@0.5"),
                       ("AP70", "AP@0.7"), ("mAP", "mAP")):
        base = float(baseline[f"{key}_mean"])
        value = float(summary[f"{key}_mean"])
        result[f"{label}_retention"] = value / base
        result[f"{label}_drop"] = base - value
    result["speedup_fullval_p50"] = (
        float(baseline["forward_p50_ms_mean"])
        / float(summary["forward_p50_ms_mean"])
    )
    result["FPS"] = 1000.0 / float(summary["forward_p50_ms_mean"])
    return result


def run(args: argparse.Namespace) -> int:
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"resource_reconciliation_requires_one_visible_gpu:"
            f"{torch.cuda.device_count()}"
        )
    device = torch.device("cuda:0")
    model, adapter, hypes, _ = _load("v2xvit", device)
    batch, _, _ = _multi_agent_validation_batch(adapter, hypes, device)
    built = _build_full_space(model, adapter, hypes, batch)
    space = built["space"]
    baseline_genotype = _baseline_candidate(space)
    baseline_phenotype = canonicalize_candidate(baseline_genotype, space)
    baseline_resource = _resource(baseline_phenotype, built)

    main_eval = _read(args.main_summary.resolve())
    main_controls = dict(main_eval["controls"])
    main_summaries = dict(main_eval["summaries"])
    b0_main = main_summaries["B0"]
    b0_engine_size = int(main_controls["B0"]["engine_size_bytes"])
    main_rows: list[dict[str, Any]] = [{
        "experiment": "Greedy_GA",
        "budget": 1.0,
        "method": "strict_FP32",
        "candidate_hash": "B0-strict",
        **_ap_fields(b0_main, b0_main),
        **baseline_resource,
        "physical_parameter_count": 13_453_197,
        "parameter_retention_physical": 1.0,
        "parameter_prune_rate_physical": 0.0,
        "parameter_compression_physical": 1.0,
        "engine_size_bytes": b0_engine_size,
        "engine_file_compression": 1.0,
        "mutable_FP32": len(space.precision_gene_ids),
        "mutable_FP16": 0,
        "mutable_INT8": 0,
        "requested_realized_exact": True,
        "evaluated_per_repeat": 1789,
        "repetitions": 3,
        "skipped_total": 0,
    }]
    for label in LABELS:
        source = _source_root(
            label, args.main_search_root.resolve(), args.frozen005_root.resolve()
        )
        budget = _read(source / f"ga/budget_{label}/seed_0/budget_summary.json")
        for method, key in (("Greedy", "greedy_anchor"), ("GA-final", "final_winner")):
            candidate = budget[key]
            candidate_hash = str(candidate["complete_phenotype_hash"])
            phenotype = _phenotype(candidate["genotype"], space)
            control_id = f"budget_{label}/{method}"
            evaluation = main_summaries[control_id]
            control = main_controls[control_id]
            parameters = _physical_count(source, label, candidate_hash)
            precision = _precision_counts(candidate["genotype"])
            main_rows.append({
                "experiment": "Greedy_GA",
                "budget": int(label) / 100.0,
                "method": method,
                "candidate_hash": candidate_hash,
                **_ap_fields(evaluation, b0_main),
                **_resource(phenotype, built),
                "physical_parameter_count": parameters,
                "parameter_retention_physical": parameters / 13_453_197,
                "parameter_prune_rate_physical": 1.0 - parameters / 13_453_197,
                "parameter_compression_physical": 13_453_197 / parameters,
                "engine_size_bytes": int(control["engine_size_bytes"]),
                "engine_file_compression": (
                    b0_engine_size / int(control["engine_size_bytes"])
                ),
                "mutable_FP32": precision["FP32"],
                "mutable_FP16": precision["FP16"],
                "mutable_INT8": precision["INT8"],
                "requested_realized_exact": bool(control["requested_realized_exact"]),
                "evaluated_per_repeat": 1789,
                "repetitions": 3,
                "skipped_total": 0,
            })

    pq_eval = _read(args.pq_summary.resolve())
    pq_b0 = _read(args.pq_b0_summary.resolve())
    pq_build = _read(args.pq_build_summary.resolve())
    pq_b0_normalized = {
        key: value for key, value in pq_b0.items()
        if key.endswith("_mean") or key.endswith("_std")
    }
    pq_rows: list[dict[str, Any]] = [{
        "experiment": "P_Q_only",
        "budget": 1.0,
        "method": "strict_FP32",
        "candidate_hash": "B0-strict",
        **_ap_fields(pq_b0_normalized, pq_b0_normalized),
        **baseline_resource,
        "physical_parameter_count": 13_453_197,
        "parameter_retention_physical": 1.0,
        "parameter_prune_rate_physical": 0.0,
        "parameter_compression_physical": 1.0,
        "engine_size_bytes": b0_engine_size,
        "engine_file_compression": 1.0,
        "mutable_FP32": len(space.precision_gene_ids),
        "mutable_FP16": 0,
        "mutable_INT8": 0,
        "requested_realized_exact": True,
        "evaluated_per_repeat": 1789,
        "repetitions": 3,
        "skipped_total": 0,
    }]
    for label in LABELS:
        source = _source_root(
            label, args.main_search_root.resolve(), args.frozen005_root.resolve()
        )
        budget = _read(source / f"ga/budget_{label}/seed_0/budget_summary.json")
        candidate = budget["final_winner"]
        source_hash = str(candidate["complete_phenotype_hash"])
        ga = CandidateGenotype.from_dict(candidate["genotype"])
        controls = {
            "P-only": CandidateGenotype(
                pruning_width_genes=dict(ga.pruning_width_genes),
                precision_genes=dict(baseline_genotype.precision_genes),
                meta={"diagnostic_control": "P-only"},
            ),
            "Q-only": CandidateGenotype(
                pruning_width_genes=dict(baseline_genotype.pruning_width_genes),
                precision_genes=dict(ga.precision_genes),
                meta={"diagnostic_control": "Q-only"},
            ),
        }
        physical = _physical_count(source, label, source_hash)
        for method, genotype in controls.items():
            control_id = f"budget_{label}/{method}"
            control = pq_build["controls"][control_id]
            summary = pq_eval["summaries"][control_id]
            phenotype = canonicalize_candidate(genotype, space)
            parameters = physical if method == "P-only" else 13_453_197
            precision = _precision_counts(genotype.to_dict())
            engine_size = int(control["engine_size_bytes"])
            pq_rows.append({
                "experiment": "P_Q_only",
                "budget": int(label) / 100.0,
                "method": method,
                "candidate_hash": str(control.get("candidate_hash") or source_hash),
                "source_ga_candidate_hash": source_hash,
                **_ap_fields(summary, pq_b0_normalized),
                **_resource(phenotype, built),
                "physical_parameter_count": parameters,
                "parameter_retention_physical": parameters / 13_453_197,
                "parameter_prune_rate_physical": 1.0 - parameters / 13_453_197,
                "parameter_compression_physical": 13_453_197 / parameters,
                "engine_size_bytes": engine_size,
                "engine_file_compression": b0_engine_size / engine_size,
                "mutable_FP32": precision["FP32"],
                "mutable_FP16": precision["FP16"],
                "mutable_INT8": precision["INT8"],
                "requested_realized_exact": bool(control["requested_realized_exact"]),
                "evaluated_per_repeat": 1789,
                "repetitions": 3,
                "skipped_total": 0,
            })

    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    all_rows = [*main_rows, *pq_rows]
    fields = sorted({str(key) for row in all_rows for key in row})
    with (output / "v2xvit_complete_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)
    _write(output / "v2xvit_complete_metrics.json", {
        "status": "complete",
        "model": "V2X-ViT",
        "validation_frames_per_repeat": 1789,
        "repetitions": 3,
        "main_rows": main_rows,
        "pq_only_rows": pq_rows,
        "resource_proxy_schema": "unified-bops-v2-hgt-relation-closure",
        "same_gpu_baseline_for_main": True,
        "same_gpu_baseline_for_pq_only": True,
    })
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-search-root", type=Path, required=True)
    parser.add_argument("--frozen005-root", type=Path, required=True)
    parser.add_argument("--main-summary", type=Path, required=True)
    parser.add_argument("--pq-summary", type=Path, required=True)
    parser.add_argument("--pq-build-summary", type=Path, required=True)
    parser.add_argument("--pq-b0-summary", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
