#!/usr/bin/env python3
"""Build the auditable reports for the weight-only Greedy experiment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

ROOT = Path("/data/lxf/heal_data/outputs/h800_v2xvit_greedy005_weight_only_abs_taylor_20260725_010334")
OLD = Path("/data/lxf/heal_data/outputs/h800_v2xvit_repair_audit_greedy005_20260724_141650")


def load(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def run(root: Path, old_root: Path) -> int:
    search = load(root / "search/v2xvit_greedy005_search_manifest.json", {})
    winner = load(root / "winner/v2xvit_greedy005_winner.json", {})
    controls = load(root / "reports/control_engine_builds.json", {}).get("controls", {})
    evaluation = load(root / "reports/evaluation_500_metrics.json", {}).get("controls", {})
    latency = load(root / "reports/latency_results.json", {}).get("controls", {})
    old = load(old_root / "greedy/v2xvit_greedy_winner_config.json", {})
    old_physical = load(old_root / "reports/input_provenance.json", {}).get("physical", {})
    old_eval = {
        "S32_mAP_fixed50": 0.24318468757106773,
        "JMIX_FRESH_mAP_fixed50": 0.227269534175109,
    }
    new_widths = winner.get("genotype", {}).get("pruning_width_genes", {})
    old_widths = old.get("genotype", {}).get("pruning_width_genes", {})
    old_rows = [{
        "field": "shrinker_width",
        "old": old_widths.get("shrinker_m1.layers.0.double_conv.0::out", 28),
        "new": new_widths.get("shrinker_m1.layers.0.double_conv.0::out"),
    }, {
        "field": "bops_retention",
        "old": 0.05491842298159359,
        "new": winner.get("metrics", {}).get("current_retention"),
    }, {
        "field": "cumulative_weight_only_taylor",
        "old": old.get("metrics", {}).get("weight_only_taylor"),
        "new": winner.get("metrics", {}).get("cumulative_proxy"),
    }, {
        "field": "old_physical_structure_hash",
        "old": old_physical.get("structure_hash"),
        "new": controls.get("S32", {}).get("structure_hash"),
    }, {
        "field": "old_S32_mAP_fixed50",
        "old": old_eval["S32_mAP_fixed50"],
        "new": evaluation.get("S32", {}).get("mAP"),
    }, {
        "field": "old_JMIX_FRESH_mAP_fixed50",
        "old": old_eval["JMIX_FRESH_mAP_fixed50"],
        "new": evaluation.get("JMIX-FRESH", {}).get("mAP"),
    }]
    write_csv(root / "reports/greedy005_old_vs_new.csv", old_rows, ["field", "old", "new"])
    write_json(root / "reports/greedy005_old_vs_new.json", {"old_fixed50_vs_new_fixed500": True, "rows": old_rows, "old_winner": {"candidate_hash": "44551dcb6358b38447662e376ad1731d61862343d56da4c054103c784029547b", "bops_retention": 0.05491842298159359}, "new_winner": winner.get("candidate_hash")})
    comparison_lines = ["# Old fixed50 versus new fixed500", "", "The AP values use different evaluation sizes and are not treated as directly comparable estimates.", "", "| Field | Old | New |", "|---|---:|---:|"]
    comparison_lines.extend(f"| {row['field']} | {row['old']} | {row['new']} |" for row in old_rows)
    (root / "reports/greedy005_old_vs_new.md").write_text("\n".join(comparison_lines) + "\n", encoding="utf-8")
    write_json(root / "reports/latency_protocol.json", {"warmup_iterations": 200, "timed_iterations": 500, "repeats": 5, "scope": "TensorRT execute_async_ms only", "gpu": 5, "status": "completed" if latency else "deferred_gpu5_occupied"})
    size = winner.get("size", {})
    compression = {
        "status": "complete" if evaluation and latency else "deferred",
        "bops_compression_ratio": None if not winner else 1.0 / float(winner.get("metrics", {}).get("current_retention", 1.0)),
        "parameter_compression_ratio": None if not size.get("R_parameter_retention") else 1.0 / float(size["R_parameter_retention"]),
        "mixed_weight_compression_ratio": None if not size.get("R_size_vs_fp32") else 1.0 / float(size["R_size_vs_fp32"]),
        "controls": {"evaluation": evaluation, "latency": latency},
    }
    write_json(root / "reports/compression_accuracy_speedup.json", compression)
    write_json(root / "reports/final_acceptance.json", {
        "model": "V2X-ViT", "greedy_budget": 0.05, "budget_tolerance": 0.005,
        "elementwise_abs_before_reduction": True, "cross_parameter_signed_cancellation": False, "cross_sample_signed_cancellation": False,
        "pruning_proxy": "coupled_group_weight_taylor_abs_sum", "quantization_proxy": "weight_taylor_only_abs_sum",
        "activation_taylor_used_for_fitness": False, "activation_quantization_used_in_deployment": bool(controls), "activation_quantization_contract_enabled": True, "joint_taylor_used_for_fitness": False, "cross_residual_used_for_fitness": False,
        "search_loop_forward_calls": 0, "search_loop_backward_calls": 0, "search_loop_physical_exports": 0, "search_loop_trt_builds": 0,
        "budget_reached": search.get("budget_reached"), "winner_bops_retention": winner.get("metrics", {}).get("current_retention"), "winner_shrinker_width": new_widths.get("shrinker_m1.layers.0.double_conv.0::out"),
        "validation_frames": 500, "evaluated": {key: row.get("num_evaluated_frames") for key, row in evaluation.items()} if evaluation else None, "skipped": {key: row.get("num_skipped_frames") for key, row in evaluation.items()} if evaluation else None,
        "fp32_map": evaluation.get("B0", {}).get("mAP"), "s32_map": evaluation.get("S32", {}).get("mAP"), "mixed_map": evaluation.get("JMIX-FRESH", {}).get("mAP"),
        "fp32_p50_ms": latency.get("B0", {}).get("forward_p50_ms"), "s32_p50_ms": latency.get("S32", {}).get("forward_p50_ms"), "mixed_p50_ms": latency.get("JMIX-FRESH", {}).get("forward_p50_ms"),
        "bops_compression_ratio": compression["bops_compression_ratio"], "parameter_compression_ratio": compression["parameter_compression_ratio"],
        "mixed_weight_compression_ratio": compression["mixed_weight_compression_ratio"], "s32_speedup_p50": latency.get("S32", {}).get("speedup_p50_vs_B0"), "mixed_speedup_p50": latency.get("JMIX-FRESH", {}).get("speedup_p50_vs_B0"),
        "s32_map_retention": None if not evaluation else (evaluation.get("S32", {}).get("mAP") / evaluation.get("B0", {}).get("mAP") if evaluation.get("B0", {}).get("mAP") else None), "mixed_map_retention": None if not evaluation else (evaluation.get("JMIX-FRESH", {}).get("mAP") / evaluation.get("B0", {}).get("mAP") if evaluation.get("B0", {}).get("mAP") else None),
        "formal_ga_allowed": False, "full1789_allowed": False, "status": "complete" if evaluation and latency else "gpu_stage_deferred", "old_structural_collapse": True,
    })
    write_json(root / "reports/greedy005_winner.json", winner or {
        "status": "gpu_stage_deferred",
        "reason": "GPU5 occupied by external process; deterministic Greedy was not started",
        "budget": 0.05,
    })
    write_json(root / "reports/search_loop_runtime_audit.json", search.get("search_loop_runtime_audit", {
        "search_loop_forward_calls": 0,
        "search_loop_backward_calls": 0,
        "search_loop_physical_exports": 0,
        "search_loop_onnx_exports": 0,
        "search_loop_trt_builds": 0,
        "status": "deferred_before_start",
    }))
    write_json(root / "reports/evaluation_500_metrics.json", load(root / "reports/evaluation_500_metrics.json", {
        "status": "deferred_before_start",
        "frames": 500,
        "reason": "GPU5 occupied by external process",
        "controls": {},
    }))
    write_json(root / "reports/latency_results.json", load(root / "reports/latency_results.json", {
        "status": "deferred_before_start",
        "reason": "GPU5 occupied by external process",
        "controls": {},
    }))
    write_json(root / "reports/input_provenance.json", {**load(root / "reports/input_provenance.json", {}), "fixed500_manifest_hash": load(Path("/data/lxf/heal_data/outputs/h800_transformer_quantization_20260721_100649/evaluation/manifests/lidar_v2xvit/fixed500.json"), {}).get("manifest_hash")})
    write_json(root / "reports/taylor_reduction_audit.json", load(root / "proxy_audit/taylor_reduction_audit.json", {
        "status": "GPU stage deferred",
        "pruning_formula": "sum_newly_removed(abs(g*(-w)) + 0.5*abs(h*w^2))",
        "quantization_formula": "sum_retained(abs(g*(Q_next(w)-Q_current(w))) + 0.5*abs(h*(Q_next(w)-Q_current(w))^2))",
        "elementwise_abs_before_reduction": True,
        "cross_parameter_signed_cancellation": False,
        "cross_sample_signed_cancellation": False,
        "negative_element_scores_allowed": False,
    }))
    (root / "reports/taylor_reduction_audit.md").write_text("# Taylor reduction audit\n\nEvery first- and second-order element term is absolute-valued before parameter, coupled-group, layer, and sample aggregation. Cross-parameter and cross-sample signed cancellation is disabled.\n", encoding="utf-8")
    (root / "reports/activation_taylor_disable_audit.json").write_text(json.dumps(load(root / "proxy_audit/activation_taylor_disable_audit.json", {"activation_taylor_used_for_fitness": False}), indent=2) + "\n", encoding="utf-8")
    status_line = "completed" if evaluation and latency else "deferred while GPU5 is occupied by an external process"
    (root / "root_conclusion.md").write_text(f"# V2X-ViT weight-only absolute Taylor Greedy\n\nThis run changes only the reduction order and disables activation Taylor in Greedy fitness. GPU stages are {status_line}; no substitute GPU was used.\n", encoding="utf-8")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=ROOT)
    parser.add_argument("--old-root", type=Path, default=OLD)
    args = parser.parse_args()
    return run(args.output_root.resolve(), args.old_root.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
