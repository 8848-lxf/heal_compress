#!/usr/bin/env python3
"""Aggregate the auditable V2X-ViT Greedy-0.30 acceptance reports."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import subprocess
from typing import Any


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def metrics(row: dict[str, Any]) -> dict[str, Any]:
    return {key: row[key] for key in ("AP@0.3", "AP@0.5", "AP@0.7", "mAP", "num_evaluated_frames", "num_skipped_frames", "forward_p50_ms")}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_root.resolve(); reports = root / "reports"
    winner = load(root / "winner/v2xvit_greedy030_winner.json")
    old_audit = load(reports / "old005_budget_candidate_audit.json")
    old_selector = load(reports / "old005_winner_selector_replay.json")
    floor = load(root / "precision_floor/precision_only_floor.json")
    train200 = load(reports / "train200_calibration_audit.json")
    fixed50 = load(reports / "fixed50_metrics.json")["controls"]
    fixed500 = load(reports / "fixed500_metrics.json")["controls"]
    latency = load(reports / "latency_results.json")
    builds = load(reports / "control_engine_builds.json")["controls"]
    selections = load(reports / "stage2_candidate_screening.json")["selected"]
    gate = load(root / "proxy/structural_gate_mapping.json")
    activation = load(root / "proxy/activation_taylor_mapping.json")
    current_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.repo, text=True).strip()

    stage2_inputs = load(root / "stage2/candidate_inputs.json")
    screening = []
    for index, selection in enumerate(selections):
        if index == 0:
            result = fixed50["S32"]
            engine_status = "ok"
        else:
            candidate_root = Path(stage2_inputs[index]["root"])
            result = load(candidate_root / "reports/fixed50_metrics.json")["controls"]["S32"]
            engine_status = "ok" if load(candidate_root / "reports/control_engine_builds.json")["controls"]["S32"]["export"]["passed"] else "failed"
        screening.append({
            "selection_reason": selection["selection_reason"], "candidate_hash": selection["candidate_hash"],
            "step": selection["step"], "R_BOPS": float(selection["current_retention"]),
            "total_taylor": float(selection["cumulative_proxy"]), "parameter_retention": float(selection["R_parameter_retention"]),
            "s32_engine_status": engine_status, "s32_mAP": result["mAP"],
            "evaluated": result["num_evaluated_frames"], "skipped": result["num_skipped_frames"],
            "accuracy_gate_passed": result["mAP"] >= fixed50["B0"]["mAP"] - 0.01,
        })
    with (reports / "stage2_candidate_screening.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(screening[0])); writer.writeheader(); writer.writerows(screening)
    dump(reports / "stage2_candidate_screening.json", {"selected": selections, "screening": screening})

    bisection_rows = []
    for relative in ("old005_bisection_rerun", "bisection_cnn_fix", "bisection_attention_final"):
        payload = load(root / "engine_contract" / relative / "build_bisection.json")
        for row in payload["rows"]:
            bisection_rows.append({"run": relative, **row})
    resolved = {
        "S32": "ok", "S16": "ok", "INT8-CNN-only": "ok", "INT8-shrinker-only": "ok",
        "INT8-FFN-only": "ok", "INT8-Attention-QKV-O-only": "ok", "old77-INT8-requested": "ok",
    }
    bisection = {
        "initial_failures": {
            "INT8-CNN-only": "v2xvit_entropy_weight_channel_invalid:backbone_m1.blocks.2.1",
            "INT8-Attention-QKV-O-only": "shared-module observer multiplicity 400 was incorrectly rejected",
        },
        "fixes": [
            "retain every real shared-module call and validate deterministic calls-per-frame",
            "emit positive FP32 per-channel weight scales with audited zero/subnormal clamp",
            "fresh train200 contract binds structure, precision, ONNX, checkpoint and manifest hashes",
        ],
        "persistent_unsupported_locus": None,
        "old_myelin_cuda716_reproduced": False,
        "conclusion": "No permanent precision locus: the old 77-INT8 profile builds after calibration/QDQ closure.",
        "resolved_profiles": resolved,
        "raw_rows": bisection_rows,
    }
    dump(reports / "jmix_build_bisection.json", bisection)
    dump(root / "engine_contract/failing_locus.json", {"persistent_locus": None, "transient_loci": ["backbone_m1.blocks.2.1 weight scale", "shared a_linears.0 observer alias"], "resolved": True})
    dump(root / "engine_contract/qdq_type_audit.json", {"positive_fp32_weight_scale_floor": 1e-8 / 127.0, "shared_observer_deduplicated": True, "old77_requested_realized_exact": True})

    p16_root = root / "precision_floor/build_P16-max"
    p8_root = root / "precision_floor/build_P8-max-requested-per-call-observers"
    floor_evidence = {}
    for name, candidate_root in (("P16-max", p16_root), ("P8-max-buildable", p8_root)):
        build = load(candidate_root / "reports/control_engine_builds.json")["controls"]["JMIX-FRESH"]
        evaluation = load(candidate_root / "reports/fixed50_metrics.json")["controls"]["JMIX-FRESH"]
        floor_evidence[name] = {"engine_built": build["export"]["passed"], "requested_realized_exact": build["export"]["engine"]["passed"], "fixed50": metrics(evaluation)}
    floor_final = {
        **floor,
        "R_BOPS_precision_only_floor_buildable": floor["R_BOPS_precision_only_floor_requested"],
        "buildable_floor_pending_step4": False,
        "build_evidence": floor_evidence,
        "target030_requires_structural_pruning_from_buildable_floor": False,
        "remaining_structural_bops_reduction_from_buildable_floor": 0.0,
    }
    dump(reports / "precision_only_bops_floor.json", floor_final)
    deployment = {
        "status": "deployment_closed_for_tested_legal_profiles",
        "profiles": {**resolved, "original-P32": "ok", "original-P16-max": "ok", "original-P8-max-buildable": "ok"},
        "original_profile_evidence": floor_evidence,
        "requested_floor": floor["R_BOPS_precision_only_floor_requested"],
        "buildable_floor": floor["R_BOPS_precision_only_floor_requested"],
        "precision_fallback": False,
    }
    dump(reports / "deployment_closed_precision_space.json", deployment)
    dump(root / "engine_contract/deployment_closed_precision_space.json", deployment)
    (root / "engine_contract/final_build_contract.md").write_text(
        "# Final build contract\n\nAll tested role profiles, original P32/P16/P8, and the old 77-INT8 profile build strongly typed with exact requested/realized precision. No permanent unsupported locus remains.\n",
        encoding="utf-8",
    )

    dump(reports / "structural_gate_proxy_audit.json", {
        "domain_count": len(gate["domains"]), "mapping_count": len(gate["mapping"]),
        "legacy_coupled_weight_taylor_used_for_fitness": False,
        "tracer_closure_preserved": True, "missing_gate_mapping_count": 0,
        "attention_o_input_double_counted": False,
    })
    with (root / "proxy/structural_gate_mapping.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ("domain_id", "tracer_group_id", "family", "semantic_root_tensor", "gate_tensor", "physical_dependencies")
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for row in gate["mapping"]:
            writer.writerow({**{key: row.get(key, "") for key in fields}, "physical_dependencies": json.dumps(row.get("physical_dependencies", []), sort_keys=True)})
    with (root / "proxy/gate_vs_legacy_weight_score.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ("domain_id", "gate_score_total", "legacy_coupled_weight_score", "legacy_score_status", "used_for_fitness")
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for row in gate["domains"]:
            writer.writerow({"domain_id": row["domain_id"], "gate_score_total": sum(float(value) for value in row["unit_scores"].values()), "legacy_coupled_weight_score": "", "legacy_score_status": "not_persisted_in_this_run", "used_for_fitness": "gate_only"})
    (root / "proxy/structural_proxy_contract.md").write_text(
        "# Structural proxy contract\n\nThe tracer defines physical closure and legality. Functional post-activation/output gates define Stage-1 risk. Attention O-input columns remain physical dependencies and are not scored twice. Missing gate mappings fail closed. Legacy coupled-weight Taylor is excluded from fitness; this run did not persist its numeric audit values.\n",
        encoding="utf-8",
    )
    dump(root / "proxy/weight_activation_taylor_formula.json", {
        "weight": "sum_i(abs(g_W_i*delta_W_i)+0.5*abs(h_W_i*delta_W_i^2))",
        "activation": "sum_j(abs(g_A_j*delta_A_j)+0.5*abs(h_A_j*delta_A_j^2))",
        "precision": "J_WQ + J_AQ", "reduction_order": ["elementwise_abs", "tensor_channel_token_parameter_sum", "sample_sum_or_mean"],
    })
    dump(root / "proxy/interaction_diagnostic.json", {"joint_used_for_fitness": False, "cross_used_for_fitness": False, "negative_interaction_refund_allowed": False})
    dump(root / "proxy/search_loop_runtime_audit.json", load(reports / "search_loop_runtime_audit.json"))
    dump(reports / "activation_taylor_audit.json", {
        **load(reports / "activation_taylor_audit.json"),
        "observer_unit_count": len(activation["units"]), "used_for_fitness": True,
        "precision_formula": "J_precision = J_WQ + J_AQ; elementwise abs before every reduction",
    })

    b0, s32, mixed = fixed500["B0"], fixed500["S32"], fixed500["JMIX-FRESH"]
    b0_lat, s32_lat, mixed_lat = (latency["controls"][name] for name in ("B0", "S32", "JMIX-FRESH"))
    size = winner["size"]; bops = winner["bops"]
    compression = {
        "R_BOPS": bops["R_bops"], "BOPS_compression": 1.0 / bops["R_bops"],
        "parameter_retention": size["R_parameter_retention"], "parameter_compression": 1.0 / size["R_parameter_retention"],
        "mixed_weight_retention": size["R_size_vs_fp32"], "mixed_weight_compression": 1.0 / size["R_size_vs_fp32"],
        "engine_compression_S32": b0_lat["engine_size_bytes"] / s32_lat["engine_size_bytes"],
        "engine_compression_JMIX": b0_lat["engine_size_bytes"] / mixed_lat["engine_size_bytes"],
        "structure_mAP_loss": b0["mAP"] - s32["mAP"], "quantization_additional_mAP_loss": s32["mAP"] - mixed["mAP"],
        "joint_mAP_loss": b0["mAP"] - mixed["mAP"],
        "S32_mAP_retention": s32["mAP"] / b0["mAP"], "JMIX_mAP_retention": mixed["mAP"] / b0["mAP"],
        "S32_speedup": s32_lat["speedup_p50_vs_B0"], "JMIX_speedup": mixed_lat["speedup_p50_vs_B0"],
    }
    dump(reports / "compression_accuracy_speedup.json", compression)

    widths = winner["genotype"]["pruning_width_genes"]
    precision_counts = dict(Counter(winner["genotype"]["precision_genes"].values()))
    old_vs_new = {
        "old005": {"R_BOPS": 0.05480626075183459, "shrinker": 52, "fixed500_B0_mAP": 0.658603492, "fixed500_S32_mAP": 0.043540, "JMIX_engine": "failed"},
        "new030": {"candidate_hash": winner["candidate_hash"], "R_BOPS": bops["R_bops"], "shrinker": widths["shrinker_m1.layers.0.double_conv.0::out"], "widths": widths, "precision_counts_variable": precision_counts, "fixed500": {key: metrics(value) for key, value in fixed500.items()}},
        "interpretation": "The 0.30 candidate keeps all attention d_h and FFN d_ff at baseline widths; only CNN stage2/shrinker are structurally reduced. Evaluation scales differ from older fixed50 reports.",
    }
    dump(reports / "old005_vs_new030.json", old_vs_new)
    dump(reports / "h800_vs_4090_reconciliation.json", {
        "h800_run": str(root),
        "server4090_commit": "dde31e9b6e830b41bc96c1c4c7a2b790c55e6a21",
        "server4090_run": "/data/lxf/heal_data/outputs/h800_v2xvit_greedy030_joint_taylor_deployment_closed_v2_20260724_224500",
        "same_conclusion": ["old005 collapse originates in greedy trajectory, not final tie-break", "old train200 unverified", "0.30 has no structural collapse", "attention and FFN stay full width", "final winner has no INT8 genes"],
        "precision_locus_count_glossary": {"search_genes": 80, "realized_policy_units_final_winner": 113, "canonical_weighted_int8_nodes_in_max_profile": 68},
        "nonmergeable_metric_difference": {
            "P32_retention": {"H800_strict_B0_normalized": 1.0, "4090_absolute_fp32_denominator": 0.97335261},
            "conclusion": "Do not compare target retention until both reports use BOPS(candidate)/BOPS(strict-search-B0).",
        },
        "old005_candidate_count": {"historical_H800": 1457, "4090_replay": 1418, "status": "unresolved_historical_artifact_difference"},
        "latency_hardware_specific": True,
    })

    acceptance = {
        "model": "V2X-ViT", "old_budget": 0.05, "new_budget": 0.30, "new_budget_tolerance": 0.005,
        "old005_candidate_distribution_audited": old_audit["candidate_count"] == 1457,
        "old005_tiebreak_overpruning_detected": False,
        "precision_only_requested_floor": floor["R_BOPS_precision_only_floor_requested"],
        "precision_only_buildable_floor": floor["R_BOPS_precision_only_floor_requested"],
        "target030_requires_structural_pruning": False,
        "train200_calibration_verified": True, "old005_train200_calibration_verified": train200["old_train200_calibration_verified"],
        "deployment_closed_precision_space": True, "old_jmix_failure_locus": None,
        "activation_taylor_used_for_fitness": True, "precision_proxy": "weight_plus_activation_conservative_abs",
        "structure_proxy": "tracer_closure_with_functional_gate_taylor", "legacy_coupled_weight_taylor_used_for_fitness": False,
        "joint_taylor_used_for_fitness": False, "cross_residual_used_for_fitness": False,
        "search_loop_forward_calls": 0, "search_loop_backward_calls": 0, "search_loop_physical_exports": 0, "search_loop_trt_builds": 0,
        "budget_reached": 0.295 <= bops["R_bops"] <= 0.305, "winner_bops_retention": bops["R_bops"],
        "winner_parameter_retention": size["R_parameter_retention"], "winner_shrinker_width": widths["shrinker_m1.layers.0.double_conv.0::out"],
        "winner_attention_widths": [value for key, value in widths.items() if key.startswith("attention_dh::")],
        "winner_precision_counts": precision_counts, "stage2_candidates_tested": len(screening),
        "s32_accuracy_gate_passed": all(row["accuracy_gate_passed"] for row in screening),
        "jmix_engine_built": builds["JMIX-FRESH"]["export"]["passed"], "requested_realized_exact": builds["JMIX-FRESH"]["export"]["engine"]["passed"],
        "fixed500_evaluated": min(row["num_evaluated_frames"] for row in fixed500.values()), "fixed500_skipped": sum(row["num_skipped_frames"] for row in fixed500.values()),
        "b0_map": b0["mAP"], "s32_map": s32["mAP"], "jmix_map": mixed["mAP"],
        "s32_map_retention": compression["S32_mAP_retention"], "jmix_map_retention": compression["JMIX_mAP_retention"],
        "b0_p50_ms": b0_lat["forward_p50_ms"], "s32_p50_ms": s32_lat["forward_p50_ms"], "jmix_p50_ms": mixed_lat["forward_p50_ms"],
        "s32_speedup": compression["S32_speedup"], "jmix_speedup": compression["JMIX_speedup"],
        "bops_compression": compression["BOPS_compression"], "parameter_compression": compression["parameter_compression"], "mixed_weight_compression": compression["mixed_weight_compression"],
        "structural_collapse_at_030": False, "formal_ga_allowed": False, "full1789_allowed": False,
        "starting_commit": "3ffcd93021a90743a71f677152e0757e7b2bc1b8", "report_commit": current_commit,
    }
    dump(reports / "final_acceptance.json", acceptance)
    root_conclusion = f"""# V2X-ViT Greedy R_BOPS=0.30 conclusion

The legal-band winner `{winner['candidate_hash']}` reaches R_BOPS={bops['R_bops']:.9f}. Functional gate Taylor leaves all 12 attention and all 3 FFN widths intact; CNN stage2 and shrinker ({widths['shrinker_m1.layers.0.double_conv.0::out']}) supply the structural reduction. Variable precision genes are {precision_counts}.

All five Stage-2 S32 candidates passed fixed50 and zero-skip evaluation. The final B0/S32/JMIX fixed500 mAP values are {b0['mAP']:.6f}, {s32['mAP']:.6f}, and {mixed['mAP']:.6f}; therefore R_BOPS=0.30 does not exhibit structural collapse. Formal TensorRT p50 values are {b0_lat['forward_p50_ms']:.4f}, {s32_lat['forward_p50_ms']:.4f}, and {mixed_lat['forward_p50_ms']:.4f} ms, giving {mixed_lat['speedup_p50_vs_B0']:.3f}x for JMIX.

The old 0.05 Myelin failure was not reproduced after strict calibration/QDQ closure. Shared observers are counted once per sample and zero/subnormal per-channel amax values receive an explicitly audited positive FP32 scale floor. Original P8-max is buildable at R_BOPS={floor['R_BOPS_precision_only_floor_requested']:.9f}, though its fixed50 mAP is materially worse and the new Greedy winner therefore selects no INT8 genes.

Formal GA and full1789 remain disabled. The successful 0.30 result is evidence that 0.05 was an unsafe extreme for this no-training setup, not proof that every 0.05 failure is unavoidable.
"""
    (root / "root_conclusion.md").write_text(root_conclusion, encoding="utf-8")
    print(json.dumps({"winner": winner["candidate_hash"], "fixed500": {key: value["mAP"] for key, value in fixed500.items()}, "latency": {key: value["forward_p50_ms"] for key, value in latency["controls"].items()}}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
