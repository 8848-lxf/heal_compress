#!/usr/bin/env python3
"""Collect one 32-sample Taylor cache and run one six-budget V2X-ViT Greedy path."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import random
import sys
from typing import Any, Mapping

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.audit_heal_transformer_search_models import MODEL_SPECS, _load, _sha256
from scripts.run_v2xvit_greedy005_full import _baseline_candidate, _build_full_space, _formal_space
from scripts.run_v2xvit_greedy005_weight_only_abs import _type_coverage_forward
from scripts.run_v2xvit_greedy030_conservative import (
    _git,
    _legacy_gate_audit,
    _load_fixed_proxy_batches,
    _physical_gpu_uuid,
    write_csv,
    write_json,
    write_json_gzip,
)
from search.canonicalization import canonicalize_candidate
from search.ga.stage12_v3 import (
    RealAnchorArchive,
    Stage1Policy,
    evaluate_stage1_population,
    score_stage2_candidates,
    select_stage2_new_candidates,
)
from search.greedy import GreedyBudgetSearch, GreedySearchConfig
from search.greedy.conservative_joint import (
    combine_precision_action_risk,
    run_conservative_joint_greedy_multi_budget,
    select_stage2_budget_pool,
)
from search.hashing import candidate_hash, canonical_json_hash
from search.proxy.conservative_action_taylor import (
    ActivationActionTaylorProxy,
    StructuralGateTaylorProxy,
    build_activation_taylor_units,
    collect_streaming_activation_action_statistics,
    collect_structural_gate_statistics,
)
from search.proxy.fisher_proxy import FisherStatistics
from search.proxy.joint_weight_taylor import JointWeightTaylorProxy
from search.proxy.taylor_convergence import audit_prefix_action_convergence
from search.quantization_space.v2xvit_deployment_closed import deployment_close_v2xvit_quantization_groups


TARGETS = (0.30, 0.25, 0.20, 0.15, 0.10, 0.05)
PREFIXES = (8, 16, 32)


def _budget_id(target: float) -> str:
    return f"{int(round(float(target) * 100)):03d}"


def _write_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f"refusing_to_overwrite:{path}")
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _activation_prefix_terms(manifest: Mapping[str, Any], prefix: int) -> dict[tuple[str, str, str], tuple[float, float, int]]:
    rows = manifest["prefix_transition_terms"][str(prefix)]
    return {
        (str(row["unit_id"]), str(row["current_precision"]), str(row["next_precision"])): (
            float(row["first_order_abs_sample_mean"]),
            float(row["second_order_abs_sample_mean"]),
            int(row["element_count_prefix_samples"]),
        )
        for row in rows
    }


def _compact_genotype(candidate: Any) -> dict[str, Any]:
    """Widths fully determine pruning units; do not repeat expanded binary genes."""

    return {
        "pruning_width_genes": dict(sorted(candidate.pruning_width_genes.items())),
        "precision_genes": dict(sorted(candidate.precision_genes.items())),
        "meta": {"created_by": "six_budget_compact_genotype", "repair_count": 0},
    }


def _compact_band_row(row: Mapping[str, Any]) -> dict[str, Any]:
    candidate = row["candidate"]
    phenotype = row["phenotype"]
    counts = {
        precision: sum(
            value == precision for value in phenotype.realized_precision_profile.values()
        )
        for precision in ("FP32", "FP16", "INT8")
    }
    return {
        **{
            key: value
            for key, value in row.items()
            if key
            not in {
                "candidate",
                "phenotype",
                "bops_breakdown",
                "risk",
                "matched_budget_targets",
            }
        },
        "genotype": _compact_genotype(candidate),
        "precision_counts": counts,
    }


def _prefix_action_rows(
    *,
    prefixes: tuple[int, ...],
    space: Any,
    baseline: Any,
    formal: Mapping[str, Any],
    gate_manifest: Mapping[str, Any],
    activation_manifest: Mapping[str, Any],
    activation_units: Any,
    gene_to_units: Mapping[str, Any],
    model: torch.nn.Module,
) -> dict[int, list[dict[str, Any]]]:
    engine = GreedyBudgetSearch(
        space,
        config=GreedySearchConfig(bops_targets=TARGETS, bops_tolerance_abs=0.005),
    )
    current = canonicalize_candidate(baseline, space)
    domain_units = {
        domain.domain_id: tuple(domain.ordered_unit_ids) for domain in space.pruning_domains
    }
    rows: dict[int, list[dict[str, Any]]] = {}
    for prefix in prefixes:
        fisher = FisherStatistics(
            gradients=formal["fisher"].prefix_gradients[prefix],
            absolute_gradients=formal["fisher"].prefix_absolute_gradients[prefix],
            fisher_diag=formal["fisher"].prefix_fisher_diag[prefix],
            manifest_hash=f"{formal['fisher'].manifest_hash}:prefix:{prefix}",
        )
        weight = JointWeightTaylorProxy(
            model,
            statistics=fisher,
            unit_to_parameter_slices=formal["slices"],
            strict=True,
        )
        structure = StructuralGateTaylorProxy(
            unit_terms=gate_manifest["prefix_unit_terms"][str(prefix)],
            domain_units=domain_units,
        )
        activation = ActivationActionTaylorProxy(
            statistics=None,
            units=activation_units,
            gene_to_unit_ids=gene_to_units,
            precomputed_transition_terms=_activation_prefix_terms(activation_manifest, prefix),
        )
        prefix_rows = []
        for successor, action in engine._neighbors(baseline):
            phenotype = canonicalize_candidate(successor, space)
            if action["kind"] == "domain_width":
                item = structure.pruning_action_breakdown(current, phenotype)
                j_struct, j_wq, j_aq = float(item["delta_J_struct"]), 0.0, 0.0
            else:
                wrow = weight.weight_quantization_action_breakdown(current, phenotype)
                arow = activation.quantization_action_breakdown(
                    current, phenotype, changed_gene_id=str(action["gene_id"])
                )
                item = combine_precision_action_risk(wrow, arow)
                j_struct, j_wq, j_aq = 0.0, float(item["delta_J_WQ"]), float(item["delta_J_AQ"])
            prefix_rows.append(
                {
                    "action_id": f"{action['kind']}::{action['gene_id']}::{action['previous']}->{action['selected']}",
                    "action_type": action["kind"],
                    "domain_type": action.get("domain_type", "precision"),
                    "J_struct": j_struct,
                    "J_WQ": j_wq,
                    "J_AQ": j_aq,
                    "J_total": j_struct + j_wq + j_aq,
                }
            )
        rows[prefix] = prefix_rows
    return rows


def _stage1_dry_run(result: Mapping[str, Any], space: Any) -> dict[str, Any]:
    band = list(result["budget_candidates"][0.30])[:64]
    by_hash = {str(row["phenotype_hash"]): row for row in band}

    def resources(phenotype: Any) -> dict[str, float]:
        row = by_hash[candidate_hash(phenotype, space)]
        return {
            "BOPS": row["BOPS_after"],
            "R_BOPS": row["current_retention"],
            "params": row["parameter_count"],
            "parameter_retention": row["R_parameter_retention"],
            "mixed_weight_size": row["mixed_weight_size_bytes"],
            "mixed_weight_retention": row["mixed_weight_retention"],
        }

    def proxy(phenotype: Any) -> dict[str, float]:
        row = by_hash[candidate_hash(phenotype, space)]
        return {
            "J_struct": row["cumulative_structural_taylor"],
            "J_WQ": row["cumulative_weight_quant_taylor"],
            "J_AQ": row["cumulative_activation_quant_taylor"],
        }

    population = [row["candidate"] for row in band]
    stage1 = evaluate_stage1_population(
        population,
        space,
        policy=Stage1Policy(target_bops_retention=0.30),
        resource_evaluator=resources,
        proxy_evaluator=proxy,
    )
    synthetic = [
        {
            "physical_hash": row["physical_hash"],
            "status": "ok",
            "requested_realized_exact": True,
            "mAP": 0.650 - index * 0.0002,
            "p50_ms": 23.0 - index * 0.1,
            "parameter_retention": row["parameter_retention"],
            "mixed_weight_retention": row["mixed_weight_retention"],
            "BOPS_deviation": row["BOPS_deviation"],
        }
        for index, row in enumerate(stage1["eligible"][:5])
    ]
    greedy = dict(synthetic[0])
    stage2 = score_stage2_candidates(synthetic, greedy_anchor=greedy)
    archive = RealAnchorArchive({**greedy, "F_S2": 0.8, "stage2_eligible": True})
    archive.update(stage2["rows"])
    quota = select_stage2_new_candidates(
        stage1["eligible"],
        evaluated_hashes=set(),
        historical_real_elites=archive.anchors()["anchors"],
        quota=5,
    )
    def compact(row: Mapping[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in row.items() if key not in {"genotype", "phenotype"}}

    stage1_report = {key: value for key, value in stage1.items() if key not in {"records", "eligible"}}
    stage1_report["eligible"] = [compact(row) for row in stage1["eligible"]]
    quota_report = dict(quota)
    quota_report["new_candidates"] = [compact(row) for row in quota["new_candidates"]]
    return {
        "synthetic_population": True,
        "one_generation_smoke": True,
        "formal_ga_executed": False,
        "stage1": stage1_report,
        "stage2": stage2,
        "anchors": archive.anchors(),
        "quota": quota_report,
        "repair_count": 0,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    for relative in ("reports", "greedy", "proxy", "logs"):
        (root / relative).mkdir(parents=True, exist_ok=True)
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"six_budget_requires_one_visible_gpu:{torch.cuda.device_count()}")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    print("[six-budget] load model and fixed 32-sample prefix", flush=True)
    model, adapter, hypes, _ = _load("v2xvit", device)
    batches, evidence, source_manifest = _load_fixed_proxy_batches(
        adapter=adapter,
        hypes=hypes,
        manifest_path=args.proxy_manifest,
        sample_count=32,
        device=device,
    )
    calibration_hash = canonical_json_hash(
        {
            "model": "lidar_v2xvit",
            "purpose": "stage1_stage2_v3_32_sample_common_taylor",
            "config_sha256": _sha256(MODEL_SPECS["v2xvit"]["config"]),
            "checkpoint_sha256": _sha256(MODEL_SPECS["v2xvit"]["checkpoint"]),
            "source_manifest_hash": source_manifest["manifest_hash"],
            "samples": evidence,
            "seed": args.seed,
        }
    )
    print("[six-budget] trace identity search space", flush=True)
    identity = _build_full_space(model, adapter, hypes, batches[0])
    print("[six-budget] collect 32-sample weight Fisher", flush=True)
    formal = _formal_space(
        model,
        adapter,
        hypes,
        batches[0],
        identity,
        calibration_hash,
        fisher_forward_fn=lambda module, values: _type_coverage_forward(adapter, module, values),
        fisher_batches=batches,
        fisher_audit_prefixes=PREFIXES,
    )
    space = replace(
        formal["space"],
        quantization_groups=deployment_close_v2xvit_quantization_groups(formal["space"].quantization_groups),
    )
    formal["space"] = space
    baseline = _baseline_candidate(space)
    print("[six-budget] collect 32-sample functional gate Taylor", flush=True)
    gate_proxy, gate_manifest = collect_structural_gate_statistics(
        model,
        space.pruning_domains,
        batches,
        forward_fn=lambda module, values: _type_coverage_forward(adapter, module, values),
        loss_fn=adapter.compute_task_loss,
        audit_prefixes=PREFIXES,
    )
    activation_units, gene_to_units, activation_mapping = build_activation_taylor_units(
        model,
        space.quantization_groups,
        mutable_gene_ids=space.precision_gene_ids,
        transformer_precision_units=formal["components"].precision_units,
    )
    groups = {group.group_id: group for group in space.quantization_groups}
    precision_ladders = {
        gene: tuple(value for value in ("FP32", "FP16", "INT8") if value in groups[gene].allowed_precisions)
        for gene in space.precision_gene_ids
    }
    print("[six-budget] collect 32-sample activation Q/DQ Taylor", flush=True)
    activation_proxy, activation_manifest = collect_streaming_activation_action_statistics(
        model,
        batches,
        forward_fn=lambda module, values: _type_coverage_forward(adapter, module, values),
        loss_fn=adapter.compute_task_loss,
        units=activation_units,
        gene_to_unit_ids=gene_to_units,
        precision_ladders=precision_ladders,
        calibration_manifest_hash=calibration_hash,
        audit_prefixes=PREFIXES,
    )
    weight_proxy = JointWeightTaylorProxy(
        model,
        statistics=formal["fisher"],
        unit_to_parameter_slices=formal["slices"],
        strict=True,
    )
    print("[six-budget] audit 8/16/32 prefix convergence", flush=True)
    prefix_rows = _prefix_action_rows(
        prefixes=PREFIXES,
        space=space,
        baseline=baseline,
        formal=formal,
        gate_manifest=gate_manifest,
        activation_manifest=activation_manifest,
        activation_units=activation_units,
        gene_to_units=gene_to_units,
        model=model,
    )
    convergence = audit_prefix_action_convergence(prefix_rows)
    print("[six-budget] run one cache-only Greedy trajectory", flush=True)
    result = run_conservative_joint_greedy_multi_budget(
        space,
        structural_proxy=gate_proxy,
        weight_proxy=weight_proxy,
        activation_proxy=activation_proxy,
        bops_evaluator=formal["bops"].evaluate_breakdown,
        size_evaluator=formal["size"].evaluate_breakdown,
        targets=TARGETS,
        tolerance_abs=0.005,
        maximum_steps=args.maximum_steps,
    )

    print("[six-budget] write compact audit artifacts", flush=True)
    all_public = {}
    summary = []
    for target in TARGETS:
        budget_id = _budget_id(target)
        budget_dir = root / "budgets" / budget_id
        budget_dir.mkdir(parents=True, exist_ok=True)
        public = [_compact_band_row(row) for row in result["budget_candidates"][target]]
        all_public[target] = public
        winner_row = result["winners"][target]
        candidate = winner_row["candidate"]
        phenotype = winner_row["phenotype"]
        winner = {
            "budget": target,
            "candidate_hash": candidate_hash(phenotype, space),
            "genotype": _compact_genotype(candidate),
            "structure_hash": winner_row["physical_hash"],
            "precision_hash": winner_row["precision_hash"],
            "phenotype_hash": winner_row["phenotype_hash"],
            "metrics": {key: value for key, value in winner_row.items() if key not in {"candidate", "phenotype", "bops_breakdown", "risk"}},
            "bops": winner_row["bops_breakdown"],
            "size": formal["size"].evaluate_breakdown(phenotype),
        }
        selected = select_stage2_budget_pool(public, target=target, maximum=5)
        write_json(budget_dir / "winner.json", winner)
        write_csv(
            budget_dir / "budget_candidates.csv",
            [
                {
                    **row,
                    "genotype": json.dumps(row["genotype"], sort_keys=True, separators=(",", ":")),
                    "precision_counts": json.dumps(row["precision_counts"], sort_keys=True, separators=(",", ":")),
                }
                for row in public
            ],
        )
        write_json(budget_dir / "stage2_candidates.json", {"candidates": selected})
        for name in ("S32", "JMIX-FRESH", "train200", "fixed500", "latency"):
            (budget_dir / name).mkdir(exist_ok=True)
        counts = {
            precision: sum(value == precision for value in phenotype.realized_precision_profile.values())
            for precision in ("FP32", "FP16", "INT8")
        }
        summary.append(
            {
                "budget": target,
                "candidate_hash": winner["candidate_hash"],
                "R_BOPS": winner["bops"]["R_bops_vs_fp32"],
                "BOPS_deviation": abs(float(winner["bops"]["R_bops_vs_fp32"]) - target),
                "J_struct": winner["metrics"]["cumulative_structural_taylor"],
                "J_WQ": winner["metrics"]["cumulative_weight_quant_taylor"],
                "J_AQ": winner["metrics"]["cumulative_activation_quant_taylor"],
                "J_total": winner["metrics"]["cumulative_total_taylor"],
                "parameter_retention": winner["size"]["R_parameter_retention"],
                "mixed_weight_retention": winner["size"]["R_size_vs_fp32"],
                **{f"{key}_count": value for key, value in counts.items()},
            }
        )

    dry_run = _stage1_dry_run(result, space)
    write_csv(root / "greedy/full_trace.csv", result["trace"])
    write_json_gzip(root / "greedy/all_budget_candidates.json.gz", {str(key): value for key, value in all_public.items()})
    write_csv(root / "reports/six_budget_greedy_summary.csv", summary)
    write_json(root / "reports/taylor_sample_convergence.json", convergence)
    write_json(root / "reports/structural_gate_mapping.json", gate_manifest)
    write_json(root / "reports/activation_qdq_mapping.json", {"mapping": activation_mapping, "statistics": activation_manifest})
    write_json(root / "reports/fisher_statistics_audit.json", formal["fisher_report"])
    write_json(root / "reports/ga_dry_run.json", dry_run)
    stage1_contract = {
        "order": [
            "static_contract_validation",
            "physical_hash_deduplication",
            "hard_bops_gate",
            "common_taylor_proxy",
            "taylor_sort",
            "strict_numerical_tie_break",
            "stage2_selection",
        ],
        "hard_gate": "abs(R_BOPS-target)<=0.005",
        "objective": "J_struct_gate + J_WQ + J_AQ",
        "taylor_equivalence_band": 0.0,
        "tie_break": [
            "minimum_bops_deviation",
            "higher_parameter_retention",
            "higher_mixed_weight_retention",
            "canonical_hash",
        ],
        "repair_allowed": False,
        "fixed_loci_in_chromosome": False,
    }
    stage2_contract = {
        "steps": [
            "physical_materialization",
            "s32_physical_model",
            "s32_fixed50",
            "fresh_train200_activation_calibration",
            "jmix_strongly_typed_onnx",
            "tensorrt_engine_build",
            "requested_realized_precision",
            "jmix_fixed50",
            "screening_latency",
        ],
        "accuracy_gate": "mAP_candidate >= mAP_greedy - 0.005",
        "score": "0.2*clip((mAP_greedy-mAP_candidate)/0.005,-1,1)+0.8*p50_candidate/p50_greedy",
        "precision_fallback": False,
        "maximum_new_engines_per_generation": 5,
    }
    write_json(root / "reports/ga_stage1_contract.json", stage1_contract)
    write_json(root / "reports/ga_stage2_contract.json", stage2_contract)
    write_json(root / "reports/ga_v1_v2_v3_anchor_audit.json", dry_run["anchors"])
    _write_markdown(
        root / "reports/ga_stage1_contract.md",
        """# GA Stage-1 contract

Candidate legality and canonical identity are checked before the hard BOPS gate. Only candidates inside the exact absolute budget band are scored. The formal objective is `J_struct_gate + J_WQ + J_AQ`; resource fields cannot improve a non-tied Taylor score. The configured Taylor equivalence band is zero.
""",
    )
    _write_markdown(
        root / "reports/ga_stage2_contract.md",
        """# GA Stage-2 contract

Every new candidate must complete physical materialization, S32 fixed50, candidate-bound train200 calibration, strongly typed JMIX build, exact requested/realized audit, JMIX fixed50, and real screening latency. A candidate below the Greedy anchor by more than 0.005 mAP is ineligible. Failures never trigger precision fallback.
""",
    )
    write_json(
        root / "reports/taylor_formula_audit.json",
        {
            "J_total": "J_struct_gate + J_WQ + J_AQ",
            "J_struct": "mean_samples(sum(abs(g_u*u)+0.5*abs(g_u^2*u^2)))",
            "J_WQ": "sum_retained(abs(E_abs_g*delta_W)+0.5*abs(E_g2*delta_W^2))",
            "J_AQ": "mean_samples(sum(abs(g_A*delta_A)+0.5*abs(g_A^2*delta_A^2)))",
            "fisher": "E[g^2]",
            "signed_mean_squared_used": False,
            "elementwise_abs_before_reduction": True,
            "legacy_weight_taylor_used_for_structure_fitness": False,
            "joint_or_cross_used_for_fitness": False,
        },
    )
    _write_markdown(
        root / "reports/taylor_formula_audit.md",
        """# Taylor formula audit

The search objective is `J_struct_gate + J_WQ + J_AQ`. Each component applies absolute value elementwise before tensor, coordinate, and sample reductions. The Fisher diagonal is `E[g^2]`, while `E[g]` remains diagnostic only. Legacy coupled-weight structure Taylor and joint/cross interaction terms do not enter fitness or tie-breaks.
""",
    )
    write_json(
        root / "search_space.json",
        {
            "pruning_domains": [domain.to_dict() for domain in space.pruning_domains],
            "quantization_groups": [group.to_dict() for group in space.quantization_groups],
            "calibration_manifest_hash": calibration_hash,
            "trace_snapshot_hash": space.trace_snapshot_hash,
            "fixed_rankings": {
                "fisher": formal["fisher_report"],
                "atomic": formal["ranking_report"],
                "transformer": formal["transformer_ranking_report"],
            },
        },
    )
    gate_legacy = _legacy_gate_audit(
        space=space, baseline=baseline, gate_proxy=gate_proxy, weight_proxy=weight_proxy
    )
    write_csv(root / "proxy/gate_vs_legacy_weight_score.csv", gate_legacy)
    manifest = {
        "schema_version": "v2xvit-six-budget-stage1-v1",
        "branch": _git("branch", "--show-current"),
        "starting_commit": args.starting_commit,
        "working_commit": _git("rev-parse", "HEAD"),
        "formal_ga_executed": False,
        "targets": list(TARGETS),
        "absolute_tolerance": 0.005,
        "physical_gpu_index": args.physical_gpu,
        "gpu_uuid": _physical_gpu_uuid(args.physical_gpu),
        "proxy_samples": 32,
        "proxy_manifest": str(args.proxy_manifest.resolve()),
        "proxy_manifest_hash": source_manifest["manifest_hash"],
        "calibration_manifest_hash": calibration_hash,
        "checkpoint_sha256": _sha256(MODEL_SPECS["v2xvit"]["checkpoint"]),
        "config_sha256": _sha256(MODEL_SPECS["v2xvit"]["config"]),
        "search_loop_runtime_audit": {key: result[key] for key in result if key.startswith("search_loop_")},
        "budget_band_candidate_counts": result["budget_band_candidate_counts"],
        "termination_reason": result["termination_reason"],
        "taylor_convergence_passed": convergence["passed"],
    }
    write_json(root / "run_manifest.json", manifest)
    print(json.dumps({"status": "ok", "output_root": str(root), "summary": summary}, sort_keys=True), flush=True)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--maximum-steps", type=int, default=10000)
    parser.add_argument("--starting-commit", default="dde31e9b6e830b41bc96c1c4c7a2b790c55e6a21")
    parser.add_argument(
        "--proxy-manifest",
        type=Path,
        default=REPO / "search/model_family/manifests/heal_lidar_v2xvit_train200_fixed_k.json",
    )
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
