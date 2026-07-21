#!/usr/bin/env python3
"""Run bounded real-data GA and greedy framework smoke tests for HEAL V2X-ViT."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from search.cache.proxy_cache import ProxyCache
from search.candidate import CandidateGenotype
from search.canonicalization import SearchSpaceSpec, canonicalize_candidate
from search.ga.engine import GAConfig, GeneticSearchEngine
from search.ga.initialization import baseline_candidate
from search.greedy import GreedyBudgetSearch, GreedySearchConfig
from search.hashing import candidate_hash, canonical_json_hash, search_hash
from search.model_family.calibration_manifest import load_v2xvit_train_manifest
from search.model_family.model_provider import load_heal_model_family
from search.model_family.search_smoke import (
    collect_v2xvit_manifest_fisher_statistics,
    load_v2xvit_manifest_batches,
    run_v2xvit_physical_stage2_smoke,
)
from search.model_family.search_space import (
    build_ranked_v2xvit_ffn_domains,
    build_v2xvit_ffn_atomic_units,
    build_v2xvit_quantization_groups,
)
from search.proxy.bops_proxy import BOPSProxy
from search.proxy.gpu_batch_proxy import TorchBatchedProxyScorer
from search.proxy.joint_weight_taylor import JointWeightTaylorProxy
from search.proxy.normalization import NormalizationStats
from search.proxy.objective import ProxyObjective, ProxyObjectiveConfig
from search.proxy.parameter_slice_resolver import build_unit_parameter_slices
from search.proxy.runtime_shape_profiler import profile_runtime_layer_shapes
from search.proxy.size_proxy import SizeProxy
from search.pruning_space.domain_importance import score_atomic_units_for_fixed_ranking
from search.stage1.proxy_evaluator import Stage1ProxyEvaluator


DEFAULT_CONFIG = Path(
    "/home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/"
    "LiDAROnly/lidar_v2xvit/config.yaml"
)
DEFAULT_CHECKPOINT = DEFAULT_CONFIG.parent / "net_epoch_bestval_at27.pth"
DEFAULT_MANIFEST = (
    REPO_ROOT
    / "search/model_family/manifests/heal_lidar_v2xvit_train200_fixed_k.json"
)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f"refusing_to_overwrite_v2xvit_smoke_artifact:{path}")
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _compact_scored(rows: list[tuple[CandidateGenotype, float, dict[str, Any]]]) -> list[dict[str, Any]]:
    result = []
    seen: set[str] = set()
    for genotype, score, metrics in rows:
        identity = str(metrics.get("candidate_hash", ""))
        if identity in seen:
            continue
        seen.add(identity)
        phenotype = metrics.get("phenotype", {})
        result.append(
            {
                "candidate_hash": identity,
                "score": float(score),
                "F1": float(metrics["F1"]),
                "L_joint_weight_taylor": float(metrics["L_joint_weight_taylor"]),
                "L_pruning_only_taylor": float(metrics["L_pruning_only_taylor"]),
                "L_retained_weight_quant_taylor": float(
                    metrics["L_retained_weight_quant_taylor"]
                ),
                "R_bops_vs_fp32": float(metrics["R_bops_vs_fp32"]),
                "R_parameter_retention": float(metrics["R_parameter_retention"]),
                "bops_feasible": bool(metrics["bops_feasible"]),
                "bops_violation": float(metrics["bops_violation"]),
                "pruned_unit_count": len(phenotype.get("pruned_unit_ids", [])),
                "domain_width_profile": dict(
                    phenotype.get("metadata", {}).get("domain_width_profile", {})
                ),
                "precision_counts": {
                    precision: list(
                        phenotype.get("realized_precision_profile", {}).values()
                    ).count(precision)
                    for precision in ("FP32", "FP16", "INT8")
                },
                "genotype": genotype.to_dict(),
            }
        )
    return result


def _greedy_summary(result: Any) -> dict[str, Any]:
    return {
        "termination_reason": result.termination_reason,
        "step_count": len(result.steps),
        "evaluated_neighbor_count": result.evaluated_neighbor_count,
        "budget_targets_reached": sorted(result.budget_candidates),
        "unreachable_targets": list(result.unreachable_targets),
        "initial_metrics": dict(result.initial_metrics),
        "steps": [row.to_dict() for row in result.steps],
        "budget_candidates": {
            f"{target:.9f}": candidate.to_dict()
            for target, candidate in sorted(result.budget_candidates.items())
        },
        "budget_metrics": {
            f"{target:.9f}": dict(metrics)
            for target, metrics in sorted(result.budget_metrics.items())
        },
        "nearest_budget_candidates": {
            f"{target:.9f}": candidate.to_dict()
            for target, candidate in sorted(result.nearest_budget_candidates.items())
        },
        "nearest_budget_metrics": {
            f"{target:.9f}": dict(metrics)
            for target, metrics in sorted(result.nearest_budget_metrics.items())
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--heal-root", type=Path, default=Path("/home/lixingfeng/UniAD_examine/HEAL"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fisher-samples", type=int, default=1)
    parser.add_argument("--ga-population", type=int, default=12)
    parser.add_argument("--ga-generations", type=int, default=2)
    parser.add_argument("--proxy-batch-size", type=int, default=64)
    parser.add_argument("--formal-bops-target", type=float, default=0.25)
    parser.add_argument("--bops-tolerance", type=float, default=0.005)
    parser.add_argument("--greedy-maximum-steps", type=int, default=128)
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("v2xvit_framework_smoke_requires_cuda_batched_proxy")
    torch.cuda.set_device(device)

    manifest = load_v2xvit_train_manifest(args.manifest)
    bundle = load_heal_model_family(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        heal_root=args.heal_root,
        device=args.device,
        family_id="heal_lidar_v2xvit",
        forward_smoke=False,
    )
    batches, sample_evidence = load_v2xvit_manifest_batches(
        bundle,
        manifest,
        sample_count=int(args.fisher_samples),
        device=device,
    )
    statistics, fisher_report = collect_v2xvit_manifest_fisher_statistics(
        bundle, manifest, batches, sample_evidence
    )
    _write_json(output_dir / "fisher_manifest.json", fisher_report)

    runtime_profile = profile_runtime_layer_shapes(
        bundle.model,
        batches[0],
        forward_fn=bundle.adapter.forward_for_task,
    )
    _write_json(output_dir / "runtime_shapes.json", runtime_profile.to_dict())
    active_module_paths = sorted({row.module_path for row in runtime_profile.shapes})
    atomic_units, _capabilities = build_v2xvit_ffn_atomic_units(
        bundle.model, bundle.audit
    )
    unit_slices = build_unit_parameter_slices(bundle.model, atomic_units)
    importance_scores, ranking_manifest = score_atomic_units_for_fixed_ranking(
        bundle.model, statistics, unit_slices, strict=True
    )
    domains = build_ranked_v2xvit_ffn_domains(atomic_units, importance_scores)
    _write_json(output_dir / "fixed_ffn_taylor_ranking.json", ranking_manifest)
    quantization_groups = build_v2xvit_quantization_groups(
        bundle.model,
        bundle.audit,
        active_module_paths=active_module_paths,
    )
    capability = torch.cuda.get_device_capability(device)
    space = SearchSpaceSpec(
        pruning_unit_ids=[row.stable_id for row in atomic_units],
        precision_layer_ids=active_module_paths,
        quantization_groups=tuple(quantization_groups),
        pruning_domains=tuple(domains),
        default_precision="FP32",
        pruning_policy_version="v2xvit-ffn-legal-domain-width-fixed-ranking-v1",
        precision_policy_version="v2xvit-capability-smoke-not-qdq-deployment-v1",
        trace_snapshot_hash=bundle.audit.to_dict()["audit_hash"],
        calibration_manifest_hash=str(manifest["manifest_hash"]),
        onnx_export_config_hash=canonical_json_hash(
            {
                "fixed_k": manifest["fixed_k_contract"]["value"],
                "max_agents": manifest["input_contract"]["max_agents"],
                "explicit_qdq": "required_but_not_part_of_framework_smoke",
            }
        ),
        tensorrt_version="not_invoked_by_framework_smoke",
        gpu_compute_capability=f"{capability[0]}.{capability[1]}",
        builder_flags={
            "stage2_backend": "real_pytorch_physical_plus_weight_fake_quant_smoke",
            "strongly_typed_required_for_production": True,
            "explicit_qdq_required_for_production": True,
        },
    )
    search_space_report = {
        "schema_version": "v2xvit-joint-search-space-smoke-v1",
        "checkpoint_hash": bundle.checkpoint_hash,
        "calibration_manifest_hash": manifest["manifest_hash"],
        "fixed_k": manifest["fixed_k_contract"]["value"],
        "real_runtime_weighted_module_count": len(active_module_paths),
        "precision_gene_count": len(space.precision_gene_ids),
        "int8_capable_gene_count": sum(
            "INT8" in row.allowed_precisions for row in quantization_groups
        ),
        "fp32_fp16_only_gene_count": sum(
            "INT8" not in row.allowed_precisions for row in quantization_groups
        ),
        "functional_weight_count_fixed_fp16": 6,
        "functional_weight_bops_in_stage1": False,
        "atomic_ffn_unit_count": len(atomic_units),
        "pruning_domain_count": len(domains),
        "pruning_gene_count": len(space.pruning_gene_ids),
        "pruning_domains": [row.to_dict() for row in domains],
        "quantization_groups": [row.to_dict() for row in quantization_groups],
        "deployment_admission": {
            "ffn_physical_materialization": "enabled_for_smoke_and_real_forward",
            "explicit_qdq": False,
            "tensorrt": False,
            "formal_accuracy_evaluation": False,
        },
    }
    _write_json(output_dir / "search_space.json", search_space_report)

    objective_config = ProxyObjectiveConfig(
        objective_mode="joint_weight_taylor_hard_bops",
        bops_threshold=float(args.formal_bops_target),
        bops_constraint_mode="hard_band_feasibility",
        bops_tolerance_abs=float(args.bops_tolerance),
        parameter_retention_tiebreak_epsilon=0.0,
    )
    normalization = NormalizationStats()
    joint = JointWeightTaylorProxy(
        bundle.model,
        statistics=statistics,
        unit_to_parameter_slices=unit_slices,
        strict=True,
    )
    objective = ProxyObjective(
        size=SizeProxy(
            model=bundle.model, unit_to_parameter_slices=unit_slices
        ),
        bops=BOPSProxy(
            model=bundle.model,
            unit_to_parameter_slices=unit_slices,
            runtime_shapes=runtime_profile.shapes,
        ),
        joint_weight_taylor=joint,
        normalization=normalization,
        config=objective_config,
    )
    gpu_scorer = TorchBatchedProxyScorer.from_components(
        model=bundle.model,
        space=space,
        unit_to_parameter_slices=unit_slices,
        fisher_statistics=statistics,
        runtime_shapes=runtime_profile.shapes,
        normalization=normalization,
        config=objective_config,
        device=device,
        batch_size=int(args.proxy_batch_size),
    )
    proxy = Stage1ProxyEvaluator(
        space,
        objective=objective,
        cache=ProxyCache(output_dir / "proxy_archive.jsonl"),
        cache_key_fn=lambda phenotype, _space: search_hash(
            phenotype,
            trace_hash=space.trace_snapshot_hash,
            proxy_version="v2xvit-joint-weight-taylor-gpu-smoke-v1",
            calibration_statistics_version=(
                statistics.statistics_version + ":" + statistics.manifest_hash
            ),
        ),
        batch_scorer=gpu_scorer,
        proxy_backend="cuda_batched",
        proxy_device=str(device),
        proxy_batch_size=int(args.proxy_batch_size),
    )

    baseline = baseline_candidate(space)
    baseline_phenotype = canonicalize_candidate(baseline, space)
    scalar_baseline = objective.evaluate(baseline_phenotype)
    gpu_baseline = proxy.evaluate_batch(
        [baseline], generation=-1, outer_round=-1
    ).metrics[0]
    scalar_gpu_parity = {
        key: {
            "scalar": float(scalar_baseline[key]),
            "gpu": float(gpu_baseline[key]),
            "abs_delta": abs(
                float(scalar_baseline[key]) - float(gpu_baseline[key])
            ),
        }
        for key in ("L_joint_weight_taylor", "R_bops_vs_fp32")
    }
    parity_passed = all(
        row["abs_delta"] <= 1.0e-5 for row in scalar_gpu_parity.values()
    )
    _write_json(
        output_dir / "proxy_parity.json",
        {
            "passed": parity_passed,
            "metrics": scalar_gpu_parity,
            "gpu_backend": "cuda_batched",
        },
    )

    ga = GeneticSearchEngine(
        space,
        GAConfig(
            initial_population_size=int(args.ga_population),
            population_size=int(args.ga_population),
            offspring_size=int(args.ga_population),
            num_generations=int(args.ga_generations),
            constraint_first_ranking=True,
            random_seed=20260717,
            mutation_action_min=1,
            mutation_action_max=2,
        ),
    )
    ga_rows = ga.run(
        batch_evaluator=lambda candidates, generation: proxy.evaluate_batch(
            candidates, generation=generation, outer_round=0
        ),
        candidate_key_fn=lambda genotype: candidate_hash(
            canonicalize_candidate(genotype, space), space
        ),
    )
    ga_compact = _compact_scored(ga_rows)
    ga_feasible = [row for row in ga_rows if bool(row[2].get("bops_feasible"))]
    if not ga_rows or not ga_feasible:
        raise RuntimeError("v2xvit_ga_smoke_has_no_feasible_candidate")
    ga_best_genotype, _ga_score, ga_best_metrics = ga_feasible[0]
    _write_json(
        output_dir / "ga_smoke.json",
        {
            "passed": True,
            "population": int(args.ga_population),
            "generations": int(args.ga_generations),
            "evaluated_row_count": len(ga_rows),
            "unique_candidate_count": len(ga_compact),
            "feasible_row_count": len(ga_feasible),
            "best": ga_compact[0],
            "candidates": ga_compact,
        },
    )

    greedy_exploration = GreedyBudgetSearch(
        space,
        config=GreedySearchConfig(
            bops_targets=(float(args.formal_bops_target),),
            bops_tolerance_abs=float(args.bops_tolerance),
            maximum_steps=int(args.greedy_maximum_steps),
        ),
    ).run(
        lambda candidates, step: proxy.evaluate_batch(
            candidates, generation=step, outer_round=1
        ).metrics
    )
    if not greedy_exploration.steps:
        raise RuntimeError("v2xvit_greedy_smoke_has_no_selected_action")
    exact_capture_target = float(greedy_exploration.steps[0].bops_after)
    greedy_capture = GreedyBudgetSearch(
        space,
        config=GreedySearchConfig(
            bops_targets=(exact_capture_target,),
            bops_tolerance_abs=float(args.bops_tolerance),
            maximum_steps=2,
        ),
    ).run(
        lambda candidates, step: proxy.evaluate_batch(
            candidates, generation=step, outer_round=2
        ).metrics
    )
    if exact_capture_target not in greedy_capture.budget_candidates:
        raise RuntimeError("v2xvit_greedy_exact_budget_capture_failed")
    greedy_best_genotype = greedy_capture.budget_candidates[exact_capture_target]
    _write_json(
        output_dir / "greedy_smoke.json",
        {
            "passed": True,
            "formal_exploration": _greedy_summary(greedy_exploration),
            "exact_budget_capture_target": exact_capture_target,
            "exact_budget_capture": _greedy_summary(greedy_capture),
        },
    )

    forced_widths = {
        domain.domain_id: domain.original_width for domain in domains
    }
    for domain in domains:
        forced_widths[domain.domain_id] = domain.legal_widths[-2]
    forced_prune = CandidateGenotype(
        pruning_width_genes=forced_widths,
        precision_genes={gene_id: "FP16" for gene_id in space.precision_gene_ids},
        meta={"created_by": "forced_nonzero_ffn_physical_smoke"},
    )
    candidates = {
        "ga_best": canonicalize_candidate(ga_best_genotype, space),
        "greedy_exact_budget": canonicalize_candidate(greedy_best_genotype, space),
        "forced_nonzero_ffn": canonicalize_candidate(forced_prune, space),
    }
    candidates["forced_nonzero_ffn_replay"] = candidates["forced_nonzero_ffn"]
    stage2_results = {}
    by_identity: dict[str, str] = {}
    for label, phenotype in candidates.items():
        identity = candidate_hash(phenotype, space)
        if identity in by_identity:
            stage2_results[label] = {
                "passed": True,
                "cache_hit": True,
                "reused_from": by_identity[identity],
                "candidate_hash": identity,
            }
            continue
        by_identity[identity] = label
        result = run_v2xvit_physical_stage2_smoke(
            bundle,
            phenotype,
            domains,
            batches[0],
            latency_rounds=3,
        )
        stage2_results[label] = {
            **result,
            "cache_hit": False,
            "candidate_hash": identity,
            "pruned_unit_count": len(phenotype.pruned_unit_ids),
            "domain_width_profile": dict(
                phenotype.metadata.get("domain_width_profile", {})
            ),
        }
    _write_json(output_dir / "stage2_pytorch_smoke.json", stage2_results)

    forced_stage2 = stage2_results["forced_nonzero_ffn"]
    acceptance = {
        "schema_version": "v2xvit-ga-greedy-framework-smoke-acceptance-v1",
        "checkpoint_strict_load": True,
        "frozen_train200_manifest_verified": True,
        "fixed_k": manifest["fixed_k_contract"]["value"],
        "real_task_loss_fisher_collected": bool(
            fisher_report["all_gradients_finite"]
            and fisher_report["all_fisher_finite"]
        ),
        "ffn_domain_width_gene_count": len(space.pruning_gene_ids),
        "canonical_precision_gene_count": len(space.precision_gene_ids),
        "int8_capable_gene_count": search_space_report["int8_capable_gene_count"],
        "gpu_batched_proxy_verified": bool(
            proxy.gpu_batch_count > 0 and proxy.scalar_evaluate_call_count == 0
        ),
        "scalar_gpu_proxy_parity": parity_passed,
        "ga_framework_smoke_passed": bool(ga_rows and ga_feasible),
        "greedy_framework_smoke_passed": bool(
            greedy_exploration.steps and greedy_capture.budget_candidates
        ),
        "hard_bops_tolerance": float(args.bops_tolerance),
        "ga_formal_target": float(args.formal_bops_target),
        "ga_best_bops": float(ga_best_metrics["R_bops_vs_fp32"]),
        "ga_best_bops_feasible": bool(ga_best_metrics["bops_feasible"]),
        "greedy_formal_target_reached": bool(
            greedy_exploration.budget_candidates
        ),
        "greedy_exact_budget_capture_passed": bool(
            greedy_capture.budget_candidates
        ),
        "nonzero_physical_pruning_smoke_passed": bool(
            forced_stage2.get("passed")
            and int(forced_stage2.get("pruned_unit_count", 0)) > 0
            and int(forced_stage2.get("parameter_reduction", 0)) > 0
        ),
        "physical_checkpoint_strict_reload": all(
            bool(row.get("checkpoint_strict_reload", row.get("cache_hit", False)))
            for row in stage2_results.values()
        ),
        "real_pytorch_forward": all(
            bool(row.get("real_forward", row.get("cache_hit", False)))
            for row in stage2_results.values()
        ),
        "proxy_cache_hit_count": int(proxy.cache_hit_count),
        "proxy_cache_verified": bool(proxy.cache_hit_count > 0),
        "stage2_identity_cache_verified": any(
            bool(row.get("cache_hit")) for row in stage2_results.values()
        ),
        "candidate_identity_cache_verified": bool(
            proxy.cache_hit_count > 0
            and any(bool(row.get("cache_hit")) for row in stage2_results.values())
        ),
        "explicit_qdq_complete": False,
        "tensorrt_engine_complete": False,
        "full_accuracy_evaluation_complete": False,
        "deployment_search_ready": False,
    }
    acceptance["framework_smoke_passed"] = all(
        bool(acceptance[key])
        for key in (
            "checkpoint_strict_load",
            "frozen_train200_manifest_verified",
            "real_task_loss_fisher_collected",
            "gpu_batched_proxy_verified",
            "scalar_gpu_proxy_parity",
            "ga_framework_smoke_passed",
            "greedy_framework_smoke_passed",
            "nonzero_physical_pruning_smoke_passed",
            "physical_checkpoint_strict_reload",
            "real_pytorch_forward",
            "proxy_cache_verified",
            "stage2_identity_cache_verified",
            "candidate_identity_cache_verified",
        )
    )
    _write_json(output_dir / "acceptance.json", acceptance)
    print(json.dumps(acceptance, indent=2))
    return 0 if acceptance["framework_smoke_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
