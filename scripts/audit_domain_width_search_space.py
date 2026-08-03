#!/usr/bin/env python3
"""Build the real lidar_pyramid domain-width search space without running GA."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import random
import sys
from typing import Any

import torch
import yaml


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Static production audit for legal domain-width and precision genes."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fisher-batches", type=int, default=None)
    parser.add_argument("--population-smoke-size", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if os.environ.get("CONDA_DEFAULT_ENV") != "univ2x-opt":
        raise RuntimeError(
            f"univ2x_opt_environment_required:{os.environ.get('CONDA_DEFAULT_ENV', '')}"
        )
    config_path = Path(args.config).resolve()
    config = dict(yaml.safe_load(config_path.read_text(encoding="utf-8")) or {})
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"audit_output_must_be_new_or_empty:{output}")
    output.mkdir(parents=True, exist_ok=True)
    model_cfg = dict(config.get("model", {}) or {})
    runtime_cfg = dict(config.get("runtime", {}) or {})
    pruning_cfg = dict(config.get("pruning", {}) or {})
    proxy_cfg = dict(config.get("proxy", {}) or {})
    stage2_cfg = dict(config.get("stage2", {}) or {})
    full_validation_cfg = dict(config.get("full_validation", {}) or {})
    fisher_batches = int(
        args.fisher_batches
        if args.fisher_batches is not None
        else proxy_cfg.get("fisher_calibration_batches", 8)
    )

    from search.integration.calibration_provider import collect_or_load_fisher_statistics
    from search.integration.lidar_pyramid_context import build_lidar_pyramid_context
    from search.candidate import CandidateGenotype
    from search.canonicalization import canonicalize_candidate, repair_genotype
    from search.ga.initialization import baseline_candidate
    from search.proxy.gpu_batch_proxy import TorchBatchedProxyScorer
    from search.proxy.normalization import NormalizationStats
    from search.proxy.objective import ProxyObjectiveConfig
    from search.proxy.parameter_slice_resolver import build_unit_parameter_slices
    from search.proxy.runtime_shape_profiler import profile_runtime_layer_shapes
    from search.pruning_space.domain_importance import score_atomic_units_for_fixed_ranking
    from search.pruning_space.local_domains import build_local_pruning_domains

    checkpoint = Path(model_cfg["checkpoint"]).resolve()
    model_config = Path(model_cfg["config"]).resolve()
    context = build_lidar_pyramid_context(
        checkpoint_path=checkpoint,
        model_config_path=model_config,
        heal_root=runtime_cfg.get(
            "heal_root", "../../HEAL"
        ),
        tensorrt_root=runtime_cfg["tensorrt_root"],
        plugin_path=runtime_cfg.get("plugin_path"),
        output_dir=output,
        gpu_id=str(runtime_cfg.get("gpu_id", 6)),
        exclude_gpu_ids=[int(value) for value in runtime_cfg.get("exclude_gpu_ids", [])],
        tensorrt_env=str(runtime_cfg.get("tensorrt_env", "modelopt")),
        fisher_calibration_batches=fisher_batches,
        quant_calibration_batches=int(proxy_cfg.get("quant_calibration_batches", 200)),
        quant_calibration_npz_manifest=proxy_cfg.get("quant_calibration_npz_manifest"),
        quant_activation_calibration_backend=str(
            proxy_cfg.get(
                "quant_activation_calibration_backend",
                "tensorrt_entropy_calibration2",
            )
        ),
        quant_calibration_force_rebuild=bool(
            proxy_cfg.get("quant_calibration_force_rebuild", False)
        ),
        num_frames=max(
            int(stage2_cfg.get("num_frames", 500)),
            int(full_validation_cfg.get("num_frames", 0) or 0),
        ),
        warmup_frames=max(
            int(stage2_cfg.get("warmup_frames", 200)),
            int(full_validation_cfg.get("warmup_frames", 0) or 0),
        ),
        reset_after_warmup=bool(
            stage2_cfg.get("reset_after_warmup", True)
            or full_validation_cfg.get("reset_after_warmup", False)
        ),
        default_precision=str(config.get("precision", {}).get("default", "FP32")),
        pruning_gene_type="legal_domain_width",
    )
    slices = build_unit_parameter_slices(context.model, context.atomic_prune_units)
    statistics = collect_or_load_fisher_statistics(
        model=context.model,
        adapter=context.model_bundle.adapter,
        model_config_path=context.model_config,
        device=torch.device(context.runtime_device),
        cache_path=output / "fisher_statistics.pt",
        num_batches=fisher_batches,
    )
    scores, ranking_manifest = score_atomic_units_for_fixed_ranking(
        context.model,
        statistics,
        slices,
        strict=True,
    )
    grouped_cfg = dict(pruning_cfg.get("grouped_conv", {}) or {})
    dense_cfg = dict(pruning_cfg.get("dense", {}) or {})
    domains = build_local_pruning_domains(
        context.atomic_prune_units,
        importance_scores=scores,
        ranking_method="pruning_only_first_plus_second_order_fisher_taylor",
        minimum_retained_ratio=float(pruning_cfg.get("minimum_retained_ratio", 0.10)),
        dense_alignment=int(dense_cfg.get("alignment", 4)),
        grouped_allowed_channels_per_group=[
            int(value)
            for value in grouped_cfg.get(
                "allowed_channels_per_group",
                [4, 8, 16, 32, 64, 128, 256, 512],
            )
        ],
    )
    context.search_space = replace(
        context.search_space,
        pruning_domains=tuple(domains),
        pruning_policy_version="legal-domain-width-fixed-ranking-v1",
    )
    runtime_shapes = profile_runtime_layer_shapes(
        context.model,
        context.trace_example_inputs,
        forward_fn=context.model_bundle.adapter.forward_for_task,
    )
    scorer = TorchBatchedProxyScorer.from_components(
        model=context.model,
        space=context.search_space,
        unit_to_parameter_slices=slices,
        fisher_statistics=statistics,
        runtime_shapes=runtime_shapes.shapes,
        normalization=NormalizationStats(version="none-v1"),
        config=ProxyObjectiveConfig(
            objective_mode="joint_weight_taylor_hard_bops",
            bops_threshold=None,
            bops_constraint_mode="hard_feasibility",
            parameter_retention_tiebreak_epsilon=0.0,
        ),
        device=context.runtime_device,
        batch_size=int(proxy_cfg.get("batch_size", 128)),
    )
    all_keep = repair_genotype(baseline_candidate(context.search_space), context.search_space)
    first_domain = next(domain for domain in domains if len(domain.legal_widths) > 1)
    lower_width = max(
        width for width in first_domain.legal_widths if width < first_domain.original_width
    )
    changed_widths = dict(all_keep.pruning_width_genes)
    changed_widths[first_domain.domain_id] = int(lower_width)
    changed_precisions = dict(all_keep.precision_genes)
    first_int8_group = next(
        group
        for group in context.search_space.quantization_groups
        if "INT8" in group.allowed_precisions and not group.protected
    )
    changed_precisions[first_int8_group.group_id] = "INT8"
    changed = repair_genotype(
        CandidateGenotype(
            pruning_genes=all_keep.pruning_genes,
            pruning_width_genes=changed_widths,
            precision_genes=changed_precisions,
            meta={"created_by": "static_real_proxy_smoke"},
        ),
        context.search_space,
    )
    proxy_result = scorer.evaluate_batch(
        [
            canonicalize_candidate(all_keep, context.search_space),
            canonicalize_candidate(changed, context.search_space),
        ],
        generation=0,
        outer_round=0,
    )
    all_keep_metrics, changed_metrics = proxy_result.metrics
    if abs(float(all_keep_metrics["L_joint_weight_taylor"])) > 1.0e-12:
        raise RuntimeError(f"all_keep_fp32_joint_taylor_not_zero:{all_keep_metrics}")
    if abs(float(all_keep_metrics["R_bops_vs_fp32"]) - 1.0) > 1.0e-6:
        raise RuntimeError(f"all_keep_fp32_bops_not_one:{all_keep_metrics}")
    if not float(changed_metrics["R_bops_vs_fp32"]) < 1.0:
        raise RuntimeError(f"changed_candidate_bops_not_reduced:{changed_metrics}")
    if not float(changed_metrics["L_joint_weight_taylor"]) > 0.0:
        raise RuntimeError(f"changed_candidate_joint_taylor_nonpositive:{changed_metrics}")
    proxy_smoke = {
        "status": "passed",
        "backend": proxy_result.stats.get("proxy_backend", ""),
        "device": context.runtime_device,
        "uses_explicit_candidate_masks": scorer.uses_explicit_candidate_masks,
        "dense_atomic_action_channel_tensor_allocated": bool(
            scorer.channel_resolver.action_out_mask is not None
            or scorer.channel_resolver.action_in_mask is not None
        ),
        "compute_legacy_sqnr_metrics": scorer.compute_legacy_sqnr_metrics,
        "layer_count": len(scorer.layer_ids),
        "atomic_action_count": len(scorer.action_ids),
        "max_channels": scorer.channel_resolver.max_channels,
        "table_memory_allocated_bytes_after_build": int(
            torch.cuda.memory_allocated(torch.device(context.runtime_device))
        ),
        "all_keep_fp32": all_keep_metrics,
        "changed_candidate": {
            "domain_id": first_domain.domain_id,
            "original_width": first_domain.original_width,
            "selected_width": lower_width,
            "int8_group_id": first_int8_group.group_id,
            "metrics": changed_metrics,
        },
        "batch_stats": proxy_result.stats,
    }
    if int(args.population_smoke_size) > 0:
        from search.ga.immigrants import random_immigrant

        rng = random.Random(20260717)
        population = [
            random_immigrant(context.search_space, rng)
            for _ in range(int(args.population_smoke_size))
        ]
        population_result = scorer.evaluate_batch(
            [canonicalize_candidate(row, context.search_space) for row in population],
            generation=1,
            outer_round=0,
        )
        if len(population_result.metrics) != int(args.population_smoke_size):
            raise RuntimeError("real_population_proxy_smoke_length_mismatch")
        if any(
            not torch.isfinite(torch.tensor(float(row["L_joint_weight_taylor"])))
            or not torch.isfinite(torch.tensor(float(row["R_bops_vs_fp32"])))
            for row in population_result.metrics
        ):
            raise RuntimeError("real_population_proxy_smoke_nonfinite")
        proxy_smoke["population_smoke"] = {
            "candidate_count": int(args.population_smoke_size),
            "scalar_evaluate_call_count": 0,
            "batch_stats": population_result.stats,
            "minimum_R_bops_vs_fp32": min(
                float(row["R_bops_vs_fp32"]) for row in population_result.metrics
            ),
            "maximum_R_bops_vs_fp32": max(
                float(row["R_bops_vs_fp32"]) for row in population_result.metrics
            ),
            "minimum_L_joint_weight_taylor": min(
                float(row["L_joint_weight_taylor"])
                for row in population_result.metrics
            ),
            "maximum_L_joint_weight_taylor": max(
                float(row["L_joint_weight_taylor"])
                for row in population_result.metrics
            ),
        }
    groups = tuple(context.search_space.quantization_groups)
    protected = [group.to_dict() for group in groups if group.protected]
    summary = {
        "status": "ok",
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "model_config": str(model_config),
        "model_config_sha256": _sha256(model_config),
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV", ""),
        "python": sys.executable,
        "runtime_device": context.runtime_device,
        "physical_gpu_id": context.physical_gpu_id,
        "trace_atomic_unit_count": len(context.trace_result.atomic_prune_units),
        "selected_safe_atomic_unit_count": len(context.atomic_prune_units),
        "domain_count": len(domains),
        "nontrivial_domain_width_gene_count": len(
            context.search_space.pruning_gene_ids
        ),
        "dense_domain_count": sum(domain.kind == "dense" for domain in domains),
        "regular_grouped_domain_count": sum(
            domain.kind == "regular_grouped" for domain in domains
        ),
        "legal_width_choice_count": sum(
            len(domain.legal_widths) for domain in domains
        ),
        "post_search_alignment_repair_required": False,
        "fixed_ranking_precision_independent": True,
        "fixed_ranking_formula": ranking_manifest.get("formula", ""),
        "fisher_statistics_manifest_hash": statistics.manifest_hash,
        "precision_parameterized_layer_count": len(
            context.search_space.precision_layer_ids
        ),
        "precision_gene_count": len(context.search_space.precision_gene_ids),
        "maximal_legal_parameterized_int8_gene_count": sum(
            "INT8" in group.allowed_precisions and not group.protected
            for group in groups
        ),
        "protected_parameterized_precision_group_count": len(protected),
        "protected_parameterized_precision_groups": protected,
        "multi_member_force_same_precision_group_count": sum(
            len(group.module_paths) > 1
            and bool(group.metadata.get("force_same_precision", True))
            for group in groups
        ),
        "pruning_scope_used_as_precision_group_count": 0,
        "functional_compute_contract": {
            "canonical_layer": "pyramid_backbone.functional_affine_grid_matmul",
            "precision_gene": False,
            "precision": "FP16",
            "reason": "parameter_free_geometry_torch_bmm_with_two_runtime_inputs; legacy_and_explicit_engines_realize_FP16",
            "canonical_weighted_count_accounting": "69 parameterized + 1 parameter-free functional compute = 70",
        },
        "domain_ranking_hashes": {
            domain.domain_id: domain.ranking_hash for domain in domains
        },
        "real_cuda_batched_proxy_smoke": proxy_smoke,
    }
    _write_json(output / "fixed_pruning_taylor_ranking.json", ranking_manifest)
    _write_json(
        output / "legal_pruning_domains.json",
        [domain.to_dict() for domain in domains],
    )
    _write_json(
        output / "quantization_precision_groups.json",
        [group.to_dict() for group in groups],
    )
    _write_json(output / "domain_width_search_space_audit.json", summary)
    _write_json(output / "real_cuda_batched_proxy_smoke.json", proxy_smoke)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
