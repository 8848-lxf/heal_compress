#!/usr/bin/env python3
"""Bounded real-checkpoint unified CNN/Transformer Greedy and GA smoke."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.audit_heal_transformer_search_models import MODEL_SPECS, _load, _sha256
from search.adapters.transformer_models import (
    TransformerSearchComponents,
    build_transformer_search_components,
    build_unified_transformer_search_space,
)
from search.candidate import CandidateGenotype
from search.canonicalization import canonicalize_candidate, repair_genotype
from search.ga.engine import GAConfig, GeneticSearchEngine
from search.greedy import GreedyBudgetSearch, GreedySearchConfig
from search.hashing import candidate_hash, canonical_json_hash
from search.proxy.bops_proxy import BOPSProxy
from search.proxy.fisher_proxy import collect_task_loss_fisher_statistics
from search.proxy.joint_weight_activation_taylor import (
    JointWeightActivationTaylorProxy,
    ModelCandidateOutputProvider,
    collect_joint_output_taylor_statistics,
    taylor_units_from_transformer_precision,
)
from search.proxy.parameter_slice_resolver import build_unit_parameter_slices
from search.proxy.runtime_shape_profiler import profile_runtime_layer_shapes
from search.proxy.size_proxy import SizeProxy
from search.proxy.transformer_bops import (
    TransformerBOPSProxy,
    UnifiedBOPSProxy,
    profile_projection_free_attention_workloads,
    profile_transformer_workloads,
)
from search.proxy.transformer_parameter_slices import (
    build_transformer_unit_parameter_slices,
)
from search.pruning_space.domain_importance import (
    score_atomic_units_for_fixed_ranking,
)
from search.pruning_space.local_domains import build_local_pruning_domains
from search.pruning_space.transformer_domains import (
    fixed_transformer_rankings_from_unit_scores,
)
from search.quantization_space.transformer_precision import (
    build_transformer_quantization_groups,
)
from search.quantization_space.types import QuantizationSearchGroup
from search.stage1.repair_selection import select_repaired_stage2_topk
from tracer.api import trace_model
from tracer.config import TraceConfig


CNN_ROOT_TYPES = (
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.ConvTranspose1d,
    nn.ConvTranspose2d,
    nn.ConvTranspose3d,
)


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f"refusing_to_overwrite_transformer_smoke_artifact:{path}")
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _merge_slices(*rows: Mapping[str, Sequence[Any]]) -> dict[str, list[Any]]:
    result: dict[str, list[Any]] = {}
    for mapping in rows:
        overlap = set(result).intersection(mapping)
        if overlap:
            raise RuntimeError(f"unified_proxy_slice_unit_overlap:{sorted(overlap)}")
        result.update({str(key): list(value) for key, value in mapping.items()})
    return result


def _next_lower(domain: Any) -> int:
    widths = tuple(int(value) for value in domain.legal_widths)
    position = widths.index(int(domain.original_width))
    if position <= 0:
        raise RuntimeError(f"smoke_domain_has_no_lower_width:{domain.domain_id}")
    return widths[position - 1]


def _choose_cnn_domain(domains: Sequence[Any]) -> Any:
    choices = [
        domain
        for domain in domains
        if len(domain.legal_widths) > 1
        and domain.domain_type in {"cnn_channel", "grouped_conv_channel"}
        and "backbone" in domain.root_module_path
    ]
    if not choices:
        choices = [domain for domain in domains if len(domain.legal_widths) > 1]
    if not choices:
        raise RuntimeError("real_runtime_trace_has_no_nontrivial_cnn_domain")
    return sorted(
        choices,
        key=lambda row: (
            row.domain_type != "cnn_channel",
            row.original_width,
            row.domain_id,
        ),
    )[0]


def _choose_transformer_domains(domains: Sequence[Any]) -> list[Any]:
    attention = [row for row in domains if row.domain_type == "attention_dh"]
    ffn = [row for row in domains if row.domain_type == "ffn_hidden"]
    selected: list[Any] = []
    if attention:
        preferred = [
            row
            for row in attention
            if "window" in row.family or "grid" in row.family
        ]
        selected.append(sorted(preferred or attention, key=lambda row: row.domain_id)[0])
    if ffn:
        if selected:
            same_block = [row for row in ffn if row.block_path == selected[0].block_path]
        else:
            same_block = []
        selected.append(sorted(same_block or ffn, key=lambda row: row.domain_id)[0])
    return selected


def _select_precision_units(
    components: TransformerSearchComponents,
    transformer_domains: Sequence[Any],
) -> tuple[Any, ...]:
    selected_paths = {row.module_path for row in transformer_domains}
    if not selected_paths and components.projection_free_attention:
        selected_paths.add(str(components.projection_free_attention[0]["module_path"]))
    rows = []
    for unit in components.precision_units:
        identity = str(unit.unit_id)
        if any(f"::{path}::" in identity for path in selected_paths):
            rows.append(unit)
    if selected_paths and not rows:
        raise RuntimeError(f"smoke_precision_units_missing:{sorted(selected_paths)}")
    return tuple(
        replace(unit, ordering=index) for index, unit in enumerate(rows)
    )


def _cnn_quantization_group(model: nn.Module, domain: Any) -> QuantizationSearchGroup:
    module = model.get_submodule(domain.root_module_path)
    parameter_count = sum(
        int(parameter.numel()) for parameter in module.parameters(recurse=False)
    )
    return QuantizationSearchGroup(
        group_id=f"cnn_precision::{domain.root_module_path}",
        module_paths=(domain.root_module_path,),
        canonical_node_ids=(domain.root_module_path,),
        allowed_precisions=("FP32", "FP16", "INT8"),
        protected=False,
        protection_reason="",
        ordering=0,
        parameter_count=parameter_count,
        baseline_macs=float(getattr(module, "weight").numel()),
        metadata={
            "source": "real_runtime_traced_cnn_smoke_domain",
            "default_precision": "FP32",
            "weight_activation_bound": True,
        },
    )


def _multi_agent_validation_batch(
    adapter: Any,
    hypes: Mapping[str, Any],
    device: torch.device,
    *,
    maximum_scan: int = 128,
) -> tuple[Any, int, int]:
    """Select a real validation frame with at least two agents."""

    from opencood.data_utils.datasets import build_dataset
    from search.integration.data_provider import move_batch_to_device

    dataset = build_dataset(
        adapter._absolutize_dataset_paths(dict(hypes)),
        visualize=True,
        train=False,
    )
    for index in range(min(len(dataset), int(maximum_scan))):
        item = dataset[index]
        batch = dataset.collate_batch_test([item])
        if batch is None:
            continue
        record_len = batch.get("ego", {}).get("record_len")
        if torch.is_tensor(record_len):
            agents = int(record_len.max().item())
        else:
            agents = max((int(value) for value in (record_len or [0])), default=0)
        if agents >= 2:
            return move_batch_to_device(batch, device), index, agents
    raise RuntimeError(
        f"real_validation_multi_agent_sample_not_found:first_{maximum_scan}"
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    if device.type != "cuda" or device.index not in {None, 0}:
        raise RuntimeError("transformer_unified_smoke_requires_visible_cuda0")
    torch.cuda.set_device(device)
    print(f"[{args.model}] load real checkpoint", flush=True)
    model, adapter, hypes, batch = _load(args.model, device)
    batch, dataset_index, agent_count = _multi_agent_validation_batch(
        adapter, hypes, device
    )
    spec = MODEL_SPECS[args.model]
    calibration_hash = canonical_json_hash(
        {
            "model": args.model,
            "config_sha256": _sha256(spec["config"]),
            "checkpoint_sha256": _sha256(spec["checkpoint"]),
            "split": "validation",
            "dataset_indices": [dataset_index],
            "agent_count": agent_count,
            "sample_count": 1,
        }
    )

    print(f"[{args.model}] trace CNN graph and runtime shapes", flush=True)
    trace = trace_model(
        model,
        batch,
        config=TraceConfig(fail_on_fx_trace_error=False),
        forward_fn=adapter.forward_for_task,
    )
    runtime = profile_runtime_layer_shapes(
        model, batch, forward_fn=adapter.forward_for_task
    )
    active_paths = sorted({row.module_path for row in runtime.shapes})
    modules = dict(model.named_modules())
    cnn_units = [
        unit
        for unit in trace.atomic_prune_units
        if not bool(unit.protected)
        and isinstance(modules.get(unit.root_module_path), CNN_ROOT_TYPES)
        and unit.root_axis in {"out", "channel"}
        and bool(unit.root_indices)
    ]
    preliminary_cnn_domains = build_local_pruning_domains(
        cnn_units,
        ranking_method="diagnostic_placeholder_before_common_task_loss_taylor",
        minimum_retained_ratio=0.10,
        dense_alignment=4,
    )
    selected_cnn_diagnostic = _choose_cnn_domain(preliminary_cnn_domains)
    selected_cnn_unit_ids = set(selected_cnn_diagnostic.ordered_unit_ids)
    selected_cnn_units = [
        unit for unit in cnn_units if unit.stable_id in selected_cnn_unit_ids
    ]

    diagnostic_components = build_transformer_search_components(
        model,
        hypes,
        allow_identity_ranking=True,
        active_module_paths=active_paths,
    )
    cnn_slices = build_unit_parameter_slices(model, selected_cnn_units)
    transformer_slices = build_transformer_unit_parameter_slices(
        model, diagnostic_components.transformer_domains
    )
    ranking_slices = _merge_slices(cnn_slices, transformer_slices)

    print(f"[{args.model}] collect common task-loss Fisher/Taylor", flush=True)
    fisher, fisher_report = collect_task_loss_fisher_statistics(
        model,
        (batch,),
        forward_fn=adapter.forward_for_task,
        loss_fn=adapter.compute_task_loss,
        calibration_manifest_hash=calibration_hash,
    )
    scores, ranking_report = score_atomic_units_for_fixed_ranking(
        model,
        fisher,
        ranking_slices,
        strict=False,
    )
    formal_cnn_domains = build_local_pruning_domains(
        selected_cnn_units,
        importance_scores=scores,
        ranking_method="raw_common_task_loss_first_plus_second_order_taylor",
        minimum_retained_ratio=0.10,
        dense_alignment=4,
    )
    if len(formal_cnn_domains) != 1:
        raise RuntimeError(f"selected_cnn_domain_rebuild_count:{len(formal_cnn_domains)}")
    attention_rankings, ffn_rankings, transformer_ranking_report = (
        fixed_transformer_rankings_from_unit_scores(
            diagnostic_components.attention_instances,
            diagnostic_components.ffn_instances,
            scores,
        )
    )
    formal_components = build_transformer_search_components(
        model,
        hypes,
        attention_rankings=attention_rankings,
        ffn_rankings=ffn_rankings,
        active_module_paths=active_paths,
    )
    formal_transformer_slices = build_transformer_unit_parameter_slices(
        model, formal_components.transformer_domains
    )
    all_slices = _merge_slices(cnn_slices, formal_transformer_slices)

    attention_workloads, ffn_workloads, workload_report = profile_transformer_workloads(
        model,
        batch,
        forward_fn=adapter.forward_for_task,
        attention_instances=formal_components.attention_instances,
        ffn_instances=formal_components.ffn_instances,
    )
    projection_free_paths = [
        str(row["module_path"]) for row in formal_components.projection_free_attention
    ]
    if projection_free_paths:
        projection_free_workloads, projection_free_report = (
            profile_projection_free_attention_workloads(
                model,
                batch,
                forward_fn=adapter.forward_for_task,
                module_paths=projection_free_paths,
            )
        )
    else:
        projection_free_workloads = ()
        projection_free_report = {
            "schema_version": "real-projection-free-attention-workload-v1",
            "workloads": [],
            "real_forward_executed": False,
        }

    selected_transformer = _choose_transformer_domains(
        formal_components.transformer_domains
    )
    selected_precision = _select_precision_units(
        formal_components, selected_transformer
    )
    smoke_components = replace(
        formal_components,
        transformer_domains=tuple(selected_transformer),
        precision_units=selected_precision,
        quantization_groups=build_transformer_quantization_groups(
            selected_precision
        ),
    )
    selected_cnn = formal_cnn_domains[0]
    space = build_unified_transformer_search_space(
        smoke_components,
        cnn_domains=(selected_cnn,),
        cnn_quantization_groups=(_cnn_quantization_group(model, selected_cnn),),
        pruning_unit_ids=[
            unit_id
            for domain in (selected_cnn, *selected_transformer)
            for unit_id in domain.ordered_unit_ids
        ],
        calibration_manifest_hash=calibration_hash,
        trace_snapshot_hash=trace.trace_hash,
        onnx_export_config_hash=canonical_json_hash(
            {"model": args.model, "fixed_shape_smoke": True}
        ),
        tensorrt_version="10.9",
        gpu_compute_capability=".".join(
            str(value) for value in torch.cuda.get_device_capability(device)
        ),
        builder_flags={
            "strongly_typed": True,
            "qk_fp32": True,
            "smoke_only": True,
        },
    )

    selected_taylor_units = taylor_units_from_transformer_precision(
        model, selected_precision, active_module_paths=active_paths
    )
    # CNN output perturbation participates in the same common loss by adding
    # the selected root output boundary and precision owner.
    from search.proxy.joint_weight_activation_taylor import TaylorDeploymentUnit

    selected_taylor_units = tuple(selected_taylor_units) + (
        TaylorDeploymentUnit(
            unit_id=f"cnn_output::{selected_cnn.root_module_path}",
            module_path=selected_cnn.root_module_path,
            unit_type=selected_cnn.domain_type,
            boundary="module_output",
            precision_owner=selected_cnn.root_module_path,
            quantizer_id=f"activation_quantizer::cnn::{selected_cnn.root_module_path}",
        ),
    )
    print(
        f"[{args.model}] collect {len(selected_taylor_units)} output Taylor boundaries",
        flush=True,
    )
    output_statistics = collect_joint_output_taylor_statistics(
        model,
        (batch,),
        forward_fn=adapter.forward_for_task,
        loss_fn=adapter.compute_task_loss,
        units=selected_taylor_units,
        calibration_manifest_hash=calibration_hash,
    )
    output_provider = ModelCandidateOutputProvider(
        model,
        (batch,),
        forward_fn=adapter.forward_for_task,
        units=selected_taylor_units,
        unit_to_parameter_slices=all_slices,
        calibration_manifest_hash=calibration_hash,
    )
    joint_proxy = JointWeightActivationTaylorProxy(
        statistics=output_statistics,
        output_provider=output_provider,
        units=selected_taylor_units,
    )
    transformer_bops = TransformerBOPSProxy(
        formal_components.transformer_domains,
        attention_workloads=attention_workloads,
        ffn_workloads=ffn_workloads,
        projection_free_workloads=projection_free_workloads,
    )
    transformer_weighted_paths = {
        path
        for domain in formal_components.transformer_domains
        for member in domain.dependency_members
        for path in (str(member["module_path"]),)
    }
    cnn_bops = BOPSProxy(
        model=model,
        unit_to_parameter_slices=all_slices,
        runtime_shapes=runtime.shapes,
        exclude_module_paths=tuple(transformer_weighted_paths),
        default_precision="FP32",
    )
    bops_proxy = UnifiedBOPSProxy(transformer_bops, cnn_bops)
    size_proxy = SizeProxy(
        model=model,
        unit_to_parameter_slices=all_slices,
        default_precision="FP32",
        include_constant_parameters_in_size=True,
    )

    metric_cache: dict[str, dict[str, Any]] = {}

    def base_metrics(genotype: CandidateGenotype) -> dict[str, Any]:
        phenotype = canonicalize_candidate(genotype, space)
        identity = candidate_hash(phenotype, space)
        cached = metric_cache.get(identity)
        if cached is not None:
            return dict(cached)
        print(f"[{args.model}] proxy replay {len(metric_cache) + 1}: {identity[:12]}", flush=True)
        joint = joint_proxy.evaluate_breakdown(phenotype)
        bops = bops_proxy.evaluate_breakdown(phenotype)
        size = size_proxy.evaluate_breakdown(phenotype)
        row = {
            **joint,
            **{key: value for key, value in bops.items() if key != "breakdown"},
            "bops_breakdown": bops["breakdown"],
            **size,
            "mixed_weight_size_bytes": float(size["size_bits_total"]) / 8.0,
            "candidate_hash": identity,
            "phenotype": phenotype.to_dict(),
            "F1": float(joint["L_joint_weight_activation_taylor"]),
            "proxy_score_raw": float(joint["L_joint_weight_activation_taylor"]),
            "activation_taylor_included": True,
        }
        metric_cache[identity] = row
        return dict(row)

    probe = GreedyBudgetSearch(
        space,
        config=GreedySearchConfig(bops_targets=(0.99,), maximum_steps=1),
    )
    initial = probe._initial_candidate()
    neighbors = probe._neighbors(initial)
    if not neighbors:
        raise RuntimeError("unified_smoke_has_no_legal_neighbor")
    initial_metrics = base_metrics(initial)
    neighbor_metrics = [base_metrics(candidate) for candidate, _action in neighbors]
    positive = [
        (candidate, action, metrics)
        for (candidate, action), metrics in zip(neighbors, neighbor_metrics)
        if float(metrics["R_bops_vs_fp32"])
        < float(initial_metrics["R_bops_vs_fp32"]) - 1.0e-12
    ]
    if not positive:
        raise RuntimeError("unified_smoke_has_no_positive_bops_action")
    target_candidate, target_action, target_metrics = min(
        positive,
        key=lambda row: (
            float(row[2]["L_joint_weight_activation_taylor"])
            / max(
                float(initial_metrics["R_bops_vs_fp32"])
                - float(row[2]["R_bops_vs_fp32"]),
                1.0e-12,
            ),
            str(row[1]["gene_id"]),
        ),
    )
    target = float(target_metrics["R_bops_vs_fp32"])
    tolerance = float(args.bops_tolerance)

    def evaluate(candidates: Sequence[CandidateGenotype], _step: int) -> list[dict[str, Any]]:
        rows = []
        for candidate in candidates:
            row = base_metrics(candidate)
            deviation = abs(float(row["R_bops_vs_fp32"]) - target)
            violation = max(0.0, deviation - tolerance)
            row.update(
                {
                    "BOPS_target": target,
                    "bops_target": target,
                    "bops_abs_delta": deviation,
                    "bops_deviation": deviation,
                    "bops_tolerance_abs": tolerance,
                    "bops_violation": violation,
                    "bops_feasible": violation <= 0.0,
                    "F1": float(row["L_joint_weight_activation_taylor"]),
                }
            )
            rows.append(row)
        return rows

    print(f"[{args.model}] greedy target={target:.9f}", flush=True)
    greedy = GreedyBudgetSearch(
        space,
        config=GreedySearchConfig(
            bops_targets=(target,),
            bops_tolerance_abs=tolerance,
            maximum_steps=int(args.greedy_steps),
            budget_recovery_max_depth=4,
            budget_recovery_beam_width=4,
            budget_recovery_seed_pool_size=8,
        ),
    ).run(evaluate)
    if target not in greedy.budget_candidates:
        raise RuntimeError("unified_greedy_smoke_budget_not_captured")

    print(f"[{args.model}] GA population={args.ga_population} generations={args.ga_generations}", flush=True)
    ga = GeneticSearchEngine(
        space,
        GAConfig(
            initial_population_size=int(args.ga_population),
            population_size=int(args.ga_population),
            offspring_size=int(args.ga_population),
            num_generations=int(args.ga_generations),
            constraint_first_ranking=True,
            random_seed=20260723,
            mutation_action_min=1,
            mutation_action_max=2,
        ),
    )
    ga_rows = ga.run(
        batch_evaluator=lambda candidates, generation: evaluate(candidates, generation),
        seed_candidates=[greedy.budget_candidates[target], target_candidate],
        candidate_key_fn=lambda genotype: candidate_hash(
            canonicalize_candidate(genotype, space), space
        ),
    )
    feasible = [row for row in ga_rows if bool(row[2]["bops_feasible"])]
    if not feasible:
        raise RuntimeError("unified_ga_smoke_has_no_feasible_candidate")

    selected, topk_report = select_repaired_stage2_topk(
        ga_rows,
        space=space,
        repair_fn=lambda genotype: (repair_genotype(genotype, space), {"status": "ok"}),
        rescore_fn=lambda phenotype: evaluate(
            [
                CandidateGenotype(
                    pruning_width_genes=dict(
                        phenotype.metadata.get("domain_width_profile") or {}
                    ),
                    precision_genes={
                        group.group_id: phenotype.realized_precision_profile[
                            group.module_paths[0]
                        ]
                        for group in space.quantization_groups
                    },
                )
            ],
            -1,
        )[0],
        topk=min(2, len(feasible)),
        repair_pool_size=max(8, int(args.ga_population)),
        selection_policy="three_plus_two_diversity",
        exploitation_count=1,
        diversity_count=1,
        eligibility_fn=lambda metrics: abs(
            float(metrics["R_bops_vs_fp32"]) - target
        ) <= tolerance,
    )
    if not selected:
        raise RuntimeError("unified_stage2_preselection_empty")

    inventory = {
        "schema_version": "real-unified-transformer-search-space-smoke-v1",
        "model": args.model,
        "canonical_name": spec["canonical_name"],
        "real_validation_dataset_index": dataset_index,
        "real_validation_agent_count": agent_count,
        "trace_backend": trace.config["realized_backend"],
        "trace_coverage": trace.trace_coverage.to_dict(),
        "cnn_domain_count": len(preliminary_cnn_domains),
        "attention_domain_count": sum(
            row.domain_type == "attention_dh"
            for row in formal_components.transformer_domains
        ),
        "ffn_domain_count": sum(
            row.domain_type == "ffn_hidden"
            for row in formal_components.transformer_domains
        ),
        "projection_free_attention_count": len(
            formal_components.projection_free_attention
        ),
        "selected_smoke_domains": [
            row.to_dict() for row in (selected_cnn, *selected_transformer)
        ],
        "selected_precision_units": [row.to_dict() for row in selected_precision],
        "all_transformer_components": formal_components.to_dict(),
    }
    _write(output / "inventory.json", inventory)
    _write(output / "runtime_shapes.json", runtime.to_dict())
    _write(output / "workloads.json", {
        "transformer": workload_report,
        "projection_free": projection_free_report,
    })
    _write(output / "fisher_manifest.json", fisher_report)
    _write(output / "ranking_manifest.json", {
        "atomic": ranking_report,
        "transformer": transformer_ranking_report,
    })
    _write(output / "joint_activation_proxy_manifest.json", {
        **output_statistics.to_manifest(),
        "deployment_units": [row.to_dict() for row in selected_taylor_units],
    })
    _write(output / "greedy_smoke.json", greedy.to_dict())
    _write(output / "ga_smoke.json", {
        "population": int(args.ga_population),
        "generations": int(args.ga_generations),
        "generation_statistics": ga.generation_statistics,
        "evaluated_row_count": len(ga_rows),
        "unique_proxy_replay_count": len(metric_cache),
        "feasible_count": len(feasible),
        "rows": [
            {
                "genotype": genotype.to_dict(),
                "score": score,
                "metrics": metrics,
            }
            for genotype, score, metrics in ga_rows
        ],
    })
    _write(output / "stage2_preselection.json", {
        "report": topk_report,
        "selected": [
            {
                "candidate_hash": row.candidate_hash,
                "genotype": row.genotype.to_dict(),
                "phenotype": row.phenotype.to_dict(),
                "metrics": row.metrics,
            }
            for row in selected
        ],
    })
    acceptance = {
        "schema_version": "real-unified-transformer-search-smoke-acceptance-v1",
        "model": args.model,
        "passed": True,
        "strict_checkpoint_load": True,
        "real_task_loss_fisher": True,
        "joint_weight_activation_taylor": True,
        "softmax_activation_proxy": any(
            row.unit_type == "softmax" for row in selected_taylor_units
        ),
        "av_activation_proxy": any(
            row.unit_type == "av_matmul" for row in selected_taylor_units
        ),
        "cnn_domain_in_smoke": True,
        "attention_domain_in_smoke": any(
            row.domain_type == "attention_dh" for row in selected_transformer
        ),
        "ffn_domain_in_smoke": any(
            row.domain_type == "ffn_hidden" for row in selected_transformer
        ),
        "projection_free_attention_semantics": bool(
            formal_components.projection_free_attention
        ),
        "greedy_budget_target": target,
        "greedy_budget_captured": target in greedy.budget_candidates,
        "ga_feasible_count": len(feasible),
        "stage2_preselection_count": len(selected),
        "full1789_executed": False,
        "formal_full_search_executed": False,
        "onnx_executed": False,
        "tensorrt_executed": False,
    }
    _write(output / "acceptance.json", acceptance)
    print(json.dumps(acceptance, indent=2, sort_keys=True), flush=True)
    return acceptance


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(MODEL_SPECS), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ga-population", type=int, default=8)
    parser.add_argument("--ga-generations", type=int, default=2)
    parser.add_argument("--greedy-steps", type=int, default=2)
    parser.add_argument("--bops-tolerance", type=float, default=0.005)
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
