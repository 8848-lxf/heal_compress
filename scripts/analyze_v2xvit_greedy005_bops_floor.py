#!/usr/bin/env python3
"""Build the full real V2X-ViT space and audit the 0.05 BOPS floor.

This is a cost-only prerequisite to the formal Greedy trajectory.  It never
uses repair or Taylor scores to move a candidate into the requested band.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.audit_heal_transformer_search_models import MODEL_SPECS, _load, _sha256
from scripts.smoke_transformer_unified_search import (
    CNN_ROOT_TYPES,
    _cnn_quantization_group,
    _merge_slices,
    _multi_agent_validation_batch,
)
from search.adapters.transformer_models import (
    build_transformer_search_components,
    build_unified_transformer_search_space,
)
from search.candidate import CandidateGenotype
from search.canonicalization import canonicalize_candidate, repair_genotype
from search.greedy import GreedyBudgetSearch, GreedySearchConfig
from search.hashing import candidate_hash, canonical_json_hash
from search.proxy.bops_proxy import BOPSProxy
from search.proxy.parameter_slice_resolver import build_unit_parameter_slices
from search.proxy.runtime_shape_profiler import profile_runtime_layer_shapes
from search.proxy.size_proxy import SizeProxy
from search.proxy.transformer_bops import (
    TransformerBOPSProxy,
    UnifiedBOPSProxy,
    profile_transformer_workloads,
)
from search.proxy.transformer_latency import (
    TransformerLatencyLUT,
    TransformerLatencyProxy,
)
from search.proxy.transformer_parameter_slices import (
    build_transformer_unit_parameter_slices,
)
from search.pruning_space.local_domains import build_local_pruning_domains
from search.quantization_space.transformer_precision import (
    build_transformer_quantization_groups,
)
from tracer.api import trace_model
from tracer.config import TraceConfig


TARGET = 0.05
TOLERANCE = 0.005


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["status"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows or [{"status": "empty"}])


def _precision_value(group: Any, *, minimum: bool) -> str:
    ordered = [
        value
        for value in ("FP32", "FP16", "INT8")
        if value in group.allowed_precisions
    ]
    if not ordered:
        raise RuntimeError(f"precision_group_has_no_contract_state:{group.group_id}")
    return ordered[-1] if minimum else ordered[0]


def _genotype_payload(candidate: CandidateGenotype) -> dict[str, Any]:
    return {
        "pruning_width_genes": dict(sorted(candidate.pruning_width_genes.items())),
        "precision_genes": dict(sorted(candidate.precision_genes.items())),
    }


def _build_full_space(model: nn.Module, adapter: Any, hypes: Mapping[str, Any], batch: Any):
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
    cnn_domains = build_local_pruning_domains(
        cnn_units,
        ranking_method="identity_only_for_cost_floor_before_taylor_ranking",
        minimum_retained_ratio=0.10,
        dense_alignment=4,
    )
    components = build_transformer_search_components(
        model,
        hypes,
        allow_identity_ranking=True,
        active_module_paths=active_paths,
    )
    transformer_groups = build_transformer_quantization_groups(
        components.precision_units
    )
    cnn_groups = tuple(
        replace(_cnn_quantization_group(model, domain), ordering=index)
        for index, domain in enumerate(cnn_domains)
    )
    transformer_groups = tuple(
        replace(group, ordering=len(cnn_groups) + index)
        for index, group in enumerate(transformer_groups)
    )
    components = replace(
        components,
        quantization_groups=transformer_groups,
    )
    calibration_hash = canonical_json_hash(
        {
            "model": "lidar_v2xvit",
            "purpose": "bops_floor_analysis",
            "sample_count": 1,
        }
    )
    all_domains = tuple(cnn_domains) + tuple(components.transformer_domains)
    space = build_unified_transformer_search_space(
        components,
        cnn_domains=cnn_domains,
        cnn_quantization_groups=cnn_groups,
        pruning_unit_ids=[
            unit_id for domain in all_domains for unit_id in domain.ordered_unit_ids
        ],
        calibration_manifest_hash=calibration_hash,
        trace_snapshot_hash=trace.trace_hash,
        onnx_export_config_hash=canonical_json_hash(
            {"model": "lidar_v2xvit", "fixed_shape": True}
        ),
        tensorrt_version="10.9",
        gpu_compute_capability=".".join(
            str(value) for value in torch.cuda.get_device_capability(torch.device("cuda:0"))
        ),
        builder_flags={"strongly_typed": True, "qk_fp32": True},
    )
    cnn_slices = build_unit_parameter_slices(model, cnn_units)
    transformer_slices = build_transformer_unit_parameter_slices(
        model, components.transformer_domains
    )
    all_slices = _merge_slices(cnn_slices, transformer_slices)
    attention_workloads, ffn_workloads, workload_report = (
        profile_transformer_workloads(
            model,
            batch,
            forward_fn=adapter.forward_for_task,
            attention_instances=components.attention_instances,
            ffn_instances=components.ffn_instances,
        )
    )
    transformer_bops = TransformerBOPSProxy(
        components.transformer_domains,
        attention_workloads=attention_workloads,
        ffn_workloads=ffn_workloads,
    )
    transformer_weighted_paths = {
        str(member["module_path"])
        for domain in components.transformer_domains
        for member in domain.dependency_members
    }
    cnn_bops = BOPSProxy(
        model=model,
        unit_to_parameter_slices=all_slices,
        runtime_shapes=runtime.shapes,
        exclude_module_paths=tuple(sorted(transformer_weighted_paths)),
        default_precision="FP32",
    )
    return {
        "trace": trace,
        "runtime": runtime,
        "cnn_units": tuple(cnn_units),
        "cnn_domains": tuple(cnn_domains),
        "components": components,
        "space": space,
        "slices": all_slices,
        "bops": UnifiedBOPSProxy(transformer_bops, cnn_bops),
        "transformer_bops": transformer_bops,
        "size": SizeProxy(
            model=model,
            unit_to_parameter_slices=all_slices,
            default_precision="FP32",
            include_constant_parameters_in_size=True,
        ),
        "workload_report": workload_report,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_root.resolve() / "greedy"
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type != "cuda" or device.index not in {None, 0}:
        raise RuntimeError("v2xvit_floor_requires_visible_cuda0")
    torch.cuda.set_device(device)
    model, adapter, hypes, _unused = _load("v2xvit", device)
    batch, dataset_index, agent_count = _multi_agent_validation_batch(
        adapter, hypes, device
    )
    built = _build_full_space(model, adapter, hypes, batch)
    space = built["space"]
    domains = tuple(space.pruning_domains)
    groups = {group.group_id: group for group in space.quantization_groups}
    type_counts = {
        domain_type: sum(domain.domain_type == domain_type for domain in domains)
        for domain_type in (
            "cnn_channel",
            "grouped_conv_channel",
            "attention_dh",
            "ffn_hidden",
        )
    }
    if type_counts["cnn_channel"] + type_counts["grouped_conv_channel"] != 20:
        raise RuntimeError(f"v2xvit_cnn_domain_count_conflict:{type_counts}")
    if type_counts["attention_dh"] != 12 or type_counts["ffn_hidden"] != 3:
        raise RuntimeError(f"v2xvit_transformer_domain_count_conflict:{type_counts}")

    baseline = repair_genotype(
        CandidateGenotype(
            pruning_width_genes={
                domain.domain_id: int(domain.original_width) for domain in domains
            },
            precision_genes={
                group_id: _precision_value(groups[group_id], minimum=False)
                for group_id in space.precision_gene_ids
            },
            meta={"created_by": "strict_b0_w32a32"},
        ),
        space,
    )
    floor = repair_genotype(
        CandidateGenotype(
            pruning_width_genes={
                domain.domain_id: int(min(domain.legal_widths)) for domain in domains
            },
            precision_genes={
                group_id: _precision_value(groups[group_id], minimum=True)
                for group_id in space.precision_gene_ids
            },
            meta={"created_by": "strict_discrete_minimum"},
        ),
        space,
    )
    baseline_pheno = canonicalize_candidate(baseline, space)
    floor_pheno = canonicalize_candidate(floor, space)
    baseline_bops = built["bops"].evaluate_breakdown(baseline_pheno)
    floor_bops = built["bops"].evaluate_breakdown(floor_pheno)
    baseline_size = built["size"].evaluate_breakdown(baseline_pheno)
    floor_size = built["size"].evaluate_breakdown(floor_pheno)
    floor_retention = float(floor_bops["R_bops_vs_fp32"])
    if abs(float(baseline_bops["R_bops_vs_fp32"]) - 1.0) > 1.0e-12:
        raise RuntimeError(
            f"v2xvit_strict_baseline_not_unity:{baseline_bops['R_bops_vs_fp32']}"
        )

    domain_rows = []
    for domain in domains:
        domain_rows.append(
            {
                "domain_id": domain.domain_id,
                "domain_type": domain.domain_type,
                "module_path": domain.module_path,
                "family": domain.family,
                "original_width": domain.original_width,
                "minimum_legal_width": min(domain.legal_widths),
                "legal_widths": "|".join(str(value) for value in domain.legal_widths),
                "minimum_precision": "",
                "protected": False,
                "constant_contract": False,
            }
        )
    for group in space.quantization_groups:
        domain_rows.append(
            {
                "domain_id": group.group_id,
                "domain_type": "precision_unit",
                "module_path": "|".join(group.module_paths),
                "family": str(group.metadata.get("family", "")),
                "original_width": "",
                "minimum_legal_width": "",
                "legal_widths": "",
                "minimum_precision": _precision_value(group, minimum=True),
                "protected": bool(group.protected),
                "constant_contract": group.group_id in space.constant_precision_group_ids,
            }
        )
    _write_csv(output / "v2xvit_domain_minimum_contract.csv", domain_rows)

    floor_report = {
        "schema_version": "v2xvit-bops-floor-analysis-v1",
        "model": "lidar_v2xvit",
        "real_checkpoint": str(MODEL_SPECS["v2xvit"]["checkpoint"]),
        "checkpoint_sha256": _sha256(MODEL_SPECS["v2xvit"]["checkpoint"]),
        "config": str(MODEL_SPECS["v2xvit"]["config"]),
        "config_sha256": _sha256(MODEL_SPECS["v2xvit"]["config"]),
        "dataset_index": dataset_index,
        "agent_count": agent_count,
        "domain_type_counts": type_counts,
        "variable_precision_locus_count": len(space.precision_gene_ids),
        "constant_precision_group_count": len(space.constant_precision_group_ids),
        "protected_units": [
            {
                "group_id": group.group_id,
                "role": group.metadata.get("transformer_role", ""),
                "allowed_precisions": list(group.allowed_precisions),
                "protection_reason": group.protection_reason,
            }
            for group in space.quantization_groups
            if group.protected
        ],
        "baseline": {
            "candidate_hash": candidate_hash(baseline_pheno, space),
            "genotype": _genotype_payload(baseline),
            "bops": baseline_bops,
            "size": baseline_size,
        },
        "discrete_minimum": {
            "candidate_hash": candidate_hash(floor_pheno, space),
            "genotype": _genotype_payload(floor),
            "bops": floor_bops,
            "size": floor_size,
        },
        "original_bops": float(baseline_bops["bops_total"]),
        "minimum_bops": float(floor_bops["bops_total"]),
        "minimum_bops_retention": floor_retention,
        "target": TARGET,
        "tolerance_abs": TOLERANCE,
        "target_band": [TARGET - TOLERANCE, TARGET + TOLERANCE],
        "target_in_theoretical_reachable_interval": floor_retention
        <= TARGET + TOLERANCE,
        "qk_fp32_included": all(
            row.get("compute_precision") == "FP32"
            for row in floor_bops["breakdown"]
            if row.get("component") == "qk_matmul"
        ),
        "softmax_activation_cost_included": any(
            row.get("component") == "softmax"
            and row.get("weight_bits") is None
            for row in floor_bops["breakdown"]
        ),
        "structural_repair_count": 0,
        "precision_repair_count": 0,
        "budget_projection_count": 0,
    }
    _write_json(output / "v2xvit_bops_floor_analysis.json", floor_report)

    # Cost-only legal-action descent produces a concrete discrete witness.  It
    # is deliberately separate from the Taylor-ranked formal Greedy run.
    enumerator = GreedyBudgetSearch(
        space,
        config=GreedySearchConfig(
            bops_targets=(TARGET,),
            bops_tolerance_abs=TOLERANCE,
            run_to_exhaustion=True,
            enable_budget_recovery=False,
        ),
    )
    current = baseline
    current_metrics = baseline_bops
    witness_rows: list[dict[str, Any]] = []
    cost_path: list[dict[str, Any]] = []
    max_steps = sum(len(domain.legal_widths) - 1 for domain in domains) + 2 * len(
        space.precision_gene_ids
    )
    for step in range(max_steps + 1):
        retention = float(current_metrics["R_bops_vs_fp32"])
        if TARGET - TOLERANCE <= retention <= TARGET + TOLERANCE:
            witness_rows.append(
                {
                    "step": step,
                    "candidate": current,
                    "metrics": current_metrics,
                }
            )
            break
        neighbors = enumerator._neighbors(current)
        if not neighbors:
            break
        rows = []
        for candidate, action in neighbors:
            phenotype = canonicalize_candidate(candidate, space)
            metrics = built["bops"].evaluate_breakdown(phenotype)
            after = float(metrics["R_bops_vs_fp32"])
            if after >= TARGET - TOLERANCE:
                rows.append((candidate, action, metrics))
        if not rows:
            break
        selected = min(
            rows,
            key=lambda row: (
                abs(float(row[2]["R_bops_vs_fp32"]) - TARGET),
                float(row[2]["R_bops_vs_fp32"]),
                str(row[1]["gene_id"]),
            ),
        )
        candidate, action, metrics = selected
        cost_path.append(
            {
                "step": step + 1,
                "action": action,
                "candidate_hash": candidate_hash(
                    canonicalize_candidate(candidate, space), space
                ),
                "bops_retention": metrics["R_bops_vs_fp32"],
            }
        )
        current, current_metrics = candidate, metrics

    witness = witness_rows[0] if witness_rows else None
    reachability = {
        "schema_version": "v2xvit-budget-reachability-v1",
        "model": "lidar_v2xvit",
        "target": TARGET,
        "tolerance_abs": TOLERANCE,
        "target_band": [TARGET - TOLERANCE, TARGET + TOLERANCE],
        "minimum_bops_retention": floor_retention,
        "target_in_theoretical_reachable_interval": floor_retention
        <= TARGET + TOLERANCE,
        "discrete_candidate_exists": witness is not None,
        "evidence": "concrete_legal_adjacent_action_witness"
        if witness is not None
        else "no_witness_on_deterministic_cost_only_path",
        "witness": None
        if witness is None
        else {
            "step": witness["step"],
            "bops_retention": witness["metrics"]["R_bops_vs_fp32"],
            "budget_deviation": abs(
                float(witness["metrics"]["R_bops_vs_fp32"]) - TARGET
            ),
            "candidate_hash": candidate_hash(
                canonicalize_candidate(witness["candidate"], space), space
            ),
            "genotype": _genotype_payload(witness["candidate"]),
        },
        "cost_only_path": cost_path,
        "cost_only_path_is_formal_greedy": False,
        "joint_taylor_used": False,
        "repair_counts": {
            "structural": 0,
            "precision": 0,
            "budget_projection": 0,
        },
    }
    _write_json(output / "v2xvit_budget_reachability.json", reachability)
    _write_json(
        output / "v2xvit_floor_space_manifest.json",
        {
            "domain_count": len(domains),
            "domains": [domain.to_dict() for domain in domains],
            "precision_groups": [
                group.to_dict() for group in space.quantization_groups
            ],
            "runtime_shapes": built["runtime"].to_dict(),
            "transformer_workloads": built["workload_report"],
            "trace_hash": built["trace"].trace_hash,
        },
    )
    print(
        json.dumps(
            {
                "original_bops": floor_report["original_bops"],
                "minimum_bops_retention": floor_retention,
                "discrete_candidate_exists": witness is not None,
                "witness_retention": None
                if witness is None
                else witness["metrics"]["R_bops_vs_fp32"],
                "domain_type_counts": type_counts,
                "variable_precision_locus_count": len(space.precision_gene_ids),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return reachability


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
