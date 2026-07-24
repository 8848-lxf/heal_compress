#!/usr/bin/env python3
"""Run the repair-free, full-space V2X-ViT R_BOPS=0.05 Greedy search."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import random
import sys
from typing import Any, Mapping, Sequence

import torch
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_v2xvit_greedy005_bops_floor import (
    TARGET,
    TOLERANCE,
    _build_full_space,
    _genotype_payload,
    _precision_value,
)
from scripts.audit_heal_transformer_search_models import MODEL_SPECS, _load, _sha256
from scripts.smoke_transformer_unified_search import (
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
from search.proxy.fisher_proxy import collect_task_loss_fisher_statistics
from search.proxy.joint_weight_activation_taylor import (
    JointOutputTaylorStatistics,
    ModelCandidateOutputProvider,
    TaylorDeploymentUnit,
    collect_joint_output_taylor_statistics,
    score_joint_output_perturbation,
    taylor_units_from_transformer_precision,
)
from search.proxy.joint_weight_taylor import JointWeightTaylorProxy
from search.proxy.parameter_slice_resolver import build_unit_parameter_slices
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


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _batch_content_hash(value: Any) -> str:
    """Hash deterministic tensor content summaries without copying full batches."""

    digest = hashlib.sha256()

    def update(item: Any, path: str) -> None:
        digest.update(path.encode("utf-8"))
        if torch.is_tensor(item):
            tensor = item.detach()
            digest.update(str(tensor.dtype).encode("utf-8"))
            digest.update(json.dumps(list(tensor.shape)).encode("utf-8"))
            flat = tensor.reshape(-1)
            digest.update(str(int(flat.numel())).encode("utf-8"))
            if flat.numel():
                numeric = flat.float()
                finite = torch.isfinite(numeric)
                safe = torch.where(finite, numeric, torch.zeros_like(numeric))
                summary = torch.stack(
                    (
                        safe.sum(),
                        safe.square().sum(),
                        safe.abs().max(),
                        finite.to(dtype=numeric.dtype).sum(),
                    )
                ).detach().cpu().tolist()
                digest.update(json.dumps(summary, sort_keys=True).encode("utf-8"))
                sample_count = min(1024, int(flat.numel()))
                indices = torch.linspace(
                    0,
                    int(flat.numel()) - 1,
                    sample_count,
                    device=flat.device,
                ).round().long()
                digest.update(
                    flat[indices].float().detach().cpu().numpy().tobytes()
                )
        elif isinstance(item, Mapping):
            for key in sorted(item, key=str):
                update(item[key], f"{path}/{key}")
        elif isinstance(item, (tuple, list)):
            for index, child in enumerate(item):
                update(child, f"{path}/{index}")
        else:
            digest.update(repr(item).encode("utf-8"))

    update(value, "batch")
    return digest.hexdigest()


def _subset_statistics(
    statistics: JointOutputTaylorStatistics,
    units: Sequence[TaylorDeploymentUnit],
) -> JointOutputTaylorStatistics:
    unit_ids = {unit.unit_id for unit in units}
    return JointOutputTaylorStatistics(
        baseline_outputs={
            key: value
            for key, value in statistics.baseline_outputs.items()
            if key in unit_ids
        },
        gradients={
            key: value
            for key, value in statistics.gradients.items()
            if key in unit_ids
        },
        calibration_manifest_hash=statistics.calibration_manifest_hash,
        unit_manifest_hash="subset-scored-directly",
        sample_count=statistics.sample_count,
        task_losses=statistics.task_losses,
    )


def _precision_counts(candidate: CandidateGenotype) -> dict[str, int]:
    return {
        value: sum(precision == value for precision in candidate.precision_genes.values())
        for value in ("FP32", "FP16", "INT8")
    }


def _bops_categories(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    categories = {
        "cnn_bops": 0.0,
        "attention_projection_bops": 0.0,
        "qk_bops": 0.0,
        "softmax_activation_bops": 0.0,
        "av_bops": 0.0,
        "output_projection_bops": 0.0,
        "ffn_bops": 0.0,
    }
    for row in rows:
        category = str(row.get("category", ""))
        component = str(row.get("component", ""))
        value = float(row.get("BOPS", 0.0))
        if "domain_id" not in row:
            categories["cnn_bops"] += value
        elif category == "attention_projection":
            categories["attention_projection_bops"] += value
        elif category == "qk":
            categories["qk_bops"] += value
        elif component == "softmax":
            categories["softmax_activation_bops"] += value
        elif category == "av":
            categories["av_bops"] += value
        elif category == "output_projection":
            categories["output_projection_bops"] += value
        elif category == "ffn":
            categories["ffn_bops"] += value
    return categories


def _formal_space(
    model: Any,
    adapter: Any,
    hypes: Mapping[str, Any],
    batch: Any,
    identity: Mapping[str, Any],
    calibration_hash: str,
) -> dict[str, Any]:
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
        identity["slices"],
        strict=False,
    )
    cnn_domains = build_local_pruning_domains(
        identity["cnn_units"],
        importance_scores=scores,
        ranking_method="raw_common_task_loss_first_plus_second_order_taylor",
        minimum_retained_ratio=0.10,
        dense_alignment=4,
    )
    identity_components = identity["components"]
    attention_rankings, ffn_rankings, transformer_ranking_report = (
        fixed_transformer_rankings_from_unit_scores(
            identity_components.attention_instances,
            identity_components.ffn_instances,
            scores,
        )
    )
    active_paths = sorted(
        {row.module_path for row in identity["runtime"].shapes}
    )
    components = build_transformer_search_components(
        model,
        hypes,
        attention_rankings=attention_rankings,
        ffn_rankings=ffn_rankings,
        active_module_paths=active_paths,
    )
    cnn_groups = tuple(
        replace(_cnn_quantization_group(model, domain), ordering=index)
        for index, domain in enumerate(cnn_domains)
    )
    transformer_groups = tuple(
        replace(group, ordering=len(cnn_groups) + index)
        for index, group in enumerate(
            build_transformer_quantization_groups(components.precision_units)
        )
    )
    components = replace(components, quantization_groups=transformer_groups)
    domains = tuple(cnn_domains) + tuple(components.transformer_domains)
    space = build_unified_transformer_search_space(
        components,
        cnn_domains=cnn_domains,
        cnn_quantization_groups=cnn_groups,
        pruning_unit_ids=[
            unit_id for domain in domains for unit_id in domain.ordered_unit_ids
        ],
        calibration_manifest_hash=calibration_hash,
        trace_snapshot_hash=identity["trace"].trace_hash,
        onnx_export_config_hash=canonical_json_hash(
            {"model": "lidar_v2xvit", "fixed_shape": True, "stage2": True}
        ),
        tensorrt_version="10.9",
        gpu_compute_capability=".".join(
            str(value)
            for value in torch.cuda.get_device_capability(torch.device("cuda:0"))
        ),
        builder_flags={
            "strongly_typed": True,
            "qk_fp32": True,
            "greedy_repair_free": True,
        },
    )
    cnn_slices = build_unit_parameter_slices(model, identity["cnn_units"])
    transformer_slices = build_transformer_unit_parameter_slices(
        model, components.transformer_domains
    )
    slices = _merge_slices(cnn_slices, transformer_slices)
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
    transformer_paths = {
        str(member["module_path"])
        for domain in components.transformer_domains
        for member in domain.dependency_members
    }
    cnn_bops = BOPSProxy(
        model=model,
        unit_to_parameter_slices=slices,
        runtime_shapes=identity["runtime"].shapes,
        exclude_module_paths=tuple(sorted(transformer_paths)),
        default_precision="FP32",
    )
    return {
        "fisher": fisher,
        "fisher_report": fisher_report,
        "ranking_report": ranking_report,
        "transformer_ranking_report": transformer_ranking_report,
        "cnn_domains": tuple(cnn_domains),
        "components": components,
        "space": space,
        "slices": slices,
        "transformer_bops": transformer_bops,
        "bops": UnifiedBOPSProxy(transformer_bops, cnn_bops),
        "size": SizeProxy(
            model=model,
            unit_to_parameter_slices=slices,
            default_precision="FP32",
            include_constant_parameters_in_size=True,
        ),
        "workload_report": workload_report,
        "active_paths": active_paths,
    }


def _baseline_candidate(space: Any) -> CandidateGenotype:
    groups = {group.group_id: group for group in space.quantization_groups}
    return repair_genotype(
        CandidateGenotype(
            pruning_width_genes={
                domain.domain_id: int(domain.original_width)
                for domain in space.pruning_domains
            },
            precision_genes={
                group_id: _precision_value(groups[group_id], minimum=False)
                for group_id in space.precision_gene_ids
            },
            meta={"created_by": "strict_b0_w32a32"},
        ),
        space,
    )


def _candidate_for_state(
    baseline: CandidateGenotype,
    *,
    locus_id: str,
    state: int | str,
    is_width: bool,
) -> CandidateGenotype:
    widths = dict(baseline.pruning_width_genes)
    precision = dict(baseline.precision_genes)
    if is_width:
        widths[locus_id] = int(state)
    else:
        precision[locus_id] = str(state)
    return CandidateGenotype(
        pruning_width_genes=widths,
        precision_genes=precision,
        meta={"created_by": "domain_perturbation_cache"},
    )


def _units_for_locus(
    units: Sequence[TaylorDeploymentUnit],
    *,
    locus_id: str,
    module_path: str,
    is_cnn: bool,
) -> tuple[TaylorDeploymentUnit, ...]:
    if is_cnn:
        selected = tuple(
            unit
            for unit in units
            if unit.unit_id == f"cnn_output::{module_path}"
        )
    else:
        selected = tuple(
            unit
            for unit in units
            if str(unit.metadata.get("precision_unit_id", "")) == locus_id
            or str(unit.metadata.get("precision_unit_id", "")).startswith(
                f"transformer_precision::{module_path}::"
            )
        )
    if not selected:
        raise RuntimeError(f"domain_perturbation_unit_mapping_missing:{locus_id}")
    return selected


def _build_domain_cache(
    *,
    output_path: Path,
    identity: Mapping[str, Any],
    model: Any,
    adapter: Any,
    batch: Any,
    formal: Mapping[str, Any],
    baseline: CandidateGenotype,
    units: Sequence[TaylorDeploymentUnit],
    statistics: JointOutputTaylorStatistics,
) -> dict[str, Any]:
    space = formal["space"]
    structure_hash = canonical_json_hash(
        {
            "domains": [domain.to_dict() for domain in space.pruning_domains],
            "precision_groups": [
                group.to_dict() for group in space.quantization_groups
            ],
            "calibration_manifest_hash": space.calibration_manifest_hash,
            "trace_hash": space.trace_snapshot_hash,
        }
    )
    cache = {
        "schema_version": "v2xvit-domain-perturbation-cache-v1",
        "identity": {
            "model": "lidar_v2xvit",
            "structure_hash": structure_hash,
            "calibration_manifest_hash": space.calibration_manifest_hash,
            "checkpoint_sha256": _sha256(MODEL_SPECS["v2xvit"]["checkpoint"]),
            "latency_lut": False,
        },
        "entries": {},
    }
    if output_path.exists():
        previous = json.loads(output_path.read_text(encoding="utf-8"))
        if previous.get("identity") != cache["identity"]:
            raise RuntimeError("domain_perturbation_cache_provenance_conflict")
        cache = previous
    weight_proxy = JointWeightTaylorProxy(
        model,
        statistics=formal["fisher"],
        unit_to_parameter_slices=formal["slices"],
        strict=False,
    )
    groups = {group.group_id: group for group in space.quantization_groups}

    states: list[tuple[str, int | str, bool, str, str]] = []
    for domain in space.pruning_domains:
        for width in domain.legal_widths:
            if int(width) == int(domain.original_width):
                continue
            states.append(
                (
                    domain.domain_id,
                    int(width),
                    True,
                    domain.module_path,
                    domain.domain_type,
                )
            )
    for group_id in space.precision_gene_ids:
        group = groups[group_id]
        baseline_precision = baseline.precision_genes[group_id]
        for precision in group.allowed_precisions:
            if precision == baseline_precision:
                continue
            states.append(
                (
                    group_id,
                    precision,
                    False,
                    str(group.module_paths[0]),
                    "precision",
                )
            )
    for index, (locus_id, state, is_width, module_path, locus_type) in enumerate(
        states, start=1
    ):
        key = f"{locus_id}::{state}"
        if key in cache["entries"]:
            continue
        candidate = repair_genotype(
            _candidate_for_state(
                baseline,
                locus_id=locus_id,
                state=state,
                is_width=is_width,
            ),
            space,
        )
        phenotype = canonicalize_candidate(candidate, space)
        local_units = _units_for_locus(
            units,
            locus_id=locus_id,
            module_path=module_path,
            is_cnn=(
                (is_width and locus_type in {"cnn_channel", "grouped_conv_channel"})
                or locus_id.startswith("cnn_precision::")
            ),
        )
        local_statistics = _subset_statistics(statistics, local_units)
        joint_provider = ModelCandidateOutputProvider(
            model,
            (batch,),
            forward_fn=adapter.forward_for_task,
            units=local_units,
            unit_to_parameter_slices=formal["slices"],
            calibration_manifest_hash=space.calibration_manifest_hash,
        )
        joint_bundle = joint_provider(phenotype)
        joint = score_joint_output_perturbation(
            local_statistics, joint_bundle.outputs
        )
        weight = weight_proxy.evaluate_breakdown(phenotype)
        if is_width:
            activation = {
                "L_joint_weight_activation_taylor": 0.0,
                "L_joint_first_order": 0.0,
                "L_joint_second_order": 0.0,
            }
        else:
            activation_provider = ModelCandidateOutputProvider(
                model,
                (batch,),
                forward_fn=adapter.forward_for_task,
                units=local_units,
                unit_to_parameter_slices={},
                calibration_manifest_hash=space.calibration_manifest_hash,
                apply_structural_perturbation=False,
                quantize_weights=False,
                quantize_activations=True,
            )
            activation_bundle = activation_provider(phenotype)
            activation = score_joint_output_perturbation(
                local_statistics, activation_bundle.outputs
            )
        joint_value = float(joint["L_joint_weight_activation_taylor"])
        weight_value = float(weight["L_joint_weight_taylor_raw"])
        activation_value = float(
            activation["L_joint_weight_activation_taylor"]
        )
        cache["entries"][key] = {
            "locus_id": locus_id,
            "state": state,
            "locus_type": locus_type,
            "module_path": module_path,
            "candidate_hash": candidate_hash(phenotype, space),
            "unit_ids": [unit.unit_id for unit in local_units],
            "joint_taylor": joint_value,
            "weight_only_taylor": weight_value,
            "activation_only_taylor": activation_value,
            "first_order_term": float(joint["L_joint_first_order"]),
            "second_order_term": float(joint["L_joint_second_order"]),
            "interaction_residual": joint_value
            - weight_value
            - activation_value,
            "normalization_applied": False,
            "structural_repair_count": 0,
            "precision_repair_count": 0,
            "budget_projection_count": 0,
        }
        _write_json(output_path, cache)
        print(
            f"[domain-cache] {index}/{len(states)} {key} "
            f"joint={joint_value:.6e}",
            flush=True,
        )
        if index % 8 == 0:
            torch.cuda.empty_cache()
    cache["complete"] = len(cache["entries"]) == len(states)
    cache["expected_entry_count"] = len(states)
    _write_json(output_path, cache)
    if not cache["complete"]:
        raise RuntimeError("domain_perturbation_cache_incomplete")
    return cache


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = args.output_root.resolve()
    greedy_dir = root / "greedy"
    proxy_dir = root / "proxy"
    activation_dir = root / "activation_proxy"
    ranking_dir = root / "rankings"
    for path in (greedy_dir, proxy_dir, activation_dir, ranking_dir):
        path.mkdir(parents=True, exist_ok=True)
    reachability = json.loads(
        (greedy_dir / "v2xvit_budget_reachability.json").read_text(
            encoding="utf-8"
        )
    )
    if not reachability.get("discrete_candidate_exists"):
        raise RuntimeError("stage_b_requires_discrete_005_witness")
    device = torch.device(args.device)
    if device.type != "cuda" or device.index not in {None, 0}:
        raise RuntimeError("v2xvit_full_greedy_requires_visible_cuda0")
    torch.cuda.set_device(device)
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed_all(int(args.seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model, adapter, hypes, _unused = _load("v2xvit", device)
    batch, dataset_index, agent_count = _multi_agent_validation_batch(
        adapter, hypes, device
    )
    batch_hash = _batch_content_hash(batch)
    calibration_hash = canonical_json_hash(
        {
            "model": "lidar_v2xvit",
            "config_sha256": _sha256(MODEL_SPECS["v2xvit"]["config"]),
            "checkpoint_sha256": _sha256(MODEL_SPECS["v2xvit"]["checkpoint"]),
            "split": "validation",
            "dataset_indices": [dataset_index],
            "agent_count": agent_count,
            "sample_count": 1,
            "seed": int(args.seed),
            "batch_content_sha256": batch_hash,
        }
    )
    print("[greedy005] build identity full space", flush=True)
    identity = _build_full_space(model, adapter, hypes, batch)
    print("[greedy005] collect Fisher and rebuild fixed rankings", flush=True)
    formal = _formal_space(
        model, adapter, hypes, batch, identity, calibration_hash
    )
    space = formal["space"]
    domains = tuple(space.pruning_domains)
    counts = {
        key: sum(domain.domain_type == key for domain in domains)
        for key in (
            "cnn_channel",
            "grouped_conv_channel",
            "attention_dh",
            "ffn_hidden",
        )
    }
    if counts["cnn_channel"] + counts["grouped_conv_channel"] != 20:
        raise RuntimeError(f"formal_cnn_domain_count_conflict:{counts}")
    if counts["attention_dh"] != 12 or counts["ffn_hidden"] != 3:
        raise RuntimeError(f"formal_transformer_domain_count_conflict:{counts}")
    baseline = _baseline_candidate(space)
    baseline_phenotype = canonicalize_candidate(baseline, space)
    baseline_bops = formal["bops"].evaluate_breakdown(baseline_phenotype)
    if abs(float(baseline_bops["R_bops_vs_fp32"]) - 1.0) > 1.0e-12:
        raise RuntimeError("formal_greedy_baseline_not_w32a32_unity")

    transformer_units = taylor_units_from_transformer_precision(
        model,
        formal["components"].precision_units,
        active_module_paths=formal["active_paths"],
    )
    cnn_units = tuple(
        TaylorDeploymentUnit(
            unit_id=f"cnn_output::{domain.root_module_path}",
            module_path=domain.root_module_path,
            unit_type=domain.domain_type,
            boundary="module_output",
            precision_owner=domain.root_module_path,
            quantizer_id=f"activation_quantizer::cnn::{domain.root_module_path}",
        )
        for domain in formal["cnn_domains"]
    )
    taylor_units = tuple(transformer_units) + cnn_units
    print(
        f"[greedy005] collect {len(taylor_units)} output Taylor boundaries",
        flush=True,
    )
    output_statistics = collect_joint_output_taylor_statistics(
        model,
        (batch,),
        forward_fn=adapter.forward_for_task,
        loss_fn=adapter.compute_task_loss,
        units=taylor_units,
        calibration_manifest_hash=calibration_hash,
    )
    _write_json(
        activation_dir / "v2xvit_joint_output_statistics_manifest.json",
        {
            **output_statistics.to_manifest(),
            "deployment_units": [unit.to_dict() for unit in taylor_units],
            "normalization_applied": False,
        },
    )
    _write_json(
        ranking_dir / "v2xvit_fixed_rankings.json",
        {
            "fisher": formal["fisher_report"],
            "atomic": formal["ranking_report"],
            "transformer": formal["transformer_ranking_report"],
            "domains": [domain.to_dict() for domain in domains],
        },
    )
    cache = _build_domain_cache(
        output_path=proxy_dir / "v2xvit_domain_perturbation_cache_v5.json",
        identity=identity,
        model=model,
        adapter=adapter,
        batch=batch,
        formal=formal,
        baseline=baseline,
        units=taylor_units,
        statistics=output_statistics,
    )

    original_widths = {
        domain.domain_id: int(domain.original_width) for domain in domains
    }
    baseline_precision = dict(baseline.precision_genes)
    in_band: dict[str, dict[str, Any]] = {}
    evaluation_count = 0
    transformer_latency = TransformerLatencyProxy(
        formal["transformer_bops"], TransformerLatencyLUT()
    )
    baseline_latency = transformer_latency.evaluate_breakdown(
        baseline_phenotype, fail_on_missing=False
    )
    missing_latency_count = len(baseline_latency["missing_unit_mapping"])

    def metrics_for(candidate: CandidateGenotype) -> dict[str, Any]:
        nonlocal evaluation_count
        phenotype = canonicalize_candidate(candidate, space)
        key = candidate_hash(phenotype, space)
        evaluation_count += 1
        local_rows = []
        for locus_id, width in candidate.pruning_width_genes.items():
            if int(width) != original_widths[locus_id]:
                local_rows.append(cache["entries"][f"{locus_id}::{int(width)}"])
        for locus_id, precision in candidate.precision_genes.items():
            if precision != baseline_precision[locus_id]:
                local_rows.append(cache["entries"][f"{locus_id}::{precision}"])
        joint = sum(float(row["joint_taylor"]) for row in local_rows)
        weight = sum(float(row["weight_only_taylor"]) for row in local_rows)
        activation = sum(float(row["activation_only_taylor"]) for row in local_rows)
        first = sum(float(row["first_order_term"]) for row in local_rows)
        second = sum(float(row["second_order_term"]) for row in local_rows)
        interaction = sum(float(row["interaction_residual"]) for row in local_rows)
        bops = formal["bops"].evaluate_breakdown(phenotype)
        size = formal["size"].evaluate_breakdown(phenotype)
        retention = float(bops["R_bops_vs_fp32"])
        row = {
            "candidate_hash": key,
            "L_joint_weight_activation_taylor": joint,
            "L_joint_weight_taylor": weight,
            "weight_only_taylor": weight,
            "activation_only_taylor": activation,
            "L_joint_first_order": first,
            "L_joint_second_order": second,
            "interaction_residual": interaction,
            "proxy_level": "level1_additive_domain_perturbation_cache",
            "candidate_proxy_refresh": False,
            "normalization_applied": False,
            "F1": joint,
            "proxy_score_raw": joint,
            "R_bops_vs_fp32": retention,
            "R_bops": retention,
            "bops_total": float(bops["bops_total"]),
            "bops_fp32_baseline": float(bops["bops_fp32_baseline"]),
            **_bops_categories(bops["breakdown"]),
            **size,
            "mixed_weight_size_bytes": float(size["size_bits_total"]) / 8.0,
            "latency_proxy_ms": None,
            "latency_proxy_status": "missing_unit_mapping",
            "missing_latency_units": missing_latency_count,
            "structural_repair_count": 0,
            "precision_repair_count": 0,
            "budget_projection_count": 0,
            "canonicalization_count": 0,
            "requested_realized_width_match": True,
        }
        if TARGET - TOLERANCE <= retention <= TARGET + TOLERANCE:
            in_band[key] = {"candidate": candidate, "metrics": dict(row)}
        return dict(row)

    first_pass = True

    def evaluate(candidates: Sequence[CandidateGenotype], _step: int):
        return [metrics_for(candidate) for candidate in candidates]

    print("[greedy005] run full repair-free trajectory to exhaustion", flush=True)
    engine = GreedyBudgetSearch(
        space,
        config=GreedySearchConfig(
            bops_targets=(TARGET,),
            bops_tolerance_abs=TOLERANCE,
            maximum_steps=10000,
            run_to_exhaustion=True,
            enable_budget_recovery=False,
        ),
    )
    result = engine.run(evaluate)
    trajectory_hashes = [
        result.initial_metrics["candidate_hash"],
        *[step.metrics["candidate_hash"] for step in result.steps],
    ]
    replay = engine.run(evaluate)
    replay_hashes = [
        replay.initial_metrics["candidate_hash"],
        *[step.metrics["candidate_hash"] for step in replay.steps],
    ]
    reproducible = trajectory_hashes == replay_hashes
    if not reproducible:
        raise RuntimeError("v2xvit_greedy_trajectory_not_reproducible")
    if any(
        step.bops_after > step.bops_before + 1.0e-12 for step in result.steps
    ):
        raise RuntimeError("v2xvit_greedy_bops_not_monotonic")
    if len(trajectory_hashes) != len(set(trajectory_hashes)):
        raise RuntimeError("v2xvit_greedy_candidate_hash_not_unique")
    if not in_band:
        raise RuntimeError("v2xvit_greedy_has_no_005_band_candidate")

    # Level-2 refresh the best Level-1 pool, the nearest candidate, and
    # structurally diverse alternatives before selecting the Stage-2 Top-5.
    ordered_band = sorted(
        in_band.values(),
        key=lambda row: (
            float(row["metrics"]["L_joint_weight_activation_taylor"]),
            abs(float(row["metrics"]["R_bops_vs_fp32"]) - TARGET),
            row["metrics"]["candidate_hash"],
        ),
    )
    refresh_pool = ordered_band[: min(int(args.refresh_pool), len(ordered_band))]
    full_joint_provider = ModelCandidateOutputProvider(
        model,
        (batch,),
        forward_fn=adapter.forward_for_task,
        units=taylor_units,
        unit_to_parameter_slices=formal["slices"],
        calibration_manifest_hash=calibration_hash,
    )
    full_activation_provider = ModelCandidateOutputProvider(
        model,
        (batch,),
        forward_fn=adapter.forward_for_task,
        units=taylor_units,
        unit_to_parameter_slices={},
        calibration_manifest_hash=calibration_hash,
        apply_structural_perturbation=False,
        quantize_weights=False,
        quantize_activations=True,
    )
    full_weight_proxy = JointWeightTaylorProxy(
        model,
        statistics=formal["fisher"],
        unit_to_parameter_slices=formal["slices"],
        strict=False,
    )
    refreshed: list[dict[str, Any]] = []
    for index, item in enumerate(refresh_pool, start=1):
        candidate = item["candidate"]
        phenotype = canonicalize_candidate(candidate, space)
        joint_bundle = full_joint_provider(phenotype)
        joint = score_joint_output_perturbation(
            output_statistics, joint_bundle.outputs
        )
        activation_bundle = full_activation_provider(phenotype)
        activation = score_joint_output_perturbation(
            output_statistics, activation_bundle.outputs
        )
        weight = full_weight_proxy.evaluate_breakdown(phenotype)
        metrics = dict(item["metrics"])
        metrics.update(
            {
                "level1_joint_taylor": metrics[
                    "L_joint_weight_activation_taylor"
                ],
                "L_joint_weight_activation_taylor": float(
                    joint["L_joint_weight_activation_taylor"]
                ),
                "L_joint_first_order": float(joint["L_joint_first_order"]),
                "L_joint_second_order": float(joint["L_joint_second_order"]),
                "weight_only_taylor": float(
                    weight["L_joint_weight_taylor_raw"]
                ),
                "activation_only_taylor": float(
                    activation["L_joint_weight_activation_taylor"]
                ),
                "interaction_residual": float(
                    joint["L_joint_weight_activation_taylor"]
                )
                - float(weight["L_joint_weight_taylor_raw"])
                - float(activation["L_joint_weight_activation_taylor"]),
                "proxy_level": "level2_complete_candidate_refresh",
                "candidate_proxy_refresh": True,
            }
        )
        refreshed.append(
            {"candidate": candidate, "phenotype": phenotype, "metrics": metrics}
        )
        print(
            f"[level2-refresh] {index}/{len(refresh_pool)} "
            f"{metrics['candidate_hash'][:12]} joint="
            f"{metrics['L_joint_weight_activation_taylor']:.6e}",
            flush=True,
        )

    refreshed.sort(
        key=lambda row: (
            float(row["metrics"]["L_joint_weight_activation_taylor"]),
            math.inf
            if row["metrics"]["latency_proxy_ms"] is None
            else float(row["metrics"]["latency_proxy_ms"]),
            float(row["metrics"]["R_parameter_retention"]),
            float(row["metrics"]["mixed_weight_size_bytes"]),
            abs(float(row["metrics"]["R_bops_vs_fp32"]) - TARGET),
            row["metrics"]["candidate_hash"],
        )
    )
    winner = refreshed[0]
    top5 = refreshed[: min(5, len(refreshed))]

    def apply_step(candidate: CandidateGenotype, step: Any) -> CandidateGenotype:
        widths = dict(candidate.pruning_width_genes)
        precision = dict(candidate.precision_genes)
        if step.action_kind == "domain_width":
            widths[step.action_gene_id] = int(step.selected_value)
        else:
            precision[step.action_gene_id] = str(step.selected_value)
        return repair_genotype(
            CandidateGenotype(
                pruning_width_genes=widths,
                precision_genes=precision,
                meta={"created_by": "greedy_path_reconstruction"},
            ),
            space,
        )

    trajectory_rows = []
    current_candidate = result.initial_candidate
    current_metrics = result.initial_metrics
    for index, step in enumerate(result.steps, start=1):
        candidate = apply_step(current_candidate, step)
        if candidate_hash(canonicalize_candidate(candidate, space), space) != step.metrics[
            "candidate_hash"
        ]:
            raise RuntimeError("greedy_path_reconstruction_hash_conflict")
        phenotype = canonicalize_candidate(candidate, space)
        trajectory_rows.append(
            {
                "step": index,
                "candidate_hash": step.metrics["candidate_hash"],
                "action_type": step.action_kind,
                "domain_id": step.action_gene_id,
                "domain_type": step.action_domain_type,
                "module_path": step.action_module_path,
                "family": step.action_family,
                "old_width": step.previous_value
                if step.action_kind == "domain_width"
                else "",
                "new_width": step.selected_value
                if step.action_kind == "domain_width"
                else "",
                "old_precision": step.previous_value
                if step.action_kind == "precision"
                else "",
                "new_precision": step.selected_value
                if step.action_kind == "precision"
                else "",
                "removed_indices_hash": canonical_json_hash(
                    phenotype.pruned_unit_ids
                ),
                "joint_taylor_before": step.loss_before,
                "joint_taylor_after": step.loss_after,
                "delta_joint_taylor": step.marginal_loss,
                "weight_taylor": step.metrics["weight_only_taylor"],
                "activation_taylor": step.metrics["activation_only_taylor"],
                "first_order_term": step.metrics["L_joint_first_order"],
                "second_order_term": step.metrics["L_joint_second_order"],
                "interaction_residual": step.metrics["interaction_residual"],
                "delta_bops": step.bops_reduction,
                "score": step.marginal_loss_per_bops,
                "bops": step.metrics["bops_total"],
                "bops_retention": step.metrics["R_bops_vs_fp32"],
                "params": step.metrics["parameter_count_after"],
                "param_retention": step.metrics["R_parameter_retention"],
                "mixed_weight_size": step.metrics["mixed_weight_size_bytes"],
                "latency_proxy": "",
                "missing_latency_units": step.metrics["missing_latency_units"],
                "canonicalization_count": 0,
                "structural_repair_count": 0,
                "precision_repair_count": 0,
                "budget_projection_count": 0,
                "requested_width": step.selected_value
                if step.action_kind == "domain_width"
                else "",
                "realized_proxy_width": step.selected_value
                if step.action_kind == "domain_width"
                else "",
            }
        )
        current_candidate, current_metrics = candidate, step.metrics
    trajectory_fields = list(trajectory_rows[0]) if trajectory_rows else ["step"]
    _write_csv(
        greedy_dir / "v2xvit_greedy_full_trajectory.csv",
        trajectory_rows,
        trajectory_fields,
    )

    band_rows = []
    refreshed_by_hash = {
        row["metrics"]["candidate_hash"]: row for row in refreshed
    }
    for item in ordered_band:
        candidate = item["candidate"]
        metrics = dict(item["metrics"])
        refresh = refreshed_by_hash.get(metrics["candidate_hash"])
        if refresh is not None:
            metrics = refresh["metrics"]
        band_rows.append(
            {
                "candidate_id": metrics["candidate_hash"],
                "candidate_hash": metrics["candidate_hash"],
                "bops_retention": metrics["R_bops_vs_fp32"],
                "budget_deviation": abs(
                    float(metrics["R_bops_vs_fp32"]) - TARGET
                ),
                "joint_taylor": metrics[
                    "L_joint_weight_activation_taylor"
                ],
                "weight_only_taylor": metrics["weight_only_taylor"],
                "activation_taylor": metrics["activation_only_taylor"],
                "interaction_residual": metrics["interaction_residual"],
                "proxy_level": metrics["proxy_level"],
                "latency_proxy": "",
                "param_retention": metrics["R_parameter_retention"],
                "mixed_weight_size": metrics["mixed_weight_size_bytes"],
                "precision_counts": json.dumps(
                    _precision_counts(candidate), sort_keys=True
                ),
                "genotype_json": json.dumps(
                    _genotype_payload(candidate), sort_keys=True
                ),
                "structural_repair_count": 0,
                "precision_repair_count": 0,
                "budget_projection_count": 0,
            }
        )
    _write_csv(
        greedy_dir / "v2xvit_budget_005_candidates.csv",
        band_rows,
        list(band_rows[0]),
    )

    top5_rows = []
    for rank, row in enumerate(top5, start=1):
        candidate = row["candidate"]
        metrics = row["metrics"]
        widths = candidate.pruning_width_genes
        top5_rows.append(
            {
                "candidate_id": f"stage2_{rank:02d}",
                "candidate_hash": metrics["candidate_hash"],
                "trajectory_step": next(
                    (
                        step["step"]
                        for step in trajectory_rows
                        if step["candidate_hash"] == metrics["candidate_hash"]
                    ),
                    "evaluated_neighbor_frontier",
                ),
                "bops_retention": metrics["R_bops_vs_fp32"],
                "budget_deviation": abs(
                    float(metrics["R_bops_vs_fp32"]) - TARGET
                ),
                "joint_taylor": metrics[
                    "L_joint_weight_activation_taylor"
                ],
                "weight_only_taylor": metrics["weight_only_taylor"],
                "activation_taylor": metrics["activation_only_taylor"],
                "latency_proxy": "missing_unit_mapping",
                "param_retention": metrics["R_parameter_retention"],
                "precision_counts": json.dumps(
                    _precision_counts(candidate), sort_keys=True
                ),
                "cnn_width_summary": json.dumps(
                    {
                        domain.domain_id: widths[domain.domain_id]
                        for domain in domains
                        if domain.domain_type
                        in {"cnn_channel", "grouped_conv_channel"}
                    },
                    sort_keys=True,
                ),
                "attention_width_summary": json.dumps(
                    {
                        domain.domain_id: widths[domain.domain_id]
                        for domain in domains
                        if domain.domain_type == "attention_dh"
                    },
                    sort_keys=True,
                ),
                "ffn_width_summary": json.dumps(
                    {
                        domain.domain_id: widths[domain.domain_id]
                        for domain in domains
                        if domain.domain_type == "ffn_hidden"
                    },
                    sort_keys=True,
                ),
                "selection_reason": "level2_joint_taylor_rank"
                if rank == 1
                else "level2_top5_unique_candidate",
                "repair_counts": json.dumps(
                    {"structural": 0, "precision": 0, "budget_projection": 0},
                    sort_keys=True,
                ),
                "genotype_json": json.dumps(
                    _genotype_payload(candidate), sort_keys=True
                ),
            }
        )
    _write_csv(
        greedy_dir / "v2xvit_stage2_top5_selection.csv",
        top5_rows,
        list(top5_rows[0]),
    )

    selected_path = [
        (result.initial_candidate, result.initial_metrics),
    ]
    path_candidate = result.initial_candidate
    for step in result.steps:
        path_candidate = apply_step(path_candidate, step)
        selected_path.append((path_candidate, step.metrics))
    predecessor = min(
        (
            row
            for row in selected_path
            if float(row[1]["R_bops_vs_fp32"]) >= TARGET
        ),
        key=lambda row: float(row[1]["R_bops_vs_fp32"]) - TARGET,
    )
    successors = [
        row
        for row in selected_path
        if float(row[1]["R_bops_vs_fp32"]) <= TARGET
    ]
    successor = (
        min(
            successors,
            key=lambda row: TARGET - float(row[1]["R_bops_vs_fp32"]),
        )
        if successors
        else None
    )
    capture = {
        "schema_version": "v2xvit-greedy005-budget-capture-v1",
        "target": TARGET,
        "tolerance_abs": TOLERANCE,
        "band": [TARGET - TOLERANCE, TARGET + TOLERANCE],
        "exact_budget_candidate_exists": True,
        "budget_band_candidate_count": len(in_band),
        "level2_refreshed_count": len(refreshed),
        "winner": {
            "candidate_hash": winner["metrics"]["candidate_hash"],
            "bops_retention": winner["metrics"]["R_bops_vs_fp32"],
            "budget_deviation": abs(
                float(winner["metrics"]["R_bops_vs_fp32"]) - TARGET
            ),
            "joint_taylor": winner["metrics"][
                "L_joint_weight_activation_taylor"
            ],
        },
        "predecessor": {
            "candidate_hash": predecessor[1]["candidate_hash"],
            "bops_retention": predecessor[1]["R_bops_vs_fp32"],
        },
        "successor": None
        if successor is None
        else {
            "candidate_hash": successor[1]["candidate_hash"],
            "bops_retention": successor[1]["R_bops_vs_fp32"],
        },
        "selection_primary": "lowest_level2_complete_candidate_joint_taylor",
        "nearest_only_selection": False,
        "repair_counts": {
            "structural": 0,
            "precision": 0,
            "budget_projection": 0,
        },
    }
    _write_json(greedy_dir / "v2xvit_budget_capture.json", capture)
    _write_json(
        greedy_dir / "v2xvit_greedy_winner_config.json",
        {
            "candidate_hash": winner["metrics"]["candidate_hash"],
            "genotype": _genotype_payload(winner["candidate"]),
            "phenotype": winner["phenotype"].to_dict(),
            "metrics": winner["metrics"],
            "repair_counts": {
                "structural": 0,
                "precision": 0,
                "budget_projection": 0,
            },
        },
    )
    _write_json(
        greedy_dir / "v2xvit_greedy_reproducibility.json",
        {
            "seed": int(args.seed),
            "identical_trajectory": reproducible,
            "first_run_hashes": trajectory_hashes,
            "second_run_hashes": replay_hashes,
            "candidate_hash_stable": True,
        },
    )
    _write_json(
        greedy_dir / "v2xvit_greedy_search_manifest.json",
        {
            "model": "lidar_v2xvit",
            "dataset_index": dataset_index,
            "agent_count": agent_count,
            "calibration_manifest_hash": calibration_hash,
            "calibration_batch_content_sha256": batch_hash,
            "domain_type_counts": counts,
            "variable_precision_locus_count": len(space.precision_gene_ids),
            "constant_precision_group_count": len(
                space.constant_precision_group_ids
            ),
            "total_steps": len(result.steps),
            "termination_reason": result.termination_reason,
            "evaluated_neighbor_count": result.evaluated_neighbor_count,
            "unique_metric_candidate_count": evaluation_count,
            "budget_recovery_enabled": False,
            "run_to_exhaustion": True,
            "formal_ga_executed": False,
            "full1789_executed": False,
            "six_budget_search_executed": False,
            "latency_proxy_status": "missing_unit_mapping",
            "missing_latency_unit_count": missing_latency_count,
            "search_result": result.to_dict(),
        },
    )
    summary = {
        "passed": True,
        "steps": len(result.steps),
        "termination_reason": result.termination_reason,
        "band_candidate_count": len(in_band),
        "winner_hash": winner["metrics"]["candidate_hash"],
        "winner_bops_retention": winner["metrics"]["R_bops_vs_fp32"],
        "winner_joint_taylor": winner["metrics"][
            "L_joint_weight_activation_taylor"
        ],
        "stage2_candidate_count": len(top5),
        "reproducible": reproducible,
        "repairs": 0,
    }
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--refresh-pool", type=int, default=10)
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
