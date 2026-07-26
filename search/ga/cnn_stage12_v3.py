"""Strict Stage-1/Stage-2/V1--V3 adapter for the three CNN LiDAR models.

This is deliberately an adapter around :mod:`search.ga.stage12_v3`; it does
not copy the historical two-stage GA.  Pyramid, DiscoNet and F-Cooper share
the same legal genotype, Taylor, BOPS, archive and generation semantics while
retaining their already-audited model loaders and TensorRT evaluators.
"""

from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from ..candidate import CandidateGenotype
from ..canonicalization import SearchSpaceSpec, canonicalize_candidate
from ..integration.calibration_provider import collect_or_load_fisher_statistics
from ..integration.data_provider import (
    build_dataset_and_loader,
    iter_limited,
    move_batch_to_device,
)
from ..integration.heal_lidar_baseline_context import (
    HEAL_RUNTIME_GRAPH_POLICY,
    build_heal_lidar_baseline_context,
)
from ..integration.lidar_pyramid_context import build_lidar_pyramid_context
from ..proxy.bops_proxy import BOPSProxy
from ..proxy.conservative_gate_activation_taylor import (
    FunctionalGateTaylorProxy,
    build_activation_units,
    collect_activation_taylor_cache_multi,
    collect_functional_gate_scores_multi,
    rerank_domains_by_gate_scores,
)
from ..proxy.joint_weight_taylor import JointWeightTaylorProxy
from ..proxy.parameter_slice_resolver import build_unit_parameter_slices
from ..proxy.runtime_shape_profiler import profile_runtime_layer_shapes
from ..proxy.size_proxy import SizeProxy
from ..stage2.heal_lidar_baseline_real_evaluator import (
    HealLidarBaselineCandidateEvaluator,
)
from ..stage2.lidar_pyramid_real_evaluator import LidarPyramidRealEvaluator
from ..stage2.objective import Stage2ObjectiveConfig
from .strict_stage12_v3 import (
    Stage2Result,
    StrictGAConfig,
    StrictStage12V3Runner,
    UnifiedTaylorStage1Evaluator,
    adjacent_mutation,
    phenotype_identity,
    score_stage2,
    validate_genotype_schema,
)


@dataclass(frozen=True)
class CNNFormalModelSpec:
    model_id: str
    family_id: str
    checkpoint: Path
    config: Path
    calibration_manifest: Path
    strict_fp32_engine: Path | None = None


MODEL_SPECS: dict[str, CNNFormalModelSpec] = {
    "pyramid": CNNFormalModelSpec(
        model_id="pyramid",
        family_id="lidar_pyramid",
        checkpoint=Path("/home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_pyramid/net_epoch_bestval_at17.pth"),
        config=Path("/home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_pyramid/config.yaml"),
        calibration_manifest=Path("/home/lixingfeng/UniAD_examine/heal_compress/tests/quant_deploy/outputs/lidar_pyramid_agent_export_strategy_compare/artifacts/calibration/train_calib_single_engine_maxK29696_200/manifest.json"),
    ),
    "disco": CNNFormalModelSpec(
        model_id="disco",
        family_id="heal_lidar_disco",
        checkpoint=Path("/home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_disco/net_epoch_bestval_at35.pth"),
        config=Path("/home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_disco/config.yaml"),
        calibration_manifest=Path("/home/lixingfeng/UniAD_examine/heal_compress/outputs/heal_lidar_baseline_train200_fixedk29696_20260719_1215_v2/calibration_manifest.json"),
        strict_fp32_engine=Path("/home/lixingfeng/UniAD_examine/heal_compress/outputs/h800_heal_lidar_fixedk29696_fp32_baselines_20260719/lidar_disco/strict_fp32.plan"),
    ),
    "fcooper": CNNFormalModelSpec(
        model_id="fcooper",
        family_id="heal_lidar_fcooper",
        checkpoint=Path("/home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_fcooper/net_epoch_bestval_at37.pth"),
        config=Path("/home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_fcooper/config.yaml"),
        calibration_manifest=Path("/home/lixingfeng/UniAD_examine/heal_compress/outputs/heal_lidar_baseline_train200_fixedk29696_20260719_1215_v2/calibration_manifest.json"),
        strict_fp32_engine=Path("/home/lixingfeng/UniAD_examine/heal_compress/outputs/h800_heal_lidar_fixedk29696_fp32_baselines_20260719/lidar_fcooper/strict_fp32.plan"),
    ),
}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _json_default(value: Any) -> Any:
    if isinstance(value, CandidateGenotype):
        return value.to_dict()
    if isinstance(value, Stage2Result):
        return stage2_payload(value)
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    return str(value)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({str(key) for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def canonical_size_metrics(size: Mapping[str, Any]) -> dict[str, Any]:
    """Expose SizeProxy values under the strict Stage-1 ranking contract."""

    result = dict(size)
    result["mixed_weight_retention"] = float(result["R_size_vs_fp32"])
    return result


def stage2_payload(result: Stage2Result) -> dict[str, Any]:
    return {
        "complete_phenotype_hash": result.complete_phenotype_hash,
        "genotype": result.genotype.to_dict(),
        "status": result.status,
        "mAP": result.map,
        "p50_ms": result.p50_ms,
        "requested_realized_exact": result.requested_realized_exact,
        "evaluated": result.evaluated,
        "skipped": result.skipped,
        "metadata": dict(result.metadata),
    }


def baseline_genotype(space: SearchSpaceSpec) -> CandidateGenotype:
    widths = {
        str(domain.domain_id): int(domain.original_width)
        for domain in space.pruning_domains
        if str(domain.domain_id) in set(space.pruning_gene_ids)
    }
    groups = {str(group.group_id): group for group in space.quantization_groups}
    precision: dict[str, str] = {}
    for group_id in space.precision_gene_ids:
        allowed = tuple(groups[group_id].allowed_precisions)
        if "FP32" not in allowed:
            raise RuntimeError(f"formal_cnn_mutable_locus_missing_fp32:{group_id}")
        precision[group_id] = "FP32"
    result = CandidateGenotype(
        pruning_width_genes=widths,
        precision_genes=precision,
        meta={"created_by": "strict_fp32_all_keep", "repair_count": 0},
    )
    validate_genotype_schema(result, space)
    return result


class DeviceBatchPrefix(Sequence[Any]):
    """Move one deterministic CPU batch at a time, avoiding a 32-batch VRAM copy."""

    def __init__(self, rows: Sequence[Any], device: torch.device) -> None:
        self.rows = tuple(rows)
        self.device = device

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Any:
        return move_batch_to_device(self.rows[index], self.device)


@dataclass
class PreparedCNNFormalSearch:
    spec: CNNFormalModelSpec
    context: Any
    space: SearchSpaceSpec
    baseline: CandidateGenotype
    bops: BOPSProxy
    size: SizeProxy
    structure: FunctionalGateTaylorProxy
    weight: JointWeightTaylorProxy
    activation: Any
    gate_mapping: list[dict[str, Any]]
    calibration_sample_count: int

    def evaluator(
        self,
        *,
        target: float,
        enforce_bops_hard_gate: bool,
    ) -> UnifiedTaylorStage1Evaluator:
        return UnifiedTaylorStage1Evaluator(
            self.space,
            baseline=self.baseline,
            structure_proxy=self.structure,
            weight_proxy=self.weight,
            activation_cache=self.activation,
            bops_evaluator=self.bops.evaluate_breakdown,
            size_evaluator=self.size.evaluate_breakdown,
            target=float(target),
            tolerance_abs=0.005,
            enforce_bops_hard_gate=enforce_bops_hard_gate,
        )


def build_context(
    spec: CNNFormalModelSpec,
    *,
    output_root: Path,
    physical_gpu: int,
    plugin: Path,
    tensorrt_root: Path,
    taylor_samples: int,
) -> Any:
    for path in (spec.checkpoint, spec.config, spec.calibration_manifest, plugin):
        if not path.is_file():
            raise RuntimeError(f"formal_cnn_required_artifact_missing:{path}")
    if spec.model_id == "pyramid":
        return build_lidar_pyramid_context(
            checkpoint_path=spec.checkpoint,
            model_config_path=spec.config,
            output_dir=output_root,
            heal_root="/home/lixingfeng/UniAD_examine/HEAL",
            tensorrt_root=tensorrt_root,
            plugin_path=plugin,
            gpu_id=str(physical_gpu),
            exclude_gpu_ids=[],
            tensorrt_env="modelopt",
            fisher_calibration_batches=int(taylor_samples),
            quant_calibration_batches=200,
            quant_calibration_npz_manifest=spec.calibration_manifest,
            quant_activation_calibration_backend="tensorrt_entropy_calibration2",
            quant_calibration_force_rebuild=True,
            num_frames=50,
            warmup_frames=20,
            reset_after_warmup=True,
            default_precision="FP32",
            pruning_gene_type="legal_domain_width",
            grouped_allowed_channels_per_group=[4, 8, 16, 32, 64, 128, 256, 512],
        )
    return build_heal_lidar_baseline_context(
        family_id=spec.family_id,
        checkpoint_path=spec.checkpoint,
        model_config_path=spec.config,
        output_dir=output_root,
        heal_root="/home/lixingfeng/UniAD_examine/HEAL",
        tensorrt_root=tensorrt_root,
        plugin_path=plugin,
        gpu_id=str(physical_gpu),
        exclude_gpu_ids=[],
        tensorrt_env="modelopt",
        fisher_calibration_batches=int(taylor_samples),
        quant_calibration_batches=200,
        quant_calibration_npz_manifest=spec.calibration_manifest,
        quant_activation_calibration_backend="tensorrt_entropy_calibration2",
        quant_calibration_force_rebuild=True,
        num_frames=50,
        warmup_frames=20,
        reset_after_warmup=True,
        default_precision="FP32",
        fixed_k=29696,
        max_agents=2,
        minimum_retained_ratio=0.10,
        dense_alignment=4,
        require_quant_calibration_manifest=True,
        search_space_policy=HEAL_RUNTIME_GRAPH_POLICY,
    )


def prepare_search(
    spec: CNNFormalModelSpec,
    *,
    output_root: Path,
    physical_gpu: int,
    plugin: Path,
    tensorrt_root: Path,
    taylor_samples: int = 8,
) -> PreparedCNNFormalSearch:
    context = build_context(
        spec,
        output_root=output_root,
        physical_gpu=physical_gpu,
        plugin=plugin,
        tensorrt_root=tensorrt_root,
        taylor_samples=taylor_samples,
    )
    device = torch.device(context.runtime_device)
    slices = build_unit_parameter_slices(context.model, context.atomic_prune_units)
    runtime = profile_runtime_layer_shapes(
        context.model,
        context.trace_example_inputs,
        forward_fn=context.model_bundle.adapter.forward_for_task,
    )
    write_json(output_root / "proxy/runtime_shapes.json", runtime.to_dict())
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    fisher = collect_or_load_fisher_statistics(
        model=context.model,
        adapter=context.model_bundle.adapter,
        model_config_path=context.model_config,
        device=device,
        cache_path=output_root / "proxy/fisher_statistics.pt",
        num_batches=int(taylor_samples),
    )
    # Reset the data RNG so gate/AQ use the same deterministic train prefix as
    # Fisher.  Each collector still performs an independent per-sample backward.
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    _dataset, loader = build_dataset_and_loader(
        context.model_bundle.adapter,
        context.model_config,
        split="train",
        num_workers=0,
        visualize=False,
    )
    cpu_batches = iter_limited(loader, int(taylor_samples))
    if len(cpu_batches) != int(taylor_samples):
        raise RuntimeError(
            f"formal_cnn_taylor_prefix_incomplete:{len(cpu_batches)}!={taylor_samples}"
        )
    batches = DeviceBatchPrefix(cpu_batches, device)
    adapter = context.model_bundle.adapter
    gate_scores, gate_mapping = collect_functional_gate_scores_multi(
        context.model,
        context.search_space.pruning_domains,
        forward_fn=adapter.forward_for_task,
        loss_fn=adapter.compute_task_loss,
        calibration_batches=batches,
    )
    domains = rerank_domains_by_gate_scores(
        context.search_space.pruning_domains,
        gate_scores,
    )
    space = replace(
        context.search_space,
        pruning_domains=tuple(domains),
        pruning_unit_ids=[
            str(unit)
            for domain in domains
            for unit in domain.ordered_unit_ids
        ],
        pruning_policy_version="cnn-functional-gate-fixed-ranking-v1",
    )
    context.search_space = space
    activation_units, group_to_units = build_activation_units(
        context.model, space, transformer_units=()
    )
    activation = collect_activation_taylor_cache_multi(
        context.model,
        activation_units,
        group_to_units,
        forward_fn=adapter.forward_for_task,
        loss_fn=adapter.compute_task_loss,
        calibration_batches=batches,
    )
    baseline = baseline_genotype(space)
    result = PreparedCNNFormalSearch(
        spec=spec,
        context=context,
        space=space,
        baseline=baseline,
        bops=BOPSProxy(
            context.model,
            unit_to_parameter_slices=slices,
            runtime_shapes=runtime.shapes,
            default_precision="FP32",
        ),
        size=SizeProxy(
            context.model,
            unit_to_parameter_slices=slices,
            default_precision="FP32",
            include_constant_parameters_in_size=True,
        ),
        structure=FunctionalGateTaylorProxy(gate_scores),
        weight=JointWeightTaylorProxy(
            context.model,
            statistics=fisher,
            unit_to_parameter_slices=slices,
            strict=True,
        ),
        activation=activation,
        gate_mapping=gate_mapping,
        calibration_sample_count=int(taylor_samples),
    )
    write_json(
        output_root / "reports/new_ga_proxy_contract.json",
        {
            "stage1_proxy": "J_struct_gate + J_WQ + J_AQ",
            "legacy_coupled_weight_taylor_used_for_fitness": False,
            "joint_cross_used_for_fitness": False,
            "sample_first": True,
            "fisher": "mean(g^2)",
            "absolute_gradient": "mean(abs(g))",
            "sample_count": int(taylor_samples),
            "domain_count": len(space.pruning_gene_ids),
            "precision_gene_count": len(space.precision_gene_ids),
            "gate_mapping_count": len(gate_mapping),
            "activation_mapping_count": len(activation.mapping),
            "search_loop_forward_calls": 0,
            "search_loop_backward_calls": 0,
            "search_loop_exports": 0,
            "search_loop_trt_builds": 0,
        },
    )
    return result


def decreasing_neighbors(
    candidate: CandidateGenotype,
    space: SearchSpaceSpec,
) -> list[tuple[str, str, CandidateGenotype]]:
    rows: list[tuple[str, str, CandidateGenotype]] = []
    for domain in space.pruning_domains:
        locus = str(domain.domain_id)
        if locus not in candidate.pruning_width_genes:
            continue
        states = tuple(sorted(int(value) for value in domain.legal_widths))
        index = states.index(int(candidate.pruning_width_genes[locus]))
        if index == 0:
            continue
        widths = dict(candidate.pruning_width_genes)
        widths[locus] = states[index - 1]
        rows.append((
            "structure",
            locus,
            CandidateGenotype(
                pruning_width_genes=widths,
                precision_genes=dict(candidate.precision_genes),
                meta={"created_by": "strict_greedy_adjacent_width", "repair_count": 0},
            ),
        ))
    groups = {str(group.group_id): group for group in space.quantization_groups}
    for locus in space.precision_gene_ids:
        states = tuple(groups[locus].allowed_precisions)
        index = states.index(candidate.precision_genes[locus])
        if index + 1 >= len(states):
            continue
        precision = dict(candidate.precision_genes)
        precision[locus] = states[index + 1]
        rows.append((
            "precision",
            locus,
            CandidateGenotype(
                pruning_width_genes=dict(candidate.pruning_width_genes),
                precision_genes=precision,
                meta={"created_by": "strict_greedy_adjacent_precision", "repair_count": 0},
            ),
        ))
    return rows


def greedy_anchors(
    prepared: PreparedCNNFormalSearch,
    *,
    targets: Sequence[float],
    output_root: Path,
    bops_tolerance_abs: float = 0.005,
    recovery_beam_width: int = 8,
    recovery_seed_pool_size: int = 32,
    recovery_max_depth: int = 64,
) -> dict[float, CandidateGenotype]:
    """Build exact-budget anchors without repair or budget projection.

    The primary trajectory remains the strict single-path Greedy search.  Two
    bounded recovery mechanisms from the audited generic Greedy engine are
    deliberately retained:

    * every already-evaluated legal neighbor may enter a budget frontier;
    * a finite target-directed beam may continue from legal above-budget
      frontier states when the selected path jumps across a budget band.

    A frontier candidate costs no additional search iteration because it was
    evaluated in the primary step.  Every beam *depth expansion* is an
    additional search iteration and is included in
    ``total_search_iteration_count``.  Candidate evaluations are reported
    separately because one iteration evaluates many neighbors in parallel.
    Neither mechanism changes a candidate to satisfy a budget.
    """

    if float(bops_tolerance_abs) <= 0.0:
        raise ValueError("strict_greedy_bops_tolerance_must_be_positive")
    if int(recovery_beam_width) <= 0:
        raise ValueError("strict_greedy_recovery_beam_width_must_be_positive")
    if int(recovery_seed_pool_size) < int(recovery_beam_width):
        raise ValueError("strict_greedy_recovery_seed_pool_smaller_than_beam")
    if int(recovery_max_depth) <= 0:
        raise ValueError("strict_greedy_recovery_max_depth_must_be_positive")
    target_values = tuple(sorted({float(value) for value in targets}, reverse=True))
    if not target_values:
        raise ValueError("strict_greedy_targets_empty")
    evaluator = prepared.evaluator(
        target=min(target_values),
        enforce_bops_hard_gate=False,
    )
    metrics_cache: dict[str, dict[str, Any]] = {}

    def resource_metrics(candidate: CandidateGenotype) -> dict[str, Any]:
        phenotype = canonicalize_candidate(candidate, prepared.space)
        identity = phenotype_identity(candidate, prepared.space)
        cache_key = str(identity["complete_phenotype_hash"])
        cached = metrics_cache.get(cache_key)
        if cached is not None:
            return dict(cached)
        size = canonical_size_metrics(prepared.size.evaluate_breakdown(phenotype))
        result = {
            **identity,
            **prepared.bops.evaluate_breakdown(phenotype),
            **size,
        }
        metrics_cache[cache_key] = dict(result)
        return result

    def action_risk(
        parent: CandidateGenotype,
        successor: CandidateGenotype,
        action_type: str,
        locus: str,
    ) -> float:
        parent_phenotype = canonicalize_candidate(parent, prepared.space)
        successor_phenotype = canonicalize_candidate(successor, prepared.space)
        if action_type == "structure":
            value = float(
                prepared.structure.pruning_action_breakdown(
                    parent_phenotype, successor_phenotype
                )["delta_J_prune"]
            )
        elif action_type == "precision":
            value = float(
                prepared.weight.weight_quantization_action_breakdown(
                    parent_phenotype, successor_phenotype
                )["delta_J_WQ"]
            ) + float(
                prepared.activation.action_breakdown(
                    parent_phenotype, successor_phenotype
                )["delta_J_AQ"]
            )
        else:
            raise RuntimeError(f"strict_greedy_unknown_action_type:{action_type}")
        if value < 0.0 or not math.isfinite(value):
            raise RuntimeError(f"strict_greedy_negative_or_nonfinite_risk:{locus}")
        return value

    def make_record(
        candidate: CandidateGenotype,
        metrics: Mapping[str, Any],
        cumulative_risk: float,
        *,
        source: str,
        primary_step: int | None,
        recovery_depth: int,
        search_iteration: int,
        selected_primary_path: bool,
        parent_hash: str | None = None,
        action_type: str | None = None,
        locus: str | None = None,
    ) -> dict[str, Any]:
        if cumulative_risk < 0.0 or not math.isfinite(cumulative_risk):
            raise RuntimeError("strict_greedy_invalid_cumulative_risk")
        return {
            "candidate": candidate,
            "metrics": dict(metrics),
            "cumulative_J_total": float(cumulative_risk),
            "capture_source": str(source),
            "capture_sources": [str(source)],
            "primary_step": primary_step,
            "recovery_depth": int(recovery_depth),
            "search_iteration": int(search_iteration),
            "selected_primary_path": bool(selected_primary_path),
            "parent_hash": parent_hash,
            "action_type": action_type,
            "locus": locus,
        }

    captured: dict[float, dict[str, dict[str, Any]]] = {
        target: {} for target in target_values
    }
    recovery_seed_pools: dict[float, dict[str, dict[str, Any]]] = {
        target: {} for target in target_values
    }
    nearest: dict[float, dict[str, Any]] = {}

    def merge_record(
        incumbent: dict[str, Any] | None,
        candidate_record: dict[str, Any],
    ) -> dict[str, Any]:
        if incumbent is None:
            return dict(candidate_record)
        incumbent_key = (
            float(incumbent["cumulative_J_total"]),
            int(incumbent["search_iteration"]),
            str(incumbent["metrics"]["complete_phenotype_hash"]),
        )
        candidate_key = (
            float(candidate_record["cumulative_J_total"]),
            int(candidate_record["search_iteration"]),
            str(candidate_record["metrics"]["complete_phenotype_hash"]),
        )
        result = dict(candidate_record if candidate_key < incumbent_key else incumbent)
        sources = sorted({
            *incumbent.get("capture_sources", [incumbent["capture_source"]]),
            *candidate_record.get(
                "capture_sources", [candidate_record["capture_source"]]
            ),
        })
        result["capture_sources"] = sources
        result["selected_primary_path"] = bool(
            incumbent.get("selected_primary_path", False)
            or candidate_record.get("selected_primary_path", False)
        )
        if result["selected_primary_path"]:
            result["capture_source"] = "primary_selected_trajectory"
        return result

    def update_budget_frontier(record: dict[str, Any]) -> None:
        retention = float(record["metrics"]["R_bops_vs_fp32"])
        candidate_hash = str(record["metrics"]["complete_phenotype_hash"])
        for target in target_values:
            nearest_key = (
                abs(retention - target),
                float(record["cumulative_J_total"]),
                candidate_hash,
            )
            incumbent_nearest = nearest.get(target)
            if incumbent_nearest is None:
                nearest[target] = dict(record)
            else:
                incumbent_key = (
                    abs(
                        float(incumbent_nearest["metrics"]["R_bops_vs_fp32"])
                        - target
                    ),
                    float(incumbent_nearest["cumulative_J_total"]),
                    str(incumbent_nearest["metrics"]["complete_phenotype_hash"]),
                )
                if nearest_key < incumbent_key:
                    nearest[target] = dict(record)

            if retention > target + float(bops_tolerance_abs):
                pool = recovery_seed_pools[target]
                pool[candidate_hash] = merge_record(pool.get(candidate_hash), record)
                if len(pool) > int(recovery_seed_pool_size):
                    retained = sorted(
                        pool.items(),
                        key=lambda row: (
                            float(row[1]["metrics"]["R_bops_vs_fp32"]) - target,
                            float(row[1]["cumulative_J_total"]),
                            row[0],
                        ),
                    )[: int(recovery_seed_pool_size)]
                    recovery_seed_pools[target] = dict(retained)

            if abs(retention - target) <= float(bops_tolerance_abs):
                pool = captured[target]
                pool[candidate_hash] = merge_record(pool.get(candidate_hash), record)

    def select_recovery_beam(
        records: Sequence[dict[str, Any]], target: float
    ) -> list[dict[str, Any]]:
        unique: dict[str, dict[str, Any]] = {}
        for record in records:
            candidate_hash = str(record["metrics"]["complete_phenotype_hash"])
            unique[candidate_hash] = merge_record(unique.get(candidate_hash), record)
        values = list(unique.values())
        closest = sorted(
            values,
            key=lambda row: (
                max(0.0, float(row["metrics"]["R_bops_vs_fp32"]) - target),
                float(row["cumulative_J_total"]),
                str(row["metrics"]["complete_phenotype_hash"]),
            ),
        )
        quality = sorted(
            values,
            key=lambda row: (
                float(row["cumulative_J_total"])
                / max(1.0 - float(row["metrics"]["R_bops_vs_fp32"]), 1.0e-18),
                float(row["cumulative_J_total"]),
                max(0.0, float(row["metrics"]["R_bops_vs_fp32"]) - target),
                str(row["metrics"]["complete_phenotype_hash"]),
            ),
        )
        selected: list[dict[str, Any]] = []
        selected_hashes: set[str] = set()
        for index in range(max(len(closest), len(quality))):
            for ordered in (closest, quality):
                if index >= len(ordered):
                    continue
                record = ordered[index]
                candidate_hash = str(record["metrics"]["complete_phenotype_hash"])
                if candidate_hash in selected_hashes:
                    continue
                selected.append(record)
                selected_hashes.add(candidate_hash)
                if len(selected) >= int(recovery_beam_width):
                    return selected
        return selected

    current = prepared.baseline
    current_metrics = resource_metrics(current)
    cumulative = 0.0
    trace: list[dict[str, Any]] = [{
        "step": 0,
        "candidate_hash": current_metrics["complete_phenotype_hash"],
        "R_bops_vs_fp32": current_metrics["R_bops_vs_fp32"],
        "cumulative_J_total": 0.0,
        "selected": True,
    }]
    initial_record = make_record(
        current,
        current_metrics,
        0.0,
        source="initial_candidate",
        primary_step=0,
        recovery_depth=0,
        search_iteration=0,
        selected_primary_path=True,
    )
    update_budget_frontier(initial_record)
    step = 0
    primary_neighbor_candidate_count = 0
    recovery_iteration_count = 0
    recovery_evaluated_candidate_count = 0
    recovery_trace: list[dict[str, Any]] = []
    epsilon = 1.0e-18
    while (
        float(current_metrics["R_bops_vs_fp32"])
        > min(target_values) - float(bops_tolerance_abs)
    ):
        actions = decreasing_neighbors(current, prepared.space)
        if not actions:
            break
        candidates: list[
            tuple[
                tuple[Any, ...], str, str, CandidateGenotype,
                dict[str, Any], float, float, dict[str, Any]
            ]
        ] = []
        for action_type, locus, successor in actions:
            validate_genotype_schema(successor, prepared.space)
            # Greedy action ordering only needs the action-local Taylor risk
            # and exact resource delta.  Computing the full candidate Taylor
            # for every unselected neighbor would repeat all earlier precision
            # transitions and is neither part of the utility nor the search
            # contract.
            successor_metrics = resource_metrics(successor)
            delta_bops = float(current_metrics["R_bops_vs_fp32"]) - float(
                successor_metrics["R_bops_vs_fp32"]
            )
            if delta_bops <= 0.0:
                continue
            delta_j = action_risk(current, successor, action_type, locus)
            utility = delta_j / max(delta_bops, epsilon)
            record = make_record(
                successor,
                successor_metrics,
                cumulative + delta_j,
                source="primary_evaluated_neighbor_frontier",
                primary_step=step + 1,
                recovery_depth=0,
                search_iteration=step + 1,
                selected_primary_path=False,
                parent_hash=str(current_metrics["complete_phenotype_hash"]),
                action_type=action_type,
                locus=locus,
            )
            update_budget_frontier(record)
            primary_neighbor_candidate_count += 1
            key = (
                utility,
                action_type,
                locus,
                str(successor_metrics["complete_phenotype_hash"]),
            )
            candidates.append((
                key, action_type, locus, successor, successor_metrics, delta_j,
                delta_bops, record
            ))
        if not candidates:
            break
        (
            _key, action_type, locus, successor, successor_metrics, delta_j,
            delta_bops, selected_record
        ) = min(
            candidates, key=lambda row: row[0]
        )
        cumulative += delta_j
        step += 1
        current = successor
        current_metrics = successor_metrics
        trace.append({
            "step": step,
            "action_type": action_type,
            "locus": locus,
            "candidate_hash": current_metrics["complete_phenotype_hash"],
            "R_bops_vs_fp32": current_metrics["R_bops_vs_fp32"],
            "delta_J_action": delta_j,
            "delta_R_bops": delta_bops,
            "utility": delta_j / max(delta_bops, epsilon),
            "cumulative_J_total": cumulative,
            "selected": True,
        })
        selected_record = {
            **selected_record,
            "capture_source": "primary_selected_trajectory",
            "capture_sources": sorted({
                *selected_record["capture_sources"],
                "primary_selected_trajectory",
            }),
            "selected_primary_path": True,
        }
        update_budget_frontier(selected_record)

    recovery_reports: dict[float, dict[str, Any]] = {}
    for target in target_values:
        if captured[target]:
            recovery_reports[target] = {
                "status": "reached_primary_frontier",
                "depth": 0,
                "search_iteration_count": 0,
                "evaluated_candidate_count": 0,
                "beam_width": int(recovery_beam_width),
                "stop_reason": "legal_primary_frontier_capture",
            }
            continue
        seeds = select_recovery_beam(
            list(recovery_seed_pools[target].values()) or [initial_record], target
        )
        beam = list(seeds)
        visited = {
            str(record["metrics"]["complete_phenotype_hash"]): record
            for record in beam
        }
        target_evaluated = 0
        target_iterations = 0
        reached_depth = 0
        stop_reason = "recovery_depth_exhausted"
        for depth in range(1, int(recovery_max_depth) + 1):
            expansion: dict[str, dict[str, Any]] = {}
            for parent_record in beam:
                parent = parent_record["candidate"]
                parent_metrics = parent_record["metrics"]
                for action_type, locus, successor in decreasing_neighbors(
                    parent, prepared.space
                ):
                    validate_genotype_schema(successor, prepared.space)
                    successor_metrics = resource_metrics(successor)
                    candidate_hash = str(
                        successor_metrics["complete_phenotype_hash"]
                    )
                    delta_bops = float(parent_metrics["R_bops_vs_fp32"]) - float(
                        successor_metrics["R_bops_vs_fp32"]
                    )
                    if delta_bops <= 0.0:
                        continue
                    delta_j = action_risk(parent, successor, action_type, locus)
                    record = make_record(
                        successor,
                        successor_metrics,
                        float(parent_record["cumulative_J_total"]) + delta_j,
                        source=f"target_directed_beam_recovery:{target:.6f}",
                        primary_step=parent_record.get("primary_step"),
                        recovery_depth=depth,
                        search_iteration=step + recovery_iteration_count + 1,
                        selected_primary_path=False,
                        parent_hash=str(parent_metrics["complete_phenotype_hash"]),
                        action_type=action_type,
                        locus=locus,
                    )
                    incumbent = expansion.get(candidate_hash)
                    expansion[candidate_hash] = merge_record(incumbent, record)
            expansion = {
                candidate_hash: record
                for candidate_hash, record in expansion.items()
                if candidate_hash not in visited
            }
            if not expansion:
                stop_reason = "no_unvisited_recovery_neighbor"
                break
            recovery_iteration_count += 1
            target_iterations += 1
            expansion_records = list(expansion.values())
            recovery_evaluated_candidate_count += len(expansion_records)
            target_evaluated += len(expansion_records)
            visited.update(expansion)
            feasible: list[dict[str, Any]] = []
            for record in expansion_records:
                update_budget_frontier(record)
                retention = float(record["metrics"]["R_bops_vs_fp32"])
                recovery_trace.append({
                    "target": target,
                    "recovery_depth": depth,
                    "search_iteration": int(record["search_iteration"]),
                    "candidate_hash": record["metrics"]["complete_phenotype_hash"],
                    "parent_hash": record["parent_hash"],
                    "action_type": record["action_type"],
                    "locus": record["locus"],
                    "R_bops_vs_fp32": retention,
                    "cumulative_J_total": record["cumulative_J_total"],
                })
                if retention >= target - float(bops_tolerance_abs):
                    feasible.append(record)
            if captured[target]:
                reached_depth = depth
                stop_reason = "strict_budget_reached"
                break
            if not feasible:
                stop_reason = "all_recovery_neighbors_undershoot_budget"
                break
            beam = select_recovery_beam(feasible, target)
            if not beam:
                stop_reason = "empty_recovery_beam"
                break
        recovery_reports[target] = {
            "status": (
                "reached_budget_recovery"
                if captured[target]
                else "unreachable_after_budget_recovery"
            ),
            "depth": int(reached_depth),
            "search_iteration_count": int(target_iterations),
            "evaluated_candidate_count": int(target_evaluated),
            "seed_count": len(seeds),
            "beam_width": int(recovery_beam_width),
            "maximum_depth": int(recovery_max_depth),
            "stop_reason": stop_reason,
        }
    write_csv(output_root / "reports/greedy_shared_trajectory.csv", trace)
    write_csv(output_root / "reports/greedy_recovery_trace.csv", recovery_trace)
    winners: dict[float, CandidateGenotype] = {}
    capture_rows: list[dict[str, Any]] = []
    winner_records: dict[float, dict[str, Any]] = {}
    for target in target_values:
        pool = list(captured[target].values())
        for record in pool:
            metrics = record["metrics"]
            capture_rows.append({
                "target": target,
                "candidate_hash": metrics["complete_phenotype_hash"],
                "R_bops_vs_fp32": metrics["R_bops_vs_fp32"],
                "bops_deviation": abs(float(metrics["R_bops_vs_fp32"]) - target),
                "cumulative_J_total": record["cumulative_J_total"],
                "capture_source": record["capture_source"],
                "capture_sources": "|".join(record["capture_sources"]),
                "selected_primary_path": record["selected_primary_path"],
                "primary_step": record["primary_step"],
                "recovery_depth": record["recovery_depth"],
                "search_iteration": record["search_iteration"],
            })
        if not pool:
            continue
        pool.sort(key=lambda row: (
            float(row["cumulative_J_total"]),
            abs(float(row["metrics"]["R_bops_vs_fp32"]) - target),
            -float(row["metrics"]["R_parameter_retention"]),
            -float(row["metrics"]["mixed_weight_retention"]),
            str(row["metrics"]["complete_phenotype_hash"]),
        ))
        winner_records[target] = pool[0]
        winners[target] = pool[0]["candidate"]
    write_csv(output_root / "reports/greedy_budget_capture.csv", capture_rows)
    write_json(
        output_root / "reports/greedy_exact_winners.json",
        {
            str(target): {
                "genotype": candidate.to_dict(),
                "identity": phenotype_identity(candidate, prepared.space),
                "metrics": evaluator(candidate),
                "cumulative_path_J_total": winner_records[target][
                    "cumulative_J_total"
                ],
                "capture_source": winner_records[target]["capture_source"],
                "capture_sources": winner_records[target]["capture_sources"],
                "selected_primary_path": winner_records[target][
                    "selected_primary_path"
                ],
                "primary_step": winner_records[target]["primary_step"],
                "recovery_depth": winner_records[target]["recovery_depth"],
                "search_iteration": winner_records[target]["search_iteration"],
            }
            for target, candidate in winners.items()
        },
    )
    write_json(
        output_root / "reports/greedy_search_audit.json",
        {
            "capture_contract": (
                "primary selected trajectory plus already-evaluated legal-neighbor "
                "frontier plus finite legal adjacent-action beam recovery"
            ),
            "budget_projection_used": False,
            "structure_repair_count": 0,
            "precision_repair_count": 0,
            "budget_repair_count": 0,
            "primary_selected_step_count": int(step),
            "primary_neighbor_candidate_count": int(
                primary_neighbor_candidate_count
            ),
            "recovery_iteration_count": int(recovery_iteration_count),
            "recovery_evaluated_candidate_count": int(
                recovery_evaluated_candidate_count
            ),
            "total_search_iteration_count": int(step + recovery_iteration_count),
            "iteration_accounting": (
                "one selected primary action or one beam depth expansion equals "
                "one search iteration; parallel neighbor candidates are counted "
                "separately"
            ),
            "bops_tolerance_abs": float(bops_tolerance_abs),
            "recovery_beam_width": int(recovery_beam_width),
            "recovery_seed_pool_size": int(recovery_seed_pool_size),
            "recovery_max_depth": int(recovery_max_depth),
            "recovery_reports": {
                str(target): report
                for target, report in recovery_reports.items()
            },
            "unreachable_targets": [
                target for target in target_values if target not in winners
            ],
        },
    )
    return winners


def build_initial_population(
    anchor: CandidateGenotype,
    *,
    space: SearchSpaceSpec,
    evaluator: UnifiedTaylorStage1Evaluator,
    seed: int,
    size: int = 64,
) -> list[CandidateGenotype]:
    rng = random.Random(seed)
    accepted: dict[str, CandidateGenotype] = {}
    anchor_metrics = dict(evaluator(anchor))
    if not anchor_metrics["bops_feasible"]:
        raise RuntimeError("formal_cnn_greedy_anchor_outside_budget_band")
    accepted[str(anchor_metrics["complete_phenotype_hash"])] = anchor
    attempts = 0
    maximum_attempts = size * 5000
    while len(accepted) < size and attempts < maximum_attempts:
        attempts += 1
        candidate = rng.choice(list(accepted.values())) if len(accepted) > 1 else anchor
        for _ in range(rng.randint(1, 12)):
            candidate = adjacent_mutation(candidate, space, rng)
        metrics = dict(evaluator(candidate))
        if not metrics["bops_feasible"]:
            continue
        accepted.setdefault(str(metrics["complete_phenotype_hash"]), candidate)
    if len(accepted) != size:
        raise RuntimeError(
            f"formal_cnn_initial_population_not_exactly_{size}:"
            f"generated={len(accepted)}:attempts={attempts}"
        )
    return list(accepted.values())


class CNNRealStage2Evaluator:
    def __init__(
        self,
        *,
        prepared: PreparedCNNFormalSearch,
        output_root: Path,
        budget_label: str,
        real_evaluator: Any,
    ) -> None:
        self.prepared = prepared
        self.output_root = output_root
        self.budget_label = budget_label
        self.real_evaluator = real_evaluator

    @staticmethod
    def _counts(raw: Mapping[str, Any]) -> tuple[int, int]:
        evaluated = int(
            raw.get("num_evaluated_frames", raw.get("evaluated", 0)) or 0
        )
        skipped = int(raw.get("num_skipped_frames", raw.get("skipped", 0)) or 0)
        return evaluated, skipped

    def __call__(self, genotype: CandidateGenotype, generation: int) -> Stage2Result:
        identity = phenotype_identity(genotype, self.prepared.space)
        complete_hash = str(identity["complete_phenotype_hash"])
        destination = self.output_root / (
            f"ga/budget_{self.budget_label}/stage2_cache/{complete_hash}"
        )
        result_path = destination / "strict_stage2_result.json"
        if result_path.is_file():
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            return Stage2Result(
                complete_phenotype_hash=complete_hash,
                genotype=genotype,
                status=str(payload["status"]),
                map=payload.get("mAP"),
                p50_ms=payload.get("p50_ms"),
                requested_realized_exact=bool(payload["requested_realized_exact"]),
                evaluated=int(payload["evaluated"]),
                skipped=int(payload["skipped"]),
                metadata=dict(payload.get("metadata") or {}),
            )
        phenotype = canonicalize_candidate(genotype, self.prepared.space)
        raw = self.real_evaluator.evaluate_candidate(
            phenotype,
            output_dir=destination,
            candidate_hash=complete_hash,
        )
        evaluated, skipped = self._counts(raw)
        qdq = dict(raw.get("qdq_realization_summary") or {})
        if self.prepared.spec.model_id == "pyramid":
            requested = int(qdq.get("requested_int8_group_count", 0))
            realized = int(qdq.get("realized_int8_group_count", 0))
            exact = bool(raw.get("status") == "ok" and requested == realized)
        else:
            exact = bool(
                raw.get("status") == "ok"
                and raw.get("precision_acceptance", False)
                and raw.get("merge_acceptance", False)
                and int(raw.get("requested_int8_count", 0))
                == int(raw.get("realized_int8_count", 0))
            )
        ok = bool(
            raw.get("status") == "ok"
            and exact
            and evaluated == 50
            and skipped == 0
            and math.isfinite(float(raw.get("mAP")))
            and math.isfinite(float(raw.get("forward_p50_ms")))
        )
        result = Stage2Result(
            complete_phenotype_hash=complete_hash,
            genotype=genotype,
            status="ok" if ok else str(raw.get("status", "failed")),
            map=float(raw["mAP"]) if ok else None,
            p50_ms=float(raw["forward_p50_ms"]) if ok else None,
            requested_realized_exact=exact,
            evaluated=evaluated,
            skipped=skipped,
            metadata={
                "generation": generation,
                "artifact_dir": str(destination),
                "engine_hash": raw.get("engine_hash", raw.get("engine_sha256", "")),
                "raw_status": raw.get("status"),
                "failure_reason": raw.get("failure_reason", ""),
                "precision_fallback": False,
            },
        )
        write_json(result_path, stage2_payload(result))
        return result


def create_real_evaluator(
    prepared: PreparedCNNFormalSearch,
    *,
    output_root: Path,
) -> Any:
    if prepared.spec.model_id == "pyramid":
        return LidarPyramidRealEvaluator(
            context=prepared.context,
            run_dir=output_root / "stage2_runtime",
            num_frames=50,
            warmup_frames=20,
            latency_rounds=1,
            stage2_config=Stage2ObjectiveConfig(
                latency_metric="forward_p50_ms",
                accuracy_reference="original_strict_fp32",
                latency_reference="original_strict_fp32",
            ),
        )
    if prepared.spec.strict_fp32_engine is None or not prepared.spec.strict_fp32_engine.is_file():
        raise RuntimeError("formal_cnn_strict_fp32_engine_missing")
    return HealLidarBaselineCandidateEvaluator(
        context=prepared.context,
        run_dir=output_root / "stage2_runtime",
        baseline_engine_path=prepared.spec.strict_fp32_engine,
        num_frames=50,
        warmup_frames=20,
        latency_rounds=1,
        objective_config=Stage2ObjectiveConfig(
            latency_metric="forward_p50_ms",
            accuracy_reference="original_strict_fp32",
            latency_reference="original_strict_fp32",
        ),
        dataloader_num_workers=8,
    )


def best_real_candidate(
    rows: Sequence[Stage2Result], greedy: Stage2Result
) -> Stage2Result:
    eligible = [
        row
        for row in rows
        if score_stage2(
            row,
            greedy_map=float(greedy.map),
            greedy_p50_ms=float(greedy.p50_ms),
        )["eligible"]
    ]
    if not eligible:
        return greedy
    best = min(
        eligible,
        key=lambda row: (
            float(score_stage2(
                row,
                greedy_map=float(greedy.map),
                greedy_p50_ms=float(greedy.p50_ms),
            )["F_S2"]),
            -float(row.map),
            float(row.p50_ms),
            row.complete_phenotype_hash,
        ),
    )
    greedy_score = float(score_stage2(
        greedy,
        greedy_map=float(greedy.map),
        greedy_p50_ms=float(greedy.p50_ms),
    )["F_S2"])
    best_score = float(score_stage2(
        best,
        greedy_map=float(greedy.map),
        greedy_p50_ms=float(greedy.p50_ms),
    )["F_S2"])
    return best if best_score < greedy_score else greedy


def run_budget(
    prepared: PreparedCNNFormalSearch,
    *,
    target: float,
    anchor_genotype: CandidateGenotype,
    output_root: Path,
    seed: int,
    generations: int,
    real_evaluator: Any,
) -> dict[str, Any]:
    label = f"{int(round(target * 100)):03d}"
    stage1 = prepared.evaluator(target=target, enforce_bops_hard_gate=True)
    initial = build_initial_population(
        anchor_genotype,
        space=prepared.space,
        evaluator=stage1,
        seed=seed + int(round(target * 1000)),
        size=64,
    )
    stage2 = CNNRealStage2Evaluator(
        prepared=prepared,
        output_root=output_root,
        budget_label=label,
        real_evaluator=real_evaluator,
    )
    greedy = stage2(anchor_genotype, 0)
    if not greedy.deployable:
        raise RuntimeError(f"formal_cnn_greedy_anchor_not_deployable:budget_{label}")
    config = StrictGAConfig(
        target_bops_retention=target,
        tolerance_abs=0.005,
        population_size=64,
        offspring_size=64,
        generations=generations,
        stage2_new_candidate_quota=5,
        random_seed=seed,
        generation_contract="formal_gen5",
    )
    runner = StrictStage12V3Runner(
        prepared.space,
        config,
        stage1_evaluator=stage1,
        stage2_evaluator=stage2,
    )
    budget_dir = output_root / f"ga/budget_{label}/seed_{seed}"

    def callback(record: Mapping[str, Any]) -> None:
        generation = int(record["generation"])
        destination = budget_dir / f"generation_{generation:02d}"
        write_json(destination / "generation_summary.json", dict(record))
        print(json.dumps({
            "event": "formal_cnn_ga_generation",
            "model": prepared.spec.model_id,
            "budget": target,
            "generation": generation,
            "stage2_new_candidate_count": record.get("stage2_new_candidate_count", 0),
            "generation_winner_hash": record.get("generation_winner_hash"),
        }, sort_keys=True), flush=True)

    result = runner.run(initial, greedy_anchor=greedy, generation_callback=callback)
    final = best_real_candidate(list(result["evaluated"].values()), greedy)
    summary = {
        "model": prepared.spec.model_id,
        "target_bops": target,
        "seed": seed,
        "generation_zero_counted": result["generation_zero_counted"],
        "completed_evolution_generations": result["completed_evolution_generations"],
        "termination_reason": result["termination_reason"],
        "greedy_anchor": stage2_payload(greedy),
        "global_anchors": [stage2_payload(row) for row in result["anchors"].unique()],
        "final_winner": stage2_payload(final),
        "ga_improved_greedy": final.complete_phenotype_hash != greedy.complete_phenotype_hash,
        "stage2_real_evaluation_count": len(result["evaluated"]) - 1,
        "repair_counts": result["formal_ga_repair_counts"],
        "population_size": 64,
        "offspring_size": 64,
        "survivor_size": 64,
        "stage2_new_candidate_quota": 5,
        "framework": "StrictStage12V3Runner",
    }
    write_json(budget_dir / "budget_summary.json", summary)
    return summary


__all__ = [
    "CNNFormalModelSpec",
    "MODEL_SPECS",
    "PreparedCNNFormalSearch",
    "baseline_genotype",
    "best_real_candidate",
    "build_initial_population",
    "canonical_size_metrics",
    "create_real_evaluator",
    "greedy_anchors",
    "prepare_search",
    "run_budget",
    "stage2_payload",
    "write_csv",
    "write_json",
]
