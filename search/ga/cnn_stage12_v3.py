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
from .stage12_v3 import (
    Stage2Result,
    StrictGAConfig,
    StrictStage12V3Runner,
    UnifiedTaylorStage1Evaluator,
    adjacent_mutation,
    phenotype_identity,
    score_stage2,
    validate_genotype_schema,
)


STAGE2_SCREENING_FRAMES = 300
STAGE2_SCREENING_WARMUP_FRAMES = 100
STAGE2_SCREENING_PROTOCOL = "top5_fixed300_warmup100_screening"
GENERATION_WINNER_FRAMES = 500
GENERATION_WINNER_WARMUP_FRAMES = 200
GENERATION_WINNER_PROTOCOL = "generation_winner_fixed500_warmup200"
EVALUATION_MANIFEST_FRAMES = GENERATION_WINNER_FRAMES
EVALUATION_MANIFEST_WARMUP_FRAMES = GENERATION_WINNER_WARMUP_FRAMES


@dataclass(frozen=True)
class CNNFormalModelSpec:
    model_id: str
    family_id: str
    checkpoint: Path
    config: Path
    calibration_manifest: Path
    strict_fp32_engine: Path | None = None


MODEL_SPECS: dict[str, CNNFormalModelSpec] = {
    "attfusion": CNNFormalModelSpec(
        model_id="attfusion",
        family_id="heal_lidar_attfusion",
        checkpoint=Path("/home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_attfuse/net_epoch_bestval_at33.pth"),
        config=Path("/home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_attfuse/config.yaml"),
        calibration_manifest=Path("/home/lixingfeng/UniAD_examine/heal_compress/outputs/heal_lidar_baseline_train200_fixedk29696_20260719_1215_v2/calibration_manifest.json"),
        strict_fp32_engine=Path("/home/lixingfeng/UniAD_examine/heal_compress/outputs/h800_dair_lidar_trt_fp32_all_models_smoke_20260718_2138/lidar_attfuse/strict_fp32.plan"),
    ),
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
            num_frames=EVALUATION_MANIFEST_FRAMES,
            warmup_frames=EVALUATION_MANIFEST_WARMUP_FRAMES,
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
        num_frames=EVALUATION_MANIFEST_FRAMES,
        warmup_frames=EVALUATION_MANIFEST_WARMUP_FRAMES,
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


def size_metrics_with_alias(size_proxy: Any, phenotype: Any) -> dict[str, Any]:
    """Expose the selector alias without changing the SizeProxy schema."""

    size = dict(size_proxy.evaluate_breakdown(phenotype))
    size.setdefault("mixed_weight_retention", float(size["R_size_vs_fp32"]))
    return size


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
        size = size_metrics_with_alias(prepared.size, phenotype)
        # ``SizeProxy`` owns the canonical report name.  Stage-1/selector code
        # uses the semantic alias so all model adapters share one tie-break
        # contract without requiring a second size calculation.
        result = {
            **identity,
            **prepared.bops.evaluate_breakdown(phenotype),
            **size,
        }
        metrics_cache[cache_key] = dict(result)
        return result

    epsilon = 1.0e-18

    def run_path(
        activation_taylor_weight: float,
    ) -> tuple[
        list[dict[str, Any]],
        dict[float, list[tuple[float, CandidateGenotype, dict[str, Any]]]],
        dict[float, dict[str, dict[str, Any]]],
        list[dict[str, Any]],
        dict[str, Any],
    ]:
        current = prepared.baseline
        current_metrics = resource_metrics(current)
        cumulative = 0.0
        cumulative_struct = 0.0
        cumulative_wq = 0.0
        cumulative_aq = 0.0
        trace: list[dict[str, Any]] = [{
            "step": 0,
            "candidate_hash": current_metrics["complete_phenotype_hash"],
            "R_bops_vs_fp32": current_metrics["R_bops_vs_fp32"],
            "activation_taylor_weight": float(activation_taylor_weight),
            "cumulative_J_struct": 0.0,
            "cumulative_J_WQ": 0.0,
            "cumulative_J_AQ": 0.0,
            "cumulative_J_total": 0.0,
            "selected": True,
        }]
        captured_records: dict[float, dict[str, dict[str, Any]]] = {
            target: {} for target in target_values
        }
        recovery_seed_pools: dict[float, dict[str, dict[str, Any]]] = {
            target: {} for target in target_values
        }
        recovery_trace: list[dict[str, Any]] = []
        recovery_iteration_count = 0
        recovery_evaluated_candidate_count = 0
        primary_neighbor_candidate_count = 0
        structure_action_count = 0
        precision_action_count = 0

        def merge_record(
            incumbent: dict[str, Any] | None,
            record: dict[str, Any],
        ) -> dict[str, Any]:
            if incumbent is None:
                return dict(record)
            incumbent_key = (
                float(incumbent["cumulative_J_total"]),
                not bool(incumbent["selected_primary_path"]),
                str(incumbent["metrics"]["complete_phenotype_hash"]),
            )
            record_key = (
                float(record["cumulative_J_total"]),
                not bool(record["selected_primary_path"]),
                str(record["metrics"]["complete_phenotype_hash"]),
            )
            result = dict(record if record_key < incumbent_key else incumbent)
            result["capture_sources"] = sorted({
                *incumbent.get("capture_sources", [incumbent["capture_source"]]),
                *record.get("capture_sources", [record["capture_source"]]),
            })
            result["selected_primary_path"] = bool(
                incumbent["selected_primary_path"]
                or record["selected_primary_path"]
            )
            if result["selected_primary_path"]:
                result["capture_source"] = "primary_selected_trajectory"
            return result

        def update_frontier(record: dict[str, Any]) -> None:
            retention = float(record["metrics"]["R_bops_vs_fp32"])
            candidate_hash = str(
                record["metrics"]["complete_phenotype_hash"]
            )
            for target in target_values:
                if retention > target + float(bops_tolerance_abs):
                    pool = recovery_seed_pools[target]
                    pool[candidate_hash] = merge_record(
                        pool.get(candidate_hash), record
                    )
                    if len(pool) > int(recovery_seed_pool_size):
                        retained = sorted(
                            pool.items(),
                            key=lambda row: (
                                float(row[1]["metrics"]["R_bops_vs_fp32"])
                                - target,
                                float(row[1]["cumulative_J_total"]),
                                row[0],
                            ),
                        )[: int(recovery_seed_pool_size)]
                        recovery_seed_pools[target] = dict(retained)
                if abs(retention - target) <= float(bops_tolerance_abs):
                    pool = captured_records[target]
                    pool[candidate_hash] = merge_record(
                        pool.get(candidate_hash), record
                    )

        def make_record(
            candidate: CandidateGenotype,
            metrics: Mapping[str, Any],
            *,
            cumulative_total: float,
            cumulative_structure: float,
            cumulative_weight: float,
            cumulative_activation: float,
            structure_actions: int,
            precision_actions: int,
            source: str,
            selected_primary_path: bool,
            primary_step: int,
            recovery_depth: int = 0,
            parent_hash: str = "",
            action_type: str = "",
            locus: str = "",
        ) -> dict[str, Any]:
            return {
                "candidate": candidate,
                "metrics": dict(metrics),
                "cumulative_J_total": float(cumulative_total),
                "cumulative_J_struct": float(cumulative_structure),
                "cumulative_J_WQ": float(cumulative_weight),
                "cumulative_J_AQ": float(cumulative_activation),
                "structure_action_count": int(structure_actions),
                "precision_action_count": int(precision_actions),
                "capture_source": source,
                "capture_sources": [source],
                "selected_primary_path": bool(selected_primary_path),
                "primary_step": int(primary_step),
                "recovery_depth": int(recovery_depth),
                "parent_hash": parent_hash,
                "action_type": action_type,
                "locus": locus,
            }

        def action_components(
            parent: CandidateGenotype,
            successor: CandidateGenotype,
            action_type: str,
        ) -> tuple[float, float, float]:
            parent_phenotype = canonicalize_candidate(parent, prepared.space)
            successor_phenotype = canonicalize_candidate(
                successor, prepared.space
            )
            delta_struct = 0.0
            delta_wq = 0.0
            delta_aq = 0.0
            if action_type == "structure":
                delta_struct = float(
                    prepared.structure.pruning_action_breakdown(
                        parent_phenotype, successor_phenotype
                    )["delta_J_prune"]
                )
            else:
                delta_wq = float(
                    prepared.weight.weight_quantization_action_breakdown(
                        parent_phenotype, successor_phenotype
                    )["delta_J_WQ"]
                )
                delta_aq = float(
                    prepared.activation.action_breakdown(
                        parent_phenotype, successor_phenotype
                    )["delta_J_AQ"]
                )
            return delta_struct, delta_wq, delta_aq

        update_frontier(make_record(
            current,
            current_metrics,
            cumulative_total=0.0,
            cumulative_structure=0.0,
            cumulative_weight=0.0,
            cumulative_activation=0.0,
            structure_actions=0,
            precision_actions=0,
            source="initial_candidate",
            selected_primary_path=True,
            primary_step=0,
        ))
        step = 0
        while (
            float(current_metrics["R_bops_vs_fp32"])
            > min(target_values) - float(bops_tolerance_abs)
        ):
            actions = decreasing_neighbors(current, prepared.space)
            if not actions:
                break
            candidates: list[tuple[Any, ...]] = []
            for action_type, locus, successor in actions:
                validate_genotype_schema(successor, prepared.space)
                successor_metrics = resource_metrics(successor)
                delta_bops = float(current_metrics["R_bops_vs_fp32"]) - float(
                    successor_metrics["R_bops_vs_fp32"]
                )
                if delta_bops <= 0.0:
                    continue
                delta_struct, delta_wq, delta_aq = action_components(
                    current, successor, action_type
                )
                delta_j = delta_struct + delta_wq + (
                    float(activation_taylor_weight) * delta_aq
                )
                if delta_j < 0.0 or not math.isfinite(delta_j):
                    raise RuntimeError(
                        f"strict_greedy_negative_or_nonfinite_risk:{locus}"
                    )
                record = make_record(
                    successor,
                    successor_metrics,
                    cumulative_total=cumulative + delta_j,
                    cumulative_structure=cumulative_struct + delta_struct,
                    cumulative_weight=cumulative_wq + delta_wq,
                    cumulative_activation=cumulative_aq + delta_aq,
                    structure_actions=(
                        structure_action_count + int(action_type == "structure")
                    ),
                    precision_actions=(
                        precision_action_count + int(action_type == "precision")
                    ),
                    source="primary_evaluated_neighbor_frontier",
                    selected_primary_path=False,
                    primary_step=step + 1,
                    parent_hash=str(
                        current_metrics["complete_phenotype_hash"]
                    ),
                    action_type=action_type,
                    locus=locus,
                )
                update_frontier(record)
                primary_neighbor_candidate_count += 1
                utility = delta_j / max(delta_bops, epsilon)
                key = (
                    utility,
                    action_type,
                    locus,
                    str(successor_metrics["complete_phenotype_hash"]),
                )
                candidates.append((
                    key,
                    action_type,
                    locus,
                    successor,
                    successor_metrics,
                    delta_j,
                    delta_bops,
                    delta_struct,
                    delta_wq,
                    delta_aq,
                    record,
                ))
            if not candidates:
                break
            (
                _key,
                action_type,
                locus,
                successor,
                successor_metrics,
                delta_j,
                delta_bops,
                delta_struct,
                delta_wq,
                delta_aq,
                selected_record,
            ) = min(candidates, key=lambda row: row[0])
            cumulative += delta_j
            cumulative_struct += delta_struct
            cumulative_wq += delta_wq
            cumulative_aq += delta_aq
            structure_action_count += int(action_type == "structure")
            precision_action_count += int(action_type == "precision")
            step += 1
            current = successor
            current_metrics = successor_metrics
            trace.append({
                "step": step,
                "action_type": action_type,
                "locus": locus,
                "candidate_hash": current_metrics["complete_phenotype_hash"],
                "R_bops_vs_fp32": current_metrics["R_bops_vs_fp32"],
                "activation_taylor_weight": float(activation_taylor_weight),
                "delta_J_struct": delta_struct,
                "delta_J_WQ": delta_wq,
                "delta_J_AQ": delta_aq,
                "delta_J_action": delta_j,
                "delta_R_bops": delta_bops,
                "utility": delta_j / max(delta_bops, epsilon),
                "cumulative_J_struct": cumulative_struct,
                "cumulative_J_WQ": cumulative_wq,
                "cumulative_J_AQ": cumulative_aq,
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
            update_frontier(selected_record)

        recovery_reports: dict[str, Any] = {}
        for target in target_values:
            if captured_records[target]:
                recovery_reports[str(target)] = {
                    "status": "reached_primary_frontier",
                    "search_iteration_count": 0,
                    "evaluated_candidate_count": 0,
                }
                continue
            seeds = sorted(
                recovery_seed_pools[target].values(),
                key=lambda row: (
                    float(row["metrics"]["R_bops_vs_fp32"]) - target,
                    float(row["cumulative_J_total"]),
                    str(row["metrics"]["complete_phenotype_hash"]),
                ),
            )[: int(recovery_beam_width)]
            beam = list(seeds)
            visited = {
                str(row["metrics"]["complete_phenotype_hash"])
                for row in beam
            }
            target_iterations = 0
            target_evaluated = 0
            stop_reason = "recovery_depth_exhausted"
            reached_depth = 0
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
                        if candidate_hash in visited:
                            continue
                        delta_bops = float(
                            parent_metrics["R_bops_vs_fp32"]
                        ) - float(successor_metrics["R_bops_vs_fp32"])
                        if delta_bops <= 0.0:
                            continue
                        delta_struct, delta_wq, delta_aq = action_components(
                            parent, successor, action_type
                        )
                        delta_j = delta_struct + delta_wq + (
                            float(activation_taylor_weight) * delta_aq
                        )
                        if delta_j < 0.0 or not math.isfinite(delta_j):
                            raise RuntimeError(
                                "strict_greedy_recovery_invalid_risk:"
                                f"{locus}"
                            )
                        record = make_record(
                            successor,
                            successor_metrics,
                            cumulative_total=(
                                float(parent_record["cumulative_J_total"])
                                + delta_j
                            ),
                            cumulative_structure=(
                                float(parent_record["cumulative_J_struct"])
                                + delta_struct
                            ),
                            cumulative_weight=(
                                float(parent_record["cumulative_J_WQ"])
                                + delta_wq
                            ),
                            cumulative_activation=(
                                float(parent_record["cumulative_J_AQ"])
                                + delta_aq
                            ),
                            structure_actions=(
                                int(parent_record["structure_action_count"])
                                + int(action_type == "structure")
                            ),
                            precision_actions=(
                                int(parent_record["precision_action_count"])
                                + int(action_type == "precision")
                            ),
                            source=(
                                "target_directed_beam_recovery:"
                                f"{target:.6f}"
                            ),
                            selected_primary_path=False,
                            primary_step=int(parent_record["primary_step"]),
                            recovery_depth=depth,
                            parent_hash=str(
                                parent_metrics["complete_phenotype_hash"]
                            ),
                            action_type=action_type,
                            locus=locus,
                        )
                        expansion[candidate_hash] = merge_record(
                            expansion.get(candidate_hash), record
                        )
                if not expansion:
                    stop_reason = "no_unvisited_recovery_neighbor"
                    break
                target_iterations += 1
                recovery_iteration_count += 1
                target_evaluated += len(expansion)
                recovery_evaluated_candidate_count += len(expansion)
                visited.update(expansion)
                feasible: list[dict[str, Any]] = []
                for record in expansion.values():
                    update_frontier(record)
                    retention = float(record["metrics"]["R_bops_vs_fp32"])
                    recovery_trace.append({
                        "target": target,
                        "activation_taylor_weight": activation_taylor_weight,
                        "recovery_depth": depth,
                        "candidate_hash": record["metrics"][
                            "complete_phenotype_hash"
                        ],
                        "parent_hash": record["parent_hash"],
                        "action_type": record["action_type"],
                        "locus": record["locus"],
                        "R_bops_vs_fp32": retention,
                        "cumulative_J_total": record["cumulative_J_total"],
                    })
                    if retention >= target - float(bops_tolerance_abs):
                        feasible.append(record)
                if captured_records[target]:
                    reached_depth = depth
                    stop_reason = "strict_budget_reached"
                    break
                if not feasible:
                    stop_reason = "all_recovery_neighbors_undershoot_budget"
                    break
                beam = sorted(
                    feasible,
                    key=lambda row: (
                        max(
                            0.0,
                            float(row["metrics"]["R_bops_vs_fp32"])
                            - target,
                        ),
                        float(row["cumulative_J_total"]),
                        str(row["metrics"]["complete_phenotype_hash"]),
                    ),
                )[: int(recovery_beam_width)]
            recovery_reports[str(target)] = {
                "status": (
                    "reached_budget_recovery"
                    if captured_records[target]
                    else "unreachable_after_budget_recovery"
                ),
                "depth": reached_depth,
                "search_iteration_count": target_iterations,
                "evaluated_candidate_count": target_evaluated,
                "stop_reason": stop_reason,
            }

        captured = {
            target: [
                (
                    float(record["cumulative_J_total"]),
                    record["candidate"],
                    dict(record["metrics"]),
                )
                for record in records.values()
            ]
            for target, records in captured_records.items()
        }
        audit = {
            "primary_selected_step_count": step,
            "primary_neighbor_candidate_count": primary_neighbor_candidate_count,
            "recovery_iteration_count": recovery_iteration_count,
            "recovery_evaluated_candidate_count": (
                recovery_evaluated_candidate_count
            ),
            "total_search_iteration_count": step + recovery_iteration_count,
            "recovery_reports": recovery_reports,
            "unreachable_targets": [
                target for target in target_values if not captured[target]
            ],
            "budget_projection_used": False,
            "structure_repair_count": 0,
            "precision_repair_count": 0,
            "budget_repair_count": 0,
        }
        return (
            trace,
            captured,
            captured_records,
            recovery_trace,
            audit,
        )

    (
        trace,
        captured,
        capture_details,
        recovery_trace,
        search_audit,
    ) = run_path(1.0)
    (
        counterfactual_trace,
        counterfactual_captured,
        counterfactual_capture_details,
        counterfactual_recovery_trace,
        counterfactual_search_audit,
    ) = run_path(0.0)
    write_csv(output_root / "reports/greedy_shared_trajectory.csv", trace)
    write_csv(
        output_root / "reports/greedy_shared_trajectory_activation_taylor_zero.csv",
        counterfactual_trace,
    )
    write_csv(
        output_root / "reports/greedy_recovery_trace.csv",
        recovery_trace,
    )
    write_csv(
        output_root / "reports/greedy_recovery_trace_activation_taylor_zero.csv",
        counterfactual_recovery_trace,
    )
    write_json(
        output_root / "reports/greedy_search_audit.json",
        {
            "capture_contract": (
                "selected trajectory plus evaluated legal-neighbor frontier "
                "plus finite adjacent-action beam recovery"
            ),
            "bops_tolerance_abs": float(bops_tolerance_abs),
            "recovery_beam_width": int(recovery_beam_width),
            "recovery_seed_pool_size": int(recovery_seed_pool_size),
            "recovery_max_depth": int(recovery_max_depth),
            "activation_taylor_enabled": search_audit,
            "activation_taylor_zero_counterfactual": (
                counterfactual_search_audit
            ),
        },
    )
    winners: dict[float, CandidateGenotype] = {}
    winner_capture_details: dict[float, dict[str, Any]] = {}
    capture_rows: list[dict[str, Any]] = []
    for target in sorted(captured, reverse=True):
        pool = captured[target]
        for risk, candidate, metrics in pool:
            detail = capture_details[target][
                str(metrics["complete_phenotype_hash"])
            ]
            capture_rows.append({
                "target": target,
                "candidate_hash": metrics["complete_phenotype_hash"],
                "R_bops_vs_fp32": metrics["R_bops_vs_fp32"],
                "bops_deviation": abs(float(metrics["R_bops_vs_fp32"]) - target),
                "cumulative_J_total": risk,
                "capture_source": detail["capture_source"],
                "capture_sources": "|".join(detail["capture_sources"]),
                "selected_primary_path": detail["selected_primary_path"],
                "primary_step": detail["primary_step"],
                "recovery_depth": detail["recovery_depth"],
            })
        if not pool:
            continue
        pool.sort(key=lambda row: (
            float(row[0]),
            abs(float(row[2]["R_bops_vs_fp32"]) - target),
            -float(row[2]["R_parameter_retention"]),
            -float(row[2]["mixed_weight_retention"]),
            str(row[2]["complete_phenotype_hash"]),
        ))
        winners[target] = pool[0][1]
        winner_hash = str(pool[0][2]["complete_phenotype_hash"])
        winner_capture_details[target] = capture_details[target][winner_hash]
    write_csv(output_root / "reports/greedy_budget_capture.csv", capture_rows)
    write_json(
        output_root / "reports/greedy_exact_winners.json",
        {
            str(target): {
                "genotype": candidate.to_dict(),
                "identity": phenotype_identity(candidate, prepared.space),
                "metrics": evaluator(candidate),
                "capture": {
                    key: winner_capture_details[target][key]
                    for key in (
                        "cumulative_J_total",
                        "cumulative_J_struct",
                        "cumulative_J_WQ",
                        "cumulative_J_AQ",
                        "structure_action_count",
                        "precision_action_count",
                        "capture_source",
                        "capture_sources",
                        "selected_primary_path",
                        "primary_step",
                        "recovery_depth",
                    )
                },
            }
            for target, candidate in winners.items()
        },
    )
    counterfactual_winners: dict[float, CandidateGenotype] = {}
    counterfactual_winner_capture_details: dict[float, dict[str, Any]] = {}
    for target, pool in counterfactual_captured.items():
        if not pool:
            continue
        pool.sort(key=lambda row: (
            float(row[0]),
            abs(float(row[2]["R_bops_vs_fp32"]) - target),
            -float(row[2]["R_parameter_retention"]),
            -float(row[2]["mixed_weight_retention"]),
            str(row[2]["complete_phenotype_hash"]),
        ))
        counterfactual_winners[target] = pool[0][1]
        winner_hash = str(pool[0][2]["complete_phenotype_hash"])
        counterfactual_winner_capture_details[target] = (
            counterfactual_capture_details[target][winner_hash]
        )

    def trace_summary(
        _rows: Sequence[Mapping[str, Any]],
        candidate: CandidateGenotype,
        detail: Mapping[str, Any],
    ) -> dict[str, Any]:
        identity = phenotype_identity(candidate, prepared.space)
        wanted = str(identity["complete_phenotype_hash"])
        metrics = resource_metrics(candidate)
        precision_counts = {
            value: sum(
                str(precision) == value
                for precision in candidate.precision_genes.values()
            )
            for value in ("FP32", "FP16", "INT8")
        }
        return {
            "candidate_hash": wanted,
            "R_bops_vs_fp32": float(metrics["R_bops_vs_fp32"]),
            "R_parameter_retention": float(metrics["R_parameter_retention"]),
            "mixed_weight_retention": float(metrics["mixed_weight_retention"]),
            "structure_action_count": int(detail["structure_action_count"]),
            "precision_action_count": int(detail["precision_action_count"]),
            "precision_counts": precision_counts,
            "cumulative_J_struct": float(detail["cumulative_J_struct"]),
            "cumulative_J_WQ": float(detail["cumulative_J_WQ"]),
            "cumulative_J_AQ": float(detail["cumulative_J_AQ"]),
            "cumulative_J_total": float(detail["cumulative_J_total"]),
            "capture_source": str(detail["capture_source"]),
            "recovery_depth": int(detail["recovery_depth"]),
        }

    comparisons: dict[str, Any] = {}
    pruning_shift_votes = 0
    comparable = 0
    for target in sorted(set(winners) | set(counterfactual_winners), reverse=True):
        main = (
            trace_summary(
                trace,
                winners[target],
                winner_capture_details[target],
            )
            if target in winners
            else None
        )
        without_aq = (
            trace_summary(
                counterfactual_trace,
                counterfactual_winners[target],
                counterfactual_winner_capture_details[target],
            )
            if target in counterfactual_winners
            else None
        )
        if main is not None and without_aq is not None:
            comparable += 1
            pushes = bool(
                main["structure_action_count"]
                > without_aq["structure_action_count"]
                or main["R_parameter_retention"]
                < without_aq["R_parameter_retention"] - 1.0e-12
            )
            pruning_shift_votes += int(pushes)
        else:
            pushes = None
        comparisons[str(target)] = {
            "activation_taylor_enabled": main,
            "activation_taylor_zero_counterfactual": without_aq,
            "activation_taylor_pushes_toward_more_pruning": pushes,
        }
    write_json(
        output_root / "reports/activation_taylor_pruning_bias_audit.json",
        {
            "schema_version": "greedy-activation-taylor-pruning-bias-v1",
            "main_search_uses_activation_taylor": True,
            "counterfactual_used_for_winner_selection": False,
            "counterfactual_forward_calls": 0,
            "counterfactual_backward_calls": 0,
            "counterfactual_physical_exports": 0,
            "counterfactual_engine_builds": 0,
            "comparable_budget_count": comparable,
            "budgets_with_more_pruning_under_activation_taylor": (
                pruning_shift_votes
            ),
            "activation_taylor_systematically_pushes_toward_pruning": bool(
                comparable > 0 and pruning_shift_votes > comparable / 2
            ),
            "budget_comparisons": comparisons,
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
        cache_namespace: str = "stage2_screening_cache",
        evaluation_frames: int = STAGE2_SCREENING_FRAMES,
        evaluation_warmup_frames: int = STAGE2_SCREENING_WARMUP_FRAMES,
        evaluation_protocol: str = STAGE2_SCREENING_PROTOCOL,
    ) -> None:
        self.prepared = prepared
        self.output_root = output_root
        self.budget_label = budget_label
        self.real_evaluator = real_evaluator
        self.cache_namespace = str(cache_namespace)
        self.evaluation_frames = int(evaluation_frames)
        self.evaluation_warmup_frames = int(evaluation_warmup_frames)
        self.evaluation_protocol = str(evaluation_protocol)

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
            f"ga/budget_{self.budget_label}/{self.cache_namespace}/{complete_hash}"
        )
        result_path = destination / "strict_stage2_result.json"
        if result_path.is_file():
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            metadata = dict(payload.get("metadata") or {})
            if (
                int(metadata.get("evaluation_frames", -1)) != self.evaluation_frames
                or int(metadata.get("evaluation_warmup_frames", -1))
                != self.evaluation_warmup_frames
                or str(metadata.get("evaluation_protocol", ""))
                != self.evaluation_protocol
            ):
                raise RuntimeError(
                    f"cnn_stage2_cache_protocol_mismatch:{complete_hash}:{metadata}"
                )
            return Stage2Result(
                complete_phenotype_hash=complete_hash,
                genotype=genotype,
                status=str(payload["status"]),
                map=payload.get("mAP"),
                p50_ms=payload.get("p50_ms"),
                requested_realized_exact=bool(payload["requested_realized_exact"]),
                evaluated=int(payload["evaluated"]),
                skipped=int(payload["skipped"]),
                metadata=metadata,
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
                and (
                    self.prepared.spec.model_id != "cobevt"
                    or (
                        raw.get("transformer_attention_fp32_acceptance", False)
                        and raw.get(
                            "transformer_functional_precision_acceptance", False
                        )
                    )
                )
                and int(raw.get("requested_int8_count", 0))
                == int(raw.get("realized_int8_count", 0))
            )
        ok = bool(
            raw.get("status") == "ok"
            and exact
            and evaluated == self.evaluation_frames
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
                "evaluation_frames": self.evaluation_frames,
                "evaluation_warmup_frames": self.evaluation_warmup_frames,
                "evaluation_protocol": self.evaluation_protocol,
            },
        )
        write_json(result_path, stage2_payload(result))
        return result


def create_real_evaluator(
    prepared: PreparedCNNFormalSearch,
    *,
    output_root: Path,
    num_frames: int = STAGE2_SCREENING_FRAMES,
    warmup_frames: int = STAGE2_SCREENING_WARMUP_FRAMES,
    run_dir_name: str = "stage2_screening_runtime",
) -> Any:
    if prepared.spec.model_id == "pyramid":
        return LidarPyramidRealEvaluator(
            context=prepared.context,
            run_dir=output_root / run_dir_name,
            num_frames=int(num_frames),
            warmup_frames=int(warmup_frames),
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
        run_dir=output_root / run_dir_name,
        baseline_engine_path=prepared.spec.strict_fp32_engine,
        num_frames=int(num_frames),
        warmup_frames=int(warmup_frames),
        latency_rounds=1,
        objective_config=Stage2ObjectiveConfig(
            latency_metric="forward_p50_ms",
            accuracy_reference="original_strict_fp32",
            latency_reference="original_strict_fp32",
        ),
        dataloader_num_workers=8,
    )


def validate_generation_winner(
    prepared: PreparedCNNFormalSearch,
    *,
    screening_result: Stage2Result,
    output_root: Path,
    budget_label: str,
    generation: int,
    validation_evaluator: Any,
) -> Stage2Result:
    """Re-evaluate one already-built screening engine on fixed500."""

    complete_hash = str(screening_result.complete_phenotype_hash)
    destination = output_root / (
        f"ga/budget_{budget_label}/generation_winner_validation/{complete_hash}"
    )
    result_path = destination / "generation_winner_result.json"
    if result_path.is_file():
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        metadata = dict(payload.get("metadata") or {})
        if (
            int(metadata.get("evaluation_frames", -1)) != GENERATION_WINNER_FRAMES
            or int(metadata.get("evaluation_warmup_frames", -1))
            != GENERATION_WINNER_WARMUP_FRAMES
            or str(metadata.get("evaluation_protocol", ""))
            != GENERATION_WINNER_PROTOCOL
        ):
            raise RuntimeError(
                f"cnn_generation_winner_cache_protocol_mismatch:{complete_hash}"
            )
        return Stage2Result(
            complete_hash,
            screening_result.genotype,
            str(payload["status"]),
            payload.get("mAP"),
            payload.get("p50_ms"),
            bool(payload["requested_realized_exact"]),
            int(payload["evaluated"]),
            int(payload["skipped"]),
            metadata,
        )
    destination.mkdir(parents=True, exist_ok=True)
    try:
        if not screening_result.deployable:
            raise RuntimeError("cnn_generation_winner_screening_not_deployable")
        source = Path(str(screening_result.metadata.get("artifact_dir", "")))
        if not source.is_dir():
            raise RuntimeError(f"cnn_generation_winner_source_missing:{source}")
        phenotype = canonicalize_candidate(
            screening_result.genotype, prepared.space
        )
        raw = validation_evaluator.reevaluate_existing_candidate_engine(
            phenotype,
            source_artifact_dir=source,
            output_dir=destination / "evaluation_fixed500",
            candidate_hash=complete_hash,
        )
        evaluated, skipped = CNNRealStage2Evaluator._counts(raw)
        exact = bool(
            raw.get("status") == "ok"
            and (
                prepared.spec.model_id == "pyramid"
                or (
                    raw.get("precision_acceptance", False)
                    and raw.get("merge_acceptance", False)
                    and (
                        prepared.spec.model_id != "cobevt"
                        or (
                            raw.get("transformer_attention_fp32_acceptance", False)
                            and raw.get(
                                "transformer_functional_precision_acceptance", False
                            )
                        )
                    )
                )
            )
        )
        ok = bool(
            raw.get("status") == "ok"
            and exact
            and evaluated == GENERATION_WINNER_FRAMES
            and skipped == 0
            and math.isfinite(float(raw.get("mAP")))
            and math.isfinite(float(raw.get("forward_p50_ms")))
        )
        result = Stage2Result(
            complete_hash,
            screening_result.genotype,
            "ok" if ok else str(raw.get("status", "failed")),
            float(raw["mAP"]) if ok else None,
            float(raw["forward_p50_ms"]) if ok else None,
            exact,
            evaluated,
            skipped,
            {
                "generation": int(generation),
                "artifact_dir": str(destination),
                "source_artifact_dir": str(source),
                "screening_result": stage2_payload(screening_result),
                "engine_hash": raw.get("engine_hash", raw.get("engine_sha256", "")),
                "engine_rebuilt_for_validation": False,
                "evaluation_frames": GENERATION_WINNER_FRAMES,
                "evaluation_warmup_frames": GENERATION_WINNER_WARMUP_FRAMES,
                "evaluation_protocol": GENERATION_WINNER_PROTOCOL,
                "precision_fallback": False,
                "raw": raw,
            },
        )
    except Exception as exc:
        result = Stage2Result(
            complete_hash,
            screening_result.genotype,
            "failed",
            None,
            None,
            False,
            0,
            0,
            {
                "generation": int(generation),
                "failure": f"{type(exc).__name__}:{exc}",
                "engine_rebuilt_for_validation": False,
                "evaluation_frames": GENERATION_WINNER_FRAMES,
                "evaluation_warmup_frames": GENERATION_WINNER_WARMUP_FRAMES,
                "evaluation_protocol": GENERATION_WINNER_PROTOCOL,
                "precision_fallback": False,
            },
        )
    write_json(result_path, stage2_payload(result))
    return result


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
    validation_evaluator: Any,
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
        generation_contract=f"formal_gen{generations}",
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
    greedy_validated = validate_generation_winner(
        prepared,
        screening_result=greedy,
        output_root=output_root,
        budget_label=label,
        generation=0,
        validation_evaluator=validation_evaluator,
    )
    if not greedy_validated.deployable:
        raise RuntimeError(
            f"formal_cnn_greedy_anchor_fixed500_failed:budget_{label}"
        )
    generation_winner_generations: dict[str, int] = {}
    for record in result["history"]:
        winner_hash = record.get("generation_winner_hash")
        if winner_hash:
            generation_winner_generations.setdefault(
                str(winner_hash), int(record["generation"])
            )
    generation_winner_validations = []
    for winner_hash, generation in generation_winner_generations.items():
        screening = result["evaluated"].get(winner_hash)
        if screening is None:
            raise RuntimeError(
                f"cnn_generation_winner_screening_missing:{winner_hash}"
            )
        generation_winner_validations.append(
            validate_generation_winner(
                prepared,
                screening_result=screening,
                output_root=output_root,
                budget_label=label,
                generation=generation,
                validation_evaluator=validation_evaluator,
            )
        )
    final = best_real_candidate(
        [greedy_validated, *generation_winner_validations], greedy_validated
    )
    summary = {
        "model": prepared.spec.model_id,
        "target_bops": target,
        "seed": seed,
        "generation_zero_counted": result["generation_zero_counted"],
        "completed_evolution_generations": result["completed_evolution_generations"],
        "termination_reason": result["termination_reason"],
        "greedy_anchor_screening": stage2_payload(greedy),
        "greedy_anchor": stage2_payload(greedy_validated),
        "global_anchors": [stage2_payload(row) for row in result["anchors"].unique()],
        "final_winner": stage2_payload(final),
        "ga_improved_greedy": final.complete_phenotype_hash != greedy.complete_phenotype_hash,
        "stage2_real_evaluation_count": len(result["evaluated"]) - 1,
        "stage2_top5_screening_frames": STAGE2_SCREENING_FRAMES,
        "stage2_top5_screening_warmup_frames": STAGE2_SCREENING_WARMUP_FRAMES,
        "generation_winner_validation_frames": GENERATION_WINNER_FRAMES,
        "generation_winner_validation_warmup_frames": GENERATION_WINNER_WARMUP_FRAMES,
        "generation_winner_validation_count": len(generation_winner_validations),
        "generation_winner_validations": [
            stage2_payload(row) for row in generation_winner_validations
        ],
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
    "EVALUATION_MANIFEST_FRAMES",
    "EVALUATION_MANIFEST_WARMUP_FRAMES",
    "GENERATION_WINNER_FRAMES",
    "GENERATION_WINNER_PROTOCOL",
    "GENERATION_WINNER_WARMUP_FRAMES",
    "MODEL_SPECS",
    "PreparedCNNFormalSearch",
    "baseline_genotype",
    "best_real_candidate",
    "build_initial_population",
    "create_real_evaluator",
    "greedy_anchors",
    "prepare_search",
    "run_budget",
    "STAGE2_SCREENING_FRAMES",
    "STAGE2_SCREENING_PROTOCOL",
    "STAGE2_SCREENING_WARMUP_FRAMES",
    "stage2_payload",
    "validate_generation_winner",
    "write_csv",
    "write_json",
]
