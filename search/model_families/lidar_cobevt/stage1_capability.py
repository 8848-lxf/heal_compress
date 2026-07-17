"""CoBEVT adapter for the shared legal-width joint Stage-1 search."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from search.cache.proxy_cache import ProxyCache
from search.canonicalization import (
    SearchSpaceSpec,
    canonicalize_legal_width_candidate,
)
from search.decoding.fixed_taylor_width_decoder import (
    FixedTaylorWidthDecoder,
    build_canonical_prune_ranking,
)
from search.encoding.legal_width_genotype import LegalWidthGenotype
from search.hashing import canonical_json_hash
from search.integration.calibration_provider import collect_or_load_fisher_statistics
from search.integration.data_provider import (
    build_dataset_and_loader,
    move_batch_to_device,
)
from search.proxy.bops_proxy import BOPSProxy
from search.proxy.fisher_proxy import FisherStatistics
from search.proxy.joint_taylor import JointTaylorProxy
from search.proxy.objective import ProxyObjective, ProxyObjectiveConfig
from search.proxy.parameter_slice_resolver import ParameterSlice
from search.proxy.runtime_shape_profiler import (
    RuntimeShapeProfile,
    profile_runtime_layer_shapes,
)
from search.quantization_space.types import QuantizationSearchGroup
from search.space.legal_width_inventory import (
    LegalWidthDomain,
    LegalWidthInventory,
    PreparedLegalWidthSearchSpace,
)
from search.stage1.proxy_evaluator import Stage1ProxyEvaluator

from .model_capability import CobevtModelBundle, CobevtModelCapability
from .pruning_recipe import CobevtPruningRecipe


def _head_unit_id(head: int) -> str:
    return f"cobevt::fusion_head::{int(head):02d}"


class CobevtStructuralSizeProxy:
    """Exact structural count for the head-aligned CoBEVT closure."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.recipe = CobevtPruningRecipe()
        self.original_parameter_count = sum(
            int(parameter.numel()) for parameter in model.parameters()
        )

    def structural_parameter_counts(self, phenotype: Any) -> tuple[int, int]:
        keep_widths = dict(phenotype.metadata.get("keep_widths", {}))
        keep_heads = int(keep_widths.get("cobevt::fusion_embed_heads", 0))
        domain = self.recipe.fusion_width_domain(self.model)
        if not 1 <= keep_heads <= domain.original_heads:
            raise RuntimeError(f"cobevt_keep_head_count_invalid:{keep_heads}")
        decoded = self.recipe.decode_fusion_width(
            self.model,
            keep_width=keep_heads * domain.dim_head,
            ranked_head_ids=tuple(range(domain.original_heads)),
        )
        candidate = self.recipe._predicted_parameter_count(self.model, decoded)
        return self.original_parameter_count, int(candidate)

    def evaluate(self, phenotype: Any) -> float:
        original, candidate = self.structural_parameter_counts(phenotype)
        return float(candidate / max(original, 1))


@dataclass
class CobevtStage1Context:
    search_space: SearchSpaceSpec
    model: nn.Module
    model_bundle: CobevtModelBundle
    checkpoint_hash: str
    code_commit: str
    bops_proxy: BOPSProxy


@dataclass
class CobevtStage1Components:
    context: CobevtStage1Context
    proxy: Stage1ProxyEvaluator
    statistics: FisherStatistics
    runtime_shapes: RuntimeShapeProfile
    joint_loss_scale: float
    prepared_space: PreparedLegalWidthSearchSpace
    unit_to_parameter_slices: dict[str, list[ParameterSlice]]


def build_cobevt_head_parameter_slices(
    model: nn.Module,
) -> dict[str, list[ParameterSlice]]:
    domain = CobevtPruningRecipe().fusion_width_domain(model)
    original = int(domain.original_width)
    dim_head = int(domain.dim_head)
    result: dict[str, list[ParameterSlice]] = {}
    for head in range(int(domain.original_heads)):
        channels = tuple(range(head * dim_head, (head + 1) * dim_head))
        qkv = tuple(
            block * original + channel
            for block in range(3)
            for channel in channels
        )
        rows: list[ParameterSlice] = []
        seen: set[tuple[str, int, tuple[int, ...]]] = set()

        def add(parameter_name: str, axis: int, indices: tuple[int, ...]) -> None:
            key = (str(parameter_name), int(axis), tuple(indices))
            if key in seen:
                return
            seen.add(key)
            rows.append(
                ParameterSlice(
                    parameter_name=str(parameter_name),
                    module_path=str(parameter_name).rsplit(".", 1)[0],
                    axis=int(axis),
                    indices=tuple(indices),
                    operation="cobevt_fusion_head_closure",
                )
            )

        for name, parameter in model.named_parameters():
            shape = tuple(int(value) for value in parameter.shape)
            if name == "shrinker_m1.layers.0.double_conv.2.weight":
                add(name, 0, channels)
            elif name == "shrinker_m1.layers.0.double_conv.2.bias":
                add(name, 0, channels)
            elif name.startswith("fusion_net"):
                if name.endswith("relative_position_bias_table.weight"):
                    add(name, 1, (head,))
                elif len(shape) == 1:
                    if shape[0] == original:
                        add(name, 0, channels)
                    elif shape[0] == original * 3:
                        add(name, 0, qkv)
                elif len(shape) == 2:
                    if shape[1] == original:
                        add(name, 1, channels)
                    if shape[0] == original:
                        add(name, 0, channels)
                    elif shape[0] == original * 3:
                        add(name, 0, qkv)
            elif name in {"cls_head.weight", "reg_head.weight", "dir_head.weight"}:
                add(name, 1, channels)
        if not rows:
            raise RuntimeError(f"cobevt_head_parameter_closure_empty:{head}")
        result[_head_unit_id(head)] = rows
    return result


def _legal_head_inventory(model: nn.Module) -> LegalWidthInventory:
    domain = CobevtPruningRecipe().fusion_width_domain(model)
    unit_ids = tuple(_head_unit_id(head) for head in range(domain.original_heads))
    legal_heads = tuple(
        int(width // domain.dim_head) for width in domain.legal_keep_widths
    )
    payload = {
        "domain_id": "cobevt::fusion_embed_heads",
        "legal_keep_heads": legal_heads,
        "unit_ids": unit_ids,
        "version": "lidar-cobevt-head-aligned-width-v1",
    }
    return LegalWidthInventory(
        domains=(
            LegalWidthDomain(
                domain_id="cobevt::fusion_embed_heads",
                root_module=domain.root_module,
                root_axis="out",
                scope_id="cobevt::fusion_embed",
                domain_kind="dense",
                original_width=int(domain.original_heads),
                legal_keep_widths=legal_heads,
                minimum_width=min(legal_heads),
                alignment=1,
                group_count=1,
                protected=False,
                prunable=True,
                physical_replay_supported=True,
                unit_ids=unit_ids,
                physical_groups={0: unit_ids},
                unit_local_indices={unit_id: index for index, unit_id in enumerate(unit_ids)},
            ),
        ),
        width_space_hash=canonical_json_hash(payload),
        inventory_version="lidar-cobevt-head-aligned-width-v1",
    )


def _precision_groups(model: nn.Module) -> tuple[QuantizationSearchGroup, ...]:
    rows = []
    weighted = (
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear))
        and getattr(module, "weight", None) is not None
    )
    for ordering, (name, module) in enumerate(weighted):
        rows.append(
            QuantizationSearchGroup(
                group_id=f"cobevt_pg::{name}",
                module_paths=(name,),
                canonical_node_ids=(f"__canonical__cobevt_stage1_{ordering:03d}",),
                allowed_precisions=("FP32", "FP16"),
                protected=False,
                protection_reason="",
                ordering=ordering,
                parameter_count=sum(
                    int(parameter.numel())
                    for parameter in module.parameters(recurse=False)
                ),
                baseline_macs=0.0,
                metadata={
                    "default_precision": "FP16",
                    "int8_deployment_available": False,
                    "model_family": "lidar_cobevt",
                },
            )
        )
    if not rows:
        raise RuntimeError("cobevt_weighted_precision_groups_empty")
    return tuple(rows)


def build_cobevt_stage1_space(
    model: nn.Module,
    *,
    statistics: FisherStatistics,
    checkpoint_hash: str,
    fisher_manifest_hash: str,
) -> PreparedLegalWidthSearchSpace:
    inventory = _legal_head_inventory(model)
    slices = build_cobevt_head_parameter_slices(model)
    first = build_canonical_prune_ranking(
        model,
        statistics=statistics,
        unit_to_parameter_slices=slices,
        inventory=inventory,
        checkpoint_hash=checkpoint_hash,
        fisher_manifest_hash=fisher_manifest_hash,
        ranking_mode="prune_only_first_order",
    )
    second = build_canonical_prune_ranking(
        model,
        statistics=statistics,
        unit_to_parameter_slices=slices,
        inventory=inventory,
        checkpoint_hash=checkpoint_hash,
        fisher_manifest_hash=fisher_manifest_hash,
        ranking_mode="prune_only_second_order_fisher",
    )
    decoder = FixedTaylorWidthDecoder(
        inventory,
        second.to_decoder_rows(),
        ranking_mode="prune_only_second_order_fisher",
    )
    groups = _precision_groups(model)
    actions = {group.group_id: group.allowed_precisions for group in groups}
    space = SearchSpaceSpec(
        pruning_unit_ids=list(inventory.unit_ids),
        precision_layer_ids=[],
        quantization_groups=groups,
        default_precision="FP16",
        pruning_policy_version="lidar-cobevt-head-aligned-width-v1",
        precision_policy_version="lidar-cobevt-fp32-fp16-capability-v1",
        trace_snapshot_hash=inventory.width_space_hash,
        calibration_manifest_hash=str(fisher_manifest_hash),
        code_commit="",
        structure_gene_type="legal_keep_width",
        legal_width_inventory=inventory,
        fixed_width_decoder=decoder,
        precision_action_space=actions,
    )
    return PreparedLegalWidthSearchSpace(
        search_space=space,
        inventory=inventory,
        first_order_ranking=first,
        second_order_ranking=second,
        decoder=decoder,
    )


def build_cobevt_stage1_components(
    *,
    checkpoint: str | Path,
    model_config: str | Path,
    heal_root: str | Path,
    device: str | torch.device,
    run_dir: str | Path,
    fisher_calibration_batches: int = 1,
    code_commit: str = "",
) -> CobevtStage1Components:
    destination = Path(run_dir)
    archives = destination / "archives"
    archives.mkdir(parents=True, exist_ok=True)
    torch_device = torch.device(device)
    bundle = CobevtModelCapability(
        checkpoint, model_config, heal_root
    ).load(device=torch_device)
    statistics = collect_or_load_fisher_statistics(
        model=bundle.model,
        adapter=bundle.adapter,
        model_config_path=model_config,
        device=torch_device,
        cache_path=archives / "fisher_statistics.pt",
        num_batches=int(fisher_calibration_batches),
        checkpoint_hash=bundle.preflight.checkpoint_sha256,
        code_commit=str(code_commit),
    )
    unit_slices = build_cobevt_head_parameter_slices(bundle.model)
    prepared = build_cobevt_stage1_space(
        bundle.model,
        statistics=statistics,
        checkpoint_hash=bundle.preflight.checkpoint_sha256,
        fisher_manifest_hash=statistics.manifest_hash,
    )
    from search.space.legal_width_inventory import (
        write_legal_width_search_space_artifacts,
    )

    write_legal_width_search_space_artifacts(prepared, destination)
    _dataset, loader = build_dataset_and_loader(
        bundle.adapter,
        model_config,
        split="train",
        num_workers=0,
        visualize=False,
    )
    try:
        calibration_batch = next(iter(loader))
    except StopIteration as exc:
        raise RuntimeError("cobevt_runtime_shape_batch_missing") from exc
    calibration_batch = move_batch_to_device(calibration_batch, torch_device)
    runtime_shapes = profile_runtime_layer_shapes(
        bundle.model,
        calibration_batch,
        forward_fn=bundle.adapter.forward_for_task,
    )
    (destination / "runtime_layer_shapes.json").write_text(
        json.dumps(runtime_shapes.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    joint = JointTaylorProxy(
        bundle.model,
        statistics=statistics,
        unit_to_parameter_slices=unit_slices,
        mode="joint_taylor_second_order_fisher_diag",
    )
    bops = BOPSProxy(
        bundle.model,
        unit_to_parameter_slices=unit_slices,
        runtime_shapes=runtime_shapes.shapes,
    )
    domain = prepared.inventory.domains[0]
    aggressive = LegalWidthGenotype(
        {domain.domain_id: 0},
        {
            group_id: "FP16"
            for group_id in prepared.search_space.precision_action_space
        },
        {"seed_family": "joint_loss_scale_anchor"},
    )
    aggressive_phenotype = canonicalize_legal_width_candidate(
        aggressive, prepared.search_space
    )
    scale_result = joint.evaluate(aggressive_phenotype)
    if not scale_result.finite or scale_result.total_importance <= 0.0:
        raise RuntimeError("cobevt_joint_loss_scale_anchor_invalid")
    joint_loss_scale = float(scale_result.total_importance)
    objective = ProxyObjective(
        joint=joint,
        size=CobevtStructuralSizeProxy(bundle.model),
        bops=bops,
        config=ProxyObjectiveConfig(
            bops_threshold=None,
            proxy_mode="joint_taylor_second_order_fisher_diag",
            task_score_mapping="linear_fixed_scale",
            joint_loss_scale=joint_loss_scale,
            task_weight=0.8,
            prune_weight=0.2,
        ),
    )
    proxy = Stage1ProxyEvaluator(
        prepared.search_space,
        objective=objective,
        cache=ProxyCache(archives / "proxy_cache.jsonl"),
        proxy_backend="scalar_cpu",
        proxy_device=str(torch_device),
        proxy_batch_size=16,
    )
    context = CobevtStage1Context(
        search_space=prepared.search_space,
        model=bundle.model,
        model_bundle=bundle,
        checkpoint_hash=bundle.preflight.checkpoint_sha256,
        code_commit=str(code_commit),
        bops_proxy=bops,
    )
    manifest = {
        "checkpoint_hash": bundle.preflight.checkpoint_sha256,
        "config_hash": bundle.preflight.config_sha256,
        "fisher_manifest_hash": statistics.manifest_hash,
        "fisher_sample_count": int(statistics.manifest.get("sample_count", 0)),
        "fisher_micro_batch_size": int(
            statistics.manifest.get("micro_batch_size", 0)
        ),
        "joint_loss_scale": joint_loss_scale,
        "precision_group_count": len(prepared.search_space.precision_action_space),
        "int8_deployment_available": False,
        "runtime_weighted_call_count": len(runtime_shapes.shapes),
        "device": str(torch_device),
    }
    (destination / "cobevt_stage1_capability.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return CobevtStage1Components(
        context=context,
        proxy=proxy,
        statistics=statistics,
        runtime_shapes=runtime_shapes,
        joint_loss_scale=joint_loss_scale,
        prepared_space=prepared,
        unit_to_parameter_slices=unit_slices,
    )


__all__ = [
    "CobevtStage1Components",
    "CobevtStage1Context",
    "CobevtStructuralSizeProxy",
    "build_cobevt_head_parameter_slices",
    "build_cobevt_stage1_components",
    "build_cobevt_stage1_space",
]
