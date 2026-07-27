"""Strict GA adapter for HEAL Transformer models other than V2X-ViT.

CoBEVT uses the common Stage-1/Stage-2/V1--V3 scheduler while retaining the
unified CNN/attention/FFN decoder and the generic HEAL fixed-K deployment
pipeline.  This module intentionally does not duplicate the GA algorithm.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from scripts.audit_heal_transformer_search_models import MODEL_SPECS, _load
from scripts.run_heal_transformer_six_budget_proxy import (
    FrozenTrainPrefix,
    _bind_model_specific_train_prefix,
)
from scripts.run_v2xvit_greedy005_full import (
    _baseline_candidate,
    _build_full_space,
    _formal_space,
)
from scripts.run_v2xvit_greedy005_weight_only_abs import _type_coverage_forward
from scripts.smoke_transformer_unified_search import _multi_agent_validation_batch
from search.integration.heal_lidar_baseline_context import (
    HEAL_RUNTIME_GRAPH_POLICY,
    build_heal_lidar_baseline_context,
)
from search.model_family.calibration_manifest import load_v2xvit_train_manifest
from search.proxy.conservative_gate_activation_taylor import (
    FunctionalGateTaylorProxy,
    build_activation_units,
    collect_activation_taylor_cache_multi,
    collect_functional_gate_scores_multi,
    rerank_domains_by_gate_scores,
)
from search.proxy.joint_weight_activation_taylor import (
    taylor_units_from_transformer_precision,
)
from search.proxy.joint_weight_taylor import JointWeightTaylorProxy

from .cnn_stage12_v3 import CNNFormalModelSpec, PreparedCNNFormalSearch, write_json


COBEVT_SPEC = CNNFormalModelSpec(
    model_id="cobevt",
    family_id="heal_lidar_cobevt",
    checkpoint=Path(MODEL_SPECS["cobevt"]["checkpoint"]),
    config=Path(MODEL_SPECS["cobevt"]["config"]),
    calibration_manifest=Path(
        "/home/lixingfeng/UniAD_examine/heal_compress/outputs/"
        "heal_lidar_baseline_train200_fixedk29696_20260719_1215_v2/"
        "calibration_manifest.json"
    ),
    strict_fp32_engine=Path(
        "/home/lixingfeng/UniAD_examine/heal_compress/outputs/"
        "h800_dair_lidar_trt_fp32_all_models_smoke_20260718_2138/"
        "lidar_cobevt/strict_fp32.plan"
    ),
)


def prepare_cobevt_search(
    *,
    output_root: Path,
    physical_gpu: int,
    plugin: Path,
    tensorrt_root: Path,
    taylor_samples: int = 32,
) -> PreparedCNNFormalSearch:
    """Build one frozen CoBEVT unified search space and Taylor cache."""

    spec = COBEVT_SPEC
    context = build_heal_lidar_baseline_context(
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
        # The immutable manifest serves 300/100 screening and 500/200
        # generation-winner validation; build it for the larger tier.
        num_frames=500,
        warmup_frames=200,
        reset_after_warmup=True,
        default_precision="FP32",
        fixed_k=29696,
        max_agents=2,
        minimum_retained_ratio=0.10,
        dense_alignment=4,
        require_quant_calibration_manifest=True,
        search_space_policy=HEAL_RUNTIME_GRAPH_POLICY,
    )
    device = torch.device(context.runtime_device)
    model = context.model
    adapter = context.model_bundle.adapter
    hypes = context.model_bundle.config
    representative, _, _ = _multi_agent_validation_batch(adapter, hypes, device)
    source_path = (
        Path(__file__).resolve().parents[1]
        / "model_family/manifests/heal_lidar_v2xvit_train200_fixed_k.json"
    )
    source = load_v2xvit_train_manifest(source_path)
    bound = _bind_model_specific_train_prefix(
        adapter=adapter,
        hypes=hypes,
        source_manifest=source,
        count=int(taylor_samples),
        model_id="cobevt",
    )
    write_json(output_root / "reports/model_specific_train32_manifest.json", bound)
    batches = FrozenTrainPrefix(
        adapter=adapter,
        hypes=hypes,
        device=device,
        manifest=bound,
        count=int(taylor_samples),
    )
    identity = _build_full_space(model, adapter, hypes, representative)
    formal = _formal_space(
        model,
        adapter,
        hypes,
        representative,
        identity,
        str(bound["manifest_hash"]),
        fisher_forward_fn=lambda current_model, batch: _type_coverage_forward(
            adapter, current_model, batch
        ),
        fisher_batches=batches,
        base_quantization_groups=context.search_space.quantization_groups,
    )
    gate_scores, gate_mapping = collect_functional_gate_scores_multi(
        model,
        formal["space"].pruning_domains,
        forward_fn=lambda current_model, batch: _type_coverage_forward(
            adapter, current_model, batch
        ),
        loss_fn=adapter.compute_task_loss,
        calibration_batches=batches,
    )
    domains = rerank_domains_by_gate_scores(
        formal["space"].pruning_domains, gate_scores
    )
    space = replace(
        formal["space"],
        pruning_domains=domains,
        pruning_unit_ids=[unit for domain in domains for unit in domain.ordered_unit_ids],
    )
    transformer_units = taylor_units_from_transformer_precision(
        model,
        formal["components"].precision_units,
        active_module_paths=formal["active_paths"],
    )
    activation_units, group_to_units = build_activation_units(
        model, space, transformer_units
    )
    activation = collect_activation_taylor_cache_multi(
        model,
        activation_units,
        group_to_units,
        forward_fn=lambda current_model, batch: _type_coverage_forward(
            adapter, current_model, batch
        ),
        loss_fn=adapter.compute_task_loss,
        calibration_batches=batches,
    )
    # Stage-2 consumes exactly the same unified domains and atomic CNN closure.
    context.search_space = space
    context.unified_pruning_domains = tuple(domains)
    context.unified_atomic_units = tuple(identity["cnn_units"])
    context.unified_model_name = "lidar_cobevt"
    context.unified_qkv_paths = tuple(
        path
        for spec_row in formal["components"].attention_instances
        for path in (spec_row.q_projection_paths + spec_row.k_projection_paths)
    )
    context.unified_precision_units = tuple(formal["components"].precision_units)
    context.unified_attention_instances = tuple(
        formal["components"].attention_instances
    )
    context.unified_ffn_instances = tuple(formal["components"].ffn_instances)
    context.unified_functional_precision_paths = tuple(sorted({
        str(path)
        for unit in formal["components"].precision_units
        if bool(unit.activation_only)
        for path in unit.module_paths
    }))
    result = PreparedCNNFormalSearch(
        spec=spec,
        context=context,
        space=space,
        baseline=_baseline_candidate(space),
        bops=formal["bops"],
        size=formal["size"],
        structure=FunctionalGateTaylorProxy(gate_scores),
        weight=JointWeightTaylorProxy(
            model,
            statistics=formal["fisher"],
            unit_to_parameter_slices=formal["slices"],
            strict=True,
        ),
        activation=activation,
        gate_mapping=gate_mapping,
        calibration_sample_count=int(taylor_samples),
    )
    write_json(
        output_root / "reports/new_ga_proxy_contract.json",
        {
            "model": "cobevt",
            "stage1_proxy": "J_struct_gate + J_WQ + J_AQ",
            "cnn_domain_count": sum(
                domain.domain_type in {"cnn_channel", "grouped_conv_channel"}
                for domain in domains
            ),
            "attention_domain_count": sum(
                domain.domain_type == "attention_dh" for domain in domains
            ),
            "ffn_domain_count": sum(
                domain.domain_type == "ffn_hidden" for domain in domains
            ),
            "precision_gene_count": len(space.precision_gene_ids),
            "sample_count": int(taylor_samples),
            "search_loop_forward_calls": 0,
            "search_loop_backward_calls": 0,
            "search_loop_exports": 0,
            "search_loop_trt_builds": 0,
        },
    )
    return result


__all__ = ["COBEVT_SPEC", "prepare_cobevt_search"]
