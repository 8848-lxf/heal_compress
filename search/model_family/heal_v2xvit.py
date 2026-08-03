"""Audited capabilities for HEAL LiDAR-only V2X-ViT models.

The production exporter remains model-family scoped and does not patch HEAL or
reuse the lidar_pyramid exporter.  The capabilities below distinguish the
fixed-K V2X-ViT path validated on H800 from intentionally protected Transformer
INT8 and attention-pruning features.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn

from .contracts import (
    DeploymentOperatorCapability,
    MergeBoundaryCapability,
    ModelFamilyAudit,
    PluginRequirement,
    PruningDomainCapability,
    WeightedOpCapability,
)


_WEIGHTED_TYPES = (
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.ConvTranspose1d,
    nn.ConvTranspose2d,
    nn.ConvTranspose3d,
    nn.Linear,
)


def _model_args(config: Mapping[str, Any]) -> Mapping[str, Any]:
    model = config.get("model", {})
    return model.get("args", {}) if isinstance(model, Mapping) else {}


def _weight_axis(module: nn.Module) -> int:
    if isinstance(module, (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
        return 1
    return 0


def _legal_aligned_widths(width: int, *, alignment: int = 4, floor: int = 64) -> tuple[int, ...]:
    lower = min(int(width), int(floor))
    values = [value for value in range(lower, int(width) + 1) if value % alignment == 0]
    if int(width) not in values:
        values.append(int(width))
    return tuple(sorted(set(values)))


def _head_counts(heads: int) -> tuple[int, ...]:
    return tuple(value for value in range(1, int(heads) + 1) if int(heads) % value == 0)


def _module_quantization_capability(name: str, module: nn.Module) -> WeightedOpCapability:
    lower = name.lower()
    op_type = type(module).__name__
    potential = ("FP32", "FP16", "INT8")
    allowed = potential
    enabled = True
    reason = ""
    input_owner = "canonical_weighted_input_tensor"
    output_boundary = "resolve_post_activation_or_post_merge_boundary_from_canonical_onnx"

    if any(token in lower for token in ("cls_head", "reg_head", "dir_head")):
        allowed = ("FP32", "FP16")
        enabled = False
        reason = "detection_head_int8_requires_model_family_accuracy_gate"
        output_boundary = "fixed_detection_output_contract"
    elif "pillar_vfe" in lower:
        allowed = ("FP32", "FP16")
        enabled = False
        reason = "pillar_vfe_matmul_requires_fixed_k_export_mapping"
        output_boundary = "post_pfn_norm_relu_max_boundary"
    elif "fusion_net" in lower:
        allowed = ("FP32", "FP16")
        enabled = False
        reason = "transformer_linear_int8_requires_qdq_parser_and_tensor_parity_gate"
        if any(token in lower for token in ("to_qkv", "q_linears", "k_linears", "v_linears")):
            output_boundary = "dequantize_to_fp16_before_chunk_einsum_softmax"
        elif any(token in lower for token in ("a_linears", "to_out")):
            output_boundary = "dequantize_to_fp16_before_transformer_residual_add"
        elif ".fn.net.0" in lower:
            output_boundary = "post_gelu_tensor_owned_by_following_linear_input"
        elif ".fn.net.3" in lower:
            output_boundary = "dequantize_to_fp16_before_feedforward_residual_add"
        elif ".split_attn.fc1" in lower:
            output_boundary = "post_layernorm_relu_tensor"
        elif ".split_attn.fc2" in lower:
            output_boundary = "dequantize_to_fp16_before_radix_softmax_weighted_sum"
        elif lower.endswith("prior_feed"):
            output_boundary = "dequantize_to_fp16_before_transformer_encoder_input"

    return WeightedOpCapability(
        canonical_id=f"module::{name}",
        module_path=name,
        op_type=op_type,
        source_kind="module",
        weight_shape=tuple(int(value) for value in module.weight.shape),
        allowed_precisions=allowed,
        potential_precisions=potential,
        default_precision="FP16",
        weight_granularity="per_output_channel",
        weight_axis=_weight_axis(module),
        input_scale_owner=input_owner,
        output_boundary=output_boundary,
        production_enabled=enabled,
        gate_reason=reason,
        metadata={
            "groups": int(getattr(module, "groups", 1)),
            "compute_output_precision_split_required": "fusion_net" in lower,
        },
    )


def _functional_weight_capabilities(model: nn.Module) -> list[WeightedOpCapability]:
    rows: list[WeightedOpCapability] = []
    for name, parameter in model.named_parameters():
        if not (name.endswith("relation_att") or name.endswith("relation_msg")):
            continue
        parent = name.rsplit(".", 1)[0]
        rows.append(
            WeightedOpCapability(
                canonical_id=f"functional::{name}",
                module_path=parent,
                op_type="FunctionalEinsumWeight",
                source_kind="functional_parameter",
                weight_shape=tuple(int(value) for value in parameter.shape),
                allowed_precisions=("FP32", "FP16"),
                potential_precisions=("FP32", "FP16", "INT8"),
                default_precision="FP16",
                weight_granularity="not_yet_legalized",
                weight_axis=None,
                input_scale_owner="einsum_operand_specific",
                output_boundary="fp16_einsum_attention_island",
                production_enabled=False,
                gate_reason="functional_einsum_weight_qdq_export_not_implemented",
                metadata={"parameter_name": name},
            )
        )
    return rows


def _pruning_domains(model: nn.Module) -> list[PruningDomainCapability]:
    rows: list[PruningDomainCapability] = []
    for name, module in model.named_modules():
        class_name = type(module).__name__
        if class_name == "FeedForward":
            first = getattr(getattr(module, "net", None), "__getitem__", lambda _index: None)(0)
            second = getattr(getattr(module, "net", None), "__getitem__", lambda _index: None)(3)
            if isinstance(first, nn.Linear) and isinstance(second, nn.Linear):
                width = int(first.out_features)
                rows.append(
                    PruningDomainCapability(
                        domain_id=f"ffn_hidden::{name}",
                        domain_kind="transformer_ffn_hidden_width",
                        member_modules=(f"{name}.net.0", f"{name}.net.3"),
                        original_width=width,
                        legal_widths=_legal_aligned_widths(width),
                        ranking_unit="fixed_task_taylor_ordered_hidden_units",
                        production_enabled=True,
                        gate_reason="",
                        constraints={
                            "prune_first_linear_output": True,
                            "prune_second_linear_input": True,
                            "embedding_width_unchanged": True,
                            "physical_materializer": "v2xvit_ffn_linear_pair_v1",
                        },
                    )
                )
        elif class_name == "SplitAttn":
            first = getattr(module, "fc1", None)
            second = getattr(module, "fc2", None)
            if isinstance(first, nn.Linear) and isinstance(second, nn.Linear):
                width = int(first.out_features)
                rows.append(
                    PruningDomainCapability(
                        domain_id=f"split_attention_hidden::{name}",
                        domain_kind="split_attention_hidden_width",
                        member_modules=(f"{name}.fc1", f"{name}.bn1", f"{name}.fc2"),
                        original_width=width,
                        legal_widths=_legal_aligned_widths(width),
                        ranking_unit="fixed_task_taylor_ordered_hidden_units",
                        production_enabled=False,
                        gate_reason="layernorm_and_radix_projection_materializer_required",
                        constraints={"fc2_output_three_way_radix_width_fixed": True},
                    )
                )
        elif class_name == "BaseWindowAttention":
            heads = int(getattr(module, "heads", 0))
            if heads > 0:
                rows.append(
                    PruningDomainCapability(
                        domain_id=f"window_attention_heads::{name}",
                        domain_kind="whole_attention_head_bundle",
                        member_modules=(f"{name}.to_qkv", f"{name}.to_out.0"),
                        original_width=heads,
                        legal_widths=_head_counts(heads),
                        ranking_unit="whole_head_joint_taylor",
                        production_enabled=False,
                        gate_reason="custom_qkv_chunk_head_materializer_required",
                        constraints={
                            "dim_head": int(getattr(module, "to_qkv").out_features // (3 * heads)),
                            "prune_q_k_v_same_head_indices": True,
                            "update_heads_attribute": True,
                        },
                    )
                )
        elif class_name == "HGTCavAttention":
            heads = int(getattr(module, "heads", 0))
            if heads > 0:
                members = tuple(
                    f"{name}.{family}.{index}"
                    for family in ("q_linears", "k_linears", "v_linears", "a_linears")
                    for index in range(len(getattr(module, family)))
                )
                rows.append(
                    PruningDomainCapability(
                        domain_id=f"hgt_attention_heads::{name}",
                        domain_kind="whole_heterogeneous_attention_head_bundle",
                        member_modules=members,
                        original_width=heads,
                        legal_widths=_head_counts(heads),
                        ranking_unit="whole_head_joint_taylor_across_agent_types",
                        production_enabled=False,
                        gate_reason="custom_hgt_relation_tensor_materializer_required",
                        constraints={
                            "couple_all_q_k_v_agent_types": True,
                            "prune_relation_att_both_head_axes": True,
                            "prune_relation_msg_both_head_axes": True,
                            "prune_output_projection_inputs": True,
                            "update_heads_attribute": True,
                        },
                    )
                )
    return sorted(rows, key=lambda row: row.domain_id)


def _merge_boundaries(model: nn.Module) -> list[MergeBoundaryCapability]:
    rows: list[MergeBoundaryCapability] = []
    for name, module in model.named_modules():
        class_name = type(module).__name__
        if class_name == "V2XFusionBlock":
            rows.append(
                MergeBoundaryCapability(
                    boundary_id=f"attention_residual::{name}",
                    merge_kind="transformer_residual_add",
                    member_modules=tuple(child for child, _ in module.named_modules() if child),
                    policy="FP16_merge",
                    scale_policy="branches_dequantized_to_fp16_before_add",
                    output_requantization="optional_post_add_qdq_owned_by_next_weighted_input",
                    production_enabled=True,
                    gate_reason="",
                )
            )
        elif class_name == "FeedForward":
            rows.append(
                MergeBoundaryCapability(
                    boundary_id=f"feedforward_residual::{name}",
                    merge_kind="transformer_residual_add",
                    member_modules=(f"{name}.net.0", f"{name}.net.3"),
                    policy="FP16_merge",
                    scale_policy="feedforward_and_identity_dequantized_to_fp16_before_add",
                    output_requantization="optional_post_add_qdq_owned_by_next_layernorm_input",
                    production_enabled=True,
                    gate_reason="",
                )
            )
        elif class_name == "SplitAttn":
            rows.append(
                MergeBoundaryCapability(
                    boundary_id=f"split_attention_sum::{name}",
                    merge_kind="three_branch_weighted_sum",
                    member_modules=(f"{name}.fc1", f"{name}.fc2"),
                    policy="FP16_merge",
                    scale_policy="three_window_branches_share_fp16_weighted_sum_boundary",
                    output_requantization="optional_post_sum_qdq_owned_by_transformer_residual",
                    production_enabled=True,
                    gate_reason="",
                )
            )
    return sorted(rows, key=lambda row: row.boundary_id)


def _deployment_operators() -> tuple[DeploymentOperatorCapability, ...]:
    return (
        DeploymentOperatorCapability(
            capability_id="pointpillar_scatter",
            op_kinds=("ScatterND", "index_put", "PointPillarScatterTRT"),
            module_paths=("encoder_m1.scatter",),
            deployment_mode="plugin_required_for_fixed_k_single_engine",
            precision_policy="FP16_boundary",
            plugin_key="pointpillar_scatter_trt",
            production_enabled=True,
            gate_reason="",
        ),
        DeploymentOperatorCapability(
            capability_id="agent_affine_warp",
            op_kinds=("GridSample", "AffineGrid", "warp_affine_simple"),
            module_paths=("fusion_net",),
            deployment_mode="native_parser_probe_then_plugin_if_required",
            precision_policy="FP16_island",
            production_enabled=True,
            gate_reason="validated_for_frozen_max_agents_2_type0_export_contract",
        ),
        DeploymentOperatorCapability(
            capability_id="transformer_attention_einsum",
            op_kinds=("Einsum", "MatMul", "Softmax", "MaskedFill"),
            module_paths=("fusion_net.fusion_net.encoder",),
            deployment_mode="native_tensorrt_fp16_island",
            precision_policy="weighted_linear_may_be_int8_but_einsum_softmax_stays_fp16",
            production_enabled=True,
            gate_reason="validated_as_strongly_typed_fp16_island",
        ),
        DeploymentOperatorCapability(
            capability_id="heterogeneous_type_dispatch",
            op_kinds=("ModuleListTensorIndex", "PythonLoop", "Gather"),
            module_paths=("fusion_net.fusion_net.encoder",),
            deployment_mode="export_rewrite_or_proven_type_specialization",
            precision_policy="FP16_island",
            production_enabled=False,
            gate_reason="onnx_trace_must_cover_all_realized_agent_types",
        ),
        DeploymentOperatorCapability(
            capability_id="multi_window_rearrange",
            op_kinds=("Reshape", "Transpose", "Slice", "Concat"),
            module_paths=("fusion_net.fusion_net.encoder",),
            deployment_mode="native_shape_contract",
            precision_policy="FP16_merge_with_optional_weighted_int8_compute",
            production_enabled=True,
            gate_reason="validated_for_fixed_bev_and_max_agents_2_contract",
        ),
    )


class HealLidarV2XViTProvider:
    family_id = "heal_lidar_v2xvit"

    def matches(self, config: Mapping[str, Any]) -> bool:
        args = _model_args(config)
        return (
            str(config.get("model", {}).get("core_method", "")).lower() == "heter_model_baseline"
            and str(args.get("fusion_method", "")).lower() == "v2xvit"
        )

    def audit(self, model: nn.Module, config: Mapping[str, Any]) -> ModelFamilyAudit:
        if not self.matches(config):
            raise RuntimeError("heal_lidar_v2xvit_provider_config_mismatch")
        module_rows = [
            _module_quantization_capability(name, module)
            for name, module in model.named_modules()
            if name and isinstance(module, _WEIGHTED_TYPES)
        ]
        functional_rows = _functional_weight_capabilities(model)
        weighted = tuple(sorted([*module_rows, *functional_rows], key=lambda row: row.canonical_id))
        args = _model_args(config)
        modality = args.get(str(args.get("ego_modality", "m1")), {})
        preprocess = config.get("heter", {}).get("modality_setting", {}).get("m1", {}).get("preprocess", {})
        voxel_size = list(preprocess.get("args", {}).get("voxel_size", []))
        lidar_range = list(args.get("lidar_range", config.get("cav_lidar_range", [])))
        grid_size = []
        if len(voxel_size) == 3 and len(lidar_range) == 6:
            grid_size = [int(round((lidar_range[index + 3] - lidar_range[index]) / voxel_size[index])) for index in range(3)]
        module_type_counts = Counter(type(module).__name__ for module in model.modules())
        relative_position_parameters = sorted(
            name for name, _parameter in model.named_parameters() if name.endswith("pos_embedding")
        )
        return ModelFamilyAudit(
            schema_version="heal-model-family-audit-v2",
            family_id=self.family_id,
            model_type=type(model).__name__,
            parameter_count=sum(int(parameter.numel()) for parameter in model.parameters()),
            weighted_ops=weighted,
            pruning_domains=tuple(_pruning_domains(model)),
            merge_boundaries=tuple(_merge_boundaries(model)),
            deployment_operators=_deployment_operators(),
            plugin_requirements=(
                PluginRequirement(
                    plugin_key="pointpillar_scatter_trt",
                    op_types=("PointPillarScatterTRT",),
                    required=True,
                    compatibility_status="validated_v2xvit_fixedk27904_h800_trt10_9",
                    reusable_implementation="quantization/plugins/pointpillar_scatter_trt",
                    compatibility_checks=(
                        "fixed_k_from_frozen_calibration_manifest",
                        "grid_size_and_feature_width",
                        "coordinate_layout_and_batch_semantics",
                        "strongly_typed_fp16_plugin_boundary",
                        "plugin_compute_capability_and_tensorRT_hash",
                    ),
                ),
            ),
            input_contract={
                "max_cav": int(config.get("train_params", {}).get("max_cav", 0)),
                "configured_max_voxel_test": int(preprocess.get("args", {}).get("max_voxel_test", 0)),
                "fixed_k_policy": "derive_from_frozen_calibration_manifest_then_freeze",
                "grid_size_xyz": grid_size,
                "voxel_size": voxel_size,
                "agent_type_policy": "prove_realized_types_before_export_specialization",
                "window_divisibility": [4, 8, 16],
            },
            blockers=(
                "representative_forward_does_not_cover_all_weighted_agent_type_branches",
                "hgt_functional_einsum_weights_not_yet_quantizable",
                "attention_pruning_materializers_not_yet_implemented",
                "transformer_parameterized_int8_remains_accuracy_protected",
                "heterogeneous_type_dispatch_only_validated_for_type0_specialization",
            ),
            metadata={
                "module_weighted_op_count": len(module_rows),
                "functional_weighted_op_count": len(functional_rows),
                "canonical_weighted_op_count": len(weighted),
                "module_type_counts": dict(sorted(module_type_counts.items())),
                "relative_position_parameter_count": len(relative_position_parameters),
                "relative_position_parameters": relative_position_parameters,
                "torch_pruning_strategy": {
                    "safe_initial_domains": ["transformer_ffn_hidden_width"],
                    "attention_head_strategy": "whole_head_bundles_with_custom_HGT_and_QKV_materializers",
                    "embedding_width_strategy": "fixed_256_until_full_residual_layernorm_contract_exists",
                    "tp_reference": "Torch-Pruning_1.6_num_heads_prune_num_heads_unwrapped_parameters",
                    "direct_nn_MultiheadAttention_pruner_reusable": False,
                    "reason": "HEAL V2X-ViT uses custom Linear/Einsum HGT and window attention modules",
                },
                "config_fusion_method": str(args.get("fusion_method", "")),
                "modality_core_method": str(modality.get("core_method", "")),
            },
        )
