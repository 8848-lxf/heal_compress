"""Audited capabilities for HEAL LiDAR-only F-Cooper and DiscoNet baselines.

The adapters identify the real HEAL PointPillar/HeterModelBaseline topology.
Dependency-closed pruning and backbone/deblock/shrinker explicit-Q/DQ are
production-enabled after real physical replay and strongly typed H800 engine
evidence. Search readiness remains fail-closed until train200 entropy scales and
small-frame INT8 accuracy evidence exist.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

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


class _HealLidarBaselineProvider:
    """Common structural audit for the two frozen PointPillar baselines."""

    fusion_method = ""
    fusion_class = ""
    family_id = ""

    @staticmethod
    def _model_args(config: Mapping[str, Any]) -> Mapping[str, Any]:
        model = config.get("model", {})
        return model.get("args", {}) if isinstance(model, Mapping) else {}

    def matches(self, config: Mapping[str, Any]) -> bool:
        args = self._model_args(config)
        model = config.get("model", {})
        return (
            isinstance(model, Mapping)
            and str(model.get("core_method", "")).lower() == "heter_model_baseline"
            and str(args.get("fusion_method", "")).lower() == self.fusion_method
            and str(args.get("ego_modality", "m1")) == "m1"
            and isinstance(args.get("m1"), Mapping)
            and str(args["m1"].get("core_method", "")).lower() == "point_pillar"
        )

    @staticmethod
    def _weight_axis(module: nn.Module) -> int:
        return 1 if isinstance(module, (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)) else 0

    def _require_real_structure(self, model: nn.Module, config: Mapping[str, Any]) -> None:
        """Reject lookalikes: audit data must describe the loaded HEAL topology."""
        args = self._model_args(config)
        required_classes = {
            "encoder_m1.pillar_vfe": "PillarVFE",
            "encoder_m1.scatter": "PointPillarScatter",
            "backbone_m1": "BaseBEVBackbone",
            "shrinker_m1": "DownsampleConv",
            "fusion_net": self.fusion_class,
        }
        if type(model).__name__ != "HeterModelBaseline":
            raise RuntimeError(f"{self.family_id}_provider_model_structure_mismatch:model_type")
        for path, expected_class in required_classes.items():
            try:
                module = model.get_submodule(path)
            except AttributeError as exc:
                raise RuntimeError(f"{self.family_id}_provider_model_structure_mismatch:{path}") from exc
            if type(module).__name__ != expected_class:
                raise RuntimeError(f"{self.family_id}_provider_model_structure_mismatch:{path}")
        anchor_number = int(args.get("anchor_number", 0))
        bins = int(args.get("dir_args", {}).get("num_bins", 0)) if isinstance(args.get("dir_args"), Mapping) else 0
        for path, out_channels in (
            ("cls_head", anchor_number),
            ("reg_head", 7 * anchor_number),
            ("dir_head", bins * anchor_number),
        ):
            try:
                head = model.get_submodule(path)
            except AttributeError as exc:
                raise RuntimeError(f"{self.family_id}_provider_model_structure_mismatch:{path}") from exc
            if not isinstance(head, nn.Conv2d) or out_channels <= 0 or head.out_channels != out_channels:
                raise RuntimeError(f"{self.family_id}_provider_model_structure_mismatch:{path}")
        self._require_fusion_structure(model)

    def _require_fusion_structure(self, model: nn.Module) -> None:
        return None

    def _weighted_capability(self, name: str, module: nn.Module) -> WeightedOpCapability:
        lower = name.lower()
        allowed = ("FP32", "FP16", "INT8")
        boundary = "canonical_weighted_output_to_next_operator"
        production_enabled = True
        reason = ""
        if name in {"cls_head", "reg_head", "dir_head"}:
            allowed = ("FP32", "FP16")
            boundary = "fixed_detection_output_semantics"
            production_enabled = False
            reason = "detection_head_output_semantics_are_accuracy_protected"
        elif ".pillar_vfe." in lower:
            allowed = ("FP32", "FP16")
            boundary = "post_pfn_norm_relu_max_boundary"
            production_enabled = False
            reason = "pfn_feature_and_voxel_max_contract_is_accuracy_protected"
        elif ".fusion_net." in lower or lower.startswith("fusion_net."):
            allowed = ("FP32", "FP16")
            boundary = self._fusion_output_boundary(name, module)
            production_enabled = False
            reason = "fusion_weighted_compute_is_protected_inside_the_fp16_fusion_island"
        return WeightedOpCapability(
            canonical_id=f"module::{name}",
            module_path=name,
            op_type=type(module).__name__,
            source_kind="module",
            weight_shape=tuple(int(value) for value in module.weight.shape),
            allowed_precisions=allowed,
            potential_precisions=("FP32", "FP16", "INT8"),
            default_precision="FP16",
            weight_granularity="per_output_channel",
            weight_axis=self._weight_axis(module),
            input_scale_owner="canonical_weighted_input_tensor",
            output_boundary=boundary,
            production_enabled=production_enabled,
            gate_reason=reason,
            metadata={
                "groups": int(getattr(module, "groups", 1)),
                "explicit_qdq_policy": "per_channel_weight_next_weighted_input_requantization",
                "strongly_typed_trt_smoke": "single_int8_realized_on_h800",
            },
        )

    def _fusion_output_boundary(self, name: str, module: nn.Module) -> str:
        return "fp16_fusion_boundary"

    def _pruning_domains(
        self,
        model: nn.Module,
        *,
        require_original_widths: bool,
    ) -> tuple[PruningDomainCapability, ...]:
        from ..pruning_space.local_domains import legal_dense_widths
        from .heal_lidar_pruning import validate_heal_lidar_baseline_pruning_topology

        topology = validate_heal_lidar_baseline_pruning_topology(
            model,
            self.family_id,
            require_original_widths=require_original_widths,
        )
        rows: list[PruningDomainCapability] = []
        for domain in topology.domain_specs:
            legal_widths = (
                (domain.original_width,)
                if domain.protected
                else legal_dense_widths(
                    original_width=domain.original_width,
                    minimum_retained_ratio=0.10,
                    alignment=4,
                )
            )
            rows.append(PruningDomainCapability(
                domain_id=domain.domain_id,
                domain_kind=domain.domain_kind,
                member_modules=tuple(dict.fromkeys(path for path, _axis, _kind in domain.members)),
                original_width=domain.original_width,
                legal_widths=legal_widths,
                ranking_unit=(
                    "protected_fixed_width"
                    if domain.protected
                    else "one_dependency_closed_channel_with_fixed_fisher_taylor_ranking"
                ),
                production_enabled=not domain.protected,
                gate_reason=domain.protection_reason,
                constraints={
                    "minimum_retained_ratio": 0.10,
                    "dense_alignment": 4,
                    "physical_materializer": "heal_lidar_baseline_plan_first_v1",
                    "fixed_deblock_outputs": True,
                    "fixed_concat_width": topology.concat_width,
                    "closure_members": [
                        {
                            "module_path": path,
                            "axis": axis,
                            "dependency_type": dependency_type,
                        }
                        for path, axis, dependency_type in domain.members
                    ],
                },
            ))
        modules = dict(model.named_modules())
        for module_path, output_width in sorted(topology.fixed_output_contracts.items()):
            if module_path == "fusion_net.pixel_weight_layer.conv1_4":
                kind = "single_channel_pixel_weight_logit_output"
            elif module_path in {"cls_head", "reg_head", "dir_head"}:
                kind = "fixed_detection_semantic_output"
            else:
                kind = "fixed_deblock_concat_output"
            rows.append(PruningDomainCapability(
                domain_id=f"protected_output::{module_path}",
                domain_kind=kind,
                member_modules=(module_path,),
                original_width=int(output_width),
                legal_widths=(int(output_width),),
                ranking_unit="protected_fixed_output_contract",
                production_enabled=False,
                gate_reason="fixed_output_contract",
                constraints={
                    "physical_module_type": type(modules[module_path]).__name__,
                    "input_dependency_pruning_allowed": True,
                    "output_pruning_allowed": False,
                },
            ))
        return tuple(sorted(rows, key=lambda row: row.domain_id))

    def _merge_boundaries(self) -> tuple[MergeBoundaryCapability, ...]:
        raise NotImplementedError

    def _deployment_operators(self) -> tuple[DeploymentOperatorCapability, ...]:
        return (
            DeploymentOperatorCapability(
                capability_id="pointpillar_scatter",
                op_kinds=("ScatterND", "index_put", "PointPillarScatter"),
                module_paths=("encoder_m1.scatter",),
                deployment_mode="plugin_required_for_fixed_k_engine",
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
                gate_reason="",
            ),
            DeploymentOperatorCapability(
                capability_id="agent_feature_fusion",
                op_kinds=self._fusion_operator_kinds(),
                module_paths=("fusion_net",),
                deployment_mode="family_specific_fp16_fusion_island",
                precision_policy="FP16_boundary",
                production_enabled=True,
                gate_reason="",
            ),
        )

    def _fusion_operator_kinds(self) -> tuple[str, ...]:
        return ("Max",)

    def _fusion_pruning_strategy(self) -> str:
        return "shared_feature_width_closed_through_fusion_without_weighted_fusion_gene"

    def audit(
        self,
        model: nn.Module,
        config: Mapping[str, Any],
        *,
        require_original_widths: bool = True,
    ) -> ModelFamilyAudit:
        if not self.matches(config):
            raise RuntimeError(f"{self.family_id}_provider_config_mismatch")
        self._require_real_structure(model, config)
        weighted = tuple(sorted(
            (self._weighted_capability(name, module) for name, module in model.named_modules()
             if name and isinstance(module, _WEIGHTED_TYPES)),
            key=lambda row: row.canonical_id,
        ))
        args = self._model_args(config)
        return ModelFamilyAudit(
            schema_version="heal-model-family-audit-v2",
            family_id=self.family_id,
            model_type=type(model).__name__,
            parameter_count=sum(int(parameter.numel()) for parameter in model.parameters()),
            weighted_ops=weighted,
            pruning_domains=self._pruning_domains(
                model,
                require_original_widths=require_original_widths,
            ),
            merge_boundaries=self._merge_boundaries(),
            deployment_operators=self._deployment_operators(),
            plugin_requirements=(PluginRequirement(
                plugin_key="pointpillar_scatter_trt",
                op_types=("PointPillarScatterTRT",),
                required=True,
                compatibility_status="verified_fixedk29696_strict_fp32_and_strongly_typed_qdq_h800",
                reusable_implementation="quantization/plugins/pointpillar_scatter_trt",
                compatibility_checks=(
                    "fixed_k_29696", "max_agents_2", "grid_size_and_feature_width",
                    "coordinate_layout_and_batch_semantics", "strongly_typed_fp16_plugin_boundary",
                ),
            ),),
            input_contract={
                "fixed_k": 29696,
                "fixed_k_policy": "max_train200_and_full_validation_then_align_256",
                "max_agents": 2,
                "ego_modality": "m1",
                "fusion_method": self.fusion_method,
                "configured_max_voxel_test": int(config.get("heter", {}).get("modality_setting", {}).get("m1", {}).get("preprocess", {}).get("args", {}).get("max_voxel_test", 0)),
            },
            blockers=(
                "baseline_train200_entropy_calibration_and_int8_accuracy_evidence_not_available",
            ),
            metadata={
                "config_fusion_method": str(args.get("fusion_method", "")),
                "canonical_weighted_op_count": len(weighted),
                "pruning_strategy": "production_plan_first_dependency_closed_domains_v1",
                "pruning_materializer": "heal_lidar_baseline_plan_first_v1",
                "pruning_real_checkpoint_smoke": "no_prune_mild_and_aggressive_forward_strict_replay_passed",
                "pfn_pruning_strategy": "protected_pfn_linear_and_voxel_max_coupling_until_materializer_is_verified",
                "fusion_pruning_strategy": self._fusion_pruning_strategy(),
                "quantization_strategy": "backbone_deblock_shrinker_explicit_qdq_with_fp16_fusion_island_v1",
                "quantization_real_engine_smoke": "single_int8_requested_and_realized_on_strongly_typed_h800",
                "real_evaluator": "heal-lidar-baseline-real-evaluator-v1",
                "physical_width_audit": not bool(require_original_widths),
            },
        )


class HealLidarFCooperProvider(_HealLidarBaselineProvider):
    family_id = "heal_lidar_fcooper"
    fusion_method = "max"
    fusion_class = "MaxFusion"

    def _merge_boundaries(self) -> tuple[MergeBoundaryCapability, ...]:
        return (MergeBoundaryCapability(
            boundary_id="fcooper_max_agent_merge",
            merge_kind="agentwise_max",
            member_modules=("fusion_net",),
            policy="FP16_merge",
            scale_policy="warp_outputs_dequantized_to_fp16_before_agentwise_max",
            output_requantization="optional_post_max_qdq_owned_by_detection_head_input",
            production_enabled=True,
            gate_reason="",
        ),)


class HealLidarAttFusionProvider(_HealLidarBaselineProvider):
    """CNN-width adapter for projection-free HEAL AttFusion."""

    family_id = "heal_lidar_attfusion"
    fusion_method = "att"
    fusion_class = "AttFusion"

    def _fusion_operator_kinds(self) -> tuple[str, ...]:
        return ("MatMul", "Softmax", "MatMul")

    def _fusion_pruning_strategy(self) -> str:
        return "shared_feature_width_closed_through_projection_free_agent_attention"

    def _merge_boundaries(self) -> tuple[MergeBoundaryCapability, ...]:
        return (MergeBoundaryCapability(
            boundary_id="attfusion_projection_free_agent_attention",
            merge_kind="activation_only_scaled_dot_product_attention",
            member_modules=("fusion_net",),
            policy="FP16_functional_island",
            scale_policy="warped_agent_features_dequantized_to_fp16_before_qk_softmax_av",
            output_requantization="optional_post_attention_qdq_owned_by_detection_head_input",
            production_enabled=True,
            gate_reason="no_trainable_qkvo_projection_and_no_attention_dh_gene",
        ),)


class HealLidarCoBEVTProvider(_HealLidarBaselineProvider):
    """HEAL CoBEVT provider for the unified CNN/Transformer closure."""

    family_id = "heal_lidar_cobevt"
    fusion_method = "cobevt"
    fusion_class = "CoBEVT"

    def _fusion_operator_kinds(self) -> tuple[str, ...]:
        return (
            "MatMul", "Softmax", "LayerNormalization", "Add", "Reshape",
            "Transpose",
        )

    def _fusion_pruning_strategy(self) -> str:
        return "unified_tracer_cnn_attention_dh_ffn_domains"

    def _merge_boundaries(self) -> tuple[MergeBoundaryCapability, ...]:
        return (MergeBoundaryCapability(
            boundary_id="cobevt_transformer_residual_window_merge",
            merge_kind="transformer_residual_window_merge",
            member_modules=("fusion_net",),
            policy="explicit_qk_softmax_av_fp32_with_weighted_qdq",
            scale_policy="canonical_transformer_precision_contract",
            output_requantization="owned_by_downstream_weighted_input_qdq",
            production_enabled=True,
            gate_reason="",
        ),)


class HealLidarDiscoNetProvider(_HealLidarBaselineProvider):
    family_id = "heal_lidar_disco"
    fusion_method = "disconet"
    fusion_class = "DiscoFusion"

    def _require_fusion_structure(self, model: nn.Module) -> None:
        required = (("fusion_net.pixel_weight_layer", "PixelWeightLayer"), ("fusion_net.pixel_weight_layer.conv1_4", "Conv2d"))
        for path, expected_class in required:
            try:
                module = model.get_submodule(path)
            except AttributeError as exc:
                raise RuntimeError(f"{self.family_id}_provider_model_structure_mismatch:{path}") from exc
            if type(module).__name__ != expected_class:
                raise RuntimeError(f"{self.family_id}_provider_model_structure_mismatch:{path}")
        last = model.get_submodule("fusion_net.pixel_weight_layer.conv1_4")
        if not isinstance(last, nn.Conv2d) or last.out_channels != 1:
            raise RuntimeError(f"{self.family_id}_provider_model_structure_mismatch:pixel_weight_logit_channels")

    def _fusion_output_boundary(self, name: str, module: nn.Module) -> str:
        if name == "fusion_net.pixel_weight_layer.conv1_4":
            return "single_channel_pixel_weight_logits_before_agent_softmax_fp16"
        return "pixel_weight_hidden_fp16_before_single_channel_logit"

    def _fusion_operator_kinds(self) -> tuple[str, ...]:
        return ("Concat", "Conv", "Softmax", "Mul", "ReduceSum")

    def _fusion_pruning_strategy(self) -> str:
        return "prune_128_and_32_hidden_domains_protect_8_channel_prelogit_and_scalar_logit"

    def _merge_boundaries(self) -> tuple[MergeBoundaryCapability, ...]:
        return (
            MergeBoundaryCapability(
                boundary_id="disconet_neighbor_ego_concat",
                merge_kind="channel_concat",
                member_modules=("fusion_net.pixel_weight_layer",),
                policy="FP16_merge",
                scale_policy="warped_neighbor_and_ego_features_dequantized_to_fp16_before_concat",
                output_requantization="none_logits_stay_fp16",
                production_enabled=True,
                gate_reason="",
            ),
            MergeBoundaryCapability(
                boundary_id="disconet_pixel_weight_softmax_sum",
                merge_kind="agent_softmax_weighted_sum",
                member_modules=("fusion_net.pixel_weight_layer.conv1_4",),
                policy="FP16_merge",
                scale_policy="single_channel_logits_softmax_over_agents_then_fp16_weighted_sum",
                output_requantization="optional_post_sum_qdq_owned_by_detection_head_input",
                production_enabled=True,
                gate_reason="",
            ),
        )
