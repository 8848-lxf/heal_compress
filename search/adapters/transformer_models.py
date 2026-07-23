"""Real-structure adapters for HEAL Transformer and attention fusion models."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Mapping, Sequence

import torch.nn as nn

from ..canonicalization import SearchSpaceSpec
from ..pruning_space.local_domains import LocalPruningDomain
from ..pruning_space.transformer_domains import (
    AttentionInstanceSpec,
    FFNInstanceSpec,
    build_transformer_pruning_domains,
)
from ..quantization_space.transformer_precision import (
    TransformerPrecisionUnit,
    build_transformer_precision_units,
    build_transformer_quantization_groups,
)
from ..quantization_space.types import QuantizationSearchGroup


def _hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode(
            "utf-8"
        )
    ).hexdigest()


def _model_args(config: Mapping[str, Any]) -> Mapping[str, Any]:
    model = config.get("model", {})
    return model.get("args", {}) if isinstance(model, Mapping) else {}


@dataclass(frozen=True)
class TransformerModelAdapter:
    model_key: str
    canonical_name: str
    core_methods: tuple[str, ...]
    fusion_method: str
    requires_trainable_attention: bool
    requires_ffn: bool
    expected_attention_families: tuple[str, ...] = ()

    def matches(self, config: Mapping[str, Any]) -> bool:
        model = config.get("model", {})
        core = str(model.get("core_method", "")).lower()
        fusion = str(_model_args(config).get("fusion_method", "")).lower()
        return core in self.core_methods and fusion == self.fusion_method

    def validate(
        self,
        attention: Sequence[AttentionInstanceSpec],
        ffn: Sequence[FFNInstanceSpec],
    ) -> None:
        if self.requires_trainable_attention and not attention:
            raise RuntimeError(f"{self.model_key}_trainable_attention_not_discovered")
        if self.requires_ffn and not ffn:
            raise RuntimeError(f"{self.model_key}_ffn_not_discovered")
        families = {row.family for row in attention}
        missing = sorted(set(self.expected_attention_families) - families)
        if missing:
            raise RuntimeError(f"{self.model_key}_attention_families_missing:{missing}")
        if not self.requires_trainable_attention and attention:
            raise RuntimeError(
                f"{self.model_key}_unexpected_trainable_qkvo_attention:"
                f"{[row.module_path for row in attention]}"
            )


MODEL_ADAPTERS = (
    TransformerModelAdapter(
        model_key="v2xvit",
        canonical_name="HeterBaseline_DAIR_lidar_v2xvit",
        core_methods=("heter_model_baseline",),
        fusion_method="v2xvit",
        requires_trainable_attention=True,
        requires_ffn=True,
        expected_attention_families=(
            "v2xvit_agent_relation",
            "v2xvit_spatial_window_w4",
            "v2xvit_spatial_window_w8",
            "v2xvit_spatial_window_w16",
        ),
    ),
    TransformerModelAdapter(
        model_key="cobevt",
        canonical_name="HeterBaseline_DAIR_lidar_cobevt",
        core_methods=("heter_model_baseline",),
        fusion_method="cobevt",
        requires_trainable_attention=True,
        requires_ffn=True,
        expected_attention_families=("cobevt_grid", "cobevt_window"),
    ),
    TransformerModelAdapter(
        model_key="attfusion",
        canonical_name="HeterBaseline_DAIR_lidar_attfuse",
        core_methods=("heter_model_baseline",),
        fusion_method="att",
        requires_trainable_attention=False,
        requires_ffn=False,
    ),
    TransformerModelAdapter(
        model_key="coalign",
        canonical_name="HeterBaseline_DAIR_lidar_coalign",
        core_methods=("heter_model_baseline_ms",),
        fusion_method="att",
        requires_trainable_attention=False,
        requires_ffn=False,
    ),
)


def detect_transformer_model_adapter(
    config: Mapping[str, Any],
) -> TransformerModelAdapter:
    matches = [adapter for adapter in MODEL_ADAPTERS if adapter.matches(config)]
    if len(matches) != 1:
        raise RuntimeError(
            "transformer_model_adapter_match_count:"
            f"{len(matches)}:{[row.model_key for row in matches]}"
        )
    return matches[0]


def projection_free_attention_inventory(model: nn.Module) -> tuple[dict[str, Any], ...]:
    """Record genuine attention modules that expose no Q/K/V/O parameters."""

    rows: list[dict[str, Any]] = []
    for path, module in model.named_modules():
        if type(module).__name__ != "ScaledDotProductAttention":
            continue
        direct_parameters = sum(
            int(parameter.numel()) for parameter in module.parameters(recurse=False)
        )
        rows.append(
            {
                "module_path": path,
                "module_type": type(module).__name__,
                "parameter_count": direct_parameters,
                "pattern": "projection_free_scaled_dot_product_attention",
                "attention_dh_domain": False,
                "reason": "query_key_value_are_input_feature_tensors_without_qkvo_projection_parameters",
                "activation_precision_boundary_supported": True,
            }
        )
    return tuple(rows)


def _projection_free_precision_units(
    adapter: TransformerModelAdapter,
    inventory: Sequence[Mapping[str, Any]],
    *,
    ordering_start: int,
) -> tuple[TransformerPrecisionUnit, ...]:
    units: list[TransformerPrecisionUnit] = []
    for row in inventory:
        path = str(row["module_path"])
        prefix = f"transformer_precision::{path}"
        common = {
            "model": adapter.model_key,
            "family": f"{adapter.model_key}_projection_free_attention",
        }
        units.extend(
            (
                TransformerPrecisionUnit(
                    unit_id=f"{prefix}::qk_matmul",
                    module_paths=(f"{path}::__qk_bmm__",),
                    role="qk_matmul",
                    allowed_states=("A32",),
                    default_state="A32",
                    activation_only=True,
                    protected=True,
                    protection_reason="operands_accumulation_and_output_fixed_fp32",
                    ordering=ordering_start + len(units),
                    metadata={
                        **common,
                        "functional_owner": path,
                        "functional_op": "bmm",
                        "functional_call_index": 0,
                        "default_precision": "FP32",
                    },
                    **common,
                ),
                TransformerPrecisionUnit(
                    unit_id=f"{prefix}::softmax",
                    module_paths=(f"{path}::__softmax_output__",),
                    role="softmax",
                    allowed_states=("A32", "A16", "A8"),
                    default_state="A32",
                    activation_only=True,
                    protected=False,
                    ordering=ordering_start + len(units) + 1,
                    metadata={
                        **common,
                        "functional_owner": path,
                        "functional_op": "softmax",
                        "functional_call_index": 0,
                        "A8_semantics": "floating_softmax_then_qdq_int8_output",
                        "default_precision": "FP32",
                    },
                    **common,
                ),
                TransformerPrecisionUnit(
                    unit_id=f"{prefix}::av",
                    module_paths=(f"{path}::__av_bmm__",),
                    role="av_matmul",
                    allowed_states=("A32", "A16", "A8"),
                    default_state="A32",
                    activation_only=True,
                    protected=False,
                    ordering=ordering_start + len(units) + 2,
                    metadata={
                        **common,
                        "functional_owner": path,
                        "functional_op": "bmm",
                        "functional_call_index": 1,
                        "default_precision": "FP32",
                    },
                    **common,
                ),
            )
        )
    return tuple(units)


@dataclass(frozen=True)
class TransformerSearchComponents:
    adapter: TransformerModelAdapter
    attention_instances: tuple[AttentionInstanceSpec, ...]
    ffn_instances: tuple[FFNInstanceSpec, ...]
    transformer_domains: tuple[LocalPruningDomain, ...]
    precision_units: tuple[TransformerPrecisionUnit, ...]
    quantization_groups: tuple[QuantizationSearchGroup, ...]
    projection_free_attention: tuple[dict[str, Any], ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": "heal-transformer-search-components-v1",
            "model_key": self.adapter.model_key,
            "canonical_name": self.adapter.canonical_name,
            "attention_instances": [row.__dict__ for row in self.attention_instances],
            "ffn_instances": [row.__dict__ for row in self.ffn_instances],
            "transformer_domains": [row.to_dict() for row in self.transformer_domains],
            "precision_units": [row.to_dict() for row in self.precision_units],
            "quantization_groups": [row.to_dict() for row in self.quantization_groups],
            "projection_free_attention": [dict(row) for row in self.projection_free_attention],
            "metadata": dict(self.metadata),
        }
        payload["component_manifest_hash"] = _hash(payload)
        return payload


def build_transformer_search_components(
    model: nn.Module,
    config: Mapping[str, Any],
    *,
    attention_rankings: Mapping[str, Mapping[str, Sequence[Sequence[int]]]] | None = None,
    ffn_rankings: Mapping[str, Sequence[int]] | None = None,
    allow_identity_ranking: bool = False,
    active_module_paths: Sequence[str] | None = None,
) -> TransformerSearchComponents:
    adapter = detect_transformer_model_adapter(config)
    domains, attention, ffn = build_transformer_pruning_domains(
        model,
        model_name=adapter.model_key,
        attention_rankings=attention_rankings,
        ffn_rankings=ffn_rankings,
        allow_identity_ranking=allow_identity_ranking,
    )
    adapter.validate(attention, ffn)
    active = set(str(value) for value in (active_module_paths or ()))
    layernorm_paths = tuple(
        path
        for path, module in model.named_modules()
        if isinstance(module, nn.LayerNorm) and (not active or path in active)
    )
    units = build_transformer_precision_units(
        attention,
        ffn,
        model=model,
        layernorm_paths=layernorm_paths,
    )
    if active:
        # Runtime shape collectors generally observe weighted projections but
        # do not necessarily hook lightweight activation modules (Softmax,
        # GELU, etc.).  Once an Attention/FFN instance is proven active by one
        # of its real projections, retain the complete deployment contract for
        # that instance.  Otherwise active-graph filtering would silently
        # remove the very activation boundaries Stage-1 must score.
        active_owners: set[str] = set()
        for spec in attention:
            projection_paths = (
                spec.q_projection_paths
                + spec.k_projection_paths
                + spec.v_projection_paths
                + spec.output_projection_paths
            )
            if spec.module_path in active or any(
                path in active for path in projection_paths
            ):
                active_owners.add(spec.module_path)
        for spec in ffn:
            projection_paths = tuple(
                path
                for path in (
                    spec.first_projection_path,
                    spec.second_projection_path,
                    spec.gate_projection_path,
                    spec.up_projection_path,
                    spec.down_projection_path,
                )
                if path
            )
            if spec.module_path in active or any(
                path in active for path in projection_paths
            ):
                active_owners.add(spec.module_path)

        def belongs_to_active_instance(unit: TransformerPrecisionUnit) -> bool:
            return any(
                unit.unit_id.startswith(f"transformer_precision::{owner}::")
                for owner in active_owners
            )

        units = tuple(
            unit
            for unit in units
            if unit.protected
            or belongs_to_active_instance(unit)
            or any("::__" in path or path in active for path in unit.module_paths)
        )
    projection_free = projection_free_attention_inventory(model)
    if not adapter.requires_trainable_attention and not projection_free:
        raise RuntimeError(f"{adapter.model_key}_projection_free_attention_not_discovered")
    units = tuple(units) + _projection_free_precision_units(
        adapter,
        projection_free,
        ordering_start=len(units),
    )
    return TransformerSearchComponents(
        adapter=adapter,
        attention_instances=tuple(attention),
        ffn_instances=tuple(ffn),
        transformer_domains=tuple(domains),
        precision_units=tuple(units),
        quantization_groups=build_transformer_quantization_groups(units),
        projection_free_attention=projection_free,
        metadata={
            "family_is_metadata_not_width_sharing": True,
            "independent_attention_instance_default": True,
            "diagnostic_identity_ranking": bool(allow_identity_ranking),
            "active_module_filter_applied": bool(active),
        },
    )


def build_unified_transformer_search_space(
    components: TransformerSearchComponents,
    *,
    cnn_domains: Sequence[LocalPruningDomain] = (),
    cnn_quantization_groups: Sequence[QuantizationSearchGroup] = (),
    pruning_unit_ids: Sequence[str] = (),
    calibration_manifest_hash: str = "",
    trace_snapshot_hash: str = "",
    onnx_export_config_hash: str = "",
    tensorrt_version: str = "",
    gpu_compute_capability: str = "",
    builder_flags: Mapping[str, Any] | None = None,
    plugin_hashes: Mapping[str, str] | None = None,
) -> SearchSpaceSpec:
    """Merge CNN and Transformer adapters into the existing scalar-gene schema."""

    domains = tuple(cnn_domains) + tuple(components.transformer_domains)
    domain_ids = [row.domain_id for row in domains]
    if len(domain_ids) != len(set(domain_ids)):
        raise RuntimeError("unified_search_duplicate_domain_id")
    transformer_paths = {
        path
        for unit in components.precision_units
        for path in unit.module_paths
        if "::__" not in path
    }
    filtered_cnn_groups = tuple(
        group
        for group in cnn_quantization_groups
        if not transformer_paths.intersection(group.module_paths)
    )
    groups = filtered_cnn_groups + tuple(components.quantization_groups)
    group_ids = [row.group_id for row in groups]
    if len(group_ids) != len(set(group_ids)):
        raise RuntimeError("unified_search_duplicate_precision_group_id")
    return SearchSpaceSpec(
        pruning_unit_ids=list(pruning_unit_ids),
        precision_layer_ids=sorted(
            {path for group in groups for path in group.module_paths}
        ),
        quantization_groups=groups,
        pruning_domains=domains,
        default_precision="FP32",
        pruning_policy_version="cnn-transformer-fixed-nested-domain-width-v1",
        precision_policy_version="cnn-transformer-w32a32-w16a16-w8a8-contract-v1",
        trace_snapshot_hash=str(trace_snapshot_hash),
        calibration_manifest_hash=str(calibration_manifest_hash),
        onnx_export_config_hash=str(onnx_export_config_hash),
        tensorrt_version=str(tensorrt_version),
        gpu_compute_capability=str(gpu_compute_capability),
        builder_flags=dict(builder_flags or {}),
        plugin_hashes=dict(plugin_hashes or {}),
    )


__all__ = [
    "MODEL_ADAPTERS",
    "TransformerModelAdapter",
    "TransformerSearchComponents",
    "build_transformer_search_components",
    "build_unified_transformer_search_space",
    "detect_transformer_model_adapter",
    "projection_free_attention_inventory",
]
