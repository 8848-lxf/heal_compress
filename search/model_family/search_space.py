"""V2X-ViT model-family search-space assembly.

This module is deliberately independent from the accepted lidar_pyramid
tracer/planner path.  The first physically supported pruning coordinate is the
hidden width of each Transformer feed-forward block.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import torch.nn as nn

from ..pruning_space.local_domains import LocalPruningDomain, build_local_pruning_domains
from ..quantization_space.types import QuantizationSearchGroup
from .contracts import ModelFamilyAudit, PruningDomainCapability, WeightedOpCapability


@dataclass(frozen=True)
class V2XViTDependencyMember:
    module_path: str
    axis: str
    indices: tuple[int, ...]


@dataclass(frozen=True)
class V2XViTAtomicPruningUnit:
    stable_id: str
    scope_id: str
    root_module_path: str
    root_axis: str
    root_indices: tuple[int, ...]
    members: tuple[V2XViTDependencyMember, ...]
    constraints: dict[str, Any] = field(default_factory=dict)
    normalized_score: float = 0.0
    protected: bool = False


def _module_capabilities(audit: ModelFamilyAudit) -> dict[str, WeightedOpCapability]:
    return {
        row.module_path: row
        for row in audit.weighted_ops
        if row.source_kind == "module"
    }


def build_v2xvit_ffn_atomic_units(
    model: nn.Module,
    audit: ModelFamilyAudit,
) -> tuple[list[V2XViTAtomicPruningUnit], dict[str, PruningDomainCapability]]:
    """Create exact first-Linear-output/second-Linear-input channel units."""

    modules = dict(model.named_modules())
    units: list[V2XViTAtomicPruningUnit] = []
    capabilities: dict[str, PruningDomainCapability] = {}
    for capability in audit.pruning_domains:
        if capability.domain_kind != "transformer_ffn_hidden_width":
            continue
        if len(capability.member_modules) != 2:
            raise RuntimeError(f"v2xvit_ffn_domain_member_count:{capability.domain_id}")
        first_path, second_path = capability.member_modules
        first = modules.get(first_path)
        second = modules.get(second_path)
        if not isinstance(first, nn.Linear) or not isinstance(second, nn.Linear):
            raise RuntimeError(f"v2xvit_ffn_domain_linear_missing:{capability.domain_id}")
        if first.out_features != second.in_features:
            raise RuntimeError(f"v2xvit_ffn_hidden_width_mismatch:{capability.domain_id}")
        if int(first.out_features) != int(capability.original_width):
            raise RuntimeError(f"v2xvit_ffn_capability_width_mismatch:{capability.domain_id}")
        scope_id = str(capability.domain_id)
        constraints = {
            **dict(capability.constraints),
            "v2xvit_domain_kind": "transformer_ffn_hidden_width",
            "capability_domain_id": capability.domain_id,
            "first_linear": first_path,
            "second_linear": second_path,
            "embedding_width_unchanged": True,
            "physical_materializer": "v2xvit_ffn_linear_pair_v1",
        }
        for channel in range(int(first.out_features)):
            units.append(
                V2XViTAtomicPruningUnit(
                    stable_id=f"v2xvit::ffn_hidden::{scope_id}::channel::{channel:04d}",
                    scope_id=scope_id,
                    root_module_path=first_path,
                    root_axis="out",
                    root_indices=(channel,),
                    members=(
                        V2XViTDependencyMember(first_path, "out", (channel,)),
                        V2XViTDependencyMember(second_path, "in", (channel,)),
                    ),
                    constraints=constraints,
                )
            )
        capabilities[first_path] = capability
    if not units:
        raise RuntimeError("v2xvit_ffn_pruning_space_empty")
    return units, capabilities


def build_ranked_v2xvit_ffn_domains(
    units: Sequence[V2XViTAtomicPruningUnit],
    importance_scores: Mapping[str, float],
) -> list[LocalPruningDomain]:
    domains = build_local_pruning_domains(
        units,
        importance_scores=importance_scores,
        ranking_method="real_train_manifest_pruning_only_first_plus_second_order_taylor",
        minimum_retained_ratio=0.25,
        dense_alignment=4,
    )
    expected = len({row.scope_id for row in units})
    if len(domains) != expected or any(
        len(row.legal_widths) <= 1 for row in domains
    ):
        raise RuntimeError(
            f"v2xvit_ffn_domain_contract_mismatch:{len(domains)}:"
            f"{[list(row.legal_widths) for row in domains]}"
        )
    return domains


def build_v2xvit_quantization_groups(
    model: nn.Module,
    audit: ModelFamilyAudit,
    *,
    active_module_paths: Iterable[str],
) -> list[QuantizationSearchGroup]:
    """Build one canonical gene per active parameterized module.

    A single-member group is still a real canonical quantization contract; its
    identity comes from the model-family audit, not a pruning dependency scope.
    Transformer/PFN/head entries currently allow only FP32/FP16.  The 24
    audited backbone/deblock/shrink entries also allow INT8; their Stage-2
    implementation uses the family-scoped semantic-boundary entropy/QDQ path
    and a strongly typed TensorRT realization check.
    """

    modules = dict(model.named_modules())
    capabilities = _module_capabilities(audit)
    groups: list[QuantizationSearchGroup] = []
    active = sorted({str(value) for value in active_module_paths})
    for ordering, module_path in enumerate(active):
        module = modules.get(module_path)
        capability = capabilities.get(module_path)
        if module is None or capability is None or getattr(module, "weight", None) is None:
            raise RuntimeError(f"v2xvit_active_weighted_module_unmapped:{module_path}")
        allowed = tuple(str(value).upper() for value in capability.allowed_precisions)
        if not allowed or "FP32" not in allowed:
            raise RuntimeError(f"v2xvit_precision_contract_missing_fp32:{module_path}")
        groups.append(
            QuantizationSearchGroup(
                group_id=f"v2xvit_qg::{capability.canonical_id}",
                module_paths=(module_path,),
                canonical_node_ids=(capability.canonical_id,),
                allowed_precisions=allowed,
                protected=len(allowed) == 1,
                protection_reason=capability.gate_reason if len(allowed) == 1 else "",
                ordering=ordering,
                parameter_count=sum(
                    int(parameter.numel())
                    for parameter in module.parameters(recurse=False)
                ),
                baseline_macs=float(module.weight.numel()),
                metadata={
                    "source": "heal_lidar_v2xvit_canonical_capability",
                    "canonical_id": capability.canonical_id,
                    "default_precision": "FP32",
                    "force_same_precision": True,
                    "production_int8_admitted": bool(
                        capability.production_enabled and "INT8" in allowed
                    ),
                    "int8_gate_reason": capability.gate_reason,
                    "weight_granularity": capability.weight_granularity,
                    "weight_axis": capability.weight_axis,
                    "input_scale_owner": capability.input_scale_owner,
                    "output_boundary": capability.output_boundary,
                    "merge_policy": "model_family_declared_fp16_merge_contract",
                    "stage2_qdq_status": (
                        "production_strongly_typed_qdq_validated"
                        if capability.production_enabled and "INT8" in allowed
                        else "mapped_fp16_fp32_protected_from_int8"
                    ),
                },
            )
        )
    return groups


__all__ = [
    "V2XViTAtomicPruningUnit",
    "V2XViTDependencyMember",
    "build_ranked_v2xvit_ffn_domains",
    "build_v2xvit_ffn_atomic_units",
    "build_v2xvit_quantization_groups",
]
