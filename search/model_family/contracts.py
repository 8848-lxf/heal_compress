"""Typed contracts for model-family search and deployment adapters.

These contracts intentionally live beside, rather than inside, the verified
``lidar_pyramid`` integration.  A new model family must describe what it can
prune, quantize, export, and deploy before any of those capabilities are
enabled in a production search space.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any


VALID_PRECISIONS = ("FP32", "FP16", "INT8")


def _precisions(values: tuple[str, ...]) -> tuple[str, ...]:
    normalized = tuple(str(value).upper() for value in values)
    if not normalized or any(value not in VALID_PRECISIONS for value in normalized):
        raise ValueError(f"invalid_precision_capability:{normalized}")
    return normalized


@dataclass(frozen=True)
class WeightedOpCapability:
    """Quantization contract for one module or functional weighted op."""

    canonical_id: str
    module_path: str
    op_type: str
    source_kind: str
    weight_shape: tuple[int, ...]
    allowed_precisions: tuple[str, ...]
    potential_precisions: tuple[str, ...]
    default_precision: str
    weight_granularity: str
    weight_axis: int | None
    input_scale_owner: str
    output_boundary: str
    production_enabled: bool
    gate_reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "weight_shape", tuple(int(value) for value in self.weight_shape))
        object.__setattr__(self, "allowed_precisions", _precisions(self.allowed_precisions))
        object.__setattr__(self, "potential_precisions", _precisions(self.potential_precisions))
        default = str(self.default_precision).upper()
        if default not in self.allowed_precisions:
            raise ValueError(f"default_precision_not_allowed:{self.canonical_id}:{default}")
        object.__setattr__(self, "default_precision", default)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_id": self.canonical_id,
            "module_path": self.module_path,
            "op_type": self.op_type,
            "source_kind": self.source_kind,
            "weight_shape": list(self.weight_shape),
            "allowed_precisions": list(self.allowed_precisions),
            "potential_precisions": list(self.potential_precisions),
            "default_precision": self.default_precision,
            "weight_granularity": self.weight_granularity,
            "weight_axis": self.weight_axis,
            "input_scale_owner": self.input_scale_owner,
            "output_boundary": self.output_boundary,
            "production_enabled": bool(self.production_enabled),
            "gate_reason": self.gate_reason,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class PruningDomainCapability:
    """A legal structured-pruning domain proposed by a model-family adapter."""

    domain_id: str
    domain_kind: str
    member_modules: tuple[str, ...]
    original_width: int
    legal_widths: tuple[int, ...]
    ranking_unit: str
    production_enabled: bool
    gate_reason: str = ""
    constraints: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        widths = tuple(sorted({int(value) for value in self.legal_widths}))
        if not widths or int(self.original_width) not in widths:
            raise ValueError(f"original_width_not_legal:{self.domain_id}")
        object.__setattr__(self, "member_modules", tuple(str(value) for value in self.member_modules))
        object.__setattr__(self, "legal_widths", widths)
        object.__setattr__(self, "constraints", dict(self.constraints))

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain_id": self.domain_id,
            "domain_kind": self.domain_kind,
            "member_modules": list(self.member_modules),
            "original_width": int(self.original_width),
            "legal_widths": list(self.legal_widths),
            "ranking_unit": self.ranking_unit,
            "production_enabled": bool(self.production_enabled),
            "gate_reason": self.gate_reason,
            "constraints": dict(self.constraints),
        }


@dataclass(frozen=True)
class MergeBoundaryCapability:
    boundary_id: str
    merge_kind: str
    member_modules: tuple[str, ...]
    policy: str
    scale_policy: str
    output_requantization: str
    production_enabled: bool
    gate_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "boundary_id": self.boundary_id,
            "merge_kind": self.merge_kind,
            "member_modules": list(self.member_modules),
            "policy": self.policy,
            "scale_policy": self.scale_policy,
            "output_requantization": self.output_requantization,
            "production_enabled": bool(self.production_enabled),
            "gate_reason": self.gate_reason,
        }


@dataclass(frozen=True)
class DeploymentOperatorCapability:
    capability_id: str
    op_kinds: tuple[str, ...]
    module_paths: tuple[str, ...]
    deployment_mode: str
    precision_policy: str
    plugin_key: str = ""
    production_enabled: bool = False
    gate_reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability_id": self.capability_id,
            "op_kinds": list(self.op_kinds),
            "module_paths": list(self.module_paths),
            "deployment_mode": self.deployment_mode,
            "precision_policy": self.precision_policy,
            "plugin_key": self.plugin_key,
            "production_enabled": bool(self.production_enabled),
            "gate_reason": self.gate_reason,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class PluginRequirement:
    plugin_key: str
    op_types: tuple[str, ...]
    required: bool
    compatibility_status: str
    reusable_implementation: str = ""
    compatibility_checks: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "plugin_key": self.plugin_key,
            "op_types": list(self.op_types),
            "required": bool(self.required),
            "compatibility_status": self.compatibility_status,
            "reusable_implementation": self.reusable_implementation,
            "compatibility_checks": list(self.compatibility_checks),
        }


@dataclass(frozen=True)
class ModelFamilyAudit:
    schema_version: str
    family_id: str
    model_type: str
    parameter_count: int
    weighted_ops: tuple[WeightedOpCapability, ...]
    pruning_domains: tuple[PruningDomainCapability, ...]
    merge_boundaries: tuple[MergeBoundaryCapability, ...]
    deployment_operators: tuple[DeploymentOperatorCapability, ...]
    plugin_requirements: tuple[PluginRequirement, ...]
    input_contract: dict[str, Any]
    blockers: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "family_id": self.family_id,
            "model_type": self.model_type,
            "parameter_count": int(self.parameter_count),
            "weighted_ops": [row.to_dict() for row in self.weighted_ops],
            "pruning_domains": [row.to_dict() for row in self.pruning_domains],
            "merge_boundaries": [row.to_dict() for row in self.merge_boundaries],
            "deployment_operators": [row.to_dict() for row in self.deployment_operators],
            "plugin_requirements": [row.to_dict() for row in self.plugin_requirements],
            "input_contract": dict(self.input_contract),
            "blockers": list(self.blockers),
            "metadata": dict(self.metadata),
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload()
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        payload["audit_hash"] = hashlib.sha256(encoded).hexdigest()
        return payload
