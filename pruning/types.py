"""Schema-versioned dataclasses shared by formal pruning APIs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

import torch.nn as nn


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _jsonable(value.to_dict())
    return value


def stable_json_hash(payload: Any) -> str:
    raw = json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass
class ImportanceResult:
    mode: str
    normalization: str
    aggregation: str
    raw_scores: dict[str, float]
    normalized_scores: dict[str, float]
    unit_parameter_costs: dict[str, int] = field(default_factory=dict)
    unit_scores: list[dict[str, Any]] = field(default_factory=list)
    calibration_batches: int = 0
    task_loss: str = ""
    gradient_accumulation: str = "mean"
    implementation_version: str = "first-order-taylor-v1"
    schema_version: str = "importance-result-v1"

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass
class AtomicPruneUnit:
    scope_id: str
    root_module_path: str
    root_axis: str
    root_indices: list[int]
    source_coupled_unit_ids: list[str]
    normalized_score: float
    raw_score: float | None = None
    parameter_cost: int = 0
    channel_cost: int = 1
    protected: bool = False
    protection_reason: str = ""
    group_keep_map: dict[int, list[int]] = field(default_factory=dict)
    group_prune_map: dict[int, list[int]] = field(default_factory=dict)
    constraints: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "atomic-prune-unit-v1"
    _stable_id: str = field(default="", repr=False, compare=False)

    def __post_init__(self) -> None:
        self.root_indices = sorted({int(index) for index in self.root_indices})
        self.source_coupled_unit_ids = sorted({str(value) for value in self.source_coupled_unit_ids})
        self.group_keep_map = {int(key): sorted({int(v) for v in values}) for key, values in self.group_keep_map.items()}
        self.group_prune_map = {int(key): sorted({int(v) for v in values}) for key, values in self.group_prune_map.items()}
        if self.channel_cost <= 0:
            self.channel_cost = max(len(self.root_indices), 1)

    @property
    def stable_id(self) -> str:
        if self._stable_id:
            return self._stable_id
        payload = {
            "schema": self.schema_version,
            "scope_id": self.scope_id,
            "root_module_path": self.root_module_path,
            "root_axis": self.root_axis,
            "root_indices": self.root_indices,
            "source_coupled_unit_ids": self.source_coupled_unit_ids,
            "group_keep_map": self.group_keep_map,
        }
        return f"apu_{stable_json_hash(payload)[:24]}"

    def to_dict(self) -> dict[str, Any]:
        row = _jsonable(asdict(self))
        row.pop("_stable_id", None)
        row["stable_id"] = self.stable_id
        return row

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AtomicPruneUnit":
        data = dict(payload)
        data["_stable_id"] = data.pop("stable_id", "")
        return cls(**data)


@dataclass
class SamplingPruningEntry:
    request_id: str
    scope_id: str
    module_path: str
    axis: str
    prune_indices: list[int]
    source_atomic_unit_ids: list[str] = field(default_factory=list)
    group_keep_map: dict[int, list[int]] = field(default_factory=dict)
    group_prune_map: dict[int, list[int]] = field(default_factory=dict)
    fixed_output_contract: bool = False
    protection_reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.prune_indices = sorted({int(index) for index in self.prune_indices})
        self.source_atomic_unit_ids = sorted({str(value) for value in self.source_atomic_unit_ids})
        self.group_keep_map = {int(k): sorted({int(v) for v in values}) for k, values in self.group_keep_map.items()}
        self.group_prune_map = {int(k): sorted({int(v) for v in values}) for k, values in self.group_prune_map.items()}

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SamplingPruningEntry":
        return cls(**dict(payload))


@dataclass
class SamplingPruningRequest:
    entries: list[SamplingPruningEntry] = field(default_factory=list)
    selected_atomic_unit_ids: list[str] = field(default_factory=list)
    requested_channel_cost: int = 0
    requested_parameter_cost: int = 0
    one_shot: bool = True
    selector: str = "global_one_shot"
    schema_version: str = "sampling-pruning-request-v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "entries": [entry.to_dict() for entry in self.entries],
            "selected_atomic_unit_ids": list(self.selected_atomic_unit_ids),
            "requested_channel_cost": int(self.requested_channel_cost),
            "requested_parameter_cost": int(self.requested_parameter_cost),
            "one_shot": bool(self.one_shot),
            "selector": self.selector,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SamplingPruningRequest":
        data = dict(payload)
        data["entries"] = [
            row if isinstance(row, SamplingPruningEntry) else SamplingPruningEntry.from_dict(row)
            for row in data.get("entries", [])
        ]
        return cls(**data)


@dataclass
class PhysicalPruningPlanEntry:
    module_path: str
    axis: str
    prune_indices: list[int]
    keep_indices: list[int]
    original_axis_size: int
    source_request_ids: list[str]
    group_keep_map: dict[int, list[int]] = field(default_factory=dict)
    group_prune_map: dict[int, list[int]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    repaired: bool = False
    repair_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PhysicalPruningPlanEntry":
        return cls(**dict(payload))


@dataclass
class PhysicalPruningPlan:
    entries: list[PhysicalPruningPlanEntry] = field(default_factory=list)
    source_request: SamplingPruningRequest | None = None
    indices_frozen_before_materialization: bool = True
    conflicts: list[dict[str, Any]] = field(default_factory=list)
    schema_version: str = "physical-pruning-plan-v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "entries": [entry.to_dict() for entry in self.entries],
            "source_request": self.source_request.to_dict() if self.source_request else None,
            "indices_frozen_before_materialization": self.indices_frozen_before_materialization,
            "conflicts": _jsonable(self.conflicts),
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PhysicalPruningPlan":
        data = dict(payload)
        data["entries"] = [
            row if isinstance(row, PhysicalPruningPlanEntry) else PhysicalPruningPlanEntry.from_dict(row)
            for row in data.get("entries", [])
        ]
        source = data.get("source_request")
        if isinstance(source, Mapping):
            data["source_request"] = SamplingPruningRequest.from_dict(source)
        return cls(**data)


@dataclass
class AlignmentRepair:
    channels_before: int
    requested_keep: int
    final_keep: int
    alignment: int
    repaired: bool
    protection_preserved: bool = False
    reason: str = ""


@dataclass
class GroupedConvShapeReport:
    legal: bool
    in_channels: int
    out_channels: int
    groups: int
    in_channels_per_group: int | None
    out_channels_per_group: int | None
    channels_per_group: int | None
    allowed_channels_per_group: tuple[int, ...]
    violations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass
class GroupedConvSelectionDecision:
    selection_policy: str
    group_keep_map: dict[int, list[int]]
    group_prune_map: dict[int, list[int]]
    per_group_raw_scores: dict[int, list[float]]
    per_group_normalized_scores: dict[int, list[float]]
    selected_local_indices: dict[int, list[int]]
    final_channels_per_group: int
    alignment_repair: dict[str, Any]
    legality_report: dict[str, Any]
    schema_version: str = "grouped-conv-selection-v1"

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(frozen=True)
class DirectionalProtectionPolicy:
    module_path: str
    root_pruning_allowed: bool = True
    input_dependency_pruning_allowed: bool = True
    output_dependency_pruning_allowed: bool = True
    fixed_output_contract: bool = False
    protection_reason: str = ""
    schema_version: str = "directional-protection-v1"


@dataclass
class ApplicationLedgerEntry:
    request_id: str
    status: str
    module_path: str
    axis: str
    requested_prune_indices: list[int]
    applied_prune_indices: list[int] = field(default_factory=list)
    applied_keep_indices: list[int] = field(default_factory=list)
    reason: str = ""
    merged_into_request_id: str = ""
    alignment: dict[str, Any] = field(default_factory=dict)
    protection: dict[str, Any] = field(default_factory=dict)
    closure: dict[str, Any] = field(default_factory=dict)


@dataclass
class PhysicalPruningApplicationLedger:
    entries: list[ApplicationLedgerEntry]
    ledger_schema_version: str = "physical-pruning-application-ledger-v2"

    def to_dict(self) -> dict[str, Any]:
        counts = {status: 0 for status in ("applied", "repaired", "merged", "skipped")}
        for entry in self.entries:
            counts[entry.status] = counts.get(entry.status, 0) + 1
        return {
            "ledger_schema_version": self.ledger_schema_version,
            "entry_count": len(self.entries),
            "status_counts": counts,
            "entries": [_jsonable(asdict(entry)) for entry in self.entries],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PhysicalPruningApplicationLedger":
        return cls(
            entries=[ApplicationLedgerEntry(**dict(row)) for row in payload.get("entries", [])],
            ledger_schema_version=str(payload.get("ledger_schema_version", "physical-pruning-application-ledger-v2")),
        )


@dataclass
class PhysicalModuleSnapshot:
    canonical_module_name: str
    module_type: str
    canonical_order: int
    in_channels: int | None = None
    out_channels: int | None = None
    in_features: int | None = None
    out_features: int | None = None
    num_features: int | None = None
    groups: int = 1
    kernel_size: list[int] = field(default_factory=list)
    stride: list[int] = field(default_factory=list)
    padding: list[int] = field(default_factory=list)
    dilation: list[int] = field(default_factory=list)
    output_padding: list[int] = field(default_factory=list)
    weight_shape: list[int] = field(default_factory=list)
    bias_shape: list[int] = field(default_factory=list)
    parameter_count: int = 0
    parameter_size_bytes: int = 0
    source_state_dict_key: str = ""
    source_bias_state_dict_key: str = ""
    protection_policy: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass
class PhysicalStructureSnapshot:
    modules: list[PhysicalModuleSnapshot]
    parameter_count: int
    parameter_size_bytes: int
    weighted_module_count: int
    snapshot_schema_version: str = "physical-structure-snapshot-v2"
    generated_from: str = "live_model_and_state_dict"

    @property
    def module_count(self) -> int:
        return len(self.modules)

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_schema_version": self.snapshot_schema_version,
            "generated_from": self.generated_from,
            "module_count": self.module_count,
            "weighted_module_count": self.weighted_module_count,
            "parameter_count": self.parameter_count,
            "parameter_size_bytes": self.parameter_size_bytes,
            "modules": [row.to_dict() for row in self.modules],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PhysicalStructureSnapshot":
        return cls(
            modules=[PhysicalModuleSnapshot(**dict(row)) for row in payload.get("modules", [])],
            parameter_count=int(payload.get("parameter_count", 0)),
            parameter_size_bytes=int(payload.get("parameter_size_bytes", 0)),
            weighted_module_count=int(payload.get("weighted_module_count", 0)),
            snapshot_schema_version=str(payload.get("snapshot_schema_version", "")),
            generated_from=str(payload.get("generated_from", "")),
        )


@dataclass(frozen=True)
class PhysicalHashes:
    structure_hash_v2: str
    shape_hash_v2: str
    snapshot_hash: str
    hash_schema_version: str = "physical-structure-v2"
    model_hash: str = ""
    config_hash: str = ""

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass
class PhysicalValidationResult:
    passed: bool
    issues: list[dict[str, Any]]
    snapshot: PhysicalStructureSnapshot
    hashes: PhysicalHashes
    forward_checked: bool = False
    output_contract_checked: bool = False
    schema_version: str = "physical-validation-v1"


@dataclass
class MaterializationResult:
    model: nn.Module
    plan: PhysicalPruningPlan
    ledger: PhysicalPruningApplicationLedger
    snapshot: PhysicalStructureSnapshot | None = None
    validation: PhysicalValidationResult | None = None


@dataclass
class ModelProvenance:
    config_path: str
    checkpoint_path: str
    checkpoint_sha256: str
    device: str
    training: bool
    strict_state_dict: bool
    schema_version: str = "model-provenance-v1"


@dataclass
class ModelLoadResult:
    model: nn.Module
    provenance: ModelProvenance
    missing_keys: list[str] = field(default_factory=list)
    unexpected_keys: list[str] = field(default_factory=list)

    def __iter__(self):
        yield self.model
        yield self.provenance


__all__ = [
    "AlignmentRepair",
    "ApplicationLedgerEntry",
    "AtomicPruneUnit",
    "DirectionalProtectionPolicy",
    "GroupedConvSelectionDecision",
    "GroupedConvShapeReport",
    "ImportanceResult",
    "MaterializationResult",
    "ModelLoadResult",
    "ModelProvenance",
    "PhysicalHashes",
    "PhysicalModuleSnapshot",
    "PhysicalPruningApplicationLedger",
    "PhysicalPruningPlan",
    "PhysicalPruningPlanEntry",
    "PhysicalStructureSnapshot",
    "PhysicalValidationResult",
    "SamplingPruningEntry",
    "SamplingPruningRequest",
    "stable_json_hash",
]
