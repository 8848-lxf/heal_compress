"""Stable typed schemas for formal channel-dependency tracing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .hashing import stable_id, to_stable_primitive


class SerializableRecord:
    """Mixin providing deterministic JSON-compatible conversion."""

    def to_dict(self) -> dict[str, Any]:
        return to_stable_primitive(self)


@dataclass
class ModuleInventoryEntry(SerializableRecord):
    module_path: str
    module_type: str
    weighted: bool = False
    in_channels: int | None = None
    out_channels: int | None = None
    in_features: int | None = None
    out_features: int | None = None
    num_features: int | None = None
    groups: int | None = None
    parameter_shapes: dict[str, tuple[int, ...]] = field(default_factory=dict)
    buffer_shapes: dict[str, tuple[int, ...]] = field(default_factory=dict)


@dataclass
class OperationInventoryEntry(SerializableRecord):
    op_id: str
    op_kind: str
    op_type: str
    target: str
    module_path: str = ""
    input_ids: list[str] = field(default_factory=list)
    output_ids: list[str] = field(default_factory=list)
    input_shapes: list[tuple[int, ...]] = field(default_factory=list)
    output_shapes: list[tuple[int, ...]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TensorInventoryEntry(SerializableRecord):
    tensor_id: str
    shape: tuple[int, ...]
    dtype: str
    producer_op_id: str
    consumer_op_ids: list[str] = field(default_factory=list)
    output_index: int = 0


@dataclass
class ModuleCallRecord(SerializableRecord):
    module_path: str
    module_type: str
    call_index: int
    module_call_index: int
    weighted: bool
    input_shapes: list[tuple[int, ...]] = field(default_factory=list)
    output_shapes: list[tuple[int, ...]] = field(default_factory=list)


@dataclass
class ProtectionPolicy(SerializableRecord):
    module_path: str
    module_type: str
    root_pruning_allowed: bool = True
    input_dependency_pruning_allowed: bool = True
    output_dependency_pruning_allowed: bool = True
    fixed_output_contract: bool = False
    protection_reason: str = ""


@dataclass
class DependencyEdge(SerializableRecord):
    source: str
    target: str
    source_axis: str
    target_axis: str
    dependency_type: str
    channel_offset: int = 0
    channel_count: int | None = None
    operation_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    stable_id: str = ""

    def __post_init__(self) -> None:
        if not self.stable_id:
            self.stable_id = stable_id(
                "edge",
                {
                    "source": self.source,
                    "target": self.target,
                    "source_axis": self.source_axis,
                    "target_axis": self.target_axis,
                    "dependency_type": self.dependency_type,
                    "channel_offset": self.channel_offset,
                    "channel_count": self.channel_count,
                    "operation_id": self.operation_id,
                },
            )


@dataclass
class DependencyMember(SerializableRecord):
    module_path: str
    axis: str
    indices: list[int]
    dependency_type: str
    channel_offset: int = 0
    module_type: str = ""
    index_map: dict[int, list[int]] = field(default_factory=dict)
    protection_reason: str = ""

    def __post_init__(self) -> None:
        self.indices = sorted({int(value) for value in self.indices})
        self.index_map = {
            int(key): sorted({int(value) for value in values})
            for key, values in self.index_map.items()
        }


@dataclass
class DependencyScope(SerializableRecord):
    root_module_path: str
    root_axis: str
    channel_count: int
    members: list[DependencyMember] = field(default_factory=list)
    root_modules: list[str] = field(default_factory=list)
    dependency_types: list[str] = field(default_factory=list)
    protected: bool = False
    protection_reason: str = ""
    schema_version: str = "dependency-scope-v1"
    stable_id: str = ""

    def __post_init__(self) -> None:
        self.root_modules = sorted(set(self.root_modules or [self.root_module_path]))
        self.dependency_types = sorted(set(self.dependency_types))
        if not self.stable_id:
            self.stable_id = stable_id(
                "scope",
                {
                    "root_axis": self.root_axis,
                    "channel_count": self.channel_count,
                    "root_modules": self.root_modules,
                    "members": [member.to_dict() for member in self.members],
                },
            )


@dataclass
class CoupledChannelUnit(SerializableRecord):
    scope_id: str
    root_module_path: str
    root_axis: str
    root_channel_index: int
    members: list[DependencyMember] = field(default_factory=list)
    dependency_types: list[str] = field(default_factory=list)
    grouped_conv_metadata: dict[str, Any] = field(default_factory=dict)
    protected: bool = False
    protection_reason: str = ""
    schema_version: str = "coupled-channel-unit-v1"
    stable_id: str = ""

    def __post_init__(self) -> None:
        self.root_channel_index = int(self.root_channel_index)
        self.dependency_types = sorted(set(self.dependency_types))
        if not self.stable_id:
            # Policy choices, importance values and alignment repairs are
            # deliberately excluded: identity describes graph membership only.
            self.stable_id = stable_id(
                "cu",
                {
                    "root_axis": self.root_axis,
                    "root_channel_index": self.root_channel_index,
                    "members": sorted(
                        [member.to_dict() for member in self.members],
                        key=lambda row: (
                            str(row.get("module_path", "")),
                            str(row.get("axis", "")),
                            tuple(row.get("indices", [])),
                        ),
                    ),
                },
            )


@dataclass
class AtomicPruneUnit(SerializableRecord):
    scope_id: str
    root_module_path: str
    root_axis: str
    root_indices: list[int]
    source_coupled_unit_ids: list[str]
    members: list[DependencyMember] = field(default_factory=list)
    constraints: dict[str, Any] = field(default_factory=dict)
    protected: bool = False
    protection_reason: str = ""
    schema_version: str = "atomic-prune-unit-v1"
    stable_id: str = ""

    def __post_init__(self) -> None:
        self.root_indices = sorted({int(value) for value in self.root_indices})
        self.source_coupled_unit_ids = sorted(set(self.source_coupled_unit_ids))
        if not self.stable_id:
            self.stable_id = stable_id(
                "apu",
                {
                    "scope_id": self.scope_id,
                    "root_module_path": self.root_module_path,
                    "root_axis": self.root_axis,
                    "root_indices": self.root_indices,
                    "source_coupled_unit_ids": self.source_coupled_unit_ids,
                },
            )


@dataclass
class ConcreteCoupledPruningGroup(SerializableRecord):
    """A dependency scope instantiated with concrete root prune indices."""

    scope: DependencyScope
    prune_indices: list[int]
    source_atomic_unit_ids: list[str] = field(default_factory=list)
    schema_version: str = "concrete-coupled-pruning-group-v1"
    stable_id: str = ""

    def __post_init__(self) -> None:
        self.prune_indices = sorted({int(value) for value in self.prune_indices})
        self.source_atomic_unit_ids = sorted(set(self.source_atomic_unit_ids))
        if not self.stable_id:
            self.stable_id = stable_id(
                "ccpg",
                {
                    "scope_id": self.scope.stable_id,
                    "prune_indices": self.prune_indices,
                    "source_atomic_unit_ids": self.source_atomic_unit_ids,
                },
            )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ConcreteCoupledPruningGroup":
        payload = dict(data)
        scope_data = dict(payload["scope"])
        scope_data["members"] = [
            DependencyMember(**dict(member)) for member in scope_data.get("members", [])
        ]
        payload["scope"] = DependencyScope(**scope_data)
        return cls(**payload)


# Formal compatibility name. The legacy live-module recipe remains available
# from ``tracer.pruning_group.PruningGroup`` for old pruning entrypoints.
PruningGroup = DependencyScope


@dataclass
class OperationIssue(SerializableRecord):
    operation_id: str
    op_type: str
    reason: str
    input_shapes: list[tuple[int, ...]] = field(default_factory=list)
    output_shapes: list[tuple[int, ...]] = field(default_factory=list)
    channel_changing: bool = False


@dataclass
class TraceCoverage(SerializableRecord):
    weighted_modules_total: int = 0
    weighted_modules_called: int = 0
    weighted_module_coverage: float = 0.0
    traced_module_count: int = 0
    traced_operation_count: int = 0
    unresolved_operation_count: int = 0
    unsupported_operation_count: int = 0
    representative_forward_executed: bool = False
    dynamic_branch_enumeration_enabled: bool = False
    coverage_notes: list[str] = field(default_factory=list)


@dataclass
class ExampleInputTensor(SerializableRecord):
    path: str
    shape: tuple[int, ...]
    dtype: str
    device_type: str


@dataclass
class ExampleInputContract(SerializableRecord):
    call_style: str
    tensors: list[ExampleInputTensor] = field(default_factory=list)
    non_tensor_paths: list[str] = field(default_factory=list)


@dataclass
class DependencyGraphResult(SerializableRecord):
    graph_schema_version: str
    module_inventory: list[ModuleInventoryEntry]
    op_inventory: list[OperationInventoryEntry]
    tensor_inventory: list[TensorInventoryEntry]
    dependency_edges: list[DependencyEdge]
    unresolved_operations: list[OperationIssue] = field(default_factory=list)
    unsupported_operations: list[OperationIssue] = field(default_factory=list)


@dataclass
class TraceResult(SerializableRecord):
    graph_schema_version: str
    module_inventory: list[ModuleInventoryEntry]
    op_inventory: list[OperationInventoryEntry]
    tensor_inventory: list[TensorInventoryEntry]
    dependency_edges: list[DependencyEdge]
    dependency_scopes: list[DependencyScope]
    coupled_channel_units: list[CoupledChannelUnit]
    atomic_prune_units: list[AtomicPruneUnit]
    protected_modules: list[str]
    protected_units: list[str]
    protection_policies: list[ProtectionPolicy]
    protection_reasons: dict[str, str]
    unresolved_operations: list[OperationIssue]
    unsupported_operations: list[OperationIssue]
    trace_coverage: TraceCoverage
    example_input_contract: ExampleInputContract
    module_call_trace: list[ModuleCallRecord]
    trace_hash: str = ""
    config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TraceResult":
        payload = dict(data)

        def rows(name: str, typ: type[Any]) -> list[Any]:
            return [typ(**dict(row)) for row in payload.get(name, [])]

        scopes: list[DependencyScope] = []
        for row in payload.get("dependency_scopes", []):
            item = dict(row)
            item["members"] = [DependencyMember(**dict(member)) for member in item.get("members", [])]
            scopes.append(DependencyScope(**item))
        coupled: list[CoupledChannelUnit] = []
        for row in payload.get("coupled_channel_units", []):
            item = dict(row)
            item["members"] = [DependencyMember(**dict(member)) for member in item.get("members", [])]
            coupled.append(CoupledChannelUnit(**item))
        atomic: list[AtomicPruneUnit] = []
        for row in payload.get("atomic_prune_units", []):
            item = dict(row)
            item["members"] = [DependencyMember(**dict(member)) for member in item.get("members", [])]
            atomic.append(AtomicPruneUnit(**item))
        contract_data = dict(payload.get("example_input_contract", {}))
        contract_data["tensors"] = [
            ExampleInputTensor(**dict(row)) for row in contract_data.get("tensors", [])
        ]
        return cls(
            graph_schema_version=str(payload["graph_schema_version"]),
            module_inventory=rows("module_inventory", ModuleInventoryEntry),
            op_inventory=rows("op_inventory", OperationInventoryEntry),
            tensor_inventory=rows("tensor_inventory", TensorInventoryEntry),
            dependency_edges=rows("dependency_edges", DependencyEdge),
            dependency_scopes=scopes,
            coupled_channel_units=coupled,
            atomic_prune_units=atomic,
            protected_modules=[str(value) for value in payload.get("protected_modules", [])],
            protected_units=[str(value) for value in payload.get("protected_units", [])],
            protection_policies=rows("protection_policies", ProtectionPolicy),
            protection_reasons={
                str(key): str(value) for key, value in payload.get("protection_reasons", {}).items()
            },
            unresolved_operations=rows("unresolved_operations", OperationIssue),
            unsupported_operations=rows("unsupported_operations", OperationIssue),
            trace_coverage=TraceCoverage(**dict(payload.get("trace_coverage", {}))),
            example_input_contract=ExampleInputContract(**contract_data),
            module_call_trace=rows("module_call_trace", ModuleCallRecord),
            trace_hash=str(payload.get("trace_hash", "")),
            config=dict(payload.get("config", {})),
        )
