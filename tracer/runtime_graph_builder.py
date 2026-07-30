"""Typed dependency graph fallback from one real tensor-flow execution."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from typing import Any

import torch.nn as nn

from .config import TraceConfig
from .generic_tracer import GenericTracer
from .module_call_tracer import is_weighted_module
from .op_graph import build_op_graph
from .types import (
    DependencyEdge,
    DependencyGraphResult,
    DependencyMember,
    DependencyScope,
    ModuleInventoryEntry,
    OperationInventoryEntry,
    TensorInventoryEntry,
)


def _module_inventory(model: nn.Module) -> list[ModuleInventoryEntry]:
    rows: list[ModuleInventoryEntry] = []
    for name, module in model.named_modules():
        if not name:
            continue
        values: dict[str, int | None] = {}
        for attribute in (
            "in_channels",
            "out_channels",
            "in_features",
            "out_features",
            "num_features",
            "groups",
        ):
            value = getattr(module, attribute, None)
            values[attribute] = int(value) if value is not None else None
        rows.append(
            ModuleInventoryEntry(
                module_path=name,
                module_type=type(module).__name__,
                weighted=is_weighted_module(module),
                parameter_shapes={
                    key: tuple(int(dim) for dim in value.shape)
                    for key, value in module.named_parameters(recurse=False)
                },
                buffer_shapes={
                    key: tuple(int(dim) for dim in value.shape)
                    for key, value in module.named_buffers(recurse=False)
                },
                **values,
            )
        )
    return rows


def _filtered_runtime_trace(trace: dict[str, Any]) -> dict[str, Any]:
    nodes = {
        str(name): dict(row)
        for name, row in trace.get("nodes", {}).items()
        if row.get("type") == "TensorOp" or bool(row.get("executed", False))
    }
    edges = [
        dict(row)
        for row in trace.get("edges", [])
        if str(row.get("src", "")) in nodes and str(row.get("dst", "")) in nodes
    ]
    return {"nodes": nodes, "edges": edges}


def _inventories(trace: dict[str, Any]) -> tuple[list[OperationInventoryEntry], list[TensorInventoryEntry]]:
    incoming: dict[str, list[str]] = defaultdict(list)
    outgoing: dict[str, list[str]] = defaultdict(list)
    for edge in trace.get("edges", []):
        source, target = str(edge["src"]), str(edge["dst"])
        incoming[target].append(source)
        outgoing[source].append(target)
    operations: list[OperationInventoryEntry] = []
    tensors: list[TensorInventoryEntry] = []
    for name, row in sorted(trace.get("nodes", {}).items()):
        tensor_op = row.get("type") == "TensorOp"
        output_shapes = [tuple(int(dim) for dim in shape) for shape in row.get("output_shapes", [])]
        input_shapes = [tuple(int(dim) for dim in shape) for shape in row.get("input_shapes", [])]
        op_id = f"runtime::{name}"
        output_ids = [f"{op_id}:out{index}" for index in range(len(output_shapes))]
        operations.append(
            OperationInventoryEntry(
                op_id=op_id,
                op_kind="call_function" if tensor_op else "call_module",
                op_type=str(row.get("op") if tensor_op else row.get("type", "Unknown")),
                target=str(row.get("op") if tensor_op else row.get("type", "")),
                module_path="" if tensor_op else str(name),
                input_ids=[f"runtime::{value}" for value in sorted(set(incoming.get(name, [])))],
                output_ids=output_ids,
                input_shapes=input_shapes,
                output_shapes=output_shapes,
                metadata={
                    key: row[key]
                    for key in (
                        "cat_dim",
                        "num_inputs",
                        "module_scope",
                        "input_dtypes",
                        "output_dtypes",
                    )
                    if key in row
                },
            )
        )
        for index, shape in enumerate(output_shapes):
            tensors.append(
                TensorInventoryEntry(
                    tensor_id=f"{op_id}:out{index}",
                    shape=shape,
                    dtype="runtime_observed",
                    producer_op_id=op_id,
                    consumer_op_ids=[f"runtime::{value}" for value in sorted(set(outgoing.get(name, [])))],
                    output_index=index,
                )
            )
    return operations, tensors


def _scope_from_group(group: Any) -> DependencyScope:
    root_modules = sorted(str(value) for value in group.meta.get("roots", []))
    if not root_modules:
        root_modules = sorted({str(item.name) for item in group.items if item.direction == "out"})
    if not root_modules:
        raise RuntimeError(f"runtime dependency group has no output root: {group.group_id}")
    members: list[DependencyMember] = []
    seen: set[tuple[str, str]] = set()
    for item in group.items:
        axis = "channel" if isinstance(item.module, nn.modules.batchnorm._BatchNorm) else str(item.direction)
        mappings = {
            root_index: [int(value) for value in item.local_keep([root_index])]
            for root_index in range(int(group.num_channels))
        }
        mappings = {key: value for key, value in mappings.items() if value}
        indices = sorted({value for values in mappings.values() for value in values})
        key = (str(item.name), axis)
        if key not in seen and indices:
            seen.add(key)
            members.append(
                DependencyMember(
                    module_path=str(item.name),
                    module_type=type(item.module).__name__,
                    axis=axis,
                    indices=indices,
                    dependency_type=str(item.reason or "runtime_dependency"),
                    index_map=mappings,
                )
            )
        if (
            axis == "out"
            and isinstance(item.module, (nn.Conv2d, nn.ConvTranspose2d))
            and int(item.module.groups) > 1
            and int(item.module.in_channels) == int(item.module.out_channels)
            and (str(item.name), "in") not in seen
        ):
            seen.add((str(item.name), "in"))
            members.append(
                DependencyMember(
                    module_path=str(item.name),
                    module_type=type(item.module).__name__,
                    axis="in",
                    indices=indices,
                    dependency_type="grouped_conv_coupled_input",
                    index_map=mappings,
                )
            )

    # A tensor may be concatenated with itself (for example, an ego feature is
    # used once as the local feature and once as the neighbour feature).  The
    # runtime propagation backend represents the concat output as one logical
    # reference space, so the same physical root channel can appear at two or
    # more logical offsets.  Leaving those aliases independent would permit a
    # structurally impossible prune: removing only one occurrence also removes
    # the shared producer channel.  Canonicalize such scopes against the single
    # physical root without relying on model names or hand-authored topology.
    channel_count = int(group.num_channels)
    alias_canonicalized = False
    if len(root_modules) == 1:
        root_name = root_modules[0]
        root_items = [
            item
            for item in group.items
            if str(item.name) == root_name and str(item.direction) == "out"
        ]
        physical_widths = {
            int(width)
            for item in root_items
            for width in (
                getattr(item.module, "out_channels", None),
                getattr(item.module, "out_features", None),
                getattr(item.module, "num_features", None),
            )
            if width is not None and int(width) > 0
        }
        if len(physical_widths) == 1:
            physical_width = next(iter(physical_widths))
            if (
                channel_count > physical_width
                and channel_count % physical_width == 0
                and str(group.meta.get("group_type", "")) == "cat"
            ):
                # The legacy concat layout stores one offset per module name,
                # so repeated uses of one producer retain only the final
                # occurrence.  Validate every retained root mapping against the
                # periodic physical index, then reconstruct all occurrences.
                aliases = {
                    physical_index: list(range(
                        physical_index,
                        channel_count,
                        physical_width,
                    ))
                    for physical_index in range(physical_width)
                }
                observed_root_mappings = [
                    (logical_index, int(value))
                    for logical_index in range(channel_count)
                    for item in root_items
                    for value in item.local_keep([logical_index])
                ]
                valid_alias_map = bool(root_items) and bool(observed_root_mappings)
                valid_alias_map = valid_alias_map and all(
                    0 <= physical_index < physical_width
                    and physical_index == logical_index % physical_width
                    for logical_index, physical_index in observed_root_mappings
                )
                valid_alias_map = (
                    valid_alias_map
                    and set(aliases) == set(range(physical_width))
                    and any(len(values) > 1 for values in aliases.values())
                )
                if valid_alias_map:
                    canonical_members: list[DependencyMember] = []
                    for member in members:
                        canonical_map = {
                            physical_index: sorted({
                                local_index
                                for logical_index in logical_indices
                                for local_index in member.index_map.get(logical_index, [])
                            })
                            for physical_index, logical_indices in aliases.items()
                        }
                        canonical_map = {
                            key: values for key, values in canonical_map.items() if values
                        }
                        canonical_members.append(
                            DependencyMember(
                                module_path=member.module_path,
                                axis=member.axis,
                                indices=sorted({
                                    value
                                    for values in canonical_map.values()
                                    for value in values
                                }),
                                dependency_type=member.dependency_type,
                                channel_offset=member.channel_offset,
                                module_type=member.module_type,
                                index_map=canonical_map,
                                protection_reason=member.protection_reason,
                            )
                        )
                    members = canonical_members
                    channel_count = physical_width
                    alias_canonicalized = True

    dependency_types = {member.dependency_type for member in members}
    if alias_canonicalized:
        dependency_types.add("repeated_root_alias_canonicalized")
    return DependencyScope(
        root_module_path=root_modules[0],
        root_axis="out",
        channel_count=channel_count,
        members=sorted(members, key=lambda row: (row.module_path, row.axis)),
        root_modules=root_modules,
        dependency_types=sorted(dependency_types),
        protected=bool(group.protected),
        protection_reason=str(group.protected_reason or ""),
    )


def build_runtime_dependency_graph(
    model: nn.Module,
    example_inputs: Any,
    *,
    config: TraceConfig,
    forward_fn: Callable[[nn.Module, Any], Any] | None = None,
) -> tuple[DependencyGraphResult, list[DependencyScope]]:
    """Execute and convert the established tensor-flow backend to typed data."""

    runtime_trace = _filtered_runtime_trace(GenericTracer(model, forward_fn=forward_fn).trace(example_inputs))
    policies = []
    from .protection import build_protection_policies

    inventory = _module_inventory(model)
    policies = build_protection_policies(inventory, config.protection)
    protected = [row.module_path for row in policies if row.fixed_output_contract]
    operation_graph = build_op_graph(runtime_trace, model, protected_layers=protected)
    # This is the established runtime dependency closure implementation. It is
    # used only as a graph backend; selection and materialization remain formal.
    from pruning.propagation import GroupBuilder

    groups = GroupBuilder(
        operation_graph,
        align=4,
        grouped_conv_mode="independent_group_topk",
        protect_residual_add=False,
    ).build()
    scopes = [_scope_from_group(group) for group in groups]
    edges: list[DependencyEdge] = []
    for scope in scopes:
        for root in scope.root_modules[1:]:
            edges.append(
                DependencyEdge(
                    source=scope.root_module_path,
                    target=root,
                    source_axis="out",
                    target_axis="out",
                    dependency_type="runtime_coupled_root",
                    channel_count=scope.channel_count,
                )
            )
        for member in scope.members:
            if member.module_path in scope.root_modules and member.axis == "out":
                continue
            edges.append(
                DependencyEdge(
                    source=scope.root_module_path,
                    target=member.module_path,
                    source_axis="out",
                    target_axis=member.axis,
                    dependency_type=member.dependency_type,
                    channel_count=scope.channel_count,
                    metadata={"index_map": member.index_map},
                )
            )
    operations, tensors = _inventories(runtime_trace)
    graph = DependencyGraphResult(
        graph_schema_version=config.graph_schema_version,
        module_inventory=inventory,
        op_inventory=operations,
        tensor_inventory=tensors,
        dependency_edges=edges,
    )
    return graph, scopes


__all__ = ["build_runtime_dependency_graph"]
