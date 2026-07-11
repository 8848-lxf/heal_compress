"""Channel-axis propagation over a shape-propagated Torch-FX graph."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import torch.nn as nn
from torch.fx import Node

from .config import DependencyConfig
from .exceptions import AmbiguousDependencyError
from .static_graph_builder import StaticGraphBuild, node_tensor_metadata
from .types import DependencyEdge, OperationIssue


@dataclass(frozen=True)
class _SourceRef:
    module_path: str
    channel_count: int
    logical_axis: int
    offset: int = 0
    index_scale: int = 1
    via: tuple[str, ...] = ()
    operation_id: str = ""


@dataclass
class ChannelMappingResult:
    """Dependency edges plus mapping-time unresolved operations."""

    edges: list[DependencyEdge]
    unresolved_operations: list[OperationIssue] = field(default_factory=list)
    unsupported_operations: list[OperationIssue] = field(default_factory=list)


def _shape(node: Node, index: int = 0) -> tuple[int, ...] | None:
    rows = node_tensor_metadata(node)
    return rows[index][0] if index < len(rows) else None


def _normalize_axis(axis: int, rank: int) -> int:
    return axis + rank if axis < 0 else axis


def _module_output_axis(module: nn.Module, shape: tuple[int, ...] | None) -> int:
    if isinstance(module, nn.Linear):
        return max((len(shape) if shape else 2) - 1, 0)
    return 1 if shape and len(shape) > 1 else 0


def _module_output_channels(module: nn.Module) -> int | None:
    for attr in ("out_channels", "out_features", "num_features"):
        value = getattr(module, attr, None)
        if value is not None:
            return int(value)
    return None


def _module_input_axis(module: nn.Module, shape: tuple[int, ...] | None) -> int:
    if isinstance(module, nn.Linear):
        return max((len(shape) if shape else 2) - 1, 0)
    return 1 if shape and len(shape) > 1 else 0


def _is_weighted(module: nn.Module) -> bool:
    return isinstance(
        module,
        (
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nn.ConvTranspose1d,
            nn.ConvTranspose2d,
            nn.ConvTranspose3d,
            nn.Linear,
        ),
    )


def _is_depthwise(module: nn.Module) -> bool:
    return isinstance(module, nn.Conv2d) and int(module.groups) == int(module.in_channels) == int(module.out_channels)


def _node_inputs(node: Node) -> list[Node]:
    return list(node.all_input_nodes)


def _cat_inputs(node: Node) -> list[Node]:
    if not node.args:
        return []
    value = node.args[0]
    if isinstance(value, (list, tuple)):
        return [item for item in value if isinstance(item, Node)]
    return [value] if isinstance(value, Node) else []


def _dedupe_sources(values: Iterable[_SourceRef]) -> list[_SourceRef]:
    rows: list[_SourceRef] = []
    seen: set[tuple[Any, ...]] = set()
    for value in values:
        key = (
            value.module_path,
            value.channel_count,
            value.logical_axis,
            value.offset,
            value.index_scale,
            value.via,
            value.operation_id,
        )
        if key not in seen:
            seen.add(key)
            rows.append(value)
    return rows


class _Mapper:
    def __init__(self, built: StaticGraphBuild, config: DependencyConfig):
        self.built = built
        self.graph = built.graph_module.graph
        self.modules = dict(built.graph_module.named_modules())
        self.op_by_id = {row.op_id: row for row in built.op_inventory}
        self.memo: dict[str, list[_SourceRef]] = {}
        self.visiting: set[str] = set()
        self.config = config
        self.unresolved: list[OperationIssue] = []
        self.unsupported: list[OperationIssue] = []

    def op_type(self, node: Node) -> str:
        return self.op_by_id[f"fx::{node.name}"].op_type

    def _issue(self, node: Node, reason: str, *, unsupported: bool) -> None:
        op = self.op_by_id[f"fx::{node.name}"]
        issue = OperationIssue(
            operation_id=op.op_id,
            op_type=op.op_type,
            reason=reason,
            input_shapes=list(op.input_shapes),
            output_shapes=list(op.output_shapes),
            channel_changing=True,
        )
        target = self.unsupported if unsupported else self.unresolved
        if not any(existing.operation_id == issue.operation_id and existing.reason == reason for existing in target):
            target.append(issue)

    def sources(self, node: Node) -> list[_SourceRef]:
        if node.name in self.memo:
            return self.memo[node.name]
        if node.name in self.visiting:
            return []
        self.visiting.add(node.name)
        try:
            result = self._sources_impl(node)
            self.memo[node.name] = _dedupe_sources(result)
            return self.memo[node.name]
        finally:
            self.visiting.discard(node.name)

    def _sources_impl(self, node: Node) -> list[_SourceRef]:
        op_type = self.op_type(node)
        if node.op == "call_module":
            module = self.modules[str(node.target)]
            if _is_weighted(module):
                count = _module_output_channels(module)
                if count is None:
                    return []
                return [
                    _SourceRef(
                        module_path=str(node.target),
                        channel_count=count,
                        logical_axis=_module_output_axis(module, _shape(node)),
                    )
                ]
            inputs = _node_inputs(node)
            return self.sources(inputs[0]) if inputs else []
        if node.op in {"placeholder", "get_attr"}:
            return []
        inputs = _node_inputs(node)
        if op_type == "Add":
            branches = [self.sources(child) for child in inputs]
            flat = [source for branch in branches for source in branch]
            counts = {source.channel_count * source.index_scale for source in flat}
            axes = {source.logical_axis for source in flat}
            if len(counts) > 1 or len(axes) > 1:
                self._issue(node, "ambiguous_residual_channel_mapping", unsupported=True)
                return []
            return [
                _SourceRef(**{**source.__dict__, "via": tuple(sorted(set(source.via + ("residual_add",))))})
                for source in flat
            ]
        if op_type == "Concat":
            branches = _cat_inputs(node)
            op = self.op_by_id[f"fx::{node.name}"]
            dim = int(op.metadata.get("dim", 0))
            output_shape = _shape(node)
            if output_shape:
                dim = _normalize_axis(dim, len(output_shape))
            offset = 0
            result: list[_SourceRef] = []
            for branch in branches:
                branch_shape = _shape(branch)
                branch_sources = self.sources(branch)
                branch_channels = int(branch_shape[dim]) if branch_shape and dim < len(branch_shape) else 0
                for source in branch_sources:
                    if source.logical_axis != dim:
                        continue
                    result.append(
                        _SourceRef(
                            module_path=source.module_path,
                            channel_count=source.channel_count,
                            logical_axis=dim,
                            offset=offset + source.offset,
                            index_scale=source.index_scale,
                            via=tuple(sorted(set(source.via + ("concat",)))),
                            operation_id=f"fx::{node.name}",
                        )
                    )
                offset += branch_channels
            return result
        if op_type == "Permute":
            if not inputs:
                return []
            sources = self.sources(inputs[0])
            input_shape = _shape(inputs[0])
            output_shape = _shape(node)
            metadata = self.op_by_id[f"fx::{node.name}"].metadata
            dims = list(metadata.get("dims", []))
            result: list[_SourceRef] = []
            for source in sources:
                new_axis: int | None = None
                if input_shape and output_shape and len(dims) == len(input_shape):
                    normalized = [_normalize_axis(int(value), len(input_shape)) for value in dims]
                    if source.logical_axis in normalized:
                        new_axis = normalized.index(source.logical_axis)
                elif input_shape and output_shape and len(dims) == 2:
                    left, right = (_normalize_axis(int(value), len(input_shape)) for value in dims)
                    new_axis = right if source.logical_axis == left else (left if source.logical_axis == right else source.logical_axis)
                if new_axis is None:
                    self._issue(node, "unresolved_permute_channel_axis", unsupported=True)
                    continue
                result.append(_SourceRef(**{**source.__dict__, "logical_axis": int(new_axis)}))
            return result
        if op_type == "View":
            if not inputs:
                return []
            sources = self.sources(inputs[0])
            input_shape = _shape(inputs[0])
            output_shape = _shape(node)
            result: list[_SourceRef] = []
            for source in sources:
                if not output_shape:
                    continue
                candidates = [idx for idx, value in enumerate(output_shape) if int(value) == source.channel_count]
                if len(candidates) == 1:
                    result.append(_SourceRef(**{**source.__dict__, "logical_axis": candidates[0]}))
                    continue
                # Common NCHW flatten: each source channel owns one contiguous
                # spatial block in the flattened feature axis.
                if input_shape and len(output_shape) == 2 and output_shape[1] % source.channel_count == 0:
                    scale = int(output_shape[1] // source.channel_count)
                    result.append(
                        _SourceRef(
                            module_path=source.module_path,
                            channel_count=source.channel_count,
                            logical_axis=1,
                            offset=source.offset * scale,
                            index_scale=source.index_scale * scale,
                            via=source.via,
                            operation_id=source.operation_id,
                        )
                    )
                    continue
                self._issue(node, "unresolved_view_channel_mapping", unsupported=True)
            return result
        if op_type == "Split":
            # The tuple itself is resolved by its getitem consumer.
            return self.sources(inputs[0]) if inputs else []
        if op_type == "Index" and inputs:
            base = inputs[0]
            if self.op_type(base) == "Split" and len(node.args) > 1 and isinstance(node.args[1], int):
                split_sources = self.sources(base)
                output_index = int(node.args[1])
                output_rows = node_tensor_metadata(base)
                result: list[_SourceRef] = []
                for source in split_sources:
                    if output_index >= len(output_rows):
                        continue
                    split_axis = int(self.op_by_id[f"fx::{base.name}"].metadata.get("dim", 0))
                    split_axis = _normalize_axis(split_axis, len(output_rows[output_index][0]))
                    if split_axis != source.logical_axis:
                        result.append(source)
                        continue
                    offset = sum(int(row[0][split_axis]) for row in output_rows[:output_index])
                    width = int(output_rows[output_index][0][split_axis])
                    result.append(
                        _SourceRef(
                            module_path=source.module_path,
                            channel_count=width,
                            logical_axis=split_axis,
                            offset=source.offset + offset,
                            index_scale=source.index_scale,
                            via=tuple(sorted(set(source.via + ("split",)))),
                            operation_id=f"fx::{base.name}",
                        )
                    )
                return result
            return self.sources(base)
        if op_type == "Reduction":
            if not inputs:
                return []
            sources = self.sources(inputs[0])
            dim = self.op_by_id[f"fx::{node.name}"].metadata.get("dim")
            if dim is None:
                return []
            input_shape = _shape(inputs[0])
            normalized = _normalize_axis(int(dim), len(input_shape or ()))
            return [source for source in sources if source.logical_axis != normalized]
        if op_type in {"Mul", "Activation", "Interpolate", "BEVWarp", "Pooling", "Index"}:
            flat = [source for child in inputs for source in self.sources(child)]
            # Multiplicative score tensors may have C=1. Preserve only feature
            # sources matching the output channel count.
            output = _shape(node)
            output_channels = int(output[1]) if output and len(output) > 1 else None
            selected = [
                source
                for source in flat
                if output_channels is None or source.channel_count * source.index_scale == output_channels
            ]
            return selected or flat
        if op_type in {"Output", "Input", "Attribute"}:
            return [source for child in inputs for source in self.sources(child)]
        # Shape-preserving unknown ops are recorded by the static builder. They
        # may be inspected, but are not allowed to silently carry a dependency.
        return []

    def build_edges(self) -> list[DependencyEdge]:
        edges: list[DependencyEdge] = []
        for node in self.graph.nodes:
            op_type = self.op_type(node)
            op_id = f"fx::{node.name}"
            if op_type == "Add" and self.config.couple_residual_add_branches:
                branches = [self.sources(child) for child in _node_inputs(node)]
                representatives = [branch[0] for branch in branches if branch]
                if len(representatives) >= 2:
                    base = representatives[0]
                    for other in representatives[1:]:
                        if base.channel_count != other.channel_count:
                            raise AmbiguousDependencyError(
                                f"residual Add {op_id} has incompatible channels "
                                f"{base.channel_count} and {other.channel_count}"
                            )
                        edges.append(
                            DependencyEdge(
                                source=base.module_path,
                                target=other.module_path,
                                source_axis="out",
                                target_axis="out",
                                dependency_type="residual_add",
                                channel_count=base.channel_count,
                                operation_id=op_id,
                            )
                        )
            if op_type == "Concat" and self.config.propagate_concat_offsets:
                op = self.op_by_id[op_id]
                dim = int(op.metadata.get("dim", 0))
                offset = 0
                for branch in _cat_inputs(node):
                    branch_shape = _shape(branch)
                    if branch_shape:
                        dim_norm = _normalize_axis(dim, len(branch_shape))
                        count = int(branch_shape[dim_norm])
                    else:
                        dim_norm, count = dim, 0
                    source_name = str(branch.target) if branch.op == "call_module" else f"fx::{branch.name}"
                    if dim_norm == 1:
                        edges.append(
                            DependencyEdge(
                                source=source_name,
                                target=op_id,
                                source_axis="out",
                                target_axis="channel",
                                dependency_type="concat",
                                channel_offset=offset,
                                channel_count=count,
                                operation_id=op_id,
                            )
                        )
                    offset += count

            if node.op != "call_module":
                continue
            module = self.modules[str(node.target)]
            inputs = _node_inputs(node)
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                sources = self.sources(inputs[0]) if inputs else []
                for source in sources:
                    edges.append(
                        DependencyEdge(
                            source=source.module_path,
                            target=str(node.target),
                            source_axis="out",
                            target_axis="channel",
                            dependency_type="conv_bn",
                            channel_offset=source.offset,
                            channel_count=source.channel_count,
                            operation_id=op_id,
                            metadata={"index_scale": source.index_scale},
                        )
                    )
                continue
            if not _is_weighted(module) or not self.config.propagate_dependency_inputs:
                continue
            sources = self.sources(inputs[0]) if inputs else []
            input_shape = _shape(inputs[0]) if inputs else None
            expected_axis = _module_input_axis(module, input_shape)
            for source in sources:
                if source.logical_axis != expected_axis:
                    self._issue(node, "producer_channel_axis_does_not_match_weighted_input", unsupported=True)
                    continue
                if source.module_path == str(node.target):
                    continue
                if isinstance(module, nn.Conv2d) and int(module.groups) > 1:
                    dep_type = "grouped_conv_input"
                elif isinstance(module, nn.ConvTranspose2d):
                    dep_type = "convtranspose_input"
                else:
                    dep_type = "concat_downstream_input" if "concat" in source.via else "downstream_input"
                edges.append(
                    DependencyEdge(
                        source=source.module_path,
                        target=str(node.target),
                        source_axis="out",
                        target_axis="in",
                        dependency_type=dep_type,
                        channel_offset=source.offset,
                        channel_count=source.channel_count,
                        operation_id=source.operation_id or op_id,
                        metadata={
                            "index_scale": source.index_scale,
                            "via": list(source.via),
                        },
                    )
                )
                if self.config.treat_depthwise_as_channel_passthrough and _is_depthwise(module):
                    edges.append(
                        DependencyEdge(
                            source=source.module_path,
                            target=str(node.target),
                            source_axis="out",
                            target_axis="out",
                            dependency_type="depthwise_coupling",
                            channel_count=source.channel_count,
                            operation_id=op_id,
                        )
                    )
        return _dedupe_edges(edges)


def _dedupe_edges(edges: Sequence[DependencyEdge]) -> list[DependencyEdge]:
    rows: list[DependencyEdge] = []
    seen: set[tuple[Any, ...]] = set()
    for edge in edges:
        key = (
            edge.source,
            edge.target,
            edge.source_axis,
            edge.target_axis,
            edge.dependency_type,
            edge.channel_offset,
            edge.channel_count,
            edge.operation_id,
            int(edge.metadata.get("index_scale", 1)),
        )
        if key not in seen:
            seen.add(key)
            rows.append(edge)
    return sorted(
        rows,
        key=lambda edge: (
            edge.operation_id,
            edge.dependency_type,
            edge.channel_offset,
            edge.source,
            edge.target,
        ),
    )


def build_channel_mappings(
    built: StaticGraphBuild,
    config: DependencyConfig | None = None,
) -> ChannelMappingResult:
    """Build channel dependencies and report every unproven mapping."""

    mapper = _Mapper(built, config or DependencyConfig())
    edges = mapper.build_edges()
    return ChannelMappingResult(
        edges=edges,
        unresolved_operations=mapper.unresolved,
        unsupported_operations=mapper.unsupported,
    )

