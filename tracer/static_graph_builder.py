"""Torch-FX static graph construction for the formal tracer."""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import torch
import torch.nn as nn
from torch.fx import GraphModule, Node, symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp

from .config import TraceConfig, UnknownOperationPolicy
from .exceptions import TraceError
from .module_call_tracer import is_weighted_module
from .types import (
    ModuleInventoryEntry,
    OperationInventoryEntry,
    OperationIssue,
    TensorInventoryEntry,
)


@dataclass
class StaticGraphBuild:
    """Internal FX graph plus its serializable inventories."""

    graph_module: GraphModule
    module_inventory: list[ModuleInventoryEntry]
    op_inventory: list[OperationInventoryEntry]
    tensor_inventory: list[TensorInventoryEntry]
    unresolved_operations: list[OperationIssue]
    unsupported_operations: list[OperationIssue]


def _shape_dtype(meta: Any) -> list[tuple[tuple[int, ...], str]]:
    if hasattr(meta, "shape") and hasattr(meta, "dtype"):
        return [
            (
                tuple(int(dim) for dim in meta.shape),
                str(meta.dtype).replace("torch.", ""),
            )
        ]
    if isinstance(meta, (list, tuple)):
        rows: list[tuple[tuple[int, ...], str]] = []
        for item in meta:
            rows.extend(_shape_dtype(item))
        return rows
    return []


def node_tensor_metadata(node: Node) -> list[tuple[tuple[int, ...], str]]:
    """Extract tensor shape/dtype rows populated by ``ShapeProp``."""

    return _shape_dtype(node.meta.get("tensor_meta"))


def _target_name(target: Any) -> str:
    if isinstance(target, str):
        return target
    module = getattr(target, "__module__", "")
    qualname = getattr(target, "__qualname__", getattr(target, "__name__", ""))
    if qualname:
        return f"{module}.{qualname}" if module else str(qualname)
    return type(target).__name__


def _module_op_type(module: nn.Module) -> str:
    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        return "Conv"
    if isinstance(module, (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
        return "ConvTranspose2d" if isinstance(module, nn.ConvTranspose2d) else "ConvTranspose"
    if isinstance(module, nn.Linear):
        return "Linear"
    if isinstance(module, nn.modules.batchnorm._BatchNorm):
        return "BatchNorm"
    if isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
        return "Norm"
    if isinstance(
        module,
        (
            nn.ReLU,
            nn.ReLU6,
            nn.LeakyReLU,
            nn.GELU,
            nn.SiLU,
            nn.ELU,
            nn.Identity,
            nn.Dropout,
        ),
    ):
        return "Activation"
    if isinstance(module, (nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d, nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d)):
        return "Pooling"
    return module.__class__.__name__


def _function_op_type(node: Node) -> str:
    target = node.target
    name = _target_name(target).lower()
    leaf = name.rsplit(".", 1)[-1]
    if target in {operator.add, torch.add} or leaf in {"add", "__add__", "iadd"}:
        return "Add"
    if target in {operator.mul, torch.mul} or leaf in {"mul", "__mul__", "imul"}:
        return "Mul"
    if target is torch.cat or leaf in {"cat", "concat"}:
        return "Concat"
    if leaf in {"split", "chunk", "tensor_split"}:
        return "Split"
    if leaf in {"reshape", "view", "flatten", "squeeze", "unsqueeze", "contiguous"}:
        return "View"
    if leaf in {"permute", "transpose", "movedim", "swapaxes"}:
        return "Permute"
    if leaf in {"interpolate", "upsample", "grid_sample"}:
        return "Interpolate" if leaf != "grid_sample" else "BEVWarp"
    if leaf in {"relu", "sigmoid", "softmax", "gelu", "silu", "where", "clone"}:
        return "Activation"
    if leaf in {"sum", "mean"}:
        return "Reduction"
    if leaf in {"matmul", "bmm", "mm"}:
        return "FunctionalMatMul"
    if leaf in {"getitem", "getattr"}:
        return "Index"
    return "Unknown"


def _node_op_type(node: Node, modules: Mapping[str, nn.Module]) -> str:
    if node.op == "call_module":
        return _module_op_type(modules[str(node.target)])
    if node.op in {"call_function", "call_method"}:
        return _function_op_type(node)
    if node.op == "placeholder":
        return "Input"
    if node.op == "get_attr":
        return "Attribute"
    if node.op == "output":
        return "Output"
    return "Unknown"


def _iter_node_inputs(value: Any) -> Iterable[Node]:
    if isinstance(value, Node):
        yield value
    elif isinstance(value, Mapping):
        for child in value.values():
            yield from _iter_node_inputs(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _iter_node_inputs(child)


def _input_nodes(node: Node) -> list[Node]:
    seen: set[str] = set()
    rows: list[Node] = []
    for child in list(_iter_node_inputs(node.args)) + list(_iter_node_inputs(node.kwargs)):
        if child.name not in seen:
            seen.add(child.name)
            rows.append(child)
    return rows


def _input_shapes(node: Node) -> list[tuple[int, ...]]:
    return [shape for child in _input_nodes(node) for shape, _dtype in node_tensor_metadata(child)]


def _channel_size(shape: tuple[int, ...]) -> int | None:
    if not shape:
        return None
    if len(shape) == 1:
        return int(shape[0])
    return int(shape[1])


def _channel_changing(input_shapes: list[tuple[int, ...]], output_shapes: list[tuple[int, ...]]) -> bool:
    if not input_shapes or not output_shapes:
        return False
    source = input_shapes[0]
    target = output_shapes[0]
    if source == target:
        return False
    source_channels = _channel_size(source)
    target_channels = _channel_size(target)
    return len(source) != len(target) or source_channels != target_channels


def _scalar_metadata(node: Node, op_type: str) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    if op_type == "Concat":
        dim = node.kwargs.get("dim", node.args[1] if len(node.args) > 1 else 0)
        if isinstance(dim, int):
            meta["dim"] = int(dim)
    elif op_type == "Permute":
        if str(node.target) == "permute":
            dims = node.args[1:] if len(node.args) > 2 else (node.args[1] if len(node.args) > 1 else ())
            if isinstance(dims, (tuple, list)) and all(isinstance(value, int) for value in dims):
                meta["dims"] = [int(value) for value in dims]
            elif all(isinstance(value, int) for value in dims):
                meta["dims"] = [int(value) for value in dims]
        elif str(node.target) == "transpose":
            values = node.args[1:3]
            if len(values) == 2 and all(isinstance(value, int) for value in values):
                meta["dims"] = [int(values[0]), int(values[1])]
    elif op_type == "View":
        meta["method"] = str(node.target)
        for key in ("start_dim", "end_dim", "dim"):
            if key in node.kwargs and isinstance(node.kwargs[key], int):
                meta[key] = int(node.kwargs[key])
    elif op_type in {"Split", "Reduction"}:
        dim = node.kwargs.get("dim")
        if isinstance(dim, int):
            meta["dim"] = int(dim)
    return meta


def _shapeprop_args(example_inputs: Any, call_style: str) -> tuple[tuple[Any, ...], dict[str, Any]]:
    style = call_style
    if style == "auto":
        style = "args" if isinstance(example_inputs, tuple) else "single"
    if style == "single":
        return (example_inputs,), {}
    if style == "args":
        if not isinstance(example_inputs, (tuple, list)):
            raise TypeError("input_call_style='args' requires tuple/list inputs")
        return tuple(example_inputs), {}
    if style == "kwargs":
        if not isinstance(example_inputs, Mapping):
            raise TypeError("input_call_style='kwargs' requires mapping inputs")
        return (), dict(example_inputs)
    raise ValueError(f"unsupported input_call_style: {call_style}")


def _module_inventory(model: nn.Module) -> list[ModuleInventoryEntry]:
    rows: list[ModuleInventoryEntry] = []
    for name, module in model.named_modules():
        if not name:
            continue
        attrs: dict[str, int | None] = {}
        for attr in (
            "in_channels",
            "out_channels",
            "in_features",
            "out_features",
            "num_features",
            "groups",
        ):
            value = getattr(module, attr, None)
            attrs[attr] = int(value) if value is not None else None
        rows.append(
            ModuleInventoryEntry(
                module_path=name,
                module_type=module.__class__.__name__,
                weighted=is_weighted_module(module),
                parameter_shapes={
                    key: tuple(int(dim) for dim in value.shape)
                    for key, value in module.named_parameters(recurse=False)
                },
                buffer_shapes={
                    key: tuple(int(dim) for dim in value.shape)
                    for key, value in module.named_buffers(recurse=False)
                },
                **attrs,
            )
        )
    return rows


def trace_static_graph(
    model: nn.Module,
    example_inputs: Any,
    *,
    config: TraceConfig | None = None,
) -> StaticGraphBuild:
    """Build and shape-propagate an FX graph or raise a typed trace error."""

    cfg = config or TraceConfig()
    try:
        graph_module = symbolic_trace(model)
    except Exception as exc:  # noqa: BLE001 - converted to a stable public exception
        raise TraceError(f"torch.fx symbolic trace failed: {type(exc).__name__}: {exc}") from exc
    args, kwargs = _shapeprop_args(example_inputs, cfg.input_call_style)
    try:
        ShapeProp(graph_module).propagate(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001
        raise TraceError(f"torch.fx shape propagation failed: {type(exc).__name__}: {exc}") from exc

    modules = dict(model.named_modules())
    op_rows: list[OperationInventoryEntry] = []
    tensor_rows: list[TensorInventoryEntry] = []
    unresolved: list[OperationIssue] = []
    unsupported: list[OperationIssue] = []
    for node in graph_module.graph.nodes:
        op_type = _node_op_type(node, modules)
        output_meta = node_tensor_metadata(node)
        output_shapes = [shape for shape, _dtype in output_meta]
        input_shapes = _input_shapes(node)
        op_id = f"fx::{node.name}"
        module_path = str(node.target) if node.op == "call_module" else ""
        outputs = [f"{op_id}:out{index}" for index in range(len(output_meta))]
        op_rows.append(
            OperationInventoryEntry(
                op_id=op_id,
                op_kind=str(node.op),
                op_type=op_type,
                target=_target_name(node.target),
                module_path=module_path,
                input_ids=[f"fx::{child.name}" for child in _input_nodes(node)],
                output_ids=outputs,
                input_shapes=input_shapes,
                output_shapes=output_shapes,
                metadata=_scalar_metadata(node, op_type),
            )
        )
        for index, (shape, dtype) in enumerate(output_meta):
            tensor_rows.append(
                TensorInventoryEntry(
                    tensor_id=f"{op_id}:out{index}",
                    shape=shape,
                    dtype=dtype,
                    producer_op_id=op_id,
                    consumer_op_ids=sorted(f"fx::{user.name}" for user in node.users),
                    output_index=index,
                )
            )
        if op_type != "Unknown" or node.op in {"placeholder", "output", "get_attr"} or not output_shapes:
            continue
        changing = _channel_changing(input_shapes, output_shapes)
        issue = OperationIssue(
            operation_id=op_id,
            op_type=_target_name(node.target),
            reason="no_registered_channel_mapping",
            input_shapes=input_shapes,
            output_shapes=output_shapes,
            channel_changing=changing,
        )
        if changing:
            unsupported.append(issue)
        else:
            unresolved.append(issue)
            if cfg.dependency.unknown_operation_policy == UnknownOperationPolicy.FAIL_ALL:
                unsupported.append(issue)

    return StaticGraphBuild(
        graph_module=graph_module,
        module_inventory=_module_inventory(model),
        op_inventory=op_rows,
        tensor_inventory=tensor_rows,
        unresolved_operations=unresolved,
        unsupported_operations=unsupported,
    )


def build_static_graph(model: nn.Module, sample_batch: Any | None = None) -> dict[str, Any]:
    """Compatibility wrapper returning the historical graph-dict shape."""

    if sample_batch is None:
        raise ValueError("sample_batch is required for formal shape propagation")
    built = trace_static_graph(model, sample_batch)
    nodes = [
        {
            "node_id": row.op_id,
            "name": row.op_id.removeprefix("fx::"),
            "op_type": row.op_type,
            "target": row.target,
            "module_name": row.module_path,
            "inputs": [value.removeprefix("fx::") for value in row.input_ids],
        }
        for row in built.op_inventory
        if row.op_kind not in {"placeholder", "output"}
    ]
    edges = [
        {"from": source, "to": node["name"], "reason": node["op_type"]}
        for node in nodes
        for source in node["inputs"]
    ]
    return {
        "nodes": nodes,
        "edges": edges,
        "modules": [row.module_path for row in built.module_inventory],
        "summary": {
            "num_nodes": len(nodes),
            "num_edges": len(edges),
            "num_modules": len(built.module_inventory),
        },
    }
