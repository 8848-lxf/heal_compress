"""Formal dependency graph builder."""

from __future__ import annotations

import operator
from typing import Any

import torch
import torch.nn as nn
from torch.fx import Node, symbolic_trace

from heal_compress.tracer.graph_schema import DependencyGraph, GraphNode


def _target_name(target: Any) -> str:
    if target is operator.add:
        return "add"
    if target is torch.add:
        return "add"
    if target is torch.cat:
        return "cat"
    return getattr(target, "__name__", str(target))


def _op_type(node: Node) -> str:
    if node.op == "call_module":
        return "module"
    if node.op == "call_function":
        name = _target_name(node.target)
        if name in {"add"}:
            return "elementwise_add"
        if name in {"cat", "concat"}:
            return "concat"
        return f"function:{name}"
    if node.op == "call_method":
        if str(node.target) in {"add", "__add__"}:
            return "elementwise_add"
        if str(node.target) in {"cat", "concat"}:
            return "concat"
        return f"method:{node.target}"
    return str(node.op)


def _input_names(node: Node) -> list[str]:
    names: list[str] = []

    def visit(value: Any) -> None:
        if isinstance(value, Node):
            names.append(str(value.name))
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)

    visit(node.args)
    visit(node.kwargs)
    return names


def build_dependency_graph(model: nn.Module, sample_batch: Any | None = None) -> dict[str, Any]:
    """Build a static module/function dependency graph.

    ``sample_batch`` is accepted for API compatibility. The current
    implementation uses ``torch.fx.symbolic_trace`` and therefore does not run
    the model.
    """

    modules = dict(model.named_modules())
    try:
        gm = symbolic_trace(model)
        fx_nodes = list(gm.graph.nodes)
    except Exception:  # noqa: BLE001
        nodes = [
            GraphNode(
                node_id=name,
                name=name,
                op_type=module.__class__.__name__,
                target=name,
                module_name=name,
                inputs=[],
            )
            for name, module in modules.items()
            if name
        ]
        graph = DependencyGraph(nodes=nodes, edges=[], modules=[name for name in modules if name])
        return graph.to_dict()

    nodes: list[GraphNode] = []
    edges: list[dict[str, str]] = []
    for node in fx_nodes:
        if node.op in {"placeholder", "output"}:
            continue
        module_name = str(node.target) if node.op == "call_module" else ""
        op_type = _op_type(node)
        if module_name and module_name in modules:
            op_type = modules[module_name].__class__.__name__
        inputs = _input_names(node)
        nodes.append(
            GraphNode(
                node_id=str(node.name),
                name=str(node.name),
                op_type=op_type,
                target=_target_name(node.target),
                module_name=module_name,
                inputs=inputs,
            )
        )
        for src in inputs:
            edges.append({"from": src, "to": str(node.name), "reason": op_type})
    graph = DependencyGraph(nodes=nodes, edges=edges, modules=[name for name in modules if name])
    return graph.to_dict()
