"""Formal tracing tools and compatibility exports."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "HealForwardWrapper",
    "DependencyGraphBuilder",
    "CoupledChannelGroupBuilder",
    "GenericTracer",
    "trace_model",
    "OpGraph",
    "OpNode",
    "build_op_graph",
    "GroupItem",
    "PruningGroup",
    "offset_transform",
    "identity_transform",
    "TransformerAnalyzer",
    "TransformerGroupBuilder",
    "build_transformer_pruning_groups",
]

_EXPORTS = {
    "HealForwardWrapper": ".forward_wrapper",
    "DependencyGraphBuilder": ".dependency_graph",
    "CoupledChannelGroupBuilder": ".coupled_channel_group",
    "GenericTracer": ".generic_tracer",
    "trace_model": ".generic_tracer",
    "OpGraph": ".op_graph",
    "OpNode": ".op_graph",
    "build_op_graph": ".op_graph",
    "GroupItem": ".pruning_group",
    "PruningGroup": ".pruning_group",
    "offset_transform": ".pruning_group",
    "identity_transform": ".pruning_group",
    "TransformerAnalyzer": ".transformer_analyzer",
    "TransformerGroupBuilder": ".transformer_groups",
    "build_transformer_pruning_groups": ".transformer_groups",
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if not module_name:
        raise AttributeError(name)
    module = import_module(module_name, __name__)
    return getattr(module, name)
