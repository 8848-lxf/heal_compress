"""Formal tracing API with explicit legacy compatibility exports."""

from __future__ import annotations

from importlib import import_module
from typing import Any
import warnings

__all__ = [
    # Canonical formal API.
    "trace_model",
    "build_dependency_graph",
    "build_dependency_scopes",
    "build_coupled_channel_units",
    "build_atomic_prune_units",
    "serialize_trace_result",
    "load_trace_result",
    "TraceResult",
    "DependencyScope",
    "PruningGroup",
    "CoupledChannelUnit",
    "AtomicPruneUnit",
    "ConcreteCoupledPruningGroup",
    # Historical helpers retained under non-canonical names/classes.
    "HealForwardWrapper",
    "DependencyGraphBuilder",
    "CoupledChannelGroupBuilder",
    "GenericTracer",
    "legacy_trace_model",
    "OpGraph",
    "OpNode",
    "build_op_graph",
    "GroupItem",
    "LegacyPruningGroup",
    "offset_transform",
    "identity_transform",
    "TransformerAnalyzer",
    "TransformerGroupBuilder",
    "build_transformer_pruning_groups",
    "build_precision_coupling_groups",
    "build_runtime_precision_coupling",
    "PrecisionGroup",
    "PrecisionRelation",
    "RuntimePrecisionCouplingResult",
]

from .api import (
    build_atomic_prune_units,
    build_coupled_channel_units,
    build_dependency_graph,
    build_dependency_scopes,
    load_trace_result,
    serialize_trace_result,
    trace_model,
)
from .types import (
    AtomicPruneUnit,
    ConcreteCoupledPruningGroup,
    CoupledChannelUnit,
    DependencyScope,
    PruningGroup,
    TraceResult,
)

_EXPORTS = {
    "HealForwardWrapper": ".forward_wrapper",
    "DependencyGraphBuilder": ".dependency_graph",
    "CoupledChannelGroupBuilder": ".coupled_channel_group",
    "GenericTracer": ".generic_tracer",
    "legacy_trace_model": ".generic_tracer",
    "OpGraph": ".op_graph",
    "OpNode": ".op_graph",
    "build_op_graph": ".op_graph",
    "GroupItem": ".pruning_group",
    "LegacyPruningGroup": ".pruning_group",
    "offset_transform": ".pruning_group",
    "identity_transform": ".pruning_group",
    "TransformerAnalyzer": ".transformer_analyzer",
    "TransformerGroupBuilder": ".transformer_groups",
    "build_transformer_pruning_groups": ".transformer_groups",
    "build_precision_coupling_groups": ".precision_coupling_tracer",
    "build_runtime_precision_coupling": ".precision_coupling_tracer",
    "PrecisionGroup": ".precision_coupling_tracer",
    "PrecisionRelation": ".precision_coupling_tracer",
    "RuntimePrecisionCouplingResult": ".precision_coupling_tracer",
}

_EXPORT_ATTRIBUTES = {
    "legacy_trace_model": "trace_model",
    "LegacyPruningGroup": "PruningGroup",
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if not module_name:
        raise AttributeError(name)
    if name in _EXPORT_ATTRIBUTES:
        warnings.warn(
            f"tracer.{name} is deprecated; use the typed formal tracer API",
            DeprecationWarning,
            stacklevel=2,
        )
    module = import_module(module_name, __name__)
    return getattr(module, _EXPORT_ATTRIBUTES.get(name, name))
