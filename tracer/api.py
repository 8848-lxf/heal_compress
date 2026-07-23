"""Public, typed API for formal model and channel-dependency tracing."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Callable

import torch.nn as nn

from .atomic_units import build_atomic_prune_units_from_coupled
from .channel_mapping import build_channel_mappings
from .config import TraceConfig
from .coupled_units import build_coupled_channel_units_from_scopes
from .dependency_graph import build_dependency_scopes_from_edges
from .exceptions import TraceError, UnsupportedOperationError
from .hashing import stable_hash
from .module_call_tracer import build_example_input_contract, trace_module_calls
from .protection import build_protection_policies
from .serialization import load_trace_result, serialize_trace_result
from .static_graph_builder import trace_static_graph
from .types import (
    AtomicPruneUnit,
    CoupledChannelUnit,
    DependencyGraphResult,
    DependencyScope,
    ProtectionPolicy,
    TraceCoverage,
    TraceResult,
)


def _config(value: TraceConfig | Mapping[str, Any] | None) -> TraceConfig:
    if value is None:
        return TraceConfig()
    if isinstance(value, TraceConfig):
        return value
    return TraceConfig.from_dict(value)


def _dedupe_issues(rows: Sequence[Any]) -> list[Any]:
    result: list[Any] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (str(row.operation_id), str(row.reason))
        if key not in seen:
            seen.add(key)
            result.append(row)
    return sorted(result, key=lambda row: (row.operation_id, row.reason))


def build_dependency_graph(
    model: nn.Module,
    example_inputs: Any,
    *,
    config: TraceConfig | Mapping[str, Any] | None = None,
    forward_fn: Callable[[nn.Module, Any], Any] | None = None,
) -> DependencyGraphResult:
    """Build a typed FX graph and proven channel dependency mappings.

    Any unresolved operation that changes the logical channel layout raises
    :class:`UnsupportedOperationError`; it is never treated as a transparent
    pass-through.
    """

    cfg = _config(config)
    try:
        built = trace_static_graph(model, example_inputs, config=cfg)
    except TraceError:
        if cfg.fail_on_fx_trace_error:
            raise
        from .runtime_graph_builder import build_runtime_dependency_graph

        graph, _scopes = build_runtime_dependency_graph(
            model,
            example_inputs,
            config=cfg,
            forward_fn=forward_fn,
        )
        setattr(graph, "_runtime_scopes", _scopes)
        setattr(graph, "_realized_backend", "runtime_tensor_flow")
        return graph
    mappings = build_channel_mappings(built, cfg.dependency)
    unresolved = _dedupe_issues(
        [*built.unresolved_operations, *mappings.unresolved_operations]
    )
    unsupported = _dedupe_issues(
        [*built.unsupported_operations, *mappings.unsupported_operations]
    )
    if unsupported:
        details = "; ".join(
            f"{row.operation_id}:{row.op_type}:{row.reason}" for row in unsupported[:8]
        )
        raise UnsupportedOperationError(
            f"unresolved channel-changing operation(s); trace rejected: {details}"
        )
    return DependencyGraphResult(
        graph_schema_version=cfg.graph_schema_version,
        module_inventory=built.module_inventory,
        op_inventory=built.op_inventory,
        tensor_inventory=built.tensor_inventory,
        dependency_edges=mappings.edges,
        unresolved_operations=unresolved,
        unsupported_operations=unsupported,
    )


def build_dependency_scopes(
    graph: DependencyGraphResult,
    *,
    protection_policies: Sequence[ProtectionPolicy] | None = None,
    config: TraceConfig | Mapping[str, Any] | None = None,
) -> list[DependencyScope]:
    """Build full residual/concat/downstream channel dependency closures."""

    cfg = _config(config)
    policies = list(protection_policies or build_protection_policies(
        graph.module_inventory, cfg.protection
    ))
    return build_dependency_scopes_from_edges(
        graph.module_inventory,
        graph.dependency_edges,
        policies,
    )


def build_coupled_channel_units(
    scopes: Sequence[DependencyScope],
    *,
    module_inventory: Sequence[Any] = (),
) -> list[CoupledChannelUnit]:
    """Expand scopes into deterministic module-axis-index membership units."""

    return build_coupled_channel_units_from_scopes(scopes, module_inventory)


def build_atomic_prune_units(
    coupled_units: Sequence[CoupledChannelUnit],
) -> list[AtomicPruneUnit]:
    """Create trace-time atoms without imposing shared grouped local indices."""

    return build_atomic_prune_units_from_coupled(coupled_units)


def trace_model(
    model: nn.Module,
    example_inputs: Any,
    *,
    config: TraceConfig | Mapping[str, Any] | None = None,
    forward_fn: Callable[[nn.Module, Any], Any] | None = None,
) -> TraceResult:
    """Trace ``model`` and return a complete typed :class:`TraceResult`.

    The representative forward records repeated module calls and tensor shape
    contracts. Torch-FX independently supplies operation provenance and shape
    propagation for dependency analysis. No model parameter or global state is
    modified.
    """

    cfg = _config(config)
    call_trace = trace_module_calls(
        model,
        example_inputs,
        call_style=cfg.input_call_style,
        forward_fn=forward_fn,
        include_non_weighted_leaf_calls=cfg.record_non_weighted_leaf_calls,
    )
    realized_backend = "torch_fx"
    runtime_scopes: list[DependencyScope] | None = None
    try:
        graph = build_dependency_graph(
            model,
            example_inputs,
            config=cfg,
            forward_fn=forward_fn,
        )
        runtime_scopes = getattr(graph, "_runtime_scopes", None)
        realized_backend = str(getattr(graph, "_realized_backend", "torch_fx"))
    except TraceError:
        if cfg.fail_on_fx_trace_error:
            raise
        from .runtime_graph_builder import build_runtime_dependency_graph

        graph, runtime_scopes = build_runtime_dependency_graph(
            model,
            example_inputs,
            config=cfg,
            forward_fn=forward_fn,
        )
        realized_backend = "runtime_tensor_flow"
    policies = build_protection_policies(graph.module_inventory, cfg.protection)
    scopes = runtime_scopes or build_dependency_scopes(
        graph,
        protection_policies=policies,
        config=cfg,
    )
    coupled_units = build_coupled_channel_units(
        scopes,
        module_inventory=graph.module_inventory,
    )
    atomic_units = build_atomic_prune_units(coupled_units)
    protected_modules = sorted(
        policy.module_path for policy in policies if policy.fixed_output_contract
    )
    protected_units = sorted(
        unit.stable_id for unit in coupled_units if unit.protected
    )
    coverage = (
        float(call_trace.weighted_modules_called) / float(call_trace.weighted_modules_total)
        if call_trace.weighted_modules_total
        else 1.0
    )
    trace_coverage = TraceCoverage(
        weighted_modules_total=call_trace.weighted_modules_total,
        weighted_modules_called=call_trace.weighted_modules_called,
        weighted_module_coverage=coverage,
        traced_module_count=len(graph.module_inventory),
        traced_operation_count=len(graph.op_inventory),
        unresolved_operation_count=len(graph.unresolved_operations),
        unsupported_operation_count=len(graph.unsupported_operations),
        representative_forward_executed=True,
        dynamic_branch_enumeration_enabled=False,
        coverage_notes=[
            "One representative forward is recorded per trace_model call.",
            "Unexecuted dynamic branches are not claimed as covered.",
        ],
    )
    result = TraceResult(
        graph_schema_version=cfg.graph_schema_version,
        module_inventory=graph.module_inventory,
        op_inventory=graph.op_inventory,
        tensor_inventory=graph.tensor_inventory,
        dependency_edges=graph.dependency_edges,
        dependency_scopes=scopes,
        coupled_channel_units=coupled_units,
        atomic_prune_units=atomic_units,
        protected_modules=protected_modules,
        protected_units=protected_units,
        protection_policies=policies,
        protection_reasons={
            policy.module_path: policy.protection_reason
            for policy in policies
            if policy.protection_reason
        },
        unresolved_operations=graph.unresolved_operations,
        unsupported_operations=graph.unsupported_operations,
        trace_coverage=trace_coverage,
        example_input_contract=build_example_input_contract(
            example_inputs, call_style=cfg.input_call_style
        ),
        module_call_trace=call_trace.records,
        config={**cfg.to_dict(), "realized_backend": realized_backend},
    )
    hash_payload = result.to_dict()
    hash_payload["trace_hash"] = ""
    result.trace_hash = stable_hash(hash_payload)
    return result


__all__ = [
    "trace_model",
    "build_dependency_graph",
    "build_dependency_scopes",
    "build_coupled_channel_units",
    "build_atomic_prune_units",
    "serialize_trace_result",
    "load_trace_result",
]
