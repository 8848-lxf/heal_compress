"""Reverse trace Q/DQ-wrapped weights to their original initializer."""

from __future__ import annotations

from pathlib import Path

from ..exceptions import QDQValidationError
from ..types import QDQRootTrace
from ..export.origin_trace import build_weight_trace_index, trace_compute_node_weight


def trace_qdq_root_initializer(onnx_path: str | Path, canonical_node_name: str) -> QDQRootTrace:
    """Trace a canonical compute node through Q/DQ and legal passthrough ops."""

    index = build_weight_trace_index(onnx_path)
    matches = [node for node in index["nodes"] if str(node.name) == str(canonical_node_name)]
    if len(matches) != 1:
        raise QDQValidationError(f"canonical compute node must resolve exactly once: {canonical_node_name}")
    trace = trace_compute_node_weight(index, matches[0])
    if not trace.get("success"):
        raise QDQValidationError(
            f"Q/DQ root initializer unresolved for {canonical_node_name}: {trace.get('unresolved_tensor', '')}"
        )
    return QDQRootTrace(
        compute_node_name=str(matches[0].name),
        compute_op_type=str(matches[0].op_type),
        consumed_weight_tensor=str(trace.get("consumed_weight_tensor", "")),
        root_initializer=str(trace.get("root_initializer", "")),
        root_initializer_shape=tuple(trace.get("root_initializer_shape", ())),
        trace_chain=list(trace.get("trace_chain", [])),
        transpose_permutations=[tuple(value) for value in trace.get("transpose_permutations", [])],
        trans_b=int(trace.get("trans_b", 0) or 0),
    )
