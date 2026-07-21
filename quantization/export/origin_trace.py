"""Trace an ONNX compute node's weight input to a real initializer."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


PASSTHROUGH_WEIGHT_OPS = frozenset(
    {
        "QuantizeLinear",
        "DequantizeLinear",
        # ModelOpt 0.29 emits these TensorRT-domain operators for explicit
        # E4M3 graphs.  For weight provenance they have the same root-preserving
        # semantics as standard ONNX Q/DQ and must not turn real Linear weights
        # into apparent functional MatMul nodes.
        "TRT_FP8QuantizeLinear",
        "TRT_FP8DequantizeLinear",
        "Cast",
        "Identity",
        "Transpose",
        "Reshape",
        "Squeeze",
        "Unsqueeze",
    }
)


def onnx_attribute(node: Any, name: str, default: Any = None) -> Any:
    """Read a scalar or integer-list ONNX node attribute."""

    for attribute in node.attribute:
        if attribute.name != name:
            continue
        if attribute.ints:
            return [int(value) for value in attribute.ints]
        if attribute.s:
            return attribute.s.decode("utf-8", errors="replace")
        return int(attribute.i)
    return default


def build_weight_trace_index(model_or_path: Any) -> dict[str, Any]:
    """Build immutable lookup tables for repeated Q/DQ root traces."""

    import onnx

    model = onnx.load(str(model_or_path)) if isinstance(model_or_path, (str, Path)) else model_or_path
    nodes = list(model.graph.node)
    return {
        "model": model,
        "nodes": nodes,
        "nodes_by_name": {str(node.name): node for node in nodes},
        "initializers": {str(item.name): tuple(int(dim) for dim in item.dims) for item in model.graph.initializer},
        "producers": {str(output): node for node in nodes for output in node.output},
    }


def trace_weight_tensor(index: Mapping[str, Any], tensor_name: str) -> dict[str, Any]:
    """Trace a tensor backwards through allowed wrappers to an initializer."""

    initializers = index.get("initializers", {})
    producers = index.get("producers", {})
    current = str(tensor_name)
    chain: list[dict[str, Any]] = []
    transpose_permutations: list[tuple[int, ...]] = []
    visited: set[str] = set()
    while current and current not in visited:
        visited.add(current)
        if current in initializers:
            return {
                "success": True,
                "root_initializer": current,
                "root_initializer_shape": tuple(initializers[current]),
                "trace_chain": chain,
                "transpose_permutations": transpose_permutations,
            }
        producer = producers.get(current)
        if producer is None:
            break
        chain.append(
            {
                "node_name": str(producer.name),
                "op_type": str(producer.op_type),
                "output_tensor": current,
                "inputs": [str(value) for value in producer.input],
            }
        )
        if str(producer.op_type) == "Transpose":
            transpose_permutations.append(tuple(onnx_attribute(producer, "perm", []) or []))
        if str(producer.op_type) not in PASSTHROUGH_WEIGHT_OPS or not producer.input:
            break
        current = str(producer.input[0])
    return {
        "success": False,
        "root_initializer": "",
        "root_initializer_shape": (),
        "trace_chain": chain,
        "transpose_permutations": transpose_permutations,
        "unresolved_tensor": current,
    }


def trace_compute_node_weight(index: Mapping[str, Any], node: Any) -> dict[str, Any]:
    """Trace input 1 of Conv/ConvTranspose/Gemm/weighted MatMul."""

    if len(node.input) < 2:
        return {"success": False, "failure_reason": "weight_input_missing"}
    result = trace_weight_tensor(index, str(node.input[1]))
    result.update(
        {
            "compute_node_name": str(node.name),
            "compute_op_type": str(node.op_type),
            "consumed_weight_tensor": str(node.input[1]),
            "trans_b": int(onnx_attribute(node, "transB", 0) or 0),
        }
    )
    return result
