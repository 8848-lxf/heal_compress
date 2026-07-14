"""Resolve the tensor that owns activation-output quantization semantics."""

from __future__ import annotations

from typing import Any


def resolve_activation_output_boundary(model: Any, weighted_node_name: str) -> dict[str, Any]:
    """Return the stable post-op boundary for one weighted ONNX node.

    A direct Conv/Gemm/MatMul -> Relu path owns its output scale at the Relu
    output, not at the raw weighted output. Residual Add/Concat paths are also
    described here, although the FP16 merge contract normally disables the
    producer output Q/DQ and lets the downstream weighted input requantize.
    Ambiguous fan-out fails closed to the raw weighted output.
    """

    nodes = {str(node.name): node for node in model.graph.node}
    consumers: dict[str, list[Any]] = {}
    for node in model.graph.node:
        for input_name in node.input:
            consumers.setdefault(str(input_name), []).append(node)
    weighted = nodes.get(str(weighted_node_name))
    if weighted is None or len(weighted.output) != 1:
        raise RuntimeError(f"activation_boundary_weighted_node_missing_or_multi_output:{weighted_node_name}")
    weighted_output = str(weighted.output[0])
    direct = consumers.get(weighted_output, [])
    boundary_node = weighted
    boundary_tensor = weighted_output
    following_ops: list[dict[str, Any]] = []
    resolution = "raw_weighted_output_no_unique_semantic_successor"

    if len(direct) == 1 and str(direct[0].op_type) == "Relu" and len(direct[0].output) == 1:
        boundary_node = direct[0]
        boundary_tensor = str(boundary_node.output[0])
        following_ops.append(
            {
                "name": str(boundary_node.name),
                "op_type": "Relu",
                "input_tensor": weighted_output,
                "output_tensor": boundary_tensor,
            }
        )
        resolution = "post_relu_semantic_boundary"
    elif len(direct) == 1 and str(direct[0].op_type) in {"Add", "Concat"} and len(direct[0].output) == 1:
        merge = direct[0]
        merge_output = str(merge.output[0])
        following_ops.append(
            {
                "name": str(merge.name),
                "op_type": str(merge.op_type),
                "input_tensor": weighted_output,
                "output_tensor": merge_output,
            }
        )
        merge_consumers = consumers.get(merge_output, [])
        if len(merge_consumers) == 1 and str(merge_consumers[0].op_type) == "Relu" and len(merge_consumers[0].output) == 1:
            boundary_node = merge_consumers[0]
            boundary_tensor = str(boundary_node.output[0])
            following_ops.append(
                {
                    "name": str(boundary_node.name),
                    "op_type": "Relu",
                    "input_tensor": merge_output,
                    "output_tensor": boundary_tensor,
                }
            )
            resolution = f"post_{str(merge.op_type).lower()}_relu_semantic_boundary"
        else:
            boundary_node = merge
            boundary_tensor = merge_output
            resolution = f"post_{str(merge.op_type).lower()}_semantic_boundary"

    return {
        "weighted_node_name": str(weighted_node_name),
        "weighted_output_tensor": weighted_output,
        "boundary_node_name": str(boundary_node.name),
        "boundary_op_type": str(boundary_node.op_type),
        "boundary_output_tensor": boundary_tensor,
        "following_ops": following_ops,
        "resolution": resolution,
        "moved_after_weighted_node": str(boundary_node.name) != str(weighted_node_name),
    }
