"""Resolve the tensor that owns activation-output quantization semantics."""

from __future__ import annotations

from typing import Any


def activation_output_boundary_stops_before_merge(
    *,
    merge_policy: str,
    realized_output_precision: str,
) -> bool:
    """Return whether one weighted output must own scale before a merge.

    Adaptive runtime merges quantize an INT8 producer before the merge and
    derive the merge precision from all incoming branches.  Calibration and
    Q/DQ insertion must therefore stop at the same raw weighted boundary.
    """

    return (
        str(merge_policy).lower() == "adaptive_upcast_merge"
        and str(realized_output_precision).lower() == "int8"
    )


def resolve_activation_output_boundary(
    model: Any,
    weighted_node_name: str,
    *,
    stop_before_merge: bool = False,
) -> dict[str, Any]:
    """Return the stable post-op boundary for one weighted ONNX node.

    A weighted op followed by a unique pre-activation chain (for example
    MatMul -> Transpose -> BatchNormalization -> Transpose) and Relu owns its
    output scale at the Relu output, not at the raw weighted output. Residual
    Add/Concat paths are also described here, although the FP16 merge contract
    normally disables producer output Q/DQ and lets the downstream weighted
    input requantize. Ambiguous fan-out fails closed to the raw weighted
    output.
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
    boundary_node = weighted
    boundary_tensor = weighted_output
    following_ops: list[dict[str, Any]] = []
    resolution = "raw_weighted_output_no_unique_semantic_successor"

    # These operations form part of an exported/fusible weighted activation
    # path.  They are traversed only when every step has exactly one consumer
    # and one output, so the resolver never guesses across fan-out.
    pre_activation_ops = {
        "BatchNormalization",
        "Flatten",
        "Identity",
        "Reshape",
        "Squeeze",
        "Transpose",
        "Unsqueeze",
    }
    current_tensor = weighted_output
    while True:
        direct = consumers.get(current_tensor, [])
        if (
            len(direct) != 1
            or str(direct[0].op_type) not in pre_activation_ops
            or len(direct[0].output) != 1
        ):
            break
        passthrough = direct[0]
        next_tensor = str(passthrough.output[0])
        following_ops.append(
            {
                "name": str(passthrough.name),
                "op_type": str(passthrough.op_type),
                "input_tensor": current_tensor,
                "output_tensor": next_tensor,
            }
        )
        current_tensor = next_tensor

    direct = consumers.get(current_tensor, [])
    if len(direct) == 1 and str(direct[0].op_type) == "Relu" and len(direct[0].output) == 1:
        boundary_node = direct[0]
        boundary_tensor = str(boundary_node.output[0])
        following_ops.append(
            {
                "name": str(boundary_node.name),
                "op_type": "Relu",
                "input_tensor": current_tensor,
                "output_tensor": boundary_tensor,
            }
        )
        resolution = (
            "post_relu_semantic_boundary"
            if len(following_ops) == 1
            else "post_relu_semantic_boundary_via_unique_pre_activation_chain"
        )
    elif (
        not stop_before_merge
        and len(direct) == 1
        and str(direct[0].op_type) in {"Add", "Concat", "Mul", "Where", "MatMul"}
        and len(direct[0].output) == 1
    ):
        merge = direct[0]
        merge_output = str(merge.output[0])
        following_ops.append(
            {
                "name": str(merge.name),
                "op_type": str(merge.op_type),
                "input_tensor": current_tensor,
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


def resolve_activation_output_boundary_for_precision(
    model: Any,
    weighted_node_name: str,
    *,
    merge_policy: str,
    realized_output_precision: str,
) -> dict[str, Any]:
    """Resolve a boundary using the realized deployment precision contract."""

    stop_before_merge = activation_output_boundary_stops_before_merge(
        merge_policy=merge_policy,
        realized_output_precision=realized_output_precision,
    )
    boundary = resolve_activation_output_boundary(
        model,
        weighted_node_name,
        stop_before_merge=stop_before_merge,
    )
    return {
        **boundary,
        "merge_policy": str(merge_policy),
        "realized_output_precision": str(realized_output_precision).lower(),
        "stop_before_merge": bool(stop_before_merge),
    }
