"""Model-family-neutral Q/DQ adjacency repair for weighted projections."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def validate_projection_qdq_adjacency(
    model: Any, projection_node_names: Sequence[str]
) -> list[dict[str, Any]]:
    """Require activation and weight DQ to directly feed each projection."""

    nodes = {str(node.name): node for node in model.graph.node if str(node.name)}
    producers = {
        str(output): node for node in model.graph.node for output in node.output
    }
    records: list[dict[str, Any]] = []

    def dequantize_source(tensor_name: str, *, weight: bool) -> Any | None:
        producer = producers.get(str(tensor_name))
        if producer is None:
            return None
        if str(producer.op_type) in {"DequantizeLinear", "TRT_FP8DequantizeLinear"}:
            return producer
        if weight and str(producer.op_type) == "Transpose" and producer.input:
            candidate = producers.get(str(producer.input[0]))
            if candidate is not None and str(candidate.op_type) in {
                "DequantizeLinear",
                "TRT_FP8DequantizeLinear",
            }:
                return candidate
        return None

    for name in projection_node_names:
        node = nodes.get(str(name))
        if node is None:
            raise ValueError(f"projection_node_missing:{name}")
        if str(node.op_type) not in {"MatMul", "Gemm"} or len(node.input) < 2:
            raise ValueError(f"projection_node_unsupported:{name}:{node.op_type}")
        input_producers = [producers.get(str(node.input[index])) for index in (0, 1)]
        dequantize_nodes = [
            dequantize_source(str(node.input[0]), weight=False),
            dequantize_source(str(node.input[1]), weight=True),
        ]
        if any(value is None for value in dequantize_nodes):
            producer_types = [
                str(producer.op_type) if producer is not None else ""
                for producer in input_producers
            ]
            raise ValueError(f"projection_qdq_not_adjacent:{name}:{producer_types}")
        records.append(
            {
                "projection_node": str(name),
                "activation_dequantize_node": str(dequantize_nodes[0].name),
                "weight_dequantize_node": str(dequantize_nodes[1].name),
                "validated": True,
            }
        )
    return records


def restore_projection_qdq_adjacency(
    model: Any,
    projection_node_names: Sequence[str],
    *,
    output_cast_precisions: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Remove only typed Casts that separate ModelOpt DQ from projections."""

    from onnx import TensorProto, helper

    nodes = {str(node.name): node for node in model.graph.node if str(node.name)}
    producers = {
        str(output): node for node in model.graph.node for output in node.output
    }
    consumers: dict[str, list[Any]] = {}
    for candidate in model.graph.node:
        for tensor_name in candidate.input:
            consumers.setdefault(str(tensor_name), []).append(candidate)

    def cast_target(node: Any) -> int | None:
        if node is None or str(node.op_type) != "Cast":
            return None
        return next(
            (int(value.i) for value in node.attribute if str(value.name) == "to"),
            None,
        )

    def is_dq_path(tensor_name: str, *, weight: bool) -> bool:
        producer = producers.get(str(tensor_name))
        if producer is None:
            return False
        if str(producer.op_type) in {"DequantizeLinear", "TRT_FP8DequantizeLinear"}:
            return True
        if weight and str(producer.op_type) == "Transpose" and producer.input:
            source = producers.get(str(producer.input[0]))
            return source is not None and str(source.op_type) in {
                "DequantizeLinear",
                "TRT_FP8DequantizeLinear",
            }
        return False

    records: list[dict[str, Any]] = []
    removed_cast_outputs: set[str] = set()
    output_casts: dict[str, Any] = {}
    requested_output_casts = {
        str(name): str(precision).upper()
        for name, precision in (output_cast_precisions or {}).items()
    }

    def set_tensor_type(name: str, element_type: int) -> None:
        for value in [*model.graph.input, *model.graph.value_info, *model.graph.output]:
            if str(value.name) == str(name):
                value.type.tensor_type.elem_type = int(element_type)
                return
        model.graph.value_info.append(
            helper.make_tensor_value_info(str(name), int(element_type), None)
        )

    for name in projection_node_names:
        node = nodes.get(str(name))
        if node is None:
            raise ValueError(f"projection_node_missing:{name}")
        removed: list[str] = []
        for input_index in (0, 1):
            cast = producers.get(str(node.input[input_index]))
            if cast_target(cast) != int(TensorProto.FLOAT16) or not cast.input:
                raise ValueError(
                    f"projection_qdq_cast_pattern_missing:{name}:{input_index}"
                )
            source = str(cast.input[0])
            if not is_dq_path(source, weight=input_index == 1):
                raise ValueError(
                    f"projection_qdq_cast_source_invalid:{name}:{input_index}:{source}"
                )
            removed.append(str(cast.name))
            removed_cast_outputs.update(str(value) for value in cast.output)
            node.input[input_index] = source
        output_precision = requested_output_casts.get(str(name), "")
        output_cast_name = ""
        if output_precision:
            if output_precision not in {"FP16", "FP32"} or len(node.output) != 1:
                raise ValueError(
                    f"projection_output_cast_invalid:{name}:{output_precision}"
                )
            public = str(node.output[0])
            existing_casts = [
                candidate
                for candidate in consumers.get(public, ())
                if str(candidate.op_type) == "Cast"
            ]
            if existing_casts:
                raise ValueError(
                    "projection_output_cast_conflicts_with_existing_typed_boundary:"
                    f"{name}:{[str(value.name) for value in existing_casts]}"
                )
            raw = f"{public}__before_quantized_projection_output_cast"
            output_cast_name = f"{name}__output_{output_precision.lower()}"
            element_type = (
                int(TensorProto.FLOAT16)
                if output_precision == "FP16"
                else int(TensorProto.FLOAT)
            )
            node.output[0] = raw
            output_casts[str(name)] = helper.make_node(
                "Cast", [raw], [public], name=output_cast_name, to=element_type
            )
            set_tensor_type(raw, int(TensorProto.FLOAT))
            set_tensor_type(public, element_type)
        records.append(
            {
                "projection_node": str(name),
                "removed_cast_nodes": removed,
                "output_cast_node": output_cast_name,
                "output_cast_precision": output_precision,
                "validated": True,
            }
        )
    used = {
        str(value)
        for node in model.graph.node
        for value in node.input
        if str(value)
    } | {str(value.name) for value in model.graph.output}
    kept = []
    for node in model.graph.node:
        if (
            str(node.op_type) == "Cast"
            and set(str(value) for value in node.output) <= removed_cast_outputs
            and not any(str(value) in used for value in node.output)
        ):
            continue
        kept.append(node)
        output_cast = output_casts.get(str(node.name))
        if output_cast is not None:
            kept.append(output_cast)
    del model.graph.node[:]
    model.graph.node.extend(kept)
    validate_projection_qdq_adjacency(model, projection_node_names)
    return records


__all__ = [
    "restore_projection_qdq_adjacency",
    "validate_projection_qdq_adjacency",
]
