"""Explicit ONNX dtype closure for strongly typed TensorRT builds."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from ..artifacts.io import file_sha256
from ..types import CanonicalPrecisionMappingResult


_FLOATING_COMPUTE_INPUTS = {
    "Conv": (0, 1, 2),
    "ConvTranspose": (0, 1, 2),
    "Gemm": (0, 1, 2),
    "MatMul": (0, 1),
}

_SAME_TYPE_ELEMENTWISE_OPS = {"Add", "Sub", "Mul", "Div", "Pow", "Min", "Max"}
_SAME_TYPE_COMPARISON_OPS = {"Equal", "Greater", "Less"}
_TYPE_PRESERVING_UNARY_OPS = {
    "Abs",
    "Clip",
    "Expand",
    "Flatten",
    "Identity",
    "Pad",
    "ReduceMean",
    "ReduceSum",
    "Relu",
    "Reshape",
    "Resize",
    "Sigmoid",
    "Slice",
    "Softmax",
    "Squeeze",
    "Transpose",
    "Unsqueeze",
}


def _boundary_type(boundary: str) -> tuple[int, str]:
    from onnx import TensorProto

    normalized = str(boundary).lower()
    if normalized == "fp16":
        return int(TensorProto.FLOAT16), "FP16"
    if normalized == "fp32":
        return int(TensorProto.FLOAT), "FP32"
    raise ValueError("scatter_boundary_must_be_fp16_or_fp32")


def _initializer_types(model: Any) -> dict[str, int]:
    return {str(value.name): int(value.data_type) for value in model.graph.initializer}


def _value_infos(model: Any) -> dict[str, Any]:
    return {
        str(value.name): value
        for value in [*model.graph.input, *model.graph.value_info, *model.graph.output]
    }


def _tensor_types(model: Any) -> dict[str, int]:
    result = _initializer_types(model)
    for name, value in _value_infos(model).items():
        tensor_type = value.type.tensor_type
        if tensor_type.elem_type:
            result[name] = int(tensor_type.elem_type)
    return result


def _shape_from_value_info(value: Any | None) -> list[int | str | None] | None:
    if value is None or not value.type.tensor_type.HasField("shape"):
        return None
    result: list[int | str | None] = []
    for dimension in value.type.tensor_type.shape.dim:
        if dimension.HasField("dim_value"):
            result.append(int(dimension.dim_value))
        elif dimension.dim_param:
            result.append(str(dimension.dim_param))
        else:
            result.append(None)
    return result


def _set_tensor_type(model: Any, name: str, element_type: int, *, like: str = "") -> None:
    from onnx import helper

    infos = _value_infos(model)
    existing = infos.get(str(name))
    if existing is not None:
        existing.type.tensor_type.elem_type = int(element_type)
        return
    shape = _shape_from_value_info(infos.get(str(like))) if like else None
    model.graph.value_info.append(
        helper.make_tensor_value_info(str(name), int(element_type), shape)
    )


def _cast_target(node: Any) -> int | None:
    if str(node.op_type) != "Cast":
        return None
    for attribute in node.attribute:
        if str(attribute.name) == "to":
            return int(attribute.i)
    return None


def _safe_cast_name(owner: str, input_index: int, target_type: int) -> str:
    digest = hashlib.sha256(
        f"{owner}\0{input_index}\0{target_type}".encode("utf-8")
    ).hexdigest()[:12]
    return f"__typed__{digest}__input{input_index:02d}__Cast"


def _insert_input_cast(
    *,
    model: Any,
    node: Any,
    input_index: int,
    target_type: int,
    producers: dict[str, Any],
    rewritten: list[Any],
    records: list[dict[str, Any]],
) -> None:
    from onnx import helper

    if input_index >= len(node.input) or not str(node.input[input_index]):
        return
    source = str(node.input[input_index])
    producer = producers.get(source)
    if producer is not None and _cast_target(producer) == int(target_type):
        return
    cast_name = _safe_cast_name(str(node.name), int(input_index), int(target_type))
    cast_output = f"{cast_name}__output"
    cast = helper.make_node(
            "Cast",
            [source],
            [cast_output],
            name=cast_name,
            to=int(target_type),
        )
    rewritten.append(cast)
    producers[cast_output] = cast
    node.input[input_index] = cast_output
    _set_tensor_type(model, cast_output, int(target_type), like=source)
    records.append(
        {
            "owner_node": str(node.name),
            "owner_op_type": str(node.op_type),
            "input_index": int(input_index),
            "source_tensor": source,
            "cast_node": cast_name,
            "cast_output_tensor": cast_output,
            "onnx_element_type": int(target_type),
        }
    )


def _constant_output_type(node: Any) -> int | None:
    if str(node.op_type) not in {"Constant", "ConstantOfShape"}:
        return None
    for attribute in node.attribute:
        if str(attribute.name) == "value" and attribute.HasField("t"):
            return int(attribute.t.data_type)
        if str(attribute.name) == "value_float":
            from onnx import TensorProto

            return int(TensorProto.FLOAT)
        if str(attribute.name) == "value_int":
            from onnx import TensorProto

            return int(TensorProto.INT64)
    if str(node.op_type) == "ConstantOfShape":
        from onnx import TensorProto

        return int(TensorProto.FLOAT)
    return None


def _append_output_casts(
    *,
    model: Any,
    node: Any,
    target_type: int,
    source_type: int,
    producers: dict[str, Any],
    rewritten: list[Any],
    records: list[dict[str, Any]],
) -> None:
    from onnx import helper

    if int(target_type) == int(source_type):
        return
    for output_index, value in enumerate(list(node.output)):
        public_name = str(value)
        cast_name = _safe_cast_name(
            f"{node.name}__output", int(output_index), int(target_type)
        )
        raw_name = f"{cast_name}__input"
        node.output[output_index] = raw_name
        cast = helper.make_node(
            "Cast",
            [raw_name],
            [public_name],
            name=cast_name,
            to=int(target_type),
        )
        rewritten.append(cast)
        producers[raw_name] = node
        producers[public_name] = cast
        _set_tensor_type(model, raw_name, int(source_type), like=public_name)
        _set_tensor_type(model, public_name, int(target_type), like=raw_name)
        records.append(
            {
                "owner_node": str(node.name),
                "owner_op_type": str(node.op_type),
                "output_index": int(output_index),
                "source_tensor": raw_name,
                "cast_node": cast_name,
                "cast_output_tensor": public_name,
                "onnx_element_type": int(target_type),
                "direction": "output_contract",
            }
        )


def _propagate_known_types(model: Any) -> dict[str, int]:
    from onnx import TensorProto

    types = _tensor_types(model)
    changed = True
    while changed:
        changed = False
        for node in model.graph.node:
            op_type = str(node.op_type)
            inferred: int | None = None
            if op_type == "Cast":
                inferred = _cast_target(node)
            elif op_type in {"Constant", "ConstantOfShape"}:
                inferred = _constant_output_type(node)
            elif op_type == "QuantizeLinear" and len(node.input) >= 3:
                inferred = types.get(str(node.input[2]), int(TensorProto.INT8))
            elif op_type == "DequantizeLinear" and len(node.input) >= 2:
                inferred = types.get(str(node.input[1]))
            elif op_type in {"Shape", "NonZero", "ArgMax", "ArgMin"}:
                inferred = int(TensorProto.INT64)
            elif op_type in (_SAME_TYPE_COMPARISON_OPS | {"And", "Or", "Not"}):
                inferred = int(TensorProto.BOOL)
            elif op_type == "Where" and len(node.input) >= 3:
                inferred = types.get(str(node.input[2]), types.get(str(node.input[1])))
            elif op_type in (
                _TYPE_PRESERVING_UNARY_OPS
                | _SAME_TYPE_ELEMENTWISE_OPS
                | {
                    "AveragePool",
                    "Concat",
                    "Conv",
                    "ConvTranspose",
                    "Gather",
                    "GatherElements",
                    "GlobalAveragePool",
                    "GridSample",
                    "MatMul",
                    "MaxPool",
                    "ScatterND",
                    "Split",
                    "Tile",
                }
            ) and node.input:
                inferred = types.get(str(node.input[0]))
            if inferred is None:
                continue
            for output in node.output:
                name = str(output)
                if types.get(name) != int(inferred):
                    types[name] = int(inferred)
                    _set_tensor_type(model, name, int(inferred), like=str(node.input[0]) if node.input else "")
                    changed = True
    return types


def _plugin_qdq_count(model: Any, plugin: Any) -> int:
    producers = {
        str(output): node for node in model.graph.node for output in node.output
    }
    consumers: dict[str, list[Any]] = {}
    for node in model.graph.node:
        for value in node.input:
            consumers.setdefault(str(value), []).append(node)
    adjacent = []
    for value in plugin.input:
        producer = producers.get(str(value))
        if producer is not None:
            adjacent.append(producer)
    for value in plugin.output:
        adjacent.extend(consumers.get(str(value), []))
    return sum(
        str(node.op_type) in {"QuantizeLinear", "DequantizeLinear"}
        for node in adjacent
    )


def apply_strongly_typed_precision_contract(
    input_onnx: str | Path,
    output_onnx: str | Path,
    mapping: CanonicalPrecisionMappingResult,
    *,
    plugin_boundary: str,
) -> dict[str, Any]:
    """Insert all floating dtype transitions needed by a strongly typed build."""

    import onnx
    from onnx import TensorProto

    plugin_type, plugin_precision = _boundary_type(plugin_boundary)
    model = onnx.load(str(input_onnx))
    entries: dict[str, Any] = {}
    for entry in mapping.entries:
        entries[str(entry.canonical_node_name)] = entry
        for node_name in entry.constraint_node_names:
            existing = entries.get(str(node_name))
            if existing is not None and existing is not entry:
                raise ValueError(f"duplicate_typed_constraint_node:{node_name}")
            entries[str(node_name)] = entry
    producers = {
        str(output): node for node in model.graph.node for output in node.output
    }
    initializer_types = _initializer_types(model)
    cast_records: list[dict[str, Any]] = []
    plugins: list[Any] = []
    rewritten: list[Any] = []
    for node in model.graph.node:
        node_name = str(node.name)
        op_type = str(node.op_type)
        entry = entries.get(node_name)
        output_type: int | None = None
        if op_type == "PointPillarScatterTRT":
            plugins.append(node)
            _insert_input_cast(
                model=model,
                node=node,
                input_index=0,
                target_type=plugin_type,
                producers=producers,
                rewritten=rewritten,
                records=cast_records,
            )
            output_type = plugin_type
        elif op_type == "QuantizeLinear":
            scale_type = initializer_types.get(
                str(node.input[1]), int(TensorProto.FLOAT)
            )
            _insert_input_cast(
                model=model,
                node=node,
                input_index=0,
                target_type=scale_type,
                producers=producers,
                rewritten=rewritten,
                records=cast_records,
            )
            output_type = initializer_types.get(
                str(node.input[2]), int(TensorProto.INT8)
            )
        if entry is not None and str(entry.realized_request_precision) in {"fp16", "fp32"}:
            target = (
                int(TensorProto.FLOAT16)
                if str(entry.realized_request_precision) == "fp16"
                else int(TensorProto.FLOAT)
            )
            for input_index in _FLOATING_COMPUTE_INPUTS.get(op_type, (0,)):
                _insert_input_cast(
                    model=model,
                    node=node,
                    input_index=input_index,
                    target_type=target,
                    producers=producers,
                    rewritten=rewritten,
                    records=cast_records,
                )
            output_type = target
        auxiliary_precision = mapping.auxiliary_layer_precisions.get(node_name)
        if str(auxiliary_precision) in {"fp16", "fp32"}:
            target = (
                int(TensorProto.FLOAT16)
                if str(auxiliary_precision) == "fp16"
                else int(TensorProto.FLOAT)
            )
            for input_index in range(len(node.input)):
                _insert_input_cast(
                    model=model,
                    node=node,
                    input_index=input_index,
                    target_type=target,
                    producers=producers,
                    rewritten=rewritten,
                    records=cast_records,
                )
            output_type = target

        current_types = _tensor_types(model)
        if op_type in (_SAME_TYPE_ELEMENTWISE_OPS | _SAME_TYPE_COMPARISON_OPS) and node.input:
            target = current_types.get(str(node.input[0]))
            if target is not None:
                for input_index in range(1, len(node.input)):
                    other = current_types.get(str(node.input[input_index]))
                    if other is not None and int(other) != int(target):
                        _insert_input_cast(
                            model=model,
                            node=node,
                            input_index=input_index,
                            target_type=target,
                            producers=producers,
                            rewritten=rewritten,
                            records=cast_records,
                        )
                output_type = (
                    int(TensorProto.BOOL)
                    if op_type in _SAME_TYPE_COMPARISON_OPS
                    else target
                )
        elif op_type == "Where" and len(node.input) >= 3:
            target = current_types.get(
                str(node.input[2]), current_types.get(str(node.input[1]))
            )
            then_type = current_types.get(str(node.input[1]))
            if target is not None and then_type is not None and int(then_type) != int(target):
                _insert_input_cast(
                    model=model,
                    node=node,
                    input_index=1,
                    target_type=target,
                    producers=producers,
                    rewritten=rewritten,
                    records=cast_records,
                )
            output_type = target
        elif op_type in {"Concat", "GridSample"} and node.input:
            target = current_types.get(str(node.input[0]))
            if target is None and op_type == "GridSample":
                raise ValueError(f"grid_sample_feature_dtype_unresolved:{node_name}")
            if target is not None:
                start_index = 1
                for input_index in range(start_index, len(node.input)):
                    other = current_types.get(str(node.input[input_index]))
                    if other is not None and int(other) != int(target):
                        _insert_input_cast(
                            model=model,
                            node=node,
                            input_index=input_index,
                            target_type=target,
                            producers=producers,
                            rewritten=rewritten,
                            records=cast_records,
                        )
                output_type = target

        if output_type is None:
            if op_type == "Cast":
                output_type = _cast_target(node)
            elif op_type == "Constant":
                output_type = _constant_output_type(node)
            elif op_type == "ConstantOfShape":
                output_type = _constant_output_type(node)
            elif op_type == "DequantizeLinear" and len(node.input) >= 2:
                output_type = current_types.get(str(node.input[1]))
            elif op_type in _TYPE_PRESERVING_UNARY_OPS and node.input:
                output_type = current_types.get(str(node.input[0]))
            elif entry is not None and str(entry.realized_request_precision) == "int8" and node.input:
                output_type = current_types.get(str(node.input[0]), int(TensorProto.FLOAT))
            elif op_type in {"Conv", "Gemm", "MatMul"} and node.input:
                output_type = current_types.get(str(node.input[0]))
            elif op_type in {"Shape", "NonZero", "ArgMax", "ArgMin"}:
                output_type = int(TensorProto.INT64)
            elif op_type in {"Equal", "Greater", "Less", "And", "Or", "Not"}:
                output_type = int(TensorProto.BOOL)

        rewritten.append(node)
        if output_type is not None:
            for output in node.output:
                _set_tensor_type(model, str(output), int(output_type))
        output_precision = str(getattr(entry, "realized_output_precision", "") or "")
        if output_precision in {"fp16", "fp32"}:
            target = (
                int(TensorProto.FLOAT16)
                if output_precision == "fp16"
                else int(TensorProto.FLOAT)
            )
            source = output_type if output_type is not None else int(TensorProto.FLOAT)
            _append_output_casts(
                model=model,
                node=node,
                target_type=target,
                source_type=source,
                producers=producers,
                rewritten=rewritten,
                records=cast_records,
            )
    if len(plugins) != 1:
        raise ValueError(f"expected_one_pointpillar_scatter_plugin:{len(plugins)}")
    del model.graph.node[:]
    model.graph.node.extend(rewritten)
    try:
        model = onnx.shape_inference.infer_shapes(model, strict_mode=False, data_prop=True)
    except Exception:
        pass
    types = _propagate_known_types(model)
    node_outputs = {
        str(output) for node in model.graph.node for output in node.output
    }
    unresolved = sorted(name for name in node_outputs if name not in types)
    destination = Path(output_onnx)
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(destination))
    counts = {"INT8": 0, "FP16": 0, "FP32": 0}
    for entry in mapping.entries:
        key = str(entry.realized_request_precision).upper()
        if key in counts:
            counts[key] += 1
    return {
        "input_onnx": str(input_onnx),
        "output_onnx": str(destination),
        "output_sha256": file_sha256(destination),
        "strongly_typed": True,
        "plugin_boundary_dtype": plugin_precision,
        "plugin_qdq_count": _plugin_qdq_count(model, plugins[0]),
        "canonical_precision_counts": counts,
        "cast_count": len(cast_records),
        "cast_records": cast_records,
        "unresolved_tensor_dtypes": unresolved,
        "unresolved_tensor_dtype_count": len(unresolved),
        "policy_version": "strongly-typed-explicit-qdq-cast-v1",
    }


__all__ = ["apply_strongly_typed_precision_contract"]
