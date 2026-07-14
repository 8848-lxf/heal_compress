"""Canonical explicit activation/weight/output Q/DQ insertion."""

from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

from ..artifacts.io import file_sha256
from ..config import QDQConfig
from ..exceptions import QDQInsertionError
from ..types import CanonicalPrecisionMappingResult, QDQInsertionRecord, QDQInsertionResult, stable_json_hash
from ..export.origin_trace import build_weight_trace_index, trace_compute_node_weight
from .activation_boundary import resolve_activation_output_boundary
from .merge_contract import fp16_merge_cast_name


def _positive_scale(value: Any, module_path: str, kind: str) -> float | list[float]:
    if value is None:
        raise QDQInsertionError(f"{kind} calibration scale missing for {module_path}")
    raw = value.tolist() if hasattr(value, "tolist") else value
    if isinstance(raw, (list, tuple)):
        result = [float(item) for item in raw]
        if not result or any(not math.isfinite(item) or item <= 0.0 for item in result):
            raise QDQInsertionError(f"{kind} calibration scale must be positive and finite for {module_path}")
        return result
    result = float(raw)
    if not math.isfinite(result) or result <= 0.0:
        raise QDQInsertionError(f"{kind} calibration scale must be positive and finite for {module_path}")
    return result


def _scales_for(
    scales: Mapping[str, Any], module_path: str, canonical_name: str
) -> tuple[float, float | list[float], float, dict[str, Any]]:
    raw = scales.get(module_path, scales.get(canonical_name))
    metadata: dict[str, Any] = {}
    if isinstance(raw, Mapping):
        metadata = dict(raw)
        common = raw.get("scale", raw.get("activation_scale", raw.get("value")))
        activation_input = raw.get("activation_input_scale", common)
        weight = raw.get("weight_scale", common if common is not None else activation_input)
        activation_output = raw.get("activation_output_scale", raw.get("output_scale", common if common is not None else activation_input))
    else:
        activation_input = weight = activation_output = raw
    if raw is None:
        raise QDQInsertionError(f"calibration scale missing for {module_path}")
    return (
        float(_positive_scale(activation_input, module_path, "activation input")),
        _positive_scale(weight, module_path, "weight"),
        float(_positive_scale(activation_output, module_path, "activation output")),
        metadata,
    )


def _insert_output_qdq_for(entry: Any, metadata: Mapping[str, Any], policy: QDQConfig) -> bool:
    """Resolve output-Q ownership, with FP16 merge contracts taking priority."""

    fp16_output_contract = str(entry.realized_output_precision or "").lower() == "fp16"
    explicitly_requested = metadata.get("insert_activation_output_qdq")
    if fp16_output_contract:
        if explicitly_requested is True:
            raise QDQInsertionError(
                f"activation output Q/DQ conflicts with FP16 output contract for {entry.module_path}"
            )
        return False
    if explicitly_requested is not None:
        return bool(explicitly_requested)
    return bool(policy.insert_activation_output_qdq)


def _qdq_pair(
    helper: Any,
    numpy_helper: Any,
    np: Any,
    graph: Any,
    source: str,
    prefix: str,
    scale: float | list[float],
    zero_point: int,
    *,
    axis: int | None = None,
) -> tuple[list[Any], str, str, str]:
    scale_name = f"{prefix}__scale"
    zero_name = f"{prefix}__zero_point"
    quantized = f"{prefix}__quantized"
    dequantized = f"{prefix}__dequantized"
    q_name = f"{prefix}__QuantizeLinear"
    dq_name = f"{prefix}__DequantizeLinear"
    scale_array = np.asarray(scale, dtype=np.float32)
    if scale_array.ndim > 1:
        raise QDQInsertionError(f"Q/DQ scale must be scalar or one-dimensional: {prefix}")
    if axis is None and scale_array.ndim != 0:
        raise QDQInsertionError(f"per-channel Q/DQ scale requires an axis: {prefix}")
    if axis is not None and scale_array.ndim != 1:
        raise QDQInsertionError(f"Q/DQ axis is only valid for a one-dimensional scale: {prefix}")
    zero_array = (
        np.full(scale_array.shape, int(zero_point), dtype=np.int8)
        if scale_array.ndim
        else np.asarray(int(zero_point), dtype=np.int8)
    )
    graph.initializer.extend(
        [
            numpy_helper.from_array(scale_array, name=scale_name),
            numpy_helper.from_array(zero_array, name=zero_name),
        ]
    )
    attributes = {} if axis is None else {"axis": int(axis)}
    return (
        [
            helper.make_node("QuantizeLinear", [source, scale_name, zero_name], [quantized], name=q_name, **attributes),
            helper.make_node("DequantizeLinear", [quantized, scale_name, zero_name], [dequantized], name=dq_name, **attributes),
        ],
        dequantized,
        q_name,
        dq_name,
    )


def _audit_fp16_merge_boundaries(
    model: Any,
    merge_policy: str,
    mapping: CanonicalPrecisionMappingResult,
) -> list[dict[str, Any]]:
    import numpy as np
    from onnx import numpy_helper

    if merge_policy != "fp16_merge":
        raise QDQInsertionError(f"unsupported_explicit_qdq_merge_policy:{merge_policy}")
    producers = {
        str(output): node
        for node in model.graph.node
        for output in node.output
    }
    consumers: dict[str, list[Any]] = {}
    for node in model.graph.node:
        for input_name in node.input:
            consumers.setdefault(str(input_name), []).append(node)
    entries = {str(row.canonical_node_name): row for row in mapping.entries}
    initializers = {
        str(row.name): row
        for row in model.graph.initializer
    }

    def dq_scale_payload(producer: Any | None) -> dict[str, Any] | None:
        if producer is None or str(producer.op_type) != "DequantizeLinear" or len(producer.input) < 2:
            return None
        initializer = initializers.get(str(producer.input[1]))
        if initializer is None:
            return None
        values = np.asarray(numpy_helper.to_array(initializer), dtype=np.float64)
        return {
            "initializer": str(producer.input[1]),
            "shape": list(values.shape),
            "min": float(values.min()),
            "max": float(values.max()),
        }

    def entry_payload(node: Any) -> dict[str, Any] | None:
        entry = entries.get(str(getattr(node, "name", "")))
        if entry is None:
            return None
        return {
            "canonical_layer": entry.module_path,
            "canonical_node": entry.canonical_node_name,
            "quantization_group": entry.precision_group,
            "requested_precision": entry.requested_precision,
            "legalized_precision": entry.realized_request_precision,
        }

    def nearest_upstream(tensor_name: str, seen: set[str] | None = None) -> list[dict[str, Any]]:
        seen = set(seen or ())
        if tensor_name in seen:
            return []
        seen.add(tensor_name)
        producer = producers.get(str(tensor_name))
        if producer is None:
            return []
        payload = entry_payload(producer)
        if payload is not None:
            return [payload]
        result = []
        for input_name in producer.input:
            if str(input_name) in initializers:
                continue
            result.extend(nearest_upstream(str(input_name), seen))
        unique = {row["canonical_node"]: row for row in result}
        return [unique[key] for key in sorted(unique)]

    def nearest_downstream(tensor_names: list[str]) -> list[dict[str, Any]]:
        queue = list(tensor_names)
        seen: set[str] = set()
        found: dict[str, dict[str, Any]] = {}
        while queue:
            tensor_name = queue.pop(0)
            if tensor_name in seen:
                continue
            seen.add(tensor_name)
            for consumer in consumers.get(tensor_name, []):
                payload = entry_payload(consumer)
                if payload is not None:
                    found[payload["canonical_node"]] = payload
                else:
                    queue.extend(str(output) for output in consumer.output)
        return [found[key] for key in sorted(found)]
    rows: list[dict[str, Any]] = []
    for node in model.graph.node:
        if str(node.op_type) not in {"Add", "Concat"}:
            continue
        branches = []
        for input_name in node.input:
            producer = producers.get(str(input_name))
            producer_type = str(producer.op_type) if producer is not None else "graph_input_or_initializer"
            if producer_type == "QuantizeLinear":
                raise QDQInsertionError(f"fp16_merge_received_quantized_tensor:{node.name}:{input_name}")
            branches.append(
                {
                    "tensor": str(input_name),
                    "producer": str(producer.name) if producer is not None else "",
                    "producer_op_type": producer_type,
                    "boundary": (
                        "explicit_cast_to_fp16"
                        if producer_type == "Cast"
                        else "explicit_dequantize_to_float"
                        if producer_type == "DequantizeLinear"
                        else "existing_float_path"
                    ),
                    "cast_to_fp16": bool(
                        producer_type == "Cast"
                        and any(str(attribute.name) == "to" and int(attribute.i) == 10 for attribute in producer.attribute)
                    ) if producer is not None else False,
                    "activation_scale": dq_scale_payload(producer),
                    "qdq_tensor_before_merge": str(input_name) if producer_type == "DequantizeLinear" else "",
                    "nearest_weighted_producers": nearest_upstream(str(input_name)),
                }
            )
        downstream = [
            {
                "consumer": str(consumer.name),
                "op_type": str(consumer.op_type),
                "requantized": str(consumer.op_type) == "QuantizeLinear",
            }
            for output_name in node.output
            for consumer in consumers.get(str(output_name), [])
        ]
        downstream_weighted = nearest_downstream([str(value) for value in node.output])
        row = (
            {
                "merge_op_name": str(node.name),
                "merge_op_type": str(node.op_type),
                "output_tensors": [str(value) for value in node.output],
                "policy": "A_fp16_merge",
                "input_branches": branches,
                "merge_scale_policy": "independent_branch_scales_then_DQ_to_float; optional_downstream_requantization",
                "downstream": downstream,
                "downstream_weighted_layers": downstream_weighted,
                "partial_explicit_qdq_input_count": sum(
                    branch["boundary"] == "explicit_dequantize_to_float" for branch in branches
                ),
                "input_count": len(branches),
                "weighted_input_branch_count": sum(
                    bool(branch["nearest_weighted_producers"]) for branch in branches
                ),
            }
        )
        row["quantization_relevant_merge"] = row["weighted_input_branch_count"] >= 2
        row["merge_role"] = (
            "residual_activation_merge"
            if str(node.op_type) == "Add" and row["quantization_relevant_merge"]
            else "concat_activation_merge"
            if str(node.op_type) == "Concat" and row["quantization_relevant_merge"]
            else "non_activation_shape_or_functional_merge"
        )
        if row["quantization_relevant_merge"]:
            rows.append(row)
    return rows


def _insert_explicit_fp16_merge_casts(model: Any, mapping: CanonicalPrecisionMappingResult) -> list[dict[str, Any]]:
    from onnx import TensorProto, helper

    target_merges = {
        str(name)
        for name, precision in mapping.auxiliary_layer_precisions.items()
        if str(precision) == "fp16"
    }
    records: list[dict[str, Any]] = []
    rewritten: list[Any] = []
    for node in model.graph.node:
        if str(node.name) not in target_merges or str(node.op_type) not in {"Add", "Concat"}:
            rewritten.append(node)
            continue
        for input_index, input_name in enumerate(list(node.input)):
            cast_name = fp16_merge_cast_name(str(node.name), input_index)
            cast_output = f"{cast_name}__output"
            rewritten.append(
                helper.make_node(
                    "Cast",
                    [str(input_name)],
                    [cast_output],
                    name=cast_name,
                    to=TensorProto.FLOAT16,
                )
            )
            node.input[input_index] = cast_output
            records.append(
                {
                    "merge_op_name": str(node.name),
                    "merge_op_type": str(node.op_type),
                    "input_index": int(input_index),
                    "source_tensor": str(input_name),
                    "cast_node": cast_name,
                    "cast_output_tensor": cast_output,
                    "cast_dtype": "FP16",
                }
            )
        rewritten.append(node)
    del model.graph.node[:]
    model.graph.node.extend(rewritten)
    return records


def _snapshot_weighted_following_ops(
    model: Any,
    mapping: CanonicalPrecisionMappingResult,
) -> dict[str, dict[str, Any]]:
    """Capture the pre-Q/DQ semantic path following each weighted node."""

    nodes_by_name = {str(node.name): node for node in model.graph.node}
    consumers: dict[str, list[Any]] = {}
    for node in model.graph.node:
        for input_name in node.input:
            consumers.setdefault(str(input_name), []).append(node)
    weighted_names = {str(row.canonical_node_name) for row in mapping.entries}
    snapshots: dict[str, dict[str, Any]] = {}
    for entry in mapping.entries:
        node = nodes_by_name.get(str(entry.canonical_node_name))
        if node is None:
            continue
        following: list[dict[str, Any]] = []
        queue = [(str(output), 0) for output in node.output]
        seen_tensors: set[str] = set()
        while queue:
            tensor_name, depth = queue.pop(0)
            if tensor_name in seen_tensors or depth > 8:
                continue
            seen_tensors.add(tensor_name)
            for consumer in consumers.get(tensor_name, []):
                is_weighted = str(consumer.name) in weighted_names
                following.append(
                    {
                        "name": str(consumer.name),
                        "op_type": str(consumer.op_type),
                        "input_tensor": tensor_name,
                        "output_tensors": [str(value) for value in consumer.output],
                        "depth": depth,
                        "next_weighted_node": is_weighted,
                    }
                )
                if not is_weighted:
                    queue.extend((str(output), depth + 1) for output in consumer.output)
        snapshots[str(entry.canonical_node_name)] = {
            "weighted_output_tensors": [str(value) for value in node.output],
            "following_ops": following,
        }
    return snapshots


def _audit_weighted_qdq_boundaries(
    model: Any,
    mapping: CanonicalPrecisionMappingResult,
    records: list[QDQInsertionRecord],
    scales: Mapping[str, Any],
    snapshots: Mapping[str, Mapping[str, Any]],
    policy: QDQConfig,
) -> tuple[list[dict[str, Any]], str]:
    nodes_by_name = {str(node.name): node for node in model.graph.node}
    entries = {str(row.canonical_node_name): row for row in mapping.entries}
    rows: list[dict[str, Any]] = []
    for record in records:
        entry = entries[str(record.canonical_node_name)]
        snapshot = dict(snapshots.get(str(record.canonical_node_name), {}))
        output_q_nodes = [nodes_by_name.get(str(name)) for name in record.output_quantize_nodes]
        q_inputs = [str(node.input[0]) for node in output_q_nodes if node is not None and node.input]
        normalized_q_inputs = {value.replace("__before_output_qdq", "") for value in q_inputs}
        weighted_outputs = set(str(value) for value in snapshot.get("weighted_output_tensors", []))
        scale_payload = scales.get(record.module_path, scales.get(record.canonical_node_name, {}))
        scale_metadata = dict(scale_payload) if isinstance(scale_payload, Mapping) else {}
        if q_inputs and normalized_q_inputs == weighted_outputs:
            placement = "weighted_output_before_following_ops"
        elif not q_inputs and entry.realized_output_precision == "fp16":
            placement = "fp16_weighted_output_no_output_qdq"
        elif q_inputs:
            placement = "post_weighted_semantic_boundary"
        else:
            placement = "no_activation_output_qdq"
        rows.append(
            {
                "canonical_layer": entry.module_path,
                "weighted_node": entry.canonical_node_name,
                "weighted_output_tensor": list(snapshot.get("weighted_output_tensors", [])),
                "following_ops": list(snapshot.get("following_ops", [])),
                "q_node_actual_input_tensor": q_inputs,
                "qdq_placement": placement,
                "activation_scale_owner": str(scale_metadata.get("activation_output_tensor", "")),
                "activation_output_scale": float(record.activation_output_scale),
                "merge_policy": policy.merge_policy,
                "requested_precision": entry.requested_precision,
                "legalized_precision": entry.realized_request_precision,
                "requested_output_precision": entry.realized_output_precision or entry.realized_request_precision,
                "realized_precision": "not_yet_inspected",
                "engine_fused_layer": "",
                "engine_effective_boundary": "not_yet_inspected",
                "weight_granularity": record.weight_granularity,
                "weight_axis": record.weight_axis,
                "activation_output_boundary_policy": policy.activation_output_boundary_policy,
            }
        )
    topology_payload = [
        {
            key: row[key]
            for key in (
                "canonical_layer",
                "weighted_node",
                "weighted_output_tensor",
                "following_ops",
                "q_node_actual_input_tensor",
                "qdq_placement",
                "activation_scale_owner",
                "merge_policy",
                "requested_precision",
                "legalized_precision",
                "requested_output_precision",
                "weight_granularity",
                "weight_axis",
                "activation_output_boundary_policy",
            )
        }
        for row in rows
    ]
    return rows, stable_json_hash(topology_payload)


def insert_explicit_qdq(
    input_onnx: str | Path,
    output_onnx: str | Path,
    mapping: CanonicalPrecisionMappingResult,
    *,
    scales: Mapping[str, Any],
    config: QDQConfig | None = None,
    calibration_metadata: Mapping[str, Any] | None = None,
) -> QDQInsertionResult:
    """Insert explicit INT8 Q/DQ only at exact canonical weighted nodes."""

    import numpy as np
    import onnx
    from onnx import helper, numpy_helper

    policy = config or QDQConfig()
    model = onnx.load(str(input_onnx))
    boundary_snapshots = _snapshot_weighted_following_ops(model, mapping)
    index = build_weight_trace_index(model)
    nodes_by_name = index["nodes_by_name"]
    requested = [row for row in mapping.entries if row.requested_precision == "int8"]
    targets = {row.canonical_node_name: row for row in mapping.entries if row.realized_request_precision == "int8"}
    if len(targets) != sum(row.realized_request_precision == "int8" for row in mapping.entries):
        raise QDQInsertionError("canonical INT8 target names are not unique")
    initializers = {str(row.name): numpy_helper.to_array(row) for row in model.graph.initializer}
    prepared: dict[str, tuple[Any, float, float | list[float], float, dict[str, Any], int | None, dict[str, Any]]] = {}
    for name, entry in targets.items():
        node = nodes_by_name.get(name)
        if node is None or str(node.op_type) not in {"Conv", "ConvTranspose", "Gemm", "MatMul"}:
            raise QDQInsertionError(f"canonical INT8 compute node missing or unsupported: {name}")
        trace = trace_compute_node_weight(index, node)
        if not trace.get("success") or str(trace.get("root_initializer", "")) != entry.weight_initializer:
            raise QDQInsertionError(f"canonical weight root mismatch for {entry.module_path}")
        activation_input_scale, weight_scale, activation_output_scale, metadata = _scales_for(
            scales, entry.module_path, name
        )
        weight_axis_raw = metadata.get("weight_axis")
        weight_axis = int(weight_axis_raw) if weight_axis_raw not in (None, "") else None
        if isinstance(weight_scale, list):
            weight = initializers.get(str(entry.weight_initializer))
            if weight is None:
                raise QDQInsertionError(f"per-channel weight initializer missing for {entry.module_path}")
            if weight_axis is None:
                raise QDQInsertionError(f"per-channel weight scale axis missing for {entry.module_path}")
            normalized_axis = weight_axis if weight_axis >= 0 else weight.ndim + weight_axis
            if normalized_axis < 0 or normalized_axis >= weight.ndim:
                raise QDQInsertionError(f"per-channel weight scale axis out of range for {entry.module_path}")
            if len(weight_scale) != int(weight.shape[normalized_axis]):
                raise QDQInsertionError(
                    f"per-channel weight scale length mismatch for {entry.module_path}: "
                    f"scale={len(weight_scale)} axis={weight_axis} weight_shape={list(weight.shape)}"
                )
            weight_axis = normalized_axis
        elif weight_axis is not None:
            raise QDQInsertionError(f"scalar weight scale must not declare an axis for {entry.module_path}")
        output_boundary = resolve_activation_output_boundary(model, name)
        insert_output_qdq = _insert_output_qdq_for(entry, metadata, policy)
        scale_owner = str(metadata.get("activation_output_tensor", ""))
        if insert_output_qdq and scale_owner and scale_owner != str(output_boundary["boundary_output_tensor"]):
            raise QDQInsertionError(
                f"activation output scale owner does not match resolved Q boundary for {entry.module_path}: "
                f"scale={scale_owner} boundary={output_boundary['boundary_output_tensor']}"
            )
        prepared[name] = (
            entry,
            activation_input_scale,
            weight_scale,
            activation_output_scale,
            metadata,
            weight_axis,
            output_boundary,
        )

    new_nodes: list[Any] = []
    records: list[QDQInsertionRecord] = []
    pending_output_qdq: dict[str, list[dict[str, Any]]] = {}
    for node in model.graph.node:
        target = prepared.get(str(node.name))
        if target is None:
            new_nodes.append(node)
            continue
        entry, activation_input_scale, weight_scale, activation_output_scale, _metadata, weight_axis, output_boundary = target
        safe = str(node.name).replace("/", "_").replace(".", "_")
        record = QDQInsertionRecord(
            module_path=entry.module_path,
            canonical_node_name=entry.canonical_node_name,
            weight_initializer=entry.weight_initializer,
            scale=activation_input_scale,
            activation_input_scale=activation_input_scale,
            weight_scale=weight_scale,
            weight_scale_shape=[len(weight_scale)] if isinstance(weight_scale, list) else [],
            weight_axis=weight_axis,
            weight_granularity="per_channel" if isinstance(weight_scale, list) else "per_tensor",
            activation_output_scale=activation_output_scale,
            zero_point=int(policy.zero_point),
            activation_output_boundary_policy=policy.activation_output_boundary_policy,
        )
        if policy.insert_activation_input_qdq:
            pair, dequantized, q_name, dq_name = _qdq_pair(
                helper, numpy_helper, np, model.graph, str(node.input[0]), f"{safe}__activation_input", activation_input_scale, policy.zero_point
            )
            new_nodes.extend(pair)
            node.input[0] = dequantized
            record.activation_quantize_node = q_name
            record.activation_dequantize_node = dq_name
        if policy.insert_weight_qdq:
            pair, dequantized, q_name, dq_name = _qdq_pair(
                helper,
                numpy_helper,
                np,
                model.graph,
                str(node.input[1]),
                f"{safe}__weight",
                weight_scale,
                policy.zero_point,
                axis=weight_axis,
            )
            new_nodes.extend(pair)
            node.input[1] = dequantized
            record.weight_quantize_node = q_name
            record.weight_dequantize_node = dq_name
        insert_output_qdq = _insert_output_qdq_for(entry, _metadata, policy)
        if insert_output_qdq:
            boundary_node_name = str(output_boundary["boundary_node_name"])
            boundary_tensor = str(output_boundary["boundary_output_tensor"])
            pending_output_qdq.setdefault(boundary_node_name, []).append(
                {
                    "record": record,
                    "public_name": boundary_tensor,
                    "prefix": f"{safe}__activation_output_0",
                    "scale": activation_output_scale,
                    "resolution": str(output_boundary["resolution"]),
                }
            )
        new_nodes.append(node)
        records.append(record)
    if len(records) != len(targets):
        raise QDQInsertionError("not every canonical INT8 target received Q/DQ")

    output_nodes: list[Any] = []
    applied_output_qdq = 0
    claimed_boundary_tensors: set[str] = set()
    for node in new_nodes:
        specs = pending_output_qdq.get(str(node.name), [])
        for spec in specs:
            public_name = str(spec["public_name"])
            if public_name in claimed_boundary_tensors:
                raise QDQInsertionError(f"activation output Q/DQ boundary claimed more than once: {public_name}")
            output_indices = [index for index, value in enumerate(node.output) if str(value) == public_name]
            if len(output_indices) != 1:
                raise QDQInsertionError(
                    f"resolved activation output boundary missing or ambiguous: {node.name}:{public_name}"
                )
            raw_name = f"{public_name}__before_output_qdq"
            node.output[output_indices[0]] = raw_name
            claimed_boundary_tensors.add(public_name)
            spec["raw_name"] = raw_name
        output_nodes.append(node)
        for spec in specs:
            pair, _dequantized, q_name, dq_name = _qdq_pair(
                helper,
                numpy_helper,
                np,
                model.graph,
                str(spec["raw_name"]),
                str(spec["prefix"]),
                float(spec["scale"]),
                policy.zero_point,
            )
            pair[-1].output[0] = str(spec["public_name"])
            output_nodes.extend(pair)
            record = spec["record"]
            record.output_quantize_nodes.append(q_name)
            record.output_dequantize_nodes.append(dq_name)
            record.activation_output_q_inputs.append(str(spec["raw_name"]))
            applied_output_qdq += 1
    expected_output_qdq = sum(len(value) for value in pending_output_qdq.values())
    if applied_output_qdq != expected_output_qdq:
        raise QDQInsertionError(
            f"not every resolved activation output boundary received Q/DQ: {applied_output_qdq}!={expected_output_qdq}"
        )
    del model.graph.node[:]
    model.graph.node.extend(output_nodes)
    merge_cast_records = _insert_explicit_fp16_merge_casts(model, mapping)
    merge_audit = _audit_fp16_merge_boundaries(model, policy.merge_policy, mapping)
    boundary_audit, topology_hash = _audit_weighted_qdq_boundaries(
        model,
        mapping,
        records,
        scales,
        boundary_snapshots,
        policy,
    )
    destination = Path(output_onnx)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".onnx", dir=destination.parent)
    os.close(descriptor)
    try:
        onnx.save(model, temporary)
        try:
            onnx.checker.check_model(onnx.load(temporary))
        except Exception as exc:
            has_custom = any(str(node.domain) for node in model.graph.node)
            if not has_custom:
                raise QDQInsertionError(f"Q/DQ ONNX checker failed: {exc}") from exc
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    fallbacks = [
        {
            "module_path": row.module_path,
            "canonical_node_name": row.canonical_node_name,
            "requested_precision": row.requested_precision,
            "realized_request_precision": row.realized_request_precision,
            "fallback_reason": row.fallback_reason,
        }
        for row in mapping.entries
        if row.requested_precision != row.realized_request_precision
    ]
    metadata = dict(calibration_metadata or {})
    metadata.setdefault("scale_modules", sorted(str(key) for key in scales))
    metadata["merge_policy"] = policy.merge_policy
    metadata["fp16_merge_cast_records"] = merge_cast_records
    metadata["auxiliary_layer_precisions"] = dict(mapping.auxiliary_layer_precisions)
    metadata["auxiliary_layer_output_types"] = dict(mapping.auxiliary_layer_output_types)
    metadata["merge_quantization_audit"] = merge_audit
    metadata["weighted_qdq_boundary_audit"] = boundary_audit
    metadata["qdq_topology_hash"] = topology_hash
    return QDQInsertionResult(
        input_onnx=str(input_onnx),
        output_onnx=str(destination),
        inserted_layer_count=len(records),
        requested_int8_count=len(requested),
        records=records,
        fallback_entries=fallbacks,
        calibration_metadata=metadata,
        policy_version=policy.policy_version,
        output_sha256=file_sha256(destination),
    )
