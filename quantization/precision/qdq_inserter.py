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
from ..types import CanonicalPrecisionMappingResult, QDQInsertionRecord, QDQInsertionResult
from ..export.origin_trace import build_weight_trace_index, trace_compute_node_weight


def _positive_scale(value: Any, module_path: str, kind: str) -> float:
    if value is None:
        raise QDQInsertionError(f"{kind} calibration scale missing for {module_path}")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise QDQInsertionError(f"{kind} calibration scale must be positive and finite for {module_path}")
    return result


def _scales_for(
    scales: Mapping[str, Any], module_path: str, canonical_name: str
) -> tuple[float, float, float, dict[str, Any]]:
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
        _positive_scale(activation_input, module_path, "activation input"),
        _positive_scale(weight, module_path, "weight"),
        _positive_scale(activation_output, module_path, "activation output"),
        metadata,
    )


def _qdq_pair(helper: Any, numpy_helper: Any, np: Any, graph: Any, source: str, prefix: str, scale: float, zero_point: int) -> tuple[list[Any], str, str, str]:
    scale_name = f"{prefix}__scale"
    zero_name = f"{prefix}__zero_point"
    quantized = f"{prefix}__quantized"
    dequantized = f"{prefix}__dequantized"
    q_name = f"{prefix}__QuantizeLinear"
    dq_name = f"{prefix}__DequantizeLinear"
    graph.initializer.extend(
        [
            numpy_helper.from_array(np.asarray(float(scale), dtype=np.float32), name=scale_name),
            numpy_helper.from_array(np.asarray(int(zero_point), dtype=np.int8), name=zero_name),
        ]
    )
    return (
        [
            helper.make_node("QuantizeLinear", [source, scale_name, zero_name], [quantized], name=q_name),
            helper.make_node("DequantizeLinear", [quantized, scale_name, zero_name], [dequantized], name=dq_name),
        ],
        dequantized,
        q_name,
        dq_name,
    )


def _consumer_index(nodes: Any) -> dict[str, list[Any]]:
    consumers: dict[str, list[Any]] = {}
    for node in nodes:
        for name in node.input:
            consumers.setdefault(str(name), []).append(node)
    return consumers


def _activation_output_boundary(node: Any, output_name: str, consumers: Mapping[str, list[Any]], policy: QDQConfig) -> tuple[Any, str]:
    if not policy.move_activation_output_qdq_after_relu:
        return node, output_name
    direct = consumers.get(str(output_name), [])
    if len(direct) != 1:
        return node, output_name
    first = direct[0]
    if str(first.op_type) == "Relu" and len(first.output) == 1:
        return first, str(first.output[0])
    if str(first.op_type) == "Add" and len(first.output) == 1:
        add_output = str(first.output[0])
        second = consumers.get(add_output, [])
        if len(second) == 1 and str(second[0].op_type) == "Relu" and len(second[0].output) == 1:
            return second[0], str(second[0].output[0])
        return first, add_output
    return node, output_name


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
    index = build_weight_trace_index(model)
    nodes_by_name = index["nodes_by_name"]
    consumers = _consumer_index(model.graph.node)
    requested = [row for row in mapping.entries if row.requested_precision == "int8"]
    targets = {row.canonical_node_name: row for row in mapping.entries if row.realized_request_precision == "int8"}
    if len(targets) != sum(row.realized_request_precision == "int8" for row in mapping.entries):
        raise QDQInsertionError("canonical INT8 target names are not unique")
    prepared: dict[str, tuple[Any, float, float, float, dict[str, Any]]] = {}
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
        prepared[name] = (entry, activation_input_scale, weight_scale, activation_output_scale, metadata)

    new_nodes: list[Any] = []
    records: list[QDQInsertionRecord] = []
    deferred_output_qdq: dict[str, list[tuple[str, str, float, int, QDQInsertionRecord]]] = {}
    planned_output_boundaries: set[tuple[str, str]] = set()
    for node in model.graph.node:
        target = prepared.get(str(node.name))
        local_output_specs: list[tuple[str, str, str, float, int, QDQInsertionRecord]] = []
        if target is None:
            for output_index, output_name in enumerate(list(node.output)):
                boundary_key = (str(node.name), str(output_name))
                pending = deferred_output_qdq.get(str(node.name), [])
                matching = [row for row in pending if row[0] == str(output_name)]
                if not matching:
                    continue
                raw_name = f"{output_name}__before_output_qdq"
                node.output[output_index] = raw_name
                for _public_name, safe, activation_output_scale, zero_point, record in matching:
                    local_output_specs.append((raw_name, str(output_name), safe, activation_output_scale, zero_point, record))
                planned_output_boundaries.add(boundary_key)
            new_nodes.append(node)
            for raw_name, public_name, safe, activation_output_scale, zero_point, record in local_output_specs:
                pair, _dequantized, q_name, dq_name = _qdq_pair(
                    helper, numpy_helper, np, model.graph, raw_name, f"{safe}__activation_output_moved", activation_output_scale, zero_point
                )
                pair[-1].output[0] = public_name
                new_nodes.extend(pair)
                record.output_quantize_nodes.append(q_name)
                record.output_dequantize_nodes.append(dq_name)
            continue
        entry, activation_input_scale, weight_scale, activation_output_scale, _metadata = target
        safe = str(node.name).replace("/", "_").replace(".", "_")
        record = QDQInsertionRecord(
            module_path=entry.module_path,
            canonical_node_name=entry.canonical_node_name,
            weight_initializer=entry.weight_initializer,
            scale=activation_input_scale,
            activation_input_scale=activation_input_scale,
            weight_scale=weight_scale,
            activation_output_scale=activation_output_scale,
            zero_point=int(policy.zero_point),
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
                helper, numpy_helper, np, model.graph, str(node.input[1]), f"{safe}__weight", weight_scale, policy.zero_point
            )
            new_nodes.extend(pair)
            node.input[1] = dequantized
            record.weight_quantize_node = q_name
            record.weight_dequantize_node = dq_name
        output_qdq_enabled = (
            policy.insert_activation_output_qdq
            and entry.module_path not in set(policy.activation_output_qdq_excluded_modules)
        )
        if output_qdq_enabled:
            for output_index, output_name in enumerate(list(node.output)):
                boundary_node, boundary_output = _activation_output_boundary(node, str(output_name), consumers, policy)
                boundary_key = (str(boundary_node.name), str(boundary_output))
                if boundary_key in planned_output_boundaries:
                    continue
                planned_output_boundaries.add(boundary_key)
                if str(boundary_node.name) == str(node.name) and str(boundary_output) == str(output_name):
                    raw_name = f"{output_name}__before_output_qdq"
                    node.output[output_index] = raw_name
                    local_output_specs.append((raw_name, str(output_name), safe, activation_output_scale, int(policy.zero_point), record))
                else:
                    deferred_output_qdq.setdefault(str(boundary_node.name), []).append(
                        (str(boundary_output), safe, activation_output_scale, int(policy.zero_point), record)
                    )
        new_nodes.append(node)
        for output_index, (raw_name, public_name, output_safe, output_scale, zero_point, output_record) in enumerate(local_output_specs):
            pair, dequantized, q_name, dq_name = _qdq_pair(
                helper, numpy_helper, np, model.graph, raw_name, f"{output_safe}__activation_output_{output_index}", output_scale, zero_point
            )
            pair[-1].output[0] = public_name
            new_nodes.extend(pair)
            output_record.output_quantize_nodes.append(q_name)
            output_record.output_dequantize_nodes.append(dq_name)
        records.append(record)
    if len(records) != len(targets):
        raise QDQInsertionError("not every canonical INT8 target received Q/DQ")
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
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
