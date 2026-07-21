"""Requested, ONNX, and TensorRT dtype provenance for CoBEVT Attention."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def _dtype_name(value: int | None) -> str:
    from onnx import TensorProto

    return {
        int(TensorProto.FLOAT): "FP32",
        int(TensorProto.FLOAT16): "FP16",
        int(TensorProto.INT8): "INT8",
        int(TensorProto.INT32): "INT32",
        int(TensorProto.INT64): "INT64",
        int(TensorProto.BOOL): "BOOL",
    }.get(int(value) if value is not None else -1, "UNRESOLVED")


def _onnx_types(model: Any) -> dict[str, int]:
    types = {
        str(initializer.name): int(initializer.data_type)
        for initializer in model.graph.initializer
    }
    for value in [*model.graph.input, *model.graph.value_info, *model.graph.output]:
        if value.type.tensor_type.elem_type:
            types[str(value.name)] = int(value.type.tensor_type.elem_type)
    return types


def _trt_dtype(value: str) -> str:
    normalized = str(value).strip().lower()
    if "half" in normalized:
        return "FP16"
    if "float" in normalized:
        return "FP32"
    if "int8" in normalized:
        return "INT8"
    if "int32" in normalized:
        return "INT32"
    if "bool" in normalized:
        return "BOOL"
    return str(value) or "UNRESOLVED"


def _module_path(block_id: str, role: str) -> str:
    prefix = f"fusion_net.{block_id}"
    projection = {
        "q_projection": "q_proj",
        "k_projection": "k_proj",
        "v_projection": "v_proj",
        "output_projection": "out_proj",
    }.get(str(role))
    if projection:
        return f"{prefix}.fn.{projection}"
    if role == "layernorm":
        return f"{prefix}.norm"
    if role == "residual_add":
        return f"{prefix}.residual_add"
    return f"{prefix}.fn::{role}"


def _matching_engine_layers(layer_rows: Sequence[Mapping[str, Any]], node_name: str):
    result = []
    for row in layer_rows:
        text = " ".join(
            str(row.get(key, "")) for key in ("Name", "Metadata", "TacticName")
        )
        if str(node_name) in text:
            result.append(dict(row))
    return result


def _primary_engine_layer(rows: Sequence[Mapping[str, Any]], op_type: str):
    if str(op_type) in {"MatMul", "Einsum"}:
        gemm = [row for row in rows if str(row.get("LayerType", "")).lower() == "gemm"]
        if gemm:
            return gemm[0]
    return rows[0] if rows else None


def _tactic_precision(tactic: str) -> str:
    value = str(tactic).lower()
    if "i8" in value or "int8" in value:
        return "INT8"
    if "f16" in value or "half" in value:
        return "FP16"
    if "f32" in value or "float" in value:
        return "FP32"
    return "UNRESOLVED"


def build_attention_precision_inventory(
    typed_onnx: str | Path,
    boundary_report: Mapping[str, Any],
    layer_info_path: str | Path | None = None,
    *,
    requested_precision_overrides: Mapping[str, str] | None = None,
    node_name_overrides: Mapping[str, str] | None = None,
    ignored_node_names: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    import onnx

    model = onnx.load(str(typed_onnx))
    try:
        model = onnx.shape_inference.infer_shapes(
            model, strict_mode=False, data_prop=True
        )
    except Exception:
        pass
    types = _onnx_types(model)
    nodes = {str(node.name): node for node in model.graph.node if str(node.name)}
    layer_rows: list[Mapping[str, Any]] = []
    if layer_info_path is not None:
        payload = json.loads(Path(layer_info_path).read_text(encoding="utf-8"))
        layer_rows = list(payload.get("Layers", payload if isinstance(payload, list) else []))
    records = []
    requested_overrides = {
        str(module_path): str(precision).upper()
        for module_path, precision in (requested_precision_overrides or {}).items()
    }
    node_overrides = {
        str(node_name): str(replacement)
        for node_name, replacement in (node_name_overrides or {}).items()
    }
    ignored_nodes = {str(value) for value in (ignored_node_names or ())}
    weighted_roles = {
        "q_projection",
        "k_projection",
        "v_projection",
        "output_projection",
    }
    for boundary in boundary_report.get("node_records", []):
        role = str(boundary["role"])
        module_path = _module_path(str(boundary["block_id"]), role)
        original_node_name = str(boundary["node_name"])
        if original_node_name in ignored_nodes:
            continue
        node_name = node_overrides.get(original_node_name, original_node_name)
        node = nodes.get(node_name)
        if node is None:
            raise ValueError(f"attention_inventory_onnx_node_missing:{node_name}")
        matched = _matching_engine_layers(layer_rows, node_name)
        primary = _primary_engine_layer(matched, str(node.op_type))
        metadata = " ".join(str(row.get("Metadata", "")) for row in matched)
        fused = metadata.count("[ONNX Layer:") > 1
        requested = requested_overrides.get(
            module_path, str(boundary["compute_dtype"])
        )
        onnx_inputs = [_dtype_name(types.get(str(value))) for value in node.input]
        onnx_outputs = [_dtype_name(types.get(str(value))) for value in node.output]
        realized_inputs = []
        realized_outputs = []
        tactic_precision = "UNRESOLVED"
        if primary is not None:
            realized_rows = (
                [primary]
                if str(node.op_type) in {"MatMul", "Einsum"}
                else list(matched)
            )
            realized_inputs = sorted(
                {
                    _trt_dtype(value.get("Format/Datatype", ""))
                    for row in realized_rows
                    for value in row.get("Inputs", [])
                }
            )
            realized_outputs = sorted(
                {
                    _trt_dtype(value.get("Format/Datatype", ""))
                    for row in realized_rows
                    for value in row.get("Outputs", [])
                }
            )
            tactic_precision = _tactic_precision(str(primary.get("TacticName", "")))
        if primary is None:
            status = "unresolved"
            failure_reason = "engine_layer_not_found"
        else:
            compute_match = tactic_precision == requested or requested in realized_inputs
            expected_output = str(boundary.get("output_dtype", requested))
            output_match = expected_output in realized_outputs
            if compute_match and output_match:
                status = "matched"
                failure_reason = ""
            elif (
                compute_match
                and bool(boundary.get("output_cast_nodes", []))
                and requested in realized_outputs
                and set(onnx_outputs) == {requested}
            ):
                status = "matched_separate_output_cast"
                failure_reason = ""
            elif compute_match and fused:
                status = "matched_fused_output_unexposed"
                failure_reason = ""
            elif (
                fused
                and output_match
                and {
                    value
                    for value in onnx_inputs
                    if value in {"FP16", "FP32", "INT8"}
                }
                == {requested}
            ):
                status = "matched_fused_input_unexposed"
                failure_reason = ""
            else:
                status = "mismatched"
                failure_reason = "realized_dtype_mismatch"
        cast_names = [
            *boundary.get("input_cast_nodes", []),
            *boundary.get("output_cast_nodes", []),
        ]
        cast_records = []
        for cast_name in cast_names:
            cast = nodes.get(str(cast_name))
            if cast is None:
                raise ValueError(f"attention_inventory_cast_node_missing:{cast_name}")
            target = next(
                (
                    int(attribute.i)
                    for attribute in cast.attribute
                    if str(attribute.name) == "to"
                ),
                None,
            )
            cast_records.append(
                {
                    "cast_node": str(cast.name),
                    "input_tensors": list(cast.input),
                    "output_tensors": list(cast.output),
                    "source_dtype": _dtype_name(types.get(str(cast.input[0]))),
                    "target_dtype": _dtype_name(target),
                }
            )
        records.append(
            {
                "block_id": str(boundary["block_id"]),
                "cast_records": cast_records,
                "engine_layer_names": [str(row.get("Name", "")) for row in matched],
                "explicit_cast": bool(cast_records),
                "explicit_input_cast": bool(boundary.get("input_cast_nodes", [])),
                "explicit_output_cast": bool(boundary.get("output_cast_nodes", [])),
                "functional_op": role not in weighted_roles,
                "fused": fused,
                "module_path": module_path,
                "node_name": node_name,
                "original_node_name": original_node_name,
                "onnx_input_dtypes": onnx_inputs,
                "onnx_input_tensors": list(node.input),
                "onnx_output_dtypes": onnx_outputs,
                "onnx_output_tensors": list(node.output),
                "op_type": str(node.op_type),
                "realization_failure_reason": failure_reason,
                "realization_status": status,
                "realized_input_dtypes": realized_inputs,
                "realized_output_dtypes": realized_outputs,
                "requested_dtype": requested,
                "role": role,
                "tactic_names": [str(row.get("TacticName", "")) for row in matched],
                "tactic_precision": tactic_precision,
                "weighted_op": role in weighted_roles,
            }
        )
    return records


def write_attention_precision_inventory(
    rows: Sequence[Mapping[str, Any]],
    json_path: str | Path,
    markdown_path: str | Path,
) -> dict[str, Any]:
    payload = [dict(row) for row in rows]
    json_destination = Path(json_path)
    markdown_destination = Path(markdown_path)
    json_destination.parent.mkdir(parents=True, exist_ok=True)
    markdown_destination.parent.mkdir(parents=True, exist_ok=True)
    json_destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = ["# CoBEVT Attention Precision Inventory", ""]
    current = None
    for row in payload:
        block = str(row.get("block_id", ""))
        if block != current:
            lines.extend(
                [
                    f"## {block}",
                    "",
                    "| role | ONNX node | op | requested | ONNX input/output | TRT input/output | fused | status |",
                    "|---|---|---|---|---|---|---:|---|",
                ]
            )
            current = block
        lines.append(
            "| {role} | `{node}` | {op} | {requested} | {onnx_in} / {onnx_out} | "
            "{trt_in} / {trt_out} | {fused} | {status} |".format(
                role=row.get("role", ""),
                node=row.get("node_name", ""),
                op=row.get("op_type", ""),
                requested=row.get("requested_dtype", ""),
                onnx_in=row.get("onnx_input_dtypes", []),
                onnx_out=row.get("onnx_output_dtypes", []),
                trt_in=row.get("realized_input_dtypes", []),
                trt_out=row.get("realized_output_dtypes", []),
                fused=row.get("fused", False),
                status=row.get("realization_status", ""),
            )
        )
    markdown_destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "json_path": str(json_destination),
        "markdown_path": str(markdown_destination),
        "row_count": len(payload),
        "unresolved_count": sum(
            str(row.get("realization_status", "")) == "unresolved" for row in payload
        ),
        "mismatch_count": sum(
            str(row.get("realization_status", "")) == "mismatched" for row in payload
        ),
    }


__all__ = [
    "build_attention_precision_inventory",
    "write_attention_precision_inventory",
]
