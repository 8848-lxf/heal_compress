"""ONNX/TensorRT precision evidence for Transformer Stage-2 candidates."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, Mapping, Sequence

from quantization.tensorrt.layer_info import (
    has_canonical_identity,
    layer_metadata,
    load_layer_info,
    precision_name,
)
from quantization.types import (
    CanonicalPrecisionEntry,
    CanonicalPrecisionMappingResult,
    stable_json_hash,
)


def build_transformer_precision_mapping(
    origin_map: Any,
    module_precision_profile: Mapping[str, str],
    *,
    profile_id: str,
    policy_version: str = "transformer-explicit-fp32-fp16-qdq-v1",
) -> CanonicalPrecisionMappingResult:
    """Expand exact module precision requests to all realized ONNX calls."""

    profile = {
        str(path): str(precision).lower()
        for path, precision in module_precision_profile.items()
    }
    origin_modules = {str(row.module_path) for row in origin_map.entries}
    missing = sorted(origin_modules - set(profile))
    unknown = sorted(set(profile) - origin_modules)
    if missing or unknown:
        raise RuntimeError(
            "transformer_precision_profile_origin_mismatch:"
            f"missing={missing}:unknown={unknown}"
        )
    invalid = {
        path: precision
        for path, precision in profile.items()
        if precision not in {"fp32", "fp16", "int8"}
    }
    if invalid:
        raise RuntimeError(f"transformer_precision_profile_invalid:{invalid}")
    entries = []
    for origin in sorted(
        origin_map.entries, key=lambda row: (int(row.graph_index), int(row.call_index))
    ):
        precision = profile[str(origin.module_path)]
        entries.append(
            CanonicalPrecisionEntry(
                module_path=str(origin.module_path),
                canonical_node_name=str(origin.canonical_node_name),
                original_node_name=str(origin.original_node_name),
                weight_initializer=str(origin.weight_initializer),
                onnx_op_type=str(origin.onnx_op_type),
                call_index=int(origin.call_index),
                precision_group=f"transformer_qg::module::{origin.module_path}",
                requested_precision=precision,
                realized_request_precision=precision,
                realized_output_precision=(
                    "fp16" if precision in {"fp16", "int8"} else "fp32"
                ),
            )
        )
    return CanonicalPrecisionMappingResult(
        entries=entries,
        profile_id=str(profile_id),
        profile_hash=stable_json_hash(profile),
        origin_map_hash=str(origin_map.origin_map_hash),
        policy_version=str(policy_version),
    )


def _tensor_element_types(model: Any) -> dict[str, int]:
    import onnx

    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=False)
    result: dict[str, int] = {}
    values = (
        list(inferred.graph.input)
        + list(inferred.graph.output)
        + list(inferred.graph.value_info)
    )
    for value in values:
        tensor_type = value.type.tensor_type
        if tensor_type.HasField("elem_type"):
            result[str(value.name)] = int(tensor_type.elem_type)
    for initializer in inferred.graph.initializer:
        result[str(initializer.name)] = int(initializer.data_type)
    return result


def _nearest_nodes(
    start_tensors: Sequence[str],
    *,
    consumers: Mapping[str, Sequence[Any]],
    accepted_ops: set[str],
    stop_ops: set[str] | None = None,
    maximum_depth: int = 64,
) -> list[Any]:
    queue = deque((str(tensor), 0) for tensor in start_tensors)
    seen: set[str] = set()
    found: dict[str, Any] = {}
    nearest_depth: int | None = None
    while queue:
        tensor, depth = queue.popleft()
        if tensor in seen or depth > int(maximum_depth):
            continue
        if nearest_depth is not None and depth >= nearest_depth:
            continue
        seen.add(tensor)
        for node in consumers.get(tensor, ()):
            op_type = str(node.op_type)
            if op_type in accepted_ops:
                accepted_depth = depth + 1
                if nearest_depth is None:
                    nearest_depth = accepted_depth
                if accepted_depth != nearest_depth:
                    continue
                found[str(node.name)] = node
                continue
            if stop_ops and op_type in stop_ops:
                continue
            queue.extend((str(output), depth + 1) for output in node.output)
    return [found[key] for key in sorted(found)]


def audit_onnx_attention_fp32_contract(
    onnx_path: str | Path,
    *,
    qkv_canonical_node_names: Sequence[str],
) -> dict[str, Any]:
    """Prove selected QK and Softmax tensors are FLOAT by graph topology."""

    import onnx
    from onnx import TensorProto

    model = onnx.load(str(onnx_path), load_external_data=False)
    nodes = {str(node.name): node for node in model.graph.node}
    requested_qkv = [str(value) for value in qkv_canonical_node_names]
    missing_qkv = sorted(set(requested_qkv) - set(nodes))
    if missing_qkv:
        raise RuntimeError(f"attention_qkv_onnx_nodes_missing:{missing_qkv}")
    consumers: dict[str, list[Any]] = {}
    producers: dict[str, Any] = {}
    for node in model.graph.node:
        for input_name in node.input:
            consumers.setdefault(str(input_name), []).append(node)
        for output_name in node.output:
            producers[str(output_name)] = node
    qkv_outputs = [
        str(output) for name in requested_qkv for output in nodes[name].output
    ]
    # Attention families can have different graph depths (for example V2X-ViT
    # HGT relation attention versus window attention).  A single multi-source
    # BFS keeps only the globally nearest depth and therefore silently drops
    # valid, deeper instances.  Resolve the nearest Softmax independently for
    # every Q/K/V projection output and then deduplicate by node identity.
    softmax_by_name: dict[str, Any] = {}
    for output in qkv_outputs:
        for node in _nearest_nodes(
            (output,),
            consumers=consumers,
            accepted_ops={"Softmax"},
            maximum_depth=96,
        ):
            softmax_by_name[str(node.name)] = node
    softmax_nodes = [softmax_by_name[key] for key in sorted(softmax_by_name)]
    if not softmax_nodes:
        raise RuntimeError("attention_softmax_not_reachable_from_qkv")

    def upstream_compute(tensor_name: str) -> Any | None:
        queue = deque([(str(tensor_name), 0)])
        seen: set[str] = set()
        while queue:
            tensor, depth = queue.popleft()
            if tensor in seen or depth > 64:
                continue
            seen.add(tensor)
            producer = producers.get(tensor)
            if producer is None:
                continue
            if str(producer.op_type) in {"MatMul", "Einsum"}:
                return producer
            queue.extend((str(value), depth + 1) for value in producer.input)
        return None

    qk_nodes: dict[str, Any] = {}
    av_nodes: dict[str, Any] = {}
    for softmax in softmax_nodes:
        qk = upstream_compute(str(softmax.input[0]))
        if qk is None:
            raise RuntimeError(f"attention_qk_not_found_before_softmax:{softmax.name}")
        qk_nodes[str(qk.name)] = qk
        for av in _nearest_nodes(
            tuple(str(value) for value in softmax.output),
            consumers=consumers,
            accepted_ops={"MatMul", "Einsum"},
            stop_ops={"Softmax"},
            maximum_depth=64,
        ):
            av_nodes[str(av.name)] = av
    if not av_nodes:
        raise RuntimeError("attention_av_not_found_after_softmax")
    element_types = _tensor_element_types(model)

    def row(node: Any) -> dict[str, Any]:
        inputs = [element_types.get(str(value), 0) for value in node.input]
        outputs = [element_types.get(str(value), 0) for value in node.output]
        return {
            "node_name": str(node.name),
            "op_type": str(node.op_type),
            "input_element_types": inputs,
            "output_element_types": outputs,
            "input_dtypes": [
                TensorProto.DataType.Name(value) if value else "UNKNOWN"
                for value in inputs
            ],
            "output_dtypes": [
                TensorProto.DataType.Name(value) if value else "UNKNOWN"
                for value in outputs
            ],
            "all_inputs_fp32": bool(inputs)
            and all(value == TensorProto.FLOAT for value in inputs),
            "all_outputs_fp32": bool(outputs)
            and all(value == TensorProto.FLOAT for value in outputs),
        }

    qk_rows = [row(qk_nodes[key]) for key in sorted(qk_nodes)]
    softmax_rows = [row(node) for node in softmax_nodes]
    av_rows = [row(av_nodes[key]) for key in sorted(av_nodes)]
    passed = bool(qk_rows and softmax_rows) and all(
        item["all_inputs_fp32"] and item["all_outputs_fp32"]
        for item in (*qk_rows, *softmax_rows)
    )
    return {
        "schema_version": "transformer-onnx-attention-fp32-contract-v1",
        "passed": passed,
        "qk_operands_fp32": all(item["all_inputs_fp32"] for item in qk_rows),
        "qk_compute_output_fp32": all(item["all_outputs_fp32"] for item in qk_rows),
        "softmax_compute_fp32": all(
            item["all_inputs_fp32"] and item["all_outputs_fp32"]
            for item in softmax_rows
        ),
        "qk_nodes": qk_rows,
        "softmax_nodes": softmax_rows,
        "av_nodes": av_rows,
    }


def audit_trt_attention_fp32_contract(
    layer_info: str | Path | Sequence[Mapping[str, Any]] | Mapping[str, Any],
    onnx_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Match ONNX QK/Softmax identities to TensorRT inspector precision."""

    rows = load_layer_info(layer_info)
    requested_nodes = [
        dict(item)
        for key in ("qk_nodes", "softmax_nodes")
        for item in onnx_contract.get(key, ())
    ]
    evidence = []
    for requested in requested_nodes:
        name = str(requested["node_name"])
        exact_identity = f"[ONNX Layer: {name}]"
        exact_matches = [
            row
            for row in rows
            if str(row.get("Metadata") or row.get("metadata") or "").strip()
            == exact_identity
        ]
        # TensorRT can fuse an upstream FP16 projection and a downstream FP32
        # QK boundary into one broad metadata record.  Such a record contains
        # the QK identity but is not the QK compute layer.  Prefer inspector
        # rows whose metadata contains exactly the requested ONNX identity;
        # retain the broader fallback for tactics that expose no exact row.
        matches = exact_matches or [row for row in rows if has_canonical_identity(row, name)]
        if not matches:
            matches = [row for row in rows if name and name in layer_metadata(row)]
        realized = sorted(
            {precision_name(row) for row in matches if precision_name(row)}
        )
        input_formats = sorted(
            {
                str(item.get("Format/Datatype") or item.get("format") or "").lower()
                for row in matches
                for item in (row.get("Inputs") or row.get("inputs") or ())
                if isinstance(item, Mapping)
            }
        )
        output_formats = sorted(
            {
                str(item.get("Format/Datatype") or item.get("format") or "").lower()
                for row in matches
                for item in (row.get("Outputs") or row.get("outputs") or ())
                if isinstance(item, Mapping)
            }
        )
        op_type = str(requested["op_type"])
        # TensorRT may fuse a FLOAT Softmax with its requested A16/A8 output
        # cast/QDQ.  In that case layer-level ``Precision`` describes the
        # fused output and cannot be used as the compute precision.  FLOAT
        # inspector inputs plus the explicit ONNX compute contract prove the
        # Softmax compute is FP32 while output_formats records A16/A8
        # separately.
        if op_type == "Softmax":
            passed = bool(matches) and (
                (bool(input_formats) and all(value == "float" for value in input_formats))
                or (not input_formats and realized == ["fp32"])
            )
        else:
            passed = bool(matches) and realized == ["fp32"]
        evidence.append(
            {
                "onnx_node_name": name,
                "onnx_op_type": op_type,
                "inspector_match_count": len(matches),
                "realized_precisions": realized,
                "input_formats": input_formats,
                "output_formats": output_formats,
                "softmax_compute_precision": "fp32" if op_type == "Softmax" and passed else "unresolved",
                "softmax_output_precision": (
                    "int8" if any("int8" in value for value in output_formats)
                    else "fp16" if any("half" in value or "float16" in value for value in output_formats)
                    else "fp32" if output_formats and all(value == "float" for value in output_formats)
                    else "unresolved"
                ) if op_type == "Softmax" else "",
                "passed": passed,
                "inspector_metadata": [layer_metadata(row) for row in matches],
            }
        )
    return {
        "schema_version": "transformer-trt-attention-fp32-contract-v1",
        "passed": bool(evidence) and all(item["passed"] for item in evidence),
        "qk_fp32_protected": all(
            item["passed"]
            for item in evidence
            if item["onnx_op_type"] in {"MatMul", "Einsum"}
        ),
        "softmax_compute_fp32": all(
            item["passed"]
            for item in evidence
            if item["onnx_op_type"] == "Softmax"
        ),
        "evidence": evidence,
    }


__all__ = [
    "audit_onnx_attention_fp32_contract",
    "audit_trt_attention_fp32_contract",
    "build_transformer_precision_mapping",
]
