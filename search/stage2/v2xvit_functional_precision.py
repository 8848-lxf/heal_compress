"""Canonical ONNX/TensorRT evidence for V2X-ViT functional precision units.

Weighted Conv/Linear realization is validated by the canonical precision
checker.  This module closes the complementary, parameter-free boundaries
that are otherwise invisible to a weighted origin map: QK, Softmax output,
AV, LayerNorm, residual/merge, and the FFN2 activation input.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from quantization.tensorrt.layer_info import (
    has_canonical_identity,
    layer_metadata,
    layer_name,
    load_layer_info,
    precision_name,
)

from .transformer_precision_export import audit_onnx_attention_fp32_contract


def requested_states_from_phenotype(
    phenotype: Any, precision_units: Sequence[Any]
) -> dict[str, str]:
    """Resolve group-level requests without inventing functional genes."""

    internal_to_weight = {"FP32": "W32A32", "FP16": "W16A16", "INT8": "W8A8"}
    internal_to_activation = {"FP32": "A32", "FP16": "A16", "INT8": "A8"}
    profile = {
        str(path): str(value).upper()
        for path, value in phenotype.realized_precision_profile.items()
    }
    result: dict[str, str] = {}
    derived = dict(phenotype.metadata.get("derived_precision_group_profile", {}))
    for unit in precision_units:
        if bool(unit.activation_only):
            if unit.role == "attention_merge" and str(unit.unit_id) in derived:
                internal = str(derived[str(unit.unit_id)]["derived_precision"]).upper()
                result[str(unit.unit_id)] = internal_to_activation[internal]
            elif unit.role == "av_matmul":
                values = {
                    value
                    for path, value in profile.items()
                    if any(
                        path == owner or path.endswith(f".{owner}")
                        for owner in unit.module_paths
                    )
                }
                if len(values) != 1:
                    raise RuntimeError(
                        f"v2xvit_av_precision_group_unresolved:"
                        f"{unit.unit_id}:{sorted(values)}"
                    )
                result[str(unit.unit_id)] = internal_to_activation[values.pop()]
            else:
                result[str(unit.unit_id)] = str(unit.default_state).upper()
            continue
        values = {
            value
            for path, value in profile.items()
            if any(path == owner or path.endswith(f".{owner}") for owner in unit.module_paths)
        }
        if len(values) != 1:
            raise RuntimeError(
                f"v2xvit_weighted_precision_group_unresolved:{unit.unit_id}:{sorted(values)}"
            )
        result[str(unit.unit_id)] = internal_to_weight[values.pop()]
    return result


def _dtype_name(value: int) -> str:
    from onnx import TensorProto

    return {
        int(TensorProto.FLOAT): "FP32",
        int(TensorProto.FLOAT16): "FP16",
        int(TensorProto.INT8): "INT8",
        int(TensorProto.BOOL): "BOOL",
        int(TensorProto.INT64): "INT64",
    }.get(int(value), "UNRESOLVED")


def _tensor_element_types(model: Any) -> dict[str, int]:
    import onnx

    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=False)
    values = (
        list(inferred.graph.input)
        + list(inferred.graph.output)
        + list(inferred.graph.value_info)
    )
    result = {
        str(value.name): int(value.type.tensor_type.elem_type)
        for value in values
        if value.type.tensor_type.HasField("elem_type")
    }
    result.update(
        {str(value.name): int(value.data_type) for value in inferred.graph.initializer}
    )
    return result


def _node_row(node: Any, element_types: Mapping[str, int]) -> dict[str, Any]:
    inputs = [_dtype_name(element_types.get(str(value), 0)) for value in node.input]
    outputs = [_dtype_name(element_types.get(str(value), 0)) for value in node.output]
    return {
        "onnx_node": str(node.name),
        "onnx_op_type": str(node.op_type),
        "input_types": inputs,
        "output_types": outputs,
    }


def _encoder_layer_index(path: str) -> int:
    match = re.search(
        r"(?:^|\.)(?:encoder\.)?layers\.(\d+)(?:\.|$)", str(path)
    )
    if match is None:
        raise RuntimeError(f"v2xvit_encoder_layer_index_unresolved:{path}")
    return int(match.group(1))


def _module_onnx_prefix(path: str) -> str:
    marker = ".encoder."
    text = str(path)
    if marker in text:
        tokens = text.split(marker, 1)[1].split(".")
    elif text.startswith("fusion_net.mlp_head."):
        # The standalone CoBEVT head is exported below an additional
        # ``/mlp_head`` scope while preserving ``mlp_head.<index>`` as the
        # module token.
        return f"/mlp_head/{text.split('fusion_net.', 1)[1]}"
    elif text.startswith("fusion_net.layers."):
        # CoBEVT exports ``fusion_net`` as the graph root.  Its module path
        # ``fusion_net.layers.0.window_attention.fn`` therefore becomes the
        # stable ONNX prefix ``/layers.0/window_attention/fn``.
        tokens = text.split("fusion_net.", 1)[1].split(".")
    else:
        raise RuntimeError(f"transformer_module_owner_unresolved:{path}")
    chunks: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == "layers":
            values = [token]
            index += 1
            while index < len(tokens) and tokens[index].isdigit():
                values.append(tokens[index])
                index += 1
            chunks.append(".".join(values))
            continue
        chunks.append(token)
        index += 1
    return f"/{'/'.join(chunks)}"


def _layernorm_onnx_name(path: str) -> str:
    return f"{_module_onnx_prefix(path)}/LayerNormalization"


def _residual_onnx_name(unit: Any) -> str:
    owner = str(unit.metadata.get("functional_owner", ""))
    if owner.startswith("fusion_net.layers."):
        # CoBEVT emits the residual Add at the normalized wrapper level, one
        # level above ``.fn`` for both Attention and FFN blocks.
        wrapper = owner.rsplit(".fn", 1)[0] if owner.endswith(".fn") else owner
        return f"{_module_onnx_prefix(wrapper)}/Add"
    layer = _encoder_layer_index(owner)
    kind = str(unit.metadata.get("boundary_kind", ""))
    if kind == "ffn_residual_add":
        return f"/Add_{layer + 2}"
    adapter = str(unit.metadata.get("attention_adapter", ""))
    return f"/layers.{layer}.0/{'Add' if adapter == 'v2xvit_hgt' else 'Add_1'}"


def fixed_functional_onnx_precision_overrides(
    precision_units: Sequence[Any],
    requested_states: Mapping[str, str],
) -> dict[str, str]:
    """Return exact fixed residual precision overrides for canonical ONNX.

    Runtime merge tracing derives a safe join from weighted producer dtypes,
    but a protected residual unit is a stronger deployment contract.  This
    function bridges the functional unit identity to the concrete ONNX Add so
    Q/DQ insertion casts the Add inputs to the already-requested A16/A32
    precision.  It does not create a gene or change any requested state.
    """

    state_to_precision = {"A16": "fp16", "A32": "fp32"}
    overrides: dict[str, str] = {}
    for unit in precision_units:
        if str(unit.role) != "residual_add":
            continue
        state = str(requested_states.get(unit.unit_id, unit.default_state)).upper()
        if state not in state_to_precision:
            raise RuntimeError(
                f"functional_residual_precision_unsupported:{unit.unit_id}:{state}"
            )
        node_name = _residual_onnx_name(unit)
        precision = state_to_precision[state]
        previous = overrides.get(node_name)
        if previous is not None and previous != precision:
            raise RuntimeError(
                f"functional_residual_precision_conflict:{node_name}:{previous}:{precision}"
            )
        overrides[node_name] = precision
    return dict(sorted(overrides.items()))


def _functional_expected(unit: Any, requested_state: str | None = None) -> tuple[str, str]:
    state = str(requested_state or unit.default_state).upper()
    if unit.role in {"qk_matmul", "softmax", "layernorm"}:
        return "FP32", "FP32"
    if unit.role == "av_matmul":
        precision = {"A32": "FP32", "A16": "FP16", "A8": "INT8"}[state]
        output = "FP16" if state == "A8" else precision
        return precision, output
    if unit.role == "residual_add":
        return "FP16", "FP16"
    if unit.role == "attention_merge":
        precision = {"A32": "FP32", "A16": "FP16", "A8": "INT8"}[state]
        return precision, precision
    raise RuntimeError(f"v2xvit_functional_role_unsupported:{unit.unit_id}:{unit.role}")


def build_v2xvit_functional_onnx_mapping(
    onnx_path: str | Path,
    *,
    origin_map: Any,
    precision_units: Sequence[Any],
    attention_instances: Sequence[Any],
    ffn_instances: Sequence[Any],
    requested_states: Mapping[str, str],
) -> dict[str, Any]:
    """Map every functional contract to one exact ONNX identity, fail closed."""

    import onnx

    model = onnx.load(str(onnx_path), load_external_data=False)
    nodes = {str(node.name): node for node in model.graph.node}
    element_types = _tensor_element_types(model)
    units = {str(unit.unit_id): unit for unit in precision_units}
    rows: list[dict[str, Any]] = []

    def add(unit: Any, node: Any, *, source: str) -> None:
        requested_state = str(
            requested_states.get(unit.unit_id, unit.default_state)
        ).upper()
        compute, output = _functional_expected(unit, requested_state)
        row = {
            "unit_id": str(unit.unit_id),
            "role": str(unit.role),
            "requested_state": requested_state,
            "requested_compute_precision": compute,
            "requested_output_precision": output,
            "source": source,
            **_node_row(node, element_types),
        }
        row["onnx_contract_exact"] = (
            compute in row["input_types"]
            and bool(row["output_types"])
            and all(value == output for value in row["output_types"])
        )
        if unit.role == "av_matmul" and requested_state == "A8":
            producers = {
                str(output_name): producer
                for producer in model.graph.node
                for output_name in producer.output
            }
            dq_inputs = [
                str(producers.get(str(input_name), object()).op_type)
                if producers.get(str(input_name)) is not None else ""
                for input_name in node.input
            ]
            row["av_operand_qdq"] = dq_inputs
            consumers = [
                consumer
                for output_name in node.output
                for consumer in model.graph.node
                if str(output_name) in tuple(str(value) for value in consumer.input)
            ]
            output_casts = [
                consumer for consumer in consumers if str(consumer.op_type) == "Cast"
            ]
            cast_output_types = [
                _dtype_name(element_types.get(str(output_name), 0))
                for consumer in output_casts
                for output_name in consumer.output
            ]
            row["av_output_cast_nodes"] = [str(value.name) for value in output_casts]
            row["av_output_boundary_types"] = cast_output_types
            row["onnx_contract_exact"] = (
                len(dq_inputs) == 2
                and all(value == "DequantizeLinear" for value in dq_inputs)
                and len(output_casts) == 1
                and cast_output_types == ["FP16"]
            )
        rows.append(row)

    origins_by_module: dict[str, list[str]] = {}
    for entry in origin_map.entries:
        origins_by_module.setdefault(str(entry.module_path), []).append(
            str(entry.canonical_node_name)
        )

    for spec in attention_instances:
        paths = set(
            spec.q_projection_paths + spec.k_projection_paths + spec.v_projection_paths
        )
        qkv_names = sorted(
            name
            for path, names in origins_by_module.items()
            if path in paths or any(path.endswith(f".{wanted}") for wanted in paths)
            for name in names
        )
        if not qkv_names:
            raise RuntimeError(f"v2xvit_attention_qkv_mapping_missing:{spec.module_path}")
        audit = audit_onnx_attention_fp32_contract(
            onnx_path, qkv_canonical_node_names=qkv_names
        )
        for role, key in (
            ("qk_matmul", "qk_nodes"),
            ("softmax", "softmax_nodes"),
            ("av_matmul", "av_nodes"),
        ):
            unit_id = f"transformer_precision::{spec.module_path}::{'qk_matmul' if role == 'qk_matmul' else 'softmax' if role == 'softmax' else 'av'}"
            unit = units.get(unit_id)
            expected_prefix = _module_onnx_prefix(str(spec.module_path))
            all_matches = list(audit.get(key, ()))
            matches = [
                row for row in audit.get(key, ())
                if str(row.get("node_name", "")).startswith(f"{expected_prefix}/")
            ]
            if not matches and len(all_matches) == 1:
                matches = all_matches
            if unit is None or len(matches) != 1:
                raise RuntimeError(
                    f"v2xvit_attention_functional_mapping_not_unique:{unit_id}:{len(matches)}"
                )
            node = nodes.get(str(matches[0]["node_name"]))
            if node is None:
                raise RuntimeError(f"v2xvit_functional_onnx_node_missing:{unit_id}")
            add(unit, node, source="attention_topology_from_canonical_qkv")

    for unit in precision_units:
        node_name = ""
        if unit.role == "layernorm":
            node_name = _layernorm_onnx_name(str(unit.metadata["functional_owner"]))
        elif unit.role == "residual_add":
            node_name = _residual_onnx_name(unit)
        elif unit.role == "attention_merge":
            layer = _encoder_layer_index(str(unit.metadata["functional_owner"]))
            node_name = f"/layers.{layer}.0/layers.0.1/fn/split_attn/Add_3"
        else:
            continue
        node = nodes.get(node_name)
        if node is None:
            raise RuntimeError(f"v2xvit_functional_onnx_node_missing:{unit.unit_id}:{node_name}")
        add(unit, node, source="explicit_v2xvit_graph_contract")

    # FFN activation is not an independent gene.  It is the activation input
    # of FFN2 and must inherit that weighted unit's requested precision.
    for spec in ffn_instances:
        unit_id = f"transformer_precision::{spec.module_path}::ffn2"
        unit = units.get(unit_id)
        if unit is None:
            raise RuntimeError(f"v2xvit_ffn2_unit_missing:{unit_id}")
        paths = (spec.down_projection_path,) if spec.ffn_type == "gated" else (spec.second_projection_path,)
        canonical = sorted(
            name
            for path, names in origins_by_module.items()
            if path in paths or any(path.endswith(f".{wanted}") for wanted in paths)
            for name in names
        )
        if len(canonical) != 1 or canonical[0] not in nodes:
            raise RuntimeError(f"v2xvit_ffn2_canonical_mapping_not_unique:{unit_id}:{canonical}")
        state = str(requested_states.get(unit_id, unit.default_state)).upper()
        requested = {"W32A32": "FP32", "W16A16": "FP16", "W8A8": "INT8"}[state]
        node = nodes[canonical[0]]
        input_name = str(node.input[0])
        producer = next(
            (row for row in model.graph.node if input_name in tuple(str(v) for v in row.output)),
            None,
        )
        qdq = bool(producer is not None and str(producer.op_type) == "DequantizeLinear")
        rows.append(
            {
                "unit_id": unit_id,
                "role": "ffn_activation_input",
                "requested_state": state,
                "requested_compute_precision": requested,
                "requested_output_precision": requested,
                "source": "bound_to_ffn2_weighted_unit",
                **_node_row(node, element_types),
                "activation_input_tensor": input_name,
                "activation_input_qdq": qdq,
                "onnx_contract_exact": requested != "INT8" or qdq,
            }
        )

    activation_unit_ids = {
        str(unit.unit_id) for unit in precision_units if bool(unit.activation_only)
    }
    mapped_unit_ids = {str(row["unit_id"]) for row in rows if row["role"] != "ffn_activation_input"}
    missing = sorted(activation_unit_ids - mapped_unit_ids)
    duplicates = sorted(
        unit_id
        for unit_id in mapped_unit_ids
        if sum(row["unit_id"] == unit_id and row["role"] != "ffn_activation_input" for row in rows) != 1
    )
    passed = not missing and not duplicates and all(bool(row["onnx_contract_exact"]) for row in rows)
    return {
        "schema_version": "v2xvit-functional-onnx-precision-v1",
        "passed": passed,
        "rows": rows,
        "missing_unit_ids": missing,
        "duplicate_unit_ids": duplicates,
        "functional_unit_count": len(activation_unit_ids),
        "ffn_activation_binding_count": len(ffn_instances),
    }


def audit_trt_v2xvit_functional_precision(
    layer_info: str | Path | Sequence[Mapping[str, Any]] | Mapping[str, Any],
    onnx_mapping: Mapping[str, Any],
) -> dict[str, Any]:
    """Match every functional ONNX identity to TensorRT inspector evidence."""

    layers = load_layer_info(layer_info)
    evidence: list[dict[str, Any]] = []
    for source in onnx_mapping.get("rows", ()):
        node_name = str(source["onnx_node"])
        matches = [row for row in layers if has_canonical_identity(row, node_name)]
        if not matches:
            matches = [row for row in layers if node_name in layer_metadata(row)]
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
        realized = sorted({precision_name(row) for row in matches if precision_name(row)})
        requested = str(source["requested_compute_precision"])
        role = str(source["role"])
        if role in {"qk_matmul", "softmax", "layernorm"}:
            format_ok = requested in source.get("input_types", ()) and all(
                value == "FP32" for value in source.get("output_types", ())
            )
        elif role == "ffn_activation_input":
            token = {"FP32": "float", "FP16": "half", "INT8": "int8"}[requested]
            format_ok = any(token in value for value in input_formats)
        elif role == "residual_add":
            format_ok = all(value == "FP16" for value in source.get("output_types", ()))
        elif role == "attention_merge":
            format_ok = all(
                value == requested for value in source.get("output_types", ())
            )
        else:
            format_ok = bool(source.get("onnx_contract_exact"))
        passed = bool(matches) and bool(source.get("onnx_contract_exact")) and format_ok
        evidence.append(
            {
                **dict(source),
                "trt_layer_names": [layer_name(row) for row in matches],
                "trt_layer_precisions": realized,
                "trt_input_formats": input_formats,
                "trt_output_formats": output_formats,
                "inspector_match_count": len(matches),
                "conflict": not passed,
                "fallback": False,
                "unmapped": not bool(matches),
                "passed": passed,
            }
        )
    return {
        "schema_version": "v2xvit-functional-trt-precision-v1",
        "passed": bool(evidence) and all(row["passed"] for row in evidence),
        "rows": evidence,
        "conflict_count": sum(bool(row["conflict"]) for row in evidence),
        "fallback_count": sum(bool(row["fallback"]) for row in evidence),
        "unmapped_count": sum(bool(row["unmapped"]) for row in evidence),
    }


__all__ = [
    "audit_trt_v2xvit_functional_precision",
    "build_v2xvit_functional_onnx_mapping",
    "fixed_functional_onnx_precision_overrides",
    "requested_states_from_phenotype",
]
