"""Build model/module/ONNX Transformer role inventories from real graphs."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

from .canonical_roles import (
    CANONICAL_TRANSFORMER_ROLES,
    CanonicalRole,
    classify_onnx_primitive,
    classify_weighted_module,
)


@dataclass(frozen=True)
class InventoryRow:
    model_family: str
    canonical_role: str
    module_path: str
    module_type: str
    onnx_node: str
    onnx_op_type: str
    input_dtype: str
    output_dtype: str
    input_shape: tuple[Any, ...]
    output_shape: tuple[Any, ...]
    heads: int | None
    d_qk: int | None
    d_v: int | None
    sequence_length: int | str | None
    attention_kind: str
    residual_scope: str
    confidence: str
    mapping_reason: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["input_shape"] = list(self.input_shape)
        payload["output_shape"] = list(self.output_shape)
        return payload


def _origin_entries(origin_map: Any) -> list[Any]:
    if isinstance(origin_map, Mapping):
        return list(origin_map.get("entries", ()))
    return list(getattr(origin_map, "entries", ()))


def _field(row: Any, name: str, default: Any = "") -> Any:
    return row.get(name, default) if isinstance(row, Mapping) else getattr(row, name, default)


def _dtype_name(element_type: int | None) -> str:
    if not element_type:
        return "unknown"
    import onnx

    try:
        return str(onnx.TensorProto.DataType.Name(int(element_type)))
    except ValueError:
        return f"onnx_dtype_{element_type}"


def _is_floating_dtype(name: str) -> bool:
    return str(name).upper() in {
        "FLOAT",
        "FLOAT16",
        "DOUBLE",
        "BFLOAT16",
        "FLOAT8E4M3FN",
        "FLOAT8E4M3FNUZ",
        "FLOAT8E5M2",
        "FLOAT8E5M2FNUZ",
    }


def _shape(value: Any | None) -> tuple[Any, ...]:
    if value is None or not value.type.tensor_type.HasField("shape"):
        return ()
    result = []
    for dim in value.type.tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            result.append(int(dim.dim_value))
        elif dim.dim_param:
            result.append(str(dim.dim_param))
        else:
            result.append(None)
    return tuple(result)


def _tensor_metadata(model: Any) -> dict[str, tuple[str, tuple[Any, ...]]]:
    result: dict[str, tuple[str, tuple[Any, ...]]] = {}
    for value in [*model.graph.input, *model.graph.value_info, *model.graph.output]:
        result[str(value.name)] = (
            _dtype_name(int(value.type.tensor_type.elem_type)),
            _shape(value),
        )
    for initializer in model.graph.initializer:
        result[str(initializer.name)] = (
            _dtype_name(int(initializer.data_type)),
            tuple(int(value) for value in initializer.dims),
        )
    return result


def _attribute_text(node: Any, name: str) -> str:
    for attribute in node.attribute:
        if str(attribute.name) == str(name):
            return bytes(attribute.s).decode("utf-8")
    return ""


def _semantic_primitive_roles(
    graph: Any,
    *,
    model_family: str,
    metadata: Mapping[str, tuple[str, tuple[Any, ...]]],
) -> dict[str, str]:
    """Resolve attention primitives by graph flow, not path substrings alone."""

    nodes = {str(node.name): node for node in graph.graph.node}
    producers = {
        str(output): node for node in graph.graph.node for output in node.output
    }
    qk_nodes: set[str] = set()
    roles: dict[str, str] = {}
    for node in graph.graph.node:
        if str(node.op_type) != "Einsum":
            continue
        equation = _attribute_text(node, "equation").replace(" ", "")
        output_subscripts = equation.split("->", 1)[-1] if "->" in equation else ""
        input_subscripts = equation.split("->", 1)[0].split(",")
        if output_subscripts.endswith("ij"):
            roles[str(node.name)] = "qk_matmul"
            qk_nodes.add(str(node.name))
        elif input_subscripts and "ij" in input_subscripts[0]:
            roles[str(node.name)] = "av_matmul"

    # Some implementations apply 1/sqrt(d) to Q before QK, while others
    # multiply the completed QK matrix.  Both are semantic QK scale sites.
    for qk_name in qk_nodes:
        qk_node = nodes[qk_name]
        for value in qk_node.input:
            producer = producers.get(str(value))
            if producer is None or str(producer.op_type) not in {"Mul", "Div"}:
                continue
            output_name = str(producer.output[0]) if producer.output else ""
            if _is_floating_dtype(metadata.get(output_name, ("unknown", ()))[0]):
                roles[str(producer.name)] = "qk_scale"

    def paths_to_qk(tensor: str, depth: int, visited: frozenset[str]) -> list[list[Any]]:
        if depth < 0:
            return []
        producer = producers.get(str(tensor))
        if producer is None or str(producer.name) in visited:
            return []
        if str(producer.name) in qk_nodes:
            return [[producer]]
        paths: list[list[Any]] = []
        next_visited = visited | {str(producer.name)}
        for value in producer.input:
            for suffix in paths_to_qk(str(value), depth - 1, next_visited):
                paths.append([producer, *suffix])
        return paths

    for node in graph.graph.node:
        name = str(node.name)
        op_type = str(node.op_type)
        if op_type == "LayerNormalization":
            roles[name] = "layernorm"
        elif op_type == "Softmax":
            roles[name] = "split_attention_gate" if "split_attn" in name.lower() else "softmax"
            if roles[name] != "softmax":
                continue
            for value in node.input:
                for path in paths_to_qk(str(value), 5, frozenset()):
                    for member in path[:-1]:
                        member_name = str(member.name)
                        member_op = str(member.op_type)
                        output_name = str(member.output[0]) if member.output else ""
                        output_dtype = metadata.get(output_name, ("unknown", ()))[0]
                        if not _is_floating_dtype(output_dtype):
                            continue
                        if member_op in {"Mul", "Div"}:
                            roles[member_name] = "qk_scale"
                        elif member_op in {"Add", "Where"}:
                            roles[member_name] = "mask_relation_add"
        elif op_type == "Concat" and any(
            token in name.lower() for token in ("split_attn", "fusion")
        ):
            roles[name] = "communication_fusion"

    # Residual sites are stable source-code boundaries and deliberately differ
    # between the two model families.  Bias Add and dynamic-shape Add nodes are
    # excluded even if their exported name contains ``/fn/`` or ``/Add_N``.
    for name, node in nodes.items():
        if str(node.op_type) != "Add":
            continue
        if model_family == "lidar_cobevt" and re.fullmatch(
            r"/layers\.\d+/(?:window_attention|window_ffd|grid_attention|grid_ffd)/Add",
            name,
        ):
            roles[name] = "residual_add"
        elif model_family == "lidar_v2xvit" and (
            re.fullmatch(r"/layers\.\d+\.0/Add(?:_1)?", name)
            or name in {"/Add_2", "/Add_3", "/Add_4"}
        ):
            roles[name] = "residual_add"
    return roles


def _attention_dimensions(model: Any, path: str) -> tuple[int | None, int | None, int | None]:
    modules = dict(model.named_modules())
    parts = str(path).split(".")
    for stop in range(len(parts), 0, -1):
        module = modules.get(".".join(parts[:stop]))
        if module is None or not hasattr(module, "heads"):
            continue
        heads = int(module.heads)
        d_qk = getattr(module, "d_qk", None)
        d_v = getattr(module, "d_v", None)
        if d_qk is None:
            projection = getattr(module, "to_qkv", None)
            if projection is not None and hasattr(projection, "out_features"):
                d_qk = int(projection.out_features) // 3 // heads
                d_v = d_qk
            elif any(hasattr(module, name) for name in ("q_proj", "q_linears")):
                q = getattr(module, "q_proj", None)
                if q is None and len(getattr(module, "q_linears", ())):
                    q = module.q_linears[0]
                v = getattr(module, "v_proj", None)
                if v is None and len(getattr(module, "v_linears", ())):
                    v = module.v_linears[0]
                d_qk = int(q.out_features) // heads if q is not None else None
                d_v = int(v.out_features) // heads if v is not None else d_qk
        return heads, int(d_qk) if d_qk is not None else None, int(d_v) if d_v is not None else None
    return None, None, None


def _sequence_length(shape: Iterable[Any], role: str) -> int | str | None:
    values = tuple(shape)
    if not values:
        return None
    if role in {"qk_matmul", "softmax", "av_matmul"} and len(values) >= 2:
        return values[-2]
    return None


def build_model_inventory(
    *,
    model_family: str,
    model: Any,
    onnx_path: str | Path,
    origin_map: Any,
) -> dict[str, Any]:
    import onnx

    graph = onnx.load(str(onnx_path), load_external_data=False)
    try:
        graph = onnx.shape_inference.infer_shapes(graph, strict_mode=False, data_prop=True)
    except Exception:
        pass
    nodes = {str(node.name): node for node in graph.graph.node if str(node.name)}
    metadata = _tensor_metadata(graph)
    semantic_primitive_roles = _semantic_primitive_roles(
        graph, model_family=str(model_family), metadata=metadata
    )
    modules = dict(model.named_modules())
    rows: list[InventoryRow] = []
    realized_module_paths: set[str] = set()
    for entry in _origin_entries(origin_map):
        path = str(_field(entry, "module_path"))
        realized_module_paths.add(path)
        node_name = str(_field(entry, "canonical_node_name"))
        node = nodes.get(node_name)
        role = classify_weighted_module(model_family, path)
        input_name = str(node.input[0]) if node is not None and node.input else ""
        output_name = str(node.output[0]) if node is not None and node.output else ""
        input_dtype, input_shape = metadata.get(input_name, ("unknown", ()))
        output_dtype, output_shape = metadata.get(output_name, ("unknown", ()))
        heads, d_qk, d_v = _attention_dimensions(model, path)
        rows.append(
            InventoryRow(
                model_family=str(model_family),
                canonical_role=role.canonical_role,
                module_path=path,
                module_type=type(modules[path]).__name__ if path in modules else "functional_weighted",
                onnx_node=node_name,
                onnx_op_type=str(node.op_type) if node is not None else str(_field(entry, "onnx_op_type")),
                input_dtype=input_dtype,
                output_dtype=output_dtype,
                input_shape=input_shape,
                output_shape=output_shape,
                heads=heads,
                d_qk=d_qk,
                d_v=d_v,
                sequence_length=_sequence_length(output_shape, role.canonical_role),
                attention_kind=role.attention_kind,
                residual_scope=role.block,
                confidence=role.confidence,
                mapping_reason=role.mapping_reason,
            )
        )
    # A traced ONNX graph can specialize Python control flow (notably the two
    # heterogeneous agent-type branches in V2X-ViT).  The module inventory is
    # a model inventory, not merely an ONNX inventory, so retain parameterized
    # modules which were not realized by the accepted export trace.  They are
    # explicit non-realized rows and can never silently inherit a precision.
    weighted_types = (
        __import__("torch").nn.Conv1d,
        __import__("torch").nn.Conv2d,
        __import__("torch").nn.Conv3d,
        __import__("torch").nn.ConvTranspose1d,
        __import__("torch").nn.ConvTranspose2d,
        __import__("torch").nn.ConvTranspose3d,
        __import__("torch").nn.Linear,
    )
    for path, module in modules.items():
        if not path or path in realized_module_paths or not isinstance(module, weighted_types):
            continue
        role = classify_weighted_module(model_family, path)
        heads, d_qk, d_v = _attention_dimensions(model, path)
        rows.append(
            InventoryRow(
                model_family=str(model_family),
                canonical_role=role.canonical_role,
                module_path=path,
                module_type=type(module).__name__,
                onnx_node="",
                onnx_op_type="not_realized_in_export_trace",
                input_dtype="unknown",
                output_dtype="unknown",
                input_shape=(),
                output_shape=(),
                heads=heads,
                d_qk=d_qk,
                d_v=d_v,
                sequence_length=None,
                attention_kind=role.attention_kind,
                residual_scope=role.block,
                confidence="missing",
                mapping_reason=(
                    "parameterized module not realized by the accepted ONNX trace; "
                    "precision is fail-closed"
                ),
            )
        )
    for node in graph.graph.node:
        equation = ""
        for attribute in node.attribute:
            if str(attribute.name) == "equation":
                equation = bytes(attribute.s).decode("utf-8")
        semantic_role = semantic_primitive_roles.get(str(node.name))
        role = (
            CanonicalRole(
                model_family=str(model_family),
                canonical_role=semantic_role,
                onnx_node=str(node.name),
                onnx_op_type=str(node.op_type),
                attention_kind=(
                    "window" if "window_attention" in str(node.name).lower()
                    else "grid" if "grid_attention" in str(node.name).lower()
                    else "spatial_window" if "pwmsa" in str(node.name).lower()
                    else "agent_relation" if "/layers.0.0/fn/" in str(node.name).lower()
                    else ""
                ),
                confidence="high",
                mapping_reason="ONNX producer/consumer attention topology",
            )
            if semantic_role
            else None
        )
        if role is None and str(node.op_type) not in {"Add", "Where", "Mul", "Div", "Einsum", "Softmax", "LayerNormalization", "Concat"}:
            role = classify_onnx_primitive(
                model_family,
                node_name=str(node.name),
                op_type=str(node.op_type),
                equation=equation,
            )
        if role is None:
            continue
        input_name = str(node.input[0]) if node.input else ""
        output_name = str(node.output[0]) if node.output else ""
        input_dtype, input_shape = metadata.get(input_name, ("unknown", ()))
        output_dtype, output_shape = metadata.get(output_name, ("unknown", ()))
        # Names such as ``attention/fn/Mul`` are also emitted for dynamic shape
        # arithmetic.  They are not QK scaling or another compute role.  A
        # precision contract may only be attached to a floating data path.
        floating_inputs = [
            metadata.get(str(value), ("unknown", ()))[0] for value in node.input
        ]
        if (
            output_dtype != "unknown"
            and not (
                _is_floating_dtype(output_dtype)
                and any(_is_floating_dtype(value) for value in floating_inputs)
            )
        ):
            continue
        rows.append(
            InventoryRow(
                model_family=str(model_family),
                canonical_role=role.canonical_role,
                module_path="",
                module_type="functional",
                onnx_node=str(node.name),
                onnx_op_type=str(node.op_type),
                input_dtype=input_dtype,
                output_dtype=output_dtype,
                input_shape=input_shape,
                output_shape=output_shape,
                heads=None,
                d_qk=None,
                d_v=None,
                sequence_length=_sequence_length(output_shape, role.canonical_role),
                attention_kind=role.attention_kind,
                residual_scope=role.block,
                confidence=role.confidence,
                mapping_reason=role.mapping_reason,
            )
        )
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row.attention_kind or row.canonical_role in {
            "q_projection", "k_projection", "v_projection", "fused_qkv_projection",
            "output_projection", "qk_matmul", "softmax", "av_matmul", "layernorm", "residual_add",
        }:
            grouped[row.residual_scope or row.attention_kind or "unscoped"][row.canonical_role].append(row.to_dict())
    missing = [
        row.to_dict()
        for row in rows
        if row.canonical_role == "missing_role_mapping" or not row.onnx_node
    ]
    return {
        "model_family": str(model_family),
        "rows": [row.to_dict() for row in rows],
        "role_counts": dict(sorted(Counter(row.canonical_role for row in rows).items())),
        "attention_role_map": {
            block: dict(sorted(role_rows.items()))
            for block, role_rows in sorted(grouped.items())
        },
        "missing_role_mapping": missing,
        "unsupported_role_mapping_count": len(missing),
        "schema": {
            "canonical_roles": list(CANONICAL_TRANSFORMER_ROLES),
            "schema_version": "canonical-transformer-role-schema-v1",
        },
    }


__all__ = ["InventoryRow", "build_model_inventory"]
