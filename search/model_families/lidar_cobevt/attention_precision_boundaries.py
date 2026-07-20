"""Explicit numerical precision boundaries for CoBEVT Attention diagnostics."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping


ATTENTION_ROLES = (
    "layernorm",
    "q_projection",
    "k_projection",
    "v_projection",
    "qk_scale",
    "qk_matmul",
    "softmax",
    "av_matmul",
    "output_projection",
    "residual_add",
)

SINGLE_BOUNDARY_PROFILE_NAMES = (
    "A0_strict_fp32_reference",
    "A1_qkv_projection_fp16_core_fp32",
    "A2_layernorm_fp16_only",
    "A3_qk_matmul_fp16_only",
    "A4_softmax_fp16_only",
    "A5_av_matmul_fp16_only",
    "A6_output_projection_fp16_only",
    "A7_residual_add_fp16_only",
)

COMBINATION_PROFILE_NAMES = (
    "M1_projection_fp16_core_fp32",
    "M2_projection_qk_av_fp16_softmax_fp32",
    "M3_projection_av_fp16_qk_softmax_fp32",
    "M4_projection_boundary_qk_av_fp16",
    "M5_projection_softmax_av_add_fp16_qk_fp32",
)

FINAL_PROFILE_NAMES = (
    "P0_rest_fp16_attention_fp32",
    "F1_rest_fp16_projection_fp16_core_fp32",
    "F2_rest_fp16_projection_av_fp16_qk_core_fp32",
    "F3_rest_fp16_qk_fp32_minimal_island",
)

ATTENTION_BOUNDARY_PROFILE_NAMES = (
    *SINGLE_BOUNDARY_PROFILE_NAMES,
    *COMBINATION_PROFILE_NAMES,
    *FINAL_PROFILE_NAMES,
)


@dataclass(frozen=True)
class AttentionBoundaryProfile:
    profile_name: str
    role_dtypes: Mapping[str, str]
    output_recovery_roles: tuple[str, ...]
    external_weighted_dtype: str = "FP32"

    def __post_init__(self) -> None:
        normalized = {str(key): str(value).upper() for key, value in self.role_dtypes.items()}
        if set(normalized) != set(ATTENTION_ROLES):
            raise ValueError(f"attention_profile_role_mismatch:{self.profile_name}")
        invalid = sorted(
            role for role, precision in normalized.items() if precision not in {"FP32", "FP16"}
        )
        if invalid:
            raise ValueError(
                f"attention_profile_precision_invalid:{self.profile_name}:{invalid}"
            )
        recovery = tuple(str(role) for role in self.output_recovery_roles)
        external = str(self.external_weighted_dtype).upper()
        if external not in {"FP32", "FP16"}:
            raise ValueError(
                f"attention_profile_external_precision_invalid:{self.profile_name}"
            )
        if not set(recovery) <= set(ATTENTION_ROLES):
            raise ValueError(f"attention_profile_recovery_role_invalid:{self.profile_name}")
        if not set(recovery) <= {
            role for role, precision in normalized.items() if precision == "FP16"
        }:
            raise ValueError(
                f"attention_profile_recovery_role_not_fp16:{self.profile_name}"
            )
        object.__setattr__(self, "role_dtypes", MappingProxyType(normalized))
        object.__setattr__(self, "output_recovery_roles", recovery)
        object.__setattr__(self, "external_weighted_dtype", external)

    @property
    def profile_hash(self) -> str:
        payload = {
            "output_recovery_roles": list(self.output_recovery_roles),
            "profile_name": self.profile_name,
            "role_dtypes": dict(sorted(self.role_dtypes.items())),
            "schema_version": "cobevt-attention-boundary-profile-v1",
        }
        if self.external_weighted_dtype != "FP32":
            payload["external_weighted_dtype"] = self.external_weighted_dtype
            payload["schema_version"] = "cobevt-attention-boundary-profile-v2"
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def to_dict(self) -> dict[str, object]:
        return {
            "external_weighted_dtype": self.external_weighted_dtype,
            "output_recovery_roles": list(self.output_recovery_roles),
            "profile_hash": self.profile_hash,
            "profile_name": self.profile_name,
            "role_dtypes": dict(sorted(self.role_dtypes.items())),
            "schema_version": "cobevt-attention-boundary-profile-v1",
        }


def _single_boundary_profile(
    profile_name: str, fp16_roles: Iterable[str]
) -> AttentionBoundaryProfile:
    selected = {str(role) for role in fp16_roles}
    return AttentionBoundaryProfile(
        profile_name=profile_name,
        role_dtypes={
            role: ("FP16" if role in selected else "FP32") for role in ATTENTION_ROLES
        },
        output_recovery_roles=tuple(role for role in ATTENTION_ROLES if role in selected),
    )


def _combination_profile(
    profile_name: str,
    *,
    fp16_roles: Iterable[str],
    output_recovery_roles: Iterable[str],
    external_weighted_dtype: str = "FP32",
) -> AttentionBoundaryProfile:
    selected = {str(role) for role in fp16_roles}
    return AttentionBoundaryProfile(
        profile_name=profile_name,
        role_dtypes={
            role: ("FP16" if role in selected else "FP32")
            for role in ATTENTION_ROLES
        },
        output_recovery_roles=tuple(str(role) for role in output_recovery_roles),
        external_weighted_dtype=external_weighted_dtype,
    )


_PROFILES = {
    name: _single_boundary_profile(name, roles)
    for name, roles in (
        ("A0_strict_fp32_reference", ()),
        (
            "A1_qkv_projection_fp16_core_fp32",
            ("q_projection", "k_projection", "v_projection"),
        ),
        ("A2_layernorm_fp16_only", ("layernorm",)),
        ("A3_qk_matmul_fp16_only", ("qk_matmul",)),
        ("A4_softmax_fp16_only", ("softmax",)),
        ("A5_av_matmul_fp16_only", ("av_matmul",)),
        ("A6_output_projection_fp16_only", ("output_projection",)),
        ("A7_residual_add_fp16_only", ("residual_add",)),
    )
}
_PROFILES.update(
    {
        "P0_rest_fp16_attention_fp32": _combination_profile(
            "P0_rest_fp16_attention_fp32",
            fp16_roles=(),
            output_recovery_roles=(),
            external_weighted_dtype="FP16",
        ),
        "R1_attention_fp16_default": _combination_profile(
            "R1_attention_fp16_default",
            fp16_roles=ATTENTION_ROLES,
            output_recovery_roles=ATTENTION_ROLES,
            external_weighted_dtype="FP16",
        ),
        "R1_fp16_operands_default_accum_fixed": _combination_profile(
            "R1_fp16_operands_default_accum_fixed",
            fp16_roles=ATTENTION_ROLES,
            output_recovery_roles=ATTENTION_ROLES,
            external_weighted_dtype="FP16",
        ),
        "M1_projection_fp16_core_fp32": _combination_profile(
            "M1_projection_fp16_core_fp32",
            fp16_roles=(
                "q_projection",
                "k_projection",
                "v_projection",
                "output_projection",
            ),
            output_recovery_roles=(
                "q_projection",
                "k_projection",
                "v_projection",
                "output_projection",
            ),
        ),
        "M2_projection_qk_av_fp16_softmax_fp32": _combination_profile(
            "M2_projection_qk_av_fp16_softmax_fp32",
            fp16_roles=(
                "q_projection",
                "k_projection",
                "v_projection",
                "qk_scale",
                "qk_matmul",
                "av_matmul",
                "output_projection",
            ),
            output_recovery_roles=("qk_matmul", "output_projection"),
        ),
        "M3_projection_av_fp16_qk_softmax_fp32": _combination_profile(
            "M3_projection_av_fp16_qk_softmax_fp32",
            fp16_roles=(
                "q_projection",
                "k_projection",
                "v_projection",
                "av_matmul",
                "output_projection",
            ),
            output_recovery_roles=(
                "q_projection",
                "k_projection",
                "output_projection",
            ),
        ),
        "M4_projection_boundary_qk_av_fp16": _combination_profile(
            "M4_projection_boundary_qk_av_fp16",
            fp16_roles=(
                "q_projection",
                "k_projection",
                "v_projection",
                "qk_scale",
                "qk_matmul",
                "av_matmul",
                "output_projection",
            ),
            output_recovery_roles=(
                "q_projection",
                "k_projection",
                "qk_matmul",
                "output_projection",
            ),
        ),
        "M5_projection_softmax_av_add_fp16_qk_fp32": _combination_profile(
            "M5_projection_softmax_av_add_fp16_qk_fp32",
            fp16_roles=(
                "q_projection",
                "k_projection",
                "v_projection",
                "softmax",
                "av_matmul",
                "output_projection",
                "residual_add",
            ),
            output_recovery_roles=(
                "q_projection",
                "k_projection",
                "residual_add",
            ),
        ),
        "F1_rest_fp16_projection_fp16_core_fp32": _combination_profile(
            "F1_rest_fp16_projection_fp16_core_fp32",
            fp16_roles=(
                "q_projection",
                "k_projection",
                "v_projection",
                "output_projection",
            ),
            output_recovery_roles=(
                "q_projection",
                "k_projection",
                "v_projection",
                "output_projection",
            ),
            external_weighted_dtype="FP16",
        ),
        "F2_rest_fp16_projection_av_fp16_qk_core_fp32": _combination_profile(
            "F2_rest_fp16_projection_av_fp16_qk_core_fp32",
            fp16_roles=(
                "q_projection",
                "k_projection",
                "v_projection",
                "av_matmul",
                "output_projection",
            ),
            output_recovery_roles=(
                "q_projection",
                "k_projection",
                "output_projection",
            ),
            external_weighted_dtype="FP16",
        ),
        "F3_rest_fp16_qk_fp32_minimal_island": _combination_profile(
            "F3_rest_fp16_qk_fp32_minimal_island",
            fp16_roles=(
                "q_projection",
                "k_projection",
                "v_projection",
                "softmax",
                "av_matmul",
                "output_projection",
                "residual_add",
            ),
            output_recovery_roles=(
                "q_projection",
                "k_projection",
                "residual_add",
            ),
            external_weighted_dtype="FP16",
        ),
    }
)


def attention_boundary_profile(profile_name: str) -> AttentionBoundaryProfile:
    name = str(profile_name).strip()
    try:
        return _PROFILES[name]
    except KeyError as exc:
        raise ValueError(f"unknown_attention_boundary_profile:{profile_name}") from exc


def attention_boundary_base_precision(profile_name: str) -> str:
    return attention_boundary_profile(profile_name).external_weighted_dtype


_PROJECTION_SUFFIX_TO_ROLE = {
    ".q_proj": "q_projection",
    ".k_proj": "k_projection",
    ".v_proj": "v_projection",
    ".out_proj": "output_projection",
}


def requested_weighted_precision(
    module_paths: Iterable[str], profile_name: str
) -> dict[str, str]:
    profile = attention_boundary_profile(profile_name)
    result = {}
    for value in module_paths:
        module_path = str(value)
        role = next(
            (
                candidate_role
                for suffix, candidate_role in _PROJECTION_SUFFIX_TO_ROLE.items()
                if module_path.endswith(suffix) and "_attention.fn." in module_path
            ),
            None,
        )
        result[module_path] = (
            profile.role_dtypes[role] if role else profile.external_weighted_dtype
        )
    return result


@dataclass(frozen=True)
class AttentionBlockNodes:
    block_id: str
    layer_index: int
    attention_kind: str
    role_nodes: Mapping[str, str]
    auxiliary_nodes: Mapping[str, str]

    def __post_init__(self) -> None:
        if set(self.role_nodes) != set(ATTENTION_ROLES):
            raise ValueError(f"attention_block_role_mismatch:{self.block_id}")
        object.__setattr__(self, "role_nodes", MappingProxyType(dict(self.role_nodes)))
        object.__setattr__(
            self, "auxiliary_nodes", MappingProxyType(dict(self.auxiliary_nodes))
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "attention_kind": self.attention_kind,
            "auxiliary_nodes": dict(sorted(self.auxiliary_nodes.items())),
            "block_id": self.block_id,
            "layer_index": self.layer_index,
            "role_nodes": dict(sorted(self.role_nodes.items())),
        }


_PROJECTION_MODULE_RE = re.compile(
    r"^fusion_net\.layers\.(?P<layer>\d+)\."
    r"(?P<kind>window|grid)_attention\.fn\."
    r"(?P<projection>q_proj|k_proj|v_proj|out_proj)$"
)


def _mapping_entries(entries_or_mapping: Any) -> tuple[Any, ...]:
    values = getattr(entries_or_mapping, "entries", entries_or_mapping)
    return tuple(values)


def discover_attention_nodes(
    model: Any,
    entries_or_mapping: Any,
    *,
    expected_block_count: int = 6,
) -> tuple[AttentionBlockNodes, ...]:
    nodes_by_name = {str(node.name): node for node in model.graph.node if str(node.name)}
    duplicate_names = len(nodes_by_name) != sum(
        bool(str(node.name)) for node in model.graph.node
    )
    if duplicate_names:
        raise ValueError("attention_graph_duplicate_node_name")
    projections: dict[tuple[int, str], dict[str, str]] = {}
    for entry in _mapping_entries(entries_or_mapping):
        module_path = str(entry.module_path)
        matched = _PROJECTION_MODULE_RE.fullmatch(module_path)
        if not matched:
            continue
        key = (int(matched.group("layer")), str(matched.group("kind")))
        role = _PROJECTION_SUFFIX_TO_ROLE[f".{matched.group('projection')}"]
        if role in projections.setdefault(key, {}):
            raise ValueError(f"attention_projection_mapping_duplicate:{module_path}")
        projections[key][role] = str(entry.canonical_node_name)
    if len(projections) != int(expected_block_count):
        raise ValueError(
            f"attention_block_count_mismatch:{len(projections)}:{expected_block_count}"
        )
    result = []
    expected_ops = {
        "layernorm": "LayerNormalization",
        "q_projection": "MatMul",
        "k_projection": "MatMul",
        "v_projection": "MatMul",
        "qk_scale": "Mul",
        "qk_matmul": "Einsum",
        "softmax": "Softmax",
        "av_matmul": "Einsum",
        "output_projection": "MatMul",
        "residual_add": "Add",
    }
    for (layer, kind), projection_nodes in sorted(projections.items()):
        base = f"/layers.{layer}/{kind}_attention"
        role_nodes = {
            "layernorm": f"{base}/norm/LayerNormalization",
            **projection_nodes,
            "qk_scale": f"{base}/fn/Mul_6",
            "qk_matmul": f"{base}/fn/Einsum",
            "softmax": f"{base}/fn/attend/Softmax",
            "av_matmul": f"{base}/fn/Einsum_1",
            "residual_add": f"{base}/Add",
        }
        missing_roles = sorted(set(ATTENTION_ROLES) - set(role_nodes))
        if missing_roles:
            raise ValueError(f"attention_role_mapping_missing:{base}:{missing_roles}")
        for role, node_name in role_nodes.items():
            node = nodes_by_name.get(node_name)
            if node is None:
                raise ValueError(f"attention_node_missing:{base}:{role}:{node_name}")
            if str(node.op_type) != expected_ops[role]:
                raise ValueError(
                    f"attention_node_op_mismatch:{base}:{role}:"
                    f"{node.op_type}:{expected_ops[role]}"
                )
        auxiliaries = {
            "rpe_add": f"{base}/fn/Add",
            "mask_where": f"{base}/fn/Where",
        }
        for role, node_name in auxiliaries.items():
            if node_name not in nodes_by_name:
                raise ValueError(f"attention_auxiliary_node_missing:{base}:{role}:{node_name}")
        result.append(
            AttentionBlockNodes(
                block_id=f"layers.{layer}.{kind}_attention",
                layer_index=layer,
                attention_kind=kind,
                role_nodes=role_nodes,
                auxiliary_nodes=auxiliaries,
            )
        )
    return tuple(result)


def _cast_target(node: Any) -> int | None:
    if str(node.op_type) != "Cast":
        return None
    return next(
        (int(attribute.i) for attribute in node.attribute if attribute.name == "to"),
        None,
    )


def _tensor_types(model: Any) -> dict[str, int]:
    from onnx import TensorProto

    types = {
        str(initializer.name): int(initializer.data_type)
        for initializer in model.graph.initializer
    }
    for value in [*model.graph.input, *model.graph.value_info, *model.graph.output]:
        if value.type.tensor_type.elem_type:
            types[str(value.name)] = int(value.type.tensor_type.elem_type)
    changed = True
    while changed:
        changed = False
        for node in model.graph.node:
            op_type = str(node.op_type)
            output_type = None
            if op_type == "Cast":
                output_type = _cast_target(node)
            elif op_type == "Where" and len(node.input) >= 3:
                output_type = types.get(str(node.input[2]))
            elif op_type in {
                "Add",
                "Einsum",
                "Identity",
                "LayerNormalization",
                "MatMul",
                "Mul",
                "Reshape",
                "Softmax",
                "Transpose",
            } and node.input:
                output_type = types.get(str(node.input[0]))
            elif op_type in {"Equal", "Greater", "Less"}:
                output_type = int(TensorProto.BOOL)
            if output_type is None:
                continue
            for output in node.output:
                name = str(output)
                if types.get(name) != int(output_type):
                    types[name] = int(output_type)
                    changed = True
    return types


def _set_tensor_type(model: Any, name: str, element_type: int) -> None:
    from onnx import helper

    for value in [*model.graph.input, *model.graph.value_info, *model.graph.output]:
        if str(value.name) == str(name):
            value.type.tensor_type.elem_type = int(element_type)
            return
    model.graph.value_info.append(
        helper.make_tensor_value_info(str(name), int(element_type), None)
    )


def _propagate_node_output_types(node: Any, types: dict[str, int]) -> None:
    """Refresh pass-through output types after an upstream boundary rewrite."""

    from onnx import TensorProto

    op_type = str(node.op_type)
    output_type = None
    if op_type == "Cast":
        output_type = _cast_target(node)
    elif op_type == "Where" and len(node.input) >= 3:
        output_type = types.get(str(node.input[2]))
    elif op_type in {
        "Add",
        "Concat",
        "Einsum",
        "Identity",
        "LayerNormalization",
        "MatMul",
        "Mul",
        "Reshape",
        "Softmax",
        "Squeeze",
        "Transpose",
        "Unsqueeze",
    } and node.input:
        output_type = types.get(str(node.input[0]))
    elif op_type in {"Equal", "Greater", "Less"}:
        output_type = int(TensorProto.BOOL)
    if output_type is None:
        return
    for output in node.output:
        types[str(output)] = int(output_type)


def validate_qk_operand_dtypes(
    model: Any,
    entries_or_mapping: Any,
    *,
    expected_block_count: int = 6,
) -> list[dict[str, object]]:
    """Fail before TensorRT parsing when either QK operand has a different dtype."""

    from onnx import TensorProto

    blocks = discover_attention_nodes(
        model, entries_or_mapping, expected_block_count=expected_block_count
    )
    nodes = {str(node.name): node for node in model.graph.node if str(node.name)}
    types = _tensor_types(model)
    precision_names = {
        int(TensorProto.FLOAT): "FP32",
        int(TensorProto.FLOAT16): "FP16",
    }
    rows: list[dict[str, object]] = []
    for block in blocks:
        node_name = block.role_nodes["qk_matmul"]
        node = nodes[node_name]
        if len(node.input) != 2:
            raise ValueError(
                f"attention_qk_operand_count_invalid:{node_name}:{len(node.input)}"
            )
        input_types = [types.get(str(value)) for value in node.input]
        if input_types[0] is None or input_types[1] is None:
            raise ValueError(
                f"attention_qk_operand_dtype_unresolved:{node_name}:{input_types}"
            )
        if input_types[0] != input_types[1]:
            raise ValueError(
                "attention_qk_operand_dtype_mismatch:"
                f"{node_name}:{input_types[0]}:{input_types[1]}"
            )
        rows.append(
            {
                "block_id": block.block_id,
                "input_0_dtype": precision_names.get(int(input_types[0]), "OTHER"),
                "input_1_dtype": precision_names.get(int(input_types[1]), "OTHER"),
                "qk_node": node_name,
                "validated": True,
            }
        )
    return rows


def _unique_cast_name(
    *, profile_name: str, block_id: str, role: str, boundary: str, index: int
) -> str:
    digest = hashlib.sha256(
        f"{profile_name}\0{block_id}\0{role}\0{boundary}\0{index}".encode("utf-8")
    ).hexdigest()[:12]
    return f"__cobevt_attention_boundary__{digest}__Cast"


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def apply_attention_boundary_contract(
    input_onnx: str | Path,
    output_onnx: str | Path,
    entries_or_mapping: Any,
    profile_name: str,
    *,
    expected_block_count: int = 6,
) -> dict[str, Any]:
    """Insert explicit per-role Cast boundaries into a canonical typed graph."""

    import onnx
    from onnx import TensorProto, helper

    profile = attention_boundary_profile(profile_name)
    source_path = Path(input_onnx)
    model = onnx.load(str(source_path))
    try:
        model = onnx.shape_inference.infer_shapes(
            model, strict_mode=False, data_prop=True
        )
    except Exception:
        pass
    blocks = discover_attention_nodes(
        model, entries_or_mapping, expected_block_count=expected_block_count
    )
    role_owner: dict[str, tuple[AttentionBlockNodes, str]] = {}
    for block in blocks:
        for role, node_name in block.role_nodes.items():
            if node_name in role_owner:
                raise ValueError(f"attention_node_role_alias:{node_name}")
            role_owner[node_name] = (block, role)
    types = _tensor_types(model)
    producers = {
        str(output): node for node in model.graph.node for output in node.output
    }
    rewritten = []
    node_records = []
    input_cast_count = 0
    output_cast_count = 0
    for node in model.graph.node:
        owner = role_owner.get(str(node.name))
        if owner is None:
            _propagate_node_output_types(node, types)
            rewritten.append(node)
            continue
        block, role = owner
        precision = str(profile.role_dtypes[role])
        compute_type = (
            int(TensorProto.FLOAT16)
            if precision == "FP16"
            else int(TensorProto.FLOAT)
        )
        input_cast_nodes = []
        original_inputs = list(node.input)
        if role == "qk_matmul" and len(node.input) != 2:
            raise ValueError(
                f"attention_qk_operand_count_invalid:{node.name}:{len(node.input)}"
            )
        for input_index, source_value in enumerate(list(node.input)):
            source = str(source_value)
            source_type = types.get(source)
            if source_type is None:
                raise ValueError(
                    f"attention_boundary_input_dtype_unresolved:"
                    f"{block.block_id}:{role}:{input_index}:{source}"
                )
            producer = producers.get(source)
            if producer is not None and _cast_target(producer) == compute_type:
                types[source] = compute_type
                continue
            force_explicit_boundary = role == "softmax" and precision == "FP16"
            if int(source_type) == compute_type and not force_explicit_boundary:
                continue
            cast_name = _unique_cast_name(
                profile_name=profile.profile_name,
                block_id=block.block_id,
                role=role,
                boundary="input",
                index=input_index,
            )
            cast_output = f"{cast_name}__output"
            cast = helper.make_node(
                "Cast", [source], [cast_output], name=cast_name, to=compute_type
            )
            rewritten.append(cast)
            producers[cast_output] = cast
            node.input[input_index] = cast_output
            types[cast_output] = compute_type
            _set_tensor_type(model, cast_output, compute_type)
            input_cast_nodes.append(cast_name)
            input_cast_count += 1
        rewritten.append(node)
        original_outputs = list(node.output)
        for output in node.output:
            types[str(output)] = compute_type
            _set_tensor_type(model, str(output), compute_type)
        output_cast_nodes = []
        output_dtype = precision
        if role in profile.output_recovery_roles and precision == "FP16":
            for output_index, public_output in enumerate(list(node.output)):
                public = str(public_output)
                cast_name = _unique_cast_name(
                    profile_name=profile.profile_name,
                    block_id=block.block_id,
                    role=role,
                    boundary="output",
                    index=output_index,
                )
                raw_output = f"{cast_name}__input"
                node.output[output_index] = raw_output
                cast = helper.make_node(
                    "Cast",
                    [raw_output],
                    [public],
                    name=cast_name,
                    to=int(TensorProto.FLOAT),
                )
                rewritten.append(cast)
                producers[raw_output] = node
                producers[public] = cast
                types[raw_output] = compute_type
                types[public] = int(TensorProto.FLOAT)
                _set_tensor_type(model, raw_output, compute_type)
                _set_tensor_type(model, public, int(TensorProto.FLOAT))
                output_cast_nodes.append(cast_name)
                output_cast_count += 1
            output_dtype = "FP32"
        node_records.append(
            {
                "attention_kind": block.attention_kind,
                "block_id": block.block_id,
                "compute_dtype": precision,
                "input_cast_count": len(input_cast_nodes),
                "input_cast_nodes": input_cast_nodes,
                "input_tensors_after": list(node.input),
                "input_tensors_before": original_inputs,
                "node_name": str(node.name),
                "op_type": str(node.op_type),
                "output_cast_count": len(output_cast_nodes),
                "output_cast_nodes": output_cast_nodes,
                "output_dtype": output_dtype,
                "output_tensors_after": list(node.output),
                "output_tensors_before": original_outputs,
                "role": role,
            }
        )
    del model.graph.node[:]
    model.graph.node.extend(rewritten)
    try:
        model = onnx.shape_inference.infer_shapes(
            model, strict_mode=False, data_prop=True
        )
    except Exception:
        pass
    qk_operand_dtype_audit = validate_qk_operand_dtypes(
        model,
        entries_or_mapping,
        expected_block_count=expected_block_count,
    )
    onnx.checker.check_model(model)
    destination = Path(output_onnx)
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(destination))
    return {
        "blocks": [block.to_dict() for block in blocks],
        "external_node_change_count": 0,
        "input_onnx": str(source_path),
        "input_onnx_sha256": _file_sha256(source_path),
        "inserted_input_cast_count": input_cast_count,
        "inserted_output_cast_count": output_cast_count,
        "missing_role_count": 0,
        "node_records": node_records,
        "output_onnx": str(destination),
        "output_onnx_sha256": _file_sha256(destination),
        "profile": profile.to_dict(),
        "qk_operand_dtype_audit": qk_operand_dtype_audit,
        "schema_version": "cobevt-attention-boundary-contract-v1",
    }


def describe_existing_attention_boundaries(
    input_onnx: str | Path,
    entries_or_mapping: Any,
    *,
    profile_name: str,
    expected_block_count: int = 6,
) -> dict[str, Any]:
    """Describe existing Attention dtypes without modifying the graph."""

    import onnx
    from onnx import TensorProto

    source = Path(input_onnx)
    model = onnx.load(str(source))
    try:
        model = onnx.shape_inference.infer_shapes(
            model, strict_mode=False, data_prop=True
        )
    except Exception:
        pass
    blocks = discover_attention_nodes(
        model, entries_or_mapping, expected_block_count=expected_block_count
    )
    types = _tensor_types(model)
    nodes = {str(node.name): node for node in model.graph.node if str(node.name)}
    producers = {
        str(output): node for node in model.graph.node for output in node.output
    }
    consumers: dict[str, list[Any]] = {}
    for node in model.graph.node:
        for input_name in node.input:
            consumers.setdefault(str(input_name), []).append(node)

    def precision_name(values: Iterable[int | None]) -> str:
        floating = {
            int(value)
            for value in values
            if value in {int(TensorProto.FLOAT), int(TensorProto.FLOAT16)}
        }
        if floating == {int(TensorProto.FLOAT)}:
            return "FP32"
        if floating == {int(TensorProto.FLOAT16)}:
            return "FP16"
        if floating:
            return "MIXED"
        return "UNRESOLVED"

    records = []
    for block in blocks:
        for role in ATTENTION_ROLES:
            node = nodes[block.role_nodes[role]]
            input_casts = [
                str(producer.name)
                for value in node.input
                if (producer := producers.get(str(value))) is not None
                and str(producer.op_type) == "Cast"
            ]
            output_casts = [
                str(consumer.name)
                for value in node.output
                for consumer in consumers.get(str(value), [])
                if str(consumer.op_type) == "Cast"
            ]
            compute_dtype = precision_name(types.get(str(value)) for value in node.input)
            output_dtype = precision_name(types.get(str(value)) for value in node.output)
            records.append(
                {
                    "attention_kind": block.attention_kind,
                    "block_id": block.block_id,
                    "compute_dtype": compute_dtype,
                    "input_cast_count": len(input_casts),
                    "input_cast_nodes": input_casts,
                    "input_tensors_after": list(node.input),
                    "input_tensors_before": list(node.input),
                    "node_name": str(node.name),
                    "op_type": str(node.op_type),
                    "output_cast_count": len(output_casts),
                    "output_cast_nodes": output_casts,
                    "output_dtype": output_dtype,
                    "output_tensors_after": list(node.output),
                    "output_tensors_before": list(node.output),
                    "role": role,
                }
            )
    profile_hash = hashlib.sha256(
        json.dumps(
            {
                "input_onnx_sha256": _file_sha256(source),
                "profile_name": str(profile_name),
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return {
        "blocks": [block.to_dict() for block in blocks],
        "graph_rewritten": False,
        "input_onnx": str(source),
        "input_onnx_sha256": _file_sha256(source),
        "missing_role_count": 0,
        "node_records": records,
        "profile": {
            "profile_hash": profile_hash,
            "profile_name": str(profile_name),
        },
        "schema_version": "cobevt-existing-attention-boundaries-v1",
    }


__all__ = [
    "ATTENTION_BOUNDARY_PROFILE_NAMES",
    "COMBINATION_PROFILE_NAMES",
    "FINAL_PROFILE_NAMES",
    "SINGLE_BOUNDARY_PROFILE_NAMES",
    "attention_boundary_base_precision",
    "ATTENTION_ROLES",
    "AttentionBlockNodes",
    "AttentionBoundaryProfile",
    "apply_attention_boundary_contract",
    "attention_boundary_profile",
    "describe_existing_attention_boundaries",
    "discover_attention_nodes",
    "requested_weighted_precision",
    "validate_qk_operand_dtypes",
]
