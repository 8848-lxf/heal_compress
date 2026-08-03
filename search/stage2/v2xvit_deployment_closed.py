"""Deployment-closed V2X-ViT precision mapping and calibration identity."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from quantization.types import (
    CanonicalPrecisionEntry,
    CanonicalPrecisionMappingResult,
    stable_json_hash,
)
from search.model_families.transformer.canonical_roles import classify_weighted_module
from search.model_family.deployment import file_sha256


_F3_FUNCTIONAL_ROLE_PRECISION = {
    "layernorm": ("fp32", "fp32", "FP32"),
    "qk_matmul": ("fp32", "fp32", "FP32"),
    "qk_scale": ("fp32", "fp32", "FP32"),
    "mask_relation_add": ("fp32", "fp32", "FP32"),
    "softmax": ("fp16", "fp16", "unknown"),
    "av_matmul": ("fp16", "fp16", "unknown"),
    "residual_add": ("fp16", "fp16", "unknown"),
    "split_attention_gate": ("fp16", "fp16", "unknown"),
    "communication_fusion": ("fp16", "fp16", "unknown"),
}


def _functional_role_contract(contract: str) -> dict[str, tuple[str, str, str]]:
    name = str(contract).upper()
    if name == "F3":
        return dict(_F3_FUNCTIONAL_ROLE_PRECISION)
    if name == "P32":
        return {
            role: ("fp32", "fp32", "FP32")
            for role in _F3_FUNCTIONAL_ROLE_PRECISION
        }
    raise ValueError(f"v2xvit_functional_contract_unknown:{contract}")


def _origin_entries(origin: Any) -> list[Any]:
    return list(origin.get("entries", ())) if isinstance(origin, Mapping) else list(origin.entries)


def _field(row: Any, name: str, default: Any = "") -> Any:
    return row.get(name, default) if isinstance(row, Mapping) else getattr(row, name, default)


def resolve_origin_module_precision(
    origin: Any,
    module_precision_profile: Mapping[str, str],
    *,
    fixed_fp32_module_paths: Sequence[str] = (),
    allowed_unexported_profile_paths: Sequence[str] = (),
) -> dict[str, str]:
    """Resolve exact or suffix-qualified module paths without silent defaults."""

    requested = {str(key): str(value).upper() for key, value in module_precision_profile.items()}
    invalid = {key: value for key, value in requested.items() if value not in {"FP32", "FP16", "INT8"}}
    if invalid:
        raise ValueError(f"v2xvit_deployment_precision_invalid:{invalid}")
    resolved: dict[str, str] = {}
    consumed: set[str] = set()
    fixed = {str(value) for value in fixed_fp32_module_paths}
    allowed_unexported = {str(value) for value in allowed_unexported_profile_paths}
    for entry in _origin_entries(origin):
        path = str(_field(entry, "module_path"))
        matches = [
            (key, value)
            for key, value in requested.items()
            if path == key or path.endswith(f".{key}") or key.endswith(f".{path}")
        ]
        values = {value for _, value in matches}
        if not values and path in fixed:
            resolved[path] = "FP32"
            continue
        if len(values) != 1:
            raise RuntimeError(
                f"v2xvit_deployment_module_precision_unresolved:{path}:{matches}"
            )
        resolved[path] = next(iter(values))
        consumed.update(key for key, _ in matches)
    unknown = sorted(set(requested) - consumed - allowed_unexported)
    if unknown:
        raise RuntimeError(f"v2xvit_deployment_precision_loci_not_exported:{unknown}")
    return resolved


def build_deployment_closed_precision_mapping(
    *,
    origin: Any,
    inventory: Mapping[str, Any],
    graph_nodes: Mapping[str, Any],
    module_precision_profile: Mapping[str, str],
    profile_id: str,
    fixed_fp32_module_paths: Sequence[str] = (),
    allowed_unexported_profile_paths: Sequence[str] = (),
    functional_contract: str = "F3",
) -> tuple[CanonicalPrecisionMappingResult, list[dict[str, Any]]]:
    """Bind weighted genes and protected functional roles to one typed graph."""

    fixed = {str(value) for value in fixed_fp32_module_paths}
    resolved = resolve_origin_module_precision(
        origin,
        module_precision_profile,
        fixed_fp32_module_paths=fixed,
        allowed_unexported_profile_paths=allowed_unexported_profile_paths,
    )
    inventory_by_node = {
        str(row["onnx_node"]): dict(row)
        for row in inventory["rows"]
        if str(row.get("onnx_node", ""))
    }
    functional_roles = _functional_role_contract(functional_contract)

    def tensors(node_name: str) -> list[str]:
        node = graph_nodes.get(str(node_name))
        return [*map(str, node.input), *map(str, node.output)] if node is not None else []

    entries: list[CanonicalPrecisionEntry] = []
    requested_rows: list[dict[str, Any]] = []
    for source in _origin_entries(origin):
        path = str(_field(source, "module_path"))
        node_name = str(_field(source, "canonical_node_name"))
        role_row = inventory_by_node.get(node_name, {})
        role = str(
            role_row.get("canonical_role")
            or classify_weighted_module("lidar_v2xvit", path).canonical_role
        )
        precision = resolved[path].lower()
        output = (
            "fp32"
            if role in {"q_projection", "k_projection", "fused_qkv_projection"}
            else "fp16"
            if precision in {"fp16", "int8"}
            else "fp32"
        )
        entries.append(
            CanonicalPrecisionEntry(
                module_path=path,
                canonical_node_name=node_name,
                original_node_name=str(_field(source, "original_node_name")),
                weight_initializer=str(_field(source, "weight_initializer")),
                onnx_op_type=str(_field(source, "onnx_op_type")),
                call_index=int(_field(source, "call_index", 0)),
                precision_group=f"v2xvit_deployment_closed::{path}",
                requested_precision=precision,
                realized_request_precision=precision,
                realized_output_precision=output,
            )
        )
        requested_rows.append(
            {
                "role": role,
                "module_path": path,
                "onnx_node": node_name,
                "requested_precision": precision.upper(),
                "requested_output_precision": output.upper(),
                "requested_accumulator": "FP32" if precision == "fp32" else "INT32" if precision == "int8" else "unknown",
                "tensor_names": tensors(node_name),
                "profile_contract_source": (
                    "fixed_fp32_not_search_gene"
                    if path in fixed
                    else "candidate_precision_gene"
                ),
            }
        )

    auxiliary: dict[str, str] = {}
    auxiliary_outputs: dict[str, str] = {}
    for row in inventory["rows"]:
        node_name = str(row.get("onnx_node", ""))
        role = str(row.get("canonical_role", ""))
        if not node_name or str(row.get("module_path", "")) or role not in functional_roles:
            continue
        compute, output, accumulator = functional_roles[role]
        auxiliary[node_name] = compute
        auxiliary_outputs[node_name] = output
        requested_rows.append(
            {
                "role": role,
                "module_path": "",
                "onnx_node": node_name,
                "requested_precision": compute.upper(),
                "requested_output_precision": output.upper(),
                "requested_accumulator": accumulator,
                "tensor_names": tensors(node_name),
                "profile_contract_source": f"protected_functional_role::{role}",
            }
        )
    for group in (
        origin.get("functional_compute_groups", ())
        if isinstance(origin, Mapping)
        else getattr(origin, "functional_compute_groups", ())
    ):
        canonical = str(_field(group, "canonical_node_name"))
        members = sorted(
            name for name in graph_nodes if name == canonical or name.startswith(f"{canonical}__member")
        )
        if not members:
            raise RuntimeError(f"v2xvit_functional_group_unresolved:{canonical}")
        for node_name in members:
            auxiliary[node_name] = "fp16"
            auxiliary_outputs[node_name] = "fp16"
            requested_rows.append(
                {
                    "role": "communication_fusion",
                    "module_path": str(_field(group, "module_path")),
                    "onnx_node": node_name,
                    "requested_precision": "FP16",
                    "requested_output_precision": "FP16",
                    "requested_accumulator": "unknown",
                    "tensor_names": tensors(node_name),
                    "profile_contract_source": "functional_compute_group",
                }
            )
    mapping = CanonicalPrecisionMappingResult(
        entries=entries,
        profile_id=str(profile_id),
        profile_hash=stable_json_hash(
            {
                "profile_id": str(profile_id),
                "weighted": resolved,
                "functional": auxiliary,
                "outputs": auxiliary_outputs,
                "functional_contract": str(functional_contract).upper(),
            }
        ),
        origin_map_hash=str(_field(origin, "origin_map_hash")),
        policy_version="v2xvit-deployment-closed-f3-qk-fp32-v1",
        auxiliary_layer_precisions=auxiliary,
        auxiliary_layer_output_types=auxiliary_outputs,
    )
    return mapping, requested_rows


def bind_train200_calibration_identity(
    *,
    metadata: Mapping[str, Any],
    scales: Mapping[str, Any],
    train200_manifest: str | Path,
    checkpoint: str | Path,
    physical_structure_hash: str,
    state_dict_shape_hash: str,
    precision_map_hash: str,
    onnx_path: str | Path,
    calibration_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind every cache-invalidating input to one fresh calibration identity."""

    base = dict(metadata)
    frame_count = int(base.get("frame_count", 0))
    if frame_count != 200 or len(base.get("sample_evidence", ())) != 200:
        raise RuntimeError(f"train200_calibration_incomplete:{frame_count}")
    identity = {
        **base,
        "algorithm": "EntropyCalibration2",
        "algorithm_implementation": "ModelOpt entropy/KL histogram",
        "dataset_split": "train",
        "requested_frames": 200,
        "processed_frames": 200,
        "skipped_frames": 0,
        "manifest_path": str(Path(train200_manifest).resolve()),
        "manifest_hash": str(base["manifest_hash"]),
        "checkpoint_path": str(Path(checkpoint).resolve()),
        "checkpoint_hash": file_sha256(checkpoint),
        "physical_hash": str(physical_structure_hash),
        "state_dict_shape_hash": str(state_dict_shape_hash),
        "precision_map_hash": str(precision_map_hash),
        "onnx_path": str(Path(onnx_path).resolve()),
        "onnx_hash": file_sha256(onnx_path),
        "calibration_algorithm_config": dict(calibration_config),
        "calibration_algorithm_config_hash": stable_json_hash(dict(calibration_config)),
        "scale_hash": stable_json_hash(dict(scales)),
        "fresh_for_exact_candidate": True,
    }
    identity["cache_hash"] = stable_json_hash(identity)
    return identity


def audit_deployment_closed_profile(
    *,
    requested_rows: Sequence[Mapping[str, Any]],
    realized_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    conflicts = [dict(row) for row in realized_rows if bool(row.get("conflict"))]
    requested_int8 = sorted(
        str(row.get("onnx_node"))
        for row in requested_rows
        if str(row.get("requested_precision", "")).upper() == "INT8"
    )
    realized_int8 = sorted(
        str(row.get("onnx_node"))
        for row in realized_rows
        if str(row.get("realized_precision", "")).upper() == "INT8"
    )
    return {
        "requested_realized_exact": not conflicts and requested_int8 == realized_int8,
        "conflict_count": len(conflicts),
        "conflicts": conflicts,
        "requested_int8_nodes": requested_int8,
        "realized_int8_nodes": realized_int8,
        "silent_fallback": bool(conflicts or requested_int8 != realized_int8),
    }


__all__ = [
    "audit_deployment_closed_profile",
    "bind_train200_calibration_identity",
    "build_deployment_closed_precision_mapping",
    "resolve_origin_module_precision",
]
