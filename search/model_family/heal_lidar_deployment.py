"""Explicit-Q/DQ deployment contracts for HEAL LiDAR F-Cooper and DiscoNet."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
import threading
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn

from quantization.config import QDQConfig
from quantization.export.origin_mapping import apply_canonical_node_names, build_onnx_origin_map
from quantization.export.signal_maxk import capture_weighted_module_calls
from quantization.precision.qdq_inserter import insert_explicit_qdq
from quantization.precision.merge_contract import (
    adaptive_merge_cast_name,
    apply_adaptive_merge_output_contract,
)
from quantization.tensorrt.layer_info import (
    has_canonical_identity,
    load_layer_info,
    precision_name,
)
from quantization.tensorrt.precision_checker import validate_precision_realization
from quantization.types import (
    CanonicalPrecisionEntry,
    CanonicalPrecisionMappingResult,
    OnnxExportResult,
    stable_json_hash,
)

from ..quantization_space.types import QuantizationSearchGroup
from .contracts import ModelFamilyAudit
from .export.heal_lidar_baselines import (
    HealLidarBaselineExportPolicy,
    build_heal_lidar_baseline_export_module,
    prepare_heal_lidar_baseline_inputs,
)
from .onnx_mapping import ModelFamilyOnnxMapping, ModelFamilyOnnxWeightedEntry


HEAL_LIDAR_BASELINE_INPUT_NAMES = (
    "voxel_features",
    "voxel_coords",
    "voxel_num_points",
    "pairwise_t_matrix",
    "valid_voxel_mask",
    "agent_mask",
)
HEAL_LIDAR_BASELINE_OUTPUT_NAMES = ("cls_preds", "reg_preds", "dir_preds")
SUPPORTED_DEPLOYMENT_FAMILIES = (
    "heal_lidar_fcooper",
    "heal_lidar_disco",
    "heal_lidar_attfusion",
    "heal_lidar_cobevt",
)
WRAPPER_PARITY_MAX_ABS_TOL = 5.0e-3
WRAPPER_PARITY_MEAN_ABS_TOL = 5.0e-5

# ``torch.onnx.export`` mutates process-global exporter state in PyTorch 2.0.
# HEAL's PFNLayer.forward also toggles process-global ``torch.backends.cudnn``.
# GA Stage-2 owns one model per GPU but runs them in threads inside one process,
# so parity forwards and export must share one serialized critical section.
# Calibration, Q/DQ, TensorRT builds and evaluation remain parallel across GPUs.
_HEAL_LIDAR_ONNX_EXPORT_LOCK = threading.Lock()


@contextmanager
def _serialized_heal_lidar_onnx_export():
    with _HEAL_LIDAR_ONNX_EXPORT_LOCK:
        yield


@dataclass(frozen=True)
class HealLidarBaselineOnnxExport:
    export: OnnxExportResult
    family_mapping: ModelFamilyOnnxMapping
    input_shapes: dict[str, tuple[int, ...]]
    plugin_nodes: tuple[str, ...]
    wrapper_parity: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": "heal-lidar-baseline-onnx-export-v1",
            "export": self.export.to_dict(),
            "family_mapping": self.family_mapping.to_dict(),
            "input_shapes": {key: list(value) for key, value in self.input_shapes.items()},
            "plugin_nodes": list(self.plugin_nodes),
            "wrapper_parity": dict(self.wrapper_parity),
        }
        payload["artifact_hash"] = stable_json_hash(payload)
        return payload


def _family_id(value: str | ModelFamilyAudit) -> str:
    family_id = str(value.family_id if isinstance(value, ModelFamilyAudit) else value)
    if family_id not in SUPPORTED_DEPLOYMENT_FAMILIES:
        raise RuntimeError(f"unsupported_heal_lidar_deployment_family:{family_id}")
    return family_id


def _parity(
    reference: Mapping[str, torch.Tensor],
    actual: Sequence[torch.Tensor],
    output_names: Sequence[str],
) -> dict[str, Any]:
    expected_names = tuple(str(value) for value in output_names)
    observed_outputs = tuple(actual)
    missing_reference = sorted(set(expected_names) - set(reference))
    if missing_reference or len(observed_outputs) != len(expected_names):
        return {
            "passed": False,
            "outputs": {},
            "failure_reason": (
                f"output_contract_mismatch:expected={len(expected_names)}:"
                f"actual={len(observed_outputs)}:missing_reference={missing_reference}"
            ),
        }
    rows: dict[str, Any] = {}
    for name, observed in zip(expected_names, observed_outputs):
        expected = reference[str(name)]
        difference = (expected.float() - observed.float()).abs()
        maximum = float(difference.max().item())
        mean = float(difference.mean().item())
        rows[str(name)] = {
            "expected_shape": list(expected.shape),
            "actual_shape": list(observed.shape),
            "max_abs": maximum,
            "mean_abs": mean,
            "max_abs_tolerance": WRAPPER_PARITY_MAX_ABS_TOL,
            "mean_abs_tolerance": WRAPPER_PARITY_MEAN_ABS_TOL,
            "allclose": bool(
                expected.shape == observed.shape
                and torch.allclose(
                    expected.float(),
                    observed.float(),
                    atol=WRAPPER_PARITY_MAX_ABS_TOL,
                    rtol=1.0e-4,
                )
                and maximum <= WRAPPER_PARITY_MAX_ABS_TOL
                and mean <= WRAPPER_PARITY_MEAN_ABS_TOL
            ),
        }
    return {"passed": all(bool(row["allclose"]) for row in rows.values()), "outputs": rows}


def _check_onnx_with_scatter_plugin(model: Any) -> None:
    """Validate all standard ONNX structure after replacing the known plugin."""

    import onnx

    checkable = onnx.ModelProto()
    checkable.ParseFromString(model.SerializeToString())
    replaced = 0
    for node in checkable.graph.node:
        if str(node.domain) == "trt" and str(node.op_type) == "PointPillarScatterTRT":
            if not node.input or len(node.output) != 1:
                raise RuntimeError("heal_lidar_scatter_plugin_boundary_invalid")
            first_input = str(node.input[0])
            node.domain = ""
            node.op_type = "Identity"
            del node.input[:]
            node.input.extend([first_input])
            del node.attribute[:]
            replaced += 1
    if replaced != 1:
        raise RuntimeError(f"heal_lidar_scatter_plugin_checker_replacement_count:{replaced}")
    onnx.checker.check_model(checkable)


def build_heal_lidar_baseline_onnx_mapping(
    onnx_path: str | Path,
    audit: ModelFamilyAudit,
    origin_map: Any,
) -> ModelFamilyOnnxMapping:
    """Require every static weighted capability to map to realized ONNX truth."""

    family_id = _family_id(audit)
    origins_by_module: dict[str, list[Any]] = {}
    for row in origin_map.entries:
        origins_by_module.setdefault(str(row.module_path), []).append(row)
    entries: list[ModelFamilyOnnxWeightedEntry] = []
    unresolved: list[dict[str, Any]] = []
    for capability in sorted(audit.weighted_ops, key=lambda row: row.canonical_id):
        origins = sorted(
            origins_by_module.get(capability.module_path, ()),
            key=lambda row: (int(row.call_index), int(row.graph_index)),
        )
        if not origins:
            unresolved.append({
                "canonical_id": capability.canonical_id,
                "module_path": capability.module_path,
                "reason": "weighted_capability_not_realized_in_baseline_export",
            })
            entries.append(ModelFamilyOnnxWeightedEntry(
                canonical_id=capability.canonical_id,
                module_path=capability.module_path,
                source_kind=capability.source_kind,
                mapping_status="unresolved_active_module",
                active=False,
                reason="weighted_capability_not_realized_in_baseline_export",
            ))
            continue
        entries.append(ModelFamilyOnnxWeightedEntry(
            canonical_id=capability.canonical_id,
            module_path=capability.module_path,
            source_kind=capability.source_kind,
            mapping_status="active_mapped",
            active=True,
            onnx_op_types=tuple(str(row.onnx_op_type) for row in origins),
            onnx_node_names=tuple(str(row.canonical_node_name) for row in origins),
            graph_indices=tuple(int(row.graph_index) for row in origins),
            weight_initializers=tuple(sorted({str(row.weight_initializer) for row in origins})),
            call_count=len(origins),
        ))
    unknown = sorted(set(origins_by_module) - {row.module_path for row in audit.weighted_ops})
    if unknown:
        unresolved.extend(
            {"module_path": module_path, "reason": "onnx_weighted_call_missing_from_family_audit"}
            for module_path in unknown
        )
    functional_groups = tuple(
        {
            "canonical_id": str(row.canonical_node_name),
            "mapping_status": "mapped_but_not_a_precision_gene",
            "op_type": str(row.onnx_op_type),
            "node_names": list(row.original_node_names),
            "graph_indices": list(row.graph_indices),
            "precision_policy": str(row.protected_precision),
            "reason": str(row.protection_reason),
        }
        for row in origin_map.functional_compute_groups
    )
    mapping = ModelFamilyOnnxMapping(
        schema_version="heal-lidar-baseline-onnx-mapping-v1",
        family_id=family_id,
        source_onnx=str(Path(onnx_path).resolve()),
        audit_hash=audit.to_dict()["audit_hash"],
        weighted_entries=tuple(entries),
        parameter_free_compute_groups=functional_groups,
        activation_only_einsum_nodes=(),
        unresolved=tuple(unresolved),
        metadata={
            "static_weighted_capability_count": len(audit.weighted_ops),
            "active_weighted_capability_count": sum(row.active for row in entries),
            "active_weighted_compute_node_count": sum(row.call_count for row in entries if row.active),
            "realized_graph_mapping_complete": not unresolved,
            "parameter_free_grid_matmul_node_count": len(origin_map.functional_matmul_nodes),
            "input_contract": "heal_lidar_baseline_fixed_k",
        },
    )
    if unresolved:
        raise RuntimeError(f"heal_lidar_baseline_onnx_mapping_incomplete:{unresolved}")
    return mapping


def canonicalize_heal_lidar_baseline_onnx(
    onnx_path: str | Path,
    module_calls: Sequence[Mapping[str, Any] | Any],
    audit: ModelFamilyAudit,
    *,
    output_path: str | Path | None = None,
) -> tuple[Any, ModelFamilyOnnxMapping]:
    origin = build_onnx_origin_map(onnx_path, module_calls)
    destination = output_path or onnx_path
    apply_canonical_node_names(
        onnx_path,
        origin,
        output_path=destination,
        allow_custom_ops=True,
    )
    return origin, build_heal_lidar_baseline_onnx_mapping(destination, audit, origin)


def export_heal_lidar_baseline_fixed_k_onnx(
    model: nn.Module,
    ego_batch: Mapping[str, Any],
    output_path: str | Path,
    *,
    audit: ModelFamilyAudit,
    policy: HealLidarBaselineExportPolicy,
    opset_version: int = 17,
) -> HealLidarBaselineOnnxExport:
    """Export the six-input wrapper, prove parity, and canonicalize all weights."""

    family_id = _family_id(audit)
    expected_fusion = {
        "heal_lidar_fcooper": "MaxFusion",
        "heal_lidar_disco": "DiscoFusion",
        "heal_lidar_attfusion": "AttFusion",
        "heal_lidar_cobevt": "CoBEVT",
    }[family_id]
    if type(model.fusion_net).__name__ != expected_fusion:
        raise RuntimeError(f"heal_lidar_baseline_export_fusion_mismatch:{type(model.fusion_net).__name__}")
    wrapper = build_heal_lidar_baseline_export_module(model, policy=policy).eval()
    prepared = prepare_heal_lidar_baseline_inputs(ego_batch, policy=policy)
    if tuple(prepared) != HEAL_LIDAR_BASELINE_INPUT_NAMES:
        raise RuntimeError(f"heal_lidar_baseline_export_input_order:{tuple(prepared)}")
    tensors = tuple(prepared[name] for name in HEAL_LIDAR_BASELINE_INPUT_NAMES)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with _serialized_heal_lidar_onnx_export():
        with torch.inference_mode():
            reference = model(dict(ego_batch))
            actual = wrapper(*tensors)
        parity = _parity(reference, actual, HEAL_LIDAR_BASELINE_OUTPUT_NAMES)
        if not parity["passed"]:
            raise RuntimeError(f"heal_lidar_baseline_wrapper_parity_failed:{parity}")
        with capture_weighted_module_calls(wrapper) as calls:
            torch.onnx.export(
                wrapper,
                tensors,
                str(destination),
                export_params=True,
                opset_version=int(opset_version),
                do_constant_folding=True,
                input_names=list(HEAL_LIDAR_BASELINE_INPUT_NAMES),
                output_names=list(HEAL_LIDAR_BASELINE_OUTPUT_NAMES),
                custom_opsets={"trt": 1},
            )
    origin, family_mapping = canonicalize_heal_lidar_baseline_onnx(
        destination,
        calls,
        audit,
        output_path=destination,
    )

    import onnx

    graph = onnx.load(str(destination), load_external_data=False)
    inputs = tuple(str(row.name) for row in graph.graph.input)
    outputs = tuple(str(row.name) for row in graph.graph.output)
    plugin_nodes = tuple(
        str(node.name)
        for node in graph.graph.node
        if str(node.domain) == "trt" and str(node.op_type) == "PointPillarScatterTRT"
    )
    if inputs != HEAL_LIDAR_BASELINE_INPUT_NAMES or outputs != HEAL_LIDAR_BASELINE_OUTPUT_NAMES:
        raise RuntimeError(f"heal_lidar_baseline_onnx_contract:{inputs}:{outputs}")
    if len(plugin_nodes) != 1:
        raise RuntimeError(f"heal_lidar_baseline_scatter_plugin_count:{len(plugin_nodes)}")
    try:
        _check_onnx_with_scatter_plugin(graph)
    except Exception as exc:
        raise RuntimeError(
            f"heal_lidar_baseline_onnx_checker_failed:{type(exc).__name__}:{exc}"
        ) from exc
    export = OnnxExportResult(
        onnx_path=str(destination),
        input_names=list(inputs),
        output_names=list(outputs),
        fixed_k=int(policy.fixed_k),
        dynamic_agent_dimension=False,
        checker_passed=True,
        origin_map=origin,
    )
    return HealLidarBaselineOnnxExport(
        export=export,
        family_mapping=family_mapping,
        input_shapes={name: tuple(int(value) for value in prepared[name].shape) for name in inputs},
        plugin_nodes=plugin_nodes,
        wrapper_parity=parity,
    )


def build_heal_lidar_baseline_quantization_groups(
    model: nn.Module,
    audit: ModelFamilyAudit,
    *,
    active_module_paths: Sequence[str] | None = None,
) -> list[QuantizationSearchGroup]:
    """Build one audited quantization gene per realized weighted module."""

    _family_id(audit)
    modules = dict(model.named_modules())
    capabilities = {row.module_path: row for row in audit.weighted_ops if row.source_kind == "module"}
    active = sorted(capabilities if active_module_paths is None else {str(value) for value in active_module_paths})
    unknown = sorted(set(active) - set(capabilities))
    if unknown:
        raise RuntimeError(f"heal_lidar_quantization_active_modules_unknown:{unknown}")
    groups: list[QuantizationSearchGroup] = []
    for ordering, module_path in enumerate(active):
        module = modules.get(module_path)
        capability = capabilities[module_path]
        if module is None or getattr(module, "weight", None) is None:
            raise RuntimeError(f"heal_lidar_quantization_module_missing:{module_path}")
        allowed = tuple(str(value).upper() for value in capability.allowed_precisions)
        protected = "INT8" not in allowed
        groups.append(QuantizationSearchGroup(
            group_id=f"heal_lidar_qg::module::{module_path}",
            module_paths=(module_path,),
            canonical_node_ids=(),
            allowed_precisions=allowed,
            protected=protected,
            protection_reason=capability.gate_reason if protected else "",
            ordering=ordering,
            parameter_count=sum(int(parameter.numel()) for parameter in module.parameters(recurse=False)),
            baseline_macs=float(module.weight.numel()),
            metadata={
                "family_id": audit.family_id,
                "capability_id": capability.canonical_id,
                "default_precision": capability.default_precision,
                "output_boundary": capability.output_boundary,
                "production_enabled": capability.production_enabled,
                "force_same_precision": True,
            },
        ))
    return groups


def _semantic_merge_nodes(
    canonical_onnx_path: str | Path,
    family_id: str,
    *,
    precision: str = "fp16",
) -> dict[str, str]:
    """Select feature merges explicitly; shape-construction Concats are excluded."""

    import onnx

    graph = onnx.load(str(canonical_onnx_path), load_external_data=False)
    actual = {str(node.name): str(node.op_type) for node in graph.graph.node}
    required = {"/Concat": "Concat"}
    if family_id == "heal_lidar_disco":
        required["/Concat_5"] = "Concat"
    missing = {
        name: {"expected": op_type, "actual": actual.get(name, "missing")}
        for name, op_type in required.items()
        if actual.get(name) != op_type
    }
    if missing:
        raise RuntimeError(f"heal_lidar_semantic_merge_nodes_missing:{family_id}:{missing}")
    return {name: _normalize_auxiliary_precision(precision) for name in required}


def _fusion_island_nodes(
    canonical_onnx_path: str | Path,
    family_id: str,
    *,
    precision: str = "fp16",
) -> dict[str, str]:
    import onnx

    graph = onnx.load(str(canonical_onnx_path), load_external_data=False)
    actual = {str(node.name): str(node.op_type) for node in graph.graph.node}
    if family_id == "heal_lidar_fcooper":
        required = {
            "/GridSample": "GridSample",
            "/Mul_6": "Mul",
            "/Where_2": "Where",
            "/ReduceMax": "ReduceMax",
        }
    else:
        required = {
            "/GridSample": "GridSample",
            "/Mul_6": "Mul",
            "/Concat_5": "Concat",
            "/Where_3": "Where",
            "/Softmax": "Softmax",
            "/Expand_2": "Expand",
            "/Mul_9": "Mul",
            "/ReduceSum": "ReduceSum",
        }
    missing = {
        name: {"expected": op_type, "actual": actual.get(name, "missing")}
        for name, op_type in required.items()
        if actual.get(name) != op_type
    }
    if missing:
        raise RuntimeError(f"heal_lidar_fusion_island_nodes_missing:{family_id}:{missing}")
    return {name: _normalize_auxiliary_precision(precision) for name in required}


def _normalize_auxiliary_precision(value: str) -> str:
    precision = str(value).strip().lower()
    if precision not in {"fp16", "fp32"}:
        raise RuntimeError(
            f"unsupported_heal_lidar_auxiliary_precision:{value}"
        )
    return precision


def partition_heal_lidar_precision_profile(
    origin_modules: Sequence[str],
    module_precision_profile: Mapping[str, str],
    *,
    functional_precision_paths: Sequence[str] = (),
) -> tuple[dict[str, str], dict[str, str]]:
    """Separate weighted origins from parameter-free deployment boundaries.

    Both partitions remain exhaustive and fail closed.  Functional paths are
    accepted only when explicitly supplied by the frozen Transformer
    precision-unit inventory; an arbitrary unknown key is never ignored.
    """

    complete = {
        str(key): str(value).lower()
        for key, value in module_precision_profile.items()
    }
    weighted_paths = {str(value) for value in origin_modules}
    functional_paths = {str(value) for value in functional_precision_paths}
    overlap = sorted(weighted_paths & functional_paths)
    if overlap:
        raise RuntimeError(
            f"heal_lidar_weighted_functional_precision_overlap:{overlap}"
        )
    weighted = {
        key: value for key, value in complete.items() if key in weighted_paths
    }
    functional = {
        key: value for key, value in complete.items() if key in functional_paths
    }
    missing = sorted(weighted_paths - set(weighted))
    missing_functional = sorted(functional_paths - set(functional))
    unknown = sorted(set(complete) - weighted_paths - functional_paths)
    if missing or missing_functional or unknown:
        raise RuntimeError(
            "heal_lidar_precision_profile_origin_mismatch:"
            f"missing={missing}:missing_functional={missing_functional}:"
            f"unknown={unknown}"
        )
    return weighted, functional


def build_heal_lidar_baseline_precision_mapping(
    origin_map: Any,
    module_precision_profile: Mapping[str, str],
    *,
    audit: ModelFamilyAudit,
    canonical_onnx_path: str | Path,
    profile_id: str,
    auxiliary_precision: str = "fp16",
    precision_policy: str = "legacy_family_static_dependency_closure_v1",
    runtime_precision_relations: Sequence[Any] = (),
    module_to_precision_group: Mapping[str, str] | None = None,
    functional_precision_paths: Sequence[str] = (),
    functional_auxiliary_precision_overrides: Mapping[str, str] | None = None,
    weighted_output_precision_overrides: Mapping[str, str] | None = None,
) -> tuple[CanonicalPrecisionMappingResult, dict[str, Any]]:
    """Expand module genes and enforce the family auxiliary precision contract."""

    family_id = _family_id(audit)
    policy_name = str(precision_policy).strip().lower()
    generic_runtime = policy_name in {
        "heal_runtime_graph_v1",
        "pyramid_runtime_tracer_v1",
    }
    if policy_name not in {
        "legacy_family_static_dependency_closure_v1",
        "heal_runtime_graph_v1",
        "pyramid_runtime_tracer_v1",
    }:
        raise RuntimeError(f"unsupported_heal_lidar_precision_policy:{policy_name}")
    auxiliary_precision = _normalize_auxiliary_precision(auxiliary_precision)
    origin_modules = {str(row.module_path) for row in origin_map.entries}
    profile, functional_profile = partition_heal_lidar_precision_profile(
        sorted(origin_modules),
        module_precision_profile,
        functional_precision_paths=functional_precision_paths,
    )
    complete_profile_count = len(module_precision_profile)
    capabilities = {row.module_path: row for row in audit.weighted_ops}
    unaudited = sorted(origin_modules - set(capabilities))
    if unaudited and not generic_runtime:
        raise RuntimeError(f"heal_lidar_precision_origin_modules_unaudited:{unaudited}")
    invalid: dict[str, str] = {}
    for module_path, precision in profile.items():
        allowed = (
            {"fp32", "fp16", "int8"}
            if generic_runtime
            else {
                str(value).lower()
                for value in capabilities[module_path].allowed_precisions
            }
        )
        if precision not in allowed:
            invalid[module_path] = precision
    if invalid:
        raise RuntimeError(f"heal_lidar_precision_profile_capability_violation:{invalid}")
    precision_groups = {
        str(key): str(value)
        for key, value in dict(module_to_precision_group or {}).items()
    }
    entries = [
        CanonicalPrecisionEntry(
            module_path=str(origin.module_path),
            canonical_node_name=str(origin.canonical_node_name),
            original_node_name=str(origin.original_node_name),
            weight_initializer=str(origin.weight_initializer),
            onnx_op_type=str(origin.onnx_op_type),
            call_index=int(origin.call_index),
            precision_group=precision_groups.get(
                str(origin.module_path),
                f"heal_lidar_qg::module::{origin.module_path}",
            ),
            requested_precision=profile[str(origin.module_path)],
            realized_request_precision=profile[str(origin.module_path)],
            realized_output_precision=(
                profile[str(origin.module_path)]
                if generic_runtime
                else "fp16" if profile[str(origin.module_path)] in {"fp16", "int8"} else "fp32"
            ),
            protected_precision=(
                ""
                if generic_runtime
                else "fp16"
                if "INT8" not in capabilities[str(origin.module_path)].allowed_precisions
                else ""
            ),
        )
        for origin in sorted(origin_map.entries, key=lambda row: (row.call_index, row.graph_index))
    ]
    mapping = CanonicalPrecisionMappingResult(
        entries=entries,
        profile_id=str(profile_id),
        profile_hash=stable_json_hash(profile),
        origin_map_hash=str(origin_map.origin_map_hash),
        policy_version="heal-lidar-baseline-explicit-qdq-strong-type-v1",
    )
    if generic_runtime:
        mapping, adaptive_report = apply_adaptive_merge_output_contract(
            canonical_onnx_path,
            mapping,
            runtime_relations=runtime_precision_relations,
        )
        weighted_output_overrides = {
            str(name): _normalize_auxiliary_precision(precision)
            for name, precision in dict(
                weighted_output_precision_overrides or {}
            ).items()
        }
        missing_weighted_output_modules = sorted(
            set(weighted_output_overrides)
            - {str(row.module_path) for row in mapping.entries}
        )
        if missing_weighted_output_modules:
            raise RuntimeError(
                "weighted_output_precision_override_invalid:"
                f"missing={missing_weighted_output_modules}"
            )
        if weighted_output_overrides:
            mapping = replace(
                mapping,
                entries=[
                    replace(
                        row,
                        realized_output_precision=weighted_output_overrides.get(
                            str(row.module_path), row.realized_output_precision
                        ),
                    )
                    for row in mapping.entries
                ],
                policy_version=(
                    "canonical-precision-mapping-adaptive-runtime-merge-"
                    "protected-weighted-output-v3"
                ),
                mapping_hash="",
            )
            adaptive_report = {
                **dict(adaptive_report),
                "weighted_output_precision_overrides": weighted_output_overrides,
                "mapping_hash_after": mapping.mapping_hash,
            }
            adaptive_report["contract_hash"] = stable_json_hash(adaptive_report)
        functional_overrides = {
            str(name): _normalize_auxiliary_precision(precision)
            for name, precision in dict(
                functional_auxiliary_precision_overrides or {}
            ).items()
        }
        if functional_overrides:
            import onnx

            graph = onnx.load(str(canonical_onnx_path), load_external_data=False)
            nodes = {str(node.name): str(node.op_type) for node in graph.graph.node}
            missing_nodes = sorted(set(functional_overrides) - set(nodes))
            unsupported_nodes = sorted(
                name
                for name in functional_overrides
                if nodes.get(name) not in {"Add", "Concat"}
            )
            if missing_nodes or unsupported_nodes:
                raise RuntimeError(
                    "functional_auxiliary_precision_override_invalid:"
                    f"missing={missing_nodes}:unsupported={unsupported_nodes}"
                )
            mapping = replace(
                mapping,
                policy_version=(
                    "canonical-precision-mapping-adaptive-runtime-merge-"
                    + (
                        "protected-weighted-output-fixed-functional-v4"
                        if weighted_output_overrides
                        else "fixed-functional-v3"
                    )
                ),
                auxiliary_layer_precisions={
                    **dict(mapping.auxiliary_layer_precisions),
                    **functional_overrides,
                },
                auxiliary_layer_output_types={
                    **dict(mapping.auxiliary_layer_output_types),
                    **functional_overrides,
                },
                mapping_hash="",
            )
            updated_merges = []
            for row in adaptive_report.get("merges", ()):
                updated = dict(row)
                node_name = str(updated.get("merge_op_name", ""))
                if node_name in functional_overrides:
                    updated["runtime_derived_precision_before_functional_override"] = str(
                        updated.get("derived_merge_precision", "")
                    )
                    updated["derived_merge_precision"] = functional_overrides[
                        node_name
                    ]
                    updated["input_cast_nodes"] = [
                        adaptive_merge_cast_name(
                            node_name,
                            int(index),
                            functional_overrides[node_name],
                        )
                        for index in updated.get("dtype_audit", {}).get(
                            "data_input_indices", ()
                        )
                    ]
                    updated["fixed_functional_precision_override"] = True
                updated_merges.append(updated)
            adaptive_report = {
                **dict(adaptive_report),
                "merges": updated_merges,
                "fixed_functional_precision_overrides": functional_overrides,
                "mapping_hash_after": mapping.mapping_hash,
            }
            adaptive_report["contract_hash"] = stable_json_hash(adaptive_report)
        report = {
            "schema_version": "heal-runtime-adaptive-merge-contract-v1",
            "family_id": family_id,
            "weighted_compute_precision_policy": "heal_runtime_graph_v1",
            "family_audit_used_for_precision_protection": False,
            "family_named_node_rules_used": False,
            "adaptive_merge_contract": adaptive_report,
            "mapping_hash": mapping.mapping_hash,
            "weighted_precision_profile": dict(sorted(profile.items())),
            "functional_precision_profile": dict(
                sorted(functional_profile.items())
            ),
            "functional_auxiliary_precision_overrides": dict(
                sorted(functional_overrides.items())
            ),
            "weighted_output_precision_overrides": dict(
                sorted(weighted_output_overrides.items())
            ),
            "weighted_precision_path_count": len(profile),
            "functional_precision_path_count": len(functional_profile),
            "precision_profile_partition_exact": (
                len(profile) + len(functional_profile) == complete_profile_count
            ),
        }
        report["island_hash"] = stable_json_hash(report)
        return mapping, report

    semantic_merges = _semantic_merge_nodes(
        canonical_onnx_path,
        family_id,
        precision=auxiliary_precision,
    )
    island_nodes = _fusion_island_nodes(
        canonical_onnx_path,
        family_id,
        precision=auxiliary_precision,
    )
    auxiliary = {**semantic_merges, **island_nodes}
    mapping = CanonicalPrecisionMappingResult(
        entries=list(mapping.entries),
        profile_id=mapping.profile_id,
        profile_hash=mapping.profile_hash,
        origin_map_hash=mapping.origin_map_hash,
        policy_version=(
            "heal-lidar-baseline-explicit-qdq-"
            f"{auxiliary_precision}-fusion-island-v1"
        ),
        auxiliary_layer_precisions=auxiliary,
        auxiliary_layer_output_types=auxiliary,
    )
    protected_fusion_modules = sorted(
        module_path
        for module_path in profile
        if module_path.startswith("fusion_net.") and profile[module_path] == "int8"
    )
    if protected_fusion_modules:
        raise RuntimeError(f"heal_lidar_fusion_weighted_int8_forbidden:{protected_fusion_modules}")
    report = {
        "schema_version": "heal-lidar-fusion-island-v1",
        "family_id": family_id,
        "auxiliary_precision": auxiliary_precision,
        "fusion_nodes": island_nodes,
        "weighted_fusion_modules": sorted(
            module_path for module_path in profile if module_path.startswith("fusion_net.")
        ),
        "weighted_fusion_int8_forbidden": True,
        "weighted_compute_precision_policy": policy_name,
        "semantic_merge_policy": "explicit_named_feature_merges_only_shape_concats_excluded",
        "semantic_merge_nodes": semantic_merges,
        "mapping_hash": mapping.mapping_hash,
        "weighted_precision_profile": dict(sorted(profile.items())),
        "functional_precision_profile": dict(sorted(functional_profile.items())),
        "weighted_precision_path_count": len(profile),
        "functional_precision_path_count": len(functional_profile),
        "precision_profile_partition_exact": (
            len(profile) + len(functional_profile) == complete_profile_count
        ),
    }
    report["island_hash"] = stable_json_hash(report)
    return mapping, report


def insert_heal_lidar_baseline_explicit_qdq(
    input_onnx: str | Path,
    output_onnx: str | Path,
    mapping: CanonicalPrecisionMappingResult,
    *,
    family: str | ModelFamilyAudit,
    scales: Mapping[str, Any],
    config: QDQConfig | None = None,
    calibration_metadata: Mapping[str, Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Insert exact Q/DQ and re-audit the semantic fusion island."""

    family_id = _family_id(family)
    adaptive_runtime = "adaptive-runtime-merge" in str(mapping.policy_version)
    if adaptive_runtime:
        adaptive_config = (
            replace(config, merge_policy="adaptive_upcast_merge")
            if config is not None
            else QDQConfig(
                allowed_precisions=("fp32", "fp16", "int8"),
                merge_policy="adaptive_upcast_merge",
                grouped_conv_int8_allowed_channels_per_group=(
                    4, 8, 16, 32, 64, 128, 256, 512
                ),
                policy_version="explicit-qdq-adaptive-runtime-merge-v1",
            )
        )
        result = insert_explicit_qdq(
            input_onnx,
            output_onnx,
            mapping,
            scales=scales,
            config=adaptive_config,
            calibration_metadata=calibration_metadata,
        )
        inserted_auxiliary = dict(
            result.calibration_metadata.get("auxiliary_layer_precisions", {})
        )
        inserted_outputs = dict(
            result.calibration_metadata.get("auxiliary_layer_output_types", {})
        )
        merge_audit = list(
            result.calibration_metadata.get("merge_quantization_audit", [])
        )
        expected_auxiliary = dict(mapping.auxiliary_layer_precisions)
        expected_outputs = dict(mapping.auxiliary_layer_output_types)
        audited_merges = {
            str(row.get("merge_op_name", "")): str(
                row.get("derived_merge_precision", "")
            ).lower()
            for row in merge_audit
        }
        expected_audited_merges = {
            str(name): str(precision).lower()
            for name, precision in expected_auxiliary.items()
        }
        audit = {
            "schema_version": "heal-runtime-adaptive-merge-qdq-audit-v1",
            "family_id": family_id,
            "passed": bool(
                inserted_auxiliary == expected_auxiliary
                and inserted_outputs == expected_outputs
                and audited_merges == expected_audited_merges
            ),
            "family_named_node_audit_used": False,
            "merge_policy": "adaptive_upcast_merge",
            "promotion_order": ["int8", "fp16", "fp32"],
            "expected_auxiliary_precisions": expected_auxiliary,
            "inserted_auxiliary_precisions": inserted_auxiliary,
            "inserted_auxiliary_output_types": inserted_outputs,
            "audited_merge_precisions": audited_merges,
            "merge_quantization_audit": merge_audit,
            "weighted_qdq_boundary_audit": result.calibration_metadata.get(
                "weighted_qdq_boundary_audit", []
            ),
            "inserted_int8_layer_count": int(result.inserted_layer_count),
            "requested_int8_count": int(result.requested_int8_count),
            "qdq_output_sha256": str(result.output_sha256),
        }
        audit["audit_hash"] = stable_json_hash(audit)
        if not audit["passed"]:
            raise RuntimeError("heal_runtime_adaptive_merge_qdq_audit_failed")
        return result, audit

    required_node_names = set(_semantic_merge_nodes(input_onnx, family_id)) | set(
        _fusion_island_nodes(input_onnx, family_id)
    )
    missing_auxiliary = sorted(
        required_node_names - set(mapping.auxiliary_layer_precisions)
    )
    missing_outputs = sorted(
        required_node_names - set(mapping.auxiliary_layer_output_types)
    )
    if missing_auxiliary or missing_outputs:
        raise RuntimeError(
            "heal_lidar_fusion_island_missing_from_mapping:"
            f"precision={missing_auxiliary}:output={missing_outputs}"
        )
    auxiliary_precisions = {
        _normalize_auxiliary_precision(mapping.auxiliary_layer_precisions[name])
        for name in required_node_names
    }
    if len(auxiliary_precisions) != 1:
        raise RuntimeError(
            "heal_lidar_auxiliary_precision_not_uniform:"
            f"{sorted(auxiliary_precisions)}"
        )
    auxiliary_precision = next(iter(auxiliary_precisions))
    auxiliary_outputs = {
        _normalize_auxiliary_precision(mapping.auxiliary_layer_output_types[name])
        for name in required_node_names
    }
    if auxiliary_outputs != {auxiliary_precision}:
        raise RuntimeError(
            "heal_lidar_auxiliary_output_precision_mismatch:"
            f"compute={auxiliary_precision}:output={sorted(auxiliary_outputs)}"
        )
    expected_nodes = _fusion_island_nodes(
        input_onnx,
        family_id,
        precision=auxiliary_precision,
    )
    expected_merges = _semantic_merge_nodes(
        input_onnx,
        family_id,
        precision=auxiliary_precision,
    )
    result = insert_explicit_qdq(
        input_onnx,
        output_onnx,
        mapping,
        scales=scales,
        config=config,
        calibration_metadata=calibration_metadata,
    )
    realized_nodes = _fusion_island_nodes(
        output_onnx,
        family_id,
        precision=auxiliary_precision,
    )
    realized_merges = _semantic_merge_nodes(
        output_onnx,
        family_id,
        precision=auxiliary_precision,
    )
    inserted_auxiliary = {
        str(name): str(precision)
        for name, precision in dict(
            result.calibration_metadata.get("auxiliary_layer_precisions", {})
        ).items()
        if str(name) in required_node_names
    }
    inserted_outputs = {
        str(name): str(precision)
        for name, precision in dict(
            result.calibration_metadata.get("auxiliary_layer_output_types", {})
        ).items()
        if str(name) in required_node_names
    }
    expected_auxiliary = {**expected_merges, **expected_nodes}
    audit = {
        "schema_version": "heal-lidar-qdq-fusion-island-audit-v1",
        "family_id": family_id,
        "passed": bool(
            realized_nodes == expected_nodes
            and realized_merges == expected_merges
            and inserted_auxiliary == expected_auxiliary
            and inserted_outputs == expected_auxiliary
        ),
        "auxiliary_precision": auxiliary_precision,
        "fusion_nodes": realized_nodes,
        "semantic_merge_nodes": realized_merges,
        "expected_auxiliary_precisions": expected_auxiliary,
        "inserted_auxiliary_precisions": inserted_auxiliary,
        "inserted_auxiliary_output_types": inserted_outputs,
        "inserted_int8_layer_count": int(result.inserted_layer_count),
        "requested_int8_count": int(result.requested_int8_count),
        "qdq_output_sha256": str(result.output_sha256),
        "merge_quantization_audit": result.calibration_metadata.get("merge_quantization_audit", []),
        "weighted_qdq_boundary_audit": result.calibration_metadata.get("weighted_qdq_boundary_audit", []),
    }
    audit["audit_hash"] = stable_json_hash(audit)
    if not audit["passed"]:
        raise RuntimeError("heal_lidar_qdq_fusion_island_audit_failed")
    return result, audit


def validate_heal_lidar_fusion_island_realization(
    layer_info: Any,
    *,
    family: str | ModelFamilyAudit,
    required_precision: str = "fp16",
    required_nodes: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Prove fused TensorRT warp/fusion kernels realize the mapping contract."""

    family_id = _family_id(family)
    required_precision = _normalize_auxiliary_precision(required_precision)
    default_nodes = (
        ("/GridSample", "/Mul_6", "/Where_2", "/ReduceMax")
        if family_id == "heal_lidar_fcooper"
        else ("/GridSample", "/Mul_6", "/Concat_5", "/Where_3", "/Softmax", "/Expand_2", "/Mul_9", "/ReduceSum")
    )
    nodes_to_check = tuple(required_nodes) if required_nodes is not None else default_nodes
    rows = load_layer_info(layer_info)
    findings = []
    issues = []
    for node_name in nodes_to_check:
        matches = [row for row in rows if has_canonical_identity(row, node_name)]
        precisions = sorted({precision_name(row) for row in matches if precision_name(row)})
        findings.append({
            "onnx_node_name": node_name,
            "matched_layer_count": len(matches),
            "matched_layer_names": [str(row.get("Name") or row.get("name") or "") for row in matches],
            "realized_precisions": precisions,
        })
        if not matches:
            issues.append(f"fusion_node_unresolved:{node_name}")
        elif "int8" in precisions:
            issues.append(f"fusion_node_realized_int8:{node_name}")
        elif required_precision and required_precision not in precisions:
            issues.append(f"fusion_node_not_{required_precision}:{node_name}:{precisions}")
    return {
        "schema_version": "heal-lidar-trt-fusion-island-realization-v1",
        "family_id": family_id,
        "required_precision": required_precision,
        "passed": not issues,
        "issues": issues,
        "findings": findings,
        "unique_matched_layer_count": len({name for row in findings for name in row["matched_layer_names"]}),
    }


def validate_heal_lidar_precision_realization(
    layer_info: Any,
    mapping: CanonicalPrecisionMappingResult,
    *,
    family: str | ModelFamilyAudit,
) -> dict[str, Any]:
    weighted = validate_precision_realization(layer_info, mapping)
    family_id = _family_id(family)
    auxiliary = {
        str(name): _normalize_auxiliary_precision(precision)
        for name, precision in mapping.auxiliary_layer_precisions.items()
    }
    if not auxiliary:
        raise RuntimeError("heal_lidar_auxiliary_precision_mapping_missing")
    auxiliary_outputs = {
        str(name): _normalize_auxiliary_precision(precision)
        for name, precision in mapping.auxiliary_layer_output_types.items()
    }
    if auxiliary_outputs != auxiliary:
        raise RuntimeError(
            "heal_lidar_auxiliary_output_precision_mismatch:"
            f"compute={auxiliary}:output={auxiliary_outputs}"
        )
    required_precisions = set(auxiliary.values())
    if len(required_precisions) != 1:
        raise RuntimeError(
            "heal_lidar_auxiliary_precision_not_uniform:"
            f"{sorted(required_precisions)}"
        )
    required_precision = next(iter(required_precisions))
    fusion = validate_heal_lidar_fusion_island_realization(
        layer_info,
        family=family_id,
        required_precision=required_precision,
    )
    return {
        "schema_version": "heal-lidar-trt-precision-acceptance-v1",
        "passed": bool(weighted.passed and fusion["passed"]),
        "weighted_precision": weighted.to_dict(),
        "fusion_island": fusion,
    }


__all__ = [
    "HEAL_LIDAR_BASELINE_INPUT_NAMES",
    "HEAL_LIDAR_BASELINE_OUTPUT_NAMES",
    "HealLidarBaselineOnnxExport",
    "build_heal_lidar_baseline_onnx_mapping",
    "build_heal_lidar_baseline_precision_mapping",
    "build_heal_lidar_baseline_quantization_groups",
    "canonicalize_heal_lidar_baseline_onnx",
    "export_heal_lidar_baseline_fixed_k_onnx",
    "insert_heal_lidar_baseline_explicit_qdq",
    "partition_heal_lidar_precision_profile",
    "validate_heal_lidar_fusion_island_realization",
    "validate_heal_lidar_precision_realization",
]
