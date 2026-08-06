"""Real lidar_pyramid search context assembly."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn

from deploy.post_scatter import (
    POST_SCATTER_CONTRACT,
    filter_post_scatter_module_paths,
    filter_post_scatter_pruning_units,
    filter_post_scatter_quantization_groups,
    post_scatter_shape_profiles,
)

from ..canonicalization import SearchSpaceSpec
from ..hashing import canonical_json_hash
from ..pruning_space.action_catalog import PruningActionCatalog, build_pruning_action_catalog
from ..pruning_space.local_domains import build_local_pruning_domains
from ..quantization_space.group_builder import build_quantization_search_groups
from .data_provider import EvaluationManifest, load_split_frame_ids, write_eval_manifest
from .model_provider import LidarPyramidModelBundle, load_lidar_pyramid_model
from .runtime_environment import GPUSelection, TensorRTEnvironment, discover_trt_environment, plugin_hashes, select_gpu


DEFAULT_CHECKPOINT = Path("${MODEL_ROOT}/lidar_pyramid/net_epoch_bestval_at17.pth")
DEFAULT_CONFIG = Path("${MODEL_ROOT}/lidar_pyramid/config.yaml")
DEFAULT_HEAL_ROOT = Path("../../HEAL")
DEFAULT_TRT_ROOT = Path("${TENSORRT_ROOT}")
DEFAULT_PLUGIN = Path("quantization/plugins/pointpillar_scatter_trt/build/libpointpillar_scatter_trt.so")


@dataclass
class LidarPyramidSearchContext:
    checkpoint_path: Path
    model: torch.nn.Module
    model_config: Path
    model_bundle: LidarPyramidModelBundle
    trace_result: Any
    atomic_prune_units: list[Any]
    coupled_channel_units: list[Any]
    pruning_action_catalog: PruningActionCatalog | None
    trace_example_inputs: Any
    export_example_inputs: Any
    fisher_calibration_batches: int
    quant_calibration_batches: int
    quant_calibration_npz_manifest: Path | None
    quant_activation_calibration_backend: str
    quant_activation_calibration_cache_path: Path | None
    quant_calibration_force_rebuild: bool
    eval_manifest_path: Path
    eval_frame_ids: list[str]
    postprocess_fn: Callable[..., Any] | None
    metric_adapter: Any
    trt_config: Any
    builder_flags: dict[str, Any]
    plugin_paths: list[Path]
    physical_gpu_id: int
    runtime_device: str
    gpu_selection: GPUSelection
    tensorrt: TensorRTEnvironment
    search_space: SearchSpaceSpec
    eval_manifest_hash: str
    checkpoint_hash: str


def _module_is_weighted(module: nn.Module) -> bool:
    return isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear))


def _safe_pruning_unit(unit: Any) -> bool:
    if bool(getattr(unit, "protected", False)):
        return False
    constraints = dict(getattr(unit, "constraints", {}) or {})
    path = str(getattr(unit, "root_module_path", ""))
    lower = path.lower()
    if any(token in lower for token in ("pillar_vfe", "scatter", "single_head", "cls_head", "reg_head", "dir_head")):
        return False
    if not any(token in lower for token in ("pyramid_backbone", "backbone", "shrink", "compressor", "aligner")):
        return False
    indices = list(getattr(unit, "root_indices", []) or [])
    return bool(indices)


def _select_safe_atomic_units(model: nn.Module, atomic_units: list[Any], *, max_units: int = 96) -> list[Any]:
    modules = dict(model.named_modules())
    grouped_by_root: dict[str, list[Any]] = {}
    for unit in atomic_units:
        if not _safe_pruning_unit(unit):
            continue
        module = modules.get(str(getattr(unit, "root_module_path", "")))
        if module is None or not _module_is_weighted(module):
            continue
        if str(getattr(unit, "root_axis", "")) not in {"out", "channel"}:
            continue
        width = int(getattr(module, "out_channels", getattr(module, "out_features", 0)) or 0)
        if width <= 16:
            continue
        grouped_by_root.setdefault(str(getattr(unit, "root_module_path")), []).append(unit)
    if not grouped_by_root:
        raise RuntimeError("real_trace_search_space_empty")
    def rank(item: tuple[str, list[Any]]) -> tuple[int, str]:
        path, units = item
        preferred = 0 if "pyramid_backbone.resnet.layer1.0.conv3" in path else 1
        return (preferred, path)
    _path, units = sorted(grouped_by_root.items(), key=rank)[0]
    units = sorted(units, key=lambda row: (min(getattr(row, "root_indices", [0]) or [0]), str(getattr(row, "stable_id"))))
    keep = units[: max(1, min(int(max_units), len(units)))]
    if any(str(getattr(unit, "stable_id", "")).startswith("unit") for unit in keep):
        raise RuntimeError("real_trace_contains_dryrun_unit_id")
    return keep


def _select_search_atomic_units(model: nn.Module, atomic_units: list[Any], *, max_dense_units: int = 96) -> list[Any]:
    dense = _select_safe_atomic_units(model, atomic_units, max_units=max_dense_units)
    modules = dict(model.named_modules())
    grouped_by_scope: dict[str, list[Any]] = {}
    for unit in atomic_units:
        if bool(getattr(unit, "protected", False)):
            continue
        constraints = dict(getattr(unit, "constraints", {}) or {})
        if not (constraints.get("grouped_conv") and not constraints.get("depthwise")):
            continue
        module = modules.get(str(getattr(unit, "root_module_path", "")))
        if module is None or not _module_is_weighted(module):
            continue
        if str(getattr(unit, "root_axis", "")) not in {"out", "channel"}:
            continue
        grouped_by_scope.setdefault(str(getattr(unit, "scope_id", "")), []).append(unit)
    grouped: list[Any] = []
    for _scope, rows in sorted(grouped_by_scope.items())[:4]:
        grouped.extend(sorted(rows, key=lambda row: (min(getattr(row, "root_indices", [0]) or [0]), str(getattr(row, "stable_id")))))
    by_id: dict[str, Any] = {}
    for unit in [*dense, *grouped]:
        by_id[str(getattr(unit, "stable_id"))] = unit
    return list(by_id.values())


def _select_all_legal_domain_units(model: nn.Module, atomic_units: list[Any]) -> list[Any]:
    """Keep every safe formal atomic unit; width genes remove the 96-bit cap."""

    modules = dict(model.named_modules())
    selected: dict[str, Any] = {}
    for unit in atomic_units:
        if not _safe_pruning_unit(unit):
            continue
        module = modules.get(str(getattr(unit, "root_module_path", "")))
        if module is None or not _module_is_weighted(module):
            continue
        if str(getattr(unit, "root_axis", "")) not in {"out", "channel"}:
            continue
        if not list(getattr(unit, "root_indices", []) or []):
            continue
        unit_id = str(getattr(unit, "stable_id", ""))
        if not unit_id or unit_id.startswith("unit"):
            raise RuntimeError(f"invalid_formal_atomic_unit_id:{unit_id}")
        selected[unit_id] = unit
    if not selected:
        raise RuntimeError("real_trace_domain_width_search_space_empty")
    return [selected[key] for key in sorted(selected)]


def _precision_layer_ids(model: nn.Module) -> list[str]:
    rows = [name for name, module in model.named_modules() if name and _module_is_weighted(module)]
    if any(name.startswith("dryrun_model") for name in rows):
        raise RuntimeError("real_precision_space_contains_dryrun_layer")
    return filter_post_scatter_module_paths(rows)


_FUNCTIONAL_FP16_OUTPUT_BOUNDARIES: dict[str, dict[str, Any]] = {
    "pyramid_backbone.single_head_0": {
        "merge_kind": "functional_sigmoid_weight_merge",
        "following_ops": ["Sigmoid", "Add", "GridSample"],
    },
    "pyramid_backbone.single_head_1": {
        "merge_kind": "functional_sigmoid_weight_merge",
        "following_ops": ["Sigmoid", "Add", "GridSample"],
    },
    "pyramid_backbone.single_head_2": {
        "merge_kind": "functional_sigmoid_weight_merge",
        "following_ops": ["Sigmoid", "Add", "GridSample"],
    },
}

_MODEL_SPECIFIC_INT8_OVERRIDES: dict[str, str] = {
    "encoder_m1.pillar_vfe.pfn_layers.0.linear": (
        "canonical_onnx_matmul_with_weight_initializer_and_explicit_per_channel_qdq"
    ),
}


def _model_specific_int8_override(module_path: str) -> str:
    """Return audited lidar_pyramid evidence overriding blanket name filters."""

    return _MODEL_SPECIFIC_INT8_OVERRIDES.get(str(module_path), "")


def _functional_fp16_output_boundary(module_path: str) -> dict[str, Any] | None:
    """Return the model-specific compute/output split verified by the H800 ablation."""

    boundary = _FUNCTIONAL_FP16_OUTPUT_BOUNDARIES.get(str(module_path))
    if boundary is None:
        return None
    return {
        "parent_precision_group_id": "canonical_onnx_pending",
        "member_layers": [str(module_path)],
        "branch_compute_precision_independent": True,
        "merge_policy": "A_fp16_merge",
        "input_qdq_placement": "INT8 weighted compute input and per-channel weight Q/DQ",
        "output_qdq_placement": "no weighted-output Q/DQ; FP16 functional path",
        "activation_scale_ownership": "canonical weighted input; output scale is diagnostic only",
        "merge_scale_policy": "INT8 compute then FP16 Sigmoid/Add/GridSample functional weighting path",
        "boundary_source": "lidar_pyramid_export_contract_verified_by_canonical_onnx",
        **boundary,
    }


def _build_precision_groups(model: nn.Module, trace_result: Any) -> list[Any]:
    try:
        from tracer.dependency_tracer import build_dependency_graph
        from tracer.precision_coupling_tracer import PrecisionGroup, build_precision_coupling_groups
    except ImportError:
        from heal_compress.tracer.dependency_tracer import build_dependency_graph
        from heal_compress.tracer.precision_coupling_tracer import PrecisionGroup, build_precision_coupling_groups
    graph = build_dependency_graph(model, None)
    # Precision coupling is a deployment contract, not a pruning dependency
    # scope.  In particular, residual/concat branches may choose independent
    # compute precision when policy A dequantizes them before an FP16 merge.
    base_groups = build_precision_coupling_groups(model, graph, sample_batch=None, allow_head_int8=True)
    weighted = set(_precision_layer_ids(model))
    # Legacy kept PFN Linear and single_head_2 in FP16, but that historical
    # profile is not evidence of a TensorRT limitation.  Exact E67 matching is
    # expressed by its baseline profile; the legal search space keeps both as
    # independently selectable INT8 genes.
    protected: dict[str, str] = {}
    memberships: dict[str, list[tuple[int, Any, list[str]]]] = {name: [] for name in weighted}
    for group_index, group in enumerate(base_groups):
        members = [str(module) for module in group.member_modules if str(module) in weighted]
        for module in members:
            memberships[module].append((group_index, group, members))

    filtered: list[Any] = []
    used_ids: set[str] = set()
    for module in sorted(weighted):
        rows = memberships.get(module) or []
        if not rows:
            raise RuntimeError(f"weighted_module_missing_precision_contract:{module}")
        group_index, source, source_members = rows[0]
        independent_merge_branch = len(source_members) > 1 and not bool(source.force_same_precision)
        if independent_merge_branch:
            branch_index = source_members.index(module)
            group_id = f"{source.precision_group_id}__branch_{branch_index:02d}"
        else:
            group_id = str(source.precision_group_id)
        if group_id in used_ids:
            group_id = f"{group_id}__{canonical_json_hash({'module': module})[:8]}"
        used_ids.add(group_id)
        merge_boundaries = [
            {
                "parent_precision_group_id": str(parent.precision_group_id),
                "merge_kind": str(parent.reason),
                "member_layers": list(parent_members),
                "branch_compute_precision_independent": not bool(parent.force_same_precision),
                "merge_policy": "A_fp16_merge",
                "input_qdq_placement": "Q/DQ at each INT8 weighted branch; all merge inputs are float after DQ",
                "output_qdq_placement": "downstream weighted-node input owns optional requantization",
                "activation_scale_ownership": "canonical ONNX tensor boundary",
                "merge_scale_policy": "independent_branch_scales_then_DQ_to_FP16_merge_then_optional_requantization",
            }
            for _index, parent, parent_members in rows
            if len(parent_members) > 1 and str(parent.reason) in {"residual", "concat"}
        ]
        functional_output_boundary = _functional_fp16_output_boundary(module)
        if functional_output_boundary is not None:
            merge_boundaries.append(functional_output_boundary)
        fp16_output_boundary = functional_output_boundary is not None or any(
            str(boundary.get("merge_kind", "")) == "concat"
            and bool(boundary.get("branch_compute_precision_independent", False))
            for boundary in merge_boundaries
        )
        allowed = list(source.allowed_precisions)
        reason = str(source.reason)
        int8_override_evidence = _model_specific_int8_override(module)
        if int8_override_evidence:
            allowed = ["fp32", "fp16", "int8"]
            reason = "model_specific_audited_int8_override"
        if module in protected:
            allowed = ["fp32", "fp16"]
            reason = protected[module]
        contract = PrecisionGroup(
            precision_group_id=group_id,
            member_modules=[module],
            reason=reason,
            allowed_precisions=allowed,
            default_precision="fp16",
            force_same_precision=True,
        )
        setattr(
            contract,
            "deployment_contract",
            {
                "member_layers": [module],
                "merge_boundaries": merge_boundaries,
                "merge_boundary_resolution": "resolved_from_canonical_onnx_during_qdq_export",
                "input_output_qdq_placement": (
                    "canonical_weighted_input_and_weight; FP16 output before functional/merge boundary"
                    if fp16_output_boundary
                    else "canonical_weighted_input_weight_and_output"
                ),
                "activation_scale_ownership": "canonical ONNX tensor boundary",
                "merge_scale_policy": "A_fp16_merge",
                "compute_precision_policy": "owned_by_quantization_gene",
                "output_precision_policy": "FP16" if fp16_output_boundary else "same_as_compute",
                "insert_activation_output_qdq": not fp16_output_boundary,
                "weight_granularity": "per_channel",
                "weight_axis_policy": {"Conv": 0, "ConvTranspose": 1, "MatMul": 1, "Gemm": "0 if transB else 1"},
                "int8_override_evidence": int8_override_evidence,
            },
        )
        filtered.append(contract)
    return filtered


def build_lidar_pyramid_context(
    *,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    model_config_path: str | Path | None = None,
    heal_root: str | Path = DEFAULT_HEAL_ROOT,
    tensorrt_root: str | Path = DEFAULT_TRT_ROOT,
    plugin_path: str | Path | None = None,
    gpu_id: str = "auto",
    exclude_gpu_ids: list[int] | None = None,
    tensorrt_env: str = "modelopt",
    fisher_calibration_batches: int = 8,
    quant_calibration_batches: int = 16,
    quant_calibration_npz_manifest: str | Path | None = None,
    quant_activation_calibration_backend: str = "modelopt_histogram_entropy",
    quant_activation_calibration_cache_path: str | Path | None = None,
    quant_calibration_force_rebuild: bool = False,
    num_frames: int = 5,
    warmup_frames: int = 10,
    reset_after_warmup: bool = False,
    default_precision: str = "FP16",
    max_pruning_units: int = 96,
    grouped_conv_mode: str = "shared_local_mean",
    grouped_conv_align: int = 8,
    grouped_allowed_channels_per_group: list[int] | None = None,
    pruning_gene_type: str = "legal_pruning_action",
) -> LidarPyramidSearchContext:
    gpu = select_gpu(gpu_id, exclude_gpu_ids)
    device = torch.device(gpu.runtime_device)
    torch.cuda.set_device(device)
    capability_major, capability_minor = torch.cuda.get_device_capability(device)
    config_path = Path(model_config_path or DEFAULT_CONFIG).expanduser().resolve()
    bundle = load_lidar_pyramid_model(
        checkpoint_path=checkpoint_path,
        model_config_path=config_path,
        heal_root=heal_root,
        device=gpu.runtime_device,
        trace=True,
    )
    if bundle.trace_result is None:
        raise RuntimeError("real_trace_missing")
    trace_hash = str(getattr(bundle.trace_result, "trace_hash", ""))
    atomic_units = filter_post_scatter_pruning_units(
        list(getattr(bundle.trace_result, "atomic_prune_units", []) or [])
    )
    coupled_units = filter_post_scatter_pruning_units(
        list(getattr(bundle.trace_result, "coupled_channel_units", []) or []),
        require_nonempty=False,
    )
    domain_width_mode = str(pruning_gene_type) in {
        "legal_domain_width",
        "domain_width",
        "coupled_domain_width",
    }
    selected_units = (
        _select_all_legal_domain_units(bundle.model, atomic_units)
        if domain_width_mode
        else _select_search_atomic_units(bundle.model, atomic_units, max_dense_units=max_pruning_units)
    )
    action_catalog = None if domain_width_mode else build_pruning_action_catalog(
        selected_units,
        grouped_conv_mode=grouped_conv_mode,
        grouped_conv_align=grouped_conv_align,
        grouped_allowed_channels_per_group=grouped_allowed_channels_per_group or [4, 8, 16, 32, 64, 128, 256, 512],
    )
    preliminary_domains = (
        build_local_pruning_domains(
            selected_units,
            ranking_method="trace_score_placeholder_replaced_after_fisher_calibration",
            grouped_allowed_channels_per_group=grouped_allowed_channels_per_group
            or [4, 8, 16, 32, 64, 128, 256, 512],
        )
        if domain_width_mode
        else []
    )
    if domain_width_mode:
        search_pruning_ids = sorted(str(getattr(unit, "stable_id")) for unit in selected_units)
    elif str(pruning_gene_type) == "coupled_channel_keep_mask":
        search_pruning_ids = sorted(str(getattr(unit, "stable_id")) for unit in selected_units)
    else:
        assert action_catalog is not None
        search_pruning_ids = action_catalog.action_ids
    pruning_unit_metadata = {
        str(getattr(unit, "stable_id")): {
            "scope_id": str(getattr(unit, "scope_id", "")),
            "root_module_path": str(getattr(unit, "root_module_path", "")),
            "root_axis": str(getattr(unit, "root_axis", "")),
            "root_indices": [int(value) for value in getattr(unit, "root_indices", []) or []],
            "constraints": dict(getattr(unit, "constraints", {}) or {}),
            "normalized_score": float(getattr(unit, "normalized_score", 0.0)),
        }
        for unit in selected_units
    }
    precision_layers = _precision_layer_ids(bundle.model)
    precision_groups = _build_precision_groups(bundle.model, bundle.trace_result)
    search_quant_groups = filter_post_scatter_quantization_groups(
        build_quantization_search_groups(
            bundle.model, precision_groups=precision_groups
        )
    )
    tensorrt = discover_trt_environment(
        tensorrt_root, plugin_path=None, conda_env=tensorrt_env
    )
    manifest = write_eval_manifest(
        Path(output_dir) / "baseline" / "eval_manifest.json",
        num_frames=num_frames,
        warmup_frames=warmup_frames,
        available_frame_ids=load_split_frame_ids(bundle.adapter, config_path, split="val"),
        reset_after_warmup=reset_after_warmup,
    )
    builder_flags = {
        "strongly_typed": True,
        "no_tf32": True,
        "shape_profiles": post_scatter_shape_profiles(),
        "engine_contract": POST_SCATTER_CONTRACT,
        "point_frontend": "dynamic_voxelization_pfn_scatter_outside_tensorrt",
        "runtime_max_k_dependency": False,
    }
    calibration_npz_manifest = None
    calibration_npz_manifest_hash = ""
    if quant_calibration_npz_manifest is not None:
        raise RuntimeError("post_scatter_calibration_rejects_fixed_k_npz_manifest")
    calibration_backend = str(quant_activation_calibration_backend).strip().lower()
    if calibration_backend != "modelopt_histogram_entropy":
        raise RuntimeError(
            "post_scatter_calibration_requires_modelopt_histogram_entropy:"
            f"requested={calibration_backend}"
        )
    if quant_activation_calibration_cache_path is not None:
        raise RuntimeError("post_scatter_calibration_rejects_legacy_entropy_cache")
    activation_calibration_cache = None
    search_space = SearchSpaceSpec(
        pruning_unit_ids=search_pruning_ids,
        precision_layer_ids=precision_layers,
        quantization_groups=tuple(search_quant_groups),
        pruning_domains=tuple(preliminary_domains),
        pruning_unit_metadata=pruning_unit_metadata,
        protected_pruning_unit_ids=set(),
        default_precision=default_precision,
        trace_snapshot_hash=trace_hash,
        calibration_manifest_hash=canonical_json_hash(
            {
                "fisher_batches": int(fisher_calibration_batches),
                "quant_batches": int(quant_calibration_batches),
                "quant_calibration_npz_manifest_hash": calibration_npz_manifest_hash,
                "quant_activation_calibration_backend": calibration_backend,
                "quant_activation_calibration_cache_hash": (
                    canonical_json_hash(activation_calibration_cache.read_bytes().hex())
                    if activation_calibration_cache is not None
                    else ""
                ),
                "config": str(config_path),
            }
        ),
        onnx_export_config_hash=canonical_json_hash({
            "engine_contract": POST_SCATTER_CONTRACT,
            "inputs": ["spatial_features", "pairwise_t_matrix"],
            "min_agents": 1,
            "opt_agents": 2,
            "max_agents": 2,
            "runtime_max_k_dependency": False,
        }),
        tensorrt_version="10.9",
        gpu_compute_capability=f"{capability_major}.{capability_minor}",
        builder_flags=builder_flags,
        plugin_hashes=plugin_hashes([]),
        pruning_policy_version=(
            "legal-domain-width-fixed-ranking-v1"
            if domain_width_mode
            else "formal-plan-first-v1"
        ),
    )
    context = LidarPyramidSearchContext(
        checkpoint_path=Path(checkpoint_path).expanduser().resolve(),
        model=bundle.model,
        model_config=config_path,
        model_bundle=bundle,
        trace_result=bundle.trace_result,
        atomic_prune_units=selected_units,
        coupled_channel_units=coupled_units,
        pruning_action_catalog=action_catalog,
        trace_example_inputs=bundle.trace_example_inputs,
        export_example_inputs=bundle.trace_example_inputs,
        fisher_calibration_batches=int(fisher_calibration_batches),
        quant_calibration_batches=int(quant_calibration_batches),
        quant_calibration_npz_manifest=calibration_npz_manifest,
        quant_activation_calibration_backend=calibration_backend,
        quant_activation_calibration_cache_path=activation_calibration_cache,
        quant_calibration_force_rebuild=bool(quant_calibration_force_rebuild),
        eval_manifest_path=manifest.path,
        eval_frame_ids=manifest.frame_ids,
        postprocess_fn=None,
        metric_adapter=None,
        trt_config=None,
        builder_flags=builder_flags,
        plugin_paths=[],
        physical_gpu_id=gpu.physical_gpu_id,
        runtime_device=gpu.runtime_device,
        gpu_selection=gpu,
        tensorrt=tensorrt,
        search_space=search_space,
        eval_manifest_hash=manifest.manifest_hash,
        checkpoint_hash=bundle.checkpoint_hash,
    )
    _write_context_report(Path(output_dir) / "context_report.json", context)
    return context


def _write_context_report(path: Path, context: LidarPyramidSearchContext) -> None:
    payload = {
        "checkpoint_path": str(context.checkpoint_path),
        "checkpoint_hash": context.checkpoint_hash,
        "model_config": str(context.model_config),
        "trace_hash": str(getattr(context.trace_result, "trace_hash", "")),
        "trace_backend": dict(getattr(context.trace_result, "config", {}) or {}).get("realized_backend", ""),
        "atomic_unit_count_total": len(getattr(context.trace_result, "atomic_prune_units", []) or []),
        "coupled_unit_count_total": len(getattr(context.trace_result, "coupled_channel_units", []) or []),
        "search_pruning_unit_count": len(context.search_space.pruning_unit_ids),
        "search_pruning_gene_count": len(context.search_space.pruning_gene_ids),
        "pruning_domain_count": len(context.search_space.pruning_domains),
        "precision_layer_count": len(context.search_space.precision_layer_ids),
        "precision_group_count": len(context.search_space.precision_gene_ids),
        "maximal_legal_int8_gene_count": sum(
            "INT8" in group.allowed_precisions and not group.protected
            for group in context.search_space.quantization_groups
        ),
        "protected_fp16_group_count": sum(group.protected for group in context.search_space.quantization_groups),
        "pruning_scope_precision_coupling_count": sum(
            len(group.module_paths) > 1 and bool(group.metadata.get("force_same_precision", True))
            for group in context.search_space.quantization_groups
        ),
        "multi_member_force_same_precision_group_count": sum(
            len(group.module_paths) > 1 and bool(group.metadata.get("force_same_precision", True))
            for group in context.search_space.quantization_groups
        ),
        "merge_boundary_resolution_status": "deferred_to_canonical_onnx_qdq_export",
        "quant_calibration_npz_manifest": (
            str(context.quant_calibration_npz_manifest)
            if context.quant_calibration_npz_manifest is not None
            else ""
        ),
        "quant_activation_calibration_backend": context.quant_activation_calibration_backend,
        "quant_activation_calibration_cache_path": (
            str(context.quant_activation_calibration_cache_path)
            if context.quant_activation_calibration_cache_path is not None
            else ""
        ),
        "quant_calibration_force_rebuild": context.quant_calibration_force_rebuild,
        "unresolved_mapping_count": "deferred_to_canonical_onnx_qdq_export",
        "quantization_groups": [
            {
                "group_id": group.group_id,
                "module_paths": list(group.module_paths),
                "allowed_precisions": list(group.allowed_precisions),
                "protected": group.protected,
                "protection_reason": group.protection_reason,
                "force_same_precision": bool(group.metadata.get("force_same_precision", True)),
                "merge_boundaries": list(group.metadata.get("merge_boundaries", [])),
                "output_precision_policy": group.metadata.get("output_precision_policy", "same_as_compute"),
            }
            for group in context.search_space.quantization_groups
        ],
        "grouped_conv_legal_action_count": context.pruning_action_catalog.grouped_action_count if context.pruning_action_catalog else 0,
        "shared_local_mean_bundle_count": context.pruning_action_catalog.shared_local_mean_bundle_count if context.pruning_action_catalog else 0,
        "independent_group_topk_bundle_count": context.pruning_action_catalog.independent_group_topk_bundle_count if context.pruning_action_catalog else 0,
        "sample_pruning_unit_ids": context.search_space.pruning_unit_ids[:8],
        "sample_precision_layers": context.search_space.precision_layer_ids[:8],
        "gpu": context.gpu_selection.to_dict(),
        "tensorrt": context.tensorrt.to_dict(),
        "eval_manifest": {
            "path": str(context.eval_manifest_path),
            "frame_count_with_warmup": len(context.eval_frame_ids),
            "hash": context.eval_manifest_hash,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
