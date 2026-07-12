"""Real lidar_pyramid search context assembly."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn

from ..canonicalization import SearchSpaceSpec
from ..hashing import canonical_json_hash
from ..pruning_space.action_catalog import PruningActionCatalog, build_pruning_action_catalog
from ..quantization_space.group_builder import build_quantization_search_groups
from .data_provider import EvaluationManifest, write_eval_manifest
from .model_provider import LidarPyramidModelBundle, load_lidar_pyramid_model
from .runtime_environment import GPUSelection, TensorRTEnvironment, discover_trt_environment, plugin_hashes, select_gpu


DEFAULT_CHECKPOINT = Path("/home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_pyramid/net_epoch_bestval_at17.pth")
DEFAULT_CONFIG = Path("/home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_pyramid/config.yaml")
DEFAULT_HEAL_ROOT = Path("/home/lixingfeng/UniAD_examine/HEAL")
DEFAULT_TRT_ROOT = Path("/home/lixingfeng/UniAD_examine/HEAL/prune_model/TensorRT-10.9_x86_cu118")
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


def _precision_layer_ids(model: nn.Module) -> list[str]:
    rows = [name for name, module in model.named_modules() if name and _module_is_weighted(module)]
    if any(name.startswith("dryrun_model") for name in rows):
        raise RuntimeError("real_precision_space_contains_dryrun_layer")
    return sorted(rows)


def _build_precision_groups(model: nn.Module, trace_result: Any) -> list[Any]:
    try:
        from tracer.dependency_tracer import build_dependency_graph
        from tracer.precision_coupling_tracer import PrecisionGroup, build_precision_coupling_groups
    except ImportError:
        from heal_compress.tracer.dependency_tracer import build_dependency_graph
        from heal_compress.tracer.precision_coupling_tracer import PrecisionGroup, build_precision_coupling_groups
    graph = build_dependency_graph(model, None)
    base_groups = build_precision_coupling_groups(model, graph, sample_batch=None, allow_head_int8=False)
    weighted = set(_precision_layer_ids(model))
    allowed_by_module: dict[str, list[str]] = {}
    default_by_module: dict[str, str] = {}
    reason_by_module: dict[str, str] = {}
    for group in base_groups:
        for module in group.member_modules:
            if module in weighted:
                allowed_by_module[module] = list(group.allowed_precisions)
                default_by_module[module] = str(group.default_precision)
                reason_by_module[module] = str(group.reason)
    filtered = []
    covered: set[str] = set()
    for scope in getattr(trace_result, "dependency_scopes", []) or []:
        members = sorted(
            {
                str(getattr(member, "module_path", ""))
                for member in getattr(scope, "members", []) or []
                if str(getattr(member, "module_path", "")) in weighted
            }
        )
        members = [module for module in members if module not in covered]
        if len(members) < 2:
            continue
        allowed_sets = [set(allowed_by_module.get(module, ["fp32", "fp16", "int8"])) for module in members]
        allowed = sorted(set.intersection(*allowed_sets), key=["fp32", "fp16", "int8"].index)
        group_id = f"pg_scope_{canonical_json_hash({'scope': getattr(scope, 'stable_id', ''), 'members': members})[:12]}"
        filtered.append(
            PrecisionGroup(
                precision_group_id=group_id,
                member_modules=members,
                reason="trace_dependency_scope",
                allowed_precisions=allowed or ["fp16"],
                default_precision="fp16",
                force_same_precision=True,
            )
        )
        covered.update(members)
    for group in base_groups:
        members = [module for module in group.member_modules if module in weighted]
        members = [module for module in members if module not in covered]
        if not members:
            continue
        if members != group.member_modules:
            group = type(group)(
                precision_group_id=group.precision_group_id,
                member_modules=members,
                reason=group.reason,
                allowed_precisions=group.allowed_precisions,
                default_precision=group.default_precision,
                force_same_precision=group.force_same_precision,
            )
        filtered.append(group)
    return filtered


def _shape_profiles(fixed_k: int = 29696) -> dict[str, dict[str, tuple[int, ...]]]:
    return {
        "pairwise_t_matrix": {"min": (1, 1, 1, 4, 4), "opt": (1, 2, 2, 4, 4), "max": (1, 2, 2, 4, 4)},
        "valid_voxel_mask": {"min": (fixed_k,), "opt": (fixed_k,), "max": (fixed_k,)},
        "voxel_coords": {"min": (fixed_k, 4), "opt": (fixed_k, 4), "max": (fixed_k, 4)},
        "voxel_features": {"min": (fixed_k, 32, 4), "opt": (fixed_k, 32, 4), "max": (fixed_k, 32, 4)},
        "voxel_num_points": {"min": (fixed_k,), "opt": (fixed_k,), "max": (fixed_k,)},
    }


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
    num_frames: int = 5,
    warmup_frames: int = 10,
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
    atomic_units = list(getattr(bundle.trace_result, "atomic_prune_units", []) or [])
    coupled_units = list(getattr(bundle.trace_result, "coupled_channel_units", []) or [])
    selected_units = _select_search_atomic_units(bundle.model, atomic_units, max_dense_units=max_pruning_units)
    action_catalog = build_pruning_action_catalog(
        selected_units,
        grouped_conv_mode=grouped_conv_mode,
        grouped_conv_align=grouped_conv_align,
        grouped_allowed_channels_per_group=grouped_allowed_channels_per_group or [4, 8, 16, 32, 64, 128, 256, 512],
    )
    if str(pruning_gene_type) == "coupled_channel_keep_mask":
        search_pruning_ids = sorted(str(getattr(unit, "stable_id")) for unit in selected_units)
    else:
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
    search_quant_groups = build_quantization_search_groups(bundle.model, precision_groups=precision_groups)
    plugin = Path(plugin_path).expanduser().resolve() if plugin_path else (Path.cwd() / DEFAULT_PLUGIN).resolve()
    tensorrt = discover_trt_environment(tensorrt_root, plugin_path=plugin, conda_env=tensorrt_env)
    manifest = write_eval_manifest(Path(output_dir) / "baseline" / "eval_manifest.json", num_frames=num_frames, warmup_frames=warmup_frames)
    builder_flags = {
        "precision_constraints": "obey",
        "fp16": True,
        "int8": True,
        "no_tf32": True,
        "shape_profiles": _shape_profiles(),
    }
    search_space = SearchSpaceSpec(
        pruning_unit_ids=search_pruning_ids,
        precision_layer_ids=precision_layers,
        quantization_groups=tuple(search_quant_groups),
        pruning_unit_metadata=pruning_unit_metadata,
        protected_pruning_unit_ids=set(),
        default_precision=default_precision,
        trace_snapshot_hash=trace_hash,
        calibration_manifest_hash=canonical_json_hash({"fisher_batches": int(fisher_calibration_batches), "quant_batches": int(quant_calibration_batches), "config": str(config_path)}),
        onnx_export_config_hash=canonical_json_hash({"fixed_k": 29696, "min_agents": 1, "opt_agents": 2, "max_agents": 2}),
        tensorrt_version="10.9",
        gpu_compute_capability="8.9",
        builder_flags=builder_flags,
        plugin_hashes=plugin_hashes([plugin]),
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
        eval_manifest_path=manifest.path,
        eval_frame_ids=manifest.frame_ids,
        postprocess_fn=None,
        metric_adapter=None,
        trt_config=None,
        builder_flags=builder_flags,
        plugin_paths=[plugin],
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
        "precision_layer_count": len(context.search_space.precision_layer_ids),
        "precision_group_count": len(context.search_space.precision_gene_ids),
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
