"""Search-context assembly for HEAL LiDAR F-Cooper and DiscoNet."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from tracer.api import trace_model
from tracer.config import TraceConfig
from tracer.precision_coupling_tracer import build_runtime_precision_coupling

from ..canonicalization import SearchSpaceSpec
from ..hashing import canonical_json_hash
from ..model_family.export.heal_lidar_baselines import (
    HealLidarBaselineExportPolicy,
    prepare_heal_lidar_baseline_inputs,
)
from ..model_family.heal_lidar_deployment import (
    HEAL_LIDAR_BASELINE_INPUT_NAMES,
    build_heal_lidar_baseline_quantization_groups,
)
from ..model_family.heal_lidar_pruning import build_heal_lidar_baseline_atomic_units
from ..model_family.model_provider import HealModelFamilyBundle, load_heal_model_family
from ..pruning_space.local_domains import build_local_pruning_domains
from ..quantization_space.group_builder import build_quantization_search_groups
from .calibration_provider import (
    BASELINE_FIXED_K_CALIBRATION_INPUT_NAMES,
    fixed_k_calibration_npz_manifest_identity,
)
from .data_provider import load_split_frame_ids, write_eval_manifest
from .runtime_environment import (
    GPUSelection,
    TensorRTEnvironment,
    discover_trt_environment,
    plugin_hashes,
    select_gpu,
)
HEAL_RUNTIME_GRAPH_POLICY = "heal_runtime_graph_v1"
_DEPRECATED_RUNTIME_POLICY_ALIASES = {
    "pyramid_runtime_tracer_v1": HEAL_RUNTIME_GRAPH_POLICY,
}
LEGACY_FAMILY_STATIC_POLICY = "legacy_family_static_dependency_closure_v1"


@dataclass
class HealLidarBaselineSearchContext:
    family_id: str
    model_name: str
    checkpoint_path: Path
    model: torch.nn.Module
    model_config: Path
    model_bundle: HealModelFamilyBundle
    trace_result: Any
    precision_coupling_result: Any
    atomic_prune_units: list[Any]
    coupled_channel_units: list[Any]
    pruning_action_catalog: None
    trace_example_inputs: Any
    export_example_inputs: dict[str, torch.Tensor]
    fisher_calibration_batches: int
    quant_calibration_batches: int
    quant_calibration_npz_manifest: Path | None
    quant_activation_calibration_backend: str
    quant_activation_calibration_cache_path: Path | None
    quant_calibration_force_rebuild: bool
    eval_manifest_path: Path
    eval_frame_ids: list[str]
    postprocess_fn: None
    metric_adapter: None
    trt_config: None
    builder_flags: dict[str, Any]
    plugin_paths: list[Path]
    physical_gpu_id: int
    runtime_device: str
    gpu_selection: GPUSelection
    tensorrt: TensorRTEnvironment
    search_space: SearchSpaceSpec
    eval_manifest_hash: str
    checkpoint_hash: str
    fixed_k: int
    max_agents: int
    search_space_policy: str


def _model_name(family_id: str) -> str:
    names = {
        "heal_lidar_fcooper": "lidar_fcooper",
        "heal_lidar_disco": "lidar_disco",
        "heal_lidar_attfusion": "lidar_attfuse",
        "heal_lidar_cobevt": "lidar_cobevt",
    }
    if family_id not in names:
        raise RuntimeError(f"unsupported_heal_lidar_baseline_context_family:{family_id}")
    return names[family_id]


def _select_runtime_traced_atomic_units(trace_result: Any) -> list[Any]:
    """Select every tracer-legal weighted output unit without path whitelists."""

    weighted = {
        str(row.module_path)
        for row in getattr(trace_result, "module_inventory", []) or []
        if bool(getattr(row, "weighted", False))
    }
    selected: dict[str, Any] = {}
    for unit in getattr(trace_result, "atomic_prune_units", []) or []:
        if bool(getattr(unit, "protected", False)):
            continue
        if str(getattr(unit, "root_module_path", "")) not in weighted:
            continue
        if str(getattr(unit, "root_axis", "")) not in {"out", "channel"}:
            continue
        if not list(getattr(unit, "root_indices", []) or []):
            continue
        unit_id = str(getattr(unit, "stable_id", ""))
        if not unit_id or unit_id.startswith("unit"):
            raise RuntimeError(f"invalid_runtime_atomic_prune_unit_id:{unit_id}")
        selected[unit_id] = unit
    if not selected:
        raise RuntimeError("runtime_full_graph_pruning_space_empty")
    return [selected[key] for key in sorted(selected)]


def build_heal_lidar_baseline_context(
    *,
    family_id: str,
    checkpoint_path: str | Path,
    model_config_path: str | Path,
    output_dir: str | Path,
    heal_root: str | Path,
    tensorrt_root: str | Path,
    plugin_path: str | Path,
    gpu_id: str = "auto",
    exclude_gpu_ids: list[int] | None = None,
    tensorrt_env: str = "modelopt",
    fisher_calibration_batches: int = 8,
    quant_calibration_batches: int = 200,
    quant_calibration_npz_manifest: str | Path | None = None,
    quant_activation_calibration_backend: str = "tensorrt_entropy_calibration2",
    quant_activation_calibration_cache_path: str | Path | None = None,
    quant_calibration_force_rebuild: bool = False,
    num_frames: int = 500,
    warmup_frames: int = 200,
    reset_after_warmup: bool = True,
    default_precision: str = "FP16",
    fixed_k: int = 29696,
    max_agents: int = 2,
    minimum_retained_ratio: float = 0.10,
    dense_alignment: int = 4,
    require_quant_calibration_manifest: bool = True,
    search_space_policy: str = LEGACY_FAMILY_STATIC_POLICY,
) -> HealLidarBaselineSearchContext:
    model_name = _model_name(family_id)
    if int(fixed_k) <= 0 or int(max_agents) != 2:
        raise RuntimeError(
            f"heal_lidar_baseline_context_contract_mismatch:{fixed_k}:{max_agents}"
        )
    gpu = select_gpu(gpu_id, exclude_gpu_ids)
    device = torch.device(gpu.runtime_device)
    if device.type != "cuda":
        raise RuntimeError("heal_lidar_baseline_search_requires_cuda")
    torch.cuda.set_device(device)
    capability_major, capability_minor = torch.cuda.get_device_capability(device)
    bundle = load_heal_model_family(
        config_path=model_config_path,
        checkpoint_path=checkpoint_path,
        heal_root=heal_root,
        device=gpu.runtime_device,
        family_id=family_id,
        forward_smoke=True,
    )
    if bundle.uncalled_weighted_modules:
        raise RuntimeError(
            f"heal_lidar_baseline_runtime_weighted_coverage_incomplete:"
            f"{bundle.uncalled_weighted_modules}"
        )
    requested_policy_name = str(search_space_policy).strip().lower()
    policy_name = _DEPRECATED_RUNTIME_POLICY_ALIASES.get(
        requested_policy_name,
        requested_policy_name,
    )
    if policy_name not in {
        LEGACY_FAMILY_STATIC_POLICY,
        HEAL_RUNTIME_GRAPH_POLICY,
    }:
        raise RuntimeError(f"unsupported_heal_lidar_search_space_policy:{policy_name}")
    precision_coupling_result = None
    if policy_name == HEAL_RUNTIME_GRAPH_POLICY:
        trace_result = trace_model(
            bundle.model,
            bundle.example_batch,
            config=TraceConfig(fail_on_fx_trace_error=False),
            forward_fn=bundle.adapter.forward_for_task,
        )
        coverage = trace_result.trace_coverage
        if (
            float(coverage.weighted_module_coverage) != 1.0
            or int(coverage.unresolved_operation_count) != 0
            or int(coverage.unsupported_operation_count) != 0
        ):
            raise RuntimeError(
                "heal_lidar_runtime_trace_coverage_rejected:"
                f"{coverage.to_dict()}"
            )
        all_atomic_units = list(trace_result.atomic_prune_units)
        atomic_units = _select_runtime_traced_atomic_units(trace_result)
        coupled_units = list(trace_result.coupled_channel_units)
    else:
        atomic_units = build_heal_lidar_baseline_atomic_units(bundle.model, bundle.audit)
        all_atomic_units = list(atomic_units)
        coupled_units = []
        trace_result = None
    preliminary_domains = build_local_pruning_domains(
        atomic_units,
        ranking_method="trace_score_placeholder_replaced_after_fisher_calibration",
        minimum_retained_ratio=float(minimum_retained_ratio),
        dense_alignment=int(dense_alignment),
    )
    if policy_name == LEGACY_FAMILY_STATIC_POLICY:
        expected_domains = sum(row.production_enabled for row in bundle.audit.pruning_domains)
        if len(preliminary_domains) != expected_domains:
            raise RuntimeError(
                f"heal_lidar_baseline_pruning_domain_count_mismatch:"
                f"{len(preliminary_domains)}!={expected_domains}"
            )
        quantization_groups = build_heal_lidar_baseline_quantization_groups(
            bundle.model,
            bundle.audit,
        )
        precision_layer_ids = [group.module_paths[0] for group in quantization_groups]
    else:
        precision_coupling_result = build_runtime_precision_coupling(
            bundle.model,
            trace_result,
        )
        precision_layer_ids = list(precision_coupling_result.weighted_modules)
        quantization_groups = build_quantization_search_groups(
            bundle.model,
            precision_groups=precision_coupling_result.groups,
        )
    production_int8 = [
        group for group in quantization_groups
        if "INT8" in group.allowed_precisions and not group.protected
    ]
    expected_int8 = 24 if policy_name == LEGACY_FAMILY_STATIC_POLICY else len(quantization_groups)
    if len(production_int8) != expected_int8:
        raise RuntimeError(
            "heal_lidar_baseline_int8_group_count_mismatch:"
            f"{len(production_int8)}!={expected_int8}"
        )
    if policy_name == HEAL_RUNTIME_GRAPH_POLICY and any(
        group.protected for group in quantization_groups
    ):
        raise RuntimeError("runtime_graph_quantization_contains_protected_group")

    policy = HealLidarBaselineExportPolicy(
        fixed_k=int(fixed_k),
        max_agents=int(max_agents),
    )
    example_ego = bundle.example_batch
    if not isinstance(example_ego, dict):
        raise RuntimeError("heal_lidar_baseline_synthetic_example_missing")
    export_inputs = prepare_heal_lidar_baseline_inputs(example_ego, policy=policy)
    if tuple(export_inputs) != HEAL_LIDAR_BASELINE_INPUT_NAMES:
        raise RuntimeError("heal_lidar_baseline_export_input_contract_mismatch")

    plugin = Path(plugin_path).expanduser().resolve()
    tensorrt = discover_trt_environment(
        tensorrt_root,
        plugin_path=plugin,
        conda_env=tensorrt_env,
    )
    config_path = Path(model_config_path).expanduser().resolve()
    manifest = write_eval_manifest(
        Path(output_dir) / "baseline" / "eval_manifest.json",
        num_frames=int(num_frames),
        warmup_frames=int(warmup_frames),
        available_frame_ids=load_split_frame_ids(bundle.adapter, config_path, split="val"),
        reset_after_warmup=bool(reset_after_warmup),
    )
    calibration_manifest = (
        Path(quant_calibration_npz_manifest).expanduser().resolve()
        if quant_calibration_npz_manifest
        else None
    )
    calibration_identity: dict[str, Any] = {}
    backend = str(quant_activation_calibration_backend).strip().lower()
    if backend not in {
        "tensorrt_entropy_calibration2",
        "external_tensorrt_entropy_cache_exact_match",
        "modelopt_histogram_entropy",
    }:
        raise RuntimeError(f"unsupported_baseline_quant_calibration_backend:{backend}")
    if calibration_manifest is not None:
        if not calibration_manifest.is_file() and not require_quant_calibration_manifest:
            calibration_manifest = None
        else:
            calibration_identity = fixed_k_calibration_npz_manifest_identity(
                calibration_manifest,
                num_batches=int(quant_calibration_batches),
                fixed_k=int(fixed_k),
                input_names=BASELINE_FIXED_K_CALIBRATION_INPUT_NAMES,
            )
    if (
        backend == "tensorrt_entropy_calibration2"
        and calibration_manifest is None
        and require_quant_calibration_manifest
    ):
        raise RuntimeError("baseline_tensorrt_entropy_calibration_requires_six_input_npz_manifest")
    calibration_cache = (
        Path(quant_activation_calibration_cache_path).expanduser().resolve()
        if quant_activation_calibration_cache_path
        else None
    )
    if backend == "external_tensorrt_entropy_cache_exact_match" and (
        calibration_cache is None or not calibration_cache.is_file()
    ):
        raise RuntimeError(f"baseline_external_entropy_cache_missing:{calibration_cache}")

    if trace_result is None:
        trace_hash = canonical_json_hash({
            "family_id": family_id,
            "audit_hash": bundle.audit.to_dict()["audit_hash"],
            "atomic_units": [unit.to_dict() for unit in atomic_units],
        })
        from types import SimpleNamespace

        trace_result = SimpleNamespace(
            trace_hash=trace_hash,
            atomic_prune_units=atomic_units,
            coupled_channel_units=[],
            config={"realized_backend": "family_static_dependency_closure_v1"},
        )
    else:
        trace_hash = str(trace_result.trace_hash)
    pruning_metadata = {
        str(unit.stable_id): {
            "scope_id": str(unit.scope_id),
            "root_module_path": str(unit.root_module_path),
            "root_axis": str(unit.root_axis),
            "root_indices": list(unit.root_indices),
            "constraints": dict(unit.constraints),
            "normalized_score": 0.0,
        }
        for unit in atomic_units
    }
    builder_flags = {
        "strongly_typed": True,
        "no_tf32": True,
        "shape_profiles": {},
        "fixed_k": int(fixed_k),
        "max_agents": int(max_agents),
        "input_contract": "heal_lidar_baseline_fixed_k",
    }
    search_space = SearchSpaceSpec(
        pruning_unit_ids=[str(unit.stable_id) for unit in atomic_units],
        precision_layer_ids=precision_layer_ids,
        quantization_groups=tuple(quantization_groups),
        pruning_domains=tuple(preliminary_domains),
        pruning_unit_metadata=pruning_metadata,
        protected_pruning_unit_ids=set(),
        default_precision=default_precision,
        pruning_policy_version=(
            "generic-runtime-tracer-protection-contract-full-graph-width-v1"
            if policy_name == HEAL_RUNTIME_GRAPH_POLICY
            else "heal-lidar-legal-domain-width-fixed-ranking-v1"
        ),
        precision_policy_version=(
            "runtime-tensor-flow-adaptive-merge-precision-groups-v1"
            if policy_name == HEAL_RUNTIME_GRAPH_POLICY
            else "heal-lidar-explicit-qdq-fp16-fusion-island-v1"
        ),
        trace_snapshot_hash=trace_hash,
        calibration_manifest_hash=canonical_json_hash({
            "fisher_batches": int(fisher_calibration_batches),
            "quant_batches": int(quant_calibration_batches),
            "calibration_identity": calibration_identity,
            "backend": backend,
        }),
        onnx_export_config_hash=canonical_json_hash({
            "fixed_k": int(fixed_k),
            "max_agents": int(max_agents),
            "input_names": list(HEAL_LIDAR_BASELINE_INPUT_NAMES),
        }),
        tensorrt_version="10.9",
        gpu_compute_capability=f"{capability_major}.{capability_minor}",
        builder_flags=builder_flags,
        plugin_hashes=plugin_hashes([plugin]),
    )
    context = HealLidarBaselineSearchContext(
        family_id=family_id,
        model_name=model_name,
        checkpoint_path=Path(checkpoint_path).expanduser().resolve(),
        model=bundle.model,
        model_config=config_path,
        model_bundle=bundle,
        trace_result=trace_result,
        precision_coupling_result=precision_coupling_result,
        atomic_prune_units=atomic_units,
        coupled_channel_units=coupled_units,
        pruning_action_catalog=None,
        trace_example_inputs=bundle.example_batch,
        export_example_inputs=export_inputs,
        fisher_calibration_batches=int(fisher_calibration_batches),
        quant_calibration_batches=int(quant_calibration_batches),
        quant_calibration_npz_manifest=calibration_manifest,
        quant_activation_calibration_backend=backend,
        quant_activation_calibration_cache_path=calibration_cache,
        quant_calibration_force_rebuild=bool(quant_calibration_force_rebuild),
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
        fixed_k=int(fixed_k),
        max_agents=int(max_agents),
        search_space_policy=policy_name,
    )
    report = {
        "family_id": family_id,
        "model_name": model_name,
        "checkpoint_path": str(context.checkpoint_path),
        "checkpoint_hash": context.checkpoint_hash,
        "model_config": str(config_path),
        "audit": bundle.audit.to_dict(),
        "family_audit_used_for_search_space": policy_name == LEGACY_FAMILY_STATIC_POLICY,
        "search_space_policy": policy_name,
        "requested_search_space_policy": requested_policy_name,
        "trace_backend": dict(getattr(trace_result, "config", {}) or {}).get(
            "realized_backend", ""
        ),
        "runtime_weighted_module_count": bundle.weighted_modules_total,
        "runtime_weighted_modules_called": list(bundle.weighted_modules_called),
        "uncalled_weighted_modules": list(bundle.uncalled_weighted_modules),
        "atomic_unit_count_total": len(all_atomic_units),
        "atomic_unit_count": len(atomic_units),
        "coupled_channel_unit_count": len(coupled_units),
        "pruning_domain_count": len(preliminary_domains),
        "quantization_group_count": len(quantization_groups),
        "production_int8_group_count": len(production_int8),
        "protected_quantization_group_count": sum(group.protected for group in quantization_groups),
        "runtime_precision_relation_count": (
            len(precision_coupling_result.relations)
            if precision_coupling_result is not None
            else 0
        ),
        "runtime_precision_relations": (
            [relation.to_dict() for relation in precision_coupling_result.relations]
            if precision_coupling_result is not None
            else []
        ),
        "runtime_precision_groups": (
            [group.to_dict() for group in precision_coupling_result.groups]
            if precision_coupling_result is not None
            else []
        ),
        "pruning_protection_reasons": dict(
            getattr(trace_result, "protection_reasons", {}) or {}
        ),
        "fixed_k": int(fixed_k),
        "max_agents": int(max_agents),
        "input_names": list(export_inputs),
        "calibration_backend": backend,
        "calibration_identity": calibration_identity,
        "trace_hash": trace_hash,
    }
    destination = Path(output_dir) / "context_report.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return context


__all__ = [
    "HealLidarBaselineSearchContext",
    "build_heal_lidar_baseline_context",
]
