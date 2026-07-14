"""Real lidar_pyramid Stage-2 evaluator."""

from __future__ import annotations

import csv
import json
import shutil
from dataclasses import asdict, is_dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from ..adapters.pruning_adapter import FormalPruningAdapter
from ..cache.artifact_cache import ArtifactCache
from ..cache.real_eval_cache import RealEvalCache
from ..candidate import CandidatePhenotype
from ..hashing import build_deployment_signature, canonical_json_hash, deployment_hash, eval_hash, physical_hash
from ..integration.calibration_provider import (
    QDQ_CALIBRATION_SEMANTICS_VERSION,
    TENSORRT_ENTROPY_CALIBRATION_SEMANTICS_VERSION,
    build_tensorrt_entropy_calibration_cache_modelopt,
    collect_or_load_qdq_calibration_scales,
    fixed_k_calibration_npz_manifest_identity,
    qdq_scales_from_tensorrt_entropy_cache,
    save_calibration_scales,
)
from ..integration.data_provider import load_split_frame_ids
from ..integration.evaluation_provider import evaluate_engine_modelopt
from ..integration.lidar_pyramid_context import LidarPyramidSearchContext
from ..integration.trt_compatible_export import build_search_trt_compatible_export_module, make_pointpillar_domain_compatible
from ..pruning_space.action_codec import selected_actions_from_genes
from ..pruning_space.grouped_bundle_adapter import request_from_pruning_actions
from ..proxy.runtime_shape_profiler import profile_runtime_layer_shapes
from ..baselines.original_engines import (
    TRUSTED_EXPLICIT_QDQ_INT8_V1_MODULES,
    make_baseline_trt_build_config,
    validate_baseline_layer_precisions,
)
from .candidate_artifacts import write_candidate_summary_artifacts
from .objective import Stage2ObjectiveConfig, compute_stage2_score
from .physical_validation import validate_repaired_physical_plan
from .realized_bops import compute_realized_bops, engine_realized_precision_profile
from .mixed_precision_export import summarize_qdq_realization
from .trt_modelopt import build_engine_modelopt


def _plain(value: Any) -> Any:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _plain(value.to_dict())
    if is_dataclass(value):
        return _plain(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_plain(item) for item in value]
    return value


def _write_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(_plain(payload), indent=2, sort_keys=True), encoding="utf-8")


def _file_hash(path: str | Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _qdq_deployment_lineage(
    *,
    base_onnx_path: str | Path,
    canonical_mapping: Any,
    legalized_precision_profile: dict[str, str],
    realized_precision_profile: dict[str, str],
    calibration_recipe: dict[str, Any],
    qdq_result: Any,
) -> dict[str, str]:
    metadata = dict(getattr(qdq_result, "calibration_metadata", {}) or {})
    return {
        "base_onnx_hash": _file_hash(base_onnx_path),
        "qdq_topology_hash": str(metadata.get("qdq_topology_hash", "")),
        "canonical_mapping_hash": canonical_json_hash(_plain(canonical_mapping)),
        "legalized_precision_profile_hash": canonical_json_hash(legalized_precision_profile),
        "realized_precision_profile_hash": canonical_json_hash(realized_precision_profile),
        "calibration_manifest_hash": str(metadata.get("calibration_manifest_hash", "")),
        "calibration_recipe_hash": canonical_json_hash(calibration_recipe),
    }


def _candidate_deployment_signature(
    *,
    context: Any,
    physical: dict[str, Any],
    qdq: dict[str, Any],
    trt: dict[str, Any],
) -> dict[str, str]:
    runtime = dict(trt.get("runtime_provenance") or {})
    plugin_hashes = dict(context.search_space.plugin_hashes)
    plugin_binary_hash = (
        next(iter(plugin_hashes.values()))
        if len(plugin_hashes) == 1
        else canonical_json_hash(plugin_hashes)
    )
    return build_deployment_signature(
        code_commit=str(context.code_commit),
        physical_model_hash=str(physical.get("physical_hash", "")),
        base_onnx_hash=str(qdq.get("base_onnx_hash", "")),
        qdq_topology_hash=str(qdq.get("qdq_topology_hash", "")),
        canonical_mapping_hash=str(qdq.get("canonical_mapping_hash", "")),
        legalized_precision_profile_hash=str(qdq.get("legalized_precision_profile_hash", "")),
        realized_precision_profile_hash=str(qdq.get("realized_precision_profile_hash", "")),
        calibration_manifest_hash=str(qdq.get("calibration_manifest_hash", "")),
        calibration_recipe_hash=str(qdq.get("calibration_recipe_hash", "")),
        tensorrt_version=str(runtime.get("tensorrt_version", "")),
        cuda_version=str(runtime.get("cuda_version", "")),
        gpu_architecture=str(runtime.get("gpu_architecture", "")),
        plugin_binary_hash=plugin_binary_hash,
    )


def _engine_merge_precision_realization(layer_info_path: str | Path, qdq_result: Any) -> dict[str, Any]:
    path = Path(layer_info_path)
    graph_rows = list(getattr(qdq_result, "calibration_metadata", {}).get("merge_quantization_audit", []))
    if not path.is_file():
        return {"status": "layer_info_missing", "merges": graph_rows}
    payload = json.loads(path.read_text(encoding="utf-8"))
    layers = list(payload.get("Layers", []))

    def formats(rows: list[dict[str, Any]], field: str) -> list[str]:
        return sorted(
            {
                str(tensor.get("Format/Datatype", ""))
                for layer in rows
                for tensor in layer.get(field, [])
                if str(tensor.get("Format/Datatype", ""))
            }
        )

    realized = []
    for merge in graph_rows:
        name = str(merge.get("merge_op_name", ""))
        matched = [layer for layer in layers if name and name in str(layer.get("Name", ""))]
        optimization = "direct_or_fused_layer_name_match"
        if not matched and str(merge.get("merge_op_type")) == "Concat":
            downstream_q = [
                str(row.get("consumer", ""))
                for row in merge.get("downstream", [])
                if str(row.get("op_type", "")) == "QuantizeLinear"
            ]
            matched = [
                layer
                for layer in layers
                if any(q_name and q_name in str(layer.get("Name", "")) for q_name in downstream_q)
            ]
            optimization = "concat_fused_with_common_downstream_quantize" if matched else "not_independently_inspectable"
        input_formats = formats(matched, "Inputs")
        output_formats = formats(matched, "Outputs")
        all_formats = input_formats + output_formats
        graph_fp16_casts = bool(merge.get("input_branches")) and all(
            bool(branch.get("cast_to_fp16", False))
            for branch in merge.get("input_branches", [])
        )
        fused_weighted_compute = any(
            any(token in str(layer.get("LayerType", "")).lower() for token in ("conv", "gemm", "matmul"))
            for layer in matched
        )
        if (
            str(merge.get("merge_op_type")) == "Add"
            and graph_fp16_casts
            and fused_weighted_compute
            and output_formats
            and set(output_formats) <= {"Half"}
        ):
            precision = "FP16"
            optimization = "int8_weighted_compute_fused_with_graph_constrained_fp16_add"
        elif (
            optimization == "concat_fused_with_common_downstream_quantize"
            and graph_fp16_casts
            and "Int8" in output_formats
        ):
            precision = "FP16"
            optimization = "graph_constrained_fp16_concat_fused_with_downstream_int8_requantization"
        elif optimization == "concat_fused_with_common_downstream_quantize" and "Int8" in output_formats:
            precision = "INT8_common_scale_fused_concat"
        elif all_formats and set(all_formats) <= {"Half"}:
            precision = "FP16"
        elif "Int8" in all_formats:
            precision = "INT8_or_mixed"
        elif "Float" in all_formats:
            precision = "FP32_or_mixed"
        else:
            precision = "not_yet_verified"
        realized.append(
            {
                **dict(merge),
                "engine_layer_names": [str(layer.get("Name", "")) for layer in matched],
                "engine_input_formats": input_formats,
                "engine_output_formats": output_formats,
                "engine_optimization": optimization,
                "realized_merge_precision": precision,
            }
        )
    issues = []
    for row in realized:
        precision = str(row.get("realized_merge_precision", ""))
        op_type = str(row.get("merge_op_type", ""))
        if op_type == "Add" and precision != "FP16":
            issues.append(f"residual_add_not_fp16:{row.get('merge_op_name')}:{precision}")
        elif op_type == "Concat" and precision not in {"FP16", "INT8_common_scale_fused_concat"}:
            issues.append(f"concat_merge_not_compatible:{row.get('merge_op_name')}:{precision}")
    return {
        "status": "ok",
        "passed": not issues,
        "issues": issues,
        "policy": "A_fp16_merge_with_equivalent_common_scale_concat_fusion_allowed",
        "merges": realized,
    }


def _load_origin_map_result(path: str | Path) -> Any:
    try:
        from quantization.types import CanonicalFunctionalComputeGroup, CanonicalMappingEntry, OnnxOriginMapResult
    except ImportError:
        from heal_compress.quantization.types import CanonicalFunctionalComputeGroup, CanonicalMappingEntry, OnnxOriginMapResult

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    payload["entries"] = [CanonicalMappingEntry(**dict(row)) for row in payload.get("entries", [])]
    payload["functional_compute_groups"] = [
        CanonicalFunctionalComputeGroup(**dict(row))
        for row in payload.get("functional_compute_groups", [])
    ]
    return OnnxOriginMapResult(**payload)


def _shape_profiles(fixed_k: int = 29696) -> dict[str, dict[str, tuple[int, ...]]]:
    return {
        "pairwise_t_matrix": {"min": (1, 1, 1, 4, 4), "opt": (1, 2, 2, 4, 4), "max": (1, 2, 2, 4, 4)},
        "valid_voxel_mask": {"min": (fixed_k,), "opt": (fixed_k,), "max": (fixed_k,)},
        "voxel_coords": {"min": (fixed_k, 4), "opt": (fixed_k, 4), "max": (fixed_k, 4)},
        "voxel_features": {"min": (fixed_k, 32, 4), "opt": (fixed_k, 32, 4), "max": (fixed_k, 32, 4)},
        "voxel_num_points": {"min": (fixed_k,), "opt": (fixed_k,), "max": (fixed_k,)},
    }


def _param_count(model: torch.nn.Module) -> int:
    return sum(int(param.numel()) for param in model.parameters())


def _tensor_sha256(value: torch.Tensor) -> str:
    import hashlib

    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("utf-8"))
    digest.update(json.dumps(list(tensor.shape)).encode("utf-8"))
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _all_keep_model_identity(original: torch.nn.Module, physical: torch.nn.Module) -> dict[str, Any]:
    original_state = original.state_dict()
    physical_state = physical.state_dict()
    keys_match = list(original_state) == list(physical_state)
    rows = []
    for key in sorted(set(original_state) | set(physical_state)):
        left = original_state.get(key)
        right = physical_state.get(key)
        row: dict[str, Any] = {
            "key": key,
            "present_original": left is not None,
            "present_physical": right is not None,
        }
        if left is not None and right is not None:
            row.update(
                {
                    "original_shape": list(left.shape),
                    "physical_shape": list(right.shape),
                    "original_dtype": str(left.dtype),
                    "physical_dtype": str(right.dtype),
                    "shape_equal": tuple(left.shape) == tuple(right.shape),
                    "dtype_equal": left.dtype == right.dtype,
                    "exact_equal": bool(torch.equal(left.detach().cpu(), right.detach().cpu())),
                    "original_sha256": _tensor_sha256(left),
                    "physical_sha256": _tensor_sha256(right),
                }
            )
        rows.append(row)
    issues = [
        row["key"]
        for row in rows
        if not (
            row.get("present_original")
            and row.get("present_physical")
            and row.get("shape_equal")
            and row.get("dtype_equal")
            and row.get("exact_equal")
            and row.get("original_sha256") == row.get("physical_sha256")
        )
    ]
    original_count = _param_count(original)
    physical_count = _param_count(physical)
    return {
        "passed": bool(keys_match and not issues and original_count == physical_count),
        "pruned_unit_count": 0,
        "state_dict_key_order_equal": keys_match,
        "original_parameter_count": original_count,
        "physical_parameter_count": physical_count,
        "parameter_count_equal": original_count == physical_count,
        "mismatched_tensor_keys": issues,
        "tensors": rows,
    }


def _apply_group_output_precision_contract(mapping: Any, contracts: dict[str, Any]) -> Any:
    """Keep compute precision independent from an explicit FP16 merge output."""

    return type(mapping)(
        entries=[
            replace(
                row,
                realized_output_precision=(
                    "fp16"
                    if row.realized_request_precision == "int8"
                    and str(contracts.get(row.precision_group, {}).get("output_precision_policy", "")) == "FP16"
                    else row.realized_output_precision
                ),
            )
            for row in mapping.entries
        ],
        profile_id=mapping.profile_id,
        profile_hash=mapping.profile_hash,
        origin_map_hash=mapping.origin_map_hash,
        policy_version=mapping.policy_version,
        auxiliary_layer_precisions=dict(mapping.auxiliary_layer_precisions),
        auxiliary_layer_output_types=dict(mapping.auxiliary_layer_output_types),
    )


def _physical_selection_key(phenotype: CandidatePhenotype, checkpoint_hash: str) -> str:
    metadata = dict(phenotype.metadata or {})
    return canonical_json_hash(
        {
            "checkpoint_hash": checkpoint_hash,
            "pruned_unit_ids": sorted(phenotype.pruned_unit_ids),
            "resolved_prune_unit_ids": metadata.get("resolved_prune_unit_ids", []),
            "resolved_prune_indices": metadata.get("resolved_prune_indices", {}),
            "resolved_prune_indices_by_scope": metadata.get("resolved_prune_indices_by_scope", {}),
            "group_keep_map_by_scope": metadata.get("group_keep_map_by_scope", {}),
            "group_prune_map_by_scope": metadata.get("group_prune_map_by_scope", {}),
            "pruning_policy_version": phenotype.pruning_policy_version,
        }
    )


def _tensorrt_cache_identity(tensorrt: Any) -> dict[str, Any]:
    payload = dict(tensorrt.to_dict() if hasattr(tensorrt, "to_dict") else tensorrt)
    payload.pop("env_hash", None)
    return payload


def _as_dict(value: Any) -> Any:
    return value.to_dict() if hasattr(value, "to_dict") else value


def _width_value(row: dict[str, Any], *names: str) -> int | None:
    for name in names:
        value = row.get(name)
        if value is not None:
            return int(value)
    return None


def _write_physical_widths_csv(path: str | Path, *, snapshot_payload: dict[str, Any], plan_payload: dict[str, Any]) -> None:
    entries_by_module: dict[str, dict[str, dict[str, Any]]] = {}
    for entry in plan_payload.get("entries", []) or []:
        module_path = str(entry.get("module_path", ""))
        axis = str(entry.get("axis", ""))
        if module_path and axis:
            entries_by_module.setdefault(module_path, {})[axis] = dict(entry)
    fields = [
        "module path",
        "op type",
        "original C_in",
        "original C_out",
        "pruned C_in",
        "pruned C_out",
        "groups",
        "channels per group before",
        "channels per group after",
        "alignment status",
    ]
    rows: list[dict[str, Any]] = []
    allowed_group_widths = {4, 8, 16, 32, 64, 128, 256, 512}
    for module in snapshot_payload.get("modules", []) or []:
        module_path = str(module.get("canonical_module_name", ""))
        op_type = str(module.get("module_type", ""))
        groups = int(module.get("groups") or 1)
        original_in = _width_value(module, "in_channels", "in_features", "num_features")
        original_out = _width_value(module, "out_channels", "out_features", "num_features")
        pruned_in = original_in
        pruned_out = original_out
        plan_axes = entries_by_module.get(module_path, {})
        in_entry = plan_axes.get("in")
        out_entry = plan_axes.get("out") or plan_axes.get("channel")
        if in_entry is not None:
            original_in = int(in_entry.get("original_axis_size", original_in or 0))
            pruned_in = len(in_entry.get("keep_indices", []) or [])
        if out_entry is not None:
            original_out = int(out_entry.get("original_axis_size", original_out or 0))
            pruned_out = len(out_entry.get("keep_indices", []) or [])
        before = ""
        after = ""
        status = "not_channel_pruned"
        changed_widths = []
        if in_entry is not None and original_in is not None and pruned_in is not None:
            changed_widths.append((original_in, pruned_in))
        if out_entry is not None and original_out is not None and pruned_out is not None:
            changed_widths.append((original_out, pruned_out))
        if op_type in {"Conv2d", "ConvTranspose2d"} and changed_widths:
            if groups > 1:
                display_before, display_after = changed_widths[-1]
                if display_before % groups == 0 and display_after % groups == 0:
                    before_value = display_before // groups
                    after_value = display_after // groups
                else:
                    before_value = 0
                    after_value = 0
                before = before_value
                after = after_value
                if all(
                    original % groups == 0
                    and pruned % groups == 0
                    and (pruned // groups) in allowed_group_widths
                    for original, pruned in changed_widths
                ):
                    status = "grouped_width_safe"
                else:
                    status = "grouped_width_not_in_safe_set"
            else:
                status = "dense_width_aligned" if all(pruned % 4 == 0 for _original, pruned in changed_widths) else "dense_width_not_multiple_of_4"
        elif op_type in {"Linear", "BatchNorm1d", "BatchNorm2d"}:
            status = "not_conv_alignment_target"
        rows.append(
            {
                "module path": module_path,
                "op type": op_type,
                "original C_in": "" if original_in is None else original_in,
                "original C_out": "" if original_out is None else original_out,
                "pruned C_in": "" if pruned_in is None else pruned_in,
                "pruned C_out": "" if pruned_out is None else pruned_out,
                "groups": groups,
                "channels per group before": before,
                "channels per group after": after,
                "alignment status": status,
            }
        )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_physical_artifact_files(
    *,
    output_dir: str | Path,
    request: Any,
    plan: Any,
    ledger: Any,
    snapshot: Any,
    validation: Any,
    model: torch.nn.Module,
    checkpoint_hash: str,
    physical_hash_value: str,
    parameter_count_base: int,
    parameter_count_pruned: int,
    plan_validation: dict[str, Any],
) -> None:
    destination = Path(output_dir)
    request_payload = _as_dict(request)
    plan_payload = _as_dict(plan)
    ledger_payload = _as_dict(ledger)
    snapshot_payload = _as_dict(snapshot)
    validation_payload = _as_dict(validation)
    _write_json(destination / "pruning_request.json", request_payload)
    _write_json(destination / "sampling_pruning_request.json", request_payload)
    _write_json(destination / "physical_plan_validation.json", plan_validation)
    _write_json(destination / "physical_plan.json", plan_payload)
    _write_json(destination / "physical_pruning_plan.json", plan_payload)
    _write_json(destination / "legalized_plan.json", plan_payload)
    _write_json(destination / "materialization_ledger.json", ledger_payload)
    _write_json(destination / "materialization_report.json", ledger_payload)
    _write_json(destination / "physical_snapshot.json", snapshot_payload)
    _write_json(destination / "physical_structure_snapshot.json", snapshot_payload)
    _write_json(destination / "physical_validation.json", validation_payload)
    _write_physical_widths_csv(destination / "physical_widths.csv", snapshot_payload=snapshot_payload, plan_payload=plan_payload)
    checkpoint_payload = {"model": model.state_dict(), "checkpoint_hash": checkpoint_hash}
    torch.save(checkpoint_payload, destination / "pruned_state_dict.pth")
    torch.save(checkpoint_payload, destination / "pruned_checkpoint.pth")
    _write_json(
        destination / "physical_hash.json",
        {
            "physical_hash": physical_hash_value,
            "parameter_count_base": int(parameter_count_base),
            "parameter_count_pruned": int(parameter_count_pruned),
            "parameter_reduction": 1.0 - (int(parameter_count_pruned) / max(int(parameter_count_base), 1)),
        },
    )


class LidarPyramidRealEvaluator:
    """Evaluate one phenotype through the formal pruning and Q/DQ deployment chain."""

    def __init__(
        self,
        *,
        context: LidarPyramidSearchContext,
        run_dir: str | Path,
        num_frames: int,
        warmup_frames: int,
        latency_rounds: int,
        target_bops_retention: float | None = None,
        bops_tolerance: float = 0.005,
        stage2_config: Stage2ObjectiveConfig | None = None,
        artifact_cache: ArtifactCache | None = None,
        real_cache: RealEvalCache | None = None,
    ) -> None:
        self.context = context
        self.run_dir = Path(run_dir)
        self.num_frames = int(num_frames)
        self.warmup_frames = int(warmup_frames)
        self.latency_rounds = int(latency_rounds)
        self.target_bops_retention = (
            None if target_bops_retention is None else float(target_bops_retention)
        )
        self.bops_tolerance = float(bops_tolerance)
        self.objective_config = stage2_config or Stage2ObjectiveConfig()
        archives = self.run_dir / "archives"
        self.artifacts = artifact_cache or ArtifactCache(archives / "artifact_index.jsonl")
        self.real_cache = real_cache or RealEvalCache(archives / "real_eval_archive.jsonl")
        self.pruning = FormalPruningAdapter()
        self._baseline: dict[str, Any] | None = None
        self._physical_memory: dict[str, dict[str, Any]] = {}
        self._baseline_bops_shapes: tuple[Any, ...] | None = None
        self._baseline_bops_snapshot: Any | None = None

    def evaluate_baseline(self) -> dict[str, Any]:
        if self._baseline is not None:
            return dict(self._baseline)
        key = canonical_json_hash(
            {
                "kind": "baseline",
                "checkpoint_hash": self.context.checkpoint_hash,
                "eval_manifest_hash": self.context.eval_manifest_hash,
                "num_frames": self.num_frames,
                "warmup_frames": self.warmup_frames,
                "latency_rounds": self.latency_rounds,
                "gpu": self.context.physical_gpu_id,
                "tensorrt": _tensorrt_cache_identity(self.context.tensorrt),
            }
        )
        cached = self.real_cache.get(key)
        if cached is not None:
            cached["cache_hit"] = True
            self._baseline = cached
            return dict(cached)
        baseline_dir = self.run_dir / "baseline"
        phenotype = self._default_precision_phenotype([])
        raw = self._deploy_and_evaluate(
            phenotype=phenotype,
            output_dir=baseline_dir / "baseline_engine",
            candidate_label="baseline",
            pruned_unit_ids=[],
        )
        if raw.get("status") != "ok":
            raise RuntimeError(f"baseline_evaluation_failed:{raw.get('failure_reason', raw.get('status'))}")
        baseline_eval = dict(raw["evaluation"])
        baseline_eval["status"] = "ok"
        baseline_eval["cache_key"] = key
        _write_json(baseline_dir / "baseline_eval.json", baseline_eval)
        self._copy_latency(raw["evaluation"], baseline_dir / "baseline_latency.csv")
        self.real_cache.put(key, baseline_eval)
        self._baseline = baseline_eval
        return dict(baseline_eval)

    def evaluate_original_baseline(self, baseline_precision: str, *, full_validation: bool = False) -> dict[str, Any]:
        kind = str(baseline_precision).lower()
        baseline_dir = self.run_dir / "baselines" / f"original_{kind}"
        baseline_dir.mkdir(parents=True, exist_ok=True)
        key = canonical_json_hash(
            {
                "kind": "original_precision_baseline",
                "precision": kind,
                "checkpoint_hash": self.context.checkpoint_hash,
                "eval_manifest_hash": self.context.eval_manifest_hash,
                "num_frames": self.num_frames,
                "warmup_frames": self.warmup_frames,
                "latency_rounds": self.latency_rounds,
                "full_validation": bool(full_validation),
                "qdq_calibration_semantics_version": QDQ_CALIBRATION_SEMANTICS_VERSION,
                "trusted_explicit_qdq_profile_hash": canonical_json_hash(
                    list(TRUSTED_EXPLICIT_QDQ_INT8_V1_MODULES)
                ),
                "explicit_qdq_boundary_policy": "weighted_output_with_engine_fusion_verification",
                "gpu": self.context.physical_gpu_id,
                "tensorrt": _tensorrt_cache_identity(self.context.tensorrt),
            }
        )
        cached = self.real_cache.get(key)
        if cached is not None:
            cached["cache_hit"] = True
            _write_json(baseline_dir / "baseline_cache_hit.json", {"cache_key": key, "engine_hash": cached.get("engine_hash", "")})
            return dict(cached)
        existing = self._load_existing_original_baseline(baseline_dir, kind, key)
        if existing is not None:
            self.real_cache.put(key, existing)
            _write_json(baseline_dir / "baseline_cache_hit.json", {"cache_key": key, "engine_hash": existing.get("engine_hash", ""), "cache_source": "existing_baseline_eval"})
            return existing
        phenotype = self._baseline_precision_phenotype(kind)
        raw = self._deploy_and_evaluate(
            phenotype=phenotype,
            output_dir=baseline_dir,
            candidate_label=f"original_{kind}",
            pruned_unit_ids=[],
            baseline_precision=kind,
        )
        if raw.get("status") != "ok":
            result = {
                "status": str(raw.get("status", "baseline_failed")),
                "failure_reason": str(raw.get("failure_reason", "")),
                "baseline_precision": kind,
                "cache_key": key,
            }
        else:
            result = {
                **raw["evaluation"],
                "baseline_precision": kind,
                "cache_key": key,
                "deployment_hash": raw.get("deployment_hash", ""),
                "eval_hash": raw.get("eval_hash", ""),
                "physical_hash": raw.get("physical_hash", ""),
                "engine_hash": raw.get("engine_hash", ""),
                "engine_path": raw.get("engine_path", ""),
                "precision_validation": raw.get("baseline_precision_validation", {}),
                "qdq_realization_summary": raw.get("qdq_realization_summary", {}),
                "status": "ok",
            }
        _write_json(baseline_dir / "baseline_eval.json", result)
        self.real_cache.put(key, result)
        return result

    def evaluate_original_baselines(self, precisions: list[str] | tuple[str, ...]) -> dict[str, Any]:
        rows = {str(precision): self.evaluate_original_baseline(str(precision), full_validation=True) for precision in precisions}
        table_dir = self.run_dir / "baselines"
        _write_json(table_dir / "full_validation_baselines.json", rows)
        self._write_baseline_csv(table_dir / "full_validation_baselines.csv", rows)
        self._write_baseline_markdown(table_dir / "full_validation_baselines.md", rows)
        return rows

    def evaluate_candidate(self, phenotype: CandidatePhenotype, *, output_dir: str | Path, candidate_hash: str) -> dict[str, Any]:
        baseline = self._stage2_reference_baseline()
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        cache_key = canonical_json_hash(
            {
                "candidate_hash": candidate_hash,
                "deployment_pipeline_version": "lidar-pyramid-stage2-bn-fold-aware-v2",
                "qdq_calibration_semantics_version": QDQ_CALIBRATION_SEMANTICS_VERSION,
                "eval_manifest_hash": self.context.eval_manifest_hash,
                "num_frames": self.num_frames,
                "warmup_frames": self.warmup_frames,
                "latency_rounds": self.latency_rounds,
                "stage2_reference_policy": "strict_fp32_ap_strict_fp16_latency_v1",
                "objective_config": asdict(self.objective_config),
                "gpu": self.context.physical_gpu_id,
                "tensorrt": _tensorrt_cache_identity(self.context.tensorrt),
            }
        )
        cached = self.real_cache.get(cache_key)
        if cached is not None:
            cached["cache_hit"] = True
            if not (destination / "stage2_score.json").exists():
                _write_json(destination / "stage2_score.json", cached)
            write_candidate_summary_artifacts(
                destination,
                candidate_hash=candidate_hash,
                phenotype=phenotype,
                stage2_score=cached,
                objective_config=self.objective_config,
                stage1_manifest_record=self._stage1_manifest_record(candidate_hash),
                overwrite=False,
            )
            return dict(cached)
        _write_json(destination / "phenotype.json", phenotype.to_dict())
        raw = self._load_existing_deployment_evaluation(destination)
        if raw is None:
            raw = self._deploy_and_evaluate(
                phenotype=phenotype,
                output_dir=destination,
                candidate_label=candidate_hash,
                pruned_unit_ids=phenotype.pruned_unit_ids,
            )
        if raw.get("status") == "ok":
            scored = compute_stage2_score(raw["evaluation"], baseline=baseline, config=self.objective_config)
            result = {
                **raw["evaluation"],
                **scored,
                "candidate_hash": candidate_hash,
                "cache_key": cache_key,
                "deployment_hash": raw.get("deployment_hash", ""),
                "eval_hash": raw.get("eval_hash", ""),
                "physical_hash": raw.get("physical_hash", ""),
                "engine_hash": raw.get("engine_hash", ""),
                "engine_path": raw.get("engine_path", ""),
                "status": str(scored.get("status", "ok")),
                "artifact_dir": str(destination),
            }
        else:
            result = {
                "candidate_hash": candidate_hash,
                "cache_key": cache_key,
                "status": str(raw.get("status", "evaluation_failed")),
                "failure_reason": str(raw.get("failure_reason", raw.get("status", ""))),
                "F2": float("inf"),
                "artifact_dir": str(destination),
            }
        _write_json(destination / "stage2_score.json", result)
        write_candidate_summary_artifacts(
            destination,
            candidate_hash=candidate_hash,
            phenotype=phenotype,
            stage2_score=result,
            objective_config=self.objective_config,
            stage1_manifest_record=self._stage1_manifest_record(candidate_hash),
        )
        self.real_cache.put(cache_key, result)
        return result

    def _stage2_reference_baseline(self) -> dict[str, Any]:
        accuracy = self.evaluate_original_baseline("strict_fp32", full_validation=False)
        latency = self.evaluate_original_baseline("strict_fp16", full_validation=False)
        if str(accuracy.get("status", "")) != "ok":
            raise RuntimeError(f"strict_fp32_accuracy_baseline_failed:{accuracy.get('failure_reason', accuracy.get('status'))}")
        if str(latency.get("status", "")) != "ok":
            raise RuntimeError(f"strict_fp16_latency_baseline_failed:{latency.get('failure_reason', latency.get('status'))}")
        metric = self.objective_config.latency_metric
        combined = {
            "status": "ok",
            "mAP": float(accuracy.get("mAP", accuracy.get("map", 0.0)) or 0.0),
            metric: float(latency.get(metric, 0.0) or 0.0),
            "accuracy_reference": "original_strict_fp32",
            "latency_reference": "original_strict_fp16",
            "strict_fp32": accuracy,
            "strict_fp16": latency,
        }
        _write_json(self.run_dir / "stage2_reference_baseline.json", combined)
        return combined

    @staticmethod
    def _load_existing_deployment_evaluation(output_dir: Path) -> dict[str, Any] | None:
        evaluation_path = output_dir / "evaluation.json"
        manifest_path = output_dir / "deployment_manifest.json"
        engine_path = output_dir / "engine.plan"
        if not (evaluation_path.is_file() and manifest_path.is_file() and engine_path.is_file()):
            return None
        try:
            evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        if str(evaluation.get("status", "")) != "ok":
            return None
        return {
            "status": "ok",
            "evaluation": evaluation,
            "physical_hash": manifest.get("physical_hash", ""),
            "deployment_hash": manifest.get("deployment_hash", ""),
            "eval_hash": manifest.get("eval_hash", ""),
            "engine_hash": manifest.get("engine_hash", _file_hash(engine_path)),
            "engine_path": str(engine_path),
            "baseline_precision_validation": {},
            "qdq_realization_summary": json.loads((output_dir / "qdq_realization_summary.json").read_text(encoding="utf-8")) if (output_dir / "qdq_realization_summary.json").is_file() else {},
        }

    @staticmethod
    def _load_existing_original_baseline(baseline_dir: Path, kind: str, cache_key: str) -> dict[str, Any] | None:
        baseline_eval_path = baseline_dir / "baseline_eval.json"
        engine_path = baseline_dir / "engine.plan"
        evaluation_path = baseline_dir / "evaluation.json"
        if not (baseline_eval_path.is_file() and engine_path.is_file() and evaluation_path.is_file()):
            return None
        try:
            result = json.loads(baseline_eval_path.read_text(encoding="utf-8"))
            evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        if str(result.get("status", "")) != "ok" or str(evaluation.get("status", "")) != "ok":
            return None
        recorded_kind = str(result.get("baseline_precision", kind)).lower()
        if recorded_kind and recorded_kind != str(kind).lower():
            return None
        loaded = dict(result)
        loaded["cache_key"] = cache_key
        loaded["cache_hit"] = True
        loaded["cache_source"] = "existing_baseline_eval"
        loaded.setdefault("baseline_precision", str(kind).lower())
        loaded.setdefault("engine_path", str(engine_path))
        loaded.setdefault("engine_hash", _file_hash(engine_path))
        return loaded

    def _stage1_manifest_record(self, candidate_hash: str) -> dict[str, Any]:
        for path in sorted(self.run_dir.glob("round_*/repaired_top5_manifest.json")):
            try:
                manifest = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            for row in manifest.get("candidates", []) or []:
                if str(row.get("repaired_phenotype_hash")) == str(candidate_hash):
                    return dict(row)
        return {}

    def _default_precision_phenotype(self, pruned_unit_ids: list[str]) -> CandidatePhenotype:
        from ..candidate import PrecisionDecision
        from ..quantization_space.legalizer import legalize_group_precision_genes

        if self.context.search_space.quantization_groups:
            genes = {
                group.group_id: self.context.search_space.default_precision
                for group in self.context.search_space.quantization_groups
            }
            legalization = legalize_group_precision_genes(
                genes,
                self.context.search_space.quantization_groups,
                default_precision=self.context.search_space.default_precision,
            )
            return CandidatePhenotype(
                pruned_unit_ids=pruned_unit_ids,
                precision_profile=legalization.expand_to_module_profile(),
                metadata=legalization.to_dict(),
            )
        return CandidatePhenotype(
            pruned_unit_ids=pruned_unit_ids,
            precision_profile={
                layer: PrecisionDecision("FP16", "FP16", "")
                for layer in self.context.search_space.precision_layer_ids
            },
        )

    def _baseline_precision_phenotype(self, baseline_precision: str) -> CandidatePhenotype:
        from ..candidate import PrecisionDecision
        from ..quantization_space.legalizer import legalize_group_precision_genes

        kind = str(baseline_precision).lower()
        if not self.context.search_space.quantization_groups:
            precision = "FP32" if kind == "strict_fp32" else "FP16"
            if kind in {"maximal_legal_int8", "pure_strict_int8"}:
                precision = "INT8"
            return CandidatePhenotype(
                pruned_unit_ids=[],
                precision_profile={
                    layer: PrecisionDecision(precision, precision, "")
                    for layer in self.context.search_space.precision_layer_ids
                },
                metadata={"baseline_precision": kind},
            )
        requested: dict[str, str] = {}
        trusted_modules = set(TRUSTED_EXPLICIT_QDQ_INT8_V1_MODULES)
        known_modules = {
            module_path
            for group in self.context.search_space.quantization_groups
            for module_path in group.module_paths
        }
        if kind == "trusted_explicit_qdq_int8":
            missing = sorted(trusted_modules - known_modules)
            if missing:
                raise RuntimeError(f"trusted_explicit_qdq_profile_modules_missing:{missing}")
        for group in self.context.search_space.quantization_groups:
            if kind == "strict_fp32":
                requested[group.group_id] = "FP32"
            elif kind == "strict_fp16":
                requested[group.group_id] = "FP16"
            elif kind == "pure_strict_int8":
                requested[group.group_id] = "INT8"
            elif kind == "trusted_explicit_qdq_int8":
                selected = any(module_path in trusted_modules for module_path in group.module_paths)
                if selected and ("INT8" not in group.allowed_precisions or group.protected):
                    raise RuntimeError(f"trusted_explicit_qdq_profile_group_not_legal:{group.group_id}")
                requested[group.group_id] = "INT8" if selected else "FP16"
            else:
                requested[group.group_id] = "INT8" if "INT8" in group.allowed_precisions and not group.protected else "FP16"
        legalization = legalize_group_precision_genes(
            requested,
            self.context.search_space.quantization_groups,
            default_precision=self.context.search_space.default_precision,
        )
        return CandidatePhenotype(
            pruned_unit_ids=[],
            precision_profile=legalization.expand_to_module_profile(),
            metadata={**legalization.to_dict(), "baseline_precision": kind},
        )

    def _deploy_and_evaluate(
        self,
        *,
        phenotype: CandidatePhenotype,
        output_dir: Path,
        candidate_label: str,
        pruned_unit_ids: list[str],
        baseline_precision: str | None = None,
    ) -> dict[str, Any]:
        try:
            physical = self._materialize_physical(phenotype, output_dir)
            qdq = self._export_qdq(phenotype, physical, output_dir)
            trt = self._build_engine(qdq, physical, output_dir, baseline_precision=baseline_precision)
            if trt.get("status") != "ok":
                return {"status": trt.get("status", "engine_build_failed"), "failure_reason": trt.get("failure_reason", trt.get("status", ""))}
            engine_precision = engine_realized_precision_profile(
                output_dir / "engine_layer_info.json",
                qdq["precision_mapping"],
            )
            _write_json(output_dir / "engine_realized_precision_profile.json", engine_precision)
            if not engine_precision["passed"]:
                return {
                    "status": "precision_realization_failure",
                    "failure_reason": ",".join(engine_precision["issues"]),
                }
            lineage = _qdq_deployment_lineage(
                base_onnx_path=qdq["base_onnx_path"],
                canonical_mapping=qdq["precision_mapping"],
                legalized_precision_profile=qdq["legalized_precision_profile"],
                realized_precision_profile=engine_precision["realized_precision_profile"],
                calibration_recipe=qdq["calibration_recipe"],
                qdq_result=qdq["qdq"],
            )
            qdq.update(lineage)
            _write_json(output_dir / "deployment_lineage.json", lineage)
            physical_shapes = profile_runtime_layer_shapes(
                physical["model"],
                self.context.trace_example_inputs,
                forward_fn=self.context.model_bundle.adapter.forward_for_task,
            )
            _write_json(output_dir / "physical_runtime_layer_shapes.json", physical_shapes.to_dict())
            if self._baseline_bops_shapes is None or self._baseline_bops_snapshot is None:
                baseline_shapes = profile_runtime_layer_shapes(
                    self.context.model,
                    self.context.trace_example_inputs,
                    forward_fn=self.context.model_bundle.adapter.forward_for_task,
                )
                self._baseline_bops_shapes = baseline_shapes.shapes
                self._baseline_bops_snapshot = self.pruning.snapshot_fn(self.context.model)
                _write_json(self.run_dir / "baseline" / "fp32_bops_runtime_layer_shapes.json", baseline_shapes.to_dict())
                _write_json(self.run_dir / "baseline" / "fp32_bops_physical_snapshot.json", self._baseline_bops_snapshot)
            admission_target = (
                None
                if baseline_precision is not None or candidate_label == "baseline"
                else self.target_bops_retention
            )
            bops_audit = compute_realized_bops(
                physical_runtime_shapes=physical_shapes.shapes,
                baseline_runtime_shapes=self._baseline_bops_shapes,
                realized_precision_profile=engine_precision["realized_precision_profile"],
                physical_snapshot=physical["snapshot"],
                baseline_snapshot=self._baseline_bops_snapshot,
                target_retention=admission_target,
                tolerance=self.bops_tolerance,
            )
            _write_json(output_dir / "realized_bops_audit.json", bops_audit)
            if not bops_audit["passed"]:
                return {
                    "status": "realized_BOPS_out_of_budget",
                    "failure_reason": (
                        f"{bops_audit['bops_retention']} not in "
                        f"{bops_audit['legal_interval']}"
                    ),
                    "realized_bops_audit": bops_audit,
                }
            evaluation = self._evaluate_engine(trt["engine_path"], output_dir)
            if evaluation.get("status") != "ok":
                return {"status": "evaluation_failed", "failure_reason": evaluation.get("failure_reason", evaluation.get("status", "")), "evaluation": evaluation}
            evaluation.update(
                {
                    "realized_BOPS": bops_audit["realized_bops"],
                    "BOPS_retention": bops_audit["bops_retention"],
                    "physical_params": bops_audit["physical_params"],
                    "parameter_retention": bops_audit["parameter_retention"],
                    "weight_storage_retention": bops_audit["weight_storage_retention"],
                    "realized_precision_counts": bops_audit["realized_precision_counts"],
                }
            )
            _write_json(output_dir / "evaluation.json", evaluation)
            calibration_scale_hash = canonical_json_hash(qdq.get("calibration_scales", {}))
            deployment_signature = _candidate_deployment_signature(
                context=self.context,
                physical=physical,
                qdq=qdq,
                trt=trt,
            )
            _write_json(output_dir / "deployment_signature.json", deployment_signature)
            deploy_hash = deployment_hash(
                deployment_signature=deployment_signature,
                calibration_scale_hash=calibration_scale_hash,
                onnx_export_config_hash=self.context.search_space.onnx_export_config_hash,
                builder_flags=self.context.search_space.builder_flags,
                optimization_profiles=_shape_profiles(),
                plugin_hashes=self.context.search_space.plugin_hashes,
                quantization_contract_hash=canonical_json_hash(
                    {
                        "groups": qdq.get("quantization_group_contracts", {}),
                        "merge_precision_realization": qdq.get("merge_precision_realization", {}),
                    }
                ),
            )
            eval_key = eval_hash(
                deployment_hash_value=deploy_hash,
                validation_manifest_hash=self.context.eval_manifest_hash,
                evaluation_config_hash=canonical_json_hash({"num_frames": self.num_frames, "warmup": self.warmup_frames, "rounds": self.latency_rounds}),
                postprocess_config={"source": "HEAL dataset.post_process"},
                warmup=self.warmup_frames,
                rounds=self.latency_rounds,
                latency_metric_definition=self.objective_config.latency_metric,
            )
            _write_json(
                output_dir / "deployment_manifest.json",
                {
                    "candidate_label": candidate_label,
                    "pruned_unit_ids": pruned_unit_ids,
                    "physical_hash": physical["physical_hash"],
                    "deployment_hash": deploy_hash,
                    "deployment_signature": deployment_signature,
                    "deployment_signature_hash": canonical_json_hash(deployment_signature),
                    "eval_hash": eval_key,
                    "engine_hash": trt.get("engine_hash", ""),
                    "quantization_contract_hash": canonical_json_hash(
                        {
                            "groups": qdq.get("quantization_group_contracts", {}),
                            "merge_precision_realization": qdq.get("merge_precision_realization", {}),
                        }
                    ),
                },
            )
            return {
                "status": "ok",
                "evaluation": evaluation,
                "physical_hash": physical["physical_hash"],
                "deployment_hash": deploy_hash,
                "eval_hash": eval_key,
                "engine_hash": trt.get("engine_hash", ""),
                "engine_path": trt.get("engine_path", ""),
                "baseline_precision_validation": trt.get("baseline_precision_validation", {}),
                "qdq_realization_summary": qdq.get("qdq_realization_summary", {}),
            }
        except Exception as exc:  # noqa: BLE001
            return {"status": "evaluation_failed", "failure_reason": f"{type(exc).__name__}: {exc}"}

    def _materialize_physical(self, phenotype: CandidatePhenotype, output_dir: Path) -> dict[str, Any]:
        selection_key = _physical_selection_key(phenotype, self.context.checkpoint_hash)
        if selection_key in self._physical_memory:
            result = dict(self._physical_memory[selection_key])
            result["cache_hit"] = True
            _write_json(output_dir / "physical_cache_hit.json", {"selection_key": selection_key, "physical_hash": result["physical_hash"]})
            _write_physical_artifact_files(
                output_dir=output_dir,
                request=result["request"],
                plan=result["plan"],
                ledger=result["ledger"],
                snapshot=result["snapshot"],
                validation=result["validation"],
                model=result["model"],
                checkpoint_hash=self.context.checkpoint_hash,
                physical_hash_value=result["physical_hash"],
                parameter_count_base=int(result["parameter_count_base"]),
                parameter_count_pruned=int(result["parameter_count_pruned"]),
                plan_validation=dict(result.get("plan_validation") or {"passed": True, "cache_hit": True}),
            )
            if not phenotype.pruned_unit_ids:
                identity = _all_keep_model_identity(self.context.model, result["model"])
                _write_json(output_dir / "all_keep_model_identity.json", identity)
                if not identity["passed"]:
                    raise RuntimeError("all_keep_physical_model_identity_failed")
            return result
        action_ids = set(getattr(self.context.pruning_action_catalog, "action_ids", []) or []) if getattr(self.context, "pruning_action_catalog", None) is not None else set()
        if action_ids and set(phenotype.pruned_unit_ids).issubset(action_ids):
            genes = {action_id: (0 if action_id in set(phenotype.pruned_unit_ids) else 1) for action_id in self.context.search_space.pruning_unit_ids}
            actions = selected_actions_from_genes(genes, self.context.pruning_action_catalog.actions)
            request = request_from_pruning_actions(actions)
        else:
            request = self.pruning.request_from_phenotype(phenotype, self.context.atomic_prune_units)
        _write_json(output_dir / "pruning_request.json", request.to_dict())
        _write_json(output_dir / "sampling_pruning_request.json", request.to_dict())
        plan = self.pruning.build_plan_fn(self.context.model, request)
        legal_plan = self.pruning.legalize_plan_fn(self.context.model, plan)
        plan_validation = validate_repaired_physical_plan(phenotype, request, legal_plan)
        _write_json(output_dir / "physical_plan_validation.json", plan_validation)
        if not bool(plan_validation.get("passed", False)):
            raise RuntimeError("repaired_physical_plan_mismatch")
        materialized_result = self.pruning.materialize_fn(self.context.model, legal_plan, in_place=False, build_snapshot=True)
        snapshot = materialized_result.snapshot if getattr(materialized_result, "snapshot", None) is not None else self.pruning.snapshot_fn(materialized_result.model)
        hashes = self.pruning.hash_fn(snapshot)
        validation = self.pruning.validate_fn(
            materialized_result.model,
            expected_snapshot=snapshot,
            grouped_config=None,
            fixed_output_contracts=None,
            example_inputs=(self.context.trace_example_inputs,),
        )
        materialized = {
            "model": materialized_result.model,
            "plan": legal_plan,
            "ledger": materialized_result.ledger,
            "snapshot": snapshot,
            "hashes": hashes,
            "validation": validation,
        }
        if hasattr(validation, "passed") and not bool(validation.passed):
            raise RuntimeError("prune_failed:physical_validation")
        physical_key = physical_hash(
            legal_physical_plan=materialized["plan"],
            physical_snapshot=materialized["snapshot"],
            checkpoint_hash=self.context.checkpoint_hash,
            pruning_policy_version=phenotype.pruning_policy_version,
        )
        result = {
            **materialized,
            "request": request,
            "plan_validation": plan_validation,
            "physical_hash": physical_key,
            "parameter_count_base": _param_count(self.context.model),
            "parameter_count_pruned": _param_count(materialized["model"]),
        }
        _write_physical_artifact_files(
            output_dir=output_dir,
            request=request,
            plan=materialized["plan"],
            ledger=materialized["ledger"],
            snapshot=materialized["snapshot"],
            validation=validation,
            model=materialized["model"],
            checkpoint_hash=self.context.checkpoint_hash,
            physical_hash_value=physical_key,
            parameter_count_base=int(result["parameter_count_base"]),
            parameter_count_pruned=int(result["parameter_count_pruned"]),
            plan_validation=plan_validation,
        )
        if not phenotype.pruned_unit_ids:
            identity = _all_keep_model_identity(self.context.model, materialized["model"])
            _write_json(output_dir / "all_keep_model_identity.json", identity)
            if not identity["passed"]:
                raise RuntimeError("all_keep_physical_model_identity_failed")
        self.artifacts.put_physical(physical_key, {"artifact_dir": str(output_dir), "pruned_state_dict": str(output_dir / "pruned_state_dict.pth")})
        self._physical_memory[selection_key] = result
        return result

    def _export_qdq(self, phenotype: CandidatePhenotype, physical: dict[str, Any], output_dir: Path) -> dict[str, Any]:
        try:
            from quantization.api import apply_fp16_merge_output_contract, build_canonical_precision_mapping, export_pruned_signal_maxk_onnx, insert_explicit_qdq, prepare_signal_maxk_inputs
            from quantization.config import CalibrationConfig, CanonicalNamingConfig, OnnxExportConfig, QDQConfig
            from quantization.types import CanonicalPrecisionMappingResult, PrecisionAssignment, PrecisionProfileResult
        except ImportError:
            from heal_compress.quantization.api import apply_fp16_merge_output_contract, build_canonical_precision_mapping, export_pruned_signal_maxk_onnx, insert_explicit_qdq, prepare_signal_maxk_inputs
            from heal_compress.quantization.config import CalibrationConfig, CanonicalNamingConfig, OnnxExportConfig, QDQConfig
            from heal_compress.quantization.types import CanonicalPrecisionMappingResult, PrecisionAssignment, PrecisionProfileResult

        qcfg = OnnxExportConfig(fixed_k=29696, min_agents=1, opt_agents=2, max_agents=2)
        cached = self.artifacts.get_onnx(str(physical["physical_hash"]))
        cached_onnx = Path(str(cached.get("onnx_path", ""))) if cached else Path()
        cached_origin = Path(str(cached.get("origin_map", ""))) if cached else Path()
        cached_hash = str(cached.get("onnx_sha256", "")) if cached else ""
        cache_valid = (
            cached_onnx.is_file()
            and cached_origin.is_file()
            and bool(cached_hash)
            and _file_hash(cached_onnx) == cached_hash
            and str(cached.get("code_commit", "")) == self.context.code_commit
            and str(cached.get("onnx_export_config_hash", ""))
            == self.context.search_space.onnx_export_config_hash
        )
        if cache_valid:
            target_onnx = output_dir / "exported.onnx"
            if cached_onnx.resolve() != target_onnx.resolve():
                shutil.copyfile(cached_onnx, target_onnx)
            elif not target_onnx.is_file():
                raise RuntimeError("onnx_cache_target_missing")
            export = SimpleNamespace(onnx_path=str(target_onnx), origin_map=_load_origin_map_result(cached_origin))
            _write_json(output_dir / "onnx_cache_hit.json", {"physical_hash": physical["physical_hash"], "onnx_sha256": cached_hash})
        else:
            wrapper = build_search_trt_compatible_export_module(
                physical["model"],
                output_names=qcfg.output_names,
                fixed_k=qcfg.fixed_k,
                modality="m1",
            ).to(self.context.runtime_device).eval()
            inputs = prepare_signal_maxk_inputs(self.context.trace_example_inputs, config=qcfg, modality="m1")
            export = export_pruned_signal_maxk_onnx(
                wrapper,
                inputs,
                output_dir / "exported.onnx",
                physical["snapshot"],
                config=qcfg,
                naming_config=CanonicalNamingConfig(),
                report_path=output_dir / "onnx_export_report.json",
            )
        origin_map = export.origin_map
        if origin_map is None:
            raise RuntimeError("onnx_export_failed:no_origin_map")
        if not (output_dir / "pruned_fp32.onnx").exists():
            shutil.copyfile(output_dir / "exported.onnx", output_dir / "pruned_fp32.onnx")
        _write_json(output_dir / "origin_map.json", origin_map.to_dict())
        if not cache_valid:
            self.artifacts.put_onnx(
                str(physical["physical_hash"]),
                {
                    "onnx_path": str(output_dir / "pruned_fp32.onnx"),
                    "onnx_sha256": _file_hash(output_dir / "pruned_fp32.onnx"),
                    "origin_map": str(output_dir / "origin_map.json"),
                    "code_commit": self.context.code_commit,
                    "onnx_export_config_hash": self.context.search_space.onnx_export_config_hash,
                },
            )
        assignments = []
        requested_profile = {}
        for order, origin in enumerate(sorted(origin_map.entries, key=lambda row: (row.call_index, row.graph_index))):
            decision = phenotype.precision_profile.get(origin.module_path)
            requested = (decision.requested_precision if decision is not None else self.context.search_space.default_precision).lower()
            module_to_group = dict(phenotype.metadata.get("module_to_precision_group") or {})
            if not module_to_group and self.context.search_space.quantization_groups:
                module_to_group = {
                    module_path: group.group_id
                    for group in self.context.search_space.quantization_groups
                    for module_path in group.module_paths
                }
            group_id = str(module_to_group.get(origin.module_path, ""))
            if not group_id:
                if self.context.search_space.quantization_groups:
                    raise RuntimeError(f"missing_quantization_group_member_mapping:{origin.module_path}")
                group_id = f"pg::{origin.module_path}"
            requested_profile[origin.module_path] = requested.upper()
            assignments.append(
                PrecisionAssignment(
                    module_path=origin.module_path,
                    precision_group=group_id,
                    requested_precision=requested,
                    ordering=order,
                )
            )
        profile = PrecisionProfileResult(
            profile_id="search_candidate",
            assignments=assignments,
            requested_int8_count=sum(row.requested_precision == "int8" for row in assignments),
            requested_int8_ratio=sum(row.requested_precision == "int8" for row in assignments) / max(len(assignments), 1),
            policy_version=phenotype.precision_policy_version,
        )
        qdq_config = QDQConfig(
            allowed_precisions=("fp32", "fp16", "int8"),
            weight_granularity="per_channel",
            merge_policy="fp16_merge",
            grouped_conv_int8_allowed_channels_per_group=(4, 8, 16, 32, 64, 128, 256, 512),
        )
        raw_group_contracts = dict(phenotype.metadata.get("quantization_group_contracts") or {})
        if not raw_group_contracts and self.context.search_space.quantization_groups:
            raw_group_contracts = {
                group.group_id: {
                    **dict(group.metadata),
                    "member_layers": list(group.module_paths),
                }
                for group in self.context.search_space.quantization_groups
            }
        mapping = build_canonical_precision_mapping(origin_map, profile, config=qdq_config)
        mapping = _apply_group_output_precision_contract(mapping, raw_group_contracts)
        mapping, canonical_merge_output_contract = apply_fp16_merge_output_contract(
            export.onnx_path,
            mapping,
        )
        from ..group_separation_audit import write_group_separation_audit

        write_group_separation_audit(
            output_dir / "pruning_quantization_group_audit.json",
            pruning_group_ids=self.context.search_space.pruning_unit_ids,
            pruning_group_metadata=self.context.search_space.pruning_unit_metadata,
            quantization_groups=self.context.search_space.quantization_groups,
            canonical_mapping=mapping,
        )
        for row in mapping.entries:
            if row.realized_output_precision != "fp16":
                continue
            contract = raw_group_contracts.get(row.precision_group)
            if contract is None:
                continue
            contract["output_precision_policy"] = "FP16"
            contract["insert_activation_output_qdq"] = False
            contract["merge_boundary_resolution"] = "resolved_from_canonical_onnx_nearest_weighted_producer"
        _write_json(
            output_dir / "canonical_merge_output_contract.json",
            canonical_merge_output_contract,
        )
        realized_profile = {row.module_path: str(row.realized_request_precision).upper() for row in mapping.entries}
        realized_output_profile = {
            row.module_path: str(row.realized_output_precision or row.realized_request_precision).upper()
            for row in mapping.entries
        }
        realized_group_profile: dict[str, str] = {}
        requested_group_profile: dict[str, str] = {}
        for row in mapping.entries:
            requested_group_profile.setdefault(row.precision_group, str(row.requested_precision).upper())
            previous = realized_group_profile.get(row.precision_group)
            realized = str(row.realized_request_precision).upper()
            if previous is None:
                realized_group_profile[row.precision_group] = realized
            elif previous != realized:
                realized_group_profile[row.precision_group] = "FP16"
        fallback_report = {
            row.module_path: row.fallback_reason
            for row in mapping.entries
            if row.requested_precision != row.realized_request_precision or row.fallback_reason
        }
        _write_json(output_dir / "requested_precision_profile.json", requested_profile)
        _write_json(output_dir / "requested_quantization_groups.json", requested_group_profile)
        _write_json(output_dir / "stage1_legalized_quantization_groups.json", phenotype.metadata.get("stage1_legalized_group_profile", requested_group_profile))
        _write_json(output_dir / "legalized_precision_profile.json", phenotype.metadata.get("stage1_legalized_group_profile", requested_group_profile))
        _write_json(output_dir / "realized_precision_profile.json", realized_profile)
        _write_json(output_dir / "stage2_realized_precision_profile.json", realized_profile)
        _write_json(output_dir / "realized_output_precision_profile.json", realized_output_profile)
        _write_json(output_dir / "stage2_realized_quantization_groups.json", realized_group_profile)
        _write_json(output_dir / "precision_group_expansion.json", phenotype.metadata.get("precision_group_expansion", {}))
        group_contracts = {
            str(group_id): {
                **dict(contract),
                "realized_precision": realized_group_profile.get(str(group_id), ""),
                "merge_policy": "A_fp16_merge",
            }
            for group_id, contract in raw_group_contracts.items()
        }
        _write_json(output_dir / "quantization_group_contracts.json", group_contracts)
        _write_json(output_dir / "precision_fallback_report.json", fallback_report)
        _write_json(output_dir / "canonical_layer_map.json", mapping.to_dict())
        int8_modules = sorted(row.module_path for row in mapping.entries if row.realized_request_precision == "int8")
        scales: dict[str, Any] = {}
        calibration_seed = 20260713
        calibration_npz_manifest = getattr(self.context, "quant_calibration_npz_manifest", None)
        calibration_input_identity: dict[str, Any] = {}
        if int8_modules and calibration_npz_manifest is not None:
            calibration_input_identity = fixed_k_calibration_npz_manifest_identity(
                calibration_npz_manifest,
                num_batches=self.context.quant_calibration_batches,
                fixed_k=29696,
            )
            train_indices = list(calibration_input_identity.get("train_dataset_indices", []))
            train_frame_ids = [
                str(train_indices[index]) if index < len(train_indices) else f"npz:{index:06d}"
                for index in range(self.context.quant_calibration_batches)
            ]
            calibration_order = "npz_manifest_file_order"
        else:
            train_frame_ids = load_split_frame_ids(
                self.context.model_bundle.adapter,
                self.context.model_config,
                split="train",
            )[: self.context.quant_calibration_batches]
            calibration_order = "dataset_manifest_order_shuffle_false"
            calibration_input_identity = {
                "source": "dataset_manifest_with_seeded_train_augmentation",
                "sample_count": len(train_frame_ids),
            }
        if int8_modules and len(train_frame_ids) != self.context.quant_calibration_batches:
            raise RuntimeError(
                f"insufficient_train_calibration_manifest_frames:{len(train_frame_ids)}<{self.context.quant_calibration_batches}"
            )
        calibration_frame_manifest_hash = str(
            calibration_input_identity.get("tensor_manifest_hash")
            or canonical_json_hash(
                {"split": "train", "frame_ids": train_frame_ids, "order": "dataset_manifest_order"}
            )
        )
        calibration_backend = str(
            getattr(self.context, "quant_activation_calibration_backend", "modelopt_histogram_entropy")
        ).lower()
        calibration_semantics = (
            TENSORRT_ENTROPY_CALIBRATION_SEMANTICS_VERSION
            if calibration_backend in {
                "external_tensorrt_entropy_cache_exact_match",
                "tensorrt_entropy_calibration2",
            }
            else QDQ_CALIBRATION_SEMANTICS_VERSION
        )
        activation_calibration_method = (
            "tensorrt_entropy_calibration2"
            if calibration_backend in {
                "external_tensorrt_entropy_cache_exact_match",
                "tensorrt_entropy_calibration2",
            }
            else "entropy"
        )
        histogram_bins: int | None = (
            None
            if calibration_backend in {
                "external_tensorrt_entropy_cache_exact_match",
                "tensorrt_entropy_calibration2",
            }
            else 2048
        )
        calibration_identity = {
            "modules": int8_modules,
            "semantics": calibration_semantics,
            "onnx_sha256": _file_hash(export.onnx_path),
            "weight_granularity": qdq_config.weight_granularity,
            "merge_policy": qdq_config.merge_policy,
            "activation_output_boundary_policy": qdq_config.activation_output_boundary_policy,
            "activation_calibration_backend": calibration_backend,
            "activation_calibration_method": activation_calibration_method,
            "histogram_bins": histogram_bins,
            "fixed_k": 29696,
            "calibration_seed": calibration_seed,
            "calibration_frame_manifest_hash": calibration_frame_manifest_hash,
            "calibration_input_manifest_sha256": calibration_input_identity.get("manifest_sha256", ""),
            "calibration_input_source": calibration_input_identity.get("source", ""),
        }
        calibration_cache_path = (
            self.run_dir
            / "archives"
            / "calibration"
            / f"{physical['physical_hash']}_{canonical_json_hash(calibration_identity)}.json"
        )
        calibration_backend_result: dict[str, Any] = {}
        calibration_compatibility: dict[str, Any] = {}
        if int8_modules:
            if calibration_backend == "external_tensorrt_entropy_cache_exact_match":
                external_cache = getattr(
                    self.context,
                    "quant_activation_calibration_cache_path",
                    None,
                )
                if external_cache is None or not Path(external_cache).is_file():
                    raise RuntimeError(
                        f"external_tensorrt_entropy_cache_missing:{external_cache}"
                    )
                calibration_backend_result = {
                    "status": "ok",
                    "external_reference_cache": True,
                    "calibration_cache_path": str(Path(external_cache).resolve()),
                    "calibration_cache_sha256": _file_hash(Path(external_cache)),
                    "calibration_cache_reused_as_reference": True,
                    "tensor_matching": "exact_name_only",
                }
                scales, scale_details = qdq_scales_from_tensorrt_entropy_cache(
                    onnx_path=export.onnx_path,
                    origin_map=origin_map,
                    module_paths=int8_modules,
                    cache_path=external_cache,
                    weight_granularity=qdq_config.weight_granularity,
                    activation_scale_source=(
                        "legacy_single_engine_maxK_fixedK29696_train200_"
                        "TensorRT_EntropyCalibration2_exact_tensor_match"
                    ),
                )
                save_calibration_scales(
                    calibration_cache_path,
                    scales,
                    {
                        **scale_details,
                        "frame_count": self.context.quant_calibration_batches,
                        "fixed_k": 29696,
                        "external_reference_cache": True,
                        "calibration_input_provenance": calibration_input_identity,
                    },
                )
            elif calibration_backend == "tensorrt_entropy_calibration2":
                calibration_onnx = output_dir / "calibration_trt_compatible.onnx"
                calibration_compatibility = make_pointpillar_domain_compatible(
                    export.onnx_path,
                    calibration_onnx,
                )
                _write_json(
                    output_dir / "calibration_onnx_domain_compatibility_report.json",
                    calibration_compatibility,
                )
                calibration_backend_result = build_tensorrt_entropy_calibration_cache_modelopt(
                    onnx_path=calibration_onnx,
                    calibration_npz_manifest=calibration_npz_manifest,
                    output_dir=output_dir / "tensorrt_entropy_calibration",
                    tensorrt_root=self.context.tensorrt.tensorrt_root,
                    plugin_path=self.context.tensorrt.plugin_path,
                    physical_gpu_id=self.context.physical_gpu_id,
                    num_batches=self.context.quant_calibration_batches,
                    fixed_k=29696,
                    conda_env=self.context.tensorrt.conda_env,
                    force_rebuild=bool(self.context.quant_calibration_force_rebuild),
                )
                scales, scale_details = qdq_scales_from_tensorrt_entropy_cache(
                    onnx_path=export.onnx_path,
                    origin_map=origin_map,
                    module_paths=int8_modules,
                    cache_path=calibration_backend_result["calibration_cache_path"],
                    weight_granularity=qdq_config.weight_granularity,
                )
                save_calibration_scales(
                    calibration_cache_path,
                    scales,
                    {
                        **scale_details,
                        "frame_count": self.context.quant_calibration_batches,
                        "fixed_k": 29696,
                        "calibration_input_provenance": calibration_input_identity,
                        "calibration_backend_result_hash": canonical_json_hash(
                            calibration_backend_result
                        ),
                    },
                )
            elif calibration_backend == "modelopt_histogram_entropy":
                scales = collect_or_load_qdq_calibration_scales(
                    model=physical["model"],
                    adapter=self.context.model_bundle.adapter,
                    model_config_path=self.context.model_config,
                    module_paths=int8_modules,
                    device=torch.device(self.context.runtime_device),
                    cache_path=calibration_cache_path,
                    num_batches=self.context.quant_calibration_batches,
                    onnx_path=export.onnx_path,
                    origin_map=origin_map,
                    weight_granularity=qdq_config.weight_granularity,
                    activation_calibration_method="entropy",
                    histogram_bins=2048,
                    fixed_k=29696,
                    calibration_frame_ids=train_frame_ids,
                    calibration_seed=calibration_seed,
                    calibration_npz_manifest=calibration_npz_manifest,
                )
            else:
                raise RuntimeError(
                    f"unsupported_quant_activation_calibration_backend:{calibration_backend}"
                )
        for row in mapping.entries:
            if row.realized_request_precision == "int8" and row.realized_output_precision == "fp16":
                scale = scales.get(row.module_path)
                if scale is None:
                    raise RuntimeError(f"fp16_output_contract_calibration_missing:{row.module_path}")
                scale["insert_activation_output_qdq"] = False
                scale["activation_output_boundary_policy"] = "fp16_output_before_functional_or_merge_boundary"
        _write_json(
            output_dir / "calibration_manifest.json",
            {
                "module_paths": int8_modules,
                "batches": self.context.quant_calibration_batches,
                "semantics_version": calibration_semantics,
                "onnx_sha256": _file_hash(export.onnx_path),
                "weight_granularity": qdq_config.weight_granularity,
                "merge_policy": qdq_config.merge_policy,
                "activation_calibration_backend": calibration_backend,
                "activation_calibration_method": activation_calibration_method,
                "histogram_bins": histogram_bins,
                "fixed_k": 29696,
                "calibration_seed": calibration_seed,
                "calibration_frame_ids": train_frame_ids,
                "calibration_frame_manifest_hash": calibration_frame_manifest_hash,
                "calibration_order": calibration_order,
                "calibration_input_provenance": calibration_input_identity,
                "calibration_cache_path": str(calibration_cache_path),
                "calibration_identity": calibration_identity,
                "calibration_backend_result": calibration_backend_result,
                "calibration_onnx_compatibility": calibration_compatibility,
            },
        )
        _write_json(output_dir / "calibration_scales.json", scales)
        qdq = insert_explicit_qdq(
            export.onnx_path,
            output_dir / "qdq.onnx",
            mapping,
            scales=scales,
            config=qdq_config,
            calibration_metadata={
                "calibration_manifest_hash": canonical_json_hash(
                    {
                        "module_paths": int8_modules,
                        "batches": self.context.quant_calibration_batches,
                        "frame_manifest_hash": calibration_frame_manifest_hash,
                        "calibration_input_manifest_sha256": calibration_input_identity.get("manifest_sha256", ""),
                        "calibration_seed": calibration_seed,
                        "activation_calibration_backend": calibration_backend,
                        "calibration_backend_cache_sha256": calibration_backend_result.get(
                            "calibration_cache_sha256", ""
                        ),
                    }
                ),
                "calibration_config": CalibrationConfig(
                    frame_count=max(1, self.context.quant_calibration_batches),
                    weight_granularity=qdq_config.weight_granularity,
                ).to_dict(),
                "qdq_config": qdq_config.to_dict(),
                "quantization_group_contract_hash": canonical_json_hash(group_contracts),
                "quantization_group_contracts": group_contracts,
            },
        )
        for merge in qdq.calibration_metadata.get("merge_quantization_audit", []):
            related_groups = {
                str(producer.get("quantization_group", ""))
                for branch in merge.get("input_branches", [])
                for producer in branch.get("nearest_weighted_producers", [])
                if str(producer.get("quantization_group", ""))
            }
            related_groups.update(
                str(layer.get("quantization_group", ""))
                for layer in merge.get("downstream_weighted_layers", [])
                if str(layer.get("quantization_group", ""))
            )
            for group_id in sorted(related_groups):
                contract = group_contracts.get(group_id)
                if contract is None:
                    raise RuntimeError(f"merge_contract_group_missing:{merge.get('merge_op_name')}:{group_id}")
                contract.setdefault("merge_boundaries", []).append(
                    {
                        "merge_op_name": merge.get("merge_op_name"),
                        "merge_op_type": merge.get("merge_op_type"),
                        "input_branches": merge.get("input_branches", []),
                        "downstream_weighted_layers": merge.get("downstream_weighted_layers", []),
                        "merge_scale_policy": merge.get("merge_scale_policy"),
                        "graph_policy": merge.get("policy"),
                    }
                )
                contract["merge_boundary_resolution"] = "resolved_from_canonical_onnx_during_qdq_export"
        qdq.calibration_metadata["quantization_group_contracts"] = group_contracts
        topology_hash = str(qdq.calibration_metadata.get("qdq_topology_hash", ""))
        for contract in group_contracts.values():
            contract["qdq_topology_hash"] = topology_hash
            contract["activation_output_boundary_policy"] = qdq_config.activation_output_boundary_policy
        qdq.calibration_metadata["quantization_group_contract_hash"] = canonical_json_hash(group_contracts)
        _write_json(output_dir / "quantization_group_contracts.json", group_contracts)
        if not (output_dir / "pruned_qdq.onnx").exists():
            shutil.copyfile(output_dir / "qdq.onnx", output_dir / "pruned_qdq.onnx")
        compatibility = make_pointpillar_domain_compatible(output_dir / "qdq.onnx", output_dir / "qdq_trt_compatible.onnx")
        _write_json(output_dir / "onnx_domain_compatibility_report.json", compatibility)
        _write_json(output_dir / "qdq_report.json", qdq.to_dict())
        group_macs = {group.group_id: float(group.baseline_macs) for group in self.context.search_space.quantization_groups}
        total_group_macs = sum(group_macs.values()) or 1.0
        int8_macs_ratio = sum(
            group_macs.get(group_id, 0.0)
            for group_id, precision in realized_group_profile.items()
            if str(precision).upper() == "INT8"
        ) / total_group_macs
        qdq_summary = summarize_qdq_realization(
            requested_group_profile=requested_group_profile,
            realized_group_profile=realized_group_profile,
            realized_canonical_profile=realized_profile,
            qdq_report=qdq.to_dict(),
            int8_macs_ratio=int8_macs_ratio,
        )
        _write_json(output_dir / "qdq_realization_summary.json", qdq_summary)
        legalized_group_profile = {
            str(key): str(value).upper()
            for key, value in dict(
                phenotype.metadata.get("stage1_legalized_group_profile", requested_group_profile)
            ).items()
        }
        return {
            "export": export,
            "origin_map": origin_map,
            "precision_profile": profile,
            "precision_mapping": mapping,
            "realized_precision_profile": realized_profile,
            "realized_group_profile": realized_group_profile,
            "fallback_report": fallback_report,
            "calibration_scales": scales,
            "qdq_realization_summary": qdq_summary,
            "quantization_group_contracts": group_contracts,
            "qdq": qdq,
            "qdq_onnx": str(output_dir / "qdq.onnx"),
            "trt_build_onnx": str(output_dir / "qdq_trt_compatible.onnx"),
            "base_onnx_path": str(export.onnx_path),
            "legalized_precision_profile": legalized_group_profile,
            "calibration_recipe": calibration_identity,
        }

    def _build_engine(self, qdq: dict[str, Any], physical: dict[str, Any], output_dir: Path, *, baseline_precision: str | None = None) -> dict[str, Any]:
        try:
            from quantization.config import TensorRTBuildConfig
        except ImportError:
            from heal_compress.quantization.config import TensorRTBuildConfig
        if baseline_precision:
            build_config = make_baseline_trt_build_config(
                baseline_precision,
                trtexec_path=self.context.tensorrt.trtexec_path,
                plugin_path=self.context.tensorrt.plugin_path,
                shape_profiles=_shape_profiles(),
            )
        else:
            build_config = TensorRTBuildConfig(
                trtexec_path=self.context.tensorrt.trtexec_path,
                plugin_path=self.context.tensorrt.plugin_path,
                shape_profiles=_shape_profiles(),
                enable_fp16=True,
                enable_int8=any(row.realized_request_precision == "int8" for row in qdq["precision_mapping"].entries),
                no_tf32=True,
                precision_constraints="obey",
                skip_inference=True,
                export_layer_info=True,
            )
        engine_path = output_dir / "engine.plan"
        result = build_engine_modelopt(
            qdq_onnx=qdq.get("trt_build_onnx", qdq["qdq_onnx"]),
            engine_path=engine_path,
            precision_mapping=qdq["precision_mapping"],
            build_config=build_config,
            physical_snapshot=physical["snapshot"],
            output_dir=output_dir,
            tensorrt_root=self.context.tensorrt.tensorrt_root,
            conda_env=self.context.tensorrt.conda_env,
            gpu_id=self.context.physical_gpu_id,
        )
        if engine_path.is_file():
            result["engine_path"] = str(engine_path)
            result["engine_hash"] = _file_hash(engine_path)
        merge_realization = _engine_merge_precision_realization(
            output_dir / "engine_layer_info.json",
            qdq["qdq"],
        )
        result["merge_precision_realization"] = merge_realization
        qdq["merge_precision_realization"] = merge_realization
        contracts = qdq.get("quantization_group_contracts", {})
        realized_by_name = {
            str(row.get("merge_op_name", "")): row
            for row in merge_realization.get("merges", [])
        }
        for contract in contracts.values():
            for boundary in contract.get("merge_boundaries", []):
                realized_boundary = realized_by_name.get(str(boundary.get("merge_op_name", "")), {})
                boundary["realized_merge_precision"] = realized_boundary.get(
                    "realized_merge_precision",
                    "not_yet_verified",
                )
                boundary["engine_optimization"] = realized_boundary.get("engine_optimization", "")
        try:
            from quantization.reports.qdq_boundary import write_production_qdq_boundary_reports
        except ImportError:
            from heal_compress.quantization.reports.qdq_boundary import write_production_qdq_boundary_reports
        boundary_report = write_production_qdq_boundary_reports(
            output_dir,
            qdq["qdq"],
            output_dir / "engine_layer_info.json",
        )
        qdq["qdq"].calibration_metadata["weighted_qdq_boundary_audit"] = boundary_report.get("layers", [])
        qdq["qdq"].calibration_metadata["boundary_audit_passed"] = bool(boundary_report.get("passed", False))
        result["qdq_boundary_audit"] = boundary_report
        _write_json(output_dir / "qdq_report.json", qdq["qdq"].to_dict())
        _write_json(output_dir / "quantization_group_contracts.json", contracts)
        _write_json(output_dir / "merge_precision_realization.json", merge_realization)
        _write_json(output_dir / "engine_manifest.json", result)
        if "engine_structure_validation" in result:
            _write_json(output_dir / "engine_structure_validation.json", result["engine_structure_validation"])
        if "precision_realization_validation" in result:
            _write_json(output_dir / "precision_realization_validation.json", result["precision_realization_validation"])
        if baseline_precision and (output_dir / "engine_layer_info.json").is_file():
            baseline_report = validate_baseline_layer_precisions(
                baseline_precision,
                output_dir / "engine_layer_info.json",
                canonical_precision_realization=result.get("precision_realization_validation"),
            )
            result["baseline_precision_validation"] = baseline_report
            _write_json(output_dir / "baseline_precision_validation.json", baseline_report)
            if not baseline_report.get("passed", False):
                result["status"] = baseline_report.get("status", "baseline_precision_validation_failed")
                result["failure_reason"] = ",".join(baseline_report.get("issues", []))
        if result.get("status") == "ok" and not bool(merge_realization.get("passed", False)):
            result["status"] = "merge_precision_realization_failed"
            result["failure_reason"] = ",".join(merge_realization.get("issues", []))
        if result.get("status") == "ok" and not bool(boundary_report.get("passed", False)):
            result["status"] = "qdq_boundary_audit_failed"
            result["failure_reason"] = ",".join(
                issue
                for row in boundary_report.get("issues", [])
                for issue in row.get("issues", [])
            )
        return result

    def _evaluate_engine(self, engine_path: str | Path, output_dir: Path) -> dict[str, Any]:
        from ..integration.runtime_environment import require_gpu_isolation

        require_gpu_isolation(
            self.context.physical_gpu_id,
            report_path=output_dir / "gpu_preflight.json",
        )
        result = evaluate_engine_modelopt(
            engine_path=engine_path,
            checkpoint=self.context.checkpoint_path,
            model_config=self.context.model_config,
            heal_root="/home/lixingfeng/UniAD_examine/HEAL",
            device=self.context.runtime_device,
            output_dir=output_dir,
            tensorrt_root=self.context.tensorrt.tensorrt_root,
            plugin_path=self.context.tensorrt.plugin_path,
            num_frames=self.num_frames,
            warmup_frames=self.warmup_frames,
            fixed_k=29696,
            latency_rounds=self.latency_rounds,
            conda_env=self.context.tensorrt.conda_env,
            eval_manifest_path=self.context.eval_manifest_path,
        )
        require_gpu_isolation(
            self.context.physical_gpu_id,
            report_path=output_dir / "gpu_postflight.json",
        )
        _write_json(output_dir / "evaluation.json", result)
        self._copy_latency(result, output_dir / "latency.csv")
        return result

    @staticmethod
    def _copy_latency(evaluation: dict[str, Any], path: str | Path) -> None:
        rows = list(evaluation.get("latency_rows") or [])
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not rows:
            destination.write_text("", encoding="utf-8")
            return
        fields = sorted({key for row in rows for key in row})
        with destination.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _write_baseline_csv(path: str | Path, rows: dict[str, dict[str, Any]]) -> None:
        fields = [
            "precision baseline",
            "engine status",
            "INT8 group count",
            "INT8 MACs ratio",
            "Q/DQ node count",
            "AP@0.3",
            "AP@0.5",
            "AP@0.7",
            "mAP",
            "forward_mean_ms",
            "forward_p50_ms",
            "forward_p90_ms",
            "forward_p95_ms",
            "forward_p99_ms",
            "evaluated frames",
            "skipped frames",
            "engine hash",
        ]
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for name, row in rows.items():
                qdq = dict(row.get("qdq_realization_summary") or {})
                writer.writerow(
                    {
                        "precision baseline": name,
                        "engine status": row.get("status"),
                        "INT8 group count": qdq.get("realized_int8_group_count", 0),
                        "INT8 MACs ratio": qdq.get("realized_int8_macs_ratio", 0.0),
                        "Q/DQ node count": int(qdq.get("QuantizeLinear_count", 0) or 0) + int(qdq.get("DequantizeLinear_count", 0) or 0),
                        "AP@0.3": row.get("AP@0.3"),
                        "AP@0.5": row.get("AP@0.5"),
                        "AP@0.7": row.get("AP@0.7"),
                        "mAP": row.get("mAP"),
                        "forward_mean_ms": row.get("forward_mean_ms"),
                        "forward_p50_ms": row.get("forward_p50_ms"),
                        "forward_p90_ms": row.get("forward_p90_ms"),
                        "forward_p95_ms": row.get("forward_p95_ms"),
                        "forward_p99_ms": row.get("forward_p99_ms"),
                        "evaluated frames": row.get("num_evaluated_frames"),
                        "skipped frames": row.get("num_skipped_frames"),
                        "engine hash": row.get("engine_hash"),
                    }
                )

    @staticmethod
    def _write_baseline_markdown(path: str | Path, rows: dict[str, dict[str, Any]]) -> None:
        lines = [
            "| baseline | status | mAP | p50 ms | p90 ms | p95 ms | evaluated | skipped | engine hash |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
        for name, row in rows.items():
            lines.append(
                "| {name} | {status} | {map} | {p50} | {p90} | {p95} | {evaluated} | {skipped} | {hash} |".format(
                    name=name,
                    status=row.get("status", ""),
                    map=row.get("mAP", ""),
                    p50=row.get("forward_p50_ms", ""),
                    p90=row.get("forward_p90_ms", ""),
                    p95=row.get("forward_p95_ms", ""),
                    evaluated=row.get("num_evaluated_frames", ""),
                    skipped=row.get("num_skipped_frames", ""),
                    hash=row.get("engine_hash", ""),
                )
            )
        Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
