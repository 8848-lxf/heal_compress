"""Production deployment contracts for searched HEAL V2X-ViT candidates.

This module is intentionally model-family scoped.  It reuses the audited
generic Q/DQ and TensorRT primitives without changing the accepted
LiDAR-pyramid deployment path.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import random
import re
from typing import Any

import numpy as np
import torch

from quantization.export.origin_mapping import apply_canonical_node_names, build_onnx_origin_map
from quantization.precision.activation_boundary import resolve_activation_output_boundary
from quantization.types import CanonicalPrecisionEntry, CanonicalPrecisionMappingResult, stable_json_hash


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_physical_structure_snapshot_v2(
    model: torch.nn.Module,
    *,
    model_family: str = "heal_lidar_v2xvit",
) -> dict[str, Any]:
    """Describe every parameterized module in the generic structure schema."""

    weighted = (
        torch.nn.Conv1d,
        torch.nn.Conv2d,
        torch.nn.Conv3d,
        torch.nn.ConvTranspose1d,
        torch.nn.ConvTranspose2d,
        torch.nn.ConvTranspose3d,
        torch.nn.Linear,
    )
    modules = []
    for name, module in model.named_modules():
        if not name or not isinstance(module, weighted):
            continue
        modules.append(
            {
                "canonical_module_name": name,
                "module_path": name,
                "module_type": type(module).__name__,
                "weight_shape": [int(value) for value in module.weight.shape],
                "weight_dtype": str(module.weight.dtype),
                "groups": int(getattr(module, "groups", 1)),
                "parameter_count": sum(
                    int(parameter.numel()) for parameter in module.parameters(recurse=False)
                ),
            }
        )
    payload = {
        "snapshot_schema_version": "physical-structure-snapshot-v2",
        "schema_version": "physical-structure-snapshot-v2",
        "model_family": str(model_family),
        "modules": modules,
        "parameter_count": sum(int(parameter.numel()) for parameter in model.parameters()),
    }
    payload["snapshot_hash"] = stable_json_hash(payload)
    return payload


def canonicalize_v2xvit_onnx(
    onnx_path: str | Path,
    module_calls: Sequence[Mapping[str, Any] | Any],
    *,
    output_path: str | Path | None = None,
) -> Any:
    """Build the exact call map and apply stable canonical node names."""

    origin_map = build_onnx_origin_map(onnx_path, module_calls)
    apply_canonical_node_names(
        onnx_path,
        origin_map,
        output_path=output_path,
        allow_custom_ops=True,
    )
    return origin_map


def _v2xvit_fp16_merge_nodes(onnx_path: str | Path) -> dict[str, str]:
    """Return only semantic feature merges, never shape/bias arithmetic.

    The export wrapper has one three-scale BEV Concat, two residual Adds per
    transformer stage, one FFN residual Add per stage, and four explicit
    three-window SplitAttn sums per stage.  Their stable exporter names are
    verified against op types before they enter the deployment signature.
    """

    import onnx

    model = onnx.load(str(onnx_path), load_external_data=False)
    nodes = {str(node.name): str(node.op_type) for node in model.graph.node}
    expected = {"/Concat": "Concat"}
    for stage in range(3):
        expected[f"/layers.{stage}.0/Add"] = "Add"
        expected[f"/layers.{stage}.0/Add_1"] = "Add"
        expected[f"/Add_{stage + 2}"] = "Add"
        prefix = f"/layers.{stage}.0/layers.0.1/fn/split_attn"
        for suffix in ("Add", "Add_1", "Add_2", "Add_3"):
            expected[f"{prefix}/{suffix}"] = "Add"
    missing = {
        name: {"expected": op_type, "actual": nodes.get(name, "missing")}
        for name, op_type in expected.items()
        if nodes.get(name) != op_type
    }
    if missing:
        raise RuntimeError(f"v2xvit_semantic_merge_nodes_missing:{missing}")
    return {name: "fp16" for name in sorted(expected)}


def build_v2xvit_precision_mapping(
    origin_map: Any,
    module_precision_profile: Mapping[str, str],
    *,
    canonical_onnx_path: str | Path,
    profile_id: str,
) -> CanonicalPrecisionMappingResult:
    """Expand module genes to every realized weighted ONNX call.

    INT8 compute is explicitly closed to an FP16 output.  Requantization is
    owned by the next weighted input, not the raw Conv output.  All semantic
    residual/concat merges are explicitly FP16.
    """

    profile = {str(key): str(value).lower() for key, value in module_precision_profile.items()}
    origin_modules = {str(row.module_path) for row in origin_map.entries}
    missing = sorted(origin_modules - set(profile))
    unknown = sorted(set(profile) - origin_modules)
    if missing or unknown:
        raise RuntimeError(f"v2xvit_precision_profile_origin_mismatch:missing={missing}:unknown={unknown}")
    invalid = {key: value for key, value in profile.items() if value not in {"fp32", "fp16", "int8"}}
    if invalid:
        raise RuntimeError(f"v2xvit_precision_profile_invalid:{invalid}")
    entries = []
    for origin in sorted(origin_map.entries, key=lambda row: (row.call_index, row.graph_index)):
        precision = profile[str(origin.module_path)]
        entries.append(
            CanonicalPrecisionEntry(
                module_path=str(origin.module_path),
                canonical_node_name=str(origin.canonical_node_name),
                original_node_name=str(origin.original_node_name),
                weight_initializer=str(origin.weight_initializer),
                onnx_op_type=str(origin.onnx_op_type),
                call_index=int(origin.call_index),
                precision_group=f"v2xvit_qg::module::{origin.module_path}",
                requested_precision=precision,
                realized_request_precision=precision,
                realized_output_precision="fp16" if precision in {"fp16", "int8"} else "fp32",
            )
        )
    # Parameter-free affine-grid MatMuls remain canonically named in the ONNX
    # origin audit, but are deliberately not precision genes and therefore do
    # not belong in the weighted structure/precision checker.  One functional
    # group expands to several real MatMul members; representing it as one
    # weighted entry would create a false ambiguous-layer failure.
    return CanonicalPrecisionMappingResult(
        entries=entries,
        profile_id=str(profile_id),
        profile_hash=stable_json_hash(profile),
        origin_map_hash=str(origin_map.origin_map_hash),
        policy_version="heal-v2xvit-explicit-qdq-strong-type-v1",
        auxiliary_layer_precisions=_v2xvit_fp16_merge_nodes(canonical_onnx_path),
        auxiliary_layer_output_types={
            name: "fp16" for name in _v2xvit_fp16_merge_nodes(canonical_onnx_path)
        },
    )


def _first_tensor(value: Any) -> torch.Tensor | None:
    if torch.is_tensor(value):
        return value
    if isinstance(value, Mapping):
        for item in value.values():
            found = _first_tensor(item)
            if found is not None:
                return found
    if isinstance(value, (tuple, list)):
        for item in value:
            found = _first_tensor(item)
            if found is not None:
                return found
    return None


def _semantic_output_module_path(model: torch.nn.Module, module_path: str) -> str:
    modules = dict(model.named_modules())
    if "." not in module_path:
        return module_path
    parent_path, leaf = module_path.rsplit(".", 1)
    parent = modules.get(parent_path)
    if isinstance(parent, torch.nn.Sequential) and leaf.isdigit():
        start = int(leaf)
        selected = module_path
        for index in range(start + 1, len(parent)):
            child = parent[index]
            if isinstance(child, (torch.nn.modules.batchnorm._BatchNorm, torch.nn.ReLU)):
                selected = f"{parent_path}.{index}"
                continue
            break
        return selected
    return module_path


def _entropy_threshold(histogram: torch.Tensor, absolute_maximum: float) -> tuple[float, dict[str, Any]]:
    from modelopt.torch.quantization.calib.histogram import _compute_amax_entropy

    values = histogram.detach().cpu().numpy().astype(np.int64)
    edges = np.linspace(0.0, absolute_maximum, values.size + 1, dtype=np.float64)
    threshold = float(
        _compute_amax_entropy(
            values.copy(), edges, num_bits=8, unsigned=False, stride=1, start_bin=128
        ).item()
    )
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise RuntimeError(f"v2xvit_entropy_threshold_invalid:{threshold}")
    selected = min(max(int(round(threshold / absolute_maximum * values.size)), 128), values.size)
    return threshold, {
        "method": "entropy",
        "implementation": "modelopt.torch.quantization.calib.histogram._compute_amax_entropy",
        "absolute_maximum": absolute_maximum,
        "clipping_threshold": threshold,
        "selected_bin": selected,
        "clipped_fraction": float(values[selected:].sum() / max(values.sum(), 1)),
    }


def _sanitize_weight_channel_amax(
    channel_amax: np.ndarray, *, module_path: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return positive per-channel amax values and audited floored indices."""

    values = np.asarray(channel_amax)
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise RuntimeError(f"v2xvit_entropy_weight_channel_invalid:{module_path}")
    zero_indices = np.flatnonzero(values == 0.0).astype(np.int64)
    floor_indices = np.flatnonzero(values < 1.0e-8).astype(np.int64)
    safe = values.astype(np.float32, copy=True)
    safe[floor_indices] = np.float32(1.0e-8)
    return safe, floor_indices, zero_indices


def _validate_calibration_observation_count(
    module_path: str, *, input_count: int, output_count: int, frame_count: int
) -> int:
    """Validate deterministic reused-module observations per processed frame."""

    if (
        input_count <= 0
        or input_count != output_count
        or input_count % frame_count != 0
    ):
        raise RuntimeError(
            f"v2xvit_entropy_observation_count:{module_path}:"
            f"{input_count}:{output_count}:{frame_count}"
        )
    return input_count // frame_count


def collect_v2xvit_train200_entropy_scales(
    *,
    bundle: Any,
    manifest: Mapping[str, Any],
    mapping: CanonicalPrecisionMappingResult,
    canonical_onnx_path: str | Path,
    device: torch.device,
    histogram_bins: int = 2048,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Stream the frozen train200 manifest twice without retaining batches."""

    from onnx import numpy_helper
    import onnx
    from opencood.data_utils.datasets import build_dataset
    from opencood.hypes_yaml import yaml_utils
    from search.integration.data_provider import move_batch_to_device

    int8_entries = [row for row in mapping.entries if row.realized_request_precision == "int8"]
    module_paths = sorted({row.module_path for row in int8_entries if row.weight_initializer})
    if not module_paths:
        return {}, {"frame_count": 0, "reason": "candidate_has_no_int8_layers"}
    entries_by_module: dict[str, list[Any]] = {}
    for entry in int8_entries:
        if entry.weight_initializer:
            entries_by_module.setdefault(entry.module_path, []).append(entry)
    modules = dict(bundle.model.named_modules())
    missing = sorted(set(module_paths) - set(modules))
    if missing:
        raise RuntimeError(f"v2xvit_entropy_modules_missing:{missing}")

    hypes = yaml_utils.load_yaml(str(bundle.config_path))
    hypes = bundle.adapter._absolutize_dataset_paths(hypes)
    dataset = build_dataset(hypes, visualize=False, train=True)
    rows = [dict(row) for row in manifest["samples"]]
    if len(rows) != 200:
        raise RuntimeError(f"v2xvit_entropy_manifest_not_train200:{len(rows)}")
    state = {
        name: {
            "input_amax": None,
            "output_amax": None,
            "input_hist": None,
            "output_hist": None,
            "input_count": 0,
            "output_count": 0,
            "output_module_path": _semantic_output_module_path(bundle.model, name),
        }
        for name in module_paths
    }
    phase = {"name": "amax"}
    handles = []

    def observe(name: str, role: str, tensor: torch.Tensor) -> None:
        value = tensor.detach().float().abs()
        row = state[name]
        if phase["name"] == "amax":
            maximum = value.amax()
            key = f"{role}_amax"
            row[key] = maximum if row[key] is None else torch.maximum(row[key], maximum)
            row[f"{role}_count"] += 1
            return
        maximum = float(row[f"{role}_amax"].item())
        histogram = torch.histc(value, bins=int(histogram_bins), min=0.0, max=maximum)
        key = f"{role}_hist"
        row[key] = histogram if row[key] is None else row[key] + histogram

    for name in module_paths:
        def pre_hook(_module: Any, inputs: tuple[Any, ...], module_path: str = name) -> None:
            tensor = _first_tensor(inputs)
            if tensor is None:
                raise RuntimeError(f"v2xvit_entropy_input_missing:{module_path}")
            observe(module_path, "input", tensor)

        def output_hook(_module: Any, _inputs: tuple[Any, ...], output: Any, module_path: str = name) -> None:
            tensor = _first_tensor(output)
            if tensor is None:
                raise RuntimeError(f"v2xvit_entropy_output_missing:{module_path}")
            observe(module_path, "output", tensor)

        handles.append(modules[name].register_forward_pre_hook(pre_hook))
        handles.append(modules[state[name]["output_module_path"]].register_forward_hook(output_hook))

    evidence = []
    bundle.model.eval()
    try:
        with torch.inference_mode():
            for pass_name in ("amax", "histogram"):
                phase["name"] = pass_name
                for row in rows:
                    seed = int(row["sample_seed"])
                    random.seed(seed)
                    np.random.seed(seed % (2**32))
                    torch.manual_seed(seed)
                    item = dataset[int(row["dataset_index"])]
                    batch = dataset.collate_batch_train([item])
                    if batch is None:
                        raise RuntimeError(f"v2xvit_entropy_empty_batch:{row['dataset_index']}")
                    observed_k = int(batch["ego"]["inputs_m1"]["voxel_features"].shape[0])
                    if observed_k != int(row["voxel_count"]):
                        raise RuntimeError(
                            f"v2xvit_entropy_manifest_k_mismatch:{row['dataset_index']}:"
                            f"{observed_k}!={row['voxel_count']}"
                        )
                    batch = move_batch_to_device(batch, device)
                    bundle.adapter.forward_for_task(bundle.model, batch)
                    if pass_name == "amax":
                        evidence.append(
                            {
                                "ordinal": int(row["ordinal"]),
                                "dataset_index": int(row["dataset_index"]),
                                "vehicle_frame_id": str(row["vehicle_frame_id"]),
                                "voxel_count": observed_k,
                                "sample_seed": seed,
                            }
                        )
                    del batch
    finally:
        for handle in handles:
            handle.remove()

    onnx_model = onnx.load(str(canonical_onnx_path), load_external_data=False)
    initializers = {
        str(value.name): numpy_helper.to_array(value) for value in onnx_model.graph.initializer
    }
    nodes = {str(node.name): node for node in onnx_model.graph.node}
    scales = {}
    threshold_audit = {}
    module_calls_per_frame = {}
    for name in module_paths:
        row = state[name]
        module_calls_per_frame[name] = _validate_calibration_observation_count(
            name,
            input_count=int(row["input_count"]),
            output_count=int(row["output_count"]),
            frame_count=len(rows),
        )
        input_threshold, input_audit = _entropy_threshold(
            row["input_hist"], float(row["input_amax"].item())
        )
        output_threshold, output_audit = _entropy_threshold(
            row["output_hist"], float(row["output_amax"].item())
        )
        entry = entries_by_module[name][0]
        node = nodes.get(entry.canonical_node_name)
        weight = initializers.get(entry.weight_initializer)
        if node is None or weight is None:
            raise RuntimeError(f"v2xvit_entropy_onnx_mapping_missing:{name}")
        if entry.onnx_op_type == "Conv":
            weight_axis = 0
        elif entry.onnx_op_type == "ConvTranspose":
            weight_axis = 1
        elif entry.onnx_op_type == "MatMul":
            weight_axis = 1
        elif entry.onnx_op_type == "Gemm":
            trans_b = next((int(attr.i) for attr in node.attribute if attr.name == "transB"), 0)
            weight_axis = 0 if trans_b else 1
        else:
            raise RuntimeError(f"v2xvit_entropy_weight_axis_unsupported:{name}:{entry.onnx_op_type}")
        reduce_axes = tuple(index for index in range(weight.ndim) if index != weight_axis)
        channel_amax = np.max(np.abs(weight), axis=reduce_axes)
        (
            safe_channel_amax,
            floored_channel_indices,
            zero_channel_indices,
        ) = _sanitize_weight_channel_amax(channel_amax, module_path=name)
        # A physically all-zero output channel is represented exactly by any
        # positive Q/DQ scale.  ONNX forbids a zero scale, so use a deterministic
        # floor only for those exact-zero channels and retain explicit evidence.
        boundary = resolve_activation_output_boundary(onnx_model, entry.canonical_node_name)
        scales[name] = {
            "activation_input_scale": input_threshold / 127.0,
            "activation_output_scale": output_threshold / 127.0,
            "weight_scale": (safe_channel_amax / 127.0).astype(np.float32).tolist(),
            "weight_axis": weight_axis,
            "weight_granularity": "per_channel",
            "weight_scale_shape": [int(channel_amax.size)],
            "activation_input_tensor": str(node.input[0]),
            "activation_output_tensor": str(boundary["boundary_output_tensor"]),
            "activation_scale_source": "frozen_train200_modelopt_entropy_KL_2048_to_128",
            "activation_input_observer_module": name,
            "activation_output_observer_module": row["output_module_path"],
            "insert_activation_output_qdq": False,
            "output_qdq_policy": "next_weighted_input_owns_requantization_after_fp16_output",
            "weight_scale_source": "final_canonical_onnx_initializer",
            "zero_weight_channel_indices": zero_channel_indices.tolist(),
            "zero_weight_channel_count": int(zero_channel_indices.size),
            "floored_weight_channel_indices": floored_channel_indices.tolist(),
            "floored_weight_channel_count": int(floored_channel_indices.size),
            "weight_channel_scale_floor_policy": "amax_below_1e-8_to_1e-8_before_div127",
        }
        threshold_audit[name] = {
            "input": input_audit,
            "output": output_audit,
            "resolved_output_boundary": boundary,
            "zero_weight_channel_indices": zero_channel_indices.tolist(),
            "zero_weight_channel_count": int(zero_channel_indices.size),
            "floored_weight_channel_indices": floored_channel_indices.tolist(),
            "floored_weight_channel_count": int(floored_channel_indices.size),
        }
    metadata = {
        "schema_version": "heal-v2xvit-train200-entropy-calibration-v1",
        "manifest_hash": str(manifest["manifest_hash"]),
        "frame_count": 200,
        "passes": 2,
        "histogram_bins": int(histogram_bins),
        "module_count": len(module_paths),
        "module_paths": module_paths,
        "module_calls_per_frame": module_calls_per_frame,
        "sample_evidence": evidence,
        "thresholds": threshold_audit,
        "observer_q_input_exact_match": True,
        "activation_output_qdq_inserted": False,
        "weight_granularity": "per_channel",
    }
    metadata["calibration_hash"] = stable_json_hash({"metadata": metadata, "scales": scales})
    return scales, metadata


def load_searched_candidate_profile(
    artifact_path: str | Path,
    *,
    target: float = 0.25,
    candidate_hash: str = "",
) -> dict[str, Any]:
    """Load a greedy budget winner or one exact GA archive member."""

    source = Path(artifact_path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    key = f"{float(target):.9f}"
    if "candidates" in payload:
        rows = [dict(row) for row in payload.get("candidates", [])]
        matches = [row for row in rows if str(row.get("candidate_hash", "")) == str(candidate_hash)]
        if len(matches) != 1:
            raise RuntimeError(
                f"v2xvit_ga_candidate_hash_not_unique:{source}:{candidate_hash}:{len(matches)}"
            )
        selected = matches[0]
        candidate = dict(selected.get("genotype") or {})
        metrics = {
            key: value
            for key, value in selected.items()
            if key not in {"genotype", "domain_width_profile", "precision_counts"}
        }
        algorithm = "ga"
        identity = str(selected["candidate_hash"])
    else:
        section = payload.get("formal_exploration", {})
        candidate = dict(section.get("budget_candidates", {}).get(key) or {})
        metrics = dict(section.get("budget_metrics", {}).get(key) or {})
        if not candidate or not metrics:
            raise RuntimeError(f"v2xvit_searched_candidate_missing:{source}:{key}")
        algorithm = "greedy"
        identity = stable_json_hash(candidate)
    group_profile = {
        str(name): str(value).upper()
        for name, value in dict(candidate.get("precision_genes") or {}).items()
    }
    module_profile = {
        name.removeprefix("v2xvit_qg::module::"): value
        for name, value in group_profile.items()
    }
    widths = {str(name): int(value) for name, value in dict(candidate.get("pruning_width_genes") or {}).items()}
    return {
        "source_artifact": str(source.resolve()),
        "source_artifact_sha256": file_sha256(source),
        "target": float(target),
        "candidate_genotype": candidate,
        "candidate_metrics": metrics,
        "module_precision_profile": module_profile,
        "domain_width_profile": widths,
        "candidate_identity": identity,
        "search_algorithm": algorithm,
    }


def load_v2xvit_pruning_domains(search_space_path: str | Path) -> list[Any]:
    """Replay the immutable domain rankings serialized by the search run."""

    from search.pruning_space.local_domains import LocalPruningDomain

    source = Path(search_space_path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    domains = []
    for raw in payload.get("pruning_domains", []):
        row = dict(raw)
        row.pop("width_semantics", None)
        for key in ("ordered_unit_ids", "legal_widths"):
            row[key] = tuple(row.get(key, ()))
        row["width_to_pruned_unit_ids"] = {
            int(width): tuple(values)
            for width, values in dict(row.get("width_to_pruned_unit_ids") or {}).items()
        }
        row["unit_root_indices"] = {
            str(unit): tuple(int(value) for value in values)
            for unit, values in dict(row.get("unit_root_indices") or {}).items()
        }
        row["ordered_unit_ids_by_group"] = {
            int(group): tuple(values)
            for group, values in dict(row.get("ordered_unit_ids_by_group") or {}).items()
        }
        row["group_local_indices"] = {
            int(group): {str(unit): int(index) for unit, index in values.items()}
            for group, values in dict(row.get("group_local_indices") or {}).items()
        }
        for key in ("group_keep_maps", "group_prune_maps"):
            row[key] = {
                int(width): {
                    int(group): [int(value) for value in values]
                    for group, values in groups.items()
                }
                for width, groups in dict(row.get(key) or {}).items()
            }
        domains.append(LocalPruningDomain(**row))
    if not domains:
        raise RuntimeError(f"v2xvit_serialized_pruning_domains_empty:{source}")
    return domains


__all__ = [
    "build_physical_structure_snapshot_v2",
    "build_v2xvit_precision_mapping",
    "canonicalize_v2xvit_onnx",
    "collect_v2xvit_train200_entropy_scales",
    "file_sha256",
    "load_searched_candidate_profile",
    "load_v2xvit_pruning_domains",
]
