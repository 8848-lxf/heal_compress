"""Deterministic calibration-manifest contracts for HEAL model families.

The formal V2X-ViT path uses the dynamic-frontend v2 contract. The fixed-K v1
helpers remain only to verify and project historical frozen sample selections;
they are not consumed by the formal TensorRT exporter or runtime.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


V2XVIT_TRAIN200_SCHEMA = "heal_lidar_v2xvit_train200_fixed_k_v1"
V2XVIT_DYNAMIC_TRAIN200_SCHEMA = "heal_lidar_v2xvit_train200_dynamic_frontend_v2"
V2XVIT_TRAIN200_SELECTION_POLICY = "evenly_spaced_valid_train_indices_v1"
V2XVIT_TRAIN200_BASE_SEED = 20260717
V2XVIT_FIXED_K_ALIGNMENT = 256


def evenly_spaced_indices(dataset_size: int, sample_count: int) -> list[int]:
    """Select a stable, ordered subset spanning the complete valid split."""

    size = int(dataset_size)
    count = int(sample_count)
    if size <= 0:
        raise ValueError("calibration_dataset_must_not_be_empty")
    if count <= 0:
        raise ValueError("calibration_sample_count_must_be_positive")
    if count > size:
        raise ValueError(f"calibration_sample_count_exceeds_dataset:{count}>{size}")
    if count == 1:
        return [0]
    denominator = count - 1
    # Integer nearest rounding avoids a NumPy-version-dependent selection.
    indices = [
        (ordinal * (size - 1) + denominator // 2) // denominator
        for ordinal in range(count)
    ]
    if len(indices) != len(set(indices)):
        raise RuntimeError("calibration_selection_contains_duplicate_indices")
    return indices


def sample_seed(base_seed: int, dataset_index: int) -> int:
    """Return the stable uint32 RNG seed owned by one calibration sample."""

    return int((int(base_seed) + int(dataset_index)) % (2**32))


def ceil_to_alignment(value: int, alignment: int = V2XVIT_FIXED_K_ALIGNMENT) -> int:
    value = int(value)
    alignment = int(alignment)
    if value <= 0:
        raise ValueError("fixed_k_source_value_must_be_positive")
    if alignment <= 0:
        raise ValueError("fixed_k_alignment_must_be_positive")
    return ((value + alignment - 1) // alignment) * alignment


def numeric_distribution(values: Iterable[int | float]) -> dict[str, float | int | None]:
    data = np.asarray(list(values), dtype=np.float64)
    if data.size == 0:
        return {
            "min": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
            "mean": None,
        }
    return {
        "min": int(data.min()) if np.all(data == np.floor(data)) else float(data.min()),
        "p50": float(np.percentile(data, 50)),
        "p90": float(np.percentile(data, 90)),
        "p95": float(np.percentile(data, 95)),
        "p99": float(np.percentile(data, 99)),
        "max": int(data.max()) if np.all(data == np.floor(data)) else float(data.max()),
        "mean": float(data.mean()),
    }


def canonical_payload_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def manifest_identity_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Strip volatile provenance before computing the reproducible identity."""

    return {
        key: value
        for key, value in manifest.items()
        if key not in {"manifest_hash", "generated_at", "source_control"}
    }


def finalize_v2xvit_train_manifest(
    manifest: Mapping[str, Any],
    *,
    expected_sample_count: int = 200,
) -> dict[str, Any]:
    """Legacy: validate sample evidence and reproduce a historical fixed-K manifest."""

    result = dict(manifest)
    samples = [dict(row) for row in result.get("samples", [])]
    if len(samples) != int(expected_sample_count):
        raise ValueError(
            f"v2xvit_train_manifest_sample_count_mismatch:{len(samples)}!={expected_sample_count}"
        )
    dataset_indices = [int(row["dataset_index"]) for row in samples]
    frame_ids = [str(row["vehicle_frame_id"]) for row in samples]
    if len(dataset_indices) != len(set(dataset_indices)):
        raise ValueError("v2xvit_train_manifest_duplicate_dataset_index")
    if len(frame_ids) != len(set(frame_ids)):
        raise ValueError("v2xvit_train_manifest_duplicate_frame_id")
    voxel_counts = [int(row["voxel_count"]) for row in samples]
    if any(value <= 0 for value in voxel_counts):
        raise ValueError("v2xvit_train_manifest_nonpositive_voxel_count")

    fixed_contract = dict(result.get("fixed_k_contract", {}))
    alignment = int(fixed_contract.get("alignment", V2XVIT_FIXED_K_ALIGNMENT))
    observed_max = max(voxel_counts)
    fixed_k = ceil_to_alignment(observed_max, alignment)
    for row in samples:
        count = int(row["voxel_count"])
        row["fixed_k"] = fixed_k
        row["padding_voxels"] = fixed_k - count
        row["padding_ratio"] = float((fixed_k - count) / fixed_k)

    result["samples"] = samples
    result["selected_dataset_indices"] = dataset_indices
    result["selected_vehicle_frame_ids"] = frame_ids
    result["sample_count"] = len(samples)
    result["voxel_count_distribution"] = numeric_distribution(voxel_counts)
    result["padding_ratio_distribution"] = numeric_distribution(
        row["padding_ratio"] for row in samples
    )
    result["record_len_distribution"] = dict(
        sorted(Counter(str(int(row["record_len"])) for row in samples).items())
    )
    result["fixed_k_contract"] = {
        **fixed_contract,
        "value": fixed_k,
        "alignment": alignment,
        "observed_max_voxel_count": observed_max,
        "alignment_margin_voxels": fixed_k - observed_max,
        "derivation": f"ceil(max_observed_train200_voxel_count/{alignment})*{alignment}",
        "truncated_sample_count": sum(value > fixed_k for value in voxel_counts),
        "coverage_scope": "only_the_exact_frozen_train200_manifest",
        "full_train_split_upper_bound_claimed": False,
    }
    input_contract = dict(result.get("input_contract", {}))
    max_points = int(input_contract.get("max_points_per_voxel", 32))
    max_agents = int(input_contract.get("max_agents", 2))
    result["input_contract"] = {
        **input_contract,
        "fixed_k": fixed_k,
        "voxel_features_shape": [fixed_k, max_points, 4],
        "voxel_coords_shape": [fixed_k, 4],
        "voxel_num_points_shape": [fixed_k],
        "valid_voxel_mask_shape": [fixed_k],
        "pairwise_t_matrix_shape": [1, max_agents, max_agents, 4, 4],
        "agent_mask_shape": [1, max_agents],
    }
    result["manifest_hash_excluded_fields"] = [
        "generated_at",
        "manifest_hash",
        "source_control",
    ]
    result["manifest_hash"] = canonical_payload_hash(manifest_identity_payload(result))
    validate_v2xvit_train_manifest(result, expected_sample_count=expected_sample_count)
    return result


def validate_v2xvit_train_manifest(
    manifest: Mapping[str, Any],
    *,
    expected_sample_count: int = 200,
) -> None:
    if str(manifest.get("schema_version")) != V2XVIT_TRAIN200_SCHEMA:
        raise ValueError("v2xvit_train_manifest_schema_mismatch")
    if str(manifest.get("family_id")) != "heal_lidar_v2xvit":
        raise ValueError("v2xvit_train_manifest_family_mismatch")
    if str(manifest.get("split")) != "train":
        raise ValueError("v2xvit_calibration_manifest_must_use_train_split")
    samples = list(manifest.get("samples", []))
    if len(samples) != int(expected_sample_count):
        raise ValueError("v2xvit_train_manifest_incomplete")
    fixed_k = int(manifest.get("fixed_k_contract", {}).get("value", 0))
    alignment = int(manifest.get("fixed_k_contract", {}).get("alignment", 0))
    if fixed_k <= 0 or alignment <= 0 or fixed_k % alignment:
        raise ValueError("v2xvit_fixed_k_contract_invalid")
    if any(int(row["voxel_count"]) > fixed_k for row in samples):
        raise ValueError("v2xvit_fixed_k_truncates_frozen_manifest")
    expected_hash = canonical_payload_hash(manifest_identity_payload(manifest))
    if str(manifest.get("manifest_hash")) != expected_hash:
        raise ValueError("v2xvit_train_manifest_hash_mismatch")


def load_v2xvit_train_manifest(path: str | Path) -> dict[str, Any]:
    """Legacy loader for pre-scatter artifacts; formal search uses the dynamic loader."""

    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise RuntimeError(f"v2xvit_frozen_train_manifest_missing:{manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("v2xvit_train_manifest_root_must_be_object")
    validate_v2xvit_train_manifest(payload)
    return payload


def finalize_v2xvit_dynamic_train_manifest(
    manifest: Mapping[str, Any],
    *,
    expected_sample_count: int = 200,
) -> dict[str, Any]:
    """Freeze sample selection without deriving a padded voxel capacity."""

    result = dict(manifest)
    result["schema_version"] = V2XVIT_DYNAMIC_TRAIN200_SCHEMA
    samples = []
    for source in result.get("samples", []):
        row = {
            key: value
            for key, value in dict(source).items()
            if key not in {"fixed_k", "padding_voxels", "padding_ratio"}
        }
        samples.append(row)
    if len(samples) != int(expected_sample_count):
        raise ValueError(
            "v2xvit_dynamic_train_manifest_sample_count_mismatch:"
            f"{len(samples)}!={expected_sample_count}"
        )
    dataset_indices = [int(row["dataset_index"]) for row in samples]
    frame_ids = [str(row["vehicle_frame_id"]) for row in samples]
    voxel_counts = [int(row["voxel_count"]) for row in samples]
    if len(dataset_indices) != len(set(dataset_indices)):
        raise ValueError("v2xvit_dynamic_train_manifest_duplicate_dataset_index")
    if len(frame_ids) != len(set(frame_ids)):
        raise ValueError("v2xvit_dynamic_train_manifest_duplicate_frame_id")
    if any(value <= 0 for value in voxel_counts):
        raise ValueError("v2xvit_dynamic_train_manifest_nonpositive_voxel_count")
    if any("sample_seed" not in row for row in samples):
        raise ValueError("v2xvit_dynamic_train_manifest_sample_seed_missing")

    result.pop("fixed_k_contract", None)
    result.pop("padding_ratio_distribution", None)
    input_contract = dict(result.get("input_contract", {}) or {})
    for key in tuple(input_contract):
        if key == "fixed_k" or key.startswith("voxel_") or key == "valid_voxel_mask_shape":
            input_contract.pop(key, None)
    result["input_contract"] = {
        **input_contract,
        "engine_contract": "heal_post_scatter_dynamic_frontend_v1",
        "point_frontend": "dynamic_gpu_voxelization_pfn_scatter_outside_tensorrt",
        "voxel_capacity": None,
        "padding_policy": "none",
        "overflow_policy": "not_applicable_dynamic_frontend",
    }
    result["purpose"] = "task_loss_and_modelopt_calibration_dynamic_frontend"
    result["samples"] = samples
    result["selected_dataset_indices"] = dataset_indices
    result["selected_vehicle_frame_ids"] = frame_ids
    result["sample_count"] = len(samples)
    result["voxel_count_distribution"] = numeric_distribution(voxel_counts)
    result["record_len_distribution"] = dict(
        sorted(Counter(str(int(row["record_len"])) for row in samples).items())
    )
    result["runtime_max_k_dependency"] = False
    result["manifest_hash_excluded_fields"] = [
        "generated_at",
        "manifest_hash",
        "source_control",
    ]
    result["manifest_hash"] = canonical_payload_hash(manifest_identity_payload(result))
    validate_v2xvit_dynamic_train_manifest(
        result, expected_sample_count=expected_sample_count
    )
    return result


def validate_v2xvit_dynamic_train_manifest(
    manifest: Mapping[str, Any],
    *,
    expected_sample_count: int = 200,
) -> None:
    if str(manifest.get("schema_version")) != V2XVIT_DYNAMIC_TRAIN200_SCHEMA:
        raise ValueError("v2xvit_dynamic_train_manifest_schema_mismatch")
    if str(manifest.get("family_id")) != "heal_lidar_v2xvit":
        raise ValueError("v2xvit_dynamic_train_manifest_family_mismatch")
    if str(manifest.get("split")) != "train":
        raise ValueError("v2xvit_dynamic_calibration_manifest_must_use_train_split")
    samples = list(manifest.get("samples", []))
    if len(samples) != int(expected_sample_count):
        raise ValueError("v2xvit_dynamic_train_manifest_incomplete")
    if "fixed_k_contract" in manifest:
        raise ValueError("v2xvit_dynamic_train_manifest_contains_fixed_k")
    if any("fixed_k" in row for row in samples):
        raise ValueError("v2xvit_dynamic_train_sample_contains_fixed_k")
    contract = dict(manifest.get("input_contract", {}) or {})
    if contract.get("voxel_capacity", "missing") is not None:
        raise ValueError("v2xvit_dynamic_train_manifest_voxel_capacity_not_dynamic")
    if bool(manifest.get("runtime_max_k_dependency", True)):
        raise ValueError("v2xvit_dynamic_train_manifest_runtime_max_k_dependency")
    expected_hash = canonical_payload_hash(manifest_identity_payload(manifest))
    if str(manifest.get("manifest_hash")) != expected_hash:
        raise ValueError("v2xvit_dynamic_train_manifest_hash_mismatch")


def load_v2xvit_dynamic_train_manifest(path: str | Path) -> dict[str, Any]:
    """Load the formal dynamic manifest, projecting a verified legacy list if needed."""

    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise RuntimeError(f"v2xvit_frozen_train_manifest_missing:{manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("v2xvit_train_manifest_root_must_be_object")
    schema = str(payload.get("schema_version", ""))
    if schema == V2XVIT_DYNAMIC_TRAIN200_SCHEMA:
        validate_v2xvit_dynamic_train_manifest(payload)
        return payload
    if schema != V2XVIT_TRAIN200_SCHEMA:
        raise ValueError("v2xvit_dynamic_train_manifest_schema_mismatch")
    validate_v2xvit_train_manifest(payload)
    projected = {
        **payload,
        "source_legacy_manifest": {
            "schema_version": schema,
            "manifest_hash": str(payload["manifest_hash"]),
            "use": "sample_selection_only",
        },
    }
    return finalize_v2xvit_dynamic_train_manifest(projected)


__all__ = [
    "V2XVIT_FIXED_K_ALIGNMENT",
    "V2XVIT_DYNAMIC_TRAIN200_SCHEMA",
    "V2XVIT_TRAIN200_BASE_SEED",
    "V2XVIT_TRAIN200_SCHEMA",
    "V2XVIT_TRAIN200_SELECTION_POLICY",
    "canonical_payload_hash",
    "ceil_to_alignment",
    "evenly_spaced_indices",
    "finalize_v2xvit_train_manifest",
    "finalize_v2xvit_dynamic_train_manifest",
    "load_v2xvit_dynamic_train_manifest",
    "load_v2xvit_train_manifest",
    "manifest_identity_payload",
    "numeric_distribution",
    "sample_seed",
    "validate_v2xvit_train_manifest",
    "validate_v2xvit_dynamic_train_manifest",
]
