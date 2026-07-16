"""Canonical SHA-256 hashing for search candidates and manifests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from .candidate import CandidatePhenotype
from .canonicalization import SearchSpaceSpec


def _plain(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return _plain(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple, set)):
        return [_plain(item) for item in value]
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _plain(value.to_dict())
    return value


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(_plain(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def canonical_json_hash(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def candidate_hash_payload(phenotype: CandidatePhenotype, space: SearchSpaceSpec) -> dict[str, Any]:
    """Return the exact canonical payload used for candidate identity."""

    return {
        "pruned_unit_ids": sorted(phenotype.pruned_unit_ids),
        "domain_width_profile": dict(phenotype.metadata.get("domain_width_profile") or {}),
        "domain_width_expansion_hash": str(
            phenotype.metadata.get("domain_width_expansion_hash", "")
        ),
        "realized_precision_profile": phenotype.realized_precision_profile,
        "pruning_policy_version": phenotype.pruning_policy_version,
        "precision_policy_version": phenotype.precision_policy_version,
        "trace_snapshot_hash": space.trace_snapshot_hash,
        "calibration_manifest_hash": space.calibration_manifest_hash,
        "onnx_export_config_hash": space.onnx_export_config_hash,
        "tensorrt_version": space.tensorrt_version,
        "gpu_compute_capability": space.gpu_compute_capability,
        "builder_flags": space.builder_flags,
        "plugin_hashes": space.plugin_hashes,
    }


def candidate_hash(phenotype: CandidatePhenotype, space: SearchSpaceSpec) -> str:
    return canonical_json_hash(candidate_hash_payload(phenotype, space))


def search_hash(phenotype: CandidatePhenotype, *, trace_hash: str, proxy_version: str, calibration_statistics_version: str) -> str:
    return canonical_json_hash(
        {
            "pruned_unit_ids": sorted(phenotype.pruned_unit_ids),
            "domain_width_profile": dict(phenotype.metadata.get("domain_width_profile") or {}),
            "domain_width_expansion_hash": str(
                phenotype.metadata.get("domain_width_expansion_hash", "")
            ),
            "requested_precision_profile": phenotype.requested_precision_profile,
            "trace_hash": trace_hash,
            "proxy_version": proxy_version,
            "calibration_statistics_version": calibration_statistics_version,
        }
    )


def physical_hash(*, legal_physical_plan: Any, physical_snapshot: Any, checkpoint_hash: str, pruning_policy_version: str) -> str:
    return canonical_json_hash(
        {
            "legal_physical_plan": legal_physical_plan.to_dict() if hasattr(legal_physical_plan, "to_dict") else legal_physical_plan,
            "physical_snapshot": physical_snapshot.to_dict() if hasattr(physical_snapshot, "to_dict") else physical_snapshot,
            "checkpoint_hash": checkpoint_hash,
            "pruning_policy_version": pruning_policy_version,
        }
    )


def deployment_hash(
    *,
    physical_hash_value: str,
    realized_precision_profile: dict[str, str],
    calibration_scale_hash: str,
    onnx_export_config_hash: str,
    tensorrt_version: str,
    gpu_compute_capability: str,
    builder_flags: dict[str, Any],
    optimization_profiles: dict[str, Any],
    plugin_hashes: dict[str, str],
    quantization_contract_hash: str,
) -> str:
    return canonical_json_hash(
        {
            "physical_hash": physical_hash_value,
            "realized_precision_profile": realized_precision_profile,
            "calibration_scale_hash": calibration_scale_hash,
            "onnx_export_config_hash": onnx_export_config_hash,
            "tensorrt_version": tensorrt_version,
            "gpu_compute_capability": gpu_compute_capability,
            "builder_flags": builder_flags,
            "optimization_profiles": optimization_profiles,
            "plugin_hashes": plugin_hashes,
            "quantization_contract_hash": quantization_contract_hash,
        }
    )


def eval_hash(
    *,
    deployment_hash_value: str,
    validation_manifest_hash: str,
    evaluation_config_hash: str,
    postprocess_config: dict[str, Any],
    warmup: int,
    rounds: int,
    latency_metric_definition: str,
) -> str:
    return canonical_json_hash(
        {
            "deployment_hash": deployment_hash_value,
            "validation_manifest_hash": validation_manifest_hash,
            "evaluation_config_hash": evaluation_config_hash,
            "postprocess_config": postprocess_config,
            "warmup": int(warmup),
            "rounds": int(rounds),
            "latency_metric_definition": latency_metric_definition,
        }
    )
