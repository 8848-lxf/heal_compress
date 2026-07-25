"""Strict dependency-bound V2X-ViT train200 calibration provenance."""

from __future__ import annotations

from typing import Any, Mapping

from quantization.types import stable_json_hash


REQUIRED_HASH_FIELDS = (
    "manifest_hash",
    "checkpoint_hash",
    "physical_hash",
    "state_dict_shape_hash",
    "precision_map_hash",
    "onnx_hash",
    "calibration_algorithm_config_hash",
    "cache_hash",
    "scale_hash",
)


def calibration_dependency_key(payload: Mapping[str, Any]) -> str:
    """Hash every dependency that invalidates activation calibration reuse."""

    required = {
        key: str(payload.get(key, ""))
        for key in REQUIRED_HASH_FIELDS[:-2]
    }
    missing = [key for key, value in required.items() if not value]
    if missing:
        raise RuntimeError(f"v2xvit_train200_dependency_hash_missing:{missing}")
    return stable_json_hash(required)


def build_train200_contract(
    *,
    manifest_hash: str,
    checkpoint_hash: str,
    physical_hash: str,
    state_dict_shape_hash: str,
    precision_map_hash: str,
    onnx_hash: str,
    calibration_algorithm_config_hash: str,
    cache_hash: str,
    scale_hash: str,
    processed_frames: int,
    skipped_frames: int,
    algorithm: str = "EntropyCalibration2",
    dataset_split: str = "train",
) -> dict[str, Any]:
    payload = {
        "schema_version": "v2xvit-train200-dependency-contract-v1",
        "algorithm": str(algorithm),
        "dataset_split": str(dataset_split),
        "requested_frames": 200,
        "processed_frames": int(processed_frames),
        "skipped_frames": int(skipped_frames),
        "manifest_hash": str(manifest_hash),
        "checkpoint_hash": str(checkpoint_hash),
        "physical_hash": str(physical_hash),
        "state_dict_shape_hash": str(state_dict_shape_hash),
        "precision_map_hash": str(precision_map_hash),
        "onnx_hash": str(onnx_hash),
        "calibration_algorithm_config_hash": str(calibration_algorithm_config_hash),
        "cache_hash": str(cache_hash),
        "scale_hash": str(scale_hash),
    }
    payload["dependency_key"] = calibration_dependency_key(payload)
    validate_train200_contract(payload)
    payload["contract_hash"] = stable_json_hash(payload)
    return payload


def validate_train200_contract(payload: Mapping[str, Any]) -> None:
    missing = [key for key in REQUIRED_HASH_FIELDS if not str(payload.get(key, ""))]
    if missing:
        raise RuntimeError(f"v2xvit_train200_contract_hash_missing:{missing}")
    if str(payload.get("algorithm")) not in {
        "EntropyCalibration2",
        "ModelOptEntropyKL2048To128",
    }:
        raise RuntimeError(f"v2xvit_train200_algorithm_invalid:{payload.get('algorithm')}")
    if payload.get("dataset_split") != "train":
        raise RuntimeError("v2xvit_train200_split_must_be_train")
    if int(payload.get("requested_frames", -1)) != 200:
        raise RuntimeError("v2xvit_train200_requested_count_mismatch")
    if int(payload.get("processed_frames", -1)) != 200:
        raise RuntimeError("v2xvit_train200_processed_count_mismatch")
    if int(payload.get("skipped_frames", -1)) != 0:
        raise RuntimeError("v2xvit_train200_skipped_frames_nonzero")
    expected = calibration_dependency_key(payload)
    if payload.get("dependency_key") != expected:
        raise RuntimeError("v2xvit_train200_dependency_key_mismatch")


def calibration_reusable(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    """Reuse is legal only for an exact dependency key and valid contracts."""

    validate_train200_contract(left)
    validate_train200_contract(right)
    return str(left["dependency_key"]) == str(right["dependency_key"])
