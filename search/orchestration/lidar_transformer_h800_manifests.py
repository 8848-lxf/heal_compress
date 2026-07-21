"""Freeze calibration/evaluation manifests and model-specific fixed-K values."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import DataLoader

from search.integration.data_provider import write_eval_manifest
from search.model_family.calibration_manifest import ceil_to_alignment, evenly_spaced_indices, sample_seed


MODEL_CONFIGS = {
    "lidar_cobevt": "/home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_cobevt/config.yaml",
    "lidar_v2xvit": "/home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_v2xvit/config.yaml",
}
BASE_SEED = 20260721


def _hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _absolute_hypes(config: str) -> tuple[dict[str, Any], Any]:
    from adapters.heal_lidar_adapter import HEALLiDARAdapter
    from opencood.hypes_yaml import yaml_utils

    adapter = HEALLiDARAdapter(
        heal_repo="/home/lixingfeng/UniAD_examine/HEAL",
        config={"model": {"hypes_yaml": config}},
    )
    hypes = yaml_utils.load_yaml(config)
    return adapter._absolutize_dataset_paths(hypes), adapter


def _frame_ids(path: str) -> list[str]:
    result = [str(value) for value in json.loads(Path(path).read_text(encoding="utf-8"))]
    if len(result) != len(set(result)):
        raise RuntimeError(f"transformer_manifest_duplicate_frame_id:{path}")
    return result


def _voxel_count(batch: Any) -> int:
    if batch is None:
        raise RuntimeError("transformer_manifest_empty_batch")
    ego = batch["ego"]
    return int(ego["inputs_m1"]["voxel_features"].shape[0])


def _calibration_rows(hypes: dict[str, Any], count: int = 200) -> tuple[list[dict[str, Any]], int]:
    from opencood.data_utils.datasets import build_dataset

    dataset = build_dataset(hypes, visualize=False, train=True)
    ids = _frame_ids(hypes["root_dir"])
    indices = evenly_spaced_indices(len(dataset), count)
    rows: list[dict[str, Any]] = []
    for ordinal, index in enumerate(indices):
        seed = sample_seed(BASE_SEED, index)
        random.seed(seed)
        np.random.seed(seed % (2**32))
        import torch

        torch.manual_seed(seed)
        item = dataset[index]
        batch = dataset.collate_batch_train([item])
        rows.append(
            {
                "ordinal": ordinal,
                "dataset_index": int(index),
                "frame_id": ids[index],
                "sample_seed": seed,
                "voxel_count": _voxel_count(batch),
                "record_len": int(batch["ego"]["record_len"][0]),
            }
        )
    return rows, len(dataset)


def _validation_rows(hypes: dict[str, Any], workers: int) -> tuple[list[dict[str, Any]], int]:
    from opencood.data_utils.datasets import build_dataset

    dataset = build_dataset(hypes, visualize=False, train=False)
    ids = _frame_ids(hypes["validate_dir"])
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(workers),
        collate_fn=dataset.collate_batch_test,
        persistent_workers=bool(workers),
        prefetch_factor=2 if workers else None,
    )
    rows = []
    for index, batch in enumerate(loader):
        rows.append(
            {
                "ordinal": index,
                "dataset_index": index,
                "frame_id": ids[index],
                "voxel_count": _voxel_count(batch),
                "record_len": int(batch["ego"]["record_len"][0]),
            }
        )
    if len(rows) != len(ids):
        raise RuntimeError(f"transformer_validation_scan_incomplete:{len(rows)}:{len(ids)}")
    return rows, len(dataset)


def _calibration_manifest(model: str, rows: list[dict[str, Any]], *, sample_count: int, fixed_k: int, dataset_size: int) -> dict[str, Any]:
    selected = rows if sample_count == len(rows) else [rows[index] for index in evenly_spaced_indices(len(rows), sample_count)]
    payload: dict[str, Any] = {
        "schema_version": "h800-transformer-calibration-manifest-v1",
        "model_family": model,
        "split": "train",
        "selection_policy": "evenly_spaced_valid_train_indices_v1",
        "base_seed": BASE_SEED,
        "dataset_size": dataset_size,
        "sample_count": len(selected),
        "samples": selected,
        "fixed_k": fixed_k,
        "max_cav": 2,
        "workers": 8,
    }
    payload["manifest_hash"] = _hash(payload)
    return payload


def run_model(model: str, output_root: Path, workers: int) -> dict[str, Any]:
    hypes, _ = _absolute_hypes(MODEL_CONFIGS[model])
    calibration_rows, train_size = _calibration_rows(hypes, 200)
    validation_rows, validation_size = _validation_rows(hypes, workers)
    max_train = max(row["voxel_count"] for row in calibration_rows)
    max_validation = max(row["voxel_count"] for row in validation_rows)
    fixed_k = ceil_to_alignment(max(max_train, max_validation), 256)
    destination = output_root / "evaluation" / "manifests" / model
    destination.mkdir(parents=True, exist_ok=True)
    calibration200 = _calibration_manifest(
        model, calibration_rows, sample_count=200, fixed_k=fixed_k, dataset_size=train_size
    )
    calibration50 = _calibration_manifest(
        model, calibration_rows, sample_count=50, fixed_k=fixed_k, dataset_size=train_size
    )
    _write(destination / "calibration200.json", calibration200)
    _write(destination / "calibration50.json", calibration50)
    _write(destination / "full1789_voxel_scan.json", validation_rows)
    ids = [row["frame_id"] for row in validation_rows]
    evals = {}
    for name, frames in (("smoke10", 10), ("fixed50", 50), ("fixed500", 500), ("full1789", len(ids))):
        manifest = write_eval_manifest(
            destination / f"{name}.json",
            num_frames=frames,
            warmup_frames=200,
            split="val",
            available_frame_ids=ids,
            reset_after_warmup=True,
            evaluation_offset=0,
        )
        evals[name] = manifest.to_dict()
    contract: dict[str, Any] = {
        "schema_version": "h800-transformer-fixed-k-contract-v1",
        "model_family": model,
        "fixed_k": fixed_k,
        "alignment": 256,
        "train200_max": max_train,
        "full_validation_max": max_validation,
        "full_validation_count": len(validation_rows),
        "overflow_count": sum(row["voxel_count"] > fixed_k for row in validation_rows),
        "max_cav": 2,
        "calibration50_hash": calibration50["manifest_hash"],
        "calibration200_hash": calibration200["manifest_hash"],
        "evaluation_manifests": evals,
        "dataset": {
            "root_dir": str(hypes["root_dir"]),
            "validate_dir": str(hypes["validate_dir"]),
            "train_size": train_size,
            "validation_size": validation_size,
        },
    }
    contract["contract_hash"] = _hash(contract)
    _write(destination / "fixed_k_contract.json", contract)
    return contract


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--model", choices=tuple(MODEL_CONFIGS) + ("all",), default="all")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    models = tuple(MODEL_CONFIGS) if args.model == "all" else (args.model,)
    results = [run_model(model, Path(args.output_root).resolve(), args.workers) for model in models]
    _write(Path(args.output_root).resolve() / "evaluation" / "dataset_manifest.json", results)
    print(json.dumps(results, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
