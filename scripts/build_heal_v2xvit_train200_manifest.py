#!/usr/bin/env python3
"""Build and freeze the real HEAL V2X-ViT train200 fixed-K manifest."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adapters.heal_lidar_adapter import HEALLiDARAdapter
from search.model_family.calibration_manifest import (
    V2XVIT_FIXED_K_ALIGNMENT,
    V2XVIT_TRAIN200_BASE_SEED,
    V2XVIT_TRAIN200_SCHEMA,
    V2XVIT_TRAIN200_SELECTION_POLICY,
    evenly_spaced_indices,
    finalize_v2xvit_train_manifest,
    sample_seed,
)
from search.model_family.model_provider import sha256_file


DEFAULT_CONFIG = Path(
    "../../Auto_Search/original_models/dairv2s/"
    "LiDAROnly/lidar_v2xvit/config.yaml"
)
DEFAULT_CHECKPOINT = DEFAULT_CONFIG.parent / "net_epoch_bestval_at27.pth"
DEFAULT_HEAL_ROOT = Path("../../HEAL")


def _git_value(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _set_sample_rng(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    torch.manual_seed(int(seed))


def _infrastructure_frame_id(frame_info: dict[str, Any]) -> str:
    return Path(str(frame_info["infrastructure_image_path"])).stem


def _process_sample(
    dataset: Any,
    *,
    dataset_index: int,
    ordinal: int,
    base_seed: int,
) -> dict[str, Any]:
    seed = sample_seed(base_seed, dataset_index)
    _set_sample_rng(seed)
    item = dataset[int(dataset_index)]
    if item is None:
        raise RuntimeError(f"v2xvit_train_manifest_sample_is_none:{dataset_index}")
    batch = dataset.collate_batch_train([item])
    if not isinstance(batch, dict) or "ego" not in batch:
        raise RuntimeError(f"v2xvit_train_manifest_invalid_batch:{dataset_index}")
    ego = batch["ego"]
    inputs = ego.get("inputs_m1")
    if not isinstance(inputs, dict):
        raise RuntimeError(f"v2xvit_train_manifest_missing_inputs_m1:{dataset_index}")
    features = inputs["voxel_features"]
    coords = inputs["voxel_coords"]
    points = inputs["voxel_num_points"]
    voxel_count = int(features.shape[0])
    if int(coords.shape[0]) != voxel_count or int(points.shape[0]) != voxel_count:
        raise RuntimeError(f"v2xvit_train_manifest_voxel_tensor_mismatch:{dataset_index}")
    record_len = int(ego["record_len"].sum().item())
    agent_modalities = [str(value) for value in ego["agent_modality_list"]]
    if record_len != len(agent_modalities):
        raise RuntimeError(f"v2xvit_train_manifest_agent_count_mismatch:{dataset_index}")
    per_agent = [int((coords[:, 0] == index).sum().item()) for index in range(record_len)]
    if sum(per_agent) != voxel_count:
        raise RuntimeError(f"v2xvit_train_manifest_per_agent_k_mismatch:{dataset_index}")

    vehicle_frame_id = str(dataset.split_info[int(dataset_index)])
    frame_info = dataset.co_data[vehicle_frame_id]
    item_ego = item["ego"]
    return {
        "ordinal": int(ordinal),
        "dataset_index": int(dataset_index),
        "vehicle_frame_id": vehicle_frame_id,
        "infrastructure_frame_id": _infrastructure_frame_id(frame_info),
        "source_pair_id": f"{vehicle_frame_id}:{_infrastructure_frame_id(frame_info)}",
        "sample_seed": int(seed),
        "record_len": record_len,
        "active_cav_ids": [int(value) for value in item_ego.get("cav_id_list", [])],
        "agent_modalities": agent_modalities,
        "voxel_count": voxel_count,
        "per_agent_voxel_counts": per_agent,
        "voxel_features_shape_unpadded": [int(value) for value in features.shape],
        "voxel_coords_dtype": str(coords.dtype).replace("torch.", ""),
        "voxel_features_dtype": str(features.dtype).replace("torch.", ""),
        "voxel_num_points_dtype": str(points.dtype).replace("torch.", ""),
    }


def _write_new_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise RuntimeError(f"refusing_to_overwrite_existing_manifest_artifact:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _freeze_manifest(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        current = json.loads(path.read_text(encoding="utf-8"))
        if current.get("manifest_hash") != payload.get("manifest_hash"):
            raise RuntimeError(f"refusing_to_replace_different_frozen_manifest:{path}")
        return "already_identical"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return "created"


def _summary_markdown(manifest: dict[str, Any]) -> str:
    distribution = manifest["voxel_count_distribution"]
    fixed = manifest["fixed_k_contract"]
    provenance = manifest["provenance"]
    return "\n".join(
        [
            "# HEAL LiDAR V2X-ViT train200 fixed-K manifest",
            "",
            f"- Manifest hash: `{manifest['manifest_hash']}`",
            f"- Train split SHA256: `{provenance['train_split']['sha256']}`",
            f"- Valid train samples: {manifest['selection']['dataset_size_after_validation']}",
            f"- Selection: `{manifest['selection']['policy']}` ({manifest['sample_count']} frames)",
            f"- K min / p50 / p90 / p95 / p99 / max: {distribution['min']} / "
            f"{distribution['p50']:.2f} / {distribution['p90']:.2f} / "
            f"{distribution['p95']:.2f} / {distribution['p99']:.2f} / {distribution['max']}",
            f"- Frozen fixed-K: **{fixed['value']}** (alignment {fixed['alignment']})",
            f"- Maximum-sample alignment margin: {fixed['alignment_margin_voxels']} voxels",
            f"- Truncated frozen samples: {fixed['truncated_sample_count']}",
            f"- Coverage scope: `{fixed['coverage_scope']}`",
            "",
            "This value is not a claimed upper bound for the other 4,611 train samples, the "
            "validation split, or a different preprocessing/augmentation contract.",
            "",
        ]
    )


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    if int(args.sample_count) != 200:
        raise ValueError(f"v2xvit_train200_requires_exactly_200_samples:{args.sample_count}")
    config_path = Path(args.config).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    heal_root = Path(args.heal_root).expanduser().resolve()
    for path, label in (
        (config_path, "config"),
        (checkpoint_path, "checkpoint"),
        (heal_root, "heal_root"),
    ):
        if not path.exists():
            raise RuntimeError(f"missing_{label}:{path}")

    adapter = HEALLiDARAdapter(
        heal_repo=str(heal_root), config={"model": {"hypes_yaml": str(config_path)}}
    )
    from opencood.data_utils.datasets import build_dataset
    from opencood.hypes_yaml import yaml_utils

    hypes = adapter._absolutize_dataset_paths(yaml_utils.load_yaml(str(config_path)))
    dataset = build_dataset(hypes, visualize=False, train=True)
    indices = evenly_spaced_indices(len(dataset), int(args.sample_count))
    samples = [
        _process_sample(
            dataset,
            dataset_index=index,
            ordinal=ordinal,
            base_seed=int(args.base_seed),
        )
        for ordinal, index in enumerate(indices)
    ]

    replay_ordinals = sorted({0, len(samples) // 2, len(samples) - 1})
    replay_checks = []
    for ordinal in replay_ordinals:
        expected = samples[ordinal]
        observed = _process_sample(
            dataset,
            dataset_index=int(expected["dataset_index"]),
            ordinal=ordinal,
            base_seed=int(args.base_seed),
        )
        fields = ("vehicle_frame_id", "record_len", "agent_modalities", "voxel_count", "per_agent_voxel_counts")
        passed = all(observed[field] == expected[field] for field in fields)
        if not passed:
            raise RuntimeError(f"v2xvit_train_manifest_replay_mismatch:{ordinal}")
        replay_checks.append(
            {
                "ordinal": ordinal,
                "dataset_index": int(expected["dataset_index"]),
                "fields": list(fields),
                "passed": True,
            }
        )

    train_split = Path(str(hypes["root_dir"])).resolve()
    cooperative_info = Path(str(hypes["data_dir"])) / "cooperative" / "data_info.json"
    raw_split = json.loads(train_split.read_text(encoding="utf-8"))
    manifest = finalize_v2xvit_train_manifest(
        {
            "schema_version": V2XVIT_TRAIN200_SCHEMA,
            "family_id": "heal_lidar_v2xvit",
            "model_name": str(hypes.get("name", "")),
            "purpose": "entropy_activation_calibration_and_fixed_k_export_contract",
            "split": "train",
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "source_control": {
                "repo": str(REPO_ROOT),
                "branch": _git_value("branch", "--show-current"),
                "head": _git_value("rev-parse", "HEAD"),
            },
            "provenance": {
                "config": {
                    "path": str(config_path),
                    "sha256": sha256_file(config_path),
                },
                "checkpoint": {
                    "path": str(checkpoint_path),
                    "sha256": sha256_file(checkpoint_path),
                    "note": "identity provenance only; voxel counts do not depend on weights",
                },
                "heal_root": str(heal_root),
                "dataset_root": str(Path(str(hypes["data_dir"])).resolve()),
                "train_split": {
                    "path": str(train_split),
                    "sha256": sha256_file(train_split),
                    "raw_entry_count": len(raw_split),
                },
                "cooperative_data_info": {
                    "path": str(cooperative_info.resolve()),
                    "sha256": sha256_file(cooperative_info),
                },
                "dataset_class": f"{type(dataset).__module__}.{type(dataset).__qualname__}",
            },
            "selection": {
                "policy": V2XVIT_TRAIN200_SELECTION_POLICY,
                "dataset_size_after_validation": len(dataset),
                "sample_count": int(args.sample_count),
                "base_seed": int(args.base_seed),
                "per_sample_seed_formula": "uint32(base_seed + dataset_index)",
                "order": "ascending_dataset_index",
                "shuffle": False,
            },
            "preprocessing": {
                "dataset_train_mode": True,
                "visualize": False,
                "direct_indexed_loading": True,
                "dataloader_workers": 0,
                "determinism": "python_numpy_torch_rng_reset_before_each_dataset_index",
                "voxel_size": hypes["heter"]["modality_setting"]["m1"]["preprocess"]["args"]["voxel_size"],
                "max_points_per_voxel": int(
                    hypes["heter"]["modality_setting"]["m1"]["preprocess"]["args"]["max_points_per_voxel"]
                ),
                "max_voxel_train_per_agent": int(
                    hypes["heter"]["modality_setting"]["m1"]["preprocess"]["args"]["max_voxel_train"]
                ),
                "project_first": bool(hypes["fusion"]["args"].get("proj_first", False)),
                "ego_selection": "HEAL train-time seeded DAIR heterogeneous ego assignment",
            },
            "fixed_k_contract": {
                "alignment": int(args.alignment),
                "padding_policy": "append_zero_voxels_and_explicit_valid_voxel_mask",
                "overflow_policy": "fail_closed_no_truncation",
            },
            "input_contract": {
                "max_points_per_voxel": int(
                    hypes["heter"]["modality_setting"]["m1"]["preprocess"]["args"]["max_points_per_voxel"]
                ),
                "max_agents": int(hypes["train_params"]["max_cav"]),
                "modality": "m1",
            },
            "replay_checks": replay_checks,
            "samples": samples,
        },
        expected_sample_count=int(args.sample_count),
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--heal-root", default=str(DEFAULT_HEAL_ROOT))
    parser.add_argument("--sample-count", type=int, default=200)
    parser.add_argument("--base-seed", type=int, default=V2XVIT_TRAIN200_BASE_SEED)
    parser.add_argument("--alignment", type=int, default=V2XVIT_FIXED_K_ALIGNMENT)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--frozen-manifest", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    manifest_path = output_dir / "v2xvit_train200_fixed_k_manifest.json"
    summary_path = output_dir / "v2xvit_train200_fixed_k_summary.md"
    if manifest_path.exists() or summary_path.exists():
        raise RuntimeError(f"refusing_to_overwrite_existing_output_directory:{output_dir}")
    manifest = build_manifest(args)
    _write_new_json(manifest_path, manifest)
    summary_path.write_text(_summary_markdown(manifest), encoding="utf-8")
    frozen_status = None
    if args.frozen_manifest:
        frozen_status = _freeze_manifest(Path(args.frozen_manifest).expanduser().resolve(), manifest)
    print(
        json.dumps(
            {
                "success": True,
                "manifest": str(manifest_path),
                "manifest_hash": manifest["manifest_hash"],
                "fixed_k": manifest["fixed_k_contract"]["value"],
                "observed_max": manifest["fixed_k_contract"]["observed_max_voxel_count"],
                "sample_count": manifest["sample_count"],
                "frozen_manifest": args.frozen_manifest,
                "frozen_status": frozen_status,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
