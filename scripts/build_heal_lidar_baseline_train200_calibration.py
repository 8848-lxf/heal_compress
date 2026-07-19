"""Convert the accepted Pyramid train200 tensors to the baseline six-input contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


INPUT_NAMES = (
    "voxel_features",
    "voxel_coords",
    "voxel_num_points",
    "pairwise_t_matrix",
    "valid_voxel_mask",
    "agent_mask",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_path(manifest: Path, row: dict[str, Any]) -> Path:
    raw = Path(str(row.get("path", ""))).expanduser()
    candidates = (raw, manifest.parent / str(row.get("name", "")), manifest.parent / raw.name)
    source = next((path.resolve() for path in candidates if path.is_file()), None)
    if source is None:
        raise RuntimeError(f"source_calibration_file_missing:{row}")
    return source


def build(source_manifest: Path, output_dir: Path, *, fixed_k: int = 29696) -> Path:
    payload = json.loads(source_manifest.read_text(encoding="utf-8"))
    files = list(payload.get("files", []) or [])
    if len(files) != 200:
        raise RuntimeError(f"source_manifest_not_train200:{len(files)}")
    output_dir.mkdir(parents=True, exist_ok=False)
    output_rows = []
    observed_max = 0
    record_lengths = []
    for index, row in enumerate(files):
        source = _source_path(source_manifest, dict(row))
        if _sha256(source) != str(row.get("sha256", "")):
            raise RuntimeError(f"source_calibration_hash_mismatch:{source}")
        with np.load(source) as values:
            required = [name for name in INPUT_NAMES[:-1] if name not in values.files]
            if required:
                raise RuntimeError(f"source_calibration_inputs_missing:{source}:{required}")
            valid = np.asarray(values["valid_voxel_mask"]).reshape(-1)
            voxel_count = int(np.count_nonzero(valid > 0.5))
            if voxel_count > int(fixed_k):
                raise RuntimeError(
                    f"baseline_fixed_k_would_truncate_real_voxels:{source}:{voxel_count}>{fixed_k}"
                )
            observed_max = max(observed_max, voxel_count)
            pairwise_source = np.asarray(values["pairwise_t_matrix"])
            if pairwise_source.ndim != 5 or pairwise_source.shape[0] != 1 or pairwise_source.shape[1] != pairwise_source.shape[2]:
                raise RuntimeError(f"source_pairwise_shape_invalid:{source}:{pairwise_source.shape}")
            agents = int(pairwise_source.shape[1])
            if agents not in {1, 2}:
                raise RuntimeError(f"baseline_calibration_agent_count_invalid:{source}:{agents}")
            record_lengths.append(agents)
            pairwise = np.eye(4, dtype=pairwise_source.dtype).reshape(1, 1, 1, 4, 4)
            pairwise = np.tile(pairwise, (1, 2, 2, 1, 1))
            pairwise[:, :agents, :agents] = pairwise_source
            agent_mask = np.zeros((1, 2), dtype=np.float32)
            agent_mask[:, :agents] = 1.0
            arrays = {
                "voxel_features": np.ascontiguousarray(values["voxel_features"][:fixed_k]),
                "voxel_coords": np.ascontiguousarray(values["voxel_coords"][:fixed_k]),
                "voxel_num_points": np.ascontiguousarray(values["voxel_num_points"][:fixed_k]),
                "pairwise_t_matrix": np.ascontiguousarray(pairwise),
                "valid_voxel_mask": np.ascontiguousarray(values["valid_voxel_mask"][:fixed_k]),
                "agent_mask": agent_mask,
            }
        destination = output_dir / f"sample_{index:06d}_N{agents}.npz"
        np.savez_compressed(destination, **arrays)
        output_rows.append({
            "name": destination.name,
            "path": str(destination.resolve()),
            "sha256": _sha256(destination),
            "bytes": destination.stat().st_size,
            "source_name": source.name,
            "source_sha256": str(row.get("sha256", "")),
            "voxel_count": voxel_count,
            "record_len": agents,
        })
    manifest = {
        "schema_version": "heal-lidar-baseline-fixed-k-train200-v2",
        "strategy": "single_engine_maxK",
        "calibration_split": "train",
        "fixed_K": int(fixed_k),
        "num_samples": len(output_rows),
        "input_names": list(INPUT_NAMES),
        "train_dataset_indices": [int(value) for value in payload.get("train_dataset_indices", [])],
        "frame_ids": [str(value) for value in payload.get("frame_ids", [])],
        "files": output_rows,
        "observed_max_voxel_count": observed_max,
        "record_len_distribution": {
            str(value): record_lengths.count(value) for value in sorted(set(record_lengths))
        },
        "padding_policy": "crop_only_source_zero_padding_then_pad_agents_to_two_with_explicit_agent_mask",
        "overflow_policy": "fail_closed_no_real_voxel_truncation",
        "source_manifest": str(source_manifest.resolve()),
        "source_manifest_sha256": _sha256(source_manifest),
    }
    destination = output_dir / "calibration_manifest.json"
    destination.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return destination


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-manifest",
        default="tests/quant_deploy/outputs/lidar_pyramid_agent_export_strategy_compare/artifacts/calibration/train_calib_single_engine_maxK29696_200/manifest.json",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/heal_lidar_baseline_train200_fixedk29696",
    )
    parser.add_argument("--fixed-k", type=int, default=29696)
    args = parser.parse_args()
    manifest = build(
        Path(args.source_manifest).expanduser().resolve(),
        Path(args.output_dir).expanduser().resolve(),
        fixed_k=int(args.fixed_k),
    )
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
