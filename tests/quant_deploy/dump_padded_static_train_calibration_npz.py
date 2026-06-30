from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bucketed_padded_agent_latency import pad_voxel_tensors_to_fixed_k, select_voxel_bucket
from dump_train_calibration_npz_for_all_strategies import (
    _git_commit,
    _loader,
    _load_train_k_samples,
    _numeric,
    enforce_train_split,
    file_sha256,
    fixed_k_buckets,
    selected_train_indices,
)
from export_lidar_pyramid_onnx import _extract_inputs
from exportable_lidar_pyramid_fixed_k_scatter_plugin import fixed_k_scatter_plugin_input_names, safe_voxel_num_points_for_fixed_k
from exportable_lidar_pyramid_padded_agent import make_valid_agent_mask
from quant_deploy_utils import DEFAULT_CHECKPOINT, DEFAULT_HEAL_REPO, DEFAULT_HYPES_YAML, ensure_quant_deploy_run_dirs, save_json


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dump padded_agent_static fixed-K train calibration NPZ files.")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--calib_split", default="train", choices=["train"])
    parser.add_argument("--calibration_frames", "--num_calib_frames", dest="calibration_frames", type=int, default=200)
    parser.add_argument("--fixed_K", type=int, default=29696)
    parser.add_argument("--max_cav", type=int, default=2)
    parser.add_argument("--hypes_yaml", default=str(DEFAULT_HYPES_YAML))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--heal_repo", default=str(DEFAULT_HEAL_REPO))
    parser.add_argument("--modality", default="m1")
    parser.add_argument("--ensure_agent_coverage", default="1,2")
    parser.add_argument("--ensure_k_coverage", default="p50,p90,p95,p99,max")
    parser.add_argument("--max_scan_samples", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _to_numpy(tensor: torch.Tensor, *, dtype: np.dtype | type | None = None) -> np.ndarray:
    array = tensor.detach().cpu().contiguous().numpy()
    if dtype is not None:
        array = array.astype(dtype, copy=False)
    return np.ascontiguousarray(array)


def calibration_npz_dir(output_root: str | Path, fixed_k: int, frames: int) -> Path:
    return Path(output_root) / "artifacts" / "calibration" / f"train_calib_padded_agent_static_fixedK{int(fixed_k)}_{int(frames)}"


def _record_len_value(ego: dict[str, Any]) -> int:
    value = ego["record_len"]
    if hasattr(value, "detach"):
        return int(value.detach().sum().item())
    if hasattr(value, "sum"):
        return int(value.sum())
    return int(value)


def _prepare_padded_arrays(
    ego: dict[str, Any],
    modality: str,
    *,
    fixed_k: int,
    max_cav: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    original_tensors, _agent_modalities = _extract_inputs(ego, modality)
    voxel_features, voxel_coords, voxel_num_points, record_len_tensor, pairwise_t_matrix = original_tensors
    record_len = _record_len_value(ego)
    original_k = int(voxel_features.shape[0])
    if record_len not in (1, 2):
        raise ValueError(f"unsupported record_len={record_len}")
    if record_len > int(max_cav):
        raise ValueError(f"record_len={record_len} exceeds max_cav={max_cav}")
    if original_k > int(fixed_k):
        raise ValueError(f"K_exceeds_fixed_K: {original_k}>{fixed_k}")

    tensors = {
        "voxel_features": voxel_features.float(),
        "voxel_coords": voxel_coords.to(torch.int32),
        "voxel_num_points": voxel_num_points.to(torch.int32),
    }
    padded, valid_voxel_mask = pad_voxel_tensors_to_fixed_k(tensors, int(fixed_k))
    padded["voxel_num_points"] = safe_voxel_num_points_for_fixed_k(padded["voxel_num_points"], valid_voxel_mask).to(torch.int32)

    pairwise = torch.eye(4, dtype=pairwise_t_matrix.dtype, device=pairwise_t_matrix.device).view(1, 1, 1, 4, 4).repeat(1, int(max_cav), int(max_cav), 1, 1)
    pairwise[:, :record_len, :record_len, :, :] = pairwise_t_matrix[:, :record_len, :record_len, :, :]
    valid_agent_mask = make_valid_agent_mask(record_len_tensor, max_cav=int(max_cav), dtype=voxel_features.dtype)

    arrays = {
        "voxel_features": _to_numpy(padded["voxel_features"], dtype=np.float32),
        "voxel_coords": _to_numpy(padded["voxel_coords"], dtype=np.int32),
        "voxel_num_points": _to_numpy(padded["voxel_num_points"], dtype=np.int32),
        "valid_agent_mask": _to_numpy(valid_agent_mask, dtype=np.float32),
        "pairwise_t_matrix": _to_numpy(pairwise.float(), dtype=np.float32),
        "valid_voxel_mask": _to_numpy(valid_voxel_mask.float(), dtype=np.float32),
    }
    input_names = fixed_k_scatter_plugin_input_names()
    meta = {
        "record_len": int(record_len),
        "N": int(record_len),
        "fixed_K": int(fixed_k),
        "max_cav": int(max_cav),
        "original_num_voxels": int(original_k),
        "padding_ratio": float((int(fixed_k) - original_k) / int(fixed_k)),
        "valid_agent_mask": arrays["valid_agent_mask"].astype(float).tolist(),
        "input_shapes": {name: list(arrays[name].shape) for name in input_names},
        "input_dtypes": {name: str(arrays[name].dtype) for name in input_names},
    }
    return arrays, meta


def _write_manifest(
    *,
    npz_dir: Path,
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    files = []
    for row in rows:
        path = Path(row["path"])
        files.append({"path": str(path), "name": path.name, "sha256": file_sha256(path), "bytes": path.stat().st_size})
    record_counter = Counter(str(row["record_len"]) for row in rows)
    mask_counter = Counter(json.dumps(row["valid_agent_mask"]) for row in rows)
    manifest = {
        "strategy": "padded_agent_static",
        "calibration_split": "train",
        "calibration_frames": int(args.calibration_frames),
        "fixed_K": int(args.fixed_K),
        "max_cav": int(args.max_cav),
        "calibration_eval_overlap": False,
        "npz_file_count": len(rows),
        "sample_idx list": [int(row["sample_idx"]) for row in rows],
        "sample_idx_list": [int(row["sample_idx"]) for row in rows],
        "train_dataset_indices": [int(row["train_dataset_index"]) for row in rows],
        "scenario_frame_ids": [
            {"scenario_id": row.get("scenario_id"), "frame_id": row.get("frame_id"), "sample_idx": int(row["sample_idx"])}
            for row in rows
        ],
        "record_len distribution": dict(record_counter),
        "record_len_distribution": dict(record_counter),
        "valid_agent_mask distribution": dict(mask_counter),
        "valid_agent_mask_distribution": dict(mask_counter),
        "original_num_voxels distribution": _numeric([int(row["original_num_voxels"]) for row in rows]),
        "original_num_voxels_distribution": _numeric([int(row["original_num_voxels"]) for row in rows]),
        "padding_ratio distribution": _numeric([float(row["padding_ratio"]) for row in rows]),
        "padding_ratio_distribution": _numeric([float(row["padding_ratio"]) for row in rows]),
        "input_names": fixed_k_scatter_plugin_input_names(),
        "input_shapes": rows[0]["input_shapes"] if rows else {},
        "input_dtypes": rows[0]["input_dtypes"] if rows else {},
        "sha256 for each NPZ": {Path(item["path"]).name: item["sha256"] for item in files},
        "files": files,
        "generated_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "repo_commit": _git_commit(),
        "config_path": str(args.hypes_yaml),
        "checkpoint_path": str(args.checkpoint),
    }
    save_json(manifest, npz_dir / "manifest.json")
    return manifest


def run(args: argparse.Namespace) -> dict[str, Any]:
    enforce_train_split(args.calib_split)
    dirs = ensure_quant_deploy_run_dirs(args.output_root)
    npz_dir = calibration_npz_dir(args.output_root, int(args.fixed_K), int(args.calibration_frames))
    npz_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for stale in npz_dir.glob("sample_*.npz"):
            stale.unlink()
    existing = sorted(npz_dir.glob("sample_*.npz"))
    manifest_path = npz_dir / "manifest.json"
    if len(existing) == int(args.calibration_frames) and manifest_path.exists() and not args.overwrite:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        report = {
            "success": True,
            "reused_existing": True,
            "strategy": "padded_agent_static",
            "calibration_split": "train",
            "calibration_frames": int(args.calibration_frames),
            "fixed_K": int(args.fixed_K),
            "max_cav": int(args.max_cav),
            "npz_dir": str(npz_dir),
            "npz_file_count": len(existing),
            "manifest_path": str(manifest_path),
            "manifest": manifest,
        }
        save_json(report, dirs["debug"] / f"train_calibration_dump_padded_agent_static_fixedK{int(args.fixed_K)}_calib{int(args.calibration_frames)}.json")
        return report

    buckets = fixed_k_buckets(int(args.fixed_K))
    dataset, _loader_obj = _loader(args)
    k_samples = _load_train_k_samples(args.output_root)
    selected_indices = selected_train_indices(
        samples=k_samples,
        frames=int(args.calibration_frames),
        strategy="single_engine_maxK",
        buckets=buckets,
        ensure_agent_coverage=args.ensure_agent_coverage,
        ensure_k_coverage=args.ensure_k_coverage,
    )
    if not selected_indices:
        selected_indices = list(range(min(int(args.calibration_frames), int(args.max_scan_samples or len(dataset)))))

    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    collate = getattr(dataset, "collate_batch_train", None) or getattr(dataset, "collate_batch_test")
    for sample_idx, dataset_index in enumerate(selected_indices[: int(args.calibration_frames)]):
        try:
            item = dataset[int(dataset_index)]
            batch = collate([item])
            ego = batch["ego"] if isinstance(batch, dict) and "ego" in batch else batch
            arrays, meta = _prepare_padded_arrays(ego, args.modality, fixed_k=int(args.fixed_K), max_cav=int(args.max_cav))
            bucket = select_voxel_bucket(int(meta["original_num_voxels"]), buckets)
            out_path = npz_dir / f"sample_{sample_idx:06d}_N{int(meta['record_len'])}_bucket{int(bucket['bucket_id'])}.npz"
            np.savez_compressed(
                out_path,
                **arrays,
                sample_idx=np.asarray(sample_idx, dtype=np.int32),
                train_dataset_index=np.asarray(dataset_index, dtype=np.int32),
                record_len=np.asarray(int(meta["record_len"]), dtype=np.int32),
                original_num_voxels=np.asarray(int(meta["original_num_voxels"]), dtype=np.int32),
                fixed_K=np.asarray(int(args.fixed_K), dtype=np.int32),
                max_cav=np.asarray(int(args.max_cav), dtype=np.int32),
                bucket_id=np.asarray(int(bucket["bucket_id"]), dtype=np.int32),
                padding_ratio=np.asarray(float(meta["padding_ratio"]), dtype=np.float32),
            )
            rows.append(
                {
                    **meta,
                    "path": str(out_path),
                    "sample_idx": int(sample_idx),
                    "train_dataset_index": int(dataset_index),
                    "bucket_id": int(bucket["bucket_id"]),
                    "scenario_id": ego.get("scenario_id") if isinstance(ego, dict) else None,
                    "frame_id": ego.get("frame_id") if isinstance(ego, dict) else None,
                }
            )
        except Exception as exc:
            skipped.append({"train_dataset_index": int(dataset_index), "reason": "exception", "error": str(exc), "traceback": traceback.format_exc()})

    manifest = _write_manifest(npz_dir=npz_dir, rows=rows, args=args)
    report = {
        "success": len(rows) == int(args.calibration_frames),
        "strategy": "padded_agent_static",
        "calibration_split": "train",
        "calibration_frames": int(args.calibration_frames),
        "fixed_K": int(args.fixed_K),
        "max_cav": int(args.max_cav),
        "calibration_eval_overlap": False,
        "npz_dir": str(npz_dir),
        "npz_file_count": len(rows),
        "manifest_path": str(npz_dir / "manifest.json"),
        "selected_train_indices": selected_indices[: int(args.calibration_frames)],
        "record_len_distribution": manifest["record_len_distribution"],
        "valid_agent_mask_distribution": manifest["valid_agent_mask_distribution"],
        "original_num_voxels_distribution": manifest["original_num_voxels_distribution"],
        "padding_ratio_distribution": manifest["padding_ratio_distribution"],
        "input_names": manifest["input_names"],
        "input_shapes": manifest["input_shapes"],
        "input_dtypes": manifest["input_dtypes"],
        "samples": rows,
        "skipped_samples": skipped,
        "skip_reasons": dict(Counter(str(item.get("reason", "unknown")) for item in skipped)),
    }
    save_json(report, dirs["debug"] / f"train_calibration_dump_padded_agent_static_fixedK{int(args.fixed_K)}_calib{int(args.calibration_frames)}.json")
    lines = [
        f"# Padded Agent Static Train Calibration fixedK{int(args.fixed_K)} calib{int(args.calibration_frames)}",
        "",
        f"- strategy: padded_agent_static",
        f"- calibration_split: train",
        f"- calibration_frames: {int(args.calibration_frames)}",
        f"- fixed_K: {int(args.fixed_K)}",
        f"- max_cav: {int(args.max_cav)}",
        f"- npz_file_count: {len(rows)}",
        f"- npz_dir: {npz_dir}",
        f"- manifest: {npz_dir / 'manifest.json'}",
        f"- skipped_samples: {len(skipped)}",
        f"- record_len_distribution: {manifest['record_len_distribution']}",
        f"- valid_agent_mask_distribution: {manifest['valid_agent_mask_distribution']}",
    ]
    (dirs["summary"] / f"train_calibration_dump_padded_agent_static_fixedK{int(args.fixed_K)}_calib{int(args.calibration_frames)}.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    return report


def main(argv: list[str] | None = None) -> int:
    report = run(parse_args(argv))
    print(json.dumps({"success": report.get("success"), "npz_file_count": report.get("npz_file_count")}, indent=2))
    return 0 if report.get("success") else 2


if __name__ == "__main__":
    raise SystemExit(main())
