from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bucketed_padded_agent_latency import pad_voxel_tensors_to_fixed_k
from deployment_equivalence import _load_model_context, _record_len_value
from dynamic_single_engine_maxk_common import FIXED_K, calibration_npz_dir, numeric_summary, single_engine_input_names
from export_lidar_pyramid_onnx import _extract_inputs, _to_device
from exportable_lidar_pyramid_fixed_k_scatter_plugin import safe_voxel_num_points_for_fixed_k
from quant_deploy_utils import DEFAULT_CHECKPOINT, DEFAULT_HEAL_REPO, DEFAULT_HYPES_YAML, ensure_quant_deploy_run_dirs, save_json


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dump train-split dynamic single-engine maxK INT8 calibration NPZ inputs.")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--num_calib_frames", type=int, required=True)
    parser.add_argument("--calib_split", default="train", choices=["train"])
    parser.add_argument("--ensure_agent_coverage", default="1,2")
    parser.add_argument("--max_scan_samples", type=int, default=None)
    parser.add_argument("--output_npz_dir", default=None)
    parser.add_argument("--hypes_yaml", default=str(DEFAULT_HYPES_YAML))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--heal_repo", default=str(DEFAULT_HEAL_REPO))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_cav", type=int, default=2)
    return parser.parse_args(argv)


def _required_agent_counts(text: str) -> set[int]:
    return {int(item.strip()) for item in str(text).split(",") if item.strip()}


def _to_numpy(tensor: torch.Tensor, *, dtype: np.dtype | type | None = None) -> np.ndarray:
    arr = tensor.detach().cpu().contiguous().numpy()
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return np.ascontiguousarray(arr)


def _sample_path(npz_dir: Path, sample_index: int, n_agents: int) -> Path:
    return npz_dir / f"sample_{sample_index:06d}_N{int(n_agents)}.npz"


def _replacement_index_for_agent(rows: list[dict[str, Any]]) -> int | None:
    counts: Counter[int] = Counter(int(row["record_len"]) for row in rows)
    for idx in range(len(rows) - 1, -1, -1):
        if counts[int(rows[idx]["record_len"])] > 1:
            return idx
    return None


def _dataset_loader(args: argparse.Namespace):
    hypes, device, model, modality = _load_model_context(args)
    from opencood.data_utils.datasets import build_dataset

    dataset = build_dataset(hypes, visualize=True, train=True)
    collate = getattr(dataset, "collate_batch_train", None) or getattr(dataset, "collate_batch_test")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate)
    return hypes, device, model, modality, dataset, loader


def dump_calibration(args: argparse.Namespace) -> dict[str, Any]:
    dirs = ensure_quant_deploy_run_dirs(args.output_root)
    npz_dir = Path(args.output_npz_dir).expanduser() if args.output_npz_dir else calibration_npz_dir(dirs, int(args.num_calib_frames))
    npz_dir.mkdir(parents=True, exist_ok=True)
    for stale in npz_dir.glob("sample_*.npz"):
        stale.unlink()

    required_agents = _required_agent_counts(args.ensure_agent_coverage)
    scan_limit = int(args.max_scan_samples or max(int(args.num_calib_frames) * 16, int(args.num_calib_frames) + 512))
    _hypes, device, _model, modality, _dataset, loader = _dataset_loader(args)
    input_names = single_engine_input_names()
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    coverage_events: list[dict[str, Any]] = []

    actual_seen = 0
    for frame_idx, batch in enumerate(loader):
        if actual_seen >= scan_limit:
            break
        actual_seen += 1
        if len(rows) >= int(args.num_calib_frames) and required_agents.issubset({int(row["record_len"]) for row in rows}):
            break
        if batch is None:
            skipped.append({"frame_id": int(frame_idx), "reason": "batch_is_none"})
            continue
        try:
            ego = batch["ego"] if isinstance(batch, dict) and "ego" in batch else batch
            ego = _to_device(ego, device)
            original_tensors, _agent_modalities = _extract_inputs(ego, modality)
            voxel_features, voxel_coords, voxel_num_points, _record_len, pairwise_t_matrix = original_tensors
            n_agents = int(_record_len_value(ego))
            if n_agents <= 0 or n_agents > int(args.max_cav):
                skipped.append({"frame_id": int(frame_idx), "reason": f"unsupported_record_len_{n_agents}"})
                continue
            original_num_voxels = int(voxel_features.shape[0])
            if original_num_voxels > FIXED_K:
                skipped.append({"frame_id": int(frame_idx), "record_len": n_agents, "original_num_voxels": original_num_voxels, "reason": "K_exceeds_fixed_maxK"})
                continue
            replace_idx: int | None = None
            covered_agents = {int(row["record_len"]) for row in rows}
            if len(rows) >= int(args.num_calib_frames):
                if n_agents not in required_agents or n_agents in covered_agents:
                    continue
                replace_idx = _replacement_index_for_agent(rows)
                if replace_idx is None:
                    continue

            tensors_by_name = {
                "voxel_features": voxel_features.float(),
                "voxel_coords": voxel_coords.to(torch.int32),
                "voxel_num_points": voxel_num_points.to(torch.int32),
                "pairwise_t_matrix": pairwise_t_matrix[:, :n_agents, :n_agents, :, :].float(),
            }
            padded, valid_mask = pad_voxel_tensors_to_fixed_k(tensors_by_name, FIXED_K)
            padded["voxel_num_points"] = safe_voxel_num_points_for_fixed_k(padded["voxel_num_points"], valid_mask).to(torch.int32)
            padded["valid_voxel_mask"] = valid_mask.float()
            arrays = {
                "voxel_features": _to_numpy(padded["voxel_features"], dtype=np.float32),
                "voxel_coords": _to_numpy(padded["voxel_coords"], dtype=np.int32),
                "voxel_num_points": _to_numpy(padded["voxel_num_points"], dtype=np.int32),
                "pairwise_t_matrix": _to_numpy(padded["pairwise_t_matrix"], dtype=np.float32),
                "valid_voxel_mask": _to_numpy(padded["valid_voxel_mask"], dtype=np.float32),
            }
            sample_index = len(rows) if replace_idx is None else replace_idx
            if replace_idx is not None:
                old_path = Path(rows[replace_idx]["path"])
                if old_path.exists():
                    old_path.unlink()
            out_path = _sample_path(npz_dir, sample_index, n_agents)
            padding_ratio = float((FIXED_K - original_num_voxels) / FIXED_K)
            np.savez_compressed(
                out_path,
                **arrays,
                frame_id=np.asarray(frame_idx, dtype=np.int32),
                sample_idx=np.asarray(sample_index, dtype=np.int32),
                train_scan_index=np.asarray(actual_seen - 1, dtype=np.int32),
                record_len=np.asarray(n_agents, dtype=np.int32),
                original_num_voxels=np.asarray(original_num_voxels, dtype=np.int32),
                fixed_K=np.asarray(FIXED_K, dtype=np.int32),
                padding_ratio=np.asarray(padding_ratio, dtype=np.float32),
            )
            row = {
                "path": str(out_path),
                "frame_id": int(frame_idx),
                "sample_idx": int(sample_index),
                "train_scan_index": int(actual_seen - 1),
                "record_len": int(n_agents),
                "N": int(n_agents),
                "fixed_K": int(FIXED_K),
                "original_num_voxels": int(original_num_voxels),
                "padding_ratio": padding_ratio,
                "input_shapes": {name: list(arrays[name].shape) for name in input_names},
                "input_dtypes": {name: str(arrays[name].dtype) for name in input_names},
            }
            if replace_idx is None:
                rows.append(row)
            else:
                replaced = rows[replace_idx]
                rows[replace_idx] = row
                coverage_events.append(
                    {
                        "event": "replace_for_required_agent_coverage",
                        "sample_index": int(replace_idx),
                        "old_record_len": int(replaced["record_len"]),
                        "new_record_len": int(n_agents),
                    }
                )
        except Exception as exc:
            skipped.append({"frame_id": int(frame_idx), "reason": "exception", "error": str(exc)})

    record_counter = Counter(str(row["record_len"]) for row in rows)
    report = {
        "calibration_split": "train",
        "num_calibration_frames": int(args.num_calib_frames),
        "actual_calibration_frames": len(rows),
        "max_scan_samples": scan_limit,
        "scanned_train_samples": actual_seen,
        "calibration_from_train": True,
        "evaluation_from_val": False,
        "calibration_eval_overlap": False,
        "record_len_distribution": dict(record_counter),
        "N_distribution": dict(record_counter),
        "original_num_voxels_distribution": numeric_summary([row["original_num_voxels"] for row in rows]),
        "padding_ratio_distribution": numeric_summary([row["padding_ratio"] for row in rows]),
        "skipped_samples": skipped,
        "skip_reasons": dict(Counter(str(item.get("reason", "unknown")) for item in skipped)),
        "train_sample_indices": [int(row["train_scan_index"]) for row in rows],
        "sample_idx_list": [int(row["sample_idx"]) for row in rows],
        "overlap_check_available": False,
        "npz_dir": str(npz_dir),
        "sample_files_count": len(rows),
        "input_names": input_names,
        "input_shapes": rows[0]["input_shapes"] if rows else {},
        "input_dtypes": rows[0]["input_dtypes"] if rows else {},
        "fixed_K": int(FIXED_K),
        "valid_voxel_mask_enabled": True,
        "pointpillar_scatter_plugin_enabled": True,
        "agent_export_mode": "dynamic_agent_single_engine_maxK",
        "single_engine": True,
        "no_bucket_router": True,
        "no_N_engine_router": True,
        "coverage_events": coverage_events,
        "required_agent_coverage": sorted(required_agents),
        "missing_agent_coverage": sorted(required_agents - {int(row["record_len"]) for row in rows}),
        "samples": rows,
    }
    save_json(report, dirs["debug"] / f"dynamic_single_engine_maxK_train_calibration_dump_{int(args.num_calib_frames)}.json")
    return report


def main(argv: list[str] | None = None) -> int:
    report = dump_calibration(parse_args(argv))
    print(report)
    return 0 if int(report.get("sample_files_count", 0)) > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
