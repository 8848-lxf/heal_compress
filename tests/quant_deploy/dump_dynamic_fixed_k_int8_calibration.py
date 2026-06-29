from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent_mode_fixed_k_plugin_ablation import FIXED_K_BUCKETS, _frame_iter
from bucketed_padded_agent_latency import pad_voxel_tensors_to_fixed_k, select_voxel_bucket
from export_lidar_pyramid_onnx import _input_names_for_export_mode, _prepare_export_tensors
from exportable_lidar_pyramid_fixed_k_scatter_plugin import safe_voxel_num_points_for_fixed_k
from quant_deploy_utils import DEFAULT_CHECKPOINT, DEFAULT_HEAL_REPO, DEFAULT_HYPES_YAML, ensure_quant_deploy_run_dirs, save_json


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dump real dynamic fixed-K TensorRT INT8 calibration NPZ inputs.")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--num_frames", type=int, required=True)
    parser.add_argument("--hypes_yaml", default=str(DEFAULT_HYPES_YAML))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--heal_repo", default=str(DEFAULT_HEAL_REPO))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_cav", type=int, default=2)
    return parser.parse_args(argv)


def _to_numpy(tensor) -> np.ndarray:
    return tensor.detach().cpu().contiguous().numpy()


def _numeric(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"min": None, "p50": None, "p90": None, "p95": None, "p99": None, "mean": None, "max": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "min": float(arr.min()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "mean": float(arr.mean()),
        "max": float(arr.max()),
    }


def _required_calibration_combos() -> set[tuple[int, int]]:
    return {(1, 0), (1, 1), (2, 1), (2, 2), (2, 3)}


def _covered_combos(rows: list[dict[str, Any]], required: set[tuple[int, int]]) -> set[tuple[int, int]]:
    return {(int(row["record_len"]), int(row["bucket_id"])) for row in rows} & required


def _replacement_index_for_combo(rows: list[dict[str, Any]]) -> int | None:
    counts: Counter[tuple[int, int]] = Counter((int(row["record_len"]), int(row["bucket_id"])) for row in rows)
    for idx in range(len(rows) - 1, -1, -1):
        combo = (int(rows[idx]["record_len"]), int(rows[idx]["bucket_id"]))
        if counts[combo] > 1:
            return idx
    return None


def _sample_npz_path(npz_dir: Path, sample_index: int, fixed_n: int, bucket_id: int) -> Path:
    return npz_dir / f"sample_{sample_index:06d}_N{fixed_n}_bucket{bucket_id}.npz"


def dump_calibration(args: argparse.Namespace) -> dict[str, Any]:
    dirs = ensure_quant_deploy_run_dirs(args.output_root)
    npz_dir = dirs["output_root"] / "artifacts" / "calibration" / f"dynamic_fixed_k_int8_calib_{int(args.num_frames)}"
    npz_dir.mkdir(parents=True, exist_ok=True)
    for stale in npz_dir.glob("sample_*.npz"):
        stale.unlink()
    input_names = _input_names_for_export_mode("padded_agent_static")
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    coverage_events: list[dict[str, Any]] = []
    required_combos = _required_calibration_combos()
    scan_limit = max(int(args.num_frames) * 16, int(args.num_frames) + 512)
    for item in _frame_iter(args, scan_limit):
        if len(rows) >= int(args.num_frames) and _covered_combos(rows, required_combos) == required_combos:
            break
        try:
            tensors = _prepare_export_tensors(item["original_tensors"], export_mode="padded_agent_static", max_cav=int(args.max_cav))
            tensors_by_name = {name: tensor for name, tensor in zip(input_names, tensors)}
            original_num_voxels = int(tensors_by_name["voxel_features"].shape[0])
            bucket = select_voxel_bucket(original_num_voxels, FIXED_K_BUCKETS)
            fixed_k = int(bucket["max_voxels"])
            fixed_n = int(item["record_len"])
            combo = (fixed_n, int(bucket["bucket_id"]))
            replace_idx: int | None = None
            if len(rows) >= int(args.num_frames):
                if combo not in required_combos or combo in _covered_combos(rows, required_combos):
                    continue
                replace_idx = _replacement_index_for_combo(rows)
                if replace_idx is None:
                    continue
            padded, valid_mask = pad_voxel_tensors_to_fixed_k(tensors_by_name, fixed_k)
            padded["voxel_num_points"] = safe_voxel_num_points_for_fixed_k(padded["voxel_num_points"], valid_mask)
            padded["valid_voxel_mask"] = valid_mask
            padded.pop("valid_agent_mask", None)
            padded["pairwise_t_matrix"] = padded["pairwise_t_matrix"][:, :fixed_n, :fixed_n, :, :]
            arrays = {name: _to_numpy(padded[name]) for name in ("voxel_features", "voxel_coords", "voxel_num_points", "pairwise_t_matrix", "valid_voxel_mask")}
            sample_index = len(rows) if replace_idx is None else replace_idx
            if replace_idx is not None:
                old_path = Path(rows[replace_idx]["path"])
                if old_path.exists():
                    old_path.unlink()
            out_path = _sample_npz_path(npz_dir, sample_index, fixed_n, int(bucket["bucket_id"]))
            np.savez_compressed(
                out_path,
                **arrays,
                frame_id=np.asarray(item["frame_id"], dtype=np.int32),
                sample_idx=np.asarray(item["sample_idx"], dtype=np.int32),
                record_len=np.asarray(fixed_n, dtype=np.int32),
                original_num_voxels=np.asarray(original_num_voxels, dtype=np.int32),
                bucket_id=np.asarray(int(bucket["bucket_id"]), dtype=np.int32),
                fixed_K=np.asarray(fixed_k, dtype=np.int32),
                padding_ratio=np.asarray((fixed_k - original_num_voxels) / fixed_k, dtype=np.float32),
            )
            row = {
                "path": str(out_path),
                "frame_id": int(item["frame_id"]),
                "sample_idx": int(item["sample_idx"]),
                "record_len": fixed_n,
                "bucket_id": int(bucket["bucket_id"]),
                "fixed_K": fixed_k,
                "original_num_voxels": original_num_voxels,
                "padding_ratio": float((fixed_k - original_num_voxels) / fixed_k),
                "input_shapes": {name: list(arr.shape) for name, arr in arrays.items()},
                "input_dtypes": {name: str(arr.dtype) for name, arr in arrays.items()},
            }
            if replace_idx is None:
                rows.append(row)
            else:
                replaced = rows[replace_idx]
                rows[replace_idx] = row
                coverage_events.append(
                    {
                        "event": "replace_for_required_combo",
                        "sample_index": replace_idx,
                        "old_combo": [int(replaced["record_len"]), int(replaced["bucket_id"])],
                        "new_combo": [fixed_n, int(bucket["bucket_id"])],
                    }
                )
        except Exception as exc:
            skipped.append({"frame_id": int(item.get("frame_id", -1)), "error": str(exc)})
    record_counter = Counter(str(row["record_len"]) for row in rows)
    bucket_counter = Counter(str(row["bucket_id"]) for row in rows)
    report = {
        "num_calibration_frames": int(args.num_frames),
        "sample_files_count": len(rows),
        "npz_dir": str(npz_dir),
        "input_names": ["voxel_features", "voxel_coords", "voxel_num_points", "pairwise_t_matrix", "valid_voxel_mask"],
        "input_shapes": rows[0]["input_shapes"] if rows else {},
        "input_dtypes": rows[0]["input_dtypes"] if rows else {},
        "record_len_distribution": dict(record_counter),
        "bucket_distribution": dict(bucket_counter),
        "original_num_voxels_distribution": _numeric([row["original_num_voxels"] for row in rows]),
        "padding_ratio_distribution": _numeric([row["padding_ratio"] for row in rows]),
        "fixed_k_buckets": FIXED_K_BUCKETS,
        "samples": rows,
        "any_skipped_samples": skipped,
        "required_combo_coverage": sorted([list(combo) for combo in _covered_combos(rows, required_combos)]),
        "missing_required_combos": sorted([list(combo) for combo in (required_combos - _covered_combos(rows, required_combos))]),
        "coverage_events": coverage_events,
        "agent_export_mode": "dynamic_agent_dim",
        "fixed_k_bucket_router": True,
        "valid_voxel_mask_enabled": True,
        "pointpillar_scatter_plugin_enabled": True,
    }
    save_json(report, dirs["debug"] / f"dynamic_fixed_k_int8_calibration_dump_{int(args.num_frames)}.json")
    return report


def main(argv: list[str] | None = None) -> int:
    report = dump_calibration(parse_args(argv))
    print(report)
    return 0 if report.get("sample_files_count", 0) else 2


if __name__ == "__main__":
    raise SystemExit(main())
