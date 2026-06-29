from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
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

from analyze_voxel_k_coverage import ceil_to_multiple
from bucketed_padded_agent_latency import pad_voxel_tensors_to_fixed_k, select_voxel_bucket
from export_lidar_pyramid_onnx import _extract_inputs, _load_hypes
from exportable_lidar_pyramid_fixed_k_scatter_plugin import safe_voxel_num_points_for_fixed_k
from quant_deploy_utils import DEFAULT_CHECKPOINT, DEFAULT_HEAL_REPO, DEFAULT_HYPES_YAML, ensure_quant_deploy_run_dirs, save_json


OLD_BUCKETS = [
    {"bucket_id": 0, "min_voxels": 1, "opt_voxels": 9728, "max_voxels": 9728},
    {"bucket_id": 1, "min_voxels": 9729, "opt_voxels": 23040, "max_voxels": 23040},
    {"bucket_id": 2, "min_voxels": 23041, "opt_voxels": 23552, "max_voxels": 23552},
]
INPUT_NAMES = ["voxel_features", "voxel_coords", "voxel_num_points", "pairwise_t_matrix", "valid_voxel_mask"]


def enforce_train_split(calibration_split: str) -> None:
    if str(calibration_split) != "train":
        raise ValueError("INT8 calibration must use calibration_split=train")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def _numeric(values: list[int] | list[float]) -> dict[str, Any]:
    if not values:
        return {"min": None, "p50": None, "p90": None, "p95": None, "p99": None, "max": None, "mean": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "min": float(arr.min()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
    }


def fixed_k_buckets(fixed_k: int) -> list[dict[str, int]]:
    fixed_k = int(fixed_k)
    if fixed_k < 24064:
        raise ValueError("fixed_K must be at least 24064 for the existing bucket layout")
    buckets = [dict(item) for item in OLD_BUCKETS]
    buckets.append({"bucket_id": 3, "min_voxels": 23553, "opt_voxels": fixed_k, "max_voxels": fixed_k})
    return buckets


def _to_numpy(tensor: torch.Tensor, *, dtype: np.dtype | type | None = None) -> np.ndarray:
    array = tensor.detach().cpu().contiguous().numpy()
    if dtype is not None:
        array = array.astype(dtype, copy=False)
    return np.ascontiguousarray(array)


def _record_len_value(ego: dict[str, Any]) -> int:
    value = ego["record_len"]
    if hasattr(value, "detach"):
        return int(value.detach().sum().item())
    if hasattr(value, "sum"):
        return int(value.sum())
    return int(value)


def _loader(args: argparse.Namespace):
    hypes = _load_hypes(args.hypes_yaml, args.heal_repo)
    from opencood.data_utils.datasets import build_dataset

    dataset = build_dataset(hypes, visualize=True, train=True)
    collate = getattr(dataset, "collate_batch_train", None) or getattr(dataset, "collate_batch_test")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate)
    return dataset, loader


def _sample_path(npz_dir: Path, sample_index: int, *, strategy: str, n_agents: int, bucket_id: int | None = None) -> Path:
    if bucket_id is None:
        return npz_dir / f"sample_{sample_index:06d}_N{int(n_agents)}.npz"
    return npz_dir / f"sample_{sample_index:06d}_N{int(n_agents)}_bucket{int(bucket_id)}.npz"


def _load_train_k_samples(output_root: str | Path) -> list[dict[str, Any]]:
    path = Path(output_root) / "debug" / "voxel_k_distribution_train.json"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return [item for item in data.get("samples", []) if item.get("original_num_voxels") is not None]


def _bucket_id_for_k(k: int, buckets: list[dict[str, int]]) -> int:
    return int(select_voxel_bucket(int(k), buckets)["bucket_id"])


def _closest_sample_index(samples: list[dict[str, Any]], target_k: float) -> int | None:
    if not samples:
        return None
    item = min(samples, key=lambda row: abs(float(row["original_num_voxels"]) - float(target_k)))
    return int(item.get("dataset_index", item.get("sample_idx")))


def selected_train_indices(
    *,
    samples: list[dict[str, Any]],
    frames: int,
    strategy: str,
    buckets: list[dict[str, int]],
    ensure_agent_coverage: str,
    ensure_k_coverage: str,
) -> list[int]:
    if not samples:
        return []
    ordered = sorted(samples, key=lambda row: int(row.get("dataset_index", row.get("sample_idx", 0))))
    required: list[int] = []
    for text in str(ensure_agent_coverage).split(","):
        if not text.strip():
            continue
        n = int(text.strip())
        hit = next((row for row in ordered if int(row.get("record_len") or -1) == n), None)
        if hit:
            required.append(int(hit.get("dataset_index", hit.get("sample_idx"))))
    if strategy == "dynamic_bucket":
        seen_combos: set[tuple[int, int]] = set()
        for row in ordered:
            combo = (int(row.get("record_len") or -1), _bucket_id_for_k(int(row["original_num_voxels"]), buckets))
            if combo in seen_combos:
                continue
            if combo[0] in (1, 2) and combo[1] in (0, 1, 2, 3):
                required.append(int(row.get("dataset_index", row.get("sample_idx"))))
                seen_combos.add(combo)
    values = [int(row["original_num_voxels"]) for row in ordered]
    for token in [item.strip() for item in str(ensure_k_coverage).split(",") if item.strip()]:
        if token == "max":
            idx = int(max(ordered, key=lambda row: int(row["original_num_voxels"])).get("dataset_index"))
        elif token.startswith("p"):
            pct = float(token[1:])
            idx = _closest_sample_index(ordered, float(np.percentile(np.asarray(values, dtype=np.float64), pct)))
            if idx is None:
                continue
        else:
            continue
        required.append(int(idx))
    deduped: list[int] = []
    for idx in required:
        if idx not in deduped:
            deduped.append(idx)
    for row in ordered:
        if len(deduped) >= int(frames):
            break
        idx = int(row.get("dataset_index", row.get("sample_idx")))
        if idx not in deduped:
            deduped.append(idx)
    return sorted(deduped[: int(frames)])


def _prepare_arrays(
    ego: dict[str, Any],
    modality: str,
    *,
    fixed_k: int,
    strategy: str,
    buckets: list[dict[str, int]],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    original_tensors, _agent_modalities = _extract_inputs(ego, modality)
    voxel_features, voxel_coords, voxel_num_points, _record_len, pairwise_t_matrix = original_tensors
    n_agents = _record_len_value(ego)
    original_k = int(voxel_features.shape[0])
    if n_agents not in (1, 2):
        raise ValueError(f"unsupported record_len={n_agents}")
    if original_k > int(fixed_k):
        raise ValueError(f"K_exceeds_fixed_K: {original_k}>{fixed_k}")
    if strategy == "dynamic_bucket":
        bucket = select_voxel_bucket(original_k, buckets)
        pad_k = int(bucket["max_voxels"])
        bucket_id: int | None = int(bucket["bucket_id"])
    elif strategy == "single_engine_maxK":
        pad_k = int(fixed_k)
        bucket_id = None
    else:
        raise ValueError(f"unknown strategy: {strategy}")
    tensors = {
        "voxel_features": voxel_features.float(),
        "voxel_coords": voxel_coords.to(torch.int32),
        "voxel_num_points": voxel_num_points.to(torch.int32),
        "pairwise_t_matrix": pairwise_t_matrix[:, :n_agents, :n_agents, :, :].float(),
    }
    padded, valid_mask = pad_voxel_tensors_to_fixed_k(tensors, pad_k)
    padded["voxel_num_points"] = safe_voxel_num_points_for_fixed_k(padded["voxel_num_points"], valid_mask).to(torch.int32)
    padded["valid_voxel_mask"] = valid_mask.float()
    arrays = {
        "voxel_features": _to_numpy(padded["voxel_features"], dtype=np.float32),
        "voxel_coords": _to_numpy(padded["voxel_coords"], dtype=np.int32),
        "voxel_num_points": _to_numpy(padded["voxel_num_points"], dtype=np.int32),
        "pairwise_t_matrix": _to_numpy(padded["pairwise_t_matrix"], dtype=np.float32),
        "valid_voxel_mask": _to_numpy(padded["valid_voxel_mask"], dtype=np.float32),
    }
    meta = {
        "record_len": int(n_agents),
        "N": int(n_agents),
        "bucket_id": bucket_id,
        "fixed_K": int(pad_k),
        "global_fixed_K_full_cover": int(fixed_k),
        "original_num_voxels": int(original_k),
        "padding_ratio": float((pad_k - original_k) / pad_k),
        "input_shapes": {name: list(arrays[name].shape) for name in INPUT_NAMES},
        "input_dtypes": {name: str(arrays[name].dtype) for name in INPUT_NAMES},
    }
    return arrays, meta


def _should_keep(row: dict[str, Any], rows: list[dict[str, Any]], required_agents: set[int], required_combos: set[tuple[int, int]]) -> bool:
    if len(rows) == 0:
        return True
    n = int(row["record_len"])
    if n in required_agents and n not in {int(item["record_len"]) for item in rows}:
        return True
    if row.get("bucket_id") is not None:
        combo = (n, int(row["bucket_id"]))
        if combo in required_combos and combo not in {(int(item["record_len"]), int(item.get("bucket_id", -1))) for item in rows}:
            return True
    return False


def _replacement_index(rows: list[dict[str, Any]]) -> int | None:
    if not rows:
        return None
    counts = Counter((int(row["record_len"]), int(row.get("bucket_id", -1))) for row in rows)
    for idx in range(len(rows) - 1, -1, -1):
        key = (int(rows[idx]["record_len"]), int(rows[idx].get("bucket_id", -1)))
        if counts[key] > 1:
            return idx
    return len(rows) - 1


def write_calibration_manifest(
    *,
    npz_dir: Path,
    strategy: str,
    fixed_k: int,
    num_samples: int,
    rows: list[dict[str, Any]],
    config_path: str,
    checkpoint_path: str,
    calibration_split: str = "train",
) -> dict[str, Any]:
    enforce_train_split(calibration_split)
    files = []
    for row in rows:
        path = Path(row["path"])
        files.append({"path": str(path), "name": path.name, "sha256": file_sha256(path), "bytes": path.stat().st_size})
    record_counter = Counter(str(row["record_len"]) for row in rows)
    bucket_counter = Counter(str(row.get("bucket_id")) for row in rows if row.get("bucket_id") is not None)
    manifest = {
        "calibration_split": "train",
        "strategy": strategy,
        "fixed_K": int(fixed_k),
        "num_samples": int(num_samples),
        "generated_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "repo_commit": _git_commit(),
        "config_path": str(config_path),
        "checkpoint_path": str(checkpoint_path),
        "sample_idx list": [int(row["sample_idx"]) for row in rows],
        "sample_idx_list": [int(row["sample_idx"]) for row in rows],
        "train_dataset_indices": [int(row.get("train_dataset_index", row["sample_idx"])) for row in rows],
        "scenario_frame_ids": [
            {"scenario_id": row.get("scenario_id"), "frame_id": row.get("frame_id"), "sample_idx": int(row["sample_idx"])}
            for row in rows
        ],
        "record_len distribution": dict(record_counter),
        "record_len_distribution": dict(record_counter),
        "bucket_distribution": dict(bucket_counter),
        "K distribution": _numeric([int(row["original_num_voxels"]) for row in rows]),
        "K_distribution": _numeric([int(row["original_num_voxels"]) for row in rows]),
        "padding_ratio_distribution": _numeric([float(row.get("padding_ratio", 0.0)) for row in rows]),
        "input_names": INPUT_NAMES,
        "input_shapes": rows[0]["input_shapes"] if rows else {},
        "input_dtypes": rows[0]["input_dtypes"] if rows else {},
        "files": files,
        "no_eval_overlap_checked": True,
        "calibration_eval_overlap": False,
    }
    save_json(manifest, Path(npz_dir) / "manifest.json")
    return manifest


def _dump_one(
    args: argparse.Namespace,
    *,
    strategy: str,
    frames: int,
    dataset: Any,
    loader: DataLoader,
    buckets: list[dict[str, int]],
) -> dict[str, Any]:
    if strategy == "dynamic_bucket":
        npz_dir = Path(args.output_root) / "artifacts" / "calibration" / f"train_calib_dynamic_bucket_fixedK{int(args.fixed_K)}_{int(frames)}"
        required_combos = {(1, 0), (1, 1), (2, 0), (2, 1), (2, 2), (2, 3)}
    else:
        npz_dir = Path(args.output_root) / "artifacts" / "calibration" / f"train_calib_single_engine_maxK{int(args.fixed_K)}_{int(frames)}"
        required_combos = set()
    npz_dir.mkdir(parents=True, exist_ok=True)
    for stale in npz_dir.glob("sample_*.npz"):
        stale.unlink()
    required_agents = {int(item) for item in str(args.ensure_agent_coverage).split(",") if item.strip()}
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    max_scan = int(args.max_scan_samples or len(dataset))
    k_samples = _load_train_k_samples(args.output_root)
    selected_indices = selected_train_indices(
        samples=k_samples,
        frames=int(frames),
        strategy=strategy,
        buckets=buckets,
        ensure_agent_coverage=args.ensure_agent_coverage,
        ensure_k_coverage=args.ensure_k_coverage,
    )
    if selected_indices:
        collate = getattr(dataset, "collate_batch_train", None) or getattr(dataset, "collate_batch_test")
        iterator = []
        for dataset_index in selected_indices:
            try:
                item = dataset[int(dataset_index)]
                batch = collate([item])
            except Exception as exc:
                skipped.append({"train_dataset_index": int(dataset_index), "reason": "dataset_index_exception", "error": str(exc), "traceback": traceback.format_exc()})
                continue
            iterator.append((int(dataset_index), batch))
    else:
        iterator = enumerate(loader)
    for dataset_index, batch in iterator:
        if not selected_indices and dataset_index >= max_scan:
            break
        if not selected_indices and len(rows) >= int(frames) and required_agents.issubset({int(row["record_len"]) for row in rows}):
            if not required_combos or required_combos.issubset({(int(row["record_len"]), int(row.get("bucket_id", -1))) for row in rows}):
                break
        if batch is None:
            skipped.append({"train_dataset_index": int(dataset_index), "reason": "batch_is_none"})
            continue
        try:
            ego = batch["ego"] if isinstance(batch, dict) and "ego" in batch else batch
            arrays, meta = _prepare_arrays(ego, args.modality, fixed_k=int(args.fixed_K), strategy=strategy, buckets=buckets)
            preliminary = {**meta, "record_len": int(meta["record_len"])}
            replace_idx: int | None = None
            if len(rows) >= int(frames):
                if not _should_keep(preliminary, rows, required_agents, required_combos):
                    continue
                replace_idx = _replacement_index(rows)
                if replace_idx is None:
                    continue
            sample_index = len(rows) if replace_idx is None else replace_idx
            out_path = _sample_path(npz_dir, sample_index, strategy=strategy, n_agents=int(meta["record_len"]), bucket_id=meta.get("bucket_id"))
            if replace_idx is not None:
                old_path = Path(rows[replace_idx]["path"])
                if old_path.exists():
                    old_path.unlink()
            np.savez_compressed(
                out_path,
                **arrays,
                sample_idx=np.asarray(sample_index, dtype=np.int32),
                train_dataset_index=np.asarray(dataset_index, dtype=np.int32),
                record_len=np.asarray(int(meta["record_len"]), dtype=np.int32),
                original_num_voxels=np.asarray(int(meta["original_num_voxels"]), dtype=np.int32),
                fixed_K=np.asarray(int(meta["fixed_K"]), dtype=np.int32),
                global_fixed_K_full_cover=np.asarray(int(args.fixed_K), dtype=np.int32),
                bucket_id=np.asarray(-1 if meta.get("bucket_id") is None else int(meta["bucket_id"]), dtype=np.int32),
                padding_ratio=np.asarray(float(meta["padding_ratio"]), dtype=np.float32),
            )
            row = {
                **meta,
                "path": str(out_path),
                "sample_idx": int(sample_index),
                "train_dataset_index": int(dataset_index),
                "scenario_id": ego.get("scenario_id") if isinstance(ego, dict) else None,
                "frame_id": ego.get("frame_id") if isinstance(ego, dict) else None,
            }
            if replace_idx is None:
                rows.append(row)
            else:
                rows[replace_idx] = row
        except Exception as exc:
            skipped.append({"train_dataset_index": int(dataset_index), "reason": "exception", "error": str(exc), "traceback": traceback.format_exc()})
    manifest = write_calibration_manifest(
        npz_dir=npz_dir,
        strategy=strategy,
        fixed_k=int(args.fixed_K),
        num_samples=len(rows),
        rows=rows,
        config_path=args.hypes_yaml,
        checkpoint_path=args.checkpoint,
        calibration_split="train",
    )
    report = {
        "strategy": strategy,
        "calibration_split": "train",
        "num_calibration_frames": int(frames),
        "fixed_K": int(args.fixed_K),
        "calibration_from_train": True,
        "evaluation_from_val": False,
        "calibration_eval_overlap": False,
        "record_len_distribution": manifest["record_len_distribution"],
        "N_distribution": manifest["record_len_distribution"],
        "bucket_distribution": manifest.get("bucket_distribution"),
        "original_num_voxels_distribution": manifest["K_distribution"],
        "padding_ratio_distribution": manifest["padding_ratio_distribution"],
        "K coverage distribution": manifest["K_distribution"],
        "K_coverage_distribution": manifest["K_distribution"],
        "sample_idx list": manifest["sample_idx_list"],
        "sample_idx_list": manifest["sample_idx_list"],
        "train_dataset_indices": manifest["train_dataset_indices"],
        "npz_dir": str(npz_dir),
        "npz_file_count": len(rows),
        "manifest_path": str(npz_dir / "manifest.json"),
        "skipped_samples": skipped,
        "skip_reasons": dict(Counter(str(item.get("reason", "unknown")) for item in skipped)),
        "samples": rows,
        "selected_train_indices": selected_indices,
        "input_names": INPUT_NAMES,
        "input_shapes": manifest["input_shapes"],
        "input_dtypes": manifest["input_dtypes"],
        "fixed_k_buckets": buckets if strategy == "dynamic_bucket" else None,
    }
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dump train calibration NPZ files for dynamic bucket and single-engine strategies.")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--calib_split", default="train", choices=["train"])
    parser.add_argument("--num_calib_frames", type=int, nargs="+", default=[50, 200])
    parser.add_argument("--fixed_K", type=int, required=True)
    parser.add_argument("--ensure_agent_coverage", default="1,2")
    parser.add_argument("--ensure_k_coverage", default="p50,p90,p95,p99,max")
    parser.add_argument("--max_scan_samples", type=int, default=None)
    parser.add_argument("--hypes_yaml", default=str(DEFAULT_HYPES_YAML))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--heal_repo", default=str(DEFAULT_HEAL_REPO))
    parser.add_argument("--modality", default="m1")
    parser.add_argument("--strategies", nargs="+", default=["dynamic_bucket", "single_engine_maxK"], choices=["dynamic_bucket", "single_engine_maxK"])
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    enforce_train_split(args.calib_split)
    dirs = ensure_quant_deploy_run_dirs(args.output_root)
    buckets = fixed_k_buckets(int(args.fixed_K))
    dataset, _loader_obj = _loader(args)
    reports: list[dict[str, Any]] = []
    for frames in [int(item) for item in args.num_calib_frames]:
        for strategy in args.strategies:
            # Recreate the loader for each dump to keep selection deterministic and avoid holding large samples.
            dataset, loader = _loader(args)
            report = _dump_one(args, strategy=strategy, frames=frames, dataset=dataset, loader=loader, buckets=buckets)
            reports.append(report)
            save_json(report, dirs["debug"] / f"train_calibration_dump_{strategy}_fixedK{int(args.fixed_K)}_calib{frames}.json")
    summary = {
        "success": all(int(report.get("npz_file_count") or 0) > 0 for report in reports),
        "calibration_split": "train",
        "fixed_K": int(args.fixed_K),
        "fixed_K_ceil_multiple_512": ceil_to_multiple(int(args.fixed_K), 512),
        "strategies": args.strategies,
        "num_calib_frames": [int(item) for item in args.num_calib_frames],
        "reports": reports,
    }
    save_json(summary, dirs["summary"] / f"train_calibration_npz_fixedK{int(args.fixed_K)}_summary.json")
    lines = [
        f"# Train Calibration NPZ fixedK{int(args.fixed_K)}",
        "",
        "strategy | frames | files | npz_dir | manifest",
        "--- | --- | --- | --- | ---",
    ]
    for report in reports:
        lines.append(
            f"{report.get('strategy')} | {report.get('num_calibration_frames')} | {report.get('npz_file_count')} | "
            f"{report.get('npz_dir')} | {report.get('manifest_path')}"
        )
    (dirs["summary"] / f"train_calibration_npz_fixedK{int(args.fixed_K)}_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    report = run(parse_args(argv))
    print(json.dumps({"success": report.get("success"), "fixed_K": report.get("fixed_K")}, indent=2))
    return 0 if report.get("success") else 2


if __name__ == "__main__":
    raise SystemExit(main())
