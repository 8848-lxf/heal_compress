from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from export_lidar_pyramid_onnx import _extract_inputs, _load_hypes
from quant_deploy_utils import DEFAULT_HEAL_REPO, DEFAULT_HYPES_YAML, ensure_quant_deploy_run_dirs, save_json


OLD_FIXED_K = 24064


def ceil_to_multiple(value: int | float, multiple: int) -> int:
    multiple = int(multiple)
    if multiple <= 0:
        raise ValueError("multiple must be positive")
    return int(math.ceil(float(value) / float(multiple)) * multiple)


def _percentile(values: list[int], pct: float) -> float | int | None:
    if not values:
        return None
    result = float(np.percentile(np.asarray(values, dtype=np.float64), pct))
    return int(result) if result.is_integer() else result


def _mean(values: list[int]) -> float | None:
    return float(np.asarray(values, dtype=np.float64).mean()) if values else None


def _metadata_from_ego(ego: dict[str, Any]) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    for key in ("scenario_id", "scenario", "frame_id", "timestamp", "sample_id", "cav_id"):
        value = ego.get(key) if isinstance(ego, dict) else None
        if value is None:
            continue
        try:
            if hasattr(value, "tolist"):
                value = value.tolist()
        except Exception:
            pass
        if isinstance(value, (str, int, float, bool)) or value is None:
            meta[key] = value
        else:
            meta[key] = str(value)
    return meta


def k_distribution_summary(samples: list[dict[str, Any]], *, split: str, old_fixed_k: int = OLD_FIXED_K) -> dict[str, Any]:
    valid = [item for item in samples if item.get("original_num_voxels") is not None]
    values = [int(item["original_num_voxels"]) for item in valid]
    gt_old = [item for item in valid if int(item["original_num_voxels"]) > int(old_fixed_k)]
    top = sorted(valid, key=lambda item: int(item["original_num_voxels"]), reverse=True)[:50]
    k_stats = {
        "min": min(values) if values else None,
        "p25": _percentile(values, 25),
        "p50": _percentile(values, 50),
        "p75": _percentile(values, 75),
        "p90": _percentile(values, 90),
        "p95": _percentile(values, 95),
        "p99": _percentile(values, 99),
        "p999": _percentile(values, 99.9),
        "max": max(values) if values else None,
        "mean": _mean(values),
    }
    return {
        "split": split,
        "total_samples": len(samples),
        "valid_samples": len(valid),
        "failed_samples": len(samples) - len(valid),
        "old_fixed_K": int(old_fixed_k),
        "K": k_stats,
        "samples_with_K_gt_24064": len(gt_old),
        "ratio_K_gt_24064": float(len(gt_old) / len(valid)) if valid else None,
        "record_len_distribution": dict(Counter(str(item.get("record_len")) for item in valid)),
        "top_50_largest_K_samples": [
            {
                "sample_idx": int(item.get("sample_idx", -1)),
                "dataset_index": item.get("dataset_index"),
                "scenario_id": item.get("scenario_id"),
                "frame_id": item.get("frame_id"),
                "record_len": item.get("record_len"),
                "original_num_voxels": int(item["original_num_voxels"]),
            }
            for item in top
        ],
        "samples": valid,
        "failed": [item for item in samples if item.get("original_num_voxels") is None],
    }


def recommended_fixed_k_from_reports(train_report: dict[str, Any], val_report: dict[str, Any], *, multiple: int = 512) -> int:
    train_max = int((train_report.get("K") or {}).get("max") or 0)
    val_max = int((val_report.get("K") or {}).get("max") or 0)
    return ceil_to_multiple(max(train_max, val_max), int(multiple))


def _loader_for_split(hypes: dict[str, Any], *, split: str):
    from opencood.data_utils.datasets import build_dataset
    from torch.utils.data import DataLoader

    train = split == "train"
    dataset = build_dataset(hypes, visualize=True, train=train)
    collate = getattr(dataset, "collate_batch_train", None) if train else getattr(dataset, "collate_batch_test", None)
    if collate is None:
        collate = getattr(dataset, "collate_batch_test", None) or getattr(dataset, "collate_batch_train")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate)
    return dataset, loader


def _record_len_value(ego: dict[str, Any]) -> int | None:
    value = ego.get("record_len") if isinstance(ego, dict) else None
    if value is None:
        return None
    try:
        if hasattr(value, "detach"):
            return int(value.detach().sum().item())
        if hasattr(value, "sum"):
            return int(value.sum())
        return int(value)
    except Exception:
        return None


def scan_split(args: argparse.Namespace, *, split: str) -> dict[str, Any]:
    hypes = _load_hypes(args.hypes_yaml, args.heal_repo)
    _dataset, loader = _loader_for_split(hypes, split=split)
    samples: list[dict[str, Any]] = []
    max_samples = args.max_samples if args.max_samples is None else int(args.max_samples)
    for dataset_index, batch in enumerate(loader):
        if max_samples is not None and dataset_index >= max_samples:
            break
        if batch is None:
            samples.append({"sample_idx": len(samples), "dataset_index": int(dataset_index), "reason": "batch_is_none", "original_num_voxels": None})
            continue
        try:
            ego = batch["ego"] if isinstance(batch, dict) and "ego" in batch else batch
            tensors, _agent_modalities = _extract_inputs(ego, args.modality)
            row = {
                "sample_idx": len(samples),
                "dataset_index": int(dataset_index),
                "record_len": _record_len_value(ego),
                "original_num_voxels": int(tensors[0].shape[0]),
            }
            row.update(_metadata_from_ego(ego))
            samples.append(row)
        except Exception as exc:
            samples.append(
                {
                    "sample_idx": len(samples),
                    "dataset_index": int(dataset_index),
                    "reason": "exception",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "original_num_voxels": None,
                }
            )
    return k_distribution_summary(samples, split=split, old_fixed_k=int(args.old_fixed_k))


def build_combined_report(train: dict[str, Any], val: dict[str, Any], *, old_fixed_k: int) -> dict[str, Any]:
    val_cover = ceil_to_multiple(int((val.get("K") or {}).get("max") or 0), 512)
    train_cover = ceil_to_multiple(int((train.get("K") or {}).get("max") or 0), 512)
    all_cover_512 = recommended_fixed_k_from_reports(train, val, multiple=512)
    all_cover_1024 = recommended_fixed_k_from_reports(train, val, multiple=1024)
    return {
        "old_fixed_K": int(old_fixed_k),
        "root_cause_of_filtered_full_val_skips": "K_exceeds_fixed_K" if val.get("samples_with_K_gt_24064") else "not_confirmed",
        "train": {key: value for key, value in train.items() if key not in {"samples", "failed"}},
        "val": {key: value for key, value in val.items() if key not in {"samples", "failed"}},
        "recommended_fixed_K_cover_val_all": val_cover,
        "recommended_fixed_K_cover_train_calib_all": train_cover,
        "recommended_fixed_K_cover_train_val_all": all_cover_512,
        "recommended_fixed_K_ceil_multiple_512": all_cover_512,
        "recommended_fixed_K_ceil_multiple_1024": all_cover_1024,
        "fixed_K_full_cover": all_cover_512,
        "covers_full_val": all_cover_512 >= int((val.get("K") or {}).get("max") or 0),
        "generated_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Voxel K Coverage Report",
        "",
        f"- old_fixed_K: {report.get('old_fixed_K')}",
        f"- root_cause_of_filtered_full_val_skips: {report.get('root_cause_of_filtered_full_val_skips')}",
        f"- fixed_K_full_cover: {report.get('fixed_K_full_cover')}",
        f"- covers_full_val: {report.get('covers_full_val')}",
        "",
        "split | total | valid | failed | K max | K p99 | K>old_fixed_K | ratio",
        "--- | --- | --- | --- | --- | --- | --- | ---",
    ]
    for split in ("train", "val"):
        item = report.get(split) or {}
        k = item.get("K") or {}
        lines.append(
            f"{split} | {item.get('total_samples')} | {item.get('valid_samples')} | {item.get('failed_samples')} | "
            f"{k.get('max')} | {k.get('p99')} | {item.get('samples_with_K_gt_24064')} | {item.get('ratio_K_gt_24064')}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze train/val voxel K coverage for fixed-K TensorRT deployment.")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--hypes_yaml", default=str(DEFAULT_HYPES_YAML))
    parser.add_argument("--heal_repo", default=str(DEFAULT_HEAL_REPO))
    parser.add_argument("--old_fixed_k", type=int, default=OLD_FIXED_K)
    parser.add_argument("--modality", default="m1")
    parser.add_argument("--max_samples", type=int, default=None)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    dirs = ensure_quant_deploy_run_dirs(args.output_root)
    train = scan_split(args, split="train")
    val = scan_split(args, split="val")
    report = build_combined_report(train, val, old_fixed_k=int(args.old_fixed_k))
    save_json(train, dirs["debug"] / "voxel_k_distribution_train.json")
    save_json(val, dirs["debug"] / "voxel_k_distribution_val.json")
    save_json(report, dirs["summary"] / "voxel_k_coverage_report.json")
    write_markdown(report, dirs["summary"] / "voxel_k_coverage_report.md")
    return report


def main(argv: list[str] | None = None) -> int:
    report = run(parse_args(argv))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report.get("covers_full_val") else 2


if __name__ == "__main__":
    raise SystemExit(main())
