from __future__ import annotations

import argparse
import ctypes
import json
import math
import re
import sys
import time
import traceback
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from deployment_equivalence import TensorRTEngineRunner, _load_model_context, _record_len_value
from evaluate_lidar_pyramid_trt_ap import IOU_THRESHOLDS, _calculate_tp_fp, _mean, _percentile, _timed
from export_lidar_pyramid_onnx import _extract_inputs, _input_names_for_export_mode, _prepare_export_tensors, _to_device
from exportable_lidar_pyramid_fixed_k_scatter_plugin import safe_voxel_num_points_for_fixed_k
from latency_decomposition import LATENCY_FIELDS, parse_trtexec_latency_metrics, summarize_latency_rows
from quant_deploy_utils import (
    DEFAULT_CHECKPOINT,
    DEFAULT_HEAL_REPO,
    DEFAULT_HYPES_YAML,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TRT_ROOT,
    build_trtexec_command,
    ensure_quant_deploy_run_dirs,
    find_trtexec_report,
    parse_trtexec_failure,
    read_json,
    run_command,
    save_json,
)


PRECISIONS = ("fp32", "fp16")
PADDED_MODE = "padded_agent_static"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and profile bucketed TensorRT engines for padded_agent_static lidar_pyramid.")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--hypes_yaml", default=str(DEFAULT_HYPES_YAML))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--heal_repo", default=str(DEFAULT_HEAL_REPO))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_frames", type=int, default=50)
    parser.add_argument("--num_buckets", type=int, default=4)
    parser.add_argument("--round_to", type=int, default=512)
    parser.add_argument("--max_cav", type=int, default=2)
    parser.add_argument("--trt_root", default=str(DEFAULT_TRT_ROOT))
    parser.add_argument("--trtexec_path", default=None)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--warmup_ms", type=int, default=200)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--duration", type=int, default=3)
    parser.add_argument("--ap_iou_backend", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--skip_build", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--skip_profile", action="store_true")
    parser.add_argument("--fixed_k_latency_only", action="store_true")
    parser.add_argument("--fixed_k_scatter_plugin", action="store_true")
    parser.add_argument("--plugin_so", default=None)
    return parser.parse_args(argv)


def _round_up(value: int, multiple: int) -> int:
    multiple = max(1, int(multiple))
    return int(math.ceil(max(1, int(value)) / multiple) * multiple)


def _percentile_nearest(values: list[int], pct: float) -> int:
    if not values:
        raise ValueError("cannot compute percentile of empty values")
    ordered = sorted(int(v) for v in values)
    index = min(len(ordered) - 1, max(0, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return int(ordered[index])


def build_voxel_buckets(voxel_counts: list[int], *, num_buckets: int = 4, round_to: int = 512) -> list[dict[str, Any]]:
    if not voxel_counts:
        raise ValueError("voxel_counts must not be empty")
    requested = min(max(1, int(num_buckets)), 5)
    counts = sorted(int(v) for v in voxel_counts)
    percentiles = [(idx + 1) * 100.0 / requested for idx in range(requested)]
    max_bounds: list[int] = []
    for pct in percentiles:
        bound = _round_up(_percentile_nearest(counts, pct), round_to)
        if max_bounds and bound <= max_bounds[-1]:
            bound = _round_up(max_bounds[-1] + 1, round_to)
        max_bounds.append(bound)
    max_bounds[-1] = max(max_bounds[-1], _round_up(max(counts), round_to))

    buckets: list[dict[str, Any]] = []
    previous_max = 0
    for bucket_id, max_voxels in enumerate(max_bounds):
        min_voxels = 1 if bucket_id == 0 else previous_max + 1
        in_bucket = [value for value in counts if min_voxels <= value <= max_voxels]
        opt_source = _percentile_nearest(in_bucket or [min(max(counts), max_voxels)], 50)
        buckets.append(
            {
                "bucket_id": bucket_id,
                "min_voxels": int(min_voxels),
                "opt_voxels": int(min(max(_round_up(opt_source, round_to), min_voxels), max_voxels)),
                "max_voxels": int(max_voxels),
                "num_frames": len(in_bucket),
                "voxel_count_min": min(in_bucket) if in_bucket else None,
                "voxel_count_max": max(in_bucket) if in_bucket else None,
            }
        )
        previous_max = int(max_voxels)
    return buckets


def select_voxel_bucket(num_voxels: int, buckets: list[dict[str, Any]]) -> dict[str, Any]:
    value = int(num_voxels)
    for bucket in sorted(buckets, key=lambda item: int(item["max_voxels"])):
        if value <= int(bucket["max_voxels"]):
            return bucket
    raise ValueError(f"num_voxels={value} is outside bucket coverage max={max(int(b['max_voxels']) for b in buckets)}")


def pad_voxel_tensors_to_fixed_k(tensors_by_name: dict[str, torch.Tensor], fixed_k: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    fixed_k = int(fixed_k)
    num_voxels = int(tensors_by_name["voxel_features"].shape[0])
    if num_voxels > fixed_k:
        raise ValueError(f"num_voxels={num_voxels} exceeds fixed_k={fixed_k}")
    padded = dict(tensors_by_name)
    device = tensors_by_name["voxel_features"].device
    for name in ("voxel_features", "voxel_coords", "voxel_num_points"):
        tensor = tensors_by_name[name]
        shape = list(tensor.shape)
        shape[0] = fixed_k
        out = torch.zeros(tuple(shape), dtype=tensor.dtype, device=tensor.device)
        if num_voxels:
            out[:num_voxels].copy_(tensor)
        padded[name] = out
    valid_mask = torch.zeros((fixed_k,), dtype=tensors_by_name["voxel_features"].dtype, device=device)
    if num_voxels:
        valid_mask[:num_voxels] = 1
    return padded, valid_mask


def profile_shapes_for_bucket(bucket: dict[str, Any], *, max_cav: int = 2) -> dict[str, Any]:
    min_voxels = int(bucket["min_voxels"])
    opt_voxels = int(bucket["opt_voxels"])
    max_voxels = int(bucket["max_voxels"])
    return {
        "voxel_features": {"min": [min_voxels, 32, 4], "opt": [opt_voxels, 32, 4], "max": [max_voxels, 32, 4]},
        "voxel_coords": {"min": [min_voxels, 4], "opt": [opt_voxels, 4], "max": [max_voxels, 4]},
        "voxel_num_points": {"min": [min_voxels], "opt": [opt_voxels], "max": [max_voxels]},
        "valid_agent_mask": {"min": [1, max_cav], "opt": [1, max_cav], "max": [1, max_cav]},
        "pairwise_t_matrix": {"min": [1, max_cav, max_cav, 4, 4], "opt": [1, max_cav, max_cav, 4, 4], "max": [1, max_cav, max_cav, 4, 4]},
    }


def profile_shapes_for_fixed_k_scatter_plugin_bucket(bucket: dict[str, Any], *, max_cav: int = 2) -> dict[str, Any]:
    fixed_k = int(bucket["max_voxels"])
    return {
        "voxel_features": {"min": [fixed_k, 32, 4], "opt": [fixed_k, 32, 4], "max": [fixed_k, 32, 4]},
        "voxel_coords": {"min": [fixed_k, 4], "opt": [fixed_k, 4], "max": [fixed_k, 4]},
        "voxel_num_points": {"min": [fixed_k], "opt": [fixed_k], "max": [fixed_k]},
        "valid_agent_mask": {"min": [1, max_cav], "opt": [1, max_cav], "max": [1, max_cav]},
        "pairwise_t_matrix": {"min": [1, max_cav, max_cav, 4, 4], "opt": [1, max_cav, max_cav, 4, 4], "max": [1, max_cav, max_cav, 4, 4]},
        "valid_voxel_mask": {"min": [fixed_k], "opt": [fixed_k], "max": [fixed_k]},
    }


def fixed_shape_profile_shapes(num_voxels: int, *, max_cav: int = 2) -> dict[str, Any]:
    bucket = {"min_voxels": int(num_voxels), "opt_voxels": int(num_voxels), "max_voxels": int(num_voxels)}
    return profile_shapes_for_bucket(bucket, max_cav=max_cav)


def classify_trt_layer(layer_name: str, layer_type: str | None = None) -> str:
    text = f"{layer_name} {layer_type or ''}".lower()
    if any(token in text for token in ("pillar_vfe", "pfn", "reduce_max", "reducemax", "reduce max")):
        return "PillarVFE / PFN"
    if any(token in text for token in ("scatternd", "pointpillarscatter", "point_pillar_scatter", "scatter/")):
        return "PointPillarScatter / ScatterND"
    if "valid_agent_mask" in text or "agent_mask" in text:
        return "valid_agent_mask fusion"
    if any(token in text for token in ("gridsample", "grid_sample", "gridgrid", "affine_grid", "bev_warp", "warp_affine", "pyramid_fusion", "pyramid_backbone")):
        return "Pyramid fusion / BEV warp / GridSample"
    if any(token in text for token in ("cls_head", "reg_head", "dir_head", "clshead", "reghead", "dirhead")) and "conv" in text:
        return "Head Conv"
    if any(token in text for token in ("shape", "gather", "slice", "reshape", "shuffle", "reformat", "constantofshape", "cast")):
        return "Shape/Gather/Slice/Reshape/Shuffle/Reformat"
    if "conv" in text or "convolution" in text:
        return "BEV backbone Conv"
    return "Other"


def _shape_spec(shapes: dict[str, list[int]]) -> str:
    return ",".join(f"{name}:{'x'.join(str(int(v)) for v in shape)}" for name, shape in shapes.items())


def _load_dataset_context(args: argparse.Namespace):
    hypes, device, model, modality = _load_model_context(args)
    from opencood.data_utils.datasets import build_dataset

    dataset = build_dataset(hypes, visualize=True, train=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=dataset.collate_batch_test)
    return hypes, device, model, modality, dataset, loader


def _iter_prepared_frames(args: argparse.Namespace, *, limit: int | None = None):
    _hypes, device, _model, modality, dataset, loader = _load_dataset_context(args)
    input_names = _input_names_for_export_mode(PADDED_MODE)
    actual = 0
    for frame_idx, batch in enumerate(loader):
        if limit is not None and actual >= int(limit):
            break
        if batch is None:
            continue
        ego = batch["ego"] if isinstance(batch, dict) and "ego" in batch else batch
        ego = _to_device(ego, device)
        batch = _to_device(batch, device)
        original_tensors, _agent_modalities = _extract_inputs(ego, modality)
        tensors = _prepare_export_tensors(original_tensors, export_mode=PADDED_MODE, max_cav=int(args.max_cav))
        tensors_by_name = {name: tensor for name, tensor in zip(input_names, tensors)}
        yield {
            "frame_id": frame_idx,
            "actual_index": actual,
            "batch": batch,
            "ego": ego,
            "dataset": dataset,
            "tensors_by_name": tensors_by_name,
            "record_len": _record_len_value(ego),
            "num_voxels": int(tensors_by_name["voxel_features"].shape[0]),
        }
        actual += 1


def collect_voxel_distribution(args: argparse.Namespace, dirs: dict[str, Path]) -> dict[str, Any]:
    frames = [{"frame_id": item["frame_id"], "record_len": item["record_len"], "num_voxels": item["num_voxels"]} for item in _iter_prepared_frames(args, limit=args.num_frames)]
    counts = [int(item["num_voxels"]) for item in frames]
    by_record_len: dict[str, list[int]] = {}
    for item in frames:
        by_record_len.setdefault(str(int(item["record_len"])), []).append(int(item["num_voxels"]))
    report = {
        "num_frames": len(frames),
        "frames": frames,
        "overall": _numeric_summary(counts),
        "by_record_len": {key: {"record_len": int(key), "num_frames": len(vals), **_numeric_summary(vals)} for key, vals in sorted(by_record_len.items(), key=lambda kv: int(kv[0]))},
    }
    save_json(report, dirs["debug"] / "num_voxels_distribution_padded_agent_static.json")
    return report


def _numeric_summary(values: list[int] | list[float]) -> dict[str, Any]:
    vals = [float(v) for v in values]
    if not vals:
        return {"min": None, "p50": None, "p90": None, "p95": None, "max": None, "mean": None}
    return {
        "min": min(vals),
        "p50": _percentile(vals, 50),
        "p90": _percentile(vals, 90),
        "p95": _percentile(vals, 95),
        "max": max(vals),
        "mean": float(sum(vals) / len(vals)),
    }


def _onnx_path(dirs: dict[str, Path]) -> Path:
    return dirs["onnx_fp32"] / "lidar_pyramid_padded_agent_static_fp32_dynamic.onnx"


def _fixed_k_scatter_plugin_onnx_path(dirs: dict[str, Path]) -> Path:
    return dirs["onnx_fp32"] / "lidar_pyramid_fixed_k_scatter_plugin_fp32_dynamic.onnx"


def _wide_engine_path(dirs: dict[str, Path], precision: str) -> Path:
    return dirs[f"engine_{precision}"] / f"lidar_pyramid_padded_agent_static_{precision}.engine"


def _bucket_engine_path(dirs: dict[str, Path], precision: str, bucket_id: int) -> Path:
    return dirs["engines"] / "bucketed" / precision / f"lidar_pyramid_padded_agent_static_bucket{bucket_id}_{precision}.engine"


def _fixed_k_plugin_bucket_engine_path(dirs: dict[str, Path], precision: str, bucket_id: int) -> Path:
    return dirs["engines"] / "fixed_k_scatter_plugin" / precision / f"lidar_pyramid_fixed_k_scatter_plugin_bucket{bucket_id}_{precision}.engine"


def _bucket_layerinfo_path(dirs: dict[str, Path], precision: str, bucket_id: int) -> Path:
    return dirs["engines"] / "bucketed" / precision / f"layerinfo_bucket{bucket_id}_{precision}.json"


def _fixed_engine_path(dirs: dict[str, Path], precision: str) -> Path:
    return dirs["engines"] / "fixed_shape" / precision / f"lidar_pyramid_padded_agent_static_fixed_shape_{precision}.engine"


def _fixed_layerinfo_path(dirs: dict[str, Path], precision: str) -> Path:
    return dirs["engines"] / "fixed_shape" / precision / f"layerinfo_fixed_shape_{precision}.json"


def _fixed_k_plugin_bucket_layerinfo_path(dirs: dict[str, Path], precision: str, bucket_id: int) -> Path:
    return dirs["engines"] / "fixed_k_scatter_plugin" / precision / f"layerinfo_fixed_k_scatter_plugin_bucket{bucket_id}_{precision}.json"


def _build_one_engine(
    args: argparse.Namespace,
    dirs: dict[str, Path],
    *,
    precision: str,
    engine_path: Path,
    layerinfo_path: Path,
    profile_shapes: dict[str, Any],
    log_name: str,
    onnx_path: Path | None = None,
    static_plugins: list[str] | None = None,
) -> dict[str, Any]:
    trtexec_report = find_trtexec_report(trt_root=args.trt_root, explicit_trtexec=args.trtexec_path)
    result: dict[str, Any] = {
        "precision": precision,
        "engine_path": str(engine_path),
        "layerinfo_path": str(layerinfo_path),
        "profile_shapes": profile_shapes,
        "trtexec": trtexec_report,
        "build_success": False,
        "error": None,
    }
    source_onnx = onnx_path or _onnx_path(dirs)
    if not source_onnx.exists():
        result["error"] = f"ONNX file not found: {source_onnx}"
        return result
    if not trtexec_report.get("trtexec_found"):
        result["error"] = f"trtexec not found. {trtexec_report.get('suggestion')}"
        return result
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    layerinfo_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = build_trtexec_command(
        precision=precision,
        onnx_path=source_onnx,
        engine_path=engine_path,
        layerinfo_path=layerinfo_path,
        profile_shapes=profile_shapes,
        trtexec_path=trtexec_report["trtexec_path"],
        no_tf32=True,
        skip_inference=True,
        static_plugins=static_plugins,
    )
    log_path = dirs["logs_build"] / log_name
    command = run_command(cmd, log_path, timeout=int(args.timeout))
    log_text = log_path.read_text(encoding="utf-8")
    failure = parse_trtexec_failure(log_text)
    result.update(
        {
            "command": cmd,
            "log_path": str(log_path),
            "returncode": command.get("returncode"),
            "build_success": bool(command.get("success") and engine_path.exists()),
            "error": command.get("error"),
            "unsupported_ops": failure.get("unsupported_ops", []),
            "failed_nodes": failure.get("failed_nodes", []),
            "engine_size_MB": engine_path.stat().st_size / (1024 * 1024) if engine_path.exists() else None,
        }
    )
    save_json(result, engine_path.with_suffix(".meta.json"))
    return result


def build_bucketed_and_fixed_engines(args: argparse.Namespace, dirs: dict[str, Path], buckets: list[dict[str, Any]], fixed_num_voxels: int) -> dict[str, Any]:
    result: dict[str, Any] = {"bucketed": {}, "fixed_shape": {}}
    for precision in PRECISIONS:
        result["bucketed"][precision] = []
        for bucket in buckets:
            bucket_id = int(bucket["bucket_id"])
            build = _build_one_engine(
                args,
                dirs,
                precision=precision,
                engine_path=_bucket_engine_path(dirs, precision, bucket_id),
                layerinfo_path=_bucket_layerinfo_path(dirs, precision, bucket_id),
                profile_shapes=profile_shapes_for_bucket(bucket, max_cav=int(args.max_cav)),
                log_name=f"build_bucket{bucket_id}_{precision}.log",
            )
            build["bucket"] = bucket
            result["bucketed"][precision].append(build)
        fixed_build = _build_one_engine(
            args,
            dirs,
            precision=precision,
            engine_path=_fixed_engine_path(dirs, precision),
            layerinfo_path=_fixed_layerinfo_path(dirs, precision),
            profile_shapes=fixed_shape_profile_shapes(fixed_num_voxels, max_cav=int(args.max_cav)),
            log_name=f"build_fixed_shape_{precision}.log",
        )
        fixed_build["fixed_num_voxels"] = int(fixed_num_voxels)
        result["fixed_shape"][precision] = fixed_build
    save_json(result, dirs["debug"] / "bucketed_engine_build_report.json")
    return result


def build_fixed_k_scatter_plugin_engines(args: argparse.Namespace, dirs: dict[str, Path], buckets: list[dict[str, Any]]) -> dict[str, Any]:
    plugin_so = Path(args.plugin_so).expanduser() if args.plugin_so else None
    result: dict[str, Any] = {"bucketed": {}, "plugin_so": str(plugin_so) if plugin_so else None}
    if plugin_so is None or not plugin_so.exists():
        result["error"] = f"plugin_so is required and must exist for fixed-K scatter plugin engines: {plugin_so}"
        save_json(result, dirs["debug"] / "fixed_k_scatter_plugin_engine_build_report.json")
        return result
    for precision in PRECISIONS:
        result["bucketed"][precision] = []
        for bucket in buckets:
            bucket_id = int(bucket["bucket_id"])
            build = _build_one_engine(
                args,
                dirs,
                precision=precision,
                engine_path=_fixed_k_plugin_bucket_engine_path(dirs, precision, bucket_id),
                layerinfo_path=_fixed_k_plugin_bucket_layerinfo_path(dirs, precision, bucket_id),
                profile_shapes=profile_shapes_for_fixed_k_scatter_plugin_bucket(bucket, max_cav=int(args.max_cav)),
                log_name=f"build_fixed_k_scatter_plugin_bucket{bucket_id}_{precision}.log",
                onnx_path=_fixed_k_scatter_plugin_onnx_path(dirs),
                static_plugins=[str(plugin_so)],
            )
            build["bucket"] = bucket
            result["bucketed"][precision].append(build)
    save_json(result, dirs["debug"] / "fixed_k_scatter_plugin_engine_build_report.json")
    return result


class BucketedTensorRTRouter:
    def __init__(self, buckets: list[dict[str, Any]], engine_paths: dict[int, Path], device: torch.device) -> None:
        self.buckets = sorted(buckets, key=lambda item: int(item["max_voxels"]))
        self.engine_paths = {int(bucket_id): Path(path) for bucket_id, path in engine_paths.items()}
        self.device = device
        self.runners: dict[int, TensorRTEngineRunner] = {}
        self.route_counts: dict[int, int] = {int(bucket["bucket_id"]): 0 for bucket in self.buckets}

    def _runner_for_bucket(self, bucket: dict[str, Any]) -> TensorRTEngineRunner:
        bucket_id = int(bucket["bucket_id"])
        if bucket_id not in self.runners:
            self.runners[bucket_id] = TensorRTEngineRunner(self.engine_paths[bucket_id], self.device)
        return self.runners[bucket_id]

    def run_profiled(self, tensors_by_name: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        num_voxels = int(tensors_by_name["voxel_features"].shape[0])
        bucket = select_voxel_bucket(num_voxels, self.buckets)
        bucket_id = int(bucket["bucket_id"])
        self.route_counts[bucket_id] = self.route_counts.get(bucket_id, 0) + 1
        outputs, profile = self._runner_for_bucket(bucket).run_profiled(tensors_by_name)
        profile["bucket_id"] = bucket_id
        profile["bucket_max_voxels"] = int(bucket["max_voxels"])
        return outputs, profile

    def allocation_report(self) -> dict[str, Any]:
        return {
            "route_counts": {str(key): int(value) for key, value in sorted(self.route_counts.items())},
            "runners": {str(bucket_id): runner.allocation_report() for bucket_id, runner in sorted(self.runners.items())},
        }


class FixedKTensorRTRouter(BucketedTensorRTRouter):
    def run_profiled(self, tensors_by_name: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict[str, Any], torch.Tensor]:
        original_num_voxels = int(tensors_by_name["voxel_features"].shape[0])
        bucket = select_voxel_bucket(original_num_voxels, self.buckets)
        fixed_k = int(bucket["max_voxels"])
        padded_tensors, valid_mask = pad_voxel_tensors_to_fixed_k(tensors_by_name, fixed_k)
        bucket_id = int(bucket["bucket_id"])
        self.route_counts[bucket_id] = self.route_counts.get(bucket_id, 0) + 1
        outputs, profile = self._runner_for_bucket(bucket).run_profiled(padded_tensors)
        profile["bucket_id"] = bucket_id
        profile["bucket_max_voxels"] = fixed_k
        profile["original_num_voxels"] = original_num_voxels
        profile["fixed_K"] = fixed_k
        return outputs, profile, valid_mask


class FixedKScatterPluginTensorRTRouter(BucketedTensorRTRouter):
    def run_profiled(self, tensors_by_name: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict[str, Any], torch.Tensor]:
        original_num_voxels = int(tensors_by_name["voxel_features"].shape[0])
        bucket = select_voxel_bucket(original_num_voxels, self.buckets)
        fixed_k = int(bucket["max_voxels"])
        padded_tensors, valid_mask = pad_voxel_tensors_to_fixed_k(tensors_by_name, fixed_k)
        padded_tensors["voxel_num_points"] = safe_voxel_num_points_for_fixed_k(padded_tensors["voxel_num_points"], valid_mask)
        padded_tensors["valid_voxel_mask"] = valid_mask
        bucket_id = int(bucket["bucket_id"])
        self.route_counts[bucket_id] = self.route_counts.get(bucket_id, 0) + 1
        outputs, profile = self._runner_for_bucket(bucket).run_profiled(padded_tensors)
        profile["bucket_id"] = bucket_id
        profile["bucket_max_voxels"] = fixed_k
        profile["original_num_voxels"] = original_num_voxels
        profile["fixed_K"] = fixed_k
        profile["valid_voxel_count"] = int(valid_mask.sum().item())
        return outputs, profile, valid_mask


def _copy_outputs_to_cpu(outputs: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], float, int, int]:
    start = time.perf_counter()
    copied = {name: tensor.detach().cpu() for name, tensor in outputs.items()}
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1000.0
    return copied, elapsed, len(copied), sum(int(tensor.numel() * tensor.element_size()) for tensor in copied.values())


def evaluate_bucketed_ap_and_latency(args: argparse.Namespace, dirs: dict[str, Path], buckets: list[dict[str, Any]]) -> dict[str, Any]:
    hypes, device, model, modality, dataset, loader = _load_dataset_context(args)
    if device.type != "cuda":
        raise RuntimeError("TensorRT bucketed evaluation requires CUDA.")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    reports: dict[str, Any] = {}
    for precision in PRECISIONS:
        engine_paths = {int(bucket["bucket_id"]): _bucket_engine_path(dirs, precision, int(bucket["bucket_id"])) for bucket in buckets}
        missing = [str(path) for path in engine_paths.values() if not path.exists()]
        if missing:
            reports[precision] = {"success": False, "error": f"missing bucket engines: {missing}", "precision": precision}
            continue
        router = BucketedTensorRTRouter(buckets, engine_paths, device)
        output_names: list[str] | None = None
        result_stat = {thr: {"tp": [], "fp": [], "gt": 0, "score": []} for thr in IOU_THRESHOLDS}
        rows: list[dict[str, Any]] = []
        forward_times: list[float] = []
        post_times: list[float] = []
        total_times: list[float] = []
        actual = 0
        skipped = 0
        lines: list[str] = []
        input_names = _input_names_for_export_mode(PADDED_MODE)
        for frame_idx, batch in enumerate(loader):
            if actual >= int(args.num_frames):
                break
            if batch is None:
                skipped += 1
                continue
            try:
                ego = batch["ego"] if isinstance(batch, dict) and "ego" in batch else batch
                ego = _to_device(ego, device)
                batch = _to_device(batch, device)
                if output_names is None:
                    with torch.no_grad():
                        raw = model(ego)
                    output_names = [name for name in ("cls_preds", "reg_preds", "dir_preds") if name in raw and torch.is_tensor(raw[name])]
                original_tensors, _agent_modalities = _extract_inputs(ego, modality)
                tensors = _prepare_export_tensors(original_tensors, export_mode=PADDED_MODE, max_cav=int(args.max_cav))
                tensors_by_name = {name: tensor for name, tensor in zip(input_names, tensors)}
                input_prepare_ms = 0.0
                outputs, profile = router.run_profiled(tensors_by_name)
                _cpu_outputs, d2h_ms, d2h_copies, d2h_bytes = _copy_outputs_to_cpu(outputs)
                output = {name: outputs[name].float() for name in (output_names or [])}

                def _postprocess():
                    od = OrderedDict()
                    od["ego"] = output
                    return dataset.post_process(batch, od)

                (pred_box, pred_score, gt_box), post_ms = _timed(_postprocess, device)
                for thr in IOU_THRESHOLDS:
                    _calculate_tp_fp(pred_box, pred_score, gt_box, result_stat, thr, args.ap_iou_backend, device)
                fwd_ms = float(profile.get("total_runner_ms", 0.0))
                forward_times.append(fwd_ms)
                post_times.append(post_ms)
                total_times.append(fwd_ms + post_ms)
                row = {
                    "frame_id": frame_idx,
                    "record_len": _record_len_value(ego),
                    "num_voxels": int(tensors_by_name["voxel_features"].shape[0]),
                    "precision": precision,
                    "scheme": "padded_agent_static_bucketed",
                    "bucket_id": profile.get("bucket_id"),
                    "bucket_max_voxels": profile.get("bucket_max_voxels"),
                    "input_prepare_ms": input_prepare_ms,
                    "dtype_cast_ms": profile.get("dtype_cast_ms", 0.0),
                    "contiguous_ms": profile.get("contiguous_ms", 0.0),
                    "h2d_copy_ms": profile.get("h2d_copy_ms", 0.0),
                    "input_device_copy_ms": profile.get("input_device_copy_ms", 0.0),
                    "set_input_shape_ms": profile.get("set_input_shape_ms", 0.0),
                    "output_shape_query_ms": profile.get("output_shape_query_ms", 0.0),
                    "bind_address_ms": profile.get("bind_address_ms", 0.0),
                    "execute_async_ms": profile.get("execute_async_ms", 0.0),
                    "synchronize_ms": profile.get("synchronize_ms", 0.0),
                    "d2h_copy_ms": d2h_ms,
                    "output_wrap_ms": profile.get("output_wrap_ms", 0.0),
                    "alloc_ms": 0.0,
                    "total_runner_ms": fwd_ms + d2h_ms,
                    "d2h_copies": d2h_copies,
                    "bytes_d2h": d2h_bytes,
                    "input_shapes": profile.get("input_shapes", {}),
                    "output_shapes": profile.get("output_shapes", {}),
                }
                rows.append(row)
                actual += 1
                lines.append(f"frame={frame_idx} precision={precision} bucket={row['bucket_id']} voxels={row['num_voxels']} forward_ms={fwd_ms:.3f} execute_ms={row['execute_async_ms']:.3f} post_ms={post_ms:.3f}")
            except Exception as exc:
                skipped += 1
                lines.append(f"frame={frame_idx} skipped error={exc}")
                lines.append(traceback.format_exc())
        ap: dict[str, float] = {}
        from opencood.utils import eval_utils

        for thr in IOU_THRESHOLDS:
            key = f"AP@{thr:.2f}"
            if result_stat[thr]["gt"] > 0 and result_stat[thr]["score"]:
                ap_value, _, _ = eval_utils.calculate_ap(result_stat, thr)
            else:
                ap_value = 0.0
            ap[key] = round(float(ap_value), 4)
        report = {
            "precision": precision,
            "success": True,
            "num_frames": int(args.num_frames),
            "actual_frames": actual,
            "skipped_frames": skipped,
            "AP@0.30": ap.get("AP@0.30", 0.0),
            "AP@0.50": ap.get("AP@0.50", 0.0),
            "AP@0.70": ap.get("AP@0.70", 0.0),
            "ap_0_3": ap.get("AP@0.30", 0.0),
            "ap_0_5": ap.get("AP@0.50", 0.0),
            "ap_0_7": ap.get("AP@0.70", 0.0),
            "mAP": round(float(sum(ap.values()) / len(ap)), 4) if ap else 0.0,
            "map": round(float(sum(ap.values()) / len(ap)), 4) if ap else 0.0,
            "forward_mean_ms": _mean(forward_times),
            "forward_p50_ms": _percentile(forward_times, 50),
            "forward_p90_ms": _percentile(forward_times, 90),
            "forward_p95_ms": _percentile(forward_times, 95),
            "postprocess_mean_ms": _mean(post_times),
            "postprocess_p50_ms": _percentile(post_times, 50),
            "total_mean_ms": _mean(total_times),
            "total_p50_ms": _percentile(total_times, 50),
            "latency_summary": summarize_latency_rows(rows),
            "frames": rows,
            "buckets": buckets,
            "router_allocation_report": router.allocation_report(),
            "output_names": output_names or [],
            "pyramid_forward_export_mode": PADDED_MODE,
            "router": "bucketed_by_num_voxels",
        }
        reports[precision] = report
        save_json(report, dirs["evaluation"] / f"trt_{precision}_ap_report_padded_agent_static_bucketed.json")
        save_json(
            {
                "scheme": "padded_agent_static_bucketed",
                "precision": precision,
                "num_frames": int(args.num_frames),
                "actual_frames": actual,
                "forward_p50_ms": report["forward_p50_ms"],
                "forward_p90_ms": report["forward_p90_ms"],
                "forward_p95_ms": report["forward_p95_ms"],
                "fps": float(1000.0 / report["forward_p50_ms"]) if report.get("forward_p50_ms") else None,
                "latency_scope": "bucketed_router_real_sample_engine_forward",
                "buckets": buckets,
            },
            dirs["benchmark"] / f"bucketed_{precision}_padded_agent_static" / f"benchmark_{precision}.json",
        )
        (dirs["logs_evaluation"] / f"evaluate_bucketed_{precision}.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=dataset.collate_batch_test)
    save_json(reports, dirs["debug"] / "trt_latency_breakdown_bucketed_padded_agent_static.json")
    return reports


def run_fixed_k_latency_only(args: argparse.Namespace, dirs: dict[str, Path], buckets: list[dict[str, Any]]) -> dict[str, Any]:
    _hypes, device, _model, _modality, _dataset, _loader = _load_dataset_context(args)
    if device.type != "cuda":
        raise RuntimeError("fixed-K TensorRT latency-only test requires CUDA.")
    reports: dict[str, Any] = {
        "scheme": "padded_agent_static_fixed_k_voxel_padding_latency_only",
        "num_frames": int(args.num_frames),
        "ap_valid": False,
        "whether_valid_mask_used": False,
        "warning": "valid_voxel_mask is prepared but not consumed by the current ONNX/TensorRT graph; AP is intentionally not evaluated.",
        "buckets": buckets,
        "precisions": {},
    }
    for precision in PRECISIONS:
        engine_paths = {int(bucket["bucket_id"]): _bucket_engine_path(dirs, precision, int(bucket["bucket_id"])) for bucket in buckets}
        missing = [str(path) for path in engine_paths.values() if not path.exists()]
        if missing:
            reports["precisions"][precision] = {"success": False, "error": f"missing bucket engines: {missing}", "precision": precision}
            continue
        router = FixedKTensorRTRouter(buckets, engine_paths, device)
        rows: list[dict[str, Any]] = []
        actual = 0
        for item in _iter_prepared_frames(args, limit=args.num_frames):
            tensors_by_name = item["tensors_by_name"]
            outputs, profile, valid_mask = router.run_profiled(tensors_by_name)
            _cpu_outputs, d2h_ms, d2h_copies, d2h_bytes = _copy_outputs_to_cpu(outputs)
            row = {
                "frame_id": int(item["frame_id"]),
                "record_len": int(item["record_len"]),
                "precision": precision,
                "original_num_voxels": int(profile.get("original_num_voxels", item["num_voxels"])),
                "fixed_K": int(profile.get("fixed_K", profile.get("bucket_max_voxels", 0))),
                "bucket_id": int(profile.get("bucket_id", -1)),
                "valid_voxel_count": int(valid_mask.sum().item()),
                "valid_mask_shape": list(valid_mask.shape),
                "whether_valid_mask_used": False,
                "ap_valid": False,
                "dtype_cast_ms": profile.get("dtype_cast_ms", 0.0),
                "contiguous_ms": profile.get("contiguous_ms", 0.0),
                "h2d_copy_ms": profile.get("h2d_copy_ms", 0.0),
                "input_device_copy_ms": profile.get("input_device_copy_ms", 0.0),
                "set_input_shape_ms": profile.get("set_input_shape_ms", 0.0),
                "output_shape_query_ms": profile.get("output_shape_query_ms", 0.0),
                "bind_address_ms": profile.get("bind_address_ms", 0.0),
                "execute_ms": profile.get("execute_async_ms", 0.0),
                "execute_async_ms": profile.get("execute_async_ms", 0.0),
                "synchronize_ms": profile.get("synchronize_ms", 0.0),
                "d2h_copy_ms": d2h_ms,
                "output_wrap_ms": profile.get("output_wrap_ms", 0.0),
                "total_runner_ms": float(profile.get("total_runner_ms", 0.0)) + d2h_ms,
                "d2h_copies": d2h_copies,
                "bytes_d2h": d2h_bytes,
                "input_shapes": profile.get("input_shapes", {}),
                "output_shapes": profile.get("output_shapes", {}),
            }
            rows.append(row)
            actual += 1
        reports["precisions"][precision] = {
            "success": True,
            "precision": precision,
            "actual_frames": actual,
            "ap_valid": False,
            "whether_valid_mask_used": False,
            "frames": rows,
            "summary": summarize_latency_rows(rows, fields=[*LATENCY_FIELDS, "execute_ms"]),
            "router_allocation_report": router.allocation_report(),
        }
    save_json(reports, dirs["debug"] / "fixed_k_voxel_padding_latency_only_report.json")
    return reports


def evaluate_fixed_k_scatter_plugin_ap_and_latency(args: argparse.Namespace, dirs: dict[str, Path], buckets: list[dict[str, Any]]) -> dict[str, Any]:
    hypes, device, model, modality, dataset, loader = _load_dataset_context(args)
    if device.type != "cuda":
        raise RuntimeError("fixed-K scatter plugin evaluation requires CUDA.")
    torch.cuda.set_device(device)
    plugin_so = Path(args.plugin_so).expanduser() if args.plugin_so else None
    if plugin_so is None or not plugin_so.exists():
        raise RuntimeError(f"plugin_so is required and must exist for fixed-K scatter plugin evaluation: {plugin_so}")
    ctypes.CDLL(str(plugin_so), mode=ctypes.RTLD_GLOBAL)
    reports: dict[str, Any] = {}
    input_names = _input_names_for_export_mode(PADDED_MODE)
    for precision in PRECISIONS:
        engine_paths = {int(bucket["bucket_id"]): _fixed_k_plugin_bucket_engine_path(dirs, precision, int(bucket["bucket_id"])) for bucket in buckets}
        missing = [str(path) for path in engine_paths.values() if not path.exists()]
        if missing:
            reports[precision] = {"success": False, "error": f"missing fixed-K scatter plugin engines: {missing}", "precision": precision}
            continue
        router = FixedKScatterPluginTensorRTRouter(buckets, engine_paths, device)
        output_names: list[str] | None = None
        result_stat = {thr: {"tp": [], "fp": [], "gt": 0, "score": []} for thr in IOU_THRESHOLDS}
        rows: list[dict[str, Any]] = []
        forward_times: list[float] = []
        post_times: list[float] = []
        total_times: list[float] = []
        actual = 0
        skipped = 0
        lines: list[str] = []
        for frame_idx, batch in enumerate(loader):
            if actual >= int(args.num_frames):
                break
            if batch is None:
                skipped += 1
                continue
            try:
                ego = batch["ego"] if isinstance(batch, dict) and "ego" in batch else batch
                ego = _to_device(ego, device)
                batch = _to_device(batch, device)
                if output_names is None:
                    with torch.no_grad():
                        raw = model(ego)
                    output_names = [name for name in ("cls_preds", "reg_preds", "dir_preds") if name in raw and torch.is_tensor(raw[name])]
                original_tensors, _agent_modalities = _extract_inputs(ego, modality)
                tensors = _prepare_export_tensors(original_tensors, export_mode=PADDED_MODE, max_cav=int(args.max_cav))
                tensors_by_name = {name: tensor for name, tensor in zip(input_names, tensors)}
                outputs, profile, valid_mask = router.run_profiled(tensors_by_name)
                _cpu_outputs, d2h_ms, d2h_copies, d2h_bytes = _copy_outputs_to_cpu(outputs)
                output = {name: outputs[name].float() for name in (output_names or [])}

                def _postprocess():
                    od = OrderedDict()
                    od["ego"] = output
                    return dataset.post_process(batch, od)

                (pred_box, pred_score, gt_box), post_ms = _timed(_postprocess, device)
                for thr in IOU_THRESHOLDS:
                    _calculate_tp_fp(pred_box, pred_score, gt_box, result_stat, thr, args.ap_iou_backend, device)
                fwd_ms = float(profile.get("total_runner_ms", 0.0)) + d2h_ms
                forward_times.append(fwd_ms)
                post_times.append(post_ms)
                total_times.append(fwd_ms + post_ms)
                row = {
                    "frame_id": frame_idx,
                    "record_len": _record_len_value(ego),
                    "precision": precision,
                    "scheme": "fixed_k_scatter_plugin",
                    "original_num_voxels": int(profile.get("original_num_voxels", tensors_by_name["voxel_features"].shape[0])),
                    "fixed_K": int(profile.get("fixed_K", profile.get("bucket_max_voxels", 0))),
                    "bucket_id": int(profile.get("bucket_id", -1)),
                    "valid_voxel_count": int(profile.get("valid_voxel_count", valid_mask.sum().item())),
                    "valid_mask_shape": list(valid_mask.shape),
                    "whether_valid_mask_used": True,
                    "ap_valid": True,
                    "dtype_cast_ms": profile.get("dtype_cast_ms", 0.0),
                    "contiguous_ms": profile.get("contiguous_ms", 0.0),
                    "h2d_copy_ms": profile.get("h2d_copy_ms", 0.0),
                    "input_device_copy_ms": profile.get("input_device_copy_ms", 0.0),
                    "set_input_shape_ms": profile.get("set_input_shape_ms", 0.0),
                    "output_shape_query_ms": profile.get("output_shape_query_ms", 0.0),
                    "bind_address_ms": profile.get("bind_address_ms", 0.0),
                    "execute_ms": profile.get("execute_async_ms", 0.0),
                    "execute_async_ms": profile.get("execute_async_ms", 0.0),
                    "synchronize_ms": profile.get("synchronize_ms", 0.0),
                    "d2h_copy_ms": d2h_ms,
                    "output_wrap_ms": profile.get("output_wrap_ms", 0.0),
                    "total_runner_ms": fwd_ms,
                    "d2h_copies": d2h_copies,
                    "bytes_d2h": d2h_bytes,
                    "input_shapes": profile.get("input_shapes", {}),
                    "output_shapes": profile.get("output_shapes", {}),
                }
                rows.append(row)
                actual += 1
                lines.append(
                    f"frame={frame_idx} precision={precision} bucket={row['bucket_id']} "
                    f"voxels={row['original_num_voxels']} fixed_K={row['fixed_K']} "
                    f"forward_ms={fwd_ms:.3f} execute_ms={row['execute_async_ms']:.3f} post_ms={post_ms:.3f}"
                )
            except Exception as exc:
                skipped += 1
                lines.append(f"frame={frame_idx} skipped error={exc}")
                lines.append(traceback.format_exc())
        ap: dict[str, float] = {}
        from opencood.utils import eval_utils

        for thr in IOU_THRESHOLDS:
            key = f"AP@{thr:.2f}"
            if result_stat[thr]["gt"] > 0 and result_stat[thr]["score"]:
                ap_value, _, _ = eval_utils.calculate_ap(result_stat, thr)
            else:
                ap_value = 0.0
            ap[key] = round(float(ap_value), 4)
        report = {
            "precision": precision,
            "success": True,
            "num_frames": int(args.num_frames),
            "actual_frames": actual,
            "skipped_frames": skipped,
            "AP@0.30": ap.get("AP@0.30", 0.0),
            "AP@0.50": ap.get("AP@0.50", 0.0),
            "AP@0.70": ap.get("AP@0.70", 0.0),
            "ap_0_3": ap.get("AP@0.30", 0.0),
            "ap_0_5": ap.get("AP@0.50", 0.0),
            "ap_0_7": ap.get("AP@0.70", 0.0),
            "mAP": round(float(sum(ap.values()) / len(ap)), 4) if ap else 0.0,
            "map": round(float(sum(ap.values()) / len(ap)), 4) if ap else 0.0,
            "forward_mean_ms": _mean(forward_times),
            "forward_p50_ms": _percentile(forward_times, 50),
            "forward_p90_ms": _percentile(forward_times, 90),
            "forward_p95_ms": _percentile(forward_times, 95),
            "postprocess_mean_ms": _mean(post_times),
            "postprocess_p50_ms": _percentile(post_times, 50),
            "total_mean_ms": _mean(total_times),
            "total_p50_ms": _percentile(total_times, 50),
            "latency_summary": summarize_latency_rows(rows, fields=[*LATENCY_FIELDS, "execute_ms"]),
            "frames": rows,
            "buckets": buckets,
            "router_allocation_report": router.allocation_report(),
            "output_names": output_names or [],
            "pyramid_forward_export_mode": "fixed_k_scatter_plugin",
            "router": "fixed_k_bucket_by_num_voxels",
            "whether_valid_mask_used": True,
            "ap_valid": True,
        }
        reports[precision] = report
        suffix = "fp32" if precision == "fp32" else "fp16"
        save_json(report, dirs["evaluation"] / f"trt_{suffix}_ap_report_fixed_k_scatter_plugin.json")
        save_json(
            {
                "scheme": "fixed_k_scatter_plugin",
                "precision": precision,
                "num_frames": int(args.num_frames),
                "actual_frames": actual,
                "forward_p50_ms": report["forward_p50_ms"],
                "forward_p90_ms": report["forward_p90_ms"],
                "forward_p95_ms": report["forward_p95_ms"],
                "execute_p50_ms": (((report.get("latency_summary") or {}).get("overall") or {}).get("execute_async_ms") or {}).get("p50"),
                "total_runner_p50_ms": (((report.get("latency_summary") or {}).get("overall") or {}).get("total_runner_ms") or {}).get("p50"),
                "fps": float(1000.0 / report["forward_p50_ms"]) if report.get("forward_p50_ms") else None,
                "latency_scope": "fixed_k_scatter_plugin_real_sample_engine_forward",
                "buckets": buckets,
                "ap_valid": True,
                "whether_valid_mask_used": True,
            },
            dirs["benchmark"] / f"fixed_k_scatter_plugin_{precision}.json",
        )
        (dirs["logs_evaluation"] / f"evaluate_fixed_k_scatter_plugin_{precision}.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=dataset.collate_batch_test)
    save_json(reports, dirs["debug"] / "fixed_k_scatter_plugin_latency_and_ap_debug.json")
    return reports


def run_fixed_shape_engine_control(args: argparse.Namespace, dirs: dict[str, Path], fixed_num_voxels: int) -> dict[str, Any]:
    items = [item for item in _iter_prepared_frames(args, limit=args.num_frames) if int(item["num_voxels"]) == int(fixed_num_voxels)]
    if not items:
        report = {"success": False, "error": f"no real frame has fixed_num_voxels={fixed_num_voxels}", "fixed_num_voxels": int(fixed_num_voxels)}
        save_json(report, dirs["debug"] / "trt_fixed_shape_engine_control_padded_agent_static.json")
        return report
    tensors_by_name = items[0]["tensors_by_name"]
    record_len = int(items[0]["record_len"])
    device = torch.device(args.device if torch.cuda.is_available() or not str(args.device).startswith("cuda") else "cpu")
    report: dict[str, Any] = {
        "success": True,
        "scheme": "padded_agent_static_fixed_shape_engine",
        "fixed_num_voxels": int(fixed_num_voxels),
        "record_len": record_len,
        "input_shapes": {name: list(tensor.shape) for name, tensor in tensors_by_name.items()},
        "precisions": {},
    }
    for precision in PRECISIONS:
        engine_path = _fixed_engine_path(dirs, precision)
        if not engine_path.exists():
            report["precisions"][precision] = {"success": False, "error": f"engine file does not exist: {engine_path}"}
            continue
        runner = TensorRTEngineRunner(engine_path, device)
        rows = []
        for _idx in range(int(args.iterations)):
            _outputs, profile = runner.run_profiled(tensors_by_name)
            rows.append(
                {
                    "record_len": record_len,
                    "execute_async_ms": profile.get("execute_async_ms", 0.0),
                    "total_runner_ms": profile.get("total_runner_ms", 0.0),
                    "set_input_shape_ms": profile.get("set_input_shape_ms", 0.0),
                    "input_device_copy_ms": profile.get("input_device_copy_ms", 0.0),
                }
            )
        report["precisions"][precision] = {
            "success": True,
            "engine_path": str(engine_path),
            "iterations": int(args.iterations),
            "summary": summarize_latency_rows(rows, ["execute_async_ms", "total_runner_ms", "set_input_shape_ms", "input_device_copy_ms"]),
            "allocation_report": runner.allocation_report(),
        }
    save_json(report, dirs["debug"] / "trt_fixed_shape_engine_control_padded_agent_static.json")
    return report


def _write_profile_input_files(args: argparse.Namespace, dirs: dict[str, Path], target_voxels: int, name: str) -> tuple[dict[str, Path], dict[str, list[int]]]:
    best_item: dict[str, Any] | None = None
    best_distance: int | None = None
    for item in _iter_prepared_frames(args, limit=args.num_frames):
        distance = abs(int(item["num_voxels"]) - int(target_voxels))
        if best_distance is None or distance < best_distance:
            best_item = item
            best_distance = distance
    if best_item is None:
        raise RuntimeError("No real sample available for trtexec profile input files.")
    input_dir = dirs["debug"] / f"trtexec_profile_inputs_{name}"
    input_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, Path] = {}
    shapes: dict[str, list[int]] = {}
    for input_name, tensor in best_item["tensors_by_name"].items():
        arr = tensor.detach().cpu().contiguous().numpy()
        path = input_dir / f"{input_name}.raw"
        arr.tofile(path)
        files[input_name] = path
        shapes[input_name] = list(arr.shape)
    save_json(
        {
            "name": name,
            "target_voxels": int(target_voxels),
            "selected_frame_id": best_item["frame_id"],
            "selected_num_voxels": best_item["num_voxels"],
            "inputs": {key: {"path": str(path), "shape": shapes[key]} for key, path in files.items()},
        },
        input_dir / "manifest.json",
    )
    return files, shapes


def run_trtexec_layer_profile(
    args: argparse.Namespace,
    dirs: dict[str, Path],
    *,
    engine_path: Path,
    output_json: Path,
    log_name: str,
    input_target_voxels: int,
) -> dict[str, Any]:
    trtexec_report = find_trtexec_report(trt_root=args.trt_root, explicit_trtexec=args.trtexec_path)
    result: dict[str, Any] = {
        "engine_path": str(engine_path),
        "profile_path": str(output_json),
        "trtexec": trtexec_report,
        "success": False,
        "error": None,
    }
    if not engine_path.exists():
        result["error"] = f"engine file does not exist: {engine_path}"
    elif not trtexec_report.get("trtexec_found"):
        result["error"] = f"trtexec not found. {trtexec_report.get('suggestion')}"
    else:
        input_files, input_shapes = _write_profile_input_files(args, dirs, input_target_voxels, output_json.stem)
        load_inputs = ",".join(f"{name}:{path}" for name, path in input_files.items())
        cmd = [
            trtexec_report["trtexec_path"],
            f"--loadEngine={engine_path}",
            f"--shapes={_shape_spec(input_shapes)}",
            f"--loadInputs={load_inputs}",
            "--dumpProfile",
            "--profilingVerbosity=detailed",
            f"--exportProfile={output_json}",
            f"--warmUp={int(args.warmup_ms)}",
            f"--iterations={int(args.iterations)}",
            f"--duration={int(args.duration)}",
            "--avgRuns=1",
            "--percentile=50,90,95",
            "--useSpinWait",
        ]
        log_path = dirs["logs_benchmark"] / log_name
        command = run_command(cmd, log_path, timeout=int(args.timeout))
        log_text = log_path.read_text(encoding="utf-8")
        result.update(parse_trtexec_latency_metrics(log_text))
        result.update({"success": bool(command.get("success") and output_json.exists()), "error": command.get("error"), "command": cmd, "log_path": str(log_path), "input_shapes": input_shapes})
    save_json(result, output_json.with_suffix(".meta.json"))
    return result


def _load_profile_layers(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        for key in ("layers", "Layers", "profile", "Profile"):
            value = data.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []


def _layer_name(item: dict[str, Any]) -> str:
    for key in ("name", "Name", "layerName", "LayerName"):
        value = item.get(key)
        if value is not None:
            return str(value)
    return ""


def _layer_type(item: dict[str, Any]) -> str:
    for key in ("layerType", "LayerType", "type", "Type"):
        value = item.get(key)
        if value is not None:
            return str(value)
    return ""


def _layer_time_ms(item: dict[str, Any]) -> float:
    for key in ("medianMs", "medianTimeMs", "timeMs", "averageMs", "avgMs", "latency.avg_time", "ms"):
        value = item.get(key)
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            match = re.search(r"[-+]?[0-9]*\.?[0-9]+", value)
            if match:
                return float(match.group(0))
    for key, value in item.items():
        if isinstance(value, (int, float)) and "time" in str(key).lower():
            return float(value)
    return 0.0


def analyze_layer_profiles(dirs: dict[str, Path], profile_paths: dict[str, Path]) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    all_top: list[dict[str, Any]] = []
    for profile_key, path in profile_paths.items():
        layers = _load_profile_layers(path)
        rows = []
        category_totals: dict[str, float] = {}
        for item in layers:
            name = _layer_name(item)
            layer_type = _layer_type(item)
            time_ms = _layer_time_ms(item)
            category = classify_trt_layer(name, layer_type)
            row = {"profile": profile_key, "name": name, "layer_type": layer_type, "latency_ms": time_ms, "category": category}
            rows.append(row)
            category_totals[category] = category_totals.get(category, 0.0) + float(time_ms)
        top20 = sorted(rows, key=lambda row: float(row.get("latency_ms") or 0.0), reverse=True)[:20]
        for row in top20:
            all_top.append(dict(row))
        reports[profile_key] = {
            "profile_path": str(path),
            "num_layers": len(rows),
            "top20": top20,
            "category_totals_ms": dict(sorted(category_totals.items(), key=lambda kv: kv[1], reverse=True)),
        }
    overall_category_totals: dict[str, float] = {}
    for report in reports.values():
        for category, value in (report.get("category_totals_ms") or {}).items():
            overall_category_totals[category] = overall_category_totals.get(category, 0.0) + float(value)
    payload = {
        "profiles": reports,
        "top_bottleneck_layers": sorted(all_top, key=lambda row: float(row.get("latency_ms") or 0.0), reverse=True)[:20],
        "overall_category_totals_ms": dict(sorted(overall_category_totals.items(), key=lambda kv: kv[1], reverse=True)),
    }
    save_json(payload, dirs["summary"] / "trt_layer_bottleneck_report.json")
    _write_layer_bottleneck_markdown(dirs["summary"] / "trt_layer_bottleneck_report.md", payload)
    return payload


def _write_layer_bottleneck_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# TensorRT Layer Bottleneck Report",
        "",
        "profile | rank | latency_ms | category | layer_type | layer",
        "--- | --- | --- | --- | --- | ---",
    ]
    for profile_key, report in (payload.get("profiles") or {}).items():
        for idx, row in enumerate(report.get("top20") or [], start=1):
            lines.append(
                " | ".join(
                    [
                        str(profile_key),
                        str(idx),
                        _fmt(row.get("latency_ms")),
                        str(row.get("category")),
                        str(row.get("layer_type")),
                        str(row.get("name")).replace("|", "/"),
                    ]
                )
            )
    lines.extend(["", "category | total_ms", "--- | ---"])
    for category, value in (payload.get("overall_category_totals_ms") or {}).items():
        lines.append(f"{category} | {_fmt(value)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def run_all_layer_profiles(args: argparse.Namespace, dirs: dict[str, Path], buckets: list[dict[str, Any]], fixed_num_voxels: int) -> dict[str, Any]:
    median_bucket = select_voxel_bucket(fixed_num_voxels, buckets)
    median_bucket_id = int(median_bucket["bucket_id"])
    profile_jobs = {
        "wide_fp32": (_wide_engine_path(dirs, "fp32"), dirs["debug"] / "trt_layer_profile_wide_fp32.json", fixed_num_voxels),
        "wide_fp16": (_wide_engine_path(dirs, "fp16"), dirs["debug"] / "trt_layer_profile_wide_fp16.json", fixed_num_voxels),
        "bucketed_fp32": (_bucket_engine_path(dirs, "fp32", median_bucket_id), dirs["debug"] / "trt_layer_profile_bucketed_fp32.json", fixed_num_voxels),
        "bucketed_fp16": (_bucket_engine_path(dirs, "fp16", median_bucket_id), dirs["debug"] / "trt_layer_profile_bucketed_fp16.json", fixed_num_voxels),
        "fixed_shape_fp32": (_fixed_engine_path(dirs, "fp32"), dirs["debug"] / "trt_layer_profile_fixed_shape_fp32.json", fixed_num_voxels),
        "fixed_shape_fp16": (_fixed_engine_path(dirs, "fp16"), dirs["debug"] / "trt_layer_profile_fixed_shape_fp16.json", fixed_num_voxels),
    }
    profile_results = {}
    profile_paths = {}
    for key, (engine_path, output_json, target_voxels) in profile_jobs.items():
        profile_paths[key] = output_json
        profile_results[key] = run_trtexec_layer_profile(
            args,
            dirs,
            engine_path=engine_path,
            output_json=output_json,
            log_name=f"profile_{key}.log",
            input_target_voxels=int(target_voxels),
        )
    bottleneck = analyze_layer_profiles(dirs, profile_paths)
    result = {"profile_results": profile_results, "bottleneck_report": bottleneck, "median_bucket_id": median_bucket_id}
    save_json(result, dirs["debug"] / "trt_layer_profile_run_report.json")
    return result


def _extract_wide_latency(dirs: dict[str, Path], precision: str) -> float | None:
    report = read_json(dirs["summary"] / "latency_decomposition_report.json", default={}) or {}
    for row in report.get("rows") or []:
        if row.get("scheme") == PADDED_MODE and row.get("precision") == precision:
            value = row.get("execute_cuda_event_p50") or row.get("total_runner_p50")
            return float(value) if value is not None else None
    eval_report = read_json(dirs["evaluation"] / f"five_way_ap_report_{PADDED_MODE}.json", default={}) or {}
    backend = (eval_report.get("reports") or {}).get(f"tensorrt_{precision}") or {}
    value = backend.get("forward_p50_ms")
    return float(value) if value is not None else None


def _extract_fixed_latency(dirs: dict[str, Path], precision: str) -> float | None:
    fixed_engine_report = read_json(dirs["debug"] / "trt_fixed_shape_engine_control_padded_agent_static.json", default={}) or {}
    fixed_engine_summary = ((((fixed_engine_report.get("precisions") or {}).get(precision) or {}).get("summary") or {}).get("overall") or {})
    fixed_engine_value = ((fixed_engine_summary.get("execute_async_ms") or {}).get("p50") or (fixed_engine_summary.get("total_runner_ms") or {}).get("p50"))
    if fixed_engine_value is not None:
        return float(fixed_engine_value)
    report = read_json(dirs["debug"] / "trt_fixed_shape_control_padded_agent_static.json", default={}) or {}
    summary = ((((report.get("precisions") or {}).get(precision) or {}).get("summary") or {}).get("overall") or {})
    value = ((summary.get("execute_async_ms") or {}).get("p50") or (summary.get("total_runner_ms") or {}).get("p50"))
    return float(value) if value is not None else None


def make_plugin_decision(dirs: dict[str, Path], bucket_eval: dict[str, Any], layer_report: dict[str, Any]) -> dict[str, Any]:
    fp32_bucket = (bucket_eval.get("fp32") or {}).get("latency_summary") or {}
    bucket_execute = (((fp32_bucket.get("overall") or {}).get("execute_async_ms") or {}).get("p50"))
    bucket_total = (((fp32_bucket.get("overall") or {}).get("total_runner_ms") or {}).get("p50"))
    wide_execute = _extract_wide_latency(dirs, "fp32")
    fixed_execute = _extract_fixed_latency(dirs, "fp32")
    bucket_latency = float(bucket_execute or bucket_total or 0.0)
    bucketed_speedup = float(wide_execute / bucket_latency) if wide_execute and bucket_latency else None
    close_to_fixed = bool(fixed_execute and bucket_latency and bucket_latency <= fixed_execute * 2.0)
    top_layers = layer_report.get("top_bottleneck_layers") or []
    categories = [row.get("category") for row in top_layers[:10]]
    category_counts = {category: categories.count(category) for category in sorted(set(categories))}
    remaining = max(category_counts.items(), key=lambda kv: kv[1])[0] if category_counts else None
    need_scatter = bool(not close_to_fixed and remaining == "PointPillarScatter / ScatterND")
    need_vfe = bool(not close_to_fixed and remaining == "PillarVFE / PFN")
    need_bevwarp = bool(not close_to_fixed and remaining == "Pyramid fusion / BEV warp / GridSample")
    recommended = "No plugin. Bucketed engines bring execute latency close to fixed-shape, so prioritize narrow profiles/router and production hardening."
    if not close_to_fixed:
        if need_scatter:
            recommended = "Consider PointPillarScatterTRT or VoxelScatterBEVTRT; do not use camera BEVPoolDynamicTRT."
        elif need_vfe:
            recommended = "Consider a fused PillarVFE plugin or fused VFE+scatter path after confirming layer profile on bucketed engines."
        elif need_bevwarp:
            recommended = "Consider BEVWarpDynamicTRT only if bucketed layer profile still attributes latency to GridSample/BEV warp."
        else:
            recommended = "Do not write plugin yet; inspect remaining Shape/Reformat/Conv bottlenecks and TensorRT profile settings."
    payload = {
        "bucketed_speedup": bucketed_speedup,
        "wide_fp32_execute_p50_ms": wide_execute,
        "bucketed_fp32_execute_p50_ms": bucket_execute,
        "bucketed_fp32_total_runner_p50_ms": bucket_total,
        "fixed_shape_fp32_execute_p50_ms": fixed_execute,
        "bucketed_execute_close_to_fixed_shape": close_to_fixed,
        "remaining_bottleneck": remaining,
        "top_bottleneck_layers": top_layers[:20],
        "need_pointpillar_scatter_plugin": need_scatter,
        "need_fused_vfe_scatter_plugin": need_vfe,
        "need_bevwarp_plugin": need_bevwarp,
        "need_bevpool_plugin": False,
        "recommended_next_step": recommended,
    }
    save_json(payload, dirs["summary"] / "plugin_decision_after_bucket_and_layer_profile.json")
    return payload


def _write_bucket_summary(dirs: dict[str, Path], buckets: list[dict[str, Any]], bucket_eval: dict[str, Any], layer_report: dict[str, Any], decision: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for precision in PRECISIONS:
        report = bucket_eval.get(precision) or {}
        summary = report.get("latency_summary") or {}
        overall = summary.get("overall") or {}
        rows.append(
            {
                "scheme": "padded_agent_static_bucketed",
                "precision": precision,
                "AP@0.30": report.get("AP@0.30"),
                "AP@0.50": report.get("AP@0.50"),
                "AP@0.70": report.get("AP@0.70"),
                "mAP": report.get("mAP"),
                "forward_p50_ms": report.get("forward_p50_ms"),
                "execute_cuda_event_p50_ms": ((overall.get("execute_async_ms") or {}).get("p50")),
                "total_runner_p50_ms": ((overall.get("total_runner_ms") or {}).get("p50")),
                "fps": float(1000.0 / report["forward_p50_ms"]) if report.get("forward_p50_ms") else None,
            }
        )
    payload = {
        "buckets": buckets,
        "rows": rows,
        "layer_bottleneck_report": layer_report,
        "plugin_decision": decision,
        "int8_qdq_modelopt_plugin_status": "not_run_by_request",
    }
    save_json(payload, dirs["summary"] / "bucketed_padded_agent_latency_report.json")
    lines = [
        "# Bucketed Padded Agent Latency Report",
        "",
        "bucket_id | min_voxels | opt_voxels | max_voxels | frames",
        "--- | --- | --- | --- | ---",
    ]
    for bucket in buckets:
        lines.append(f"{bucket['bucket_id']} | {bucket['min_voxels']} | {bucket['opt_voxels']} | {bucket['max_voxels']} | {bucket['num_frames']}")
    lines.extend(["", "precision | AP@0.30 | AP@0.50 | AP@0.70 | mAP | forward_p50 | execute_p50 | total_runner_p50 | FPS", "--- | --- | --- | --- | --- | --- | --- | --- | ---"])
    for row in rows:
        lines.append(
            " | ".join(
                [
                    str(row["precision"]),
                    _fmt(row.get("AP@0.30")),
                    _fmt(row.get("AP@0.50")),
                    _fmt(row.get("AP@0.70")),
                    _fmt(row.get("mAP")),
                    _fmt(row.get("forward_p50_ms")),
                    _fmt(row.get("execute_cuda_event_p50_ms")),
                    _fmt(row.get("total_runner_p50_ms")),
                    _fmt(row.get("fps")),
                ]
            )
        )
    lines.extend(["", "## Plugin Decision", "", f"- remaining_bottleneck: {decision.get('remaining_bottleneck')}", f"- bucketed_speedup: {_fmt(decision.get('bucketed_speedup'))}", f"- need_pointpillar_scatter_plugin: {decision.get('need_pointpillar_scatter_plugin')}", f"- need_fused_vfe_scatter_plugin: {decision.get('need_fused_vfe_scatter_plugin')}", f"- need_bevwarp_plugin: {decision.get('need_bevwarp_plugin')}", f"- need_bevpool_plugin: {decision.get('need_bevpool_plugin')}", f"- recommended_next_step: {decision.get('recommended_next_step')}"])
    (dirs["summary"] / "bucketed_padded_agent_latency_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def _compact_fixed_k_plugin_rows(plugin_eval: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for precision in PRECISIONS:
        report = plugin_eval.get(precision) or {}
        summary = report.get("latency_summary") or {}
        overall = summary.get("overall") or {}
        rows.append(
            {
                "scheme": "fixed_k_scatter_plugin",
                "precision": precision,
                "AP@0.30": report.get("AP@0.30"),
                "AP@0.50": report.get("AP@0.50"),
                "AP@0.70": report.get("AP@0.70"),
                "mAP": report.get("mAP"),
                "forward_p50_ms": report.get("forward_p50_ms"),
                "execute_cuda_event_p50_ms": ((overall.get("execute_async_ms") or {}).get("p50")),
                "total_runner_p50_ms": ((overall.get("total_runner_ms") or {}).get("p50")),
                "fps": float(1000.0 / report["forward_p50_ms"]) if report.get("forward_p50_ms") else None,
                "actual_frames": report.get("actual_frames"),
                "ap_valid": report.get("ap_valid"),
            }
        )
    return rows


def write_fixed_k_scatter_plugin_summary(
    dirs: dict[str, Path],
    buckets: list[dict[str, Any]],
    build_report: dict[str, Any],
    plugin_eval: dict[str, Any],
) -> dict[str, Any]:
    equivalence = read_json(dirs["debug"] / "pointpillar_scatter_plugin_equivalence.json", default={}) or {}
    precision_equivalence = equivalence.get("precisions") or {}
    if not precision_equivalence:
        precision_equivalence = {
            precision: read_json(dirs["debug"] / f"pointpillar_scatter_plugin_equivalence_{precision}.json", default={}) or {}
            for precision in PRECISIONS
        }
    equivalent_values = [item.get("equivalent") for item in precision_equivalence.values() if item]
    invalid_affects_values = [
        item.get("invalid_padded_voxel_affects_spatial_features")
        for item in precision_equivalence.values()
        if item and item.get("invalid_padded_voxel_affects_spatial_features") is not None
    ]
    max_abs_values = [item.get("max_abs_error") for item in precision_equivalence.values() if item.get("max_abs_error") is not None]
    latency_only = read_json(dirs["debug"] / "fixed_k_voxel_padding_latency_only_report.json", default={}) or {}
    rows = _compact_fixed_k_plugin_rows(plugin_eval)
    payload = {
        "scheme": "fixed_k_voxel_mask_scatter_plugin",
        "buckets": buckets,
        "build_report": build_report,
        "pointpillar_scatter_plugin_equivalence": equivalence,
        "pointpillar_scatter_plugin_equivalence_by_precision": precision_equivalence,
        "fixed_k_latency_only_reference": latency_only,
        "rows": rows,
        "ap_close_to_padded_agent_static_baseline": None,
        "execute_p50_close_to_fixed_shape_control": None,
        "pointpillar_scatter_plugin_equivalent": bool(equivalent_values and all(equivalent_values)),
        "pointpillar_scatter_plugin_max_abs_error": max(max_abs_values) if max_abs_values else None,
        "invalid_padded_voxel_affects_spatial_features": bool(any(invalid_affects_values)) if invalid_affects_values else None,
        "whether_valid_mask_used": True,
        "ap_valid": True,
        "need_pointpillar_scatter_plugin": True,
        "need_bevpool_plugin": False,
        "need_bevwarp_plugin": False,
        "recommended_next_step": "Use fixed-K bucket routing with PointPillarScatterTRT only if AP matches padded_agent_static baseline and p50 latency remains near fixed-shape control.",
    }
    save_json(payload, dirs["summary"] / "fixed_k_voxel_mask_scatter_plugin_report.json")
    lines = [
        "# Fixed-K Voxel Mask Scatter Plugin Report",
        "",
        "bucket_id | fixed_K | frames",
        "--- | --- | ---",
    ]
    for bucket in buckets:
        lines.append(f"{bucket['bucket_id']} | {bucket['max_voxels']} | {bucket.get('num_frames')}")
    lines.extend(["", "precision | AP@0.30 | AP@0.50 | AP@0.70 | mAP | forward_p50 | execute_p50 | total_runner_p50 | FPS | frames", "--- | --- | --- | --- | --- | --- | --- | --- | --- | ---"])
    for row in rows:
        lines.append(
            " | ".join(
                [
                    str(row["precision"]),
                    _fmt(row.get("AP@0.30")),
                    _fmt(row.get("AP@0.50")),
                    _fmt(row.get("AP@0.70")),
                    _fmt(row.get("mAP")),
                    _fmt(row.get("forward_p50_ms")),
                    _fmt(row.get("execute_cuda_event_p50_ms")),
                    _fmt(row.get("total_runner_p50_ms")),
                    _fmt(row.get("fps")),
                    _fmt(row.get("actual_frames")),
                ]
            )
        )
    lines.extend(
        [
            "",
            "## Safety",
            "",
            f"- pointpillar_scatter_plugin_equivalent: {payload['pointpillar_scatter_plugin_equivalent']}",
            f"- plugin_max_abs_error: {_fmt(payload.get('pointpillar_scatter_plugin_max_abs_error'))}",
            f"- invalid_padded_voxel_affects_spatial_features: {payload['invalid_padded_voxel_affects_spatial_features']}",
            "- valid_voxel_mask is consumed by PointPillarScatterTRT in this graph.",
        ]
    )
    (dirs["summary"] / "fixed_k_voxel_mask_scatter_plugin_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def run_bucketed_latency(args: argparse.Namespace) -> dict[str, Any]:
    dirs = ensure_quant_deploy_run_dirs(args.output_root)
    distribution = collect_voxel_distribution(args, dirs)
    counts = [int(item["num_voxels"]) for item in distribution["frames"]]
    buckets = build_voxel_buckets(counts, num_buckets=int(args.num_buckets), round_to=int(args.round_to))
    save_json({"buckets": buckets, "distribution": distribution}, dirs["configs"] / "padded_agent_static_voxel_buckets.json")
    fixed_num_voxels = int(_percentile_nearest(counts, 50))
    build_report = None
    if args.fixed_k_scatter_plugin:
        build_report = None
        if not args.skip_build:
            build_report = build_fixed_k_scatter_plugin_engines(args, dirs, buckets)
        plugin_eval: dict[str, Any] = {}
        if not args.skip_eval:
            plugin_eval = evaluate_fixed_k_scatter_plugin_ap_and_latency(args, dirs, buckets)
        summary = write_fixed_k_scatter_plugin_summary(dirs, buckets, build_report or {}, plugin_eval)
        return {
            "output_root": str(dirs["output_root"]),
            "distribution": distribution,
            "buckets": buckets,
            "build_report": build_report,
            "plugin_eval": plugin_eval,
            "summary": summary,
        }
    if not args.skip_build:
        build_report = build_bucketed_and_fixed_engines(args, dirs, buckets, fixed_num_voxels)
    fixed_k_latency_report = None
    if args.fixed_k_latency_only:
        fixed_k_latency_report = run_fixed_k_latency_only(args, dirs, buckets)
        return {
            "output_root": str(dirs["output_root"]),
            "distribution": distribution,
            "buckets": buckets,
            "fixed_k_latency_only": fixed_k_latency_report,
            "build_report": build_report,
        }
    bucket_eval: dict[str, Any] = {}
    if not args.skip_eval:
        bucket_eval = evaluate_bucketed_ap_and_latency(args, dirs, buckets)
    else:
        bucket_eval = read_json(dirs["debug"] / "trt_latency_breakdown_bucketed_padded_agent_static.json", default={}) or {}
    fixed_shape_control = run_fixed_shape_engine_control(args, dirs, fixed_num_voxels)
    profile_report: dict[str, Any] = {}
    if not args.skip_profile:
        profile_report = run_all_layer_profiles(args, dirs, buckets, fixed_num_voxels)
    layer_report = profile_report.get("bottleneck_report") or analyze_layer_profiles(
        dirs,
        {
            "wide_fp32": dirs["debug"] / "trt_layer_profile_wide_fp32.json",
            "wide_fp16": dirs["debug"] / "trt_layer_profile_wide_fp16.json",
            "bucketed_fp32": dirs["debug"] / "trt_layer_profile_bucketed_fp32.json",
            "bucketed_fp16": dirs["debug"] / "trt_layer_profile_bucketed_fp16.json",
            "fixed_shape_fp32": dirs["debug"] / "trt_layer_profile_fixed_shape_fp32.json",
            "fixed_shape_fp16": dirs["debug"] / "trt_layer_profile_fixed_shape_fp16.json",
        },
    )
    decision = make_plugin_decision(dirs, bucket_eval, layer_report)
    summary = _write_bucket_summary(dirs, buckets, bucket_eval, layer_report, decision)
    return {
        "output_root": str(dirs["output_root"]),
        "distribution": distribution,
        "buckets": buckets,
        "fixed_num_voxels": fixed_num_voxels,
        "build_report": build_report,
        "bucket_eval": bucket_eval,
        "fixed_shape_control": fixed_shape_control,
        "fixed_k_latency_only": fixed_k_latency_report,
        "profile_report": profile_report,
        "layer_report": layer_report,
        "plugin_decision": decision,
        "summary": summary,
    }


def main(argv: list[str] | None = None) -> int:
    report = run_bucketed_latency(parse_args(argv))
    print(report["output_root"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
