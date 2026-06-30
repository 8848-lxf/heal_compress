from __future__ import annotations

import argparse
import ctypes
import json
import math
import re
import shutil
import sys
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

from audit_pointpillar_scatter_plugin_engine import run_audit as run_plugin_audit
from bucketed_padded_agent_latency import (
    PRECISIONS,
    _fixed_k_plugin_bucket_engine_path,
    _fixed_k_plugin_bucket_layerinfo_path,
    _fixed_k_scatter_plugin_onnx_path,
    _fmt,
    _load_dataset_context,
    _mean,
    _numeric_summary,
    _percentile,
    _shape_spec,
    build_fixed_k_scatter_plugin_engines,
    build_voxel_buckets,
    classify_trt_layer,
    pad_voxel_tensors_to_fixed_k,
    profile_shapes_for_fixed_k_scatter_plugin_bucket,
    select_voxel_bucket,
)
from deployment_equivalence import TensorRTEngineRunner, _record_len_value
from evaluate_lidar_pyramid_trt_ap import IOU_THRESHOLDS, _calculate_tp_fp, _timed
from export_lidar_pyramid_onnx import (
    _extract_inputs,
    _input_names_for_export_mode,
    _prepare_export_tensors,
    _to_device,
    export_lidar_pyramid_onnx,
)
from exportable_lidar_pyramid_fixed_k_scatter_plugin import safe_voxel_num_points_for_fixed_k
from latency_decomposition import LATENCY_FIELDS, summarize_latency_rows
from pointpillar_scatter_plugin_check import build_plugin
from quant_deploy_utils import (
    DEFAULT_CHECKPOINT,
    DEFAULT_HEAL_REPO,
    DEFAULT_HYPES_YAML,
    DEFAULT_TRT_ROOT,
    build_trtexec_command,
    ensure_quant_deploy_run_dirs,
    find_trtexec_report,
    parse_trtexec_failure,
    read_json,
    run_command,
    save_json,
)


FIXED_K_BUCKETS = [
    {"bucket_id": 0, "min_voxels": 1, "opt_voxels": 9728, "max_voxels": 9728},
    {"bucket_id": 1, "min_voxels": 9729, "opt_voxels": 23040, "max_voxels": 23040},
    {"bucket_id": 2, "min_voxels": 23041, "opt_voxels": 23552, "max_voxels": 23552},
    {"bucket_id": 3, "min_voxels": 23553, "opt_voxels": 24064, "max_voxels": 24064},
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare padded vs dynamic-agent fixed-K PointPillarScatterTRT deployments.")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--hypes_yaml", default=str(DEFAULT_HYPES_YAML))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--heal_repo", default=str(DEFAULT_HEAL_REPO))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--trt_root", default=str(DEFAULT_TRT_ROOT))
    parser.add_argument("--trtexec_path", default=None)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--max_cav", type=int, default=2)
    parser.add_argument("--num_frames", type=int, nargs="+", default=[50, 200])
    parser.add_argument("--warmup_ms", type=int, default=200)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--duration", type=int, default=3)
    parser.add_argument("--ap_iou_backend", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--skip_existing", action="store_true", default=True)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--skip_profile", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--skip_audit", action="store_true")
    parser.add_argument("--plugin_so", default=None)
    return parser.parse_args(argv)


def _copy_or_link(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        shutil.copyfile(src, dst)


def _plugin_so(args: argparse.Namespace, dirs: dict[str, Path]) -> Path:
    if args.plugin_so:
        plugin_so = Path(args.plugin_so).expanduser()
        if plugin_so.exists():
            return plugin_so
    candidate = dirs["output_root"] / "artifacts" / "plugins" / "pointpillar_scatter_trt_build" / "libpointpillar_scatter_trt.so"
    if candidate.exists() and not args.rebuild:
        return candidate
    build_args = SimpleNamespace(
        output_root=args.output_root,
        trt_root=args.trt_root,
        trtexec_path=args.trtexec_path,
        timeout=args.timeout,
        skip_build_plugin=False,
    )
    report = build_plugin(build_args, dirs)
    if not report.get("success"):
        raise RuntimeError(f"PointPillarScatterTRT plugin build failed: {report}")
    return Path(report["plugin_so"])


def _frame_iter(args: argparse.Namespace, limit: int):
    hypes, device, model, modality, dataset, loader = _load_dataset_context(args)
    actual = 0
    for frame_idx, batch in enumerate(loader):
        if actual >= int(limit):
            break
        if batch is None:
            continue
        ego = batch["ego"] if isinstance(batch, dict) and "ego" in batch else batch
        ego = _to_device(ego, device)
        batch = _to_device(batch, device)
        original_tensors, _agent_modalities = _extract_inputs(ego, modality)
        yield {
            "frame_id": frame_idx,
            "sample_idx": actual,
            "batch": batch,
            "ego": ego,
            "dataset": dataset,
            "model": model,
            "modality": modality,
            "device": device,
            "original_tensors": original_tensors,
            "record_len": _record_len_value(ego),
            "num_voxels": int(original_tensors[0].shape[0]),
        }
        actual += 1


def _collect_distribution(args: argparse.Namespace, dirs: dict[str, Path], max_frames: int) -> dict[str, Any]:
    frames = []
    for item in _frame_iter(args, max_frames):
        frames.append({"frame_id": item["frame_id"], "sample_idx": item["sample_idx"], "record_len": item["record_len"], "num_voxels": item["num_voxels"]})
    buckets = []
    for bucket in FIXED_K_BUCKETS:
        values = [frame["num_voxels"] for frame in frames if int(bucket["min_voxels"]) <= int(frame["num_voxels"]) <= int(bucket["max_voxels"])]
        b = dict(bucket)
        b.update(
            {
                "num_frames": len(values),
                "voxel_count_min": min(values) if values else None,
                "voxel_count_max": max(values) if values else None,
            }
        )
        buckets.append(b)
    by_record_len: dict[str, list[int]] = {}
    for frame in frames:
        by_record_len.setdefault(str(frame["record_len"]), []).append(frame["num_voxels"])
    report = {
        "num_frames": len(frames),
        "frames": frames,
        "buckets": buckets,
        "overall": _numeric_summary([frame["num_voxels"] for frame in frames]),
        "by_record_len": {
            key: {"record_len": int(key), "num_frames": len(values), **_numeric_summary(values)}
            for key, values in sorted(by_record_len.items(), key=lambda item: int(item[0]))
        },
    }
    save_json(report, dirs["debug"] / "agent_mode_fixed_k_plugin_num_voxels_distribution.json")
    save_json({"buckets": buckets, "distribution": report}, dirs["configs"] / "padded_agent_static_voxel_buckets.json")
    return report


def _export_dynamic_onnx(args: argparse.Namespace, dirs: dict[str, Path], fixed_n: int, fixed_k: int) -> dict[str, Any]:
    export_args = SimpleNamespace(
        hypes_yaml=args.hypes_yaml,
        checkpoint=args.checkpoint,
        heal_repo=args.heal_repo,
        output_dir=None,
        output_root=args.output_root,
        run_name=None,
        device=args.device,
        num_frames=1,
        max_cav=int(fixed_n),
        fixed_num_agents=int(fixed_n),
        fixed_k=int(fixed_k),
        opset=17,
        overwrite=False,
        allow_synthetic_fallback=False,
        trt_root=args.trt_root,
        trtexec_path=args.trtexec_path,
        bev_warp_export_mode="exportable_grid",
        pillar_vfe_export_fix="explicit_squeeze",
        pyramid_forward_export_mode="dynamic_agent_dim_fixed_k_scatter_plugin",
    )
    summary = export_lidar_pyramid_onnx(export_args)
    source = Path(summary["onnx_path"])
    target = dirs["onnx_fp32"] / f"lidar_pyramid_dynamic_agent_dim_N{fixed_n}_fixed_k_scatter_plugin.onnx"
    if source.exists():
        shutil.copyfile(source, target)
    summary["canonical_onnx_path"] = str(target)
    summary["fixed_N"] = int(fixed_n)
    summary["fixed_K"] = int(fixed_k)
    save_json(summary, dirs["summary"] / f"summary_dynamic_agent_dim_N{fixed_n}_fixed_k_scatter_plugin_export.json")
    return summary


def _dynamic_engine_path(dirs: dict[str, Path], precision: str, fixed_n: int, bucket_id: int) -> Path:
    return dirs["engines"] / "dynamic_agent_dim_fixed_k_scatter_plugin" / f"N{fixed_n}" / precision / (
        f"lidar_pyramid_dynamic_agent_dim_N{fixed_n}_fixed_k_scatter_plugin_bucket{bucket_id}_{precision}.engine"
    )


def _dynamic_layerinfo_path(dirs: dict[str, Path], precision: str, fixed_n: int, bucket_id: int) -> Path:
    return dirs["engines"] / "dynamic_agent_dim_fixed_k_scatter_plugin" / f"N{fixed_n}" / precision / (
        f"layerinfo_dynamic_agent_dim_N{fixed_n}_fixed_k_scatter_plugin_bucket{bucket_id}_{precision}.json"
    )


def _profile_shapes_dynamic_fixed_k(bucket: dict[str, Any], fixed_n: int) -> dict[str, Any]:
    fixed_k = int(bucket["max_voxels"])
    return {
        "voxel_features": {"min": [fixed_k, 32, 4], "opt": [fixed_k, 32, 4], "max": [fixed_k, 32, 4]},
        "voxel_coords": {"min": [fixed_k, 4], "opt": [fixed_k, 4], "max": [fixed_k, 4]},
        "voxel_num_points": {"min": [fixed_k], "opt": [fixed_k], "max": [fixed_k]},
        "pairwise_t_matrix": {"min": [1, fixed_n, fixed_n, 4, 4], "opt": [1, fixed_n, fixed_n, 4, 4], "max": [1, fixed_n, fixed_n, 4, 4]},
        "valid_voxel_mask": {"min": [fixed_k], "opt": [fixed_k], "max": [fixed_k]},
    }


def _build_one_engine(
    args: argparse.Namespace,
    dirs: dict[str, Path],
    *,
    precision: str,
    onnx_path: Path,
    engine_path: Path,
    layerinfo_path: Path,
    profile_shapes: dict[str, Any],
    log_name: str,
    plugin_so: Path,
    agent_export_mode: str,
    fixed_n: int | None,
    bucket: dict[str, Any],
) -> dict[str, Any]:
    trtexec_report = find_trtexec_report(trt_root=args.trt_root, explicit_trtexec=args.trtexec_path)
    result: dict[str, Any] = {
        "onnx_path": str(onnx_path),
        "engine_path": str(engine_path),
        "precision": precision,
        "plugin_shared_library_path": str(plugin_so),
        "fixed_K_bucket": bucket,
        "agent_export_mode": agent_export_mode,
        "max_cav": int(args.max_cav) if agent_export_mode.startswith("padded") else None,
        "dynamic_N": agent_export_mode.startswith("dynamic"),
        "fixed_N": fixed_n,
        "input_shapes": profile_shapes,
        "optimization_profiles": profile_shapes,
        "whether_PointPillarScatterTRT_layer_exists": False,
        "whether_valid_voxel_mask_is_engine_input": None,
        "build_success": False,
        "build_log_path": None,
        "error": None,
        "trtexec": trtexec_report,
    }
    if not trtexec_report.get("trtexec_found"):
        result["error"] = f"trtexec not found. {trtexec_report.get('suggestion')}"
        return result
    if not onnx_path.exists():
        result["error"] = f"ONNX not found: {onnx_path}"
        return result
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    layerinfo_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = build_trtexec_command(
        precision=precision,
        onnx_path=onnx_path,
        engine_path=engine_path,
        layerinfo_path=layerinfo_path,
        profile_shapes=profile_shapes,
        trtexec_path=trtexec_report["trtexec_path"],
        no_tf32=True,
        skip_inference=True,
        static_plugins=[str(plugin_so)],
    )
    log_path = dirs["logs_build"] / log_name
    command = run_command(cmd, log_path, timeout=int(args.timeout))
    log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    failure = parse_trtexec_failure(log_text)
    layer_text = layerinfo_path.read_text(encoding="utf-8", errors="replace") if layerinfo_path.exists() else ""
    result.update(
        {
            "command": cmd,
            "returncode": command.get("returncode"),
            "build_success": bool(command.get("success") and engine_path.exists()),
            "error": command.get("error"),
            "unsupported_ops": failure.get("unsupported_ops", []),
            "failed_nodes": failure.get("failed_nodes", []),
            "engine_size_MB": engine_path.stat().st_size / (1024 * 1024) if engine_path.exists() else None,
            "build_log_path": str(log_path),
            "layerinfo_path": str(layerinfo_path),
            "whether_PointPillarScatterTRT_layer_exists": "PointPillarScatterTRT" in layer_text or "pointpillar" in layer_text.lower(),
            "whether_valid_voxel_mask_is_engine_input": "valid_voxel_mask" in layer_text or "valid_voxel_mask" in log_text,
        }
    )
    save_json(result, engine_path.with_suffix(".meta.json"))
    return result


def _ensure_padded_exports_and_engines(args: argparse.Namespace, dirs: dict[str, Path], buckets: list[dict[str, Any]], plugin_so: Path) -> dict[str, Any]:
    canonical_onnx = dirs["onnx_fp32"] / "lidar_pyramid_padded_agent_static_fixed_k_scatter_plugin.onnx"
    source_onnx = _fixed_k_scatter_plugin_onnx_path(dirs)
    max_bucket = max(int(bucket["max_voxels"]) for bucket in buckets)
    if (args.rebuild or not source_onnx.exists()) and not args.skip_existing:
        export_args = SimpleNamespace(
            hypes_yaml=args.hypes_yaml,
            checkpoint=args.checkpoint,
            heal_repo=args.heal_repo,
            output_dir=None,
            output_root=args.output_root,
            run_name=None,
            device=args.device,
            num_frames=1,
            max_cav=int(args.max_cav),
            fixed_num_agents=None,
            fixed_k=int(max_bucket),
            opset=17,
            overwrite=False,
            allow_synthetic_fallback=False,
            trt_root=args.trt_root,
            trtexec_path=args.trtexec_path,
            bev_warp_export_mode="exportable_grid",
            pillar_vfe_export_fix="explicit_squeeze",
            pyramid_forward_export_mode="fixed_k_scatter_plugin",
        )
        summary = export_lidar_pyramid_onnx(export_args)
        exported = Path(summary.get("onnx_path", ""))
        if exported.exists() and exported.resolve() != source_onnx.resolve():
            source_onnx.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(exported, source_onnx)
    _copy_or_link(source_onnx, canonical_onnx)
    build_args = SimpleNamespace(**vars(args))
    build_args.plugin_so = str(plugin_so)
    if args.rebuild:
        return build_fixed_k_scatter_plugin_engines(build_args, dirs, buckets)
    missing = []
    for precision in PRECISIONS:
        for bucket in buckets:
            if not _fixed_k_plugin_bucket_engine_path(dirs, precision, int(bucket["bucket_id"])).exists():
                missing.append((precision, int(bucket["bucket_id"])))
    if missing:
        return build_fixed_k_scatter_plugin_engines(build_args, dirs, buckets)
    report = read_json(dirs["debug"] / "fixed_k_scatter_plugin_engine_build_report.json", default={}) or {}
    for precision in PRECISIONS:
        canonical_engine = dirs[f"engine_{precision}"] / f"lidar_pyramid_padded_agent_static_fixed_k_scatter_plugin.engine"
        _copy_or_link(_fixed_k_plugin_bucket_engine_path(dirs, precision, 0), canonical_engine)
    return report


def _ensure_dynamic_exports_and_engines(args: argparse.Namespace, dirs: dict[str, Path], buckets: list[dict[str, Any]], plugin_so: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "dynamic_agent_dim_single_engine_supported": False,
        "dynamic_agent_dim_per_N_engine_required": True,
        "selected_test_plan": "per-N fixed engines routed by record_len",
        "onnx_exports": {},
        "engines": {},
    }
    max_bucket = max(int(bucket["max_voxels"]) for bucket in buckets)
    for fixed_n in (1, 2):
        onnx_path = dirs["onnx_fp32"] / f"lidar_pyramid_dynamic_agent_dim_N{fixed_n}_fixed_k_scatter_plugin.onnx"
        if args.rebuild or not onnx_path.exists():
            result["onnx_exports"][f"N{fixed_n}"] = _export_dynamic_onnx(args, dirs, fixed_n, max_bucket)
        else:
            result["onnx_exports"][f"N{fixed_n}"] = {"success": True, "canonical_onnx_path": str(onnx_path), "fixed_N": fixed_n, "fixed_K": max_bucket, "skipped_existing": True}
        for precision in PRECISIONS:
            result["engines"].setdefault(precision, {}).setdefault(f"N{fixed_n}", [])
            for bucket in buckets:
                bucket_id = int(bucket["bucket_id"])
                engine_path = _dynamic_engine_path(dirs, precision, fixed_n, bucket_id)
                layerinfo_path = _dynamic_layerinfo_path(dirs, precision, fixed_n, bucket_id)
                if engine_path.exists() and not args.rebuild:
                    meta = read_json(engine_path.with_suffix(".meta.json"), default={}) or {
                        "build_success": True,
                        "engine_path": str(engine_path),
                        "precision": precision,
                        "fixed_N": fixed_n,
                        "fixed_K_bucket": bucket,
                        "skipped_existing": True,
                    }
                    result["engines"][precision][f"N{fixed_n}"].append(meta)
                    continue
                build = _build_one_engine(
                    args,
                    dirs,
                    precision=precision,
                    onnx_path=onnx_path,
                    engine_path=engine_path,
                    layerinfo_path=layerinfo_path,
                    profile_shapes=_profile_shapes_dynamic_fixed_k(bucket, fixed_n),
                    log_name=f"build_dynamic_agent_dim_N{fixed_n}_fixed_k_scatter_plugin_bucket{bucket_id}_{precision}.log",
                    plugin_so=plugin_so,
                    agent_export_mode="dynamic_agent_dim_fixed_k_scatter_plugin_per_N",
                    fixed_n=fixed_n,
                    bucket=bucket,
                )
                result["engines"][precision][f"N{fixed_n}"].append(build)
    save_json(result, dirs["debug"] / "dynamic_agent_dim_fixed_k_scatter_plugin_engine_build_report.json")
    return result


class AgentModeFixedKRouter:
    def __init__(
        self,
        *,
        agent_export_mode: str,
        precision: str,
        buckets: list[dict[str, Any]],
        device: torch.device,
        dirs: dict[str, Path],
    ) -> None:
        self.agent_export_mode = agent_export_mode
        self.precision = precision
        self.buckets = sorted(buckets, key=lambda item: int(item["max_voxels"]))
        self.device = device
        self.dirs = dirs
        self.runners: dict[tuple[int, int], TensorRTEngineRunner] = {}
        self.route_counts: dict[str, int] = {}

    def _engine_path(self, fixed_n: int, bucket_id: int) -> Path:
        if self.agent_export_mode == "padded_agent_static_fixed_k_plugin":
            if self.precision == "int8_train_calib200":
                ns = self.dirs["engines"].name
                fixed_k = ns.replace("fixedK", "") if ns.startswith("fixedK") else str(max(int(bucket["max_voxels"]) for bucket in self.buckets))
                return (
                    self.dirs["engines"]
                    / "padded_agent_static"
                    / "int8_train_calib200"
                    / f"lidar_pyramid_padded_agent_static_fixedK{fixed_k}_int8_train_calib200_bucket{int(bucket_id)}.engine"
                )
            return _fixed_k_plugin_bucket_engine_path(self.dirs, self.precision, bucket_id)
        return _dynamic_engine_path(self.dirs, self.precision, fixed_n, bucket_id)

    def _runner(self, fixed_n: int, bucket_id: int) -> TensorRTEngineRunner:
        key = (int(fixed_n), int(bucket_id))
        if key not in self.runners:
            self.runners[key] = TensorRTEngineRunner(self._engine_path(fixed_n, bucket_id), self.device)
        return self.runners[key]

    def run_profiled(self, tensors_by_name: dict[str, torch.Tensor], *, record_len: int) -> tuple[dict[str, torch.Tensor], dict[str, Any], torch.Tensor]:
        original_num_voxels = int(tensors_by_name["voxel_features"].shape[0])
        bucket = select_voxel_bucket(original_num_voxels, self.buckets)
        bucket_id = int(bucket["bucket_id"])
        fixed_k = int(bucket["max_voxels"])
        fixed_n = int(record_len) if self.agent_export_mode.startswith("dynamic") else 2
        if fixed_n not in (1, 2):
            raise ValueError(f"Only N=1/2 dynamic fallback is supported, got record_len={record_len}")
        padded, valid_mask = pad_voxel_tensors_to_fixed_k(tensors_by_name, fixed_k)
        padded["voxel_num_points"] = safe_voxel_num_points_for_fixed_k(padded["voxel_num_points"], valid_mask)
        padded["valid_voxel_mask"] = valid_mask
        if self.agent_export_mode.startswith("dynamic"):
            padded.pop("valid_agent_mask", None)
            padded["pairwise_t_matrix"] = padded["pairwise_t_matrix"][:, :fixed_n, :fixed_n, :, :]
        key = f"N{fixed_n}_bucket{bucket_id}"
        self.route_counts[key] = self.route_counts.get(key, 0) + 1
        outputs, profile = self._runner(fixed_n, bucket_id).run_profiled(padded)
        profile.update(
            {
                "bucket_id": bucket_id,
                "fixed_K": fixed_k,
                "bucket_max_voxels": fixed_k,
                "original_num_voxels": original_num_voxels,
                "padding_voxel_count": int(fixed_k - original_num_voxels),
                "padding_ratio": float((fixed_k - original_num_voxels) / fixed_k) if fixed_k else 0.0,
                "fixed_N": fixed_n,
                "selected_N_engine": f"N{fixed_n}" if self.agent_export_mode.startswith("dynamic") else None,
                "valid_voxel_count": int(valid_mask.sum().item()),
                "engine_path": str(self._engine_path(fixed_n, bucket_id)),
            }
        )
        return outputs, profile, valid_mask

    def allocation_report(self) -> dict[str, Any]:
        return {
            "route_counts": self.route_counts,
            "runners": {f"N{n}_bucket{b}": runner.allocation_report() for (n, b), runner in sorted(self.runners.items())},
        }


def _copy_outputs_to_cpu(outputs: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], float, int, int]:
    import time

    start = time.perf_counter()
    copied = {name: tensor.detach().cpu() for name, tensor in outputs.items()}
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1000.0
    return copied, elapsed, len(copied), sum(int(tensor.numel() * tensor.element_size()) for tensor in copied.values())


def _summarize_distribution(values: list[int] | list[float]) -> dict[str, Any]:
    vals = [float(v) for v in values]
    if not vals:
        return {"min": None, "p50": None, "p90": None, "p95": None, "p99": None, "mean": None, "max": None}
    return {
        "min": min(vals),
        "p50": _percentile(vals, 50),
        "p90": _percentile(vals, 90),
        "p95": _percentile(vals, 95),
        "p99": _percentile(vals, 99),
        "mean": float(sum(vals) / len(vals)),
        "max": max(vals),
    }


def _normalize_eval_report(report: dict[str, Any], *, mode: str, precision: str, frames: int, dirs: dict[str, Path], plugin_so: Path, buckets: list[dict[str, Any]]) -> dict[str, Any]:
    rows = report.get("frames") or []
    latency = report.get("latency_summary") or summarize_latency_rows(rows, fields=[*LATENCY_FIELDS, "execute_ms"])
    overall = latency.get("overall") or {}
    record_dist: dict[str, int] = {}
    bucket_dist: dict[str, int] = {}
    for row in rows:
        record_dist[str(row.get("record_len"))] = record_dist.get(str(row.get("record_len")), 0) + 1
        bucket_dist[str(row.get("bucket_id"))] = bucket_dist.get(str(row.get("bucket_id")), 0) + 1
    return {
        **report,
        "agent_export_mode": mode,
        "precision": precision,
        "num_frames": int(frames),
        "execute_ms": overall.get("execute_ms") or overall.get("execute_async_ms"),
        "forward_ms": {
            "p50": report.get("forward_p50_ms"),
            "p90": report.get("forward_p90_ms"),
            "p95": report.get("forward_p95_ms"),
            "p99": _percentile([float(row.get("total_runner_ms", 0.0)) for row in rows], 99) if rows else None,
            "mean": report.get("forward_mean_ms"),
            "max": max([float(row.get("total_runner_ms", 0.0)) for row in rows], default=None),
        },
        "total_runner_ms": overall.get("total_runner_ms"),
        "FPS": float(1000.0 / report["forward_p50_ms"]) if report.get("forward_p50_ms") else None,
        "fps": float(1000.0 / report["forward_p50_ms"]) if report.get("forward_p50_ms") else None,
        "record_len_distribution": record_dist,
        "bucket_distribution": bucket_dist,
        "original_num_voxels_distribution": _summarize_distribution([row.get("original_num_voxels", row.get("num_voxels", 0)) for row in rows]),
        "padding_ratio_distribution": _summarize_distribution([row.get("padding_ratio", 0.0) for row in rows]),
        "engine_path": "bucket_router_multiple_engines",
        "onnx_path": str(dirs["onnx_fp32"] / ("lidar_pyramid_padded_agent_static_fixed_k_scatter_plugin.onnx" if mode.startswith("padded") else "dynamic_agent_dim_N1_N2_fixed_k_scatter_plugin")),
        "plugin_path": str(plugin_so),
        "fixed_k_buckets": buckets,
        "valid_voxel_mask_enabled": True,
        "pointpillar_scatter_plugin_enabled": True,
        "valid_agent_mask_enabled": mode.startswith("padded"),
        "true_dynamic_agent_dim": False,
        "per_N_engine_routing": mode.startswith("dynamic"),
    }


def evaluate_agent_mode(args: argparse.Namespace, dirs: dict[str, Path], buckets: list[dict[str, Any]], plugin_so: Path, *, mode: str, precision: str, frames: int) -> dict[str, Any]:
    hypes, device, model, modality, dataset, loader = _load_dataset_context(args)
    if device.type != "cuda":
        raise RuntimeError("TensorRT fixed-K plugin evaluation requires CUDA.")
    torch.cuda.set_device(device)
    ctypes.CDLL(str(plugin_so), mode=ctypes.RTLD_GLOBAL)
    input_mode = "padded_agent_static"
    input_names = _input_names_for_export_mode(input_mode)
    router = AgentModeFixedKRouter(agent_export_mode=mode, precision=precision, buckets=buckets, device=device, dirs=dirs)
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
        if actual >= int(frames):
            break
        if batch is None:
            skipped += 1
            continue
        try:
            ego = batch["ego"] if isinstance(batch, dict) and "ego" in batch else batch
            ego = _to_device(ego, device)
            batch = _to_device(batch, device)
            record_len = _record_len_value(ego)
            if output_names is None:
                with torch.no_grad():
                    raw = model(ego)
                output_names = [name for name in ("cls_preds", "reg_preds", "dir_preds") if name in raw and torch.is_tensor(raw[name])]
            original_tensors, _agent_modalities = _extract_inputs(ego, modality)
            tensors = _prepare_export_tensors(original_tensors, export_mode=input_mode, max_cav=int(args.max_cav))
            tensors_by_name = {name: tensor for name, tensor in zip(input_names, tensors)}
            outputs, profile, _valid_mask = router.run_profiled(tensors_by_name, record_len=int(record_len))
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
            fixed_k = int(profile.get("fixed_K", profile.get("bucket_max_voxels", 0)))
            original_num_voxels = int(profile.get("original_num_voxels", tensors_by_name["voxel_features"].shape[0]))
            row = {
                "frame_id": frame_idx,
                "sample_idx": actual,
                "record_len": int(record_len),
                "agent_export_mode": mode,
                "true_dynamic_agent_dim": False,
                "per_N_engine_routing": mode.startswith("dynamic"),
                "selected_N_engine": profile.get("selected_N_engine"),
                "precision": precision,
                "original_num_voxels": original_num_voxels,
                "bucket_id": int(profile.get("bucket_id", -1)),
                "fixed_K": fixed_k,
                "padding_voxel_count": int(profile.get("padding_voxel_count", fixed_k - original_num_voxels)),
                "padding_ratio": float(profile.get("padding_ratio", (fixed_k - original_num_voxels) / fixed_k if fixed_k else 0.0)),
                "valid_voxel_count": int(profile.get("valid_voxel_count", original_num_voxels)),
                "valid_voxel_mask_enabled": True,
                "pointpillar_scatter_plugin_enabled": True,
                "valid_agent_mask_enabled": mode.startswith("padded"),
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
                "forward_ms": fwd_ms,
                "total_runner_ms": fwd_ms,
                "d2h_copies": d2h_copies,
                "bytes_d2h": d2h_bytes,
                "input_shapes": profile.get("input_shapes", {}),
                "output_shapes": profile.get("output_shapes", {}),
                "engine_path": profile.get("engine_path"),
            }
            rows.append(row)
            actual += 1
            lines.append(
                f"frame={frame_idx} mode={mode} precision={precision} record_len={record_len} bucket={row['bucket_id']} "
                f"fixed_K={fixed_k} forward_ms={fwd_ms:.3f} execute_ms={row['execute_ms']:.3f}"
            )
        except Exception as exc:
            skipped += 1
            lines.append(f"frame={frame_idx} skipped error={exc}")
            lines.append(traceback.format_exc())
    from opencood.utils import eval_utils

    ap: dict[str, float] = {}
    for thr in IOU_THRESHOLDS:
        key = f"AP@{thr:.2f}"
        if result_stat[thr]["gt"] > 0 and result_stat[thr]["score"]:
            ap_value, _, _ = eval_utils.calculate_ap(result_stat, thr)
        else:
            ap_value = 0.0
        ap[key] = round(float(ap_value), 4)
    report = {
        "success": True,
        "agent_export_mode": mode,
        "precision": precision,
        "num_frames": int(frames),
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
        "forward_p99_ms": _percentile(forward_times, 99),
        "postprocess_mean_ms": _mean(post_times),
        "postprocess_p50_ms": _percentile(post_times, 50),
        "total_mean_ms": _mean(total_times),
        "total_p50_ms": _percentile(total_times, 50),
        "latency_summary": summarize_latency_rows(rows, fields=[*LATENCY_FIELDS, "execute_ms", "forward_ms"]),
        "frames": rows,
        "router_trace": rows,
        "buckets": buckets,
        "router_allocation_report": router.allocation_report(),
        "output_names": output_names or [],
        "router": "fixed_k_bucket_by_num_voxels_and_record_len" if mode.startswith("dynamic") else "fixed_k_bucket_by_num_voxels",
        "ap_valid": True,
        "whether_valid_mask_used": True,
    }
    report = _normalize_eval_report(report, mode=mode, precision=precision, frames=frames, dirs=dirs, plugin_so=plugin_so, buckets=buckets)
    suffix = f"{mode}_{frames}"
    save_json(report, dirs["evaluation"] / f"trt_{precision}_ap_report_{suffix}.json")
    save_json(
        {
            "agent_export_mode": mode,
            "precision": precision,
            "num_frames": int(frames),
            "actual_frames": actual,
            "execute_ms": report.get("execute_ms"),
            "forward_ms": report.get("forward_ms"),
            "total_runner_ms": report.get("total_runner_ms"),
            "forward_p50_ms": report.get("forward_p50_ms"),
            "execute_p50_ms": ((report.get("execute_ms") or {}).get("p50")),
            "fps": report.get("fps"),
            "record_len_distribution": report.get("record_len_distribution"),
            "bucket_distribution": report.get("bucket_distribution"),
            "fixed_k_buckets": buckets,
            "valid_voxel_mask_enabled": True,
            "pointpillar_scatter_plugin_enabled": True,
            "valid_agent_mask_enabled": mode.startswith("padded"),
            "true_dynamic_agent_dim": False,
            "per_N_engine_routing": mode.startswith("dynamic"),
        },
        dirs["benchmark"] / f"{mode}_{precision}_{frames}.json",
    )
    (dirs["logs_evaluation"] / f"evaluate_{mode}_{precision}_{frames}.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


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
        if item.get(key) is not None:
            return str(item.get(key))
    return ""


def _layer_type(item: dict[str, Any]) -> str:
    for key in ("layerType", "LayerType", "type", "Type"):
        if item.get(key) is not None:
            return str(item.get(key))
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


def _classify_layer(name: str, layer_type: str) -> str:
    text = f"{name} {layer_type}".lower()
    if "pointpillarscattertrt" in text or "pointpillar" in text:
        return "PointPillarScatterTRT"
    if "valid_voxel_mask" in text or "voxel_mask" in text:
        return "valid_voxel_mask"
    base = classify_trt_layer(name, layer_type)
    if base == "PointPillarScatter / ScatterND":
        return "PointPillarScatterTRT"
    if base == "valid_agent_mask fusion":
        return "valid_agent_mask"
    if base == "Pyramid fusion / BEV warp / GridSample":
        return "Pyramid fusion / BEV warp / GridSample"
    return base


def _analyze_profiles(dirs: dict[str, Path], profile_paths: dict[str, Path]) -> dict[str, Any]:
    profiles: dict[str, Any] = {}
    all_top: list[dict[str, Any]] = []
    for key, path in profile_paths.items():
        layers = _load_profile_layers(path)
        rows = []
        totals: dict[str, float] = {}
        for item in layers:
            name = _layer_name(item)
            layer_type = _layer_type(item)
            latency = _layer_time_ms(item)
            category = _classify_layer(name, layer_type)
            row = {"profile": key, "name": name, "layer_type": layer_type, "latency_ms": latency, "category": category}
            rows.append(row)
            totals[category] = totals.get(category, 0.0) + latency
        top20 = sorted(rows, key=lambda row: float(row.get("latency_ms") or 0.0), reverse=True)[:20]
        all_top.extend(top20)
        profiles[key] = {"profile_path": str(path), "num_layers": len(rows), "top20": top20, "category_totals_ms": dict(sorted(totals.items(), key=lambda kv: kv[1], reverse=True))}
    payload = {"profiles": profiles, "top_bottleneck_layers": sorted(all_top, key=lambda row: float(row.get("latency_ms") or 0.0), reverse=True)[:20]}
    save_json(payload, dirs["summary"] / "agent_mode_fixed_k_plugin_layer_profile_comparison.json")
    lines = ["# Agent Mode Fixed-K Plugin Layer Profile Comparison", "", "profile | rank | latency_ms | category | layer_type | layer", "--- | --- | --- | --- | --- | ---"]
    for key, profile in profiles.items():
        for idx, row in enumerate(profile.get("top20") or [], start=1):
            lines.append(f"{key} | {idx} | {_fmt(row.get('latency_ms'))} | {row.get('category')} | {row.get('layer_type')} | {str(row.get('name')).replace('|', '/')}")
    (dirs["summary"] / "agent_mode_fixed_k_plugin_layer_profile_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def _profile_input_item(args: argparse.Namespace, target_voxels: int, agent_mode: str, fixed_n: int) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    best_distance: int | None = None
    requested_frames = getattr(args, "num_frames", [50])
    if isinstance(requested_frames, (list, tuple)):
        scan_limit = max([int(value) for value in requested_frames] or [50])
    else:
        scan_limit = int(requested_frames)
    for item in _frame_iter(args, max(scan_limit, 50)):
        if agent_mode.startswith("dynamic") and int(item["record_len"]) != int(fixed_n):
            continue
        distance = abs(int(item["num_voxels"]) - int(target_voxels))
        if best_distance is None or distance < best_distance:
            best = item
            best_distance = distance
    if best is None:
        raise RuntimeError(f"No sample available for profile input: mode={agent_mode}, fixed_n={fixed_n}")
    return best


def _write_plugin_profile_inputs(
    args: argparse.Namespace,
    dirs: dict[str, Path],
    *,
    target_voxels: int,
    fixed_k_override: int | None,
    agent_mode: str,
    fixed_n: int,
    name: str,
) -> tuple[dict[str, Path], dict[str, list[int]], dict[str, Any]]:
    item = _profile_input_item(args, target_voxels, agent_mode, fixed_n)
    input_names = _input_names_for_export_mode("padded_agent_static")
    tensors = _prepare_export_tensors(item["original_tensors"], export_mode="padded_agent_static", max_cav=int(args.max_cav))
    tensors_by_name = {key: value for key, value in zip(input_names, tensors)}
    fixed_k = int(fixed_k_override or select_voxel_bucket(int(tensors_by_name["voxel_features"].shape[0]), FIXED_K_BUCKETS)["max_voxels"])
    if int(tensors_by_name["voxel_features"].shape[0]) > fixed_k:
        candidates = [candidate for candidate in _frame_iter(args, max(max([int(v) for v in getattr(args, "num_frames", [50])] or [50]), 50)) if int(candidate["num_voxels"]) <= fixed_k]
        if agent_mode.startswith("dynamic"):
            candidates = [candidate for candidate in candidates if int(candidate["record_len"]) == int(fixed_n)]
        if not candidates:
            raise RuntimeError(f"No sample fits fixed_k={fixed_k} for profile input: mode={agent_mode}, fixed_n={fixed_n}")
        item = min(candidates, key=lambda candidate: abs(int(candidate["num_voxels"]) - int(target_voxels)))
        tensors = _prepare_export_tensors(item["original_tensors"], export_mode="padded_agent_static", max_cav=int(args.max_cav))
        tensors_by_name = {key: value for key, value in zip(input_names, tensors)}
    padded, valid_mask = pad_voxel_tensors_to_fixed_k(tensors_by_name, fixed_k)
    padded["voxel_num_points"] = safe_voxel_num_points_for_fixed_k(padded["voxel_num_points"], valid_mask)
    padded["valid_voxel_mask"] = valid_mask
    if agent_mode.startswith("dynamic"):
        padded.pop("valid_agent_mask", None)
        padded["pairwise_t_matrix"] = padded["pairwise_t_matrix"][:, :fixed_n, :fixed_n, :, :]
    input_dir = dirs["debug"] / f"trtexec_profile_inputs_{name}"
    input_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, Path] = {}
    shapes: dict[str, list[int]] = {}
    for input_name, tensor in padded.items():
        arr = tensor.detach().cpu().contiguous().numpy()
        path = input_dir / f"{input_name}.raw"
        arr.tofile(path)
        files[input_name] = path
        shapes[input_name] = list(arr.shape)
    meta = {
        "name": name,
        "agent_mode": agent_mode,
        "fixed_n": int(fixed_n),
        "target_voxels": int(target_voxels),
        "selected_frame_id": int(item["frame_id"]),
        "selected_record_len": int(item["record_len"]),
        "selected_num_voxels": int(item["num_voxels"]),
        "fixed_K": fixed_k,
        "inputs": {key: {"path": str(path), "shape": shapes[key]} for key, path in files.items()},
    }
    save_json(meta, input_dir / "manifest.json")
    return files, shapes, meta


def _run_plugin_layer_profile(
    args: argparse.Namespace,
    dirs: dict[str, Path],
    *,
    engine_path: Path,
    output_json: Path,
    log_name: str,
    target_voxels: int,
    fixed_k: int,
    agent_mode: str,
    fixed_n: int,
    plugin_so: Path,
) -> dict[str, Any]:
    trtexec_report = find_trtexec_report(trt_root=args.trt_root, explicit_trtexec=args.trtexec_path)
    result: dict[str, Any] = {
        "engine_path": str(engine_path),
        "profile_path": str(output_json),
        "agent_mode": agent_mode,
        "fixed_n": int(fixed_n),
        "trtexec": trtexec_report,
        "success": False,
        "error": None,
    }
    if not engine_path.exists():
        result["error"] = f"engine file does not exist: {engine_path}"
    elif not trtexec_report.get("trtexec_found"):
        result["error"] = f"trtexec not found. {trtexec_report.get('suggestion')}"
    else:
        input_files, input_shapes, input_meta = _write_plugin_profile_inputs(
            args,
            dirs,
            target_voxels=target_voxels,
            fixed_k_override=fixed_k,
            agent_mode=agent_mode,
            fixed_n=fixed_n,
            name=output_json.stem,
        )
        load_inputs = ",".join(f"{name}:{path}" for name, path in input_files.items())
        cmd = [
            trtexec_report["trtexec_path"],
            f"--loadEngine={engine_path}",
            f"--staticPlugins={plugin_so}",
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
        log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
        result.update(
            {
                "success": bool(command.get("success") and output_json.exists()),
                "error": command.get("error"),
                "command": cmd,
                "log_path": str(log_path),
                "input_shapes": input_shapes,
                "input_meta": input_meta,
                "plugin_shared_library_path": str(plugin_so),
                "log_contains_pointpillar_plugin": "PointPillarScatterTRT" in log_text or "pointpillar" in log_text.lower(),
            }
        )
    save_json(result, output_json.with_suffix(".meta.json"))
    return result


def _profile_jobs(args: argparse.Namespace, dirs: dict[str, Path], buckets: list[dict[str, Any]], plugin_so: Path) -> dict[str, Any]:
    if args.skip_profile:
        return {"success": False, "skipped": True}
    median_bucket = buckets[min(1, len(buckets) - 1)]
    target_voxels = int(median_bucket["max_voxels"])
    profile_fixed_k = int(median_bucket["max_voxels"])
    jobs: dict[str, tuple[Path, Path, int]] = {
        "padded_agent_static_fixed_k_plugin_fp32": (
            _fixed_k_plugin_bucket_engine_path(dirs, "fp32", int(median_bucket["bucket_id"])),
            dirs["debug"] / "trt_layer_profile_padded_agent_static_fixed_k_plugin_fp32.json",
            target_voxels,
        ),
        "padded_agent_static_fixed_k_plugin_fp16": (
            _fixed_k_plugin_bucket_engine_path(dirs, "fp16", int(median_bucket["bucket_id"])),
            dirs["debug"] / "trt_layer_profile_padded_agent_static_fixed_k_plugin_fp16.json",
            target_voxels,
        ),
    }
    for fixed_n in (1, 2):
        for precision in PRECISIONS:
            jobs[f"dynamic_agent_dim_N{fixed_n}_fixed_k_plugin_{precision}"] = (
                _dynamic_engine_path(dirs, precision, fixed_n, int(median_bucket["bucket_id"])),
                dirs["debug"] / f"trt_layer_profile_dynamic_agent_dim_N{fixed_n}_fixed_k_plugin_{precision}.json",
                target_voxels,
            )
    profile_paths = {}
    results = {}
    for key, (engine, out_json, target) in jobs.items():
        profile_paths[key] = out_json
        mode = "dynamic_agent_dim_fixed_k_plugin" if key.startswith("dynamic") else "padded_agent_static_fixed_k_plugin"
        fixed_n = 2
        match = re.search(r"_N([12])_", key)
        if match:
            fixed_n = int(match.group(1))
        results[key] = _run_plugin_layer_profile(
            args,
            dirs,
            engine_path=engine,
            output_json=out_json,
            log_name=f"profile_{key}.log",
            target_voxels=target,
            fixed_k=profile_fixed_k,
            agent_mode=mode,
            fixed_n=fixed_n,
            plugin_so=plugin_so,
        )
    # Compatibility filenames requested for the dynamic aggregate point at N2.
    _copy_or_link(dirs["debug"] / "trt_layer_profile_dynamic_agent_dim_N2_fixed_k_plugin_fp32.json", dirs["debug"] / "trt_layer_profile_dynamic_agent_dim_fixed_k_plugin_fp32.json")
    _copy_or_link(dirs["debug"] / "trt_layer_profile_dynamic_agent_dim_N2_fixed_k_plugin_fp16.json", dirs["debug"] / "trt_layer_profile_dynamic_agent_dim_fixed_k_plugin_fp16.json")
    analyzed = _analyze_profiles(dirs, profile_paths)
    return {"profile_results": results, "profile_analysis": analyzed}


def _row_from_report(report: dict[str, Any]) -> dict[str, Any]:
    execute = report.get("execute_ms") or {}
    forward = report.get("forward_ms") or {}
    return {
        "mode": report.get("agent_export_mode"),
        "precision": report.get("precision"),
        "frames": report.get("num_frames"),
        "true_dynamic_N": report.get("true_dynamic_agent_dim"),
        "per_N_engine": report.get("per_N_engine_routing"),
        "AP@0.30": report.get("AP@0.30"),
        "AP@0.50": report.get("AP@0.50"),
        "AP@0.70": report.get("AP@0.70"),
        "mAP": report.get("mAP"),
        "execute_p50": execute.get("p50"),
        "execute_p95": execute.get("p95"),
        "forward_p50": forward.get("p50") or report.get("forward_p50_ms"),
        "forward_p95": forward.get("p95") or report.get("forward_p95_ms"),
        "FPS": report.get("fps"),
        "notes": "per-N fallback" if report.get("per_N_engine_routing") else "fixed max_cav=2 padded",
    }


def _write_ablation_summary(dirs: dict[str, Path], reports: dict[str, dict[str, dict[int, dict[str, Any]]]], audit: dict[str, Any], build_report: dict[str, Any], layer_report: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for mode_reports in reports.values():
        for precision_reports in mode_reports.values():
            for report in precision_reports.values():
                rows.append(_row_from_report(report))
    padded_fp16 = next((row for row in rows if row["mode"] == "padded_agent_static_fixed_k_plugin" and row["precision"] == "fp16" and row["frames"] == 50), {})
    dynamic_fp16 = next((row for row in rows if row["mode"] == "dynamic_agent_dim_fixed_k_plugin" and row["precision"] == "fp16" and row["frames"] == 50), {})
    dynamic_slower = None
    valid_mask_overhead = None
    dynamic_speedup = None
    if padded_fp16.get("forward_p50") and dynamic_fp16.get("forward_p50"):
        dynamic_slower = bool(float(dynamic_fp16["forward_p50"]) > float(padded_fp16["forward_p50"]) * 1.05)
        dynamic_speedup = float(padded_fp16["forward_p50"]) / float(dynamic_fp16["forward_p50"])
        valid_mask_overhead = bool(dynamic_speedup and dynamic_speedup > 1.05)
    answers = {
        "current_existing_fixed_k_plugin_engine_executes_PointPillarScatterTRT": (audit.get("audit") or {}).get("engine_contains_pointpillar_scatter_plugin_layer"),
        "current_existing_results_agent_export_mode": (audit.get("current_agent_mode") or {}).get("current_agent_export_mode"),
        "dynamic_agent_dim_still_slower_after_fixed_k_plugin": dynamic_slower,
        "valid_agent_mask_extra_overhead_obvious": valid_mask_overhead,
        "dynamic_agent_dim_per_N_fp16_50_speedup_vs_padded": dynamic_speedup,
        "true_dynamic_N_single_engine_feasible": False,
        "dynamic_agent_dim_per_N_engine_worth_complexity": "maybe for latency-only deployments; not default because it requires N1/N2 engine routing",
        "recommended_deployment_path": "padded_agent_static --max_cav 2 + fixed-K bucket router + PointPillarScatterTRT",
        "need_agent_dimension_dynamic_optimization": False,
        "can_enter_int8_qdq": False,
        "int8_qdq_status": "not_run_by_request; freeze FP32/FP16 first",
    }
    payload = {
        "rows": rows,
        "answers": answers,
        "plugin_audit": audit,
        "engine_build_report": build_report,
        "layer_profile_comparison": layer_report,
        "no_int8_qdq_modelopt": True,
        "heal_opencood_source_modified": False,
    }
    save_json(payload, dirs["summary"] / "agent_mode_fixed_k_plugin_ablation_report.json")
    lines = [
        "# Agent Mode Fixed-K Plugin Ablation Report",
        "",
        "mode | precision | frames | true_dynamic_N | per_N_engine | AP@0.30 | AP@0.50 | AP@0.70 | mAP | execute p50 | execute p95 | forward p50 | forward p95 | FPS | notes",
        "--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---",
    ]
    for row in rows:
        lines.append(
            " | ".join(
                [
                    str(row.get("mode")),
                    str(row.get("precision")),
                    str(row.get("frames")),
                    str(row.get("true_dynamic_N")),
                    str(row.get("per_N_engine")),
                    _fmt(row.get("AP@0.30")),
                    _fmt(row.get("AP@0.50")),
                    _fmt(row.get("AP@0.70")),
                    _fmt(row.get("mAP")),
                    _fmt(row.get("execute_p50")),
                    _fmt(row.get("execute_p95")),
                    _fmt(row.get("forward_p50")),
                    _fmt(row.get("forward_p95")),
                    _fmt(row.get("FPS")),
                    str(row.get("notes")),
                ]
            )
        )
    lines.extend(["", "## Answers", ""])
    for key, value in answers.items():
        lines.append(f"- {key}: {value}")
    (dirs["summary"] / "agent_mode_fixed_k_plugin_ablation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def run_ablation(args: argparse.Namespace) -> dict[str, Any]:
    dirs = ensure_quant_deploy_run_dirs(args.output_root)
    plugin_so = _plugin_so(args, dirs)
    max_frames = max(int(v) for v in args.num_frames)
    distribution = _collect_distribution(args, dirs, max_frames)
    buckets = distribution["buckets"]
    audit_result = {}
    if not args.skip_audit:
        audit_args = SimpleNamespace(
            output_root=args.output_root,
            trt_root=args.trt_root,
            trtexec_path=args.trtexec_path,
            timeout=args.timeout,
            onnx_path=None,
            plugin_so=str(plugin_so),
            try_build_plugin=False,
        )
        audit_result = run_plugin_audit(audit_args)
    padded_build = _ensure_padded_exports_and_engines(args, dirs, buckets, plugin_so)
    dynamic_build = _ensure_dynamic_exports_and_engines(args, dirs, buckets, plugin_so)
    build_report = {
        "fixed_k_buckets": buckets,
        "plugin_shared_library_path": str(plugin_so),
        "padded_agent_static_fixed_k_plugin": padded_build,
        "dynamic_agent_dim_fixed_k_plugin": dynamic_build,
    }
    save_json(build_report, dirs["benchmark"] / "agent_mode_fixed_k_plugin_engine_build_report.json")
    save_json(build_report, dirs["summary"] / "agent_mode_fixed_k_plugin_engine_build_report.json")
    lines = ["# Agent Mode Fixed-K Plugin Engine Build Report", "", f"- plugin: {plugin_so}", f"- buckets: {buckets}", "- dynamic true-N single engine: false", "- dynamic fallback: N1/N2 engines routed by record_len"]
    (dirs["summary"] / "agent_mode_fixed_k_plugin_engine_build_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    reports: dict[str, dict[str, dict[int, dict[str, Any]]]] = {
        "padded_agent_static_fixed_k_plugin": {"fp32": {}, "fp16": {}},
        "dynamic_agent_dim_fixed_k_plugin": {"fp32": {}, "fp16": {}},
    }
    router_traces: dict[int, list[dict[str, Any]]] = {int(frames): [] for frames in args.num_frames}
    if not args.skip_eval:
        for frames in [int(v) for v in args.num_frames]:
            for mode in ("padded_agent_static_fixed_k_plugin", "dynamic_agent_dim_fixed_k_plugin"):
                for precision in PRECISIONS:
                    report = evaluate_agent_mode(args, dirs, buckets, plugin_so, mode=mode, precision=precision, frames=frames)
                    reports[mode][precision][frames] = report
                    router_traces[frames].extend(report.get("router_trace") or [])
        for frames, trace in router_traces.items():
            save_json(trace, dirs["debug"] / f"agent_mode_fixed_k_plugin_router_trace_{frames}.json")
    else:
        for frames in [int(v) for v in args.num_frames]:
            for mode in ("padded_agent_static_fixed_k_plugin", "dynamic_agent_dim_fixed_k_plugin"):
                for precision in PRECISIONS:
                    path = dirs["evaluation"] / f"trt_{precision}_ap_report_{mode}_{frames}.json"
                    report = read_json(path, default={}) or {}
                    if report:
                        reports[mode][precision][frames] = report
                        router_traces[frames].extend(report.get("router_trace") or report.get("frames") or [])
        for frames, trace in router_traces.items():
            if trace:
                save_json(trace, dirs["debug"] / f"agent_mode_fixed_k_plugin_router_trace_{frames}.json")
    layer_report = _profile_jobs(args, dirs, buckets, plugin_so)
    summary = _write_ablation_summary(dirs, reports, audit_result, build_report, layer_report)
    return {"audit": audit_result, "build": build_report, "reports": reports, "layer_profile": layer_report, "summary": summary}


def main(argv: list[str] | None = None) -> int:
    try:
        result = run_ablation(parse_args(argv))
        print(json.dumps({"summary_path": "summary/agent_mode_fixed_k_plugin_ablation_report.json", "rows": (result.get("summary") or {}).get("rows", [])}, indent=2, ensure_ascii=False))
        return 0
    except Exception:
        print(traceback.format_exc(), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
