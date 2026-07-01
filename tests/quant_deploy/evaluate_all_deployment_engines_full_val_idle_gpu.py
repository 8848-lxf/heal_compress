from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
import time
import traceback
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from quant_deploy_utils import DEFAULT_CHECKPOINT, DEFAULT_HEAL_REPO, DEFAULT_HYPES_YAML, DEFAULT_OUTPUT_DIR, DEFAULT_TRT_ROOT, ensure_quant_deploy_run_dirs, save_json
from select_idle_gpu import (
    GpuTelemetryMonitor,
    query_gpu_snapshots,
    snapshot_to_dict,
    snapshots_to_dicts,
    wait_for_idle_gpu,
    write_gpu_selection_reports,
)


FIXED_K = 24064
FIXED_K_BUCKETS = [
    {"bucket_id": 0, "min_voxels": 1, "opt_voxels": 9728, "max_voxels": 9728},
    {"bucket_id": 1, "min_voxels": 9729, "opt_voxels": 23040, "max_voxels": 23040},
    {"bucket_id": 2, "min_voxels": 23041, "opt_voxels": 23552, "max_voxels": 23552},
    {"bucket_id": 3, "min_voxels": 23553, "opt_voxels": 24064, "max_voxels": 24064},
]


def buckets_for_fixed_k(fixed_k: int) -> list[dict[str, int]]:
    fixed_k = int(fixed_k)
    if fixed_k <= FIXED_K:
        return [dict(item) for item in FIXED_K_BUCKETS]
    buckets = [dict(item) for item in FIXED_K_BUCKETS[:-1]]
    buckets.append({"bucket_id": 3, "min_voxels": 23553, "opt_voxels": fixed_k, "max_voxels": fixed_k})
    return buckets


@dataclass(frozen=True)
class EvalMode:
    key: str
    scheme: str
    engine_strategy: str
    precision: str
    calibration_mode: str | None
    calibration_split: str | None
    runner_kind: str
    router_mode: str | None
    router_precision: str | None
    single_precision: str | None
    single_calibration_frames: int | None
    dynamic_N: bool
    bucket_router: bool
    N_engine_router: bool
    single_engine: bool
    fixed_K: int
    max_K: int


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate all built TensorRT deployment engines on full val using an idle GPU.")
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_DIR / "lidar_pyramid_agent_export_strategy_compare"))
    parser.add_argument("--plugin_path", default="tests/quant_deploy/outputs/lidar_pyramid_agent_export_strategy_compare/artifacts/plugins/pointpillar_scatter_trt_build/libpointpillar_scatter_trt.so")
    parser.add_argument("--hypes_yaml", default=str(DEFAULT_HYPES_YAML))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--heal_repo", default=str(DEFAULT_HEAL_REPO))
    parser.add_argument("--trt_root", default=str(DEFAULT_TRT_ROOT))
    parser.add_argument("--max_cav", type=int, default=2)
    parser.add_argument("--eval_split", default="val", choices=["val"])
    parser.add_argument("--eval_all", action="store_true", default=True)
    parser.add_argument("--latency_warmup_frames", type=int, default=20)
    parser.add_argument("--latency_repeat", type=int, default=1)
    parser.add_argument("--ap_iou_backend", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--gpu_idle_util_threshold", type=int, default=5)
    parser.add_argument("--gpu_idle_mem_threshold_mb", type=int, default=2000)
    parser.add_argument("--gpu_wait_timeout_sec", type=int, default=3600)
    parser.add_argument("--gpu_poll_interval_sec", type=int, default=30)
    parser.add_argument("--gpu_index", type=int, default=None)
    parser.add_argument("--allow_busy_gpu", action="store_true")
    parser.add_argument("--schemes", nargs="*", default=None, help="Optional subset of mode keys to run.")
    parser.add_argument("--summarize_existing", action="store_true", help="Regenerate full-val summary from existing per-mode reports without running inference.")
    parser.add_argument("--fixed_k", type=int, default=FIXED_K)
    parser.add_argument("--output_tag", default="full_val_idle_gpu")
    parser.add_argument("--fixedk_engine_namespace", action="store_true", help="Read ONNX/engine files from artifacts/onnx|engines/fixedK<fixed_k>.")
    parser.add_argument("--dynamic_int8_calibration_split", default=None, choices=[None, "train"])
    parser.add_argument("--include_mixed_heads_fp16", action="store_true")
    parser.add_argument("--force_gpu_index_no_nvidia_smi", type=int, default=None, help="Force a physical GPU id and skip all nvidia-smi based selection/telemetry.")
    parser.add_argument("--excluded_gpu_indices", default="", help="Comma-separated physical GPU ids excluded from selection, recorded in reports.")
    parser.add_argument("--progress_every_frames", type=int, default=1, help="Print and log progress every N frames. Default 1 prints every frame.")
    parser.add_argument("--no_progress_stdout", action="store_true", help="Write progress logs only to files, not stdout.")
    return parser.parse_args(argv)


def full_val_dirs(output_root: str | Path, *, output_tag: str = "full_val_idle_gpu", fixed_k: int = FIXED_K, fixedk_engine_namespace: bool = False) -> dict[str, Path]:
    dirs = ensure_quant_deploy_run_dirs(output_root)
    for key in ("evaluation", "benchmark", "debug", "summary"):
        dirs[f"{key}_full_val_idle_gpu"] = dirs[key] / str(output_tag)
        dirs[f"{key}_full_val_idle_gpu"].mkdir(parents=True, exist_ok=True)
    if fixedk_engine_namespace and int(fixed_k) != FIXED_K:
        namespace = f"fixedK{int(fixed_k)}"
        dirs["onnx"] = dirs["output_root"] / "artifacts" / "onnx" / namespace
        dirs["onnx_fp32"] = dirs["onnx"] / "fp32"
        dirs["engines"] = dirs["output_root"] / "artifacts" / "engines" / namespace
        dirs["engine_fp32"] = dirs["engines"] / "fp32"
        dirs["engine_fp16"] = dirs["engines"] / "fp16"
        dirs["engine_int8"] = dirs["engines"] / "int8"
        dirs["logs_build"] = dirs["output_root"] / "logs" / "build" / namespace
        for key in ("onnx", "onnx_fp32", "engines", "engine_fp32", "engine_fp16", "engine_int8", "logs_build"):
            dirs[key].mkdir(parents=True, exist_ok=True)
    return dirs


def eval_modes(*, fixed_k: int = FIXED_K, dynamic_int8_calibration_split: str | None = None, include_mixed_heads_fp16: bool = False) -> list[EvalMode]:
    fixed_k = int(fixed_k)
    modes = [
        EvalMode("padded_agent_static_fp32", "padded_agent_static", "padded_agent_static_fixed_k_plugin", "fp32", None, None, "router", "padded_agent_static_fixed_k_plugin", "fp32", None, None, False, True, False, False, fixed_k, fixed_k),
        EvalMode("padded_agent_static_fp16", "padded_agent_static", "padded_agent_static_fixed_k_plugin", "fp16", None, None, "router", "padded_agent_static_fixed_k_plugin", "fp16", None, None, False, True, False, False, fixed_k, fixed_k),
        EvalMode("padded_agent_static_int8_train_calib200", "padded_agent_static", "padded_agent_static_fixed_k_plugin", "int8", "train_calib200", "train", "router", "padded_agent_static_fixed_k_plugin", "int8_train_calib200", None, None, False, True, False, False, fixed_k, fixed_k),
        EvalMode("dynamic_bucket_fp32", "dynamic_agent_dim", "dynamic_agent_dim_bucket_fixed_k_plugin", "fp32", None, None, "router", "dynamic_agent_dim_fixed_k_plugin", "fp32", None, None, False, True, True, False, fixed_k, fixed_k),
        EvalMode("dynamic_bucket_fp16", "dynamic_agent_dim", "dynamic_agent_dim_bucket_fixed_k_plugin", "fp16", None, None, "router", "dynamic_agent_dim_fixed_k_plugin", "fp16", None, None, False, True, True, False, fixed_k, fixed_k),
        EvalMode("dynamic_bucket_int8_calib50", "dynamic_agent_dim", "dynamic_agent_dim_bucket_int8", "int8", "train_calib50" if dynamic_int8_calibration_split == "train" else "calib50", dynamic_int8_calibration_split, "router", "dynamic_agent_dim_fixed_k_plugin", "int8_calib50", None, None, False, True, True, False, fixed_k, fixed_k),
        EvalMode("dynamic_bucket_int8_calib200", "dynamic_agent_dim", "dynamic_agent_dim_bucket_int8", "int8", "train_calib200" if dynamic_int8_calibration_split == "train" else "calib200", dynamic_int8_calibration_split, "router", "dynamic_agent_dim_fixed_k_plugin", "int8_calib200", None, None, False, True, True, False, fixed_k, fixed_k),
        EvalMode("single_engine_maxK_fp32", "dynamic_agent_single_engine_maxK", "single TensorRT engine", "fp32", None, None, "single", None, None, "fp32", None, True, False, False, True, fixed_k, fixed_k),
        EvalMode("single_engine_maxK_fp16", "dynamic_agent_single_engine_maxK", "single TensorRT engine", "fp16", None, None, "single", None, None, "fp16", None, True, False, False, True, fixed_k, fixed_k),
        EvalMode("single_engine_maxK_int8_train_calib50", "dynamic_agent_single_engine_maxK", "single TensorRT engine", "int8", "train_calib50", "train", "single", None, None, "int8", 50, True, False, False, True, fixed_k, fixed_k),
        EvalMode("single_engine_maxK_int8_train_calib200", "dynamic_agent_single_engine_maxK", "single TensorRT engine", "int8", "train_calib200", "train", "single", None, None, "int8", 200, True, False, False, True, fixed_k, fixed_k),
    ]
    if include_mixed_heads_fp16:
        modes.insert(
            6,
            EvalMode("dynamic_bucket_int8_mixed_heads_fp16", "dynamic_agent_dim", "dynamic_agent_dim_bucket_int8", "int8", "calib200_mixed_heads_fp16", dynamic_int8_calibration_split, "router", "dynamic_agent_dim_fixed_k_plugin", "int8_calib200_mixed_heads_fp16", None, None, False, True, True, False, fixed_k, fixed_k),
        )
    return modes


def stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"p50": None, "p90": None, "p95": None, "p99": None, "mean": None, "max": None}
    ordered = sorted(float(v) for v in values)

    def pct(pct_value: float) -> float:
        idx = min(len(ordered) - 1, max(0, int(round((pct_value / 100.0) * (len(ordered) - 1)))))
        return float(ordered[idx])

    return {
        "p50": pct(50),
        "p90": pct(90),
        "p95": pct(95),
        "p99": pct(99),
        "mean": float(sum(ordered) / len(ordered)),
        "max": float(max(ordered)),
    }


class ProgressLogger:
    def __init__(self, dirs: dict[str, Path], *, stdout: bool = True, every_frames: int = 1) -> None:
        self.root = dirs["evaluation_full_val_idle_gpu"]
        self.stdout = bool(stdout)
        self.every_frames = max(1, int(every_frames or 1))
        self.text_path = self.root / "full_val_progress.log"
        self.jsonl_path = self.root / "full_val_progress.jsonl"
        self.root.mkdir(parents=True, exist_ok=True)

    def _paths_for_mode(self, mode: str | None) -> tuple[Path | None, Path | None]:
        if not mode:
            return None, None
        return self.root / f"{mode}_progress.log", self.root / f"{mode}_progress.jsonl"

    def emit(self, stage: str, message: str, *, mode: str | None = None, **fields: Any) -> None:
        ts = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        prefix = f"[{ts}] [{stage}]"
        if mode:
            prefix += f" [{mode}]"
        line = f"{prefix} {message}"
        record = {"time": ts, "stage": stage, "mode": mode, "message": message, **fields}
        if self.stdout:
            print(line, flush=True)
        with self.text_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        mode_text, mode_jsonl = self._paths_for_mode(mode)
        if mode_text is not None and mode_jsonl is not None:
            with mode_text.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
            with mode_jsonl.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def should_emit_frame(self, frame_idx: int) -> bool:
        return int(frame_idx) == 0 or (int(frame_idx) + 1) % self.every_frames == 0


def mode_report_name(mode: EvalMode) -> str:
    return f"{mode.key}_full_val.json"


def _copy_outputs_to_cpu(outputs: dict[str, Any], device: Any) -> tuple[dict[str, Any], float, int, int]:
    import torch

    start = time.perf_counter()
    copied = {name: tensor.detach().cpu() for name, tensor in outputs.items()}
    torch.cuda.synchronize(device)
    elapsed = (time.perf_counter() - start) * 1000.0
    return copied, elapsed, len(copied), sum(int(tensor.numel() * tensor.element_size()) for tensor in copied.values())


def _engine_paths_for_mode(dirs: dict[str, Path], mode: EvalMode, route_requirements: dict[str, Any]) -> tuple[list[Path], list[Path]]:
    if mode.runner_kind == "single":
        from dynamic_single_engine_maxk_common import engine_path as single_engine_path

        path = single_engine_path(dirs, str(mode.single_precision), mode.single_calibration_frames, fixed_k=int(mode.fixed_K))
        return [path] if path.exists() else [], [] if path.exists() else [path]
    if mode.runner_kind != "router":
        return [], []
    from agent_mode_fixed_k_plugin_ablation import _dynamic_engine_path
    from bucketed_padded_agent_latency import _fixed_k_plugin_bucket_engine_path

    required: list[Path] = []
    if mode.router_mode == "padded_agent_static_fixed_k_plugin":
        for bucket_id in sorted(route_requirements["required_buckets"]):
            if str(mode.router_precision) == "int8_train_calib200":
                required.append(
                    dirs["engines"]
                    / "padded_agent_static"
                    / "int8_train_calib200"
                    / f"lidar_pyramid_padded_agent_static_fixedK{int(mode.fixed_K)}_int8_train_calib200_bucket{int(bucket_id)}.engine"
                )
            else:
                required.append(_fixed_k_plugin_bucket_engine_path(dirs, str(mode.router_precision), int(bucket_id)))
    else:
        for fixed_n, bucket_id in sorted(route_requirements["required_dynamic_routes"]):
            required.append(_dynamic_engine_path(dirs, str(mode.router_precision), int(fixed_n), int(bucket_id)))
    existing = [path for path in required if path.exists()]
    missing = [path for path in required if not path.exists()]
    return existing, missing


def _onnx_path_for_mode(dirs: dict[str, Path], mode: EvalMode) -> str | None:
    if mode.runner_kind == "single":
        from dynamic_single_engine_maxk_common import onnx_path

        return str(onnx_path(dirs, fixed_k=int(mode.fixed_K)))
    if mode.router_mode == "padded_agent_static_fixed_k_plugin":
        return str(dirs["onnx_fp32"] / "lidar_pyramid_padded_agent_static_fixed_k_scatter_plugin.onnx")
    return str(dirs["onnx_fp32"] / "dynamic_agent_dim_N1_N2_fixed_k_scatter_plugin")


def _load_context(args: argparse.Namespace):
    from deployment_equivalence import _load_model_context
    from opencood.data_utils.datasets import build_dataset
    from torch.utils.data import DataLoader

    context_args = SimpleNamespace(
        hypes_yaml=args.hypes_yaml,
        checkpoint=args.checkpoint,
        heal_repo=args.heal_repo,
        device="cuda:0",
    )
    hypes, device, model, modality = _load_model_context(context_args)
    dataset = build_dataset(hypes, visualize=True, train=False)
    return hypes, device, model, modality, dataset, DataLoader


def _new_loader(dataset: Any, DataLoader: Any) -> Any:
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=dataset.collate_batch_test)


def scan_val_requirements(args: argparse.Namespace, context: tuple[Any, Any, Any, str, Any, Any], dirs: dict[str, Path], progress: ProgressLogger | None = None) -> dict[str, Any]:
    from bucketed_padded_agent_latency import select_voxel_bucket
    from deployment_equivalence import _record_len_value
    from export_lidar_pyramid_onnx import _extract_inputs

    _hypes, _device, _model, modality, dataset, DataLoader = context
    fixed_k = int(args.fixed_k)
    buckets = buckets_for_fixed_k(fixed_k)
    required_buckets: set[int] = set()
    required_dynamic_routes: set[tuple[int, int]] = set()
    samples: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    total_samples = len(dataset)
    if progress is not None:
        progress.emit(
            "scan_start",
            f"scanning validation split for K/route requirements: total={total_samples}, fixed_K={fixed_k}",
            total_val_samples=total_samples,
            fixed_K=fixed_k,
        )
    for frame_idx, batch in enumerate(_new_loader(dataset, DataLoader)):
        frame_start = time.perf_counter()
        if batch is None:
            skipped.append({"sample_idx": int(frame_idx), "reason": "batch_is_none"})
            if progress is not None and progress.should_emit_frame(frame_idx):
                progress.emit(
                    "scan_frame",
                    f"frame {frame_idx + 1}/{total_samples}: skipped batch_is_none",
                    sample_idx=int(frame_idx),
                    frame_number=int(frame_idx + 1),
                    total_val_samples=total_samples,
                    reason="batch_is_none",
                    elapsed_ms=(time.perf_counter() - frame_start) * 1000.0,
                )
            continue
        try:
            ego = batch["ego"] if isinstance(batch, dict) and "ego" in batch else batch
            original_tensors, _agent_modalities = _extract_inputs(ego, modality)
            k = int(original_tensors[0].shape[0])
            record_len = int(_record_len_value(ego))
            row = {"sample_idx": int(frame_idx), "record_len": record_len, "original_num_voxels": k}
            if k > fixed_k:
                skipped.append({**row, "reason": "K_exceeds_fixed_K"})
            else:
                bucket = select_voxel_bucket(k, buckets)
                bucket_id = int(bucket["bucket_id"])
                required_buckets.add(bucket_id)
                required_dynamic_routes.add((record_len, bucket_id))
                row["bucket_id"] = bucket_id
            samples.append(row)
            if progress is not None and progress.should_emit_frame(frame_idx):
                progress.emit(
                    "scan_frame",
                    f"frame {frame_idx + 1}/{total_samples}: K={k}, N={record_len}, bucket={row.get('bucket_id')}, skipped={k > fixed_k}",
                    sample_idx=int(frame_idx),
                    frame_number=int(frame_idx + 1),
                    total_val_samples=total_samples,
                    original_num_voxels=k,
                    record_len=record_len,
                    bucket_id=row.get("bucket_id"),
                    skipped=bool(k > fixed_k),
                    elapsed_ms=(time.perf_counter() - frame_start) * 1000.0,
                )
        except Exception as exc:
            skipped.append({"sample_idx": int(frame_idx), "reason": "scan_exception", "error": str(exc)})
            if progress is not None:
                progress.emit(
                    "scan_frame_error",
                    f"frame {frame_idx + 1}/{total_samples}: scan_exception: {exc}",
                    sample_idx=int(frame_idx),
                    frame_number=int(frame_idx + 1),
                    total_val_samples=total_samples,
                    error=str(exc),
                    elapsed_ms=(time.perf_counter() - frame_start) * 1000.0,
                )
    report = {
        "evaluation_split": "val",
        "total_val_samples": len(dataset),
        "scanned_samples": len(samples),
        "required_buckets": sorted(required_buckets),
        "required_dynamic_routes": [[n, b] for n, b in sorted(required_dynamic_routes)],
        "fixed_K": fixed_k,
        "fixed_k_buckets": buckets,
        "record_len_distribution": dict(Counter(str(row.get("record_len")) for row in samples if row.get("record_len") is not None)),
        "bucket_distribution": dict(Counter(str(row.get("bucket_id")) for row in samples if row.get("bucket_id") is not None)),
        "skipped_samples": skipped,
        "skip_reasons": dict(Counter(str(item.get("reason", "unknown")) for item in skipped)),
    }
    save_json(report, dirs["debug_full_val_idle_gpu"] / "full_val_route_requirements.json")
    if progress is not None:
        progress.emit(
            "scan_done",
            f"scan done: scanned={len(samples)}, skipped={len(skipped)}, required_buckets={sorted(required_buckets)}, required_dynamic_routes={sorted(required_dynamic_routes)}",
            scanned_samples=len(samples),
            skipped_samples=len(skipped),
            skip_reasons=report["skip_reasons"],
            required_buckets=sorted(required_buckets),
            required_dynamic_routes=[[n, b] for n, b in sorted(required_dynamic_routes)],
            output_path=str(dirs["debug_full_val_idle_gpu"] / "full_val_route_requirements.json"),
        )
    return {
        **report,
        "required_buckets": required_buckets,
        "required_dynamic_routes": required_dynamic_routes,
    }


def _wait_for_no_other_processes(args: argparse.Namespace, selected_gpu: Any, own_pid: int) -> dict[str, Any]:
    if getattr(args, "force_gpu_index_no_nvidia_smi", None) is not None:
        return {
            "ok": True,
            "checks": [],
            "forced_no_nvidia_smi": True,
            "note": "Skipped process check because nvidia-smi is disabled for this run.",
        }
    if getattr(args, "allow_busy_gpu", False):
        snapshots = query_gpu_snapshots()
        selected = next((item for item in snapshots if item.index == selected_gpu.index), None)
        other = []
        if selected is not None:
            other = [process for process in selected.processes if int(process.pid) != int(own_pid)]
        return {
            "ok": True,
            "allow_busy_gpu": True,
            "checks": [
                {
                    "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "snapshot": snapshot_to_dict(selected) if selected is not None else None,
                    "other_processes": [process.__dict__ for process in other],
                }
            ],
            "note": "Proceeding on user-approved busy GPU; latency may be marked unreliable if telemetry sees other processes.",
        }
    start = time.time()
    checks: list[dict[str, Any]] = []
    while True:
        snapshots = query_gpu_snapshots()
        selected = next((item for item in snapshots if item.index == selected_gpu.index), None)
        other = []
        if selected is not None:
            other = [process for process in selected.processes if int(process.pid) != int(own_pid)]
        checks.append(
            {
                "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "snapshot": snapshot_to_dict(selected) if selected is not None else None,
                "other_processes": [process.__dict__ for process in other],
            }
        )
        if not other:
            return {"ok": True, "checks": checks}
        if time.time() - start >= int(args.gpu_wait_timeout_sec):
            return {"ok": False, "checks": checks, "error": "other process remained on selected GPU until timeout"}
        time.sleep(max(1, int(args.gpu_poll_interval_sec)))


def _base_report(
    *,
    mode: EvalMode,
    args: argparse.Namespace,
    dirs: dict[str, Path],
    selected_gpu: Any,
    engine_paths: list[Path],
    missing_paths: list[Path],
    route_requirements: dict[str, Any],
    nvidia_smi_before: list[dict[str, Any]],
    nvidia_smi_after: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    total = int(route_requirements.get("total_val_samples") or 0)
    return {
        "scheme": mode.scheme,
        "engine_strategy": mode.engine_strategy,
        "precision": mode.precision,
        "calibration_mode": mode.calibration_mode,
        "calibration_split": mode.calibration_split,
        "evaluation_split": "val",
        "calibration_eval_overlap": False if mode.calibration_split == "train" else None,
        "full_val": True,
        "total_val_samples": total,
        "evaluated_samples": 0,
        "skipped_samples": list(route_requirements.get("skipped_samples") or []),
        "skipped_ratio": None,
        "skip_reasons": dict(route_requirements.get("skip_reasons") or {}),
        "AP@0.30": None,
        "AP@0.50": None,
        "AP@0.70": None,
        "mAP": None,
        "execute_ms": stats([]),
        "forward_ms": stats([]),
        "total_runner_ms": stats([]),
        "FPS": None,
        "engine_count": len(engine_paths),
        "engine_paths": [str(path) for path in engine_paths],
        "missing_engine_paths": [str(path) for path in missing_paths],
        "onnx_path": _onnx_path_for_mode(dirs, mode),
        "plugin_path": str(args.plugin_path),
        "fixed_K": int(mode.fixed_K),
        "max_K": int(mode.max_K),
        "dynamic_N": bool(mode.dynamic_N),
        "bucket_router": bool(mode.bucket_router),
        "N_engine_router": bool(mode.N_engine_router),
        "single_engine": bool(mode.single_engine),
        "valid_voxel_mask_enabled": True,
        "valid_agent_mask_enabled": mode.scheme == "padded_agent_static",
        "PointPillarScatterTRT_present": True,
        "selected_gpu": int(selected_gpu.index),
        "selected_gpu_physical_id": int(selected_gpu.index),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "CUDA_DEVICE_ORDER": os.environ.get("CUDA_DEVICE_ORDER"),
        "gpu_selection_method": "forced_no_nvidia_smi" if getattr(args, "force_gpu_index_no_nvidia_smi", None) is not None else "nvidia_smi_idle_selection",
        "excluded_gpu_indices": str(getattr(args, "excluded_gpu_indices", "")),
        "gpu_contention_detected": False,
        "unreliable_latency": False,
        "nvidia_smi_before": nvidia_smi_before,
        "nvidia_smi_after": nvidia_smi_after,
        "latency_warmup_frames": int(args.latency_warmup_frames),
        "latency_repeat": int(args.latency_repeat),
        "previous_results_under_gpu_contention_should_not_be_trusted": True,
        "output_tag": str(getattr(args, "output_tag", "full_val_idle_gpu")),
    }


def _write_mode_reports(dirs: dict[str, Path], mode: EvalMode, report: dict[str, Any], gpu_trace: dict[str, Any]) -> None:
    eval_path = dirs["evaluation_full_val_idle_gpu"] / mode_report_name(mode)
    bench_path = dirs["benchmark_full_val_idle_gpu"] / mode_report_name(mode)
    trace_path = dirs["debug_full_val_idle_gpu"] / f"{mode.key}_gpu_trace.json"
    save_json(report, eval_path)
    save_json(
        {
            key: report.get(key)
            for key in [
                "scheme",
                "engine_strategy",
                "precision",
                "calibration_mode",
                "calibration_split",
                "evaluation_split",
                "full_val",
                "status",
                "total_val_samples",
                "evaluated_samples",
                "skipped_samples",
                "skipped_ratio",
                "skip_reasons",
                "AP@0.30",
                "AP@0.50",
                "AP@0.70",
                "mAP",
                "execute_ms",
                "forward_ms",
                "total_runner_ms",
                "FPS",
                "engine_count",
                "engine_paths",
                "selected_gpu",
                "gpu_contention_detected",
                "unreliable_latency",
            ]
        },
        bench_path,
    )
    save_json(gpu_trace, trace_path)


def _finalize_report(report: dict[str, Any], *, rows: list[dict[str, Any]], latency_rows: list[dict[str, Any]], skipped: list[dict[str, Any]], ap: dict[str, float], monitor_report: dict[str, Any], nvidia_smi_after: list[dict[str, Any]]) -> dict[str, Any]:
    execute_values = [float(row.get("execute_ms", 0.0) or 0.0) for row in latency_rows]
    forward_values = [float(row.get("forward_ms", 0.0) or 0.0) for row in latency_rows]
    total_values = [float(row.get("total_runner_ms", 0.0) or 0.0) for row in latency_rows]
    evaluated = len(rows)
    total = int(report.get("total_val_samples") or 0)
    report.update(
        {
            "status": "success",
            "evaluated_samples": evaluated,
            "skipped_samples": skipped,
            "skipped_ratio": float(len(skipped) / total) if total else None,
            "skip_reasons": dict(Counter(str(item.get("reason", "unknown")) for item in skipped)),
            "AP@0.30": ap.get("AP@0.30", 0.0),
            "AP@0.50": ap.get("AP@0.50", 0.0),
            "AP@0.70": ap.get("AP@0.70", 0.0),
            "mAP": round(float(sum(ap.values()) / len(ap)), 4) if ap else 0.0,
            "execute_ms": stats(execute_values),
            "forward_ms": stats(forward_values),
            "total_runner_ms": stats(total_values),
            "FPS": float(1000.0 / stats(forward_values)["p50"]) if stats(forward_values)["p50"] else None,
            "latency_samples": len(latency_rows),
            "warmup_samples_excluded_from_latency": max(0, evaluated - len(latency_rows)),
            "frames": rows,
            "gpu_contention_detected": bool(monitor_report.get("gpu_contention_detected")),
            "unreliable_latency": bool(monitor_report.get("unreliable_latency")),
            "other_processes_detected": bool(monitor_report.get("other_processes_detected")),
            "contention_events": monitor_report.get("contention_events") or [],
            "nvidia_smi_after": nvidia_smi_after,
        }
    )
    return report


def evaluate_router_mode(args: argparse.Namespace, dirs: dict[str, Path], context: tuple[Any, Any, Any, str, Any, Any], mode: EvalMode, selected_gpu: Any, route_requirements: dict[str, Any], nvidia_smi_before: list[dict[str, Any]], progress: ProgressLogger | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch
    from agent_mode_fixed_k_plugin_ablation import AgentModeFixedKRouter
    from deployment_equivalence import _record_len_value
    from evaluate_lidar_pyramid_trt_ap import IOU_THRESHOLDS, _calculate_tp_fp, _timed
    from export_lidar_pyramid_onnx import _extract_inputs, _input_names_for_export_mode, _prepare_export_tensors, _to_device
    from opencood.utils import eval_utils

    engine_paths, missing_paths = _engine_paths_for_mode(dirs, mode, route_requirements)
    report = _base_report(mode=mode, args=args, dirs=dirs, selected_gpu=selected_gpu, engine_paths=engine_paths, missing_paths=missing_paths, route_requirements=route_requirements, nvidia_smi_before=nvidia_smi_before)
    if progress is not None:
        progress.emit(
            "mode_start",
            f"loading router mode: scheme={mode.scheme}, precision={mode.precision}, calibration={mode.calibration_mode}, engine_count={len(engine_paths)}",
            mode=mode.key,
            scheme=mode.scheme,
            precision=mode.precision,
            calibration_mode=mode.calibration_mode,
            engine_strategy=mode.engine_strategy,
            engine_paths=[str(path) for path in engine_paths],
        )
    if missing_paths:
        report.update({"status": "missing_engine", "error": "required route engine files are missing"})
        if progress is not None:
            progress.emit(
                "mode_missing",
                f"missing router engine files: {len(missing_paths)}",
                mode=mode.key,
                missing_engine_paths=[str(path) for path in missing_paths],
            )
        after = [] if getattr(args, "force_gpu_index_no_nvidia_smi", None) is not None else snapshots_to_dicts(query_gpu_snapshots())
        report["nvidia_smi_after"] = after
        trace = {"status": "missing_engine", "mode": mode.key, "missing_engine_paths": [str(path) for path in missing_paths], "nvidia_smi_before": nvidia_smi_before, "nvidia_smi_after": after}
        return report, trace
    _hypes, device, _model, modality, dataset, DataLoader = context
    fixed_k = int(mode.fixed_K)
    buckets = buckets_for_fixed_k(fixed_k)
    torch.cuda.set_device(device)
    ctypes.CDLL(str(Path(args.plugin_path).resolve()), mode=ctypes.RTLD_GLOBAL)
    router = AgentModeFixedKRouter(agent_export_mode=str(mode.router_mode), precision=str(mode.router_precision), buckets=buckets, device=device, dirs=dirs)
    input_names = _input_names_for_export_mode("padded_agent_static")
    result_stat = {thr: {"tp": [], "fp": [], "gt": 0, "score": []} for thr in IOU_THRESHOLDS}
    rows: list[dict[str, Any]] = []
    latency_rows: list[dict[str, Any]] = []
    skipped = [dict(item) for item in route_requirements.get("skipped_samples") or []]
    monitor = None if getattr(args, "force_gpu_index_no_nvidia_smi", None) is not None else GpuTelemetryMonitor(selected_gpu_index=selected_gpu.index, selected_gpu_uuid=selected_gpu.uuid, own_pid=os.getpid(), poll_interval_sec=args.gpu_poll_interval_sec)
    if monitor is not None:
        monitor.start()
    try:
        evaluated = 0
        total_samples = int(route_requirements.get("total_val_samples") or len(dataset))
        for frame_idx, batch in enumerate(_new_loader(dataset, DataLoader)):
            frame_start = time.perf_counter()
            if batch is None:
                skipped.append({"sample_idx": int(frame_idx), "reason": "batch_is_none"})
                if progress is not None and progress.should_emit_frame(frame_idx):
                    progress.emit(
                        "eval_frame",
                        f"frame {frame_idx + 1}/{total_samples}: skipped batch_is_none",
                        mode=mode.key,
                        sample_idx=int(frame_idx),
                        frame_number=int(frame_idx + 1),
                        total_val_samples=total_samples,
                        evaluated_samples=evaluated,
                        reason="batch_is_none",
                        elapsed_ms=(time.perf_counter() - frame_start) * 1000.0,
                    )
                continue
            try:
                ego = batch["ego"] if isinstance(batch, dict) and "ego" in batch else batch
                original_tensors_cpu, _agent_modalities = _extract_inputs(ego, modality)
                original_num_voxels = int(original_tensors_cpu[0].shape[0])
                if original_num_voxels > fixed_k:
                    if progress is not None and progress.should_emit_frame(frame_idx):
                        progress.emit(
                            "eval_frame",
                            f"frame {frame_idx + 1}/{total_samples}: skipped K_exceeds_fixed_K K={original_num_voxels}",
                            mode=mode.key,
                            sample_idx=int(frame_idx),
                            frame_number=int(frame_idx + 1),
                            total_val_samples=total_samples,
                            evaluated_samples=evaluated,
                            original_num_voxels=original_num_voxels,
                            reason="K_exceeds_fixed_K",
                            elapsed_ms=(time.perf_counter() - frame_start) * 1000.0,
                        )
                    continue
                ego = _to_device(ego, device)
                batch = _to_device(batch, device)
                original_tensors, _agent_modalities = _extract_inputs(ego, modality)
                tensors = _prepare_export_tensors(original_tensors, export_mode="padded_agent_static", max_cav=int(args.max_cav))
                tensors_by_name = {name: tensor for name, tensor in zip(input_names, tensors)}
                record_len = int(_record_len_value(ego))
                outputs, profile, _valid_mask = router.run_profiled(tensors_by_name, record_len=record_len)
                _cpu_outputs, d2h_ms, d2h_copies, d2h_bytes = _copy_outputs_to_cpu(outputs, device)
                output = {name: outputs[name].float() for name in ("cls_preds", "reg_preds", "dir_preds") if name in outputs}

                def _postprocess():
                    od = OrderedDict()
                    od["ego"] = output
                    return dataset.post_process(batch, od)

                (pred_box, pred_score, gt_box), post_ms = _timed(_postprocess, device)
                for thr in IOU_THRESHOLDS:
                    _calculate_tp_fp(pred_box, pred_score, gt_box, result_stat, thr, args.ap_iou_backend, device)
                fwd_ms = float(profile.get("total_runner_ms", 0.0)) + d2h_ms
                row = {
                    "sample_idx": int(frame_idx),
                    "record_len": record_len,
                    "original_num_voxels": original_num_voxels,
                    "bucket_id": int(profile.get("bucket_id", -1)),
                    "fixed_K": int(profile.get("fixed_K", 0)),
                    "padding_ratio": float(profile.get("padding_ratio", 0.0)),
                    "engine_path": profile.get("engine_path"),
                    "execute_ms": float(profile.get("execute_async_ms", 0.0)),
                    "execute_async_ms": float(profile.get("execute_async_ms", 0.0)),
                    "forward_ms": fwd_ms,
                    "total_runner_ms": fwd_ms,
                    "postprocess_ms": post_ms,
                    "d2h_copy_ms": d2h_ms,
                    "d2h_copies": d2h_copies,
                    "bytes_d2h": d2h_bytes,
                    "latency_warmup": evaluated < int(args.latency_warmup_frames),
                    "input_shapes": profile.get("input_shapes", {}),
                    "output_shapes": profile.get("output_shapes", {}),
                }
                rows.append(row)
                if not row["latency_warmup"]:
                    latency_rows.append(row)
                evaluated += 1
                if progress is not None and progress.should_emit_frame(frame_idx):
                    progress.emit(
                        "eval_frame",
                        (
                            f"frame {frame_idx + 1}/{total_samples}: evaluated={evaluated}, "
                            f"K={original_num_voxels}, N={record_len}, bucket={row['bucket_id']}, "
                            f"execute={row['execute_ms']:.4f} ms, forward={row['forward_ms']:.4f} ms, "
                            f"post={post_ms:.4f} ms"
                        ),
                        mode=mode.key,
                        sample_idx=int(frame_idx),
                        frame_number=int(frame_idx + 1),
                        total_val_samples=total_samples,
                        evaluated_samples=evaluated,
                        record_len=record_len,
                        original_num_voxels=original_num_voxels,
                        bucket_id=row["bucket_id"],
                        engine_path=row["engine_path"],
                        execute_ms=row["execute_ms"],
                        forward_ms=row["forward_ms"],
                        total_runner_ms=row["total_runner_ms"],
                        postprocess_ms=post_ms,
                        d2h_copy_ms=d2h_ms,
                        latency_warmup=row["latency_warmup"],
                        elapsed_ms=(time.perf_counter() - frame_start) * 1000.0,
                    )
            except Exception as exc:
                skipped.append({"sample_idx": int(frame_idx), "reason": "eval_exception", "error": str(exc), "traceback": traceback.format_exc()})
                if progress is not None:
                    progress.emit(
                        "eval_frame_error",
                        f"frame {frame_idx + 1}/{total_samples}: eval_exception: {exc}",
                        mode=mode.key,
                        sample_idx=int(frame_idx),
                        frame_number=int(frame_idx + 1),
                        total_val_samples=total_samples,
                        error=str(exc),
                        elapsed_ms=(time.perf_counter() - frame_start) * 1000.0,
                    )
    finally:
        if monitor is not None:
            monitor.stop()
    ap: dict[str, float] = {}
    for thr in IOU_THRESHOLDS:
        key = f"AP@{thr:.2f}"
        if result_stat[thr]["gt"] > 0 and result_stat[thr]["score"]:
            ap_value, _, _ = eval_utils.calculate_ap(result_stat, thr)
        else:
            ap_value = 0.0
        ap[key] = round(float(ap_value), 4)
    after = [] if getattr(args, "force_gpu_index_no_nvidia_smi", None) is not None else snapshots_to_dicts(query_gpu_snapshots())
    trace = (
        {
            "selected_gpu": int(selected_gpu.index),
            "selected_gpu_uuid": selected_gpu.uuid,
            "polling_interval": int(args.gpu_poll_interval_sec),
            "samples": [],
            "contention_events": [],
            "other_processes_detected": None,
            "other_processes": [],
            "gpu_contention_detected": False,
            "unreliable_latency": False,
            "forced_no_nvidia_smi": True,
        }
        if monitor is None
        else monitor.report()
    )
    report = _finalize_report(report, rows=rows, latency_rows=latency_rows, skipped=skipped, ap=ap, monitor_report=trace, nvidia_smi_after=after)
    if progress is not None:
        progress.emit(
            "mode_done",
            f"mode done: evaluated={report.get('evaluated_samples')}/{report.get('total_val_samples')}, mAP={report.get('mAP')}, AP70={report.get('AP@0.70')}, forward_p50={(report.get('forward_ms') or {}).get('p50')}",
            mode=mode.key,
            evaluated_samples=report.get("evaluated_samples"),
            total_val_samples=report.get("total_val_samples"),
            skipped_samples=len(report.get("skipped_samples") or []),
            AP_030=report.get("AP@0.30"),
            AP_050=report.get("AP@0.50"),
            AP_070=report.get("AP@0.70"),
            mAP=report.get("mAP"),
            execute_ms=report.get("execute_ms"),
            forward_ms=report.get("forward_ms"),
            FPS=report.get("FPS"),
            gpu_contention_detected=report.get("gpu_contention_detected"),
            unreliable_latency=report.get("unreliable_latency"),
        )
    trace.update({"mode": mode.key, "nvidia_smi_before": nvidia_smi_before, "nvidia_smi_after": after})
    return report, trace


def evaluate_single_mode(args: argparse.Namespace, dirs: dict[str, Path], context: tuple[Any, Any, Any, str, Any, Any], mode: EvalMode, selected_gpu: Any, route_requirements: dict[str, Any], nvidia_smi_before: list[dict[str, Any]], progress: ProgressLogger | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch
    from deployment_equivalence import TensorRTEngineRunner
    from evaluate_lidar_pyramid_trt_ap import IOU_THRESHOLDS, _calculate_tp_fp, _timed
    from opencood.utils import eval_utils
    from run_dynamic_single_engine_maxk import _prepare_inputs

    engine_paths, missing_paths = _engine_paths_for_mode(dirs, mode, route_requirements)
    report = _base_report(mode=mode, args=args, dirs=dirs, selected_gpu=selected_gpu, engine_paths=engine_paths, missing_paths=missing_paths, route_requirements=route_requirements, nvidia_smi_before=nvidia_smi_before)
    if progress is not None:
        progress.emit(
            "mode_start",
            f"loading single-engine mode: scheme={mode.scheme}, precision={mode.precision}, calibration={mode.calibration_mode}, engine={engine_paths[0] if engine_paths else None}",
            mode=mode.key,
            scheme=mode.scheme,
            precision=mode.precision,
            calibration_mode=mode.calibration_mode,
            engine_strategy=mode.engine_strategy,
            engine_paths=[str(path) for path in engine_paths],
        )
    if missing_paths:
        report.update({"status": "missing_engine", "error": "single engine file is missing"})
        if progress is not None:
            progress.emit(
                "mode_missing",
                f"missing single engine file: {missing_paths[0] if missing_paths else None}",
                mode=mode.key,
                missing_engine_paths=[str(path) for path in missing_paths],
            )
        after = [] if getattr(args, "force_gpu_index_no_nvidia_smi", None) is not None else snapshots_to_dicts(query_gpu_snapshots())
        report["nvidia_smi_after"] = after
        trace = {"status": "missing_engine", "mode": mode.key, "missing_engine_paths": [str(path) for path in missing_paths], "nvidia_smi_before": nvidia_smi_before, "nvidia_smi_after": after}
        return report, trace
    from export_lidar_pyramid_onnx import _to_device

    _hypes, device, _model, _modality, dataset, DataLoader = context
    fixed_k = int(mode.fixed_K)
    torch.cuda.set_device(device)
    ctypes.CDLL(str(Path(args.plugin_path).resolve()), mode=ctypes.RTLD_GLOBAL)
    runner = TensorRTEngineRunner(engine_paths[0], device)
    result_stat = {thr: {"tp": [], "fp": [], "gt": 0, "score": []} for thr in IOU_THRESHOLDS}
    rows: list[dict[str, Any]] = []
    latency_rows: list[dict[str, Any]] = []
    skipped = [dict(item) for item in route_requirements.get("skipped_samples") or []]
    monitor = None if getattr(args, "force_gpu_index_no_nvidia_smi", None) is not None else GpuTelemetryMonitor(selected_gpu_index=selected_gpu.index, selected_gpu_uuid=selected_gpu.uuid, own_pid=os.getpid(), poll_interval_sec=args.gpu_poll_interval_sec)
    if monitor is not None:
        monitor.start()
    try:
        evaluated = 0
        total_samples = int(route_requirements.get("total_val_samples") or len(dataset))
        for frame_idx, batch in enumerate(_new_loader(dataset, DataLoader)):
            frame_start = time.perf_counter()
            if batch is None:
                skipped.append({"sample_idx": int(frame_idx), "reason": "batch_is_none"})
                if progress is not None and progress.should_emit_frame(frame_idx):
                    progress.emit(
                        "eval_frame",
                        f"frame {frame_idx + 1}/{total_samples}: skipped batch_is_none",
                        mode=mode.key,
                        sample_idx=int(frame_idx),
                        frame_number=int(frame_idx + 1),
                        total_val_samples=total_samples,
                        evaluated_samples=evaluated,
                        reason="batch_is_none",
                        elapsed_ms=(time.perf_counter() - frame_start) * 1000.0,
                    )
                continue
            try:
                ego_cpu = batch["ego"] if isinstance(batch, dict) and "ego" in batch else batch
                # _prepare_inputs needs tensors on the active device; pre-scan already records K skips.
                from export_lidar_pyramid_onnx import _extract_inputs

                original_tensors_cpu, _agent_modalities = _extract_inputs(ego_cpu, _modality)
                original_num_voxels = int(original_tensors_cpu[0].shape[0])
                if original_num_voxels > fixed_k:
                    if progress is not None and progress.should_emit_frame(frame_idx):
                        progress.emit(
                            "eval_frame",
                            f"frame {frame_idx + 1}/{total_samples}: skipped K_exceeds_fixed_K K={original_num_voxels}",
                            mode=mode.key,
                            sample_idx=int(frame_idx),
                            frame_number=int(frame_idx + 1),
                            total_val_samples=total_samples,
                            evaluated_samples=evaluated,
                            original_num_voxels=original_num_voxels,
                            reason="K_exceeds_fixed_K",
                            elapsed_ms=(time.perf_counter() - frame_start) * 1000.0,
                        )
                    continue
                ego = _to_device(ego_cpu, device)
                batch = _to_device(batch, device)
                inputs, meta = _prepare_inputs(ego, _modality, fixed_k=fixed_k)
                outputs, profile = runner.run_profiled(inputs)
                _cpu_outputs, d2h_ms, d2h_copies, d2h_bytes = _copy_outputs_to_cpu(outputs, device)
                output = {name: outputs[name].float() for name in ("cls_preds", "reg_preds", "dir_preds") if name in outputs}

                def _postprocess():
                    od = OrderedDict()
                    od["ego"] = output
                    return dataset.post_process(batch, od)

                (pred_box, pred_score, gt_box), post_ms = _timed(_postprocess, device)
                for thr in IOU_THRESHOLDS:
                    _calculate_tp_fp(pred_box, pred_score, gt_box, result_stat, thr, args.ap_iou_backend, device)
                fwd_ms = float(profile.get("total_runner_ms", 0.0)) + d2h_ms
                row = {
                    "sample_idx": int(frame_idx),
                    "record_len": int(meta["record_len"]),
                    "N_runtime_shape": int(meta["N"]),
                    "original_num_voxels": int(meta["original_num_voxels"]),
                    "fixed_K": int(meta["fixed_K"]),
                    "padding_ratio": float(meta["padding_ratio"]),
                    "engine_path": str(engine_paths[0]),
                    "execute_ms": float(profile.get("execute_async_ms", 0.0)),
                    "execute_async_ms": float(profile.get("execute_async_ms", 0.0)),
                    "forward_ms": fwd_ms,
                    "total_runner_ms": fwd_ms,
                    "postprocess_ms": post_ms,
                    "d2h_copy_ms": d2h_ms,
                    "d2h_copies": d2h_copies,
                    "bytes_d2h": d2h_bytes,
                    "shape_setup_ms": float(profile.get("set_input_shape_ms", 0.0)),
                    "padding_ms": float(meta.get("padding_ms", 0.0)),
                    "latency_warmup": evaluated < int(args.latency_warmup_frames),
                    "input_shapes": profile.get("input_shapes", {}),
                    "output_shapes": profile.get("output_shapes", {}),
                }
                rows.append(row)
                if not row["latency_warmup"]:
                    latency_rows.append(row)
                evaluated += 1
                if progress is not None and progress.should_emit_frame(frame_idx):
                    progress.emit(
                        "eval_frame",
                        (
                            f"frame {frame_idx + 1}/{total_samples}: evaluated={evaluated}, "
                            f"K={row['original_num_voxels']}, N={row['record_len']}, "
                            f"execute={row['execute_ms']:.4f} ms, forward={row['forward_ms']:.4f} ms, "
                            f"post={post_ms:.4f} ms"
                        ),
                        mode=mode.key,
                        sample_idx=int(frame_idx),
                        frame_number=int(frame_idx + 1),
                        total_val_samples=total_samples,
                        evaluated_samples=evaluated,
                        record_len=row["record_len"],
                        N_runtime_shape=row["N_runtime_shape"],
                        original_num_voxels=row["original_num_voxels"],
                        engine_path=row["engine_path"],
                        execute_ms=row["execute_ms"],
                        forward_ms=row["forward_ms"],
                        total_runner_ms=row["total_runner_ms"],
                        postprocess_ms=post_ms,
                        d2h_copy_ms=d2h_ms,
                        shape_setup_ms=row["shape_setup_ms"],
                        padding_ms=row["padding_ms"],
                        latency_warmup=row["latency_warmup"],
                        elapsed_ms=(time.perf_counter() - frame_start) * 1000.0,
                    )
            except Exception as exc:
                skipped.append({"sample_idx": int(frame_idx), "reason": "eval_exception", "error": str(exc), "traceback": traceback.format_exc()})
                if progress is not None:
                    progress.emit(
                        "eval_frame_error",
                        f"frame {frame_idx + 1}/{total_samples}: eval_exception: {exc}",
                        mode=mode.key,
                        sample_idx=int(frame_idx),
                        frame_number=int(frame_idx + 1),
                        total_val_samples=total_samples,
                        error=str(exc),
                        elapsed_ms=(time.perf_counter() - frame_start) * 1000.0,
                    )
    finally:
        if monitor is not None:
            monitor.stop()
    ap: dict[str, float] = {}
    for thr in IOU_THRESHOLDS:
        key = f"AP@{thr:.2f}"
        if result_stat[thr]["gt"] > 0 and result_stat[thr]["score"]:
            ap_value, _, _ = eval_utils.calculate_ap(result_stat, thr)
        else:
            ap_value = 0.0
        ap[key] = round(float(ap_value), 4)
    after = [] if getattr(args, "force_gpu_index_no_nvidia_smi", None) is not None else snapshots_to_dicts(query_gpu_snapshots())
    trace = (
        {
            "selected_gpu": int(selected_gpu.index),
            "selected_gpu_uuid": selected_gpu.uuid,
            "polling_interval": int(args.gpu_poll_interval_sec),
            "samples": [],
            "contention_events": [],
            "other_processes_detected": None,
            "other_processes": [],
            "gpu_contention_detected": False,
            "unreliable_latency": False,
            "forced_no_nvidia_smi": True,
        }
        if monitor is None
        else monitor.report()
    )
    report = _finalize_report(report, rows=rows, latency_rows=latency_rows, skipped=skipped, ap=ap, monitor_report=trace, nvidia_smi_after=after)
    if progress is not None:
        progress.emit(
            "mode_done",
            f"mode done: evaluated={report.get('evaluated_samples')}/{report.get('total_val_samples')}, mAP={report.get('mAP')}, AP70={report.get('AP@0.70')}, forward_p50={(report.get('forward_ms') or {}).get('p50')}",
            mode=mode.key,
            evaluated_samples=report.get("evaluated_samples"),
            total_val_samples=report.get("total_val_samples"),
            skipped_samples=len(report.get("skipped_samples") or []),
            AP_030=report.get("AP@0.30"),
            AP_050=report.get("AP@0.50"),
            AP_070=report.get("AP@0.70"),
            mAP=report.get("mAP"),
            execute_ms=report.get("execute_ms"),
            forward_ms=report.get("forward_ms"),
            FPS=report.get("FPS"),
            gpu_contention_detected=report.get("gpu_contention_detected"),
            unreliable_latency=report.get("unreliable_latency"),
        )
    trace.update({"mode": mode.key, "nvidia_smi_before": nvidia_smi_before, "nvidia_smi_after": after})
    return report, trace


def missing_report_for_mode(args: argparse.Namespace, dirs: dict[str, Path], mode: EvalMode, selected_gpu: Any, route_requirements: dict[str, Any], nvidia_smi_before: list[dict[str, Any]], reason: str) -> tuple[dict[str, Any], dict[str, Any]]:
    engine_paths, missing_paths = _engine_paths_for_mode(dirs, mode, route_requirements)
    report = _base_report(mode=mode, args=args, dirs=dirs, selected_gpu=selected_gpu, engine_paths=engine_paths, missing_paths=missing_paths, route_requirements=route_requirements, nvidia_smi_before=nvidia_smi_before)
    after = [] if getattr(args, "force_gpu_index_no_nvidia_smi", None) is not None else snapshots_to_dicts(query_gpu_snapshots())
    report.update({"status": "missing_engine", "error": reason, "nvidia_smi_after": after})
    trace = {"mode": mode.key, "status": "missing_engine", "reason": reason, "nvidia_smi_before": nvidia_smi_before, "nvidia_smi_after": after}
    return report, trace


def summarize_reports(dirs: dict[str, Path], reports: list[dict[str, Any]], gpu_report: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for report in reports:
        execute = report.get("execute_ms") or {}
        forward = report.get("forward_ms") or {}
        row = {
            "scheme": report.get("scheme"),
            "engine_strategy": report.get("engine_strategy"),
            "precision": report.get("precision"),
            "calibration": report.get("calibration_mode"),
            "full_val": report.get("full_val"),
            "total_val": report.get("total_val_samples"),
            "evaluated": report.get("evaluated_samples"),
            "skipped": len(report.get("skipped_samples") or []),
            "engine_count": report.get("engine_count"),
            "dynamic_N": report.get("dynamic_N"),
            "bucket_router": report.get("bucket_router"),
            "single_engine": report.get("single_engine"),
            "AP@0.30": report.get("AP@0.30"),
            "AP@0.50": report.get("AP@0.50"),
            "AP@0.70": report.get("AP@0.70"),
            "mAP": report.get("mAP"),
            "execute_p50": execute.get("p50"),
            "execute_p95": execute.get("p95"),
            "forward_p50": forward.get("p50"),
            "forward_p95": forward.get("p95"),
            "FPS": report.get("FPS"),
            "selected_gpu": report.get("selected_gpu"),
            "contention": report.get("gpu_contention_detected"),
            "reliable_latency": not bool(report.get("unreliable_latency")),
            "status": report.get("status"),
            "notes": report.get("error")
            or (
                f"AP over evaluated_samples; fixed_K={report.get('fixed_K')}; skipped={len(report.get('skipped_samples') or [])}"
                if report.get("status") == "success"
                else ""
            ),
        }
        rows.append(row)

    successful = [row for row in rows if row.get("status") == "success" and row.get("mAP") is not None]
    fp16_success = [row for row in successful if row.get("precision") == "fp16"]
    int8_success = [row for row in successful if row.get("precision") == "int8"]

    def best_by(metric: str, candidates: list[dict[str, Any]], reverse: bool = False) -> dict[str, Any] | None:
        usable = [row for row in candidates if row.get(metric) is not None]
        if not usable:
            return None
        return sorted(usable, key=lambda row: float(row[metric]), reverse=reverse)[0]

    single_fp16 = next((row for row in rows if row.get("scheme") == "dynamic_agent_single_engine_maxK" and row.get("precision") == "fp16"), None)
    dynamic_bucket_fp16 = next((row for row in rows if row.get("scheme") == "dynamic_agent_dim" and row.get("precision") == "fp16" and row.get("engine_strategy") == "dynamic_agent_dim_bucket_fixed_k_plugin"), None)
    single_16ms_still_present = bool(single_fp16 and single_fp16.get("forward_p50") and float(single_fp16["forward_p50"]) > 10.0)
    single_slower_than_dynamic = None
    if single_fp16 and dynamic_bucket_fp16 and single_fp16.get("forward_p50") and dynamic_bucket_fp16.get("forward_p50"):
        single_slower_than_dynamic = float(single_fp16["forward_p50"]) / max(float(dynamic_bucket_fp16["forward_p50"]), 1e-9)
    single_fp16_map_delta = None
    if single_fp16 and dynamic_bucket_fp16 and single_fp16.get("mAP") is not None and dynamic_bucket_fp16.get("mAP") is not None:
        single_fp16_map_delta = abs(float(single_fp16["mAP"]) - float(dynamic_bucket_fp16["mAP"]))
    single_engine_fp16_accept = bool(
        single_fp16
        and dynamic_bucket_fp16
        and single_fp16_map_delta is not None
        and single_fp16_map_delta <= 0.002
        and single_slower_than_dynamic is not None
        and single_slower_than_dynamic <= 1.10
    )

    def fp16_reference_for(row: dict[str, Any]) -> dict[str, Any] | None:
        if row.get("scheme") == "dynamic_agent_single_engine_maxK":
            return single_fp16
        if row.get("scheme") == "dynamic_agent_dim":
            return dynamic_bucket_fp16
        return None

    int8_candidates: list[dict[str, Any]] = []
    for row in int8_success:
        ref = fp16_reference_for(row)
        if not ref or row.get("mAP") is None or ref.get("mAP") is None or row.get("forward_p50") is None or ref.get("forward_p50") is None:
            continue
        drop = float(ref["mAP"]) - float(row["mAP"])
        row["mAP_drop_vs_reference_FP16"] = round(drop, 4)
        if drop <= 0.1 and float(row["forward_p50"]) < float(ref["forward_p50"]):
            int8_candidates.append(row)

    fastest_fp32 = best_by("forward_p50", [row for row in successful if row.get("precision") == "fp32"])
    fastest_fp16 = best_by("forward_p50", fp16_success)
    fastest_int8 = best_by("forward_p50", int8_success)
    best_ap = best_by("mAP", successful, reverse=True)
    recommended_path = (
        "dynamic_agent_single_engine_maxK FP16"
        if single_engine_fp16_accept
        else "dynamic_agent_dim fixed-K bucket router PointPillarScatterTRT FP16"
    )

    report = {
        "success": True,
        "previous_results_under_gpu_contention_should_not_be_trusted": True,
        "selected_gpu": gpu_report.get("selected_gpu"),
        "selected_gpu_physical_id": gpu_report.get("selected_gpu_physical_id"),
        "CUDA_VISIBLE_DEVICES": gpu_report.get("CUDA_VISIBLE_DEVICES"),
        "gpu_contention_detected": gpu_report.get("gpu_contention_detected"),
        "unreliable_latency": gpu_report.get("unreliable_latency"),
        "final_recommended_deployment_path": recommended_path,
        "gpu_report": gpu_report,
        "rows": rows,
        "analysis": {
            "single_engine_maxK_16ms_forward_p50_still_present": single_16ms_still_present,
            "single_engine_maxK_vs_dynamic_bucket_fp16_forward_ratio": single_slower_than_dynamic,
            "single_engine_maxK_fp16_mAP_delta_vs_dynamic_bucket_fp16": single_fp16_map_delta,
            "single_engine_slowdown_interpretation": "GPU contention is excluded when reliable_latency=true; remaining likely causes are fixed_K input copy/padding overhead, dynamic profile shape setup, TensorRT tactic choices, and dynamic-N plugin island overhead.",
            "dynamic_bucket_fp16_fastest_high_precision": (best_by("forward_p50", fp16_success) or {}).get("scheme") == "dynamic_agent_dim",
            "single_engine_maxK_fp16_acceptance": single_engine_fp16_accept,
            "single_engine_maxK_as_engineering_simplification": "recommend as default FP16 deployment path when AP matches dynamic bucket and forward p50 slowdown stays within 10%; dynamic bucket remains the fastest FP16 reference",
            "int8_changed_conclusion_after_full_val_idle_gpu": "see rows; accept only if mAP drop and latency criteria are met",
            "int8_smallest_mAP_drop_mode": best_by("mAP", int8_success, reverse=True),
            "int8_fastest_latency_mode": best_by("forward_p50", int8_success),
            "int8_acceptable_speed_candidates": int8_candidates,
            "recommend_int8_as_deployment_candidate": bool(int8_candidates),
            "recommend_int8_as_default_deployment": False,
            "recommend_qdq_modelopt": "yes for INT8, especially single_engine_maxK where train_calib200 mAP drop exceeds 0.1; try mixed precision whitelist before full ModelOpt Q/DQ",
            "fastest_fp32_mode": fastest_fp32,
            "fastest_fp16_mode": fastest_fp16,
            "fastest_int8_mode": fastest_int8,
            "best_ap_mode": best_ap,
            "recommended_deployment_path": recommended_path,
        },
    }
    save_json(report, dirs["summary_full_val_idle_gpu"] / "all_deployment_engines_full_val_idle_gpu_report.json")
    lines = [
        "# All Deployment Engines Full-Val Idle-GPU Report",
        "",
        "previous results under GPU contention should not be trusted.",
        "",
        "scheme | engine_strategy | precision | calibration | full_val | total_val | evaluated | skipped | engine_count | dynamic_N | bucket_router | single_engine | AP@0.30 | AP@0.50 | AP@0.70 | mAP | execute_p50 | execute_p95 | forward_p50 | forward_p95 | FPS | selected_gpu | contention | reliable_latency | notes",
        "--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---",
    ]
    for row in rows:
        lines.append(
            " | ".join(
                str(row.get(key, ""))
                for key in [
                    "scheme",
                    "engine_strategy",
                    "precision",
                    "calibration",
                    "full_val",
                    "total_val",
                    "evaluated",
                    "skipped",
                    "engine_count",
                    "dynamic_N",
                    "bucket_router",
                    "single_engine",
                    "AP@0.30",
                    "AP@0.50",
                    "AP@0.70",
                    "mAP",
                    "execute_p50",
                    "execute_p95",
                    "forward_p50",
                    "forward_p95",
                    "FPS",
                    "selected_gpu",
                    "contention",
                    "reliable_latency",
                    "notes",
                ]
            )
        )
    lines.extend(
        [
            "",
            "## Analysis",
            "",
            f"- single_engine_maxK_16ms_forward_p50_still_present: {report['analysis']['single_engine_maxK_16ms_forward_p50_still_present']}",
            f"- single_engine_maxK_vs_dynamic_bucket_fp16_forward_ratio: {report['analysis']['single_engine_maxK_vs_dynamic_bucket_fp16_forward_ratio']}",
            f"- single_engine_maxK_fp16_mAP_delta_vs_dynamic_bucket_fp16: {report['analysis']['single_engine_maxK_fp16_mAP_delta_vs_dynamic_bucket_fp16']}",
            f"- single_engine_maxK_fp16_acceptance: {report['analysis']['single_engine_maxK_fp16_acceptance']}",
            f"- dynamic_bucket_fp16_fastest_high_precision: {report['analysis']['dynamic_bucket_fp16_fastest_high_precision']}",
            f"- int8_smallest_mAP_drop_mode: {(report['analysis']['int8_smallest_mAP_drop_mode'] or {}).get('calibration')}",
            f"- int8_fastest_latency_mode: {(report['analysis']['int8_fastest_latency_mode'] or {}).get('calibration')}",
            f"- recommend_int8_as_deployment_candidate: {report['analysis']['recommend_int8_as_deployment_candidate']}",
            f"- recommend_int8_as_default_deployment: {report['analysis']['recommend_int8_as_default_deployment']}",
            f"- recommend_qdq_modelopt: {report['analysis']['recommend_qdq_modelopt']}",
            f"- recommended_deployment_path: {report['analysis']['recommended_deployment_path']}",
        ]
    )
    (dirs["summary_full_val_idle_gpu"] / "all_deployment_engines_full_val_idle_gpu_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def summarize_existing(args: argparse.Namespace) -> dict[str, Any]:
    dirs = full_val_dirs(args.output_root, output_tag=args.output_tag, fixed_k=int(args.fixed_k), fixedk_engine_namespace=bool(args.fixedk_engine_namespace))
    reports: list[dict[str, Any]] = []
    for mode in eval_modes(fixed_k=int(args.fixed_k), dynamic_int8_calibration_split=args.dynamic_int8_calibration_split, include_mixed_heads_fp16=bool(args.include_mixed_heads_fp16)):
        path = dirs["evaluation_full_val_idle_gpu"] / mode_report_name(mode)
        if path.exists():
            reports.append(json.loads(path.read_text(encoding="utf-8")))
    gpu_report_path = dirs["debug_full_val_idle_gpu"] / "gpu_selection_and_contention_report.json"
    gpu_report = json.loads(gpu_report_path.read_text(encoding="utf-8")) if gpu_report_path.exists() else {}
    return summarize_reports(dirs, reports, gpu_report)


def run(args: argparse.Namespace) -> dict[str, Any]:
    dirs = full_val_dirs(args.output_root, output_tag=args.output_tag, fixed_k=int(args.fixed_k), fixedk_engine_namespace=bool(args.fixedk_engine_namespace))
    progress = ProgressLogger(
        dirs,
        stdout=not bool(getattr(args, "no_progress_stdout", False)),
        every_frames=int(getattr(args, "progress_every_frames", 1)),
    )
    run_start = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    progress.emit(
        "run_start",
        f"starting full-val evaluation: fixed_K={args.fixed_k}, output_tag={args.output_tag}, schemes={args.schemes or 'all'}",
        fixed_K=int(args.fixed_k),
        output_tag=str(args.output_tag),
        schemes=args.schemes or "all",
        output_root=str(args.output_root),
    )
    if args.force_gpu_index_no_nvidia_smi is not None:
        selected_gpu = SimpleNamespace(index=int(args.force_gpu_index_no_nvidia_smi), uuid=f"forced-gpu-{int(args.force_gpu_index_no_nvidia_smi)}")
        selected_reason = "forced physical GPU id; nvidia-smi disabled because a bad GPU makes NVML queries slow"
        selection_attempts = []
        progress.emit(
            "gpu_selected",
            f"selected forced GPU {selected_gpu.index}: {selected_reason}",
            selected_gpu=int(selected_gpu.index),
            selected_gpu_reason=selected_reason,
        )
    else:
        try:
            selected_gpu, selected_reason, selection_attempts = wait_for_idle_gpu(
                util_threshold=args.gpu_idle_util_threshold,
                mem_threshold_mb=args.gpu_idle_mem_threshold_mb,
                wait_timeout_sec=args.gpu_wait_timeout_sec,
                poll_interval_sec=args.gpu_poll_interval_sec,
                gpu_index=args.gpu_index,
                allow_busy_gpu=args.allow_busy_gpu,
            )
            progress.emit(
                "gpu_selected",
                f"selected GPU {selected_gpu.index}: {selected_reason}",
                selected_gpu=int(selected_gpu.index),
                selected_gpu_uuid=selected_gpu.uuid,
                selected_gpu_reason=selected_reason,
                allow_busy_gpu=bool(args.allow_busy_gpu),
            )
        except Exception as exc:
            gpu_report = {
                "status": "no_idle_gpu",
                "error": str(exc),
                "all_gpu_snapshot_before": [],
                "all_gpu_snapshot_after": [],
                "selected_gpu": None,
                "selected_gpu_physical_id": None,
                "selected_gpu_uuid": None,
                "selected_gpu_reason": "idle GPU selection failed",
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "CUDA_DEVICE_ORDER": os.environ.get("CUDA_DEVICE_ORDER"),
                "polling_interval": int(args.gpu_poll_interval_sec),
                "selection_attempts": [],
                "mode_traces": [],
                "contention_events": [{"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "reason": "idle_gpu_selection_failed", "error": str(exc)}],
                "other_processes_detected": None,
                "other_processes": [],
                "gpu_contention_detected": True,
                "unreliable_latency": True,
                "run_start_time": run_start,
                "run_end_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }
            write_gpu_selection_reports(
                gpu_report,
                debug_path=dirs["debug_full_val_idle_gpu"] / "gpu_selection_and_contention_report.json",
                summary_path=dirs["summary_full_val_idle_gpu"] / "gpu_selection_and_contention_report.md",
            )
            save_json(gpu_report, dirs["debug"] / "gpu_selection_and_contention_report.json")
            progress.emit("gpu_selection_failed", f"idle GPU selection failed: {exc}", error=str(exc))
            return {"success": False, "gpu_report": gpu_report, "summary": {}}
    os.environ["CUDA_VISIBLE_DEVICES"] = str(selected_gpu.index)
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    progress.emit(
        "env_ready",
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}, CUDA_DEVICE_ORDER={os.environ.get('CUDA_DEVICE_ORDER')}",
        CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES"),
        CUDA_DEVICE_ORDER=os.environ.get("CUDA_DEVICE_ORDER"),
    )
    nvidia_before = (
        [
            {
                "source": "user_provided_or_forced_no_nvidia_smi",
                "selected_gpu": int(selected_gpu.index),
                "excluded_gpu_indices": str(args.excluded_gpu_indices),
                "note": "nvidia-smi telemetry disabled; GPU 2 reported error by user.",
            }
        ]
        if args.force_gpu_index_no_nvidia_smi is not None
        else snapshots_to_dicts(query_gpu_snapshots())
    )
    progress.emit(
        "context_load_start",
        f"loading HEAL context: config={args.hypes_yaml}, checkpoint={args.checkpoint}",
        hypes_yaml=str(args.hypes_yaml),
        checkpoint=str(args.checkpoint),
        heal_repo=str(args.heal_repo),
    )
    context = _load_context(args)
    _hypes, device, _model, modality, dataset, _DataLoader = context
    progress.emit(
        "context_load_done",
        f"context loaded: device={device}, modality={modality}, val_samples={len(dataset)}",
        device=str(device),
        modality=str(modality),
        total_val_samples=len(dataset),
    )
    route_requirements = scan_val_requirements(args, context, dirs, progress)
    modes = eval_modes(fixed_k=int(args.fixed_k), dynamic_int8_calibration_split=args.dynamic_int8_calibration_split, include_mixed_heads_fp16=bool(args.include_mixed_heads_fp16))
    requested = set(args.schemes or [mode.key for mode in modes])
    reports: list[dict[str, Any]] = []
    mode_traces: list[dict[str, Any]] = []
    for mode in modes:
        if mode.key not in requested:
            continue
        progress.emit(
            "mode_prepare",
            f"preparing mode {mode.key}: scheme={mode.scheme}, precision={mode.precision}, calibration={mode.calibration_mode}",
            mode=mode.key,
            scheme=mode.scheme,
            precision=mode.precision,
            calibration_mode=mode.calibration_mode,
            engine_strategy=mode.engine_strategy,
        )
        mode_start_check = _wait_for_no_other_processes(args, selected_gpu, os.getpid())
        if not mode_start_check.get("ok"):
            report, trace = missing_report_for_mode(args, dirs, mode, selected_gpu, route_requirements, nvidia_before, str(mode_start_check.get("error")))
            report["status"] = "gpu_busy"
            report["unreliable_latency"] = True
            trace["mode_start_check"] = mode_start_check
            progress.emit(
                "mode_gpu_busy",
                f"mode {mode.key} not run because selected GPU stayed busy: {mode_start_check.get('error')}",
                mode=mode.key,
                error=str(mode_start_check.get("error")),
            )
        else:
            try:
                if mode.runner_kind == "single":
                    report, trace = evaluate_single_mode(args, dirs, context, mode, selected_gpu, route_requirements, nvidia_before, progress)
                else:
                    report, trace = evaluate_router_mode(args, dirs, context, mode, selected_gpu, route_requirements, nvidia_before, progress)
                trace["mode_start_check"] = mode_start_check
            except Exception as exc:
                report, trace = missing_report_for_mode(args, dirs, mode, selected_gpu, route_requirements, nvidia_before, str(exc))
                report["status"] = "failed"
                report["traceback"] = traceback.format_exc()
                trace["traceback"] = report["traceback"]
                progress.emit(
                    "mode_failed",
                    f"mode {mode.key} failed: {exc}",
                    mode=mode.key,
                    error=str(exc),
                    traceback=trace["traceback"],
                )
        _write_mode_reports(dirs, mode, report, trace)
        progress.emit(
            "mode_report_written",
            f"reports written for {mode.key}: status={report.get('status')}, eval={dirs['evaluation_full_val_idle_gpu'] / mode_report_name(mode)}",
            mode=mode.key,
            status=report.get("status"),
            evaluation_path=str(dirs["evaluation_full_val_idle_gpu"] / mode_report_name(mode)),
            benchmark_path=str(dirs["benchmark_full_val_idle_gpu"] / mode_report_name(mode)),
            trace_path=str(dirs["debug_full_val_idle_gpu"] / f"{mode.key}_gpu_trace.json"),
        )
        reports.append(report)
        mode_traces.append(trace)
    run_end = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    nvidia_after = [] if args.force_gpu_index_no_nvidia_smi is not None else snapshots_to_dicts(query_gpu_snapshots())
    contention_events = [event for trace in mode_traces for event in (trace.get("contention_events") or [])]
    other_processes = [process for trace in mode_traces for process in (trace.get("other_processes") or [])]
    gpu_report = {
        "all_gpu_snapshot_before": nvidia_before,
        "all_gpu_snapshot_after": nvidia_after,
        "selected_gpu": int(selected_gpu.index),
        "selected_gpu_physical_id": int(selected_gpu.index),
        "selected_gpu_uuid": selected_gpu.uuid,
        "selected_gpu_reason": selected_reason,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "CUDA_DEVICE_ORDER": os.environ.get("CUDA_DEVICE_ORDER"),
        "gpu_selection_method": "forced_no_nvidia_smi" if args.force_gpu_index_no_nvidia_smi is not None else "nvidia_smi_idle_selection",
        "excluded_gpu_indices": str(args.excluded_gpu_indices),
        "polling_interval": int(args.gpu_poll_interval_sec),
        "selection_attempts": selection_attempts,
        "mode_traces": mode_traces,
        "contention_events": contention_events,
        "other_processes_detected": bool(other_processes),
        "other_processes": other_processes,
        "gpu_contention_detected": bool(contention_events),
        "unreliable_latency": bool(contention_events),
        "run_start_time": run_start,
        "run_end_time": run_end,
    }
    write_gpu_selection_reports(
        gpu_report,
        debug_path=dirs["debug_full_val_idle_gpu"] / "gpu_selection_and_contention_report.json",
        summary_path=dirs["summary_full_val_idle_gpu"] / "gpu_selection_and_contention_report.md",
    )
    save_json(gpu_report, dirs["debug"] / "gpu_selection_and_contention_report.json")
    (dirs["summary"] / "gpu_selection_and_contention_report.md").write_text(
        (dirs["summary_full_val_idle_gpu"] / "gpu_selection_and_contention_report.md").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    summary = summarize_reports(dirs, reports, gpu_report)
    progress.emit(
        "run_done",
        f"full-val evaluation done: modes={len(reports)}, summary={dirs['summary_full_val_idle_gpu'] / 'all_deployment_engines_full_val_idle_gpu_report.json'}",
        modes=len(reports),
        summary_json=str(dirs["summary_full_val_idle_gpu"] / "all_deployment_engines_full_val_idle_gpu_report.json"),
        summary_md=str(dirs["summary_full_val_idle_gpu"] / "all_deployment_engines_full_val_idle_gpu_report.md"),
    )
    return {"success": True, "gpu_report": gpu_report, "summary": summary}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.summarize_existing:
        summary = summarize_existing(args)
        print(json.dumps({"success": summary.get("success"), "selected_gpu": summary.get("selected_gpu")}, indent=2))
        return 0
    result = run(args)
    print(json.dumps({"success": result.get("success"), "selected_gpu": result["gpu_report"].get("selected_gpu")}, indent=2))
    return 0 if result.get("success") else 2


if __name__ == "__main__":
    raise SystemExit(main())
