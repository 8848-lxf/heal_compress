"""Persistent Stage-2 candidate evaluator bound to one physical GPU."""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any


def _atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def _context_kwargs(request: dict[str, Any]) -> dict[str, Any]:
    config = dict(request.get("config", {}))
    runtime = dict(config.get("runtime", {}))
    model = dict(config.get("model", {}))
    search = dict(config.get("search", {}))
    pruning = dict(config.get("pruning", {}))
    grouped = dict(pruning.get("grouped_conv", {}) or {})
    precision = dict(config.get("precision", {}))
    proxy = dict(config.get("proxy", config.get("proxy_objective", {})))
    stage2 = dict(
        config.get("stage2")
        or config.get("stage2_smoke")
        or config.get("evaluation", {})
    )
    stage2_parallel = dict(config.get("stage2_parallel", {}) or {})
    gpu_id = int(request["gpu_id"])
    controller_pid = int(request.get("controller_pid", 0) or 0)
    try:
        stage1_gpu_id = int(runtime.get("gpu_id", -1))
    except (TypeError, ValueError):
        stage1_gpu_id = -1
    allowed_gpu_pids = (
        {controller_pid}
        if controller_pid > 0
        and gpu_id == stage1_gpu_id
        and bool(
            stage2_parallel.get(
                "allow_controller_process_on_stage1_gpu", False
            )
        )
        else set()
    )
    return {
        "checkpoint_path": request["checkpoint"],
        "output_dir": request["worker_dir"],
        "model_config_path": model.get("config") or model.get("hypes_yaml"),
        "heal_root": runtime.get(
            "heal_root", "/home/lixingfeng/UniAD_examine/HEAL"
        ),
        "tensorrt_root": runtime.get(
            "tensorrt_root",
            "/home/lixingfeng/UniAD_examine/HEAL/prune_model/TensorRT-10.9_x86_cu118",
        ),
        "plugin_path": runtime.get("plugin_path"),
        "plugin_boundary_dtype": str(runtime.get("plugin_boundary_dtype", "")),
        "gpu_id": str(gpu_id),
        "exclude_gpu_ids": [],
        "tensorrt_env": str(runtime.get("tensorrt_env", "modelopt")),
        "fisher_calibration_batches": int(
            proxy.get("fisher_calibration_batches", 8)
        ),
        "quant_calibration_batches": int(
            proxy.get("quant_calibration_batches", 16)
        ),
        "quant_calibration_npz_manifest": proxy.get(
            "quant_calibration_npz_manifest"
        ),
        "quant_activation_calibration_backend": str(
            proxy.get(
                "quant_activation_calibration_backend",
                "modelopt_histogram_entropy",
            )
        ),
        "quant_activation_calibration_cache_path": proxy.get(
            "quant_activation_calibration_cache_path"
        ),
        "quant_calibration_force_rebuild": bool(
            proxy.get("quant_calibration_force_rebuild", False)
        ),
        "num_frames": int(stage2.get("num_frames", 5)),
        "warmup_frames": int(stage2.get("warmup_frames", 10)),
        "reset_after_warmup": bool(stage2.get("reset_after_warmup", False)),
        "default_precision": str(precision.get("default", "FP16")),
        "max_pruning_units": int(search.get("max_pruning_units", 96)),
        "grouped_conv_mode": str(
            pruning.get("grouped_conv_mode")
            or grouped.get("position_mode", "shared_local_mean")
        ),
        "grouped_conv_align": int(
            pruning.get("grouped_conv_align")
            or grouped.get("default_channels_per_group", 8)
        ),
        "grouped_allowed_channels_per_group": [
            int(value)
            for value in grouped.get(
                "allowed_channels_per_group",
                [4, 8, 16, 32, 64, 128, 256, 512],
            )
        ],
        "pruning_gene_type": str(
            pruning.get(
                "gene_type", pruning.get("search_variable", "legal_pruning_action")
            )
        ),
        "allow_foreign_gpu_processes": bool(
            runtime.get("allow_foreign_gpu_processes", False)
        ),
        "allowed_gpu_pids": allowed_gpu_pids,
        "max_gpu_utilization_pct": int(
            runtime.get("max_gpu_utilization_pct", 20)
        ),
    }


def _gpu_isolation_kwargs(request: dict[str, Any]) -> dict[str, Any]:
    context = _context_kwargs(request)
    return {
        "allowed_pids": set(context["allowed_gpu_pids"]),
        "allow_foreign_processes": bool(
            context["allow_foreign_gpu_processes"]
        ),
        "max_gpu_utilization_pct": int(
            context["max_gpu_utilization_pct"]
        ),
    }


def _build_evaluator(request: dict[str, Any]) -> tuple[Any, Any]:
    from ..constrained.context import apply_constrained_pruning_context
    from ..integration.lidar_pyramid_context import build_lidar_pyramid_context
    from .lidar_pyramid_real_evaluator import LidarPyramidRealEvaluator
    from .objective import Stage2ObjectiveConfig

    context = build_lidar_pyramid_context(**_context_kwargs(request))
    config = dict(request.get("config", {}))
    constrained = dict(config.get("constrained_search", {}) or {})
    if bool(constrained.get("enabled", False)):
        pruning = dict(config.get("pruning", {}) or {})
        grouped = dict(pruning.get("grouped_conv", {}) or {})
        context, _projection = apply_constrained_pruning_context(
            context,
            allowed_root_patterns=[
                str(value)
                for value in constrained.get("allowed_pruning_root_patterns", [])
            ],
            grouped_conv_mode=str(
                grouped.get("position_mode", "independent_group_topk")
            ),
            grouped_conv_align=int(
                dict(pruning.get("dense", {}) or {}).get("alignment", 4)
            ),
            grouped_allowed_channels_per_group=[
                int(value)
                for value in grouped.get(
                    "allowed_channels_per_group",
                    [4, 8, 16, 32, 64, 128, 256, 512],
                )
            ],
            allowed_precision_values=[
                str(value)
                for value in constrained.get(
                    "allowed_precision_values", ["FP16", "INT8"]
                )
            ],
        )
    stage2 = dict(
        config.get("stage2")
        or config.get("stage2_smoke")
        or config.get("evaluation", {})
    )
    evaluator = LidarPyramidRealEvaluator(
        context=context,
        run_dir=request["worker_dir"],
        num_frames=int(stage2.get("num_frames", 5)),
        warmup_frames=int(stage2.get("warmup_frames", 10)),
        latency_rounds=int(
            stage2.get("latency_rounds", stage2.get("rounds", 1))
        ),
        target_bops_retention=stage2.get("target_bops_retention"),
        bops_tolerance=float(
            stage2.get("bops_tolerance", stage2.get("tolerance", 0.005))
        ),
        stage2_config=Stage2ObjectiveConfig(
            eta_map=float(stage2.get("eta_ap", stage2.get("eta_map", 1.0))),
            eta_latency=float(stage2.get("eta_latency", 1.0)),
            latency_metric=str(
                stage2.get("latency_metric", "forward_mean_ms")
            ),
            tau_ap=stage2.get("tau_ap"),
            max_map_drop=stage2.get("max_map_drop"),
            min_map=stage2.get("min_map"),
            min_ap07=stage2.get("min_ap07"),
            required_evaluated_frames=stage2.get(
                "required_evaluated_frames"
            ),
            required_skipped_frames=stage2.get("required_skipped_frames"),
            r_mac_floor=stage2.get("r_mac_floor"),
            int8_mac_share_min=(
                list(stage2.get("int8_mac_share", []))[0]
                if len(list(stage2.get("int8_mac_share", []))) == 2
                else None
            ),
            int8_mac_share_max=(
                list(stage2.get("int8_mac_share", []))[1]
                if len(list(stage2.get("int8_mac_share", []))) == 2
                else None
            ),
        ),
    )
    return context, evaluator


def _worker_environment(context: Any, request: dict[str, Any]) -> dict[str, Any]:
    from ..integration.runtime_environment import query_gpus

    physical_gpu_id = int(request["gpu_id"])
    gpu_rows = [
        row for row in query_gpus() if int(row.get("index", -1)) == physical_gpu_id
    ]
    return {
        "pid": os.getpid(),
        "controller_pid": int(request.get("controller_pid", 0) or 0),
        "physical_gpu_id": physical_gpu_id,
        "runtime_device": str(context.runtime_device),
        "gpu": gpu_rows[0] if gpu_rows else {},
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "nested_tensorrt_cuda_visible_devices": str(physical_gpu_id),
        "allowed_gpu_pids": sorted(
            int(value) for value in getattr(context, "allowed_gpu_pids", set())
        ),
    }


def _evaluate_task(evaluator: Any, task: dict[str, Any], gpu_id: int) -> dict[str, Any]:
    from ..candidate import CandidatePhenotype

    phenotype = CandidatePhenotype.from_dict(dict(task["phenotype"]))
    smoke_frames = int(task.get("smoke_frames", 0) or 0)
    if smoke_frames > 0:
        result = evaluator.evaluate_candidate_two_level(
            phenotype,
            output_dir=task["output_dir"],
            candidate_hash=str(task["candidate_hash"]),
            smoke_frames=smoke_frames,
            smoke_warmup_frames=int(task.get("smoke_warmup_frames", 10)),
        )
    else:
        result = evaluator.evaluate_candidate(
            phenotype,
            output_dir=task["output_dir"],
            candidate_hash=str(task["candidate_hash"]),
        )
    merged = {
        **dict(result),
        "seed_family": str(task.get("seed_family", "")),
        "stage1_metrics": dict(task.get("stage1_metrics", {}) or {}),
        "raw_precision_gene_hash": str(
            task.get("raw_precision_gene_hash", "")
        ),
        "repaired_precision_gene_hash": str(
            task.get("repaired_precision_gene_hash", "")
        ),
        "saturation_ratio": float(task.get("saturation_ratio", 0.0) or 0.0),
        "worker_gpu_id": int(gpu_id),
        "worker_pid": os.getpid(),
    }
    hashes = [
        str(merged.get(key, ""))
        for key in (
            "raw_precision_gene_hash",
            "repaired_precision_gene_hash",
            "requested_precision_profile_hash",
            "realized_precision_profile_hash",
        )
    ]
    merged["precision_identity_passed"] = bool(
        merged.get("precision_identity_passed", False)
        and all(hashes)
        and len(set(hashes)) == 1
    )
    if str(merged.get("status", "")) == "ok" and not merged[
        "precision_identity_passed"
    ]:
        merged["status"] = "precision_profile_hash_mismatch"
        merged["failure_reason"] = "raw_repaired_requested_realized_hash_mismatch"
        merged["F2"] = float("inf")
    return merged


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    worker_dir = Path(request["worker_dir"])
    queue_dir = Path(request["queue_dir"])
    stop_path = Path(request["stop_path"])
    poll_interval = float(request.get("poll_interval_seconds", 0.25))
    try:
        from ..integration.runtime_environment import require_gpu_isolation

        require_gpu_isolation(
            int(request["gpu_id"]),
            report_path=worker_dir / "gpu_startup_preflight.json",
            **_gpu_isolation_kwargs(request),
        )
        context, evaluator = _build_evaluator(request)
        environment = _worker_environment(context, request)
        _atomic_write_json(worker_dir / "environment.json", environment)
        _atomic_write_json(
            request["ready_path"],
            {"status": "ready", **environment},
        )
    except Exception as exc:  # noqa: BLE001
        _atomic_write_json(
            worker_dir / "startup_failure.json",
            {
                "status": "worker_start_failed",
                "failure_reason": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            },
        )
        raise

    while not stop_path.is_file():
        tasks = sorted(queue_dir.glob("*.task.json"))
        if not tasks:
            time.sleep(poll_interval)
            continue
        task_path = tasks[0]
        running_path = task_path.with_suffix(".running")
        try:
            os.replace(task_path, running_path)
        except FileNotFoundError:
            continue
        task = json.loads(running_path.read_text(encoding="utf-8"))
        try:
            result = _evaluate_task(evaluator, task, int(request["gpu_id"]))
        except Exception as exc:  # noqa: BLE001
            result = {
                "status": "stage2_worker_task_failed",
                "failure_reason": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "candidate_hash": str(task.get("candidate_hash", "")),
                "worker_gpu_id": int(request["gpu_id"]),
                "worker_pid": os.getpid(),
                "F2": float("inf"),
            }
        _atomic_write_json(task["result_path"], result)
        running_path.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
