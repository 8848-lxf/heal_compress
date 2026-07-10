#!/usr/bin/env python3
"""Dual-GPU queue runner for v11 mixed-precision profile evaluation.

The coordinator owns all top-level dataset indexes. Each worker subprocess owns
exactly one profile directory and runs the existing per-engine pipeline there.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[2]
_UNIAD = _ROOT.parent
for _p in (_UNIAD, _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tools.latency_lut import run_v11_mixed_precision_lut_dataset_builder as builder


GATE_FAILURE_STAGES = {
    "profile_legality_failed",
    "real_onnx_export_failed",
    "onnx_or_qdq_failed",
    "engine_build_failed",
    "engine_structure_mismatch",
    "engine_precision_mismatch",
    "trt_smoke_failed",
    "eval_failed",
}


@dataclass(frozen=True)
class ProfileJob:
    subnet_dir: Path
    subnet_id: str
    subnet_index: int
    structure_hash: str
    profile_id: str
    profile_index: int
    profile_dir: Path


@dataclass
class RunningWorker:
    job: ProfileJob
    gpu: str
    process: subprocess.Popen
    log_handle: Any
    log_path: Path
    result_path: Path


@dataclass
class IndexState:
    output_dir: Path
    args: argparse.Namespace
    subnet_rows: list[dict[str, Any]]
    profile_rows: list[dict[str, Any]] = field(default_factory=list)
    eval_rows: list[dict[str, Any]] = field(default_factory=list)
    component_rows: list[dict[str, Any]] = field(default_factory=list)
    training_rows: list[dict[str, Any]] = field(default_factory=list)
    failure_rows: list[dict[str, Any]] = field(default_factory=list)
    progress_state: dict[str, Any] = field(default_factory=dict)
    consecutive_failures: int = 0
    total_gate_failures: int = 0
    failed_profiles: list[dict[str, Any]] = field(default_factory=list)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="v11 dual-GPU per-profile worker queue")
    parser.add_argument("--source-dir", default="outputs/latency_lut/v11_mixed_precision_lut_dataset_trt_full")
    parser.add_argument("--max-subnets", type=int, default=50)
    parser.add_argument("--precision-profiles-per-subnet", type=int, default=4)
    parser.add_argument("--gpus", default="6,7")
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--log-dir", default="")
    parser.add_argument("--profile-sampling-policy", choices=["stratified", "random"], default="stratified")
    parser.add_argument("--profile-seed", type=int, default=20260708)
    parser.add_argument("--precision-modes", default="fp32,fp16,int8")
    parser.add_argument("--overwrite-profiles", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite-invalid-profiles", action="store_true")
    parser.add_argument("--force-reeval", action="store_true")
    parser.add_argument("--allow-precision-mismatch-eval", action="store_true")
    parser.add_argument("--fail-fast-on-profile-error", action="store_true")
    parser.add_argument("--quarantine-invalid-profile", action="store_true", default=True)
    parser.add_argument("--max-consecutive-failures", type=int, default=5)
    parser.add_argument("--max-total-gate-failures", type=int, default=10)
    parser.add_argument("--build-engines", type=builder.str2bool, default=True)
    parser.add_argument("--eval-engines", type=builder.str2bool, default=True)
    parser.add_argument("--target-pruning-mode", default="param", choices=["param", "channel"])
    parser.add_argument("--round-to", type=int, default=4)
    parser.add_argument("--max-ch-sparsity", type=float, default=0.60)
    parser.add_argument("--stage1-min-per-group", type=int, default=8)
    parser.add_argument("--stage1-max-ch-sparsity", type=float, default=0.30)
    parser.add_argument("--protect-fpn-output", action="store_true", default=True)
    parser.add_argument("--protect-head-output", action="store_true", default=True)
    parser.add_argument("--no-extra-output-protection", action="store_true", default=True)
    parser.add_argument("--calib-train-frames", type=int, default=200)
    parser.add_argument("--warmup-frames", type=int, default=100)
    parser.add_argument("--eval-frames", type=int, default=1000)
    parser.add_argument("--smoke-frames", type=int, default=5)
    parser.add_argument("--ap-thresholds", default="0.03,0.30,0.50,0.70")
    parser.add_argument("--scale-method", default="percentile")
    parser.add_argument("--random-seed", type=int, default=1100)
    parser.add_argument("--device", default="")
    parser.add_argument("--num-calib-batches", type=int, default=8)
    parser.add_argument("--trt-root", default="/home/lixingfeng/UniAD_examine/HEAL/prune_model/TensorRT-10.9_x86_cu118")
    parser.add_argument("--trtexec", default="")
    parser.add_argument("--trt-build-timeout-seconds", type=int, default=180)
    parser.add_argument("--fixed-k", "--fixed_k", dest="fixed_k", type=int, default=29696)
    parser.add_argument("--plugin", default="quantization/plugins/pointpillar_scatter_trt/build/libpointpillar_scatter_trt.so")
    parser.add_argument("--enable-synthetic-trt-eval", action="store_true")
    parser.add_argument("--cuda-visible-devices", default="")
    parser.add_argument("--heal-root", default="/home/lixingfeng/UniAD_examine/HEAL")
    parser.add_argument("--model-config", default="/home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_pyramid/config.yaml")
    parser.add_argument("--checkpoint", default="/home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_pyramid/net_epoch_bestval_at17.pth")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--auto-select-idle-gpu", action="store_true")
    parser.add_argument("--max-gpu-utilization", type=int, default=5)
    parser.add_argument("--max-gpu-memory-ratio", type=float, default=0.20)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--mode", default="dual-gpu-profile-workers")

    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-subnet-dir", default="", help=argparse.SUPPRESS)
    parser.add_argument("--worker-profile-index", type=int, default=-1, help=argparse.SUPPRESS)
    parser.add_argument("--worker-gpu", default="", help=argparse.SUPPRESS)
    parser.add_argument("--worker-result-path", default="", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not args.output_dir:
        args.output_dir = args.source_dir
    return args


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return value


def _read_manifest(subnet_dir: Path) -> dict[str, Any]:
    return json.loads((subnet_dir / "pruning_manifest.json").read_text(encoding="utf-8"))


def _subnet_index_from_id(subnet_id: str, fallback: int) -> int:
    try:
        return int(str(subnet_id).split("_")[-1])
    except ValueError:
        return fallback


def _subnet_row(subnet_dir: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    subnet_id = str(manifest.get("subnet_id", subnet_dir.name))
    return {
        "subnet_id": subnet_id,
        "structure_hash": manifest.get("structure_hash", ""),
        "target_param_prune_ratio": manifest.get("target_pruning_ratio", manifest.get("actual_param_prune_ratio", "")),
        "actual_param_prune_ratio": manifest.get("actual_param_prune_ratio", ""),
        "actual_channel_prune_ratio": manifest.get("actual_channel_prune_ratio", ""),
        "params_after": manifest.get("params_after", ""),
        "pruning_manifest_path": str(subnet_dir / "pruning_manifest.json"),
        "shape_invariant_passed": manifest.get("shape_invariant_passed", True),
        "precision_coupling_groups_path": str(subnet_dir / "precision_coupling_groups.json"),
    }


def build_pending_jobs(args: argparse.Namespace) -> tuple[list[ProfileJob], list[dict[str, Any]]]:
    subnet_dirs = builder._subnet_dirs_for_expand(args)
    jobs: list[ProfileJob] = []
    subnet_rows: list[dict[str, Any]] = []
    for fallback_index, subnet_dir in enumerate(subnet_dirs):
        manifest = _read_manifest(subnet_dir)
        subnet_id = str(manifest.get("subnet_id", subnet_dir.name))
        subnet_index = _subnet_index_from_id(subnet_id, fallback_index)
        structure_hash = str(manifest.get("structure_hash", ""))
        subnet_rows.append(_subnet_row(subnet_dir, manifest))
        for profile_index in range(int(args.precision_profiles_per_subnet)):
            profile_id = f"profile_{profile_index:03d}"
            profile_dir = subnet_dir / profile_id
            if bool(args.resume) and not bool(args.force_reeval) and builder._profile_success_complete(profile_dir):
                continue
            jobs.append(
                ProfileJob(
                    subnet_dir=subnet_dir,
                    subnet_id=subnet_id,
                    subnet_index=subnet_index,
                    structure_hash=structure_hash,
                    profile_id=profile_id,
                    profile_index=profile_index,
                    profile_dir=profile_dir,
                )
            )
    return jobs, subnet_rows


def _load_or_generate_profile(args: argparse.Namespace, job: ProfileJob) -> tuple[dict[str, Any], list[Any], set[str]]:
    groups = builder._precision_groups_from_json(job.subnet_dir / "precision_coupling_groups.json")
    normalized_existing: dict[int, dict[str, Any]] = {}
    for profile_path in builder._profile_paths(job.subnet_dir):
        try:
            idx = int(profile_path.parent.name.split("_")[-1])
        except ValueError:
            continue
        normalized_existing[idx] = builder._normalize_existing_profile(
            builder._read_profile(profile_path),
            profile_index=idx,
            subnet_id=job.subnet_id,
            structure_hash=job.structure_hash,
        )
    other_hashes = {
        str(profile.get("precision_assignment_hash", ""))
        for idx, profile in normalized_existing.items()
        if idx != job.profile_index
    }
    if job.profile_index == 0 and job.profile_index in normalized_existing and not bool(args.overwrite_profiles):
        profile = normalized_existing[job.profile_index]
    elif (
        job.profile_index in normalized_existing
        and not bool(args.overwrite_profiles)
        and builder._profile_matches_per_engine_template(normalized_existing[job.profile_index], job.profile_index)
    ):
        profile = normalized_existing[job.profile_index]
    else:
        profile = builder._generate_missing_profile(
            args=args,
            subnet_id=job.subnet_id,
            subnet_index=job.subnet_index,
            structure_hash=job.structure_hash,
            groups=groups,
            profile_index=job.profile_index,
            existing_hashes=other_hashes,
        )
        if profile is None:
            profile = builder.sample_stratified_mixed_precision_profile(
                subnet_id=job.subnet_id,
                structure_hash=job.structure_hash,
                groups=groups,
                profile_index=job.profile_index,
                profile_seed=int(args.profile_seed) + 99991,
                subnet_index=job.subnet_index,
                precision_modes=[p.strip().lower() for p in str(args.precision_modes).split(",") if p.strip()],
            )
    profile["profile_id"] = job.profile_id
    profile["profile_index"] = job.profile_index
    return profile, groups, other_hashes


def create_empty_index_state(output_dir: Path, *, subnet_rows: list[dict[str, Any]], args: argparse.Namespace) -> IndexState:
    output_dir.mkdir(parents=True, exist_ok=True)
    if bool(args.resume):
        profile_rows, eval_rows, component_rows, training_rows, failure_rows = builder._load_existing_index_state(output_dir)
    else:
        profile_rows, eval_rows, component_rows, training_rows, failure_rows = [], [], [], [], []
    progress_state = {
        "mode": "dual-gpu-profile-workers",
        "reuse_existing_subnets": True,
        "new_pruning_performed": False,
        "total_subnets": len(subnet_rows),
        "profiles_per_subnet": int(args.precision_profiles_per_subnet),
        "profiles_started": 0,
        "profiles_completed": 0,
        "profiles_failed": 0,
        "engines_built": 0,
        "engines_structure_checked": 0,
        "engines_precision_checked": 0,
        "engines_smoke_passed": 0,
        "engines_eval_success": 0,
        "last_completed_profile": "",
        "current_profile": "",
        "last_failure": None,
        "failure_stage_counts": {},
        "failed_profiles": [],
        "stopped_early": False,
        "stop_reason": "",
    }
    return IndexState(
        output_dir=output_dir,
        args=args,
        subnet_rows=subnet_rows,
        profile_rows=[dict(row) for row in profile_rows],
        eval_rows=[dict(row) for row in eval_rows],
        component_rows=[dict(row) for row in component_rows],
        training_rows=[dict(row) for row in training_rows],
        failure_rows=[dict(row) for row in failure_rows],
        progress_state=progress_state,
    )


def _write_jsonl_overwrite(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(_jsonable(dict(row)), ensure_ascii=False, default=str) + "\n")


def _write_pipeline_report(state: IndexState) -> None:
    progress = state.progress_state
    lines = [
        "# v11 Dual-GPU Per-Engine Pipeline Report",
        "",
        f"- stopped_early: {str(bool(progress.get('stopped_early'))).lower()}",
        f"- stop_reason: {progress.get('stop_reason', '')}",
        f"- successful_profile_count: {sum(builder.truthy(row.get('eval_success')) for row in state.profile_rows)}",
        f"- build_success_count: {progress.get('engines_built', 0)}",
        f"- structure_check_passed_count: {progress.get('engines_structure_checked', 0)}",
        f"- precision_check_passed_count: {progress.get('engines_precision_checked', 0)}",
        f"- smoke_passed_count: {progress.get('engines_smoke_passed', 0)}",
        f"- real_eval_success_count: {progress.get('engines_eval_success', 0)}",
        f"- failure_stage_counts: {progress.get('failure_stage_counts', {})}",
    ]
    (state.output_dir / "per_engine_pipeline_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_index_state(state: IndexState) -> None:
    output_dir = state.output_dir
    builder.write_csv(output_dir / "subnet_index.csv", state.subnet_rows)
    builder.write_csv(output_dir / "mixed_precision_profile_index.csv", state.profile_rows)
    builder.write_csv(output_dir / "engine_eval_summary.csv", state.eval_rows)
    builder.write_csv(output_dir / "component_lut_samples.csv", state.component_rows)
    _write_jsonl_overwrite(output_dir / "full_engine_training_samples.jsonl", state.training_rows)
    builder.write_csv(output_dir / "failure_summary.csv", state.failure_rows)
    builder.write_json(output_dir / "progress_state.json", state.progress_state)
    manifest = {
        "dataset_version": builder.DATASET_VERSION,
        "mode": "dual-gpu-profile-workers",
        "reuse_existing_subnets": True,
        "new_pruning_performed": False,
        "successful_subnet_count": len(state.subnet_rows),
        "profiles_per_subnet_requested": int(state.args.precision_profiles_per_subnet),
        "successful_profile_count": len(state.profile_rows),
        "successful_engine_count": sum(builder.truthy(row.get("build_success")) for row in state.profile_rows),
        "engine_structure_checked_count": sum(builder.truthy(row.get("engine_structure_check_passed")) for row in state.profile_rows),
        "engine_precision_checked_count": sum(builder.truthy(row.get("precision_realization_passed")) for row in state.profile_rows),
        "engine_smoke_passed_count": sum(builder.truthy(row.get("smoke_success")) for row in state.profile_rows),
        "engine_eval_success_count": sum(
            builder.truthy(row.get("eval_success"))
            and not builder.truthy(row.get("synthetic_used"))
            and builder.truthy(row.get("validation_dataloader_used"))
            for row in state.eval_rows
        ),
        "stopped_early": bool(state.progress_state.get("stopped_early")),
        "stop_reason": state.progress_state.get("stop_reason", ""),
        "failure_summary": dict(state.progress_state.get("failure_stage_counts", {})),
        "failed_profiles": list(state.progress_state.get("failed_profiles", [])),
    }
    builder.write_json(output_dir / "dataset_manifest.json", manifest)
    builder.write_json(output_dir / "manifest.json", manifest)
    _write_pipeline_report(state)


def _failure_from_result(result: Mapping[str, Any]) -> dict[str, Any]:
    index_row = result.get("index_row") if isinstance(result.get("index_row"), Mapping) else {}
    eval_row = result.get("eval_row") if isinstance(result.get("eval_row"), Mapping) else {}
    subnet_id = str(index_row.get("subnet_id") or eval_row.get("subnet_id") or result.get("subnet_id", ""))
    profile_id = str(index_row.get("profile_id") or eval_row.get("profile_id") or result.get("profile_id", ""))
    status = str(result.get("status", "worker_failed"))
    failure = result.get("failure")
    if isinstance(failure, Mapping):
        return dict(failure)
    return {
        "subnet_id": subnet_id,
        "profile_id": profile_id,
        "stage_failed": status,
        "failure_reason": str(result.get("failure_reason") or status),
        "traceback": str(result.get("traceback", "")),
        "recovery_action": "resume_after_fix",
    }


def _remove_profile_rows(rows: Sequence[Mapping[str, Any]], subnet_id: str, profile_id: str) -> list[dict[str, Any]]:
    return [dict(row) for row in rows if builder._row_key(row) != (subnet_id, profile_id)]


def _refresh_progress_counts(state: IndexState) -> None:
    progress = state.progress_state
    progress["engines_built"] = sum(builder.truthy(row.get("build_success")) for row in state.profile_rows)
    progress["engines_structure_checked"] = sum(builder.truthy(row.get("engine_structure_check_passed")) for row in state.profile_rows)
    progress["engines_precision_checked"] = sum(builder.truthy(row.get("precision_realization_passed")) for row in state.profile_rows)
    progress["engines_smoke_passed"] = sum(builder.truthy(row.get("smoke_success")) for row in state.profile_rows)
    progress["engines_eval_success"] = sum(
        builder.truthy(row.get("eval_success"))
        and not builder.truthy(row.get("synthetic_used"))
        and builder.truthy(row.get("validation_dataloader_used"))
        for row in state.eval_rows
    )


def apply_worker_result(
    state: IndexState,
    result: Mapping[str, Any],
    *,
    total_subnets: int,
    profiles_per_subnet: int,
) -> bool:
    progress = state.progress_state
    progress["total_subnets"] = total_subnets
    progress["profiles_per_subnet"] = profiles_per_subnet
    status = str(result.get("status", "worker_failed"))
    index_row = dict(result.get("index_row") or {})
    eval_row = dict(result.get("eval_row") or {})
    subnet_id = str(index_row.get("subnet_id") or eval_row.get("subnet_id") or result.get("subnet_id", ""))
    profile_id = str(index_row.get("profile_id") or eval_row.get("profile_id") or result.get("profile_id", ""))

    if index_row:
        state.profile_rows = builder._replace_row(state.profile_rows, index_row)
    if eval_row:
        state.eval_rows = builder._replace_row(state.eval_rows, eval_row)
    state.component_rows = _remove_profile_rows(state.component_rows, subnet_id, profile_id) + [dict(row) for row in result.get("component_rows", [])]
    if result.get("training_row"):
        state.training_rows = _remove_profile_rows(state.training_rows, subnet_id, profile_id) + [dict(result["training_row"])]

    if status == "eval_success":
        state.consecutive_failures = 0
        state.failure_rows = builder._remove_row(state.failure_rows, subnet_id=subnet_id, profile_id=profile_id)
        if progress.get("last_failure") and builder._row_key(progress["last_failure"]) == (subnet_id, profile_id):
            progress["last_failure"] = None
    else:
        state.consecutive_failures += 1
        failure = _failure_from_result(result)
        if status in GATE_FAILURE_STAGES:
            state.total_gate_failures += 1
        state.failure_rows = builder._replace_row(state.failure_rows, failure)
        state.failed_profiles.append(failure)
        progress["profiles_failed"] = int(progress.get("profiles_failed", 0)) + 1
        progress["last_failure"] = failure
        counts = dict(progress.get("failure_stage_counts", {}))
        stage = str(failure.get("stage_failed") or status)
        counts[stage] = counts.get(stage, 0) + 1
        progress["failure_stage_counts"] = counts
        progress["failed_profiles"] = state.failed_profiles

    progress["profiles_completed"] = int(progress.get("profiles_completed", 0)) + 1
    progress["last_completed_profile"] = f"{subnet_id}/{profile_id}"
    progress["current_profile"] = ""
    _refresh_progress_counts(state)

    stopped = (
        state.consecutive_failures >= int(state.args.max_consecutive_failures)
        or state.total_gate_failures >= int(state.args.max_total_gate_failures)
    )
    if stopped:
        progress["stopped_early"] = True
        progress["stop_reason"] = (
            f"failure_threshold:consecutive={state.consecutive_failures},"
            f"total_gate={state.total_gate_failures}"
        )
    write_index_state(state)
    return stopped


def build_worker_command(
    args: argparse.Namespace,
    job: ProfileJob,
    *,
    gpu: str,
    result_path: Path,
) -> tuple[list[str], dict[str, str], Path]:
    log_dir = Path(args.log_dir) if str(args.log_dir) else Path(args.source_dir) / "dual_gpu_worker_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{job.subnet_id}_{job.profile_id}_gpu{gpu}.log"
    cmd = [
        sys.executable,
        str(_THIS),
        "--worker",
        "--source-dir",
        str(args.source_dir),
        "--max-subnets",
        str(args.max_subnets),
        "--precision-profiles-per-subnet",
        str(args.precision_profiles_per_subnet),
        "--profile-sampling-policy",
        str(args.profile_sampling_policy),
        "--profile-seed",
        str(args.profile_seed),
        "--precision-modes",
        str(args.precision_modes),
        "--build-engines",
        str(bool(args.build_engines)).lower(),
        "--eval-engines",
        str(bool(args.eval_engines)).lower(),
        "--warmup-frames",
        str(args.warmup_frames),
        "--eval-frames",
        str(args.eval_frames),
        "--smoke-frames",
        str(args.smoke_frames),
        "--ap-thresholds",
        str(args.ap_thresholds),
        "--calib-train-frames",
        str(args.calib_train_frames),
        "--trt-root",
        str(args.trt_root),
        "--trt-build-timeout-seconds",
        str(args.trt_build_timeout_seconds),
        "--fixed-k",
        str(args.fixed_k),
        "--plugin",
        str(args.plugin),
        "--heal-root",
        str(args.heal_root),
        "--model-config",
        str(args.model_config),
        "--checkpoint",
        str(args.checkpoint),
        "--num-workers",
        str(args.num_workers),
        "--worker-subnet-dir",
        str(job.subnet_dir),
        "--worker-profile-index",
        str(job.profile_index),
        "--worker-gpu",
        str(gpu),
        "--worker-result-path",
        str(result_path),
        "--cuda-visible-devices",
        str(gpu),
    ]
    if args.trtexec:
        cmd.extend(["--trtexec", str(args.trtexec)])
    if bool(args.resume):
        cmd.append("--resume")
    if bool(args.overwrite_invalid_profiles):
        cmd.append("--overwrite-invalid-profiles")
    if bool(args.overwrite_profiles):
        cmd.append("--overwrite-profiles")
    if bool(args.force_reeval):
        cmd.append("--force-reeval")
    if bool(args.allow_precision_mismatch_eval):
        cmd.append("--allow-precision-mismatch-eval")
    if bool(args.enable_synthetic_trt_eval):
        cmd.append("--enable-synthetic-trt-eval")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    trt_root = Path(str(args.trt_root))
    ld_parts = [
        str(trt_root / "lib"),
        str(trt_root / "targets/x86_64-linux-gnu/lib"),
        "/home/lixingfeng/anaconda3/envs/modelopt/lib",
    ]
    if env.get("LD_LIBRARY_PATH"):
        ld_parts.append(env["LD_LIBRARY_PATH"])
    env["LD_LIBRARY_PATH"] = ":".join(ld_parts)
    pythonpath_parts = [str(_UNIAD), str(_ROOT), str(Path(args.heal_root))]
    if env.get("PYTHONPATH"):
        pythonpath_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = ":".join(pythonpath_parts)
    return cmd, env, log_path


def _minimal_failure_result(job: ProfileJob, *, status: str, reason: str, traceback_text: str = "") -> dict[str, Any]:
    failure = {
        "subnet_id": job.subnet_id,
        "profile_id": job.profile_id,
        "stage_failed": status,
        "failure_reason": reason,
        "traceback": traceback_text,
        "recovery_action": "inspect_worker_log_and_resume",
    }
    return {
        "status": status,
        "subnet_id": job.subnet_id,
        "profile_id": job.profile_id,
        "index_row": {
            "subnet_id": job.subnet_id,
            "profile_id": job.profile_id,
            "structure_hash": job.structure_hash,
            "build_success": False,
            "eval_success": False,
            "failure_reason": reason,
            "status": status,
        },
        "eval_row": {
            "subnet_id": job.subnet_id,
            "profile_id": job.profile_id,
            "structure_hash": job.structure_hash,
            "build_success": False,
            "eval_success": False,
            "failure_reason": reason,
            "status": status,
            "synthetic_used": False,
            "validation_dataloader_used": False,
        },
        "component_rows": [],
        "training_row": {
            "subnet_id": job.subnet_id,
            "profile_id": job.profile_id,
            "structure_hash": job.structure_hash,
            "build_success": False,
            "eval_success": False,
            "label_available": False,
            "failure_reason": reason,
        },
        "failure": failure,
    }


def run_worker(args: argparse.Namespace) -> int:
    if args.worker_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.worker_gpu)
        args.cuda_visible_devices = str(args.worker_gpu)
    subnet_dir = Path(args.worker_subnet_dir)
    manifest = _read_manifest(subnet_dir)
    subnet_id = str(manifest.get("subnet_id", subnet_dir.name))
    subnet_index = _subnet_index_from_id(subnet_id, 0)
    structure_hash = str(manifest.get("structure_hash", ""))
    profile_index = int(args.worker_profile_index)
    profile_id = f"profile_{profile_index:03d}"
    profile_dir = subnet_dir / profile_id
    result_path = Path(args.worker_result_path) if args.worker_result_path else profile_dir / "worker_result.json"
    job = ProfileJob(
        subnet_dir=subnet_dir,
        subnet_id=subnet_id,
        subnet_index=subnet_index,
        structure_hash=structure_hash,
        profile_id=profile_id,
        profile_index=profile_index,
        profile_dir=profile_dir,
    )
    profile_dir.mkdir(parents=True, exist_ok=True)
    try:
        profile, groups, other_hashes = _load_or_generate_profile(args, job)
        ctx = {
            "args": args,
            "subnet_dir": subnet_dir,
            "subnet_id": subnet_id,
            "subnet_index": subnet_index,
            "structure_hash": structure_hash,
            "groups": groups,
            "profile": profile,
            "profile_id": profile_id,
            "profile_index": profile_index,
            "profile_dir": profile_dir,
            "existing_hashes": other_hashes,
        }
        result = builder.run_one_profile_pipeline(ctx)
        result["worker_gpu"] = str(args.worker_gpu)
        result["subnet_id"] = subnet_id
        result["profile_id"] = profile_id
        builder.write_json(result_path, _jsonable(result))
        return 0
    except Exception as exc:  # noqa: BLE001
        result = _minimal_failure_result(
            job,
            status="worker_crashed",
            reason=f"{type(exc).__name__}: {exc}",
            traceback_text=traceback.format_exc(),
        )
        builder.write_json(profile_dir / "profile_failure_report.json", result["failure"])
        builder.write_json(result_path, result)
        return 1


def _read_worker_result(path: Path, job: ProfileJob, returncode: int) -> dict[str, Any]:
    if path.is_file():
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
            if returncode != 0 and str(result.get("status", "")) == "eval_success":
                result["status"] = "worker_returncode_failed"
                result["failure_reason"] = f"worker returncode={returncode}"
            return result
        except Exception as exc:  # noqa: BLE001
            return _minimal_failure_result(job, status="worker_result_parse_failed", reason=f"{type(exc).__name__}: {exc}")
    return _minimal_failure_result(job, status="worker_result_missing", reason=f"worker returncode={returncode}, result missing: {path}")


def _gpus(args: argparse.Namespace) -> list[str]:
    return [gpu.strip() for gpu in str(args.gpus).split(",") if gpu.strip()]


def _update_current_profiles(state: IndexState, active: Mapping[str, RunningWorker]) -> None:
    state.progress_state["current_profile"] = ",".join(
        f"{worker.job.subnet_id}/{worker.job.profile_id}@gpu{gpu}" for gpu, worker in sorted(active.items())
    )
    builder.write_json(state.output_dir / "progress_state.json", state.progress_state)


def terminate_active_workers(active: Mapping[str, RunningWorker]) -> None:
    for worker in active.values():
        if worker.process.poll() is None:
            worker.process.terminate()
    deadline = time.time() + 30
    for worker in active.values():
        if worker.process.poll() is not None:
            continue
        remaining = max(0.0, deadline - time.time())
        try:
            worker.process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            worker.process.kill()


def run_coordinator(args: argparse.Namespace) -> int:
    output_dir = Path(args.source_dir)
    jobs, subnet_rows = build_pending_jobs(args)
    state = create_empty_index_state(output_dir, subnet_rows=subnet_rows, args=args)
    state.progress_state["queued_profiles"] = len(jobs)
    write_index_state(state)
    gpus = _gpus(args)
    if not gpus:
        raise ValueError("--gpus must contain at least one GPU id")
    max_workers = max(1, min(int(args.max_workers), len(gpus)))
    pending = list(jobs)
    active: dict[str, RunningWorker] = {}
    stopped = False

    while pending or active:
        while pending and len(active) < max_workers:
            gpu = next((candidate for candidate in gpus if candidate not in active), None)
            if gpu is None:
                break
            job = pending.pop(0)
            result_path = job.profile_dir / "worker_result.json"
            if result_path.exists():
                result_path.unlink()
            cmd, env, log_path = build_worker_command(args, job, gpu=gpu, result_path=result_path)
            log_handle = log_path.open("w", encoding="utf-8")
            state.progress_state["profiles_started"] = int(state.progress_state.get("profiles_started", 0)) + 1
            process = subprocess.Popen(cmd, cwd=str(_ROOT), env=env, stdout=log_handle, stderr=subprocess.STDOUT)
            active[gpu] = RunningWorker(job=job, gpu=gpu, process=process, log_handle=log_handle, log_path=log_path, result_path=result_path)
            _update_current_profiles(state, active)

        completed_gpus: list[str] = []
        for gpu, worker in list(active.items()):
            returncode = worker.process.poll()
            if returncode is None:
                continue
            worker.log_handle.close()
            result = _read_worker_result(worker.result_path, worker.job, int(returncode))
            completed_gpus.append(gpu)
            stopped = apply_worker_result(
                state,
                result,
                total_subnets=len(subnet_rows),
                profiles_per_subnet=int(args.precision_profiles_per_subnet),
            )
            if bool(args.fail_fast_on_profile_error) and str(result.get("status")) != "eval_success":
                state.progress_state["stopped_early"] = True
                state.progress_state["stop_reason"] = f"fail_fast_on_profile_error:{worker.job.subnet_id}/{worker.job.profile_id}"
                write_index_state(state)
                stopped = True
            if stopped:
                break
        for gpu in completed_gpus:
            active.pop(gpu, None)
        if stopped:
            terminate_active_workers(active)
            for worker in active.values():
                worker.log_handle.close()
            active.clear()
            write_index_state(state)
            print(json.dumps({"success": False, "stopped_early": True, "stop_reason": state.progress_state.get("stop_reason", ""), "output_dir": str(output_dir)}, indent=2))
            return 2
        if active:
            time.sleep(float(args.poll_seconds))

    state.progress_state["current_profile"] = ""
    write_index_state(state)
    print(json.dumps({"success": True, "output_dir": str(output_dir), "profiles_completed": state.progress_state.get("profiles_completed", 0)}, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.worker:
        return run_worker(args)
    return run_coordinator(args)


if __name__ == "__main__":
    raise SystemExit(main())
