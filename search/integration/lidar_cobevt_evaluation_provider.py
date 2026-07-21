"""Subprocess provider for CoBEVT TensorRT AP and latency evaluation."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from .runtime_environment import modelopt_python_command, modelopt_subprocess_env


def worker_module() -> str:
    return "search.integration.lidar_cobevt_evaluation_worker"


def _logical_device(device: str) -> tuple[str | None, str]:
    text = str(device)
    if text.startswith("cuda:") and text.split(":", 1)[1].isdigit():
        return text.split(":", 1)[1], "cuda:0"
    return None, text


def build_cobevt_evaluation_request(
    *,
    engine_path: str | Path,
    checkpoint: str | Path,
    model_config: str | Path,
    heal_root: str | Path,
    device: str,
    output_path: str | Path,
    plugin_path: str | Path,
    additional_plugin_paths: list[str | Path] | tuple[str | Path, ...] = (),
    fixed_k: int,
    num_frames: int,
    warmup_frames: int,
    eval_manifest_path: str | Path,
    num_workers: int = 8,
    ap_iou_backend: str = "gpu",
    latency_rounds: int = 1,
    warmup_latency_rounds: int | None = None,
) -> dict[str, Any]:
    workers = int(num_workers)
    backend = str(ap_iou_backend).lower()
    if workers != 8:
        raise ValueError("cobevt_evaluation_num_workers_must_equal_8")
    if backend != "gpu":
        raise ValueError("cobevt_evaluation_gpu_ap_iou_required")
    visible, logical = _logical_device(device)
    timed_rounds = max(1, int(latency_rounds))
    warmup_rounds = timed_rounds if warmup_latency_rounds is None else max(
        1, int(warmup_latency_rounds)
    )
    return {
        "ap_iou_backend": backend,
        "checkpoint": str(checkpoint),
        "cuda_visible_devices": visible or "",
        "device": logical,
        "engine_path": str(engine_path),
        "eval_manifest_path": str(eval_manifest_path),
        "fixed_k": int(fixed_k),
        "heal_root": str(heal_root),
        "latency_rounds": timed_rounds,
        "warmup_latency_rounds": warmup_rounds,
        "model_config": str(model_config),
        "model_family": "lidar_cobevt",
        "num_frames": int(num_frames),
        "num_workers": workers,
        "output_path": str(output_path),
        "physical_device": str(device),
        "plugin_path": str(plugin_path),
        "additional_plugin_paths": [str(value) for value in additional_plugin_paths],
        "strict_gpu_ap_iou": True,
        "warmup_frames": int(warmup_frames),
    }


def evaluate_cobevt_engine_modelopt(
    *,
    engine_path: str | Path,
    checkpoint: str | Path,
    model_config: str | Path,
    heal_root: str | Path,
    device: str,
    output_dir: str | Path,
    tensorrt_root: str | Path,
    plugin_path: str | Path,
    additional_plugin_paths: list[str | Path] | tuple[str | Path, ...] = (),
    fixed_k: int,
    num_frames: int,
    warmup_frames: int,
    eval_manifest_path: str | Path,
    num_workers: int = 8,
    ap_iou_backend: str = "gpu",
    strict_gpu_ap_iou: bool = True,
    latency_rounds: int = 1,
    warmup_latency_rounds: int | None = None,
    conda_env: str = "modelopt",
) -> dict[str, Any]:
    if not strict_gpu_ap_iou:
        raise ValueError("cobevt_evaluation_strict_gpu_ap_iou_required")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    output_path = destination / "evaluation.json"
    request = build_cobevt_evaluation_request(
        engine_path=engine_path,
        checkpoint=checkpoint,
        model_config=model_config,
        heal_root=heal_root,
        device=device,
        output_path=output_path,
        plugin_path=plugin_path,
        additional_plugin_paths=additional_plugin_paths,
        fixed_k=fixed_k,
        num_frames=num_frames,
        warmup_frames=warmup_frames,
        eval_manifest_path=eval_manifest_path,
        num_workers=num_workers,
        ap_iou_backend=ap_iou_backend,
        latency_rounds=latency_rounds,
        warmup_latency_rounds=warmup_latency_rounds,
    )
    request_path = destination / "evaluation_request.json"
    request_path.write_text(
        json.dumps(request, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    root = Path(tensorrt_root)
    repo_root = Path(__file__).resolve().parents[2]
    env = modelopt_subprocess_env(
        tensorrt_root=root,
        conda_env=conda_env,
        pythonpath_entries=[
            str(repo_root),
            str(Path(heal_root)),
            str(Path(heal_root).parent),
        ],
        cuda_visible_devices=request["cuda_visible_devices"] or None,
    )
    env["LD_LIBRARY_PATH"] = ":".join(
        (
            str(root / "targets" / "x86_64-linux-gnu" / "lib"),
            str(root / "lib"),
            env.get("LD_LIBRARY_PATH", ""),
        )
    )
    command = modelopt_python_command(conda_env) + [
        "-m",
        worker_module(),
        "--request",
        str(request_path),
    ]
    completed = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        check=False,
    )
    log_path = destination / "evaluation_worker.log"
    log_path.write_text(completed.stdout or "", encoding="utf-8")
    if not output_path.is_file():
        return {
            "status": "evaluation_failed",
            "failure_reason": f"worker_no_output_rc_{completed.returncode}",
            "log_path": str(log_path),
        }
    result = json.loads(output_path.read_text(encoding="utf-8"))
    result["worker_returncode"] = int(completed.returncode)
    result["log_path"] = str(log_path)
    return result


__all__ = [
    "build_cobevt_evaluation_request",
    "evaluate_cobevt_engine_modelopt",
    "worker_module",
]
