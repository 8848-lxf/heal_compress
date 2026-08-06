"""Subprocess provider for TensorRT AP/latency evaluation."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from deploy.post_scatter import POST_SCATTER_CONTRACT

from .runtime_environment import modelopt_python_command, modelopt_subprocess_env


DEFAULT_AP_IOU_BACKEND = "gpu"
DEFAULT_TORCH_NUM_THREADS = 4
DEFAULT_DATALOADER_NUM_WORKERS = 8
DEFAULT_VOXELIZATION_BACKEND = "gpu"


def _cuda_visible_and_logical_device(device: str) -> tuple[str | None, str]:
    text = str(device)
    if text.startswith("cuda:"):
        suffix = text.split(":", 1)[1]
        if suffix.isdigit():
            return suffix, "cuda:0"
    return None, text


def evaluate_engine_modelopt(
    *,
    engine_path: str | Path,
    checkpoint: str | Path,
    model_config: str | Path,
    heal_root: str | Path,
    device: str,
    physical_gpu_id: int | None = None,
    output_dir: str | Path,
    tensorrt_root: str | Path,
    plugin_path: str | Path | None,
    num_frames: int,
    warmup_frames: int,
    latency_rounds: int = 1,
    conda_env: str = "modelopt",
    eval_manifest_path: str | Path | None = None,
    ap_iou_backend: str = DEFAULT_AP_IOU_BACKEND,
    require_cuda_postprocess: bool = True,
    torch_num_threads: int = DEFAULT_TORCH_NUM_THREADS,
    dataloader_num_workers: int = DEFAULT_DATALOADER_NUM_WORKERS,
    voxelization_backend: str = DEFAULT_VOXELIZATION_BACKEND,
    evaluation_seed: int = 0,
    checkpoint_path: str | Path | None = None,
) -> dict[str, Any]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    request_path = destination / "evaluation_request.json"
    output_path = destination / "evaluation.json"
    if physical_gpu_id is None:
        cuda_visible_devices, worker_device = _cuda_visible_and_logical_device(
            str(device)
        )
        physical_device = str(device)
    else:
        cuda_visible_devices = str(int(physical_gpu_id))
        worker_device = "cuda:0" if str(device).startswith("cuda:") else str(device)
        physical_device = f"cuda:{int(physical_gpu_id)}"
    request = {
        "engine_path": str(engine_path),
        "checkpoint": str(checkpoint),
        "model_config": str(model_config),
        "heal_root": str(heal_root),
        "device": worker_device,
        "physical_device": physical_device,
        "output_path": str(output_path),
        "plugin_path": str(plugin_path) if plugin_path else "",
        "num_frames": int(num_frames),
        "warmup_frames": int(warmup_frames),
        "latency_rounds": int(latency_rounds),
        "fixed_k": None,
        "input_contract": POST_SCATTER_CONTRACT,
        "frontend_checkpoint": str(checkpoint_path or checkpoint),
        "eval_manifest_path": str(eval_manifest_path) if eval_manifest_path else "",
        "evaluation_protocol_version": (
            "heal-post-scatter-dynamic-gpu-voxel-pfn-scatter-external-v1"
        ),
        "ap_iou_backend": str(ap_iou_backend),
        "require_cuda_postprocess": bool(require_cuda_postprocess),
        "torch_num_threads": max(1, int(torch_num_threads)),
        "dataloader_num_workers": max(0, int(dataloader_num_workers)),
        "voxelization_backend": str(voxelization_backend),
        "evaluation_seed": int(evaluation_seed),
    }
    request_path.write_text(json.dumps(request, indent=2, sort_keys=True), encoding="utf-8")
    root = Path(tensorrt_root)
    env = modelopt_subprocess_env(
        tensorrt_root=root,
        conda_env=conda_env,
        pythonpath_entries=[
            "../HEAL",
            "../../HEAL",
            "..",
            ".",
            "../..",
        ],
        cuda_visible_devices=cuda_visible_devices,
    )
    env["LD_LIBRARY_PATH"] = ":".join(
        [
            str(root / "targets" / "x86_64-linux-gnu" / "lib"),
            str(root / "lib"),
            env.get("LD_LIBRARY_PATH", ""),
        ]
    )
    thread_limit = str(max(1, int(torch_num_threads)))
    env.update(
        {
            "OMP_NUM_THREADS": thread_limit,
            "MKL_NUM_THREADS": thread_limit,
            "OPENBLAS_NUM_THREADS": thread_limit,
            "NUMEXPR_NUM_THREADS": thread_limit,
        }
    )
    cmd = modelopt_python_command(conda_env) + [
        "-m",
        "search.integration.evaluation_worker",
        "--request",
        str(request_path),
    ]
    log_path = destination / "evaluation_worker.log"
    completed = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, check=False)
    log_path.write_text(completed.stdout or "", encoding="utf-8")
    if not output_path.is_file():
        return {"status": "evaluation_failed", "failure_reason": f"worker_no_output_rc_{completed.returncode}", "log_path": str(log_path)}
    result = json.loads(output_path.read_text(encoding="utf-8"))
    result["worker_returncode"] = completed.returncode
    result["log_path"] = str(log_path)
    return result
