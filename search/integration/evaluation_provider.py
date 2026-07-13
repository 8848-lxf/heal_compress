"""Subprocess provider for TensorRT AP/latency evaluation."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from .runtime_environment import modelopt_python_command, modelopt_subprocess_env


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
    output_dir: str | Path,
    tensorrt_root: str | Path,
    plugin_path: str | Path | None,
    num_frames: int,
    warmup_frames: int,
    fixed_k: int = 29696,
    latency_rounds: int = 1,
    conda_env: str = "modelopt",
    eval_manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    request_path = destination / "evaluation_request.json"
    output_path = destination / "evaluation.json"
    cuda_visible_devices, worker_device = _cuda_visible_and_logical_device(str(device))
    request = {
        "engine_path": str(engine_path),
        "checkpoint": str(checkpoint),
        "model_config": str(model_config),
        "heal_root": str(heal_root),
        "device": worker_device,
        "physical_device": str(device),
        "output_path": str(output_path),
        "plugin_path": str(plugin_path) if plugin_path else "",
        "num_frames": int(num_frames),
        "warmup_frames": int(warmup_frames),
        "latency_rounds": int(latency_rounds),
        "fixed_k": int(fixed_k),
        "eval_manifest_path": str(eval_manifest_path) if eval_manifest_path else "",
    }
    request_path.write_text(json.dumps(request, indent=2, sort_keys=True), encoding="utf-8")
    root = Path(tensorrt_root)
    env = modelopt_subprocess_env(
        tensorrt_root=root,
        conda_env=conda_env,
        pythonpath_entries=[
            "/home/lixingfeng/UniAD_examine/HEAL",
            "/home/lixingfeng/UniAD_examine/heal_compress",
            "/home/lixingfeng/UniAD_examine",
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
