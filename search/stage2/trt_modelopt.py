"""Host-side ModelOpt TensorRT build invocation."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from ..integration.runtime_environment import modelopt_python_command, modelopt_subprocess_env


def build_engine_modelopt(
    *,
    qdq_onnx: str | Path,
    engine_path: str | Path,
    precision_mapping: Any,
    build_config: Any,
    physical_snapshot: Any,
    output_dir: str | Path,
    tensorrt_root: str | Path,
    conda_env: str = "modelopt",
    gpu_id: int | str | None = None,
) -> dict[str, Any]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    request_path = destination / "trt_build_request.json"
    output_path = destination / "trt_build_result.json"
    layer_info_path = destination / "engine_layer_info.json"
    log_path = destination / "engine_build.log"
    root = Path(tensorrt_root)
    ld = ":".join(
        str(path)
        for path in (
            root / "targets" / "x86_64-linux-gnu" / "lib",
            root / "lib",
        )
        if path.exists()
    )
    request = {
        "repo_root": str(Path(__file__).resolve().parents[2]),
        "qdq_onnx": str(qdq_onnx),
        "engine_path": str(engine_path),
        "precision_mapping": precision_mapping.to_dict() if hasattr(precision_mapping, "to_dict") else precision_mapping,
        "build_config": build_config.to_dict() if hasattr(build_config, "to_dict") else build_config,
        "physical_snapshot": physical_snapshot.to_dict() if hasattr(physical_snapshot, "to_dict") else physical_snapshot,
        "layer_info_path": str(layer_info_path),
        "log_path": str(log_path),
        "output_path": str(output_path),
        "ld_library_path": ld,
    }
    request_path.write_text(json.dumps(request, indent=2, sort_keys=True, default=str), encoding="utf-8")
    env = modelopt_subprocess_env(
        tensorrt_root=root,
        conda_env=conda_env,
        pythonpath_entries=[
            "/home/lixingfeng/UniAD_examine/HEAL",
            "/home/lixingfeng/UniAD_examine/heal_compress",
            "/home/lixingfeng/UniAD_examine",
        ],
        cuda_visible_devices=gpu_id,
    )
    if ld:
        env["LD_LIBRARY_PATH"] = ld + ":" + env.get("LD_LIBRARY_PATH", "")
    cmd = modelopt_python_command(conda_env) + [
        "-m",
        "search.stage2.trt_build_worker",
        "--request",
        str(request_path),
    ]
    completed = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, check=False)
    (destination / "trt_build_worker.log").write_text(completed.stdout or "", encoding="utf-8")
    if not output_path.is_file():
        return {
            "status": "engine_build_failed",
            "failure_reason": f"trt_build_worker_no_output_rc_{completed.returncode}",
            "worker_log_path": str(destination / "trt_build_worker.log"),
        }
    result = json.loads(output_path.read_text(encoding="utf-8"))
    result["worker_returncode"] = completed.returncode
    result["worker_log_path"] = str(destination / "trt_build_worker.log")
    result["worker_invocation"] = {
        "mode": "explicit_conda_activate_modelopt_python",
        "conda_env": str(conda_env),
        "cuda_visible_devices": str(gpu_id) if gpu_id is not None else "",
    }
    return result
