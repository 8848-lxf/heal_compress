"""Host-side V2X-ViT TensorRT evaluation launcher."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Any

from deploy.post_scatter import POST_SCATTER_CONTRACT

from search.integration.runtime_environment import modelopt_python_command, modelopt_subprocess_env


def evaluate_v2xvit_engine_modelopt(
    *,
    engine_path: str | Path,
    model_config: str | Path,
    heal_root: str | Path,
    output_dir: str | Path,
    tensorrt_root: str | Path,
    plugin_path: str | Path | None,
    eval_manifest_path: str | Path,
    physical_gpu_id: int,
    fixed_k: int | None = None,
    max_agents: int = 2,
    num_frames: int = 20,
    warmup_frames: int = 5,
    latency_rounds: int = 1,
    dataloader_num_workers: int = 8,
    input_contract: str = POST_SCATTER_CONTRACT,
    checkpoint_path: str | Path | None = None,
    voxelization_backend: str = "gpu",
    evaluation_seed: int = 0,
) -> dict[str, Any]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parents[2]
    # This branch intentionally lives in a worktree whose directory is not
    # named ``heal_compress``.  Some legacy evaluation dependencies import the
    # package by that canonical name, so provide an output-local alias instead
    # of falling back to the formal search worktree.  The alias is contained in
    # the independent evaluation directory and is recorded in the request.
    python_package_root = destination / "python_package"
    python_package_root.mkdir(parents=True, exist_ok=True)
    package_alias = python_package_root / "heal_compress"
    if package_alias.is_symlink():
        if package_alias.resolve() != repo_root:
            raise RuntimeError(
                f"evaluation_package_alias_mismatch:{package_alias.resolve()}!={repo_root}"
            )
    elif package_alias.exists():
        raise RuntimeError(f"evaluation_package_alias_not_symlink:{package_alias}")
    else:
        package_alias.symlink_to(repo_root, target_is_directory=True)
    request_path = destination / "evaluation_request.json"
    output_path = destination / "evaluation.json"
    if str(input_contract) == POST_SCATTER_CONTRACT and checkpoint_path is None:
        raise RuntimeError("post_scatter_evaluation_requires_frontend_checkpoint")
    request = {
        "repo_root": str(repo_root),
        "python_package_root": str(python_package_root),
        "engine_path": str(Path(engine_path).resolve()),
        "model_config": str(Path(model_config).resolve()),
        "heal_root": str(Path(heal_root).resolve()),
        "device": "cuda:0",
        "physical_gpu_id": int(physical_gpu_id),
        "output_path": str(output_path.resolve()),
        "plugin_path": (
            str(Path(plugin_path).resolve()) if plugin_path is not None else ""
        ),
        "checkpoint_path": (
            str(Path(checkpoint_path).resolve())
            if checkpoint_path is not None
            else ""
        ),
        "num_frames": int(num_frames),
        "warmup_frames": int(warmup_frames),
        "latency_rounds": int(latency_rounds),
        "fixed_k": int(fixed_k) if fixed_k is not None else None,
        "max_agents": int(max_agents),
        "input_contract": str(input_contract),
        "voxelization_backend": str(voxelization_backend),
        "evaluation_seed": int(evaluation_seed),
        "eval_manifest_path": str(Path(eval_manifest_path).resolve()),
        "dataloader_num_workers": int(dataloader_num_workers),
        "torch_num_threads": 4,
        "evaluation_protocol_version": (
            "heal-post-scatter-dynamic-gpu-voxel-pfn-scatter-external-v1"
            if str(input_contract) == POST_SCATTER_CONTRACT
            else "heal-v2xvit-fixed-manifest-deterministic-gpu-voxel-postprocess-workers8-v3"
        ),
    }
    request_path.write_text(json.dumps(request, indent=2, sort_keys=True), encoding="utf-8")
    env = modelopt_subprocess_env(
        tensorrt_root=tensorrt_root,
        conda_env="modelopt",
        pythonpath_entries=[
            str(python_package_root),
            str(repo_root),
            str(repo_root.parent),
            "../../HEAL",
        ],
        cuda_visible_devices=int(physical_gpu_id),
    )
    env.update(
        {
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "OPENBLAS_NUM_THREADS": "4",
            "NUMEXPR_NUM_THREADS": "4",
        }
    )
    command = modelopt_python_command("modelopt") + [
        "-m",
        "search.model_family.evaluation_worker",
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
            "failure_reason": f"v2xvit_worker_no_output_rc_{completed.returncode}",
            "log_path": str(log_path),
        }
    result = json.loads(output_path.read_text(encoding="utf-8"))
    result["worker_returncode"] = int(completed.returncode)
    result["log_path"] = str(log_path)
    return result


__all__ = ["evaluate_v2xvit_engine_modelopt"]
