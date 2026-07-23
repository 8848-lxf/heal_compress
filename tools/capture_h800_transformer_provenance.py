#!/usr/bin/env python3
"""Capture Git and fail-closed modelopt/TensorRT provenance for H800 work."""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def _run(
    command: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout: int = 30,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        env=env,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
    )
    if check and completed.returncode != 0:
        raise RuntimeError(
            f"command_failed:{command!r}:returncode={completed.returncode}:"
            f"stderr={completed.stderr.strip()}"
        )
    return completed


def _git(cwd: Path, *arguments: str, check: bool = True) -> str:
    return _run(["git", *arguments], cwd=cwd, check=check).stdout.strip()


def _timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def capture_git(args: argparse.Namespace) -> int:
    worktree = args.worktree.resolve()
    primary = args.primary_repo.resolve()
    current_branch = _git(worktree, "branch", "--show-current")
    head = _git(worktree, "rev-parse", "HEAD")
    remote_ref = f"origin/{current_branch}"
    remote_exists = bool(
        _git(worktree, "show-ref", "--verify", "--quiet", f"refs/remotes/{remote_ref}", check=False)
        or _run(
            ["git", "show-ref", "--verify", "--quiet", f"refs/remotes/{remote_ref}"],
            cwd=worktree,
            check=False,
        ).returncode
        == 0
    )
    ahead = behind = None
    if remote_exists:
        counts = _git(worktree, "rev-list", "--left-right", "--count", f"HEAD...{remote_ref}").split()
        if len(counts) == 2:
            ahead, behind = (int(counts[0]), int(counts[1]))
    common_dir = Path(_git(worktree, "rev-parse", "--git-common-dir")).resolve()
    payload = {
        "schema_version": "h800-transformer-git-start-manifest-v1",
        "timestamp": _timestamp(),
        "repo_root": str(primary),
        "repository_common_dir": str(common_dir),
        "worktree_root": str(worktree),
        "branch": current_branch,
        "base_branch": args.base_branch,
        "base_full_commit": _git(worktree, "rev-parse", f"origin/{args.base_branch}"),
        "head": head,
        "formal_search_branch": args.base_branch,
        "formal_search_branch_commit": _git(worktree, "rev-parse", f"origin/{args.base_branch}"),
        "transformer_alignment_experiment_branch": args.alignment_branch,
        "transformer_alignment_experiment_branch_commit": _git(
            worktree, "rev-parse", f"origin/{args.alignment_branch}"
        ),
        "remote_url": _git(worktree, "config", "--get", "remote.origin.url"),
        "remote_tracking_ref": remote_ref if remote_exists else None,
        "ahead": ahead,
        "behind": behind,
        "remote_sync": bool(remote_exists and ahead == 0 and behind == 0),
        "dirty_status": _git(worktree, "status", "--porcelain=v1").splitlines(),
        "primary_worktree": {
            "root": str(primary),
            "head": _git(primary, "rev-parse", "HEAD"),
            "branch": _git(primary, "branch", "--show-current"),
            "dirty_status": _git(primary, "status", "--porcelain=v1").splitlines(),
        },
        "worktree_list_porcelain": _git(worktree, "worktree", "list", "--porcelain").splitlines(),
    }
    _write_json(args.output, payload)
    return 0


def _package_version(names: tuple[str, ...]) -> dict[str, str | None]:
    for name in names:
        try:
            return {"distribution": name, "version": importlib.metadata.version(name)}
        except importlib.metadata.PackageNotFoundError:
            continue
    return {"distribution": None, "version": None}


def _tool_record(name: str, prefix: Path, version_args: list[str]) -> dict[str, Any]:
    resolved = shutil.which(name)
    if resolved is None:
        raise RuntimeError(f"required_tool_missing:{name}")
    path = Path(resolved).resolve()
    try:
        path.relative_to(prefix)
    except ValueError as exc:
        raise RuntimeError(f"tool_outside_modelopt_prefix:{name}:{path}") from exc
    completed = _run([str(path), *version_args], check=False)
    return {
        "path": str(path),
        "returncode": completed.returncode,
        "version_stdout": completed.stdout.strip(),
        "version_stderr": completed.stderr.strip(),
    }


def _runtime_worker(args: argparse.Namespace) -> int:
    prefix = Path(os.environ.get("CONDA_PREFIX", "")).resolve()
    expected_prefix = args.modelopt_prefix.resolve()
    if prefix != expected_prefix:
        raise RuntimeError(f"modelopt_prefix_mismatch:{prefix}!={expected_prefix}")
    executable = Path(sys.executable).resolve()
    try:
        executable.relative_to(expected_prefix)
    except ValueError as exc:
        raise RuntimeError(f"python_outside_modelopt_prefix:{executable}") from exc

    nvcc = _tool_record("nvcc", expected_prefix, ["--version"])
    gcc = _tool_record("gcc", expected_prefix, ["--version"])
    gxx = _tool_record("g++", expected_prefix, ["--version"])
    if nvcc["path"] == "/usr/bin/nvcc":
        raise RuntimeError("system_nvcc_leak:/usr/bin/nvcc")

    tensorrt_root = args.tensorrt_root.resolve()
    trtexec_candidates = (
        tensorrt_root / "targets/x86_64-linux-gnu/bin/trtexec",
        tensorrt_root / "bin/trtexec",
    )
    trtexec_path = next((path for path in trtexec_candidates if path.is_file()), None)
    if trtexec_path is None:
        raise RuntimeError(f"trtexec_missing:{tensorrt_root}")
    trtexec = _run([str(trtexec_path), "--version"], check=False, timeout=60)

    import tensorrt  # type: ignore[import-not-found]
    import torch

    gpu_rows = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
    ).stdout.splitlines()
    gpu_zero = None
    for row in gpu_rows:
        parts = [part.strip() for part in row.split(",")]
        if parts and parts[0] == str(args.physical_gpu):
            gpu_zero = {
                "index": int(parts[0]),
                "uuid": parts[1],
                "name": parts[2],
                "memory_total_mib": int(parts[3]),
                "memory_used_mib": int(parts[4]),
                "memory_free_mib": int(parts[5]),
                "utilization_gpu_pct": int(parts[6]),
            }
            break
    if gpu_zero is None:
        raise RuntimeError(f"physical_gpu_missing:{args.physical_gpu}")

    payload = {
        "schema_version": "h800-transformer-runtime-environment-v1",
        "timestamp": _timestamp(),
        "conda_environment": "modelopt",
        "conda_prefix": str(expected_prefix),
        "python": {"path": str(executable), "version": sys.version},
        "nvcc": nvcc,
        "gcc": gcc,
        "g++": gxx,
        "tensorrt": {
            "root": str(tensorrt_root),
            "python_version": str(tensorrt.__version__),
            "trtexec_path": str(trtexec_path.resolve()),
            "trtexec_returncode": trtexec.returncode,
            "trtexec_stdout": trtexec.stdout.strip(),
            "trtexec_stderr": trtexec.stderr.strip(),
        },
        "torch": {
            "version": str(torch.__version__),
            "cuda_build_version": str(torch.version.cuda),
            "cuda_available": bool(torch.cuda.is_available()),
        },
        "cuda_runtime": {
            "driver_reported_version": _run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                check=True,
            ).stdout.splitlines()[0].strip(),
            "torch_cuda_build_version": str(torch.version.cuda),
        },
        "gpu": gpu_zero,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "modelopt": _package_version(("nvidia-modelopt", "modelopt")),
        "fail_closed_checks": {
            "python_inside_modelopt": True,
            "nvcc_inside_modelopt": True,
            "gcc_inside_modelopt": True,
            "gxx_inside_modelopt": True,
            "system_nvcc_used": False,
        },
    }
    _write_json(args.output, payload)
    return 0


def capture_runtime(args: argparse.Namespace) -> int:
    if args.worker:
        return _runtime_worker(args)
    repo = args.worktree.resolve()
    module_path = repo / "search/integration/runtime_environment.py"
    spec = importlib.util.spec_from_file_location(
        "h800_transformer_runtime_environment", module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"runtime_environment_load_failed:{module_path}")
    runtime_environment = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = runtime_environment
    spec.loader.exec_module(runtime_environment)
    modelopt_python_command = runtime_environment.modelopt_python_command
    modelopt_subprocess_env = runtime_environment.modelopt_subprocess_env
    resolve_conda_env_prefix = runtime_environment.resolve_conda_env_prefix

    prefix = resolve_conda_env_prefix("modelopt")
    env = modelopt_subprocess_env(
        tensorrt_root=args.tensorrt_root,
        conda_env="modelopt",
        pythonpath_entries=[repo],
        cuda_visible_devices=args.physical_gpu,
    )
    command = modelopt_python_command("modelopt") + [
        str(Path(__file__).resolve()),
        "runtime",
        "--worker",
        "--worktree",
        str(repo),
        "--modelopt-prefix",
        str(prefix),
        "--tensorrt-root",
        str(args.tensorrt_root.resolve()),
        "--physical-gpu",
        str(args.physical_gpu),
        "--output",
        str(args.output.resolve()),
    ]
    completed = _run(command, cwd=repo, env=env, timeout=180, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"runtime_worker_failed:returncode={completed.returncode}:"
            f"stdout={completed.stdout.strip()}:stderr={completed.stderr.strip()}"
        )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    git_parser = subparsers.add_parser("git")
    git_parser.add_argument("--worktree", type=Path, required=True)
    git_parser.add_argument("--primary-repo", type=Path, required=True)
    git_parser.add_argument("--base-branch", default="feature/heal-unified-search-h800")
    git_parser.add_argument(
        "--alignment-branch", default="feature/h800-transformer-dh-alignment-sweep"
    )
    git_parser.add_argument("--output", type=Path, required=True)

    runtime_parser = subparsers.add_parser("runtime")
    runtime_parser.add_argument("--worker", action="store_true")
    runtime_parser.add_argument("--worktree", type=Path, required=True)
    runtime_parser.add_argument("--modelopt-prefix", type=Path, default=Path("/invalid"))
    runtime_parser.add_argument("--tensorrt-root", type=Path, required=True)
    runtime_parser.add_argument("--physical-gpu", type=int, default=0)
    runtime_parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "git":
        return capture_git(args)
    return capture_runtime(args)


if __name__ == "__main__":
    raise SystemExit(main())
