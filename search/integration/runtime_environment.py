"""Runtime environment and GPU/TensorRT helpers."""

from __future__ import annotations

import hashlib
import json
import os
import pwd
import shlex
import sys
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GPUSelection:
    gpu_id_arg: str
    physical_gpu_id: int
    runtime_device: str
    excluded_gpu_ids: list[int]
    gpu_report: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "gpu_id_arg": self.gpu_id_arg,
            "physical_gpu_id": self.physical_gpu_id,
            "runtime_device": self.runtime_device,
            "excluded_gpu_ids": list(self.excluded_gpu_ids),
            "gpu_report": list(self.gpu_report),
        }


@dataclass(frozen=True)
class TensorRTEnvironment:
    tensorrt_root: Path
    trtexec_path: Path
    plugin_path: Path | None = None
    conda_env: str = "modelopt"
    env: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tensorrt_root": str(self.tensorrt_root),
            "trtexec_path": str(self.trtexec_path),
            "plugin_path": str(self.plugin_path) if self.plugin_path else "",
            "conda_env": self.conda_env,
            "env_hash": hashlib.sha256(json.dumps(self.env, sort_keys=True).encode()).hexdigest(),
        }


def query_gpus() -> list[dict[str, Any]]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,driver_version,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=15,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"nvidia-smi_failed:{completed.stderr.strip()}")
    rows = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 10:
            continue
        rows.append(
            {
                "index": int(parts[0]),
                "uuid": parts[1],
                "name": parts[2],
                "driver_version": parts[3],
                "memory_total_mib": int(parts[4]),
                "memory_used_mib": int(parts[5]),
                "memory_free_mib": int(parts[6]),
                "utilization_gpu_pct": int(parts[7]),
                "temperature_c": int(parts[8]),
                "power_draw_w": float(parts[9]) if parts[9] not in {"N/A", "[N/A]"} else None,
                "processes": [],
            }
        )
    by_uuid = {str(row["uuid"]): row for row in rows}
    processes = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=15,
    )
    if processes.returncode == 0:
        for line in processes.stdout.splitlines():
            parts = [part.strip() for part in line.split(",", 3)]
            if len(parts) != 4 or parts[0] not in by_uuid:
                continue
            pid = int(parts[1])
            process = {
                "pid": pid,
                "process_name": parts[2],
                "used_memory_mib": int(parts[3]),
            }
            proc_path = Path("/proc") / str(pid)
            try:
                process["user"] = pwd.getpwuid(proc_path.stat().st_uid).pw_name
                command = (proc_path / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                    "utf-8", errors="replace"
                ).strip()
                process["command"] = command
            except (FileNotFoundError, PermissionError, KeyError):
                process["user"] = ""
                process["command"] = ""
            elapsed = subprocess.run(
                ["ps", "-o", "etimes=", "-p", str(pid)],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=5,
            )
            process["elapsed_seconds"] = int(elapsed.stdout.strip() or 0)
            by_uuid[parts[0]]["processes"].append(process)
    return rows


def audit_gpu_isolation(
    gpu_report: list[dict[str, Any]],
    *,
    gpu_index: int,
    allowed_pids: set[int] | None = None,
    unexplained_utilization_limit_pct: int = 20,
    allow_foreign_processes: bool = False,
    max_gpu_utilization_pct: int = 20,
) -> dict[str, Any]:
    selected = next(
        (dict(row) for row in gpu_report if int(row.get("index", -1)) == int(gpu_index)),
        None,
    )
    if selected is None:
        raise RuntimeError(f"gpu_telemetry_missing_for_index:{gpu_index}")
    allowed = {int(value) for value in (allowed_pids or set())}
    processes = [dict(row) for row in selected.get("processes", [])]
    foreign = [row for row in processes if int(row.get("pid", -1)) not in allowed]
    issues: list[str] = []
    if foreign and not allow_foreign_processes:
        issues.append("foreign_compute_processes_present")
    if (
        allow_foreign_processes
        and int(selected.get("utilization_gpu_pct", 0)) > int(max_gpu_utilization_pct)
    ):
        issues.append("gpu_utilization_above_shared_limit")
    if (
        not processes
        and int(selected.get("utilization_gpu_pct", 0))
        > int(unexplained_utilization_limit_pct)
    ):
        issues.append("unexplained_gpu_utilization")
    return {
        "passed": not issues,
        "gpu_index": int(gpu_index),
        "gpu_uuid": str(selected.get("uuid", "")),
        "allowed_pids": sorted(allowed),
        "shared_gpu_authorized": bool(allow_foreign_processes),
        "max_gpu_utilization_pct": int(max_gpu_utilization_pct),
        "foreign_compute_processes": foreign,
        "issues": issues,
        "telemetry": selected,
    }


def require_gpu_isolation(
    gpu_index: int,
    *,
    report_path: str | Path | None = None,
    allowed_pids: set[int] | None = None,
    allow_foreign_processes: bool = False,
    max_gpu_utilization_pct: int = 20,
) -> dict[str, Any]:
    allowed = {os.getpid(), *(allowed_pids or set())}
    report = audit_gpu_isolation(
        query_gpus(),
        gpu_index=int(gpu_index),
        allowed_pids=allowed,
        allow_foreign_processes=allow_foreign_processes,
        max_gpu_utilization_pct=max_gpu_utilization_pct,
    )
    if report_path is not None:
        destination = Path(report_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(report, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
    if not report["passed"]:
        raise RuntimeError(
            f"gpu_competition_detected:gpu={gpu_index}:issues={report['issues']}"
        )
    return report


def select_gpu(gpu_id: str = "auto", exclude_gpu_ids: list[int] | None = None) -> GPUSelection:
    excluded = [int(value) for value in (exclude_gpu_ids or [5, 6, 7])]
    report = query_gpus()
    if str(gpu_id) != "auto":
        selected = int(gpu_id)
        return GPUSelection(str(gpu_id), selected, f"cuda:{selected}", excluded, report)
    candidates = [row for row in report if int(row["index"]) not in set(excluded)]
    if not candidates:
        raise RuntimeError(f"no_usable_gpu_after_exclusion:{excluded}")
    candidates.sort(key=lambda row: (-int(row["memory_free_mib"]), int(row["utilization_gpu_pct"]), int(row["memory_used_mib"]), int(row["index"])))
    selected = int(candidates[0]["index"])
    return GPUSelection(str(gpu_id), selected, f"cuda:{selected}", excluded, report)


def discover_trt_environment(
    tensorrt_root: str | Path,
    *,
    plugin_path: str | Path | None = None,
    conda_env: str = "modelopt",
) -> TensorRTEnvironment:
    root = Path(tensorrt_root).expanduser().resolve()
    trtexec = root / "targets" / "x86_64-linux-gnu" / "bin" / "trtexec"
    if not trtexec.is_file():
        trtexec = root / "bin" / "trtexec"
    if not trtexec.is_file():
        raise RuntimeError(f"trtexec_missing:{root}")
    lib_dirs = [
        root / "targets" / "x86_64-linux-gnu" / "lib",
        root / "lib",
    ]
    bin_dirs = [
        root / "targets" / "x86_64-linux-gnu" / "bin",
        root / "bin",
    ]
    env = dict(os.environ)
    env["PATH"] = ":".join(str(path) for path in bin_dirs if path.exists()) + ":" + env.get("PATH", "")
    env["LD_LIBRARY_PATH"] = ":".join(str(path) for path in lib_dirs if path.exists()) + ":" + env.get("LD_LIBRARY_PATH", "")
    plugin = Path(plugin_path).expanduser().resolve() if plugin_path else None
    if plugin is not None and not plugin.is_file():
        raise RuntimeError(f"plugin_missing:{plugin}")
    return TensorRTEnvironment(root, trtexec, plugin, conda_env=conda_env, env=env)


def resolve_conda_env_prefix(conda_env: str = "modelopt") -> Path:
    """Resolve a conda environment prefix without activating the caller shell."""

    env_name = str(conda_env)
    active_prefix = os.environ.get("CONDA_PREFIX")
    if active_prefix and Path(active_prefix).name == env_name:
        return Path(active_prefix).resolve()

    candidates: list[Path] = []
    executable = Path(sys.executable).resolve()
    if len(executable.parents) >= 3 and executable.parents[1].name == "envs":
        candidates.append(executable.parents[1] / env_name)
    home = Path.home()
    candidates.extend(
        [
            home / "anaconda3" / "envs" / env_name,
            home / "miniconda3" / "envs" / env_name,
            Path("/home/lixingfeng/anaconda3/envs") / env_name,
        ]
    )
    for prefix in candidates:
        if (prefix / "bin" / "python").is_file():
            return prefix.resolve()

    completed = subprocess.run(
        ["conda", "env", "list", "--json"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )
    if completed.returncode == 0:
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError:
            payload = {}
        for item in payload.get("envs", []):
            prefix = Path(str(item))
            if prefix.name == env_name and (prefix / "bin" / "python").is_file():
                return prefix.resolve()
    raise RuntimeError(f"conda_env_prefix_missing:{env_name}")


def modelopt_python_command(conda_env: str = "modelopt") -> list[str]:
    """Return an explicitly activated modelopt Python command.

    ``conda run`` is not used because compiler activation hooks can resolve
    cross-compiler wrappers against the base prefix on migrated hosts.  The
    login shell activates the requested environment and then pins every
    compiler/CUDA entry to that environment before executing Python.
    """

    prefix = resolve_conda_env_prefix(conda_env)
    conda_sh = prefix.parents[1] / "etc" / "profile.d" / "conda.sh"
    if not conda_sh.is_file():
        raise RuntimeError(f"conda_activation_script_missing:{conda_sh}")
    script = (
        "requested_path=\"${PATH:-}\"; "
        "requested_ld_library_path=\"${LD_LIBRARY_PATH:-}\"; "
        f"source {shlex.quote(str(conda_sh))}; "
        "conda activate \"$1\"; "
        "clean_path=\"$CONDA_PREFIX/bin\"; "
        "IFS=':' read -r -a requested_path_entries <<< \"$requested_path\"; "
        "for entry in \"${requested_path_entries[@]}\"; do "
        "case \"$entry\" in ''|/usr/local/cuda*|*/envs/univ2x-opt/bin) continue ;; esac; "
        "case \":$clean_path:\" in *\":$entry:\"*) ;; *) clean_path=\"$clean_path:$entry\" ;; esac; "
        "done; "
        "export PATH=\"$clean_path\"; "
        "export CUDA_HOME=\"$CONDA_PREFIX\"; "
        "export CC=\"$CONDA_PREFIX/bin/gcc\"; "
        "export CXX=\"$CONDA_PREFIX/bin/g++\"; "
        "export CUDACXX=\"$CONDA_PREFIX/bin/nvcc\"; "
        "clean_ld_library_path=\"$CONDA_PREFIX/lib\"; "
        "IFS=':' read -r -a requested_ld_entries <<< \"$requested_ld_library_path\"; "
        "for entry in \"${requested_ld_entries[@]}\"; do "
        "case \"$entry\" in ''|/usr/local/cuda*|*/envs/univ2x-opt/*) continue ;; esac; "
        "case \":$clean_ld_library_path:\" in *\":$entry:\"*) ;; *) clean_ld_library_path=\"$clean_ld_library_path:$entry\" ;; esac; "
        "done; "
        "export LD_LIBRARY_PATH=\"$clean_ld_library_path\"; "
        "export CMAKE_PREFIX_PATH=\"$CONDA_PREFIX${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}\"; "
        "shift; exec python \"$@\""
    )
    return ["bash", "-lc", script, "modelopt-python", str(conda_env)]


def modelopt_subprocess_env(
    *,
    tensorrt_root: str | Path,
    conda_env: str = "modelopt",
    pythonpath_entries: list[str | Path] | None = None,
    cuda_visible_devices: str | int | None = None,
) -> dict[str, str]:
    """Build an isolated environment for modelopt/TensorRT subprocesses."""

    prefix = resolve_conda_env_prefix(conda_env)
    root = Path(tensorrt_root).expanduser().resolve()
    lib_dirs = [
        root / "targets" / "x86_64-linux-gnu" / "lib",
        root / "lib",
        prefix / "lib",
    ]
    bin_dirs = [
        prefix / "bin",
        root / "targets" / "x86_64-linux-gnu" / "bin",
        root / "bin",
    ]
    env = dict(os.environ)
    env["CONDA_PREFIX"] = str(prefix)
    env["CONDA_DEFAULT_ENV"] = str(conda_env)
    env["PATH"] = ":".join(str(path) for path in bin_dirs if path.exists()) + ":" + env.get("PATH", "")
    env["LD_LIBRARY_PATH"] = ":".join(str(path) for path in lib_dirs if path.exists()) + ":" + env.get("LD_LIBRARY_PATH", "")
    env["CUDA_HOME"] = str(prefix)
    env["CC"] = str(prefix / "bin" / "gcc")
    env["CXX"] = str(prefix / "bin" / "g++")
    env["CUDACXX"] = str(prefix / "bin" / "nvcc")
    env["CMAKE_PREFIX_PATH"] = ":".join(
        value for value in (str(prefix), env.get("CMAKE_PREFIX_PATH", "")) if value
    )
    entries = [str(Path(item)) for item in (pythonpath_entries or [])]
    if entries:
        env["PYTHONPATH"] = ":".join(entries + [env.get("PYTHONPATH", "")])
    if cuda_visible_devices is not None and str(cuda_visible_devices) != "":
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
    return env


def plugin_hashes(paths: list[Path]) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in paths:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        result[str(path)] = digest.hexdigest()
    return result
