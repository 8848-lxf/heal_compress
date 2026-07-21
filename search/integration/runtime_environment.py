"""Runtime environment and GPU/TensorRT helpers."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import subprocess
import ctypes
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_MODELOPT_SOURCE_ROOT = Path(
    "/home/lixingfeng/UniAD_examine/Model-Optimizer-0.29.0"
)


def load_tensorrt_runtime(tensorrt_root: str | Path) -> tuple[str, ...]:
    """Load only the audited TensorRT runtime libraries into this process."""

    root = Path(tensorrt_root).expanduser().resolve()
    candidates = (
        root / "targets" / "x86_64-linux-gnu" / "lib",
        root / "lib",
    )
    library_root = next(
        (path for path in candidates if (path / "libnvinfer.so.10").is_file()),
        None,
    )
    if library_root is None:
        raise FileNotFoundError(f"tensorrt_runtime_library_root_missing:{root}")
    libraries = tuple(
        library_root / name
        for name in (
            "libnvinfer.so.10",
            "libnvinfer_plugin.so.10",
            "libnvonnxparser.so.10",
        )
    )
    for library in libraries:
        if not library.is_file():
            raise FileNotFoundError(str(library))
        ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
    return tuple(str(path) for path in libraries)


def ensure_modelopt_source_available() -> Path:
    """Resolve the audited ModelOpt 0.29 source without mutating Conda.

    The migrated ``modelopt`` environment contains a stale editable-install
    pointer, so in-process production entry points must use the same verified
    source tree that subprocess environments receive through ``PYTHONPATH``.
    Fail closed instead of importing an unrelated site package.
    """

    root = Path(
        os.environ.get("MODELOPT_SOURCE_ROOT", str(DEFAULT_MODELOPT_SOURCE_ROOT))
    ).expanduser().resolve()
    package = root / "modelopt" / "__init__.py"
    if not package.is_file():
        raise RuntimeError(f"modelopt_source_missing:{root}")
    value = str(root)
    if value not in sys.path:
        sys.path.insert(0, value)
    return root


def runtime_cuda_index_for_physical(physical_gpu_id: int) -> int:
    """Map a physical GPU id through CUDA_VISIBLE_DEVICES, fail closed."""

    physical = int(physical_gpu_id)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible:
        return physical
    tokens = [token.strip() for token in visible.split(",") if token.strip()]
    if not tokens or any(not token.isdigit() for token in tokens):
        raise RuntimeError(f"cuda_visible_devices_not_physical_indices:{visible}")
    try:
        return tokens.index(str(physical))
    except ValueError as exc:
        raise RuntimeError(
            f"physical_gpu_not_visible:{physical}:visible={visible}"
        ) from exc


def configure_modelopt_inprocess(
    *, output_root: str | Path, cache_namespace: str
) -> dict[str, str]:
    """Configure and verify the current ModelOpt Conda toolchain.

    This changes only the current child process.  It neither installs packages
    nor changes system CUDA/compiler links.  A per-experiment extension cache
    prevents concurrent SM90 builds from racing on ``modelopt_cuda_ext``.
    """

    prefix = Path(sys.executable).resolve().parent.parent
    if prefix.name != "modelopt":
        raise RuntimeError(f"modelopt_python_required:{sys.executable}")
    tools = {
        "python": Path(sys.executable).resolve(),
        "nvcc": prefix / "bin" / "nvcc",
        "gcc": prefix / "bin" / "gcc",
        "g++": prefix / "bin" / "g++",
        "ninja": prefix / "bin" / "ninja",
    }
    missing = [name for name, path in tools.items() if not path.is_file()]
    if missing:
        raise RuntimeError(f"modelopt_toolchain_missing:{missing}")
    cache = (
        Path(output_root).expanduser().resolve()
        / "environment"
        / "torch_extensions_modelopt_sm90"
        / str(cache_namespace)
    )
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.update(
        {
            "CONDA_PREFIX": str(prefix),
            "CONDA_DEFAULT_ENV": "modelopt",
            "PATH": f"{prefix / 'bin'}:{os.environ.get('PATH', '')}",
            "CUDA_HOME": str(prefix),
            "CUDA_PATH": str(prefix),
            "CC": str(tools["gcc"]),
            "CXX": str(tools["g++"]),
            "CUDACXX": str(tools["nvcc"]),
            "CMAKE_CUDA_COMPILER": str(tools["nvcc"]),
            "TORCH_CUDA_ARCH_LIST": "9.0",
            "TORCH_EXTENSIONS_DIR": str(cache),
        }
    )
    ensure_modelopt_source_available()
    return {
        "conda_prefix": str(prefix),
        **{name: str(path) for name, path in tools.items()},
        "torch_cuda_arch_list": "9.0",
        "torch_extensions_dir": str(cache),
        "cpu_fallback": "false",
    }


def require_modelopt_cuda_extension(kind: str = "int8") -> dict[str, str]:
    """Compile/load the requested ModelOpt CUDA extension without fallback."""

    from modelopt.torch.quantization.extensions import (  # type: ignore
        get_cuda_ext,
        get_cuda_ext_fp8,
    )

    loaders = {"int8": get_cuda_ext, "fp8": get_cuda_ext_fp8}
    if kind not in loaders:
        raise ValueError(f"unknown_modelopt_cuda_extension:{kind}")
    extension = loaders[kind](raise_if_failed=True)
    if extension is None:
        raise RuntimeError(f"modelopt_cuda_extension_cpu_fallback:{kind}")
    extension_file = Path(str(getattr(extension, "__file__", ""))).resolve()
    if not extension_file.is_file():
        raise RuntimeError(f"modelopt_cuda_extension_file_missing:{kind}")
    digest = hashlib.sha256(extension_file.read_bytes()).hexdigest()
    return {
        "kind": kind,
        "module": str(getattr(extension, "__name__", "")),
        "path": str(extension_file),
        "sha256": digest,
        "cpu_fallback": "false",
    }


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
            "--query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu",
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
        if len(parts) != 5:
            continue
        rows.append(
            {
                "index": int(parts[0]),
                "memory_total_mib": int(parts[1]),
                "memory_used_mib": int(parts[2]),
                "memory_free_mib": int(parts[3]),
                "utilization_gpu_pct": int(parts[4]),
            }
        )
    return rows


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

    script = (
        "requested_path=\"${PATH:-}\"; "
        "requested_ld_library_path=\"${LD_LIBRARY_PATH:-}\"; "
        "source /home/lixingfeng/miniconda3/etc/profile.d/conda.sh; "
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
    # This server's ModelOpt 0.29 environment was migrated with a stale
    # editable-install .pth entry.  Prefer the verified local source tree when
    # it exists; this keeps the package version fixed without modifying the
    # Conda environment or installing anything globally.
    modelopt_source = Path(
        os.environ.get("MODELOPT_SOURCE_ROOT", str(DEFAULT_MODELOPT_SOURCE_ROOT))
    )
    if (modelopt_source / "modelopt" / "__init__.py").is_file():
        entries.insert(0, str(modelopt_source))
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
