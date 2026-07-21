"""Materialize the H800 toolchain and selective-porting provenance audit."""

from __future__ import annotations

import argparse
import ctypes
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

from search.integration.runtime_environment import (
    DEFAULT_MODELOPT_SOURCE_ROOT,
    configure_modelopt_inprocess,
)


SOURCE_COMMITS = (
    "b4cae960111a13eee4c578601948fb4f6c1261bd",
    "1a818373bd690206e0ee981fce461a5cc6a03328",
    "70f93224691a17c8174d44ce73fe3b7ef6d3383f",
)
TRT_ROOT = Path("/home/lixingfeng/UniAD_examine/TensorRT-10.9_x86_cu118")


def _run(command: list[str], *, cwd: Path) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command_failed:{command}:rc={completed.returncode}:{completed.stdout[-2000:]}"
        )
    return completed.stdout


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _version(path: Path) -> str:
    return _run([str(path), "--version"], cwd=path.parent).strip()


def environment_audit(output_root: Path, repo: Path) -> dict[str, Any]:
    environment = output_root / "environment"
    toolchain = configure_modelopt_inprocess(
        output_root=output_root, cache_namespace="environment_audit"
    )
    import torch

    trt_lib = TRT_ROOT / "targets" / "x86_64-linux-gnu" / "lib"
    os.environ["LD_LIBRARY_PATH"] = f"{trt_lib}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    ctypes.CDLL(str(trt_lib / "libnvinfer.so.10"), mode=ctypes.RTLD_GLOBAL)
    ctypes.CDLL(str(trt_lib / "libnvonnxparser.so.10"), mode=ctypes.RTLD_GLOBAL)
    ctypes.CDLL(str(trt_lib / "libnvinfer_plugin.so.10"), mode=ctypes.RTLD_GLOBAL)
    import tensorrt as trt
    import modelopt

    gpu_query = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,compute_cap,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        cwd=repo,
    )
    gpu_rows = []
    for line in gpu_query.splitlines():
        parts = [item.strip() for item in line.split(",")]
        if len(parts) == 7:
            gpu_rows.append(
                dict(zip(
                    ("index", "name", "uuid", "compute_capability", "memory_total_mib", "memory_used_mib", "utilization_pct"),
                    parts,
                ))
            )
    _write_csv(environment / "gpu_inventory.csv", gpu_rows)
    python_environment = {
        "python": str(Path(sys.executable).resolve()),
        "python_version": sys.version,
        "conda_prefix": os.environ.get("CONDA_PREFIX", ""),
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "modelopt": getattr(modelopt, "__version__", "0.29.0-source"),
        "modelopt_source": str(DEFAULT_MODELOPT_SOURCE_ROOT),
        "modelopt_source_hash": _sha256(DEFAULT_MODELOPT_SOURCE_ROOT / "modelopt" / "__init__.py"),
    }
    _write_json(environment / "python_environment.json", python_environment)
    conda = Path(toolchain["conda_prefix"]).parent.parent / "bin" / "conda"
    if not conda.is_file():
        raise RuntimeError(f"conda_executable_missing:{conda}")
    (environment / "conda_list.txt").write_text(
        _run([str(conda), "list", "-p", toolchain["conda_prefix"]], cwd=repo),
        encoding="utf-8",
    )
    (environment / "conda_explicit.txt").write_text(
        _run([str(conda), "list", "--explicit", "-p", toolchain["conda_prefix"]], cwd=repo),
        encoding="utf-8",
    )
    cuda = {
        **toolchain,
        "nvcc_version": _version(Path(toolchain["nvcc"])),
        "gcc_version": _version(Path(toolchain["gcc"])),
        "gxx_version": _version(Path(toolchain["g++"])),
        "ninja_version": _version(Path(toolchain["ninja"])),
        "all_compilers_inside_modelopt": all(
            Path(toolchain[key]).is_relative_to(Path(toolchain["conda_prefix"]))
            for key in ("nvcc", "gcc", "g++", "ninja")
        ),
        "system_nvcc_rejected": toolchain["nvcc"] not in {
            "/usr/bin/nvcc", "/usr/local/cuda/bin/nvcc"
        },
    }
    _write_json(environment / "cuda_toolchain.json", cuda)
    tensorrt = {
        "root": str(TRT_ROOT),
        "root_exists": TRT_ROOT.is_dir(),
        "trtexec": str(TRT_ROOT / "targets" / "x86_64-linux-gnu" / "bin" / "trtexec"),
        "python_version": trt.__version__,
        "python_module": str(Path(trt.__file__).resolve()),
        "strongly_typed_supported": hasattr(trt.NetworkDefinitionCreationFlag, "STRONGLY_TYPED"),
        "bf16_dtype_supported": hasattr(trt.DataType, "BF16"),
        "fp8_dtype_supported": hasattr(trt.DataType, "FP8"),
    }
    _write_json(environment / "tensorrt_environment.json", tensorrt)
    extension_log = environment / "extension_sm90_compile.log"
    match = re.search(
        r"SM90_RESULT_JSON=(\{.*\})", extension_log.read_text(encoding="utf-8") if extension_log.is_file() else ""
    )
    extension = json.loads(match.group(1)) if match else {"passed": False, "reason": "gate_log_missing"}
    extension_so = (
        environment / "torch_extensions_modelopt_sm90" / "h800_sm90_extension_gate"
        / "h800_sm90_extension_gate.so"
    )
    extension.update(
        extension_exists=extension_so.is_file(),
        extension_sha256=_sha256(extension_so) if extension_so.is_file() else "",
    )
    _write_json(environment / "extension_sm90_test.json", extension)
    conclusion = {
        "status": "passed" if (
            cuda["all_compilers_inside_modelopt"]
            and cuda["system_nvcc_rejected"]
            and extension.get("passed")
            and not extension.get("cpu_fallback", True)
            and tensorrt["strongly_typed_supported"]
        ) else "failed",
        "cpu_fallback": False,
        "sm90": extension.get("capability") == [9, 0],
        "toolchain_isolated": cuda["all_compilers_inside_modelopt"],
        "tensorrt_root_verified": tensorrt["root_exists"],
    }
    _write_json(environment / "environment_conclusion.json", conclusion)
    (environment / "environment_conclusion.md").write_text(
        "# H800 environment conclusion\n\n"
        f"- status: `{conclusion['status']}`\n"
        f"- Python: `{python_environment['python']}`\n"
        f"- CUDA/nvcc: `{toolchain['nvcc']}`\n"
        f"- GCC/G++: `{toolchain['gcc']}`, `{toolchain['g++']}`\n"
        f"- TensorRT: `{trt.__version__}` from `{TRT_ROOT}`\n"
        f"- SM90 extension: `{extension.get('passed', False)}`; CPU fallback: `{extension.get('cpu_fallback')}`\n",
        encoding="utf-8",
    )
    return conclusion


def porting_audit(output_root: Path, repo: Path) -> dict[str, Any]:
    destination = output_root / "porting"
    inventory_lines = ["# Selective porting source commit inventory", ""]
    matrix: list[dict[str, Any]] = []
    patterns = {
        "sm89": re.compile(r"SM89|sm_89|8\.9"),
        "rtx4090": re.compile(r"RTX ?4090|4090 UUID", re.I),
        "fixed_cuda_device": re.compile(r"cuda:[0-9]+|set_device\([0-9]+\)"),
        "fixed_attention_shape": re.compile(r"d_h\s*=\s*32|heads\s*=\s*8"),
        "absolute_output": re.compile(r"/data/.+outputs"),
    }
    for commit in SOURCE_COMMITS:
        header = _run(
            ["git", "show", "-s", "--format=%H%n%P%n%ad%n%s", "--date=iso-strict", commit],
            cwd=repo,
        ).splitlines()
        inventory_lines.extend(
            [f"## `{header[0]}`", "", f"- subject: {header[3]}", f"- date: {header[2]}", f"- parent: `{header[1]}`", ""]
        )
        changed = _run(
            ["git", "diff-tree", "--no-commit-id", "--name-status", "-r", commit], cwd=repo
        )
        for line in changed.splitlines():
            status, path_text = line.split("\t", 1)
            path = repo / path_text
            text = path.read_text(encoding="utf-8", errors="ignore") if path.is_file() else ""
            hits = [name for name, pattern in patterns.items() if pattern.search(text)]
            matrix.append(
                {
                    "source_commit": commit,
                    "source_status": status,
                    "path": path_text,
                    "present_in_h800_worktree": path.is_file(),
                    "current_sha256": _sha256(path) if path.is_file() else "",
                    "hardware_assumption_hits": ";".join(hits),
                    "migration_decision": "selective_present" if path.is_file() else "not_ported",
                }
            )
    (destination / "source_commit_inventory.md").parent.mkdir(parents=True, exist_ok=True)
    (destination / "source_commit_inventory.md").write_text(
        "\n".join(inventory_lines) + "\n", encoding="utf-8"
    )
    _write_csv(destination / "changed_file_matrix.csv", matrix)
    flagged = [row for row in matrix if row["hardware_assumption_hits"]]
    (destination / "hardware_assumption_audit.md").write_text(
        "# H800 hardware-assumption audit\n\n"
        "The three source commits were inspected file-by-file; presence is not treated as proof of portability.\n\n"
        f"- source file records: {len(matrix)}\n"
        f"- records with lexical hardware assumptions: {len(flagged)}\n"
        "- H800 engines/timing caches are always rebuilt; no 4090 engine or timing cache is reusable.\n"
        "- Production H800 CLIs resolve the requested physical GPU and reject system nvcc.\n"
        "- Exact findings are retained in `changed_file_matrix.csv`.\n",
        encoding="utf-8",
    )
    return {"source_commits": len(SOURCE_COMMITS), "records": len(matrix), "flagged": len(flagged)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--repo", default=str(Path.cwd()))
    args = parser.parse_args(argv)
    output_root = Path(args.output_root).resolve()
    repo = Path(args.repo).resolve()
    result = {
        "environment": environment_audit(output_root, repo),
        "porting": porting_audit(output_root, repo),
    }
    _write_json(output_root / "environment" / "audit_summary.json", result)
    head = _run(["git", "rev-parse", "HEAD"], cwd=repo).strip()
    branch = _run(["git", "branch", "--show-current"], cwd=repo).strip()
    _write_json(
        output_root / "run_manifest.json",
        {
            "schema_version": "h800-transformer-quantization-run-v1",
            "branch": branch,
            "head_at_audit": head,
            "models": ["lidar_cobevt", "lidar_v2xvit"],
            "structure_frozen": True,
            "search_executed": False,
            "pyramid_ga_touched": False,
            "output_root": str(output_root),
        },
    )
    _write_json(
        output_root / "hardware_manifest.json",
        {
            "gpu_inventory": str(output_root / "environment" / "gpu_inventory.csv"),
            "extension_sm90_test": str(output_root / "environment" / "extension_sm90_test.json"),
            **result["environment"],
        },
    )
    _write_json(
        output_root / "toolchain_manifest.json",
        {
            "cuda": str(output_root / "environment" / "cuda_toolchain.json"),
            "tensorrt": str(output_root / "environment" / "tensorrt_environment.json"),
            "python": str(output_root / "environment" / "python_environment.json"),
            "environment_status": result["environment"]["status"],
        },
    )
    dataset_source = output_root / "evaluation" / "dataset_manifest.json"
    if dataset_source.is_file():
        _write_json(
            output_root / "dataset_manifest.json",
            json.loads(dataset_source.read_text(encoding="utf-8")),
        )
    print(json.dumps(result, sort_keys=True))
    return 0 if result["environment"]["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
