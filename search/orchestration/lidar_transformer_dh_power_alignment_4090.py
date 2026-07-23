"""RTX 4090 runtime adapter for the Transformer head-alignment sweep.

This module is intentionally additive.  It validates and configures the local
SM89 toolchain, then emits candidate and queue manifests consumed by the
existing Transformer physical build/evaluation entry points.  It never edits
the H800 implementation or the formal unified-search branch.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

from search.model_families.transformer.dh_power_alignment_4090 import (
    joint_candidates,
    power_alignment_widths,
)


PLATFORM_ID = "RTX4090_SM89"
FORMAL_SEARCH_BRANCH = "feature/heal-unified-search-h800"
DEFAULT_GPU_IDS = (4, 5, 6, 7)

_FAMILIES: dict[str, tuple[tuple[str, int, int], ...]] = {
    "lidar_cobevt": (
        ("cobevt_grid_h8_d32", 8, 32),
        ("cobevt_window_h8_d32", 8, 32),
    ),
    "lidar_v2xvit": (
        ("v2xvit_agent_relation_h8_d32", 8, 32),
        ("v2xvit_spatial_window_w16_h4_d64", 4, 64),
        ("v2xvit_spatial_window_w4_h16_d16", 16, 16),
        ("v2xvit_spatial_window_w8_h8_d32", 8, 32),
    ),
}


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class Runtime4090Paths:
    modelopt_prefix: Path
    tensorrt_root: Path
    plugin_path: Path
    nvcc_path: Path | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "modelopt_prefix", Path(self.modelopt_prefix))
        object.__setattr__(self, "tensorrt_root", Path(self.tensorrt_root))
        object.__setattr__(self, "plugin_path", Path(self.plugin_path))
        if self.nvcc_path is not None:
            object.__setattr__(self, "nvcc_path", Path(self.nvcc_path))

    @property
    def python_path(self) -> Path:
        return self.modelopt_prefix / "bin" / "python"

    @property
    def compiler_path(self) -> Path:
        return self.modelopt_prefix / "bin" / "g++"

    @property
    def resolved_nvcc_path(self) -> Path:
        return self.nvcc_path or self.modelopt_prefix / "bin" / "nvcc"

    @property
    def trtexec_path(self) -> Path:
        candidates = (
            self.tensorrt_root / "targets" / "x86_64-linux-gnu" / "bin" / "trtexec",
            self.tensorrt_root / "bin" / "trtexec",
        )
        return next((path for path in candidates if path.is_file()), candidates[0])

    @property
    def tensorrt_library_path(self) -> Path:
        candidates = (
            self.tensorrt_root / "targets" / "x86_64-linux-gnu" / "lib",
            self.tensorrt_root / "lib",
        )
        return next((path for path in candidates if path.is_dir()), candidates[0])


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def validate_4090_runtime(
    paths: Runtime4090Paths, *, nvcc_archs: Sequence[str]
) -> dict[str, Any]:
    required = {
        "python": paths.python_path,
        "nvcc": paths.resolved_nvcc_path,
        "g++": paths.compiler_path,
        "trtexec": paths.trtexec_path,
        "tensorrt_library_path": paths.tensorrt_library_path,
        "plugin": paths.plugin_path,
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        raise RuntimeError(f"4090_runtime_path_missing:{','.join(missing)}")
    if not _is_relative_to(paths.resolved_nvcc_path, paths.modelopt_prefix):
        raise RuntimeError(f"nvcc_outside_modelopt_prefix:{paths.resolved_nvcc_path}")
    architectures = {str(value).strip().lower() for value in nvcc_archs}
    if not architectures.intersection({"compute_89", "sm_89"}):
        raise RuntimeError(f"sm89_not_supported:{sorted(architectures)}")
    return {
        "platform": PLATFORM_ID,
        "modelopt_prefix": str(paths.modelopt_prefix.resolve()),
        "python": str(paths.python_path.resolve()),
        "nvcc": str(paths.resolved_nvcc_path.resolve()),
        "g++": str(paths.compiler_path.resolve()),
        "tensorrt_root": str(paths.tensorrt_root.resolve()),
        "trtexec": str(paths.trtexec_path.resolve()),
        "tensorrt_library_path": str(paths.tensorrt_library_path.resolve()),
        "plugin": str(paths.plugin_path.resolve()),
        "nvcc_inside_conda": True,
        "sm89_supported": True,
        "nvcc_architectures": sorted(architectures),
        "cpu_fallback": False,
    }


def configure_4090_runtime(
    paths: Runtime4090Paths, *, output_root: Path, nvcc_archs: Sequence[str]
) -> dict[str, Any]:
    manifest = validate_4090_runtime(paths, nvcc_archs=nvcc_archs)
    extension_root = Path(output_root).resolve() / "environment" / "torch_extensions_modelopt_sm89"
    extension_root.mkdir(parents=True, exist_ok=True)
    prefix = paths.modelopt_prefix.resolve()
    library_path = paths.tensorrt_library_path.resolve()
    os.environ.update(
        {
            "CONDA_PREFIX": str(prefix),
            "CONDA_DEFAULT_ENV": "modelopt",
            "PATH": f"{prefix / 'bin'}:{os.environ.get('PATH', '')}",
            "CUDA_HOME": str(prefix),
            "CUDA_PATH": str(prefix),
            "CC": str(prefix / "bin" / "gcc"),
            "CXX": str(paths.compiler_path.resolve()),
            "CUDACXX": str(paths.resolved_nvcc_path.resolve()),
            "CMAKE_CUDA_COMPILER": str(paths.resolved_nvcc_path.resolve()),
            "TORCH_CUDA_ARCH_LIST": "8.9",
            "TORCH_EXTENSIONS_DIR": str(extension_root),
            "LD_LIBRARY_PATH": f"{library_path}:{prefix / 'lib'}:{os.environ.get('LD_LIBRARY_PATH', '')}",
        }
    )
    manifest.update(
        {
            "torch_cuda_arch_list": "8.9",
            "torch_extensions_dir": str(extension_root),
        }
    )
    return manifest


def formal_branch_guard(expected_remote_head: str, current_remote_head: str) -> bool:
    if str(expected_remote_head) != str(current_remote_head):
        raise RuntimeError(
            f"formal_search_branch_changed:expected={expected_remote_head}:current={current_remote_head}"
        )
    return True


def calibration_identity(
    model: str, structure_hash: str, calibration_manifest_hash: str, profile: str
) -> str:
    if str(profile) != "P8":
        return "not_int8"
    return _stable_hash(
        {
            "model": str(model),
            "structure_hash": str(structure_hash),
            "calibration_manifest_hash": str(calibration_manifest_hash),
            "profile": "P8_SQ1_INT8",
            "cross_structure_reuse": False,
        }
    )


def fresh_build_contract() -> dict[str, Any]:
    return {
        "timing_cache_reused": False,
        "engine_reused": False,
        "onnx_reused_across_structures": False,
        "calibration_reused_across_structures": False,
        "h800_artifact_reused": False,
        "strongly_typed": True,
        "no_tf32": True,
        "execution_platform": PLATFORM_ID,
    }


def _family_defaults(model: str) -> dict[str, int]:
    return {family: original for family, _heads, original in _FAMILIES[model]}


def single_family_candidate_manifest() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model, families in _FAMILIES.items():
        baseline = _family_defaults(model)
        signature = _stable_hash({"model": model, "targets": baseline})
        rows.append(
            {
                "candidate_id": f"{model}__B0",
                "model": model,
                "family": "baseline",
                "d_h": None,
                "structure_kind": "baseline",
                "target_d_h_by_family": baseline,
                "structure_signature": signature,
                "execution_platform": PLATFORM_ID,
            }
        )
        for family, heads, original in families:
            for candidate in power_alignment_widths(original, heads=heads):
                if candidate.d_h == original:
                    continue
                targets = dict(baseline)
                targets[family] = int(candidate.d_h)
                payload = candidate.to_dict()
                rows.append(
                    {
                        "candidate_id": f"{model}__{family}__dh_{candidate.d_h:03d}",
                        "model": model,
                        "family": family,
                        "d_h": int(candidate.d_h),
                        "structure_kind": "single_family",
                        "target_d_h_by_family": targets,
                        "structure_signature": _stable_hash({"model": model, "targets": targets}),
                        "execution_platform": PLATFORM_ID,
                        **payload,
                    }
                )
    return rows


def joint_candidate_manifest() -> list[dict[str, Any]]:
    return [
        {
            **candidate.to_dict(),
            "structure_kind": "joint",
            "structure_signature": _stable_hash(
                {"model": candidate.model, "targets": candidate.target_d_h_by_family}
            ),
            "execution_platform": PLATFORM_ID,
        }
        for model in _FAMILIES
        for candidate in joint_candidates(model)
    ]


def assign_queue_owners(
    rows: Iterable[Mapping[str, Any]], *, gpu_ids: Sequence[int] = DEFAULT_GPU_IDS
) -> list[dict[str, Any]]:
    owners = tuple(int(value) for value in gpu_ids)
    if not owners or len(set(owners)) != len(owners):
        raise ValueError("invalid_4090_gpu_owner_set")
    return [
        {**dict(row), "physical_gpu": owners[index % len(owners)], "queue_index": index}
        for index, row in enumerate(rows)
    ]


def write_candidate_manifests(
    output_root: Path, *, gpu_ids: Sequence[int] = DEFAULT_GPU_IDS
) -> dict[str, Any]:
    destination = Path(output_root).resolve() / "candidate_manifests"
    single = single_family_candidate_manifest()
    joint = joint_candidate_manifest()
    queue_rows = assign_queue_owners([*single, *joint], gpu_ids=gpu_ids)
    _write_json(destination / "single_family_candidates.json", single)
    _write_json(destination / "joint_candidate_selection.json", joint)
    _write_json(destination / "gpu_queue_manifest.json", queue_rows)
    return {
        "single_family_candidates": len(single),
        "single_family_unique_structures": len({row["structure_signature"] for row in single}),
        "joint_candidates": len(joint),
        "joint_unique_structures": len({row["structure_signature"] for row in joint}),
        "queue_rows": len(queue_rows),
        "gpu_ids": list(map(int, gpu_ids)),
    }


def write_4090_provenance(
    output_root: Path,
    *,
    paths: Runtime4090Paths,
    nvcc_archs: Sequence[str],
    expected_formal_head: str,
    current_formal_head: str,
) -> dict[str, Any]:
    formal_branch_guard(expected_formal_head, current_formal_head)
    root = Path(output_root).resolve()
    runtime = configure_4090_runtime(paths, output_root=root, nvcc_archs=nvcc_archs)
    repo = Path(__file__).resolve().parents[2]
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=repo, text=True
    ).strip()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    result = {
        "schema_version": "transformer-dh-power-alignment-4090-v1",
        "branch": branch,
        "head": head,
        "execution_platform": PLATFORM_ID,
        "runtime": runtime,
        "formal_search_branch": FORMAL_SEARCH_BRANCH,
        "formal_search_branch_expected_head": expected_formal_head,
        "formal_search_branch_current_head": current_formal_head,
        "formal_search_branch_unchanged": True,
        "formal_search_branch_modified": False,
        "ga_executed": False,
        "greedy_executed": False,
        "full1789_executed": False,
        "build_contract": fresh_build_contract(),
    }
    _write_json(root / "provenance" / "4090_execution_manifest.json", result)
    return result


def _nvcc_architectures(nvcc: Path) -> tuple[str, ...]:
    completed = subprocess.run(
        [str(nvcc), "--list-gpu-arch"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=30,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"nvcc_arch_query_failed:{completed.stdout.strip()}")
    return tuple(value.strip() for value in completed.stdout.splitlines() if value.strip())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--modelopt-prefix", default="/home/lixingfeng/anaconda3/envs/modelopt")
    parser.add_argument("--tensorrt-root", required=True)
    parser.add_argument("--plugin", required=True)
    parser.add_argument("--expected-formal-head", required=True)
    parser.add_argument("--current-formal-head", required=True)
    parser.add_argument("--gpus", default="4,5,6,7")
    args = parser.parse_args(argv)
    paths = Runtime4090Paths(
        Path(args.modelopt_prefix), Path(args.tensorrt_root), Path(args.plugin)
    )
    nvcc_archs = _nvcc_architectures(paths.resolved_nvcc_path)
    provenance = write_4090_provenance(
        Path(args.output_root),
        paths=paths,
        nvcc_archs=nvcc_archs,
        expected_formal_head=args.expected_formal_head,
        current_formal_head=args.current_formal_head,
    )
    candidates = write_candidate_manifests(
        Path(args.output_root),
        gpu_ids=tuple(int(value) for value in args.gpus.split(",") if value),
    )
    print(json.dumps({"provenance": provenance, "candidates": candidates}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "Runtime4090Paths",
    "assign_queue_owners",
    "calibration_identity",
    "configure_4090_runtime",
    "formal_branch_guard",
    "fresh_build_contract",
    "joint_candidate_manifest",
    "single_family_candidate_manifest",
    "validate_4090_runtime",
    "write_4090_provenance",
    "write_candidate_manifests",
]
