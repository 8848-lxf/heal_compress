"""CoBEVT head-dimension TensorRT capability orchestration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from search.model_families.lidar_cobevt.head_dim_capability import (
    HeadDimCandidate,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _version_pair(value: str) -> tuple[int, int]:
    fields = str(value).strip().split(".")
    if len(fields) < 2:
        raise RuntimeError(f"invalid_tensorrt_version:{value}")
    try:
        return int(fields[0]), int(fields[1])
    except ValueError as exc:
        raise RuntimeError(f"invalid_tensorrt_version:{value}") from exc


def _trt_root_from_executable(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    parents = list(resolved.parents)
    for parent in parents:
        if parent.name == "targets":
            return parent.parent
    raise RuntimeError(f"trtexec_root_unrecognized:{resolved}")


def resolve_tensorrt_evidence(
    previous_output: str | Path,
    *,
    python_tensorrt_version: str,
    trtexec_version: str,
) -> dict[str, Any]:
    root = Path(previous_output).expanduser().resolve()
    run_manifest_path = root / "run_manifest.json"
    if not run_manifest_path.is_file():
        raise RuntimeError("tensorrt_run_manifest_missing")
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    environment = dict(run_manifest.get("environment", {}))
    manifest_root_text = str(environment.get("resolved_tensorrt_root", ""))
    if not manifest_root_text:
        raise RuntimeError("resolved_tensorrt_root_missing")
    manifest_root = Path(manifest_root_text).expanduser().resolve()
    build_reports = sorted(root.rglob("build_report.json"))
    if not build_reports:
        raise RuntimeError("successful_build_report_missing")
    executable_roots: set[Path] = set()
    executable_paths: set[Path] = set()
    for report_path in build_reports:
        try:
            payload = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        command_record = payload.get("builder_command", {})
        command = (
            command_record.get("command", ())
            if isinstance(command_record, dict)
            else command_record
        )
        if not command:
            continue
        executable = Path(str(command[0])).expanduser().resolve()
        if executable.name != "trtexec":
            continue
        executable_paths.add(executable)
        executable_roots.add(_trt_root_from_executable(executable))
    if not executable_paths:
        raise RuntimeError("trtexec_provenance_missing")
    all_roots = {*executable_roots, manifest_root}
    if len(all_roots) != 1:
        raise RuntimeError(
            "tensorrt_root_mismatch:"
            + json.dumps(sorted(str(value) for value in all_roots))
        )
    if len(executable_paths) != 1:
        raise RuntimeError("multiple_trtexec_binaries_detected")
    trtexec = next(iter(executable_paths))
    if not trtexec.is_file():
        raise RuntimeError(f"trtexec_missing:{trtexec}")
    recorded_version = str(environment.get("tensorrt_version", ""))
    versions = {
        _version_pair(recorded_version),
        _version_pair(python_tensorrt_version),
        _version_pair(trtexec_version),
    }
    if len(versions) != 1:
        raise RuntimeError(
            "tensorrt_version_mismatch:"
            + json.dumps(
                {
                    "recorded": recorded_version,
                    "python": python_tensorrt_version,
                    "trtexec": trtexec_version,
                },
                sort_keys=True,
            )
        )
    return {
        "cuda_version": str(environment.get("cuda_version", "unknown")),
        "python_tensorrt_version": str(python_tensorrt_version),
        "tensorrt_root": str(manifest_root),
        "tensorrt_version": str(python_tensorrt_version),
        "trtexec_path": str(trtexec),
        "trtexec_sha256": _sha256(trtexec),
        "trtexec_version": str(trtexec_version),
    }


def capability_build_signature(
    *,
    candidate_hash: str,
    onnx_sha256: str,
    qdq_scale_hash: str,
    trtexec_sha256: str,
    tensorrt_version: str,
    gpu_architecture: str,
) -> str:
    payload = {
        "candidate_hash": str(candidate_hash),
        "gpu_architecture": str(gpu_architecture),
        "onnx_sha256": str(onnx_sha256),
        "qdq_scale_hash": str(qdq_scale_hash),
        "schema": "cobevt-head-dim-capability-build-v1",
        "tensorrt_version": str(tensorrt_version),
        "trtexec_sha256": str(trtexec_sha256),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "ascii"
    )
    return hashlib.sha256(encoded).hexdigest()


def candidate_input_shapes(
    candidate: HeadDimCandidate,
) -> dict[str, tuple[int, ...]]:
    groups = int(candidate.window_groups)
    tokens = int(candidate.token_length)
    heads = int(candidate.num_heads)
    if candidate.graph_variant == "core_attention":
        return {
            "q": (groups, heads, tokens, int(candidate.d_qk)),
            "k": (groups, heads, tokens, int(candidate.d_qk)),
            "v": (groups, heads, tokens, int(candidate.d_v)),
        }
    return {
        "x": (groups, tokens, int(candidate.embed_dim)),
        "attention_mask": (groups, tokens, tokens),
        "relative_position_bias": (1, heads, tokens, tokens),
    }


__all__ = [
    "candidate_input_shapes",
    "capability_build_signature",
    "resolve_tensorrt_evidence",
]
