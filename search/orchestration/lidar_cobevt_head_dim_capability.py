"""CoBEVT head-dimension TensorRT capability orchestration."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import time
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from search.model_families.lidar_cobevt.head_dim_capability import (
    HeadDimCandidate,
    build_synthetic_candidate_matrix,
)
from search.model_families.lidar_cobevt.head_dim_synthetic import (
    SYNTHETIC_INPUT_CASES,
    applicable_requested_precision,
    export_synthetic_diagnostic_onnx,
    export_synthetic_onnx,
    requested_precision_manifest,
)
from search.integration.lidar_cobevt_head_dim_runtime import (
    evaluate_synthetic_runtime,
)
from search.reporting.cobevt_head_dim_capability import (
    audit_requested_realized_precision,
    classify_support,
    derive_head_dim_search_contract,
    inspect_attention_layers,
    write_capability_matrix,
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


def _manifest_tensorrt_root(previous_output: Path) -> Path:
    payload = json.loads(
        (previous_output / "run_manifest.json").read_text(encoding="utf-8")
    )
    value = str(payload.get("environment", {}).get("resolved_tensorrt_root", ""))
    if not value:
        raise RuntimeError("resolved_tensorrt_root_missing")
    return Path(value).expanduser().resolve()


def _probe_tensorrt_versions(
    *, previous_output: Path, modelopt_python: Path
) -> tuple[str, str]:
    trt_root = _manifest_tensorrt_root(previous_output)
    env = _tensorrt_environment(
        {"tensorrt_root": str(trt_root)}, physical_gpu=0
    )
    python_probe = subprocess.run(
        [str(modelopt_python), "-c", "import tensorrt; print(tensorrt.__version__)"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        check=False,
        timeout=60,
    )
    if python_probe.returncode != 0:
        raise RuntimeError(
            "python_tensorrt_probe_failed:" + str(python_probe.stdout).strip()
        )
    python_version = str(python_probe.stdout).strip().splitlines()[-1]
    trtexec = trt_root / "targets/x86_64-linux-gnu/bin/trtexec"
    help_probe = subprocess.run(
        [str(trtexec), "--help"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        check=False,
        timeout=60,
    )
    match = re.search(r"TensorRT v(\d+)", str(help_probe.stdout))
    if not match:
        raise RuntimeError("trtexec_version_probe_failed")
    encoded = int(match.group(1))
    trtexec_version = f"{encoded // 10000}.{(encoded % 10000) // 100}.{encoded % 100}"
    return python_version, trtexec_version


def _gpu_rows() -> list[dict[str, Any]]:
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,compute_cap,driver_version,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=30,
    )
    if query.returncode != 0:
        raise RuntimeError("nvidia_smi_gpu_query_failed")
    rows = []
    for line in str(query.stdout).splitlines():
        fields = [value.strip() for value in line.split(",")]
        if len(fields) != 10:
            continue
        rows.append(
            {
                "index": int(fields[0]),
                "gpu_model": fields[1],
                "gpu_uuid": fields[2],
                "compute_capability": fields[3],
                "driver_version": fields[4],
                "utilization_gpu_pct": float(fields[5]),
                "memory_used_mib": float(fields[6]),
                "memory_total_mib": float(fields[7]),
                "temperature_c": float(fields[8]),
                "power_draw_w": float(fields[9]),
            }
        )
    return rows


def audit_gpu_idle(physical_gpu: int) -> dict[str, Any]:
    gpu = next(
        (row for row in _gpu_rows() if row["index"] == int(physical_gpu)),
        None,
    )
    if gpu is None:
        raise RuntimeError(f"physical_gpu_not_found:{physical_gpu}")
    process_query = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=30,
    )
    processes = []
    if process_query.returncode == 0:
        for line in str(process_query.stdout).splitlines():
            fields = [value.strip() for value in line.split(",")]
            if len(fields) < 4 or fields[0] != gpu["gpu_uuid"]:
                continue
            pid = int(fields[1])
            try:
                command = (
                    Path(f"/proc/{pid}/cmdline")
                    .read_bytes()
                    .replace(b"\x00", b" ")
                    .decode("utf-8", errors="replace")
                    .strip()
                )
            except OSError:
                command = fields[2]
            processes.append(
                {
                    "command": command,
                    "pid": pid,
                    "process_name": fields[2],
                    "used_memory_mib": fields[3],
                    "is_pyramid": any(
                        token in command
                        for token in (
                            "4090_joint_six_budget_ga",
                            "search.stage2.candidate_worker",
                            "lidar_pyramid",
                        )
                    ),
                }
            )
    return {
        **gpu,
        "compute_processes": processes,
        "gpu_idle": not processes and float(gpu["utilization_gpu_pct"]) <= 5.0,
    }


def discover_capability_environment(
    *,
    previous_output: str | Path,
    modelopt_python: str | Path,
    physical_gpu: int,
) -> dict[str, Any]:
    previous = Path(previous_output).expanduser().resolve()
    python_version, trtexec_version = _probe_tensorrt_versions(
        previous_output=previous,
        modelopt_python=Path(modelopt_python).expanduser().resolve(),
    )
    trt = resolve_tensorrt_evidence(
        previous,
        python_tensorrt_version=python_version,
        trtexec_version=trtexec_version,
    )
    gpu = audit_gpu_idle(physical_gpu)
    trt.update(
        {
            **gpu,
            "gpu_architecture": "sm" + str(gpu["compute_capability"]).replace(".", ""),
            "hardware_id": hashlib.sha256(
                json.dumps(
                    {
                        "compute_capability": gpu["compute_capability"],
                        "cuda_version": trt["cuda_version"],
                        "driver_version": gpu["driver_version"],
                        "gpu_model": gpu["gpu_model"],
                        "tensorrt_version": trt["tensorrt_version"],
                    },
                    sort_keys=True,
                ).encode("ascii")
            ).hexdigest(),
            "physical_gpu": int(physical_gpu),
        }
    )
    return trt


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


def candidate_ids_for_shard(
    candidates: Any, *, shard_index: int, shard_count: int
) -> set[str]:
    if int(shard_count) <= 0:
        raise ValueError("candidate_shard_count_must_be_positive")
    if not 0 <= int(shard_index) < int(shard_count):
        raise ValueError("candidate_shard_index_out_of_range")
    ordered = sorted(str(row.candidate_id) for row in candidates)
    return {
        candidate_id
        for index, candidate_id in enumerate(ordered)
        if index % int(shard_count) == int(shard_index)
    }


def _shape_spec(candidate: HeadDimCandidate) -> str:
    shapes = candidate_input_shapes(candidate)
    dynamic_names = (
        ("q", "k", "v")
        if candidate.graph_variant == "core_attention"
        else ("x", "attention_mask")
    )
    return ",".join(
        f"{name}:" + "x".join(str(value) for value in shapes[name])
        for name in dynamic_names
    )


def build_trtexec_command(
    candidate: HeadDimCandidate,
    *,
    trtexec: str | Path,
    onnx_path: str | Path,
    engine_path: str | Path,
    layer_info_path: str | Path,
    profile_path: str | Path,
) -> list[str]:
    shapes = _shape_spec(candidate)
    command = [
        str(Path(trtexec).expanduser().resolve()),
        f"--onnx={Path(onnx_path).expanduser().resolve()}",
        f"--saveEngine={Path(engine_path).expanduser().resolve()}",
        "--skipInference",
        "--stronglyTyped",
        "--noTF32",
        "--noBuilderCache",
        "--profilingVerbosity=detailed",
        "--dumpLayerInfo",
        "--dumpProfile",
        f"--exportLayerInfo={Path(layer_info_path).expanduser().resolve()}",
        f"--exportProfile={Path(profile_path).expanduser().resolve()}",
        "--memPoolSize=workspace:512",
        f"--minShapes={shapes}",
        f"--optShapes={shapes}",
        f"--maxShapes={shapes}",
        "--verbose",
    ]
    forbidden = (
        "--fp16",
        "--int8",
        "precisionConstraints",
        "layerPrecisions",
        "layerOutputTypes",
    )
    if any(any(token in value for token in forbidden) for value in command):
        raise RuntimeError("weak_precision_flag_in_capability_builder")
    return command


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def prepare_capability_run(
    output_dir: str | Path,
    *,
    environment: dict[str, Any],
    code_commit: str,
) -> dict[str, Any]:
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise RuntimeError(f"capability_output_not_empty:{destination}")
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("synthetic", "real_cobevt", "failures", "logs"):
        (destination / name).mkdir(exist_ok=True)
    tensorrt_version = str(environment.get("tensorrt_version", ""))
    gpu_architecture = str(environment.get("gpu_architecture", ""))
    if not tensorrt_version or not gpu_architecture:
        raise RuntimeError("capability_environment_identity_incomplete")
    candidates = build_synthetic_candidate_matrix(
        tensorrt_version=tensorrt_version,
        gpu_architecture=gpu_architecture,
    )
    candidate_rows = [candidate.to_dict() for candidate in candidates]
    run_manifest = {
        "candidate_count": len(candidate_rows),
        "candidate_matrix_hash": hashlib.sha256(
            json.dumps(
                candidate_rows, sort_keys=True, separators=(",", ":")
            ).encode("ascii")
        ).hexdigest(),
        "code_commit": str(code_commit),
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "environment": dict(environment),
        "graph_variants": ["core_attention", "projection_attention"],
        "output_dir": str(destination),
        "real_attention_shape": {
            "activation": [1, 2, 16, 32, 4, 4, 256],
            "embed_dim": 256,
            "num_heads": 8,
            "token_length": 32,
            "window_groups": 512,
            "window_shape": [4, 4],
        },
        "schema_version": "cobevt-head-dim-trt-capability-v1",
    }
    _write_json(destination / "candidate_matrix.json", candidate_rows)
    _write_json(destination / "hardware_manifest.json", environment)
    _write_json(destination / "run_manifest.json", run_manifest)
    (destination / "resolved_config.yaml").write_text(
        yaml.safe_dump(
            {
                "candidate_count": len(candidate_rows),
                "environment": dict(environment),
                "real_attention_shape": run_manifest["real_attention_shape"],
                "trace_window_groups": 1,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return {
        "candidate_count": len(candidate_rows),
        "candidate_matrix_hash": run_manifest["candidate_matrix_hash"],
        "output_dir": str(destination),
    }


def prepare_real_cobevt_integration(
    output_dir: str | Path, source_experiment: str | Path
) -> dict[str, Any]:
    """Create a real-model inventory that reuses the validated fixedK protocol."""

    destination = Path(output_dir).expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise RuntimeError(f"real_capability_output_not_empty:{destination}")
    destination.mkdir(parents=True, exist_ok=True)
    source_root = Path(source_experiment).expanduser().resolve()
    source = json.loads(
        (source_root / "experiment_config.json").read_text(encoding="utf-8")
    )
    if int(source.get("fixed_k", 0)) != 29696:
        raise RuntimeError("real_capability_requires_fixedk29696")
    if not bool(source.get("fixed_k_validated")):
        raise RuntimeError("real_capability_fixedk_not_validated")
    if int(source.get("fixed_k_contract", {}).get("overflow_count", -1)) != 0:
        raise RuntimeError("real_capability_fixedk_overflow_detected")
    widths = (8, 12, 16, 24, 32, 48, 64)
    candidates = []
    for family in ("uniform", "qk_only", "v_only"):
        for width in widths:
            d_qk = width if family != "v_only" else 32
            d_v = width if family != "qk_only" else 32
            candidates.append(
                {
                    "candidate_id": f"{family}_qk{d_qk}_v{d_v}",
                    "d_qk": int(d_qk),
                    "d_v": int(d_v),
                    "embed_dim": 256,
                    "experiment": "capability",
                    "heads": 8,
                    "mask_path": "",
                    "mask_sha256": "not_applicable_capability_resize",
                    "variant": family,
                }
            )
    config = {
        "candidates": candidates,
        "deployment_policy": dict(source.get("deployment_policy", {})),
        "fixed_k": 29696,
        "fixed_k_contract": dict(source["fixed_k_contract"]),
        "fixed_k_validated": True,
        "manifests": dict(source["manifests"]),
        "protocol": dict(source.get("protocol", {})),
        "source_experiment": str(source_root),
        "structure_recipe": "capability_only_explicit_qk_v_resize_v1",
    }
    _write_json(destination / "experiment_config.json", config)
    return {
        "candidate_count": len(candidates),
        "fixed_k": 29696,
        "output_dir": str(destination),
    }


def audit_real_cobevt_structures(
    output_dir: str | Path,
    *,
    checkpoint: str | Path,
    config: str | Path,
    heal_root: str | Path,
) -> dict[str, int]:
    """Materialize all representative real-model shapes without using CUDA."""

    from dataclasses import asdict

    import torch

    from search.model_families.lidar_cobevt.head_dim_materializer import (
        materialize_model_attention_for_capability,
    )
    from search.model_families.lidar_cobevt.model_capability import (
        CobevtModelCapability,
    )

    destination = Path(output_dir).expanduser().resolve()
    experiment = json.loads(
        (destination / "experiment_config.json").read_text(encoding="utf-8")
    )
    report_path = destination / "structure_audit.json"
    if report_path.exists():
        raise RuntimeError("real_capability_structure_audit_not_fresh")
    rows = []
    for record in experiment["candidates"]:
        try:
            bundle = CobevtModelCapability(
                Path(checkpoint), Path(config), Path(heal_root)
            ).load(device=torch.device("cpu"))
            report = materialize_model_attention_for_capability(
                bundle.model,
                d_qk=int(record["d_qk"]),
                d_v=int(record["d_v"]),
            )
            modules = [
                {
                    "module_path": name,
                    "q_weight_shape": list(module.q_proj.weight.shape),
                    "k_weight_shape": list(module.k_proj.weight.shape),
                    "v_weight_shape": list(module.v_proj.weight.shape),
                    "out_weight_shape": list(module.out_proj.weight.shape),
                    "scale": float(module.scale),
                }
                for name, module in bundle.model.named_modules()
                if module.__class__.__name__ == "PrunableCobevtAttention"
            ]
            row = {
                **record,
                "actual_parameter_count": sum(
                    parameter.numel() for parameter in bundle.model.parameters()
                ),
                "checkpoint_strict_load": True,
                "modules": modules,
                "report": asdict(report),
                "structure_legal": bool(report.passed),
            }
        except Exception as exc:  # noqa: BLE001
            row = {
                **record,
                "failure_reason": f"{type(exc).__name__}:{exc}",
                "structure_legal": False,
            }
        rows.append(row)
    _write_json(report_path, rows)
    fieldnames = sorted(
        {key for row in rows for key in row if key not in {"modules", "report"}}
    )
    with (destination / "structure_audit.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return {
        "attempted": len(rows),
        "failed": sum(not row["structure_legal"] for row in rows),
        "succeeded": sum(bool(row["structure_legal"]) for row in rows),
    }


def _candidate_from_record(record: dict[str, Any]) -> HeadDimCandidate:
    owned = {field.name for field in fields(HeadDimCandidate)}
    return HeadDimCandidate(**{key: record[key] for key in owned})


def _candidate_directory(output_dir: Path, candidate: HeadDimCandidate) -> Path:
    return (
        output_dir
        / "synthetic"
        / candidate.graph_variant
        / candidate.structure_family
        / f"qk{candidate.d_qk}_v{candidate.d_v}"
        / candidate.precision_profile
        / candidate.candidate_hash[:16]
    )


def _qdq_scale_manifest(candidate: HeadDimCandidate) -> dict[str, Any]:
    return {
        "activation_granularity": "per_tensor_symmetric",
        "attention_probability_scale": 1.0 / 255.0,
        "default_activation_scale": 0.03125,
        "profile": candidate.precision_profile,
        "schema": "cobevt-synthetic-fixed-qdq-v1",
        "seed": 20260718,
        "weight_granularity": "per_output_channel_symmetric",
        "weight_scale_source": "deterministic_weight_absmax_div_127",
        "zero_point": 0,
    }


def export_capability_candidates(
    output_dir: str | Path,
    *,
    only_candidate_ids: set[str] | None = None,
    trace_window_groups: int = 1,
) -> dict[str, int]:
    destination = Path(output_dir).expanduser().resolve()
    matrix_path = destination / "candidate_matrix.json"
    if not matrix_path.is_file():
        raise RuntimeError("capability_candidate_matrix_missing")
    records = json.loads(matrix_path.read_text(encoding="utf-8"))
    attempted = succeeded = failed = 0
    for record in records:
        candidate = _candidate_from_record(dict(record))
        if only_candidate_ids and candidate.candidate_id not in only_candidate_ids:
            continue
        attempted += 1
        candidate_dir = _candidate_directory(destination, candidate)
        candidate_dir.mkdir(parents=True, exist_ok=True)
        _write_json(candidate_dir / "candidate.json", candidate.to_dict())
        requested = requested_precision_manifest(candidate.precision_profile)
        _write_json(candidate_dir / "requested_precision.json", requested)
        qdq_manifest = _qdq_scale_manifest(candidate)
        _write_json(candidate_dir / "qdq_scale_manifest.json", qdq_manifest)
        qdq_scale_hash = hashlib.sha256(
            json.dumps(
                qdq_manifest, sort_keys=True, separators=(",", ":")
            ).encode("ascii")
        ).hexdigest()
        try:
            production_path = candidate_dir / "attention.onnx"
            export_report_path = candidate_dir / "export_report.json"
            if production_path.is_file() and export_report_path.is_file():
                export_report = json.loads(
                    export_report_path.read_text(encoding="utf-8")
                )
                if export_report.get("candidate_hash") != candidate.candidate_hash:
                    raise RuntimeError("same_run_export_candidate_hash_mismatch")
                if _sha256(production_path) != export_report.get("onnx_sha256"):
                    raise RuntimeError("same_run_export_onnx_hash_mismatch")
                export_report["production_export_reused_same_run"] = True
                export_report[
                    "production_export_reuse_reason"
                ] = "add_diagnostic_graph_without_overwriting_production"
            else:
                export_report = export_synthetic_onnx(
                    candidate,
                    production_path,
                    trace_window_groups=int(trace_window_groups),
                )
                export_report["production_export_reused_same_run"] = False
            diagnostic_path = candidate_dir / "diagnostic_attention.onnx"
            diagnostic_report_path = candidate_dir / "diagnostic_export_report.json"
            if diagnostic_path.is_file() and diagnostic_report_path.is_file():
                diagnostic_report = json.loads(
                    diagnostic_report_path.read_text(encoding="utf-8")
                )
                if diagnostic_report.get("candidate_hash") != candidate.candidate_hash:
                    raise RuntimeError("same_run_diagnostic_candidate_hash_mismatch")
                if _sha256(diagnostic_path) != diagnostic_report.get("onnx_sha256"):
                    raise RuntimeError("same_run_diagnostic_onnx_hash_mismatch")
                diagnostic_report["diagnostic_export_reused_same_run"] = True
            else:
                diagnostic_report = export_synthetic_diagnostic_onnx(
                    candidate,
                    diagnostic_path,
                    trace_window_groups=int(trace_window_groups),
                )
                diagnostic_report["diagnostic_export_reused_same_run"] = False
            export_report["evidence_directory"] = str(candidate_dir)
            export_report["qdq_scale_hash"] = qdq_scale_hash
            export_report["trace_window_groups"] = int(trace_window_groups)
            _write_json(diagnostic_report_path, diagnostic_report)
            succeeded += 1
        except Exception as exc:  # noqa: BLE001
            export_report = {
                "candidate_hash": candidate.candidate_hash,
                "candidate_id": candidate.candidate_id,
                "evidence_directory": str(candidate_dir),
                "failure_reason": f"{type(exc).__name__}:{exc}",
                "onnx_export_success": False,
                "qdq_scale_hash": qdq_scale_hash,
                "support_class": "unsupported_export",
            }
            _write_json(
                destination / "failures" / f"{candidate.candidate_hash}.json",
                export_report,
            )
            failed += 1
        _write_json(candidate_dir / "export_report.json", export_report)
    return {"attempted": attempted, "failed": failed, "succeeded": succeeded}


def _tensorrt_environment(
    environment: dict[str, Any], physical_gpu: int
) -> dict[str, str]:
    root = Path(str(environment["tensorrt_root"])).expanduser().resolve()
    inherited = os.environ.get("LD_LIBRARY_PATH", "")
    libraries = [
        root / "targets/x86_64-linux-gnu/lib",
        root / "lib",
        Path("/home/lixingfeng/anaconda3/envs/modelopt/lib"),
    ]
    result = dict(os.environ)
    result["CUDA_VISIBLE_DEVICES"] = str(int(physical_gpu))
    result["LD_LIBRARY_PATH"] = ":".join(
        [*(str(path) for path in libraries), inherited]
    ).rstrip(":")
    return result


def build_capability_candidates(
    output_dir: str | Path,
    *,
    environment: dict[str, Any],
    physical_gpu: int,
    only_candidate_ids: set[str] | None = None,
    command_runner: Any = subprocess.run,
    diagnostic: bool = False,
) -> dict[str, int]:
    destination = Path(output_dir).expanduser().resolve()
    trtexec = Path(str(environment["trtexec_path"])).expanduser().resolve()
    if not trtexec.is_file():
        raise RuntimeError(f"capability_trtexec_missing:{trtexec}")
    attempted = succeeded = failed = 0
    for candidate_path in sorted((destination / "synthetic").rglob("candidate.json")):
        candidate_dir = candidate_path.parent
        candidate = _candidate_from_record(
            json.loads(candidate_path.read_text(encoding="utf-8"))
        )
        if only_candidate_ids and candidate.candidate_id not in only_candidate_ids:
            continue
        artifact_prefix = "diagnostic_" if diagnostic else ""
        export_path = candidate_dir / f"{artifact_prefix}export_report.json"
        if not export_path.is_file():
            continue
        export = json.loads(export_path.read_text(encoding="utf-8"))
        if not bool(export.get("onnx_export_success")):
            continue
        attempted += 1
        onnx_path = candidate_dir / f"{artifact_prefix}attention.onnx"
        engine_path = candidate_dir / f"{artifact_prefix}engine.plan"
        layer_info_path = candidate_dir / f"{artifact_prefix}engine_layer_info.json"
        profile_path = candidate_dir / f"{artifact_prefix}engine_profile.json"
        build_log_path = candidate_dir / f"{artifact_prefix}build.log"
        build_report_path = candidate_dir / f"{artifact_prefix}build_report.json"
        if engine_path.exists() or build_report_path.exists():
            raise RuntimeError(
                f"capability_candidate_build_not_fresh:{candidate.candidate_id}"
            )
        command = build_trtexec_command(
            candidate,
            trtexec=trtexec,
            onnx_path=onnx_path,
            engine_path=engine_path,
            layer_info_path=layer_info_path,
            profile_path=profile_path,
        )
        started = time.monotonic()
        try:
            completed = command_runner(
                command,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=_tensorrt_environment(environment, physical_gpu),
                check=False,
                timeout=1200,
            )
            output = str(completed.stdout or "")
            build_log_path.write_text(output, encoding="utf-8")
            build_success = bool(
                int(completed.returncode) == 0
                and engine_path.is_file()
                and layer_info_path.is_file()
            )
            inspector: dict[str, Any] = {}
            if layer_info_path.is_file():
                layer_payload = json.loads(
                    layer_info_path.read_text(encoding="utf-8")
                )
                inspector = inspect_attention_layers(
                    layer_payload.get("Layers", layer_payload)
                )
            build_report = {
                "build_elapsed_seconds": time.monotonic() - started,
                "build_returncode": int(completed.returncode),
                "builder_command": command,
                "candidate_hash": candidate.candidate_hash,
                "candidate_id": candidate.candidate_id,
                "diagnostic_only": bool(diagnostic),
                "engine_path": str(engine_path),
                "engine_sha256": _sha256(engine_path)
                if engine_path.is_file()
                else "",
                "failure_reason": ""
                if build_success
                else "trtexec_strongly_typed_build_failed",
                "layer_info_path": str(layer_info_path),
                "latency_eligible": not diagnostic,
                "onnx_sha256": str(export.get("onnx_sha256", "")),
                "physical_gpu": int(physical_gpu),
                "plugin_used": False,
                "profile_path": str(profile_path),
                "trt_build_success": build_success,
                **inspector,
            }
            build_report["build_signature"] = capability_build_signature(
                candidate_hash=candidate.candidate_hash,
                onnx_sha256=str(export.get("onnx_sha256", "")),
                qdq_scale_hash=str(export.get("qdq_scale_hash", "")),
                trtexec_sha256=str(environment.get("trtexec_sha256", "")),
                tensorrt_version=str(environment.get("tensorrt_version", "")),
                gpu_architecture=str(environment.get("gpu_architecture", "")),
            )
        except Exception as exc:  # noqa: BLE001
            build_report = {
                "build_elapsed_seconds": time.monotonic() - started,
                "builder_command": command,
                "candidate_hash": candidate.candidate_hash,
                "candidate_id": candidate.candidate_id,
                "diagnostic_only": bool(diagnostic),
                "failure_reason": f"{type(exc).__name__}:{exc}",
                "latency_eligible": not diagnostic,
                "physical_gpu": int(physical_gpu),
                "plugin_used": False,
                "trt_build_success": False,
            }
        _write_json(build_report_path, build_report)
        if bool(build_report["trt_build_success"]):
            succeeded += 1
        else:
            failed += 1
            _write_json(
                destination / "failures" / f"{candidate.candidate_hash}_build.json",
                build_report,
            )
    return {"attempted": attempted, "failed": failed, "succeeded": succeeded}


def run_capability_candidates(
    output_dir: str | Path,
    *,
    environment: dict[str, Any],
    physical_gpu: int,
    only_candidate_ids: set[str] | None = None,
    runner_factory: Any | None = None,
    warmup_iterations: int = 20,
    measured_iterations: int = 100,
) -> dict[str, int]:
    """Run production latency and diagnostic parity for built candidates."""

    import torch

    destination = Path(output_dir).expanduser().resolve()
    records = json.loads(
        (destination / "candidate_matrix.json").read_text(encoding="utf-8")
    )
    candidates = {
        candidate.candidate_id: candidate
        for candidate in (_candidate_from_record(dict(row)) for row in records)
    }
    candidate_dirs = {
        candidate.candidate_id: _candidate_directory(destination, candidate)
        for candidate in candidates.values()
    }
    if runner_factory is None:
        from search.orchestration.lidar_cobevt_attention_microbenchmark import (
            load_tensorrt_runtime,
        )
        from tests.quant_deploy.deployment_equivalence import (
            TensorRTEngineRunner,
        )

        load_tensorrt_runtime(
            Path(str(environment["tensorrt_root"])).expanduser().resolve()
        )
        device = torch.device("cuda", int(physical_gpu))
        if not torch.cuda.is_available():
            raise RuntimeError("capability_runtime_cuda_unavailable")
        torch.cuda.set_device(device)
        runner_factory = lambda path: TensorRTEngineRunner(str(path), device)

    attempted = succeeded = failed = 0
    runtime_reports: dict[str, dict[str, Any]] = {}
    cached_reference_id = ""
    cached_reference_runner: Any | None = None
    for candidate_id, candidate in candidates.items():
        if only_candidate_ids and candidate_id not in only_candidate_ids:
            continue
        candidate_dir = candidate_dirs[candidate_id]
        production_build_path = candidate_dir / "build_report.json"
        diagnostic_build_path = candidate_dir / "diagnostic_build_report.json"
        if not (production_build_path.is_file() and diagnostic_build_path.is_file()):
            continue
        production_build = json.loads(
            production_build_path.read_text(encoding="utf-8")
        )
        diagnostic_build = json.loads(
            diagnostic_build_path.read_text(encoding="utf-8")
        )
        if not (
            production_build.get("trt_build_success")
            and diagnostic_build.get("trt_build_success")
        ):
            continue
        attempted += 1
        report_path = candidate_dir / "runtime_report.json"
        if report_path.exists():
            raise RuntimeError(
                f"capability_candidate_runtime_not_fresh:{candidate_id}"
            )
        reference_id = candidate.same_shape_fp32_reference_id
        reference_dir = candidate_dirs.get(reference_id)
        try:
            if reference_dir is None:
                raise RuntimeError("same_shape_fp32_reference_missing")
            reference_build_path = reference_dir / "diagnostic_build_report.json"
            if not reference_build_path.is_file() or not json.loads(
                reference_build_path.read_text(encoding="utf-8")
            ).get("trt_build_success"):
                raise RuntimeError("same_shape_fp32_reference_not_built")
            production_runner = runner_factory(candidate_dir / "engine.plan")
            diagnostic_runner = runner_factory(
                candidate_dir / "diagnostic_engine.plan"
            )
            if reference_id == candidate_id:
                reference_runner = diagnostic_runner
            else:
                if cached_reference_id != reference_id:
                    cached_reference_runner = runner_factory(
                        reference_dir / "diagnostic_engine.plan"
                    )
                    cached_reference_id = reference_id
                reference_runner = cached_reference_runner
            report = evaluate_synthetic_runtime(
                candidate,
                production_runner=production_runner,
                diagnostic_runner=diagnostic_runner,
                reference_diagnostic_runner=reference_runner,
                input_cases=SYNTHETIC_INPUT_CASES,
                warmup_iterations=int(warmup_iterations),
                measured_iterations=int(measured_iterations),
            )
            report.update(
                {
                    "candidate_hash": candidate.candidate_hash,
                    "candidate_id": candidate_id,
                    "physical_gpu": int(physical_gpu),
                    "same_shape_fp32_reference_id": reference_id,
                }
            )
            succeeded += 1
        except Exception as exc:  # noqa: BLE001
            report = {
                "candidate_hash": candidate.candidate_hash,
                "candidate_id": candidate_id,
                "failure_reason": f"{type(exc).__name__}:{exc}",
                "physical_gpu": int(physical_gpu),
                "runtime_success": False,
                "same_shape_fp32_reference_id": reference_id,
            }
            failed += 1
            _write_json(
                destination / "failures" / f"{candidate.candidate_hash}_runtime.json",
                report,
            )
        runtime_reports[candidate_id] = report
        _write_json(report_path, report)

    for candidate_id, report in runtime_reports.items():
        if not report.get("runtime_success"):
            continue
        reference = runtime_reports.get(
            str(report["same_shape_fp32_reference_id"])
        )
        if reference and reference.get("runtime_success"):
            reference_p50 = float(reference["p50_ms"])
            candidate_p50 = float(report["p50_ms"])
            report["same_shape_fp32_p50_ms"] = reference_p50
            report["same_shape_fp32_speedup"] = reference_p50 / candidate_p50
            _write_json(candidate_dirs[candidate_id] / "runtime_report.json", report)
    return {"attempted": attempted, "failed": failed, "succeeded": succeeded}


def _read_optional_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return dict(json.loads(path.read_text(encoding="utf-8")))


def assemble_capability_matrix(output_dir: str | Path) -> dict[str, str]:
    """Assemble one evidence row per declared candidate without inventing facts."""

    destination = Path(output_dir).expanduser().resolve()
    records = json.loads(
        (destination / "candidate_matrix.json").read_text(encoding="utf-8")
    )
    hardware = _read_optional_json(destination / "hardware_manifest.json")
    rows: list[dict[str, Any]] = []
    for record in records:
        candidate = _candidate_from_record(dict(record))
        candidate_dir = _candidate_directory(destination, candidate)
        export = _read_optional_json(candidate_dir / "export_report.json")
        build = _read_optional_json(candidate_dir / "build_report.json")
        runtime = _read_optional_json(candidate_dir / "runtime_report.json")
        requested_manifest = requested_precision_manifest(
            candidate.precision_profile
        )
        requested = applicable_requested_precision(
            candidate, requested_manifest
        )
        realized = dict(build.get("realized_precision", {}))
        precision_audit = (
            audit_requested_realized_precision(requested, realized)
            if build.get("trt_build_success")
            else {
                "fallback_count": 0,
                "fallback_roles": [],
                "precision_identity": False,
                "requested_precision": requested,
                "realized_precision": realized,
            }
        )
        executed = bool(runtime.get("runtime_success"))
        if export.get("onnx_export_success") is False:
            support_class = "unsupported_export"
        elif build and not build.get("trt_build_success"):
            support_class = "unsupported_build"
        elif runtime:
            support_class = classify_support(
                {
                    "onnx_export_success": bool(
                        export.get("onnx_export_success")
                    ),
                    "trt_build_success": bool(build.get("trt_build_success")),
                    "runtime_success": executed,
                    "precision_identity": bool(
                        precision_audit["precision_identity"]
                    ),
                    "fused_mha_detected": bool(
                        build.get("fused_mha_detected")
                    ),
                }
            )
        else:
            support_class = "pending_not_executed"
        failure_reason = next(
            (
                str(payload.get("failure_reason"))
                for payload in (runtime, build, export)
                if payload.get("failure_reason")
            ),
            "",
        )
        row = {
            **candidate.to_dict(),
            "accuracy_safe": runtime.get(
                "accuracy_safe", "accuracy_unresolved"
            ),
            "av_accumulator_precision": build.get(
                "av_accumulator_precision", "unknown"
            ),
            "build_status": support_class,
            "cast_count": int(build.get("cast_count", 0)),
            "compute_capability": str(
                hardware.get("compute_capability", "unknown")
            ),
            "cuda_version": str(hardware.get("cuda_version", "unknown")),
            "engine_sha256": str(build.get("engine_sha256", "")),
            "evidence_directory": str(candidate_dir),
            "failure_reason": failure_reason,
            "fallback_count": int(precision_audit["fallback_count"]),
            "fallback_roles": list(precision_audit["fallback_roles"]),
            "fused_mha_detected": bool(build.get("fused_mha_detected", False)),
            "fusion_kind": str(build.get("fusion_kind", "unknown")),
            "gpu_model": str(hardware.get("gpu_model", "unknown")),
            "hardware_id": str(hardware.get("hardware_id", "unknown")),
            "numerical_safe": runtime.get("numerical_safe"),
            "onnx_export_success": bool(export.get("onnx_export_success", False)),
            "onnx_sha256": str(export.get("onnx_sha256", "")),
            "p50_ms": runtime.get("p50_ms"),
            "p90_ms": runtime.get("p90_ms"),
            "p99_ms": runtime.get("p99_ms"),
            "plugin_used": bool(build.get("plugin_used", False)),
            "precision_identity": bool(precision_audit["precision_identity"]),
            "qk_accumulator_precision": build.get(
                "qk_accumulator_precision", "unknown"
            ),
            "realized_av_precision": realized.get("av_matmul", "unknown"),
            "realized_qk_precision": realized.get("qk_matmul", "unknown"),
            "realized_qkv_precision": {
                role: realized.get(role, "not_applicable")
                for role in ("q_projection", "k_projection", "v_projection")
            },
            "realized_softmax_precision": realized.get("softmax", "unknown"),
            "reformat_count": int(build.get("reformat_count", 0)),
            "requested_av_precision": requested.get("av_matmul", "not_applicable"),
            "requested_qk_precision": requested.get("qk_matmul", "not_applicable"),
            "requested_qkv_precision": {
                role: requested.get(role, "not_applicable")
                for role in ("q_projection", "k_projection", "v_projection")
            },
            "requested_softmax_precision": requested.get(
                "softmax", "not_applicable"
            ),
            "runtime_success": executed,
            "same_shape_fp32_speedup": runtime.get(
                "same_shape_fp32_speedup"
            ),
            "support_class": support_class,
            "trt_build_success": bool(build.get("trt_build_success", False)),
        }
        rows.append(row)
    paths = write_capability_matrix(destination, rows)
    contract_path = destination / "final_head_dim_search_contract.json"
    _write_json(
        contract_path,
        derive_head_dim_search_contract(
            rows,
            hardware_scope={
                key: hardware.get(key)
                for key in (
                    "gpu_model",
                    "compute_capability",
                    "tensorrt_version",
                    "cuda_version",
                    "driver_version",
                )
            },
        ),
    )
    parity_rows = []
    for row in rows:
        runtime = _read_optional_json(
            Path(str(row["evidence_directory"])) / "runtime_report.json"
        )
        parity_rows.extend(
            {"candidate_id": row["candidate_id"], **dict(metrics)}
            for metrics in runtime.get("parity", [])
        )
    parity_json = destination / "synthetic_numerical_parity.json"
    _write_json(parity_json, parity_rows)
    parity_csv = destination / "synthetic_numerical_parity.csv"
    parity_fields = sorted({key for row in parity_rows for key in row})
    with parity_csv.open("w", encoding="utf-8", newline="") as handle:
        if parity_fields:
            writer = csv.DictWriter(handle, fieldnames=parity_fields)
            writer.writeheader()
            for row in parity_rows:
                writer.writerow(
                    {
                        key: (
                            json.dumps(row.get(key), sort_keys=True)
                            if isinstance(row.get(key), (dict, list, tuple))
                            else row.get(key)
                        )
                        for key in parity_fields
                    }
                )
    return {
        **{key: str(value) for key, value in paths.items()},
        "contract": str(contract_path),
        "parity_csv": str(parity_csv),
        "parity_json": str(parity_json),
    }


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return str(completed.stdout).strip()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CoBEVT TensorRT head-dimension capability matrix"
    )
    parser.add_argument(
        "--phase",
        choices=(
            "prepare",
            "prepare-real",
            "audit-real-structure",
            "export",
            "build",
            "build-diagnostic",
            "runtime",
            "assemble",
        ),
        required=True,
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--previous-output",
        default=(
            "/data/lxf/heal_data/outputs/"
            "cobevt_attention_fp16_boundary_audit_20260719_030621"
        ),
    )
    parser.add_argument(
        "--modelopt-python",
        default="/home/lixingfeng/anaconda3/envs/modelopt/bin/python",
    )
    parser.add_argument("--physical-gpu", type=int, default=2)
    parser.add_argument("--only-candidate", action="append", default=[])
    parser.add_argument("--trace-window-groups", type=int, default=1)
    parser.add_argument(
        "--source-real-experiment",
        default=(
            "/data/lxf/heal_data/outputs/"
            "cobevt_attention_dim_pruning_20260718_085644"
        ),
    )
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shard-count", type=int)
    parser.add_argument("--allow-shared-gpu", action="store_true")
    parser.add_argument(
        "--checkpoint",
        default=(
            "/home/lixingfeng/UniAD_examine/Auto_Search/original_models/"
            "dairv2s/LiDAROnly/lidar_cobevt/net_epoch_bestval_at19.pth"
        ),
    )
    parser.add_argument(
        "--config",
        default=(
            "/home/lixingfeng/UniAD_examine/Auto_Search/original_models/"
            "dairv2s/LiDAROnly/lidar_cobevt/config.yaml"
        ),
    )
    parser.add_argument(
        "--heal-root", default="/home/lixingfeng/UniAD_examine/HEAL"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output_dir = Path(args.output_dir).expanduser().resolve()
    only = set(str(value) for value in args.only_candidate) or None
    if (args.shard_index is None) != (args.shard_count is None):
        raise ValueError("candidate_shard_index_and_count_required_together")
    if args.shard_index is not None:
        records = json.loads(
            (output_dir / "candidate_matrix.json").read_text(encoding="utf-8")
        )
        candidates = [_candidate_from_record(dict(row)) for row in records]
        shard = candidate_ids_for_shard(
            candidates,
            shard_index=int(args.shard_index),
            shard_count=int(args.shard_count),
        )
        only = shard if only is None else only & shard
    if args.phase == "prepare":
        environment = discover_capability_environment(
            previous_output=args.previous_output,
            modelopt_python=args.modelopt_python,
            physical_gpu=int(args.physical_gpu),
        )
        result = prepare_capability_run(
            output_dir,
            environment=environment,
            code_commit=_git_commit(),
        )
    elif args.phase == "prepare-real":
        result = prepare_real_cobevt_integration(
            output_dir / "real_cobevt",
            args.source_real_experiment,
        )
    elif args.phase == "audit-real-structure":
        result = audit_real_cobevt_structures(
            output_dir / "real_cobevt",
            checkpoint=args.checkpoint,
            config=args.config,
            heal_root=args.heal_root,
        )
    elif args.phase == "export":
        result = export_capability_candidates(
            output_dir,
            only_candidate_ids=only,
            trace_window_groups=int(args.trace_window_groups),
        )
    elif args.phase == "assemble":
        result = assemble_capability_matrix(output_dir)
    else:
        manifest = json.loads(
            (output_dir / "run_manifest.json").read_text(encoding="utf-8")
        )
        environment = dict(manifest["environment"])
        gpu_audit = audit_gpu_idle(int(args.physical_gpu))
        if not gpu_audit["gpu_idle"] and not bool(args.allow_shared_gpu):
            raise RuntimeError(
                "capability_gpu_not_idle:" + json.dumps(gpu_audit, sort_keys=True)
            )
        if any(row.get("is_pyramid") for row in gpu_audit["compute_processes"]):
            raise RuntimeError("capability_refuses_pyramid_gpu")
        if args.phase == "runtime":
            result = run_capability_candidates(
                output_dir,
                environment=environment,
                physical_gpu=int(args.physical_gpu),
                only_candidate_ids=only,
            )
        else:
            result = build_capability_candidates(
                output_dir,
                environment=environment,
                physical_gpu=int(args.physical_gpu),
                only_candidate_ids=only,
                diagnostic=args.phase == "build-diagnostic",
            )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if int(result.get("failed", 0)) == 0 else 2


__all__ = [
    "assemble_capability_matrix",
    "audit_real_cobevt_structures",
    "build_trtexec_command",
    "build_capability_candidates",
    "candidate_ids_for_shard",
    "candidate_input_shapes",
    "capability_build_signature",
    "export_capability_candidates",
    "prepare_capability_run",
    "prepare_real_cobevt_integration",
    "resolve_tensorrt_evidence",
    "run_capability_candidates",
]


if __name__ == "__main__":
    raise SystemExit(main())
