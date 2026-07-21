"""Fresh full-CoBEVT native tactic pinning helpers.

The module deliberately treats the full model as a separate graph from the
canonical micrographs.  A micrograph cache key is never reused for a full
model layer; full graph keys are collected from its own editable-cache build.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Iterable

from search.model_families.lidar_cobevt.native_tactic_evidence import (
    parse_editable_timing_log,
    select_f16a32_tactic,
)
from search.orchestration.lidar_cobevt_native_tactic_microbench import (
    _RecordingLogger,
    edit_timing_cache,
)


_ROLE = re.compile(
    r"layers[._](?P<block>[0-9]+)[_/](?P<kind>window|grid)_attention[^/]*[/]fn[/]Einsum(?P<av>_1)?",
    re.IGNORECASE,
)


def parse_attention_role(op_name: str) -> tuple[str, int, str] | None:
    """Return (QK/AV, block, window/grid) for an ONNX/TRT op name."""

    match = _ROLE.search(str(op_name))
    if match is None:
        return None
    return ("AV" if match.group("av") else "QK", int(match.group("block")), match.group("kind").lower())


def choose_full_graph_targets(records: Iterable[dict[str, Any]], *, role: str) -> list[dict[str, Any]]:
    """Choose one F16A32 kernel for each of the six semantic full-graph layers."""

    role = role.upper()
    selected: dict[tuple[str, int, str], dict[str, Any]] = {}
    for record in records:
        parsed = parse_attention_role(str(record.get("op", "")))
        if parsed is None or parsed[0] != role:
            continue
        candidate = select_f16a32_tactic(record)
        if candidate is None:
            continue
        key = (parsed[0], parsed[1], parsed[2])
        value = dict(record)
        value.update(
            {
                "role": parsed[0],
                "block": parsed[1],
                "attention_kind": parsed[2],
                "requested_tactic_hash": candidate["tactic_hash"],
                "requested_kernel_name": candidate["kernel_name"],
                "requested_kernel_evidence": candidate.get("kernel_evidence", {}),
            }
        )
        selected[key] = value
    ordered = [selected[key] for key in sorted(selected, key=lambda item: (item[1], item[2]))]
    if len(ordered) != 6:
        raise ValueError(f"full_graph_expected_six_targets:{role}:{len(ordered)}")
    return ordered


def _layer_output_precision(layer: dict[str, Any]) -> str:
    outputs = layer.get("Outputs", [])
    if not outputs:
        return "unknown"
    text = str(outputs[0].get("Format/Datatype", "")).upper()
    if "HALF" in text:
        return "FP16"
    if "FLOAT" in text:
        return "FP32"
    if "BF16" in text:
        return "BF16"
    return text or "unknown"


def _layer_tactic_hash(name: str) -> str | None:
    values = re.findall(r"0x[0-9a-fA-F]+", str(name))
    return values[-1] if values else None


def verify_full_engine_preservation(
    targets: Iterable[dict[str, Any]], inspector: dict[str, Any]
) -> dict[str, Any]:
    """Check layer identity, tactic identity and reject plugin/fused replacement."""

    layers = inspector.get("Layers", []) if isinstance(inspector, dict) else []
    rows: list[dict[str, Any]] = []
    for target in targets:
        match = None
        target_role = (str(target.get("role")), int(target.get("block", -1)), str(target.get("attention_kind")))
        for layer in layers:
            parsed = parse_attention_role(str(layer.get("Name", "")))
            if parsed == target_role:
                match = layer
                break
        if match is None:
            rows.append({**target, "matched": False, "preserved": False, "reason": "layer_missing"})
            continue
        tactic_name = str(match.get("TacticName", ""))
        tactic_hash = _layer_tactic_hash(tactic_name)
        layer_type = str(match.get("LayerType", ""))
        fused_or_plugin = "gemm_mha" in tactic_name.lower() or "plugin" in layer_type.lower()
        name_match = str(target.get("requested_kernel_name", "")) == tactic_name
        hash_match = bool(tactic_hash and tactic_hash == target.get("requested_tactic_hash"))
        # Full graph implementations often decorate a kernel with a generated
        # suffix.  When the exact hash is absent, kernel token equality is the
        # conservative fallback; no token equality means not preserved.
        preserved = (name_match or hash_match) and not fused_or_plugin
        rows.append(
            {
                **target,
                "matched": True,
                "realized_tactic_name": tactic_name,
                "realized_tactic_hash": tactic_hash,
                "realized_output_precision": _layer_output_precision(match),
                "layer_type": layer_type,
                "fused_or_plugin": fused_or_plugin,
                "tactic_match": bool(name_match or hash_match),
                "preserved": preserved,
                "reason": "ok" if preserved else ("fused_or_plugin" if fused_or_plugin else "tactic_mismatch"),
            }
        )
    return {
        "preserved": len(rows) == 6 and all(bool(row.get("preserved")) for row in rows),
        "matched_count": sum(bool(row.get("matched")) for row in rows),
        "rows": rows,
    }


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key, value in row.items() if not isinstance(value, (dict, list))})
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


class FullGraphBuildError(RuntimeError):
    pass


def build_full_graph_engine(
    *,
    onnx_path: Path,
    engine_path: Path,
    cache_path: Path,
    build_log_path: Path,
    inspector_path: Path,
    plugin_path: Path | None = None,
    input_shapes: dict[str, tuple[int, ...]] | None = None,
    cache_blob: bytes | None = None,
    error_on_cache_miss: bool = False,
    workspace_limit_bytes: int = 4 << 30,
) -> dict[str, Any]:
    """Build a fresh strongly-typed full graph using only TensorRT Python API."""

    import ctypes
    import tensorrt as trt

    if plugin_path is not None:
        ctypes.CDLL(str(plugin_path), mode=ctypes.RTLD_GLOBAL)
    logger = _RecordingLogger()
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, logger)
    parser.clear_errors()
    payload = onnx_path.read_bytes()
    if not parser.parse(payload):
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise FullGraphBuildError("onnx_parse_failed:" + " | ".join(errors[-10:]))
    config = builder.create_builder_config()
    config.clear_flag(trt.BuilderFlag.TF32)
    config.set_flag(trt.BuilderFlag.EDITABLE_TIMING_CACHE)
    if error_on_cache_miss:
        config.set_flag(trt.BuilderFlag.ERROR_ON_TIMING_CACHE_MISS)
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, int(workspace_limit_bytes)
    )
    cache = config.create_timing_cache(cache_blob or b"")
    if cache is None or not config.set_timing_cache(cache, True):
        raise FullGraphBuildError("timing_cache_attach_failed")
    started = time.monotonic()
    serialized = builder.build_serialized_network(network, config)
    elapsed = time.monotonic() - started
    build_log_path.parent.mkdir(parents=True, exist_ok=True)
    build_log_path.write_text("\n".join(logger.lines) + "\n")
    if serialized is None:
        raise FullGraphBuildError("full_graph_build_failed")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(bytes(serialized))
    cache_blob_out = bytes(cache.serialize())
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(cache_blob_out)
    runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
    engine = runtime.deserialize_cuda_engine(serialized)
    if engine is None:
        raise FullGraphBuildError("full_graph_deserialize_failed")
    inspector = engine.create_engine_inspector()
    info = json.loads(inspector.get_engine_information(trt.LayerInformationFormat.JSON))
    _write_json(inspector_path, info)
    return {
        "build_seconds": elapsed,
        "onnx_sha256": hashlib.sha256(payload).hexdigest(),
        "engine_sha256": _sha256(engine_path),
        "cache_sha256": hashlib.sha256(cache_blob_out).hexdigest(),
        "engine_size_bytes": engine_path.stat().st_size,
        "cache_blob": cache_blob_out,
        "layer_info": info,
        "num_layers": len(info.get("Layers", [])),
        "workspace_limit_bytes": int(workspace_limit_bytes),
    }


def edit_full_graph_cache(cache_blob: bytes, targets: Iterable[dict[str, Any]]) -> bytes:
    edited = cache_blob
    for target in targets:
        edited = edit_timing_cache(edited, str(target["key"]), str(target["requested_tactic_hash"]))
    return edited


def parse_full_graph_build_targets(build_log: Path, *, role: str) -> list[dict[str, Any]]:
    records = parse_editable_timing_log(build_log.read_text(errors="replace").splitlines())
    targets = choose_full_graph_targets(records, role=role)
    return targets


def write_full_target_inventory(path: Path, targets: Iterable[dict[str, Any]]) -> None:
    rows = []
    for target in targets:
        rows.append(
            {
                "role": target.get("role"),
                "block": target.get("block"),
                "attention_kind": target.get("attention_kind"),
                "op": target.get("op"),
                "key": target.get("key"),
                "requested_tactic_hash": target.get("requested_tactic_hash"),
                "requested_kernel_name": target.get("requested_kernel_name"),
                "requested_compute_phenotype": target.get("requested_kernel_evidence", {}).get("compute_phenotype"),
                "requested_evidence_level": target.get("requested_kernel_evidence", {}).get("evidence_level"),
            }
        )
    _write_csv(path, rows)


__all__ = [
    "FullGraphBuildError",
    "build_full_graph_engine",
    "choose_full_graph_targets",
    "edit_full_graph_cache",
    "parse_attention_role",
    "parse_full_graph_build_targets",
    "verify_full_engine_preservation",
    "write_full_target_inventory",
]
