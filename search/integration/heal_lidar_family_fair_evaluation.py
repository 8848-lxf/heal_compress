"""Fair full-validation for existing HEAL LiDAR family ablation engines.

This module is intentionally evaluation-only.  It does not import any engine
builder and never copies a TensorRT engine into the evaluation directory.  A
source engine is locked by path, SHA256, size, and mtime before evaluation and
verified again after the fresh modelopt worker exits.

The inventory contract is shared by F-Cooper and DiscoNet: each search method
has one strict-FP32 baseline followed by six BOPS budgets and the joint,
pruning-only, and quantization-only variants.  Methods may run concurrently on
different GPUs; every item assigned to one method is evaluated serially.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

from .runtime_environment import modelopt_python_command
from ..model_family.evaluation import evaluate_v2xvit_engine_modelopt


METHODS = ("ga", "greedy")
VARIANTS = ("prune_quant", "prune_only", "quant_only")
BUDGETS = (0.30, 0.25, 0.20, 0.15, 0.10, 0.05)
NUMERIC_METRICS = (
    "AP@0.3",
    "AP@0.5",
    "AP@0.7",
    "mAP",
    "forward_mean_ms",
    "forward_p50_ms",
    "forward_p90_ms",
    "forward_p99_ms",
    "postprocess_mean_ms",
    "postprocess_p50_ms",
    "postprocess_p90_ms",
    "postprocess_p99_ms",
    "total_mean_ms",
    "total_p50_ms",
    "total_p90_ms",
    "total_p99_ms",
    "speedup_vs_same_gpu_fp32",
)
FORBIDDEN_OUTPUT_SUFFIXES = (".plan", ".engine", ".onnx", ".pth")


def read_json(path: str | Path) -> dict[str, Any]:
    return dict(json.loads(Path(path).read_text(encoding="utf-8")))


def write_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    temporary.replace(destination)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _float_budget(value: Any) -> float:
    return round(float(value), 2)


def _resolve_engine(row: Mapping[str, Any], build_root: Path) -> Path:
    explicit = str(row.get("engine_path", "")).strip()
    if explicit:
        path = Path(explicit).expanduser()
        return (path if path.is_absolute() else build_root / path).resolve()
    artifact_value = str(row.get("artifact_dir", "")).strip()
    if not artifact_value:
        raise RuntimeError(f"family_ablation_engine_path_missing:{row.get('row_id', '')}")
    artifact = Path(artifact_value).expanduser()
    artifact = (artifact if artifact.is_absolute() else build_root / artifact).resolve()
    candidates = (artifact / "deployment/candidate.plan", artifact / "engine.plan")
    existing = [path for path in candidates if path.is_file()]
    if len(existing) != 1:
        raise RuntimeError(
            f"family_ablation_engine_resolution_failed:{artifact}:{existing}"
        )
    return existing[0]


def lock_engine(
    engine_path: str | Path, *, expected_sha256: str = ""
) -> dict[str, Any]:
    engine = Path(engine_path).expanduser().resolve()
    if not engine.is_file() or engine.stat().st_size <= 0:
        raise RuntimeError(f"family_ablation_engine_missing_or_empty:{engine}")
    stat = engine.stat()
    digest = sha256_file(engine)
    if expected_sha256 and digest != str(expected_sha256):
        raise RuntimeError(
            f"family_ablation_engine_hash_mismatch:{engine}:{expected_sha256}:{digest}"
        )
    return {
        "engine_path": str(engine),
        "engine_sha256": digest,
        "engine_size_bytes": int(stat.st_size),
        "engine_mtime_ns": int(stat.st_mtime_ns),
    }


def verify_engine_lock(lock: Mapping[str, Any]) -> None:
    engine = Path(str(lock["engine_path"]))
    if not engine.is_file():
        raise RuntimeError(f"family_ablation_engine_disappeared:{engine}")
    stat = engine.stat()
    errors = []
    if int(stat.st_size) != int(lock["engine_size_bytes"]):
        errors.append("size")
    if int(stat.st_mtime_ns) != int(lock["engine_mtime_ns"]):
        errors.append("mtime")
    if sha256_file(engine) != str(lock["engine_sha256"]):
        errors.append("sha256")
    if errors:
        raise RuntimeError(f"family_ablation_engine_changed:{engine}:{errors}")


def _precision_counts(row: Mapping[str, Any]) -> dict[str, int]:
    nested = dict(row.get("precision_counts") or {})
    aliases = {
        "int8_count": ("int8_count", "realized_int8_count"),
        "fp16_count": ("fp16_count", "realized_fp16_count"),
        "fp32_count": ("fp32_count", "realized_fp32_count"),
    }
    result: dict[str, int] = {}
    for output, keys in aliases.items():
        value: Any = 0
        for key in keys:
            if key in row:
                value = row[key]
                break
            if key in nested:
                value = nested[key]
                break
        result[output] = int(value or 0)
    return result


def _resolve_optional_path(value: Any, *, root: Path) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def _precision_counts_with_artifacts(
    row: Mapping[str, Any], *, build_root: Path
) -> dict[str, int]:
    counts = _precision_counts(row)
    if sum(counts.values()) > 0:
        return counts
    phenotype_path = _resolve_optional_path(
        row.get("phenotype_path"), root=build_root
    )
    if phenotype_path is None:
        artifact = _resolve_optional_path(row.get("artifact_dir"), root=build_root)
        if artifact is not None:
            phenotype_path = artifact / "phenotype.json"
    if phenotype_path is None or not phenotype_path.is_file():
        raise RuntimeError(
            f"family_ablation_precision_profile_missing:{row.get('row_id', '')}:{phenotype_path}"
        )
    phenotype = read_json(phenotype_path)
    profile = dict(phenotype.get("realized_precision_profile") or {})
    if not profile:
        # CandidatePhenotype.to_dict also carries the decision-rich profile.
        decisions = dict(phenotype.get("precision_profile") or {})
        profile = {
            str(key): str(dict(value).get("realized_precision", ""))
            for key, value in decisions.items()
        }
    values = [str(value).strip().upper() for value in profile.values()]
    if not values or any(value not in {"INT8", "FP16", "FP32"} for value in values):
        raise RuntimeError(
            f"family_ablation_precision_profile_invalid:{phenotype_path}"
        )
    return {
        "int8_count": sum(value == "INT8" for value in values),
        "fp16_count": sum(value == "FP16" for value in values),
        "fp32_count": sum(value == "FP32" for value in values),
    }


def _parameter_fields(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "parameter_count_base": row.get(
            "parameter_count_base", row.get("physical_parameter_count_before")
        ),
        "parameter_count_pruned": row.get(
            "parameter_count_pruned", row.get("physical_parameter_count_after")
        ),
        "parameter_reduction": row.get(
            "parameter_reduction", row.get("physical_parameter_pruning_ratio")
        ),
    }


def _parameter_fields_with_artifacts(
    row: Mapping[str, Any], *, build_root: Path
) -> dict[str, Any]:
    values = _parameter_fields(row)
    if values["parameter_count_base"] is not None:
        return values
    candidates: list[Path] = []
    variant = str(row.get("variant", "")).lower()
    artifact = _resolve_optional_path(row.get("artifact_dir"), root=build_root)
    source = _resolve_optional_path(row.get("source_artifact_dir"), root=build_root)
    if variant == "prune_quant":
        for root in (artifact, source):
            if root is not None:
                candidates.extend(
                    (root / "candidate_stage2_result.json", root / "stage2_score.json")
                )
    else:
        if artifact is not None:
            candidates.append(artifact / "candidate_build_result.json")
    for path in candidates:
        if not path.is_file():
            continue
        payload = read_json(path)
        if str(payload.get("status", "")) != "ok":
            raise RuntimeError(
                f"family_ablation_parameter_result_not_accepted:{path}:"
                f"{payload.get('status')}"
            )
        if path.name == "candidate_build_result.json" and payload.get(
            "evaluation_invoked"
        ) is not False:
            raise RuntimeError(
                f"family_ablation_build_result_evaluation_contract:{path}"
            )
        fields = _parameter_fields(payload)
        if fields["parameter_count_base"] is not None:
            return fields
    raise RuntimeError(
        f"family_ablation_parameter_counts_missing:{row.get('row_id', '')}:{candidates}"
    )


def build_family_evaluation_inventories(
    *,
    build_root: str | Path,
    baseline_engine_path: str | Path,
    family_id: str,
    methods: Sequence[str] = METHODS,
    budgets: Sequence[float] = BUDGETS,
    variants: Sequence[str] = VARIANTS,
) -> dict[str, list[dict[str, Any]]]:
    """Validate the logical 2x6x3 matrix and prepend one FP32 per method."""

    if family_id not in {
        "heal_lidar_attfusion",
        "heal_lidar_cobevt",
        "heal_lidar_fcooper",
        "heal_lidar_disco",
    }:
        raise ValueError(f"unsupported_heal_lidar_family:{family_id}")
    root = Path(build_root).expanduser().resolve()
    inventory_path = root / "engine_inventory.json"
    fallback_path = root / "ablation_results.json"
    payload_path = inventory_path if inventory_path.is_file() else fallback_path
    if not payload_path.is_file():
        raise RuntimeError(
            f"family_ablation_results_missing:{inventory_path}:{fallback_path}"
        )
    payload = read_json(payload_path)
    payload_family = str(payload.get("family_id", ""))
    if payload_family and payload_family != family_id:
        raise RuntimeError(
            f"family_ablation_results_family_mismatch:{payload_family}:{family_id}"
        )
    rows = [dict(row) for row in payload.get("rows", [])]
    expected = {
        (str(method), _float_budget(budget), str(variant))
        for method in methods
        for budget in budgets
        for variant in variants
    }
    actual = {
        (
            str(row.get("method", "")).lower(),
            _float_budget(row["budget"]),
            str(row.get("variant", "")).lower(),
        )
        for row in rows
    }
    if actual != expected or len(rows) != len(expected):
        raise RuntimeError(
            "family_ablation_inventory_mismatch:"
            f"missing={sorted(expected-actual)}:extra={sorted(actual-expected)}:"
            f"rows={len(rows)}:unique={len(actual)}"
        )

    baseline_lock = lock_engine(baseline_engine_path)
    resolved_precision = {
        str(row.get("row_id", index)): _precision_counts_with_artifacts(
            row, build_root=root
        )
        for index, row in enumerate(rows)
    }
    resolved_parameters = {
        str(row.get("row_id", index)): _parameter_fields_with_artifacts(
            row, build_root=root
        )
        for index, row in enumerate(rows)
    }
    weighted_counts = [sum(counts.values()) for counts in resolved_precision.values()]
    baseline_weighted_count = max(weighted_counts, default=0)
    parameter_bases = [
        fields["parameter_count_base"]
        for fields in resolved_parameters.values()
        if fields["parameter_count_base"] is not None
    ]
    baseline_parameters = parameter_bases[0] if parameter_bases else None
    inventories: dict[str, list[dict[str, Any]]] = {}
    variant_order = {name: index for index, name in enumerate(variants)}
    for method in methods:
        method_rows = [
            row for row in rows if str(row.get("method", "")).lower() == method
        ]
        method_rows.sort(
            key=lambda row: (
                -_float_budget(row["budget"]),
                variant_order[str(row["variant"]).lower()],
            )
        )
        inventory: list[dict[str, Any]] = [
            {
                "sequence_index": 0,
                "item_id": f"{method}_fp32_original",
                "family_id": family_id,
                "assigned_method": method,
                "method": "fp32",
                "variant": "fp32",
                "budget": None,
                "actual_bops": 1.0,
                **baseline_lock,
                "int8_count": 0,
                "fp16_count": 0,
                "fp32_count": baseline_weighted_count,
                "precision_counts_inferred": True,
                "parameter_count_base": baseline_parameters,
                "parameter_count_pruned": baseline_parameters,
                "parameter_reduction": 0.0,
                "source_row_id": "original_strict_fp32",
            }
        ]
        for index, row in enumerate(method_rows, start=1):
            resolution_key = str(row.get("row_id", rows.index(row)))
            expected_hash = str(
                row.get("engine_sha256", row.get("engine_hash", ""))
            )
            if not expected_hash:
                raise RuntimeError(
                    f"family_ablation_logical_row_engine_hash_missing:{row.get('row_id', '')}"
                )
            engine_lock = lock_engine(
                _resolve_engine(row, root), expected_sha256=expected_hash
            )
            budget = _float_budget(row["budget"])
            variant = str(row["variant"]).lower()
            inventory.append(
                {
                    **row,
                    "sequence_index": index,
                    "item_id": f"{method}_bops_{budget:.2f}_{variant}",
                    "family_id": family_id,
                    "assigned_method": method,
                    "method": method,
                    "variant": variant,
                    "budget": budget,
                    **engine_lock,
                    **resolved_precision[resolution_key],
                    **resolved_parameters[resolution_key],
                }
            )
        expected_count = 1 + len(budgets) * len(variants)
        if len(inventory) != expected_count:
            raise RuntimeError(
                f"family_ablation_method_inventory_count:{method}:{len(inventory)}"
            )
        inventories[method] = inventory
    return inventories


def gpu_snapshot(gpu_id: int) -> dict[str, Any]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            f"--id={int(gpu_id)}",
            "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"family_gpu_snapshot_failed:{completed.stderr.strip()}")
    fields = [value.strip() for value in completed.stdout.strip().split(",")]
    if len(fields) != 7:
        raise RuntimeError(f"family_gpu_snapshot_malformed:{completed.stdout!r}")
    return {
        "gpu_id": int(fields[0]),
        "uuid": fields[1],
        "name": fields[2],
        "memory_total_mib": int(fields[3]),
        "memory_used_mib": int(fields[4]),
        "memory_free_mib": int(fields[5]),
        "utilization_percent": int(fields[6]),
        "captured_at_epoch": time.time(),
    }


def clear_gpu_cache_and_wait(
    *,
    gpu_id: int,
    before: Mapping[str, Any],
    conda_env: str = "modelopt",
    allowed_residual_mib: int = 128,
    timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    """Destroy the worker context, flush allocators, and prove memory rollback."""

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(int(gpu_id))
    code = (
        "import torch; torch.cuda.set_device(0); torch.cuda.synchronize(); "
        "torch.cuda.empty_cache(); torch.cuda.ipc_collect(); "
        "torch.cuda.synchronize(); "
        "print(torch.cuda.memory_allocated(0), torch.cuda.memory_reserved(0))"
    )
    completed = subprocess.run(
        modelopt_python_command(conda_env) + ["-c", code],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
    )
    threshold = int(before["memory_used_mib"]) + int(allowed_residual_mib)
    deadline = time.monotonic() + float(timeout_seconds)
    snapshots: list[dict[str, Any]] = []
    while True:
        current = gpu_snapshot(gpu_id)
        snapshots.append(current)
        if current["memory_used_mib"] <= threshold or time.monotonic() >= deadline:
            break
        time.sleep(0.5)
    final = snapshots[-1]
    passed = completed.returncode == 0 and final["memory_used_mib"] <= threshold
    return {
        "passed": passed,
        "cleanup_returncode": int(completed.returncode),
        "cleanup_output": completed.stdout.strip(),
        "process_exit_released_context": True,
        "before_memory_used_mib": int(before["memory_used_mib"]),
        "allowed_residual_mib": int(allowed_residual_mib),
        "threshold_memory_used_mib": threshold,
        "after": final,
        "poll_count": len(snapshots),
    }


def _is_warmup_latency_row(row: Mapping[str, Any]) -> bool:
    if "phase" in row:
        phase = str(row["phase"]).strip().lower()
        if phase not in {"warmup", "evaluation"}:
            raise RuntimeError(f"family_latency_row_phase_invalid:{phase}")
        return phase == "warmup"
    return bool(row.get("warmup", False))


def validate_family_evaluation_result(
    result: Mapping[str, Any],
    *,
    num_frames: int,
    warmup_frames: int,
    manifest_hash: str,
    fixed_k: int,
) -> None:
    errors: list[str] = []
    if result.get("status") != "ok":
        errors.append(f"status={result.get('status')}:{result.get('failure_reason', '')}")
    if int(result.get("num_warmup_frames", -1)) != int(warmup_frames):
        errors.append(f"warmup={result.get('num_warmup_frames')}")
    if int(result.get("num_evaluated_frames", -1)) != int(num_frames):
        errors.append(f"evaluated={result.get('num_evaluated_frames')}")
    if int(result.get("num_skipped_frames", -1)) != 0:
        errors.append(f"skipped={result.get('num_skipped_frames')}")
    if int(result.get("dataloader_num_workers", -1)) != 8:
        errors.append(f"workers={result.get('dataloader_num_workers')}")
    if not bool(result.get("reset_after_warmup", False)):
        errors.append("warmup_not_reset")
    if str(result.get("eval_manifest_hash", "")) != str(manifest_hash):
        errors.append(f"manifest={result.get('eval_manifest_hash')}")
    if int(result.get("fixed_k", -1)) != int(fixed_k):
        errors.append(f"fixed_k={result.get('fixed_k')}")
    if str(result.get("input_contract", "")) != "heal_lidar_baseline_fixed_k":
        errors.append(f"input_contract={result.get('input_contract')}")
    if not bool(dict(result.get("cuda_postprocess_audit") or {}).get("passed", False)):
        errors.append("cuda_postprocess_not_passed")
    latency_rows = list(result.get("latency_rows") or [])
    warmup_count = sum(_is_warmup_latency_row(row) for row in latency_rows)
    evaluation_count = len(latency_rows) - warmup_count
    if warmup_count != int(warmup_frames):
        errors.append(f"latency_warmup_rows={warmup_count}")
    if evaluation_count != int(num_frames):
        errors.append(f"latency_evaluation_rows={evaluation_count}")
    if errors:
        raise RuntimeError(f"family_fair_evaluation_protocol_failed:{errors}")


def evaluation_frame_latency_rows(
    result: Mapping[str, Any], *, metadata: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Return only measured frames, accepting both phase and warmup schemas."""

    output: list[dict[str, Any]] = []
    for raw in list(result.get("latency_rows") or []):
        row = dict(raw)
        if _is_warmup_latency_row(row) or row.get("success") is False:
            continue
        forward = float(row["forward_ms"])
        postprocess = float(row["postprocess_ms"])
        output.append(
            {
                **dict(metadata),
                "frame_id": str(row["frame_id"]),
                "input_prepare_ms": row.get("input_prepare_ms"),
                "host_to_device_ms": row.get("host_to_device_ms"),
                "forward_ms": forward,
                "postprocess_ms": postprocess,
                "total_ms": float(row.get("total_ms", forward + postprocess)),
            }
        )
    expected = int(result.get("num_evaluated_frames", -1))
    if len(output) != expected:
        raise RuntimeError(
            f"family_per_frame_latency_count_mismatch:{len(output)}:{expected}"
        )
    return output


def write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot_write_empty_family_evaluation_csv")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def _ensure_evaluation_only_output(output_dir: Path) -> None:
    forbidden = [
        str(path)
        for path in output_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in FORBIDDEN_OUTPUT_SUFFIXES
    ]
    if forbidden:
        raise RuntimeError(f"family_evaluation_output_contains_build_artifact:{forbidden}")


def evaluate_existing_family_engine(
    *,
    source: Mapping[str, Any],
    gpu_id: int,
    repeat_index: int,
    output_dir: str | Path,
    model_config: str | Path,
    heal_root: str | Path,
    tensorrt_root: str | Path,
    plugin_path: str | Path,
    eval_manifest_path: str | Path,
    num_frames: int = 1789,
    warmup_frames: int = 200,
    latency_rounds: int = 3,
    fixed_k: int = 29696,
    max_agents: int = 2,
    evaluator: Callable[..., dict[str, Any]] = evaluate_v2xvit_engine_modelopt,
    snapshotter: Callable[[int], dict[str, Any]] = gpu_snapshot,
    cleaner: Callable[..., dict[str, Any]] = clear_gpu_cache_and_wait,
) -> dict[str, Any]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    manifest = read_json(eval_manifest_path)
    manifest_hash = str(manifest.get("manifest_hash", ""))
    if not manifest_hash:
        raise RuntimeError(f"family_eval_manifest_hash_missing:{eval_manifest_path}")
    if not bool(manifest.get("reset_after_warmup", False)):
        raise RuntimeError("family_eval_manifest_requires_warmup_reset")
    engine_lock = lock_engine(
        source["engine_path"], expected_sha256=str(source["engine_sha256"])
    )
    before = snapshotter(int(gpu_id))
    write_json(destination / "gpu_before.json", before)
    started_at = time.time()
    result: dict[str, Any] = {}
    evaluation_error: BaseException | None = None
    try:
        result = evaluator(
            engine_path=engine_lock["engine_path"],
            model_config=model_config,
            heal_root=heal_root,
            output_dir=destination,
            tensorrt_root=tensorrt_root,
            plugin_path=plugin_path,
            eval_manifest_path=eval_manifest_path,
            physical_gpu_id=int(gpu_id),
            fixed_k=int(fixed_k),
            max_agents=int(max_agents),
            num_frames=int(num_frames),
            warmup_frames=int(warmup_frames),
            latency_rounds=int(latency_rounds),
            dataloader_num_workers=8,
            input_contract="heal_lidar_baseline_fixed_k",
        )
    except BaseException as exc:  # cleanup must run before propagation
        evaluation_error = exc
    finished_at = time.time()
    cleanup = cleaner(gpu_id=int(gpu_id), before=before, conda_env="modelopt")
    write_json(destination / "gpu_cleanup_audit.json", cleanup)
    verify_engine_lock(engine_lock)
    if evaluation_error is not None:
        raise evaluation_error
    if not bool(cleanup.get("passed", False)):
        raise RuntimeError(f"family_gpu_cleanup_failed:{gpu_id}:{cleanup}")
    validate_family_evaluation_result(
        result,
        num_frames=num_frames,
        warmup_frames=warmup_frames,
        manifest_hash=manifest_hash,
        fixed_k=fixed_k,
    )
    metadata = {
        "family_id": source["family_id"],
        "assigned_method": source["assigned_method"],
        "repeat_index": int(repeat_index),
        "gpu_id": int(gpu_id),
        "sequence_index": int(source["sequence_index"]),
        "item_id": source["item_id"],
        "variant": source["variant"],
        "budget": source.get("budget"),
        "actual_bops": source.get("actual_bops"),
        "engine_sha256": engine_lock["engine_sha256"],
    }
    frame_rows = evaluation_frame_latency_rows(result, metadata=metadata)
    per_frame_path = destination / "per_frame_latency.csv"
    write_csv(per_frame_path, frame_rows)
    _ensure_evaluation_only_output(destination)
    provenance = {
        "schema_version": "heal-lidar-family-fair-evaluation-v1",
        **metadata,
        **engine_lock,
        "source_engine_unchanged": True,
        "engine_build_invoked": False,
        "eval_manifest_path": str(Path(eval_manifest_path).resolve()),
        "eval_manifest_hash": manifest_hash,
        "num_frames": int(num_frames),
        "warmup_frames": int(warmup_frames),
        "latency_rounds": int(latency_rounds),
        "dataloader_num_workers": 8,
        "cuda_postprocess": True,
        "started_at_epoch": started_at,
        "finished_at_epoch": finished_at,
        "wall_seconds": finished_at - started_at,
        "gpu_before": before,
        "gpu_cleanup": cleanup,
        "per_frame_csv": str(per_frame_path),
        "per_frame_csv_sha256": sha256_file(per_frame_path),
        "per_frame_row_count": len(frame_rows),
    }
    write_json(destination / "evaluation_provenance.json", provenance)
    return {**dict(source), **result, **provenance, "output_dir": str(destination)}


def load_resumable_family_evaluation(
    *,
    source: Mapping[str, Any],
    gpu_id: int,
    repeat_index: int,
    output_dir: str | Path,
    eval_manifest_path: str | Path,
    num_frames: int = 1789,
    warmup_frames: int = 200,
    latency_rounds: int = 3,
    fixed_k: int = 29696,
) -> dict[str, Any]:
    """Load one complete result or fail closed on any partial/stale directory."""

    destination = Path(output_dir).resolve()
    required = {
        "evaluation": destination / "evaluation.json",
        "provenance": destination / "evaluation_provenance.json",
        "cleanup": destination / "gpu_cleanup_audit.json",
        "per_frame": destination / "per_frame_latency.csv",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise RuntimeError(
            f"family_resume_incomplete_evaluation:{destination}:missing={missing}"
        )
    manifest = read_json(eval_manifest_path)
    manifest_hash = str(manifest.get("manifest_hash", ""))
    evaluation = read_json(required["evaluation"])
    provenance = read_json(required["provenance"])
    cleanup = read_json(required["cleanup"])
    expected_identity = {
        "schema_version": "heal-lidar-family-fair-evaluation-v1",
        "family_id": str(source["family_id"]),
        "assigned_method": str(source["assigned_method"]),
        "repeat_index": int(repeat_index),
        "gpu_id": int(gpu_id),
        "sequence_index": int(source["sequence_index"]),
        "item_id": str(source["item_id"]),
        "variant": str(source["variant"]),
        "engine_sha256": str(source["engine_sha256"]),
        "eval_manifest_hash": manifest_hash,
        "num_frames": int(num_frames),
        "warmup_frames": int(warmup_frames),
        "latency_rounds": int(latency_rounds),
        "dataloader_num_workers": 8,
    }
    mismatches = {
        key: {"expected": expected, "actual": provenance.get(key)}
        for key, expected in expected_identity.items()
        if provenance.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(
            f"family_resume_identity_mismatch:{destination}:{mismatches}"
        )
    if not bool(provenance.get("source_engine_unchanged", False)):
        raise RuntimeError(f"family_resume_source_engine_not_locked:{destination}")
    if bool(provenance.get("engine_build_invoked", True)):
        raise RuntimeError(f"family_resume_contains_engine_build:{destination}")
    source_path = Path(str(source["engine_path"])).resolve()
    if Path(str(provenance.get("engine_path", ""))).resolve() != source_path:
        raise RuntimeError(f"family_resume_engine_path_mismatch:{destination}")
    verify_engine_lock(provenance)
    if not bool(cleanup.get("passed", False)):
        raise RuntimeError(f"family_resume_cleanup_not_accepted:{destination}")
    validate_family_evaluation_result(
        evaluation,
        num_frames=num_frames,
        warmup_frames=warmup_frames,
        manifest_hash=manifest_hash,
        fixed_k=fixed_k,
    )
    expected_csv_hash = str(provenance.get("per_frame_csv_sha256", ""))
    actual_csv_hash = sha256_file(required["per_frame"])
    if not expected_csv_hash or actual_csv_hash != expected_csv_hash:
        raise RuntimeError(
            f"family_resume_per_frame_csv_hash_mismatch:{destination}:"
            f"{expected_csv_hash}:{actual_csv_hash}"
        )
    with required["per_frame"].open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != int(num_frames):
        raise RuntimeError(
            f"family_resume_per_frame_csv_count:{destination}:{len(rows)}:{num_frames}"
        )
    csv_identity_fields = {
        "family_id": str(source["family_id"]),
        "assigned_method": str(source["assigned_method"]),
        "repeat_index": str(int(repeat_index)),
        "gpu_id": str(int(gpu_id)),
        "item_id": str(source["item_id"]),
        "variant": str(source["variant"]),
        "engine_sha256": str(source["engine_sha256"]),
    }
    bad_rows = [
        index
        for index, row in enumerate(rows)
        if any(str(row.get(key, "")) != value for key, value in csv_identity_fields.items())
    ]
    if bad_rows:
        raise RuntimeError(
            f"family_resume_per_frame_csv_identity:{destination}:{bad_rows[:8]}"
        )
    _ensure_evaluation_only_output(destination)
    return {
        **dict(source),
        **evaluation,
        **provenance,
        "gpu_cleanup": cleanup,
        "output_dir": str(destination),
        "resume_hit": True,
    }


def frame_order_hash(result: Mapping[str, Any]) -> str:
    existing = str(result.get("frame_order_hash", ""))
    if existing:
        return existing
    return canonical_json_hash(list(result.get("evaluated_frame_ids") or []))


def compact_result(result: Mapping[str, Any]) -> dict[str, Any]:
    compact = {
        key: value
        for key, value in result.items()
        if key
        not in {
            "latency_rows",
            "warmup_frame_ids",
            "evaluated_frame_ids",
            "skipped_frame_ids",
        }
    }
    compact["frame_order_hash"] = frame_order_hash(result)
    compact["latency_row_count"] = len(list(result.get("latency_rows") or []))
    return compact


def summarize_repeat_results(
    results: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    values = [dict(row) for row in results]
    baselines = {
        (str(row["assigned_method"]), int(row["repeat_index"])): float(
            row["forward_p50_ms"]
        )
        for row in values
        if str(row["variant"]) == "fp32"
    }
    output: list[dict[str, Any]] = []
    for row in values:
        key = (str(row["assigned_method"]), int(row["repeat_index"]))
        baseline = baselines.get(key)
        if baseline is None:
            raise RuntimeError(f"family_same_gpu_fp32_baseline_missing:{key}")
        p50 = float(row["forward_p50_ms"])
        output.append(
            {
                "family_id": row["family_id"],
                "assigned_method": row["assigned_method"],
                "gpu_id": int(row["gpu_id"]),
                "repeat_index": int(row["repeat_index"]),
                "sequence_index": int(row["sequence_index"]),
                "item_id": row["item_id"],
                "variant": row["variant"],
                "budget": row.get("budget"),
                "actual_bops": row.get("actual_bops"),
                "parameter_count_base": row.get("parameter_count_base"),
                "parameter_count_pruned": row.get("parameter_count_pruned"),
                "parameter_reduction": row.get("parameter_reduction"),
                "int8_count": row.get("int8_count"),
                "fp16_count": row.get("fp16_count"),
                "fp32_count": row.get("fp32_count"),
                "AP@0.3": row["AP@0.3"],
                "AP@0.5": row["AP@0.5"],
                "AP@0.7": row["AP@0.7"],
                "mAP": row["mAP"],
                "forward_mean_ms": row["forward_mean_ms"],
                "forward_p50_ms": p50,
                "forward_p90_ms": row["forward_p90_ms"],
                "forward_p99_ms": row["forward_p99_ms"],
                "postprocess_mean_ms": row["postprocess_mean_ms"],
                "postprocess_p50_ms": row["postprocess_p50_ms"],
                "postprocess_p90_ms": row["postprocess_p90_ms"],
                "postprocess_p99_ms": row["postprocess_p99_ms"],
                "total_mean_ms": row["total_mean_ms"],
                "total_p50_ms": row["total_p50_ms"],
                "total_p90_ms": row["total_p90_ms"],
                "total_p99_ms": row["total_p99_ms"],
                "speedup_vs_same_gpu_fp32": baseline / p50,
                "num_evaluated_frames": row["num_evaluated_frames"],
                "num_skipped_frames": row["num_skipped_frames"],
                "frame_order_hash": frame_order_hash(row),
                "engine_sha256": row["engine_sha256"],
                "source_engine_unchanged": row["source_engine_unchanged"],
                "cache_cleanup_passed": bool(
                    dict(row.get("gpu_cleanup") or {}).get("passed", False)
                ),
            }
        )
    return sorted(
        output,
        key=lambda row: (
            str(row["assigned_method"]),
            int(row["repeat_index"]),
            int(row["sequence_index"]),
        ),
    )


def aggregate_five_repeat_results(
    rows: Iterable[Mapping[str, Any]], *, repeat_count: int = 5
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, float | None], list[dict[str, Any]]] = {}
    for raw in rows:
        row = dict(raw)
        budget = row.get("budget")
        key = (
            str(row["assigned_method"]),
            str(row["variant"]),
            None if budget is None else _float_budget(budget),
        )
        grouped.setdefault(key, []).append(row)
    output: list[dict[str, Any]] = []
    for (method, variant, budget), values in grouped.items():
        repeats = sorted(int(row["repeat_index"]) for row in values)
        if repeats != list(range(int(repeat_count))) or len(values) != int(repeat_count):
            raise RuntimeError(
                f"family_repeat_count_mismatch:{method}:{variant}:{budget}:{repeats}"
            )
        if len({str(row["engine_sha256"]) for row in values}) != 1:
            raise RuntimeError(
                f"family_repeat_engine_identity_mismatch:{method}:{variant}:{budget}"
            )
        if len({str(row["frame_order_hash"]) for row in values}) != 1:
            raise RuntimeError(
                f"family_repeat_frame_order_mismatch:{method}:{variant}:{budget}"
            )
        first = values[0]
        result: dict[str, Any] = {
            "family_id": first["family_id"],
            "assigned_method": method,
            "gpu_id": int(first["gpu_id"]),
            "variant": variant,
            "budget": budget,
            "actual_bops": first.get("actual_bops"),
            "repeat_count": int(repeat_count),
            "engine_sha256": first["engine_sha256"],
            "frame_order_hash": first["frame_order_hash"],
            "parameter_count_base": first.get("parameter_count_base"),
            "parameter_count_pruned": first.get("parameter_count_pruned"),
            "parameter_reduction": first.get("parameter_reduction"),
            "int8_count": first.get("int8_count"),
            "fp16_count": first.get("fp16_count"),
            "fp32_count": first.get("fp32_count"),
        }
        for metric in NUMERIC_METRICS:
            metric_values = [float(row[metric]) for row in values]
            result[f"{metric}_across_runs_mean"] = statistics.fmean(metric_values)
            result[f"{metric}_across_runs_std"] = statistics.pstdev(metric_values)
        output.append(result)
    variant_order = {name: index for index, name in enumerate(("fp32", *VARIANTS))}
    return sorted(
        output,
        key=lambda row: (
            str(row["assigned_method"]),
            0 if row["budget"] is None else 1,
            0.0 if row["budget"] is None else -float(row["budget"]),
            variant_order[str(row["variant"])],
        ),
    )


def ablation_contribution_rows(
    rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Decompose accuracy and latency for every method/repeat/budget."""

    values = [dict(row) for row in rows]
    baselines = {
        (str(row["assigned_method"]), int(row["repeat_index"])): row
        for row in values
        if str(row["variant"]) == "fp32"
    }
    grouped: dict[tuple[str, int, float], dict[str, dict[str, Any]]] = {}
    for row in values:
        if str(row["variant"]) == "fp32":
            continue
        key = (
            str(row["assigned_method"]),
            int(row["repeat_index"]),
            _float_budget(row["budget"]),
        )
        grouped.setdefault(key, {})[str(row["variant"])] = row
    output: list[dict[str, Any]] = []
    for (method, repeat, budget), variants in grouped.items():
        if set(variants) != set(VARIANTS):
            raise RuntimeError(
                f"family_ablation_variants_missing:{method}:{repeat}:{budget}:{sorted(variants)}"
            )
        baseline = baselines[(method, repeat)]
        pq, p_only, q_only = (
            variants["prune_quant"],
            variants["prune_only"],
            variants["quant_only"],
        )
        fp32_map = float(baseline["mAP"])
        output.append(
            {
                "family_id": baseline["family_id"],
                "assigned_method": method,
                "gpu_id": int(baseline["gpu_id"]),
                "repeat_index": repeat,
                "budget": budget,
                "actual_bops": pq.get("actual_bops"),
                "fp32_mAP": fp32_map,
                "prune_quant_mAP": float(pq["mAP"]),
                "prune_only_mAP": float(p_only["mAP"]),
                "quant_only_mAP": float(q_only["mAP"]),
                "prune_quant_delta_vs_fp32": float(pq["mAP"]) - fp32_map,
                "prune_only_delta_vs_fp32": float(p_only["mAP"]) - fp32_map,
                "quant_only_delta_vs_fp32": float(q_only["mAP"]) - fp32_map,
                "pq_interaction_mAP": (
                    float(pq["mAP"])
                    - float(p_only["mAP"])
                    - float(q_only["mAP"])
                    + fp32_map
                ),
                "fp32_forward_p50_ms": float(baseline["forward_p50_ms"]),
                "prune_quant_forward_p50_ms": float(pq["forward_p50_ms"]),
                "prune_only_forward_p50_ms": float(p_only["forward_p50_ms"]),
                "quant_only_forward_p50_ms": float(q_only["forward_p50_ms"]),
                "prune_quant_speedup": float(pq["speedup_vs_same_gpu_fp32"]),
                "prune_only_speedup": float(p_only["speedup_vs_same_gpu_fp32"]),
                "quant_only_speedup": float(q_only["speedup_vs_same_gpu_fp32"]),
            }
        )
    return sorted(output, key=lambda row: (row["assigned_method"], row["repeat_index"], -row["budget"]))


def aggregate_contribution_rows(
    rows: Iterable[Mapping[str, Any]], *, repeat_count: int = 5
) -> list[dict[str, Any]]:
    values = [dict(row) for row in rows]
    grouped: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for row in values:
        grouped.setdefault(
            (str(row["assigned_method"]), _float_budget(row["budget"])), []
        ).append(row)
    output: list[dict[str, Any]] = []
    identity = {
        "family_id",
        "assigned_method",
        "gpu_id",
        "repeat_index",
        "budget",
        "actual_bops",
    }
    for (method, budget), group in grouped.items():
        repeats = sorted(int(row["repeat_index"]) for row in group)
        if repeats != list(range(int(repeat_count))):
            raise RuntimeError(
                f"family_contribution_repeat_count:{method}:{budget}:{repeats}"
            )
        first = group[0]
        result: dict[str, Any] = {
            "family_id": first["family_id"],
            "assigned_method": method,
            "gpu_id": int(first["gpu_id"]),
            "budget": budget,
            "actual_bops": first.get("actual_bops"),
            "repeat_count": int(repeat_count),
        }
        numeric_fields = [key for key in first if key not in identity]
        for field in numeric_fields:
            field_values = [float(row[field]) for row in group]
            result[f"{field}_across_runs_mean"] = statistics.fmean(field_values)
            result[f"{field}_across_runs_std"] = statistics.pstdev(field_values)
        output.append(result)
    return sorted(output, key=lambda row: (row["assigned_method"], -row["budget"]))


__all__ = [
    "BUDGETS",
    "METHODS",
    "VARIANTS",
    "ablation_contribution_rows",
    "aggregate_contribution_rows",
    "aggregate_five_repeat_results",
    "build_family_evaluation_inventories",
    "canonical_json_hash",
    "clear_gpu_cache_and_wait",
    "compact_result",
    "evaluate_existing_family_engine",
    "evaluation_frame_latency_rows",
    "gpu_snapshot",
    "lock_engine",
    "load_resumable_family_evaluation",
    "read_json",
    "sha256_file",
    "summarize_repeat_results",
    "validate_family_evaluation_result",
    "verify_engine_lock",
    "write_csv",
    "write_json",
]
