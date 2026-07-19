"""Evaluation-only fairness audit for existing TensorRT engines.

This module deliberately has no engine-build imports.  It validates immutable
source engines, evaluates them in fresh subprocesses, and verifies that GPU
memory returns to its pre-evaluation level between samples.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Iterable, Mapping

from .evaluation_provider import evaluate_engine_modelopt
from .runtime_environment import modelopt_python_command


ACCEPTANCE_REPORTS = (
    "physical_validation.json",
    "physical_plan_validation.json",
    "engine_structure_validation.json",
    "precision_realization_validation.json",
    "merge_precision_realization.json",
    "production_qdq_boundary_audit.json",
)
FORBIDDEN_EVALUATION_ARTIFACT_SUFFIXES = (".plan", ".engine", ".onnx", ".pth")
METHOD_ORDER = {"fp32": 0, "ga": 1, "greedy": 2}
ABLATION_VARIANT_ORDER = {"prune_quant": 0, "prune_only": 1, "quant_only": 2}


def read_json(path: str | Path) -> dict[str, Any]:
    return dict(json.loads(Path(path).read_text(encoding="utf-8")))


def write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    temporary.replace(destination)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _artifact_dir_for_candidate(
    candidate: Mapping[str, Any], *, ablation_root: Path
) -> Path:
    source = str(candidate.get("source_artifact_dir", "")).strip()
    if source:
        return Path(source).resolve()
    method = str(candidate["method"])
    budget = float(candidate["budget"])
    if method == "greedy" and abs(budget - 0.30) < 1.0e-9:
        return (
            ablation_root
            / "candidates"
            / "greedy"
            / "bops_0.30"
            / "prune_quant"
        ).resolve()
    raise RuntimeError(f"candidate_has_no_existing_artifact:{method}:{budget:.2f}")


def _candidate_sort_key(row: Mapping[str, Any]) -> tuple[int, float]:
    return METHOD_ORDER[str(row["method"])], -float(row.get("budget", 0.0))


def build_evaluation_inventory(
    *,
    ablation_root: str | Path,
    baseline_artifact_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Build the fixed FP32 -> GA -> greedy evaluation order.

    The ablation manifest is used only to identify already accepted winners;
    no candidate reconstruction or engine construction is performed here.
    """

    root = Path(ablation_root).resolve()
    manifest = read_json(root / "ablation_manifest.json")
    baseline = (
        Path(baseline_artifact_dir).resolve()
        if baseline_artifact_dir is not None
        else (root / "full_validation/baselines/original_strict_fp32").resolve()
    )
    rows: list[dict[str, Any]] = [
        {
            "sequence_index": 0,
            "item_id": "fp32_original",
            "method": "fp32",
            "budget": None,
            "actual_bops": 1.0,
            "artifact_dir": str(baseline),
            "source_candidate_hash": "original_strict_fp32",
        }
    ]
    candidates = sorted(manifest.get("candidates", []), key=_candidate_sort_key)
    expected = {
        (method, round(budget, 2))
        for method in ("ga", "greedy")
        for budget in (0.30, 0.25, 0.20, 0.15, 0.10, 0.05)
    }
    actual = {
        (str(row["method"]), round(float(row["budget"]), 2)) for row in candidates
    }
    if actual != expected:
        raise RuntimeError(
            f"winner_inventory_mismatch:missing={sorted(expected-actual)}:extra={sorted(actual-expected)}"
        )
    for index, candidate in enumerate(candidates, start=1):
        method = str(candidate["method"])
        budget = float(candidate["budget"])
        artifact_dir = _artifact_dir_for_candidate(candidate, ablation_root=root)
        rows.append(
            {
                "sequence_index": index,
                "item_id": f"{method}_bops_{budget:.2f}",
                "method": method,
                "budget": budget,
                "actual_bops": float(candidate["actual_bops"]),
                "artifact_dir": str(artifact_dir),
                "source_candidate_hash": str(candidate.get("candidate_hash", "")),
            }
        )
    return rows


def build_method_ablation_inventory(
    *, ablation_root: str | Path, method: str
) -> list[dict[str, Any]]:
    """Load FP32 plus the 18 P+Q/P-only/Q-only rows for one search method."""

    root = Path(ablation_root).resolve()
    method_name = str(method).lower()
    if method_name not in {"ga", "greedy"}:
        raise ValueError(f"unsupported_ablation_method:{method}")
    baseline = (root / "full_validation/baselines/original_strict_fp32").resolve()
    result_payload = read_json(root / "ablation_results.json")
    candidates = [
        dict(row)
        for row in result_payload.get("rows", [])
        if str(row.get("method", "")).lower() == method_name
    ]
    expected = {
        (round(budget, 2), variant)
        for budget in (0.30, 0.25, 0.20, 0.15, 0.10, 0.05)
        for variant in ABLATION_VARIANT_ORDER
    }
    actual = {
        (round(float(row["budget"]), 2), str(row["variant"])) for row in candidates
    }
    if actual != expected:
        raise RuntimeError(
            f"ablation_inventory_mismatch:{method_name}:missing={sorted(expected-actual)}:extra={sorted(actual-expected)}"
        )
    candidates.sort(
        key=lambda row: (
            -float(row["budget"]),
            ABLATION_VARIANT_ORDER[str(row["variant"])],
        )
    )
    rows: list[dict[str, Any]] = [
        {
            "sequence_index": 0,
            "item_id": f"{method_name}_fp32_original",
            "method": "fp32",
            "assigned_method": method_name,
            "variant": "fp32",
            "budget": None,
            "actual_bops": 1.0,
            "artifact_dir": str(baseline),
            "source_candidate_hash": "original_strict_fp32",
            "parameter_count_base": None,
            "parameter_count_pruned": None,
            "parameter_reduction": 0.0,
        }
    ]
    for index, candidate in enumerate(candidates, start=1):
        variant = str(candidate["variant"])
        source_dir = str(candidate.get("source_artifact_dir", "")).strip()
        artifact_dir = Path(str(candidate["artifact_dir"])).resolve()
        if variant == "prune_quant" and source_dir:
            artifact_dir = Path(source_dir).resolve()
        rows.append(
            {
                "sequence_index": index,
                "item_id": (
                    f"{method_name}_bops_{float(candidate['budget']):.2f}_{variant}"
                ),
                "method": method_name,
                "assigned_method": method_name,
                "variant": variant,
                "budget": float(candidate["budget"]),
                "actual_bops": float(candidate["actual_bops"]),
                "artifact_dir": str(artifact_dir),
                "source_candidate_hash": str(
                    candidate.get("source_candidate_hash", "")
                ),
                "source_row_id": str(candidate.get("row_id", "")),
                "parameter_count_base": candidate.get("parameter_count_base"),
                "parameter_count_pruned": candidate.get("parameter_count_pruned"),
                "parameter_reduction": candidate.get("parameter_reduction"),
            }
        )
    return rows


def _report_passed(path: Path) -> bool:
    payload = read_json(path)
    return bool(payload.get("passed", False))


def _precision_counts(artifact_dir: Path) -> dict[str, int]:
    path = artifact_dir / "realized_precision_profile.json"
    if not path.is_file():
        return {"int8_count": 0, "fp16_count": 0, "fp32_count": 0}
    values = [str(value).upper() for value in read_json(path).values()]
    return {
        "int8_count": sum(value == "INT8" for value in values),
        "fp16_count": sum(value == "FP16" for value in values),
        "fp32_count": sum(value == "FP32" for value in values),
    }


def verify_existing_engine(row: Mapping[str, Any]) -> dict[str, Any]:
    """Verify that a source engine is complete, accepted, and hash-addressed."""

    artifact_dir = Path(str(row["artifact_dir"])).resolve()
    engine = artifact_dir / "engine.plan"
    deployment = artifact_dir / "deployment_manifest.json"
    missing = [
        str(path)
        for path in (engine, deployment, *(artifact_dir / name for name in ACCEPTANCE_REPORTS))
        if not path.is_file()
    ]
    if missing:
        raise RuntimeError(f"evaluation_source_artifact_missing:{missing}")
    failed = [
        name for name in ACCEPTANCE_REPORTS if not _report_passed(artifact_dir / name)
    ]
    if failed:
        raise RuntimeError(f"evaluation_source_acceptance_failed:{artifact_dir}:{failed}")
    stat = engine.stat()
    if stat.st_size <= 0:
        raise RuntimeError(f"evaluation_source_engine_empty:{engine}")
    digest = sha256_file(engine)
    manifest = read_json(deployment)
    expected_hash = str(manifest.get("engine_hash", ""))
    if not expected_hash:
        raise RuntimeError(f"evaluation_source_engine_hash_missing:{deployment}")
    if digest != expected_hash:
        raise RuntimeError(
            f"evaluation_source_engine_hash_mismatch:{engine}:{expected_hash}:{digest}"
        )
    return {
        **dict(row),
        "engine_path": str(engine),
        "engine_sha256": digest,
        "engine_size_bytes": stat.st_size,
        "engine_mtime_ns": stat.st_mtime_ns,
        "deployment_hash": str(manifest.get("deployment_hash", "")),
        "physical_hash": str(manifest.get("physical_hash", "")),
        **_precision_counts(artifact_dir),
        "acceptance_reports": {name: True for name in ACCEPTANCE_REPORTS},
    }


def verify_engine_unchanged(provenance: Mapping[str, Any]) -> None:
    engine = Path(str(provenance["engine_path"]))
    stat = engine.stat()
    if stat.st_size != int(provenance["engine_size_bytes"]):
        raise RuntimeError(f"source_engine_size_changed_during_evaluation:{engine}")
    if stat.st_mtime_ns != int(provenance["engine_mtime_ns"]):
        raise RuntimeError(f"source_engine_mtime_changed_during_evaluation:{engine}")
    if sha256_file(engine) != str(provenance["engine_sha256"]):
        raise RuntimeError(f"source_engine_hash_changed_during_evaluation:{engine}")


def gpu_snapshot(gpu_id: int) -> dict[str, Any]:
    command = [
        "nvidia-smi",
        f"--id={int(gpu_id)}",
        "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(
        command, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    if completed.returncode != 0:
        raise RuntimeError(f"nvidia_smi_failed:{completed.stderr.strip()}")
    fields = [value.strip() for value in completed.stdout.strip().split(",")]
    if len(fields) != 7:
        raise RuntimeError(f"nvidia_smi_unexpected_output:{completed.stdout!r}")
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
    """Run explicit CUDA cleanup after the evaluation worker has exited.

    Process exit is the operation that destroys the TensorRT/PyTorch CUDA
    context.  The short cleanup subprocess is an additional allocator flush;
    the nvidia-smi poll proves that no allocation leaked across evaluations.
    """

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(int(gpu_id))
    code = (
        "import torch; "
        "torch.cuda.set_device(0); "
        "torch.cuda.synchronize(); "
        "torch.cuda.empty_cache(); "
        "torch.cuda.ipc_collect(); "
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
    deadline = time.monotonic() + float(timeout_seconds)
    threshold = int(before["memory_used_mib"]) + int(allowed_residual_mib)
    snapshots: list[dict[str, Any]] = []
    while True:
        current = gpu_snapshot(gpu_id)
        snapshots.append(current)
        if current["memory_used_mib"] <= threshold:
            break
        if time.monotonic() >= deadline:
            break
        time.sleep(0.5)
    final = snapshots[-1]
    passed = completed.returncode == 0 and final["memory_used_mib"] <= threshold
    return {
        "passed": passed,
        "cleanup_returncode": completed.returncode,
        "cleanup_output": completed.stdout.strip(),
        "process_exit_released_context": True,
        "before_memory_used_mib": int(before["memory_used_mib"]),
        "allowed_residual_mib": int(allowed_residual_mib),
        "threshold_memory_used_mib": threshold,
        "after": final,
        "poll_count": len(snapshots),
    }


def validate_evaluation_result(
    result: Mapping[str, Any], *, num_frames: int, warmup_frames: int, manifest_hash: str
) -> None:
    errors: list[str] = []
    if result.get("status") != "ok":
        errors.append(f"status={result.get('status')}:{result.get('failure_reason', '')}")
    if int(result.get("num_evaluated_frames", -1)) != int(num_frames):
        errors.append(f"evaluated={result.get('num_evaluated_frames')}")
    if int(result.get("num_skipped_frames", -1)) != 0:
        errors.append(f"skipped={result.get('num_skipped_frames')}")
    if int(result.get("dataloader_num_workers", -1)) != 8:
        errors.append(f"workers={result.get('dataloader_num_workers')}")
    if not bool(result.get("reset_after_warmup", False)):
        errors.append("warmup_not_reset")
    if str(result.get("eval_manifest_hash", "")) != str(manifest_hash):
        errors.append(f"manifest_hash={result.get('eval_manifest_hash')}")
    cuda_audit = dict(result.get("cuda_postprocess_audit") or {})
    if not bool(cuda_audit.get("passed", False)):
        errors.append("cuda_postprocess_not_passed")
    latency_rows = list(result.get("latency_rows") or [])
    warmup_rows = sum(bool(row.get("warmup", False)) for row in latency_rows)
    if warmup_rows != int(warmup_frames):
        errors.append(f"warmup_rows={warmup_rows}")
    if errors:
        raise RuntimeError(f"fair_evaluation_protocol_failed:{errors}")


def ensure_evaluation_only_output(output_dir: str | Path) -> None:
    unexpected = [
        str(path)
        for path in Path(output_dir).rglob("*")
        if path.is_file() and path.suffix.lower() in FORBIDDEN_EVALUATION_ARTIFACT_SUFFIXES
    ]
    if unexpected:
        raise RuntimeError(f"evaluation_output_contains_build_artifact:{unexpected}")


def run_one_evaluation(
    *,
    source: Mapping[str, Any],
    gpu_id: int,
    output_dir: str | Path,
    checkpoint: str | Path,
    model_config: str | Path,
    heal_root: str | Path,
    tensorrt_root: str | Path,
    plugin_path: str | Path,
    eval_manifest_path: str | Path,
    num_frames: int = 1789,
    warmup_frames: int = 200,
    latency_rounds: int = 3,
    fixed_k: int = 29696,
    conda_env: str = "modelopt",
) -> dict[str, Any]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    manifest = read_json(eval_manifest_path)
    manifest_hash = str(manifest.get("manifest_hash", ""))
    if not manifest_hash:
        raise RuntimeError(f"eval_manifest_hash_missing:{eval_manifest_path}")
    before = gpu_snapshot(gpu_id)
    write_json(destination / "gpu_before.json", before)
    started_at = time.time()
    evaluation_error: BaseException | None = None
    result: dict[str, Any] = {}
    try:
        result = evaluate_engine_modelopt(
            engine_path=source["engine_path"],
            checkpoint=checkpoint,
            model_config=model_config,
            heal_root=heal_root,
            device=f"cuda:{int(gpu_id)}",
            output_dir=destination,
            tensorrt_root=tensorrt_root,
            plugin_path=plugin_path,
            num_frames=num_frames,
            warmup_frames=warmup_frames,
            fixed_k=fixed_k,
            latency_rounds=latency_rounds,
            conda_env=conda_env,
            eval_manifest_path=eval_manifest_path,
            ap_iou_backend="gpu",
            require_cuda_postprocess=True,
            torch_num_threads=4,
            dataloader_num_workers=8,
        )
    except BaseException as exc:  # guarantee cleanup before propagating
        evaluation_error = exc
    finished_at = time.time()
    cleanup = clear_gpu_cache_and_wait(
        gpu_id=gpu_id, before=before, conda_env=conda_env
    )
    write_json(destination / "gpu_cleanup_audit.json", cleanup)
    if evaluation_error is not None:
        raise evaluation_error
    if not cleanup["passed"]:
        raise RuntimeError(f"gpu_cache_cleanup_failed:gpu={gpu_id}:{cleanup}")
    validate_evaluation_result(
        result,
        num_frames=num_frames,
        warmup_frames=warmup_frames,
        manifest_hash=manifest_hash,
    )
    verify_engine_unchanged(source)
    ensure_evaluation_only_output(destination)
    provenance = {
        "schema_version": "h800-evaluation-only-provenance-v1",
        "gpu_id": int(gpu_id),
        "item_id": source["item_id"],
        "engine_path": source["engine_path"],
        "engine_sha256": source["engine_sha256"],
        "source_engine_unchanged": True,
        "engine_build_invoked": False,
        "eval_manifest_path": str(Path(eval_manifest_path).resolve()),
        "eval_manifest_hash": manifest_hash,
        "fixed_k": int(fixed_k),
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
    }
    write_json(destination / "evaluation_provenance.json", provenance)
    return {**dict(source), **dict(result), "output_dir": str(destination), **provenance}


def frame_order_hash(result: Mapping[str, Any]) -> str:
    return canonical_json_hash(list(result.get("evaluated_frame_ids") or []))


def summarize_results(results: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = [dict(row) for row in results]
    baselines = {
        int(row["gpu_id"]): float(row["forward_p50_ms"])
        for row in rows
        if row["method"] == "fp32"
    }
    summary: list[dict[str, Any]] = []
    for row in rows:
        p50 = float(row["forward_p50_ms"])
        baseline = baselines[int(row["gpu_id"])]
        summary.append(
            {
                "gpu_id": int(row["gpu_id"]),
                "sequence_index": int(row["sequence_index"]),
                "item_id": row["item_id"],
                "method": row["method"],
                "assigned_method": row.get("assigned_method", row["method"]),
                "variant": row.get("variant", "prune_quant"),
                "budget": row.get("budget"),
                "actual_bops": row.get("actual_bops"),
                "parameter_count_base": row.get("parameter_count_base"),
                "parameter_count_pruned": row.get("parameter_count_pruned"),
                "parameter_reduction": row.get("parameter_reduction"),
                "AP@0.3": row.get("AP@0.3"),
                "AP@0.5": row.get("AP@0.5"),
                "AP@0.7": row.get("AP@0.7"),
                "mAP": row.get("mAP"),
                "forward_p50_ms": p50,
                "forward_p90_ms": row.get("forward_p90_ms"),
                "forward_p99_ms": row.get("forward_p99_ms"),
                "speedup_vs_same_gpu_fp32": baseline / p50,
                "int8_count": row.get("int8_count"),
                "fp16_count": row.get("fp16_count"),
                "fp32_count": row.get("fp32_count"),
                "evaluated_frames": row.get("num_evaluated_frames"),
                "skipped_frames": row.get("num_skipped_frames"),
                "frame_order_hash": frame_order_hash(row),
                "engine_sha256": row.get("engine_sha256"),
                "cache_cleanup_passed": bool(
                    dict(row.get("gpu_cleanup") or {}).get("passed", False)
                ),
                "engine_build_invoked": False,
            }
        )
    return summary


def ablation_contribution_rows(
    rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Compute P/Q accuracy contributions within each assigned GPU baseline."""

    values = [dict(row) for row in rows]
    baseline_by_method = {
        str(row["assigned_method"]): row
        for row in values
        if str(row["variant"]) == "fp32"
    }
    grouped: dict[tuple[str, float], dict[str, dict[str, Any]]] = {}
    for row in values:
        if str(row["variant"]) == "fp32":
            continue
        key = (str(row["assigned_method"]), round(float(row["budget"]), 2))
        grouped.setdefault(key, {})[str(row["variant"])] = row
    result: list[dict[str, Any]] = []
    for (method, budget), variants in sorted(
        grouped.items(), key=lambda item: (METHOD_ORDER[item[0][0]], -item[0][1])
    ):
        if set(variants) != set(ABLATION_VARIANT_ORDER):
            raise RuntimeError(
                f"ablation_variants_missing:{method}:{budget}:{sorted(variants)}"
            )
        baseline = baseline_by_method[method]
        p_q = variants["prune_quant"]
        p_only = variants["prune_only"]
        q_only = variants["quant_only"]
        fp32_map = float(baseline["mAP"])
        result.append(
            {
                "assigned_method": method,
                "gpu_id": int(p_q["gpu_id"]),
                "budget": budget,
                "actual_bops": float(p_q["actual_bops"]),
                "fp32_mAP": fp32_map,
                "prune_quant_mAP": float(p_q["mAP"]),
                "prune_only_mAP": float(p_only["mAP"]),
                "quant_only_mAP": float(q_only["mAP"]),
                "prune_quant_delta_vs_fp32": float(p_q["mAP"]) - fp32_map,
                "prune_only_delta_vs_fp32": float(p_only["mAP"]) - fp32_map,
                "quant_only_delta_vs_fp32": float(q_only["mAP"]) - fp32_map,
                "pq_interaction_mAP": (
                    float(p_q["mAP"])
                    - float(p_only["mAP"])
                    - float(q_only["mAP"])
                    + fp32_map
                ),
                "prune_quant_p50_ms": float(p_q["forward_p50_ms"]),
                "prune_only_p50_ms": float(p_only["forward_p50_ms"]),
                "quant_only_p50_ms": float(q_only["forward_p50_ms"]),
                "prune_quant_speedup": float(p_q["speedup_vs_same_gpu_fp32"]),
                "prune_only_speedup": float(p_only["speedup_vs_same_gpu_fp32"]),
                "quant_only_speedup": float(q_only["speedup_vs_same_gpu_fp32"]),
            }
        )
    return result


def write_summary_csv(path: str | Path, rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("summary rows must not be empty")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def cross_gpu_differences(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Compare identical engines across the two physical GPUs."""

    by_item: dict[str, dict[int, Mapping[str, Any]]] = {}
    for row in rows:
        by_item.setdefault(str(row["item_id"]), {})[int(row["gpu_id"])] = row
    result: list[dict[str, Any]] = []
    for item_id, values in sorted(
        by_item.items(), key=lambda item: int(item[1].get(0, item[1].get(1, {})).get("sequence_index", 0))
    ):
        if set(values) != {0, 1}:
            raise RuntimeError(f"cross_gpu_result_missing:{item_id}:{sorted(values)}")
        gpu0, gpu1 = values[0], values[1]
        p50_0 = float(gpu0["forward_p50_ms"])
        p50_1 = float(gpu1["forward_p50_ms"])
        result.append(
            {
                "sequence_index": int(gpu0["sequence_index"]),
                "item_id": item_id,
                "method": gpu0["method"],
                "budget": gpu0.get("budget"),
                "mAP_gpu0": float(gpu0["mAP"]),
                "mAP_gpu1": float(gpu1["mAP"]),
                "mAP_abs_delta": abs(float(gpu0["mAP"]) - float(gpu1["mAP"])),
                "p50_gpu0_ms": p50_0,
                "p50_gpu1_ms": p50_1,
                "p50_abs_delta_ms": abs(p50_0 - p50_1),
                "p50_ratio_gpu1_vs_gpu0": p50_1 / p50_0,
                "speedup_gpu0": float(gpu0["speedup_vs_same_gpu_fp32"]),
                "speedup_gpu1": float(gpu1["speedup_vs_same_gpu_fp32"]),
                "same_engine_hash": str(gpu0["engine_sha256"])
                == str(gpu1["engine_sha256"]),
                "same_frame_order_hash": str(gpu0["frame_order_hash"])
                == str(gpu1["frame_order_hash"]),
            }
        )
    return result
