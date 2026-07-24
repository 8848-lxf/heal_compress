"""Execute the Transformer head-alignment matrix on the RTX 4090 host.

The H800 experiment implementation remains unchanged.  This adapter removes
physical-structure aliases before dispatch and records every alias so reports
can still distinguish the requested single-family and joint candidates.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import traceback
from typing import Any, Iterable, Mapping, Sequence

from search.orchestration.lidar_transformer_dh_power_alignment_4090 import (
    Runtime4090Paths,
    configure_4090_runtime,
)


def unique_structure_queue(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return the first candidate for each physical structure signature."""

    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source in rows:
        row = dict(source)
        signature = str(row.get("structure_signature", ""))
        if not signature:
            raise ValueError("candidate_structure_signature_missing")
        if signature in seen:
            continue
        seen.add(signature)
        unique.append(row)
    return unique


def candidate_alias_map(rows: Iterable[Mapping[str, Any]]) -> dict[str, list[str]]:
    """Map each canonical physical candidate to all requested candidate IDs."""

    materialized = [dict(row) for row in rows]
    canonical_by_signature = {
        str(row["structure_signature"]): str(row["candidate_id"])
        for row in unique_structure_queue(materialized)
    }
    aliases: dict[str, list[str]] = defaultdict(list)
    for row in materialized:
        signature = str(row.get("structure_signature", ""))
        candidate_id = str(row.get("candidate_id", ""))
        if not candidate_id:
            raise ValueError("candidate_id_missing")
        aliases[canonical_by_signature[signature]].append(candidate_id)
    return {candidate: sorted(set(values)) for candidate, values in aliases.items()}


def remaining_structure_queue(
    rows: Iterable[Mapping[str, Any]],
    *,
    completed_signatures: set[str],
    gpu_ids: Sequence[int] = (4, 5, 6, 7),
) -> list[dict[str, Any]]:
    """Deduplicate physical structures and assign only incomplete work."""

    owners = tuple(int(gpu) for gpu in gpu_ids)
    if not owners or len(set(owners)) != len(owners):
        raise ValueError("invalid_remaining_gpu_set")
    pending = [
        row
        for row in unique_structure_queue(rows)
        if str(row["structure_signature"]) not in completed_signatures
    ]
    return [
        {
            **row,
            "queue_index": index,
            "physical_gpu": owners[index % len(owners)],
        }
        for index, row in enumerate(pending)
    ]


def priority_aligned_single_family_queue(
    rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Select P16-first single-family structures aligned to at least 8 channels.

    Baselines are first, followed by exact d_h values 8/16/32, then the
    remaining multiple-of-eight ladder.  Four-aligned neighbor controls are
    intentionally deferred until the initial same-profile speed result exists.
    """

    selected: list[dict[str, Any]] = []
    for source in unique_structure_queue(rows):
        row = dict(source)
        kind = str(row.get("structure_kind", ""))
        if kind == "baseline":
            row["priority_tier"] = 0
        elif kind == "single_family" and int(row.get("d_h", 0)) % 8 == 0:
            row["priority_tier"] = 1 if int(row["d_h"]) in {8, 16, 32} else 2
        else:
            continue
        selected.append(row)
    return sorted(
        selected,
        key=lambda row: (
            int(row["priority_tier"]),
            str(row["model"]),
            str(row["candidate_id"]),
        ),
    )


def priority_execution_queue(
    rows: Iterable[Mapping[str, Any]], *, gpu_ids: Sequence[int] = (4, 5, 6, 7)
) -> list[dict[str, Any]]:
    """Assign the priority matrix to physical GPUs in deterministic order."""

    owners = tuple(int(gpu) for gpu in gpu_ids)
    if not owners or len(set(owners)) != len(owners):
        raise ValueError("invalid_priority_gpu_set")
    return [
        {
            **row,
            "queue_index": index,
            "physical_gpu": owners[index % len(owners)],
        }
        for index, row in enumerate(priority_aligned_single_family_queue(rows))
    ]


def worker_queue(
    rows: Iterable[Mapping[str, Any]], physical_gpu: int
) -> list[dict[str, Any]]:
    """Select one worker's ordered queue and fail if the GPU is not assigned."""

    materialized = [dict(row) for row in rows]
    gpu = int(physical_gpu)
    available = {int(row["physical_gpu"]) for row in materialized}
    if gpu not in available:
        raise ValueError(f"gpu_not_in_priority_queue:{gpu}")
    return [row for row in materialized if int(row["physical_gpu"]) == gpu]


def formal_latency_evidence_ready(
    build: Mapping[str, Any],
    fixed500: Mapping[str, Any],
    *,
    profile: str | None = None,
) -> bool:
    """Require exact realized precision and a complete fixed500 evaluation."""

    evidence_ready = (
        str(build.get("status")) == "ok"
        and int(build.get("requested_realized_conflict_count", -1)) == 0
        and str(fixed500.get("status")) == "ok"
        and int(fixed500.get("evaluated", -1)) == 500
        and int(fixed500.get("skipped", -1)) == 0
    )
    if profile is None:
        return evidence_ready
    expected = str(profile)
    return (
        evidence_ready
        and str(build.get("profile")) == expected
        and str(fixed500.get("profile")) == expected
    )


def all_matrix_formal_latency_candidates(
    rows: Iterable[Mapping[str, Any]],
    *,
    profiles: Sequence[str],
    evidence_directory: Any,
    evidence_loader: Any,
    fail_on_missing: bool = True,
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Bind every unique accepted structure to its same-profile timing group."""

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    missing: list[str] = []
    for source in unique_structure_queue(rows):
        row = dict(source)
        model = str(row["model"])
        candidate_id = str(row["candidate_id"])
        for profile in profiles:
            profile_id = str(profile)
            directory = Path(evidence_directory(row, profile_id)).resolve()
            build, fixed = evidence_loader(directory, profile_id)
            if not formal_latency_evidence_ready(build, fixed, profile=profile_id):
                missing.append(f"{candidate_id}:{profile_id}")
                continue
            if not (directory / "engine.plan").is_file():
                missing.append(f"{candidate_id}:{profile_id}:engine_missing")
                continue
            grouped[(model, profile_id)].append(
                {
                    **row,
                    "profile": profile_id,
                    "fixed500_mAP": float(fixed["mAP"]),
                    "structure_hash": str(fixed["structure_hash"]),
                    "engine_sha256": str(build["engine_sha256"]),
                    "engine_directory": str(directory),
                }
            )
    if missing and fail_on_missing:
        raise RuntimeError(f"all_matrix_formal_latency_evidence_incomplete:{missing}")
    return {
        key: sorted(
            values,
            key=lambda row: (
                str(row["candidate_id"]) != f"{key[0]}__B0",
                str(row["candidate_id"]),
            ),
        )
        for key, values in sorted(grouped.items())
    }


def formal_latency_candidate_batches(
    rows: Sequence[Mapping[str, Any]],
    *,
    baseline_id: str,
    maximum_candidates: int = 8,
) -> list[list[dict[str, Any]]]:
    """Bound baseline replay duration while preserving one timing per candidate."""

    limit = int(maximum_candidates)
    if limit < 1:
        raise ValueError("formal_latency_batch_size_must_be_positive")
    materialized = [dict(row) for row in rows]
    baselines = [row for row in materialized if str(row.get("candidate_id")) == baseline_id]
    if len(baselines) != 1:
        raise RuntimeError(f"formal_latency_baseline_count:{baseline_id}:{len(baselines)}")
    candidates = [
        row for row in materialized if str(row.get("candidate_id")) != baseline_id
    ]
    return [
        [dict(baselines[0]), *candidates[offset : offset + limit]]
        for offset in range(0, len(candidates), limit)
    ] or [[dict(baselines[0])]]


def formal_latency_evidence_index(
    rows: Iterable[Mapping[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    """Index candidates once and collapse repeated baseline measurements."""

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for source in rows:
        row = dict(source)
        signature = str(row.get("structure_signature", ""))
        profile = str(row.get("profile", ""))
        if not signature or profile not in {"P32", "P16", "P8"}:
            raise RuntimeError("formal_latency_row_identity_missing")
        grouped[(signature, profile)].append(row)
    result: dict[tuple[str, str], dict[str, Any]] = {}
    percentile_fields = ("p50_ms", "p90_ms", "p95_ms", "p99_ms", "mean_ms", "std_ms")
    for key, values in grouped.items():
        candidates = [row for row in values if not bool(row.get("baseline_replay"))]
        if candidates:
            if len(candidates) != 1:
                raise RuntimeError(f"formal_latency_candidate_duplicate:{key}:{len(candidates)}")
            result[key] = dict(candidates[0])
            continue
        aggregate = dict(values[0])
        for field in percentile_fields:
            present = [float(row[field]) for row in values if row.get(field) is not None]
            if present:
                aggregate[field] = statistics.median(present)
        aggregate["baseline_measurement_count"] = len(values)
        aggregate["baseline_replay"] = True
        result[key] = aggregate
    return result


def formal_latency_batch_result_reusable(
    actual_rows: Sequence[Mapping[str, Any]],
    expected_rows: Sequence[Mapping[str, Any]],
) -> bool:
    """Accept a completed timing batch only when its engine contract is exact."""

    expected = {
        (str(row.get("candidate_id")), str(row.get("profile"))): str(
            row.get("engine_sha256")
        )
        for row in expected_rows
    }
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for source in actual_rows:
        row = dict(source)
        if row.get("formal") is not True:
            return False
        key = (str(row.get("candidate_id")), str(row.get("profile")))
        if key not in expected or str(row.get("engine_sha256")) != expected[key]:
            return False
        counts[key] += 1
    if set(counts) != set(expected):
        return False
    baseline_keys = [
        key for key in expected if key[0].endswith("__B0")
    ]
    if len(baseline_keys) != 1:
        return False
    return all(
        count == (2 if key == baseline_keys[0] else 1)
        for key, count in counts.items()
    )


def fresh_build_repeat_plan(
    *,
    baseline_directory: Path,
    candidate_directory: Path,
    output_directory: Path,
    profile: str,
    repeats: int = 3,
) -> list[dict[str, Any]]:
    """Describe isolated baseline/candidate rebuilds without timing-cache reuse."""

    profile_id = str(profile)
    if profile_id not in {"P32", "P16", "P8"}:
        raise ValueError(f"unsupported_fresh_build_profile:{profile_id}")
    sources = {
        "baseline": Path(baseline_directory).resolve(),
        "candidate": Path(candidate_directory).resolve(),
    }
    mismatched = [
        role for role, directory in sources.items() if directory.name != profile_id
    ]
    if mismatched:
        raise ValueError(
            f"fresh_build_source_profile_mismatch:{profile_id}:{','.join(mismatched)}"
        )
    count = int(repeats)
    if count < 1:
        raise ValueError("fresh_build_repeat_count_must_be_positive")
    root = Path(output_directory).resolve()
    return [
        {
            "repeat_index": repeat_index,
            "role": role,
            "profile": profile_id,
            "source_directory": str(source),
            "output_directory": str(root / f"repeat_{repeat_index}" / role),
            "engine_reused": False,
            "timing_cache_reused": False,
        }
        for repeat_index in range(1, count + 1)
        for role, source in sources.items()
    ]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def execute_fresh_build_plan(
    plan: Iterable[Mapping[str, Any]],
    *,
    physical_gpu: int,
    build_engine_fn: Any | None = None,
) -> list[dict[str, Any]]:
    """Execute fresh TensorRT builds while preserving the accepted ONNX contract."""

    if build_engine_fn is None:
        from search.stage2.trt_modelopt import build_engine_modelopt

        build_engine_fn = build_engine_modelopt
    results: list[dict[str, Any]] = []
    for source_row in plan:
        row = dict(source_row)
        source = Path(str(row["source_directory"])).resolve()
        request_path = source / "engine_build" / "trt_build_request.json"
        if not request_path.is_file():
            raise RuntimeError(f"fresh_build_source_incomplete:{source}")
        request = _read_json(request_path)
        typed = Path(str(request.get("qdq_onnx", ""))).resolve()
        if not typed.is_file() or not typed.is_relative_to(source):
            raise RuntimeError(f"fresh_build_typed_onnx_invalid:{source}:{typed}")
        config = dict(request.get("build_config", {}))
        if any(config.get(key) for key in ("timing_cache", "timing_cache_path", "load_timing_cache")):
            raise RuntimeError(f"fresh_build_timing_cache_forbidden:{source}")
        destination = Path(str(row["output_directory"])).resolve()
        destination.mkdir(parents=True, exist_ok=True)
        engine = destination / "engine.plan"
        build = build_engine_fn(
            qdq_onnx=typed,
            engine_path=engine,
            precision_mapping=request["precision_mapping"],
            build_config=config,
            physical_snapshot=request["physical_snapshot"],
            output_dir=destination / "engine_build",
            tensorrt_root=Path(str(request["tensorrt_root"])).resolve(),
            conda_env="modelopt",
            gpu_id=int(physical_gpu),
        )
        if str(build.get("status")) != "ok" or not engine.is_file():
            raise RuntimeError(
                f"fresh_build_failed:{row['repeat_index']}:{row['role']}:{build.get('failure_reason', build)}"
            )
        result = {
            **row,
            "status": "ok",
            "physical_gpu": int(physical_gpu),
            "typed_onnx_sha256": _sha256_file(typed),
            "engine_sha256": _sha256_file(engine),
            "engine_directory": str(destination),
            "build": build,
        }
        _write_json(destination / "fresh_build_result.json", result)
        results.append(result)
    return results


def fresh_build_latency_candidates(
    repeat_directory: Path,
    *,
    profile: str,
    repeat_index: int,
    candidate_id: str = "C1",
) -> list[dict[str, Any]]:
    """Bind one independent build pair to the formal latency protocol."""

    root = Path(repeat_directory).resolve()
    rows = []
    for role, identity in (("baseline", "baseline"), ("candidate", candidate_id)):
        directory = root / role
        if not (directory / "engine.plan").is_file():
            raise RuntimeError(f"fresh_build_repeat_engine_missing:{directory}")
        rows.append(
            {
                "candidate_id": str(identity),
                "profile": str(profile),
                "repeat_index": int(repeat_index),
                "engine_directory": str(directory),
            }
        )
    return rows


def priority_rows_through_tier(
    rows: Iterable[Mapping[str, Any]], max_priority_tier: int
) -> list[dict[str, Any]]:
    """Select an initial priority prefix without changing queue order."""

    limit = int(max_priority_tier)
    if limit < 0:
        raise ValueError("priority_tier_must_be_nonnegative")
    return [
        dict(row)
        for row in rows
        if int(row.get("priority_tier", 0)) <= limit
    ]


def _read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f"{destination.suffix}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(destination)


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(value), sort_keys=True) + "\n")
        stream.flush()


def prepare_priority_execution_manifest(
    output_root: Path, *, gpu_ids: Sequence[int] = (4, 5, 6, 7)
) -> dict[str, Any]:
    """Write the P16-first physical queue and its alias provenance."""

    root = Path(output_root).resolve()
    single = _read_json(root / "candidate_manifests" / "single_family_candidates.json")
    joint = _read_json(root / "candidate_manifests" / "joint_candidate_selection.json")
    queue = priority_execution_queue(single, gpu_ids=gpu_ids)
    aliases = candidate_alias_map([*single, *joint])
    payload = {
        "schema_version": "transformer-dh-power-alignment-4090-priority-p16-v1",
        "profile": "P16",
        "protocols": ["smoke10", "fixed50", "fixed500"],
        "physical_structure_count": len(queue),
        "gpu_ids": [int(gpu) for gpu in gpu_ids],
        "rows": queue,
        "aliases": {
            str(row["candidate_id"]): aliases.get(str(row["candidate_id"]), [str(row["candidate_id"])])
            for row in queue
        },
    }
    _write_json(root / "scheduler" / "priority_p16_queue.json", payload)
    return payload


def prepare_remaining_execution_manifest(
    output_root: Path, *, gpu_ids: Sequence[int] = (4, 5, 6, 7)
) -> dict[str, Any]:
    """Discover incomplete physical structures from accepted on-disk evidence."""

    root = Path(output_root).resolve()
    single = _read_json(root / "candidate_manifests" / "single_family_candidates.json")
    joint = _read_json(root / "candidate_manifests" / "joint_candidate_selection.json")
    combined = [*single, *joint]
    from search.orchestration.lidar_transformer_dh_joint import _engine_dir

    completed: set[str] = set()
    for row in unique_structure_queue(combined):
        accepted = True
        for profile in ("P32", "P16", "P8"):
            directory = _engine_dir(
                root,
                str(row["model"]),
                {str(key): int(value) for key, value in row["target_d_h_by_family"].items()},
                profile,
            )
            build_path = directory / "baseline_result.json"
            fixed_path = directory / "evaluation" / "fixed500" / "evaluation_acceptance.json"
            build = _read_json(build_path) if build_path.is_file() else {}
            fixed = _read_json(fixed_path) if fixed_path.is_file() else {}
            accepted = accepted and formal_latency_evidence_ready(
                build, fixed, profile=profile
            )
        if accepted:
            completed.add(str(row["structure_signature"]))
    queue = remaining_structure_queue(
        combined,
        completed_signatures=completed,
        gpu_ids=gpu_ids,
    )
    payload = {
        "schema_version": "transformer-dh-power-alignment-4090-remaining-v1",
        "profiles": ["P32", "P16", "P8"],
        "protocols": ["smoke10", "fixed50", "fixed500"],
        "completed_structure_count": len(completed),
        "remaining_structure_count": len(queue),
        "gpu_ids": [int(gpu) for gpu in gpu_ids],
        "rows": queue,
        "aliases": candidate_alias_map(combined),
    }
    _write_json(root / "scheduler" / "remaining_structure_queue.json", payload)
    return payload


def _configure_runtime_modules(
    *, paths: Runtime4090Paths, output_root: Path, nvcc_archs: Sequence[str]
) -> dict[str, Any]:
    """Patch imported H800 constants inside this 4090-only child process."""

    runtime = configure_4090_runtime(
        paths, output_root=output_root, nvcc_archs=nvcc_archs
    )
    from search.integration.runtime_environment import ensure_modelopt_source_available
    import search.orchestration.lidar_transformer_dh_build as dh_build
    import search.orchestration.lidar_transformer_dh_evaluate as dh_evaluate
    import search.orchestration.lidar_transformer_dh_joint as joint
    import search.orchestration.lidar_transformer_h800_baselines as baselines

    root = paths.tensorrt_root.resolve()
    trtexec = paths.trtexec_path.resolve()
    for module in (dh_build, joint, dh_evaluate, baselines):
        if hasattr(module, "TRT_ROOT"):
            module.TRT_ROOT = root
    for module in (dh_build, baselines):
        if hasattr(module, "TRTEXEC"):
            module.TRTEXEC = trtexec

    def configure_sm89(*, output_root: str | Path, cache_namespace: str) -> dict[str, str]:
        ensure_modelopt_source_available()
        manifest = configure_4090_runtime(
            paths,
            output_root=Path(output_root),
            nvcc_archs=nvcc_archs,
        )
        cache = (
            Path(output_root).resolve()
            / "environment"
            / "torch_extensions_modelopt_sm89"
            / str(cache_namespace)
        )
        cache.mkdir(parents=True, exist_ok=True)
        os.environ["TORCH_EXTENSIONS_DIR"] = str(cache)
        return {
            "conda_prefix": manifest["modelopt_prefix"],
            "python": manifest["python"],
            "nvcc": manifest["nvcc"],
            "g++": manifest["g++"],
            "torch_cuda_arch_list": "8.9",
            "torch_extensions_dir": str(cache),
            "cpu_fallback": "false",
        }

    joint.configure_modelopt_inprocess = configure_sm89
    return runtime


def _result_status(result: Mapping[str, Any]) -> str:
    rows = [*result.get("builds", ()), *result.get("evaluations", ())]
    return "ok" if rows and all(str(row.get("status")) == "ok" for row in rows) else "failed"


def execute_priority_worker(
    *,
    output_root: Path,
    physical_gpu: int,
    paths: Runtime4090Paths,
    nvcc_archs: Sequence[str],
) -> dict[str, Any]:
    """Run one GPU's P16 priority queue with resumable per-candidate evidence."""

    root = Path(output_root).resolve()
    queue_payload = _read_json(root / "scheduler" / "priority_p16_queue.json")
    rows = worker_queue(queue_payload["rows"], physical_gpu)
    runtime = _configure_runtime_modules(
        paths=paths, output_root=root, nvcc_archs=nvcc_archs
    )
    from search.orchestration.lidar_transformer_dh_joint import run_joint

    journal = root / "scheduler" / f"priority_p16_gpu{physical_gpu}.jsonl"
    completed = 0
    failed = 0
    for row in rows:
        candidate_id = str(row["candidate_id"])
        started = datetime.now(timezone.utc).isoformat()
        _append_jsonl(
            journal,
            {
                "event": "candidate_started",
                "candidate_id": candidate_id,
                "queue_index": int(row["queue_index"]),
                "physical_gpu": int(physical_gpu),
                "started_at": started,
            },
        )
        try:
            result = run_joint(
                output_root=root,
                model_name=str(row["model"]),
                targets={str(key): int(value) for key, value in row["target_d_h_by_family"].items()},
                physical_gpu=int(physical_gpu),
                plugin=paths.plugin_path.resolve(),
                profiles=("P16",),
                protocols=("smoke10", "fixed50", "fixed500"),
            )
            status = _result_status(result)
            if status == "ok":
                completed += 1
            else:
                failed += 1
            record = {
                "event": "candidate_finished",
                "candidate_id": candidate_id,
                "queue_index": int(row["queue_index"]),
                "physical_gpu": int(physical_gpu),
                "status": status,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "build_statuses": [item.get("status") for item in result.get("builds", ())],
                "evaluation_statuses": [item.get("status") for item in result.get("evaluations", ())],
            }
            _write_json(
                root / "scheduler" / "priority_p16_results" / f"{candidate_id}.json",
                {**record, "result": result},
            )
            _append_jsonl(journal, record)
        except Exception as error:  # preserve one failure without dropping the matrix
            failed += 1
            failure = {
                "event": "candidate_failed",
                "candidate_id": candidate_id,
                "queue_index": int(row["queue_index"]),
                "physical_gpu": int(physical_gpu),
                "status": "exception",
                "failure_type": type(error).__name__,
                "failure_reason": str(error),
                "traceback": traceback.format_exc(),
                "finished_at": datetime.now(timezone.utc).isoformat(),
            }
            _write_json(
                root / "failures" / "priority_p16" / f"{candidate_id}.json", failure
            )
            _append_jsonl(journal, failure)
    summary = {
        "status": "ok" if failed == 0 else "completed_with_failures",
        "physical_gpu": int(physical_gpu),
        "assigned": len(rows),
        "completed": completed,
        "failed": failed,
        "runtime": runtime,
    }
    _write_json(root / "scheduler" / f"priority_p16_gpu{physical_gpu}_summary.json", summary)
    return summary


def execute_remaining_worker(
    *,
    output_root: Path,
    physical_gpu: int,
    paths: Runtime4090Paths,
    nvcc_archs: Sequence[str],
) -> dict[str, Any]:
    """Run one resumable shard of the remaining P32/P16/P8 matrix."""

    root = Path(output_root).resolve()
    payload = _read_json(root / "scheduler" / "remaining_structure_queue.json")
    rows = [
        dict(row)
        for row in payload["rows"]
        if int(row["physical_gpu"]) == int(physical_gpu)
    ]
    runtime = _configure_runtime_modules(
        paths=paths, output_root=root, nvcc_archs=nvcc_archs
    )
    from search.orchestration.lidar_transformer_dh_joint import run_joint

    journal = root / "scheduler" / f"remaining_gpu{physical_gpu}.jsonl"
    completed = 0
    failed = 0
    for row in rows:
        candidate_id = str(row["candidate_id"])
        started = {
            "event": "candidate_started",
            "candidate_id": candidate_id,
            "structure_signature": str(row["structure_signature"]),
            "queue_index": int(row["queue_index"]),
            "physical_gpu": int(physical_gpu),
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        _append_jsonl(journal, started)
        try:
            result = run_joint(
                output_root=root,
                model_name=str(row["model"]),
                targets={
                    str(key): int(value)
                    for key, value in row["target_d_h_by_family"].items()
                },
                physical_gpu=int(physical_gpu),
                plugin=paths.plugin_path.resolve(),
                profiles=("P32", "P16", "P8"),
                protocols=("smoke10", "fixed50", "fixed500"),
            )
            status = _result_status(result)
            completed += status == "ok"
            failed += status != "ok"
            record = {
                **started,
                "event": "candidate_finished",
                "status": status,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "build_statuses": [item.get("status") for item in result.get("builds", ())],
                "evaluation_statuses": [
                    item.get("status") for item in result.get("evaluations", ())
                ],
            }
            _write_json(
                root / "scheduler" / "remaining_results" / f"{candidate_id}.json",
                {**record, "result": result},
            )
            _append_jsonl(journal, record)
        except Exception as error:
            failed += 1
            failure = {
                **started,
                "event": "candidate_failed",
                "status": "exception",
                "failure_type": type(error).__name__,
                "failure_reason": str(error),
                "traceback": traceback.format_exc(),
                "finished_at": datetime.now(timezone.utc).isoformat(),
            }
            _write_json(
                root / "failures" / "remaining" / f"{candidate_id}.json", failure
            )
            _append_jsonl(journal, failure)
    summary = {
        "status": "ok" if failed == 0 else "completed_with_failures",
        "physical_gpu": int(physical_gpu),
        "assigned": len(rows),
        "completed": completed,
        "failed": failed,
        "runtime": runtime,
    }
    _write_json(root / "scheduler" / f"remaining_gpu{physical_gpu}_summary.json", summary)
    return summary


def run_priority_formal_latency(
    *,
    output_root: Path,
    physical_gpu: int,
    paths: Runtime4090Paths,
    nvcc_archs: Sequence[str],
    isolation_seconds: int = 300,
    warmup: int = 200,
    iterations: int = 500,
    repeats: int = 5,
    max_priority_tier: int = 2,
    profile: str = "P16",
) -> list[dict[str, Any]]:
    """Measure structures against a same-profile baseline with replay."""

    root = Path(output_root).resolve()
    profile_id = str(profile)
    if profile_id not in {"P32", "P16", "P8"}:
        raise ValueError(f"unsupported_priority_latency_profile:{profile_id}")
    phase = f"priority_{profile_id.lower()}_4090"
    queue_payload = _read_json(root / "scheduler" / "priority_p16_queue.json")
    _configure_runtime_modules(paths=paths, output_root=root, nvcc_archs=nvcc_archs)
    import search.orchestration.lidar_transformer_dh_formal_latency as formal
    import search.orchestration.lidar_transformer_h800_latency as latency
    from search.integration.runtime_environment import (
        load_tensorrt_runtime,
        runtime_cuda_index_for_physical,
    )
    from search.orchestration.lidar_transformer_dh_joint import _engine_dir

    formal.TRT_ROOT = paths.tensorrt_root.resolve()
    latency.TRT_ROOT = paths.tensorrt_root.resolve()
    candidates_by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    missing: list[str] = []
    selected_queue = priority_rows_through_tier(
        queue_payload["rows"], max_priority_tier
    )
    for source in selected_queue:
        row = dict(source)
        directory = _engine_dir(
            root,
            str(row["model"]),
            {str(key): int(value) for key, value in row["target_d_h_by_family"].items()},
            profile_id,
        )
        build = _read_json(directory / "baseline_result.json") if (directory / "baseline_result.json").is_file() else {}
        fixed_path = directory / "evaluation" / "fixed500" / "evaluation_acceptance.json"
        fixed = _read_json(fixed_path) if fixed_path.is_file() else {}
        if not formal_latency_evidence_ready(build, fixed, profile=profile_id):
            missing.append(str(row["candidate_id"]))
            continue
        candidates_by_model[str(row["model"])].append(
            {
                **row,
                "fixed500_mAP": fixed["mAP"],
                "structure_hash": fixed["structure_hash"],
                "engine_sha256": build["engine_sha256"],
                "engine_directory": str(directory),
            }
        )
    if missing:
        raise RuntimeError(f"priority_formal_latency_evidence_incomplete:{missing}")

    formal._isolation_gate(root, phase, physical_gpu, isolation_seconds)
    load_tensorrt_runtime(paths.tensorrt_root)
    ctypes.CDLL(str(paths.plugin_path.resolve()), mode=ctypes.RTLD_GLOBAL)
    import torch

    runtime_gpu = runtime_cuda_index_for_physical(physical_gpu)
    torch.cuda.set_device(runtime_gpu)
    device = torch.device(f"cuda:{runtime_gpu}")
    all_rows: list[dict[str, Any]] = []
    for model, candidates in sorted(candidates_by_model.items()):
        baseline_id = f"{model}__B0"
        if not any(str(row["candidate_id"]) == baseline_id for row in candidates):
            raise RuntimeError(f"priority_formal_latency_baseline_missing:{model}")
        inputs = latency._real_inputs(model, device)
        rows = formal._time_group(
            output_root=root,
            phase=phase,
            model=model,
            profile=profile_id,
            candidate_rows=candidates,
            directory_for=lambda row: Path(str(row["engine_directory"])),
            candidate_key="candidate_id",
            baseline_key=baseline_id,
            physical_gpu=physical_gpu,
            device=device,
            inputs=inputs,
            warmup=warmup,
            iterations=iterations,
            repeats=repeats,
        )
        all_rows.extend(rows)
        del inputs
        torch.cuda.empty_cache()
    destination = root / "formal_latency" / phase
    _write_json(destination / "formal_latency.json", all_rows)
    formal._write_csv(
        root / f"power_alignment_priority_{profile_id.lower()}_formal_latency.csv",
        all_rows,
    )
    return all_rows


def run_all_matrix_formal_latency(
    *,
    output_root: Path,
    physical_gpu: int,
    paths: Runtime4090Paths,
    nvcc_archs: Sequence[str],
    isolation_seconds: int = 300,
    warmup: int = 200,
    iterations: int = 500,
    repeats: int = 5,
    maximum_candidates_per_batch: int = 8,
    require_complete_evidence: bool = True,
) -> list[dict[str, Any]]:
    """Measure all accepted structures on one GPU with same-profile replay."""

    root = Path(output_root).resolve()
    single = _read_json(root / "candidate_manifests" / "single_family_candidates.json")
    joint = _read_json(root / "candidate_manifests" / "joint_candidate_selection.json")
    combined = [*single, *joint]
    _configure_runtime_modules(paths=paths, output_root=root, nvcc_archs=nvcc_archs)
    import search.orchestration.lidar_transformer_dh_formal_latency as formal
    import search.orchestration.lidar_transformer_h800_latency as latency
    from search.integration.runtime_environment import (
        load_tensorrt_runtime,
        runtime_cuda_index_for_physical,
    )
    from search.orchestration.lidar_transformer_dh_joint import _engine_dir

    formal.TRT_ROOT = paths.tensorrt_root.resolve()
    latency.TRT_ROOT = paths.tensorrt_root.resolve()

    def directory_for_evidence(row: Mapping[str, Any], profile: str) -> Path:
        return _engine_dir(
            root,
            str(row["model"]),
            {
                str(key): int(value)
                for key, value in row["target_d_h_by_family"].items()
            },
            profile,
        )

    def load_evidence(directory: Path, profile: str) -> tuple[dict[str, Any], dict[str, Any]]:
        build_path = directory / "baseline_result.json"
        fixed_path = directory / "evaluation" / "fixed500" / "evaluation_acceptance.json"
        return (
            _read_json(build_path) if build_path.is_file() else {},
            _read_json(fixed_path) if fixed_path.is_file() else {},
        )

    grouped = all_matrix_formal_latency_candidates(
        combined,
        profiles=("P32", "P16", "P8"),
        evidence_directory=directory_for_evidence,
        evidence_loader=load_evidence,
        fail_on_missing=require_complete_evidence,
    )
    phase = "all_matrix_4090"
    formal._isolation_gate(root, phase, physical_gpu, isolation_seconds)
    load_tensorrt_runtime(paths.tensorrt_root)
    ctypes.CDLL(str(paths.plugin_path.resolve()), mode=ctypes.RTLD_GLOBAL)
    import torch

    runtime_gpu = runtime_cuda_index_for_physical(physical_gpu)
    torch.cuda.set_device(runtime_gpu)
    device = torch.device(f"cuda:{runtime_gpu}")
    all_rows: list[dict[str, Any]] = []
    destination = root / "formal_latency" / phase
    for (model, profile), candidates in grouped.items():
        inputs = latency._real_inputs(model, device)
        group_rows: list[dict[str, Any]] = []
        batches = formal_latency_candidate_batches(
            candidates,
            baseline_id=f"{model}__B0",
            maximum_candidates=maximum_candidates_per_batch,
        )
        for batch_index, batch in enumerate(batches, start=1):
            batch_path = (
                destination
                / model
                / profile.lower()
                / f"batch_{batch_index}.json"
            )
            cached = _read_json(batch_path) if batch_path.is_file() else []
            if formal_latency_batch_result_reusable(cached, batch):
                batch_rows = [dict(row) for row in cached]
            else:
                batch_rows = formal._time_group(
                    output_root=root,
                    phase=f"{phase}_{model}_{profile.lower()}_batch_{batch_index}",
                    model=model,
                    profile=profile,
                    candidate_rows=batch,
                    directory_for=lambda row: Path(str(row["engine_directory"])),
                    candidate_key="candidate_id",
                    baseline_key=f"{model}__B0",
                    physical_gpu=physical_gpu,
                    device=device,
                    inputs=inputs,
                    warmup=warmup,
                    iterations=iterations,
                    repeats=repeats,
                )
            for row in batch_rows:
                row["formal_batch_index"] = batch_index
                row["formal_batch_count"] = len(batches)
            group_rows.extend(batch_rows)
            _write_json(batch_path, batch_rows)
        all_rows.extend(group_rows)
        _write_json(destination / model / f"{profile.lower()}.json", group_rows)
        del inputs
        torch.cuda.empty_cache()
    _write_json(destination / "formal_latency.json", all_rows)
    formal._write_csv(root / "power_alignment_all_matrix_formal_latency.csv", all_rows)
    return all_rows


def collect_power_alignment_report(output_root: Path) -> dict[str, Any]:
    """Join physical, precision, fixed500, and formal-latency evidence."""

    root = Path(output_root).resolve()
    formal_path = root / "formal_latency" / "all_matrix_4090" / "formal_latency.json"
    if not formal_path.is_file():
        raise RuntimeError(f"all_matrix_formal_latency_missing:{formal_path}")
    latency_index = formal_latency_evidence_index(_read_json(formal_path))
    single = _read_json(root / "candidate_manifests" / "single_family_candidates.json")
    joint = _read_json(root / "candidate_manifests" / "joint_candidate_selection.json")
    from search.model_families.transformer.dh_power_alignment_4090 import (
        neighbor_advantage,
        neighbor_controls,
        speedup_metrics,
    )
    from search.orchestration.lidar_transformer_dh_joint import _engine_dir
    from search.reporting.transformer_dh_power_alignment_4090 import (
        accuracy_class,
        compact_evidence_record,
        precision_interaction,
        write_power_alignment_reports,
    )

    def evidence_row(manifest: Mapping[str, Any], profile: str) -> dict[str, Any]:
        directory = _engine_dir(
            root,
            str(manifest["model"]),
            {
                str(key): int(value)
                for key, value in manifest["target_d_h_by_family"].items()
            },
            profile,
        )
        build = _read_json(directory / "baseline_result.json")
        fixed = _read_json(
            directory / "evaluation" / "fixed500" / "evaluation_acceptance.json"
        )
        if not formal_latency_evidence_ready(build, fixed, profile=profile):
            raise RuntimeError(
                f"report_evidence_incomplete:{manifest['candidate_id']}:{profile}"
            )
        latency = latency_index.get((str(manifest["structure_signature"]), profile))
        if latency is None:
            raise RuntimeError(
                f"report_latency_missing:{manifest['candidate_id']}:{profile}"
            )
        return {
            **dict(manifest),
            "profile": profile,
            "structure_hash": str(fixed["structure_hash"]),
            "onnx_sha256": str(build["typed_onnx_sha256"]),
            "engine_sha256": str(build["engine_sha256"]),
            "scale_hash": str(fixed.get("scale_hash", "not_quantized")),
            "requested_realized_conflict_count": int(
                build["requested_realized_conflict_count"]
            ),
            "evaluated": int(fixed["evaluated"]),
            "skipped": int(fixed["skipped"]),
            "AP30": float(fixed["AP@0.3"]),
            "AP50": float(fixed["AP@0.5"]),
            "AP70": float(fixed["AP@0.7"]),
            "mAP": float(fixed["mAP"]),
            "p50_ms": float(latency["p50_ms"]),
            "p90_ms": float(latency["p90_ms"]),
            "p95_ms": float(latency["p95_ms"]),
            "p99_ms": float(latency["p99_ms"]),
            "tactic_signature": latency.get("tactic_signature"),
            "fusion_signature": latency.get("fusion_signature"),
            "kernel_count": latency.get("kernel_count"),
            "cast_count": latency.get("cast_count"),
            "reformat_count": latency.get("reformat_count"),
            "baseline_replay_drift_ratio": latency.get(
                "baseline_replay_p50_drift_ratio"
            ),
            "repeat_p50_cv": latency.get("repeat_p50_cv"),
            "latency_beneficial": bool(latency.get("latency_beneficial", False)),
            "required_latency_reduction": latency.get("required_reduction"),
            "observed_latency_reduction": latency.get("latency_reduction"),
            "diagnostic_only": bool(manifest.get("diagnostic", False)),
            "engine_directory": str(directory),
        }

    rows = [
        evidence_row(manifest, profile)
        for manifest in [*single, *joint]
        for profile in ("P32", "P16", "P8")
    ]
    baseline_rows = {
        (str(row["model"]), str(row["profile"])): row
        for row in rows
        if row.get("structure_kind") == "baseline"
    }
    p32_baselines = {
        model: row
        for (model, profile), row in baseline_rows.items()
        if profile == "P32"
    }
    for row in rows:
        model = str(row["model"])
        profile = str(row["profile"])
        baseline = baseline_rows[(model, profile)]
        baseline_p32 = p32_baselines[model]
        delta_structure = float(row["mAP"]) - float(baseline["mAP"])
        row["delta_structure"] = delta_structure
        row["accuracy_class"] = accuracy_class(
            delta_structure,
            evaluated=int(row["evaluated"]),
            skipped=int(row["skipped"]),
            finite=True,
        )
        row.update(
            speedup_metrics(
                baseline_p32_ms=float(baseline_p32["p50_ms"]),
                baseline_profile_ms=float(baseline["p50_ms"]),
                candidate_profile_ms=float(row["p50_ms"]),
            )
        )
        row["latency_class"] = (
            "HIGH_ALIGNMENT_BENEFICIAL"
            if row["structure_kind"] != "baseline" and row["latency_beneficial"]
            else "NO_BENEFIT"
        )
        if row["accuracy_class"] == "UNSAFE":
            row["diagnostic_only"] = True

    structure_profile_index = {
        (str(row["model"]), str(row["structure_signature"]), str(row["profile"])): row
        for row in rows
    }
    for row in rows:
        model = str(row["model"])
        profile = str(row["profile"])
        candidate_p32 = structure_profile_index[
            (model, str(row["structure_signature"]), "P32")
        ]
        metrics = precision_interaction(
            baseline_p32_map=float(baseline_rows[(model, "P32")]["mAP"]),
            baseline_profile_map=float(baseline_rows[(model, profile)]["mAP"]),
            candidate_p32_map=float(candidate_p32["mAP"]),
            candidate_profile_map=float(row["mAP"]),
        )
        row.update(metrics)

    single_index = {
        (str(row["model"]), str(row["family"]), int(row["d_h"]), str(row["profile"])): row
        for row in rows
        if row.get("structure_kind") == "single_family" and row.get("d_h") is not None
    }
    for row in rows:
        if row.get("structure_kind") != "single_family" or row.get("d_h") is None:
            continue
        controls = neighbor_controls(int(row["d_h"]), int(row["original_d_h"]))
        control_rows = [
            single_index.get(
                (str(row["model"]), str(row["family"]), control, str(row["profile"]))
            )
            for control in controls
        ]
        lower = next(
            (item for item in control_rows if item and int(item["d_h"]) < int(row["d_h"])),
            None,
        )
        upper = next(
            (item for item in control_rows if item and int(item["d_h"]) > int(row["d_h"])),
            None,
        )
        comparison = neighbor_advantage(
            candidate_ms=float(row["p50_ms"]),
            lower_control_ms=float(lower["p50_ms"]) if lower else None,
            upper_control_ms=float(upper["p50_ms"]) if upper else None,
        )
        row["neighbor_control_speedup"] = comparison.get("minimum_control_speedup")
        row["neighbor_control_advantage"] = bool(comparison["advantage"])
        if row["latency_beneficial"] and row["neighbor_control_advantage"]:
            row["latency_class"] = "ALIGNMENT_ADVANTAGE"

    # Search admission remains conservative until a candidate also has explicit
    # independent-build and joint evidence; unknown is never promoted.
    for row in rows:
        row.setdefault("build_repeat_stable", None)
        row["search_space_candidate"] = False

    compact = [compact_evidence_record(row) for row in rows]
    single_rows = [
        row for row in compact if row.get("structure_kind") in {"baseline", "single_family"}
    ]
    joint_rows = [row for row in compact if row.get("structure_kind") == "joint"]
    result = write_power_alignment_reports(
        root,
        candidate_rows=[*single, *joint],
        single_family_rows=single_rows,
        joint_rows=joint_rows,
    )
    _write_json(root / "reports" / "complete_evidence_rows.json", rows)
    return result


def run_fresh_build_repeat_latency(
    *,
    output_root: Path,
    build_repeat_directory: Path,
    physical_gpu: int,
    paths: Runtime4090Paths,
    nvcc_archs: Sequence[str],
    profile: str,
    isolation_seconds: int = 300,
    warmup: int = 200,
    iterations: int = 500,
    repeats: int = 5,
    build_repeat_count: int = 3,
    candidate_id: str = "C1",
) -> list[dict[str, Any]]:
    """Measure each independently built candidate against its paired baseline."""

    root = Path(output_root).resolve()
    build_root = Path(build_repeat_directory).resolve()
    profile_id = str(profile)
    if build_root.name != profile_id:
        raise ValueError(
            f"fresh_build_latency_profile_mismatch:{profile_id}:{build_root.name}"
        )
    _configure_runtime_modules(paths=paths, output_root=root, nvcc_archs=nvcc_archs)
    import search.orchestration.lidar_transformer_dh_formal_latency as formal
    import search.orchestration.lidar_transformer_h800_latency as latency
    from search.integration.runtime_environment import (
        load_tensorrt_runtime,
        runtime_cuda_index_for_physical,
    )

    formal.TRT_ROOT = paths.tensorrt_root.resolve()
    latency.TRT_ROOT = paths.tensorrt_root.resolve()
    phase = f"{str(candidate_id).lower()}_{profile_id.lower()}_fresh_build_stability"
    formal._isolation_gate(root, phase, physical_gpu, isolation_seconds)
    load_tensorrt_runtime(paths.tensorrt_root)
    ctypes.CDLL(str(paths.plugin_path.resolve()), mode=ctypes.RTLD_GLOBAL)
    import torch

    runtime_gpu = runtime_cuda_index_for_physical(physical_gpu)
    torch.cuda.set_device(runtime_gpu)
    device = torch.device(f"cuda:{runtime_gpu}")
    inputs = latency._real_inputs("lidar_cobevt", device)
    all_rows: list[dict[str, Any]] = []
    for repeat_index in range(1, int(build_repeat_count) + 1):
        candidates = fresh_build_latency_candidates(
            build_root / f"repeat_{repeat_index}",
            profile=profile_id,
            repeat_index=repeat_index,
            candidate_id=candidate_id,
        )
        all_rows.extend(
            formal._time_group(
                output_root=root,
                phase=f"{phase}_repeat_{repeat_index}",
                model="lidar_cobevt",
                profile=profile_id,
                candidate_rows=candidates,
                directory_for=lambda row: Path(str(row["engine_directory"])),
                candidate_key="candidate_id",
                baseline_key="baseline",
                physical_gpu=physical_gpu,
                device=device,
                inputs=inputs,
                warmup=warmup,
                iterations=iterations,
                repeats=repeats,
            )
        )
    del inputs
    torch.cuda.empty_cache()
    _write_json(build_root / "formal_latency_build_repeats.json", all_rows)
    formal._write_csv(build_root / "formal_latency_build_repeats.csv", all_rows)
    return all_rows


def _parse_gpu_ids(value: str) -> tuple[int, ...]:
    return tuple(int(token) for token in value.split(",") if token.strip())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--modelopt-prefix", required=True)
    parser.add_argument("--tensorrt-root", required=True)
    parser.add_argument("--plugin", required=True)
    parser.add_argument("--gpu-ids", default="4,5,6,7")
    parser.add_argument("--physical-gpu", type=int)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--prepare-remaining", action="store_true")
    parser.add_argument("--remaining-worker", action="store_true")
    parser.add_argument("--formal-latency", action="store_true")
    parser.add_argument("--all-formal-latency", action="store_true")
    parser.add_argument("--write-reports", action="store_true")
    parser.add_argument("--fresh-build-latency", action="store_true")
    parser.add_argument("--fresh-build-directory")
    parser.add_argument("--build-repeat-count", type=int, default=3)
    parser.add_argument("--isolation-seconds", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--formal-batch-size", type=int, default=8)
    parser.add_argument("--allow-incomplete-formal", action="store_true")
    parser.add_argument("--max-priority-tier", type=int, default=2)
    parser.add_argument("--profile", choices=("P32", "P16", "P8"), default="P16")
    args = parser.parse_args(argv)
    root = Path(args.output_root).resolve()
    gpu_ids = _parse_gpu_ids(args.gpu_ids)
    if args.write_reports:
        result = collect_power_alignment_report(root)
    elif args.prepare_remaining:
        result = prepare_remaining_execution_manifest(root, gpu_ids=gpu_ids)
    elif args.prepare_only:
        result = prepare_priority_execution_manifest(root, gpu_ids=gpu_ids)
    else:
        if args.physical_gpu is None:
            parser.error("--physical-gpu is required unless --prepare-only is set")
        paths = Runtime4090Paths(
            Path(args.modelopt_prefix), Path(args.tensorrt_root), Path(args.plugin)
        )
        nvcc = paths.resolved_nvcc_path
        import subprocess

        archs = subprocess.check_output(
            [str(nvcc), "--list-gpu-arch"], text=True
        ).splitlines()
        if args.remaining_worker:
            result = execute_remaining_worker(
                output_root=root,
                physical_gpu=args.physical_gpu,
                paths=paths,
                nvcc_archs=archs,
            )
        elif args.fresh_build_latency:
            if not args.fresh_build_directory:
                parser.error("--fresh-build-directory is required")
            rows = run_fresh_build_repeat_latency(
                output_root=root,
                build_repeat_directory=Path(args.fresh_build_directory),
                physical_gpu=args.physical_gpu,
                paths=paths,
                nvcc_archs=archs,
                profile=args.profile,
                isolation_seconds=args.isolation_seconds,
                warmup=args.warmup,
                iterations=args.iterations,
                repeats=args.repeats,
                build_repeat_count=args.build_repeat_count,
            )
            result = {"status": "ok", "fresh_build_latency_rows": len(rows)}
        elif args.all_formal_latency:
            rows = run_all_matrix_formal_latency(
                output_root=root,
                physical_gpu=args.physical_gpu,
                paths=paths,
                nvcc_archs=archs,
                isolation_seconds=args.isolation_seconds,
                warmup=args.warmup,
                iterations=args.iterations,
                repeats=args.repeats,
                maximum_candidates_per_batch=args.formal_batch_size,
                require_complete_evidence=not args.allow_incomplete_formal,
            )
            result = {"status": "ok", "all_formal_latency_rows": len(rows)}
        elif args.formal_latency:
            rows = run_priority_formal_latency(
                output_root=root,
                physical_gpu=args.physical_gpu,
                paths=paths,
                nvcc_archs=archs,
                isolation_seconds=args.isolation_seconds,
                warmup=args.warmup,
                iterations=args.iterations,
                repeats=args.repeats,
                max_priority_tier=args.max_priority_tier,
                profile=args.profile,
            )
            result = {"status": "ok", "formal_latency_rows": len(rows)}
        else:
            result = execute_priority_worker(
                output_root=root,
                physical_gpu=args.physical_gpu,
                paths=paths,
                nvcc_archs=archs,
            )
    print(json.dumps(result, sort_keys=True))
    return 0 if result.get("status", "ok") == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
