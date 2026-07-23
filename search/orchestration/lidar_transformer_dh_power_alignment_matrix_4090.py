"""Execute the Transformer head-alignment matrix on the RTX 4090 host.

The H800 experiment implementation remains unchanged.  This adapter removes
physical-structure aliases before dispatch and records every alias so reports
can still distinguish the requested single-family and joint candidates.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
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
    args = parser.parse_args(argv)
    root = Path(args.output_root).resolve()
    gpu_ids = _parse_gpu_ids(args.gpu_ids)
    if args.prepare_only:
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
