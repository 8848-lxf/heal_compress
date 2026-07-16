"""Low-occupancy GPU selection for parallel Stage-2 work."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def _process_key(row: Mapping[str, Any]) -> tuple[str, int, str]:
    return (
        str(row.get("gpu_uuid", row.get("uuid", ""))),
        int(row.get("pid", -1)),
        str(row.get("process_name", "")),
    )


def select_stage2_gpu_ids(
    gpu_rows: Sequence[Mapping[str, Any]],
    process_rows: Sequence[Mapping[str, Any]],
    *,
    max_memory_fraction: float = 0.50,
) -> dict[str, Any]:
    """Select every GPU under the memory cap, ordered by sampled load."""

    limit = float(max_memory_fraction)
    if not 0.0 < limit <= 1.0:
        raise ValueError("stage2_max_memory_fraction_must_be_in_0_1")

    processes_by_uuid: dict[str, list[dict[str, Any]]] = {}
    processes_by_index: dict[int, list[dict[str, Any]]] = {}
    for process in process_rows:
        normalized = dict(process)
        uuid = str(process.get("gpu_uuid", process.get("uuid", "")))
        if uuid:
            processes_by_uuid.setdefault(uuid, []).append(normalized)
        try:
            index = int(process.get("gpu_index", process.get("index", -1)))
        except (TypeError, ValueError):
            index = -1
        if index >= 0:
            processes_by_index.setdefault(index, []).append(normalized)

    eligible: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for gpu in gpu_rows:
        row = dict(gpu)
        index = int(row.get("index", -1))
        uuid = str(row.get("uuid", ""))
        total = float(row.get("memory_total_mib", 0.0) or 0.0)
        used = float(row.get("memory_used_mib", 0.0) or 0.0)
        if index < 0 or total <= 0.0 or used < 0.0:
            excluded.append(
                {
                    "gpu_index": index,
                    "gpu_uuid": uuid,
                    "reason": "invalid_gpu_memory_telemetry",
                    "telemetry": row,
                }
            )
            continue
        combined_processes = [
            dict(process) for process in row.get("processes", []) or []
        ]
        combined_processes.extend(processes_by_uuid.get(uuid, []))
        combined_processes.extend(processes_by_index.get(index, []))
        unique_processes: dict[tuple[str, int, str], dict[str, Any]] = {}
        for process in combined_processes:
            normalized = dict(process)
            normalized.setdefault("gpu_uuid", uuid)
            normalized.setdefault("gpu_index", index)
            unique_processes[_process_key(normalized)] = normalized
        processes = list(unique_processes.values())
        memory_fraction = used / total
        normalized_gpu = {
            **row,
            "gpu_index": index,
            "gpu_uuid": uuid,
            "memory_fraction": memory_fraction,
            "processes": processes,
            "foreign_process_count": len(processes),
        }
        if memory_fraction > limit:
            excluded.append(
                {
                    **normalized_gpu,
                    "reason": "memory_fraction_above_limit",
                }
            )
            continue
        eligible.append(normalized_gpu)

    eligible.sort(
        key=lambda row: (
            int(row.get("utilization_gpu_pct", 0) or 0),
            float(row["memory_fraction"]),
            int(row["foreign_process_count"]),
            int(row["gpu_index"]),
        )
    )
    excluded.sort(key=lambda row: int(row.get("gpu_index", -1)))
    selected = [int(row["gpu_index"]) for row in eligible]
    return {
        "status": "ready" if selected else "pending_no_eligible_gpu",
        "dispatch_allowed": bool(selected),
        "selected_gpu_ids": selected,
        "max_memory_fraction": limit,
        "eligible": eligible,
        "excluded": excluded,
        "process_policy": "audit_only_never_terminate",
        "gpu_count": len(gpu_rows),
        "eligible_count": len(eligible),
        "excluded_count": len(excluded),
    }
