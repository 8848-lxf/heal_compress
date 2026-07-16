from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_scheduler_excludes_over_half_memory_and_orders_low_load() -> None:
    from search.orchestration.gpu_scheduler import select_stage2_gpu_ids

    report = select_stage2_gpu_ids(
        [
            {
                "index": 0,
                "uuid": "a",
                "memory_used_mib": 13000,
                "memory_total_mib": 24000,
                "utilization_gpu_pct": 0,
            },
            {
                "index": 1,
                "uuid": "b",
                "memory_used_mib": 4000,
                "memory_total_mib": 24000,
                "utilization_gpu_pct": 30,
            },
            {
                "index": 2,
                "uuid": "c",
                "memory_used_mib": 2000,
                "memory_total_mib": 24000,
                "utilization_gpu_pct": 5,
            },
        ],
        [],
        max_memory_fraction=0.50,
    )

    assert report["selected_gpu_ids"] == [2, 1]
    assert report["dispatch_allowed"] is True
    assert report["excluded"][0]["gpu_index"] == 0
    assert report["excluded"][0]["reason"] == "memory_fraction_above_limit"


def test_scheduler_reports_foreign_processes_without_termination_action() -> None:
    from search.orchestration.gpu_scheduler import select_stage2_gpu_ids

    processes = [
        {
            "gpu_uuid": "gpu-a",
            "pid": 123,
            "process_name": "external-training",
            "used_memory_mib": 6000,
        }
    ]
    report = select_stage2_gpu_ids(
        [
            {
                "index": 4,
                "uuid": "gpu-a",
                "memory_used_mib": 7000,
                "memory_total_mib": 24000,
                "utilization_gpu_pct": 7,
            }
        ],
        processes,
    )

    assert report["selected_gpu_ids"] == [4]
    assert report["eligible"][0]["foreign_process_count"] == 1
    assert report["eligible"][0]["processes"][0]["pid"] == 123
    assert report["process_policy"] == "audit_only_never_terminate"


def test_scheduler_returns_pending_when_no_gpu_is_eligible() -> None:
    from search.orchestration.gpu_scheduler import select_stage2_gpu_ids

    report = select_stage2_gpu_ids(
        [
            {
                "index": 7,
                "uuid": "gpu-7",
                "memory_used_mib": 12001,
                "memory_total_mib": 24000,
                "utilization_gpu_pct": 0,
            }
        ],
        [],
    )

    assert report["selected_gpu_ids"] == []
    assert report["dispatch_allowed"] is False
    assert report["status"] == "pending_no_eligible_gpu"


def test_scheduler_includes_exactly_half_memory_and_embedded_processes() -> None:
    from search.orchestration.gpu_scheduler import select_stage2_gpu_ids

    report = select_stage2_gpu_ids(
        [
            {
                "index": 5,
                "uuid": "gpu-5",
                "memory_used_mib": 12000,
                "memory_total_mib": 24000,
                "utilization_gpu_pct": 10,
                "processes": [{"pid": 456, "used_memory_mib": 12000}],
            }
        ],
        [],
        max_memory_fraction=0.50,
    )

    assert report["selected_gpu_ids"] == [5]
    assert report["eligible"][0]["memory_fraction"] == 0.5
    assert report["eligible"][0]["foreign_process_count"] == 1


def test_runner_selection_uses_all_gpus_when_allowlist_is_empty(
    tmp_path: Path, monkeypatch
) -> None:
    import json

    from search.integration import runtime_environment
    from search.orchestration.lidar_pyramid_search import (
        _select_runtime_stage2_gpus,
    )

    monkeypatch.setattr(
        runtime_environment,
        "query_gpus",
        lambda: [
            {
                "index": index,
                "uuid": f"gpu-{index}",
                "memory_used_mib": 1000 + index,
                "memory_total_mib": 24000,
                "utilization_gpu_pct": index,
                "processes": [],
            }
            for index in range(8)
        ],
    )

    report = _select_runtime_stage2_gpus(
        run_dir=tmp_path,
        configured_gpu_ids=[],
        max_memory_fraction=0.50,
    )

    assert report["selected_gpu_ids"] == list(range(8))
    persisted = json.loads(
        (tmp_path / "stage2_gpu_selection.json").read_text(encoding="utf-8")
    )
    assert persisted["process_policy"] == "audit_only_never_terminate"
