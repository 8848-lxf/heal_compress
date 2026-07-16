from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class _FormalPool:
    def __init__(self) -> None:
        self.tasks = []

    def map_tasks(self, tasks):
        self.tasks.extend(tasks)
        return [
            {
                **dict(task.get("deployment_metadata", {})),
                "candidate_hash": task["candidate_hash"],
                "status": "ok",
                "forward_p50_ms": 5.0 if index == 0 else 4.0 + index,
                "forward_p95_ms": 6.0 + index,
                "num_evaluated_frames": 1789,
                "num_skipped_frames": 0,
                "precision_identity_passed": True,
                "worker_gpu_id": 4,
            }
            for index, task in enumerate(tasks)
        ]


def _reference(tmp_path: Path) -> dict:
    engine = tmp_path / "strict-fp32.plan"
    engine.write_bytes(b"reference")
    return {
        "status": "ok",
        "reference_precision": "strict_fp32",
        "engine_path": str(engine),
        "engine_hash": "strict-engine",
        "eval_hash": "strict-eval",
        "reference_hash": "signed-reference",
        "mAP": 0.73,
        "forward_p50_ms": 5.0,
    }


def _row(tmp_path: Path, candidate: str) -> dict:
    engine = tmp_path / f"{candidate}.plan"
    engine.write_bytes(b"engine")
    return {
        "candidate_hash": candidate,
        "engine_path": str(engine),
        "engine_hash": f"engine-{candidate}",
        "deployment_identity": f"identity-{candidate}",
        "raw_precision_gene_hash": "precision",
        "repaired_precision_gene_hash": "precision",
        "requested_precision_profile_hash": "precision",
        "realized_precision_profile_hash": "precision",
        "precision_identity_passed": True,
    }


def test_formal_latency_rejects_active_build_or_candidate_workers(
    tmp_path: Path,
) -> None:
    from search.orchestration.formal_latency import run_formal_latency_replay

    with pytest.raises(RuntimeError, match="formal_latency_parallel_worker_active"):
        run_formal_latency_replay(
            rows=[_row(tmp_path, "a")],
            strict_fp32_reference=_reference(tmp_path),
            stage2_pool=_FormalPool(),
            run_dir=tmp_path,
            selected_gpu_id=4,
            selected_gpu_uuid="GPU-4",
            active_process_commands=["python -m search.stage2.trt_build_worker"],
            gpu_processes=[],
        )


def test_formal_latency_rejects_selected_gpu_processes(tmp_path: Path) -> None:
    from search.orchestration.formal_latency import run_formal_latency_replay

    with pytest.raises(RuntimeError, match="formal_latency_selected_gpu_busy"):
        run_formal_latency_replay(
            rows=[_row(tmp_path, "a")],
            strict_fp32_reference=_reference(tmp_path),
            stage2_pool=_FormalPool(),
            run_dir=tmp_path,
            selected_gpu_id=4,
            selected_gpu_uuid="GPU-4",
            active_process_commands=[],
            gpu_processes=[{"gpu_uuid": "GPU-4", "pid": 99}],
        )


def test_formal_latency_replays_strict_reference_first_on_one_gpu(
    tmp_path: Path,
) -> None:
    from search.orchestration.formal_latency import run_formal_latency_replay

    pool = _FormalPool()
    report = run_formal_latency_replay(
        rows=[_row(tmp_path, "a"), _row(tmp_path, "b")],
        strict_fp32_reference=_reference(tmp_path),
        stage2_pool=pool,
        run_dir=tmp_path,
        selected_gpu_id=4,
        selected_gpu_uuid="GPU-4",
        active_process_commands=[],
        gpu_processes=[],
        required_evaluated_frames=1789,
        required_skipped_frames=0,
    )

    assert pool.tasks[0]["candidate_hash"] == "strict_fp32_reference"
    assert report["strict_fp32_formal_p50_ms"] == 5.0
    assert report["successful_candidate_count"] == 2
    assert {row["formal_latency_gpu_uuid"] for row in report["results"]} == {
        "GPU-4"
    }
    assert all(row["formal_latency_success"] for row in report["results"])
