from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_persistent_stage2_pool_reuses_one_process_per_gpu(tmp_path: Path) -> None:
    from search.orchestration.stage2_process_pool import PersistentStage2ProcessPool

    worker = tmp_path / "fake_worker.py"
    worker.write_text(
        """
import argparse
import json
import os
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--request', required=True)
args = parser.parse_args()
request = json.loads(Path(args.request).read_text())
Path(request['ready_path']).write_text(json.dumps({'pid': os.getpid(), 'gpu_id': request['gpu_id']}))
queue = Path(request['queue_dir'])
while not Path(request['stop_path']).exists():
    tasks = sorted(queue.glob('*.task.json'))
    if not tasks:
        time.sleep(0.01)
        continue
    task_path = tasks[0]
    task = json.loads(task_path.read_text())
    Path(task['result_path']).write_text(json.dumps({
        'status': 'ok',
        'candidate_hash': task['candidate_hash'],
        'worker_gpu_id': request['gpu_id'],
        'worker_pid': os.getpid(),
    }))
    task_path.unlink()
""",
        encoding="utf-8",
    )

    with PersistentStage2ProcessPool(
        run_dir=tmp_path / "run",
        gpu_ids=[4, 5],
        worker_payload={"kind": "test"},
        worker_command=[sys.executable, str(worker)],
        startup_timeout_seconds=5.0,
        task_timeout_seconds=5.0,
        poll_interval_seconds=0.01,
    ) as pool:
        first = pool.map_tasks(
            [{"candidate_hash": name} for name in ("a", "b", "c", "d")]
        )
        second = pool.map_tasks(
            [{"candidate_hash": name} for name in ("e", "f")]
        )
        reused = pool.map_tasks([{"candidate_hash": "a"}])

    assert [row["candidate_hash"] for row in first] == list("abcd")
    assert [row["worker_gpu_id"] for row in first] == [4, 5, 4, 5]
    assert [row["worker_gpu_id"] for row in second] == [4, 5]
    first_pids = {row["worker_gpu_id"]: row["worker_pid"] for row in first}
    second_pids = {row["worker_gpu_id"]: row["worker_pid"] for row in second}
    assert first_pids == second_pids
    assert reused[0]["pool_cache_hit"] is True
    assert reused[0]["reused_pool_task_id"] == first[0]["pool_task_id"]
    assert reused[0]["worker_pid"] == first[0]["worker_pid"]
    manifest_path = tmp_path / "run" / "stage2_workers" / "pool_manifest.json"
    assert manifest_path.is_file()
    import json

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "stopped"
    assert {row["returncode"] for row in manifest["workers"]} == {0}
    persisted = json.loads(
        sorted((tmp_path / "run" / "stage2_workers" / "results").glob("*.json"))[0].read_text(
            encoding="utf-8"
        )
    )
    assert persisted["pool_task_id"].startswith("task_")
    assert persisted["pool_signature"] == manifest["pool_signature"]


def test_persistent_stage2_pool_requires_unique_gpu_ids(tmp_path: Path) -> None:
    import pytest

    from search.orchestration.stage2_process_pool import PersistentStage2ProcessPool

    with pytest.raises(ValueError, match="stage2_worker_gpu_ids_must_be_unique"):
        PersistentStage2ProcessPool(
            run_dir=tmp_path,
            gpu_ids=[4, 4],
            worker_payload={},
        )
