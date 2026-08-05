from __future__ import annotations

import json
import inspect
import threading
import time
from types import SimpleNamespace


def test_stage2_serializes_only_process_global_torch_onnx_export() -> None:
    from search.stage2.lidar_pyramid_real_evaluator import LidarPyramidRealEvaluator

    source = inspect.getsource(LidarPyramidRealEvaluator._export_qdq)
    assert "with _ONNX_EXPORT_LOCK" in source


def test_stage2_parallel_scheduler_serializes_each_gpu_and_runs_gpus_concurrently(
    tmp_path, monkeypatch
) -> None:
    from search.orchestration.lidar_pyramid_search import LidarPyramidTwoStageSearch

    lock = threading.Lock()
    active_by_gpu = {1: 0, 3: 0}
    max_active_by_gpu = {1: 0, 3: 0}
    total_active = 0
    max_total_active = 0

    class FakeEvaluator:
        def __init__(self, gpu_id: int) -> None:
            self.gpu_id = gpu_id

        def evaluate_candidate(self, phenotype, *, output_dir, candidate_hash):
            nonlocal total_active, max_total_active
            with lock:
                active_by_gpu[self.gpu_id] += 1
                total_active += 1
                max_active_by_gpu[self.gpu_id] = max(
                    max_active_by_gpu[self.gpu_id], active_by_gpu[self.gpu_id]
                )
                max_total_active = max(max_total_active, total_active)
            time.sleep(0.05)
            with lock:
                active_by_gpu[self.gpu_id] -= 1
                total_active -= 1
            return {
                "status": "ok",
                "F2": float(int(candidate_hash[-1])),
                "artifact_dir": str(output_dir),
            }

    def item(index: int):
        genotype = SimpleNamespace(to_dict=lambda: {"candidate": index})
        record = SimpleNamespace(
            candidate_hash=f"candidate-{index}",
            F1=float(index),
            genotype=genotype,
            phenotype={"candidate": index},
        )
        return SimpleNamespace(record=record)

    search = LidarPyramidTwoStageSearch(
        config={}, checkpoint=tmp_path / "checkpoint.pth", output_root=tmp_path
    )
    evaluators = [(1, FakeEvaluator(1)), (3, FakeEvaluator(3))]
    monkeypatch.setattr(
        search, "_ga_stage2_evaluator_pool", lambda *_args: evaluators
    )
    monkeypatch.setattr("torch.cuda.set_device", lambda _device: None)

    round_dir = tmp_path / "round_000"
    started = time.perf_counter()
    rows = search._evaluate_ga_stage2_selected_parallel(
        context=object(),
        real_evaluator=SimpleNamespace(),
        run_dir=tmp_path,
        round_dir=round_dir,
        selected=[item(index) for index in range(4)],
        round_index=0,
    )
    elapsed = time.perf_counter() - started

    assert [row["candidate_hash"] for row in rows] == [
        "candidate-0",
        "candidate-1",
        "candidate-2",
        "candidate-3",
    ]
    assert [row["assigned_gpu_id"] for row in rows] == [1, 3, 1, 3]
    assert max_active_by_gpu == {1: 1, 3: 1}
    assert max_total_active == 2
    assert elapsed < 0.18
    schedule = json.loads(
        (round_dir / "stage2_parallel_schedule.json").read_text(encoding="utf-8")
    )
    assert schedule["per_gpu_execution"] == "sequential"
    assert schedule["cross_gpu_execution"] == "parallel"
