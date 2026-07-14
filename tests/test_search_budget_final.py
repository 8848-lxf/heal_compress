from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _context(tmp_path: Path) -> SimpleNamespace:
    stage2_manifest = tmp_path / "baseline" / "eval_manifest.json"
    stage2_manifest.parent.mkdir(parents=True)
    stage2_manifest.write_text(
        json.dumps({"evaluation_frame_ids": ["0", "1"], "manifest_hash": "stage2"}),
        encoding="utf-8",
    )
    return SimpleNamespace(
        checkpoint_path=tmp_path / "model.pth",
        model_config=tmp_path / "config.yaml",
        runtime_device="cuda:3",
        physical_gpu_id=3,
        eval_manifest_path=stage2_manifest,
        model_bundle=SimpleNamespace(adapter=object()),
        tensorrt=SimpleNamespace(
            tensorrt_root=tmp_path / "trt",
            plugin_path=tmp_path / "scatter.so",
            conda_env="modelopt",
        ),
    )


def _winner(tmp_path: Path, generation: int, identity: str, map_value: float) -> dict:
    artifact = tmp_path / "round_000" / f"generation_{generation:03d}" / identity
    artifact.mkdir(parents=True)
    (artifact / "engine.plan").write_bytes(f"engine-{identity}".encode())
    return {
        "generation": generation,
        "candidate_hash": f"candidate-{generation}",
        "physical_hash": f"physical-{identity}",
        "deployment_hash": f"deployment-{identity}",
        "engine_hash": f"engine-hash-{identity}",
        "artifact_dir": str(artifact),
        "mAP": map_value,
        "BOPS_retention": 0.21,
    }


def test_budget_final_evaluates_fixed_references_and_reuses_duplicate_deployment(tmp_path: Path) -> None:
    from search.orchestration.budget_final import run_budget_final_evaluation

    context = _context(tmp_path)
    for precision in ("strict_fp32", "strict_fp16"):
        path = tmp_path / "baselines" / f"original_{precision}" / "engine.plan"
        path.parent.mkdir(parents=True)
        path.write_bytes(precision.encode())
    first = _winner(tmp_path, 1, "a", 0.7)
    duplicate = {**first, "generation": 2, "candidate_hash": "candidate-2"}
    third = _winner(tmp_path, 3, "b", 0.6)
    calls: list[str] = []

    def evaluate_fn(**kwargs):
        engine = Path(kwargs["engine_path"])
        calls.append(engine.read_text(encoding="utf-8"))
        if "strict_fp32" in str(engine):
            map_value, latency = 0.80, 20.0
        elif "strict_fp16" in str(engine):
            map_value, latency = 0.79, 10.0
        elif engine.read_bytes() == b"engine-a":
            map_value, latency = 0.75, 8.0
        else:
            map_value, latency = 0.70, 7.0
        return {
            "status": "ok",
            "mAP": map_value,
            "AP@0.3": map_value + 0.1,
            "AP@0.5": map_value,
            "AP@0.7": map_value - 0.1,
            "forward_p50_ms": latency,
            "forward_p90_ms": latency + 1.0,
            "forward_p95_ms": latency + 2.0,
            "num_evaluated_frames": 5,
            "num_skipped_frames": 0,
        }

    report = run_budget_final_evaluation(
        context=context,
        run_dir=tmp_path,
        generation_winners=[first, duplicate, third],
        config={
            "num_frames": 5,
            "warmup_frames": 2,
            "reset_after_warmup": True,
            "evaluation_offset": 2,
            "latency_rounds": 3,
            "latency_metric": "forward_p50_ms",
            "eta_ap": 0.8,
            "eta_latency": 0.2,
        },
        budget=0.21,
        available_frame_ids=[str(index) for index in range(10)],
        evaluate_fn=evaluate_fn,
        gpu_isolation_fn=lambda *_args, **_kwargs: {"passed": True},
    )

    assert report["status"] == "ok"
    assert report["winner"]["generation"] == 1
    assert len(report["candidates"]) == 3
    assert report["candidates"][1]["reused_from_generation"] == 1
    assert len(calls) == 4  # two references plus two unique deployments
    assert report["manifest_overlap"]["overlap_count"] == 0
    assert (tmp_path / "budget_021_winner.json").is_file()
    assert (tmp_path / "budget_final_5frames.csv").is_file()


def test_budget_final_fails_closed_on_skipped_frame(tmp_path: Path) -> None:
    from search.orchestration.budget_final import run_budget_final_evaluation

    context = _context(tmp_path)
    for precision in ("strict_fp32", "strict_fp16"):
        path = tmp_path / "baselines" / f"original_{precision}" / "engine.plan"
        path.parent.mkdir(parents=True)
        path.write_bytes(precision.encode())
    winner = _winner(tmp_path, 1, "a", 0.7)

    def evaluate_fn(**_kwargs):
        return {
            "status": "ok",
            "mAP": 0.7,
            "forward_p50_ms": 10.0,
            "num_evaluated_frames": 4,
            "num_skipped_frames": 1,
        }

    with pytest.raises(RuntimeError, match="budget_final_evaluation_incomplete"):
        run_budget_final_evaluation(
            context=context,
            run_dir=tmp_path,
            generation_winners=[winner],
            config={"num_frames": 5, "warmup_frames": 2, "reset_after_warmup": True},
            budget=0.21,
            available_frame_ids=[str(index) for index in range(10)],
            evaluate_fn=evaluate_fn,
            gpu_isolation_fn=lambda *_args, **_kwargs: {"passed": True},
        )
