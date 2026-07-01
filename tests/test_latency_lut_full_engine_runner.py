from __future__ import annotations

import json
from pathlib import Path

from tools.latency_lut import run_full_engine_candidate_benchmark as runner
from tools.latency_lut.run_full_engine_candidate_benchmark import parse_args, run


def test_full_engine_runner_fails_unknown_channel_config_without_candidate_export_path(tmp_path: Path):
    candidate = tmp_path / "candidate.json"
    output = tmp_path / "result.json"
    candidate.write_text(
        json.dumps(
            {
                "candidate_id": "cand_test",
                "deploy_mode": "single_engine_maxK",
                "fixed_K": 29696,
                "pruning": {"enabled": True, "source": "preparsed_channel_config"},
                "channel_config": {},
                "precision_config": {},
            }
        ),
        encoding="utf-8",
    )

    status = run(
        parse_args(
            [
                "--candidate",
                str(candidate),
                "--output",
                str(output),
                "--deploy-mode",
                "single_engine_maxK",
                "--fixed-k",
                "29696",
            ]
        )
    )
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert status["success"] is False
    assert payload["status"] == "failed_candidate_export_not_connected"
    assert "T_real_p50" not in payload


def test_full_engine_runner_reuses_quant_deploy_pipeline_for_baseline_fp16(tmp_path: Path, monkeypatch):
    candidate = tmp_path / "baseline_like_fp16.json"
    output = tmp_path / "baseline_like_fp16.full_engine_result.json"
    candidate.write_text(
        json.dumps(
            {
                "candidate_id": "baseline_like_fp16",
                "deploy_mode": "single_engine_maxK",
                "fixed_K": 29696,
                "pruning": {"enabled": False},
                "precision_config": {"default": "FP16"},
            }
        ),
        encoding="utf-8",
    )

    calls: list[tuple[str, dict]] = []

    def fake_export(ctx):
        calls.append(("export", {"checkpoint": str(ctx.checkpoint)}))
        ctx.onnx_path.parent.mkdir(parents=True, exist_ok=True)
        ctx.onnx_path.write_bytes(b"fake-onnx")
        return {"success": True, "status": "success", "onnx_path": str(ctx.onnx_path)}

    def fake_build(ctx):
        calls.append(("build", {"precision": ctx.precision}))
        ctx.engine_path.parent.mkdir(parents=True, exist_ok=True)
        ctx.engine_path.write_bytes(b"fake-engine")
        return {"build_success": True, "status": "success", "engine_path": str(ctx.engine_path)}

    def fake_eval(ctx):
        calls.append(("eval", {"frames": ctx.val_subset_size, "precision": ctx.precision}))
        return {
            "success": True,
            "status": "success",
            "forward_p50_ms": 12.0,
            "forward_p90_ms": 14.0,
            "forward_p95_ms": 15.0,
            "forward_mean_ms": 13.0,
            "forward_ms": {"std": 0.5},
            "actual_frames": 50,
            "mAP": 0.7,
            "AP@0.70": 0.5,
        }

    monkeypatch.setattr(runner, "_export_single_engine_onnx", fake_export)
    monkeypatch.setattr(runner, "_build_single_engine", fake_build)
    monkeypatch.setattr(runner, "_evaluate_single_engine_subset", fake_eval)

    result = run(
        parse_args(
            [
                "--candidate",
                str(candidate),
                "--output",
                str(output),
                "--val-subset-size",
                "50",
                "--deploy-mode",
                "single_engine_maxK",
                "--fixed-k",
                "29696",
                "--precision-profile",
                "FP16",
            ]
        )
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert result["success"] is True
    assert payload["status"] == "success"
    assert payload["T_real_p50"] == 12.0
    assert payload["T_real_p95"] == 15.0
    assert payload["num_val_frames"] == 50
    assert [name for name, _ in calls] == ["export", "build", "eval"]


def test_full_engine_runner_reports_light_prune_stage_failure(tmp_path: Path, monkeypatch):
    candidate = tmp_path / "light_prune_fp16.json"
    output = tmp_path / "light_prune_fp16.full_engine_result.json"
    candidate.write_text(
        json.dumps(
            {
                "candidate_id": "light_prune_fp16",
                "deploy_mode": "single_engine_maxK",
                "fixed_K": 29696,
                "pruning": {
                    "enabled": True,
                    "source": "pruning_tool",
                    "importance": "l1",
                    "target_keep_ratio": 0.875,
                },
                "precision_config": {"default": "FP16"},
            }
        ),
        encoding="utf-8",
    )

    def fake_prune(ctx, candidate_payload):
        return {
            "success": False,
            "status": "physical_prune_failed",
            "error": "unit-test prune failure",
        }

    monkeypatch.setattr(runner, "_prepare_candidate_checkpoint", fake_prune)

    result = run(
        parse_args(
            [
                "--candidate",
                str(candidate),
                "--output",
                str(output),
                "--deploy-mode",
                "single_engine_maxK",
                "--fixed-k",
                "29696",
            ]
        )
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert result["success"] is False
    assert payload["status"] == "physical_prune_failed"
    assert payload["failed_stage"] == "physical_prune"
    assert "T_real_p50" not in payload
