from __future__ import annotations

import json
from types import SimpleNamespace


def test_evaluation_provider_requests_gpu_postprocess_and_eight_workers(tmp_path, monkeypatch) -> None:
    from search.integration import evaluation_provider

    captured = {}

    def fake_run(command, *, text, stdout, stderr, env, check):
        request_path = command[-1]
        request = json.loads(open(request_path, encoding="utf-8").read())
        captured["request"] = request
        captured["env"] = dict(env)
        with open(request["output_path"], "w", encoding="utf-8") as handle:
            json.dump({"status": "ok"}, handle)
        return SimpleNamespace(returncode=0, stdout="ok")

    monkeypatch.setattr(evaluation_provider, "modelopt_python_command", lambda _env: ["python"])
    monkeypatch.setattr(
        evaluation_provider,
        "modelopt_subprocess_env",
        lambda **_kwargs: {"LD_LIBRARY_PATH": "existing"},
    )
    monkeypatch.setattr(evaluation_provider.subprocess, "run", fake_run)

    result = evaluation_provider.evaluate_engine_modelopt(
        engine_path=tmp_path / "engine.plan",
        checkpoint=tmp_path / "model.pth",
        model_config=tmp_path / "config.yaml",
        heal_root=tmp_path / "HEAL",
        device="cuda:6",
        output_dir=tmp_path / "evaluation",
        tensorrt_root=tmp_path / "TensorRT",
        plugin_path=tmp_path / "plugin.so",
        num_frames=500,
        warmup_frames=200,
        latency_rounds=3,
    )

    assert result["status"] == "ok"
    request = captured["request"]
    assert request["device"] == "cuda:0"
    assert request["physical_device"] == "cuda:6"
    assert request["evaluation_protocol_version"] == "fixed-shared-manifest-prefix-gpu-postprocess-workers8-v4"
    assert request["ap_iou_backend"] == "gpu"
    assert request["require_cuda_postprocess"] is True
    assert request["dataloader_num_workers"] == 8
    assert request["torch_num_threads"] == 4
    assert captured["env"]["OMP_NUM_THREADS"] == "4"
    assert captured["env"]["MKL_NUM_THREADS"] == "4"


def test_shared_full_manifest_selects_stable_stage2_prefix() -> None:
    from search.integration.evaluation_worker import _fixed_manifest_subset

    warmup, evaluation = _fixed_manifest_subset(
        [f"w{index}" for index in range(200)],
        [f"e{index}" for index in range(1789)],
        warmup_frames=200,
        evaluation_frames=500,
    )

    assert len(warmup) == 200
    assert evaluation == [f"e{index}" for index in range(500)]


def test_shared_manifest_rejects_insufficient_requested_frames() -> None:
    import pytest

    from search.integration.evaluation_worker import _fixed_manifest_subset

    with pytest.raises(RuntimeError, match="eval_manifest_insufficient_frames"):
        _fixed_manifest_subset(
            ["w0"],
            ["e0", "e1"],
            warmup_frames=2,
            evaluation_frames=2,
        )
