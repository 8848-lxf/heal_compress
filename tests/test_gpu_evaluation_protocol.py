from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_evaluation_provider_request_defaults_to_gpu_iou_and_eight_workers(
    tmp_path: Path, monkeypatch
) -> None:
    from search.integration import evaluation_provider

    def fake_run(command, **kwargs):
        request_path = Path(command[-1])
        request = json.loads(request_path.read_text(encoding="utf-8"))
        Path(request["output_path"]).write_text(
            json.dumps({"status": "ok"}), encoding="utf-8"
        )
        return type("Completed", (), {"returncode": 0, "stdout": ""})()

    monkeypatch.setattr(evaluation_provider.subprocess, "run", fake_run)
    monkeypatch.setattr(
        evaluation_provider,
        "modelopt_python_command",
        lambda _env: ["python"],
    )
    monkeypatch.setattr(
        evaluation_provider,
        "modelopt_subprocess_env",
        lambda **_kwargs: {},
    )
    root = tmp_path / "trt"
    root.mkdir()

    evaluation_provider.evaluate_engine_modelopt(
        engine_path=tmp_path / "engine.plan",
        checkpoint=tmp_path / "model.pth",
        model_config=tmp_path / "config.yaml",
        heal_root=tmp_path,
        device="cuda:4",
        output_dir=tmp_path / "evaluation",
        tensorrt_root=root,
        plugin_path=None,
        num_frames=10,
        warmup_frames=2,
    )

    request = json.loads(
        (tmp_path / "evaluation" / "evaluation_request.json").read_text(
            encoding="utf-8"
        )
    )
    assert request["ap_iou_backend"] == "gpu"
    assert request["strict_gpu_ap_iou"] is True
    assert request["num_workers"] == 8


def test_formal_evaluation_rejects_cpu_ap_backend() -> None:
    from search.integration.evaluation_worker import _validate_ap_iou_protocol

    with pytest.raises(RuntimeError, match="gpu_ap_iou_backend_required"):
        _validate_ap_iou_protocol("cpu", strict_gpu=True)


def test_gpu_ap_backend_has_no_cpu_fallback_in_worker_source() -> None:
    source = Path("search/integration/evaluation_worker.py").read_text(
        encoding="utf-8"
    )
    assert 'calculate_tp_fp_for_threshold(\n                            pred_box' in source
    assert '"cpu", device' not in source


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_gpu_tp_fp_matches_cpu_on_fixed_boxes() -> None:
    sys.path.insert(0, "/home/lixingfeng/UniAD_examine/HEAL")
    from opencood.utils import box_utils
    from tests.test_baseline_eval import calculate_tp_fp_for_threshold

    device = torch.device("cuda:0")
    boxes7 = torch.tensor(
        [
            [0.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0],
            [0.3, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0],
            [10.0, 10.0, 0.0, 4.0, 2.0, 1.5, 0.0],
        ],
        dtype=torch.float32,
        device=device,
    )
    gt7 = torch.tensor(
        [
            [0.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0],
            [10.0, 10.0, 0.0, 4.0, 2.0, 1.5, 0.0],
        ],
        dtype=torch.float32,
        device=device,
    )
    predicted = box_utils.boxes_to_corners_3d(boxes7, order="hwl")
    ground_truth = box_utils.boxes_to_corners_3d(gt7, order="hwl")
    scores = torch.tensor([0.95, 0.80, 0.70], device=device)

    for threshold in (0.3, 0.5, 0.7):
        cpu = {threshold: {"tp": [], "fp": [], "gt": 0, "score": []}}
        gpu = {threshold: {"tp": [], "fp": [], "gt": 0, "score": []}}
        calculate_tp_fp_for_threshold(
            predicted, scores, ground_truth, cpu, threshold, "cpu", device
        )
        calculate_tp_fp_for_threshold(
            predicted, scores, ground_truth, gpu, threshold, "gpu", device
        )
        assert gpu[threshold]["tp"] == cpu[threshold]["tp"]
        assert gpu[threshold]["fp"] == cpu[threshold]["fp"]
        assert gpu[threshold]["gt"] == cpu[threshold]["gt"]
        assert gpu[threshold]["score"] == pytest.approx(
            cpu[threshold]["score"]
        )
