from __future__ import annotations

from pathlib import Path

import pytest


def _request(tmp_path: Path) -> dict[str, object]:
    return {
        "ap_iou_backend": "gpu",
        "checkpoint": str(tmp_path / "model.pth"),
        "device": "cuda:0",
        "engine_path": str(tmp_path / "engine.plan"),
        "eval_manifest_path": str(tmp_path / "fixed50.json"),
        "fixed_k": 24064,
        "heal_root": str(tmp_path / "HEAL"),
        "model_config": str(tmp_path / "config.yaml"),
        "model_family": "lidar_cobevt",
        "num_frames": 50,
        "num_workers": 8,
        "output_path": str(tmp_path / "evaluation.json"),
        "plugin_path": str(tmp_path / "scatter.so"),
        "strict_gpu_ap_iou": True,
        "warmup_frames": 20,
    }


def test_cobevt_worker_accepts_only_its_gpu_protocol(tmp_path: Path) -> None:
    from search.integration.lidar_cobevt_evaluation_worker import validate_request

    request = _request(tmp_path)

    validated = validate_request(request)

    assert validated["model_family"] == "lidar_cobevt"
    assert validated["device"] == "cuda:0"
    assert validated["num_workers"] == 8
    assert validated["ap_iou_backend"] == "gpu"


@pytest.mark.parametrize(
    ("key", "value", "reason"),
    [
        ("model_family", "lidar_pyramid", "cobevt_model_family_required"),
        ("device", "cpu", "cobevt_cuda_device_required"),
        ("num_workers", 4, "cobevt_evaluation_num_workers_must_equal_8"),
        ("ap_iou_backend", "cpu", "cobevt_gpu_ap_iou_backend_required"),
        ("strict_gpu_ap_iou", False, "cobevt_strict_gpu_ap_iou_required"),
    ],
)
def test_cobevt_worker_rejects_nonproduction_protocol(
    tmp_path: Path,
    key: str,
    value: object,
    reason: str,
) -> None:
    from search.integration.lidar_cobevt_evaluation_worker import validate_request

    request = _request(tmp_path)
    request[key] = value

    with pytest.raises(RuntimeError, match=reason):
        validate_request(request)


def test_cobevt_worker_uses_cobevt_input_preparer() -> None:
    source = Path("search/integration/lidar_cobevt_evaluation_worker.py").read_text(
        encoding="utf-8"
    )

    assert "prepare_cobevt_maxk_inputs" in source
    assert "prepare_signal_maxk_inputs" not in source
