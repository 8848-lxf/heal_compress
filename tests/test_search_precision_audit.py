from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_checkpoint_dtype_audit_recurses_and_identifies_model_state_dict(tmp_path: Path) -> None:
    import torch

    from search.precision_audit import audit_checkpoint_dtypes

    checkpoint = {
        "epoch": 17,
        "model": {
            "conv.weight": torch.ones(2, 3, dtype=torch.float32),
            "bn.bias": torch.ones(2, dtype=torch.float16),
        },
        "optimizer": {
            "state": {
                0: {
                    "exp_avg": torch.ones(2, 3, dtype=torch.float32),
                    "step": torch.tensor(4, dtype=torch.int64),
                }
            }
        },
        "metadata": {"nested": [torch.ones(1, dtype=torch.bfloat16)]},
    }
    path = tmp_path / "checkpoint.pth"
    torch.save(checkpoint, path)

    report = audit_checkpoint_dtypes(path)

    assert report["checkpoint_path"] == str(path)
    assert report["all_tensors"]["torch.float32"]["tensor_count"] == 2
    assert report["all_tensors"]["torch.float16"]["element_count"] == 2
    assert report["all_tensors"]["torch.bfloat16"]["bytes"] == 2
    assert report["all_tensors"]["torch.int64"]["tensor_count"] == 1
    assert report["model_state_dict_path"] == "model"
    assert report["model_state_dict"]["torch.float32"]["element_count"] == 6
    assert report["model_state_dict"]["torch.float16"]["tensor_count"] == 1


def test_loaded_model_dtype_audit_counts_parameters_and_buffers() -> None:
    import torch

    from search.precision_audit import audit_loaded_model_dtypes

    model = torch.nn.Sequential(
        torch.nn.Conv2d(1, 2, kernel_size=1, bias=False),
        torch.nn.BatchNorm2d(2),
    )
    model[0].weight.data = model[0].weight.data.half()
    model[1].running_mean.data = model[1].running_mean.data.bfloat16()

    report = audit_loaded_model_dtypes(model)

    assert report["parameters"]["torch.float16"]["tensor_count"] == 1
    assert report["parameters"]["torch.float32"]["tensor_count"] == 2
    assert report["buffers"]["torch.bfloat16"]["tensor_count"] == 1
    assert report["buffers"]["torch.float32"]["tensor_count"] >= 1


def test_training_precision_audit_detects_amp_and_grad_scaler(tmp_path: Path) -> None:
    from search.precision_audit import audit_training_precision_sources

    config = tmp_path / "config.yaml"
    train = tmp_path / "train.py"
    config.write_text("train_params:\n  mixed_precision: true\n", encoding="utf-8")
    train.write_text(
        "from torch.cuda.amp import autocast, GradScaler\n"
        "scaler = GradScaler()\n"
        "with autocast(enabled=True):\n"
        "    pass\n",
        encoding="utf-8",
    )

    report = audit_training_precision_sources([config, train])

    assert report["training_compute_precision"] == "amp"
    assert report["training_forward_compute_precision"] == "mixed_precision_autocast"
    assert report["training_backward_compute_precision"] == "mixed_precision_grad_scaler"
    assert report["gradient_scaler_enabled"] is True
    assert any(hit["pattern"] == "GradScaler" for hit in report["evidence"])


def test_pytorch_eval_precision_hooks_record_io_dtypes_and_autocast() -> None:
    import torch

    from search.precision_audit import audit_pytorch_eval_forward_precision

    class Toy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = torch.nn.Conv2d(1, 2, kernel_size=1)
            self.cls_head = torch.nn.Conv2d(2, 1, kernel_size=1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.cls_head(self.backbone(x))

    model = Toy().eval()
    sample = torch.ones(1, 1, 2, 2, dtype=torch.float32)

    report = audit_pytorch_eval_forward_precision(
        model,
        sample,
        module_name_patterns={
            "backbone": "backbone",
            "classification head": "cls_head",
        },
    )

    assert report["pytorch_eval_parameter_precision"] == {"torch.float32": 4}
    assert report["pytorch_eval_autocast_enabled"] is False
    assert report["module_samples"]["backbone"]["input_dtypes"] == ["torch.float32"]
    assert report["module_samples"]["classification head"]["output_dtypes"] == ["torch.float32"]
