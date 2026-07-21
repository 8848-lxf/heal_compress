from __future__ import annotations

from pathlib import Path

import pytest
import torch.nn as nn


CHECKPOINT = Path(
    "/home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/"
    "LiDAROnly/lidar_cobevt/net_epoch_bestval_at19.pth"
)
CONFIG = CHECKPOINT.with_name("config.yaml")
HEAL_ROOT = Path("/home/lixingfeng/UniAD_examine/HEAL")
EXPECTED_SHA = "67b0f2f00d74fea4912b4fdb902c1150146c733201e5dc36d3358f5ba605cfd4"
EXPECTED_CONFIG_SHA = "a0ee9d64fd1b01af95b1c997937ab07e0e810440236accfb598d5462ba086c33"


def test_real_checkpoint_preflight_binds_identity():
    from search.model_families.lidar_cobevt.model_capability import (
        CobevtModelCapability,
    )

    report = CobevtModelCapability(CHECKPOINT, CONFIG, HEAL_ROOT).preflight()

    assert report.model_family == "lidar_cobevt"
    assert report.checkpoint_sha256 == EXPECTED_SHA
    assert report.config_sha256 == EXPECTED_CONFIG_SHA
    assert report.model_core_method == "heter_model_baseline"
    assert report.fusion_method == "cobevt"


def test_preflight_rejects_non_cobevt_config(tmp_path):
    from search.model_families.lidar_cobevt.model_capability import (
        CobevtModelCapability,
    )

    config = tmp_path / "config.yaml"
    config.write_text(
        "model:\n  core_method: heter_model_baseline\n  args:\n"
        "    fusion_method: max\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="model_fusion_not_cobevt:max"):
        CobevtModelCapability(CHECKPOINT, config, HEAL_ROOT).preflight()


def test_real_checkpoint_load_has_full_weight_coverage():
    from search.model_families.lidar_cobevt.model_capability import (
        CobevtModelCapability,
    )

    bundle = CobevtModelCapability(CHECKPOINT, CONFIG, HEAL_ROOT).load()

    assert bundle.load_report.missing_weight_keys == ()
    assert bundle.load_report.unexpected_keys == ()
    assert bundle.load_report.loaded_state_keys == 224
    assert bundle.model.fusion_net.__class__.__name__ == "CoBEVT"
    assert bundle.model.training is False


def test_scatter_capability_is_float_and_not_quantized():
    from search.model_families.lidar_cobevt.model_capability import (
        validate_scatter_capability,
    )

    class Scatter(nn.Module):
        nx = 512
        ny = 256
        num_bev_features = 64

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.scatter = Scatter()

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder_m1 = Encoder()

    report = validate_scatter_capability(Model())

    assert report.module_name == "encoder_m1.scatter"
    assert report.allowed_boundary_dtypes == ("FP32", "FP16")
    assert report.quantization_gene is False
    assert report.output_shape == (64, 256, 512)
