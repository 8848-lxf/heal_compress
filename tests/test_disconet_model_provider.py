from __future__ import annotations

import hashlib
import importlib
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


CONFIG = Path(
    "/home/lixingfeng/UniAD_examine/Auto_Search/original_models/"
    "dairv2s/LiDAROnly/lidar_disco/config.yaml"
)
CHECKPOINT = Path(
    "/home/lixingfeng/UniAD_examine/Auto_Search/original_models/"
    "dairv2s/LiDAROnly/lidar_disco/net_epoch_bestval_at35.pth"
)


def test_disconet_compat_pixel_weight_topology_matches_checkpoint() -> None:
    from search.integration.disconet_compat import PixelWeightLayer

    layer = PixelWeightLayer(256)

    assert tuple(layer.conv1_1.weight.shape) == (128, 512, 1, 1)
    assert tuple(layer.bn1_1.weight.shape) == (128,)
    assert tuple(layer.conv1_2.weight.shape) == (32, 128, 1, 1)
    assert tuple(layer.bn1_2.weight.shape) == (32,)
    assert tuple(layer.conv1_3.weight.shape) == (8, 32, 1, 1)
    assert tuple(layer.bn1_3.weight.shape) == (8,)
    assert tuple(layer.conv1_4.weight.shape) == (1, 8, 1, 1)


def test_disconet_compat_installer_is_explicit_and_idempotent() -> None:
    from search.integration.disconet_compat import install_disconet_compat_module

    first = install_disconet_compat_module()
    second = install_disconet_compat_module()
    module = importlib.import_module("opencood.models.fuse_modules.disco_fuse")

    assert first["module_name"] == "opencood.models.fuse_modules.disco_fuse"
    assert first["implementation"] in {"local_compat", "native"}
    assert second["module_name"] == first["module_name"]
    assert module.PixelWeightLayer.__name__ == "PixelWeightLayer"


def test_real_disconet_checkpoint_load_is_weight_key_complete() -> None:
    from search.integration.model_provider import load_heal_lidar_model

    bundle = load_heal_lidar_model(
        family="lidar_disco",
        checkpoint_path=CHECKPOINT,
        model_config_path=CONFIG,
        device="cpu",
        trace=False,
        strict_checkpoint=True,
    )

    assert bundle.family_spec.name == "lidar_disco"
    assert type(bundle.model).__name__ == "HeterModelBaseline"
    assert bundle.checkpoint_hash == hashlib.sha256(
        CHECKPOINT.read_bytes()
    ).hexdigest()
    assert bundle.checkpoint_load_audit["source_state_entry_count"] == 171
    assert bundle.checkpoint_load_audit["missing_parameter_keys"] == []
    assert bundle.checkpoint_load_audit["unexpected_weighted_keys"] == []
    assert bundle.checkpoint_load_audit["strict_weighted_pass"] is True
    fusion = bundle.model.fusion_net.pixel_weight_layer
    assert tuple(fusion.conv1_1.weight.shape) == (128, 512, 1, 1)
    assert torch.equal(
        fusion.conv1_4.weight.detach().cpu(),
        torch.load(CHECKPOINT, map_location="cpu")[
            "fusion_net.pixel_weight_layer.conv1_4.weight"
        ],
    )


def test_pyramid_compatibility_loader_keeps_public_name() -> None:
    from search.integration import model_provider

    assert model_provider.LidarPyramidModelBundle is model_provider.HEALLidarModelBundle
    assert callable(model_provider.load_lidar_pyramid_model)

