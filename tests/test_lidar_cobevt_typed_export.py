from __future__ import annotations

import torch
import torch.nn as nn


def test_prepare_inputs_pads_fixed_k_and_keeps_agent_mask_input():
    from quantization.export.heal_lidar_cobevt import prepare_cobevt_maxk_inputs

    ego = {
        "inputs_m1": {
            "voxel_features": torch.ones(2, 32, 4),
            "voxel_coords": torch.tensor([[0, 0, 1, 2], [0, 0, 3, 4]]),
            "voxel_num_points": torch.tensor([32, 16]),
        },
        "record_len": torch.tensor([1]),
        "pairwise_t_matrix": torch.eye(4).view(1, 1, 1, 4, 4),
    }

    inputs = prepare_cobevt_maxk_inputs(ego, fixed_k=4, max_cav=2)

    assert inputs["voxel_features"].shape == (4, 32, 4)
    assert inputs["valid_voxel_mask"].tolist() == [1.0, 1.0, 0.0, 0.0]
    assert inputs["record_len"].dtype == torch.int32
    assert inputs["record_len"].tolist() == [1]
    assert inputs["pairwise_t_matrix"].shape == (1, 2, 2, 4, 4)
    assert torch.equal(inputs["pairwise_t_matrix"][0, 1, 1], torch.eye(4))


def test_cobevt_export_wrapper_is_family_specific():
    from quantization.export.heal_lidar_cobevt import HEALLiDARCoBEVTSignalMaxK

    assert HEALLiDARCoBEVTSignalMaxK.__module__ == "quantization.export.heal_lidar_cobevt"
    assert "Pyramid" not in HEALLiDARCoBEVTSignalMaxK.__name__


def test_export_recipe_writes_checked_six_input_onnx(tmp_path):
    from search.model_families.lidar_cobevt.export_recipe import CobevtExportRecipe

    class PillarVFE(nn.Module):
        def forward(self, batch):
            return {"pillar_features": batch["voxel_features"].mean(dim=1)}

    class Scatter(nn.Module):
        nx = 4
        ny = 4
        num_bev_features = 4

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.pillar_vfe = PillarVFE()
            self.scatter = Scatter()

    class Backbone(nn.Module):
        def forward(self, batch):
            batch["spatial_features_2d"] = batch["spatial_features"]
            return batch

    class MeanAgents(nn.Module):
        def forward(self, value):
            return value.mean(dim=1)

    class MaskConsumer(nn.Module):
        def forward(self, value, mask):
            agent_mask = mask[:, 0, 0, 0, :].reshape(1, 2, 1, 1, 1)
            return value * agent_mask.to(value)

    class Fusion(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([MaskConsumer()])
            self.mlp_head = MeanAgents()

    class Model(nn.Module):
        H = 4.0
        W = 4.0
        fake_voxel_size = 1.0

        def __init__(self):
            super().__init__()
            self.encoder_m1 = Encoder()
            self.backbone_m1 = Backbone()
            self.shrinker_m1 = nn.Identity()
            self.fusion_net = Fusion()
            self.cls_head = nn.Conv2d(4, 2, 1)
            self.reg_head = nn.Conv2d(4, 14, 1)
            self.dir_head = nn.Conv2d(4, 4, 1)

    inputs = {
        "voxel_features": torch.ones(4, 2, 4),
        "voxel_coords": torch.tensor(
            [[0, 0, 0, 0], [0, 0, 1, 1], [1, 0, 2, 2], [1, 0, 3, 3]],
            dtype=torch.int32,
        ),
        "voxel_num_points": torch.full((4,), 2, dtype=torch.int32),
        "pairwise_t_matrix": torch.eye(4).view(1, 1, 1, 4, 4).repeat(1, 2, 2, 1, 1),
        "valid_voxel_mask": torch.ones(4),
        "record_len": torch.tensor([2], dtype=torch.int32),
    }
    output = tmp_path / "cobevt.onnx"

    report = CobevtExportRecipe(fixed_k=4, max_cav=2).export(
        Model().eval(), inputs, output
    )

    assert output.is_file()
    assert report.checker_passed is True
    assert report.input_names[-1] == "record_len"
    assert report.output_names == ("cls_preds", "reg_preds", "dir_preds")
    assert report.registered_custom_ops == ("trt::PointPillarScatterTRT",)
    assert report.unregistered_custom_ops == ()
