from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from exportable_lidar_pyramid import (
    _bev_backbone_static,
    _decode_multiscale_static,
    _normalize_pairwise_exportable,
    _resnet_static_features,
    _weighted_fuse_single_batch,
)
from exportable_lidar_pyramid_padded_agent import point_pillar_scatter_fixed_agents


SOURCE_DYNAMIC_AGENT_WRAPPER = str(Path(__file__).resolve())


def dynamic_agent_input_names() -> list[str]:
    return ["voxel_features", "voxel_coords", "voxel_num_points", "pairwise_t_matrix"]


def dynamic_agent_dynamic_axes(output_names: list[str]) -> dict[str, dict[int, str]]:
    axes = {
        "voxel_features": {0: "num_voxels"},
        "voxel_coords": {0: "num_voxels"},
        "voxel_num_points": {0: "num_voxels"},
        "pairwise_t_matrix": {0: "batch", 1: "num_agents", 2: "num_agents"},
    }
    for name in output_names:
        axes[name] = {0: "batch"}
    return axes


def _encoder_dynamic_agent(
    encoder: nn.Module,
    voxel_features: torch.Tensor,
    voxel_coords: torch.Tensor,
    voxel_num_points: torch.Tensor,
    pairwise_t_matrix: torch.Tensor,
) -> torch.Tensor:
    batch_dict = {
        "voxel_features": voxel_features,
        "voxel_coords": voxel_coords,
        "voxel_num_points": voxel_num_points,
    }
    batch_dict = encoder.pillar_vfe(batch_dict)
    num_agents = pairwise_t_matrix.size(1)
    return point_pillar_scatter_fixed_agents(
        batch_dict["pillar_features"],
        voxel_coords,
        num_agents=num_agents,
        num_bev_features=int(encoder.scatter.num_bev_features),
        nx=int(encoder.scatter.nx),
        ny=int(encoder.scatter.ny),
        nz=int(encoder.scatter.nz),
    )


class ExportableLidarPyramidDynamicAgent(nn.Module):
    """Export-only lidar_pyramid wrapper where agent count is feature dim 0.

    This mode assumes deployment batch size ``B=1``. The active agent count is
    represented by the first dimension of BEV features and by
    ``pairwise_t_matrix.shape[1:3]`` instead of a record_len split/list.
    """

    def __init__(self, model: nn.Module, modality_name: str, output_names: list[str]) -> None:
        super().__init__()
        self.model = model
        self.modality_name = modality_name
        self.output_names = list(output_names)
        self.wrapper_source = SOURCE_DYNAMIC_AGENT_WRAPPER

    def _encoder(self) -> nn.Module:
        return getattr(self.model, f"encoder_{self.modality_name}")

    def _backbone(self) -> nn.Module:
        return getattr(self.model, f"backbone_{self.modality_name}")

    def _aligner(self) -> nn.Module:
        return getattr(self.model, f"aligner_{self.modality_name}")

    def _pyramid_forward_static(
        self,
        spatial_features: torch.Tensor,
        affine_matrix: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        pyramid = self.model.pyramid_backbone
        features = _resnet_static_features(pyramid.resnet, spatial_features)
        num_levels = int(getattr(pyramid, "num_levels", len(features)))
        if num_levels != 3 or len(features) != 3:
            raise RuntimeError(f"dynamic-agent LiDAR pyramid export expects 3 pyramid levels, got {num_levels}.")

        occ0 = pyramid.single_head_0(features[0])
        occ1 = pyramid.single_head_1(features[1])
        occ2 = pyramid.single_head_2(features[2])
        align_corners = bool(getattr(pyramid, "align_corners", False))
        fused0 = _weighted_fuse_single_batch(features[0], torch.sigmoid(occ0) + 1.0e-4, affine_matrix, align_corners)
        fused1 = _weighted_fuse_single_batch(features[1], torch.sigmoid(occ1) + 1.0e-4, affine_matrix, align_corners)
        fused2 = _weighted_fuse_single_batch(features[2], torch.sigmoid(occ2) + 1.0e-4, affine_matrix, align_corners)
        fused_feature = _decode_multiscale_static(pyramid, (fused0, fused1, fused2))
        return fused_feature, (occ0, occ1, occ2)

    def forward(
        self,
        voxel_features: torch.Tensor,
        voxel_coords: torch.Tensor,
        voxel_num_points: torch.Tensor,
        pairwise_t_matrix: torch.Tensor,
    ):
        feature = _encoder_dynamic_agent(
            self._encoder(),
            voxel_features,
            voxel_coords,
            voxel_num_points,
            pairwise_t_matrix,
        )
        feature = _bev_backbone_static(self._backbone(), feature)
        feature = self._aligner()(feature)

        if bool(getattr(self.model, "compress", False)):
            feature = self.model.compressor(feature)

        affine_matrix = _normalize_pairwise_exportable(
            pairwise_t_matrix,
            float(self.model.H),
            float(self.model.W),
            float(self.model.fake_voxel_size),
        )
        fused_feature, occ_outputs = self._pyramid_forward_static(feature, affine_matrix)

        if bool(getattr(self.model, "shrink_flag", False)):
            fused_feature = self.model.shrink_conv(fused_feature)

        outputs = {
            "cls_preds": self.model.cls_head(fused_feature),
            "reg_preds": self.model.reg_head(fused_feature),
            "dir_preds": self.model.dir_head(fused_feature),
            "occ0": occ_outputs[0],
            "occ1": occ_outputs[1],
            "occ2": occ_outputs[2],
        }
        return tuple(outputs[name] for name in self.output_names)


def check_dynamic_agent_wrapper_equivalence(
    wrapper: ExportableLidarPyramidDynamicAgent,
    tensors: tuple[torch.Tensor, ...],
    reference_output: dict[str, Any],
    output_names: list[str],
) -> dict[str, Any]:
    with torch.no_grad():
        outputs = wrapper(*tensors)
    report: dict[str, Any] = {"wrapper_source": SOURCE_DYNAMIC_AGENT_WRAPPER, "same_shape": True, "outputs": {}}
    for name, candidate in zip(output_names, outputs):
        ref = reference_output.get(name)
        item: dict[str, Any] = {
            "reference_shape": list(ref.shape) if torch.is_tensor(ref) else None,
            "candidate_shape": list(candidate.shape) if torch.is_tensor(candidate) else None,
            "max_abs_error": None,
            "mean_abs_error": None,
            "relative_error": None,
        }
        if torch.is_tensor(ref) and torch.is_tensor(candidate):
            same_shape = tuple(ref.shape) == tuple(candidate.shape)
            report["same_shape"] = bool(report["same_shape"] and same_shape)
            if same_shape:
                diff = (ref.detach().float() - candidate.detach().float()).abs()
                denom = ref.detach().float().abs().mean().clamp_min(1.0e-12)
                item["max_abs_error"] = float(diff.max().item()) if diff.numel() else 0.0
                item["mean_abs_error"] = float(diff.mean().item()) if diff.numel() else 0.0
                item["relative_error"] = float((diff.mean() / denom).item()) if diff.numel() else 0.0
        else:
            report["same_shape"] = False
        report["outputs"][name] = item
    return report
