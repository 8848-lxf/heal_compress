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
)
from exportable_bev_warp import warp_affine_simple_exportable


SOURCE_PADDED_WRAPPER = str(Path(__file__).resolve())


def padded_agent_input_names() -> list[str]:
    return ["voxel_features", "voxel_coords", "voxel_num_points", "valid_agent_mask", "pairwise_t_matrix"]


def padded_agent_dynamic_axes(output_names: list[str]) -> dict[str, dict[int, str]]:
    axes = {
        "voxel_features": {0: "num_voxels"},
        "voxel_coords": {0: "num_voxels"},
        "voxel_num_points": {0: "num_voxels"},
        "valid_agent_mask": {0: "batch"},
        "pairwise_t_matrix": {0: "batch", 1: "max_cav", 2: "max_cav"},
    }
    for name in output_names:
        axes[name] = {0: "batch"}
    return axes


def make_valid_agent_mask(record_len: torch.Tensor, max_cav: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    agents = torch.arange(int(max_cav), device=record_len.device).view(1, int(max_cav))
    return (agents < record_len.to(device=record_len.device, dtype=torch.long).view(-1, 1)).to(dtype=dtype)


def point_pillar_scatter_fixed_agents(
    pillar_features: torch.Tensor,
    coords: torch.Tensor,
    *,
    num_agents: int,
    num_bev_features: int,
    nx: int,
    ny: int,
    nz: int = 1,
    valid_agent_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """PointPillarScatter equivalent with a fixed agent canvas count.

    This avoids HEAL's ``coords[:, 0].max().item()`` export path. Duplicate
    indices are assumed to have been checked by the existing diagnostics.
    """
    if int(nz) != 1:
        raise ValueError(f"Only nz=1 PointPillarScatter export is supported, got nz={nz}.")
    agent_idx = coords[:, 0].to(dtype=torch.long)
    y_idx = coords[:, 2].to(dtype=torch.long)
    x_idx = coords[:, 3].to(dtype=torch.long)
    linear = agent_idx * (int(nx) * int(ny)) + y_idx * int(nx) + x_idx
    canvas = pillar_features.new_zeros((num_agents * int(nx) * int(ny), int(num_bev_features)))
    canvas[linear] = pillar_features
    spatial = canvas.view(num_agents, int(ny), int(nx), int(num_bev_features)).permute(0, 3, 1, 2).contiguous()
    if valid_agent_mask is not None:
        mask = valid_agent_mask.reshape(-1)[:num_agents].to(device=spatial.device, dtype=spatial.dtype)
        spatial = spatial * mask.view(num_agents, 1, 1, 1)
    return spatial


def _encoder_padded(
    encoder: nn.Module,
    voxel_features: torch.Tensor,
    voxel_coords: torch.Tensor,
    voxel_num_points: torch.Tensor,
    valid_agent_mask: torch.Tensor,
    max_cav: int,
) -> torch.Tensor:
    batch_dict = {
        "voxel_features": voxel_features,
        "voxel_coords": voxel_coords,
        "voxel_num_points": voxel_num_points,
    }
    batch_dict = encoder.pillar_vfe(batch_dict)
    return point_pillar_scatter_fixed_agents(
        batch_dict["pillar_features"],
        voxel_coords,
        num_agents=int(max_cav),
        num_bev_features=int(encoder.scatter.num_bev_features),
        nx=int(encoder.scatter.nx),
        ny=int(encoder.scatter.ny),
        nz=int(encoder.scatter.nz),
        valid_agent_mask=valid_agent_mask,
    )


def _weighted_fuse_padded_single_batch(
    feature: torch.Tensor,
    score: torch.Tensor,
    affine_matrix: torch.Tensor,
    valid_agent_mask: torch.Tensor,
    align_corners: bool,
) -> torch.Tensor:
    _, _, height, width = feature.shape
    theta_ego_from_all = affine_matrix[0, 0, : feature.shape[0], :, :]
    mask = valid_agent_mask[0, : feature.shape[0]].to(device=feature.device, dtype=feature.dtype).view(-1, 1, 1, 1)
    masked_feature = feature * mask
    masked_score = score * mask
    feature_in_ego = warp_affine_simple_exportable(
        masked_feature,
        theta_ego_from_all,
        (int(height), int(width)),
        align_corners=align_corners,
    )
    score_in_ego = warp_affine_simple_exportable(
        masked_score,
        theta_ego_from_all,
        (int(height), int(width)),
        align_corners=align_corners,
    )
    score_in_ego = torch.where(score_in_ego == 0, torch.full_like(score_in_ego, -1.0e20), score_in_ego)
    score_in_ego = torch.softmax(score_in_ego, dim=0)
    score_in_ego = torch.where(torch.isnan(score_in_ego), torch.zeros_like(score_in_ego), score_in_ego)
    return torch.sum(feature_in_ego * score_in_ego, dim=0, keepdim=True)


class ExportableLidarPyramidPaddedAgent(nn.Module):
    """Export-only lidar_pyramid wrapper with fixed max_cav and valid_agent_mask."""

    def __init__(
        self,
        model: nn.Module,
        modality_name: str,
        output_names: list[str],
        *,
        max_cav: int = 2,
    ) -> None:
        super().__init__()
        self.model = model
        self.modality_name = modality_name
        self.output_names = list(output_names)
        self.max_cav = int(max_cav)
        self.wrapper_source = SOURCE_PADDED_WRAPPER

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
        valid_agent_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        pyramid = self.model.pyramid_backbone
        features = _resnet_static_features(pyramid.resnet, spatial_features)
        num_levels = int(getattr(pyramid, "num_levels", len(features)))
        if num_levels != 3 or len(features) != 3:
            raise RuntimeError(f"padded LiDAR pyramid export expects 3 pyramid levels, got {num_levels}.")

        occ0 = pyramid.single_head_0(features[0])
        occ1 = pyramid.single_head_1(features[1])
        occ2 = pyramid.single_head_2(features[2])
        align_corners = bool(getattr(pyramid, "align_corners", False))
        fused0 = _weighted_fuse_padded_single_batch(features[0], torch.sigmoid(occ0) + 1.0e-4, affine_matrix, valid_agent_mask, align_corners)
        fused1 = _weighted_fuse_padded_single_batch(features[1], torch.sigmoid(occ1) + 1.0e-4, affine_matrix, valid_agent_mask, align_corners)
        fused2 = _weighted_fuse_padded_single_batch(features[2], torch.sigmoid(occ2) + 1.0e-4, affine_matrix, valid_agent_mask, align_corners)
        fused_feature = _decode_multiscale_static(pyramid, (fused0, fused1, fused2))
        return fused_feature, (occ0, occ1, occ2)

    def forward(
        self,
        voxel_features: torch.Tensor,
        voxel_coords: torch.Tensor,
        voxel_num_points: torch.Tensor,
        valid_agent_mask: torch.Tensor,
        pairwise_t_matrix: torch.Tensor,
    ):
        feature = _encoder_padded(
            self._encoder(),
            voxel_features,
            voxel_coords,
            voxel_num_points,
            valid_agent_mask,
            self.max_cav,
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
        fused_feature, occ_outputs = self._pyramid_forward_static(feature, affine_matrix, valid_agent_mask)

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


def check_padded_agent_wrapper_equivalence(
    wrapper: ExportableLidarPyramidPaddedAgent,
    tensors: tuple[torch.Tensor, ...],
    reference_output: dict[str, Any],
    output_names: list[str],
) -> dict[str, Any]:
    with torch.no_grad():
        outputs = wrapper(*tensors)
    report: dict[str, Any] = {"wrapper_source": SOURCE_PADDED_WRAPPER, "same_shape": True, "outputs": {}}
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
