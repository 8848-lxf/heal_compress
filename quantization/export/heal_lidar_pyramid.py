"""Formal HEAL LiDAR-pyramid signal-maxK export adapter."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as functional

from ..config import OnnxExportConfig
from ..exceptions import OnnxExportError


POINTPILLAR_SCATTER_PLUGIN_OP = "PointPillarScatterTRT"
POINTPILLAR_SCATTER_ONNX_DOMAIN = "trt"


def _base_grid(height: int, width: int, value: torch.Tensor, align_corners: bool) -> torch.Tensor:
    if align_corners:
        xs = torch.linspace(-1.0, 1.0, width, device=value.device, dtype=value.dtype) if width > 1 else value.new_zeros(width)
        ys = torch.linspace(-1.0, 1.0, height, device=value.device, dtype=value.dtype) if height > 1 else value.new_zeros(height)
    else:
        xs = (torch.arange(width, device=value.device, dtype=value.dtype) + 0.5) * (2.0 / float(width)) - 1.0
        ys = (torch.arange(height, device=value.device, dtype=value.dtype) + 0.5) * (2.0 / float(height)) - 1.0
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack((xx, yy, torch.ones_like(xx)), dim=-1).unsqueeze(0)


def _warp(value: torch.Tensor, theta: torch.Tensor, height: int, width: int, align_corners: bool) -> torch.Tensor:
    base = _base_grid(height, width, theta, align_corners)
    flat = base.reshape(1, height * width, 3).expand(theta.shape[0], -1, -1)
    grid = torch.bmm(flat, theta.transpose(1, 2)).reshape(theta.shape[0], height, width, 2)
    return functional.grid_sample(value, grid.to(value), align_corners=align_corners)


def _scatter_reference(
    pillar_features: torch.Tensor,
    voxel_coords: torch.Tensor,
    valid_voxel_mask: torch.Tensor,
    *,
    num_agents: int,
    height: int,
    width: int,
) -> torch.Tensor:
    valid = valid_voxel_mask.reshape(-1).to(pillar_features.device) > 0.5
    features = pillar_features[valid]
    coordinates = voxel_coords[valid].long()
    channels = int(pillar_features.shape[1])
    spatial = pillar_features.new_zeros((int(num_agents), channels, int(height), int(width)))
    for agent in range(int(num_agents)):
        selected = coordinates[:, 0] == agent
        if bool(selected.any()):
            rows = coordinates[selected, 2].clamp(0, int(height) - 1)
            columns = coordinates[selected, 3].clamp(0, int(width) - 1)
            spatial[agent, :, rows, columns] = features[selected].transpose(0, 1)
    return spatial


class DynamicPointPillarScatterTRT(torch.autograd.Function):
    """Runtime-equivalent scatter with a TensorRT custom-op symbolic."""

    @staticmethod
    def forward(ctx: Any, pillar_features: torch.Tensor, voxel_coords: torch.Tensor, valid_voxel_mask: torch.Tensor, pairwise_t_matrix: torch.Tensor, height: int, width: int) -> torch.Tensor:
        del ctx
        return _scatter_reference(
            pillar_features,
            voxel_coords,
            valid_voxel_mask,
            num_agents=int(pairwise_t_matrix.shape[1]),
            height=int(height),
            width=int(width),
        )

    @staticmethod
    def symbolic(graph: Any, pillar_features: Any, voxel_coords: Any, valid_voxel_mask: Any, pairwise_t_matrix: Any, height: int, width: int) -> Any:
        return graph.op(
            f"{POINTPILLAR_SCATTER_ONNX_DOMAIN}::{POINTPILLAR_SCATTER_PLUGIN_OP}",
            pillar_features,
            voxel_coords,
            valid_voxel_mask,
            pairwise_t_matrix,
            num_agents_i=0,
            height_i=int(height),
            width_i=int(width),
            plugin_version_s="1",
            plugin_namespace_s="",
        )


def _resnet_features(resnet: nn.Module, value: torch.Tensor) -> tuple[torch.Tensor, ...]:
    count = int(getattr(resnet, "layernum", 0))
    first = resnet.layer0(value)
    if count == 1:
        return (first,)
    second = resnet.layer1(first)
    if count == 2:
        return first, second
    if count == 3:
        return first, second, resnet.layer2(second)
    raise OnnxExportError(f"HEAL export supports one to three ResNet levels, got {count}")


def _decode(backbone: nn.Module, features: tuple[torch.Tensor, ...]) -> torch.Tensor:
    count = len(features)
    if count != int(getattr(backbone, "num_levels", count)):
        raise OnnxExportError("pyramid feature count differs from backbone.num_levels")
    upsampled = tuple(
        backbone.deblocks[index](feature) if len(backbone.deblocks) > index else feature
        for index, feature in enumerate(features)
    )
    value = upsampled[0] if len(upsampled) == 1 else torch.cat(upsampled, dim=1)
    return backbone.deblocks[-1](value) if len(backbone.deblocks) > count else value


def _bev_backbone(backbone: nn.Module, value: torch.Tensor) -> torch.Tensor:
    return _decode(backbone, _resnet_features(backbone.resnet, value))


def _normalize_pairwise(pairwise: torch.Tensor, height_m: float, width_m: float, ratio: float) -> torch.Tensor:
    matrix = pairwise[:, :, :, [0, 1], :][:, :, :, :, [0, 1, 3]]
    row0 = torch.stack((matrix[..., 0, 0], matrix[..., 0, 1] * height_m / width_m, matrix[..., 0, 2] / (ratio * width_m) * 2.0), dim=-1)
    row1 = torch.stack((matrix[..., 1, 0] * width_m / height_m, matrix[..., 1, 1], matrix[..., 1, 2] / (ratio * height_m) * 2.0), dim=-1)
    return torch.stack((row0, row1), dim=-2)


def _weighted_fuse(feature: torch.Tensor, score: torch.Tensor, affine: torch.Tensor, align_corners: bool) -> torch.Tensor:
    height, width = int(feature.shape[-2]), int(feature.shape[-1])
    theta = affine[0, 0, : feature.shape[0], :, :]
    warped_feature = _warp(feature, theta, height, width, align_corners)
    warped_score = _warp(score, theta, height, width, align_corners)
    warped_score = torch.where(warped_score == 0, torch.full_like(warped_score, -1.0e20), warped_score)
    weights = torch.softmax(warped_score, dim=0)
    weights = torch.where(torch.isnan(weights), torch.zeros_like(weights), weights)
    return torch.sum(warped_feature * weights, dim=0, keepdim=True)


class HEALLiDARPyramidSignalMaxK(nn.Module):
    """Five-input export-ready HEAL module with dynamic agent dimension."""

    def __init__(self, model: nn.Module, *, modality: str = "m1", output_names: Sequence[str] = ("cls_preds", "reg_preds", "dir_preds"), fixed_k: int = 29696) -> None:
        super().__init__()
        self.model = model
        self.modality = str(modality)
        self.output_names = tuple(str(value) for value in output_names)
        self.fixed_k = int(fixed_k)

    def forward(self, voxel_features: torch.Tensor, voxel_coords: torch.Tensor, voxel_num_points: torch.Tensor, pairwise_t_matrix: torch.Tensor, valid_voxel_mask: torch.Tensor) -> tuple[torch.Tensor, ...]:
        encoder = getattr(self.model, f"encoder_{self.modality}")
        safe_counts = torch.where(valid_voxel_mask.reshape(-1) > 0.5, voxel_num_points, torch.ones_like(voxel_num_points))
        encoded = encoder.pillar_vfe({
            "voxel_features": voxel_features,
            "voxel_coords": voxel_coords,
            "voxel_num_points": safe_counts,
        })
        feature = DynamicPointPillarScatterTRT.apply(
            encoded["pillar_features"], voxel_coords, valid_voxel_mask, pairwise_t_matrix,
            int(encoder.scatter.ny), int(encoder.scatter.nx),
        )
        feature = _bev_backbone(getattr(self.model, f"backbone_{self.modality}"), feature)
        feature = getattr(self.model, f"aligner_{self.modality}")(feature)
        if bool(getattr(self.model, "compress", False)):
            feature = self.model.compressor(feature)
        affine = _normalize_pairwise(pairwise_t_matrix, float(self.model.H), float(self.model.W), float(self.model.fake_voxel_size))
        pyramid = self.model.pyramid_backbone
        levels = _resnet_features(pyramid.resnet, feature)
        if len(levels) != 3:
            raise OnnxExportError(f"LiDAR pyramid export requires exactly three feature levels, got {len(levels)}")
        occupancies = (pyramid.single_head_0(levels[0]), pyramid.single_head_1(levels[1]), pyramid.single_head_2(levels[2]))
        align_corners = bool(getattr(pyramid, "align_corners", False))
        fused = tuple(
            _weighted_fuse(level, torch.sigmoid(occupancy) + 1.0e-4, affine, align_corners)
            for level, occupancy in zip(levels, occupancies)
        )
        value = _decode(pyramid, fused)
        if bool(getattr(self.model, "shrink_flag", False)):
            value = self.model.shrink_conv(value)
        outputs = {
            "cls_preds": self.model.cls_head(value),
            "reg_preds": self.model.reg_head(value),
            "dir_preds": self.model.dir_head(value),
            "occ0": occupancies[0], "occ1": occupancies[1], "occ2": occupancies[2],
        }
        return tuple(outputs[name] for name in self.output_names)


def build_heal_signal_maxk_export_module(model: nn.Module, *, config: OnnxExportConfig | None = None, modality: str = "m1") -> HEALLiDARPyramidSignalMaxK:
    """Build an export wrapper without loading data or model state."""

    policy = config or OnnxExportConfig()
    return HEALLiDARPyramidSignalMaxK(model, modality=modality, output_names=policy.output_names, fixed_k=policy.fixed_k)


def prepare_signal_maxk_inputs(ego_batch: Mapping[str, Any], *, config: OnnxExportConfig | None = None, modality: str = "m1") -> dict[str, torch.Tensor]:
    """Pad one real HEAL batch to fixed K and preserve dynamic agent count."""

    policy = config or OnnxExportConfig()
    source = ego_batch[f"inputs_{modality}"]
    features = source["voxel_features"].float()
    coordinates = source["voxel_coords"].to(torch.int32)
    counts = source["voxel_num_points"].to(torch.int32)
    count = int(features.shape[0])
    if count > policy.fixed_k:
        raise OnnxExportError(f"real voxel count {count} exceeds fixed K={policy.fixed_k}")
    padding = int(policy.fixed_k) - count
    features = functional.pad(features, (0, 0, 0, 0, 0, padding))
    coordinates = functional.pad(coordinates, (0, 0, 0, padding))
    counts = functional.pad(counts, (0, padding), value=1)
    mask = features.new_zeros((int(policy.fixed_k),))
    mask[:count] = 1.0
    record_len = int(ego_batch["record_len"][0].item())
    pairwise = ego_batch["pairwise_t_matrix"][:, :record_len, :record_len].float()
    return {
        "voxel_features": features.contiguous(),
        "voxel_coords": coordinates.contiguous(),
        "voxel_num_points": counts.contiguous(),
        "pairwise_t_matrix": pairwise.contiguous(),
        "valid_voxel_mask": mask.contiguous(),
    }


__all__ = ["HEALLiDARPyramidSignalMaxK", "build_heal_signal_maxk_export_module", "prepare_signal_maxk_inputs"]
