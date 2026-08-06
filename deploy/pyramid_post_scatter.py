"""Export-safe HEAL Pyramid graph beginning at the dense BEV boundary."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_base_grid_2d(
    out_h: int,
    out_w: int,
    device: torch.device,
    dtype: torch.dtype,
    align_corners: bool,
) -> torch.Tensor:
    if align_corners:
        xs = (
            torch.linspace(-1.0, 1.0, out_w, device=device, dtype=dtype)
            if out_w > 1
            else torch.zeros((out_w,), device=device, dtype=dtype)
        )
        ys = (
            torch.linspace(-1.0, 1.0, out_h, device=device, dtype=dtype)
            if out_h > 1
            else torch.zeros((out_h,), device=device, dtype=dtype)
        )
    else:
        xs = (
            (torch.arange(out_w, device=device, dtype=dtype) + 0.5)
            * (2.0 / float(out_w))
            - 1.0
        )
        ys = (
            (torch.arange(out_h, device=device, dtype=dtype) + 0.5)
            * (2.0 / float(out_h))
            - 1.0
        )
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack((xx, yy, torch.ones_like(xx)), dim=-1).unsqueeze(0)


def _warp_affine_exportable(
    source: torch.Tensor,
    theta: torch.Tensor,
    output_size: tuple[int, int],
    align_corners: bool,
) -> torch.Tensor:
    out_h, out_w = (int(output_size[0]), int(output_size[1]))
    base = _make_base_grid_2d(
        out_h, out_w, theta.device, theta.dtype, align_corners
    )
    flat = base.reshape(1, out_h * out_w, 3).expand(theta.shape[0], -1, -1)
    grid = torch.bmm(flat, theta.transpose(1, 2)).reshape(
        theta.shape[0], out_h, out_w, 2
    )
    return F.grid_sample(
        source,
        grid.to(dtype=source.dtype, device=source.device),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=align_corners,
    )


def resnet_static_features(
    resnet: nn.Module, spatial_features: torch.Tensor
) -> tuple[torch.Tensor, ...]:
    """Run HEAL ResNetModified without Python list operations in ONNX."""

    layer_count = int(getattr(resnet, "layernum", 0))
    feature0 = resnet.layer0(spatial_features)
    if layer_count == 1:
        return (feature0,)
    feature1 = resnet.layer1(feature0)
    if layer_count == 2:
        return feature0, feature1
    if layer_count == 3:
        return feature0, feature1, resnet.layer2(feature1)
    raise RuntimeError(
        f"Pyramid export supports one to three ResNet levels, got {layer_count}"
    )


def decode_multiscale_static(
    backbone: nn.Module, features: tuple[torch.Tensor, ...]
) -> torch.Tensor:
    """Decode a statically sized feature tuple into one BEV tensor."""

    level_count = int(getattr(backbone, "num_levels", len(features)))
    if level_count != len(features) or level_count not in {1, 2, 3}:
        raise RuntimeError(
            f"invalid Pyramid decode levels: configured={level_count}, "
            f"features={len(features)}"
        )
    if len(backbone.deblocks) > 0:
        upsampled = tuple(
            backbone.deblocks[index](features[index])
            for index in range(level_count)
        )
    else:
        upsampled = features
    decoded = upsampled[0] if level_count == 1 else torch.cat(upsampled, dim=1)
    if len(backbone.deblocks) > level_count:
        decoded = backbone.deblocks[-1](decoded)
    return decoded


def bev_backbone_static(
    backbone: nn.Module, spatial_features: torch.Tensor
) -> torch.Tensor:
    return decode_multiscale_static(
        backbone, resnet_static_features(backbone.resnet, spatial_features)
    )


def normalize_pairwise_exportable(
    pairwise_t_matrix: torch.Tensor,
    height_m: float,
    width_m: float,
    discrete_ratio: float,
    downsample_rate: float = 1.0,
) -> torch.Tensor:
    matrix = pairwise_t_matrix[:, :, :, [0, 1], :][:, :, :, :, [0, 1, 3]]
    row0 = torch.stack(
        (
            matrix[..., 0, 0],
            matrix[..., 0, 1] * float(height_m) / float(width_m),
            matrix[..., 0, 2]
            / (float(downsample_rate) * float(discrete_ratio) * float(width_m))
            * 2.0,
        ),
        dim=-1,
    )
    row1 = torch.stack(
        (
            matrix[..., 1, 0] * float(width_m) / float(height_m),
            matrix[..., 1, 1],
            matrix[..., 1, 2]
            / (float(downsample_rate) * float(discrete_ratio) * float(height_m))
            * 2.0,
        ),
        dim=-1,
    )
    return torch.stack((row0, row1), dim=-2)


def weighted_fuse_single_batch(
    feature: torch.Tensor,
    score: torch.Tensor,
    affine_matrix: torch.Tensor,
    align_corners: bool,
) -> torch.Tensor:
    """Fuse a dynamic number of agents for the formal batch-size-one graph."""

    _, _, height, width = feature.shape
    theta = affine_matrix[0, 0, : feature.shape[0], :, :]
    feature_ego = _warp_affine_exportable(
        feature, theta, (int(height), int(width)), align_corners
    )
    score_ego = _warp_affine_exportable(
        score, theta, (int(height), int(width)), align_corners
    )
    score_ego = torch.where(
        score_ego == 0, torch.full_like(score_ego, -1.0e20), score_ego
    )
    weights = torch.softmax(score_ego, dim=0)
    weights = torch.where(torch.isnan(weights), torch.zeros_like(weights), weights)
    return torch.sum(feature_ego * weights, dim=0, keepdim=True)


class PostScatterLidarPyramid(nn.Module):
    """TensorRT export wrapper whose inputs contain no fixed voxel dimension K."""

    def __init__(
        self,
        model: nn.Module,
        modality_name: str,
        output_names: list[str],
    ) -> None:
        super().__init__()
        self.model = model
        self.modality_name = str(modality_name)
        self.output_names = list(output_names)

    def _backbone(self) -> nn.Module:
        return getattr(self.model, f"backbone_{self.modality_name}")

    def _aligner(self) -> nn.Module:
        return getattr(self.model, f"aligner_{self.modality_name}")

    def _pyramid_forward(
        self,
        spatial_features: torch.Tensor,
        affine_matrix: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        pyramid = self.model.pyramid_backbone
        features = resnet_static_features(pyramid.resnet, spatial_features)
        if len(features) != 3:
            raise RuntimeError(
                f"formal Pyramid export expects three levels, got {len(features)}"
            )
        occupancy = (
            pyramid.single_head_0(features[0]),
            pyramid.single_head_1(features[1]),
            pyramid.single_head_2(features[2]),
        )
        align_corners = bool(getattr(pyramid, "align_corners", False))
        fused = tuple(
            weighted_fuse_single_batch(
                feature,
                torch.sigmoid(score) + 1.0e-4,
                affine_matrix,
                align_corners,
            )
            for feature, score in zip(features, occupancy)
        )
        return decode_multiscale_static(pyramid, fused), occupancy

    def forward(
        self,
        spatial_features: torch.Tensor,
        pairwise_t_matrix: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        feature = bev_backbone_static(self._backbone(), spatial_features)
        feature = self._aligner()(feature)
        if bool(getattr(self.model, "compress", False)):
            feature = self.model.compressor(feature)
        affine_matrix = normalize_pairwise_exportable(
            pairwise_t_matrix,
            float(self.model.H),
            float(self.model.W),
            float(self.model.fake_voxel_size),
        )
        fused_feature, occupancy = self._pyramid_forward(feature, affine_matrix)
        if bool(getattr(self.model, "shrink_flag", False)):
            fused_feature = self.model.shrink_conv(fused_feature)
        outputs = {
            "cls_preds": self.model.cls_head(fused_feature),
            "reg_preds": self.model.reg_head(fused_feature),
            "dir_preds": self.model.dir_head(fused_feature),
            "occ0": occupancy[0],
            "occ1": occupancy[1],
            "occ2": occupancy[2],
        }
        return tuple(outputs[name] for name in self.output_names)


__all__ = ["PostScatterLidarPyramid"]
