from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from exportable_bev_warp import warp_affine_simple_exportable


SOURCE_FIXED_WRAPPER = (
    "/home/lixingfeng/UniAD_examine/HEAL/prune_model/pyramid-trt/export_dynamic_onnx.py"
)


def _resnet_static_features(resnet: nn.Module, spatial_features: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Run HEAL ResNetModified without returning a Python list in the ONNX graph."""
    layernum = int(getattr(resnet, "layernum", 0))
    if layernum == 1:
        feat0 = resnet.layer0(spatial_features)
        return (feat0,)
    if layernum == 2:
        feat0 = resnet.layer0(spatial_features)
        feat1 = resnet.layer1(feat0)
        return feat0, feat1
    if layernum == 3:
        feat0 = resnet.layer0(spatial_features)
        feat1 = resnet.layer1(feat0)
        feat2 = resnet.layer2(feat1)
        return feat0, feat1, feat2
    raise RuntimeError(f"fixed LiDAR pyramid export supports 1-3 ResNet levels, got {layernum}.")


def _decode_multiscale_static(backbone: nn.Module, features: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """Decode multiscale features without Python list append/index in the traced graph."""
    num_levels = int(getattr(backbone, "num_levels", len(features)))
    if num_levels != len(features):
        raise RuntimeError(f"backbone.num_levels={num_levels} does not match {len(features)} static features.")

    if num_levels == 1:
        up0 = backbone.deblocks[0](features[0]) if len(backbone.deblocks) > 0 else features[0]
        decoded = up0
    elif num_levels == 2:
        if len(backbone.deblocks) > 0:
            up0 = backbone.deblocks[0](features[0])
            up1 = backbone.deblocks[1](features[1])
        else:
            up0, up1 = features
        decoded = torch.cat((up0, up1), dim=1)
    elif num_levels == 3:
        if len(backbone.deblocks) > 0:
            up0 = backbone.deblocks[0](features[0])
            up1 = backbone.deblocks[1](features[1])
            up2 = backbone.deblocks[2](features[2])
        else:
            up0, up1, up2 = features
        decoded = torch.cat((up0, up1, up2), dim=1)
    else:
        raise RuntimeError(f"fixed LiDAR pyramid export supports 1-3 decode levels, got {num_levels}.")

    if len(backbone.deblocks) > num_levels:
        decoded = backbone.deblocks[-1](decoded)
    return decoded


def _bev_backbone_static(backbone: nn.Module, spatial_features: torch.Tensor) -> torch.Tensor:
    return _decode_multiscale_static(backbone, _resnet_static_features(backbone.resnet, spatial_features))


def _normalize_pairwise_exportable(
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
            matrix[..., 0, 2] / (float(downsample_rate) * float(discrete_ratio) * float(width_m)) * 2.0,
        ),
        dim=-1,
    )
    row1 = torch.stack(
        (
            matrix[..., 1, 0] * float(width_m) / float(height_m),
            matrix[..., 1, 1],
            matrix[..., 1, 2] / (float(downsample_rate) * float(discrete_ratio) * float(height_m)) * 2.0,
        ),
        dim=-1,
    )
    return torch.stack((row0, row1), dim=-2)


def _weighted_fuse_single_batch(
    feature: torch.Tensor,
    score: torch.Tensor,
    affine_matrix: torch.Tensor,
    align_corners: bool,
) -> torch.Tensor:
    """Fixed B=1 dynamic-agent weighted fuse without regroup/tensor_split sequences."""
    _, _, height, width = feature.shape
    theta_ego_from_all = affine_matrix[0, 0, : feature.shape[0], :, :]
    feature_in_ego = warp_affine_simple_exportable(
        feature,
        theta_ego_from_all,
        (int(height), int(width)),
        align_corners=align_corners,
    )
    score_in_ego = warp_affine_simple_exportable(
        score,
        theta_ego_from_all,
        (int(height), int(width)),
        align_corners=align_corners,
    )
    score_in_ego = torch.where(score_in_ego == 0, torch.full_like(score_in_ego, -1.0e20), score_in_ego)
    score_in_ego = torch.softmax(score_in_ego, dim=0)
    score_in_ego = torch.where(torch.isnan(score_in_ego), torch.zeros_like(score_in_ego), score_in_ego)
    return torch.sum(feature_in_ego * score_in_ego, dim=0, keepdim=True)


class FixedLidarPyramidExportWrapper(nn.Module):
    """Export-only LiDAR pyramid forward with static pyramid branches.

    This follows the fixed dynamic-agent wrapper pattern from the previous
    camera pyramid TensorRT project, but keeps only the LiDAR path. It bypasses
    HEAL's Python list based pyramid fusion and returns a fixed Tensor tuple.
    """

    def __init__(
        self,
        model: nn.Module,
        modality_name: str,
        output_names: list[str],
    ) -> None:
        super().__init__()
        self.model = model
        self.modality_name = modality_name
        self.output_names = list(output_names)
        self.wrapper_source = SOURCE_FIXED_WRAPPER

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
            raise RuntimeError(f"fixed LiDAR pyramid export expects 3 pyramid levels, got {num_levels}.")

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
        record_len: torch.Tensor,
        pairwise_t_matrix: torch.Tensor,
    ):
        data_dict = {
            f"inputs_{self.modality_name}": {
                "voxel_features": voxel_features,
                "voxel_coords": voxel_coords,
                "voxel_num_points": voxel_num_points,
            }
        }
        feature = self._encoder()(data_dict, self.modality_name)
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
        record_len_keepalive = record_len.to(dtype=fused_feature.dtype).sum() * 0.0
        outputs = {name: value + record_len_keepalive for name, value in outputs.items()}
        return tuple(outputs[name] for name in self.output_names)


def check_fixed_lidar_pyramid_wrapper_equivalence(
    wrapper: FixedLidarPyramidExportWrapper,
    tensors: tuple[torch.Tensor, ...],
    reference_output: dict[str, Any],
    output_names: list[str],
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "wrapper_source": str(Path(SOURCE_FIXED_WRAPPER)),
        "same_shape": True,
        "outputs": {},
    }
    with torch.no_grad():
        fixed_outputs = wrapper(*tensors)
    for name, fixed in zip(output_names, fixed_outputs):
        ref = reference_output.get(name)
        item: dict[str, Any] = {
            "reference_shape": list(ref.shape) if torch.is_tensor(ref) else None,
            "fixed_shape": list(fixed.shape) if torch.is_tensor(fixed) else None,
            "max_abs_error": None,
            "mean_abs_error": None,
            "relative_error": None,
        }
        if torch.is_tensor(ref) and torch.is_tensor(fixed):
            same_shape = tuple(ref.shape) == tuple(fixed.shape)
            report["same_shape"] = bool(report["same_shape"] and same_shape)
            if same_shape:
                diff = (ref.detach().float() - fixed.detach().float()).abs()
                denom = ref.detach().float().abs().mean().clamp_min(1.0e-12)
                item["max_abs_error"] = float(diff.max().item()) if diff.numel() else 0.0
                item["mean_abs_error"] = float(diff.mean().item()) if diff.numel() else 0.0
                item["relative_error"] = float((diff.mean() / denom).item()) if diff.numel() else 0.0
        else:
            report["same_shape"] = False
        report["outputs"][name] = item
    return report
