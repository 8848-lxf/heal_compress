"""TensorRT-compatible fixedK export for HEAL Max/Disco soft fusion."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from .lidar_family import HEALLidarFamilySpec
from .trt_compatible_export import (
    SearchDynamicPointPillarScatterTRT,
    _normalize_pairwise_exportable,
    safe_voxel_num_points_for_fixed_k,
)

from exportable_bev_warp import warp_affine_simple_exportable  # type: ignore  # noqa: E402


def _base_bev_backbone_static(
    backbone: nn.Module,
    spatial_features: torch.Tensor,
) -> torch.Tensor:
    """Run BaseBEVBackbone without its mutable dictionary interface."""

    blocks = list(getattr(backbone, "blocks", ()))
    deblocks = list(getattr(backbone, "deblocks", ()))
    if not blocks:
        raise RuntimeError("softfusion_backbone_has_no_blocks")
    value = spatial_features
    decoded = []
    for index, block in enumerate(blocks):
        value = block(value)
        decoded.append(deblocks[index](value) if deblocks else value)
    if len(decoded) == 1:
        value = decoded[0]
    else:
        value = torch.cat(tuple(decoded), dim=1)
    if len(deblocks) > len(blocks):
        value = deblocks[-1](value)
    return value


def _pixel_weight_layer_static(
    pixel_weight_layer: nn.Module,
    value: torch.Tensor,
) -> torch.Tensor:
    """Run the 4-D PixelWeightLayer without its redundant dynamic reshape."""

    value = torch.relu(pixel_weight_layer.bn1_1(pixel_weight_layer.conv1_1(value)))
    value = torch.relu(pixel_weight_layer.bn1_2(pixel_weight_layer.conv1_2(value)))
    value = torch.relu(pixel_weight_layer.bn1_3(pixel_weight_layer.conv1_3(value)))
    return torch.relu(pixel_weight_layer.conv1_4(value))


class SearchTensorRTCompatibleSoftFusion(nn.Module):
    """B=1, dynamic-agent Max/Disco deployment wrapper."""

    def __init__(
        self,
        model: nn.Module,
        *,
        family: HEALLidarFamilySpec,
        output_names: Sequence[str],
        fixed_k: int,
        modality: str = "m1",
    ) -> None:
        super().__init__()
        if family.export_recipe != "soft_fusion_fixed_k":
            raise RuntimeError(
                f"softfusion_export_recipe_mismatch:{family.export_recipe}"
            )
        if family.fusion_kind not in {"max", "disconet"}:
            raise RuntimeError(
                f"unsupported_softfusion_kind:{family.fusion_kind}"
            )
        if family.fusion_kind == "disconet" and not hasattr(
            getattr(model, "fusion_net", None), "pixel_weight_layer"
        ):
            raise RuntimeError("disconet_pixel_weight_layer_missing")
        self.model = model
        self.family = family
        self.output_names = tuple(str(name) for name in output_names)
        self.fixed_k = int(fixed_k)
        self.modality = str(modality)
        fusion_step = (
            "disconet_fusion"
            if family.fusion_kind == "disconet"
            else "max_fusion"
        )
        self.execution_contract = (
            "pillar_vfe",
            "scatter_plugin",
            "base_bev_backbone",
            "modality_shrinker",
            fusion_step,
            "heads",
        )
        self.last_fusion_audit: dict[str, object] = {}

    def _encoder(self) -> nn.Module:
        return getattr(self.model, f"encoder_{self.modality}")

    def _backbone(self) -> nn.Module:
        return getattr(self.model, f"backbone_{self.modality}")

    def _shrinker(self) -> nn.Module:
        return getattr(self.model, f"shrinker_{self.modality}")

    def _affine(self, pairwise_t_matrix: torch.Tensor) -> torch.Tensor:
        return _normalize_pairwise_exportable(
            pairwise_t_matrix,
            float(self.model.H),
            float(self.model.W),
            float(self.model.fake_voxel_size),
        )

    def fuse_features(
        self,
        feature: torch.Tensor,
        pairwise_t_matrix: torch.Tensor,
    ) -> torch.Tensor:
        if int(pairwise_t_matrix.shape[0]) != 1:
            raise RuntimeError(
                "softfusion_single_engine_requires_batch_size_one"
            )
        affine = self._affine(pairwise_t_matrix)
        height, width = int(feature.shape[-2]), int(feature.shape[-1])
        theta = affine[0, 0, : feature.shape[0], :, :]
        warped = warp_affine_simple_exportable(
            feature,
            theta,
            (height, width),
            align_corners=False,
        )
        if self.family.fusion_kind == "max":
            fused = torch.amax(warped, dim=0, keepdim=True)
            self.last_fusion_audit = {
                "fusion_kind": "max",
                "agent_count": int(feature.shape[0]),
                "reduction_axis": 0,
            }
            return fused

        ego = feature[:1].expand(feature.shape[0], -1, -1, -1)
        fusion_input = torch.cat((warped, ego), dim=1)
        logits = _pixel_weight_layer_static(
            self.model.fusion_net.pixel_weight_layer,
            fusion_input,
        )
        weights = torch.softmax(logits, dim=0)
        fused = torch.sum(weights.expand_as(warped) * warped, dim=0, keepdim=True)
        self.last_fusion_audit = {
            "fusion_kind": "disconet",
            "agent_count": int(feature.shape[0]),
            "softmax_axis": 0,
            "pixel_weight_input_channels": int(fusion_input.shape[1]),
        }
        return fused

    def forward(
        self,
        voxel_features: torch.Tensor,
        voxel_coords: torch.Tensor,
        voxel_num_points: torch.Tensor,
        pairwise_t_matrix: torch.Tensor,
        valid_voxel_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        encoder = self._encoder()
        encoded = encoder.pillar_vfe(
            {
                "voxel_features": voxel_features,
                "voxel_coords": voxel_coords,
                "voxel_num_points": safe_voxel_num_points_for_fixed_k(
                    voxel_num_points, valid_voxel_mask
                ),
            }
        )
        feature = SearchDynamicPointPillarScatterTRT.apply(
            encoded["pillar_features"],
            voxel_coords,
            valid_voxel_mask,
            pairwise_t_matrix,
            int(encoder.scatter.ny),
            int(encoder.scatter.nx),
        )
        feature = _base_bev_backbone_static(self._backbone(), feature)
        feature = self._shrinker()(feature)
        if bool(getattr(self.model, "compress", False)):
            feature = self.model.compressor(feature)
        fused = self.fuse_features(feature, pairwise_t_matrix)
        if bool(getattr(self.model, "shrink_flag", False)):
            fused = self.model.shrink_conv(fused)
        outputs = {
            "cls_preds": self.model.cls_head(fused),
            "reg_preds": self.model.reg_head(fused),
            "dir_preds": self.model.dir_head(fused),
        }
        missing = sorted(set(self.output_names) - set(outputs))
        if missing:
            raise RuntimeError(f"unsupported_softfusion_outputs:{missing}")
        return tuple(outputs[name] for name in self.output_names)


__all__ = [
    "SearchTensorRTCompatibleSoftFusion",
    "_base_bev_backbone_static",
    "_pixel_weight_layer_static",
]
