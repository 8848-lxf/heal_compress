"""Post-scatter ONNX adapter for HEAL LiDAR V2X-ViT.

The adapter preserves V2X-ViT grid-sampling semantics while keeping dynamic
voxelization, PFN and scatter outside TensorRT.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as functional


@dataclass(frozen=True)
class HealV2XViTPostScatterPolicy:
    """Runtime policy for a TensorRT graph with no point-count dimension."""

    max_agents: int = 2
    modality: str = "m1"
    output_names: tuple[str, ...] = ("cls_preds", "reg_preds", "dir_preds")

    def __post_init__(self) -> None:
        if int(self.max_agents) <= 0:
            raise ValueError("v2xvit_max_agents_must_be_positive")
        if not self.output_names:
            raise ValueError("v2xvit_output_names_must_not_be_empty")


def _base_grid(
    height: int,
    width: int,
    reference: torch.Tensor,
    *,
    align_corners: bool,
) -> torch.Tensor:
    if align_corners:
        xs = (
            torch.linspace(-1.0, 1.0, width, device=reference.device, dtype=reference.dtype)
            if width > 1
            else reference.new_zeros(width)
        )
        ys = (
            torch.linspace(-1.0, 1.0, height, device=reference.device, dtype=reference.dtype)
            if height > 1
            else reference.new_zeros(height)
        )
    else:
        xs = (torch.arange(width, device=reference.device, dtype=reference.dtype) + 0.5) * (
            2.0 / float(width)
        ) - 1.0
        ys = (torch.arange(height, device=reference.device, dtype=reference.dtype) + 0.5) * (
            2.0 / float(height)
        ) - 1.0
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack((xx, yy, torch.ones_like(xx)), dim=-1).unsqueeze(0)


def _warp_exportable(
    value: torch.Tensor,
    theta: torch.Tensor,
    *,
    height: int,
    width: int,
    align_corners: bool,
    mode: str = "bilinear",
) -> torch.Tensor:
    base = _base_grid(height, width, theta, align_corners=align_corners)
    flat = base.reshape(1, height * width, 3).expand(theta.shape[0], -1, -1)
    grid = torch.bmm(flat, theta.transpose(1, 2)).reshape(theta.shape[0], height, width, 2)
    return functional.grid_sample(
        value,
        grid.to(value),
        mode=mode,
        padding_mode="zeros",
        align_corners=align_corners,
    )


def _identity_theta(count: int, reference: torch.Tensor) -> torch.Tensor:
    row0 = reference.new_tensor((1.0, 0.0, 0.0))
    row1 = reference.new_tensor((0.0, 1.0, 0.0))
    return torch.stack((row0, row1), dim=0).unsqueeze(0).expand(int(count), -1, -1)


def _normalize_pairwise(
    pairwise: torch.Tensor,
    *,
    height_m: float,
    width_m: float,
    ratio: float,
) -> torch.Tensor:
    matrix = pairwise[:, :, :, [0, 1], :][:, :, :, :, [0, 1, 3]]
    row0 = torch.stack(
        (
            matrix[..., 0, 0],
            matrix[..., 0, 1] * height_m / width_m,
            matrix[..., 0, 2] / (ratio * width_m) * 2.0,
        ),
        dim=-1,
    )
    row1 = torch.stack(
        (
            matrix[..., 1, 0] * width_m / height_m,
            matrix[..., 1, 1],
            matrix[..., 1, 2] / (ratio * height_m) * 2.0,
        ),
        dim=-1,
    )
    return torch.stack((row0, row1), dim=-2)


def _base_bev_backbone(backbone: nn.Module, spatial: torch.Tensor) -> torch.Tensor:
    value = spatial
    upsampled: list[torch.Tensor] = []
    for index, block in enumerate(backbone.blocks):
        value = block(value)
        upsampled.append(backbone.deblocks[index](value) if backbone.deblocks else value)
    value = upsampled[0] if len(upsampled) == 1 else torch.cat(upsampled, dim=1)
    return backbone.deblocks[-1](value) if len(backbone.deblocks) > len(backbone.blocks) else value


def _identity_sttf_and_roi(
    value: torch.Tensor,
    agent_mask: torch.Tensor,
    *,
    use_roi_mask: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mirror V2XTEncoder STTF/ROI for its hard-coded identity correction."""

    batch, agents, height, width, channels = value.shape
    channels_first = value.permute(0, 1, 4, 2, 3)
    if agents > 1:
        non_ego = channels_first[:, 1:].reshape(-1, channels, height, width)
        theta = _identity_theta(batch * (agents - 1), value)
        non_ego = _warp_exportable(
            non_ego,
            theta,
            height=height,
            width=width,
            align_corners=True,
        ).reshape(batch, agents - 1, channels, height, width)
        channels_first = torch.cat((channels_first[:, :1], non_ego), dim=1)
    value = channels_first.permute(0, 1, 3, 4, 2)

    if use_roi_mask:
        ones = value.new_ones((batch * agents, 1, height, width))
        roi = _warp_exportable(
            ones,
            _identity_theta(batch * agents, value),
            height=height,
            width=width,
            align_corners=True,
            mode="nearest",
        ).reshape(batch, agents, 1, height, width)
        roi = roi * agent_mask.reshape(batch, agents, 1, 1, 1).to(roi)
        communication_mask = roi.permute(0, 3, 4, 2, 1)
    else:
        communication_mask = agent_mask.reshape(batch, 1, 1, agents, 1).to(value)
    return value, communication_mask


def _v2xvit_transformer_export(
    transformer: nn.Module,
    value: torch.Tensor,
    agent_mask: torch.Tensor,
) -> torch.Tensor:
    encoder = transformer.encoder
    prior_encoding = value[..., -3:]
    value = value[..., :-3]
    if bool(getattr(encoder, "use_RTE", False)):
        raise RuntimeError("v2xvit_export_rte_not_supported")
    value, communication_mask = _identity_sttf_and_roi(
        value,
        agent_mask,
        use_roi_mask=bool(getattr(encoder, "use_roi_mask", False)),
    )
    for attention, feed_forward in encoder.layers:
        value = attention(value, mask=communication_mask, prior_encoding=prior_encoding)
        value = feed_forward(value) + value
    return value[:, 0]


class HEALLiDARV2XViTPostScatter(nn.Module):
    """V2X-ViT graph beginning at the shared dense BEV boundary."""

    def __init__(self, model: nn.Module, policy: HealV2XViTPostScatterPolicy) -> None:
        super().__init__()
        self.model = model
        self.policy = policy

    def forward(
        self,
        spatial_features: torch.Tensor,
        pairwise_t_matrix: torch.Tensor,
        agent_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        modality = self.policy.modality
        feature = _base_bev_backbone(
            getattr(self.model, f"backbone_{modality}"), spatial_features
        )
        feature = getattr(self.model, f"shrinker_{modality}")(feature)
        if bool(getattr(self.model, "compress", False)):
            feature = self.model.compressor(feature)
        affine = _normalize_pairwise(
            pairwise_t_matrix,
            height_m=float(self.model.H),
            width_m=float(self.model.W),
            ratio=float(self.model.fake_voxel_size),
        )
        height, width = int(feature.shape[-2]), int(feature.shape[-1])
        prior = feature.new_zeros((self.policy.max_agents, 3, height, width))
        feature_with_prior = torch.cat((feature, prior), dim=1)
        feature_with_prior = _warp_exportable(
            feature_with_prior,
            affine[0, 0, : self.policy.max_agents],
            height=height,
            width=width,
            align_corners=False,
        )
        transformer_input = feature_with_prior.unsqueeze(0).permute(0, 1, 3, 4, 2)
        fused = _v2xvit_transformer_export(
            self.model.fusion_net.fusion_net,
            transformer_input,
            agent_mask,
        ).permute(0, 3, 1, 2)
        if bool(getattr(self.model, "shrink_flag", False)):
            fused = self.model.shrink_conv(fused)
        outputs = {
            "cls_preds": self.model.cls_head(fused),
            "reg_preds": self.model.reg_head(fused),
            "dir_preds": self.model.dir_head(fused),
        }
        return tuple(outputs[name] for name in self.policy.output_names)


def build_heal_v2xvit_post_scatter_export_module(
    model: nn.Module,
    *,
    policy: HealV2XViTPostScatterPolicy,
) -> HEALLiDARV2XViTPostScatter:
    if str(getattr(model, "ego_modality", policy.modality)) != policy.modality:
        raise RuntimeError("v2xvit_export_modality_mismatch")
    return HEALLiDARV2XViTPostScatter(model, policy)


__all__ = [
    "HEALLiDARV2XViTPostScatter",
    "HealV2XViTPostScatterPolicy",
    "build_heal_v2xvit_post_scatter_export_module",
]
