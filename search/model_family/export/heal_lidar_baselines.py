"""Post-scatter export adapters for the HEAL DAIR LiDAR baseline families.

Dynamic voxelization, PFN and scatter stay outside TensorRT. Invalid padded
agents are masked so a one-agent sample remains numerically equivalent to the
original variable-agent forward.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as functional

from .heal_v2xvit import _base_bev_backbone, _normalize_pairwise, _warp_exportable


@dataclass(frozen=True)
class HealLidarBaselineExportPolicy:
    max_agents: int = 2
    modality: str = "m1"
    output_names: tuple[str, ...] = ("cls_preds", "reg_preds", "dir_preds")

    def __post_init__(self) -> None:
        if int(self.max_agents) <= 0:
            raise ValueError("baseline_max_agents_must_be_positive")


def _masked_warp(
    feature: torch.Tensor,
    affine: torch.Tensor,
    agent_mask: torch.Tensor,
) -> torch.Tensor:
    height, width = int(feature.shape[-2]), int(feature.shape[-1])
    warped = _warp_exportable(
        feature,
        affine[0, 0],
        height=height,
        width=width,
        align_corners=False,
    )
    return warped * agent_mask.reshape(-1, 1, 1, 1).to(warped)


def _max_fuse(warped: torch.Tensor, agent_mask: torch.Tensor) -> torch.Tensor:
    valid = agent_mask.reshape(-1, 1, 1, 1) > 0.5
    masked = torch.where(valid, warped, torch.full_like(warped, -1.0e20))
    return torch.amax(masked, dim=0, keepdim=True)


def _attention_fuse(warped: torch.Tensor, agent_mask: torch.Tensor) -> torch.Tensor:
    agents, channels, height, width = warped.shape
    value = warped.reshape(agents, channels, height * width).permute(2, 0, 1)
    score = torch.bmm(value, value.transpose(1, 2)) / math.sqrt(float(channels))
    key_valid = agent_mask.reshape(1, 1, agents) > 0.5
    score = torch.where(key_valid, score, torch.full_like(score, -1.0e20))
    attention = torch.softmax(score, dim=-1)
    context = torch.bmm(attention, value)
    return context[:, 0].transpose(0, 1).reshape(1, channels, height, width)


def _disco_fuse(
    fusion: nn.Module,
    warped: torch.Tensor,
    ego_feature: torch.Tensor,
    agent_mask: torch.Tensor,
) -> torch.Tensor:
    agents, channels, height, width = warped.shape
    ego = ego_feature[:1].expand(agents, -1, -1, -1)
    pixel = fusion.pixel_weight_layer
    value = torch.cat((warped, ego), dim=1)
    # HEAL's PixelWeightLayer.forward begins with a redundant
    # ``view(-1, x.size(-3), x.size(-2), x.size(-1))``.  PyTorch 2.0's ONNX
    # symbolic for a negative-dimension ``size`` can incorrectly construct a
    # scalar Slice bound (``len() of a 0-d tensor``).  The export wrapper
    # already supplies a canonical NCHW tensor, so spell out the exact four
    # registered layers and preserve module provenance without that no-op.
    if all(
        hasattr(pixel, name)
        for name in (
            "conv1_1",
            "bn1_1",
            "conv1_2",
            "bn1_2",
            "conv1_3",
            "bn1_3",
            "conv1_4",
        )
    ):
        value = functional.relu(pixel.bn1_1(pixel.conv1_1(value)))
        value = functional.relu(pixel.bn1_2(pixel.conv1_2(value)))
        value = functional.relu(pixel.bn1_3(pixel.conv1_3(value)))
        logits = functional.relu(pixel.conv1_4(value))
    else:
        logits = pixel(value)
    valid = agent_mask.reshape(agents, 1, 1, 1) > 0.5
    logits = torch.where(valid, logits, torch.full_like(logits, -1.0e20))
    weights = torch.softmax(logits, dim=0)
    return torch.sum(weights.expand(-1, channels, -1, -1) * warped, dim=0, keepdim=True)


def _cobevt_fuse(
    fusion: nn.Module,
    warped: torch.Tensor,
    agent_mask: torch.Tensor,
) -> torch.Tensor:
    value = warped.unsqueeze(0)
    _, _, _, height, width = value.shape
    merge_mask = agent_mask.reshape(1, 1, 1, 1, -1).expand(1, height, width, 1, -1)
    for stage in fusion.layers:
        value = stage(value, mask=merge_mask)
    return fusion.mlp_head(value)


def _fuse(
    fusion: nn.Module,
    feature: torch.Tensor,
    affine: torch.Tensor,
    agent_mask: torch.Tensor,
) -> torch.Tensor:
    warped = _masked_warp(feature, affine, agent_mask)
    kind = type(fusion).__name__
    if kind == "MaxFusion":
        return _max_fuse(warped, agent_mask)
    if kind == "AttFusion":
        return _attention_fuse(warped, agent_mask)
    if kind == "DiscoFusion":
        return _disco_fuse(fusion, warped, feature, agent_mask)
    if kind == "CoBEVT":
        return _cobevt_fuse(fusion, warped, agent_mask)
    raise RuntimeError(f"baseline_export_fusion_not_supported:{kind}")


class HEALLiDARBaselinePostScatter(nn.Module):
    """F-Cooper/Disco graph with dynamic voxelization, PFN and scatter external."""

    def __init__(self, model: nn.Module, policy: HealLidarBaselineExportPolicy) -> None:
        super().__init__()
        self.model = model
        self.policy = policy
        kind = type(model).__name__
        if kind not in {"HeterModelBaseline", "HeterModelBaselineMs"}:
            raise RuntimeError(f"baseline_export_model_not_supported:{kind}")
        if kind == "HeterModelBaseline" and type(model.fusion_net).__name__ == "V2XViTFusion":
            raise RuntimeError("baseline_export_v2xvit_requires_specialized_adapter")

    def forward(
        self,
        spatial_features: torch.Tensor,
        pairwise_t_matrix: torch.Tensor,
        agent_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        modality = self.policy.modality
        backbone = getattr(self.model, f"backbone_{modality}")
        if type(self.model).__name__ == "HeterModelBaselineMs":
            feature = backbone({"spatial_features": spatial_features})[
                "spatial_features_2d"
            ]
            feature = getattr(self.model, f"aligner_{modality}")(feature)
        else:
            feature = _base_bev_backbone(backbone, spatial_features)
            feature = getattr(self.model, f"shrinker_{modality}")(feature)
            if bool(getattr(self.model, "compress", False)):
                feature = self.model.compressor(feature)
        mask = agent_mask.reshape(-1)
        feature = feature * mask.reshape(-1, 1, 1, 1).to(feature)
        affine = _normalize_pairwise(
            pairwise_t_matrix,
            height_m=float(self.model.H),
            width_m=float(self.model.W),
            ratio=float(self.model.fake_voxel_size),
        )
        if type(self.model).__name__ == "HeterModelBaselineMs":
            levels = [feature]
            current = feature
            for index in range(1, len(self.model.fusion_net)):
                current = self.model.backbone.get_layer_i_feature(
                    current, layer_i=index
                )
                levels.append(current)
            fused_levels = [
                _fuse(fusion, level, affine, mask)
                for fusion, level in zip(self.model.fusion_net, levels)
            ]
            fused = self.model.backbone.decode_multiscale_feature(fused_levels)
        else:
            fused = _fuse(self.model.fusion_net, feature, affine, mask)
        if bool(getattr(self.model, "shrink_flag", False)):
            fused = self.model.shrink_conv(fused)
        outputs = {
            "cls_preds": self.model.cls_head(fused),
            "reg_preds": self.model.reg_head(fused),
            "dir_preds": self.model.dir_head(fused),
        }
        return tuple(outputs[name] for name in self.policy.output_names)


def build_heal_lidar_baseline_post_scatter_export_module(
    model: nn.Module,
    *,
    policy: HealLidarBaselineExportPolicy,
) -> HEALLiDARBaselinePostScatter:
    return HEALLiDARBaselinePostScatter(model, policy)


__all__ = [
    "HEALLiDARBaselinePostScatter",
    "HealLidarBaselineExportPolicy",
    "build_heal_lidar_baseline_post_scatter_export_module",
]
