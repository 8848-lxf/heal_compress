"""Export adapters for the original HEAL DAIR LiDAR baseline families.

The original HEAL forwards use Python ``record_len`` loops and dictionary
inputs.  This module supplies one fixed-K/fixed-max-agent tensor contract for
the single-scale Max/Attention/Disco/CoBEVT families and the multi-scale
CoAlign family.  Invalid padded agents are explicitly masked, so a one-agent
sample remains numerically equivalent to the original variable-agent forward.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as functional

from quantization.export.heal_lidar_pyramid import DynamicPointPillarScatterTRT

from .heal_v2xvit import _base_bev_backbone, _normalize_pairwise, _warp_exportable


@dataclass(frozen=True)
class HealLidarBaselineExportPolicy:
    fixed_k: int
    max_agents: int = 2
    modality: str = "m1"
    output_names: tuple[str, ...] = ("cls_preds", "reg_preds", "dir_preds")

    def __post_init__(self) -> None:
        if int(self.fixed_k) <= 0:
            raise ValueError("baseline_fixed_k_must_be_positive")
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
    logits = fusion.pixel_weight_layer(torch.cat((warped, ego), dim=1))
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


class HEALLiDARBaselineFixedK(nn.Module):
    """Six-input export wrapper for non-pyramid, non-V2XViT baselines."""

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
        voxel_features: torch.Tensor,
        voxel_coords: torch.Tensor,
        voxel_num_points: torch.Tensor,
        pairwise_t_matrix: torch.Tensor,
        valid_voxel_mask: torch.Tensor,
        agent_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        modality = self.policy.modality
        encoder = getattr(self.model, f"encoder_{modality}")
        safe_counts = torch.where(
            valid_voxel_mask.reshape(-1) > 0.5,
            voxel_num_points,
            torch.ones_like(voxel_num_points),
        )
        encoded = encoder.pillar_vfe(
            {
                "voxel_features": voxel_features,
                "voxel_coords": voxel_coords,
                "voxel_num_points": safe_counts,
            }
        )
        feature = DynamicPointPillarScatterTRT.apply(
            encoded["pillar_features"],
            voxel_coords,
            valid_voxel_mask,
            pairwise_t_matrix,
            int(encoder.scatter.ny),
            int(encoder.scatter.nx),
        )
        backbone = getattr(self.model, f"backbone_{modality}")
        if type(self.model).__name__ == "HeterModelBaselineMs":
            feature = backbone({"spatial_features": feature})["spatial_features_2d"]
            feature = getattr(self.model, f"aligner_{modality}")(feature)
        else:
            feature = _base_bev_backbone(backbone, feature)
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
                current = self.model.backbone.get_layer_i_feature(current, layer_i=index)
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


def build_heal_lidar_baseline_export_module(
    model: nn.Module,
    *,
    policy: HealLidarBaselineExportPolicy,
) -> HEALLiDARBaselineFixedK:
    return HEALLiDARBaselineFixedK(model, policy)


def prepare_heal_lidar_baseline_inputs(
    ego_batch: Mapping[str, Any],
    *,
    policy: HealLidarBaselineExportPolicy,
) -> dict[str, torch.Tensor]:
    source = ego_batch[f"inputs_{policy.modality}"]
    features = source["voxel_features"].float()
    coordinates = source["voxel_coords"].to(torch.int32)
    counts = source["voxel_num_points"].to(torch.int32)
    voxel_count = int(features.shape[0])
    if voxel_count > policy.fixed_k:
        raise RuntimeError(
            f"baseline_real_voxel_count_exceeds_fixed_k:{voxel_count}:{policy.fixed_k}"
        )
    padding = int(policy.fixed_k) - voxel_count
    features = functional.pad(features, (0, 0, 0, 0, 0, padding))
    coordinates = functional.pad(coordinates, (0, 0, 0, padding))
    counts = functional.pad(counts, (0, padding), value=1)
    valid = features.new_zeros((int(policy.fixed_k),))
    valid[:voxel_count] = 1.0

    record_len = int(ego_batch["record_len"][0].item())
    if record_len > policy.max_agents:
        raise RuntimeError(
            f"baseline_record_len_exceeds_max_agents:{record_len}:{policy.max_agents}"
        )
    raw_pairwise = ego_batch["pairwise_t_matrix"].float()
    pairwise = torch.eye(4, dtype=raw_pairwise.dtype, device=raw_pairwise.device)
    pairwise = pairwise.reshape(1, 1, 1, 4, 4).repeat(
        1, policy.max_agents, policy.max_agents, 1, 1
    )
    pairwise[:, :record_len, :record_len] = raw_pairwise[:, :record_len, :record_len]
    agent_mask = features.new_zeros((1, policy.max_agents))
    agent_mask[:, :record_len] = 1.0
    return {
        "voxel_features": features.contiguous(),
        "voxel_coords": coordinates.contiguous(),
        "voxel_num_points": counts.contiguous(),
        "pairwise_t_matrix": pairwise.contiguous(),
        "valid_voxel_mask": valid.contiguous(),
        "agent_mask": agent_mask.contiguous(),
    }


__all__ = [
    "HEALLiDARBaselineFixedK",
    "HealLidarBaselineExportPolicy",
    "build_heal_lidar_baseline_export_module",
    "prepare_heal_lidar_baseline_inputs",
]
