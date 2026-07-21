"""Fixed-K, fixed-agent ONNX adapter for HEAL LiDAR V2X-ViT.

This adapter is deliberately separate from the accepted LiDAR-pyramid path.
It preserves the V2X-ViT grid-sampling semantics while replacing the
non-exportable inverse of a *known identity* STTF correction with an explicit
identity sampling grid.  It is an ONNX smoke path, not yet a production
TensorRT or Q/DQ path.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as functional

from quantization.export.heal_lidar_pyramid import DynamicPointPillarScatterTRT


@dataclass(frozen=True)
class HealV2XViTExportPolicy:
    """Static single-engine input contract.

    ``fixed_k`` has no default on purpose: a caller must derive it from the
    frozen calibration/evaluation manifest rather than inheriting the
    LiDAR-pyramid value or the YAML preprocessing ceiling.
    """

    fixed_k: int
    max_agents: int = 2
    modality: str = "m1"
    output_names: tuple[str, ...] = ("cls_preds", "reg_preds", "dir_preds")
    calibration_manifest_hash: str | None = None

    def __post_init__(self) -> None:
        if int(self.fixed_k) <= 0:
            raise ValueError("v2xvit_fixed_k_must_be_positive")
        if int(self.max_agents) <= 0:
            raise ValueError("v2xvit_max_agents_must_be_positive")
        if not self.output_names:
            raise ValueError("v2xvit_output_names_must_not_be_empty")

    @classmethod
    def from_frozen_train_manifest(cls, path: str | Path) -> "HealV2XViTExportPolicy":
        from search.model_family.calibration_manifest import load_v2xvit_train_manifest

        manifest = load_v2xvit_train_manifest(path)
        contract = manifest["input_contract"]
        return cls(
            fixed_k=int(manifest["fixed_k_contract"]["value"]),
            max_agents=int(contract["max_agents"]),
            modality=str(contract["modality"]),
            calibration_manifest_hash=str(manifest["manifest_hash"]),
        )


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


class HEALLiDARV2XViTFixedK(nn.Module):
    """Six-input, fixed-K and fixed-agent export wrapper for V2X-ViT."""

    def __init__(self, model: nn.Module, policy: HealV2XViTExportPolicy) -> None:
        super().__init__()
        self.model = model
        self.policy = policy

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
        feature = _base_bev_backbone(getattr(self.model, f"backbone_{modality}"), feature)
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


def build_heal_v2xvit_export_module(
    model: nn.Module,
    *,
    policy: HealV2XViTExportPolicy,
) -> HEALLiDARV2XViTFixedK:
    if str(getattr(model, "ego_modality", policy.modality)) != policy.modality:
        raise RuntimeError("v2xvit_export_modality_mismatch")
    return HEALLiDARV2XViTFixedK(model, policy)


def prepare_v2xvit_fixed_k_inputs(
    ego_batch: Mapping[str, Any],
    *,
    policy: HealV2XViTExportPolicy,
) -> dict[str, torch.Tensor]:
    source = ego_batch[f"inputs_{policy.modality}"]
    features = source["voxel_features"].float()
    coordinates = source["voxel_coords"].to(torch.int32)
    counts = source["voxel_num_points"].to(torch.int32)
    voxel_count = int(features.shape[0])
    if voxel_count > policy.fixed_k:
        raise RuntimeError(f"v2xvit_real_voxel_count_exceeds_fixed_k:{voxel_count}:{policy.fixed_k}")
    padding = policy.fixed_k - voxel_count
    features = functional.pad(features, (0, 0, 0, 0, 0, padding))
    coordinates = functional.pad(coordinates, (0, 0, 0, padding))
    counts = functional.pad(counts, (0, padding), value=1)
    valid = features.new_zeros((policy.fixed_k,))
    valid[:voxel_count] = 1.0

    record_len = int(ego_batch["record_len"][0].item())
    if record_len > policy.max_agents:
        raise RuntimeError(
            f"v2xvit_record_len_exceeds_max_agents:{record_len}:{policy.max_agents}"
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
    "HEALLiDARV2XViTFixedK",
    "HealV2XViTExportPolicy",
    "build_heal_v2xvit_export_module",
    "prepare_v2xvit_fixed_k_inputs",
]
