"""Family-specific fixed-K export wrapper for HEAL LiDAR CoBEVT."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as functional

from ..exceptions import OnnxExportError

POINTPILLAR_SCATTER_PLUGIN_OP = "PointPillarScatterTRT"
POINTPILLAR_SCATTER_ONNX_DOMAIN = "trt"


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
    spatial = pillar_features.new_zeros((num_agents, channels, height, width))
    for agent in range(num_agents):
        selected = coordinates[:, 0] == agent
        if bool(selected.any()):
            rows = coordinates[selected, 2].clamp(0, height - 1)
            columns = coordinates[selected, 3].clamp(0, width - 1)
            spatial[agent, :, rows, columns] = features[selected].transpose(0, 1)
    return spatial


class CobevtPointPillarScatterTRT(torch.autograd.Function):
    """Reference scatter with the accepted TensorRT plugin symbolic."""

    @staticmethod
    def forward(
        ctx: Any,
        pillar_features: torch.Tensor,
        voxel_coords: torch.Tensor,
        valid_voxel_mask: torch.Tensor,
        pairwise_t_matrix: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
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
    def symbolic(
        graph: Any,
        pillar_features: Any,
        voxel_coords: Any,
        valid_voxel_mask: Any,
        pairwise_t_matrix: Any,
        height: int,
        width: int,
    ) -> Any:
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


def scatter_export_capability(*, boundary_dtype: str) -> dict[str, Any]:
    normalized = str(boundary_dtype).strip().upper()
    if normalized not in {"FP32", "FP16"}:
        raise ValueError("scatter_boundary_must_be_fp16_or_fp32")
    return {
        "op_type": POINTPILLAR_SCATTER_PLUGIN_OP,
        "onnx_domain": POINTPILLAR_SCATTER_ONNX_DOMAIN,
        "boundary_dtype": normalized,
        "quantization_gene": False,
        "int8_allowed": False,
    }


def _normalize_pairwise(
    pairwise: torch.Tensor,
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


def _warp_agents(feature: torch.Tensor, affine: torch.Tensor) -> torch.Tensor:
    height, width = int(feature.shape[-2]), int(feature.shape[-1])
    theta = affine[0, 0, : feature.shape[0]]
    xs = (
        torch.arange(width, device=feature.device, dtype=feature.dtype) + 0.5
    ) * (2.0 / float(width)) - 1.0
    ys = (
        torch.arange(height, device=feature.device, dtype=feature.dtype) + 0.5
    ) * (2.0 / float(height)) - 1.0
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    base = torch.stack((xx, yy, torch.ones_like(xx)), dim=-1).reshape(
        1, height * width, 3
    )
    flat = base.expand(theta.shape[0], -1, -1)
    grid = torch.bmm(flat, theta.transpose(1, 2)).reshape(
        theta.shape[0], height, width, 2
    )
    return functional.grid_sample(feature, grid, align_corners=False)


class HEALLiDARCoBEVTSignalMaxK(nn.Module):
    """Six-input CoBEVT graph with fixed K and explicit real-agent length."""

    def __init__(
        self,
        model: nn.Module,
        *,
        modality: str = "m1",
        output_names: Sequence[str] = ("cls_preds", "reg_preds", "dir_preds"),
        fixed_k: int,
        max_cav: int = 2,
    ) -> None:
        super().__init__()
        if fixed_k <= 0 or max_cav <= 0:
            raise ValueError("fixed_k_and_max_cav_must_be_positive")
        self.model = model
        self.modality = str(modality)
        self.output_names = tuple(str(value) for value in output_names)
        self.fixed_k = int(fixed_k)
        self.max_cav = int(max_cav)

    def forward(
        self,
        voxel_features: torch.Tensor,
        voxel_coords: torch.Tensor,
        voxel_num_points: torch.Tensor,
        pairwise_t_matrix: torch.Tensor,
        valid_voxel_mask: torch.Tensor,
        record_len: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        encoder = getattr(self.model, f"encoder_{self.modality}")
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
        feature = CobevtPointPillarScatterTRT.apply(
            encoded["pillar_features"],
            voxel_coords,
            valid_voxel_mask,
            pairwise_t_matrix,
            int(encoder.scatter.ny),
            int(encoder.scatter.nx),
        )
        backbone = getattr(self.model, f"backbone_{self.modality}")
        feature = backbone({"spatial_features": feature})["spatial_features_2d"]
        feature = getattr(self.model, f"shrinker_{self.modality}")(feature)
        affine = _normalize_pairwise(
            pairwise_t_matrix,
            float(self.model.H),
            float(self.model.W),
            float(self.model.fake_voxel_size),
        )
        feature = _warp_agents(feature, affine)
        height, width = int(feature.shape[-2]), int(feature.shape[-1])
        valid_agents = (
            torch.arange(self.max_cav, device=record_len.device)
            < record_len.reshape(-1)[0]
        )
        communication_mask = valid_agents.reshape(1, 1, 1, 1, self.max_cav)
        communication_mask = communication_mask.expand(
            1, height, width, 1, self.max_cav
        )
        fused = feature.unsqueeze(0)
        for stage in self.model.fusion_net.layers:
            fused = stage(fused, mask=communication_mask)
        fused = self.model.fusion_net.mlp_head(fused)
        outputs = {
            "cls_preds": self.model.cls_head(fused),
            "reg_preds": self.model.reg_head(fused),
            "dir_preds": self.model.dir_head(fused),
        }
        return tuple(outputs[name] for name in self.output_names)


def prepare_cobevt_maxk_inputs(
    ego_batch: Mapping[str, Any],
    *,
    fixed_k: int,
    max_cav: int = 2,
    modality: str = "m1",
) -> dict[str, torch.Tensor]:
    source = ego_batch[f"inputs_{modality}"]
    features = source["voxel_features"].float()
    coordinates = source["voxel_coords"].to(torch.int32)
    counts = source["voxel_num_points"].to(torch.int32)
    count = int(features.shape[0])
    if count > fixed_k:
        raise OnnxExportError(
            f"real voxel count {count} exceeds CoBEVT fixed K={fixed_k}"
        )
    record_len = ego_batch["record_len"].reshape(-1).to(torch.int32)
    if record_len.numel() != 1 or int(record_len[0]) > max_cav:
        raise OnnxExportError("CoBEVT export currently requires batch=1 and record_len<=max_cav")
    padding = fixed_k - count
    features = functional.pad(features, (0, 0, 0, 0, 0, padding))
    coordinates = functional.pad(coordinates, (0, 0, 0, padding))
    counts = functional.pad(counts, (0, padding), value=1)
    mask = features.new_zeros((fixed_k,))
    mask[:count] = 1.0

    source_pairwise = ego_batch["pairwise_t_matrix"].float()
    pairwise = torch.eye(
        4, dtype=source_pairwise.dtype, device=source_pairwise.device
    ).reshape(1, 1, 1, 4, 4).repeat(1, max_cav, max_cav, 1, 1)
    rows = min(int(source_pairwise.shape[1]), max_cav)
    columns = min(int(source_pairwise.shape[2]), max_cav)
    pairwise[:, :rows, :columns] = source_pairwise[:, :rows, :columns]
    return {
        "voxel_features": features.contiguous(),
        "voxel_coords": coordinates.contiguous(),
        "voxel_num_points": counts.contiguous(),
        "pairwise_t_matrix": pairwise.contiguous(),
        "valid_voxel_mask": mask.contiguous(),
        "record_len": record_len.contiguous(),
    }


__all__ = [
    "CobevtPointPillarScatterTRT",
    "HEALLiDARCoBEVTSignalMaxK",
    "prepare_cobevt_maxk_inputs",
    "scatter_export_capability",
]
