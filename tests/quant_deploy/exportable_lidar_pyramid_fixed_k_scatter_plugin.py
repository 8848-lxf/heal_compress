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
from exportable_lidar_pyramid_padded_agent import (
    _weighted_fuse_padded_single_batch,
    point_pillar_scatter_fixed_agents,
)


SOURCE_FIXED_K_SCATTER_PLUGIN_WRAPPER = str(Path(__file__).resolve())
POINTPILLAR_SCATTER_PLUGIN_OP = "PointPillarScatterTRT"
POINTPILLAR_SCATTER_PLUGIN_VERSION = "1"
POINTPILLAR_SCATTER_PLUGIN_NAMESPACE = ""


def fixed_k_scatter_plugin_input_names() -> list[str]:
    return [
        "voxel_features",
        "voxel_coords",
        "voxel_num_points",
        "valid_agent_mask",
        "pairwise_t_matrix",
        "valid_voxel_mask",
    ]


def fixed_k_scatter_plugin_dynamic_axes(output_names: list[str]) -> dict[str, dict[int, str]]:
    axes = {
        "voxel_features": {0: "fixed_k_num_voxels"},
        "voxel_coords": {0: "fixed_k_num_voxels"},
        "voxel_num_points": {0: "fixed_k_num_voxels"},
        "valid_voxel_mask": {0: "fixed_k_num_voxels"},
        "valid_agent_mask": {0: "batch"},
        "pairwise_t_matrix": {0: "batch", 1: "max_cav", 2: "max_cav"},
    }
    for name in output_names:
        axes[name] = {0: "batch"}
    return axes


def safe_voxel_num_points_for_fixed_k(voxel_num_points: torch.Tensor, valid_voxel_mask: torch.Tensor) -> torch.Tensor:
    mask = valid_voxel_mask.reshape(-1).to(device=voxel_num_points.device)
    return torch.where(mask > 0.5, voxel_num_points, torch.ones_like(voxel_num_points))


def point_pillar_scatter_fixed_agents_valid_voxel_mask(
    pillar_features: torch.Tensor,
    coords: torch.Tensor,
    valid_voxel_mask: torch.Tensor,
    *,
    num_agents: int,
    num_bev_features: int,
    nx: int,
    ny: int,
    nz: int = 1,
) -> torch.Tensor:
    valid = valid_voxel_mask.reshape(-1).to(device=pillar_features.device) > 0.5
    return point_pillar_scatter_fixed_agents(
        pillar_features[valid],
        coords[valid],
        num_agents=int(num_agents),
        num_bev_features=int(num_bev_features),
        nx=int(nx),
        ny=int(ny),
        nz=int(nz),
        valid_agent_mask=None,
    )


class PointPillarScatterTRTFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pillar_features, voxel_coords, valid_voxel_mask, num_agents: int, height: int, width: int):
        del ctx
        return point_pillar_scatter_fixed_agents_valid_voxel_mask(
            pillar_features,
            voxel_coords,
            valid_voxel_mask,
            num_agents=int(num_agents),
            num_bev_features=int(pillar_features.shape[1]),
            nx=int(width),
            ny=int(height),
            nz=1,
        )

    @staticmethod
    def symbolic(g, pillar_features, voxel_coords, valid_voxel_mask, num_agents: int, height: int, width: int):
        return g.op(
            POINTPILLAR_SCATTER_PLUGIN_OP,
            pillar_features,
            voxel_coords,
            valid_voxel_mask,
            num_agents_i=int(num_agents),
            height_i=int(height),
            width_i=int(width),
            plugin_version_s=POINTPILLAR_SCATTER_PLUGIN_VERSION,
            plugin_namespace_s=POINTPILLAR_SCATTER_PLUGIN_NAMESPACE,
        )


def point_pillar_scatter_trt_export(
    pillar_features: torch.Tensor,
    voxel_coords: torch.Tensor,
    valid_voxel_mask: torch.Tensor,
    *,
    num_agents: int,
    height: int,
    width: int,
) -> torch.Tensor:
    return PointPillarScatterTRTFn.apply(
        pillar_features,
        voxel_coords,
        valid_voxel_mask,
        int(num_agents),
        int(height),
        int(width),
    )


def _encoder_fixed_k_scatter_plugin(
    encoder: nn.Module,
    voxel_features: torch.Tensor,
    voxel_coords: torch.Tensor,
    voxel_num_points: torch.Tensor,
    valid_agent_mask: torch.Tensor,
    valid_voxel_mask: torch.Tensor,
    max_cav: int,
) -> torch.Tensor:
    batch_dict = {
        "voxel_features": voxel_features,
        "voxel_coords": voxel_coords,
        "voxel_num_points": safe_voxel_num_points_for_fixed_k(voxel_num_points, valid_voxel_mask),
    }
    batch_dict = encoder.pillar_vfe(batch_dict)
    spatial = point_pillar_scatter_trt_export(
        batch_dict["pillar_features"],
        voxel_coords,
        valid_voxel_mask,
        num_agents=int(max_cav),
        height=int(encoder.scatter.ny),
        width=int(encoder.scatter.nx),
    )
    mask = valid_agent_mask.reshape(-1)[: int(max_cav)].to(device=spatial.device, dtype=spatial.dtype)
    return spatial * mask.view(int(max_cav), 1, 1, 1)


class ExportableLidarPyramidFixedKScatterPlugin(nn.Module):
    """Export-only fixed max_cav lidar_pyramid wrapper with safe fixed-K scatter plugin."""

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
        self.wrapper_source = SOURCE_FIXED_K_SCATTER_PLUGIN_WRAPPER

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
            raise RuntimeError(f"fixed-K scatter plugin export expects 3 pyramid levels, got {num_levels}.")

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
        valid_voxel_mask: torch.Tensor,
    ):
        feature = _encoder_fixed_k_scatter_plugin(
            self._encoder(),
            voxel_features,
            voxel_coords,
            voxel_num_points,
            valid_agent_mask,
            valid_voxel_mask,
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


def check_fixed_k_scatter_plugin_wrapper_equivalence(
    wrapper: ExportableLidarPyramidFixedKScatterPlugin,
    tensors: tuple[torch.Tensor, ...],
    reference_output: dict[str, Any],
    output_names: list[str],
) -> dict[str, Any]:
    with torch.no_grad():
        outputs = wrapper(*tensors)
    report: dict[str, Any] = {"wrapper_source": SOURCE_FIXED_K_SCATTER_PLUGIN_WRAPPER, "same_shape": True, "outputs": {}}
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
