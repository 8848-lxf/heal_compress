from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from dynamic_single_engine_maxk_common import FIXED_K, single_engine_dynamic_axes, single_engine_input_names
from exportable_lidar_pyramid import (
    _bev_backbone_static,
    _decode_multiscale_static,
    _normalize_pairwise_exportable,
    _resnet_static_features,
    _weighted_fuse_single_batch,
)
from exportable_lidar_pyramid_fixed_k_scatter_plugin import (
    POINTPILLAR_SCATTER_PLUGIN_NAMESPACE,
    POINTPILLAR_SCATTER_PLUGIN_OP,
    POINTPILLAR_SCATTER_PLUGIN_VERSION,
    safe_voxel_num_points_for_fixed_k,
)
from exportable_lidar_pyramid_dynamic_fixed_k_scatter_plugin import point_pillar_scatter_dynamic_fixed_agents_valid_voxel_mask


SOURCE_DYNAMIC_SINGLE_ENGINE_MAXK_WRAPPER = str(Path(__file__).resolve())


def dynamic_single_engine_maxk_input_names() -> list[str]:
    return single_engine_input_names()


def dynamic_single_engine_maxk_dynamic_axes(output_names: list[str]) -> dict[str, dict[int, str]]:
    return single_engine_dynamic_axes(output_names)


class DynamicNPointPillarScatterTRTFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pillar_features, voxel_coords, valid_voxel_mask, pairwise_t_matrix, height: int, width: int):
        del ctx
        num_agents = int(pairwise_t_matrix.shape[1])
        return point_pillar_scatter_dynamic_fixed_agents_valid_voxel_mask(
            pillar_features,
            voxel_coords,
            valid_voxel_mask,
            num_agents=num_agents,
            num_bev_features=int(pillar_features.shape[1]),
            nx=int(width),
            ny=int(height),
            nz=1,
        )

    @staticmethod
    def symbolic(g, pillar_features, voxel_coords, valid_voxel_mask, pairwise_t_matrix, height: int, width: int):
        return g.op(
            POINTPILLAR_SCATTER_PLUGIN_OP,
            pillar_features,
            voxel_coords,
            valid_voxel_mask,
            pairwise_t_matrix,
            num_agents_i=0,
            height_i=int(height),
            width_i=int(width),
            plugin_version_s=POINTPILLAR_SCATTER_PLUGIN_VERSION,
            plugin_namespace_s=POINTPILLAR_SCATTER_PLUGIN_NAMESPACE,
        )


def point_pillar_scatter_trt_dynamic_n_export(
    pillar_features: torch.Tensor,
    voxel_coords: torch.Tensor,
    valid_voxel_mask: torch.Tensor,
    pairwise_t_matrix: torch.Tensor,
    *,
    height: int,
    width: int,
) -> torch.Tensor:
    return DynamicNPointPillarScatterTRTFn.apply(
        pillar_features,
        voxel_coords,
        valid_voxel_mask,
        pairwise_t_matrix,
        int(height),
        int(width),
    )


def _encoder_dynamic_single_engine_maxk(
    encoder: nn.Module,
    voxel_features: torch.Tensor,
    voxel_coords: torch.Tensor,
    voxel_num_points: torch.Tensor,
    valid_voxel_mask: torch.Tensor,
    pairwise_t_matrix: torch.Tensor,
) -> torch.Tensor:
    batch_dict = {
        "voxel_features": voxel_features,
        "voxel_coords": voxel_coords,
        "voxel_num_points": safe_voxel_num_points_for_fixed_k(voxel_num_points, valid_voxel_mask),
    }
    batch_dict = encoder.pillar_vfe(batch_dict)
    return point_pillar_scatter_trt_dynamic_n_export(
        batch_dict["pillar_features"],
        voxel_coords,
        valid_voxel_mask,
        pairwise_t_matrix,
        height=int(encoder.scatter.ny),
        width=int(encoder.scatter.nx),
    )


class ExportableLidarPyramidDynamicSingleEngineMaxK(nn.Module):
    """Export-only dynamic-agent single-engine wrapper with fixed maxK scatter plugin input."""

    def __init__(
        self,
        model: nn.Module,
        modality_name: str,
        output_names: list[str],
        *,
        fixed_k: int = FIXED_K,
    ) -> None:
        super().__init__()
        self.model = model
        self.modality_name = modality_name
        self.output_names = list(output_names)
        self.fixed_k = int(fixed_k)
        self.wrapper_source = SOURCE_DYNAMIC_SINGLE_ENGINE_MAXK_WRAPPER

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
            raise RuntimeError(f"dynamic single-engine maxK export expects 3 pyramid levels, got {num_levels}.")

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
        pairwise_t_matrix: torch.Tensor,
        valid_voxel_mask: torch.Tensor,
    ):
        feature = _encoder_dynamic_single_engine_maxk(
            self._encoder(),
            voxel_features,
            voxel_coords,
            voxel_num_points,
            valid_voxel_mask,
            pairwise_t_matrix,
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
        return tuple(outputs[name] for name in self.output_names)


def check_dynamic_single_engine_maxk_wrapper_equivalence(
    wrapper: ExportableLidarPyramidDynamicSingleEngineMaxK,
    tensors: tuple[torch.Tensor, ...],
    reference_output: dict[str, Any],
    output_names: list[str],
) -> dict[str, Any]:
    with torch.no_grad():
        outputs = wrapper(*tensors)
    report: dict[str, Any] = {
        "wrapper_source": SOURCE_DYNAMIC_SINGLE_ENGINE_MAXK_WRAPPER,
        "fixed_K": int(wrapper.fixed_k),
        "same_shape": True,
        "outputs": {},
    }
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
