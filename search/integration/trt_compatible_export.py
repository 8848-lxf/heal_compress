"""TensorRT-compatible signal-maxK wrapper for search integration.

This adapter keeps the formal ``export_pruned_signal_maxk_onnx`` API in use
while matching the dynamic-single-engine graph that existing TensorRT 10.9
artifacts successfully build.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def _ensure_quant_deploy_path() -> None:
    root = Path(__file__).resolve().parents[2]
    tests = root / "tests" / "quant_deploy"
    if str(tests) not in sys.path:
        sys.path.insert(0, str(tests))


_ensure_quant_deploy_path()

from exportable_lidar_pyramid import (  # type: ignore  # noqa: E402
    _bev_backbone_static,
    _decode_multiscale_static,
    _normalize_pairwise_exportable,
    _resnet_static_features,
    _weighted_fuse_single_batch,
)
from exportable_lidar_pyramid_dynamic_fixed_k_scatter_plugin import (  # type: ignore  # noqa: E402
    point_pillar_scatter_dynamic_fixed_agents_valid_voxel_mask,
)
from exportable_lidar_pyramid_fixed_k_scatter_plugin import (  # type: ignore  # noqa: E402
    POINTPILLAR_SCATTER_PLUGIN_NAMESPACE,
    POINTPILLAR_SCATTER_PLUGIN_OP,
    POINTPILLAR_SCATTER_PLUGIN_VERSION,
    safe_voxel_num_points_for_fixed_k,
)


class SearchDynamicPointPillarScatterTRT(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, pillar_features: torch.Tensor, voxel_coords: torch.Tensor, valid_voxel_mask: torch.Tensor, pairwise_t_matrix: torch.Tensor, height: int, width: int) -> torch.Tensor:
        del ctx
        return point_pillar_scatter_dynamic_fixed_agents_valid_voxel_mask(
            pillar_features,
            voxel_coords,
            valid_voxel_mask,
            num_agents=int(pairwise_t_matrix.shape[1]),
            num_bev_features=int(pillar_features.shape[1]),
            nx=int(width),
            ny=int(height),
            nz=1,
        )

    @staticmethod
    def symbolic(graph: Any, pillar_features: Any, voxel_coords: Any, valid_voxel_mask: Any, pairwise_t_matrix: Any, height: int, width: int) -> Any:
        return graph.op(
            f"trt::{POINTPILLAR_SCATTER_PLUGIN_OP}",
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


class SearchTensorRTCompatibleLidarPyramid(nn.Module):
    def __init__(self, model: nn.Module, modality_name: str, output_names: list[str], *, fixed_k: int) -> None:
        super().__init__()
        self.model = model
        self.modality_name = str(modality_name)
        self.output_names = list(output_names)
        self.fixed_k = int(fixed_k)

    def _encoder(self) -> nn.Module:
        return getattr(self.model, f"encoder_{self.modality_name}")

    def _backbone(self) -> nn.Module:
        return getattr(self.model, f"backbone_{self.modality_name}")

    def _aligner(self) -> nn.Module:
        return getattr(self.model, f"aligner_{self.modality_name}")

    def _pyramid_forward_static(self, spatial_features: torch.Tensor, affine_matrix: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        pyramid = self.model.pyramid_backbone
        features = _resnet_static_features(pyramid.resnet, spatial_features)
        if len(features) != 3:
            raise RuntimeError(f"dynamic single-engine maxK export expects 3 pyramid levels, got {len(features)}")
        occ0 = pyramid.single_head_0(features[0])
        occ1 = pyramid.single_head_1(features[1])
        occ2 = pyramid.single_head_2(features[2])
        align_corners = bool(getattr(pyramid, "align_corners", False))
        fused0 = _weighted_fuse_single_batch(features[0], torch.sigmoid(occ0) + 1.0e-4, affine_matrix, align_corners)
        fused1 = _weighted_fuse_single_batch(features[1], torch.sigmoid(occ1) + 1.0e-4, affine_matrix, align_corners)
        fused2 = _weighted_fuse_single_batch(features[2], torch.sigmoid(occ2) + 1.0e-4, affine_matrix, align_corners)
        return _decode_multiscale_static(pyramid, (fused0, fused1, fused2)), (occ0, occ1, occ2)

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
                "voxel_num_points": safe_voxel_num_points_for_fixed_k(voxel_num_points, valid_voxel_mask),
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


def build_search_trt_compatible_export_module(model: nn.Module, *, output_names: tuple[str, ...], fixed_k: int, modality: str = "m1") -> SearchTensorRTCompatibleLidarPyramid:
    return SearchTensorRTCompatibleLidarPyramid(model, modality, list(output_names), fixed_k=fixed_k)


def make_pointpillar_domain_compatible(input_onnx: str | Path, output_onnx: str | Path) -> dict[str, Any]:
    import onnx

    source = Path(input_onnx)
    destination = Path(output_onnx)
    model = onnx.load(str(source))
    changed = []
    for node in model.graph.node:
        if node.op_type == POINTPILLAR_SCATTER_PLUGIN_OP and node.domain:
            changed.append({"node": node.name, "old_domain": node.domain, "new_domain": ""})
            node.domain = ""
    keep = [opset for opset in model.opset_import if opset.domain != "trt"]
    del model.opset_import[:]
    model.opset_import.extend(keep)
    onnx.save(model, str(destination))
    return {"input": str(source), "output": str(destination), "changed_nodes": changed}
