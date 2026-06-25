from __future__ import annotations

from typing import Any, Callable

import torch


def explicit_squeeze_pillar_features(features: torch.Tensor) -> torch.Tensor:
    """Convert final PFN output [M, 1, C] to [M, C] with an explicit axis."""
    if features.ndim != 3 or int(features.shape[1]) != 1:
        raise ValueError(f"expected final PFN features with shape [M, 1, C], got {tuple(features.shape)}")
    return features.squeeze(1)


def pillar_vfe_forward_explicit_squeeze(self, batch_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Export-only PillarVFE.forward replacement.

    This mirrors HEAL/OpenCOOD ``PillarVFE.forward`` and changes only the final
    ``features.squeeze()`` to ``features.squeeze(1)`` so ONNX contains explicit
    Squeeze axes and preserves the voxel batch dimension when M == 1.
    """
    voxel_features, voxel_num_points, coords = (
        batch_dict["voxel_features"],
        batch_dict["voxel_num_points"],
        batch_dict["voxel_coords"],
    )

    points_mean = (
        voxel_features[:, :, :3].sum(dim=1, keepdim=True)
        / voxel_num_points.type_as(voxel_features).view(-1, 1, 1)
    )
    f_cluster = voxel_features[:, :, :3] - points_mean

    f_center = torch.zeros_like(voxel_features[:, :, :3])
    f_center[:, :, 0] = voxel_features[:, :, 0] - (
        coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset
    )
    f_center[:, :, 1] = voxel_features[:, :, 1] - (
        coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset
    )
    f_center[:, :, 2] = voxel_features[:, :, 2] - (
        coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset
    )

    if self.use_absolute_xyz:
        features = [voxel_features, f_cluster, f_center]
    else:
        features = [voxel_features[..., 3:], f_cluster, f_center]

    if self.with_distance:
        points_dist = torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True)
        features.append(points_dist)
    features = torch.cat(features, dim=-1)

    voxel_count = features.shape[1]
    mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
    mask = torch.unsqueeze(mask, -1).type_as(voxel_features)
    features *= mask
    for pfn in self.pfn_layers:
        features = pfn(features)
    features = explicit_squeeze_pillar_features(features)
    batch_dict["pillar_features"] = features
    return batch_dict


def clone_lidar_batch_dict(batch_dict: dict[str, Any]) -> dict[str, Any]:
    cloned: dict[str, Any] = {}
    for key, value in batch_dict.items():
        if torch.is_tensor(value):
            cloned[key] = value.detach().clone()
        else:
            cloned[key] = value
    return cloned


def check_pillar_vfe_export_fix_equivalence(
    pillar_vfe: Any,
    input_batch_dict: dict[str, torch.Tensor],
    original_forward: Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]] | None = None,
) -> dict[str, Any]:
    """Compare original PillarVFE.forward against explicit-squeeze forward."""
    original_forward = original_forward or pillar_vfe.forward
    with torch.no_grad():
        original_out = original_forward(clone_lidar_batch_dict(input_batch_dict))["pillar_features"]
        patched_out = pillar_vfe_forward_explicit_squeeze(pillar_vfe, clone_lidar_batch_dict(input_batch_dict))["pillar_features"]

    same_shape = tuple(original_out.shape) == tuple(patched_out.shape)
    if same_shape:
        abs_error = (original_out - patched_out).abs()
        max_abs = float(abs_error.max().item()) if abs_error.numel() else 0.0
        mean_abs = float(abs_error.mean().item()) if abs_error.numel() else 0.0
        denom = float(original_out.abs().mean().item()) + 1e-12 if original_out.numel() else 1.0
    else:
        max_abs = None
        mean_abs = None
        denom = 1.0
    return {
        "same_shape": bool(same_shape),
        "original_shape": list(original_out.shape),
        "patched_shape": list(patched_out.shape),
        "max_abs_error": max_abs,
        "mean_abs_error": mean_abs,
        "relative_error": (mean_abs / denom) if mean_abs is not None else None,
    }
