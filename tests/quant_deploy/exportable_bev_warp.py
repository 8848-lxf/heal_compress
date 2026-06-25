from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn.functional as F


def make_base_grid_2d(
    out_h: int,
    out_w: int,
    device: torch.device,
    dtype: torch.dtype,
    align_corners: bool = False,
) -> torch.Tensor:
    """Create a homogeneous normalized output grid without F.affine_grid.

    Returns a tensor with shape ``[1, out_h, out_w, 3]`` and last dimension
    ``[x_norm, y_norm, 1]``. The normalized coordinates match PyTorch
    ``affine_grid`` for static H/W.
    """
    if align_corners:
        if out_w > 1:
            xs = torch.linspace(-1.0, 1.0, out_w, device=device, dtype=dtype)
        else:
            xs = torch.zeros((out_w,), device=device, dtype=dtype)
        if out_h > 1:
            ys = torch.linspace(-1.0, 1.0, out_h, device=device, dtype=dtype)
        else:
            ys = torch.zeros((out_h,), device=device, dtype=dtype)
    else:
        xs = (torch.arange(out_w, device=device, dtype=dtype) + 0.5) * (2.0 / float(out_w)) - 1.0
        ys = (torch.arange(out_h, device=device, dtype=dtype) + 0.5) * (2.0 / float(out_h)) - 1.0
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    ones = torch.ones_like(xx)
    return torch.stack((xx, yy, ones), dim=-1).unsqueeze(0)


def affine_grid_2d_exportable(
    theta: torch.Tensor,
    out_h: int,
    out_w: int,
    align_corners: bool = False,
) -> torch.Tensor:
    """ONNX-exportable replacement for ``F.affine_grid`` for 2D grids."""
    if theta.ndim != 3 or theta.shape[1:] != (2, 3):
        raise ValueError(f"theta must have shape [N, 2, 3], got {tuple(theta.shape)}")
    base_grid = make_base_grid_2d(out_h, out_w, theta.device, theta.dtype, align_corners)
    flat_grid = base_grid.reshape(1, out_h * out_w, 3).expand(theta.shape[0], -1, -1)
    grid = torch.bmm(flat_grid, theta.transpose(1, 2))
    return grid.reshape(theta.shape[0], out_h, out_w, 2)


def warp_affine_simple_exportable(
    src: torch.Tensor,
    M: torch.Tensor,
    dsize: tuple[int, int],
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool = False,
) -> torch.Tensor:
    """HEAL-compatible BEV warp that avoids ``F.affine_grid`` during export."""
    grid = affine_grid_2d_exportable(M, int(dsize[0]), int(dsize[1]), align_corners=align_corners)
    return F.grid_sample(
        src,
        grid.to(dtype=src.dtype, device=src.device),
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )


def check_exportable_bev_warp_equivalence(
    src: torch.Tensor | None = None,
    theta: torch.Tensor | None = None,
    dsize: tuple[int, int] = (6, 8),
    align_corners: bool = False,
    original_warp: Callable[..., torch.Tensor] | None = None,
) -> dict[str, Any]:
    """Compare the exportable warp against the original affine-grid path."""
    if src is None:
        src = torch.randn(2, 3, int(dsize[0]), int(dsize[1]))
    if theta is None:
        theta = torch.tensor(
            [
                [[1.0, 0.0, 0.1], [0.0, 1.0, -0.2]],
                [[0.9, 0.2, 0.0], [-0.2, 0.9, 0.1]],
            ],
            dtype=src.dtype,
            device=src.device,
        )

    if original_warp is not None:
        expected = original_warp(src, theta, dsize, align_corners=align_corners)
    else:
        expected_grid = F.affine_grid(theta, torch.Size((theta.shape[0], src.shape[1], int(dsize[0]), int(dsize[1]))), align_corners=align_corners)
        expected = F.grid_sample(src, expected_grid, align_corners=align_corners)
    actual = warp_affine_simple_exportable(src, theta, dsize, align_corners=align_corners)
    abs_error = (actual - expected).abs()
    max_abs = float(abs_error.max().item())
    mean_abs = float(abs_error.mean().item())
    denom = float(expected.abs().mean().item()) + 1e-12
    return {
        "max_abs_error": max_abs,
        "mean_abs_error": mean_abs,
        "relative_error": mean_abs / denom,
        "align_corners": bool(align_corners),
        "H": int(dsize[0]),
        "W": int(dsize[1]),
        "N": int(theta.shape[0]),
        "C": int(src.shape[1]),
    }
