"""Numerical reconstruction and metrics for CoBEVT Attention captures."""

from __future__ import annotations

import hashlib
from typing import Any, Iterable, Mapping

import torch


def derive_attention_tensors(
    *,
    qkv: torch.Tensor,
    heads: int,
    scale: float,
    relative_bias: torch.Tensor | None,
    attention_mask: torch.Tensor | None,
) -> dict[str, torch.Tensor]:
    if qkv.ndim != 3:
        raise ValueError(f"qkv_must_be_rank3:{tuple(qkv.shape)}")
    if qkv.shape[-1] % (3 * int(heads)):
        raise ValueError("qkv_width_not_divisible_by_three_heads")
    q_raw, k_raw, v_raw = qkv.chunk(3, dim=-1)
    batch, tokens, width = q_raw.shape
    head_dim = width // int(heads)

    def split_heads(value: torch.Tensor) -> torch.Tensor:
        return value.reshape(batch, tokens, heads, head_dim).permute(0, 2, 1, 3)

    q = split_heads(q_raw)
    k = split_heads(k_raw)
    v = split_heads(v_raw)
    scaled_q = q * float(scale)
    qk_score = torch.matmul(scaled_q, k.transpose(-1, -2))
    logits = qk_score
    if relative_bias is not None:
        bias = relative_bias.to(device=logits.device, dtype=logits.dtype)
        if bias.ndim != 3 or tuple(bias.shape) != (heads, tokens, tokens):
            raise ValueError(
                f"relative_bias_shape_mismatch:{tuple(bias.shape)}:"
                f"expected={(heads, tokens, tokens)}"
            )
        logits = logits + bias.unsqueeze(0)
    masked_logits = logits
    if attention_mask is not None:
        keep = attention_mask.to(device=logits.device, dtype=torch.bool)
        try:
            masked_logits = logits.masked_fill(~keep, -float("inf"))
        except RuntimeError as exc:
            raise ValueError(
                f"attention_mask_not_broadcastable:{tuple(keep.shape)}:"
                f"{tuple(logits.shape)}"
            ) from exc
    probability = torch.softmax(masked_logits, dim=-1)
    av = torch.matmul(probability, v)
    return {
        "q": q,
        "k": k,
        "v": v,
        "scaled_q": scaled_q,
        "qk_score": qk_score,
        "masked_logits": masked_logits,
        "probability": probability,
        "av": av,
    }


def tensor_statistics(tensor: torch.Tensor) -> dict[str, Any]:
    value = tensor.detach()
    if not value.is_floating_point():
        value = value.float()
    flat = value.reshape(-1)
    finite_mask = torch.isfinite(flat)
    finite = flat[finite_mask]
    tiny = torch.finfo(value.dtype).tiny
    subnormal = finite.ne(0) & finite.abs().lt(tiny)
    count = max(int(flat.numel()), 1)
    if finite.numel():
        result = {
            "min": float(finite.min()),
            "max": float(finite.max()),
            "abs_max": float(finite.abs().max()),
            "mean": float(finite.double().mean()),
            "std": float(finite.double().std(unbiased=False)),
        }
    else:
        result = {key: float("nan") for key in ("min", "max", "abs_max", "mean", "std")}
    return {
        **result,
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "numel": int(tensor.numel()),
        "zero_ratio": float(flat.eq(0).sum()) / count,
        "subnormal_ratio": float(subnormal.sum()) / count,
        "finite_ratio": float(finite_mask.sum()) / count,
    }


def tensor_content_hash(tensor: torch.Tensor) -> str:
    value = tensor.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def validate_capture_inventory(
    rows: Iterable[Mapping[str, Any]], *, expected_frames: int
) -> dict[str, Any]:
    values = tuple(rows)
    modules = tuple(sorted({str(row["module_name"]) for row in values}))
    frames = tuple(sorted({str(row["frame_id"]) for row in values}))
    expected_modules = tuple(
        f"fusion_net.layers.{layer}.{kind}_attention.fn"
        for layer in range(3)
        for kind in ("window", "grid")
    )
    if modules != tuple(sorted(expected_modules)):
        raise ValueError(f"capture_attention_block_incomplete:{modules}")
    if len(frames) != int(expected_frames):
        raise ValueError(f"capture_frame_count_mismatch:{len(frames)}:{expected_frames}")
    expected_pairs = {(frame, module) for frame in frames for module in modules}
    actual_pairs = {(str(row["frame_id"]), str(row["module_name"])) for row in values}
    if actual_pairs != expected_pairs:
        raise ValueError("capture_attention_block_incomplete")
    return {
        "module_count": len(modules),
        "frame_count": len(frames),
        "capture_count": len(actual_pairs),
        "modules": list(modules),
        "frames": list(frames),
    }


__all__ = [
    "derive_attention_tensors",
    "tensor_content_hash",
    "tensor_statistics",
    "validate_capture_inventory",
]
