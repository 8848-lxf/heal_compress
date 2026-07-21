"""Exact Q/K/V projection decomposition for role-addressable quantization."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

import torch
from torch import nn


def _replace(model: nn.Module, path: str, module: nn.Module) -> None:
    parent_path, leaf = path.rsplit(".", 1)
    parent = model.get_submodule(parent_path)
    if leaf.isdigit() and isinstance(parent, (nn.ModuleList, nn.Sequential)):
        parent[int(leaf)] = module
    else:
        setattr(parent, leaf, module)


class SplitV2XWindowAttention(nn.Module):
    """Numerically equivalent BaseWindowAttention with addressable Q/K/V."""

    def __init__(self, source: nn.Module) -> None:
        super().__init__()
        self.heads = int(source.heads)
        self.scale = float(source.scale)
        self.window_size = int(source.window_size)
        self.relative_pos_embedding = bool(source.relative_pos_embedding)
        in_features = int(source.to_qkv.in_features)
        inner = int(source.to_qkv.out_features) // 3
        self.d_qk = inner // self.heads
        self.d_v = self.d_qk
        self.q_proj = nn.Linear(in_features, inner, bias=False)
        self.k_proj = nn.Linear(in_features, inner, bias=False)
        self.v_proj = nn.Linear(in_features, inner, bias=False)
        self.out_proj = nn.Linear(
            int(source.to_out[0].in_features),
            int(source.to_out[0].out_features),
            bias=source.to_out[0].bias is not None,
        )
        self.output_dropout = nn.Dropout(float(source.to_out[1].p))
        if self.relative_pos_embedding:
            # Match the HEAL implementation: these integer lookup indices are
            # deliberately a CPU constant, not a device buffer.  Moving them
            # to CUDA makes the PyTorch ONNX constant-folding pass combine a
            # CUDA index with CPU graph constants and fail export.
            self.relative_indices = source.relative_indices.detach().cpu().clone()
        self.pos_embedding = nn.Parameter(source.pos_embedding.detach().clone())
        with torch.no_grad():
            q, k, v = source.to_qkv.weight.detach().chunk(3, dim=0)
            self.q_proj.weight.copy_(q)
            self.k_proj.weight.copy_(k)
            self.v_proj.weight.copy_(v)
            self.out_proj.weight.copy_(source.to_out[0].weight)
            if self.out_proj.bias is not None:
                self.out_proj.bias.copy_(source.to_out[0].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, agents, height, width, _channels = x.shape
        new_h = height // self.window_size
        new_w = width // self.window_size

        def reshape(value: torch.Tensor) -> torch.Tensor:
            value = value.reshape(
                batch,
                agents,
                new_h,
                self.window_size,
                new_w,
                self.window_size,
                self.heads,
                self.d_qk,
            )
            return value.permute(0, 1, 6, 2, 4, 3, 5, 7).reshape(
                batch,
                agents,
                self.heads,
                new_h * new_w,
                self.window_size * self.window_size,
                self.d_qk,
            )

        q = reshape(self.q_proj(x))
        k = reshape(self.k_proj(x))
        v = reshape(self.v_proj(x))
        score = torch.einsum("blmhic,blmhjc->blmhij", q, k) * self.scale
        if self.relative_pos_embedding:
            score = score + self.pos_embedding[
                self.relative_indices[:, :, 0], self.relative_indices[:, :, 1]
            ]
        else:
            score = score + self.pos_embedding
        probability = torch.softmax(score, dim=-1)
        output = torch.einsum("blmhij,blmhjc->blmhic", probability, v)
        output = output.reshape(
            batch,
            agents,
            self.heads,
            new_h,
            new_w,
            self.window_size,
            self.window_size,
            self.d_v,
        ).permute(0, 1, 3, 5, 4, 6, 2, 7)
        output = output.reshape(batch, agents, height, width, self.heads * self.d_v)
        return self.output_dropout(self.out_proj(output))


@dataclass(frozen=True)
class ProjectionRewriteReport:
    model_family: str
    replacement_count: int
    records: tuple[dict[str, Any], ...]
    semantic_weight_hash: str
    head_dimension_changed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_family": self.model_family,
            "replacement_count": self.replacement_count,
            "records": list(self.records),
            "semantic_weight_hash": self.semantic_weight_hash,
            "head_dimension_changed": self.head_dimension_changed,
            "structure_frozen": not self.head_dimension_changed,
            "rewrite_kind": "exact_fused_qkv_decomposition_for_precision_roles",
        }


def split_v2xvit_fused_qkv(model: nn.Module) -> ProjectionRewriteReport:
    rows = [
        (name, module)
        for name, module in model.named_modules()
        if module.__class__.__name__ == "BaseWindowAttention" and hasattr(module, "to_qkv")
    ]
    records = []
    hash_rows = []
    for name, source in rows:
        replacement = SplitV2XWindowAttention(source).to(
            device=source.to_qkv.weight.device,
            dtype=source.to_qkv.weight.dtype,
        )
        # ``Module.to`` preserves the replacement's freshly-created training
        # mode, not the source module's mode.  These rewrites are commonly
        # applied after the checkpoint model has already been switched to
        # ``eval()``; failing to copy the flag silently re-enables the output
        # Dropout and destroys the supposedly identity rewrite.
        replacement.train(source.training)
        reconstructed = torch.cat(
            (
                replacement.q_proj.weight.detach(),
                replacement.k_proj.weight.detach(),
                replacement.v_proj.weight.detach(),
            ),
            dim=0,
        )
        if not torch.equal(reconstructed, source.to_qkv.weight.detach()):
            raise RuntimeError(f"v2xvit_qkv_decomposition_weight_mismatch:{name}")
        digest = hashlib.sha256(reconstructed.cpu().contiguous().numpy().tobytes()).hexdigest()
        hash_rows.append((name, digest))
        records.append(
            {
                "module_path": name,
                "heads": replacement.heads,
                "d_qk": replacement.d_qk,
                "d_v": replacement.d_v,
                "source_weight_shape": list(source.to_qkv.weight.shape),
                "reconstructed_weight_exact": True,
                "weight_sha256": digest,
            }
        )
        _replace(model, name, replacement)
    if len(rows) != 9:
        raise RuntimeError(f"v2xvit_window_attention_replacement_count:{len(rows)}:9")
    semantic_hash = hashlib.sha256(
        json.dumps(hash_rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return ProjectionRewriteReport(
        model_family="lidar_v2xvit",
        replacement_count=len(rows),
        records=tuple(records),
        semantic_weight_hash=semantic_hash,
    )


def split_cobevt_fused_qkv(model: nn.Module) -> ProjectionRewriteReport:
    from search.model_families.lidar_cobevt.attention_dim_pruning import (
        materialize_attention_bottleneck,
        uniform_attention_masks,
    )

    masks = uniform_attention_masks(model, d_qk=32, d_v=32)
    report = materialize_attention_bottleneck(model, masks)
    if not report.passed or report.original_parameter_count != report.physical_parameter_count:
        raise RuntimeError(f"cobevt_identity_projection_decomposition_failed:{report.issues}")
    records = tuple(
        {
            "module_path": row["module"],
            "heads": 8,
            "d_qk": int(row["after"]["d_qk"]),
            "d_v": int(row["after"]["d_v"]),
            "reconstructed_weight_exact": True,
        }
        for row in report.operations
    )
    return ProjectionRewriteReport(
        model_family="lidar_cobevt",
        replacement_count=len(records),
        records=records,
        semantic_weight_hash=str(report.structure_hash),
    )


__all__ = [
    "ProjectionRewriteReport",
    "SplitV2XWindowAttention",
    "split_cobevt_fused_qkv",
    "split_v2xvit_fused_qkv",
]
