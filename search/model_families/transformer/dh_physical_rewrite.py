"""Model-aware physical Q/K/V/O head-dimension rewrites.

The rewrite never masks or pads weights.  Projection rows, output-projection
columns, biases, and V2X-ViT relation tensors are replaced by smaller
Parameters whose shapes encode the requested logical ``d_h``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re
from typing import Any, Mapping

import torch
from torch import nn

from .dh_pruning_contract import AttentionFamilyRecord, HeadLocalMask, stable_hash


def _replace(model: nn.Module, path: str, replacement: nn.Module) -> None:
    if "." in path:
        parent_path, leaf = path.rsplit(".", 1)
        parent = model.get_submodule(parent_path)
    else:
        parent, leaf = model, path
    if leaf.isdigit() and isinstance(parent, (nn.ModuleList, nn.Sequential)):
        parent[int(leaf)] = replacement
    else:
        setattr(parent, leaf, replacement)


def _kind(path: str) -> str:
    if ".window_attention.fn" in path:
        return "window"
    if ".grid_attention.fn" in path:
        return "grid"
    if ".pwmsa." in path:
        return "spatial_window"
    if path.endswith(".fn"):
        return "agent_relation"
    return "unknown"


def discover_attention_families(model_name: str, model: nn.Module) -> tuple[AttentionFamilyRecord, ...]:
    records: list[AttentionFamilyRecord] = []
    if model_name == "lidar_cobevt":
        grouped: dict[str, list[tuple[str, nn.Module]]] = {"window": [], "grid": []}
        for path, module in model.named_modules():
            if (
                path.startswith("fusion_net")
                and module.__class__.__name__ == "Attention"
                and hasattr(module, "to_qkv")
            ):
                grouped[_kind(path)].append((path, module))
        for kind, rows in grouped.items():
            if not rows:
                continue
            source = rows[0][1]
            heads = int(source.heads)
            d_h = int(source.to_qkv.out_features) // 3 // heads
            records.append(
                AttentionFamilyRecord(
                    model=model_name,
                    family_id=f"cobevt_{kind}_h{heads}_d{d_h}",
                    attention_kind=kind,
                    module_paths=tuple(path for path, _ in rows),
                    heads=heads,
                    original_d_h=d_h,
                    embed_dim=int(source.to_qkv.in_features),
                    window_size=tuple(int(value) for value in source.window_size),
                    shared_dependency="fused_qkv_rows_and_output_projection_columns",
                )
            )
    elif model_name == "lidar_v2xvit":
        agent = [
            (path, module)
            for path, module in model.named_modules()
            if module.__class__.__name__ == "HGTCavAttention"
            and hasattr(module, "q_linears")
        ]
        if agent:
            source = agent[0][1]
            heads = int(source.heads)
            d_h = int(source.q_linears[0].out_features) // heads
            records.append(
                AttentionFamilyRecord(
                    model=model_name,
                    family_id=f"v2xvit_agent_relation_h{heads}_d{d_h}",
                    attention_kind="agent_relation",
                    module_paths=tuple(path for path, _ in agent),
                    heads=heads,
                    original_d_h=d_h,
                    embed_dim=int(source.q_linears[0].in_features),
                    shared_dependency="relation_att_and_relation_msg_both_local_axes",
                )
            )
        grouped: dict[tuple[int, int, int, bool], list[tuple[str, nn.Module]]] = {}
        for path, module in model.named_modules():
            if module.__class__.__name__ != "BaseWindowAttention" or not hasattr(module, "to_qkv"):
                continue
            heads = int(module.heads)
            d_h = int(module.to_qkv.out_features) // 3 // heads
            key = (heads, d_h, int(module.window_size), bool(module.relative_pos_embedding))
            grouped.setdefault(key, []).append((path, module))
        for (heads, d_h, window, relative), rows in sorted(grouped.items()):
            source = rows[0][1]
            records.append(
                AttentionFamilyRecord(
                    model=model_name,
                    family_id=f"v2xvit_spatial_window_w{window}_h{heads}_d{d_h}",
                    attention_kind="spatial_window",
                    module_paths=tuple(path for path, _ in rows),
                    heads=heads,
                    original_d_h=d_h,
                    embed_dim=int(source.to_qkv.in_features),
                    window_size=window,
                    shared_dependency="fused_qkv_rows_and_output_projection_columns",
                )
            )
    else:
        raise ValueError(f"unsupported_transformer_model:{model_name}")
    if not records:
        raise RuntimeError(f"attention_family_inventory_empty:{model_name}")
    return tuple(records)


def _slice_linear_output(module: nn.Linear, indices: tuple[int, ...]) -> nn.Linear:
    index = torch.as_tensor(indices, dtype=torch.long, device=module.weight.device)
    replacement = nn.Linear(
        int(module.in_features), len(indices), bias=module.bias is not None,
        device=module.weight.device, dtype=module.weight.dtype,
    )
    with torch.no_grad():
        replacement.weight.copy_(module.weight.index_select(0, index))
        if module.bias is not None:
            replacement.bias.copy_(module.bias.index_select(0, index))
    replacement.train(module.training)
    return replacement


def _slice_linear_input(module: nn.Linear, indices: tuple[int, ...]) -> nn.Linear:
    index = torch.as_tensor(indices, dtype=torch.long, device=module.weight.device)
    replacement = nn.Linear(
        len(indices), int(module.out_features), bias=module.bias is not None,
        device=module.weight.device, dtype=module.weight.dtype,
    )
    with torch.no_grad():
        replacement.weight.copy_(module.weight.index_select(1, index))
        if module.bias is not None:
            replacement.bias.copy_(module.bias)
    replacement.train(module.training)
    return replacement


class PrunedV2XWindowAttention(nn.Module):
    """Physical BaseWindowAttention with arbitrary integer head width."""

    def __init__(self, source: nn.Module, mask: HeadLocalMask) -> None:
        super().__init__()
        self.heads = int(source.heads)
        self.d_qk = int(mask.target_d_h)
        self.d_v = int(mask.target_d_h)
        self.scale = self.d_qk ** -0.5
        self.window_size = int(source.window_size)
        self.relative_pos_embedding = bool(source.relative_pos_embedding)
        projection = int(source.to_qkv.out_features) // 3
        flat = mask.flattened()
        self.q_proj = _slice_linear_output_from_fused(source.to_qkv, flat, 0)
        self.k_proj = _slice_linear_output_from_fused(source.to_qkv, flat, projection)
        self.v_proj = _slice_linear_output_from_fused(source.to_qkv, flat, 2 * projection)
        self.out_proj = _slice_linear_input(source.to_out[0], flat)
        self.output_dropout = nn.Dropout(float(source.to_out[1].p))
        if self.relative_pos_embedding:
            self.relative_indices = source.relative_indices.detach().cpu().clone()
        self.pos_embedding = nn.Parameter(source.pos_embedding.detach().clone())
        self.train(source.training)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, agents, height, width, _ = x.shape
        new_h, new_w = height // self.window_size, width // self.window_size

        def reshape(value: torch.Tensor) -> torch.Tensor:
            return value.reshape(
                batch, agents, new_h, self.window_size, new_w, self.window_size,
                self.heads, self.d_qk,
            ).permute(0, 1, 6, 2, 4, 3, 5, 7).reshape(
                batch, agents, self.heads, new_h * new_w,
                self.window_size * self.window_size, self.d_qk,
            )

        q, k, v = reshape(self.q_proj(x)), reshape(self.k_proj(x)), reshape(self.v_proj(x))
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
            batch, agents, self.heads, new_h, new_w, self.window_size,
            self.window_size, self.d_v,
        ).permute(0, 1, 3, 5, 4, 6, 2, 7).reshape(
            batch, agents, height, width, self.heads * self.d_v
        )
        return self.output_dropout(self.out_proj(output))


def _slice_linear_output_from_fused(
    module: nn.Linear, local_indices: tuple[int, ...], offset: int
) -> nn.Linear:
    return _slice_linear_output(module, tuple(int(offset) + value for value in local_indices))


def _prune_hgt(module: nn.Module, mask: HeadLocalMask) -> None:
    flat = mask.flattened()
    for name in ("q_linears", "k_linears", "v_linears"):
        source = getattr(module, name)
        setattr(module, name, nn.ModuleList([_slice_linear_output(value, flat) for value in source]))
    module.a_linears = nn.ModuleList([_slice_linear_input(value, flat) for value in module.a_linears])
    original = int(mask.original_d_h)
    target = int(mask.target_d_h)
    relation_indices = torch.stack(
        [torch.as_tensor(row, dtype=torch.long, device=module.relation_att.device) for row in mask.keep_by_head]
    )

    def relation_slice(parameter: nn.Parameter) -> nn.Parameter:
        values = []
        for head in range(mask.heads):
            index = relation_indices[head]
            values.append(parameter[:, head].index_select(1, index).index_select(2, index))
        selected = torch.stack(values, dim=1)
        return nn.Parameter(selected.clone(), requires_grad=parameter.requires_grad)

    module.relation_att = relation_slice(module.relation_att)
    module.relation_msg = relation_slice(module.relation_msg)
    module.d_qk = target
    module.d_v = target
    module.original_d_h = original
    module.scale = target ** -0.5


@dataclass(frozen=True)
class PhysicalRewriteReport:
    model: str
    family_id: str
    target_d_h: int
    original_parameter_count: int
    physical_parameter_count: int
    operations: tuple[dict[str, Any], ...]
    state_dict_shape_hash: str
    structure_hash: str
    passed: bool
    issues: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class JointPhysicalRewriteReport:
    model: str
    target_d_h_by_family: dict[str, int]
    original_parameter_count: int
    physical_parameter_count: int
    operations: tuple[dict[str, Any], ...]
    state_dict_shape_hash: str
    structure_hash: str
    passed: bool
    issues: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def state_dict_shape_hash(model: nn.Module) -> str:
    return stable_hash({name: {"shape": list(value.shape), "dtype": str(value.dtype)} for name, value in model.state_dict().items()})


def materialize_family_head_dimension(
    model_name: str,
    model: nn.Module,
    family: AttentionFamilyRecord,
    masks: Mapping[str, HeadLocalMask],
) -> PhysicalRewriteReport:
    """Physically shrink one family and identity-decompose other fused families."""

    original_count = sum(int(value.numel()) for value in model.parameters())
    operations: list[dict[str, Any]] = []
    issues: list[str] = []
    selected = set(family.module_paths)
    all_families = discover_attention_families(model_name, model)
    full_masks: dict[str, HeadLocalMask] = {}
    for current in all_families:
        for path in current.module_paths:
            if path in selected:
                full_masks[path] = masks[path]
            else:
                full_masks[path] = HeadLocalMask(
                    path,
                    current.original_d_h,
                    tuple(tuple(range(current.original_d_h)) for _ in range(current.heads)),
                )

    if model_name == "lidar_cobevt":
        from search.model_families.lidar_cobevt.attention_dim_pruning import (
            AttentionDimMask,
            materialize_attention_bottleneck,
        )
        converted = {
            path: AttentionDimMask(
                mask.keep_by_head, mask.keep_by_head,
                original_d_qk=mask.original_d_h, original_d_v=mask.original_d_h,
            )
            for path, mask in full_masks.items()
        }
        report = materialize_attention_bottleneck(model, converted)
        operations.extend(report.operations)
        issues.extend(str(value) for value in report.issues)
    else:
        original_rows = [(path, module) for path, module in model.named_modules()]
        for path, module in original_rows:
            if module.__class__.__name__ == "HGTCavAttention" and path in full_masks:
                before = int(module.q_linears[0].out_features) // int(module.heads)
                _prune_hgt(module, full_masks[path])
                operations.append({"module": path, "kind": "agent_relation", "before_d_h": before, "after_d_h": full_masks[path].target_d_h})
            elif module.__class__.__name__ == "BaseWindowAttention" and path in full_masks:
                before = int(module.to_qkv.out_features) // 3 // int(module.heads)
                replacement = PrunedV2XWindowAttention(module, full_masks[path]).to(
                    device=module.to_qkv.weight.device, dtype=module.to_qkv.weight.dtype
                )
                _replace(model, path, replacement)
                operations.append({"module": path, "kind": "spatial_window", "before_d_h": before, "after_d_h": full_masks[path].target_d_h})

    modules = dict(model.named_modules())
    for path in selected:
        module = modules[path]
        mask = full_masks[path]
        if int(getattr(module, "heads", -1)) != int(family.heads):
            issues.append(f"head_count_changed:{path}")
        if not math.isclose(float(module.scale), mask.target_d_h ** -0.5, rel_tol=0.0, abs_tol=1e-12):
            issues.append(f"scale_mismatch:{path}")
        if module.__class__.__name__ == "HGTCavAttention":
            if any(int(value.out_features) != family.heads * mask.target_d_h for value in module.q_linears):
                issues.append(f"hgt_q_shape_mismatch:{path}")
            if tuple(module.relation_att.shape[-2:]) != (mask.target_d_h, mask.target_d_h):
                issues.append(f"relation_att_shape_mismatch:{path}")
            if tuple(module.relation_msg.shape[-2:]) != (mask.target_d_h, mask.target_d_h):
                issues.append(f"relation_msg_shape_mismatch:{path}")
        else:
            for projection in (module.q_proj, module.k_proj, module.v_proj):
                if int(projection.out_features) != family.heads * mask.target_d_h:
                    issues.append(f"projection_shape_mismatch:{path}")
            if int(module.out_proj.in_features) != family.heads * mask.target_d_h:
                issues.append(f"out_shape_mismatch:{path}")
    physical_count = sum(int(value.numel()) for value in model.parameters())
    if int(family.original_d_h) != int(next(iter(masks.values())).target_d_h) and physical_count >= original_count:
        issues.append("parameter_count_did_not_decrease")
    structure_payload = {
        "model": model_name,
        "family": family.to_dict(),
        "masks": {path: value.to_dict() for path, value in sorted(full_masks.items())},
        "state_dict_shape_hash": state_dict_shape_hash(model),
    }
    return PhysicalRewriteReport(
        model=model_name,
        family_id=family.family_id,
        target_d_h=next(iter(masks.values())).target_d_h,
        original_parameter_count=original_count,
        physical_parameter_count=physical_count,
        operations=tuple(operations),
        state_dict_shape_hash=structure_payload["state_dict_shape_hash"],
        structure_hash=stable_hash(structure_payload),
        passed=not issues,
        issues=tuple(issues),
    )


def materialize_joint_head_dimensions(
    model_name: str,
    model: nn.Module,
    masks_by_family: Mapping[str, Mapping[str, HeadLocalMask]],
) -> JointPhysicalRewriteReport:
    """Physically rewrite several independently discovered families at once.

    This is the Phase-B path.  Every unselected family is decomposed at its
    original width, matching the topology used by the Phase-A baselines.
    """

    original_count = sum(int(value.numel()) for value in model.parameters())
    families = discover_attention_families(model_name, model)
    by_id = {family.family_id: family for family in families}
    unknown = sorted(set(masks_by_family) - set(by_id))
    if unknown:
        raise ValueError(f"joint_attention_family_missing:{unknown}")
    if not masks_by_family:
        raise ValueError("joint_attention_family_selection_empty")
    full_masks: dict[str, HeadLocalMask] = {}
    selected_paths: set[str] = set()
    target_by_family: dict[str, int] = {}
    for family in families:
        supplied = masks_by_family.get(family.family_id)
        if supplied is not None:
            if set(supplied) != set(family.module_paths):
                raise ValueError(f"joint_attention_mask_paths_mismatch:{family.family_id}")
            targets = {int(mask.target_d_h) for mask in supplied.values()}
            if len(targets) != 1:
                raise ValueError(f"joint_attention_family_width_mismatch:{family.family_id}")
            target_by_family[family.family_id] = next(iter(targets))
            selected_paths.update(family.module_paths)
        for path in family.module_paths:
            full_masks[path] = (
                supplied[path]
                if supplied is not None
                else HeadLocalMask(
                    path,
                    family.original_d_h,
                    tuple(tuple(range(family.original_d_h)) for _ in range(family.heads)),
                )
            )
    operations: list[dict[str, Any]] = []
    issues: list[str] = []
    if model_name == "lidar_cobevt":
        from search.model_families.lidar_cobevt.attention_dim_pruning import (
            AttentionDimMask,
            materialize_attention_bottleneck,
        )

        converted = {
            path: AttentionDimMask(
                mask.keep_by_head,
                mask.keep_by_head,
                original_d_qk=mask.original_d_h,
                original_d_v=mask.original_d_h,
            )
            for path, mask in full_masks.items()
        }
        report = materialize_attention_bottleneck(model, converted)
        operations.extend(report.operations)
        issues.extend(str(value) for value in report.issues)
    else:
        original_rows = [(path, module) for path, module in model.named_modules()]
        for path, module in original_rows:
            if module.__class__.__name__ == "HGTCavAttention" and path in full_masks:
                before = int(module.q_linears[0].out_features) // int(module.heads)
                _prune_hgt(module, full_masks[path])
                operations.append({"module": path, "kind": "agent_relation", "before_d_h": before, "after_d_h": full_masks[path].target_d_h})
            elif module.__class__.__name__ == "BaseWindowAttention" and path in full_masks:
                before = int(module.to_qkv.out_features) // 3 // int(module.heads)
                replacement = PrunedV2XWindowAttention(module, full_masks[path]).to(
                    device=module.to_qkv.weight.device, dtype=module.to_qkv.weight.dtype
                )
                _replace(model, path, replacement)
                operations.append({"module": path, "kind": "spatial_window", "before_d_h": before, "after_d_h": full_masks[path].target_d_h})
    modules = dict(model.named_modules())
    family_by_path = {
        path: family
        for family in families
        for path in family.module_paths
    }
    for path in selected_paths:
        module = modules[path]
        family = family_by_path[path]
        mask = full_masks[path]
        if int(getattr(module, "heads", -1)) != int(family.heads):
            issues.append(f"head_count_changed:{path}")
        if not math.isclose(float(module.scale), mask.target_d_h ** -0.5, rel_tol=0.0, abs_tol=1e-12):
            issues.append(f"scale_mismatch:{path}")
        if module.__class__.__name__ == "HGTCavAttention":
            if any(int(value.out_features) != family.heads * mask.target_d_h for value in module.q_linears):
                issues.append(f"hgt_q_shape_mismatch:{path}")
            if tuple(module.relation_att.shape[-2:]) != (mask.target_d_h, mask.target_d_h):
                issues.append(f"relation_att_shape_mismatch:{path}")
            if tuple(module.relation_msg.shape[-2:]) != (mask.target_d_h, mask.target_d_h):
                issues.append(f"relation_msg_shape_mismatch:{path}")
        else:
            if any(int(value.out_features) != family.heads * mask.target_d_h for value in (module.q_proj, module.k_proj, module.v_proj)):
                issues.append(f"projection_shape_mismatch:{path}")
            if int(module.out_proj.in_features) != family.heads * mask.target_d_h:
                issues.append(f"out_shape_mismatch:{path}")
    physical_count = sum(int(value.numel()) for value in model.parameters())
    if any(target_by_family[name] < by_id[name].original_d_h for name in target_by_family) and physical_count >= original_count:
        issues.append("joint_parameter_count_did_not_decrease")
    shape_hash = state_dict_shape_hash(model)
    structure_payload = {
        "model": model_name,
        "families": {name: by_id[name].to_dict() for name in sorted(target_by_family)},
        "target_d_h_by_family": target_by_family,
        "masks": {path: value.to_dict() for path, value in sorted(full_masks.items())},
        "state_dict_shape_hash": shape_hash,
        "recipe": "transformer-unified-qkv-dh-joint-v1",
    }
    return JointPhysicalRewriteReport(
        model=model_name,
        target_d_h_by_family=target_by_family,
        original_parameter_count=original_count,
        physical_parameter_count=physical_count,
        operations=tuple(operations),
        state_dict_shape_hash=shape_hash,
        structure_hash=stable_hash(structure_payload),
        passed=not issues,
        issues=tuple(issues),
    )


__all__ = [
    "JointPhysicalRewriteReport",
    "PhysicalRewriteReport",
    "PrunedV2XWindowAttention",
    "discover_attention_families",
    "materialize_family_head_dimension",
    "materialize_joint_head_dimensions",
    "state_dict_shape_hash",
]
