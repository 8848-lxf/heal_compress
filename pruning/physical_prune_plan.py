"""Global original-index-space physical prune plan and one-shot surgery."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import torch
import torch.nn as nn

from .grouped_conv import (
    grouped_conv_pruning_fn,
    prune_grouped_conv_d_compact_frontfill_reblock,
    prune_grouped_conv_input_balanced,
    prune_grouped_conv_true_group_block,
)
from .grouped_pergroup8_policy import validate_grouped_input_keep_pergroup8


@dataclass
class ModuleAxisPruneRequest:
    module_name: str
    axis: str
    prune_indices: list[int]
    source_recipe_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    source_recipe_ids: list[str] = field(default_factory=list)

    def normalized(self) -> "ModuleAxisPruneRequest":
        ids = list(self.source_recipe_ids or ([self.source_recipe_id] if self.source_recipe_id else []))
        return ModuleAxisPruneRequest(
            self.module_name,
            self.axis,
            sorted({int(v) for v in self.prune_indices}),
            self.source_recipe_id,
            dict(self.metadata),
            ids,
        )


def concat_offset_transform(indices: Iterable[int], *, offset: int) -> list[int]:
    return [int(offset) + int(idx) for idx in indices]


class GlobalPhysicalPrunePlan:
    def __init__(self) -> None:
        self._requests: dict[tuple[str, str], ModuleAxisPruneRequest] = {}
        self._request_counts: dict[tuple[str, str], int] = {}

    def add_request(self, request: ModuleAxisPruneRequest) -> None:
        request = request.normalized()
        key = (request.module_name, request.axis)
        self._request_counts[key] = self._request_counts.get(key, 0) + 1
        if key not in self._requests:
            self._requests[key] = request
            return
        existing = self._requests[key]
        existing.prune_indices = sorted(set(existing.prune_indices).union(request.prune_indices))
        for rid in request.source_recipe_ids:
            if rid and rid not in existing.source_recipe_ids:
                existing.source_recipe_ids.append(rid)
        existing.metadata.update(request.metadata)

    def add_recipe(self, recipe: Any) -> None:
        for req in getattr(recipe, "requests", []):
            self.add_request(
                ModuleAxisPruneRequest(
                    module_name=req.module_name,
                    axis=req.axis,
                    prune_indices=list(req.prune_indices),
                    source_recipe_id=getattr(recipe, "recipe_id", ""),
                    metadata={"policy": getattr(recipe, "policy", "")},
                )
            )

    def get_request(self, module_name: str, axis: str) -> ModuleAxisPruneRequest:
        return self._requests[(module_name, axis)]

    def requests(self) -> list[ModuleAxisPruneRequest]:
        return [self._requests[key] for key in sorted(self._requests)]

    def apply_residual_closure(self, *, component_id: str, module_axes: list[tuple[str, str]]) -> dict[str, Any]:
        closed: set[int] = set()
        for module_name, axis in module_axes:
            req = self._requests.get((module_name, axis))
            if req:
                closed.update(req.prune_indices)
        for module_name, axis in module_axes:
            self.add_request(
                ModuleAxisPruneRequest(
                    module_name=module_name,
                    axis=axis,
                    prune_indices=sorted(closed),
                    source_recipe_id=f"residual_closure::{component_id}",
                )
            )
        return {"component_id": component_id, "closed_indices": sorted(closed), "module_axes": module_axes}

    def audit(self) -> dict[str, Any]:
        return {
            "num_module_axis_requests": len(self._requests),
            "num_duplicate_module_axis_requests": sum(1 for count in self._request_counts.values() if count > 1),
            "requests": [
                {
                    "module_name": req.module_name,
                    "axis": req.axis,
                    "prune_indices": req.prune_indices,
                    "source_recipe_ids": req.source_recipe_ids,
                    "metadata": req.metadata,
                }
                for req in self.requests()
            ],
        }

    def to_json(self) -> dict[str, Any]:
        return self.audit()

    def apply_one_shot(self, model: nn.Module) -> dict[str, Any]:
        modules = dict(model.named_modules())
        operations: list[dict[str, Any]] = []
        for req in self.requests():
            module = modules[req.module_name]
            original = _axis_channels(module, req.axis)
            prune = [idx for idx in req.prune_indices if 0 <= idx < original]
            keep = [idx for idx in range(original) if idx not in set(prune)]
            ordered_keep = req.metadata.get("ordered_keep_indices")
            if ordered_keep is not None:
                ordered = [int(idx) for idx in ordered_keep if 0 <= int(idx) < original]
                if len(ordered) != len(set(ordered)) or set(ordered) != set(keep):
                    raise ValueError(
                        f"ordered_keep_indices_mismatch:{req.module_name}:{req.axis}:"
                        f"ordered={ordered}:natural_keep={keep}"
                    )
                keep = ordered
            op_extra = _apply_keep(module, req.axis, keep, req.metadata)
            replay_axis = str(req.metadata.get("replay_axis", req.axis))
            op = {
                "module_name": req.module_name,
                "axis": replay_axis,
                "physical_axis": req.axis,
                "original_num_channels": original,
                "prune_indices": prune,
                "keep_indices": keep,
                "new_num_channels": len(keep),
                "applied_once": True,
                "source_recipe_ids": req.source_recipe_ids,
                "before": original,
                "after": len(keep),
                "direction": req.axis,
                "layer": req.module_name,
            }
            op.update(op_extra)
            operations.append(op)
        return {
            "operations": operations,
            "num_operations": len(operations),
            "num_duplicate_module_axis_requests": sum(1 for count in self._request_counts.values() if count > 1),
        }


def _idx(indices: list[int], device: torch.device) -> torch.Tensor:
    return torch.as_tensor(indices, dtype=torch.long, device=device)


def _axis_channels(module: nn.Module, axis: str) -> int:
    if isinstance(module, nn.Conv2d):
        if axis == "grouped_true_group_block":
            return int(module.groups)
        if axis == "grouped_d_compact_frontfill_reblock":
            return int(module.out_channels)
        if axis == "grouped_independent_keep":
            return int(module.out_channels)
        if axis == "grouped_input_pergroup8":
            return int(module.in_channels)
        return int(module.out_channels if axis in {"out", "grouped_coarsen_out"} else module.in_channels)
    if isinstance(module, nn.ConvTranspose2d):
        return int(module.out_channels if axis == "out" else module.in_channels)
    if isinstance(module, nn.modules.batchnorm._BatchNorm):
        return int(module.num_features)
    if isinstance(module, nn.Linear):
        return int(module.out_features if axis == "out" else module.in_features)
    raise TypeError(f"unsupported_module_for_one_shot:{module.__class__.__name__}:{axis}")


def _apply_keep(module: nn.Module, axis: str, keep: list[int], metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    metadata = metadata or {}
    if isinstance(module, nn.Conv2d):
        if axis == "grouped_true_group_block":
            return prune_grouped_conv_true_group_block(module, keep)
        if axis == "grouped_d_compact_frontfill_reblock":
            old_output_keep = metadata.get("old_output_keep_indices", keep)
            old_input_keep = metadata.get("old_input_keep_indices", list(range(int(module.in_channels))))
            groups_new = int(metadata.get("groups_new") or 0)
            return prune_grouped_conv_d_compact_frontfill_reblock(
                module,
                old_output_keep_indices=[int(v) for v in old_output_keep],
                old_input_keep_indices=[int(v) for v in old_input_keep],
                groups_new=groups_new,
            )
        if axis == "grouped_coarsen_out":
            return _apply_grouped_coarsen_out(module, keep, metadata)
        if axis == "grouped_independent_keep":
            return grouped_conv_pruning_fn("independent_group_topk")(module, keep)
        if axis == "grouped_input_pergroup8":
            resolved = validate_grouped_input_keep_pergroup8(
                module_name=str(metadata.get("module_name", "")),
                keep_indices=keep,
                groups=int(module.groups),
                C_in_before=int(module.in_channels),
                align=int(metadata.get("align_channels", 8) or 8),
            )
            if not resolved.get("legal", False):
                raise ValueError(str(resolved.get("skipped_input_prune_reason") or "grouped_input_pergroup8_illegal"))
            op = prune_grouped_conv_input_balanced(module, keep)
            op["axis"] = "grouped_input_pergroup8"
            op["legality_passed"] = True
            return op
        if axis == "grouped_input_balanced":
            return prune_grouped_conv_input_balanced(module, keep)
        if axis == "out":
            index = _idx(keep, module.weight.device)
            module.weight = nn.Parameter(module.weight.data.index_select(0, index).clone())
            if module.bias is not None:
                module.bias = nn.Parameter(module.bias.data.index_select(0, index).clone())
            module.out_channels = len(keep)
            if module.groups == module.in_channels == _axis_channels(module, "in"):
                pass
            return {}
        if axis == "in":
            if module.groups != 1 and str(metadata.get("replay_axis", "")) == "grouped_input_balanced":
                return prune_grouped_conv_input_balanced(module, keep)
            if module.groups != 1:
                raise ValueError("one_shot grouped Conv2d input slicing requires policy-specific grouped surgery")
            index = _idx(keep, module.weight.device)
            module.weight = nn.Parameter(module.weight.data.index_select(1, index).clone())
            module.in_channels = len(keep)
            return {}
    if isinstance(module, nn.ConvTranspose2d):
        if module.groups != 1:
            raise ValueError("unsupported_grouped_convtranspose")
        index = _idx(keep, module.weight.device)
        if axis == "out":
            module.weight = nn.Parameter(module.weight.data.index_select(1, index).clone())
            if module.bias is not None:
                module.bias = nn.Parameter(module.bias.data.index_select(0, index).clone())
            module.out_channels = len(keep)
            return {}
        if axis == "in":
            module.weight = nn.Parameter(module.weight.data.index_select(0, index).clone())
            module.in_channels = len(keep)
            return {}
    if isinstance(module, nn.modules.batchnorm._BatchNorm):
        index = _idx(keep, module.weight.device if module.weight is not None else module.running_mean.device)
        if module.weight is not None:
            module.weight = nn.Parameter(module.weight.data.index_select(0, index).clone())
        if module.bias is not None:
            module.bias = nn.Parameter(module.bias.data.index_select(0, index).clone())
        if module.running_mean is not None:
            module.running_mean = module.running_mean.index_select(0, index).clone()
        if module.running_var is not None:
            module.running_var = module.running_var.index_select(0, index).clone()
        module.num_features = len(keep)
        return {}
    if isinstance(module, nn.Linear):
        index = _idx(keep, module.weight.device)
        if axis == "out":
            module.weight = nn.Parameter(module.weight.data.index_select(0, index).clone())
            if module.bias is not None:
                module.bias = nn.Parameter(module.bias.data.index_select(0, index).clone())
            module.out_features = len(keep)
            return {}
        if axis == "in":
            module.weight = nn.Parameter(module.weight.data.index_select(1, index).clone())
            module.in_features = len(keep)
            return {}
    raise TypeError(f"unsupported_one_shot_axis:{module.__class__.__name__}:{axis}")


def _apply_grouped_coarsen_out(module: nn.Conv2d, keep: list[int], metadata: dict[str, Any]) -> dict[str, Any]:
    old_groups = int(metadata.get("old_groups") or module.groups)
    groups_new = int(metadata.get("groups_new") or 0)
    if groups_new <= 0:
        raise ValueError("grouped_coarsen_out_missing_groups_new")
    if groups_new >= old_groups:
        raise ValueError("grouped_coarsen_out_requires_groups_new_lt_groups_old")
    if old_groups % groups_new != 0:
        raise ValueError("grouped_coarsen_out_requires_groups_old_divisible_by_groups_new")
    if module.in_channels % old_groups != 0 or module.in_channels % groups_new != 0:
        raise ValueError("grouped_coarsen_out_invalid_input_group_divisibility")
    if len(keep) % groups_new != 0:
        raise ValueError("grouped_coarsen_out_invalid_output_group_divisibility")

    old_weight = module.weight.data
    old_bias = module.bias.data if module.bias is not None else None
    old_out_per_group = module.out_channels // old_groups
    old_in_per_group = module.in_channels // old_groups
    new_out_per_group = len(keep) // groups_new
    new_in_per_group = module.in_channels // groups_new
    merge_factor = old_groups // groups_new
    new_weight = old_weight.new_zeros((len(keep), new_in_per_group, *old_weight.shape[2:]))
    new_bias = old_bias.new_empty((len(keep),)) if old_bias is not None else None
    copy_plan: list[dict[str, Any]] = []
    semantic_mismatch_count = 0

    for new_out_idx, old_out_idx in enumerate(keep):
        old_group = int(old_out_idx) // old_out_per_group
        target_new_group = old_group // merge_factor
        actual_new_group = new_out_idx // new_out_per_group
        offset = (old_group % merge_factor) * old_in_per_group
        if target_new_group != actual_new_group:
            semantic_mismatch_count += 1
        new_weight[new_out_idx, offset : offset + old_in_per_group] = old_weight[old_out_idx]
        if new_bias is not None and old_bias is not None:
            new_bias[new_out_idx] = old_bias[old_out_idx]
        copy_plan.append(
            {
                "old_out_idx": int(old_out_idx),
                "new_out_idx": int(new_out_idx),
                "old_group": int(old_group),
                "target_new_group": int(target_new_group),
                "actual_new_group": int(actual_new_group),
                "offset_in_new_group": int(offset),
                "copied_slice_range": [int(offset), int(offset + old_in_per_group)],
                "zero_padded_slice_count": int(new_in_per_group - old_in_per_group),
            }
        )

    module.weight = nn.Parameter(new_weight.clone())
    if new_bias is not None:
        module.bias = nn.Parameter(new_bias.clone())
    module.out_channels = len(keep)
    module.groups = groups_new
    return {
        "groups_before": old_groups,
        "groups_after": groups_new,
        "merge_factor": merge_factor,
        "in_per_group_before": old_in_per_group,
        "in_per_group_after": new_in_per_group,
        "out_per_group_before": old_out_per_group,
        "out_per_group_after": new_out_per_group,
        "zero_pad_copy_plan": copy_plan,
        "semantic_mismatch_count": semantic_mismatch_count,
        "weight_truncation_count": 0,
        "added_zero_connections": int(len(keep) * (new_in_per_group - old_in_per_group) * old_weight.shape[2] * old_weight.shape[3]),
    }
