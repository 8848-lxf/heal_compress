"""Global physical-plan legalization before execution."""

from __future__ import annotations

import copy

import torch.nn as nn

from ..config import AlignmentConfig, GroupedConvConfig
from ..exceptions import PruningLegalityError
from ..policies.alignment import legalize_dense_keep_count
from ..policies.grouped_conv import validate_grouped_conv_shape
from ..types import PhysicalPruningPlan
from .grouped_conv import validate_grouped_plan_entry


def legalize_pruning_plan(
    model: nn.Module,
    plan: PhysicalPruningPlan,
    *,
    alignment_config: AlignmentConfig | None = None,
    grouped_config: GroupedConvConfig | None = None,
) -> PhysicalPruningPlan:
    """Return a repaired copy while preserving the frozen original axes."""

    if not plan.indices_frozen_before_materialization:
        raise PruningLegalityError("plan indices were not frozen before legalization")
    alignment = alignment_config or AlignmentConfig()
    grouped = grouped_config or GroupedConvConfig()
    modules = dict(model.named_modules())
    result = copy.deepcopy(plan)
    # Grouped replay metadata is part of legality, not optional reporting.
    for entry in result.entries:
        module = modules.get(entry.module_path)
        if module is None:
            raise PruningLegalityError(f"module disappeared before legalization: {entry.module_path}")
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)) and module.groups > 1:
            if module.groups == module.in_channels == module.out_channels:
                # Depthwise pruning removes whole channel/groups. Its absolute
                # coupled in/out indices are checked below and do not use a
                # per-group local width map (the local width is always one).
                continue
            width = (
                entry.original_axis_size // module.groups
                if entry.original_axis_size % module.groups == 0
                else 0
            )
            keep_map = validate_grouped_plan_entry(
                entry,
                groups=module.groups,
                channels_per_group=width,
            )
            kept_per_group = len(next(iter(keep_map.values())))
            if kept_per_group not in grouped.allowed_channels_per_group and not (
                grouped.depthwise_special_case
                and module.groups == module.in_channels == module.out_channels
            ):
                raise PruningLegalityError(
                    f"{entry.module_path}:{entry.axis} keeps disallowed channels per group: {kept_per_group}"
                )
            continue

    # Alignment is repaired at the dependency-scope level. We first determine
    # the additional root indices, then propagate them to every saved closure
    # member before execution. Different-sized mappings must be explicit.
    root_repairs: list[tuple[object, set[int], set[str]]] = []
    for entry in result.entries:
        module = modules[entry.module_path]
        if isinstance(module, nn.Conv2d) and module.groups == 1 and entry.axis == "out":
            repair = legalize_dense_keep_count(
                entry.original_axis_size,
                len(entry.keep_indices),
                alignment=alignment.dense_conv_channel_alignment,
                fixed_output_contract=bool(entry.metadata.get("fixed_output_contract")),
            )
            if repair.final_keep != len(entry.keep_indices):
                old_keep = list(entry.keep_indices)
                new_keep = old_keep[: repair.final_keep]
                added_prune = set(old_keep[repair.final_keep :])
                scopes = {str(value) for value in entry.metadata.get("scope_ids", [])}
                if not scopes:
                    raise PruningLegalityError(
                        f"alignment repair lacks dependency scope metadata: {entry.module_path}"
                    )
                root_repairs.append((entry, added_prune, scopes))
                entry.keep_indices = new_keep
                entry.prune_indices = [
                    index for index in range(entry.original_axis_size) if index not in set(new_keep)
                ]
                entry.repaired = True
                entry.repair_reason = repair.reason
                entry.metadata["alignment_repair"] = repair.__dict__

    for root_entry, added_root_prune, scopes in root_repairs:
        for dependent in result.entries:
            if dependent is root_entry:
                continue
            dependent_scopes = {str(value) for value in dependent.metadata.get("scope_ids", [])}
            if not (scopes & dependent_scopes):
                continue
            if dependent.metadata.get("fixed_output_contract") and dependent.axis in {"out", "channel"}:
                raise PruningLegalityError(
                    f"alignment closure reaches protected output: {dependent.module_path}:{dependent.axis}"
                )
            if dependent.original_axis_size == root_entry.original_axis_size:
                mapped_additional = set(added_root_prune)
            else:
                index_map = {
                    int(key): [int(value) for value in values]
                    for key, values in dependent.metadata.get("closure_index_map", {}).items()
                }
                missing = sorted(index for index in added_root_prune if index not in index_map)
                if missing:
                    raise PruningLegalityError(
                        "non-identity alignment closure lacks a complete saved index map for "
                        f"{dependent.module_path}:{dependent.axis}; missing roots={missing}"
                    )
                mapped_additional = {
                    local
                    for root_index in added_root_prune
                    for local in index_map[root_index]
                }
            if mapped_additional and (
                min(mapped_additional) < 0 or max(mapped_additional) >= dependent.original_axis_size
            ):
                raise PruningLegalityError(
                    f"alignment closure map is out of bounds for {dependent.module_path}:{dependent.axis}"
                )
            keep = [index for index in dependent.keep_indices if index not in mapped_additional]
            if not keep:
                raise PruningLegalityError(
                    f"alignment repair removes every closure channel from {dependent.module_path}:{dependent.axis}"
                )
            dependent.keep_indices = keep
            dependent.prune_indices = [
                index for index in range(dependent.original_axis_size) if index not in set(keep)
            ]
            dependent.repaired = True
            dependent.repair_reason = "dependency_closure_alignment_repair"
            dependent.metadata["alignment_repair"] = {
                "root_module_path": root_entry.module_path,
                "root_added_prune_indices": sorted(added_root_prune),
                "mapped_added_prune_indices": sorted(mapped_additional),
                "alignment": alignment.dense_conv_channel_alignment,
                "repaired": True,
            }

    # A depthwise module is legal only when its in/out requests are coupled.
    entries_by_module: dict[str, dict[str, object]] = {}
    for entry in result.entries:
        entries_by_module.setdefault(entry.module_path, {})[entry.axis] = entry
    for module_path, axes in entries_by_module.items():
        module = modules[module_path]
        if not (
            isinstance(module, (nn.Conv2d, nn.ConvTranspose2d))
            and module.groups == module.in_channels == module.out_channels
        ):
            continue
        input_entry = axes.get("in")
        output_entry = axes.get("out") or axes.get("channel")
        if input_entry is None or output_entry is None:
            raise PruningLegalityError(
                f"depthwise pruning requires coupled in/out entries: {module_path}"
            )
        if input_entry.keep_indices != output_entry.keep_indices:
            raise PruningLegalityError(
                f"depthwise in/out keep indices differ: {module_path}"
            )
    return result


__all__ = ["legalize_dense_keep_count", "legalize_pruning_plan"]
