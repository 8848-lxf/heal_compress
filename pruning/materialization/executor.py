"""Single-transaction CPU-safe physical channel materialization."""

from __future__ import annotations

import copy
from collections import defaultdict
from collections.abc import Sequence

import torch
import torch.nn as nn

from ..exceptions import GroupedConvLegalityError, PruningLegalityError, PruningPlanError
from ..types import MaterializationResult, PhysicalPruningPlan, PhysicalPruningPlanEntry
from .grouped_conv import validate_grouped_materialization, validate_grouped_plan_entry
from .ledger import build_application_ledger
from .planner import module_axis_size


def _parameter_like(original: nn.Parameter, value: torch.Tensor) -> nn.Parameter:
    return nn.Parameter(value.detach().clone(), requires_grad=original.requires_grad)


def _index(value: torch.Tensor, axis: int, indices: Sequence[int]) -> torch.Tensor:
    index = torch.as_tensor(list(indices), dtype=torch.long, device=value.device)
    return value.index_select(axis, index)


def _entry(entries: Sequence[PhysicalPruningPlanEntry], axis: str) -> PhysicalPruningPlanEntry | None:
    return next((row for row in entries if row.axis == axis), None)


def _slice_dense_conv(module: nn.Conv2d | nn.ConvTranspose2d, entries: Sequence[PhysicalPruningPlanEntry]) -> None:
    in_entry = _entry(entries, "in")
    out_entry = _entry(entries, "out") or _entry(entries, "channel")
    in_keep = in_entry.keep_indices if in_entry else list(range(module.in_channels))
    out_keep = out_entry.keep_indices if out_entry else list(range(module.out_channels))
    weight = module.weight
    if isinstance(module, nn.ConvTranspose2d):
        weight = _index(weight, 0, in_keep)
        weight = _index(weight, 1, out_keep)
    else:
        weight = _index(weight, 0, out_keep)
        weight = _index(weight, 1, in_keep)
    module.weight = _parameter_like(module.weight, weight)
    if module.bias is not None and out_entry is not None:
        module.bias = _parameter_like(module.bias, _index(module.bias, 0, out_keep))
    module.in_channels = len(in_keep)
    module.out_channels = len(out_keep)


def _slice_grouped_conv(module: nn.Conv2d | nn.ConvTranspose2d, entries: Sequence[PhysicalPruningPlanEntry]) -> None:
    if module.groups == module.in_channels == module.out_channels:
        in_entry = _entry(entries, "in")
        out_entry = _entry(entries, "out") or _entry(entries, "channel")
        if in_entry is None or out_entry is None:
            raise GroupedConvLegalityError("depthwise pruning requires coupled in/out plan entries")
        if in_entry.keep_indices != out_entry.keep_indices:
            raise GroupedConvLegalityError("depthwise in/out keep indices must be identical")
        keep = in_entry.keep_indices
        # Both depthwise Conv2d and ConvTranspose2d store one kernel block per
        # logical channel/group on weight axis zero.
        module.weight = _parameter_like(module.weight, _index(module.weight, 0, keep))
        if module.bias is not None:
            module.bias = _parameter_like(module.bias, _index(module.bias, 0, keep))
        module.in_channels = len(keep)
        module.out_channels = len(keep)
        module.groups = len(keep)
        return
    groups = int(module.groups)
    in_per = module.in_channels // groups
    out_per = module.out_channels // groups
    in_entry = _entry(entries, "in")
    out_entry = _entry(entries, "out") or _entry(entries, "channel")
    input_maps = (
        validate_grouped_plan_entry(
            in_entry,
            groups=groups,
            channels_per_group=in_per,
        )
        if in_entry
        else {group: list(range(in_per)) for group in range(groups)}
    )
    output_maps = (
        validate_grouped_plan_entry(
            out_entry,
            groups=groups,
            channels_per_group=out_per,
        )
        if out_entry
        else {group: list(range(out_per)) for group in range(groups)}
    )
    blocks: list[torch.Tensor] = []
    output_absolute: list[int] = []
    for group in range(groups):
        input_absolute = [group * in_per + local for local in input_maps[group]]
        output_absolute.extend(group * out_per + local for local in output_maps[group])
        if isinstance(module, nn.ConvTranspose2d):
            block = _index(module.weight, 0, input_absolute)
            block = _index(block, 1, output_maps[group])
        else:
            output_rows = [group * out_per + local for local in output_maps[group]]
            block = _index(module.weight, 0, output_rows)
            block = _index(block, 1, input_maps[group])
        blocks.append(block)
    module.weight = _parameter_like(module.weight, torch.cat(blocks, dim=0))
    if module.bias is not None and out_entry is not None:
        module.bias = _parameter_like(module.bias, _index(module.bias, 0, output_absolute))
    module.in_channels = groups * len(input_maps[0])
    module.out_channels = groups * len(output_maps[0])


def _slice_linear(module: nn.Linear, entries: Sequence[PhysicalPruningPlanEntry]) -> None:
    in_entry = _entry(entries, "in")
    out_entry = _entry(entries, "out") or _entry(entries, "channel")
    in_keep = in_entry.keep_indices if in_entry else list(range(module.in_features))
    out_keep = out_entry.keep_indices if out_entry else list(range(module.out_features))
    weight = _index(_index(module.weight, 0, out_keep), 1, in_keep)
    module.weight = _parameter_like(module.weight, weight)
    if module.bias is not None and out_entry is not None:
        module.bias = _parameter_like(module.bias, _index(module.bias, 0, out_keep))
    module.in_features = len(in_keep)
    module.out_features = len(out_keep)


def _slice_batchnorm(module: nn.modules.batchnorm._BatchNorm, entries: Sequence[PhysicalPruningPlanEntry]) -> None:
    row = _entry(entries, "channel") or _entry(entries, "out") or _entry(entries, "in")
    if row is None:
        return
    keep = row.keep_indices
    for name in ("weight", "bias"):
        parameter = getattr(module, name)
        if parameter is not None:
            setattr(module, name, _parameter_like(parameter, _index(parameter, 0, keep)))
    for name in ("running_mean", "running_var"):
        buffer = getattr(module, name)
        if buffer is not None:
            setattr(module, name, _index(buffer, 0, keep).detach().clone())
    module.num_features = len(keep)


def _slice_layernorm(module: nn.LayerNorm, entries: Sequence[PhysicalPruningPlanEntry]) -> None:
    row = _entry(entries, "channel") or _entry(entries, "out") or _entry(entries, "in")
    if row is None or len(module.normalized_shape) != 1:
        raise PruningPlanError("only one-dimensional LayerNorm channel pruning is supported")
    keep = row.keep_indices
    for name in ("weight", "bias"):
        parameter = getattr(module, name)
        if parameter is not None:
            setattr(module, name, _parameter_like(parameter, _index(parameter, 0, keep)))
    module.normalized_shape = (len(keep),)


def _preflight(model: nn.Module, plan: PhysicalPruningPlan) -> None:
    if not plan.indices_frozen_before_materialization:
        raise PruningPlanError("materialization rejected a plan with unfrozen indices")
    modules = dict(model.named_modules())
    seen: set[tuple[str, str]] = set()
    for row in plan.entries:
        key = (row.module_path, row.axis)
        if key in seen:
            raise PruningPlanError(f"unmerged duplicate plan entry: {key}")
        seen.add(key)
        module = modules.get(row.module_path)
        if module is None:
            raise PruningPlanError(f"module missing at materialization: {row.module_path}")
        live_size = module_axis_size(module, row.axis)
        if live_size != row.original_axis_size:
            raise PruningPlanError(
                f"model changed after plan freeze for {row.module_path}:{row.axis}: "
                f"live={live_size}, frozen={row.original_axis_size}"
            )
        if row.keep_indices != sorted(set(row.keep_indices)):
            raise PruningPlanError(f"non-canonical keep indices for {row.module_path}:{row.axis}")
        if not row.keep_indices:
            raise PruningLegalityError(f"empty keep set for {row.module_path}:{row.axis}")
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)) and module.groups > 1:
            if module.groups == module.in_channels == module.out_channels:
                continue
            width = row.original_axis_size // module.groups
            validate_grouped_plan_entry(row, groups=module.groups, channels_per_group=width)

    # Depthwise structure changes are group removals coupled across both axes.
    for module_path, module in modules.items():
        if not module_path or not (
            isinstance(module, (nn.Conv2d, nn.ConvTranspose2d))
            and module.groups == module.in_channels == module.out_channels
        ):
            continue
        rows = [row for row in plan.entries if row.module_path == module_path]
        if not rows:
            continue
        input_row = _entry(rows, "in")
        output_row = _entry(rows, "out") or _entry(rows, "channel")
        if input_row is None or output_row is None or input_row.keep_indices != output_row.keep_indices:
            raise GroupedConvLegalityError(
                f"depthwise plan must contain matching coupled in/out keeps: {module_path}"
            )


def materialize_pruning(
    model: nn.Module,
    plan: PhysicalPruningPlan,
    *,
    in_place: bool = False,
    build_snapshot: bool = True,
) -> MaterializationResult:
    """Apply all precomputed axes once, after a complete transaction preflight."""

    _preflight(model, plan)
    target = model if in_place else copy.deepcopy(model)
    modules = dict(target.named_modules())
    grouped: dict[str, list[PhysicalPruningPlanEntry]] = defaultdict(list)
    for row in plan.entries:
        grouped[row.module_path].append(row)
    for module_path in sorted(grouped):
        module = modules[module_path]
        entries = grouped[module_path]
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            if module.groups > 1:
                _slice_grouped_conv(module, entries)
            else:
                _slice_dense_conv(module, entries)
        elif isinstance(module, nn.Linear):
            _slice_linear(module, entries)
        elif isinstance(module, nn.modules.batchnorm._BatchNorm):
            _slice_batchnorm(module, entries)
        elif isinstance(module, nn.LayerNorm):
            _slice_layernorm(module, entries)
        else:
            raise PruningPlanError(f"unsupported physical module: {module_path} ({type(module).__name__})")
    ledger = build_application_ledger(plan)
    snapshot = None
    if build_snapshot:
        from ..artifacts.snapshot import build_physical_structure_snapshot

        snapshot = build_physical_structure_snapshot(target)
    return MaterializationResult(model=target, plan=plan, ledger=ledger, snapshot=snapshot)


__all__ = ["materialize_pruning"]
