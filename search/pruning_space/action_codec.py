"""Pruning action codec helpers."""

from __future__ import annotations

from typing import Sequence

import torch

from ..proxy.parameter_slice_resolver import ParameterSlice
from .action_catalog import PruningSearchAction


def selected_actions_from_genes(
    pruning_genes: dict[str, int],
    actions: Sequence[PruningSearchAction],
) -> list[PruningSearchAction]:
    by_id = {action.action_id: action for action in actions}
    return [
        by_id[action_id]
        for action_id, keep in sorted(pruning_genes.items())
        if int(keep) == 0 and action_id in by_id
    ]


def sanitize_action_parameter_slices(
    action: PruningSearchAction,
    rows: Sequence[ParameterSlice],
    parameters: dict[str, torch.nn.Parameter],
) -> list[ParameterSlice]:
    sanitized: list[ParameterSlice] = []
    width = int(action.constraints.get("channels_per_group_before") or action.constraints.get("channels_per_group") or 0)
    for row in rows:
        parameter = parameters.get(row.parameter_name)
        if parameter is None or int(row.axis) >= parameter.ndim:
            continue
        axis_size = int(parameter.shape[int(row.axis)])
        indices = tuple(int(value) for value in row.indices)
        if indices and max(indices) >= axis_size and action.kind == "grouped_bundle" and width > 0 and int(row.axis) == 1:
            indices = tuple(sorted({value % width for value in indices}))
        indices = tuple(index for index in indices if 0 <= index < axis_size)
        if not indices:
            continue
        sanitized.append(ParameterSlice(row.parameter_name, row.module_path, row.axis, indices, row.operation))
    return sanitized
