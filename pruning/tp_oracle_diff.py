"""Optional Torch-Pruning oracle comparison for one root prune action.

The project pruner must not depend on torch-pruning.  This helper is a debug
adapter: when TP and example inputs are available it records TP's dependency
group for the same root action; otherwise it still writes a stable unavailable
report so automation can audit why oracle comparison was skipped.
"""

from __future__ import annotations

from typing import Any

import torch.nn as nn


def _request_rows(plan: Any) -> list[dict[str, Any]]:
    if not hasattr(plan, "requests"):
        return []
    rows: list[dict[str, Any]] = []
    for req in plan.requests():
        rows.append(
            {
                "module_name": str(getattr(req, "module_name", "")),
                "axis": str(getattr(req, "axis", "")),
                "prune_indices": [int(v) for v in getattr(req, "prune_indices", [])],
                "source_recipe_ids": list(getattr(req, "source_recipe_ids", [])),
                "metadata": dict(getattr(req, "metadata", {}) or {}),
            }
        )
    return rows


def _axis_name(axis: str) -> str:
    if axis == "out":
        return "out"
    if axis == "in":
        return "in"
    if axis == "grouped_coarsen_out":
        return "out"
    if axis in {
        "grouped_flat_output",
        "grouped_group_balanced_output",
        "grouped_independent_keep",
        "grouped_remove",
    }:
        return "out"
    if axis == "grouped_input_balanced":
        return "in"
    return axis


def _project_member_keys(rows: list[dict[str, Any]]) -> set[tuple[str, str]]:
    return {(row["module_name"], _axis_name(row["axis"])) for row in rows}


def _tp_pruning_fn(tp: Any, module: nn.Module, axis: str) -> Any:
    if isinstance(module, nn.Conv2d):
        return tp.prune_conv_out_channels if axis == "out" else tp.prune_conv_in_channels
    if isinstance(module, nn.ConvTranspose2d):
        return tp.prune_convtranspose_out_channels if axis == "out" else tp.prune_convtranspose_in_channels
    if isinstance(module, nn.Linear):
        return tp.prune_linear_out_channels if axis == "out" else tp.prune_linear_in_channels
    if isinstance(module, nn.modules.batchnorm._BatchNorm):
        return tp.prune_batchnorm_out_channels
    raise TypeError(f"unsupported_tp_root:{module.__class__.__name__}:{axis}")


def _extract_tp_group_members(tp_group: Any, module_names_by_id: dict[int, str]) -> list[dict[str, Any]]:
    members: list[dict[str, Any]] = []
    try:
        iterator = list(tp_group)
    except TypeError:
        iterator = []
    for item in iterator:
        module = None
        idxs: list[int] = []
        pruning_fn = ""
        dep = getattr(item, "dep", None)
        target = getattr(dep, "target", None)
        if target is not None:
            module = getattr(target, "module", None)
        if module is None:
            module = getattr(item, "module", None)
        raw_idxs = getattr(item, "idxs", getattr(item, "indices", []))
        try:
            idxs = [int(v) for v in raw_idxs]
        except TypeError:
            idxs = []
        handler = getattr(dep, "handler", None) if dep is not None else None
        if handler is not None:
            pruning_fn = getattr(handler, "__name__", str(handler))
        name = module_names_by_id.get(id(module), "") if module is not None else ""
        axis = "out"
        low_fn = pruning_fn.lower()
        if "in_channels" in low_fn or "_in_" in low_fn or low_fn.endswith("_in_channels"):
            axis = "in"
        members.append(
            {
                "module_name": name,
                "module_type": module.__class__.__name__ if module is not None else "",
                "axis": axis,
                "idxs": idxs,
                "dependency": str(dep) if dep is not None else str(item),
                "raw": str(item),
            }
        )
    return members


def build_tp_oracle_diff(
    model: nn.Module,
    plan: Any,
    *,
    root_module_name: str,
    root_axis: str,
    root_indices: list[int],
    example_inputs: Any | None = None,
    forward_fn: Any | None = None,
) -> dict[str, Any]:
    """Build a stable TP-vs-project dependency diff for one root action."""

    project_rows = _request_rows(plan)
    base = {
        "available": False,
        "status": "",
        "root_module_name": root_module_name,
        "root_axis": root_axis,
        "root_indices": [int(v) for v in root_indices],
        "project_plan_members": project_rows,
        "tp_group_members": [],
        "missing_dependencies_in_project_plan": [],
        "extra_project_plan_members": [],
        "idx_transform_mismatches": [],
        "dependency_chain": [],
        "diagnosis": {
            "project_plan_missing_dependency": False,
            "project_plan_extra_prune": False,
            "idx_transform_inconsistent_with_tp": False,
            "possible_concat_residual_grouped_input_contract_gap": False,
        },
    }
    try:
        import torch_pruning as tp  # type: ignore
    except Exception as exc:  # noqa: BLE001
        base["status"] = "torch_pruning_not_installed"
        base["error"] = f"{type(exc).__name__}: {exc}"
        return base
    if example_inputs is None:
        base["status"] = "oracle_build_failed"
        base["error"] = "example_inputs_required_for_torch_pruning_depgraph"
        return base

    modules = dict(model.named_modules())
    root_module = modules.get(root_module_name)
    if root_module is None:
        base["status"] = "oracle_build_failed"
        base["error"] = f"missing_root_module:{root_module_name}"
        return base
    try:
        pruning_fn = _tp_pruning_fn(tp, root_module, root_axis)
        dg = tp.DependencyGraph()
        kwargs: dict[str, Any] = {"example_inputs": example_inputs}
        if forward_fn is not None:
            kwargs["forward_fn"] = forward_fn
        dg.build_dependency(model, **kwargs)
        tp_group = dg.get_pruning_group(root_module, pruning_fn, idxs=[int(v) for v in root_indices])
        module_names_by_id = {id(module): name for name, module in modules.items()}
        tp_members = _extract_tp_group_members(tp_group, module_names_by_id)
    except Exception as exc:  # noqa: BLE001
        base["status"] = "oracle_build_failed"
        base["error"] = f"{type(exc).__name__}: {exc}"
        return base

    project_keys = _project_member_keys(project_rows)
    tp_keys = {(row["module_name"], _axis_name(row["axis"])) for row in tp_members if row.get("module_name")}
    missing = sorted(tp_keys - project_keys)
    extra = sorted(project_keys - tp_keys)
    base.update(
        {
            "available": True,
            "status": "success",
            "tp_group_members": tp_members,
            "missing_dependencies_in_project_plan": [{"module_name": name, "axis": axis} for name, axis in missing],
            "extra_project_plan_members": [{"module_name": name, "axis": axis} for name, axis in extra],
            "dependency_chain": [row.get("dependency", "") for row in tp_members],
            "diagnosis": {
                "project_plan_missing_dependency": bool(missing),
                "project_plan_extra_prune": bool(extra),
                "idx_transform_inconsistent_with_tp": False,
                "possible_concat_residual_grouped_input_contract_gap": any(
                    key in " ".join(name for name, _axis in missing).lower()
                    for key in ("cat", "concat", "add", "residual", "downsample", "conv2")
                ),
            },
        }
    )
    return base
