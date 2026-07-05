"""Full-model prunable surface inventory and protection for v9.5."""

from __future__ import annotations

from typing import Any

import torch.nn as nn


def is_regular_grouped_conv(module: Any) -> bool:
    return (
        isinstance(module, nn.Conv2d)
        and int(module.groups) > 1
        and not (int(module.groups) == int(module.in_channels) == int(module.out_channels))
    )


def group_has_grouped_input_contract(group: Any) -> bool:
    for item in getattr(group, "items", []):
        if is_regular_grouped_conv(getattr(item, "module", None)) and getattr(item, "direction", "") == "in":
            return True
    return False


def group_has_grouped_output(group: Any) -> bool:
    for item in getattr(group, "items", []):
        if is_regular_grouped_conv(getattr(item, "module", None)) and getattr(item, "direction", "") == "out":
            return True
    return False


def group_has_unsupported_deblock_contract(group: Any) -> bool:
    for item in getattr(group, "items", []):
        module = getattr(item, "module", None)
        name = str(getattr(item, "name", ""))
        if isinstance(module, nn.ConvTranspose2d):
            return True
        if "pyramid_backbone.deblocks" in name:
            return True
    return False


def _module_channels(module: Any, direction: str) -> int:
    if isinstance(module, nn.Conv2d):
        return int(module.out_channels if direction == "out" else module.in_channels)
    if isinstance(module, nn.modules.batchnorm._BatchNorm):
        return int(module.num_features)
    if isinstance(module, nn.Linear):
        return int(module.out_features if direction == "out" else module.in_features)
    return 0


def _module_params(module: Any) -> int:
    if not hasattr(module, "parameters"):
        return 0
    return int(sum(p.numel() for p in module.parameters(recurse=False)))


def surface_inventory(groups: list[Any], *, total_model_params: int = 0, group_conv_policy: str = "A") -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    prunable_modules: dict[int, int] = {}
    protected_modules: dict[int, int] = {}
    grouped_prunable_modules: dict[int, int] = {}
    non_grouped_prunable_modules: dict[int, int] = {}
    for group in groups:
        items = list(getattr(group, "items", []))
        group_params = sum(_module_params(getattr(item, "module", None)) for item in items)
        belongs_grouped = any(is_regular_grouped_conv(getattr(item, "module", None)) for item in items)
        protected = bool(getattr(group, "protected", False))
        for item in items:
            module = getattr(item, "module", None)
            params = _module_params(module)
            if params <= 0:
                continue
            key = id(module)
            if protected:
                protected_modules[key] = params
            else:
                prunable_modules[key] = params
                if belongs_grouped:
                    grouped_prunable_modules[key] = params
                else:
                    non_grouped_prunable_modules[key] = params
        root_item = items[0] if items else None
        rows.append(
            {
                "module_name": getattr(root_item, "name", getattr(group, "group_id", "")),
                "module_type": getattr(getattr(root_item, "module", None), "__class__", type("", (), {})).__name__,
                "is_prunable": not protected,
                "is_protected": protected,
                "protected_reason": getattr(group, "protected_reason", ""),
                "num_channels": int(getattr(group, "num_channels", 0) or 0),
                "num_params": group_params,
                "belongs_to_grouped_conv": belongs_grouped,
                "grouped_conv_policy_if_applicable": group_conv_policy if belongs_grouped else "",
                "dependency_root": getattr(group, "group_id", ""),
                "downstream_dependencies": [getattr(item, "name", "") for item in items[1:]],
                "residual_concat_constraints": (getattr(group, "meta", {}) or {}).get("group_type", ""),
            }
        )
    prunable_params = int(sum(prunable_modules.values()))
    if total_model_params:
        total = int(total_model_params)
        protected_params = max(total - prunable_params, 0)
    else:
        protected_only = {k: v for k, v in protected_modules.items() if k not in prunable_modules}
        protected_params = int(sum(protected_only.values()))
        total = int(prunable_params + protected_params)
    grouped_params = int(sum(grouped_prunable_modules.values()))
    non_grouped_params = int(sum(non_grouped_prunable_modules.values()))
    return {
        "prunable_surface": "full_model_all_safe_coupled_units",
        "groups": rows,
        "total_model_params": total,
        "total_prunable_params": prunable_params,
        "total_protected_params": protected_params,
        "total_grouped_conv_prunable_params": grouped_params,
        "total_non_grouped_conv_prunable_params": non_grouped_params,
        "prunable_param_ratio_of_full_model": prunable_params / total if total else 0.0,
        "num_prunable_coupled_units": sum(1 for row in rows if row["is_prunable"]),
        "num_protected_coupled_units": sum(1 for row in rows if row["is_protected"]),
    }


def apply_full_model_prunable_surface(groups: list[Any], *, group_conv_policy: str, total_model_params: int = 0) -> dict[str, Any]:
    policy = str(group_conv_policy).upper()
    for group in groups:
        if bool(getattr(group, "protected", False)):
            continue
        if group_has_unsupported_deblock_contract(group):
            group.protected = True
            group.protected_reason = "protected_convtranspose_deblock_or_fpn_output_contract"
            continue
        if policy in {"A", "B", "D"} and group_has_grouped_input_contract(group):
            group.protected = True
            group.protected_reason = f"protected_grouped_conv_input_contract:{policy}"
    return surface_inventory(groups, total_model_params=total_model_params, group_conv_policy=policy)
