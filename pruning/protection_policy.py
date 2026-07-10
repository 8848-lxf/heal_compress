"""Default output-channel protection policy for v10.8."""

from __future__ import annotations

from typing import Any, Iterable

import torch.nn as nn


HEAD_OUTPUT_KEYWORDS = (
    "cls_head",
    "reg_head",
    "dir_head",
    "obj_head",
    "heatmap_head",
    "hm_head",
    "box_head",
)
FPN_CONTAINER_KEYWORDS = ("fpn", "neck", "fusion_neck", "pyramid_fpn")
FPN_OUTPUT_HINTS = ("fpn_out", "out_conv", "output_conv", "final_conv", "lateral_out")


def _root_output_items(group: Any) -> list[Any]:
    return [item for item in getattr(group, "items", []) if getattr(item, "direction", "") == "out"]


def _is_head_output(name: str) -> bool:
    low = name.lower()
    return any(key in low for key in HEAD_OUTPUT_KEYWORDS)


def _is_fpn_output(name: str) -> bool:
    low = name.lower()
    if not any(key in low for key in FPN_CONTAINER_KEYWORDS):
        return False
    return any(hint in low for hint in FPN_OUTPUT_HINTS)


def _is_regular_grouped_conv(module: Any) -> bool:
    return (
        isinstance(module, nn.Conv2d)
        and int(module.groups) > 1
        and not (int(module.groups) == int(module.in_channels) == int(module.out_channels))
    )


def apply_v108_default_protection(
    groups: Iterable[Any],
    *,
    protect_fpn_output: bool = True,
    protect_head_output: bool = True,
) -> dict[str, Any]:
    """Protect only FPN output and detection-head output domains."""

    protected_fpn: list[str] = []
    protected_head: list[str] = []
    unprotected_grouped: list[str] = []
    unprotected_ordinary: list[str] = []
    reason_by_module: dict[str, str] = {}
    total_protected_units = 0
    total_searchable_units = 0
    for group in groups:
        items = _root_output_items(group)
        root = items[0] if items else (getattr(group, "items", []) or [None])[0]
        name = str(getattr(root, "name", getattr(group, "group_id", "")))
        module = getattr(root, "module", None)
        reason = ""
        if protect_head_output and _is_head_output(name):
            reason = "head_output"
            protected_head.append(name)
        elif protect_fpn_output and _is_fpn_output(name):
            reason = "fpn_output"
            protected_fpn.append(name)
        if reason:
            group.protected = True
            group.protected_reason = reason
            total_protected_units += int(getattr(group, "num_channels", 0) or 0)
            reason_by_module[name] = reason
            continue
        total_searchable_units += int(getattr(group, "num_channels", 0) or 0)
        if _is_regular_grouped_conv(module):
            unprotected_grouped.append(name)
        elif isinstance(module, nn.Conv2d):
            unprotected_ordinary.append(name)
    return {
        "protected_fpn_outputs": sorted(protected_fpn),
        "protected_head_outputs": sorted(protected_head),
        "unprotected_grouped_conv_outputs": sorted(unprotected_grouped),
        "unprotected_ordinary_conv_outputs": sorted(unprotected_ordinary),
        "total_protected_units": int(total_protected_units),
        "total_searchable_units": int(total_searchable_units),
        "reason_per_protected_module": reason_by_module,
    }
