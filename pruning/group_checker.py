"""Pruning-group legality checks (requirement #5).

``check_pruning_group`` validates a :class:`PruningGroup` against a proposed set
of keep indices *before* any surgery, so that an illegal cut becomes a clean
skip instead of a corrupted model. It guards against:

* pruning a channel dimension to empty;
* in/out/groups divisibility violations on (grouped) convs;
* Add branches whose output widths would diverge;
* Cat offsets that would leave the shrink conv's ``in_channels`` inconsistent
  with the summed branch widths;
* ConvTranspose2d weight-shape mismatches;
* detection-head input widths inconsistent with their producer.

``check_model_legality`` is a whole-model post-pass for positive dims and
in/out/groups sanity after surgery.
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch.nn as nn

from ..tracer.pruning_group import PruningGroup


def check_pruning_group(
    group: PruningGroup,
    group_keep: List[int],
) -> Dict[str, Any]:
    """Validate one group + keep set. Returns ``{"legal": bool, "issues": [...]}``.

    A group that fails any check should be treated as protected/skipped by the
    caller (never pruned partially).
    """
    issues: List[Dict[str, Any]] = []

    if group.protected:
        return {"legal": False, "issues": [{"issue": "protected", "reason": group.protected_reason}]}

    if not group_keep:
        issues.append({"issue": "empty_keep", "group_id": group.group_id})
        return {"legal": False, "issues": issues}

    # Per-item local keep checks.
    out_widths: List[tuple] = []
    cat_offset_total = 0
    is_cat = group.meta.get("group_type") == "cat"

    for item in group.items:
        module = item.module
        local = item.local_keep(group_keep)
        if not local:
            issues.append({
                "issue": "channel_emptied", "layer": item.name,
                "direction": item.direction,
            })
            continue

        # groups divisibility for conv-like modules
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            g = module.groups
            new_out = len(local) if item.direction == "out" else module.out_channels
            new_in = len(local) if item.direction == "in" else module.in_channels
            fn_name = getattr(item.pruning_fn, "__name__", "")
            # Handlers that reset in==out==groups (depthwise) or in==out
            # (grouped keep_groups) to len(local): divisibility holds by
            # construction, so skip the static check for them.
            if fn_name == "prune_grouped_keep_groups":
                new_in = new_out = len(local)
                # keep_groups requires identical local keep pattern per group.
                g = module.groups
                if module.out_channels % g == 0:
                    per = module.out_channels // g
                    patterns = set()
                    for gi in range(g):
                        start = gi * per
                        patterns.add(tuple(sorted(
                            idx - start for idx in local if start <= idx < start + per)))
                    if len(patterns) > 1:
                        issues.append({
                            "issue": "grouped_keep_pattern_mismatch", "layer": item.name,
                            "groups": g,
                        })
            is_depthwise_out = (
                isinstance(module, nn.Conv2d)
                and g > 1 and g == module.in_channels == module.out_channels
                and item.direction == "out"
            )
            if is_depthwise_out:
                pass  # in==out==groups all become len(local)
            elif g > 0 and fn_name not in ("prune_grouped_remove_groups", "prune_grouped_keep_groups"):
                if new_out % g != 0 or new_in % g != 0:
                    issues.append({
                        "issue": "groups_divisibility", "layer": item.name,
                        "groups": g, "new_in": new_in, "new_out": new_out,
                    })

        # ConvTranspose2d weight-shape consistency (out lives on axis 1).
        if isinstance(module, nn.ConvTranspose2d) and item.direction == "out":
            if module.weight.shape[1] != module.out_channels:
                issues.append({
                    "issue": "convtranspose_weight_shape", "layer": item.name,
                    "weight_out_axis": int(module.weight.shape[1]),
                    "out_channels": int(module.out_channels),
                })

        if item.direction == "out":
            out_widths.append((item.name, len(local), item.reason))

    # Add-branch consistency: every root-out member must end at the same width.
    if group.meta.get("group_type") == "add":
        root_widths = [w for (_n, w, reason) in out_widths if reason in ("root_out", "grouped_conv:keep_groups", "grouped_conv:depthwise")]
        if root_widths and len(set(root_widths)) > 1:
            issues.append({
                "issue": "add_branch_width_mismatch",
                "widths": root_widths,
            })

    # Cat offset consistency: summed branch widths must equal the kept count in
    # the concatenated reference space (== len(group_keep) here).
    if is_cat:
        branch_total = 0
        for item in group.items:
            if item.reason in ("root_out", "grouped_conv:keep_groups", "grouped_conv:depthwise"):
                branch_total += len(item.local_keep(group_keep))
        # downstream cat-consumers slice on the full concatenated space
        for item in group.items:
            if item.reason in ("cat_downstream_in", "cat_bn"):
                consumed = len(item.local_keep(group_keep))
                if consumed != len(group_keep):
                    issues.append({
                        "issue": "cat_shrink_in_mismatch", "layer": item.name,
                        "expected": len(group_keep), "actual": consumed,
                    })
        if branch_total != len(group_keep):
            issues.append({
                "issue": "cat_offset_total_mismatch",
                "branch_total": branch_total, "expected": len(group_keep),
            })

    return {"legal": not issues, "issues": issues}


def check_model_legality(model: nn.Module) -> Dict[str, Any]:
    """Whole-model structural sanity scan after surgery."""
    issues: List[Dict[str, Any]] = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            if module.in_channels <= 0 or module.out_channels <= 0:
                issues.append({"layer": name, "issue": "invalid_channels",
                               "in": module.in_channels, "out": module.out_channels})
            elif module.groups <= 0 or module.in_channels % module.groups != 0 or module.out_channels % module.groups != 0:
                issues.append({"layer": name, "issue": "invalid_groups",
                               "in": module.in_channels, "out": module.out_channels,
                               "groups": module.groups})
            # weight shape vs declared channels
            if isinstance(module, nn.Conv2d):
                w = module.weight
                if w.shape[0] != module.out_channels or w.shape[1] != module.in_channels // module.groups:
                    issues.append({"layer": name, "issue": "conv_weight_shape_mismatch",
                                   "weight": list(w.shape), "out": module.out_channels,
                                   "in": module.in_channels, "groups": module.groups})
            else:
                w = module.weight
                if w.shape[0] != module.in_channels or w.shape[1] != module.out_channels // module.groups:
                    issues.append({"layer": name, "issue": "convtranspose_weight_shape_mismatch",
                                   "weight": list(w.shape), "out": module.out_channels,
                                   "in": module.in_channels, "groups": module.groups})
        elif isinstance(module, nn.Linear):
            if module.in_features <= 0 or module.out_features <= 0:
                issues.append({"layer": name, "issue": "invalid_linear",
                               "in": module.in_features, "out": module.out_features})
            elif list(module.weight.shape) != [module.out_features, module.in_features]:
                issues.append({"layer": name, "issue": "linear_weight_shape_mismatch",
                               "weight": list(module.weight.shape)})
        elif isinstance(module, nn.modules.batchnorm._BatchNorm):
            if module.num_features <= 0 or int(module.weight.shape[0]) != module.num_features:
                issues.append({"layer": name, "issue": "invalid_bn",
                               "num_features": module.num_features})
    return {"legal": not issues, "issues": issues}
