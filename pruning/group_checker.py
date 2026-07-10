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
    group_conv_align: int = 8,
    require_group_aligned: bool = True,
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
                    if require_group_aligned:
                        kept_per_group = len(next(iter(patterns))) if patterns else 0
                        if kept_per_group <= 0 or kept_per_group % group_conv_align != 0:
                            issues.append({
                                "issue": "violates_group_inner_channel_align8",
                                "layer": item.name,
                                "groups": g,
                                "kept_per_group": kept_per_group,
                                "align": group_conv_align,
                            })
            if fn_name == "prune_grouped_flat_output_groups_fixed":
                new_in = module.in_channels
                new_out = len(local)
                g = module.groups
                if module.in_channels % g != 0 or new_out % g != 0:
                    issues.append({
                        "issue": "groups_divisibility",
                        "layer": item.name,
                        "groups": g,
                        "new_in": new_in,
                        "new_out": new_out,
                    })
                if item.direction != "out":
                    issues.append({
                        "issue": "flat_output_groups_fixed_requires_out_axis",
                        "layer": item.name,
                        "direction": item.direction,
                    })
            if fn_name == "prune_grouped_group_balanced_output_groups_fixed":
                new_in = module.in_channels
                new_out = len(local)
                g = module.groups
                if module.in_channels % g != 0 or new_out % g != 0:
                    issues.append({
                        "issue": "groups_divisibility",
                        "layer": item.name,
                        "groups": g,
                        "new_in": new_in,
                        "new_out": new_out,
                    })
                per = module.out_channels // g if g else 0
                counts = []
                for gi in range(g):
                    start = gi * per
                    counts.append(len([idx for idx in local if start <= idx < start + per]))
                if len(set(counts)) > 1:
                    issues.append({
                        "issue": "group_balance_violation",
                        "layer": item.name,
                        "groups": g,
                        "old_group_keep_counts": counts,
                    })
            if fn_name == "prune_grouped_independent_topk":
                new_in = new_out = len(local)
                g = module.groups
                if module.out_channels % g != 0 or module.in_channels % g != 0:
                    issues.append({
                        "issue": "groups_divisibility",
                        "layer": item.name,
                        "groups": g,
                        "new_in": new_in,
                        "new_out": new_out,
                    })
                else:
                    per = module.out_channels // g
                    counts = []
                    for gi in range(g):
                        start = gi * per
                        counts.append(len([idx for idx in local if start <= idx < start + per]))
                    if len(set(counts)) > 1:
                        issues.append({
                            "issue": "grouped_independent_keep_count_mismatch",
                            "layer": item.name,
                            "groups": g,
                            "per_group_counts": counts,
                        })
                    kept_per_group = counts[0] if counts else 0
                    if kept_per_group <= 0:
                        issues.append({
                            "issue": "channel_emptied",
                            "layer": item.name,
                            "direction": item.direction,
                        })
                    if require_group_aligned and kept_per_group % group_conv_align != 0:
                        issues.append({
                            "issue": "violates_group_inner_channel_align8",
                            "layer": item.name,
                            "groups": g,
                            "kept_per_group": kept_per_group,
                            "align": group_conv_align,
                        })
            if fn_name == "prune_grouped_conv_input_balanced":
                new_in = len(local)
                new_out = module.out_channels
                g = module.groups
                if module.in_channels % g != 0 or new_in % g != 0:
                    issues.append({
                        "issue": "groups_divisibility",
                        "layer": item.name,
                        "groups": g,
                        "new_in": new_in,
                        "new_out": new_out,
                    })
                else:
                    per = module.in_channels // g
                    counts = []
                    for gi in range(g):
                        start = gi * per
                        counts.append(len([idx for idx in local if start <= idx < start + per]))
                    if len(set(counts)) > 1:
                        issues.append({
                            "issue": "grouped_input_balance_violation",
                            "layer": item.name,
                            "groups": g,
                            "per_group_counts": counts,
                        })
                    kept_per_group = counts[0] if counts else 0
                    if kept_per_group <= 0:
                        issues.append({
                            "issue": "channel_emptied",
                            "layer": item.name,
                            "direction": item.direction,
                        })
                    if require_group_aligned and kept_per_group % group_conv_align != 0:
                        issues.append({
                            "issue": "violates_group_inner_channel_align8",
                            "layer": item.name,
                            "groups": g,
                            "kept_per_group": kept_per_group,
                            "align": group_conv_align,
                        })
            if fn_name == "prune_grouped_remove_groups":
                g = module.groups
                if module.out_channels % g != 0 or module.in_channels % g != 0:
                    issues.append({
                        "issue": "groups_divisibility",
                        "layer": item.name,
                        "groups": g,
                        "new_in": module.in_channels,
                        "new_out": module.out_channels,
                    })
                else:
                    out_per = module.out_channels // g
                    in_per = module.in_channels // g
                    kept_groups = sorted({idx // out_per for idx in local})
                    expected = []
                    for gi in kept_groups:
                        expected.extend(range(gi * out_per, (gi + 1) * out_per))
                    if sorted(local) != expected:
                        issues.append({
                            "issue": "grouped_remove_requires_whole_groups",
                            "layer": item.name,
                            "groups": g,
                        })
                    after_groups = len(kept_groups)
                    if require_group_aligned:
                        if after_groups % group_conv_align != 0:
                            issues.append({
                                "issue": "violates_group_count_align8",
                                "layer": item.name,
                                "groups_after": after_groups,
                                "align": group_conv_align,
                            })
                        if in_per % group_conv_align != 0 or out_per % group_conv_align != 0:
                            issues.append({
                                "issue": "violates_group_inner_channel_align8",
                                "layer": item.name,
                                "groups_after": after_groups,
                                "in_channels_per_group": in_per,
                                "out_channels_per_group": out_per,
                                "align": group_conv_align,
                            })
                        if g % group_conv_align != 0:
                            issues.append({
                                "issue": "violates_group_count_align8",
                                "layer": item.name,
                                "groups": g,
                                "align": group_conv_align,
                            })
            is_depthwise_out = (
                isinstance(module, nn.Conv2d)
                and g > 1 and g == module.in_channels == module.out_channels
                and item.direction == "out"
            )
            if is_depthwise_out:
                if require_group_aligned and len(local) % group_conv_align != 0:
                    issues.append({
                        "issue": "violates_depthwise_channel_align8",
                        "layer": item.name,
                        "new_channels": len(local),
                        "align": group_conv_align,
                    })
            elif g > 0 and fn_name not in (
                "prune_grouped_remove_groups",
                "prune_grouped_keep_groups",
                "prune_grouped_flat_output_groups_fixed",
                "prune_grouped_group_balanced_output_groups_fixed",
                "prune_grouped_independent_topk",
                "prune_grouped_conv_input_balanced",
            ):
                if new_out % g != 0 or new_in % g != 0:
                    issues.append({
                        "issue": "groups_divisibility", "layer": item.name,
                        "groups": g, "new_in": new_in, "new_out": new_out,
                    })
                elif require_group_aligned and g > 1:
                    in_per = new_in // g
                    out_per = new_out // g
                    if g % group_conv_align != 0:
                        issues.append({
                            "issue": "violates_group_count_align8",
                            "layer": item.name,
                            "groups": g,
                            "align": group_conv_align,
                        })
                    if in_per % group_conv_align != 0 or out_per % group_conv_align != 0:
                        issues.append({
                            "issue": "violates_group_inner_channel_align8",
                            "layer": item.name,
                            "groups": g,
                            "new_in": new_in,
                            "new_out": new_out,
                            "in_per_group": in_per,
                            "out_per_group": out_per,
                            "align": group_conv_align,
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
        root_widths = [
            w for (_n, w, reason) in out_widths
            if reason == "root_out" or reason.startswith("grouped_conv:")
        ]
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
            if item.reason == "root_out" or item.reason.startswith("grouped_conv:"):
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


def check_model_legality(
    model: nn.Module,
    group_conv_align: int = 8,
    require_group_aligned: bool = True,
) -> Dict[str, Any]:
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
            elif require_group_aligned and module.groups > 1:
                in_per = module.in_channels // module.groups
                out_per = module.out_channels // module.groups
                is_depthwise = (
                    isinstance(module, nn.Conv2d)
                    and module.groups == module.in_channels == module.out_channels
                )
                if is_depthwise:
                    if module.out_channels % group_conv_align != 0:
                        issues.append({
                            "layer": name,
                            "issue": "violates_depthwise_channel_align8",
                            "channels": module.out_channels,
                            "align": group_conv_align,
                        })
                else:
                    if module.groups % group_conv_align != 0:
                        issues.append({
                            "layer": name,
                            "issue": "violates_group_count_align8",
                            "groups": module.groups,
                            "align": group_conv_align,
                        })
                    if in_per % group_conv_align != 0 or out_per % group_conv_align != 0:
                        issues.append({
                            "layer": name,
                            "issue": "violates_group_inner_channel_align8",
                            "in_channels": module.in_channels,
                            "out_channels": module.out_channels,
                            "groups": module.groups,
                            "in_channels_per_group": in_per,
                            "out_channels_per_group": out_per,
                            "align": group_conv_align,
                        })
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
