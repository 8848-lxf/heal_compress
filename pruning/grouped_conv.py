"""Grouped / depthwise convolution pruning handlers (requirement #4).

Two strategies for a grouped Conv2d (``groups > 1``):

A. **keep_groups** - the number of ``groups`` stays fixed; every group keeps the
   *same* local channel indices, so the ``in/out`` per-group widths shrink
   uniformly and the ``groups`` attribute is unchanged. Requires in==out width
   (the common case for the bottleneck grouped 3x3).

B. **remove_groups** - whole groups are dropped. ``groups`` is reduced and the
   corresponding ``in/out`` slabs are removed together, keeping
   ``in_channels % groups == out_channels % groups == 0``.

Depthwise conv (``groups == in == out``) is handled by the standard
``prune_conv_out`` path (it already syncs in/out/groups), so it is not routed
here.

A grouped conv whose resulting per-group width would not be hardware-friendly
(not a power-of-two-ish regular width, or violating the requested ``align``) is
classified ``"protected"`` so the enclosing :class:`PruningGroup` is skipped
atomically.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List

import torch
import torch.nn as nn

from .pruning_fns import _index

# Per-group widths considered hardware-regular.
REGULAR_GROUP_WIDTHS = (16, 8, 4, 2, 1)


def _is_depthwise(module: nn.Conv2d) -> bool:
    return module.groups == module.in_channels == module.out_channels


def classify_grouped_conv(module: nn.Conv2d, align: int) -> str:
    """Classify how a grouped conv may be pruned.

    Returns one of ``"depthwise"``, ``"keep_groups"``, ``"remove_groups"`` or
    ``"protected"``.
    """
    if module.groups <= 1:
        return "keep_groups"  # not grouped; standard path
    if _is_depthwise(module):
        return "depthwise"
    g = module.groups
    if module.in_channels % g != 0 or module.out_channels % g != 0:
        return "protected"
    in_per = module.in_channels // g
    out_per = module.out_channels // g
    # Alignment: total channels should respect ``align`` where feasible.
    if align > 1 and g % align != 0 and module.out_channels % align != 0:
        # groups not aligned and channel count not aligned -> unsafe to reshape
        # for most fixed-shape inference kernels.
        if in_per not in REGULAR_GROUP_WIDTHS or out_per not in REGULAR_GROUP_WIDTHS:
            return "protected"
    if module.in_channels == module.out_channels:
        return "keep_groups"
    return "remove_groups"


# --------------------------------------------------------------------------- #
# Handler A0: flat output-only, groups fixed
# --------------------------------------------------------------------------- #
def _prune_grouped_flat_output_groups_fixed(module: nn.Conv2d, keep: List[int]) -> Dict[str, Any]:
    """Slice only output filters of a regular grouped Conv2d.

    This is the project-pruner implementation of the TP-like output pruning
    baseline. It intentionally does *not* prune the current grouped conv input
    axis and does *not* change ``groups``. Uneven old-group keep counts are
    allowed; they are recorded by the selector as a reinterpretation risk.
    """
    if module.groups <= 1:
        raise ValueError("flat_output_groups_fixed expects groups > 1")
    if _is_depthwise(module):
        raise ValueError("flat_output_groups_fixed excludes depthwise conv")
    if module.in_channels % module.groups != 0:
        raise ValueError(
            f"flat_output_groups_fixed: in_channels {module.in_channels} not divisible by groups {module.groups}"
        )
    if len(keep) <= 0:
        raise ValueError("flat_output_groups_fixed: must keep at least one output channel")
    if len(keep) % module.groups != 0:
        raise ValueError(
            f"flat_output_groups_fixed: C_out_after {len(keep)} not divisible by groups {module.groups}"
        )
    before_in, before_out, before_groups = module.in_channels, module.out_channels, module.groups
    idx = _index(sorted(keep), module.weight.device)
    module.weight = nn.Parameter(module.weight.data.index_select(0, idx).clone())
    if module.bias is not None:
        module.bias = nn.Parameter(module.bias.data.index_select(0, idx).clone())
    module.out_channels = len(keep)
    return {
        "axis": "grouped_flat_output",
        "before_in": before_in,
        "after_in": module.in_channels,
        "before_out": before_out,
        "after_out": module.out_channels,
        "before_groups": before_groups,
        "after_groups": module.groups,
        "input_pruned": False,
        "groups_changed": False,
    }


def _prune_grouped_group_balanced_output_groups_fixed(module: nn.Conv2d, keep: List[int]) -> Dict[str, Any]:
    """Output-only grouped Conv2d pruning with equal old-group keep counts."""
    if module.groups <= 1:
        raise ValueError("group_balanced_output_groups_fixed expects groups > 1")
    if _is_depthwise(module):
        raise ValueError("group_balanced_output_groups_fixed excludes depthwise conv")
    keep_map = _group_local_keep_map(sorted(keep), module.out_channels, module.groups)
    counts = {len(v) for v in keep_map.values()}
    if len(counts) != 1:
        raise ValueError("group_balanced_output_groups_fixed requires equal old-group keep counts")
    return _prune_grouped_flat_output_groups_fixed(module, keep) | {
        "axis": "grouped_group_balanced_output",
        "group_keep_map": keep_map,
        "group_balance_pass": True,
    }


# --------------------------------------------------------------------------- #
# Handler A: keep groups fixed, shared local keep
# --------------------------------------------------------------------------- #
def _shared_local_keep(keep: List[int], channels: int, groups: int) -> List[int]:
    """Validate that ``keep`` uses the same local pattern in every group."""
    if channels % groups != 0:
        raise ValueError(f"grouped conv channels {channels} not divisible by groups {groups}")
    per = channels // groups
    expected: List[int] | None = None
    for gi in range(groups):
        start = gi * per
        local = sorted(idx - start for idx in keep if start <= idx < start + per)
        if expected is None:
            expected = local
        elif local != expected:
            raise ValueError("keep_groups requires identical local keep indices in every group")
    return expected or []


def _prune_grouped_keep_groups(module: nn.Conv2d, keep: List[int]) -> Dict[str, Any]:
    """Slice a grouped conv keeping ``groups`` fixed (in==out width)."""
    if module.in_channels != module.out_channels:
        raise ValueError(
            f"keep_groups handler needs in==out (in={module.in_channels}, out={module.out_channels})"
        )
    before_in, before_out = module.in_channels, module.out_channels
    local = _shared_local_keep(keep, module.out_channels, module.groups)
    if not local:
        raise ValueError("keep_groups: each group must retain >=1 channel")
    out_idx = _index(keep, module.weight.device)
    in_idx = _index(local, module.weight.device)
    # weight: [out, in/groups, kH, kW]; out axis selects whole channels, axis 1
    # selects the shared per-group local indices.
    module.weight = nn.Parameter(
        module.weight.data.index_select(0, out_idx).index_select(1, in_idx).clone()
    )
    if module.bias is not None:
        module.bias = nn.Parameter(module.bias.data.index_select(0, out_idx).clone())
    module.in_channels = len(keep)
    module.out_channels = len(keep)
    # groups unchanged
    return {
        "axis": "grouped_keep", "before_in": before_in, "after_in": module.in_channels,
        "before_out": before_out, "after_out": module.out_channels, "groups": module.groups,
    }


def _group_local_keep_map(keep: List[int], channels: int, groups: int) -> Dict[int, List[int]]:
    if channels % groups != 0:
        raise ValueError(f"grouped conv channels {channels} not divisible by groups {groups}")
    per = channels // groups
    keep_map: Dict[int, List[int]] = {}
    for gi in range(groups):
        start = gi * per
        keep_map[gi] = sorted(idx - start for idx in keep if start <= idx < start + per)
    return keep_map


def _prune_grouped_independent_topk(module: nn.Conv2d, keep: List[int]) -> Dict[str, Any]:
    """Slice a grouped conv with an independent local keep map per group.

    ``groups`` remains unchanged. Each group must keep the same number of local
    channels, but the local positions may differ by group. The handler repacks
    each retained group into a contiguous output block and a contiguous
    within-group input block.
    """
    if module.in_channels != module.out_channels:
        raise ValueError(
            "independent_group_topk currently requires in_channels == out_channels "
            f"(in={module.in_channels}, out={module.out_channels})"
        )
    g = module.groups
    before_in, before_out = module.in_channels, module.out_channels
    per = module.out_channels // g
    keep_map = _group_local_keep_map(keep, module.out_channels, g)
    counts = {len(v) for v in keep_map.values()}
    if len(counts) != 1:
        raise ValueError("independent_group_topk requires the same kept count in every group")
    keep_per = counts.pop()
    if keep_per <= 0:
        raise ValueError("independent_group_topk: each group must retain >=1 channel")

    new_weight = module.weight.data.new_empty(g * keep_per, keep_per, *module.weight.shape[2:])
    for gi in range(g):
        local = keep_map[gi]
        out_abs = [gi * per + idx for idx in local]
        out_idx = _index(out_abs, module.weight.device)
        in_idx = _index(local, module.weight.device)
        block = module.weight.data.index_select(0, out_idx).index_select(1, in_idx)
        new_weight[gi * keep_per:(gi + 1) * keep_per].copy_(block)
    module.weight = nn.Parameter(new_weight.clone())
    if module.bias is not None:
        out_keep = [gi * per + idx for gi in range(g) for idx in keep_map[gi]]
        module.bias = nn.Parameter(module.bias.data.index_select(0, _index(out_keep, module.bias.device)).clone())
    module.in_channels = len(keep)
    module.out_channels = len(keep)
    return {
        "axis": "grouped_independent_keep",
        "before_in": before_in,
        "after_in": module.in_channels,
        "before_out": before_out,
        "after_out": module.out_channels,
        "groups": module.groups,
        "per_group_after": keep_per,
        "group_keep_map": keep_map,
    }


# --------------------------------------------------------------------------- #
# Handler B: remove whole groups
# --------------------------------------------------------------------------- #
def _prune_grouped_remove_groups(module: nn.Conv2d, keep: List[int]) -> Dict[str, Any]:
    """Drop whole groups; reduce ``groups`` and slice in/out slabs together.

    ``keep`` is interpreted as the set of *output* channels to retain; the
    groups they belong to are kept whole (a partially-kept group is rounded up
    to the whole group) and the matching input slabs are removed.
    """
    g = module.groups
    out_per = module.out_channels // g
    in_per = module.in_channels // g
    kept_groups = sorted({idx // out_per for idx in keep})
    if not kept_groups:
        raise ValueError("remove_groups: must keep >=1 group")
    expected_keep: List[int] = []
    for gi in kept_groups:
        expected_keep.extend(range(gi * out_per, (gi + 1) * out_per))
    if sorted(keep) != expected_keep:
        raise ValueError("remove_groups requires whole output-group slabs")
    out_keep: List[int] = []
    in_keep: List[int] = []
    for gi in kept_groups:
        out_keep.extend(range(gi * out_per, (gi + 1) * out_per))
        in_keep.extend(range(gi * in_per, (gi + 1) * in_per))
    before_in, before_out, before_g = module.in_channels, module.out_channels, g
    out_idx = _index(out_keep, module.weight.device)
    in_idx = _index(list(range(in_per)), module.weight.device)  # within-group, axis1 already per-group
    # weight axis1 length == in/groups; removing whole groups keeps axis1 size,
    # only the out axis changes, and groups shrinks.
    module.weight = nn.Parameter(module.weight.data.index_select(0, out_idx).clone())
    if module.bias is not None:
        module.bias = nn.Parameter(module.bias.data.index_select(0, out_idx).clone())
    module.out_channels = len(out_keep)
    module.in_channels = len(kept_groups) * in_per
    module.groups = len(kept_groups)
    return {
        "axis": "grouped_remove", "before_in": before_in, "after_in": module.in_channels,
        "before_out": before_out, "after_out": module.out_channels,
        "before_groups": before_g, "after_groups": module.groups,
        "kept_groups": kept_groups,
    }


# --------------------------------------------------------------------------- #
# Handler C: true old group-block removal
# --------------------------------------------------------------------------- #
def _validate_grouped_true_group_block_module(module: nn.Conv2d) -> tuple[int, int, int]:
    if not isinstance(module, nn.Conv2d) or module.groups <= 1:
        raise ValueError("true_group_block expects ordinary grouped Conv2d")
    if _is_depthwise(module):
        raise ValueError("true_group_block rejects depthwise conv")
    groups = int(module.groups)
    if module.in_channels % groups != 0 or module.out_channels % groups != 0:
        raise ValueError("true_group_block requires in/out channels divisible by groups")
    return groups, int(module.in_channels // groups), int(module.out_channels // groups)


def _complete_group_blocks_from_channel_prune(
    prune_indices: Iterable[int],
    *,
    channels: int,
    groups: int,
) -> tuple[bool, list[int], dict[int, list[int]]]:
    per = channels // groups
    prune_set = sorted({int(idx) for idx in prune_indices if 0 <= int(idx) < channels})
    by_group: dict[int, list[int]] = {group_id: [] for group_id in range(groups)}
    for idx in prune_set:
        by_group[idx // per].append(idx % per)
    pruned_groups = sorted(group_id for group_id, local in by_group.items() if local)
    complete = all(by_group[group_id] == list(range(per)) for group_id in pruned_groups)
    return complete, pruned_groups, by_group


def resolve_grouped_conv_true_group_block_keep(
    module: nn.Conv2d,
    *,
    prune_indices: Iterable[int] | None = None,
    pruned_old_groups: Iterable[int] | None = None,
    kept_old_groups: Iterable[int] | None = None,
) -> Dict[str, Any]:
    """Resolve C strategy keep groups for ordinary grouped Conv2d.

    ``prune_indices`` are interpreted as old output-channel indices and must
    cover complete old output groups.  ``pruned_old_groups`` and
    ``kept_old_groups`` are explicit old group ids and are used by one-shot
    global plans.
    """
    try:
        groups, in_per, out_per = _validate_grouped_true_group_block_module(module)
    except ValueError as exc:
        return {"legal": False, "reason": str(exc)}

    if kept_old_groups is not None:
        kept = sorted({int(group_id) for group_id in kept_old_groups if 0 <= int(group_id) < groups})
        pruned = [group_id for group_id in range(groups) if group_id not in set(kept)]
    elif pruned_old_groups is not None:
        pruned = sorted({int(group_id) for group_id in pruned_old_groups if 0 <= int(group_id) < groups})
        kept = [group_id for group_id in range(groups) if group_id not in set(pruned)]
    else:
        complete, pruned, by_group = _complete_group_blocks_from_channel_prune(
            prune_indices or [],
            channels=int(module.out_channels),
            groups=groups,
        )
        if not complete:
            return {
                "legal": False,
                "reason": "c_strategy_requires_complete_old_group_block",
                "old_groups": groups,
                "in_per_group_before": in_per,
                "out_per_group_before": out_per,
                "group_local_pruned_indices": by_group,
            }
        kept = [group_id for group_id in range(groups) if group_id not in set(pruned)]

    if not kept:
        return {
            "legal": False,
            "reason": "c_strategy_requires_at_least_one_group_after_prune",
            "old_groups": groups,
            "pruned_old_groups": pruned,
        }
    groups_after = len(kept)
    return {
        "legal": True,
        "reason": "",
        "old_groups": groups,
        "kept_old_groups": kept,
        "pruned_old_groups": pruned,
        "in_per_group_before": in_per,
        "out_per_group_before": out_per,
        "groups_after": groups_after,
        "in_channels_after": groups_after * in_per,
        "out_channels_after": groups_after * out_per,
    }


def prune_grouped_conv_true_group_block(module: nn.Conv2d, kept_old_groups: List[int]) -> Dict[str, Any]:
    """Physically remove complete old groups from an ordinary grouped Conv2d.

    The caller must separately synchronize the upstream producer output,
    following BN, and downstream consumer input.  This function changes only the
    grouped Conv2d module itself using old group ids as the stable reference
    space.
    """
    resolved = resolve_grouped_conv_true_group_block_keep(module, kept_old_groups=kept_old_groups)
    if not resolved.get("legal", False):
        raise ValueError(str(resolved.get("reason") or "invalid_true_group_block_keep"))
    groups = int(resolved["old_groups"])
    in_per = int(resolved["in_per_group_before"])
    out_per = int(resolved["out_per_group_before"])
    kept = list(resolved["kept_old_groups"])
    out_keep: list[int] = []
    for group_id in kept:
        out_keep.extend(range(group_id * out_per, (group_id + 1) * out_per))

    before_in = int(module.in_channels)
    before_out = int(module.out_channels)
    before_groups = int(module.groups)
    out_idx = _index(out_keep, module.weight.device)
    module.weight = nn.Parameter(module.weight.data.index_select(0, out_idx).clone())
    if module.bias is not None:
        module.bias = nn.Parameter(module.bias.data.index_select(0, out_idx).clone())
    module.in_channels = len(kept) * in_per
    module.out_channels = len(kept) * out_per
    module.groups = len(kept)
    return {
        "axis": "grouped_true_group_block",
        "before_in": before_in,
        "after_in": module.in_channels,
        "before_out": before_out,
        "after_out": module.out_channels,
        "before_groups": before_groups,
        "after_groups": module.groups,
        "groups_after": module.groups,
        "old_groups": groups,
        "kept_old_groups": kept,
        "pruned_old_groups": resolved["pruned_old_groups"],
        "in_per_group_before": in_per,
        "in_per_group_after": in_per,
        "out_per_group_before": out_per,
        "out_per_group_after": out_per,
        "groups_changed": True,
    }


# --------------------------------------------------------------------------- #
# Handler D: compact-first frontfill zero-padded reblock
# --------------------------------------------------------------------------- #
def _ordered_unique_valid(indices: Iterable[int], upper: int) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for raw in indices:
        idx = int(raw)
        if idx < 0 or idx >= upper or idx in seen:
            continue
        seen.add(idx)
        out.append(idx)
    return out


def resolve_grouped_conv_d_compact_frontfill_reblock(
    module: nn.Conv2d,
    *,
    old_output_keep_indices: Iterable[int],
    old_input_keep_indices: Iterable[int] | None = None,
    groups_new: int,
) -> Dict[str, Any]:
    """Resolve D compact-frontfill reblock metadata for ordinary grouped Conv2d.

    D is intentionally not semantic-preserving. Kept old output filters are
    compacted in the requested order, each old local kernel slice is copied to
    the front of the new group's input slice, and newly introduced connections
    are zero-initialized.
    """
    if not isinstance(module, nn.Conv2d) or module.groups <= 1:
        return {"legal": False, "reason": "d_strategy_requires_ordinary_grouped_conv"}
    if _is_depthwise(module):
        return {"legal": False, "reason": "depthwise_not_supported_for_d_reblock"}
    groups_old = int(module.groups)
    c_in_old = int(module.in_channels)
    c_out_old = int(module.out_channels)
    if c_in_old % groups_old != 0 or c_out_old % groups_old != 0:
        return {"legal": False, "reason": "old_grouped_conv_divisibility_violation"}
    groups_new = int(groups_new)
    if groups_new <= 0:
        return {"legal": False, "reason": "groups_new_must_be_positive"}

    out_keep = _ordered_unique_valid(old_output_keep_indices, c_out_old)
    in_keep = _ordered_unique_valid(old_input_keep_indices if old_input_keep_indices is not None else range(c_in_old), c_in_old)
    c_out_new = len(out_keep)
    c_in_new = len(in_keep)
    if c_in_new <= 0 or c_out_new <= 0:
        return {"legal": False, "reason": "d_strategy_requires_nonempty_input_and_output_keep"}
    if c_in_new % groups_new != 0:
        return {"legal": False, "reason": "d_reblock_input_divisibility_violation", "C_in_new": c_in_new, "groups_new": groups_new}
    if c_out_new % groups_new != 0:
        return {"legal": False, "reason": "d_reblock_output_divisibility_violation", "C_out_new": c_out_new, "groups_new": groups_new}

    in_per_old = c_in_old // groups_old
    out_per_old = c_out_old // groups_old
    in_per_new = c_in_new // groups_new
    out_per_new = c_out_new // groups_new
    old_slice_width = int(module.weight.shape[1])
    if in_per_new < old_slice_width:
        return {
            "legal": False,
            "reason": "frontfill_source_kernel_too_wide",
            "groups_old": groups_old,
            "groups_new": groups_new,
            "frontfill_copy_width": old_slice_width,
            "in_per_group_new": in_per_new,
        }
    if out_per_new <= 0:
        return {"legal": False, "reason": "d_reblock_out_per_group_must_be_positive"}

    mapping = {str(old_out): new_out for new_out, old_out in enumerate(out_keep)}
    copy_plan = []
    zero_pad_width = in_per_new - old_slice_width
    for new_out, old_out in enumerate(out_keep):
        copy_plan.append(
            {
                "old_out_idx": int(old_out),
                "new_out_idx": int(new_out),
                "old_group": int(old_out // out_per_old),
                "new_group": int(new_out // out_per_new),
                "frontfill_copy_range": [0, int(old_slice_width)],
                "zero_pad_range": [int(old_slice_width), int(in_per_new)],
            }
        )

    return {
        "legal": True,
        "reason": "",
        "groups_old": groups_old,
        "groups_new": groups_new,
        "C_in_old": c_in_old,
        "C_out_old": c_out_old,
        "C_in_new": c_in_new,
        "C_out_new": c_out_new,
        "in_per_group_old": in_per_old,
        "out_per_group_old": out_per_old,
        "in_per_group_new": in_per_new,
        "out_per_group_new": out_per_new,
        "old_output_to_new_output_map": mapping,
        "old_input_keep_indices": in_keep,
        "old_output_keep_indices": out_keep,
        "frontfill_copy_width": old_slice_width,
        "zero_pad_width": zero_pad_width,
        "copy_plan": copy_plan,
        "semantic_preserved": False,
        "compact_first": True,
        "frontfill_weight_transplant": True,
        "weight_values_retained": True,
        "new_connections_zero_initialized": True,
        "requires_recovery_finetune": True,
    }


def prune_grouped_conv_d_compact_frontfill_reblock(
    module: nn.Conv2d,
    *,
    old_output_keep_indices: Iterable[int],
    old_input_keep_indices: Iterable[int] | None = None,
    groups_new: int,
) -> Dict[str, Any]:
    """Apply D compact-frontfill zero-padded reblock to a grouped Conv2d."""
    resolved = resolve_grouped_conv_d_compact_frontfill_reblock(
        module,
        old_output_keep_indices=old_output_keep_indices,
        old_input_keep_indices=old_input_keep_indices,
        groups_new=groups_new,
    )
    if not resolved.get("legal", False):
        raise ValueError(str(resolved.get("reason") or "invalid_d_compact_frontfill_reblock"))

    old_weight = module.weight.data
    old_bias = module.bias.data if module.bias is not None else None
    out_keep = list(resolved["old_output_keep_indices"])
    c_out_new = int(resolved["C_out_new"])
    in_per_new = int(resolved["in_per_group_new"])
    copy_width = int(resolved["frontfill_copy_width"])
    new_weight = old_weight.new_zeros(c_out_new, in_per_new, *old_weight.shape[2:])
    for new_out, old_out in enumerate(out_keep):
        new_weight[new_out, :copy_width].copy_(old_weight[int(old_out), :copy_width])
    new_bias = None
    if old_bias is not None:
        new_bias = old_bias.index_select(0, _index(out_keep, old_bias.device)).clone()

    before_in = int(module.in_channels)
    before_out = int(module.out_channels)
    before_groups = int(module.groups)
    module.weight = nn.Parameter(new_weight.clone())
    if new_bias is not None:
        module.bias = nn.Parameter(new_bias)
    module.in_channels = int(resolved["C_in_new"])
    module.out_channels = int(resolved["C_out_new"])
    module.groups = int(resolved["groups_new"])
    report = dict(resolved)
    report.update(
        {
            "axis": "grouped_d_compact_frontfill_reblock",
            "before_in": before_in,
            "after_in": module.in_channels,
            "before_out": before_out,
            "after_out": module.out_channels,
            "before_groups": before_groups,
            "after_groups": module.groups,
            "groups_after": module.groups,
        }
    )
    return report


def _grouped_supports(module: nn.Module) -> bool:
    return isinstance(module, nn.Conv2d) and module.groups > 1


def resolve_grouped_conv_input_keep(
    module: nn.Conv2d,
    keep: List[int],
    *,
    allow_repair: bool = True,
    min_in_per_group: int = 1,
) -> Dict[str, Any]:
    """Validate or repair keep indices for ``upstream.out -> grouped_conv.in``.

    Output channels and ``groups`` are intentionally unchanged.  The only legal
    input compaction is group-balanced: every old input group keeps the same
    number of local input channels, so weight axis 1 becomes
    ``C_in_after / groups``.
    """
    if not isinstance(module, nn.Conv2d) or module.groups <= 1 or _is_depthwise(module):
        return {"legal": False, "reason": "unsupported_grouped_input_module", "keep_indices": sorted(keep)}
    groups = int(module.groups)
    c_in = int(module.in_channels)
    if c_in % groups != 0:
        return {"legal": False, "reason": "grouped_input_divisibility_violation", "keep_indices": sorted(keep)}
    per = c_in // groups
    keep_set = {int(idx) for idx in keep if 0 <= int(idx) < c_in}
    group_keep_map: Dict[int, List[int]] = {}
    for group_id in range(groups):
        start = group_id * per
        group_keep_map[group_id] = sorted(idx - start for idx in keep_set if start <= idx < start + per)
    counts = {group_id: len(local) for group_id, local in group_keep_map.items()}
    balanced = len(set(counts.values())) == 1
    keep_per = next(iter(counts.values()), 0) if counts else 0
    if balanced and keep_per >= int(min_in_per_group) and len(keep_set) == keep_per * groups:
        return {
            "legal": True,
            "repaired": False,
            "reason": "",
            "keep_indices": sorted(keep_set),
            "groups": groups,
            "in_per_group_before": per,
            "in_per_group_after": keep_per,
            "per_group_kept_count": counts,
            "group_keep_map": group_keep_map,
        }
    if not allow_repair:
        return {
            "legal": False,
            "repaired": False,
            "reason": "grouped_input_balance_violation",
            "keep_indices": sorted(keep_set),
            "groups": groups,
            "in_per_group_before": per,
            "per_group_kept_count": counts,
            "group_keep_map": group_keep_map,
        }
    target_keep_per = int(round(len(keep_set) / max(groups, 1)))
    target_keep_per = max(int(min_in_per_group), min(per, target_keep_per))
    repaired_keep: List[int] = []
    repaired_map: Dict[int, List[int]] = {}
    for group_id in range(groups):
        start = group_id * per
        preferred = set(group_keep_map[group_id])
        ranked = sorted(range(per), key=lambda local: (local not in preferred, local))
        local_keep = sorted(ranked[:target_keep_per])
        repaired_map[group_id] = local_keep
        repaired_keep.extend(start + local for local in local_keep)
    return {
        "legal": True,
        "repaired": True,
        "reason": "grouped_input_balance_violation",
        "keep_indices": repaired_keep,
        "groups": groups,
        "in_per_group_before": per,
        "in_per_group_after": target_keep_per,
        "per_group_kept_count": {group_id: target_keep_per for group_id in range(groups)},
        "group_keep_map": repaired_map,
    }


def _prune_grouped_conv_input_balanced(module: nn.Conv2d, keep: List[int]) -> Dict[str, Any]:
    resolved = resolve_grouped_conv_input_keep(module, keep, allow_repair=False)
    if not resolved.get("legal", False):
        raise ValueError(str(resolved.get("reason") or "grouped_input_balance_violation"))
    groups = int(module.groups)
    before_in = int(module.in_channels)
    before_out = int(module.out_channels)
    in_per_before = before_in // groups
    out_per = before_out // groups
    in_per_after = int(resolved["in_per_group_after"])
    keep_map: Dict[int, List[int]] = resolved["group_keep_map"]
    old_weight = module.weight.data
    new_weight = old_weight.new_empty(before_out, in_per_after, *old_weight.shape[2:])
    for group_id in range(groups):
        local_idx = _index(keep_map[group_id], old_weight.device)
        out_start = group_id * out_per
        out_end = out_start + out_per
        new_weight[out_start:out_end].copy_(old_weight[out_start:out_end].index_select(1, local_idx))
    module.weight = nn.Parameter(new_weight.clone())
    module.in_channels = int(len(resolved["keep_indices"]))
    return {
        "axis": "grouped_input_balanced",
        "before_in": before_in,
        "after_in": module.in_channels,
        "before_out": before_out,
        "after_out": module.out_channels,
        "before_groups": groups,
        "after_groups": module.groups,
        "in_per_group_before": in_per_before,
        "in_per_group_after": in_per_after,
        "group_keep_map": keep_map,
        "input_pruned": True,
        "output_pruned": False,
        "groups_changed": False,
    }


_prune_grouped_keep_groups.__name__ = "prune_grouped_keep_groups"
_prune_grouped_keep_groups.supports = _grouped_supports  # type: ignore[attr-defined]
_prune_grouped_flat_output_groups_fixed.__name__ = "prune_grouped_flat_output_groups_fixed"
_prune_grouped_flat_output_groups_fixed.supports = _grouped_supports  # type: ignore[attr-defined]
_prune_grouped_group_balanced_output_groups_fixed.__name__ = "prune_grouped_group_balanced_output_groups_fixed"
_prune_grouped_group_balanced_output_groups_fixed.supports = _grouped_supports  # type: ignore[attr-defined]
_prune_grouped_independent_topk.__name__ = "prune_grouped_independent_topk"
_prune_grouped_independent_topk.supports = _grouped_supports  # type: ignore[attr-defined]
_prune_grouped_remove_groups.__name__ = "prune_grouped_remove_groups"
_prune_grouped_remove_groups.supports = _grouped_supports  # type: ignore[attr-defined]
prune_grouped_conv_true_group_block.__name__ = "prune_grouped_conv_true_group_block"
prune_grouped_conv_true_group_block.supports = _grouped_supports  # type: ignore[attr-defined]
prune_grouped_conv_d_compact_frontfill_reblock.__name__ = "prune_grouped_conv_d_compact_frontfill_reblock"
prune_grouped_conv_d_compact_frontfill_reblock.supports = _grouped_supports  # type: ignore[attr-defined]
prune_grouped_conv_input_balanced = _prune_grouped_conv_input_balanced
prune_grouped_conv_input_balanced.__name__ = "prune_grouped_conv_input_balanced"
prune_grouped_conv_input_balanced.supports = _grouped_supports  # type: ignore[attr-defined]


def grouped_conv_pruning_fn(mode: str) -> Callable[[nn.Module, List[int]], Dict[str, Any]]:
    """Return the grouped-conv pruning_fn for ``mode``."""
    if mode in {"A", "flat_output_groups_fixed", "group_coarsening_zero_padded_reblock"}:
        return _prune_grouped_flat_output_groups_fixed
    if mode in {"B", "group_balanced_output_groups_fixed"}:
        return _prune_grouped_group_balanced_output_groups_fixed
    if mode == "remove_groups":
        return _prune_grouped_remove_groups
    if mode in {"true_group_block", "grouped_true_group_block"}:
        return prune_grouped_conv_true_group_block
    if mode in {"D", "compact_frontfill_zero_padded_reblock", "grouped_d_compact_frontfill_reblock"}:
        return prune_grouped_conv_d_compact_frontfill_reblock
    if mode == "independent_group_topk":
        return _prune_grouped_independent_topk
    if mode == "input_balanced":
        return prune_grouped_conv_input_balanced
    return _prune_grouped_keep_groups


def merge_grouped_conv_groups(module: nn.Conv2d, merge_factor: int) -> Dict[str, Any]:
    """Merge adjacent grouped-conv groups with block-diagonal weights.

    This is a structure-normalization step, not pruning. It preserves the
    original grouped-conv function by placing each old group into a diagonal
    block inside the merged group and filling cross-group connections with zero.
    """
    if not isinstance(module, nn.Conv2d) or module.groups <= 1:
        raise ValueError("merge_grouped_conv_groups expects grouped Conv2d")
    if _is_depthwise(module):
        raise ValueError("depthwise conv groups are not merge-normalized")
    if merge_factor <= 1 or module.groups % merge_factor != 0:
        raise ValueError(f"invalid merge_factor={merge_factor} for groups={module.groups}")
    old_groups = module.groups
    new_groups = old_groups // merge_factor
    if module.in_channels % old_groups != 0 or module.out_channels % old_groups != 0:
        raise ValueError("grouped conv channels must be divisible by old groups")
    old_in_per = module.in_channels // old_groups
    old_out_per = module.out_channels // old_groups
    new_in_per = old_in_per * merge_factor
    new_weight = module.weight.data.new_zeros(
        module.out_channels,
        new_in_per,
        *module.weight.shape[2:],
    )
    for new_g in range(new_groups):
        for rel in range(merge_factor):
            old_g = new_g * merge_factor + rel
            out_start = old_g * old_out_per
            out_end = out_start + old_out_per
            in_start = rel * old_in_per
            in_end = in_start + old_in_per
            new_weight[out_start:out_end, in_start:in_end].copy_(
                module.weight.data[out_start:out_end]
            )
    module.weight = nn.Parameter(new_weight.clone())
    module.groups = new_groups
    return {
        "axis": "grouped_merge",
        "before_in": module.in_channels,
        "after_in": module.in_channels,
        "before_out": module.out_channels,
        "after_out": module.out_channels,
        "before_groups": old_groups,
        "after_groups": new_groups,
        "merge_factor": merge_factor,
    }


def grouped_conv_alignment_merge_factor(module: nn.Conv2d, align: int) -> int | None:
    """Return a safe group-merge factor that satisfies group/per-group align."""
    if not isinstance(module, nn.Conv2d) or module.groups <= 1 or _is_depthwise(module):
        return None
    if module.in_channels % module.groups != 0 or module.out_channels % module.groups != 0:
        return None
    in_per = module.in_channels // module.groups
    out_per = module.out_channels // module.groups
    if module.groups % align == 0 and in_per % align == 0 and out_per % align == 0:
        return None
    for factor in range(2, module.groups + 1):
        if module.groups % factor != 0:
            continue
        new_groups = module.groups // factor
        if new_groups % align != 0:
            continue
        if (in_per * factor) % align != 0:
            continue
        if (out_per * factor) % align != 0:
            continue
        return factor
    return None
