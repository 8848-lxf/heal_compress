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

from typing import Any, Callable, Dict, List

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


def _grouped_supports(module: nn.Module) -> bool:
    return isinstance(module, nn.Conv2d) and module.groups > 1


_prune_grouped_keep_groups.__name__ = "prune_grouped_keep_groups"
_prune_grouped_keep_groups.supports = _grouped_supports  # type: ignore[attr-defined]
_prune_grouped_independent_topk.__name__ = "prune_grouped_independent_topk"
_prune_grouped_independent_topk.supports = _grouped_supports  # type: ignore[attr-defined]
_prune_grouped_remove_groups.__name__ = "prune_grouped_remove_groups"
_prune_grouped_remove_groups.supports = _grouped_supports  # type: ignore[attr-defined]


def grouped_conv_pruning_fn(mode: str) -> Callable[[nn.Module, List[int]], Dict[str, Any]]:
    """Return the grouped-conv pruning_fn for ``mode``."""
    if mode == "remove_groups":
        return _prune_grouped_remove_groups
    if mode == "independent_group_topk":
        return _prune_grouped_independent_topk
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
