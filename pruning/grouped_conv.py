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
_prune_grouped_remove_groups.__name__ = "prune_grouped_remove_groups"
_prune_grouped_remove_groups.supports = _grouped_supports  # type: ignore[attr-defined]


def grouped_conv_pruning_fn(mode: str) -> Callable[[nn.Module, List[int]], Dict[str, Any]]:
    """Return the grouped-conv pruning_fn for ``mode``."""
    if mode == "remove_groups":
        return _prune_grouped_remove_groups
    return _prune_grouped_keep_groups
