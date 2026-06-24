"""Torch-Pruning style PruningGroup abstraction.

A :class:`PruningGroup` is the unit of structured pruning. It bundles every
operation that must be modified together when a single set of channels is
removed, so the surgery is applied atomically: either every member is pruned
or none is.

Unlike a layer-level coupled group (which only records *which* layers are
coupled), a :class:`GroupItem` records the full execution recipe:

* ``module``        - the live ``nn.Module`` to mutate
* ``pruning_fn``    - the callable that physically slices the module
* ``direction``     - ``"out"`` / ``"in"`` (which side of the module is cut)
* ``idxs``          - the channel indices (in *this op's* local index space)
* ``idx_transform`` - maps group-level keep indices into this op's space
                      (e.g. a ``concat`` offset or a depthwise group remap)
* ``reason``        - human-readable provenance (which rule added this item)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

import torch.nn as nn

logger = logging.getLogger(__name__)


# An idx_transform maps a list of group-level keep indices to the local keep
# indices for one specific GroupItem. ``None`` means identity.
IdxTransform = Optional[Callable[[List[int]], List[int]]]


def identity_transform(keep: list[int]) -> list[int]:
    """Identity idx_transform: local indices == group indices."""
    return list(keep)


def offset_transform(offset: int, channels: int) -> Callable[[list[int]], list[int]]:
    """Build an idx_transform for a concat branch occupying ``[offset, offset+channels)``.

    Given group-level keep indices expressed in the *concatenated* index space,
    return the subset that falls inside this branch, rebased to local indices.

    Args:
        offset: Start channel of this branch inside the concat output.
        channels: Number of channels this branch contributes.

    Returns:
        A callable mapping concat-space keep indices -> local keep indices.
    """

    def _transform(keep: list[int]) -> list[int]:
        lo, hi = offset, offset + channels
        return sorted(idx - offset for idx in keep if lo <= idx < hi)

    return _transform


@dataclass
class GroupItem:
    """A single coupled pruning operation inside a :class:`PruningGroup`.

    Attributes:
        name: Fully qualified module name.
        module: The live module to mutate.
        pruning_fn: Callable ``(module, keep_indices) -> dict`` performing the
            physical slice. See :mod:`heal_compress.pruning.pruning_fns`.
        direction: ``"out"`` or ``"in"`` - which channel axis is sliced.
        idxs: Local channel indices owned by this item (full range before
            pruning); informational, used for checks.
        idx_transform: Maps group keep indices to this item's local keep
            indices. ``None`` == identity.
        reason: Provenance string for this dependency (rule name).
    """

    name: str
    module: nn.Module
    pruning_fn: Callable[[nn.Module, list[int]], dict[str, Any]]
    direction: str
    idxs: list[int] = field(default_factory=list)
    idx_transform: IdxTransform = None
    reason: str = ""

    def local_keep(self, group_keep: list[int]) -> list[int]:
        """Translate group-level keep indices into this item's local space."""
        if self.idx_transform is None:
            return list(group_keep)
        return self.idx_transform(group_keep)

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (
            f"GroupItem({self.name}, dir={self.direction}, "
            f"fn={getattr(self.pruning_fn, '__name__', self.pruning_fn)}, "
            f"reason={self.reason})"
        )


@dataclass
class PruningGroup:
    """An atomic set of coupled pruning operations.

    A group prunes one logical channel dimension. ``group_keep`` indices are
    expressed in the group's reference index space (the output channels of the
    group's root module). Each :class:`GroupItem` translates those indices into
    its own local space via :meth:`GroupItem.local_keep`.

    Attributes:
        group_id: Unique identifier.
        items: Ordered list of coupled operations.
        num_channels: Size of the group's reference channel dimension.
        protected: If True the group must not be pruned.
        protected_reason: Why the group is protected.
        meta: Free-form metadata (group_type, cat layout, groups count, ...).
    """

    group_id: str
    items: list[GroupItem] = field(default_factory=list)
    num_channels: int = 0
    protected: bool = False
    protected_reason: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def add_dep(
        self,
        name: str,
        module: nn.Module,
        pruning_fn: Callable[[nn.Module, list[int]], dict[str, Any]],
        direction: str,
        idxs: list[int] | None = None,
        idx_transform: IdxTransform = None,
        reason: str = "",
    ) -> GroupItem:
        """Append a coupled operation to this group.

        Deduplicates by (name, direction): a layer cannot be sliced twice on the
        same axis within one group.

        Returns:
            The created (or existing) :class:`GroupItem`.
        """
        for existing in self.items:
            if existing.name == name and existing.direction == direction:
                return existing
        item = GroupItem(
            name=name,
            module=module,
            pruning_fn=pruning_fn,
            direction=direction,
            idxs=list(idxs) if idxs is not None else [],
            idx_transform=idx_transform,
            reason=reason,
        )
        self.items.append(item)
        return item

    def protect(self, reason: str) -> None:
        """Mark the whole group as non-prunable."""
        self.protected = True
        if reason:
            self.protected_reason = reason

    @property
    def is_protected(self) -> bool:
        """Compatibility alias used by older search/importance code."""
        return self.protected

    @property
    def is_prunable(self) -> bool:
        """Compatibility alias for group-level prunability."""
        return not self.protected

    @property
    def source_modules(self) -> list[str]:
        """Compatibility alias returning member module names."""
        return [item.name for item in self.items]

    def prune(self, group_keep: list[int]) -> dict[str, Any]:
        """Atomically apply the pruning to every item in the group.

        Requirement #6: the surgery is all-or-nothing. Each item's pruning_fn
        is dry-run validated first (via its ``supports`` attribute, set by the
        dispatch table); if any item is unsupported the group is left untouched
        and reported as skipped. Only when every item is applicable do we mutate
        the modules.

        Args:
            group_keep: Channel indices to keep, in group reference space.

        Returns:
            Report dict with ``applied`` (bool), ``operations`` and ``skipped``.
        """
        if self.protected:
            return {
                "group_id": self.group_id,
                "applied": False,
                "skipped_reason": self.protected_reason or "protected",
                "operations": [],
            }

        # Phase 1: validate every item is supported before mutating anything.
        for item in self.items:
            supports = getattr(item.pruning_fn, "supports", None)
            if supports is not None and not supports(item.module):
                return {
                    "group_id": self.group_id,
                    "applied": False,
                    "skipped_reason": (
                        f"unsupported_op:{item.name}:"
                        f"{item.module.__class__.__name__}:{item.direction}"
                    ),
                    "operations": [],
                }

        # Phase 2: apply. All items validated -> safe to mutate.
        operations: list[dict[str, Any]] = []
        for item in self.items:
            local_keep = item.local_keep(group_keep)
            if not local_keep:
                # A coupled op that loses all channels is illegal; abort group.
                # (We have already mutated earlier items, so this is a hard
                # error rather than a soft skip - the checker must prevent it.)
                raise RuntimeError(
                    f"PruningGroup {self.group_id}: item {item.name} "
                    f"({item.direction}) would keep 0 channels"
                )
            op = item.pruning_fn(item.module, local_keep)
            op = dict(op)
            op.update({
                "layer": item.name,
                "direction": item.direction,
                "reason": item.reason,
            })
            operations.append(op)

        return {
            "group_id": self.group_id,
            "applied": True,
            "skipped_reason": "",
            "operations": operations,
        }

    def summary(self) -> dict[str, Any]:
        """Compact serializable description of the group (no live modules)."""
        return {
            "group_id": self.group_id,
            "num_channels": self.num_channels,
            "protected": self.protected,
            "protected_reason": self.protected_reason,
            "group_type": self.meta.get("group_type", ""),
            "items": [
                {
                    "name": it.name,
                    "type": it.module.__class__.__name__,
                    "direction": it.direction,
                    "pruning_fn": getattr(it.pruning_fn, "__name__", str(it.pruning_fn)),
                    "reason": it.reason,
                }
                for it in self.items
            ],
        }

    def __len__(self) -> int:
        return len(self.items)

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (
            f"PruningGroup({self.group_id}, channels={self.num_channels}, "
            f"items={len(self.items)}, protected={self.protected})"
        )
