"""Channel-dependency propagation rules (Torch-Pruning style).

Turns an :class:`heal_compress.tracer.op_graph.OpGraph` into a list of
:class:`heal_compress.tracer.pruning_group.PruningGroup` objects by applying the
propagation rules:

    Conv.out      -> BN.out                       (conv-bn coupling)
    BN.out        -> downstream Conv.in / Linear.in
    Add.out       <-> every Add input branch       (shared common dim)
    Cat input_i.out -> Cat.out with offset         (concatenated dim)
    Cat.out       -> downstream Conv.in
    ConvTranspose2d in/out                          (dedicated axis fns)
    Grouped / Depthwise Conv                        (delegated handler)

Reference index space of a group:
    * plain / Add-merged group  -> the common output dim (identity transforms)
    * Cat-merged group          -> the concatenated dim; branch roots use
                                    offset transforms, cat-downstream inputs use
                                    identity over the concatenated dim.

Anything whose channel semantics cannot be resolved unambiguously (mixed
add+cat reference spaces, unsupported ops, protected roots) is marked
``protected`` rather than pruned half-way (requirement #6, atomic safety).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set, Tuple

from ..tracer.op_graph import (
    OP_ADD, OP_BN, OP_CAT, OP_CONV, OP_CONVT, OP_DET_HEAD_INPUT,
    OP_LINEAR, OP_NORM, OP_SPLIT, PARAMETRIC_OPS, PASSTHROUGH_OPS,
    OpGraph, OpNode,
)
from ..tracer.pruning_group import PruningGroup, identity_transform, offset_transform
from .grouped_conv import classify_grouped_conv, grouped_conv_pruning_fn
from .pruning_fns import get_pruning_fn

logger = logging.getLogger(__name__)

# Ops we are willing to walk *through* when chasing a channel dependency. The
# channel dimension is assumed preserved across these (true for BN/Norm and the
# shape-only view/permute/interpolate ops in channel-first layouts).
_TRANSPARENT = PASSTHROUGH_OPS | {OP_ADD}


class GroupBuilder:
    """Builds :class:`PruningGroup` objects from an :class:`OpGraph`.

    Args:
        graph: The operation-level dependency graph.
        align: Hardware channel alignment (used by grouped-conv protection).
        grouped_conv_mode: ``"keep_groups"`` or ``"remove_groups"``.
    """

    def __init__(
        self,
        graph: OpGraph,
        align: int = 16,
        grouped_conv_mode: str = "keep_groups",
    ):
        self.graph = graph
        self.align = int(align)
        self.grouped_conv_mode = grouped_conv_mode
        # union-find over root node names (output-channel dims)
        self._parent: Dict[str, str] = {}
        self._reasons: Dict[str, Set[str]] = {}
        # cat layout per root: root -> (cat_node, offset, branch_size)
        self._cat_layout: Dict[str, Tuple[str, int, int]] = {}
        # roots merged via add (need identity / common-dim reference)
        self._add_roots: Set[str] = set()
        # depthwise conv nodes: transparent channel passthroughs, not roots
        self._depthwise: Set[str] = set()

    def _is_depthwise_node(self, name: str) -> bool:
        node = self.graph.nodes.get(name)
        if node is None or node.op_type != OP_CONV:
            return False
        module = self.graph.module_of(name)
        from torch.nn import Conv2d
        return (
            isinstance(module, Conv2d)
            and module.groups > 1
            and module.groups == module.in_channels == module.out_channels
        )

    # -- public ------------------------------------------------------------- #
    def build(self) -> List[PruningGroup]:
        self._depthwise = {n.name for n in self.graph.parametric_nodes()
                           if self._is_depthwise_node(n.name)}
        # Roots: parametric nodes defining an output dim. Depthwise convs are
        # excluded (transparent passthrough). Grouped convs (non-depthwise) with
        # in==out define a new dimension but will be merged with their producer.
        roots = [
            n.name for n in self.graph.parametric_nodes()
            if n.out_dim() and n.name not in self._depthwise
        ]
        for r in roots:
            self._parent[r] = r
            self._reasons[r] = set()

        self._merge_add_branches()
        self._merge_cat_branches()
        self._merge_grouped_conv_producers()

        # bucket roots by union-find root
        buckets: Dict[str, List[str]] = {}
        for r in roots:
            buckets.setdefault(self._find(r), []).append(r)

        groups: List[PruningGroup] = []
        for gid_root, members in buckets.items():
            group = self._build_group(gid_root, sorted(members))
            if group is not None:
                groups.append(group)
        logger.info(
            "Propagation produced %d groups (%d protected)",
            len(groups), sum(1 for g in groups if g.protected),
        )
        return groups

    # -- union-find --------------------------------------------------------- #
    def _find(self, name: str) -> str:
        while self._parent[name] != name:
            self._parent[name] = self._parent[self._parent[name]]
            name = self._parent[name]
        return name

    def _union(self, a: str, b: str, reason: str) -> None:
        if a not in self._parent or b not in self._parent:
            return
        ra, rb = self._find(a), self._find(b)
        if ra != rb:
            self._parent[rb] = ra
            self._reasons.setdefault(ra, set()).update(self._reasons.get(rb, set()))
        self._reasons.setdefault(ra, set()).add(reason)

    def _merge_grouped_conv_producers(self) -> None:
        """Merge producer.out ↔ grouped_conv.in/out when the conv is in==out.

        A grouped conv (non-depthwise) with in==out is prunable via keep_groups,
        but its input must come from a single producer whose output is coupled to
        the grouped conv's in/out. We merge them so both are sliced together.
        """
        for node in self.graph.nodes.values():
            if node.op_type != OP_CONV or (node.groups or 1) <= 1:
                continue
            if node.name in self._depthwise:
                continue
            module = self.graph.module_of(node.name)
            if module is None:
                continue
            from torch.nn import Conv2d
            if not isinstance(module, Conv2d):
                continue
            if module.in_channels != module.out_channels:
                continue
            # Find immediate producer(s) via transparent backward walk.
            producers = []
            for src, _idx in self.graph.incoming(node.name):
                producers.extend(self._upstream_roots(src))
            if len(producers) == 1:
                # Merge the producer root with this grouped conv root (both are
                # in roots list since grouped conv in==out defines a new dim).
                # We must add the grouped conv to roots if not present (edge
                # case: if it's the first op, it's a root).
                producer = producers[0]
                if producer in self._parent and node.name not in self._parent:
                    self._parent[node.name] = node.name
                    self._reasons[node.name] = set()
                if producer in self._parent and node.name in self._parent:
                    self._union(producer, node.name, f"grouped_conv_producer:{node.name}")

    # -- backward traversal: find producing roots --------------------------- #
    def _upstream_roots(self, node: str, visited: Optional[Set[str]] = None) -> List[str]:
        """Walk backward through transparent ops to the parametric root(s)."""
        visited = visited or set()
        if node in visited:
            return []
        visited.add(node)
        info = self.graph.nodes.get(node)
        if info is None:
            return []
        if info.op_type in PARAMETRIC_OPS and node not in self._depthwise:
            return [node]
        # depthwise convs and transparent ops forward the channel dim upstream
        out: List[str] = []
        for src, _idx in self.graph.incoming(node):
            out.extend(self._upstream_roots(src, visited))
        # dedupe preserving order
        seen: Set[str] = set()
        ordered = []
        for r in out:
            if r not in seen:
                seen.add(r)
                ordered.append(r)
        return ordered

    def _branch_roots_ordered(self, structural_node: str) -> List[List[str]]:
        """Return, per input branch (ordered by input_index), its upstream roots."""
        branches: List[List[str]] = []
        for src, _idx in self.graph.incoming(structural_node):
            branches.append(self._upstream_roots(src))
        return branches

    # -- merge rules -------------------------------------------------------- #
    def _merge_add_branches(self) -> None:
        for node in self.graph.nodes.values():
            if node.op_type != OP_ADD:
                continue
            branches = self._branch_roots_ordered(node.name)
            flat = [r for br in branches for r in br]
            if len(flat) < 2:
                continue
            base = flat[0]
            for other in flat[1:]:
                self._union(base, other, f"add:{node.name}")
            for r in flat:
                self._add_roots.add(r)

    def _merge_cat_branches(self) -> None:
        for node in self.graph.nodes.values():
            if node.op_type != OP_CAT:
                continue
            cat_dim = node.cat_dim if node.cat_dim is not None else 1
            if cat_dim != 1:
                # Only channel-dim concat participates in channel pruning.
                continue
            branches = self._branch_roots_ordered(node.name)
            in_shapes = node.input_shapes
            if not branches or len(branches) < 2:
                continue
            # Compute each branch's channel size from traced shapes when possible.
            offset = 0
            base_root: Optional[str] = None
            ok = True
            layout: List[Tuple[str, int, int]] = []
            for i, br in enumerate(branches):
                if len(br) != 1:
                    # A branch resolving to 0 or >1 roots is ambiguous for cat
                    # offset bookkeeping -> protect via marker.
                    ok = False
                root = br[0] if br else None
                size = None
                if i < len(in_shapes) and len(in_shapes[i]) > cat_dim:
                    size = int(in_shapes[i][cat_dim])
                elif root is not None:
                    size = self.graph.nodes[root].out_dim()
                if root is None or size is None:
                    ok = False
                    break
                layout.append((root, offset, size))
                offset += size
            if not ok:
                # Mark all resolvable roots as cat-ambiguous by unioning + flag.
                flat = [r for br in branches for r in br]
                for other in flat[1:] if flat else []:
                    self._union(flat[0], other, f"cat_ambiguous:{node.name}")
                for r in flat:
                    self._cat_layout.setdefault(r, (node.name, -1, -1))
                continue
            base_root = layout[0][0]
            for root, off, size in layout:
                self._union(base_root, root, f"cat:{node.name}")
                self._cat_layout[root] = (node.name, off, size)

    # -- group assembly ----------------------------------------------------- #
    def _build_group(self, gid_root: str, members: List[str]) -> Optional[PruningGroup]:
        reasons = self._reasons.get(gid_root, set())
        is_cat = any(r in self._cat_layout for r in members)
        is_add = any(r in self._add_roots for r in members)
        cat_node = None
        for r in members:
            if r in self._cat_layout:
                cat_node = self._cat_layout[r][0]
                break

        # Reference channel space size.
        if is_cat:
            # ambiguous cat (offset == -1) -> protect
            ambiguous = any(self._cat_layout.get(r, (None, 0, 0))[1] < 0 for r in members)
            ref_channels = 0
            for r in members:
                _node, off, size = self._cat_layout.get(r, (None, 0, 0))
                if off >= 0:
                    ref_channels = max(ref_channels, off + size)
        else:
            ambiguous = False
            ref_channels = self.graph.nodes[gid_root].out_dim() or 0

        group = PruningGroup(
            group_id=f"group::{gid_root}",
            num_channels=ref_channels,
            meta={
                "group_type": "cat" if is_cat else ("add" if is_add else "plain"),
                "reasons": sorted(reasons),
                "cat_node": cat_node,
                "roots": members,
            },
        )

        if ambiguous:
            group.protect(f"cat_ambiguous:{cat_node}")
            return group

        # Protect if mixed add+cat reference spaces (we cannot express one space).
        if is_cat and is_add:
            group.protect("mixed_add_cat_reference")
            return group

        # 1. Root output members (+ grouped/depthwise handling).
        for r in members:
            node = self.graph.nodes[r]
            module = self.graph.module_of(r)
            if module is None:
                group.protect(f"missing_module:{r}")
                return group
            if node.protected:
                group.protect(node.protected_reason or f"protected_root:{r}")
                return group

            transform = self._root_transform(r, is_cat)

            if node.op_type == OP_CONV and (node.groups or 1) > 1:
                kind = classify_grouped_conv(module, self.align)
                if kind == "protected":
                    group.protect(f"grouped_conv_unaligned:{r}")
                    return group
                fn = grouped_conv_pruning_fn(self.grouped_conv_mode)
                group.add_dep(r, module, fn, "out",
                              idxs=list(range(node.out_dim() or 0)),
                              idx_transform=transform, reason=f"grouped_conv:{kind}")
                continue

            fn = get_pruning_fn(module, "out")
            if fn is None:
                group.protect(f"no_pruning_fn:{r}")
                return group
            group.add_dep(r, module, fn, "out",
                          idxs=list(range(node.out_dim() or 0)),
                          idx_transform=transform, reason="root_out")

        # 2. Norm/BN members + downstream input members per root.
        for r in members:
            self._add_norm_and_downstream(group, r, is_cat, member_set=set(members))

        # 3. If this is a cat group, add the cat's downstream consumers' inputs
        #    with identity transform over the concatenated reference space.
        if is_cat and cat_node is not None:
            self._add_cat_downstream(group, cat_node)

        return group

    def _root_transform(self, root: str, is_cat: bool):
        if not is_cat:
            return None  # identity
        _node, off, size = self._cat_layout.get(root, (None, 0, 0))
        if off <= 0 and size and off == 0:
            # branch at offset 0 still needs slicing to its size
            return offset_transform(0, size)
        return offset_transform(off, size)

    def _add_norm_and_downstream(self, group: PruningGroup, root: str, is_cat: bool, member_set: Set[str]) -> None:
        """Add BN/Norm consumers (out axis) and parametric .in consumers."""
        transform = self._root_transform(root, is_cat)
        visited: Set[str] = {root}
        queue: List[str] = [d for d, _ in self.graph.outgoing(root)]
        while queue:
            cur = queue.pop(0)
            if cur in visited:
                continue
            visited.add(cur)
            node = self.graph.nodes.get(cur)
            if node is None:
                continue
            module = self.graph.module_of(cur)
            if node.op_type in (OP_BN, OP_NORM) and module is not None:
                fn = get_pruning_fn(module, "out")
                if fn is None:
                    group.protect(f"no_norm_fn:{cur}")
                    return
                group.add_dep(cur, module, fn, "out",
                              idx_transform=transform, reason="conv_bn")
                queue.extend(d for d, _ in self.graph.outgoing(cur))
                continue
            if cur in self._depthwise and module is not None:
                # Depthwise conv: channel-preserving passthrough. Slice its
                # out axis (prune_conv_out syncs in/out/groups) and keep walking.
                fn = get_pruning_fn(module, "out")
                if fn is None:
                    group.protect(f"no_depthwise_fn:{cur}")
                    return
                group.add_dep(cur, module, fn, "out",
                              idx_transform=transform, reason="depthwise_passthrough")
                queue.extend(d for d, _ in self.graph.outgoing(cur))
                continue
            if node.op_type in (OP_CONV, OP_CONVT, OP_LINEAR) and module is not None:
                # downstream consumer input side
                if node.protected and node.is_det_head:
                    # det-head input IS prunable (only its output is fixed)
                    pass
                # If this conv is a grouped conv that's already a root member of
                # this group (merged producer), skip it (already handled).
                if cur in member_set:
                    continue
                if node.op_type == OP_CONV and (node.groups or 1) > 1:
                    # grouped (non-depthwise) consumer input cannot be sliced on
                    # in-axis alone; protect the whole group to stay atomic.
                    group.protect(f"grouped_consumer_in:{cur}")
                    return
                fn = get_pruning_fn(module, "in")
                if fn is None:
                    group.protect(f"no_in_fn:{cur}")
                    return
                group.add_dep(cur, module, fn, "in",
                              idx_transform=transform, reason="downstream_in")
                # stop: consumer defines a new output dim, do not pass through
                continue
            if node.op_type in _TRANSPARENT:
                queue.extend(d for d, _ in self.graph.outgoing(cur))
            elif node.op_type in (OP_CAT, OP_SPLIT):
                # handled by cat-downstream logic / split unsupported here
                if node.op_type == OP_SPLIT:
                    group.protect(f"split_consumer:{cur}")
                    return
                # cat: do not traverse further from a branch root; the cat's own
                # downstream consumers are added separately with identity space.
                continue

    def _add_cat_downstream(self, group: PruningGroup, cat_node: str) -> None:
        """Add parametric consumers of a cat output (input side, identity space)."""
        visited: Set[str] = {cat_node}
        queue: List[str] = [d for d, _ in self.graph.outgoing(cat_node)]
        while queue:
            cur = queue.pop(0)
            if cur in visited:
                continue
            visited.add(cur)
            node = self.graph.nodes.get(cur)
            if node is None:
                continue
            module = self.graph.module_of(cur)
            if node.op_type in (OP_CONV, OP_CONVT, OP_LINEAR) and module is not None:
                if node.op_type == OP_CONV and (node.groups or 1) > 1:
                    group.protect(f"grouped_cat_consumer:{cur}")
                    return
                fn = get_pruning_fn(module, "in")
                if fn is None:
                    group.protect(f"no_cat_in_fn:{cur}")
                    return
                group.add_dep(cur, module, fn, "in",
                              idx_transform=identity_transform, reason="cat_downstream_in")
                continue
            if node.op_type in (OP_BN, OP_NORM):
                # BN directly on cat output: prune over full cat space
                if module is not None:
                    fn = get_pruning_fn(module, "out")
                    if fn is not None:
                        group.add_dep(cur, module, fn, "out",
                                      idx_transform=identity_transform, reason="cat_bn")
                queue.extend(d for d, _ in self.graph.outgoing(cur))
                continue
            if node.op_type in _TRANSPARENT:
                queue.extend(d for d, _ in self.graph.outgoing(cur))
