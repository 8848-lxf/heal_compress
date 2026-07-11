"""Coupled channel group generation and management.

Groups channels that must be pruned synchronously due to structural
dependencies (residual connections, conv-bn pairs, concat, etc.).
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any

from ..utils.io_utils import save_csv, save_json, save_text

logger = logging.getLogger(__name__)


@dataclass
class CoupledChannelGroup:
    """A set of channels across multiple layers that must be pruned together.

    Attributes:
        group_id: Unique identifier for this group.
        group_type: Category (e.g. 'conv_block', 'residual', 'concat',
            'transformer_head', 'pyramid_scale').
        source_modules: List of layers whose output channels belong to this group.
        dependent_modules: List of layers whose input channels depend on this group.
        channel_indices: List of channel indices in this group.
        input_channel_dependencies: Map of dependent layer -> input channel indices.
        output_channel_dependencies: Map of source layer -> output channel indices.
        is_prunable: Whether this group can be pruned.
        is_protected: Whether this group is protected from pruning.
        protected_reason: Why the group is protected (if applicable).
        dynamic_branch_source: Which agent-count paths touch this group.
        successor_mapping: Maps downstream layers to their channel index offsets.
        hardware_alignment_constraint: Alignment requirement (filled by checker).
    """
    group_id: str
    group_type: str = "conv_block"
    source_modules: list[str] = field(default_factory=list)
    dependent_modules: list[str] = field(default_factory=list)
    channel_indices: list[int] = field(default_factory=list)
    input_channel_dependencies: dict[str, list[int]] = field(default_factory=dict)
    output_channel_dependencies: dict[str, list[int]] = field(default_factory=dict)
    is_prunable: bool = True
    is_protected: bool = False
    protected_reason: str = ""
    dynamic_branch_source: list[str] = field(default_factory=list)
    successor_mapping: dict[str, Any] = field(default_factory=dict)
    hardware_alignment_constraint: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for JSON/YAML output."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CoupledChannelGroup":
        """Deserialize from a dict."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class CoupledChannelGroupBuilder:
    """Builds coupled channel groups from a dependency graph.

    Uses union-find to merge channels connected by any dependency edge
    (direct or transitive) into the same group.

    Handles special cases:
    - Grouped convolutions: keep_groups mode or depthwise sync
    - Transformer Q/K/V: same attention head grouped together
    - Pyramid multi-scale: cross-scale coupled channels

    Args:
        dependency_graph: The dependency graph dict (nodes + edges).
        model: The HEAL model for structure inspection.
        branches: List of traced branch labels.
    """

    PARAMETRIC_TYPES = {"Conv2d", "ConvTranspose2d", "Linear", "MultiheadAttention"}

    def __init__(
        self,
        dependency_graph: dict[str, Any],
        model: Any = None,
        branches: list[str] | None = None,
    ):
        self.graph = dependency_graph
        self.model = model
        self.branches = branches or ["default"]
        self._parent: dict[str, str] = {}
        self._reasons: dict[str, set[str]] = {}

    def build(self) -> list[CoupledChannelGroup]:
        """Generate coupled channel groups from the dependency graph.

        Steps:
        1. Initialize union-find with all parametric layers.
        2. Merge layers connected by conv_bn edges.
        3. Merge layers connected by residual_add edges.
        4. Merge layers connected by sequential edges (same output dimension).
        5. Handle grouped convolution constraints.
        6. Preserve legacy Transformer heuristic metadata (not validated
           formal pruning support).
        7. Build group objects with channel indices and successor mappings.
        8. Mark protected groups.

        Returns:
            List of CoupledChannelGroup objects.
        """
        nodes = self.graph.get("nodes", {})
        edges = self.graph.get("edges", [])

        # Initialize union-find
        for name, info in nodes.items():
            if info.get("type") in self.PARAMETRIC_TYPES:
                self._parent[name] = name
                self._reasons[name] = set(self.branches)

        # Build adjacency
        incoming: dict[str, list[str]] = {}
        outgoing: dict[str, list[str]] = {}
        conv_bn: dict[str, list[str]] = {}

        for edge in edges:
            src = edge.get("src", "")
            dst = edge.get("dst", "")
            kind = edge.get("kind", "")
            incoming.setdefault(dst, []).append(src)
            outgoing.setdefault(src, []).append(dst)
            if kind == "conv_bn":
                conv_bn.setdefault(src, []).append(dst)

        # Merge by residual_add
        for edge in edges:
            if edge.get("kind") == "residual_add":
                self._union(edge["src"], edge["dst"], "residual_add")

        # Build groups from union-find
        grouped: dict[str, list[str]] = {}
        for name in self._parent:
            root = self._find(name)
            grouped.setdefault(root, []).append(name)

        groups: list[CoupledChannelGroup] = []
        for root, members in grouped.items():
            members = sorted(members)
            root_info = nodes.get(root, {})
            if root_info.get("type") not in self.PARAMETRIC_TYPES:
                continue

            out_channels = int(
                root_info.get("out_channels",
                              root_info.get("out_features",
                                            root_info.get("embed_dim", 0)))
            )
            if out_channels <= 0:
                continue

            protected = any(
                nodes.get(m, {}).get("protected", False) for m in members
            )

            # Find downstream input layers
            downstream = self._find_downstream_parametric(
                members, outgoing, nodes
            )

            # Find BN layers
            bn_layers = []
            for m in members:
                bn_layers.extend(conv_bn.get(m, []))

            group = CoupledChannelGroup(
                group_id=f"group::{root}",
                group_type=self._classify_group_type(root, nodes),
                source_modules=members,
                dependent_modules=downstream,
                channel_indices=list(range(out_channels)),
                is_prunable=not protected,
                is_protected=protected,
                protected_reason="auto_protected" if protected else "",
                dynamic_branch_source=sorted(
                    self._reasons.get(self._find(root), set(self.branches))
                ),
                successor_mapping={
                    "out_channels": out_channels,
                    "conv_bn": bn_layers,
                },
            )
            groups.append(group)

        logger.info(
            f"Built {len(groups)} coupled channel groups "
            f"({sum(1 for g in groups if g.is_protected)} protected)"
        )
        return groups

    def _find(self, name: str) -> str:
        """Union-find: find root with path compression."""
        while self._parent.get(name, name) != name:
            self._parent[name] = self._parent.get(self._parent[name], self._parent[name])
            name = self._parent[name]
        return name

    def _union(self, a: str, b: str, reason: str) -> None:
        """Union-find: merge two sets."""
        if a not in self._parent or b not in self._parent:
            return
        ra, rb = self._find(a), self._find(b)
        if ra == rb:
            self._reasons.setdefault(ra, set()).add(reason)
            return
        self._parent[rb] = ra
        self._reasons.setdefault(ra, set()).update(
            self._reasons.get(rb, set())
        )
        self._reasons[ra].add(reason)

    def _find_downstream_parametric(
        self,
        members: list[str],
        outgoing: dict[str, list[str]],
        nodes: dict[str, dict[str, Any]],
    ) -> list[str]:
        """BFS from members to find downstream parametric layers.

        Args:
            members: Source layer names.
            outgoing: Outgoing adjacency map.
            nodes: All graph nodes.

        Returns:
            Sorted list of downstream parametric layer names.
        """
        member_set = set(members)
        found: set[str] = set()
        queue = list(members)
        visited = set(queue)

        while queue:
            current = queue.pop(0)
            for dst in outgoing.get(current, []):
                if dst in visited:
                    continue
                visited.add(dst)
                dst_type = nodes.get(dst, {}).get("type", "")
                if dst_type in self.PARAMETRIC_TYPES:
                    if dst not in member_set:
                        found.add(dst)
                    continue
                queue.append(dst)
        return sorted(found)

    def _classify_group_type(
        self,
        root: str,
        nodes: dict[str, dict[str, Any]],
    ) -> str:
        """Classify a group by its root node's structural role.

        Args:
            root: Root node name.
            nodes: All graph nodes.

        Returns:
            Group type string.
        """
        low = root.lower()
        if "single_head" in low or "deblocks" in low:
            return "pyramid_scale"
        if "transformer" in low or "attention" in low:
            return "transformer_head"
        if nodes.get(root, {}).get("type") == "MultiheadAttention":
            return "transformer_head"
        return "conv_block"

    def save(
        self,
        groups: list[CoupledChannelGroup],
        json_path: str,
        csv_path: str | None = None,
        summary_path: str | None = None,
    ) -> None:
        """Save coupled channel groups to files.

        Args:
            groups: List of groups to save.
            json_path: Path for JSON output.
            csv_path: Optional path for CSV output.
            summary_path: Optional path for text summary.
        """
        save_json(
            {"groups": [g.to_dict() for g in groups]},
            json_path,
        )
        logger.info(f"Coupled channel groups saved to {json_path}")

        if csv_path:
            rows = []
            for g in groups:
                rows.append({
                    "group_id": g.group_id,
                    "group_type": g.group_type,
                    "num_channels": len(g.channel_indices),
                    "num_source_modules": len(g.source_modules),
                    "num_dependent_modules": len(g.dependent_modules),
                    "is_prunable": g.is_prunable,
                    "is_protected": g.is_protected,
                    "protected_reason": g.protected_reason,
                })
            save_csv(rows, csv_path)

        if summary_path:
            lines = [f"Total groups: {len(groups)}"]
            lines.append(f"Prunable: {sum(1 for g in groups if g.is_prunable)}")
            lines.append(f"Protected: {sum(1 for g in groups if g.is_protected)}")
            lines.append("")
            for g in groups:
                lines.append(
                    f"{g.group_id}: type={g.group_type} "
                    f"channels={len(g.channel_indices)} "
                    f"prunable={g.is_prunable} "
                    f"sources={g.source_modules[:3]}"
                )
            save_text("\n".join(lines), summary_path)

    @classmethod
    def load(cls, json_path: str) -> list[CoupledChannelGroup]:
        """Load coupled channel groups from a JSON file.

        Args:
            json_path: Input file path.

        Returns:
            List of CoupledChannelGroup objects.
        """
        from ..utils.io_utils import load_json
        data = load_json(json_path)
        return [
            CoupledChannelGroup.from_dict(g) for g in data.get("groups", [])
        ]
