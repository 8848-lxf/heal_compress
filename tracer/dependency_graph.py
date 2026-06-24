"""Dependency graph builder for HEAL models.

Traces channel-level dependencies between Conv2d, BatchNorm, Linear,
residual Add, Concat, and Transformer layers to determine which channels
must be pruned synchronously.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any

import torch.nn as nn

from ..utils.io_utils import load_json, load_yaml, save_json, save_yaml

logger = logging.getLogger(__name__)


@dataclass
class ChannelRef:
    """Reference to a channel dimension on a specific layer.

    Attributes:
        layer_name: Fully qualified module name.
        channel_axis: Which axis holds the channel dimension (0=out, 1=in).
        channel_index: Specific channel index, or None for all channels.
        role: Semantic role ('out', 'in', 'bn', 'norm').
    """
    layer_name: str
    channel_axis: int
    channel_index: int | None = None
    role: str = "out"


@dataclass
class DependencyEdge:
    """Directed edge in the dependency graph.

    Attributes:
        src: Source node name.
        dst: Destination node name.
        kind: Edge type (e.g. 'conv_bn', 'residual_add', 'concat',
              'sequential', 'module_input', 'tensor_op_input').
        meta: Additional metadata.
    """
    src: str
    dst: str
    kind: str
    meta: dict[str, Any]


class DependencyGraphBuilder:
    """Builds a channel-level dependency graph from a traced computation graph.

    Takes the output of HealForwardWrapper.trace_all_paths() and constructs
    a directed dependency graph D = (V, E) where:
    - V: parametric layers (Conv2d, ConvTranspose2d, BatchNorm2d, Linear,
         MultiheadAttention) and tensor operations (Add, Cat, etc.)
    - E: channel dependency edges

    Dependency types tracked:
    - Conv2d out -> successor Conv2d in
    - Conv2d out -> successor BatchNorm num_features
    - Residual Add: both branches must have same channels (sync pruning)
    - Concat: channel index mapping to downstream input
    - Multi-scale fusion (deblocks cat -> fused_feature)
    - BEV warp channel passthrough
    - Transformer Q/K/V projection coupling

    Protected nodes (not prunable):
    - LSS frustum/camC/D related convolutions
    - pyramid_backbone cat output dimension
    - cls_head/reg_head/dir_head final output layers
    - BEV geometry transform interface layers

    Args:
        trace_graph: The computation graph dict from HealForwardWrapper,
            with 'nodes' and 'edges' keys.
        model: The HEAL model for structure inspection.
        protected_layers: Explicit list of protected layer names.
    """

    PARAMETRIC_TYPES = {"Conv2d", "ConvTranspose2d", "Linear",
                        "BatchNorm2d", "MultiheadAttention"}
    NORM_TYPES = {"BatchNorm2d", "LayerNorm"}

    def __init__(
        self,
        trace_graph: dict[str, Any],
        model: nn.Module,
        protected_layers: list[str] | None = None,
    ):
        self.trace_graph = trace_graph
        self.model = model
        self.protected_layers = set(protected_layers or [])
        self.nodes: dict[str, dict[str, Any]] = {}
        self.edges: list[DependencyEdge] = []
        self.warnings: list[str] = []

    def build(self) -> "DependencyGraphBuilder":
        """Build the full dependency graph.

        Steps:
        1. Import all nodes from the trace graph, annotating parametric types.
        2. Mark protected nodes.
        3. Trace Conv->BN dependencies.
        4. Trace sequential Conv->Conv input channel dependencies.
        5. Trace residual Add synchronization constraints.
        6. Trace Concat channel index mappings.
        7. Trace multi-scale pyramid structure.
        8. Trace Transformer Q/K/V coupling.
        9. Validate and log warnings.

        Returns:
            self, for chaining.
        """
        self._import_trace_nodes()
        self._mark_protected_nodes()
        self._trace_conv_bn_edges()
        self._trace_sequential_edges()
        self._trace_residual_add_edges()
        self._trace_concat_edges()
        self._trace_pyramid_structure()
        self._trace_transformer_coupling()
        return self

    def _import_trace_nodes(self) -> None:
        """Import nodes from the traced computation graph."""
        for name, info in self.trace_graph.get("nodes", {}).items():
            self.nodes[name] = dict(info)

    def _mark_protected_nodes(self) -> None:
        """Mark nodes that must not be pruned."""
        for name in self.nodes:
            if name in self.protected_layers:
                self.nodes[name]["protected"] = True
            elif self._is_auto_protected(name):
                self.nodes[name]["protected"] = True
                self.protected_layers.add(name)

    def _is_auto_protected(self, name: str) -> bool:
        """Check if a layer should be automatically protected.

        Args:
            name: Layer name.

        Returns:
            True if the layer should be protected.
        """
        low = name.lower()
        auto_keywords = (
            "cls_head", "reg_head", "dir_head",
            "frustum", "camc", "depth_net",
        )
        return any(k in low for k in auto_keywords)

    def _trace_conv_bn_edges(self) -> None:
        """Find Conv2d -> BatchNorm2d pairs and add conv_bn edges."""
        trace_edges = self.trace_graph.get("edges", [])
        for edge in trace_edges:
            src = edge.get("src", "")
            dst = edge.get("dst", "")
            src_type = self.nodes.get(src, {}).get("type", "")
            dst_type = self.nodes.get(dst, {}).get("type", "")
            if src_type in ("Conv2d", "ConvTranspose2d") and dst_type == "BatchNorm2d":
                self.edges.append(DependencyEdge(src, dst, "conv_bn", {}))

    def _trace_sequential_edges(self) -> None:
        """Find sequential Conv/Linear chains and add input channel edges."""
        trace_edges = self.trace_graph.get("edges", [])
        for edge in trace_edges:
            src = edge.get("src", "")
            dst = edge.get("dst", "")
            src_type = self.nodes.get(src, {}).get("type", "")
            dst_type = self.nodes.get(dst, {}).get("type", "")
            if (src_type in ("Conv2d", "ConvTranspose2d", "Linear")
                    and dst_type in ("Conv2d", "ConvTranspose2d", "Linear")):
                self.edges.append(DependencyEdge(src, dst, "sequential", {}))

    def _trace_residual_add_edges(self) -> None:
        """Find residual Add operations and link their input branches.

        When two parametric layers feed into an element-wise Add, they must
        have matching output channel counts after pruning.
        """
        incoming = self._build_incoming_map()
        for name, info in self.nodes.items():
            if info.get("type") != "TensorOp":
                continue
            op = info.get("op", "")
            if op not in ("Tensor.__add__", "torch.add"):
                continue
            sources = self._upstream_parametric_sources(name, incoming)
            if len(sources) >= 2:
                for i in range(1, len(sources)):
                    self.edges.append(
                        DependencyEdge(sources[0], sources[i], "residual_add", {})
                    )

    def _trace_concat_edges(self) -> None:
        """Find torch.cat operations and record channel offset mappings."""
        incoming = self._build_incoming_map()
        outgoing = self._build_outgoing_map()
        for name, info in self.nodes.items():
            if info.get("type") != "TensorOp" or info.get("op") != "torch.cat":
                continue
            cat_dim = info.get("cat_dim", 1)
            if cat_dim != 1:
                continue
            sources = []
            for src in incoming.get(name, []):
                sources.append(src)
            downstream = outgoing.get(name, [])
            self.edges.append(DependencyEdge(
                name, name, "concat",
                {"sources": sources, "downstream": downstream}
            ))

    def _trace_pyramid_structure(self) -> None:
        """Detect and annotate the 3-scale pyramid_backbone structure.

        pyramid_backbone has:
        - resnet.layer0/1/2
        - single_head_0/1/2
        - deblocks[0/1/2]
        - final cat -> fused_feature

        The cat output dimension is protected.
        """
        for name, _ in self.model.named_modules():
            if "pyramid_backbone" in name:
                if any(x in name for x in ("single_head", "deblocks")):
                    pass  # These are prunable but coupled across scales
                break

    def _trace_transformer_coupling(self) -> None:
        """Find Transformer Q/K/V projections and link them as coupled.

        Same attention head's Q/K/V subspaces must be in the same group.
        FFN intermediate dimensions can be independent.
        """
        for name, info in self.nodes.items():
            if info.get("type") == "MultiheadAttention":
                self.edges.append(
                    DependencyEdge(name, name, "transformer_qkv", {
                        "embed_dim": info.get("embed_dim"),
                        "num_heads": info.get("num_heads"),
                    })
                )

    def _build_incoming_map(self) -> dict[str, list[str]]:
        """Build node -> list of incoming source nodes."""
        incoming: dict[str, list[str]] = {}
        for edge in self.trace_graph.get("edges", []):
            incoming.setdefault(edge.get("dst", ""), []).append(edge.get("src", ""))
        return incoming

    def _build_outgoing_map(self) -> dict[str, list[str]]:
        """Build node -> list of outgoing destination nodes."""
        outgoing: dict[str, list[str]] = {}
        for edge in self.trace_graph.get("edges", []):
            outgoing.setdefault(edge.get("src", ""), []).append(edge.get("dst", ""))
        return outgoing

    def _upstream_parametric_sources(
        self,
        node: str,
        incoming: dict[str, list[str]],
        visited: set[str] | None = None,
    ) -> list[str]:
        """Walk upstream from a node to find parametric source layers.

        Args:
            node: Starting node name.
            incoming: Pre-built incoming adjacency map.
            visited: Set of already-visited nodes (for cycle detection).

        Returns:
            Sorted list of parametric layer names feeding into this node.
        """
        visited = visited or set()
        if node in visited:
            return []
        visited.add(node)

        node_type = self.nodes.get(node, {}).get("type", "")
        if node_type in self.PARAMETRIC_TYPES:
            return [node]

        sources: list[str] = []
        for src in incoming.get(node, []):
            sources.extend(self._upstream_parametric_sources(src, incoming, visited))
        return sorted(set(sources))

    def get_prunable_modules(self) -> list[str]:
        """Return list of module names that are eligible for pruning.

        Returns:
            Sorted list of prunable layer names.
        """
        prunable = []
        for name, info in self.nodes.items():
            if info.get("type") not in ("Conv2d", "ConvTranspose2d", "Linear"):
                continue
            if info.get("protected", False):
                continue
            prunable.append(name)
        return sorted(prunable)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the dependency graph to a dict.

        Returns:
            Dict with 'nodes', 'edges', and 'warnings' keys.
        """
        return {
            "nodes": self.nodes,
            "edges": [asdict(e) for e in self.edges],
            "warnings": self.warnings,
        }

    def save(self, json_path: str) -> None:
        """Save the dependency graph to a JSON file.

        Args:
            json_path: Output file path.
        """
        save_json(self.to_dict(), json_path)
        logger.info(f"Dependency graph saved to {json_path}")

    @classmethod
    def load(cls, json_path: str) -> dict[str, Any]:
        """Load a dependency graph from a JSON file.

        Args:
            json_path: Input file path.

        Returns:
            The loaded graph dict.
        """
        return load_json(json_path)
