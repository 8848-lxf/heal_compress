"""Operation-level dependency graph (Torch-Pruning style).

Where :mod:`heal_compress.tracer.dependency_graph` records *layer-level* edges,
this module records *operation-level* nodes. Every node carries an ``op_type``
drawn from a fixed taxonomy so that the propagation rules can reason about how a
channel cut flows through the network:

    Conv / BN / Linear / ConvTranspose2d / Norm / Attention
    Add / Cat / Split / View / Permute / Interpolate / BEVWarp / DetHeadInput

The graph is built from the runtime trace produced by
:class:`heal_compress.tracer.forward_wrapper.HealForwardWrapper` (or any tracer
emitting the same ``{"nodes": ..., "edges": ...}`` schema), so dynamic Python
forwards (multi-agent fusion, BEV warps) are captured as they actually execute.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch.nn as nn

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Op taxonomy
# --------------------------------------------------------------------------- #
OP_CONV = "Conv"
OP_BN = "BN"
OP_LINEAR = "Linear"
OP_CONVT = "ConvTranspose2d"
OP_NORM = "Norm"            # LayerNorm / GroupNorm
OP_ATTENTION = "Attention"  # MultiheadAttention
OP_ADD = "Add"
OP_CAT = "Cat"
OP_SPLIT = "Split"
OP_VIEW = "View"
OP_PERMUTE = "Permute"
OP_INTERPOLATE = "Interpolate"
OP_BEV_WARP = "BEVWarp"
OP_DET_HEAD_INPUT = "DetHeadInput"
OP_OTHER = "Other"

# Ops that *define* a channel dimension on their output (group roots).
PARAMETRIC_OPS = {OP_CONV, OP_LINEAR, OP_CONVT, OP_ATTENTION}
# Ops that preserve the channel dimension and forward a cut transparently.
PASSTHROUGH_OPS = {OP_BN, OP_NORM, OP_VIEW, OP_PERMUTE, OP_INTERPOLATE, OP_OTHER}
# Ops that have non-trivial channel-index semantics handled by propagation.
STRUCTURAL_OPS = {OP_ADD, OP_CAT, OP_SPLIT}

# Default name substrings that flag a detection-head module (output protected,
# input prunable). Override via build_op_graph(..., det_head_keywords=...).
DEFAULT_DET_HEAD_KEYWORDS = ("cls_head", "reg_head", "dir_head", "heatmap_head")
# Default substrings that flag fully-protected sensor/geometry interface convs.
DEFAULT_PROTECTED_KEYWORDS = ("frustum", "camc", "depth_net", "lss")
# Substrings (in op name or module scope) that flag a BEV warp / spatial
# transform whose channel dimension must pass through unchanged.
DEFAULT_BEV_WARP_KEYWORDS = ("warp", "grid_sample", "spatial_transform", "se3")


@dataclass
class OpNode:
    """A single operation-level node.

    Attributes:
        name: Unique node name (module name or ``op::NNNNN::op_name``).
        op_type: One of the taxonomy constants above.
        raw_type: Original module class name or traced op string.
        in_channels / out_channels / num_features / groups: Parametric attrs
            (``None`` when not applicable).
        cat_dim / num_inputs: Concat layout (for ``Cat`` / ``Split``).
        input_shapes / output_shapes: Traced tensor shapes.
        module_scope: Enclosing module names at trace time.
        protected: Whether this node's *output* channels must not change.
        is_det_head: Whether this node is a detection-head module.
        protected_reason: Provenance for protection.
    """

    name: str
    op_type: str
    raw_type: str = ""
    in_channels: Optional[int] = None
    out_channels: Optional[int] = None
    num_features: Optional[int] = None
    groups: Optional[int] = None
    cat_dim: Optional[int] = None
    num_inputs: Optional[int] = None
    input_shapes: List[List[int]] = field(default_factory=list)
    output_shapes: List[List[int]] = field(default_factory=list)
    module_scope: List[str] = field(default_factory=list)
    protected: bool = False
    is_det_head: bool = False
    protected_reason: str = ""

    @property
    def is_parametric(self) -> bool:
        return self.op_type in PARAMETRIC_OPS

    def out_dim(self) -> Optional[int]:
        """Best-effort output channel count."""
        for v in (self.out_channels, self.num_features):
            if v is not None:
                return int(v)
        return None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


class OpGraph:
    """Operation-level dependency graph with adjacency helpers.

    Attributes:
        nodes: Mapping name -> :class:`OpNode`.
        edges: List of ``(src, dst, input_index, kind)`` tuples.
        modules: Live module map (name -> nn.Module) for surgery.
    """

    def __init__(self, model: Optional[nn.Module] = None):
        self.nodes: Dict[str, OpNode] = {}
        self.edges: List[Tuple[str, str, int, str]] = []
        self.modules: Dict[str, nn.Module] = dict(model.named_modules()) if model else {}
        self._incoming: Dict[str, List[Tuple[str, int]]] = {}
        self._outgoing: Dict[str, List[Tuple[str, int]]] = {}
        self.warnings: List[str] = []

    # -- construction ------------------------------------------------------- #
    def add_node(self, node: OpNode) -> None:
        self.nodes[node.name] = node

    def add_edge(self, src: str, dst: str, input_index: int = 0, kind: str = "flow") -> None:
        self.edges.append((src, dst, input_index, kind))

    def finalize(self) -> "OpGraph":
        """Build adjacency maps from edges (call after all edges added)."""
        self._incoming = {}
        self._outgoing = {}
        for src, dst, idx, _kind in self.edges:
            self._outgoing.setdefault(src, []).append((dst, idx))
            self._incoming.setdefault(dst, []).append((src, idx))
        return self

    # -- adjacency ---------------------------------------------------------- #
    def incoming(self, name: str) -> List[Tuple[str, int]]:
        """List of (producer, input_index) feeding ``name``, ordered by index."""
        return sorted(self._incoming.get(name, []), key=lambda t: t[1])

    def outgoing(self, name: str) -> List[Tuple[str, int]]:
        return self._outgoing.get(name, [])

    def module_of(self, name: str) -> Optional[nn.Module]:
        return self.modules.get(name)

    # -- queries ------------------------------------------------------------ #
    def parametric_nodes(self) -> List[OpNode]:
        return [n for n in self.nodes.values() if n.is_parametric]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
            "edges": [
                {"src": s, "dst": d, "input_index": i, "kind": k}
                for (s, d, i, k) in self.edges
            ],
            "warnings": self.warnings,
        }


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #
def _classify_module_type(raw_type: str) -> str:
    if raw_type == "Conv2d" or raw_type == "Conv1d" or raw_type == "Conv3d":
        return OP_CONV
    if raw_type.startswith("BatchNorm"):
        return OP_BN
    if raw_type == "Linear":
        return OP_LINEAR
    if raw_type == "ConvTranspose2d":
        return OP_CONVT
    if raw_type in ("LayerNorm", "GroupNorm"):
        return OP_NORM
    if raw_type == "MultiheadAttention":
        return OP_ATTENTION
    return OP_OTHER


def _classify_tensor_op(op: str, scope: List[str], bev_kw: Tuple[str, ...]) -> str:
    low = op.lower()
    scope_low = " ".join(scope).lower()
    if any(k in scope_low or k in low for k in bev_kw):
        return OP_BEV_WARP
    if op in ("torch.add", "Tensor.__add__", "Tensor.__radd__", "Tensor.__iadd__"):
        return OP_ADD
    if op == "torch.cat":
        return OP_CAT
    if op in ("torch.split", "Tensor.split", "Tensor.chunk", "torch.chunk"):
        return OP_SPLIT
    if op in ("Tensor.view", "Tensor.reshape", "Tensor.flatten", "Tensor.squeeze",
              "Tensor.unsqueeze", "Tensor.expand", "Tensor.repeat", "Tensor.contiguous"):
        return OP_VIEW
    if op in ("Tensor.permute", "Tensor.transpose"):
        return OP_PERMUTE
    if op in ("F.interpolate", "torch.nn.functional.interpolate"):
        return OP_INTERPOLATE
    return OP_OTHER


def build_op_graph(
    trace_graph: Dict[str, Any],
    model: nn.Module,
    protected_layers: Optional[List[str]] = None,
    det_head_keywords: Tuple[str, ...] = DEFAULT_DET_HEAD_KEYWORDS,
    protected_keywords: Tuple[str, ...] = DEFAULT_PROTECTED_KEYWORDS,
    bev_warp_keywords: Tuple[str, ...] = DEFAULT_BEV_WARP_KEYWORDS,
) -> OpGraph:
    """Construct an :class:`OpGraph` from a runtime trace.

    Args:
        trace_graph: ``{"nodes": {...}, "edges": [...]}`` from the forward
            wrapper. Module nodes carry ``type`` (class name) + channel attrs;
            tensor-op nodes carry ``type="TensorOp"`` + ``op``.
        model: Live model (for module references during surgery).
        protected_layers: Explicit layer names whose output must not change.
        det_head_keywords: Substrings marking detection-head modules.
        protected_keywords: Substrings marking fully-protected interface modules.
        bev_warp_keywords: Substrings marking BEV-warp tensor ops.

    Returns:
        A finalized :class:`OpGraph`.
    """
    explicit_protected = set(protected_layers or [])
    graph = OpGraph(model)
    raw_nodes = trace_graph.get("nodes", {})

    for name, info in raw_nodes.items():
        raw_type = info.get("type", "")
        if raw_type == "TensorOp":
            op = info.get("op", "")
            op_type = _classify_tensor_op(op, info.get("module_scope", []), bev_warp_keywords)
            node = OpNode(
                name=name,
                op_type=op_type,
                raw_type=op,
                cat_dim=info.get("cat_dim"),
                num_inputs=info.get("num_inputs"),
                input_shapes=info.get("input_shapes", []),
                output_shapes=info.get("output_shapes", []),
                module_scope=list(info.get("module_scope", [])),
            )
        else:
            op_type = _classify_module_type(raw_type)
            node = OpNode(
                name=name,
                op_type=op_type,
                raw_type=raw_type,
                in_channels=info.get("in_channels"),
                out_channels=info.get("out_channels"),
                num_features=info.get("num_features"),
                groups=info.get("groups"),
            )
            if op_type == OP_LINEAR:
                node.in_channels = info.get("in_features")
                node.out_channels = info.get("out_features")
            if op_type == OP_ATTENTION:
                node.out_channels = info.get("embed_dim")
            low = name.lower()
            if name in explicit_protected or any(k in low for k in protected_keywords):
                node.protected = True
                node.protected_reason = "explicit" if name in explicit_protected else "interface"
            if any(k in low for k in det_head_keywords):
                node.is_det_head = True
                # Head output channels are fixed by the task; protect the out axis.
                node.protected = True
                node.protected_reason = "det_head_output"
        graph.add_node(node)

    for edge in trace_graph.get("edges", []):
        src = edge.get("src", "")
        dst = edge.get("dst", "")
        if not src or not dst:
            continue
        graph.add_edge(src, dst, int(edge.get("input_index", 0)), edge.get("kind", "flow"))

    graph.finalize()
    logger.info(
        "Built op graph: %d nodes (%d parametric), %d edges",
        len(graph.nodes), len(graph.parametric_nodes()), len(graph.edges),
    )
    return graph
