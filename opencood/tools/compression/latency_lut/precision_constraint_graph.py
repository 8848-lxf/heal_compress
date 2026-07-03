from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


MUST_SAME_PRECISION = "MUST_SAME_PRECISION"
MUST_SAME_DTYPE_BEFORE_OP = "MUST_SAME_DTYPE_BEFORE_OP"
CAST_BOUNDARY_ALLOWED = "CAST_BOUNDARY_ALLOWED"
QDQ_BOUNDARY_ALLOWED = "QDQ_BOUNDARY_ALLOWED"
FIXED_PRECISION = "FIXED_PRECISION"

MERGE_OPS = {"Add", "Sub", "Mul", "Div", "Concat", "GridSample"}


@dataclass
class PrecisionConstraintNode:
    node_id: str
    node_type: str
    precision: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PrecisionConstraintEdge:
    src: str
    dst: str
    edge_type: str
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PrecisionConstraintGraph:
    def __init__(self) -> None:
        self.nodes: dict[str, PrecisionConstraintNode] = {}
        self.edges: list[PrecisionConstraintEdge] = []

    def add_node(self, node: PrecisionConstraintNode) -> None:
        self.nodes[str(node.node_id)] = node

    def add_edge(
        self,
        src: str,
        dst: str,
        edge_type: str,
        *,
        reason: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        for node_id in (src, dst):
            if node_id not in self.nodes:
                self.add_node(PrecisionConstraintNode(node_id=node_id, node_type="precision_region"))
        self.edges.append(
            PrecisionConstraintEdge(
                src=str(src),
                dst=str(dst),
                edge_type=str(edge_type),
                reason=str(reason),
                metadata=dict(metadata or {}),
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [self.nodes[key].to_dict() for key in sorted(self.nodes)],
            "edges": [edge.to_dict() for edge in self.edges],
            "edge_type_legend": {
                MUST_SAME_PRECISION: "residual/fusion regions must share one precision profile",
                MUST_SAME_DTYPE_BEFORE_OP: "merge op inputs must have the same ONNX tensor dtype",
                CAST_BOUNDARY_ALLOWED: "FP16/FP32 cast boundary may be inserted here",
                QDQ_BOUNDARY_ALLOWED: "INT8 Q/DQ boundary may be inserted here",
                FIXED_PRECISION: "plugin or custom op is pinned to a fixed precision",
            },
        }

    def write_json(self, path: str | Path) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _node_label(node: Any) -> str:
    return str(getattr(node, "name", None) or (node.output[0] if getattr(node, "output", None) else ""))


def build_precision_constraint_graph_from_onnx(model: Any) -> PrecisionConstraintGraph:
    graph = PrecisionConstraintGraph()
    producer: dict[str, str] = {}
    for node in model.graph.node:
        node_name = _node_label(node)
        graph.add_node(PrecisionConstraintNode(node_name, "onnx_node", metadata={"op_type": node.op_type}))
        for output in node.output:
            producer[str(output)] = node_name
            graph.add_node(PrecisionConstraintNode(str(output), "tensor_edge", metadata={"producer": node_name}))
            graph.add_edge(node_name, str(output), CAST_BOUNDARY_ALLOWED, reason="node_output_tensor")

    for node in model.graph.node:
        node_name = _node_label(node)
        if node.op_type in MERGE_OPS:
            input_producers = [producer[name] for name in node.input if name in producer]
            for idx, src in enumerate(input_producers[1:], start=1):
                graph.add_edge(
                    input_producers[0],
                    src,
                    MUST_SAME_DTYPE_BEFORE_OP,
                    reason=f"{node.op_type.lower()}_requires_same_dtype",
                    metadata={"merge_node": node_name, "input_index": idx},
                )
    return graph
