"""Lightweight formal graph schema used by pruning and precision tracing."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class GraphNode:
    node_id: str
    name: str
    op_type: str
    target: str
    module_name: str = ""
    inputs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DependencyGraph:
    nodes: list[GraphNode]
    edges: list[dict[str, str]]
    modules: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": list(self.edges),
            "modules": list(self.modules),
            "summary": {
                "num_nodes": len(self.nodes),
                "num_edges": len(self.edges),
                "num_modules": len(self.modules),
            },
        }
