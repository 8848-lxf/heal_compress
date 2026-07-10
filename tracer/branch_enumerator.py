"""Branch enumeration helpers for traced dependency graphs."""

from __future__ import annotations

from typing import Any


def enumerate_branch_nodes(dependency_graph: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for node in dependency_graph.get("nodes", []) or []:
        op_type = str(node.get("op_type", "")).lower()
        if "add" in op_type or "concat" in op_type:
            rows.append(
                {
                    "node_id": node.get("node_id", node.get("name", "")),
                    "op_type": node.get("op_type", ""),
                    "inputs": list(node.get("inputs", []) or []),
                }
            )
    return rows
