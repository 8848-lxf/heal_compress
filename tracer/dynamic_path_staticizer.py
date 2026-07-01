from __future__ import annotations

from typing import Any


def detect_dynamic_paths(trace_graph: dict[str, Any]) -> list[dict[str, Any]]:
    paths = []
    for name, info in trace_graph.get("nodes", {}).items():
        text = " ".join([name, str(info.get("op", "")), " ".join(info.get("module_scope", []))]).lower()
        if any(token in text for token in ("record_len", "agent", "fusion", "pairwise", "warp")):
            paths.append({"node": name, "reason": "runtime_control_or_agent_dependent_path"})
    return paths


def staticize_paths(dynamic_paths: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            **path,
            "staticization": "recorded_from_executed_forward_path",
            "status": "staticized_for_dependency_analysis",
        }
        for path in dynamic_paths
    ]
