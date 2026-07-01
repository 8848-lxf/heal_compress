from __future__ import annotations

from typing import Any


def inspect_model_modules(model: Any) -> dict[str, Any]:
    import torch.nn as nn

    modules = list(model.named_modules())
    conv_layers = []
    bn_layers = []
    detection_heads = []
    for name, module in modules:
        if not name:
            continue
        if isinstance(module, nn.Conv2d):
            conv_layers.append(name)
        if isinstance(module, nn.BatchNorm2d):
            bn_layers.append(name)
        if any(token in name.lower() for token in ("cls_head", "reg_head", "dir_head", "heatmap_head")):
            detection_heads.append(name)
    return {
        "total_modules": max(0, len(modules) - 1),
        "conv_layers": len(conv_layers),
        "bn_layers": len(bn_layers),
        "conv_layer_names": conv_layers,
        "bn_layer_names": bn_layers,
        "detection_head_modules": detection_heads,
    }


def summarize_trace(trace_graph: dict[str, Any], dependency_graph: dict[str, Any] | None = None) -> dict[str, Any]:
    nodes = trace_graph.get("nodes", {})
    dep_edges = (dependency_graph or {}).get("edges", [])
    trace_edges = trace_graph.get("edges", [])
    return {
        "traced_modules": sum(1 for value in nodes.values() if value.get("type") != "TensorOp"),
        "residual_edges": sum(1 for edge in dep_edges if edge.get("kind") == "residual_add"),
        "concat_edges": sum(1 for edge in dep_edges if edge.get("kind") == "concat"),
        "fusion_edges": sum(1 for edge in trace_edges if "fusion" in str(edge).lower()),
        "detection_head_edges": sum(1 for edge in trace_edges if any(token in str(edge).lower() for token in ("cls_head", "reg_head", "dir_head"))),
        "invalid_or_unsupported_ops": [
            name for name, info in nodes.items()
            if info.get("type") == "TensorOp" and str(info.get("op", "")).lower() in {"unknown", ""}
        ],
    }
