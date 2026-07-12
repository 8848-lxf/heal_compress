"""Torch-Pruning-backed root-channel L2 importance for method B."""

from __future__ import annotations

from typing import Any

from torch import nn


def torch_pruning_l2_channel_scores(
    module: nn.Conv2d | nn.ConvTranspose2d,
    *,
    module_path: str,
) -> dict[str, Any]:
    """Call TP MagnitudeImportance on a root-only group without TP materialization."""

    import torch_pruning as tp
    from torch_pruning.dependency.dependency import Dependency
    from torch_pruning.dependency.group import Group
    from torch_pruning.dependency.node import Node

    if not isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        raise TypeError(f"expected Conv2d/ConvTranspose2d, got {type(module).__name__}")
    node = Node(module, None, str(module_path))
    dependency = Dependency(
        tp.prune_conv_out_channels,
        tp.prune_conv_out_channels,
        node,
        node,
    )
    group = Group()
    indices = list(range(int(module.out_channels)))
    group.add_dep(dependency, indices)
    group[0].root_idxs = list(indices)
    constructor_args = {
        "p": 2,
        "group_reduction": "mean",
        "normalizer": None,
        "bias": False,
    }
    importance = tp.importance.MagnitudeImportance(**constructor_args)
    scores = importance(group)
    if scores is None or int(scores.numel()) != int(module.out_channels):
        raise RuntimeError(
            f"Torch-Pruning did not return {module.out_channels} channel scores for {module_path}"
        )
    values = scores.detach().float().cpu().tolist()
    return {
        "module_path": str(module_path),
        "scores": values,
        "torch_pruning_api_used": True,
        "torch_pruning_version": getattr(tp, "__version__", "unknown"),
        "torch_pruning_path": tp.__file__,
        "importance_class": type(importance).__name__,
        "importance_class_path": f"{type(importance).__module__}.{type(importance).__name__}",
        "constructor_args": constructor_args,
        "reduction": "mean",
        "normalization": None,
        "call_entrypoint": "MagnitudeImportance.__call__(root_only_group)",
        "returned_per_channel_importance": True,
        "score_semantics": "sum(abs(weight_channel) ** 2); native TP p=2 magnitude",
        "tp_dependency_graph_used": False,
        "tp_physical_materialization_used": False,
    }


__all__ = ["torch_pruning_l2_channel_scores"]
