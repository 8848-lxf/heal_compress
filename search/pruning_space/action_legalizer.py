"""Legalize pruning action genes."""

from __future__ import annotations

from .action_catalog import PruningActionCatalog


def legalize_pruning_action_genes(pruning_genes: dict[str, int], catalog: PruningActionCatalog) -> dict[str, int]:
    action_ids = set(catalog.action_ids)
    return {action_id: 1 if int(pruning_genes.get(action_id, 1)) else 0 for action_id in sorted(action_ids)}
