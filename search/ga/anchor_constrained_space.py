"""Reconstruct a nested search space constrained by frozen exact anchors."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Sequence


def _nested_order(
    reference: Sequence[Any], removed_sets: Sequence[set[Any]], *, label: str
) -> tuple[Any, ...]:
    previous: set[Any] = set()
    ordered: list[Any] = []
    reference_tuple = tuple(reference)
    for removed in removed_sets:
        if not previous <= removed:
            raise RuntimeError(f"anchor_removed_sets_not_nested:{label}")
        segment = removed - previous
        ordered.extend(value for value in reference_tuple if value in segment)
        missing = segment - set(ordered)
        if missing:
            raise RuntimeError(f"anchor_ranking_reference_missing:{label}:{sorted(missing)}")
        previous = set(removed)
    ordered.extend(value for value in reference_tuple if value not in previous)
    if len(ordered) != len(reference_tuple) or set(ordered) != set(reference_tuple):
        raise RuntimeError(f"anchor_nested_order_not_permutation:{label}")
    return tuple(ordered)


def constrain_domains_to_frozen_anchors(
    domains: Sequence[Any],
    anchor_phenotypes: Sequence[Mapping[str, Any]],
) -> tuple[Any, ...]:
    """Preserve serialized anchor masks while filling unobserved widths by rank.

    CUDA reductions can produce harmless near-tie ordering changes on replay.
    Exact Greedy anchors already serialize their immutable masks, decoded keep
    coordinates, and ranking hash.  This function uses those states as nested
    constraints.  Widths not present in an anchor retain a deterministic order
    inside each constrained segment from the freshly collected reference rank.
    """

    if not anchor_phenotypes:
        raise ValueError("anchor_phenotypes_empty")
    anchor_domains = [
        dict(payload["metadata"]["domains"]) for payload in anchor_phenotypes
    ]
    result = []
    for domain in domains:
        states = [dict(rows[domain.domain_id]) for rows in anchor_domains]
        ranking_hashes = {str(state["ranking_hash"]) for state in states}
        if len(ranking_hashes) != 1:
            raise RuntimeError(f"anchor_ranking_hash_conflict:{domain.domain_id}")
        # One state per observed width; repeated budgets must agree exactly.
        by_width: dict[int, dict[str, Any]] = {}
        for state in states:
            width = int(state["retained_width"])
            if width in by_width and by_width[width] != state:
                raise RuntimeError(
                    f"anchor_same_width_state_conflict:{domain.domain_id}:{width}"
                )
            by_width[width] = state
        observed = [by_width[width] for width in sorted(by_width, reverse=True)]
        width_map = dict(domain.width_to_pruned_unit_ids)
        ranking_groups = dict(domain.ranking_groups)
        ordered_unit_ids = tuple(domain.ordered_unit_ids)

        if domain.domain_type == "attention_dh":
            qk_reference = tuple(
                tuple(int(value) for value in row)
                for row in ranking_groups["qk_low_to_high_by_head"]
            )
            vo_reference = tuple(
                tuple(int(value) for value in row)
                for row in ranking_groups["vo_low_to_high_by_head"]
            )
            qk_orders, vo_orders = [], []
            for role, reference, destination in (
                ("qk", qk_reference, qk_orders),
                ("vo", vo_reference, vo_orders),
            ):
                for head, head_reference in enumerate(reference):
                    removed_sets = []
                    for state in observed:
                        keep = set(
                            int(value)
                            for value in state["decoded_width_state"][
                                f"{role}_keep_by_head"
                            ][head]
                        )
                        removed_sets.append(set(head_reference) - keep)
                    destination.append(
                        _nested_order(
                            head_reference,
                            removed_sets,
                            label=f"{domain.domain_id}:{role}:head{head}",
                        )
                    )
            ranking_groups["qk_low_to_high_by_head"] = [list(row) for row in qk_orders]
            ranking_groups["vo_low_to_high_by_head"] = [list(row) for row in vo_orders]
            ordered = []
            unit_indices = dict(domain.unit_root_indices)
            unit_scores = dict(domain.unit_scores)
            for role, rankings in (("qk", qk_orders), ("vo", vo_orders)):
                for head, order in enumerate(rankings):
                    for rank, local in enumerate(order):
                        unit_id = f"{domain.domain_id}::head{head}::{role}::{local}"
                        ordered.append(unit_id)
                        unit_scores[unit_id] = float(rank)
            ordered_unit_ids = tuple(ordered)
            for width in domain.legal_widths:
                removed = []
                count = int(domain.original_width) - int(width)
                for role, rankings in (("qk", qk_orders), ("vo", vo_orders)):
                    for head, order in enumerate(rankings):
                        removed.extend(
                            f"{domain.domain_id}::head{head}::{role}::{local}"
                            for local in order[:count]
                        )
                width_map[int(width)] = tuple(removed)
            replacements = {"unit_scores": unit_scores, "unit_root_indices": unit_indices}
        elif domain.domain_type == "ffn_hidden":
            reference = tuple(
                int(value) for value in ranking_groups["ffn_low_to_high"]
            )
            removed_sets = []
            for state in observed:
                keep = set(
                    int(value)
                    for value in state["decoded_width_state"]["keep_indices"]
                )
                removed_sets.append(set(reference) - keep)
            order = _nested_order(
                reference, removed_sets, label=f"{domain.domain_id}:ffn"
            )
            ranking_groups["ffn_low_to_high"] = list(order)
            ordered_unit_ids = tuple(
                f"{domain.domain_id}::neuron::{index}" for index in order
            )
            for width in domain.legal_widths:
                width_map[int(width)] = tuple(
                    f"{domain.domain_id}::neuron::{index}"
                    for index in order[: int(domain.original_width) - int(width)]
                )
            replacements = {
                "unit_scores": {
                    f"{domain.domain_id}::neuron::{index}": float(rank)
                    for rank, index in enumerate(order)
                }
            }
        elif int(domain.groups) == 1:
            reference = tuple(domain.ordered_unit_ids)
            removed_sets = [set(state["pruned_unit_ids"]) for state in observed]
            order = _nested_order(
                reference, removed_sets, label=f"{domain.domain_id}:channel"
            )
            ordered_unit_ids = order
            for width in domain.legal_widths:
                width_map[int(width)] = tuple(
                    order[: int(domain.original_width) - int(width)]
                )
            replacements = {
                "unit_scores": {
                    unit_id: float(rank) for rank, unit_id in enumerate(order)
                }
            }
        else:
            # No grouped-convolution domain is present in V2X-ViT.  Preserve
            # its group-aware maps and only bind exact serialized anchor masks.
            replacements = {}

        for width, state in by_width.items():
            width_map[int(width)] = tuple(str(value) for value in state["pruned_unit_ids"])
        constrained = replace(
            domain,
            ordered_unit_ids=ordered_unit_ids,
            width_to_pruned_unit_ids=width_map,
            ranking_groups=ranking_groups,
            ranking_hash=next(iter(ranking_hashes)),
            **replacements,
        )
        # Verify every observed decoded state before accepting the space.
        for width, expected in by_width.items():
            if constrained.decode_width(width) != expected["decoded_width_state"]:
                raise RuntimeError(
                    f"anchor_decoded_state_mismatch:{domain.domain_id}:{width}"
                )
            if list(constrained.pruned_unit_ids_for_width(width)) != list(
                expected["pruned_unit_ids"]
            ):
                raise RuntimeError(
                    f"anchor_pruned_mask_mismatch:{domain.domain_id}:{width}"
                )
        result.append(constrained)
    return tuple(result)
