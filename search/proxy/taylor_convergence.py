"""Deterministic prefix-sample convergence checks for cached Taylor actions."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: (float(values[index]), index))
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and float(values[order[end]]) == float(values[order[position]]):
            end += 1
        rank = 0.5 * ((position + 1) + end)
        for offset in range(position, end):
            ranks[order[offset]] = rank
        position = end
    return ranks


def spearman_rank(left: Sequence[float], right: Sequence[float]) -> float:
    """Return tie-aware Spearman correlation without an optional SciPy dependency."""

    if len(left) != len(right) or len(left) < 2:
        raise ValueError("taylor_convergence_rank_input_invalid")
    lhs = _average_ranks(left)
    rhs = _average_ranks(right)
    lmean = sum(lhs) / len(lhs)
    rmean = sum(rhs) / len(rhs)
    numerator = sum((a - lmean) * (b - rmean) for a, b in zip(lhs, rhs))
    lnorm = math.sqrt(sum((a - lmean) ** 2 for a in lhs))
    rnorm = math.sqrt(sum((b - rmean) ** 2 for b in rhs))
    if lnorm == 0.0 or rnorm == 0.0:
        return 1.0 if lhs == rhs else 0.0
    return float(numerator / (lnorm * rnorm))


def audit_prefix_action_convergence(
    action_rows_by_prefix: Mapping[int, Sequence[Mapping[str, Any]]],
    *,
    top_k: int = 10,
    required_pair: tuple[int, int] = (16, 32),
    minimum_spearman: float = 0.95,
    minimum_top_k_overlap: float = 0.80,
) -> dict[str, Any]:
    """Compare fixed action identities across 8/16/32-style prefix caches."""

    prefixes = tuple(sorted(int(value) for value in action_rows_by_prefix))
    if any(not action_rows_by_prefix[prefix] for prefix in prefixes):
        raise ValueError("taylor_convergence_empty_prefix")
    by_prefix = {
        prefix: {str(row["action_id"]): dict(row) for row in action_rows_by_prefix[prefix]}
        for prefix in prefixes
    }
    identities = set(by_prefix[prefixes[0]])
    for prefix in prefixes[1:]:
        if set(by_prefix[prefix]) != identities:
            raise RuntimeError(f"taylor_convergence_action_identity_mismatch:{prefix}")
    ordered_ids = sorted(identities)
    pair_rows: list[dict[str, Any]] = []
    for left, right in zip(prefixes[:-1], prefixes[1:]):
        lhs = [float(by_prefix[left][action_id]["J_total"]) for action_id in ordered_ids]
        rhs = [float(by_prefix[right][action_id]["J_total"]) for action_id in ordered_ids]
        count = min(int(top_k), len(ordered_ids))
        left_top = set(sorted(ordered_ids, key=lambda key: (by_prefix[left][key]["J_total"], key))[:count])
        right_top = set(sorted(ordered_ids, key=lambda key: (by_prefix[right][key]["J_total"], key))[:count])
        pair_rows.append(
            {
                "left_samples": left,
                "right_samples": right,
                "action_count": len(ordered_ids),
                "spearman": spearman_rank(lhs, rhs),
                "top_k": count,
                "top_k_overlap": len(left_top & right_top) / max(count, 1),
            }
        )
    selected = next(
        (
            row
            for row in pair_rows
            if (row["left_samples"], row["right_samples"]) == tuple(required_pair)
        ),
        None,
    )
    if selected is None:
        raise RuntimeError(f"taylor_convergence_required_pair_missing:{required_pair}")
    insufficient = bool(
        float(selected["spearman"]) < float(minimum_spearman)
        or float(selected["top_k_overlap"]) < float(minimum_top_k_overlap)
    )
    domains: dict[str, dict[str, list[float]]] = {}
    for action_id in ordered_ids:
        domain = str(by_prefix[prefixes[-1]][action_id].get("domain_type", "unknown"))
        entry = domains.setdefault(domain, {str(prefix): [] for prefix in prefixes})
        for prefix in prefixes:
            entry[str(prefix)].append(float(by_prefix[prefix][action_id]["J_total"]))
    domain_rows = []
    left, right = required_pair
    for domain, values in sorted(domains.items()):
        if len(values[str(left)]) < 2:
            correlation = 1.0
        else:
            correlation = spearman_rank(values[str(left)], values[str(right)])
        domain_rows.append(
            {"domain_type": domain, "action_count": len(values[str(left)]), "spearman_16_32": correlation}
        )
    return {
        "schema_version": "v2xvit-taylor-prefix-convergence-v1",
        "prefixes": list(prefixes),
        "same_action_identity": True,
        "action_count": len(ordered_ids),
        "pairwise": pair_rows,
        "domain_rank_stability": domain_rows,
        "required_pair": list(required_pair),
        "minimum_spearman": float(minimum_spearman),
        "minimum_top_k_overlap": float(minimum_top_k_overlap),
        "taylor_sample_convergence_insufficient": insufficient,
        "passed": not insufficient,
    }


__all__ = ["audit_prefix_action_convergence", "spearman_rank"]
