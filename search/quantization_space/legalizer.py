"""Legalize coupled precision-group genes."""

from __future__ import annotations

from typing import Mapping, Sequence

from ..candidate import normalize_precision
from .types import GroupPrecisionLegalization, QuantizationSearchGroup


def legalize_group_precision_genes(
    precision_genes: Mapping[str, str],
    groups: Sequence[QuantizationSearchGroup],
    *,
    default_precision: str = "FP16",
) -> GroupPrecisionLegalization:
    requested: dict[str, str] = {}
    realized: dict[str, str] = {}
    fallback: dict[str, dict[str, str]] = {}
    for group in sorted(groups, key=lambda row: row.ordering):
        req = normalize_precision(precision_genes.get(group.group_id, default_precision), default=default_precision)
        requested[group.group_id] = req
        if group.protected:
            legal = normalize_precision(group.metadata.get("default_precision", default_precision), default=default_precision)
            reason = group.protection_reason or "protected_precision_group"
        elif req in group.allowed_precisions:
            legal = req
            reason = ""
        else:
            legal = normalize_precision(group.metadata.get("default_precision", default_precision), default=default_precision)
            if legal not in group.allowed_precisions:
                legal = "FP16" if "FP16" in group.allowed_precisions else group.allowed_precisions[0]
            reason = f"requested_precision_not_allowed:{req}"
        realized[group.group_id] = legal
        if legal != req or reason:
            fallback[group.group_id] = {
                "requested_precision": req,
                "realized_precision": legal,
                "fallback_reason": reason,
            }
    return GroupPrecisionLegalization(
        requested_group_profile=requested,
        stage1_legalized_group_profile=realized,
        fallback_report=fallback,
        groups=tuple(sorted(groups, key=lambda row: row.ordering)),
    )
