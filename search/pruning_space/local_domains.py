"""Local pruning-domain width choices for search genes."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class LocalPruningDomain:
    domain_id: str
    root_module_path: str
    root_axis: str
    scope_id: str
    original_width: int
    ordered_unit_ids: tuple[str, ...]
    legal_widths: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain_id": self.domain_id,
            "root_module_path": self.root_module_path,
            "root_axis": self.root_axis,
            "scope_id": self.scope_id,
            "original_width": self.original_width,
            "ordered_unit_ids": list(self.ordered_unit_ids),
            "legal_widths": list(self.legal_widths),
        }


def legal_dense_widths(
    *,
    original_width: int,
    minimum_retained_ratio: float,
    alignment: int = 4,
) -> tuple[int, ...]:
    """Return dense Conv/Linear retained widths aligned to 4 plus original."""

    width = int(original_width)
    align = int(alignment)
    if width <= 0 or align <= 0:
        return ()
    minimum = max(1, int(round(width * float(minimum_retained_ratio))))
    values = {width}
    for retained in range(align, width + 1, align):
        if retained >= minimum:
            values.add(retained)
    if not any(value != width for value in values):
        values.add(min(width, max(align, minimum)))
    return tuple(sorted(values))


def build_local_pruning_domains(
    units: Sequence[Any],
    *,
    minimum_retained_ratio: float = 0.10,
    dense_alignment: int = 4,
) -> list[LocalPruningDomain]:
    """Group units by root module/axis, not by coarse model stage."""

    grouped: dict[tuple[str, str, str], list[Any]] = defaultdict(list)
    for unit in units:
        if bool(getattr(unit, "protected", False)):
            continue
        key = (
            str(getattr(unit, "root_module_path", "")),
            str(getattr(unit, "root_axis", "")),
            str(getattr(unit, "scope_id", "")),
        )
        if key[0] and key[1]:
            grouped[key].append(unit)
    domains: list[LocalPruningDomain] = []
    for (root, axis, scope_id), rows in sorted(grouped.items()):
        sorted_rows = sorted(
            rows,
            key=lambda row: (
                float(getattr(row, "normalized_score", 0.0)),
                min(getattr(row, "root_indices", [0]) or [0]),
                str(getattr(row, "stable_id", "")),
            ),
        )
        original_width = max(
            [int(index) for row in rows for index in (getattr(row, "root_indices", []) or [])] or [-1]
        ) + 1
        domains.append(
            LocalPruningDomain(
                domain_id=f"{root}::{axis}",
                root_module_path=root,
                root_axis=axis,
                scope_id=scope_id,
                original_width=original_width,
                ordered_unit_ids=tuple(str(getattr(row, "stable_id")) for row in sorted_rows),
                legal_widths=legal_dense_widths(
                    original_width=original_width,
                    minimum_retained_ratio=minimum_retained_ratio,
                    alignment=dense_alignment,
                ),
            )
        )
    return domains
