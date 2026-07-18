"""Reporting helpers for the CoBEVT Attention deployment study."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence


def derive_kernel_friendly_widths(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, list[int]]:
    """Return uniform widths fused as MHA in the real mask/RPE graph."""

    widths: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        if str(row.get("status", "")) != "ok":
            continue
        if str(row.get("variant", "")) != "uniform":
            continue
        if str(row.get("graph_kind", "")) != "cobevt_mask_rpe":
            continue
        if not bool(row.get("mha_fused", False)):
            continue
        d_qk = int(row["d_qk"])
        d_v = int(row["d_v"])
        if d_qk != d_v:
            continue
        precision = str(row.get("requested_precision", "")).upper()
        widths[precision].add(d_qk)
    return {
        precision: sorted(values)
        for precision, values in sorted(widths.items())
    }


__all__ = ["derive_kernel_friendly_widths"]
