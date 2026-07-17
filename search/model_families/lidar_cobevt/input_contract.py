"""Fixed-K input identity for CoBEVT point-pillar deployment."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class FixedKContract:
    fixed_k: int
    source_max_k: int
    alignment: int
    record_count: int
    overflow_count: int
    manifest_sha256: str


def derive_fixed_k(records: Iterable[int], *, alignment: int) -> FixedKContract:
    """Derive a fixed-K profile from ordered manifest voxel counts."""

    if alignment <= 0:
        raise ValueError("alignment_must_be_positive")
    counts = [int(value) for value in records]
    if not counts:
        raise ValueError("voxel_count_records_empty")
    if any(value <= 0 for value in counts):
        raise ValueError("voxel_counts_must_be_positive")
    source_max = max(counts)
    fixed_k = int(math.ceil(source_max / alignment) * alignment)
    overflow_count = sum(value > fixed_k for value in counts)
    canonical = json.dumps(
        {"alignment": alignment, "voxel_counts": counts},
        sort_keys=True,
        separators=(",", ":"),
    )
    return FixedKContract(
        fixed_k=fixed_k,
        source_max_k=source_max,
        alignment=alignment,
        record_count=len(counts),
        overflow_count=overflow_count,
        manifest_sha256=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    )

