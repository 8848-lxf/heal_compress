"""Encoding helpers for coupled precision-group profiles."""

from __future__ import annotations

from typing import Mapping, Sequence

from ..candidate import CandidateGenotype
from ..hashing import canonical_json_hash
from .types import QuantizationSearchGroup


def quantization_group_profile_hash(
    group_profile: Mapping[str, str],
    groups: Sequence[QuantizationSearchGroup],
) -> str:
    payload = {
        "groups": [
            {
                "group_id": group.group_id,
                "module_paths": sorted(group.module_paths),
                "precision": str(group_profile.get(group.group_id, "")).upper(),
            }
            for group in sorted(groups, key=lambda row: row.group_id)
        ]
    }
    return canonical_json_hash(payload)


def forced_int8_group_genotype(
    groups: Sequence[QuantizationSearchGroup],
    *,
    minimum_int8_macs_ratio: float,
    default_precision: str = "FP16",
) -> CandidateGenotype:
    total_macs = sum(max(float(group.baseline_macs), 0.0) for group in groups) or 1.0
    target = float(minimum_int8_macs_ratio) * total_macs
    selected: set[str] = set()
    running = 0.0
    for group in sorted(groups, key=lambda row: (-float(row.baseline_macs), row.group_id)):
        if "INT8" not in group.allowed_precisions or group.protected:
            continue
        selected.add(group.group_id)
        running += max(float(group.baseline_macs), 0.0)
        if running >= target:
            break
    return CandidateGenotype(
        pruning_genes={},
        precision_genes={
            group.group_id: ("INT8" if group.group_id in selected else default_precision)
            for group in sorted(groups, key=lambda row: row.ordering)
        },
        meta={"created_by": "forced_int8_group_candidate", "minimum_int8_macs_ratio": float(minimum_int8_macs_ratio)},
    )
