"""Diverse Stage-1 to Stage-2 candidate selection."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from ..candidate import CandidateGenotype, CandidatePhenotype
from ..ga.diversity import min_distance_to_archive


@dataclass(frozen=True)
class ProxyCandidateRecord:
    candidate_hash: str
    genotype: CandidateGenotype
    phenotype: CandidatePhenotype
    F1: float
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TopKConfig:
    topk_real: int = 5
    exploitation_count: int = 3
    diversity_count: int = 1
    exploration_count: int = 1


@dataclass(frozen=True)
class TopKSelection:
    role: str
    record: ProxyCandidateRecord


def _eligible(records: list[ProxyCandidateRecord], real_eval_hashes: set[str]) -> list[ProxyCandidateRecord]:
    seen: set[str] = set()
    result = []
    for row in sorted(records, key=lambda item: (math.inf if not math.isfinite(item.F1) else item.F1, item.candidate_hash)):
        if row.candidate_hash in seen or row.candidate_hash in real_eval_hashes or not math.isfinite(row.F1):
            continue
        seen.add(row.candidate_hash)
        result.append(row)
    return result


def select_stage1_topk(
    records: list[ProxyCandidateRecord],
    *,
    real_eval_hashes: set[str],
    archive_genotypes: list[CandidateGenotype],
    config: TopKConfig | None = None,
) -> list[TopKSelection]:
    policy = config or TopKConfig()
    remaining = _eligible(records, real_eval_hashes)
    selected: list[TopKSelection] = []
    used: set[str] = set()

    def take(row: ProxyCandidateRecord, role: str) -> None:
        if row.candidate_hash not in used and len(selected) < policy.topk_real:
            selected.append(TopKSelection(role, row))
            used.add(row.candidate_hash)

    for row in remaining:
        if len([item for item in selected if item.role == "exploitation"]) >= policy.exploitation_count:
            break
        take(row, "exploitation")

    pool = [row for row in remaining if row.candidate_hash not in used]
    for _ in range(policy.diversity_count):
        if not pool:
            break
        row = max(
            pool,
            key=lambda item: (
                min_distance_to_archive(item.genotype, archive_genotypes + [entry.record.genotype for entry in selected]),
                -item.F1,
                item.candidate_hash,
            ),
        )
        take(row, "diversity")
        pool = [item for item in pool if item.candidate_hash not in used]

    for row in sorted(
        pool,
        key=lambda item: (
            -(1 if float(item.metrics.get("int8_macs_ratio", 0.0) or 0.0) > 0.0 else 0),
            -(1 if int(item.metrics.get("grouped_action_count", 0) or 0) > 0 else 0),
            -float(item.metrics.get("int8_macs_ratio", 0.0) or 0.0),
            -int(item.metrics.get("grouped_action_count", 0) or 0),
            -min_distance_to_archive(item.genotype, archive_genotypes),
            item.F1,
            item.candidate_hash,
        ),
    ):
        if len([item for item in selected if item.role == "exploration"]) >= policy.exploration_count:
            break
        take(row, "exploration")

    for row in remaining:
        if len(selected) >= policy.topk_real:
            break
        take(row, "backfill")
    return selected
