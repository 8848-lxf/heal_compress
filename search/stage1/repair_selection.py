"""Repair raw Stage-1 candidates, rescore phenotypes, and select unique Top-K."""

from __future__ import annotations

from typing import Any, Callable, Iterable, Optional, Union

from ..candidate import CandidateGenotype, CandidatePhenotype
from ..canonicalization import SearchSpaceSpec, canonicalize_candidate
from ..hashing import candidate_hash
from ..ga.diversity import min_distance_to_archive
from .topk_selector import ProxyCandidateRecord


RepairFn = Callable[[CandidateGenotype], tuple[Optional[Union[CandidateGenotype, CandidatePhenotype]], dict[str, Any]]]
RescoreFn = Callable[[CandidatePhenotype], dict[str, Any]]
BatchRescoreFn = Callable[[list[CandidatePhenotype]], list[dict[str, Any]]]


def select_repaired_stage2_topk(
    scored: Iterable[tuple[CandidateGenotype, float, dict[str, Any]]],
    *,
    space: SearchSpaceSpec,
    repair_fn: RepairFn,
    rescore_fn: RescoreFn,
    batch_rescore_fn: Optional[BatchRescoreFn] = None,
    topk: int,
    repair_pool_size: int | None = None,
    selection_policy: str = "lowest_f1",
    exploitation_count: int = 3,
    diversity_count: int = 2,
    taylor_relative_epsilon: float = 0.05,
    taylor_absolute_epsilon: float = 1.0e-8,
    eligibility_fn: Callable[[dict[str, Any]], bool] | None = None,
) -> tuple[list[ProxyCandidateRecord], dict[str, Any]]:
    """Select Stage-2 candidates from repaired, rescored, unique phenotypes."""

    sorted_scored = sorted(list(scored), key=lambda row: (float(row[1]), str(row[0].to_dict())))
    requested_pool_size = int(repair_pool_size or max(topk * 10, topk))
    pool = sorted_scored[:requested_pool_size]
    pending: list[tuple[str, CandidateGenotype, CandidatePhenotype, dict[str, Any], dict[str, Any]]] = []
    seen_repaired_hashes: set[str] = set()
    repair_failed_count = 0
    duplicate_count = 0
    failure_reasons: dict[str, int] = {}
    for genotype, _raw_score, raw_metrics in pool:
        repaired, repair_report = repair_fn(genotype)
        if repaired is None or str(repair_report.get("status", "ok")) != "ok":
            repair_failed_count += 1
            reason = str(repair_report.get("failure_reason", "repair_failed"))
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
            continue
        phenotype = repaired if isinstance(repaired, CandidatePhenotype) else canonicalize_candidate(repaired, space)
        key = candidate_hash(phenotype, space)
        if key in seen_repaired_hashes:
            duplicate_count += 1
            continue
        seen_repaired_hashes.add(key)
        repaired_genotype = repaired if isinstance(repaired, CandidateGenotype) else genotype
        pending.append((key, repaired_genotype, phenotype, raw_metrics, repair_report))
    if batch_rescore_fn is not None and pending:
        rescored_rows = [dict(row) for row in batch_rescore_fn([item[2] for item in pending])]
    else:
        rescored_rows = [dict(rescore_fn(item[2])) for item in pending]
    records: list[ProxyCandidateRecord] = []
    for (key, repaired_genotype, phenotype, raw_metrics, repair_report), repaired_metrics in zip(pending, rescored_rows):
        repaired_metrics.setdefault("raw_F1", float(raw_metrics.get("F1", 0.0)))
        repaired_metrics["repaired_phenotype_hash"] = key
        repaired_metrics["repair_report"] = dict(repair_report)
        records.append(
            ProxyCandidateRecord(
                candidate_hash=key,
                genotype=repaired_genotype,
                phenotype=phenotype,
                F1=float(repaired_metrics.get("F1", float("inf"))),
                metrics=repaired_metrics,
            )
        )
    rejected_after_rescore = 0
    if eligibility_fn is not None:
        eligible_records = []
        for record in records:
            if eligibility_fn(record.metrics):
                eligible_records.append(record)
            else:
                rejected_after_rescore += 1
        records = eligible_records

    def taylor(record: ProxyCandidateRecord) -> float:
        return float(
            record.metrics.get(
                "L_joint_weight_taylor",
                record.metrics.get("proxy_score_raw", record.F1),
            )
        )

    best_taylor = min((taylor(row) for row in records), default=float("inf"))
    near_limit = (
        best_taylor * (1.0 + max(0.0, float(taylor_relative_epsilon)))
        + max(0.0, float(taylor_absolute_epsilon))
    )

    def exploitation_key(record: ProxyCandidateRecord) -> tuple[Any, ...]:
        value = taylor(record)
        retention = float(record.metrics.get("R_parameter_retention", 1.0))
        if value <= near_limit:
            return (0, retention, value, record.candidate_hash)
        return (1, value, retention, record.candidate_hash)

    ordered = sorted(records, key=exploitation_key)
    selection_roles: dict[str, str] = {}
    if str(selection_policy) == "three_plus_two_diversity":
        selected = ordered[: min(int(topk), max(0, int(exploitation_count)))]
        selection_roles.update(
            {row.candidate_hash: "exploitation" for row in selected}
        )
        remaining = [row for row in ordered if row.candidate_hash not in {item.candidate_hash for item in selected}]
        for _ in range(min(max(0, int(diversity_count)), int(topk) - len(selected))):
            if not remaining:
                break
            row = max(
                remaining,
                key=lambda item: (
                    min_distance_to_archive(
                        item.genotype,
                        [entry.genotype for entry in selected],
                    ),
                    -taylor(item),
                    item.candidate_hash,
                ),
            )
            selected.append(row)
            selection_roles[row.candidate_hash] = "diversity"
            remaining = [item for item in remaining if item.candidate_hash != row.candidate_hash]
        for row in ordered:
            if len(selected) >= int(topk):
                break
            if row.candidate_hash not in {item.candidate_hash for item in selected}:
                selected.append(row)
                selection_roles[row.candidate_hash] = "backfill"
    else:
        selected = ordered[: int(topk)]
        selection_roles.update(
            {row.candidate_hash: "lowest_f1" for row in selected}
        )
    for record in selected:
        record.metrics["stage2_selection_role"] = selection_roles.get(
            record.candidate_hash, "selected"
        )
    report = {
        "repair_pool_size": min(requested_pool_size, len(sorted_scored)),
        "processed_raw_candidate_count": len(pool),
        "repair_failed_count": repair_failed_count,
        "duplicate_repaired_phenotype_count": duplicate_count,
        "legal_repaired_phenotype_count": len(records),
        "rejected_after_repaired_rescore": rejected_after_rescore,
        "selected_count": len(selected),
        "failure_reasons": failure_reasons,
        "topk_stage2": int(topk),
        "selection_policy": str(selection_policy),
        "exploitation_count": int(exploitation_count),
        "diversity_count": int(diversity_count),
        "taylor_relative_epsilon": float(taylor_relative_epsilon),
        "selected_roles": {
            record.candidate_hash: record.metrics["stage2_selection_role"]
            for record in selected
        },
    }
    return selected, report
