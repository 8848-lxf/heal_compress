"""Repair raw Stage-1 candidates, rescore phenotypes, and select unique Top-K."""

from __future__ import annotations

from typing import Any, Callable, Iterable, Optional, Union

from ..candidate import CandidateGenotype, CandidatePhenotype
from ..canonicalization import SearchSpaceSpec, canonicalize_candidate
from ..hashing import candidate_hash
from ..stage2.admission import Stage2AdmissionPolicy
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
    bops_target: float | None = None,
    bops_tolerance: float = 0.005,
    require_compression_or_int8: bool = False,
) -> tuple[list[ProxyCandidateRecord], dict[str, Any]]:
    """Select Stage-2 candidates from repaired, rescored, unique phenotypes."""

    sorted_scored = sorted(list(scored), key=lambda row: (float(row[1]), str(row[0].to_dict())))
    requested_pool_size = int(repair_pool_size or max(topk * 10, topk))
    pool = sorted_scored
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
    admission_rejected_count = 0
    admission_rejection_reasons: dict[str, int] = {}
    admission_policy = Stage2AdmissionPolicy(float(bops_target), tolerance=float(bops_tolerance)) if bops_target is not None else None
    for (key, repaired_genotype, phenotype, raw_metrics, repair_report), repaired_metrics in zip(pending, rescored_rows):
        repaired_metrics.setdefault("raw_F1", float(raw_metrics.get("F1", 0.0)))
        repaired_metrics["repaired_phenotype_hash"] = key
        repaired_metrics["repair_report"] = dict(repair_report)
        if admission_policy is not None:
            decision = admission_policy.check_repaired_candidate(phenotype, repaired_metrics)
            repaired_metrics["stage2_admission"] = decision.to_dict()
            if not decision.accepted:
                admission_rejected_count += 1
                admission_rejection_reasons[decision.reason] = admission_rejection_reasons.get(decision.reason, 0) + 1
                continue
        elif require_compression_or_int8:
            decision = Stage2AdmissionPolicy(1.0).check_repaired_candidate(phenotype, {"R_bops": 0.0})
            repaired_metrics["stage2_admission"] = decision.to_dict()
            if decision.reason == "control_only_repaired_candidate":
                admission_rejected_count += 1
                admission_rejection_reasons[decision.reason] = admission_rejection_reasons.get(decision.reason, 0) + 1
                continue
        records.append(
            ProxyCandidateRecord(
                candidate_hash=key,
                genotype=repaired_genotype,
                phenotype=phenotype,
                F1=float(repaired_metrics.get("F1", float("inf"))),
                metrics=repaired_metrics,
            )
        )
    selected = sorted(records, key=lambda row: (row.F1, row.candidate_hash))[: int(topk)]
    report = {
        "repair_pool_size": min(requested_pool_size, len(sorted_scored)),
        "processed_raw_candidate_count": len(pool),
        "repair_failed_count": repair_failed_count,
        "duplicate_repaired_phenotype_count": duplicate_count,
        "stage2_admission_rejected_count": admission_rejected_count,
        "stage2_admission_rejection_reasons": admission_rejection_reasons,
        "legal_repaired_phenotype_count": len(records),
        "selected_count": len(selected),
        "failure_reasons": failure_reasons,
        "topk_stage2": int(topk),
    }
    return selected, report
