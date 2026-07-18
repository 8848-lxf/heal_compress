"""Per-generation Stage-2 deployment with uniqueness-aware backfill."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Sequence

from ..admission.bops_band import (
    BopsBandPolicy,
    classify_bops_value,
    select_bops_candidates,
)


def fixed_bops_admission(
    retention: float,
    *,
    target: float,
    tolerance: float,
) -> dict[str, Any]:
    policy = BopsBandPolicy(
        target=float(target),
        primary_tolerance=float(tolerance),
        expanded_tolerance=float(tolerance),
    )
    classification = classify_bops_value(float(retention), policy=policy)
    lower, upper = classification["primary_interval"]
    violation = max(
        0.0,
        float(classification["absolute_error"]) - float(tolerance),
    )
    return {
        "passed": classification["classification"] == "primary",
        "retention": float(retention),
        "target": float(target),
        "tolerance": float(tolerance),
        "legal_interval": [lower, upper],
        "violation": violation,
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True, default=str),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _record_metrics(record: Any) -> dict[str, Any]:
    metrics = getattr(record, "metrics", {})
    return dict(metrics) if isinstance(metrics, dict) else {}


def _record_bops(record: Any) -> float:
    metrics = _record_metrics(record)
    for key in ("R_BOPS", "R_bops_vs_fp32", "R_bops", "BOPS_retention"):
        if key in metrics:
            return float(metrics[key])
    return float("nan")


def _result_bops(result: dict[str, Any], keys: tuple[str, ...]) -> float:
    for key in keys:
        if result.get(key) is not None:
            try:
                return float(result[key])
            except (TypeError, ValueError):
                return float("nan")
    return float("nan")


def _write_generation_artifacts(
    destination: Path,
    prefix: str,
    report: dict[str, Any],
) -> None:
    _write_json(destination / f"{prefix}_top5.json", report)
    _write_json(
        destination / f"{prefix}_failures.json",
        report.get("failure_records", []),
    )
    _write_json(
        destination / f"{prefix}_bops_admission.json",
        report.get("bops_admission", {}),
    )
    _write_json(
        destination / f"{prefix}_count_decision.json",
        {
            "status": report.get("status"),
            "generation_skipped": bool(report.get("generation_skipped", False)),
            "generation_skip_reason": str(
                report.get("generation_skip_reason", "")
            ),
            "physical_preflight_attempt_count": report.get(
                "physical_preflight_attempt_count", 0
            ),
            "physical_preflight_admitted_count": report.get(
                "physical_preflight_admitted_count", 0
            ),
            "engine_build_attempt_count": report.get(
                "engine_build_attempt_count", report.get("attempted_count", 0)
            ),
            "build_success_count": report.get("build_success_count", 0),
            "evaluated_500_count": report.get("evaluated_500_count", 0),
            "selected_count": report.get("selected_count", 0),
            "failure_stage_histogram": dict(
                report.get("failure_stage_histogram", {}) or {}
            ),
            "failure_reason_histogram": dict(
                report.get("failure_reason_histogram", {}) or {}
            ),
            "evaluation_500_skipped": bool(
                report.get("winner", {}).get("evaluation_500_skipped", False)
            ),
        },
    )
    _write_csv(
        destination / f"{prefix}_stage2.csv",
        [dict(row) for row in report.get("candidates", [])],
    )
    winner = report.get("winner")
    if winner is not None:
        _write_json(destination / f"{prefix}_winner.json", winner)


def run_generation_stage2(
    ranked_records: Sequence[Any],
    *,
    generation_index: int,
    output_dir: str | Path,
    policy: BopsBandPolicy,
    physical_preflight_batch_fn: Callable[
        [list[Any]], list[dict[str, Any]]
    ]
    | None = None,
    build_smoke_batch_fn: Callable[[list[Any]], list[dict[str, Any]]],
    evaluate_500_batch_fn: Callable[
        [list[dict[str, Any]]], list[dict[str, Any]]
    ],
    topk: int = 5,
) -> dict[str, Any]:
    """BOPS-select, preflight, cap engine builds, then apply 0/1/2-5 semantics."""

    destination = Path(output_dir)
    generation_number = int(generation_index) + 1
    prefix = f"generation_{generation_number:03d}"
    records = list(ranked_records)
    record_rank_by_id = {id(record): index for index, record in enumerate(records)}
    admission_rows = [
        {
            "record_index": index,
            "candidate_hash": str(record.candidate_hash),
            "J1": float(_record_metrics(record).get("J1", -float(record.F1))),
            "R_BOPS": _record_bops(record),
        }
        for index, record in enumerate(records)
    ]
    bops_admission = select_bops_candidates(
        admission_rows,
        policy=policy,
        retention_key="R_BOPS",
    )
    annotated = list(bops_admission["annotated"])
    primary_supply = [
        records[int(row["record_index"])]
        for row in annotated
        if row["classification"] == "primary"
    ]
    expanded_supply = [
        records[int(row["record_index"])]
        for row in annotated
        if row["classification"] == "expanded_only"
        and bool(row["eligible_for_expanded"])
    ]
    failures: list[dict[str, Any]] = []
    build_admitted: list[dict[str, Any]] = []
    preflight_admitted: list[tuple[Any, dict[str, Any], str, float]] = []
    seen_physical: set[str] = set()
    seen_deployment: set[str] = set()
    preflight_seen_physical: set[str] = set()
    physical_preflight_attempt_count = 0
    engine_build_attempt_count = 0

    def preflight_supply(
        supply: list[Any], *, mode: str, tolerance: float
    ) -> None:
        nonlocal physical_preflight_attempt_count
        if physical_preflight_batch_fn is None:
            for record in supply[: max(0, int(topk) - len(preflight_admitted))]:
                preflight_admitted.append((record, {}, mode, tolerance))
            return
        cursor = 0
        while cursor < len(supply) and len(preflight_admitted) < int(topk):
            remaining = int(topk) - len(preflight_admitted)
            wave = supply[cursor : cursor + remaining]
            cursor += len(wave)
            results = [dict(row) for row in physical_preflight_batch_fn(wave)]
            if len(results) != len(wave):
                raise RuntimeError(
                    "physical_preflight_result_count_mismatch:"
                    f"{len(results)}!={len(wave)}"
                )
            physical_preflight_attempt_count += len(wave)
            lower = float(policy.target) - float(tolerance)
            upper = float(policy.target) + float(tolerance)
            for record, result in zip(wave, results):
                candidate_hash = str(record.candidate_hash)
                metrics = _record_metrics(record)
                base = {
                    "generation": generation_number,
                    "candidate_hash": candidate_hash,
                    "stage1_rank": record_rank_by_id[id(record)],
                    "F1": float(record.F1),
                    "J1": float(metrics.get("J1", -float(record.F1))),
                    "BOPS_proxy": _record_bops(record),
                    "bops_admission_mode": mode,
                    "effective_tolerance": float(tolerance),
                }
                if str(result.get("status", "")) != "ok":
                    failures.append(
                        {
                            **base,
                            **result,
                            "status": str(
                                result.get("status", "build_smoke_failed")
                            ),
                            "failure_reason": str(
                                result.get(
                                    "failure_reason",
                                    result.get("status", "physical_preflight_failed"),
                                )
                            ),
                            "failure_stage": "physical_preflight",
                        }
                    )
                    continue
                physical_hash = str(result.get("physical_hash", ""))
                if not physical_hash:
                    failures.append(
                        {
                            **base,
                            "status": "physical_identity_missing",
                            "failure_reason": "physical_hash_missing",
                            "failure_stage": "physical_preflight",
                        }
                    )
                    continue
                if physical_hash in preflight_seen_physical:
                    failures.append(
                        {
                            **base,
                            "status": "duplicate_physical_hash",
                            "failure_reason": "duplicate_physical_hash",
                            "physical_hash": physical_hash,
                            "failure_stage": "physical_preflight",
                        }
                    )
                    continue
                physical_bops = _result_bops(
                    result,
                    (
                        "physical_BOPS_retention",
                        "BOPS_physical",
                        "physical_bops_retention",
                        "BOPS_retention",
                    ),
                )
                if not math.isfinite(physical_bops) or not lower <= physical_bops <= upper:
                    failures.append(
                        {
                            **base,
                            **result,
                            "status": "physical_bops_out_of_band",
                            "failure_reason": "physical_bops_out_of_active_interval",
                            "BOPS_physical": physical_bops,
                            "active_interval": [lower, upper],
                            "failure_stage": "physical_bops_admission",
                        }
                    )
                    continue
                preflight_seen_physical.add(physical_hash)
                preflight_admitted.append(
                    (
                        record,
                        {**result, "BOPS_physical_preflight": physical_bops},
                        mode,
                        tolerance,
                    )
                )

    active_mode = "primary_bops_tolerance"
    expanded_reason = ""
    if primary_supply:
        preflight_supply(
            primary_supply,
            mode=active_mode,
            tolerance=float(policy.primary_tolerance),
        )
        if not preflight_admitted and expanded_supply:
            active_mode = "expanded_bops_tolerance"
            expanded_reason = "primary_physical_preflight_supply_exhausted"
            preflight_supply(
                expanded_supply,
                mode=active_mode,
                tolerance=float(policy.expanded_tolerance),
            )
    elif expanded_supply:
        active_mode = "expanded_bops_tolerance"
        expanded_reason = "primary_proxy_supply_absent"
        preflight_supply(
            expanded_supply,
            mode=active_mode,
            tolerance=float(policy.expanded_tolerance),
        )

    if preflight_admitted:
        build_records = [row[0] for row in preflight_admitted[: int(topk)]]
        build_results = [dict(row) for row in build_smoke_batch_fn(build_records)]
        if len(build_results) != len(build_records):
            raise RuntimeError(
                "build_smoke_result_count_mismatch:"
                f"{len(build_results)}!={len(build_records)}"
            )
        engine_build_attempt_count = len(build_records)
        for (record, preflight, mode, tolerance), result in zip(
            preflight_admitted, build_results
        ):
            candidate_hash = str(record.candidate_hash)
            metrics = _record_metrics(record)
            lower = float(policy.target) - float(tolerance)
            upper = float(policy.target) + float(tolerance)
            base = {
                "generation": generation_number,
                "candidate_hash": candidate_hash,
                "stage1_rank": record_rank_by_id[id(record)],
                "F1": float(record.F1),
                "J1": float(metrics.get("J1", -float(record.F1))),
                "BOPS_proxy": _record_bops(record),
                "bops_admission_mode": mode,
                "effective_tolerance": float(tolerance),
                **preflight,
            }
            if str(result.get("status", "")) != "ok":
                failures.append(
                    {
                        **base,
                        **result,
                        "status": str(result.get("status", "build_smoke_failed")),
                        "failure_reason": str(
                            result.get(
                                "failure_reason",
                                result.get("status", "build_smoke_failed"),
                            )
                        ),
                        "failure_stage": "engine_build_or_smoke",
                    }
                )
                continue
            physical_hash = str(result.get("physical_hash", ""))
            deployment_hash = str(result.get("deployment_hash", ""))
            if not physical_hash or not deployment_hash:
                failures.append(
                    {
                        **base,
                        "status": "deployment_identity_missing",
                        "failure_reason": "physical_or_deployment_hash_missing",
                        "failure_stage": "deployment_identity",
                    }
                )
                continue
            if physical_hash in seen_physical or deployment_hash in seen_deployment:
                reason = (
                    "duplicate_physical_hash"
                    if physical_hash in seen_physical
                    else "duplicate_deployment_hash"
                )
                failures.append(
                    {
                        **base,
                        **result,
                        "status": reason,
                        "failure_reason": reason,
                        "failure_stage": "deployment_identity",
                    }
                )
                continue
            physical_bops = _result_bops(
                result,
                (
                    "physical_BOPS_retention",
                    "BOPS_physical",
                    "physical_bops_retention",
                    "BOPS_retention",
                ),
            )
            realized_bops = _result_bops(
                result,
                (
                    "BOPS_retention",
                    "R_BOPS",
                    "realized_BOPS_retention",
                    "realized_bops_retention",
                ),
            )
            if not math.isfinite(physical_bops) or not lower <= physical_bops <= upper:
                failures.append(
                    {
                        **base,
                        **result,
                        "status": "physical_bops_out_of_band",
                        "failure_reason": "physical_bops_out_of_active_interval",
                        "failure_stage": "post_build_physical_bops_audit",
                        "BOPS_physical": physical_bops,
                        "BOPS_realized": realized_bops,
                        "active_interval": [lower, upper],
                    }
                )
                continue
            if not math.isfinite(realized_bops) or not lower <= realized_bops <= upper:
                failures.append(
                    {
                        **base,
                        **result,
                        "status": "realized_bops_out_of_band",
                        "failure_reason": "realized_bops_out_of_active_interval",
                        "failure_stage": "realized_bops_audit",
                        "BOPS_physical": physical_bops,
                        "BOPS_realized": realized_bops,
                        "active_interval": [lower, upper],
                    }
                )
                continue
            seen_physical.add(physical_hash)
            seen_deployment.add(deployment_hash)
            build_admitted.append(
                {
                    **base,
                    **result,
                    "BOPS_physical": physical_bops,
                    "BOPS_realized": realized_bops,
                    "active_interval": [lower, upper],
                }
            )

    bops_admission["deployment_admission_mode"] = active_mode
    bops_admission["expanded_tolerance_reason"] = expanded_reason
    bops_admission[
        "physical_preflight_attempt_count"
    ] = physical_preflight_attempt_count
    bops_admission["physical_preflight_admitted_count"] = len(preflight_admitted)
    bops_admission["deployment_attempted_count"] = engine_build_attempt_count
    bops_admission["deployment_admitted_count"] = len(build_admitted)
    failure_stage_histogram = dict(
        sorted(
            Counter(
                str(row.get("failure_stage", "unknown")) for row in failures
            ).items()
        )
    )
    failure_reason_histogram = dict(
        sorted(
            Counter(
                str(
                    row.get("failure_reason")
                    or row.get("status")
                    or "unknown"
                )
                for row in failures
            ).items()
        )
    )
    common = {
        "generation": generation_number,
        "topk_limit": int(topk),
        "attempted_count": engine_build_attempt_count,
        "physical_preflight_attempt_count": physical_preflight_attempt_count,
        "physical_preflight_admitted_count": len(preflight_admitted),
        "engine_build_attempt_count": engine_build_attempt_count,
        "build_success_count": len(build_admitted),
        "active_bops_mode": active_mode,
        "expanded_tolerance_reason": expanded_reason,
        "bops_admission": bops_admission,
        "failure_records": failures,
        "failure_stage_histogram": failure_stage_histogram,
        "failure_reason_histogram": failure_reason_histogram,
    }
    if not primary_supply and not expanded_supply:
        report = {
            **common,
            "status": "no_bops_admissible_candidates",
            "generation_skipped": True,
            "generation_skip_reason": "proxy_bops_admission_exhausted",
            "selected_count": 0,
            "evaluated_500_count": 0,
            "candidates": [],
        }
        _write_generation_artifacts(destination, prefix, report)
        return report
    if not preflight_admitted:
        report = {
            **common,
            "status": "no_physical_bops_admissible_candidates",
            "generation_skipped": True,
            "generation_skip_reason": "physical_bops_preflight_exhausted",
            "selected_count": 0,
            "evaluated_500_count": 0,
            "candidates": [],
        }
        _write_generation_artifacts(destination, prefix, report)
        return report
    if not build_admitted:
        report = {
            **common,
            "status": "no_deployable_candidates",
            "generation_skipped": True,
            "generation_skip_reason": "all_capped_engine_candidates_failed",
            "selected_count": 0,
            "evaluated_500_count": 0,
            "candidates": [],
        }
        _write_generation_artifacts(destination, prefix, report)
        return report
    if len(build_admitted) == 1:
        winner = {
            **build_admitted[0],
            "evaluation_500_skipped": True,
            "winner_selection_reason": "sole_deployable_candidate",
        }
        report = {
            **common,
            "status": "single_candidate_direct_winner",
            "generation_skipped": False,
            "selected_count": 1,
            "evaluated_500_count": 0,
            "candidates": [winner],
            "winner": winner,
        }
        _write_generation_artifacts(destination, prefix, report)
        return report

    evaluation_results = [
        dict(row) for row in evaluate_500_batch_fn(build_admitted)
    ]
    if len(evaluation_results) != len(build_admitted):
        raise RuntimeError(
            "evaluate_500_result_count_mismatch:"
            f"{len(evaluation_results)}!={len(build_admitted)}"
        )
    evaluated: list[dict[str, Any]] = []
    for build, evaluation in zip(build_admitted, evaluation_results):
        row = {**build, **evaluation, "evaluation_500_skipped": False}
        evaluated_frames = int(
            row.get("evaluated", row.get("num_evaluated_frames", -1))
        )
        skipped_frames = int(
            row.get("skipped", row.get("num_skipped_frames", -1))
        )
        f2 = float(row.get("F2", float("nan")))
        reasons = []
        if str(row.get("status", "")) != "ok":
            reasons.append(str(row.get("status", "evaluation_500_failed")))
        if evaluated_frames != 500:
            reasons.append("evaluated_frames_not_500")
        if skipped_frames != 0:
            reasons.append("evaluation_skip")
        if not math.isfinite(f2):
            reasons.append("finite_F2_required")
        if reasons:
            failures.append(
                {
                    **row,
                    "status": "evaluation_500_admission_failed",
                    "failure_reason": ",".join(reasons),
                    "failure_stage": "evaluation_500",
                }
            )
            continue
        evaluated.append(row)
    if not evaluated:
        report = {
            **common,
            "status": "no_valid_500_evaluations",
            "generation_skipped": True,
            "generation_skip_reason": "all_500_evaluations_failed",
            "selected_count": 0,
            "evaluated_500_count": 0,
            "candidates": [],
            "failure_records": failures,
        }
        _write_generation_artifacts(destination, prefix, report)
        return report
    winner = sorted(
        evaluated,
        key=lambda row: (-float(row["F2"]), str(row["candidate_hash"])),
    )[0]
    report = {
        **common,
        "status": "evaluated_generation_winner",
        "generation_skipped": False,
        "selected_count": len(evaluated),
        "evaluated_500_count": len(evaluated),
        "candidates": evaluated,
        "failure_records": failures,
        "winner": winner,
    }
    _write_generation_artifacts(destination, prefix, report)
    return report


def deploy_generation_with_backfill(
    ranked_records: Sequence[Any],
    *,
    generation_index: int,
    output_dir: str | Path,
    deploy_fn: Callable[[Any, Path], dict[str, Any]] | None,
    deploy_batch_fn: Callable[
        [list[tuple[Any, Path]]], list[dict[str, Any]]
    ]
    | None = None,
    parallelism: int = 1,
    topk: int = 5,
    require_individual_hash_uniqueness: bool = False,
    raise_on_insufficient: bool = True,
) -> dict[str, Any]:
    """Deploy ranked candidates until one generation has Top-K unique artifacts."""

    destination = Path(output_dir)
    generation_number = int(generation_index) + 1
    prefix = f"generation_{generation_number:03d}"
    admitted: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    speculative: list[dict[str, Any]] = []
    seen_deployments: set[tuple[str, str]] = set()
    seen_physical_hashes: set[str] = set()
    seen_deployment_hashes: set[str] = set()
    attempted = 0
    width = max(1, int(parallelism))
    if deploy_batch_fn is None and deploy_fn is None:
        raise ValueError("deploy_fn_or_deploy_batch_fn_required")
    rank = 0
    while rank < len(ranked_records) and len(admitted) < int(topk):
        wave = list(ranked_records[rank : rank + width])
        prepared = []
        for record in wave:
            candidate_hash = str(record.candidate_hash)
            candidate_dir = destination / prefix / "stage2" / candidate_hash
            candidate_dir.mkdir(parents=True, exist_ok=True)
            prepared.append((record, candidate_dir))
        if deploy_batch_fn is not None:
            results = [dict(row) for row in deploy_batch_fn(prepared)]
            if len(results) != len(prepared):
                raise RuntimeError(
                    f"parallel_deploy_result_count_mismatch:{len(results)}!={len(prepared)}"
                )
        else:
            assert deploy_fn is not None
            results = [dict(deploy_fn(record, path)) for record, path in prepared]
        attempted += len(prepared)
        for offset, ((record, _candidate_dir), result) in enumerate(
            zip(prepared, results)
        ):
            candidate_hash = str(record.candidate_hash)
            base = {
                "generation": generation_number,
                "stage1_rank": rank + offset,
                "candidate_hash": candidate_hash,
                "F1": float(record.F1),
            }
            if len(admitted) >= int(topk):
                speculative.append({**base, **result})
                continue
            if str(result.get("status", "")) != "ok":
                failures.append(
                    {
                        **base,
                        "status": str(result.get("status", "deployment_failed")),
                        "failure_reason": str(
                            result.get("failure_reason", result.get("status", "deployment_failed"))
                        ),
                    }
                )
                continue
            physical_hash = str(result.get("physical_hash", ""))
            deployment_hash = str(result.get("deployment_hash", ""))
            if not physical_hash or not deployment_hash:
                failures.append(
                    {
                        **base,
                        "status": "deployment_identity_missing",
                        "failure_reason": "physical_or_deployment_hash_missing",
                    }
                )
                continue
            identity = (physical_hash, deployment_hash)
            if (
                require_individual_hash_uniqueness
                and physical_hash in seen_physical_hashes
            ):
                failures.append(
                    {
                        **base,
                        "status": "duplicate_physical_hash",
                        "failure_reason": "duplicate_physical_hash",
                        "physical_hash": physical_hash,
                        "deployment_hash": deployment_hash,
                    }
                )
                continue
            if (
                require_individual_hash_uniqueness
                and deployment_hash in seen_deployment_hashes
            ):
                failures.append(
                    {
                        **base,
                        "status": "duplicate_deployment_hash",
                        "failure_reason": "duplicate_deployment_hash",
                        "physical_hash": physical_hash,
                        "deployment_hash": deployment_hash,
                    }
                )
                continue
            if identity in seen_deployments:
                failures.append(
                    {
                        **base,
                        "status": "duplicate_deployment",
                        "failure_reason": "duplicate_physical_deployment_hash",
                        "physical_hash": physical_hash,
                        "deployment_hash": deployment_hash,
                    }
                )
                continue
            f2 = float(result.get("F2", float("inf")))
            if not math.isfinite(f2):
                failures.append(
                    {
                        **base,
                        "status": "stage2_score_invalid",
                        "failure_reason": "finite_F2_required",
                    }
                )
                continue
            seen_deployments.add(identity)
            seen_physical_hashes.add(physical_hash)
            seen_deployment_hashes.add(deployment_hash)
            admitted.append({**base, **result})
        rank += len(wave)

    status = "ok" if len(admitted) == int(topk) else "insufficient_unique_deployable_candidates"
    report = {
        "status": status,
        "generation": generation_number,
        "topk_required": int(topk),
        "attempted_count": attempted,
        "parallelism": width if deploy_batch_fn is not None else 1,
        "require_individual_hash_uniqueness": bool(
            require_individual_hash_uniqueness
        ),
        "selected_count": len(admitted),
        "candidates": admitted,
        "failure_records": failures,
        "speculative_deployments": speculative,
    }
    top5_path = destination / f"{prefix}_top5.json"
    _write_json(top5_path, report)
    _write_json(destination / f"{prefix}_failures.json", failures)
    _write_csv(destination / f"{prefix}_stage2.csv", admitted)
    if len(admitted) != int(topk) and raise_on_insufficient:
        raise RuntimeError(
            f"insufficient_unique_deployable_candidates:{len(admitted)}<{int(topk)}"
        )
    if len(admitted) != int(topk):
        return report
    winner = min(admitted, key=lambda row: (float(row["F2"]), str(row["candidate_hash"])))
    report["winner"] = winner
    _write_json(destination / f"{prefix}_winner.json", winner)
    _write_json(top5_path, report)
    return report
