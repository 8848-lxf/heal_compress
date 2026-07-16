"""Per-generation Stage-2 deployment with uniqueness-aware backfill."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Callable, Sequence

from ..admission.bops_band import BopsBandPolicy, classify_bops_value


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
