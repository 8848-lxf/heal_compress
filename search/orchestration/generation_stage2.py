"""Per-generation Stage-2 deployment with uniqueness-aware backfill."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Callable, Sequence


def fixed_bops_admission(
    retention: float,
    *,
    target: float,
    tolerance: float,
) -> dict[str, Any]:
    lower = float(target) - float(tolerance)
    upper = float(target) + float(tolerance)
    violation = max(0.0, abs(float(retention) - float(target)) - float(tolerance))
    return {
        "passed": violation <= 1.0e-12,
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
    deploy_fn: Callable[[Any, Path], dict[str, Any]],
    topk: int = 5,
) -> dict[str, Any]:
    """Deploy ranked candidates until one generation has Top-K unique artifacts."""

    destination = Path(output_dir)
    generation_number = int(generation_index) + 1
    prefix = f"generation_{generation_number:03d}"
    admitted: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    seen_deployments: set[tuple[str, str]] = set()
    attempted = 0
    for rank, record in enumerate(ranked_records):
        if len(admitted) >= int(topk):
            break
        attempted += 1
        candidate_hash = str(record.candidate_hash)
        candidate_dir = destination / prefix / "stage2" / candidate_hash
        candidate_dir.mkdir(parents=True, exist_ok=True)
        result = dict(deploy_fn(record, candidate_dir))
        base = {
            "generation": generation_number,
            "stage1_rank": rank,
            "candidate_hash": candidate_hash,
            "F1": float(record.F1),
        }
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
        admitted.append({**base, **result})

    status = "ok" if len(admitted) == int(topk) else "insufficient_unique_deployable_candidates"
    report = {
        "status": status,
        "generation": generation_number,
        "topk_required": int(topk),
        "attempted_count": attempted,
        "selected_count": len(admitted),
        "candidates": admitted,
        "failure_records": failures,
    }
    top5_path = destination / f"{prefix}_top5.json"
    _write_json(top5_path, report)
    _write_json(destination / f"{prefix}_failures.json", failures)
    _write_csv(destination / f"{prefix}_stage2.csv", admitted)
    if len(admitted) != int(topk):
        raise RuntimeError(
            f"insufficient_unique_deployable_candidates:{len(admitted)}<{int(topk)}"
        )
    winner = min(admitted, key=lambda row: (float(row["F2"]), str(row["candidate_hash"])))
    report["winner"] = winner
    _write_json(destination / f"{prefix}_winner.json", winner)
    _write_json(top5_path, report)
    return report
