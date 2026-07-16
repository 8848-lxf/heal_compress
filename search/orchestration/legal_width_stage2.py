"""Archive-first Stage-2 deployment for legal-width GA candidates."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from ..hashing import canonical_json_hash


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    fields = sorted({str(key) for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(value, sort_keys=True, default=str)
                        if isinstance(value, (dict, list, tuple))
                        else value
                    )
                    for key, value in row.items()
                }
            )


def _expanded_precision_hash(phenotype: Any) -> str:
    return canonical_json_hash(
        {
            str(module_path): str(decision.requested_precision).upper()
            for module_path, decision in sorted(phenotype.precision_profile.items())
        }
    )


def run_legal_width_stage2_screening(
    *,
    archive: Any,
    records_by_phenotype_hash: Mapping[str, Any],
    stage2_pool: Any,
    run_dir: str | Path,
    minimum_successful_candidates: int = 15,
    maximum_attempts: int | None = None,
    smoke_frames: int = 0,
    smoke_warmup_frames: int = 10,
) -> dict[str, Any]:
    """Deploy diverse archive candidates and backfill failures fail-closed."""

    destination = Path(run_dir)
    requested = max(1, int(minimum_successful_candidates))
    limit = int(maximum_attempts or max(requested * 3, requested))
    selected = archive.select_stage2_candidates(limit)
    results: list[dict[str, Any]] = []
    successful: list[dict[str, Any]] = []
    attempted = 0

    while len(successful) < requested and attempted < len(selected):
        wave_rows = selected[
            attempted : min(attempted + int(stage2_pool.parallelism), len(selected))
        ]
        tasks: list[dict[str, Any]] = []
        task_records: list[Any] = []
        for archive_row in wave_rows:
            phenotype_hash = str(archive_row["phenotype_hash"])
            record = records_by_phenotype_hash.get(phenotype_hash)
            if record is None:
                results.append(
                    {
                        **dict(archive_row),
                        "status": "archive_record_missing",
                        "failure_reason": "archive_record_missing",
                    }
                )
                continue
            candidate_dir = destination / "stage2_screening" / record.candidate_hash
            candidate_dir.mkdir(parents=True, exist_ok=True)
            _write_json(candidate_dir / "genotype.json", record.genotype.to_dict())
            _write_json(candidate_dir / "phenotype.json", record.phenotype.to_dict())
            precision_hash = _expanded_precision_hash(record.phenotype)
            tasks.append(
                {
                    "candidate_hash": record.candidate_hash,
                    "phenotype": record.phenotype.to_dict(),
                    "output_dir": str(candidate_dir.resolve()),
                    "seed_family": str(
                        record.phenotype.metadata.get("seed_family", "ga")
                    ),
                    "stage1_metrics": dict(record.metrics),
                    "raw_precision_gene_hash": precision_hash,
                    "repaired_precision_gene_hash": precision_hash,
                    "smoke_frames": int(smoke_frames),
                    "smoke_warmup_frames": int(smoke_warmup_frames),
                }
            )
            task_records.append(record)
        attempted += len(wave_rows)
        if tasks:
            wave_results = stage2_pool.map_tasks(tasks)
            for record, result in zip(task_records, wave_results):
                row = {
                    "candidate_hash": record.candidate_hash,
                    "structure_hash": record.phenotype.metadata.get(
                        "structure_hash", ""
                    ),
                    "precision_hash": record.phenotype.metadata.get(
                        "precision_hash", ""
                    ),
                    "phenotype_hash": record.phenotype.metadata.get(
                        "phenotype_hash", ""
                    ),
                    "F1": record.F1,
                    **dict(result),
                }
                results.append(row)
                if str(row.get("status", "")) == "ok":
                    successful.append(row)

    failures = Counter(
        str(row.get("failure_reason") or row.get("status") or "unknown")
        for row in results
        if str(row.get("status", "")) != "ok"
    )
    summary = {
        "requested_successful_count": requested,
        "archive_supply_count": len(selected),
        "attempted_count": attempted,
        "successful_count": len(successful),
        "minimum_success_reached": len(successful) >= requested,
        "failure_reason_histogram": dict(sorted(failures.items())),
        "results": results,
        "successful_candidates": successful,
    }
    _write_json(destination / "stage2_screening_results.json", summary)
    _write_csv(destination / "stage2_screening_results.csv", results)
    return summary


def _screening_nondominated(
    rows: list[Mapping[str, Any]], resource_key: str
) -> list[dict[str, Any]]:
    valid = [
        dict(row)
        for row in rows
        if str(row.get("status", "")) == "ok"
        and row.get(resource_key) is not None
        and row.get("mAP") is not None
    ]
    return [
        candidate
        for candidate in valid
        if not any(
            float(other[resource_key]) <= float(candidate[resource_key])
            and float(other["mAP"]) >= float(candidate["mAP"])
            and (
                float(other[resource_key]) < float(candidate[resource_key])
                or float(other["mAP"]) > float(candidate["mAP"])
            )
            for other in valid
            if other is not candidate
        )
    ]


def run_legal_width_full_validation(
    *,
    screening_rows: list[Mapping[str, Any]],
    stage2_pool: Any,
    run_dir: str | Path,
    minimum_successful_candidates: int = 5,
    required_evaluated_frames: int = 1789,
    required_skipped_frames: int = 0,
) -> dict[str, Any]:
    """Evaluate screening fronts on the same engines and a full manifest."""

    destination = Path(run_dir)
    successful_screening = [
        dict(row) for row in screening_rows if str(row.get("status", "")) == "ok"
    ]
    selected_by_hash: dict[str, dict[str, Any]] = {}
    for key in ("R_BOPS", "R_param", "forward_p50_ms"):
        for row in _screening_nondominated(successful_screening, key):
            selected_by_hash.setdefault(str(row["candidate_hash"]), row)
    ordered_supply = sorted(
        successful_screening,
        key=lambda row: (
            -float(row.get("mAP", 0.0)),
            float(row.get("R_BOPS", float("inf"))),
            str(row.get("candidate_hash", "")),
        ),
    )
    for row in ordered_supply:
        if len(selected_by_hash) >= int(minimum_successful_candidates):
            break
        selected_by_hash.setdefault(str(row["candidate_hash"]), row)
    selected = list(selected_by_hash.values())
    tasks = []
    metadata_fields = (
        "structure_hash",
        "precision_hash",
        "phenotype_hash",
        "physical_hash",
        "deployment_hash",
        "engine_hash",
        "R_BOPS",
        "R_param",
        "raw_precision_gene_hash",
        "repaired_precision_gene_hash",
        "requested_precision_profile_hash",
        "realized_precision_profile_hash",
        "precision_identity_passed",
        "deployment_audits_passed",
    )
    for row in selected:
        engine_path = Path(str(row.get("engine_path", "")))
        if not engine_path.is_file():
            continue
        tasks.append(
            {
                "candidate_hash": str(row["candidate_hash"]),
                "phenotype": dict(row.get("phenotype", {})) or {
                    "pruned_unit_ids": [],
                    "precision_profile": {},
                    "metadata": {},
                },
                "output_dir": str(
                    (destination / "full_validation" / str(row["candidate_hash"])).resolve()
                ),
                "evaluation_only_engine_path": str(engine_path.resolve()),
                "deployment_metadata": {
                    key: row.get(key) for key in metadata_fields
                },
            }
        )
    results = stage2_pool.map_tasks(tasks) if tasks else []
    successful: list[dict[str, Any]] = []
    normalized = []
    for result in results:
        row = dict(result)
        evaluated = int(row.get("num_evaluated_frames", row.get("evaluated", -1)))
        skipped = int(row.get("num_skipped_frames", row.get("skipped", -1)))
        passed = (
            str(row.get("status", "")) == "ok"
            and evaluated == int(required_evaluated_frames)
            and skipped == int(required_skipped_frames)
            and bool(row.get("precision_identity_passed", False))
        )
        row.update(
            {
                "evaluation_protocol": "full_validation",
                "full_validation_success": passed,
                "evaluated_frames": evaluated,
                "skipped_frames": skipped,
            }
        )
        normalized.append(row)
        if passed:
            successful.append(row)
    summary = {
        "selected_count": len(selected),
        "task_count": len(tasks),
        "successful_count": len(successful),
        "minimum_successful_candidates": int(minimum_successful_candidates),
        "minimum_success_reached": len(successful)
        >= int(minimum_successful_candidates),
        "results": normalized,
        "successful_candidates": successful,
    }
    _write_json(destination / "full_validation_results.json", summary)
    _write_csv(destination / "full_validation_results.csv", normalized)
    return summary
