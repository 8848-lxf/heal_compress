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
