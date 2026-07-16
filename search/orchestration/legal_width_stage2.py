"""Archive-first Stage-2 deployment for legal-width GA candidates."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..hashing import canonical_json_hash
from ..candidate import normalize_precision


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


def _serialized_precision_hash(phenotype: Mapping[str, Any]) -> str:
    profile = dict(phenotype.get("precision_profile", {}) or {})
    expanded = {}
    for module_path, source in sorted(profile.items()):
        if isinstance(source, Mapping):
            precision = source.get(
                "requested_precision", source.get("realized_precision", "")
            )
        else:
            precision = source
        expanded[str(module_path)] = normalize_precision(str(precision))
    return canonical_json_hash(expanded)


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
                    "task_protocol": "build_smoke",
                    "task_cache_key": canonical_json_hash(
                        {
                            "protocol": "build_smoke",
                            "candidate_hash": record.candidate_hash,
                            "precision_hash": precision_hash,
                            "smoke_frames": int(smoke_frames or 10),
                            "smoke_warmup_frames": int(smoke_warmup_frames),
                        }
                    ),
                    "candidate_hash": record.candidate_hash,
                    "phenotype": record.phenotype.to_dict(),
                    "output_dir": str(candidate_dir.resolve()),
                    "seed_family": str(
                        record.phenotype.metadata.get("seed_family", "ga")
                    ),
                    "stage1_metrics": dict(record.metrics),
                    "raw_precision_gene_hash": precision_hash,
                    "repaired_precision_gene_hash": precision_hash,
                    "smoke_frames": int(smoke_frames or 10),
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
    evaluation_protocol: str = "full_validation",
) -> dict[str, Any]:
    """Evaluate screening fronts on the same engines and a full manifest."""

    destination = Path(run_dir)
    successful_screening = [
        dict(row) for row in screening_rows if str(row.get("status", "")) == "ok"
    ]
    selected_by_hash: dict[str, dict[str, Any]] = {
        str(row["candidate_hash"]): row
        for row in successful_screening
        if bool(row.get("force_full_validation", False))
    }
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
        "anchor_id",
        "precision_variant",
        "requested_prune_rate",
        "realized_prune_rate",
        "mask_hash",
        "proxy_result",
        "candidate_source",
        "BOPS_retention",
        "parameter_retention",
        "calibration_manifest_hash",
    )
    for row in selected:
        engine_path = Path(str(row.get("engine_path", "")))
        if not engine_path.is_file():
            continue
        tasks.append(
            {
                "task_protocol": str(evaluation_protocol),
                "task_cache_key": canonical_json_hash(
                    {
                        "protocol": str(evaluation_protocol),
                        "candidate_hash": str(row["candidate_hash"]),
                        "deployment_hash": str(row.get("deployment_hash", "")),
                        "engine_hash": str(row.get("engine_hash", "")),
                        "required_evaluated_frames": int(
                            required_evaluated_frames
                        ),
                        "required_skipped_frames": int(required_skipped_frames),
                    }
                ),
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
                "engine_path": str(engine_path.resolve()),
                "deployment_metadata": {
                    **{key: row.get(key) for key in metadata_fields},
                    "candidate_hash": str(row["candidate_hash"]),
                    "evaluation_protocol": str(evaluation_protocol),
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
                "evaluation_protocol": str(evaluation_protocol),
                "full_validation_success": passed,
                "evaluated_frames": evaluated,
                "skipped_frames": skipped,
                "validation_manifest_hash": str(
                    row.get("eval_manifest_hash", "")
                ),
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


def _lineage_identity(row: Mapping[str, Any], *, built: bool) -> str:
    keys = (
        ("deployment_identity", "deployment_hash", "engine_hash", "candidate_hash")
        if built
        else ("phenotype_hash", "candidate_hash")
    )
    for key in keys:
        value = str(row.get(key, ""))
        if value:
            return value
    raise ValueError("full_validation_candidate_identity_missing")


def _lineage_reference(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: row.get(key)
        for key in (
            "budget",
            "target_bops",
            "budget_label",
            "generation",
            "candidate_hash",
            "candidate_source",
        )
        if row.get(key) is not None
    }


def _full_validation_tasks(
    rows: Sequence[Mapping[str, Any]],
    *,
    destination: Path,
    required_evaluated_frames: int,
    required_skipped_frames: int,
) -> list[dict[str, Any]]:
    tasks = []
    for source in rows:
        row = dict(source)
        engine_path = Path(str(row.get("engine_path", "")))
        if not engine_path.is_file():
            continue
        candidate_hash = str(row["candidate_hash"])
        deployment_key = _lineage_identity(row, built=True)
        tasks.append(
            {
                "task_protocol": "full_validation",
                "task_cache_key": canonical_json_hash(
                    {
                        "protocol": "full_validation",
                        "deployment_identity": deployment_key,
                        "engine_hash": str(row.get("engine_hash", "")),
                        "required_evaluated_frames": int(
                            required_evaluated_frames
                        ),
                        "required_skipped_frames": int(required_skipped_frames),
                    }
                ),
                "candidate_hash": candidate_hash,
                "engine_path": str(engine_path.resolve()),
                "output_dir": str(
                    (destination / "full_validation" / candidate_hash).resolve()
                ),
                "deployment_metadata": {
                    **row,
                    "evaluation_protocol": "full_validation",
                },
            }
        )
    return tasks


def _normalize_full_validation_results(
    results: Sequence[Mapping[str, Any]],
    *,
    required_evaluated_frames: int,
    required_skipped_frames: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    normalized = []
    successful = []
    for source in results:
        row = dict(source)
        evaluated = int(
            row.get("evaluated", row.get("num_evaluated_frames", -1))
        )
        skipped = int(row.get("skipped", row.get("num_skipped_frames", -1)))
        passed = (
            str(row.get("status", "")) == "ok"
            and evaluated == int(required_evaluated_frames)
            and skipped == int(required_skipped_frames)
            and bool(row.get("precision_identity_passed", False))
            and math.isfinite(float(row.get("mAP", float("nan"))))
        )
        row.update(
            {
                "evaluation_protocol": "full_validation",
                "full_validation_success": passed,
                "evaluated_frames": evaluated,
                "skipped_frames": skipped,
                "candidate_id": str(
                    row.get("candidate_id", row.get("candidate_hash", ""))
                ),
                "R_BOPS": row.get("R_BOPS", row.get("BOPS_retention")),
                "R_param": row.get(
                    "R_param", row.get("parameter_retention")
                ),
            }
        )
        normalized.append(row)
        if passed:
            successful.append(row)
    return normalized, successful


def run_generation_winner_full_validation(
    *,
    generation_winners: Sequence[Mapping[str, Any]],
    stage2_pool: Any,
    run_dir: str | Path,
    required_evaluated_frames: int = 1789,
    required_skipped_frames: int = 0,
) -> dict[str, Any]:
    """Full-validate every unique generation-winner deployment exactly once."""

    destination = Path(run_dir)
    unique: dict[str, dict[str, Any]] = {}
    lineage: dict[str, list[dict[str, Any]]] = {}
    for source in generation_winners:
        row = dict(source)
        identity = _lineage_identity(row, built=True)
        unique.setdefault(identity, row)
        lineage.setdefault(identity, []).append(_lineage_reference(row))
    selected = []
    for identity, row in unique.items():
        selected.append(
            {
                **row,
                "candidate_source": str(row.get("candidate_source", "ga")),
                "lineage_references": lineage[identity],
            }
        )
    tasks = _full_validation_tasks(
        selected,
        destination=destination,
        required_evaluated_frames=required_evaluated_frames,
        required_skipped_frames=required_skipped_frames,
    )
    results = stage2_pool.map_tasks(tasks) if tasks else []
    normalized, successful = _normalize_full_validation_results(
        results,
        required_evaluated_frames=required_evaluated_frames,
        required_skipped_frames=required_skipped_frames,
    )
    summary = {
        "unique_deployment_count": len(unique),
        "lineage_reference_count": sum(len(rows) for rows in lineage.values()),
        "task_count": len(tasks),
        "successful_count": len(successful),
        "results": normalized,
        "successful_candidates": successful,
    }
    _write_json(destination / "generation_winner_full_validation.json", summary)
    _write_csv(destination / "generation_winner_full_validation.csv", normalized)
    return summary


def run_greedy_endpoint_full_validation(
    *,
    endpoints: Sequence[Mapping[str, Any]],
    stage2_pool: Any,
    run_dir: str | Path,
    required_evaluated_frames: int = 1789,
    required_skipped_frames: int = 0,
) -> dict[str, Any]:
    """Build and full-validate only unique terminal greedy endpoint phenotypes."""

    destination = Path(run_dir)
    unique: dict[str, dict[str, Any]] = {}
    lineage: dict[str, list[dict[str, Any]]] = {}
    for source in endpoints:
        row = dict(source)
        identity = _lineage_identity(row, built=False)
        unique.setdefault(identity, row)
        lineage.setdefault(identity, []).append(_lineage_reference(row))
    build_tasks = []
    identities = []
    for identity, row in unique.items():
        candidate_hash = str(row["candidate_hash"])
        phenotype = dict(row.get("phenotype", {}) or {})
        precision_hash = _serialized_precision_hash(phenotype)
        build_tasks.append(
            {
                "task_protocol": "build_smoke",
                "task_cache_key": canonical_json_hash(
                    {
                        "protocol": "build_smoke",
                        "candidate_hash": candidate_hash,
                        "precision_hash": precision_hash,
                    }
                ),
                "candidate_hash": candidate_hash,
                "phenotype": phenotype,
                "output_dir": str(
                    (destination / "greedy_build" / candidate_hash).resolve()
                ),
                "raw_precision_gene_hash": precision_hash,
                "repaired_precision_gene_hash": precision_hash,
                "smoke_frames": 10,
                "smoke_warmup_frames": 10,
            }
        )
        identities.append(identity)
    build_results = stage2_pool.map_tasks(build_tasks) if build_tasks else []
    built = []
    for identity, endpoint, result in zip(
        identities, unique.values(), build_results
    ):
        if str(result.get("status", "")) != "ok":
            continue
        metrics = dict(endpoint.get("metrics", {}) or {})
        built.append(
            {
                **dict(endpoint),
                **metrics,
                **dict(result),
                "candidate_source": "greedy",
                "lineage_references": lineage[identity],
                "R_BOPS": metrics.get(
                    "R_BOPS", metrics.get("R_bops_vs_fp32")
                ),
                "R_param": 1.0
                - float(metrics.get("R_prune", 1.0 - metrics.get("R_param", 1.0))),
            }
        )
    tasks = _full_validation_tasks(
        built,
        destination=destination,
        required_evaluated_frames=required_evaluated_frames,
        required_skipped_frames=required_skipped_frames,
    )
    results = stage2_pool.map_tasks(tasks) if tasks else []
    normalized, successful = _normalize_full_validation_results(
        results,
        required_evaluated_frames=required_evaluated_frames,
        required_skipped_frames=required_skipped_frames,
    )
    summary = {
        "unique_deployment_count": len(built),
        "unique_endpoint_count": len(unique),
        "budget_lineage_count": sum(len(rows) for rows in lineage.values()),
        "build_task_count": len(build_tasks),
        "full_validation_task_count": len(tasks),
        "successful_count": len(successful),
        "build_results": [dict(row) for row in build_results],
        "results": normalized,
        "successful_candidates": successful,
    }
    _write_json(destination / "greedy_full_validation.json", summary)
    _write_csv(destination / "greedy_full_validation.csv", normalized)
    return summary
