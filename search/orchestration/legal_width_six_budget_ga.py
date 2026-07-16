"""Three-seed, six-budget legal-width GA with one Stage-2 decision per generation."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..admission.bops_band import BopsBandPolicy
from ..hashing import canonical_json_hash
from ..stage1.topk_selector import ProxyCandidateRecord
from .generation_stage2 import run_generation_stage2
from .legal_width_joint_ga import run_legal_width_stage1_seeds


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )


def _budget_label(target: float) -> str:
    return f"budget_{int(round(float(target) * 100.0)):03d}"


def _phenotype_identity(record: ProxyCandidateRecord) -> str:
    metadata = dict(getattr(record.phenotype, "metadata", {}) or {})
    return str(metadata.get("phenotype_hash") or record.candidate_hash)


def _j1(record: ProxyCandidateRecord) -> float:
    value = float(record.metrics.get("J1", -float(record.F1)))
    return value if math.isfinite(value) else -float("inf")


def merge_seed_generation_records(
    seed_outputs: Mapping[int, Sequence[ProxyCandidateRecord]],
    *,
    generation: int,
) -> list[ProxyCandidateRecord]:
    """Merge one generation from independent populations by phenotype and J1."""

    candidates: list[ProxyCandidateRecord] = []
    for seed_index, records in sorted(
        seed_outputs.items(), key=lambda item: int(item[0])
    ):
        for source in records:
            metrics = {
                **dict(source.metrics),
                "seed_index": int(seed_index),
                "generation": int(generation),
            }
            candidates.append(
                ProxyCandidateRecord(
                    candidate_hash=str(source.candidate_hash),
                    genotype=source.genotype,
                    phenotype=source.phenotype,
                    F1=float(source.F1),
                    metrics=metrics,
                )
            )
    ordered = sorted(
        candidates,
        key=lambda row: (-_j1(row), str(row.candidate_hash)),
    )
    unique: dict[str, ProxyCandidateRecord] = {}
    for record in ordered:
        unique.setdefault(_phenotype_identity(record), record)
    return list(unique.values())


def _precision_hash(record: ProxyCandidateRecord) -> str:
    value = str(getattr(record.genotype, "precision_hash", ""))
    if value:
        return value
    return str(
        dict(getattr(record.phenotype, "metadata", {}) or {}).get(
            "precision_hash", ""
        )
    )


def _build_smoke_callback(
    *,
    stage2_pool: Any,
    generation_dir: Path,
    smoke_frames: int,
    smoke_warmup_frames: int,
):
    def build(records: list[ProxyCandidateRecord]) -> list[dict[str, Any]]:
        tasks: list[dict[str, Any]] = []
        for record in records:
            candidate_dir = generation_dir / "stage2" / record.candidate_hash
            candidate_dir.mkdir(parents=True, exist_ok=True)
            genotype = record.genotype.to_dict()
            phenotype = record.phenotype.to_dict()
            _write_json(candidate_dir / "genotype.json", genotype)
            _write_json(candidate_dir / "phenotype.json", phenotype)
            precision_hash = _precision_hash(record)
            tasks.append(
                {
                    "task_protocol": "build_smoke",
                    "task_cache_key": canonical_json_hash(
                        {
                            "protocol": "build_smoke",
                            "candidate_hash": record.candidate_hash,
                            "precision_hash": precision_hash,
                            "smoke_frames": int(smoke_frames),
                            "smoke_warmup_frames": int(smoke_warmup_frames),
                        }
                    ),
                    "candidate_hash": record.candidate_hash,
                    "phenotype": phenotype,
                    "output_dir": str(candidate_dir.resolve()),
                    "seed_family": str(
                        dict(getattr(record.genotype, "meta", {}) or {}).get(
                            "seed_family", "ga"
                        )
                    ),
                    "stage1_metrics": dict(record.metrics),
                    "raw_precision_gene_hash": precision_hash,
                    "repaired_precision_gene_hash": precision_hash,
                    "smoke_frames": int(smoke_frames),
                    "smoke_warmup_frames": int(smoke_warmup_frames),
                }
            )
        return [dict(row) for row in stage2_pool.map_tasks(tasks)]

    return build


def _evaluate_500_callback(*, stage2_pool: Any, generation_dir: Path):
    def evaluate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        tasks: list[dict[str, Any]] = []
        for row in rows:
            engine_path = str(row.get("engine_path", ""))
            if not engine_path:
                raise RuntimeError("evaluate_500_engine_path_required")
            deployment_key = str(
                row.get("deployment_identity")
                or row.get("deployment_hash")
                or row.get("engine_hash")
            )
            candidate_hash = str(row["candidate_hash"])
            tasks.append(
                {
                    "task_protocol": "evaluate_500",
                    "task_cache_key": canonical_json_hash(
                        {
                            "protocol": "evaluate_500",
                            "deployment_identity": deployment_key,
                            "engine_hash": str(row.get("engine_hash", "")),
                        }
                    ),
                    "candidate_hash": candidate_hash,
                    "engine_path": engine_path,
                    "output_dir": str(
                        (
                            generation_dir
                            / "evaluation_500"
                            / candidate_hash
                        ).resolve()
                    ),
                    "deployment_metadata": dict(row),
                }
            )
        return [dict(row) for row in stage2_pool.map_tasks(tasks)]

    return evaluate


def run_six_budget_joint_ga(
    context: Any,
    proxy: Any,
    stage2_pool: Any,
    run_dir: str | Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Run three independent populations and one merged Stage-2 per generation."""

    targets = sorted({float(value) for value in config.get("targets", ())})
    if not targets:
        raise ValueError("six_budget_ga_targets_required")
    seed_count = int(config.get("independent_seeds", 3))
    initial_size = int(config.get("initial_population_size", 64))
    population_size = int(config.get("population_size", 64))
    offspring_size = int(config.get("offspring_size", 64))
    generations = int(config.get("generations", 20))
    if seed_count != 3:
        raise ValueError("six_budget_ga_requires_three_independent_seeds")
    if min(initial_size, population_size, offspring_size) < 64:
        raise ValueError("six_budget_ga_population_and_offspring_must_be_at_least_64")
    if generations != 20:
        raise ValueError("six_budget_ga_generations_must_equal_20")
    primary_tolerance = float(config.get("primary_bops_tolerance", 0.005))
    expanded_tolerance = float(config.get("expanded_bops_tolerance", 0.0075))
    topk = int(config.get("topk_stage2", 5))
    smoke_frames = int(config.get("smoke_frames", 10))
    smoke_warmup_frames = int(config.get("smoke_warmup_frames", 10))
    destination = Path(run_dir)
    budget_reports: list[dict[str, Any]] = []
    generation_winners: list[dict[str, Any]] = []
    stage2_call_count = 0
    total_proxy_evaluations = 0

    for budget_index, target in enumerate(targets):
        label = _budget_label(target)
        budget_dir = destination / "joint_six_budget_ga" / label
        budget_dir.mkdir(parents=True, exist_ok=True)
        adjacent = tuple(
            targets[index]
            for index in (budget_index - 1, budget_index + 1)
            if 0 <= index < len(targets)
        )
        search_config = {
            **dict(config),
            "target_bops_retention": target,
            "bops_tolerance": primary_tolerance,
            "independent_seeds": seed_count,
            "initial_population_size": initial_size,
            "population_size": population_size,
            "offspring_size": offspring_size,
            "generations": generations,
            "seed": int(config.get("seed", 4090)) + budget_index * 10000,
            "budget_intervals": [],
        }
        stage1 = run_legal_width_stage1_seeds(
            context=context,
            proxy=proxy,
            run_dir=budget_dir / "stage1",
            search_config=search_config,
        )
        total_proxy_evaluations += int(stage1["total_proxy_evaluations"])
        generation_reports: list[dict[str, Any]] = []
        generation_records = dict(stage1.get("generation_records", {}))
        for generation in range(generations):
            seed_records = generation_records.get(
                generation, generation_records.get(str(generation), {})
            )
            merged = merge_seed_generation_records(
                seed_records,
                generation=generation,
            )
            generation_dir = budget_dir / f"generation_{generation + 1:03d}"
            report = run_generation_stage2(
                merged,
                generation_index=generation,
                output_dir=generation_dir,
                policy=BopsBandPolicy(
                    target=target,
                    primary_tolerance=primary_tolerance,
                    expanded_tolerance=expanded_tolerance,
                    adjacent_targets=adjacent,
                ),
                build_smoke_batch_fn=_build_smoke_callback(
                    stage2_pool=stage2_pool,
                    generation_dir=generation_dir,
                    smoke_frames=smoke_frames,
                    smoke_warmup_frames=smoke_warmup_frames,
                ),
                evaluate_500_batch_fn=_evaluate_500_callback(
                    stage2_pool=stage2_pool,
                    generation_dir=generation_dir,
                ),
                topk=topk,
            )
            stage2_call_count += 1
            generation_reports.append(report)
            if report.get("winner") is not None:
                winner = {
                    **dict(report["winner"]),
                    "budget": target,
                    "budget_label": label,
                    "generation": generation + 1,
                }
                generation_winners.append(winner)
        budget_report = {
            "budget": target,
            "budget_label": label,
            "generation_count": generations,
            "generation_winner_count": sum(
                report.get("winner") is not None for report in generation_reports
            ),
            "zero_candidate_generation_count": sum(
                int(report.get("selected_count", 0)) == 0
                for report in generation_reports
            ),
            "generation_reports": generation_reports,
            "stage1": {
                key: value
                for key, value in stage1.items()
                if key
                not in {
                    "archive",
                    "records_by_phenotype_hash",
                    "all_metric_rows",
                    "generation_records",
                }
            },
        }
        budget_reports.append(budget_report)
        _write_json(budget_dir / "budget_ga_report.json", budget_report)

    summary = {
        "targets": targets,
        "independent_seeds": seed_count,
        "initial_population_size": initial_size,
        "population_size": population_size,
        "offspring_size": offspring_size,
        "generations": generations,
        "topk_stage2": topk,
        "primary_bops_tolerance": primary_tolerance,
        "expanded_bops_tolerance": expanded_tolerance,
        "generation_stage2_call_count": stage2_call_count,
        "generation_winner_count": len(generation_winners),
        "total_proxy_evaluations": total_proxy_evaluations,
        "budget_reports": budget_reports,
        "generation_winners": generation_winners,
    }
    _write_json(
        destination / "joint_six_budget_ga" / "six_budget_ga_summary.json",
        summary,
    )
    return summary
