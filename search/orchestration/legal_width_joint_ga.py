"""Archive-first legal-width GA coordination for the production runner."""

from __future__ import annotations

import csv
import json
import math
import random
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..archive.feasible_pareto_archive import FeasibleParetoArchive
from ..canonicalization import canonicalize_legal_width_candidate
from ..encoding.legal_width_genotype import (
    LegalWidthGenotype,
    random_legal_width_genotype,
)
from ..ga.engine import GAConfig, GeneticSearchEngine
from ..hashing import candidate_hash
from ..stage1.exception_repair import ExceptionOnlyRepairMonitor
from ..stage1.proxy_evaluator import BatchProxyResult
from ..stage1.topk_selector import ProxyCandidateRecord


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({str(key) for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _initial_population(
    space: Any,
    *,
    size: int,
    rng: random.Random,
    anchor_width_seeds: Sequence[Mapping[str, Any]] = (),
) -> list[LegalWidthGenotype]:
    inventory = space.legal_width_inventory
    full_widths = {
        domain.domain_id: len(domain.legal_keep_widths) - 1
        for domain in inventory.domains
    }
    default_precision = {
        group_id: (
            space.default_precision
            if space.default_precision in actions
            else actions[0]
        )
        for group_id, actions in space.precision_action_space.items()
    }
    proposals: list[LegalWidthGenotype] = [
        LegalWidthGenotype(
            full_widths,
            default_precision,
            {"seed_family": "original_width"},
        )
    ]
    for row in anchor_width_seeds:
        widths = {
            str(domain_id): int(index)
            for domain_id, index in dict(row.get("width_genes", {})).items()
        }
        if set(widths) != set(full_widths):
            continue
        proposals.append(
            LegalWidthGenotype(
                widths,
                {
                    group_id: str(
                        dict(row.get("precision_genes", {})).get(
                            group_id, default_precision[group_id]
                        )
                    )
                    for group_id in default_precision
                },
                {
                    "seed_family": "anchor_derived",
                    "anchor_id": str(row.get("anchor_id", "")),
                    "ranking_mode": str(row.get("ranking_mode", "")),
                },
            )
        )
    for domain in inventory.domains:
        full_index = full_widths[domain.domain_id]
        if full_index <= 0:
            continue
        widths = dict(full_widths)
        widths[domain.domain_id] = full_index - 1
        proposals.append(
            LegalWidthGenotype(
                widths,
                default_precision,
                {
                    "seed_family": "anchor_adjacent_width",
                    "seed_domain": domain.domain_id,
                },
            )
        )
    for group_id, actions in space.precision_action_space.items():
        for action in actions:
            if action == default_precision[group_id]:
                continue
            precision = dict(default_precision)
            precision[group_id] = action
            proposals.append(
                LegalWidthGenotype(
                    full_widths,
                    precision,
                    {
                        "seed_family": "precision_budget_seed",
                        "seed_precision_group": group_id,
                    },
                )
            )
    unique: dict[str, LegalWidthGenotype] = {}
    for candidate in proposals:
        unique.setdefault(candidate.genotype_hash, candidate)
    attempts = 0
    max_attempts = max(1000, int(size) * 500)
    while len(unique) < int(size) and attempts < max_attempts:
        attempts += 1
        candidate = random_legal_width_genotype(
            inventory,
            precision_actions=space.precision_action_space,
            rng=rng,
        )
        unique.setdefault(candidate.genotype_hash, candidate)
    if len(unique) < int(size):
        raise RuntimeError(
            f"legal_width_initial_population_supply_exhausted:{len(unique)}<{int(size)}"
        )
    return list(unique.values())[: int(size)]


def normalize_budget_intervals(
    search_config: Mapping[str, Any],
) -> list[tuple[float, float]]:
    target = float(search_config.get("target_bops_retention", 0.21))
    tolerance = float(search_config.get("bops_tolerance", 0.005))
    raw = list(search_config.get("budget_intervals", []) or [])
    raw.append([target - tolerance, target + tolerance])
    intervals = {
        (round(float(row[0]), 12), round(float(row[1]), 12))
        for row in raw
        if len(row) == 2 and float(row[0]) < float(row[1])
    }
    return sorted(intervals)


def _load_anchor_width_seeds(
    search_config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows = [dict(row) for row in search_config.get("anchor_width_seeds", []) or []]
    raw_path = str(search_config.get("anchor_structure_manifest", "")).strip()
    if not raw_path:
        return rows
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise RuntimeError(f"anchor_structure_manifest_missing:{path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    for structure in payload.get("evaluated_structures", payload.get("structures", [])):
        metadata = dict(structure.get("repair_metadata", {}) or {})
        width_genes = dict(metadata.get("width_genes", {}) or {})
        if not width_genes:
            continue
        rows.append(
            {
                "anchor_id": structure.get("anchor_id", ""),
                "ranking_mode": metadata.get("ranking_mode", ""),
                "width_genes": width_genes,
            }
        )
    return rows


def _precision_histogram(genotypes: Sequence[LegalWidthGenotype]) -> dict[str, int]:
    return dict(
        sorted(
            Counter(
                precision
                for genotype in genotypes
                for precision in genotype.precision_genes.values()
            ).items()
        )
    )


def run_legal_width_stage1_seeds(
    *,
    context: Any,
    proxy: Any,
    run_dir: str | Path,
    search_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Run independent real GA seeds and maintain one feasible archive."""

    destination = Path(run_dir)
    population_size = int(search_config.get("population_size", 64))
    initial_size = int(
        search_config.get("initial_population_size", population_size)
    )
    offspring_size = int(search_config.get("offspring_size", population_size))
    generations = int(
        search_config.get(
            "generations", search_config.get("generations_per_round", 30)
        )
    )
    seed_count = int(search_config.get("independent_seeds", 3))
    base_seed = int(search_config.get("seed", 42))
    target = float(search_config.get("target_bops_retention", 0.21))
    tolerance = float(search_config.get("bops_tolerance", 0.005))
    lower = target - tolerance
    upper = target + tolerance
    archive = FeasibleParetoArchive()
    repair_monitor = ExceptionOnlyRepairMonitor()
    records_by_hash: dict[str, ProxyCandidateRecord] = {}
    all_metric_rows: list[dict[str, Any]] = []
    generation_summaries: list[dict[str, Any]] = []
    total_evaluations = 0
    anchor_width_seeds = _load_anchor_width_seeds(search_config)

    for seed_offset in range(seed_count):
        seed = base_seed + seed_offset
        rng = random.Random(seed)
        initial = _initial_population(
            context.search_space,
            size=initial_size,
            rng=rng,
            anchor_width_seeds=anchor_width_seeds,
        )
        seed_dir = destination / f"ga_seed_{seed_offset:02d}_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        def evaluate_batch(
            genotypes: list[LegalWidthGenotype], generation: int
        ) -> BatchProxyResult:
            nonlocal total_evaluations
            for genotype in genotypes:
                repair_monitor.observe_normal_candidate(
                    genotype,
                    inventory=context.search_space.legal_width_inventory,
                    precision_actions=context.search_space.precision_action_space,
                )
            result = proxy.evaluate_batch(
                genotypes,
                generation=generation,
                outer_round=seed_offset,
            )
            total_evaluations += len(genotypes)
            metrics_rows: list[dict[str, Any]] = []
            for genotype, raw_metrics in zip(genotypes, result.metrics):
                metrics = dict(raw_metrics)
                bops = float(
                    metrics.get(
                        "R_bops_vs_fp32", metrics.get("R_bops", float("inf"))
                    )
                )
                violation = max(0.0, lower - bops, bops - upper)
                metrics["BOPS_target"] = target
                metrics["BOPS_legal_interval"] = [lower, upper]
                metrics["bops_violation"] = violation
                metrics["bops_feasible"] = violation <= 1.0e-12
                metrics["structure_legal"] = True
                metrics["precision_legal"] = True
                metrics["missing_mapping"] = 0
                metrics["finite_joint_proxy"] = math.isfinite(
                    float(metrics.get("L_joint_raw", float("nan")))
                )
                if violation > 0.0:
                    metrics["F1"] = (
                        1.0e6
                        + 1.0e3 * violation
                        + float(metrics.get("F1", 0.0))
                    )
                metrics_rows.append(metrics)
            return BatchProxyResult(metrics_rows, dict(result.stats))

        def on_generation(
            generation: int,
            scored: list[tuple[LegalWidthGenotype, float, dict[str, Any]]],
        ) -> None:
            generation_rows: list[dict[str, Any]] = []
            for genotype, score, metrics in scored:
                phenotype = canonicalize_legal_width_candidate(
                    genotype, context.search_space
                )
                deploy_hash = candidate_hash(phenotype, context.search_space)
                phenotype_hash = str(phenotype.metadata["phenotype_hash"])
                row = {
                    "seed_index": seed_offset,
                    "seed": seed,
                    "generation": generation,
                    "genotype_hash": genotype.genotype_hash,
                    "width_vector_hash": genotype.width_vector_hash,
                    "structure_hash": phenotype.metadata["structure_hash"],
                    "precision_hash": genotype.precision_hash,
                    "phenotype_hash": phenotype_hash,
                    "candidate_hash": deploy_hash,
                    "F1": float(score),
                    "J1": float(metrics.get("J1", -float(score))),
                    "S_task": float(metrics.get("S_task", 0.0)),
                    "L_joint_raw": float(metrics.get("L_joint_raw", float("nan"))),
                    "L_joint_first_order": float(
                        metrics.get("L_joint_first_order", float("nan"))
                    ),
                    "L_joint_second_order": float(
                        metrics.get("L_joint_second_order", float("nan"))
                    ),
                    "R_prune": float(metrics.get("R_prune", 0.0)),
                    "R_param": 1.0 - float(metrics.get("R_prune", 0.0)),
                    "R_BOPS": float(
                        metrics.get(
                            "R_bops_vs_fp32", metrics.get("R_bops", float("inf"))
                        )
                    ),
                    "budget_target": target,
                    "budget_lower": lower,
                    "budget_upper": upper,
                    "R_MAC": float(metrics.get("R_MAC", 1.0)),
                    "latency_proxy_value": float(metrics.get("R_MAC", 1.0)),
                    "latency_proxy_source": "MAC_retention_normalized_proxy",
                    "structure_legal": bool(metrics["structure_legal"]),
                    "precision_legal": bool(metrics["precision_legal"]),
                    "missing_mapping": int(metrics["missing_mapping"]),
                    "finite_joint_proxy": bool(metrics["finite_joint_proxy"]),
                    "bops_feasible": bool(metrics["bops_feasible"]),
                    "seed_family": str(genotype.meta.get("seed_family", "ga")),
                }
                generation_rows.append(row)
                all_metric_rows.append(row)
                records_by_hash.setdefault(
                    phenotype_hash,
                    ProxyCandidateRecord(
                        deploy_hash,
                        genotype,
                        phenotype,
                        float(score),
                        dict(metrics),
                    ),
                )
                if row["bops_feasible"]:
                    archive.add(row, active_budget=upper)
            scores = [float(row[1]) for row in scored]
            feasible = [row for row in generation_rows if row["bops_feasible"]]
            summary = {
                "seed_index": seed_offset,
                "seed": seed,
                "generation": generation,
                "candidate_count": len(scored),
                "unique_genotype_count": len(
                    {row["genotype_hash"] for row in generation_rows}
                ),
                "unique_structure_count": len(
                    {row["structure_hash"] for row in generation_rows}
                ),
                "unique_phenotype_count": len(
                    {row["phenotype_hash"] for row in generation_rows}
                ),
                "feasible_count": len(feasible),
                "BOPS_feasible_count": len(feasible),
                "repair_count": 0,
                "best_J1": max((row["J1"] for row in generation_rows), default=None),
                "median_J1": statistics.median(
                    row["J1"] for row in generation_rows
                ),
                "best_L_joint": min(
                    (row["L_joint_raw"] for row in generation_rows), default=None
                ),
                "param_retention_range": [
                    min(row["R_param"] for row in generation_rows),
                    max(row["R_param"] for row in generation_rows),
                ],
                "BOPS_retention_range": [
                    min(row["R_BOPS"] for row in generation_rows),
                    max(row["R_BOPS"] for row in generation_rows),
                ],
                "precision_action_histogram": _precision_histogram(
                    [row[0] for row in scored]
                ),
                "best_F1": min(scores) if scores else None,
            }
            generation_summaries.append(summary)
            _write_csv(
                seed_dir / f"generation_{generation:03d}.csv", generation_rows
            )
            _write_json(
                seed_dir / f"generation_{generation:03d}_summary.json", summary
            )

        engine = GeneticSearchEngine(
            context.search_space,
            GAConfig(
                initial_population_size=initial_size,
                population_size=population_size,
                offspring_size=offspring_size,
                num_generations=generations,
                elite_ratio=float(search_config.get("elite_ratio", 0.10)),
                crossover_rate=float(search_config.get("crossover_rate", 0.80)),
                prune_mutation_rate=float(
                    search_config.get("width_mutation_rate", 0.12)
                ),
                precision_mutation_rate=float(
                    search_config.get("precision_group_mutation_rate", 0.08)
                ),
                immigrant_ratio=float(search_config.get("immigrant_ratio", 0.08)),
                stagnation_generations=int(
                    search_config.get("stagnation_generations", 8)
                ),
                stagnation_immigrant_ratio=float(
                    search_config.get("stagnation_immigrant_ratio", 0.25)
                ),
                random_seed=seed,
            ),
        )
        engine.run(
            batch_evaluator=evaluate_batch,
            initial_population=initial,
            candidate_key_fn=lambda candidate: candidate.genotype_hash,
            generation_callback=on_generation,
        )

    repair_report = repair_monitor.report()
    summary = {
        "structure_gene_type": "legal_keep_width",
        "ga_seeds": seed_count,
        "ga_generations": generations,
        "ga_population_size": population_size,
        "total_proxy_evaluations": total_evaluations,
        "BOPS_target": target,
        "BOPS_legal_interval": [lower, upper],
        "generation_summaries": generation_summaries,
        "archive_summary": archive.summary(),
        "repair_report": repair_report,
        "anchor_width_seed_count": len(anchor_width_seeds),
    }
    _write_json(destination / "stage1_legal_width_summary.json", summary)
    _write_csv(destination / "stage1_all_candidates.csv", all_metric_rows)
    return {
        **summary,
        "archive": archive,
        "records_by_phenotype_hash": records_by_hash,
        "all_metric_rows": all_metric_rows,
    }


def run_legal_width_budget_sweep(
    *,
    context: Any,
    proxy: Any,
    run_dir: str | Path,
    search_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Run the primary seeds plus independently seeded BOPS bands."""

    destination = Path(run_dir)
    intervals = normalize_budget_intervals(search_config)
    primary_target = float(search_config.get("target_bops_retention", 0.21))
    primary_tolerance = float(search_config.get("bops_tolerance", 0.005))
    primary = (
        round(primary_target - primary_tolerance, 12),
        round(primary_target + primary_tolerance, 12),
    )
    minimum_per_band = int(
        search_config.get("minimum_unique_feasible_phenotypes_per_band", 5)
    )
    minimum_total = int(
        search_config.get("minimum_total_unique_feasible_phenotypes", 30)
    )
    minimum_structures = int(
        search_config.get("minimum_total_unique_physical_structures", 15)
    )
    band_seed_count = int(search_config.get("band_independent_seeds", 1))
    max_band_retries = int(search_config.get("max_band_retry_runs", 2))
    combined_archive = FeasibleParetoArchive()
    combined_records: dict[str, ProxyCandidateRecord] = {}
    all_rows: list[dict[str, Any]] = []
    band_reports = []
    total_evaluations = 0
    normal_candidate_count = 0
    repair_invocations = 0

    for band_index, (lower, upper) in enumerate(intervals):
        target = 0.5 * (lower + upper)
        tolerance = 0.5 * (upper - lower)
        is_primary = (lower, upper) == primary
        band_rows: dict[str, dict[str, Any]] = {}
        attempts = 1 if is_primary else max(1, max_band_retries + 1)
        run_reports = []
        for retry in range(attempts):
            config = dict(search_config)
            config.update(
                {
                    "target_bops_retention": target,
                    "bops_tolerance": tolerance,
                    "independent_seeds": (
                        int(search_config.get("independent_seeds", 3))
                        if is_primary and retry == 0
                        else band_seed_count
                    ),
                    "seed": int(search_config.get("seed", 42))
                    + band_index * 10000
                    + retry * 1000,
                    "budget_intervals": [],
                }
            )
            band_dir = (
                destination
                / "budget_bands"
                / f"band_{band_index:02d}_{lower:.6f}_{upper:.6f}"
                / f"retry_{retry:02d}"
            )
            result = run_legal_width_stage1_seeds(
                context=context,
                proxy=proxy,
                run_dir=band_dir,
                search_config=config,
            )
            run_reports.append(
                {
                    key: value
                    for key, value in result.items()
                    if key
                    not in {
                        "archive",
                        "records_by_phenotype_hash",
                        "all_metric_rows",
                    }
                }
            )
            total_evaluations += int(result["total_proxy_evaluations"])
            normal_candidate_count += int(
                result["repair_report"]["normal_candidate_count"]
            )
            repair_invocations += int(
                result["repair_report"]["repair_invocation_count"]
            )
            for row in result["archive"].records:
                phenotype_hash = str(row["phenotype_hash"])
                band_rows.setdefault(phenotype_hash, row)
                combined_archive.add(row, active_budget=upper)
            combined_records.update(result["records_by_phenotype_hash"])
            all_rows.extend(result["all_metric_rows"])
            if len(band_rows) >= minimum_per_band:
                break
        band_reports.append(
            {
                "band_index": band_index,
                "lower": lower,
                "upper": upper,
                "target": target,
                "primary": is_primary,
                "feasible_phenotype_count": len(band_rows),
                "minimum_required": minimum_per_band,
                "passed": len(band_rows) >= minimum_per_band,
                "runs": run_reports,
            }
        )

    archive_summary = combined_archive.summary()
    coverage_passed = (
        len(intervals) >= 5
        and all(row["passed"] for row in band_reports)
        and int(archive_summary["feasible_phenotype_count"]) >= minimum_total
        and int(archive_summary["unique_structure_count"]) >= minimum_structures
    )
    repair_report = {
        "normal_candidate_count": normal_candidate_count,
        "repair_invocation_count": repair_invocations,
        "repair_invocation_rate": (
            repair_invocations / max(normal_candidate_count, 1)
        ),
        "repair_changed_structure_count": 0,
        "repair_changed_precision_count": 0,
    }
    summary = {
        "structure_gene_type": "legal_keep_width",
        "budget_intervals": [list(row) for row in intervals],
        "budget_band_reports": band_reports,
        "budget_coverage_passed": coverage_passed,
        "total_proxy_evaluations": total_evaluations,
        "archive_summary": archive_summary,
        "repair_report": repair_report,
    }
    _write_json(destination / "legal_width_budget_sweep_summary.json", summary)
    _write_csv(destination / "legal_width_budget_sweep_candidates.csv", all_rows)
    return {
        **summary,
        "archive": combined_archive,
        "records_by_phenotype_hash": combined_records,
        "all_metric_rows": all_rows,
    }
