"""Constraint-aware discrete GA engine."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from ..candidate import CandidateGenotype
from ..canonicalization import SearchSpaceSpec
from .crossover import block_crossover
from .diversity import average_hamming_distance
from .immigrants import immigrant_ratio_for_generation, make_immigrants
from .initialization import initialize_population
from .mutation import adapt_mutation_rate, mutate_candidate
from .population import dedupe_population
from .ranking import rank_constraint_first
from .selection import tournament_select


@dataclass(frozen=True)
class GAConfig:
    initial_population_size: int = 100
    population_size: int = 100
    offspring_size: int = 100
    num_generations: int = 30
    elite_ratio: float = 0.10
    crossover_rate: float = 0.80
    prune_mutation_rate: float = 0.03
    precision_mutation_rate: float = 0.08
    immigrant_ratio: float = 0.08
    stagnation_generations: int = 8
    stagnation_immigrant_ratio: float = 0.25
    preserve_evaluated_elites: bool = False
    mutation_action_min: int = 1
    mutation_action_max: int = 2
    early_stop_patience: int = 0
    minimum_generations: int = 1
    constraint_first_ranking: bool = False
    taylor_relative_epsilon: float = 0.05
    taylor_absolute_epsilon: float = 1.0e-8
    seeded_initial_population_ratio: float = 0.90
    random_seed: int = 42
    show_progress: bool = False
    progress_description: str = "Stage1 GA 世代"


class GeneticSearchEngine:
    """Small discrete GA that evaluates repaired candidates via a callback."""

    def __init__(self, space: SearchSpaceSpec, config: GAConfig | None = None) -> None:
        self.space = space
        self.config = config or GAConfig()
        self.rng = random.Random(self.config.random_seed)
        self.generation_statistics: list[dict[str, Any]] = []

    def run(
        self,
        evaluator: Callable[[CandidateGenotype, int], dict[str, Any]] | None = None,
        *,
        batch_evaluator: Callable[[list[CandidateGenotype], int], Any] | None = None,
        previous_elite: list[CandidateGenotype] | None = None,
        previous_best: CandidateGenotype | None = None,
        seed_candidates: list[CandidateGenotype] | None = None,
        seen_candidate_keys: Iterable[str] | None = None,
        candidate_key_fn: Callable[[CandidateGenotype], str] | None = None,
    ) -> list[tuple[CandidateGenotype, float, dict[str, Any]]]:
        # Seen hashes are cache/archive evidence, not an exclusion set.  The
        # evaluator owns cache reuse; GA population construction only removes
        # duplicates inside the current population.
        _seen_cache_evidence = {str(key) for key in (seen_candidate_keys or [])}
        immigrant_seeds = list(seed_candidates or previous_elite or [])
        immigrant_generation_counter = [0]

        def generate_immigrants(count: int) -> list[CandidateGenotype]:
            if not immigrant_seeds:
                return make_immigrants(self.space, count, self.rng)
            rows: list[CandidateGenotype] = []
            for _index in range(max(0, int(count))):
                index = immigrant_generation_counter[0]
                immigrant_generation_counter[0] += 1
                base = self.rng.choice(immigrant_seeds)
                rows.append(
                    mutate_candidate(
                        base,
                        self.space,
                        self.rng,
                        prune_mutation_rate=1.0,
                        precision_mutation_rate=1.0,
                        action_count=1 + (index % 4),
                        adjacent_precision=True,
                    )
                )
            return rows

        def fresh_population(
            candidates: list[CandidateGenotype],
            target_size: int,
        ) -> list[CandidateGenotype]:
            if candidate_key_fn is None:
                return candidates[:target_size]
            fresh: list[CandidateGenotype] = []
            local_keys: set[str] = set()

            def try_add(candidate: CandidateGenotype) -> None:
                if len(fresh) >= target_size:
                    return
                key = str(candidate_key_fn(candidate))
                if key in local_keys:
                    return
                local_keys.add(key)
                fresh.append(candidate)

            for candidate in candidates:
                try_add(candidate)
            attempts = 0
            max_attempts = max(100, target_size * 100)
            while len(fresh) < target_size and attempts < max_attempts:
                attempts += 1
                for candidate in generate_immigrants(1):
                    try_add(candidate)
                    if len(fresh) >= target_size:
                        break
            if len(fresh) < target_size:
                raise RuntimeError(f"unable_to_generate_unseen_population:{len(fresh)}<{target_size}")
            return fresh

        population = initialize_population(
            self.space,
            max(self.config.population_size, self.config.initial_population_size),
            self.rng,
            previous_elite=previous_elite,
            previous_best=previous_best,
            seed_candidates=seed_candidates,
            seeded_population_ratio=self.config.seeded_initial_population_ratio,
        )
        population = fresh_population(
            list(population),
            max(self.config.population_size, self.config.initial_population_size),
        )
        best_key: tuple[Any, ...] | None = None
        stagnant = 0
        all_scored: list[tuple[CandidateGenotype, float, dict[str, Any]]] = []
        elite_count = max(1, int(round(self.config.population_size * self.config.elite_ratio)))
        gene_count = len(self.space.pruning_gene_ids) + len(self.space.precision_gene_ids)
        self.generation_statistics = []
        generations: Iterable[int] = range(self.config.num_generations)
        if self.config.show_progress:
            try:
                from tqdm.auto import tqdm
            except ImportError:
                pass
            else:
                generations = tqdm(
                    generations,
                    total=self.config.num_generations,
                    desc=self.config.progress_description,
                    unit="代",
                )
        for generation in generations:
            scored = []
            if batch_evaluator is not None:
                batch_result = batch_evaluator(list(population), generation)
                batch_metrics = batch_result.metrics if hasattr(batch_result, "metrics") else list(batch_result)
                if len(batch_metrics) != len(population):
                    raise RuntimeError(f"batch_evaluator_length_mismatch:{len(batch_metrics)}!={len(population)}")
                for candidate, metrics in zip(population, batch_metrics):
                    score = float(metrics.get("F1", metrics.get("score", float("inf"))))
                    scored.append((candidate, score, metrics))
                    all_scored.append((candidate, score, metrics))
            else:
                if evaluator is None:
                    raise RuntimeError("scalar_or_batch_evaluator_required")
                for candidate in population:
                    metrics = evaluator(candidate, generation)
                    score = float(metrics.get("F1", metrics.get("score", float("inf"))))
                    scored.append((candidate, score, metrics))
                    all_scored.append((candidate, score, metrics))
            if self.config.constraint_first_ranking:
                scored = rank_constraint_first(
                    scored,
                    taylor_relative_epsilon=self.config.taylor_relative_epsilon,
                    taylor_absolute_epsilon=self.config.taylor_absolute_epsilon,
                )
            else:
                scored.sort(key=lambda row: row[1])
            for rank, (_candidate, _score, metrics) in enumerate(scored):
                metrics["ga_selection_rank"] = int(rank)
            canonical_ids = {
                str(metrics.get("candidate_hash", ""))
                for _candidate, _score, metrics in scored
                if str(metrics.get("candidate_hash", ""))
            }
            raw_ids = {
                str(candidate_key_fn(candidate))
                if candidate_key_fn is not None
                else repr(candidate.to_dict())
                for candidate, _score, _metrics in scored
            }
            self.generation_statistics.append(
                {
                    "generation": int(generation),
                    "population_count": len(scored),
                    "raw_genotype_count": len(raw_ids),
                    "canonical_phenotype_count": len(canonical_ids),
                    "canonicalization_collision_count": max(
                        0, len(raw_ids) - len(canonical_ids)
                    ),
                    "cache_hit_count": sum(
                        bool(metrics.get("cache_hit", False))
                        for _candidate, _score, metrics in scored
                    ),
                }
            )
            current_key = None
            if scored:
                current_key = (
                    not bool(scored[0][2].get("bops_feasible", True)),
                    float(scored[0][2].get("bops_violation", 0.0)),
                    float(
                        scored[0][2].get(
                            "L_joint_weight_activation_taylor",
                            scored[0][2].get(
                                "L_joint_weight_taylor", scored[0][1]
                            ),
                        )
                    ),
                    float(
                        scored[0][2].get(
                            "latency_proxy_ms",
                            scored[0][2].get("R_latency_proxy", float("inf")),
                        )
                        or float("inf")
                    ),
                    float(scored[0][2].get("R_parameter_retention", 1.0)),
                )
            if current_key is not None and (best_key is None or current_key < best_key):
                best_key = current_key
                stagnant = 0
            else:
                stagnant += 1
            diversity = average_hamming_distance([row[0] for row in scored])
            prune_rate = adapt_mutation_rate(self.config.prune_mutation_rate, diversity, gene_count)
            precision_rate = adapt_mutation_rate(self.config.precision_mutation_rate, diversity, gene_count)
            immigrant_ratio = immigrant_ratio_for_generation(
                stagnant,
                base_ratio=self.config.immigrant_ratio,
                stagnation_generations=self.config.stagnation_generations,
                stagnant_ratio=self.config.stagnation_immigrant_ratio,
            )
            next_population = [row[0] for row in scored[:elite_count]]
            immigrant_count = max(1, int(round(self.config.population_size * immigrant_ratio)))
            target_children = max(0, self.config.population_size - elite_count - immigrant_count)
            while len(next_population) < elite_count + target_children:
                left = tournament_select(scored, self.rng)
                right = tournament_select(scored, self.rng)
                child = block_crossover(left, right, self.space, self.rng) if self.rng.random() < self.config.crossover_rate else left
                child = mutate_candidate(
                    child,
                    self.space,
                    self.rng,
                    prune_mutation_rate=prune_rate,
                    precision_mutation_rate=precision_rate,
                    action_count=self.rng.randint(
                        max(1, int(self.config.mutation_action_min)),
                        max(
                            max(1, int(self.config.mutation_action_min)),
                            int(self.config.mutation_action_max),
                        ),
                    ),
                    adjacent_precision=True,
                )
                next_population.append(child)
            next_population.extend(generate_immigrants(immigrant_count))
            population = dedupe_population(next_population)
            while len(population) < self.config.population_size:
                population.extend(generate_immigrants(1))
            population = fresh_population(
                population,
                self.config.population_size,
            )
            if (
                int(self.config.early_stop_patience) > 0
                and generation + 1 >= int(self.config.minimum_generations)
                and stagnant >= int(self.config.early_stop_patience)
            ):
                break
        if self.config.constraint_first_ranking:
            all_scored = rank_constraint_first(
                all_scored,
                taylor_relative_epsilon=self.config.taylor_relative_epsilon,
                taylor_absolute_epsilon=self.config.taylor_absolute_epsilon,
            )
        else:
            all_scored.sort(key=lambda row: row[1])
        return all_scored
