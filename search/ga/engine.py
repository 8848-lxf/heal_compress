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
    random_seed: int = 42


class GeneticSearchEngine:
    """Small discrete GA that evaluates repaired candidates via a callback."""

    def __init__(self, space: SearchSpaceSpec, config: GAConfig | None = None) -> None:
        self.space = space
        self.config = config or GAConfig()
        self.rng = random.Random(self.config.random_seed)

    def run(
        self,
        evaluator: Callable[[CandidateGenotype, int], dict[str, Any]] | None = None,
        *,
        batch_evaluator: Callable[[list[CandidateGenotype], int], Any] | None = None,
        previous_elite: list[CandidateGenotype] | None = None,
        previous_best: CandidateGenotype | None = None,
        seen_candidate_keys: Iterable[str] | None = None,
        candidate_key_fn: Callable[[CandidateGenotype], str] | None = None,
    ) -> list[tuple[CandidateGenotype, float, dict[str, Any]]]:
        seen_keys = {str(key) for key in (seen_candidate_keys or [])}

        def fresh_population(candidates: list[CandidateGenotype], target_size: int) -> list[CandidateGenotype]:
            if candidate_key_fn is None:
                return candidates[:target_size]
            fresh: list[CandidateGenotype] = []

            def try_add(candidate: CandidateGenotype) -> None:
                if len(fresh) >= target_size:
                    return
                key = str(candidate_key_fn(candidate))
                if key in seen_keys:
                    return
                seen_keys.add(key)
                fresh.append(candidate)

            for candidate in candidates:
                try_add(candidate)
            attempts = 0
            max_attempts = max(100, target_size * 100)
            while len(fresh) < target_size and attempts < max_attempts:
                attempts += 1
                for candidate in make_immigrants(self.space, 1, self.rng):
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
        )
        population = fresh_population(list(population), max(self.config.population_size, self.config.initial_population_size))
        best = float("inf")
        stagnant = 0
        all_scored: list[tuple[CandidateGenotype, float, dict[str, Any]]] = []
        elite_count = max(1, int(round(self.config.population_size * self.config.elite_ratio)))
        gene_count = len(self.space.pruning_unit_ids) + len(self.space.precision_gene_ids)
        for generation in range(self.config.num_generations):
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
            scored.sort(key=lambda row: row[1])
            if scored and scored[0][1] < best:
                best = scored[0][1]
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
                )
                next_population.append(child)
            next_population.extend(make_immigrants(self.space, immigrant_count, self.rng))
            population = dedupe_population(next_population)
            while len(population) < self.config.population_size:
                population.extend(make_immigrants(self.space, 1, self.rng))
            population = fresh_population(population, self.config.population_size)
        all_scored.sort(key=lambda row: row[1])
        return all_scored
