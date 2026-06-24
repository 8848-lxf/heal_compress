"""Genetic algorithm search engine for pruning-quantization co-optimization.

Drives the evolutionary search over the joint pruning + quantization space,
using proxy objectives for fast candidate evaluation.
"""

from __future__ import annotations

import logging
import random
import time
from pathlib import Path
from typing import Any

from .search_space import CandidateEncoding, WEIGHT_BITS
from .proxy_objective import ProxyObjectiveEvaluator
from ..utils.io_utils import ensure_dir, save_json, save_yaml

logger = logging.getLogger(__name__)


class SearchConstraints:
    """Constraint repair and feasibility checking for candidate encodings.

    Ensures:
    - Protected groups are always kept.
    - Minimum channel ratio per stage.
    - Bit-widths are valid candidates.
    - Channel alignment requirements.

    Args:
        protected_group_ids: Set of group IDs that cannot be pruned.
        weight_bits: Valid bit-width candidates.
        min_channels: Minimum channels per layer (default 8).
        min_stage_ratio: Minimum retention ratio per stage (default 0.25).
        channel_alignment: Channel alignment base (default 8).
    """

    def __init__(
        self,
        protected_group_ids: set[str],
        weight_bits: list[str] | None = None,
        min_channels: int = 8,
        min_stage_ratio: float = 0.25,
        channel_alignment: int = 8,
    ):
        self.protected_group_ids = protected_group_ids
        self.weight_bits = weight_bits or list(WEIGHT_BITS)
        self.min_channels = min_channels
        self.min_stage_ratio = min_stage_ratio
        self.channel_alignment = channel_alignment

    def repair(self, candidate: CandidateEncoding) -> CandidateEncoding:
        """Repair a candidate to satisfy all constraints.

        Steps:
        1. Force protected groups to keep=1.
        2. Ensure minimum retention ratio.
        3. Snap bit-widths to valid candidates.

        Args:
            candidate: Candidate to repair (modified in-place).

        Returns:
            The repaired candidate.
        """
        repaired = candidate.clone()

        # Protect groups
        for gid in self.protected_group_ids:
            if gid in repaired.prune_vars:
                repaired.prune_vars[gid] = 1

        # Ensure minimum retention
        total = max(len(repaired.prune_vars), 1)
        keep_min = max(1, int(total * self.min_stage_ratio))
        kept = [gid for gid, v in repaired.prune_vars.items() if v == 1]
        if len(kept) < keep_min:
            pruned = [gid for gid, v in repaired.prune_vars.items()
                       if v == 0 and gid not in self.protected_group_ids]
            for gid in pruned[:keep_min - len(kept)]:
                repaired.prune_vars[gid] = 1

        # Snap bit-widths
        for layer, bit in list(repaired.bitwidth_vars.items()):
            if bit not in self.weight_bits:
                repaired.bitwidth_vars[layer] = self.weight_bits[0]

        return repaired


class GeneticSearchEngine:
    """Genetic algorithm for joint pruning-quantization search.

    Evolves a population of candidate subnet encodings to minimize the
    joint proxy objective F(C).

    Args:
        group_ids: All coupled channel group IDs.
        layer_names: All quantizable layer names.
        evaluator: Proxy objective evaluator.
        constraints: Search constraints for repair.
        pop_size: Population size per generation.
        max_generations: Maximum number of generations.
        elite_size: Number of elite candidates to preserve.
        mutation_rate: Probability of mutating each variable.
        patience: Early stopping patience (generations without improvement).
        weight_bits: Candidate bit-width labels.
        output_dir: Directory for search outputs.
    """

    def __init__(
        self,
        group_ids: list[str],
        layer_names: list[str],
        evaluator: ProxyObjectiveEvaluator,
        constraints: SearchConstraints,
        pop_size: int = 50,
        max_generations: int = 100,
        elite_size: int = 5,
        mutation_rate: float = 0.1,
        patience: int = 20,
        weight_bits: list[str] | None = None,
        output_dir: str = "search_results",
    ):
        self.group_ids = group_ids
        self.layer_names = layer_names
        self.evaluator = evaluator
        self.constraints = constraints
        self.pop_size = pop_size
        self.max_generations = max_generations
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.patience = patience
        self.weight_bits = weight_bits or list(WEIGHT_BITS)
        self.output_dir = ensure_dir(output_dir)

    def run(self, topk: int = 5) -> tuple[CandidateEncoding, float]:
        """Execute the genetic algorithm search.

        Steps per generation:
        1. Evaluate all candidates via proxy objective.
        2. Select elite candidates.
        3. Tournament selection for parents.
        4. Uniform crossover + mutation.
        5. Constraint repair.
        6. Check early stopping.

        Args:
            topk: Number of top candidates to save.

        Returns:
            Tuple of (best_candidate, best_score).
        """
        start = time.time()
        logger.info(
            f"[GA] Starting search: pop={self.pop_size} gens={self.max_generations} "
            f"groups={len(self.group_ids)} layers={len(self.layer_names)}"
        )

        population = [self._random_candidate() for _ in range(self.pop_size)]
        all_scored: list[tuple[CandidateEncoding, float]] = []
        history: list[dict] = []
        best_score = float("inf")
        stale_count = 0

        for gen in range(self.max_generations):
            scored: list[tuple[CandidateEncoding, float, dict]] = []

            for idx, cand in enumerate(population):
                metrics = self.evaluator.evaluate(cand)
                score = float(metrics["score"])
                scored.append((cand, score, metrics))
                all_scored.append((cand.clone(), score))
                history.append({
                    "generation": gen, "index": idx,
                    "score": score, "metrics": metrics,
                })

            scored.sort(key=lambda x: x[1])
            gen_best = scored[0][1]
            feasible_count = sum(1 for _, _, m in scored if m.get("feasible"))

            if gen_best < best_score:
                best_score = gen_best
                stale_count = 0
                logger.info(
                    f"[GA] gen {gen+1}: new best score={best_score:.6g} "
                    f"feasible={feasible_count}/{self.pop_size}"
                )
            else:
                stale_count += 1

            if stale_count >= self.patience:
                logger.info(f"[GA] Early stopping at gen {gen+1} (patience={self.patience})")
                break

            # Save progress
            save_json(
                {"generation": gen, "best_score": gen_best,
                 "global_best": best_score, "feasible": feasible_count},
                str(self.output_dir / "ga_progress.json"),
            )

            # Build next generation
            next_pop = [c.clone() for c, _, _ in scored[:self.elite_size]]
            while len(next_pop) < self.pop_size:
                a = self._tournament_select(scored)
                b = self._tournament_select(scored)
                child = self._crossover(a, b)
                child = self._mutate(child)
                child = self.constraints.repair(child)
                next_pop.append(child)
            population = next_pop

        # Save results
        all_scored.sort(key=lambda x: x[1])
        best_cand, best_score = all_scored[0]

        self._save_results(best_cand, best_score, all_scored[:topk], history)

        elapsed = time.time() - start
        logger.info(
            f"[GA] Finished: best_score={best_score:.6g} elapsed={elapsed:.1f}s"
        )
        return best_cand, best_score

    def _random_candidate(self) -> CandidateEncoding:
        """Generate a random candidate and repair it."""
        cand = CandidateEncoding.random(
            self.group_ids, self.layer_names, self.weight_bits,
            protected_group_ids=self.constraints.protected_group_ids,
        )
        return self.constraints.repair(cand)

    def _tournament_select(
        self,
        scored: list[tuple[CandidateEncoding, float, dict]],
        k: int = 3,
    ) -> CandidateEncoding:
        """Tournament selection.

        Args:
            scored: List of (candidate, score, metrics) tuples.
            k: Tournament size.

        Returns:
            Selected candidate (cloned).
        """
        k = min(k, len(scored))
        tournament = random.sample(scored, k)
        return min(tournament, key=lambda x: x[1])[0].clone()

    def _crossover(
        self, a: CandidateEncoding, b: CandidateEncoding,
    ) -> CandidateEncoding:
        """Uniform crossover between two parents.

        Args:
            a: First parent.
            b: Second parent.

        Returns:
            Child candidate.
        """
        child = a.clone()
        for gid in self.group_ids:
            if random.random() < 0.5:
                child.prune_vars[gid] = b.prune_vars.get(gid, 1)
        for layer in self.layer_names:
            if random.random() < 0.5:
                child.bitwidth_vars[layer] = b.bitwidth_vars.get(layer, "FP16")
        child.meta["created_by"] = "crossover"
        return child

    def _mutate(self, cand: CandidateEncoding) -> CandidateEncoding:
        """Mutate a candidate by randomly flipping prune vars and bit-widths.

        Args:
            cand: Candidate to mutate.

        Returns:
            Mutated candidate.
        """
        out = cand.clone()
        for gid in self.group_ids:
            if gid in self.constraints.protected_group_ids:
                out.prune_vars[gid] = 1
                continue
            if random.random() < self.mutation_rate:
                out.prune_vars[gid] = 1 - out.prune_vars.get(gid, 1)
        for layer in self.layer_names:
            if random.random() < self.mutation_rate:
                # Pick an adjacent bit-width
                current = out.bitwidth_vars.get(layer, "FP16")
                candidates = [b for b in self.weight_bits if b != current]
                if candidates:
                    out.bitwidth_vars[layer] = random.choice(candidates)
        out.meta["created_by"] = "mutation"
        return out

    def _save_results(
        self,
        best: CandidateEncoding,
        best_score: float,
        topk: list[tuple[CandidateEncoding, float]],
        history: list[dict],
    ) -> None:
        """Save search results to files.

        Outputs:
        - best_subnet_config.json
        - topk_subnets.json
        - ga_history.json
        """
        save_json(
            {"best_score": best_score, **best.to_dict()},
            str(self.output_dir / "best_subnet_config.json"),
        )
        save_json(
            [{"score": s, **c.to_dict()} for c, s in topk],
            str(self.output_dir / "topk_subnets.json"),
        )
        save_json(
            {"history": history, "best_score": best_score},
            str(self.output_dir / "ga_history.json"),
        )
