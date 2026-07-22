"""Stage-1 proxy evaluator with cache integration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from ..cache.proxy_cache import ProxyCache
from ..candidate import CandidateGenotype
from ..canonicalization import SearchSpaceSpec, canonicalize_candidate
from ..hashing import candidate_hash
from ..proxy.objective import ProxyObjective


@dataclass
class BatchProxyResult:
    metrics: list[dict[str, Any]]
    stats: dict[str, Any] = field(default_factory=dict)


class Stage1ProxyEvaluator:
    def __init__(
        self,
        space: SearchSpaceSpec,
        objective: ProxyObjective | None = None,
        cache: ProxyCache | None = None,
        cache_key_fn: Callable[[Any, SearchSpaceSpec], str] | None = None,
        batch_scorer: Any | None = None,
        proxy_backend: str = "scalar_cpu",
        proxy_device: str = "cpu",
        proxy_batch_size: int = 1,
    ) -> None:
        self.space = space
        self.objective = objective or ProxyObjective()
        self.cache = cache
        self.cache_key_fn = cache_key_fn
        self.batch_scorer = batch_scorer
        self.proxy_backend = str(proxy_backend)
        self.proxy_device = str(proxy_device)
        self.proxy_batch_size = int(proxy_batch_size)
        self.scalar_evaluate_call_count = 0
        self.batch_evaluate_call_count = 0
        self.cache_hit_count = 0
        self.cache_miss_count = 0
        self.gpu_batch_count = 0
        self.initial_candidate_count = 0
        self.unique_phenotype_count = 0
        self.current_unique_phenotype_count = 0
        self._seen_cache_keys: set[str] = set()
        self.last_batch_stats: dict[str, Any] = {}

    def evaluate(self, genotype: CandidateGenotype, *, generation: int = 0, outer_round: int = 0) -> dict[str, Any]:
        self.scalar_evaluate_call_count += 1
        phenotype = canonicalize_candidate(genotype, self.space)
        deploy_key = candidate_hash(phenotype, self.space)
        key = self.cache_key_fn(phenotype, self.space) if self.cache_key_fn is not None else deploy_key
        self._seen_cache_keys.add(str(key))
        self.current_unique_phenotype_count = 1
        self.unique_phenotype_count = len(self._seen_cache_keys)

        def compute() -> dict[str, Any]:
            metrics = self.objective.evaluate(phenotype)
            return {
                **metrics,
                "candidate_hash": deploy_key,
                "proxy_cache_key": key,
                "generation": generation,
                "outer_round": outer_round,
                "phenotype": phenotype.to_dict(),
            }

        if self.cache is None:
            return compute()
        cached = self.cache.get(key)
        if cached is not None:
            self.cache_hit_count += 1
            return {**cached, "candidate_hash": deploy_key, "proxy_cache_key": key, "cache_hit": True}
        self.cache_miss_count += 1
        result = compute()
        self.cache.put(
            key,
            {
                k: v
                for k, v in result.items()
                if k not in {"candidate_hash", "proxy_cache_key", "phenotype"}
            },
        )
        return result

    def evaluate_batch(
        self,
        genotypes_or_phenotypes: Sequence[Any],
        *,
        generation: int = 0,
        outer_round: int = 0,
    ) -> BatchProxyResult:
        self.batch_evaluate_call_count += 1
        phenotypes = []
        deploy_keys = []
        cache_keys = []
        for row in genotypes_or_phenotypes:
            phenotype = row if hasattr(row, "pruned_unit_ids") and hasattr(row, "precision_profile") else canonicalize_candidate(row, self.space)
            deploy_key = candidate_hash(phenotype, self.space)
            key = self.cache_key_fn(phenotype, self.space) if self.cache_key_fn is not None else deploy_key
            phenotypes.append(phenotype)
            deploy_keys.append(deploy_key)
            cache_keys.append(key)
        current_unique_count = len(set(cache_keys))
        if self.initial_candidate_count == 0:
            self.initial_candidate_count = len(genotypes_or_phenotypes)
        self._seen_cache_keys.update(str(key) for key in cache_keys)
        self.unique_phenotype_count = len(self._seen_cache_keys)
        self.current_unique_phenotype_count = current_unique_count

        metrics_by_key: dict[str, dict[str, Any]] = {}
        miss_order: list[str] = []
        miss_phenotypes = []
        pending_miss_keys: set[str] = set()
        for key, phenotype in zip(cache_keys, phenotypes):
            if key in metrics_by_key or key in pending_miss_keys:
                continue
            cached = self.cache.get(key) if self.cache is not None else None
            if cached is not None:
                self.cache_hit_count += 1
                metrics_by_key[key] = {**cached, "cache_hit": True}
            else:
                self.cache_miss_count += 1
                pending_miss_keys.add(key)
                miss_order.append(key)
                miss_phenotypes.append(phenotype)

        batch_stats: dict[str, Any] = {}
        if miss_phenotypes:
            if self.batch_scorer is None:
                if self.proxy_backend != "scalar_cpu":
                    raise RuntimeError("gpu_proxy_required_but_not_active")
                computed = []
                for phenotype in miss_phenotypes:
                    computed.append(self.objective.evaluate(phenotype))
                batch_stats["proxy_backend"] = "scalar_cpu"
            else:
                result = self.batch_scorer.evaluate_batch(miss_phenotypes, generation=generation, outer_round=outer_round)
                computed = result.metrics
                batch_stats.update(result.stats)
                self.gpu_batch_count += int(result.stats.get("gpu_batch_count", 0) or 0)
            for key, phenotype, metrics in zip(miss_order, miss_phenotypes, computed):
                deploy_key = candidate_hash(phenotype, self.space)
                row = {
                    **metrics,
                    "candidate_hash": deploy_key,
                    "proxy_cache_key": key,
                    "generation": generation,
                    "outer_round": outer_round,
                    "phenotype": phenotype.to_dict(),
                    "cache_hit": False,
                    "proxy_backend": self.proxy_backend,
                    "proxy_device": self.proxy_device,
                }
                metrics_by_key[key] = row
                if self.cache is not None:
                    # The key already commits to the exact canonical
                    # phenotype. Persisting thousands of atomic unit ids in
                    # every cache row makes domain-width searches grow by
                    # gigabytes without adding resume information. The
                    # caller's current phenotype is restored below on hits.
                    self.cache.put(
                        key,
                        {
                            k: v
                            for k, v in row.items()
                            if k
                            not in {
                                "candidate_hash",
                                "proxy_cache_key",
                                "phenotype",
                            }
                        },
                    )

        final = []
        for deploy_key, key, phenotype in zip(deploy_keys, cache_keys, phenotypes):
            row = dict(metrics_by_key[key])
            row["candidate_hash"] = deploy_key
            row["proxy_cache_key"] = key
            row["generation"] = generation
            row["outer_round"] = outer_round
            row.setdefault("phenotype", phenotype.to_dict())
            row.setdefault("proxy_backend", self.proxy_backend)
            row.setdefault("proxy_device", self.proxy_device)
            final.append(row)
        stats = {
            **batch_stats,
            "proxy_backend": self.proxy_backend,
            "proxy_device": self.proxy_device,
            "proxy_batch_size": self.proxy_batch_size,
            "initial_candidate_count": self.initial_candidate_count,
            "batch_candidate_count": len(genotypes_or_phenotypes),
            "unique_phenotype_count": self.unique_phenotype_count,
            "current_unique_phenotype_count": self.current_unique_phenotype_count,
            "cache_hit_count": self.cache_hit_count,
            "cache_miss_count": self.cache_miss_count,
            "gpu_batch_count": self.gpu_batch_count,
            "scalar_evaluate_call_count": self.scalar_evaluate_call_count,
            "batch_evaluate_call_count": self.batch_evaluate_call_count,
        }
        merged_stats = dict(self.last_batch_stats)
        for key, value in stats.items():
            if key in {"cuda_event_elapsed_ms", "gpu_peak_memory_bytes", "candidates_per_second"}:
                if float(value or 0.0) > 0.0:
                    merged_stats[key] = value
                else:
                    merged_stats.setdefault(key, value)
            else:
                merged_stats[key] = value
        self.last_batch_stats = merged_stats
        return BatchProxyResult(metrics=final, stats=stats)
