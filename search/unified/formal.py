"""Reusable strict Stage1/Stage2 V3 composition for every registered family."""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

from ..candidate import CandidateGenotype
from ..canonicalization import SearchSpaceSpec
from ..ga.strict_stage12_v3 import (
    Stage2Result,
    StrictGAConfig,
    StrictStage12V3Runner,
    UnifiedTaylorStage1Evaluator,
)
from .config import ResolvedSearchConfig
from .progress import SearchProgress


def build_strict_stage1(
    config: ResolvedSearchConfig,
    *,
    space: SearchSpaceSpec,
    baseline: CandidateGenotype,
    structure_proxy: Any,
    weight_proxy: Any,
    activation_cache: Any | None,
    bops_evaluator: Callable[[Any], Mapping[str, Any]],
    size_evaluator: Callable[[Any], Mapping[str, Any]],
    target: float,
) -> UnifiedTaylorStage1Evaluator:
    proxy = dict(config.payload.get("proxy", {}) or {})
    return UnifiedTaylorStage1Evaluator(
        space,
        baseline=baseline,
        structure_proxy=structure_proxy,
        weight_proxy=weight_proxy,
        activation_cache=activation_cache,
        bops_evaluator=bops_evaluator,
        size_evaluator=size_evaluator,
        target=float(target),
        tolerance_abs=float(proxy.get("bops_tolerance_abs", 0.005)),
        enforce_bops_hard_gate=True,
        include_activation_taylor=bool(
            proxy.get("include_activation_taylor", False)
        ),
    )


def run_strict_formal_ga(
    config: ResolvedSearchConfig,
    *,
    space: SearchSpaceSpec,
    initial_population: Sequence[CandidateGenotype],
    greedy_anchor: Stage2Result,
    stage1_evaluator: Callable[[CandidateGenotype], Mapping[str, Any]],
    stage2_evaluator: Callable[[CandidateGenotype, int], Stage2Result],
    target: float,
    progress: SearchProgress | None = None,
) -> dict[str, Any]:
    search = dict(config.payload.get("search", {}) or {})
    gate = dict(
        dict(config.payload.get("stage2", {}) or {}).get(
            "greedy_anchor_accuracy_gate", {}
        )
        or {}
    )
    generations = int(search.get("generations_per_round", 10))
    contract = "formal_gen5" if generations == 5 else "formal_gen10"
    policy = StrictGAConfig(
        target_bops_retention=float(target),
        tolerance_abs=float(
            dict(config.payload.get("proxy", {}) or {}).get(
                "bops_tolerance_abs", 0.005
            )
        ),
        population_size=int(search.get("population_size", 64)),
        offspring_size=int(search.get("offspring_size", 64)),
        generations=generations,
        stage2_new_candidate_quota=int(search.get("topk_stage2", 5)),
        random_seed=int(search.get("seed", 42)),
        stage2_accuracy_tolerance=float(gate.get("tolerance", 0.005)),
        generation_contract=contract,
    )
    reporter = progress or SearchProgress(
        enabled=bool(search.get("show_progress", True))
    )
    reporter.phase(1, policy.generations, "正式 GA 初始化与 Greedy 锚点注入")

    def on_generation(row: Mapping[str, Any]) -> None:
        generation = int(row.get("generation", 0))
        if generation > 0:
            reporter.phase(
                generation,
                policy.generations,
                f"Stage1 进化与 Stage2 新候选评估，第 {generation} 代",
            )

    result = StrictStage12V3Runner(
        space,
        policy,
        stage1_evaluator=stage1_evaluator,
        stage2_evaluator=stage2_evaluator,
    ).run(
        initial_population,
        greedy_anchor=greedy_anchor,
        generation_callback=on_generation,
    )
    return {
        **result,
        "family_id": config.family.family_id,
        "formal_protocol": "strict_stage12_v3",
    }


__all__ = ["build_strict_stage1", "run_strict_formal_ga"]
