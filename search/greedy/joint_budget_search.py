"""Bounded deterministic greedy search for one BOPS retention target."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

from ..admission.bops_band import (
    BopsBandPolicy,
    classify_bops_value,
    select_bops_candidates,
)
from ..encoding.legal_width_genotype import LegalWidthGenotype
from .legal_actions import GreedyAction


Evaluator = Callable[[LegalWidthGenotype], Mapping[str, Any]]
BatchEvaluator = Callable[
    [Sequence[LegalWidthGenotype]], Sequence[Mapping[str, Any]]
]
SuccessorEnumerator = Callable[
    [LegalWidthGenotype],
    Sequence[tuple[GreedyAction, LegalWidthGenotype]],
]


@dataclass(frozen=True)
class GreedySearchState:
    genotype: LegalWidthGenotype
    metrics: Mapping[str, Any]
    parent_hash: str = ""
    action_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))

    @property
    def genotype_hash(self) -> str:
        return self.genotype.genotype_hash

    def to_dict(self) -> dict[str, Any]:
        return {
            "genotype": self.genotype.to_dict(),
            "genotype_hash": self.genotype_hash,
            "metrics": dict(self.metrics),
            "parent_hash": self.parent_hash,
            "action_id": self.action_id,
        }


@dataclass(frozen=True)
class GreedyBudgetResult:
    target: float
    status: str
    terminal: GreedySearchState | None
    admission_mode: str
    accepted_path_states: tuple[GreedySearchState, ...]
    expansion_count: int
    failure_reason: str = ""
    evaluated_state_count: int = 0
    evaluated_states: tuple[GreedySearchState, ...] = ()
    rejection_counts: Mapping[str, int] = field(default_factory=dict)
    bops_funnel: Mapping[str, int] = field(default_factory=dict)
    nearest_misses: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "rejection_counts", MappingProxyType(dict(self.rejection_counts))
        )
        object.__setattr__(self, "bops_funnel", MappingProxyType(dict(self.bops_funnel)))
        object.__setattr__(
            self,
            "nearest_misses",
            tuple(MappingProxyType(dict(row)) for row in self.nearest_misses),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "status": self.status,
            "terminal": self.terminal.to_dict() if self.terminal else None,
            "admission_mode": self.admission_mode,
            "accepted_path_states": [
                state.to_dict() for state in self.accepted_path_states
            ],
            "expansion_count": self.expansion_count,
            "failure_reason": self.failure_reason,
            "evaluated_state_count": self.evaluated_state_count,
            "evaluated_states": [state.to_dict() for state in self.evaluated_states],
            "rejection_counts": dict(self.rejection_counts),
            "bops_funnel": dict(self.bops_funnel),
            "nearest_misses": [dict(row) for row in self.nearest_misses],
        }


def _metric_value(metrics: Mapping[str, Any], *keys: str) -> float:
    for key in keys:
        if key in metrics:
            return float(metrics[key])
    return float("nan")


def _canonical_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(metrics)
    bops = _metric_value(row, "R_BOPS", "R_bops_vs_fp32", "R_bops")
    loss = _metric_value(row, "L_joint_raw")
    prune = _metric_value(row, "R_prune")
    if not math.isfinite(bops) or not 0.0 <= bops <= 1.0:
        raise ValueError("greedy_state_bops_invalid")
    if not math.isfinite(loss) or loss < 0.0:
        raise ValueError("greedy_state_joint_loss_invalid")
    if not math.isfinite(prune) or not 0.0 <= prune <= 1.0:
        raise ValueError("greedy_state_prune_rate_invalid")
    row.update({"R_BOPS": bops, "L_joint_raw": loss, "R_prune": prune})
    return row


def _transition_key(
    parent: GreedySearchState,
    child_metrics: Mapping[str, Any],
    action: GreedyAction,
    child_hash: str,
) -> tuple[Any, ...]:
    bops_saved = float(parent.metrics["R_BOPS"]) - float(child_metrics["R_BOPS"])
    if not math.isfinite(bops_saved) or bops_saved <= 0.0:
        raise ValueError("greedy_action_nonpositive_bops_saving")
    loss_delta = float(child_metrics["L_joint_raw"]) - float(
        parent.metrics["L_joint_raw"]
    )
    if not math.isfinite(loss_delta):
        raise ValueError("greedy_action_joint_loss_delta_nonfinite")
    prune_delta = float(child_metrics["R_prune"]) - float(
        parent.metrics["R_prune"]
    )
    rho = float(loss_delta / bops_saved)
    if not math.isfinite(rho):
        raise ValueError("greedy_action_rho_nonfinite")
    return (
        round(rho, 12),
        -round(prune_delta, 12),
        action.action_id,
        child_hash,
    )


def _terminal_key(state: GreedySearchState, target: float) -> tuple[Any, ...]:
    return (
        float(state.metrics["L_joint_raw"]),
        -float(state.metrics["R_prune"]),
        abs(float(state.metrics["R_BOPS"]) - float(target)),
        state.genotype_hash,
    )


def _accepted_path(
    terminal: GreedySearchState,
    states: Mapping[str, GreedySearchState],
) -> tuple[GreedySearchState, ...]:
    path: list[GreedySearchState] = []
    current = terminal
    seen: set[str] = set()
    while True:
        if current.genotype_hash in seen:
            raise RuntimeError("greedy_path_parent_cycle")
        seen.add(current.genotype_hash)
        path.append(current)
        if not current.parent_hash:
            break
        current = states[current.parent_hash]
    return tuple(reversed(path))


def run_targeted_greedy(
    *,
    initial_genotype: LegalWidthGenotype,
    evaluate: Evaluator,
    evaluate_batch: BatchEvaluator | None = None,
    enumerate_successors: SuccessorEnumerator,
    policy: BopsBandPolicy,
    frontier_size: int = 8,
    max_expansions: int = 4096,
) -> GreedyBudgetResult:
    """Search one target without relaxing structure or deployment legality."""

    width = int(frontier_size)
    limit = int(max_expansions)
    if width <= 0:
        raise ValueError("greedy_frontier_size_must_be_positive")
    if limit <= 0:
        raise ValueError("greedy_max_expansions_must_be_positive")

    initial_metrics = _canonical_metrics(evaluate(initial_genotype))
    initial = GreedySearchState(initial_genotype, initial_metrics)
    metrics_cache: dict[str, Mapping[str, Any]] = {
        initial.genotype_hash: initial.metrics
    }
    attempted_hashes = {initial.genotype_hash}
    states: dict[str, GreedySearchState] = {initial.genotype_hash: initial}
    evaluation_order = [initial.genotype_hash]
    frontier = [initial]
    expanded_hashes: set[str] = set()
    primary_terminals: dict[str, GreedySearchState] = {}
    expanded_terminals: dict[str, GreedySearchState] = {}
    rejection_counts: Counter[str] = Counter()

    def record_terminal(state: GreedySearchState) -> None:
        classification = classify_bops_value(
            state.metrics["R_BOPS"], policy=policy
        )
        if classification["classification"] == "primary":
            primary_terminals[state.genotype_hash] = state
        elif classification["eligible_for_expanded"]:
            expanded_terminals[state.genotype_hash] = state

    record_terminal(initial)
    while frontier and len(metrics_cache) < limit:
        raw_proposals: list[
            tuple[GreedySearchState, GreedyAction, LegalWidthGenotype]
        ] = []
        proposals: dict[
            str,
            tuple[tuple[Any, ...], GreedySearchState, GreedyAction, LegalWidthGenotype],
        ] = {}
        for parent in sorted(frontier, key=lambda row: row.genotype_hash):
            if parent.genotype_hash in expanded_hashes:
                rejection_counts["state_already_expanded"] += 1
                continue
            expanded_hashes.add(parent.genotype_hash)
            parent_bops = float(parent.metrics["R_BOPS"])
            # Compression-only actions cannot recover after crossing below the
            # primary lower boundary. Expanded-high states may still reach it.
            if parent_bops <= policy.target + policy.primary_tolerance + 1.0e-12:
                continue
            successors = sorted(
                enumerate_successors(parent.genotype),
                key=lambda row: row[0].action_id,
            )
            for action, child in successors:
                child_hash = child.genotype_hash
                if child_hash == parent.genotype_hash:
                    rejection_counts["identity_action"] += 1
                    continue
                raw_proposals.append((parent, action, child))

        pending: dict[str, LegalWidthGenotype] = {}
        available = max(0, limit - len(metrics_cache))
        for _parent, _action, child in raw_proposals:
            child_hash = child.genotype_hash
            if child_hash in metrics_cache or child_hash in pending:
                continue
            if child_hash in attempted_hashes:
                rejection_counts["duplicate_genotype"] += 1
                continue
            if len(pending) >= available:
                rejection_counts["max_expansions_reached"] += 1
                continue
            attempted_hashes.add(child_hash)
            pending[child_hash] = child

        if pending:
            pending_items = list(pending.items())
            if evaluate_batch is None:
                metric_rows: Sequence[Mapping[str, Any] | Exception]
                scalar_rows: list[Mapping[str, Any] | Exception] = []
                for _child_hash, child in pending_items:
                    try:
                        scalar_rows.append(evaluate(child))
                    except Exception as exc:  # noqa: BLE001
                        scalar_rows.append(exc)
                metric_rows = scalar_rows
            else:
                try:
                    metric_rows = list(
                        evaluate_batch([child for _child_hash, child in pending_items])
                    )
                except Exception as exc:  # noqa: BLE001
                    raise RuntimeError("greedy_batch_evaluation_failed") from exc
                if len(metric_rows) != len(pending_items):
                    raise RuntimeError(
                        "greedy_batch_evaluation_count_mismatch:"
                        f"{len(metric_rows)}!={len(pending_items)}"
                    )
            for (child_hash, _child), metrics in zip(pending_items, metric_rows):
                if isinstance(metrics, Exception):
                    rejection_counts["evaluation_or_metric_failure"] += 1
                    continue
                try:
                    metrics_cache[child_hash] = MappingProxyType(
                        _canonical_metrics(metrics)
                    )
                except (KeyError, TypeError, ValueError, RuntimeError):
                    rejection_counts["evaluation_or_metric_failure"] += 1
                    continue
                evaluation_order.append(child_hash)

        for parent, action, child in raw_proposals:
            child_hash = child.genotype_hash
            if child_hash not in metrics_cache:
                continue
            child_metrics = metrics_cache[child_hash]
            try:
                key = _transition_key(parent, child_metrics, action, child_hash)
            except ValueError as exc:
                rejection_counts[str(exc)] += 1
                continue
            previous = proposals.get(child_hash)
            proposal = (key, parent, action, child)
            if previous is None or key < previous[0]:
                proposals[child_hash] = proposal
            else:
                rejection_counts["duplicate_genotype"] += 1

        ranked: list[tuple[tuple[Any, ...], GreedySearchState]] = []
        for child_hash, (key, parent, action, child) in proposals.items():
            state = states.get(child_hash)
            if state is None:
                state = GreedySearchState(
                    genotype=child,
                    metrics=metrics_cache[child_hash],
                    parent_hash=parent.genotype_hash,
                    action_id=action.action_id,
                )
                states[child_hash] = state
            record_terminal(state)
            ranked.append((key, state))
        frontier = [state for _key, state in sorted(ranked)[:width]]

    if primary_terminals:
        terminal = min(
            primary_terminals.values(),
            key=lambda row: _terminal_key(row, policy.target),
        )
        admission_mode = "primary_bops_tolerance"
    elif expanded_terminals:
        terminal = min(
            expanded_terminals.values(),
            key=lambda row: _terminal_key(row, policy.target),
        )
        admission_mode = "expanded_bops_tolerance"
    else:
        terminal = None
        admission_mode = "no_bops_candidate"

    evaluated_states = tuple(
        states[state_hash]
        for state_hash in evaluation_order
        if state_hash in states
    )
    admission = select_bops_candidates(
        [
            {**dict(state.metrics), "genotype_hash": state.genotype_hash}
            for state in evaluated_states
        ],
        policy=policy,
    )
    if terminal is None:
        return GreedyBudgetResult(
            target=policy.target,
            status="infeasible",
            terminal=None,
            admission_mode=admission_mode,
            accepted_path_states=(),
            expansion_count=max(0, len(metrics_cache) - 1),
            failure_reason="no_state_within_expanded_bops_tolerance",
            evaluated_state_count=len(metrics_cache),
            evaluated_states=evaluated_states,
            rejection_counts=rejection_counts,
            bops_funnel=admission["funnel"],
            nearest_misses=tuple(admission["nearest_misses"]),
        )
    return GreedyBudgetResult(
        target=policy.target,
        status="ok",
        terminal=terminal,
        admission_mode=admission_mode,
        accepted_path_states=_accepted_path(terminal, states),
        expansion_count=max(0, len(metrics_cache) - 1),
        evaluated_state_count=len(metrics_cache),
        evaluated_states=evaluated_states,
        rejection_counts=rejection_counts,
        bops_funnel=admission["funnel"],
        nearest_misses=tuple(admission["nearest_misses"]),
    )
