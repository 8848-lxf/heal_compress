"""Robust iterative greedy baseline for the joint compression search space."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Any, Callable, Sequence

from ..candidate import CandidateGenotype
from ..canonicalization import SearchSpaceSpec, repair_genotype


BatchEvaluator = Callable[[list[CandidateGenotype], int], list[dict[str, Any]]]


def _identity(candidate: CandidateGenotype) -> str:
    payload = {
        "pruning_width_genes": dict(sorted(candidate.pruning_width_genes.items())),
        "precision_genes": dict(sorted(candidate.precision_genes.items())),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _bops(metrics: dict[str, Any]) -> float:
    return float(metrics.get("R_bops_vs_fp32", metrics.get("R_bops", float("inf"))))


def _loss(metrics: dict[str, Any]) -> float:
    return float(
        metrics.get(
            "L_joint_weight_taylor",
            metrics.get("proxy_score_raw", metrics.get("F1", float("inf"))),
        )
    )


@dataclass(frozen=True)
class GreedySearchConfig:
    bops_targets: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
    minimum_bops_reduction: float = 1.0e-12
    maximum_steps: int = 10000
    precision_order: tuple[str, ...] = ("FP32", "FP16", "INT8")
    parameter_retention_tiebreak: bool = True
    bops_tolerance_abs: float = 0.005

    def __post_init__(self) -> None:
        targets = tuple(sorted({float(value) for value in self.bops_targets}))
        if not targets or any(not 0.0 < value <= 1.0 for value in targets):
            raise ValueError(f"invalid_greedy_bops_targets:{targets}")
        object.__setattr__(self, "bops_targets", targets)
        object.__setattr__(
            self,
            "precision_order",
            tuple(str(value).upper() for value in self.precision_order),
        )


@dataclass(frozen=True)
class GreedyStep:
    step_index: int
    action_kind: str
    action_gene_id: str
    previous_value: str | int
    selected_value: str | int
    bops_before: float
    bops_after: float
    bops_reduction: float
    loss_before: float
    loss_after: float
    marginal_loss: float
    marginal_loss_per_bops: float
    candidate_hash: str
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "action_kind": self.action_kind,
            "action_gene_id": self.action_gene_id,
            "previous_value": self.previous_value,
            "selected_value": self.selected_value,
            "bops_before": self.bops_before,
            "bops_after": self.bops_after,
            "bops_reduction": self.bops_reduction,
            "loss_before": self.loss_before,
            "loss_after": self.loss_after,
            "marginal_loss": self.marginal_loss,
            "marginal_loss_per_bops": self.marginal_loss_per_bops,
            "candidate_hash": self.candidate_hash,
            "metrics": dict(self.metrics),
        }


@dataclass(frozen=True)
class GreedySearchResult:
    initial_candidate: CandidateGenotype
    initial_metrics: dict[str, Any]
    steps: tuple[GreedyStep, ...]
    budget_candidates: dict[float, CandidateGenotype]
    budget_metrics: dict[float, dict[str, Any]]
    nearest_budget_candidates: dict[float, CandidateGenotype]
    nearest_budget_metrics: dict[float, dict[str, Any]]
    unreachable_targets: tuple[float, ...]
    termination_reason: str
    evaluated_neighbor_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "initial_candidate": self.initial_candidate.to_dict(),
            "initial_metrics": dict(self.initial_metrics),
            "steps": [step.to_dict() for step in self.steps],
            "budget_candidates": {
                f"{target:.6f}": candidate.to_dict()
                for target, candidate in sorted(self.budget_candidates.items())
            },
            "budget_metrics": {
                f"{target:.6f}": dict(metrics)
                for target, metrics in sorted(self.budget_metrics.items())
            },
            "nearest_budget_candidates": {
                f"{target:.6f}": candidate.to_dict()
                for target, candidate in sorted(self.nearest_budget_candidates.items())
            },
            "nearest_budget_metrics": {
                f"{target:.6f}": dict(metrics)
                for target, metrics in sorted(self.nearest_budget_metrics.items())
            },
            "unreachable_targets": list(self.unreachable_targets),
            "termination_reason": self.termination_reason,
            "evaluated_neighbor_count": self.evaluated_neighbor_count,
            "search_semantics": {
                "action_set": "one_adjacent_legal_domain_width_or_precision_downgrade",
                "selection": "minimum_incremental_joint_taylor_loss_per_positive_BOPS_reduction",
                "neighbor_costs_recomputed_after_every_step": True,
                "activation_taylor_included": False,
                "budget_capture": "abs(R_BOPS-target)<=bops_tolerance_abs",
                "bops_tolerance_abs": float(self.config.bops_tolerance_abs),
                "stage2_policy": "only_unique_final_candidate_per_budget_full_validation",
            },
        }


class GreedyBudgetSearch:
    """Follow one monotonic path and snapshot the first feasible candidate/budget."""

    def __init__(
        self,
        space: SearchSpaceSpec,
        *,
        config: GreedySearchConfig | None = None,
    ) -> None:
        if not space.pruning_domains:
            raise ValueError("greedy_domain_width_search_requires_pruning_domains")
        self.space = space
        self.config = config or GreedySearchConfig()
        self._groups = {group.group_id: group for group in space.quantization_groups}

    def _initial_candidate(self) -> CandidateGenotype:
        precision: dict[str, str] = {}
        for gene_id in self.space.precision_gene_ids:
            allowed = (
                tuple(self._groups[gene_id].allowed_precisions)
                if gene_id in self._groups
                else self.config.precision_order
            )
            precision[gene_id] = next(
                (value for value in self.config.precision_order if value in allowed),
                allowed[0],
            )
        return repair_genotype(
            CandidateGenotype(
                pruning_genes={unit_id: 1 for unit_id in self.space.pruning_unit_ids},
                pruning_width_genes={
                    domain.domain_id: int(domain.original_width)
                    for domain in self.space.pruning_domains
                },
                precision_genes=precision,
                meta={"created_by": "greedy_all_keep_highest_precision"},
            ),
            self.space,
        )

    def _precision_successor(self, gene_id: str, current: str) -> str | None:
        allowed = (
            tuple(self._groups[gene_id].allowed_precisions)
            if gene_id in self._groups
            else self.config.precision_order
        )
        allowed_ordered = [value for value in self.config.precision_order if value in allowed]
        if current not in allowed_ordered:
            return allowed_ordered[0] if allowed_ordered else None
        index = allowed_ordered.index(current)
        return allowed_ordered[index + 1] if index + 1 < len(allowed_ordered) else None

    def _neighbors(
        self,
        current: CandidateGenotype,
    ) -> list[tuple[CandidateGenotype, dict[str, Any]]]:
        rows: list[tuple[CandidateGenotype, dict[str, Any]]] = []
        for domain in self.space.pruning_domains:
            width = int(
                current.pruning_width_genes.get(domain.domain_id, domain.original_width)
            )
            position = domain.legal_widths.index(width)
            if position <= 0:
                continue
            selected = int(domain.legal_widths[position - 1])
            genes = dict(current.pruning_width_genes)
            genes[domain.domain_id] = selected
            rows.append(
                (
                    repair_genotype(
                        CandidateGenotype(
                            pruning_genes=current.pruning_genes,
                            pruning_width_genes=genes,
                            precision_genes=current.precision_genes,
                            meta={"created_by": "greedy_width_neighbor"},
                        ),
                        self.space,
                    ),
                    {
                        "kind": "domain_width",
                        "gene_id": domain.domain_id,
                        "previous": width,
                        "selected": selected,
                    },
                )
            )
        for gene_id in self.space.precision_gene_ids:
            current_precision = str(current.precision_genes[gene_id]).upper()
            selected = self._precision_successor(gene_id, current_precision)
            if selected is None:
                continue
            genes = dict(current.precision_genes)
            genes[gene_id] = selected
            rows.append(
                (
                    repair_genotype(
                        CandidateGenotype(
                            pruning_genes=current.pruning_genes,
                            pruning_width_genes=current.pruning_width_genes,
                            precision_genes=genes,
                            meta={"created_by": "greedy_precision_neighbor"},
                        ),
                        self.space,
                    ),
                    {
                        "kind": "precision",
                        "gene_id": gene_id,
                        "previous": current_precision,
                        "selected": selected,
                    },
                )
            )
        unique: dict[str, tuple[CandidateGenotype, dict[str, Any]]] = {}
        for candidate, action in rows:
            unique.setdefault(_identity(candidate), (candidate, action))
        return [unique[key] for key in sorted(unique)]

    @staticmethod
    def _evaluate(
        evaluator: BatchEvaluator,
        candidates: list[CandidateGenotype],
        step: int,
    ) -> list[dict[str, Any]]:
        rows = list(evaluator(candidates, step))
        if len(rows) != len(candidates):
            raise RuntimeError(
                f"greedy_batch_evaluator_length_mismatch:{len(candidates)}:{len(rows)}"
            )
        return [dict(row) for row in rows]

    def run(self, evaluator: BatchEvaluator) -> GreedySearchResult:
        current = self._initial_candidate()
        current_metrics = self._evaluate(evaluator, [current], 0)[0]
        initial_candidate = current
        initial_metrics = dict(current_metrics)
        if not math.isfinite(_loss(current_metrics)) or not math.isfinite(_bops(current_metrics)):
            raise RuntimeError("greedy_initial_candidate_nonfinite")
        targets_desc = tuple(sorted(self.config.bops_targets, reverse=True))
        budget_candidates: dict[float, CandidateGenotype] = {}
        budget_metrics: dict[float, dict[str, Any]] = {}
        nearest_budget_candidates = {
            target: current for target in targets_desc
        }
        nearest_budget_metrics = {
            target: dict(current_metrics) for target in targets_desc
        }
        for target in targets_desc:
            if abs(_bops(current_metrics) - target) <= float(
                self.config.bops_tolerance_abs
            ):
                budget_candidates[target] = current
                budget_metrics[target] = dict(current_metrics)
        steps: list[GreedyStep] = []
        seen = {_identity(current)}
        evaluated_neighbors = 0
        termination = "minimum_target_reached" if _bops(current_metrics) <= min(targets_desc) else ""
        while not termination and len(steps) < int(self.config.maximum_steps):
            neighbors = [
                (candidate, action)
                for candidate, action in self._neighbors(current)
                if _identity(candidate) not in seen
            ]
            if not neighbors:
                termination = "no_remaining_legal_action"
                break
            metrics_rows = self._evaluate(
                evaluator,
                [candidate for candidate, _action in neighbors],
                len(steps) + 1,
            )
            evaluated_neighbors += len(neighbors)
            bops_before = _bops(current_metrics)
            loss_before = _loss(current_metrics)
            feasible_rows = []
            for (candidate, action), metrics in zip(neighbors, metrics_rows):
                bops_after = _bops(metrics)
                loss_after = _loss(metrics)
                reduction = bops_before - bops_after
                if (
                    not math.isfinite(bops_after)
                    or not math.isfinite(loss_after)
                    or reduction <= float(self.config.minimum_bops_reduction)
                ):
                    continue
                marginal = loss_after - loss_before
                ratio = marginal / reduction
                parameter_retention_tie = float(
                    metrics.get("R_parameter_retention", 1.0)
                )
                key = (
                    ratio,
                    loss_after,
                    parameter_retention_tie
                    if self.config.parameter_retention_tiebreak
                    else 0.0,
                    _identity(candidate),
                )
                feasible_rows.append(
                    (key, candidate, action, metrics, reduction, marginal, ratio)
                )
            if not feasible_rows:
                termination = "no_positive_bops_reduction_action"
                break
            (
                _key,
                selected_candidate,
                selected_action,
                selected_metrics,
                reduction,
                marginal,
                ratio,
            ) = min(feasible_rows, key=lambda row: row[0])
            candidate_id = _identity(selected_candidate)
            seen.add(candidate_id)
            step = GreedyStep(
                step_index=len(steps) + 1,
                action_kind=str(selected_action["kind"]),
                action_gene_id=str(selected_action["gene_id"]),
                previous_value=selected_action["previous"],
                selected_value=selected_action["selected"],
                bops_before=bops_before,
                bops_after=_bops(selected_metrics),
                bops_reduction=reduction,
                loss_before=loss_before,
                loss_after=_loss(selected_metrics),
                marginal_loss=marginal,
                marginal_loss_per_bops=ratio,
                candidate_hash=candidate_id,
                metrics=dict(selected_metrics),
            )
            steps.append(step)
            current = selected_candidate
            current_metrics = dict(selected_metrics)
            for target in targets_desc:
                if abs(_bops(current_metrics) - target) < abs(
                    _bops(nearest_budget_metrics[target]) - target
                ):
                    nearest_budget_candidates[target] = current
                    nearest_budget_metrics[target] = dict(current_metrics)
                if target not in budget_candidates and abs(
                    _bops(current_metrics) - target
                ) <= float(self.config.bops_tolerance_abs):
                    budget_candidates[target] = current
                    budget_metrics[target] = dict(current_metrics)
            if _bops(current_metrics) <= min(targets_desc):
                termination = "minimum_target_reached"
        if not termination:
            termination = "maximum_steps_reached"
        unreachable = tuple(
            sorted(set(self.config.bops_targets) - set(budget_candidates))
        )
        return GreedySearchResult(
            initial_candidate=initial_candidate,
            initial_metrics=initial_metrics,
            steps=tuple(steps),
            budget_candidates=budget_candidates,
            budget_metrics=budget_metrics,
            nearest_budget_candidates=nearest_budget_candidates,
            nearest_budget_metrics=nearest_budget_metrics,
            unreachable_targets=unreachable,
            termination_reason=termination,
            evaluated_neighbor_count=evaluated_neighbors,
        )
