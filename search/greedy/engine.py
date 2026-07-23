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
            "L_joint_weight_activation_taylor",
            metrics.get(
                "L_joint_weight_taylor",
                metrics.get("proxy_score_raw", metrics.get("F1", float("inf"))),
            ),
        )
    )


def _finite_metric(metrics: dict[str, Any], *names: str) -> float:
    """Return the first finite ranking metric, or infinity when unavailable."""

    for name in names:
        raw = metrics.get(name)
        if raw is None:
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            return value
    return float("inf")


def _secondary_cost_key(metrics: dict[str, Any]) -> tuple[float, float, float]:
    """Deployment tie-break: latency, physical parameters, mixed weights."""

    return (
        _finite_metric(metrics, "latency_proxy_ms", "R_latency_proxy"),
        _finite_metric(metrics, "R_parameter_retention"),
        _finite_metric(metrics, "mixed_weight_size_bytes", "R_size_vs_fp32"),
    )


@dataclass(frozen=True)
class GreedySearchConfig:
    bops_targets: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
    minimum_bops_reduction: float = 1.0e-12
    maximum_steps: int = 10000
    precision_order: tuple[str, ...] = ("FP32", "FP16", "INT8")
    parameter_retention_tiebreak: bool = True
    marginal_score_relative_epsilon: float = 1.0e-12
    marginal_score_absolute_epsilon: float = 1.0e-12
    bops_tolerance_abs: float = 0.005
    budget_recovery_beam_width: int = 8
    budget_recovery_seed_pool_size: int = 32
    budget_recovery_max_depth: int = 64

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
        if int(self.budget_recovery_beam_width) <= 0:
            raise ValueError("greedy_budget_recovery_beam_width_must_be_positive")
        if int(self.budget_recovery_seed_pool_size) < int(
            self.budget_recovery_beam_width
        ):
            raise ValueError("greedy_budget_recovery_seed_pool_smaller_than_beam")
        if int(self.budget_recovery_max_depth) <= 0:
            raise ValueError("greedy_budget_recovery_max_depth_must_be_positive")
        if float(self.marginal_score_relative_epsilon) < 0.0:
            raise ValueError("greedy_marginal_score_relative_epsilon_must_be_nonnegative")
        if float(self.marginal_score_absolute_epsilon) < 0.0:
            raise ValueError("greedy_marginal_score_absolute_epsilon_must_be_nonnegative")


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
    neighbor_count: int
    metrics: dict[str, Any] = field(default_factory=dict)
    action_domain_type: str = ""
    action_family: str = ""
    action_module_path: str = ""

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
            "neighbor_count": self.neighbor_count,
            "action_domain_type": self.action_domain_type,
            "action_family": self.action_family,
            "action_module_path": self.action_module_path,
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
    budget_recovery_evaluated_neighbor_count: int
    budget_recovery_reports: dict[float, dict[str, Any]]
    bops_tolerance_abs: float

    def to_dict(self) -> dict[str, Any]:
        activation_taylor = "L_joint_weight_activation_taylor" in self.initial_metrics
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
            "budget_recovery_evaluated_neighbor_count": (
                self.budget_recovery_evaluated_neighbor_count
            ),
            "budget_recovery_reports": {
                f"{target:.6f}": dict(report)
                for target, report in sorted(self.budget_recovery_reports.items())
            },
            "search_semantics": {
                "action_set": "one_adjacent_legal_domain_width_or_precision_downgrade",
                "domain_action_types": [
                    "cnn_channel",
                    "grouped_conv_channel",
                    "attention_dh",
                    "ffn_hidden",
                ],
                "selection": "minimum_incremental_joint_taylor_loss_per_positive_BOPS_reduction",
                "neighbor_costs_recomputed_after_every_step": True,
                "activation_taylor_included": activation_taylor,
                "near_equal_score_tiebreak": [
                    "lower_latency_proxy",
                    "larger_parameter_compression",
                    "lower_mixed_weight_size",
                    "underrepresented_domain_type",
                    "deterministic_domain_and_candidate_hash",
                ],
                "budget_capture": (
                    "lowest_joint_taylor_loss_among_all_evaluated_one-action_neighbors_"
                    "with_abs(R_BOPS-target)<=bops_tolerance_abs"
                ),
                "missing_budget_recovery": (
                    "deterministic_target_directed_beam_over_legal_adjacent_actions"
                ),
                "bops_tolerance_abs": float(self.bops_tolerance_abs),
                "stage2_policy": "only_unique_final_candidate_per_budget_full_validation",
            },
        }


class GreedyBudgetSearch:
    """Follow one monotonic path and retain its evaluated strict-budget frontier."""

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
                        "domain_type": str(getattr(domain, "domain_type", domain.kind)),
                        "family": str(getattr(domain, "family", "")),
                        "module_path": str(
                            getattr(domain, "module_path", domain.root_module_path)
                        ),
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
                        "domain_type": "precision",
                        "family": "",
                        "module_path": "",
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
        recovery_seed_pools: dict[
            float,
            dict[str, tuple[CandidateGenotype, dict[str, Any]]],
        ] = {target: {} for target in targets_desc}

        def update_budget_frontier(
            candidate: CandidateGenotype,
            metrics: dict[str, Any],
            *,
            source: str,
        ) -> None:
            """Keep the best evaluated candidate in each strict BOPS band.

            The primary greedy trajectory remains a single monotonic path.  A
            batched step nevertheless evaluates every one-action neighbor, so
            discarding an in-band non-selected neighbor makes the reported
            budget frontier depend on an unrelated action selected for a later
            target.  Retaining those already evaluated neighbors is both
            deterministic and does not add scalar proxy work.
            """

            candidate_bops = _bops(metrics)
            candidate_loss = _loss(metrics)
            candidate_id = _identity(candidate)
            for target in targets_desc:
                delta = abs(candidate_bops - target)
                if candidate_bops > target + float(self.config.bops_tolerance_abs):
                    pool = recovery_seed_pools[target]
                    pool[candidate_id] = (candidate, dict(metrics))
                    if len(pool) > int(self.config.budget_recovery_seed_pool_size):
                        retained = sorted(
                            pool.items(),
                            key=lambda row: (
                                _bops(row[1][1]) - target,
                                _loss(row[1][1]),
                                row[0],
                            ),
                        )[: int(self.config.budget_recovery_seed_pool_size)]
                        recovery_seed_pools[target] = dict(retained)
                nearest_delta = abs(_bops(nearest_budget_metrics[target]) - target)
                nearest_key = (
                    nearest_delta,
                    _loss(nearest_budget_metrics[target]),
                    _identity(nearest_budget_candidates[target]),
                )
                candidate_nearest_key = (delta, candidate_loss, candidate_id)
                if candidate_nearest_key < nearest_key:
                    nearest_budget_candidates[target] = candidate
                    nearest_budget_metrics[target] = dict(metrics)

                if delta > float(self.config.bops_tolerance_abs):
                    continue
                parameter_retention = float(
                    metrics.get("R_parameter_retention", 1.0)
                )
                candidate_key = (
                    candidate_loss,
                    *_secondary_cost_key(metrics),
                    parameter_retention
                    if self.config.parameter_retention_tiebreak
                    else 0.0,
                    delta,
                    candidate_id,
                )
                if target in budget_candidates:
                    incumbent_metrics = budget_metrics[target]
                    incumbent_key = (
                        _loss(incumbent_metrics),
                        *_secondary_cost_key(incumbent_metrics),
                        float(incumbent_metrics.get("R_parameter_retention", 1.0))
                        if self.config.parameter_retention_tiebreak
                        else 0.0,
                        abs(_bops(incumbent_metrics) - target),
                        _identity(budget_candidates[target]),
                    )
                    if candidate_key >= incumbent_key:
                        continue
                annotated = dict(metrics)
                annotated.update(
                    {
                        "bops_target": float(target),
                        "bops_abs_delta": float(delta),
                        "bops_within_tolerance": True,
                        "greedy_budget_capture_source": str(source),
                    }
                )
                budget_candidates[target] = candidate
                budget_metrics[target] = annotated

        update_budget_frontier(current, current_metrics, source="initial_candidate")
        steps: list[GreedyStep] = []
        selected_action_type_counts: dict[str, int] = {}
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
                feasible_rows.append(
                    (candidate, action, metrics, reduction, marginal, ratio)
                )
            if not feasible_rows:
                termination = "no_positive_bops_reduction_action"
                break
            for (
                frontier_candidate,
                _frontier_action,
                frontier_metrics,
                _frontier_reduction,
                _frontier_marginal,
                _frontier_ratio,
            ) in feasible_rows:
                update_budget_frontier(
                    frontier_candidate,
                    frontier_metrics,
                    source="primary_evaluated_neighbor_frontier",
                )
            best_ratio = min(row[5] for row in feasible_rows)
            score_tolerance = max(
                float(self.config.marginal_score_absolute_epsilon),
                abs(best_ratio) * float(self.config.marginal_score_relative_epsilon),
            )
            near_best_rows = [
                row for row in feasible_rows if row[5] <= best_ratio + score_tolerance
            ]

            def action_tiebreak(row: tuple[Any, ...]) -> tuple[Any, ...]:
                candidate, action, metrics, _reduction, _marginal, ratio = row
                action_type = str(action.get("domain_type", action.get("kind", "")))
                latency, parameter_retention, mixed_weight_size = _secondary_cost_key(
                    metrics
                )
                return (
                    latency,
                    parameter_retention
                    if self.config.parameter_retention_tiebreak
                    else 0.0,
                    mixed_weight_size,
                    selected_action_type_counts.get(action_type, 0),
                    _loss(metrics),
                    ratio,
                    str(action.get("gene_id", "")),
                    _identity(candidate),
                )

            (
                selected_candidate,
                selected_action,
                selected_metrics,
                reduction,
                marginal,
                ratio,
            ) = min(near_best_rows, key=action_tiebreak)
            candidate_id = _identity(selected_candidate)
            seen.add(candidate_id)
            action_type = str(
                selected_action.get("domain_type", selected_action.get("kind", ""))
            )
            selected_action_type_counts[action_type] = (
                selected_action_type_counts.get(action_type, 0) + 1
            )
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
                neighbor_count=len(neighbors),
                metrics=dict(selected_metrics),
                action_domain_type=action_type,
                action_family=str(selected_action.get("family", "")),
                action_module_path=str(selected_action.get("module_path", "")),
            )
            steps.append(step)
            current = selected_candidate
            current_metrics = dict(selected_metrics)
            update_budget_frontier(
                current,
                current_metrics,
                source="primary_selected_path",
            )
            if _bops(current_metrics) <= min(targets_desc):
                termination = "minimum_target_reached"
        if not termination:
            termination = "maximum_steps_reached"

        recovery_evaluated_neighbors = 0
        recovery_reports: dict[float, dict[str, Any]] = {
            target: {
                "status": "reached_primary_frontier",
                "depth": 0,
                "evaluated_neighbor_count": 0,
                "beam_width": int(self.config.budget_recovery_beam_width),
            }
            for target in targets_desc
            if target in budget_candidates
        }

        def select_recovery_beam(
            rows: list[tuple[CandidateGenotype, dict[str, Any]]],
            *,
            target: float,
        ) -> list[tuple[CandidateGenotype, dict[str, Any]]]:
            unique = {_identity(candidate): (candidate, metrics) for candidate, metrics in rows}
            values = list(unique.values())
            closest = sorted(
                values,
                key=lambda row: (
                    max(0.0, _bops(row[1]) - target),
                    _loss(row[1]),
                    _identity(row[0]),
                ),
            )
            initial_bops = _bops(initial_metrics)
            initial_loss = _loss(initial_metrics)

            def cumulative_ratio(row: tuple[CandidateGenotype, dict[str, Any]]) -> float:
                compression = initial_bops - _bops(row[1])
                if compression <= float(self.config.minimum_bops_reduction):
                    return float("inf")
                return (_loss(row[1]) - initial_loss) / compression

            quality = sorted(
                values,
                key=lambda row: (
                    cumulative_ratio(row),
                    _loss(row[1]),
                    max(0.0, _bops(row[1]) - target),
                    _identity(row[0]),
                ),
            )
            selected: list[tuple[CandidateGenotype, dict[str, Any]]] = []
            selected_ids: set[str] = set()
            for index in range(max(len(closest), len(quality))):
                for ordered in (closest, quality):
                    if index >= len(ordered):
                        continue
                    row = ordered[index]
                    candidate_id = _identity(row[0])
                    if candidate_id in selected_ids:
                        continue
                    selected.append(row)
                    selected_ids.add(candidate_id)
                    if len(selected) >= int(self.config.budget_recovery_beam_width):
                        return selected
            return selected

        for target in targets_desc:
            if target in budget_candidates:
                continue
            seeds = select_recovery_beam(
                list(recovery_seed_pools[target].values())
                or [(initial_candidate, initial_metrics)],
                target=target,
            )
            beam = list(seeds)
            visited = {_identity(candidate) for candidate, _metrics in beam}
            target_evaluated = 0
            reached_depth = 0
            stop_reason = "recovery_depth_exhausted"
            for depth in range(1, int(self.config.budget_recovery_max_depth) + 1):
                expansion: dict[
                    str,
                    tuple[CandidateGenotype, dict[str, Any]],
                ] = {}
                for parent, parent_metrics in beam:
                    for candidate, _action in self._neighbors(parent):
                        candidate_id = _identity(candidate)
                        if candidate_id in visited or candidate_id in expansion:
                            continue
                        expansion[candidate_id] = (candidate, parent_metrics)
                if not expansion:
                    stop_reason = "no_unvisited_recovery_neighbor"
                    break
                expansion_rows = list(expansion.values())
                visited.update(expansion)
                metrics_rows = self._evaluate(
                    evaluator,
                    [candidate for candidate, _parent_metrics in expansion_rows],
                    len(steps) + recovery_evaluated_neighbors + 1,
                )
                recovery_evaluated_neighbors += len(expansion_rows)
                target_evaluated += len(expansion_rows)
                feasible_recovery: list[
                    tuple[CandidateGenotype, dict[str, Any]]
                ] = []
                for (candidate, parent_metrics), metrics in zip(
                    expansion_rows, metrics_rows
                ):
                    candidate_bops = _bops(metrics)
                    candidate_loss = _loss(metrics)
                    reduction = _bops(parent_metrics) - candidate_bops
                    if (
                        not math.isfinite(candidate_bops)
                        or not math.isfinite(candidate_loss)
                        or reduction <= float(self.config.minimum_bops_reduction)
                    ):
                        continue
                    update_budget_frontier(
                        candidate,
                        metrics,
                        source=f"target_directed_beam_recovery:{target:.6f}",
                    )
                    if candidate_bops >= target - float(
                        self.config.bops_tolerance_abs
                    ):
                        feasible_recovery.append((candidate, metrics))
                if target in budget_candidates:
                    reached_depth = depth
                    stop_reason = "strict_budget_reached"
                    break
                if not feasible_recovery:
                    stop_reason = "all_recovery_neighbors_undershoot_budget"
                    break
                beam = select_recovery_beam(feasible_recovery, target=target)
                if not beam:
                    stop_reason = "empty_recovery_beam"
                    break
            recovery_reports[target] = {
                "status": (
                    "reached_budget_recovery"
                    if target in budget_candidates
                    else "unreachable_after_budget_recovery"
                ),
                "depth": int(reached_depth),
                "evaluated_neighbor_count": int(target_evaluated),
                "seed_count": len(seeds),
                "beam_width": int(self.config.budget_recovery_beam_width),
                "maximum_depth": int(self.config.budget_recovery_max_depth),
                "stop_reason": stop_reason,
            }
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
            evaluated_neighbor_count=(
                evaluated_neighbors + recovery_evaluated_neighbors
            ),
            budget_recovery_evaluated_neighbor_count=recovery_evaluated_neighbors,
            budget_recovery_reports=recovery_reports,
            bops_tolerance_abs=float(self.config.bops_tolerance_abs),
        )
