"""Stage-1 proxy objective."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from ..candidate import CandidatePhenotype
from .bops_proxy import BOPSProxy
from .fisher_proxy import FisherTaylorProxy
from .joint_weight_taylor import JointWeightTaylorProxy
from .joint_weight_activation_taylor import JointWeightActivationTaylorProxy
from .normalization import NormalizationStats
from .size_proxy import SizeProxy
from .sqnr_proxy import SQNRProxy


@dataclass(frozen=True)
class ProxyObjectiveConfig:
    objective_mode: str = "legacy_fisher_sqnr_size_bops"
    alpha_fisher: float = 1.0
    beta_sqnr: float = 1.0
    gamma_size: float = 1.0
    delta_bops: float = 1.0
    size_threshold: float | None = None
    bops_threshold: float | None = 0.40
    bops_constraint_mode: str = "weighted_penalty"
    bops_tolerance_abs: float = 0.0
    bops_penalty_formula: str = "absolute_excess_squared"
    lambda_bops: float = 1.0
    parameter_retention_tiebreak_epsilon: float = 0.0
    illegal_score: float = float("inf")


def bops_target_for_generation(generation: int, num_generations: int, schedule: dict[str, float] | None) -> float | None:
    if not schedule:
        return None
    start = float(schedule.get("start", schedule.get("end", 1.0)))
    end = float(schedule.get("end", start))
    if int(num_generations) <= 1:
        return end
    ratio = max(0.0, min(1.0, float(generation) / max(float(num_generations - 1), 1.0)))
    return start - (start - end) * ratio


def bops_target_for_outer_round(round_index: int, outer_rounds: int, schedule: dict[str, float] | None) -> float | None:
    if not schedule:
        return None
    start = float(schedule.get("start_target", schedule.get("start", schedule.get("end_target", schedule.get("end", 0.25)))))
    end = float(schedule.get("end_target", schedule.get("end", start)))
    ratio = max(0.0, min(1.0, float(round_index) / max(float(outer_rounds - 1), 1.0)))
    return start - (start - end) * ratio


def bops_soft_penalty(
    r_bops: float,
    target: float | None,
    *,
    formula: str = "absolute_excess_squared",
    constraint_mode: str = "weighted_penalty",
    tolerance_abs: float = 0.0,
) -> tuple[float, float]:
    if target is None:
        return 0.0, 0.0
    value = float(r_bops)
    threshold = float(target)
    if str(constraint_mode) == "hard_band_feasibility":
        violation = max(0.0, abs(value - threshold) - max(0.0, float(tolerance_abs)))
    elif str(formula) == "squared_relative_excess":
        violation = max(0.0, value / max(threshold, 1.0e-12) - 1.0)
    else:
        violation = max(0.0, value - threshold)
    return float(violation), float(violation * violation)


def feasibility_first_key(
    metrics: dict[str, Any],
    *,
    bops_target: float | None,
    constraint_mode: str = "hard_feasibility",
    tolerance_abs: float = 0.0,
) -> tuple[int, float, float]:
    if bops_target is None:
        return (0, 0.0, float(metrics.get("F1", metrics.get("score", float("inf")))))
    bops = float(metrics.get("R_bops_vs_fp32", metrics.get("R_bops", float("inf"))))
    if str(constraint_mode) == "hard_band_feasibility":
        violation = max(
            0.0,
            abs(bops - float(bops_target)) - max(0.0, float(tolerance_abs)),
        )
    else:
        violation = max(0.0, bops - float(bops_target))
    return (1 if violation > 0.0 else 0, violation, float(metrics.get("F1", metrics.get("score", float("inf")))))


class ProxyObjective:
    """Compute F1 where every term is smaller-is-better."""

    def __init__(
        self,
        fisher: FisherTaylorProxy | None = None,
        sqnr: SQNRProxy | None = None,
        size: SizeProxy | None = None,
        bops: BOPSProxy | None = None,
        joint_weight_taylor: JointWeightTaylorProxy | None = None,
        joint_weight_activation_taylor: JointWeightActivationTaylorProxy | None = None,
        *,
        normalization: NormalizationStats | None = None,
        config: ProxyObjectiveConfig | None = None,
    ) -> None:
        self.fisher = fisher or FisherTaylorProxy()
        self.sqnr = sqnr or SQNRProxy()
        self.size = size or SizeProxy(layer_parameter_counts={})
        self.bops = bops or BOPSProxy(layer_ops={})
        self.joint_weight_taylor = joint_weight_taylor
        self.joint_weight_activation_taylor = joint_weight_activation_taylor
        self.normalization = normalization or NormalizationStats()
        self.config = config or ProxyObjectiveConfig()

    def evaluate(self, phenotype: CandidatePhenotype, *, legal: bool = True) -> dict[str, Any]:
        if not legal:
            return {"F1": self.config.illegal_score, "legal": False}
        activation_joint_mode = self.config.objective_mode == "joint_weight_activation_taylor_hard_bops"
        joint_mode = self.config.objective_mode in {
            "joint_weight_taylor_hard_bops",
            "joint_weight_activation_taylor_hard_bops",
        }
        if joint_mode:
            if activation_joint_mode:
                if self.joint_weight_activation_taylor is None:
                    raise RuntimeError("joint_weight_activation_taylor_proxy_missing")
                joint_metrics = dict(self.joint_weight_activation_taylor.evaluate_breakdown(phenotype))
                fisher = float(joint_metrics["L_joint_weight_activation_taylor"])
            else:
                if self.joint_weight_taylor is None:
                    raise RuntimeError("joint_weight_taylor_proxy_missing")
                joint_metrics = dict(self.joint_weight_taylor.evaluate_breakdown(phenotype))
                fisher = float(joint_metrics["L_joint_weight_taylor"])
            sqnr = 0.0
        else:
            joint_metrics = {}
            fisher = self.fisher.evaluate(phenotype)
            sqnr = self.sqnr.evaluate(phenotype)
        if hasattr(self.size, "evaluate_breakdown"):
            size_metrics = self.size.evaluate_breakdown(phenotype)
            size = float(size_metrics.get("R_size_vs_fp32", size_metrics.get("R_size_vs_fp16_deploy", 0.0)))
        else:
            size = float(self.size.evaluate(phenotype))
            size_metrics = {"R_size_vs_fp32": size, "R_size_vs_fp16_deploy": size}
        parameter_retention = float(size_metrics.get("R_parameter_retention", 1.0))
        if hasattr(self.bops, "evaluate_breakdown"):
            bops_metrics = self.bops.evaluate_breakdown(phenotype)
        else:
            legacy_bops = float(self.bops.evaluate(phenotype))
            bops_metrics = {"R_bops_vs_fp16_deploy": legacy_bops, "R_bops_vs_fp32": legacy_bops, "R_bops": legacy_bops}
        bops = float(bops_metrics.get("R_bops_vs_fp32", bops_metrics.get("R_bops", 0.0)))
        penalty = 0.0
        if self.config.size_threshold is not None:
            penalty += max(0.0, size - float(self.config.size_threshold)) ** 2
        bops_violation, bops_penalty = bops_soft_penalty(
            bops,
            self.config.bops_threshold,
            formula=self.config.bops_penalty_formula,
            constraint_mode=self.config.bops_constraint_mode,
            tolerance_abs=self.config.bops_tolerance_abs,
        )
        if self.config.bops_threshold is not None:
            penalty += float(self.config.lambda_bops) * bops_penalty
        if joint_mode:
            raw_score = float(
                joint_metrics.get(
                    "L_joint_weight_activation_taylor",
                    joint_metrics.get("L_joint_weight_taylor", float("inf")),
                )
            )
            raw_score += (
                float(self.config.parameter_retention_tiebreak_epsilon)
                * parameter_retention
            )
        else:
            raw_score = (
                self.config.alpha_fisher * self.normalization.normalize("L_fisher", fisher)
                + self.config.beta_sqnr * self.normalization.normalize("L_sqnr", sqnr)
                + self.config.gamma_size * size
                + self.config.delta_bops * bops_penalty
                + (penalty - float(self.config.lambda_bops) * bops_penalty)
            )
        hard_one_sided = self.config.bops_constraint_mode in {
            "feasibility_first",
            "hard_feasibility",
        }
        if hard_one_sided and bops_violation > 0.0:
            score = 1.0e6 + bops_violation * 1.0e3 + raw_score
        else:
            score = raw_score
        if not math.isfinite(score):
            score = self.config.illegal_score
        return {
            **joint_metrics,
            "L_fisher": float(fisher),
            "L_sqnr": float(sqnr),
            "R_size": float(size),
            "R_size_vs_fp16_deploy": float(size_metrics.get("R_size_vs_fp16_deploy", size)),
            "R_size_vs_fp32": float(size_metrics.get("R_size_vs_fp32", size)),
            "R_size_reference": "original_fp32",
            "mixed_weight_size_bytes": float(
                size_metrics.get("size_bits_total", 0.0)
            ) / 8.0,
            "R_parameter_retention": parameter_retention,
            "parameter_pruning_rate": float(
                size_metrics.get("parameter_pruning_rate", 1.0 - parameter_retention)
            ),
            "parameter_count_base": float(size_metrics.get("parameter_count_base", 0.0)),
            "parameter_count_after": float(size_metrics.get("parameter_count_after", 0.0)),
            "constant_untracked_parameter_count": float(
                size_metrics.get("constant_untracked_parameter_count", 0.0)
            ),
            "R_bops": float(bops),
            "R_bops_vs_fp16_deploy": float(bops_metrics.get("R_bops_vs_fp16_deploy", bops)),
            "R_bops_vs_fp32": float(bops_metrics.get("R_bops_vs_fp32", bops)),
            "R_bops_reference": "original_fp32",
            "BOPS_target": float(self.config.bops_threshold) if self.config.bops_threshold is not None else None,
            "P_bops": float(bops_penalty),
            "int8_macs_ratio": float(bops_metrics.get("int8_macs_ratio", 0.0)),
            "bops_fp16_baseline": float(bops_metrics.get("bops_fp16_baseline", 0.0)),
            "bops_fp32_baseline": float(bops_metrics.get("bops_fp32_baseline", 0.0)),
            "bops_breakdown": list(bops_metrics.get("breakdown", [])),
            "constraint_penalty": float(penalty),
            "bops_violation": float(bops_violation),
            "bops_abs_delta": (
                abs(float(bops) - float(self.config.bops_threshold))
                if self.config.bops_threshold is not None
                else 0.0
            ),
            "bops_tolerance_abs": float(self.config.bops_tolerance_abs),
            "bops_feasible": bool(bops_violation <= 0.0),
            "proxy_score_raw": float(raw_score),
            "F1": float(score),
            "legal": True,
            "normalization": self.normalization.to_dict(),
            "objective_mode": self.config.objective_mode,
            "parameter_retention_role": (
                "report_only" if self.config.parameter_retention_tiebreak_epsilon == 0.0 else "epsilon_tiebreak"
            ),
        }
