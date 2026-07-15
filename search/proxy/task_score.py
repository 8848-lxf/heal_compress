"""Fixed exponential task-preservation score for the joint Taylor proxy."""

from __future__ import annotations

import math
from typing import Any


def compute_exponential_j1(
    *,
    l_joint: float,
    tau: float,
    original_params: int,
    candidate_params: int,
    proxy_mode: str = "joint_taylor_second_order_fisher_diag",
    l_joint_first_order: float | None = None,
    l_joint_second_order: float | None = None,
) -> dict[str, Any]:
    loss = float(l_joint)
    scale = float(tau)
    original = int(original_params)
    candidate = int(candidate_params)
    if not math.isfinite(loss) or loss < 0.0:
        raise ValueError("joint_taylor_loss_must_be_finite_nonnegative")
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("joint_taylor_tau_must_be_finite_positive")
    if original <= 0 or candidate < 0 or candidate > original:
        raise ValueError("invalid_structural_parameter_counts")
    raw_exponent = float(loss / scale)
    exponent = float(min(raw_exponent, 80.0))
    task_score = float(math.exp(-exponent))
    prune_rate = float(1.0 - float(candidate) / float(original))
    j1 = float(0.8 * task_score + 0.2 * prune_rate)
    return {
        "L_joint_raw": loss,
        "L_joint_first_order": float(
            loss if l_joint_first_order is None else l_joint_first_order
        ),
        "L_joint_second_order": float(
            0.0 if l_joint_second_order is None else l_joint_second_order
        ),
        "tau": scale,
        "exponent_value": exponent,
        "S_task": task_score,
        "original_params": original,
        "candidate_params": candidate,
        "R_prune": prune_rate,
        "J1": j1,
        # The existing GA engine minimizes F1. Negating J1 preserves that
        # implementation while the formal objective remains maximize(J1).
        "F1": -j1,
        "proxy_mode": str(proxy_mode),
        "task_score_saturated": bool(raw_exponent > 80.0),
        "sqnr_main_objective_contribution": 0.0,
    }

