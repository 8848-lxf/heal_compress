from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _genotype(index: int):
    from search.encoding.legal_width_genotype import LegalWidthGenotype

    return LegalWidthGenotype({"domain": index}, {"pg": "FP32"})


def _metric(bops: float, loss: float, prune: float) -> dict[str, float]:
    return {"R_BOPS": bops, "L_joint_raw": loss, "R_prune": prune}


def test_greedy_metrics_drop_reconstructable_phenotype_payload() -> None:
    from search.greedy.joint_budget_search import _canonical_metrics

    row = _canonical_metrics(
        {
            **_metric(0.5, 0.1, 0.2),
            "candidate_hash": "candidate",
            "phenotype": {"metadata": {"large": [1, 2, 3]}},
        }
    )

    assert row["candidate_hash"] == "candidate"
    assert "phenotype" not in row


def test_bounded_frontier_recovers_from_best_immediate_budget_overshoot() -> None:
    from search.admission.bops_band import BopsBandPolicy
    from search.greedy.joint_budget_search import run_targeted_greedy
    from search.greedy.legal_actions import GreedyAction

    initial, overshoot, target = (_genotype(index) for index in (3, 2, 1))
    metrics = {
        initial.genotype_hash: _metric(0.25, 0.0, 0.0),
        overshoot.genotype_hash: _metric(0.18, 0.001, 0.20),
        target.genotype_hash: _metric(0.204, 0.020, 0.10),
    }
    graph = {
        initial.genotype_hash: [
            (GreedyAction("width", "domain", 3, 2), overshoot),
            (GreedyAction("width", "domain", 3, 1), target),
            (GreedyAction("precision", "pg", "FP32", "FP16"), target),
        ]
    }
    calls: Counter[str] = Counter()

    def evaluate(genotype):
        calls[genotype.genotype_hash] += 1
        return metrics[genotype.genotype_hash]

    result = run_targeted_greedy(
        initial_genotype=initial,
        evaluate=evaluate,
        enumerate_successors=lambda genotype: graph.get(genotype.genotype_hash, ()),
        policy=BopsBandPolicy(target=0.20, adjacent_targets=(0.15, 0.25)),
        frontier_size=8,
        max_expansions=4096,
    )

    assert result.status == "ok"
    assert result.terminal is not None
    assert result.terminal.metrics["R_BOPS"] == pytest.approx(0.204)
    assert result.admission_mode == "primary_bops_tolerance"
    assert len(result.accepted_path_states) == 2
    assert all(count == 1 for count in calls.values())
    assert result.evaluated_state_count == 3


def test_primary_terminal_suppresses_earlier_expanded_terminal() -> None:
    from search.admission.bops_band import BopsBandPolicy
    from search.greedy.joint_budget_search import run_targeted_greedy
    from search.greedy.legal_actions import GreedyAction

    initial, expanded, branch, primary = (_genotype(index) for index in (4, 3, 2, 1))
    metrics = {
        initial.genotype_hash: _metric(0.25, 0.0, 0.0),
        expanded.genotype_hash: _metric(0.206, 0.001, 0.05),
        branch.genotype_hash: _metric(0.23, 0.002, 0.04),
        primary.genotype_hash: _metric(0.204, 0.010, 0.08),
    }
    graph = {
        initial.genotype_hash: [
            (GreedyAction("width", "domain", 4, 3), expanded),
            (GreedyAction("width", "domain", 4, 2), branch),
        ],
        branch.genotype_hash: [
            (GreedyAction("width", "domain", 2, 1), primary),
        ],
    }

    result = run_targeted_greedy(
        initial_genotype=initial,
        evaluate=lambda genotype: metrics[genotype.genotype_hash],
        enumerate_successors=lambda genotype: graph.get(genotype.genotype_hash, ()),
        policy=BopsBandPolicy(target=0.20),
    )

    assert result.terminal is not None
    assert result.terminal.genotype.genotype_hash == primary.genotype_hash
    assert result.admission_mode == "primary_bops_tolerance"


def test_no_state_outside_expanded_tolerance_is_returned() -> None:
    from search.admission.bops_band import BopsBandPolicy
    from search.greedy.joint_budget_search import run_targeted_greedy
    from search.greedy.legal_actions import GreedyAction

    initial, below = _genotype(2), _genotype(1)
    metrics = {
        initial.genotype_hash: _metric(0.25, 0.0, 0.0),
        below.genotype_hash: _metric(0.19, 0.01, 0.1),
    }
    result = run_targeted_greedy(
        initial_genotype=initial,
        evaluate=lambda genotype: metrics[genotype.genotype_hash],
        enumerate_successors=lambda genotype: (
            [(GreedyAction("width", "domain", 2, 1), below)]
            if genotype.genotype_hash == initial.genotype_hash
            else []
        ),
        policy=BopsBandPolicy(target=0.20),
    )

    assert result.status == "infeasible"
    assert result.terminal is None
    assert result.admission_mode == "no_bops_candidate"
    assert result.failure_reason == "no_state_within_expanded_bops_tolerance"


def test_equal_rho_prefers_higher_prune_gain_then_stable_action_id() -> None:
    from search.admission.bops_band import BopsBandPolicy
    from search.greedy.joint_budget_search import run_targeted_greedy
    from search.greedy.legal_actions import GreedyAction

    initial = _genotype(5)
    low_prune = _genotype(4)
    high_prune = _genotype(3)
    terminal = _genotype(2)
    metrics = {
        initial.genotype_hash: _metric(0.30, 0.0, 0.0),
        low_prune.genotype_hash: _metric(0.25, 0.05, 0.10),
        high_prune.genotype_hash: _metric(0.24, 0.06, 0.20),
        terminal.genotype_hash: _metric(0.204, 0.08, 0.25),
    }
    graph = {
        initial.genotype_hash: [
            (GreedyAction("width", "z", 5, 4), low_prune),
            (GreedyAction("width", "y", 5, 3), high_prune),
        ],
        high_prune.genotype_hash: [
            (GreedyAction("width", "a", 3, 2), terminal),
        ],
    }

    result = run_targeted_greedy(
        initial_genotype=initial,
        evaluate=lambda genotype: metrics[genotype.genotype_hash],
        enumerate_successors=lambda genotype: graph.get(genotype.genotype_hash, ()),
        policy=BopsBandPolicy(target=0.20),
        frontier_size=1,
    )

    assert result.terminal is not None
    assert [state.genotype.genotype_hash for state in result.accepted_path_states] == [
        initial.genotype_hash,
        high_prune.genotype_hash,
        terminal.genotype_hash,
    ]


def test_equal_rho_and_prune_gain_use_stable_action_id() -> None:
    from search.admission.bops_band import BopsBandPolicy
    from search.greedy.joint_budget_search import run_targeted_greedy
    from search.greedy.legal_actions import GreedyAction

    initial, chosen, dropped, terminal = (_genotype(index) for index in (5, 4, 3, 2))
    metrics = {
        initial.genotype_hash: _metric(0.30, 0.0, 0.0),
        chosen.genotype_hash: _metric(0.25, 0.05, 0.10),
        dropped.genotype_hash: _metric(0.25, 0.05, 0.10),
        terminal.genotype_hash: _metric(0.204, 0.08, 0.20),
    }
    graph = {
        initial.genotype_hash: [
            (GreedyAction("width", "b", 5, 3), dropped),
            (GreedyAction("width", "a", 5, 4), chosen),
        ],
        chosen.genotype_hash: [
            (GreedyAction("width", "a", 4, 2), terminal),
        ],
    }

    result = run_targeted_greedy(
        initial_genotype=initial,
        evaluate=lambda genotype: metrics[genotype.genotype_hash],
        enumerate_successors=lambda genotype: graph.get(genotype.genotype_hash, ()),
        policy=BopsBandPolicy(target=0.20),
        frontier_size=1,
    )

    assert result.terminal is not None
    assert result.accepted_path_states[1].action_id.startswith("width:a:")
