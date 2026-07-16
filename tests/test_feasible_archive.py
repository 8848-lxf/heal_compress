from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _row(
    candidate: str,
    structure: str,
    precision: str,
    *,
    bops: float,
    task: float,
    params: float,
    latency: float,
) -> dict[str, object]:
    return {
        "candidate_id": candidate,
        "phenotype_hash": candidate,
        "structure_hash": structure,
        "precision_hash": precision,
        "structure_legal": True,
        "precision_legal": True,
        "missing_mapping": 0,
        "finite_joint_proxy": True,
        "R_BOPS": bops,
        "S_task": task,
        "R_param": params,
        "latency_proxy_ms": latency,
    }


def test_archive_rejects_infeasible_and_duplicate_phenotypes() -> None:
    from search.archive.feasible_pareto_archive import FeasibleParetoArchive

    archive = FeasibleParetoArchive()
    good = _row("p0", "s0", "q0", bops=0.2, task=0.9, params=0.8, latency=4.0)
    assert archive.add(good, active_budget=0.21) is True
    assert archive.add(good, active_budget=0.21) is False
    assert archive.add(
        _row("p1", "s1", "q1", bops=0.3, task=0.95, params=0.7, latency=3.0),
        active_budget=0.21,
    ) is False

    assert len(archive.records) == 1
    assert archive.rejection_counts == {
        "duplicate_phenotype_hash": 1,
        "R_BOPS_above_active_budget": 1,
    }


def test_stage2_selection_preserves_structure_and_precision_diversity() -> None:
    from search.archive.feasible_pareto_archive import FeasibleParetoArchive

    archive = FeasibleParetoArchive()
    rows = [
        _row("p0", "s0", "q0", bops=0.20, task=0.95, params=0.80, latency=4.0),
        _row("p1", "s0", "q1", bops=0.19, task=0.94, params=0.80, latency=3.9),
        _row("p2", "s1", "q0", bops=0.18, task=0.93, params=0.70, latency=3.8),
        _row("p3", "s2", "q2", bops=0.17, task=0.92, params=0.60, latency=3.7),
    ]
    for row in rows:
        assert archive.add(row, active_budget=0.21)

    selected = archive.select_stage2_candidates(3)

    assert len({row["structure_hash"] for row in selected}) == 3
    assert len({row["phenotype_hash"] for row in selected}) == 3

