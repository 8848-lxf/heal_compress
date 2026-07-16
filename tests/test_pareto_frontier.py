from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _full(candidate: str, *, map_value: float, bops: float, params: float, latency: float):
    return {
        "candidate_id": candidate,
        "mAP": map_value,
        "R_BOPS": bops,
        "R_param": params,
        "formal_latency_p50_ms": latency,
        "latency_proxy_ms": latency - 1.0,
        "evaluation_protocol": "full_validation",
        "full_validation_success": True,
    }


def test_pareto_dominance_maximizes_map_and_minimizes_resource() -> None:
    from search.reporting.pareto_frontier import official_pareto_front

    rows = [
        _full("a", map_value=0.70, bops=0.20, params=0.8, latency=4.0),
        _full("b", map_value=0.69, bops=0.22, params=0.7, latency=3.0),
        _full("c", map_value=0.71, bops=0.24, params=0.9, latency=5.0),
        _full("dominated", map_value=0.68, bops=0.25, params=0.9, latency=6.0),
    ]

    front = official_pareto_front(rows, resource_key="R_BOPS")

    assert [row["candidate_id"] for row in front] == ["a", "c"]


def test_screening_and_failed_full_validation_never_enter_official_front() -> None:
    from search.reporting.pareto_frontier import official_pareto_front

    valid = _full("valid", map_value=0.7, bops=0.2, params=0.8, latency=4.0)
    screening = {**valid, "candidate_id": "screen", "evaluation_protocol": "screening_50"}
    failed = {**valid, "candidate_id": "failed", "full_validation_success": False}

    front = official_pareto_front(
        [valid, screening, failed], resource_key="R_param"
    )

    assert [row["candidate_id"] for row in front] == ["valid"]


def test_official_latency_front_rejects_proxy_latency_axis() -> None:
    from search.reporting.pareto_frontier import official_pareto_front

    with pytest.raises(ValueError, match="official_latency_requires_formal_real_latency"):
        official_pareto_front(
            [_full("valid", map_value=0.7, bops=0.2, params=0.8, latency=4.0)],
            resource_key="latency_proxy_ms",
        )


def test_pareto_writer_emits_three_fronts_and_combined_plots(tmp_path) -> None:
    from search.reporting.pareto_frontier import write_official_pareto_artifacts

    result = write_official_pareto_artifacts(
        [
            _full("a", map_value=0.70, bops=0.20, params=0.8, latency=4.0),
            _full("b", map_value=0.69, bops=0.18, params=0.7, latency=3.0),
        ],
        tmp_path,
    )

    assert set(result) == {"bops", "param", "latency", "combined"}
    assert (tmp_path / "pareto_combined_bops_map_latency.png").is_file()
    assert (tmp_path / "pareto_combined_param_map_latency.png").is_file()


def test_pareto_writer_accepts_greedy_source_marker(tmp_path) -> None:
    from search.reporting.pareto_frontier import write_official_pareto_artifacts

    row = {
        **_full("greedy", map_value=0.70, bops=0.20, params=0.8, latency=4.0),
        "candidate_source": "greedy",
    }

    result = write_official_pareto_artifacts([row], tmp_path)

    assert result["bops"]["point_count"] == 1
