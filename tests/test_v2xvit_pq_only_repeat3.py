from __future__ import annotations

from search.candidate import CandidateGenotype
from scripts.run_v2xvit_sixbudget_pq_only_full1789_repeat3 import (
    _summary,
    q_only_genotype,
)


def test_q_only_keeps_original_widths_and_replays_precision() -> None:
    baseline = CandidateGenotype(
        pruning_width_genes={"d0": 64}, precision_genes={"p0": "FP32"}
    )
    winner = CandidateGenotype(
        pruning_width_genes={"d0": 16}, precision_genes={"p0": "INT8"}
    )
    result = q_only_genotype(baseline, winner, ("p0",))
    assert result.pruning_width_genes == {"d0": 64}
    assert result.precision_genes == {"p0": "INT8"}


def test_pq_summary_supports_repeat5_and_tail_latency() -> None:
    rows = [
        {
            "AP@0.3": 0.7,
            "AP@0.5": 0.6,
            "AP@0.7": 0.4,
            "mAP": 0.5,
            "forward_p50_ms": 10.0 + index,
            "forward_p90_ms": 11.0 + index,
            "forward_p99_ms": 12.0 + index,
        }
        for index in range(5)
    ]
    result = _summary(rows, repeat_count=5)
    assert result["forward_p50_ms_mean"] == 12.0
    assert result["forward_p90_ms_mean"] == 13.0
    assert result["forward_p99_ms_mean"] == 14.0
    assert len(result["repetitions"]) == 5
