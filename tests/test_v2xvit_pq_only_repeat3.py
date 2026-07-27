from __future__ import annotations

from search.candidate import CandidateGenotype
from scripts.run_v2xvit_sixbudget_pq_only_full1789_repeat3 import q_only_genotype


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
