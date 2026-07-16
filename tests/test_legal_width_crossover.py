from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_crossover_uses_whole_domain_and_whole_precision_group_genes() -> None:
    from search.encoding.legal_width_genotype import LegalWidthGenotype
    from search.operators.legal_width_crossover import crossover_legal_width

    left = LegalWidthGenotype(
        {"d0": 0, "d1": 1, "d2": 2},
        {"pg0": "FP16", "pg1": "INT8"},
    )
    right = LegalWidthGenotype(
        {"d0": 3, "d1": 2, "d2": 1},
        {"pg0": "INT8", "pg1": "FP16"},
    )
    child = crossover_legal_width(left, right, rng=random.Random(9))

    assert all(
        child.width_genes[key] in {left.width_genes[key], right.width_genes[key]}
        for key in child.width_genes
    )
    assert all(
        child.precision_genes[key]
        in {left.precision_genes[key], right.precision_genes[key]}
        for key in child.precision_genes
    )


def test_width_only_crossover_does_not_change_precision_profile() -> None:
    from search.encoding.legal_width_genotype import LegalWidthGenotype
    from search.operators.legal_width_crossover import crossover_width_only

    left = LegalWidthGenotype({"d": 0}, {"pg": "FP16"})
    right = LegalWidthGenotype({"d": 1}, {"pg": "INT8"})
    child = crossover_width_only(left, right, rng=random.Random(2))

    assert child.precision_genes == left.precision_genes

