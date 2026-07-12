"""GA parent selection."""

from __future__ import annotations

import random
from typing import Any

from ..candidate import CandidateGenotype


def tournament_select(scored: list[tuple[CandidateGenotype, float, dict[str, Any]]], rng: random.Random, k: int = 3) -> CandidateGenotype:
    candidates = rng.sample(scored, min(k, len(scored)))
    return min(candidates, key=lambda row: row[1])[0]
