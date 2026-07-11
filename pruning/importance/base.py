"""Common interfaces for pruning-unit importance implementations."""

from __future__ import annotations

from typing import Protocol, Sequence

import torch.nn as nn

from ..config import ImportanceConfig
from ..types import ImportanceResult


class ImportanceScorer(Protocol):
    """Protocol implemented by formal importance scorers."""

    def __call__(
        self,
        model: nn.Module,
        units: Sequence[object],
        *,
        config: ImportanceConfig,
    ) -> ImportanceResult:
        """Return raw and normalized scores for immutable pruning units."""


__all__ = ["ImportanceScorer"]
