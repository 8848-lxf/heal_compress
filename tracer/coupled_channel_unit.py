"""Formal coupled-channel unit schema."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class CoupledChannelUnit:
    pruning_domain_id: str
    coupled_unit_id: str
    root_module_name: str
    root_channel_index: int
    importance_raw: float = 0.0
    importance_normalized: float = 0.0
    protected_reason: str = ""
    skipped_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
