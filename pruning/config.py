"""Formal structured pruning configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


SUPPORTED_ROUND_TO = {4, 8, 16, 32, 64, 128}


def validate_round_to(value: int) -> int:
    round_to = int(value)
    if round_to not in SUPPORTED_ROUND_TO or round_to <= 0 or (round_to & (round_to - 1)) != 0:
        raise ValueError(f"round_to must be one of {sorted(SUPPORTED_ROUND_TO)} and a power of two, got {round_to}")
    return round_to


@dataclass(frozen=True)
class PruningConfig:
    target_pruning_ratio: float
    target_pruning_mode: Literal["param", "channel"] = "param"
    round_to: int = 4
    max_ch_sparsity: float = 0.60
    stage1_min_per_group: int = 8
    stage1_max_ch_sparsity: float = 0.30
    protect_fpn_output: bool = True
    protect_head_output: bool = True
    no_extra_output_protection: bool = True
    fixed_shape_structural_skip: bool = True
    importance: Literal["first_order_taylor"] = "first_order_taylor"
    selector: Literal["greedy_global_ranking"] = "greedy_global_ranking"

    def __post_init__(self) -> None:
        if self.target_pruning_mode not in {"param", "channel"}:
            raise ValueError(f"unsupported target_pruning_mode: {self.target_pruning_mode}")
        if not 0.0 <= float(self.target_pruning_ratio) <= 1.0:
            raise ValueError(f"target_pruning_ratio must be in [0, 1], got {self.target_pruning_ratio}")
        object.__setattr__(self, "round_to", validate_round_to(self.round_to))
        if int(self.stage1_min_per_group) <= 0:
            raise ValueError("stage1_min_per_group must be positive")
        if not 0.0 <= float(self.max_ch_sparsity) <= 1.0:
            raise ValueError("max_ch_sparsity must be in [0, 1]")
        if not 0.0 <= float(self.stage1_max_ch_sparsity) <= 1.0:
            raise ValueError("stage1_max_ch_sparsity must be in [0, 1]")
        if self.importance != "first_order_taylor":
            raise ValueError(f"unsupported importance: {self.importance}")
        if self.selector != "greedy_global_ranking":
            raise ValueError(f"unsupported selector: {self.selector}")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
