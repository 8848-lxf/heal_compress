"""HEAL LiDAR model-family contracts used by search integration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class HEALLidarFamilySpec:
    name: str
    aliases: tuple[str, ...]
    fusion_kind: str
    export_recipe: str
    weighted_fusion: bool
    default_config: Path
    default_checkpoint: Path | None
    output_names: tuple[str, ...] = (
        "cls_preds",
        "reg_preds",
        "dir_preds",
    )
    fixed_k: int = 29696
    plugin_boundary_dtype: str = "fp32"
    scatter_is_precision_gene: bool = False
    compatibility_module: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["default_config"] = str(self.default_config)
        payload["default_checkpoint"] = (
            str(self.default_checkpoint)
            if self.default_checkpoint is not None
            else None
        )
        return payload


__all__ = ["HEALLidarFamilySpec"]

