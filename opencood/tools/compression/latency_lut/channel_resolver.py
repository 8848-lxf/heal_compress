from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .schema import DEPLOY_MODE, FIXED_K
from .full_engine_precision import apply_precision_config_to_units


@dataclass
class ResolvedUnitConfig:
    unit_id: str
    module_name: str
    block_name: str
    block_type: str
    H: int | None = None
    W: int | None = None
    C_in: int | None = None
    C_mid: int | None = None
    C_out: int | None = None
    kernel_size: int | tuple[int, ...] | None = None
    stride: int | tuple[int, ...] | None = None
    padding: int | tuple[int, ...] | None = None
    dilation: int | tuple[int, ...] | None = None
    groups: int | None = None
    precision: str = "FP16"
    plugin_name: str | None = None
    plugin_version: str | None = None
    metadata: dict[str, Any] | None = None


class ChannelResolver:
    """Resolve GA candidates into deployment-unit channel configs.

    The current implementation accepts pre-resolved ``units``. Raw
    ``group_mask`` support needs tracer/group metadata from the search runner
    and is intentionally left as an explicit integration point.
    """

    def __init__(self, deployment_units: list[dict[str, Any]] | None = None) -> None:
        self.deployment_units = deployment_units or []

    def resolve(self, candidate_config: dict[str, Any]) -> list[ResolvedUnitConfig]:
        deploy_mode = candidate_config.get("deploy_mode", DEPLOY_MODE)
        fixed_k = int(candidate_config.get("fixed_K", FIXED_K))
        if deploy_mode != DEPLOY_MODE or fixed_k != FIXED_K:
            raise ValueError(f"LatencyProxy only supports {DEPLOY_MODE} fixedK{FIXED_K}")
        if "units" in candidate_config:
            units = list(candidate_config.get("units") or [])
            if "precision_config" in candidate_config:
                units = apply_precision_config_to_units(units, dict(candidate_config.get("precision_config") or {}))
            return [self._unit_from_dict(item) for item in units]
        if "group_mask" in candidate_config:
            raise NotImplementedError(
                "raw group_mask resolution requires coupled-channel metadata; pass pre-resolved units for now"
            )
        return []

    def _unit_from_dict(self, data: dict[str, Any]) -> ResolvedUnitConfig:
        return ResolvedUnitConfig(
            unit_id=str(data.get("unit_id") or data.get("block_name")),
            module_name=str(data.get("module_name")),
            block_name=str(data.get("block_name") or data.get("unit_id")),
            block_type=str(data.get("block_type")),
            H=data.get("H"),
            W=data.get("W"),
            C_in=data.get("C_in"),
            C_mid=data.get("C_mid"),
            C_out=data.get("C_out"),
            kernel_size=data.get("kernel_size"),
            stride=data.get("stride"),
            padding=data.get("padding"),
            dilation=data.get("dilation"),
            groups=data.get("groups"),
            precision=str(data.get("precision", data.get("precision_profile", data.get("weight_precision", "FP16")))),
            plugin_name=data.get("plugin_name"),
            plugin_version=data.get("plugin_version"),
            metadata=dict(data.get("metadata") or {}),
        )
