from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .schema import DEPLOY_MODE, FIXED_K, LatencyLUTKey, precision_to_profile, profile_to_weight_precision

KEEP_RATIOS = [1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25]
BOUNDARY_BLOCK_TYPE = "precision_boundary"


@dataclass
class DeploymentUnit:
    unit_id: str
    module_name: str
    block_type: str
    block_name: str | None = None
    input_shape: list[int] | None = None
    output_shape: list[int] | None = None
    layer_names: list[str] = field(default_factory=list)
    fusable_pattern: str | None = None
    precision_profile: str | None = None
    H: int | None = None
    W: int | None = None
    C_base: int | None = None
    C_in_base: int | None = None
    C_mid_base: int | None = None
    C_out_base: int | None = None
    kernel_size: int | tuple[int, ...] | None = None
    stride: int | tuple[int, ...] | None = None
    padding: int | tuple[int, ...] | None = None
    dilation: int | tuple[int, ...] | None = None
    groups: int | None = None
    fixed_K: int = FIXED_K
    plugin_name: str | None = None
    plugin_version: str | None = None
    branch: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DeploymentUnit":
        payload = dict(data)
        known = {field_name for field_name in cls.__dataclass_fields__}
        metadata = {k: payload.pop(k) for k in list(payload.keys()) if k not in known}
        payload.setdefault("block_name", payload.get("unit_id"))
        payload["metadata"] = {**metadata, **dict(payload.get("metadata") or {})}
        return cls(**payload)

    @property
    def block(self) -> str:
        return self.block_name or self.unit_id

    def base_channels(self) -> int:
        for value in (self.C_base, self.C_out_base, self.C_mid_base, self.C_in_base):
            if value is not None:
                return int(value)
        return 1


def _align_down(value: int, align: int) -> int:
    if align <= 1:
        return int(value)
    return max(align, int(value) - (int(value) % int(align)))


def build_channel_grid(
    base_channels: int,
    keep_ratios: list[float] | tuple[float, ...] = KEEP_RATIOS,
    min_keep_ratio: float = 0.25,
    align: int = 16,
    reachable_channels: list[int] | None = None,
) -> list[int]:
    base = int(base_channels)
    if base <= 0:
        return []
    min_keep = max(1, int(round(base * float(min_keep_ratio))))
    values: set[int] = set()
    for ratio in keep_ratios:
        if float(ratio) < float(min_keep_ratio):
            continue
        raw = max(min_keep, min(base, int(round(base * float(ratio)))))
        aligned = _align_down(raw, int(align))
        if aligned < min_keep:
            aligned = min(base, max(min_keep, int(align) if base >= int(align) else base))
            if align > 1 and aligned % align != 0 and base >= align:
                aligned = min(base, _align_down(aligned + align, align))
        values.add(min(base, aligned))
    values.add(base)
    if reachable_channels:
        for value in reachable_channels:
            ivalue = int(value)
            if min_keep <= ivalue <= base:
                values.add(ivalue)
    return sorted(values)


def _sample_pairs(grid: list[int]) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for value in grid:
        pairs.append((value, value))
    if len(grid) >= 2:
        pairs.append((grid[-1], grid[0]))
        pairs.append((grid[0], grid[-1]))
    if len(grid) >= 3:
        pairs.append((grid[-1], grid[len(grid) // 2]))
        pairs.append((grid[len(grid) // 2], grid[-1]))
    deduped = []
    seen = set()
    for pair in pairs:
        if pair in seen:
            continue
        seen.add(pair)
        deduped.append(pair)
    return deduped


def _profile_fields(profile: str) -> tuple[str, str, str]:
    precision_profile = precision_to_profile(profile)
    weight = profile_to_weight_precision(precision_profile)
    if precision_profile == "TRT_FP32":
        return weight, "high_precision", "FP32"
    if precision_profile == "TRT_FP16":
        return weight, "FP16", "FP16"
    return weight, "FP16", "INT8"


def _key(
    unit: DeploymentUnit,
    *,
    deploy_mode: str,
    fixed_K: int,
    profile: str,
    c_in: int | None = None,
    c_mid: int | None = None,
    c_out: int | None = None,
    block_type: str | None = None,
    block_name: str | None = None,
    plugin_flag: bool | None = None,
    metadata: dict[str, Any] | None = None,
) -> LatencyLUTKey:
    weight, activation, compute = _profile_fields(profile)
    return LatencyLUTKey(
        deploy_mode=deploy_mode,
        fixed_K=fixed_K,
        module_name=unit.module_name,
        block_name=block_name or unit.block,
        block_type=block_type or unit.block_type,
        H=unit.H,
        W=unit.W,
        C_in=c_in,
        C_mid=c_mid,
        C_out=c_out,
        kernel_size=unit.kernel_size,
        stride=unit.stride,
        padding=unit.padding if unit.padding is not None else (unit.kernel_size // 2 if isinstance(unit.kernel_size, int) and unit.kernel_size > 1 else 0),
        dilation=unit.dilation,
        groups=unit.groups,
        batch_size=1,
        precision_profile=profile,
        weight_precision=weight,
        activation_precision=activation,
        compute_precision=compute,
        plugin_flag=bool(unit.block_type == "plugin") if plugin_flag is None else bool(plugin_flag),
        plugin_name=unit.plugin_name,
        plugin_version=unit.plugin_version,
        metadata={**unit.metadata, **dict(metadata or {})},
    )


def _unit_keys(
    unit: DeploymentUnit,
    *,
    grid: list[int],
    precision_profiles: list[str],
    deploy_mode: str,
    fixed_K: int,
) -> list[LatencyLUTKey]:
    keys: list[LatencyLUTKey] = []
    block_type = unit.block_type
    for profile in precision_profiles:
        if block_type == "plugin":
            c = unit.C_base or unit.C_out_base or unit.C_in_base
            keys.append(_key(unit, deploy_mode=deploy_mode, fixed_K=fixed_K, profile=profile, c_in=c, c_out=c))
        elif block_type == "pfn_block":
            point_feature_dim = int(unit.metadata.get("point_feature_dim") or unit.C_in_base or unit.C_base or unit.C_out_base or 1)
            c_out = int(unit.C_out_base or unit.C_base or point_feature_dim)
            keys.append(
                _key(
                    unit,
                    deploy_mode=deploy_mode,
                    fixed_K=fixed_K,
                    profile=profile,
                    c_in=point_feature_dim,
                    c_out=c_out,
                    metadata={"point_feature_dim": point_feature_dim},
                )
            )
        elif block_type in {"residual_block"}:
            for c in grid:
                mid = unit.C_mid_base or c
                if unit.C_mid_base:
                    mid_grid = build_channel_grid(int(unit.C_mid_base), KEEP_RATIOS, min_keep_ratio=0.25, align=8)
                    for c_mid in sorted({mid, mid_grid[0], mid_grid[-1]}):
                        keys.append(_key(unit, deploy_mode=deploy_mode, fixed_K=fixed_K, profile=profile, c_in=c, c_mid=c_mid, c_out=c))
                else:
                    keys.append(_key(unit, deploy_mode=deploy_mode, fixed_K=fixed_K, profile=profile, c_in=c, c_mid=mid, c_out=c))
        elif block_type in {"fusion_block"}:
            for c_ego, c_infra in _sample_pairs(grid):
                c_fused = max(c_ego, c_infra)
                keys.append(
                    _key(
                        unit,
                        deploy_mode=deploy_mode,
                        fixed_K=fixed_K,
                        profile=profile,
                        c_in=c_ego + c_infra,
                        c_mid=c_fused,
                        c_out=c_fused,
                        metadata={"C_ego": c_ego, "C_infra": c_infra, "C_fused": c_fused},
                    )
                )
        elif block_type in {"head_branch"}:
            fixed_out = unit.C_out_base
            for c in grid:
                keys.append(_key(unit, deploy_mode=deploy_mode, fixed_K=fixed_K, profile=profile, c_in=c, c_out=fixed_out))
        else:
            for c_in, c_out in _sample_pairs(grid):
                keys.append(_key(unit, deploy_mode=deploy_mode, fixed_K=fixed_K, profile=profile, c_in=c_in, c_out=c_out))
    return keys


def build_boundary_key(
    *,
    src_precision: str,
    dst_precision: str,
    H: int | None,
    W: int | None,
    C: int | None,
    fixed_K: int = FIXED_K,
    deploy_mode: str = DEPLOY_MODE,
    batch_size: int = 1,
    default_profile: str = "TRT_FP16",
) -> LatencyLUTKey:
    src_profile = precision_to_profile(src_precision)
    dst_profile = precision_to_profile(dst_precision)
    weight, activation, compute = _profile_fields(default_profile)
    return LatencyLUTKey(
        deploy_mode=deploy_mode,
        fixed_K=fixed_K,
        module_name="precision_boundary",
        block_name=f"{src_profile}_to_{dst_profile}",
        block_type=BOUNDARY_BLOCK_TYPE,
        H=H,
        W=W,
        C_in=C,
        C_out=C,
        batch_size=batch_size,
        precision_profile=default_profile,
        weight_precision=weight,
        activation_precision=activation,
        compute_precision=compute,
        src_precision=src_profile,
        dst_precision=dst_profile,
        tensor_dtype_before=src_profile,
        tensor_dtype_after=dst_profile,
        metadata={"zero_latency": src_profile == dst_profile},
    )


def build_lut_keys_for_units(
    units: list[DeploymentUnit],
    *,
    keep_ratios: list[float] | tuple[float, ...] = KEEP_RATIOS,
    min_keep_ratio: float = 0.25,
    channel_align: int = 16,
    precision_profiles: list[str] | tuple[str, ...] = ("TRT_FP32", "TRT_FP16", "TRT_INT8_QDQ"),
    deploy_mode: str = DEPLOY_MODE,
    fixed_K: int = FIXED_K,
    include_boundaries: bool = True,
) -> list[LatencyLUTKey]:
    if deploy_mode != DEPLOY_MODE or int(fixed_K) != FIXED_K:
        raise ValueError(f"latency LUT only supports {DEPLOY_MODE} fixedK{FIXED_K}")
    keys: list[LatencyLUTKey] = []
    for unit in units:
        grid = build_channel_grid(
            unit.base_channels(),
            list(keep_ratios),
            min_keep_ratio=float(min_keep_ratio),
            align=int(channel_align),
            reachable_channels=unit.metadata.get("reachable_channels") if isinstance(unit.metadata, dict) else None,
        )
        keys.extend(_unit_keys(unit, grid=grid, precision_profiles=list(precision_profiles), deploy_mode=deploy_mode, fixed_K=int(fixed_K)))
    if include_boundaries:
        for unit in units:
            c = unit.C_base or unit.C_out_base or unit.C_in_base
            for src in precision_profiles:
                for dst in precision_profiles:
                    if src == dst:
                        continue
                    keys.append(build_boundary_key(src_precision=src, dst_precision=dst, H=unit.H, W=unit.W, C=c, fixed_K=int(fixed_K), deploy_mode=deploy_mode))
    deduped: dict[str, LatencyLUTKey] = {}
    for key in keys:
        deduped[key.stable_hash()] = key
    return list(deduped.values())


def load_units_from_config(path: str | Path) -> tuple[list[DeploymentUnit], dict[str, Any]]:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    units = [DeploymentUnit.from_dict(item) for item in cfg.get("deployment_units", [])]
    return units, cfg


def collect_key_stats(keys: list[LatencyLUTKey], units: list[DeploymentUnit]) -> dict[str, Any]:
    by_module = Counter(key.module_name for key in keys)
    by_precision = Counter(key.precision_profile for key in keys)
    return {
        "num_units": len(units),
        "num_keys_total": len(keys),
        "num_keys_by_module": dict(sorted(by_module.items())),
        "num_keys_by_precision": dict(sorted(by_precision.items())),
        "num_plugin_keys": sum(1 for key in keys if key.plugin_flag),
        "num_boundary_keys": sum(1 for key in keys if key.block_type == BOUNDARY_BLOCK_TYPE),
    }


def auto_parse_deployment_units_from_model(*_args: Any, **_kwargs: Any) -> list[DeploymentUnit]:
    raise NotImplementedError("automatic HEAL/OpenCOOD graph parsing is not connected yet; use deployment_units config")
