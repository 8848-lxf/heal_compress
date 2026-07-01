from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .calibration import IdentityCalibrationModel
from .channel_resolver import ChannelResolver, ResolvedUnitConfig
from .key_builder import build_boundary_key
from .lut_database import LatencyEstimateItem, LatencyLUTDatabase
from .schema import DEPLOY_MODE, FIXED_K, LatencyLUTKey, precision_to_profile, profile_to_weight_precision


@dataclass
class LatencyEstimate:
    latency_ms: float
    latency_lut_raw_ms: float
    latency_calibrated_ms: float
    uncertainty_ms: float
    unit_items: list[LatencyEstimateItem] = field(default_factory=list)
    boundary_items: list[LatencyEstimateItem] = field(default_factory=list)
    plugin_items: list[LatencyEstimateItem] = field(default_factory=list)
    calibration_features: dict[str, Any] = field(default_factory=dict)
    missing_keys: list[dict[str, Any]] = field(default_factory=list)
    unavailable_keys: list[dict[str, Any]] = field(default_factory=list)
    interpolated_keys: list[dict[str, Any]] = field(default_factory=list)
    calibration_model: str = "identity"

    def to_dict(self) -> dict[str, Any]:
        return {
            "T_lut_raw": self.latency_lut_raw_ms,
            "T_calibrated": self.latency_calibrated_ms,
            "T_proxy": self.latency_ms,
            "uncertainty": self.uncertainty_ms,
            "latency_ms": self.latency_ms,
            "latency_lut_raw_ms": self.latency_lut_raw_ms,
            "latency_calibrated_ms": self.latency_calibrated_ms,
            "uncertainty_ms": self.uncertainty_ms,
            "unit_items": [item.to_dict() for item in self.unit_items],
            "boundary_items": [item.to_dict() for item in self.boundary_items],
            "plugin_items": [item.to_dict() for item in self.plugin_items],
            "calibration_features": self.calibration_features,
            "missing_keys": self.missing_keys,
            "unavailable_keys": self.unavailable_keys,
            "interpolated_keys": self.interpolated_keys,
            "calibration_model": self.calibration_model,
        }


class LatencyProxy:
    def __init__(
        self,
        lut_database: LatencyLUTDatabase,
        *,
        channel_resolver: ChannelResolver | None = None,
        calibration_model: Any | None = None,
        kappa: float = 1.0,
        prefer_interpolation: bool = True,
        strict_latency: bool = False,
    ) -> None:
        self.lut_database = lut_database
        self.channel_resolver = channel_resolver or ChannelResolver()
        self.calibration_model = calibration_model or IdentityCalibrationModel()
        self.kappa = float(kappa)
        self.prefer_interpolation = bool(prefer_interpolation)
        self.strict_latency = bool(strict_latency)

    def estimate(self, candidate_config: dict[str, Any]) -> LatencyEstimate:
        deploy_mode = candidate_config.get("deploy_mode", DEPLOY_MODE)
        fixed_k = int(candidate_config.get("fixed_K", FIXED_K))
        if deploy_mode != DEPLOY_MODE or fixed_k != FIXED_K:
            raise ValueError(f"LatencyProxy only supports {DEPLOY_MODE} fixedK{FIXED_K}")
        units = self.channel_resolver.resolve(candidate_config)
        estimated_items = [self.lut_database.estimate_key(self._key_from_unit(unit), prefer_interpolation=self.prefer_interpolation) for unit in units]
        unit_items = [item for item in estimated_items if not (item.key.plugin_flag or item.key.block_type == "plugin")]
        plugin_items = [item for item in estimated_items if item.key.plugin_flag or item.key.block_type == "plugin"]
        boundary_items = []
        for prev, cur in zip(units, units[1:]):
            boundary_key = build_boundary_key(
                src_precision=prev.precision,
                dst_precision=cur.precision,
                H=cur.H or prev.H,
                W=cur.W or prev.W,
                C=prev.C_out or cur.C_in,
                fixed_K=fixed_k,
                deploy_mode=deploy_mode,
            )
            boundary_items.append(self.lut_database.estimate_key(boundary_key, prefer_interpolation=False))
        all_items = unit_items + plugin_items + boundary_items
        raw = sum(item.latency_ms for item in all_items)
        uncertainty = sum(item.uncertainty_ms for item in all_items)
        missing = [
            item.to_dict()
            for item in all_items
            if item.match_type == "default" and item.key.block_type != "precision_boundary"
        ]
        unavailable = [item.to_dict() for item in all_items if item.match_type == "unavailable"]
        interpolated = [item.to_dict() for item in all_items if item.match_type == "interpolate"]
        if (candidate_config.get("strict_latency", self.strict_latency)) and (missing or unavailable):
            raise RuntimeError(f"LatencyProxy strict mode found missing/unavailable LUT keys: missing={len(missing)}, unavailable={len(unavailable)}")
        features = self._features(units, boundary_items, missing, unavailable)
        calibrated = float(self.calibration_model.predict(raw, features))
        return LatencyEstimate(
            latency_ms=calibrated + self.kappa * uncertainty,
            latency_lut_raw_ms=raw,
            latency_calibrated_ms=calibrated,
            uncertainty_ms=uncertainty,
            unit_items=unit_items,
            boundary_items=boundary_items,
            plugin_items=plugin_items,
            calibration_features=features,
            missing_keys=missing,
            unavailable_keys=unavailable,
            interpolated_keys=interpolated,
            calibration_model=str(getattr(self.calibration_model, "model_type", self.calibration_model.__class__.__name__)),
        )

    def _key_from_unit(self, unit: ResolvedUnitConfig) -> LatencyLUTKey:
        profile = precision_to_profile(unit.precision)
        weight = profile_to_weight_precision(profile)
        compute = "FP32" if profile == "TRT_FP32" else "FP16" if profile == "TRT_FP16" else "INT8"
        activation = "high_precision" if profile == "TRT_FP32" else "FP16"
        return LatencyLUTKey(
            deploy_mode=DEPLOY_MODE,
            fixed_K=FIXED_K,
            module_name=unit.module_name,
            block_name=unit.block_name,
            block_type=unit.block_type,
            H=unit.H,
            W=unit.W,
            C_in=unit.C_in,
            C_mid=unit.C_mid,
            C_out=unit.C_out,
            kernel_size=unit.kernel_size,
            stride=unit.stride,
            padding=unit.padding,
            dilation=unit.dilation,
            groups=unit.groups,
            batch_size=1,
            precision_profile=profile,
            weight_precision=weight,
            activation_precision=activation,
            compute_precision=compute,
            plugin_flag=unit.block_type == "plugin",
            plugin_name=unit.plugin_name,
            plugin_version=unit.plugin_version,
            metadata=unit.metadata or {},
        )

    def _features(
        self,
        units: list[ResolvedUnitConfig],
        boundaries: list[LatencyEstimateItem],
        missing: list[dict[str, Any]] | None = None,
        unavailable: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        precision_switches = sum(1 for item in boundaries if item.key.src_precision != item.key.dst_precision)
        total_activation_bytes = 0
        total_weight_bytes = 0
        for unit in units:
            h, w = int(unit.H or 1), int(unit.W or 1)
            c_in, c_out = int(unit.C_in or 0), int(unit.C_out or unit.C_mid or unit.C_in or 0)
            total_activation_bytes += h * w * max(c_in, c_out) * 2
            k = unit.kernel_size if isinstance(unit.kernel_size, int) else 1
            total_weight_bytes += max(c_in, 1) * max(c_out, 1) * int(k or 1) * int(k or 1) * 2
        return {
            "num_units": len(units),
            "num_precision_switch": precision_switches,
            "num_plugins": sum(1 for unit in units if unit.block_type == "plugin"),
            "total_activation_bytes": total_activation_bytes,
            "total_weight_bytes": total_weight_bytes,
            "channel_alignment_waste": 0,
            "num_concat": sum(1 for unit in units if unit.block_type == "fusion_block"),
            "num_residual": sum(1 for unit in units if unit.block_type == "residual_block"),
            "num_missing_keys": len(missing or []),
            "num_unavailable_keys": len(unavailable or []),
        }
