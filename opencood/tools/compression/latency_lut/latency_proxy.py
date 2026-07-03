from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .calibration import IdentityCalibrationModel
from .channel_resolver import ChannelResolver, ResolvedUnitConfig
from .full_engine_precision import assignment_for_unit
from .key_builder import build_boundary_key
from .lut_database import LatencyEstimateItem, LatencyLUTDatabase
from .schema import DEPLOY_MODE, FIXED_K, LatencyLUTKey


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
    unsupported_precision_regions: list[dict[str, Any]] = field(default_factory=list)
    precision_resolution_changes: list[dict[str, Any]] = field(default_factory=list)
    interpolated_keys: list[dict[str, Any]] = field(default_factory=list)
    calibration_model: str = "identity"
    decomposition: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = {
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
            "unsupported_precision_regions": self.unsupported_precision_regions,
            "precision_resolution_changes": self.precision_resolution_changes,
            "interpolated_keys": self.interpolated_keys,
            "calibration_model": self.calibration_model,
            "calibrator_model": self.calibration_model,
            "calibrator_validated": False,
            "ga_integration_allowed": False,
            "calibration_status": "preliminary",
            "readiness_gate_reason": "readiness_gate_not_satisfied",
        }
        data.update(self.decomposition)
        return data


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
        decomposition = self._decomposition(unit_items, plugin_items, boundary_items, missing, unavailable)
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
            decomposition=decomposition,
        )

    def _key_from_unit(self, unit: ResolvedUnitConfig) -> LatencyLUTKey:
        precision = assignment_for_unit(unit.unit_id, unit.precision)
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
            precision_profile=precision.precision_profile,
            weight_precision=precision.weight_precision,
            activation_precision=precision.activation_precision,
            compute_precision=precision.compute_precision,
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

    def _decomposition(
        self,
        unit_items: list[LatencyEstimateItem],
        plugin_items: list[LatencyEstimateItem],
        boundary_items: list[LatencyEstimateItem],
        missing: list[dict[str, Any]],
        unavailable: list[dict[str, Any]],
    ) -> dict[str, Any]:
        def precision_name(profile: str | None) -> str:
            if profile == "TRT_FP32":
                return "FP32"
            if profile == "TRT_INT8_QDQ":
                return "INT8"
            return "FP16"

        def unit_type(key: LatencyLUTKey) -> str:
            if key.plugin_flag or key.block_type == "plugin":
                return "plugin"
            if key.module_name == "shrink":
                return "shrink"
            if key.module_name == "pyramid_fusion":
                return "fusion"
            if key.module_name == "detection_head":
                return "head"
            if key.block_type in {"pfn_block", "gemm"}:
                return "gemm"
            if key.block_type == "precision_boundary":
                if key.src_precision == "TRT_INT8_QDQ" or key.dst_precision == "TRT_INT8_QDQ":
                    return "qdq_boundary"
                return "cast_boundary"
            if key.block_type in {"fusion_merge", "concat", "merge"}:
                return "add_concat_merge"
            if key.block_type == "grid_sample":
                return "grid_sample"
            return "conv_bn_act"

        by_precision = {"FP32": 0.0, "FP16": 0.0, "INT8": 0.0}
        by_unit_type = {
            "conv_bn_act": 0.0,
            "gemm": 0.0,
            "shrink": 0.0,
            "fusion": 0.0,
            "head": 0.0,
            "plugin": 0.0,
            "grid_sample": 0.0,
            "add_concat_merge": 0.0,
            "cast_boundary": 0.0,
            "qdq_boundary": 0.0,
            "fixed_overhead": 0.0,
        }
        by_stage: dict[str, float] = {}
        exact = 0
        coarse = 0
        missing_count = 0
        unavailable_count = 0
        coarse_keys: list[dict[str, Any]] = []

        compute = 0.0
        for item in unit_items:
            if item.match_type in {"exact", "nearest", "interpolate"}:
                compute += float(item.latency_ms)
            by_precision[precision_name(item.key.precision_profile)] += float(item.latency_ms)
            kind = unit_type(item.key)
            by_unit_type[kind] = by_unit_type.get(kind, 0.0) + float(item.latency_ms)
            stage = f"{item.key.module_name}.{item.key.block_name}"
            by_stage[stage] = by_stage.get(stage, 0.0) + float(item.latency_ms)
            if item.match_type == "exact":
                exact += 1
            elif item.match_type in {"nearest", "interpolate"}:
                coarse += 1
                coarse_keys.append(item.to_dict())
            elif item.match_type == "unavailable":
                unavailable_count += 1
            else:
                missing_count += 1

        plugin = 0.0
        for item in plugin_items:
            plugin += float(item.latency_ms)
            by_unit_type["plugin"] += float(item.latency_ms)
            if item.match_type == "exact":
                exact += 1
            elif item.match_type in {"nearest", "interpolate"}:
                coarse += 1
                coarse_keys.append(item.to_dict())
            elif item.match_type == "unavailable":
                unavailable_count += 1
            else:
                missing_count += 1

        cast = 0.0
        qdq = 0.0
        for item in boundary_items:
            if item.key.src_precision == item.key.dst_precision:
                exact += 1 if item.match_type == "exact" else 0
                continue
            if item.key.src_precision == "TRT_INT8_QDQ" or item.key.dst_precision == "TRT_INT8_QDQ":
                qdq += float(item.latency_ms)
                by_unit_type["qdq_boundary"] += float(item.latency_ms)
            else:
                cast += float(item.latency_ms)
                by_unit_type["cast_boundary"] += float(item.latency_ms)
            if item.match_type == "exact":
                exact += 1
            elif item.match_type in {"nearest", "interpolate", "default"}:
                coarse += 1
                if item.match_type != "exact":
                    coarse_keys.append(item.to_dict())

        total_units = len(unit_items) + len(plugin_items)
        covered_units = sum(1 for item in unit_items + plugin_items if item.match_type in {"exact", "nearest", "interpolate"})
        raw = compute + cast + qdq + plugin
        return {
            "T_compute_covered": compute,
            "T_boundary_cast": cast,
            "T_boundary_qdq": qdq,
            "T_plugin_or_scatter": plugin,
            "T_grid_sample_or_geometry": 0.0,
            "T_elementwise_merge": 0.0,
            "T_memory_reformat": 0.0,
            "T_fixed_overhead": 0.0,
            "T_uncovered_est": 0.0,
            "T_lut_by_precision": by_precision,
            "T_lut_by_unit_type": by_unit_type,
            "T_lut_by_stage": by_stage,
            "covered_unit_count": covered_units,
            "missing_unit_count": len(missing),
            "coverage_ratio_by_units": float(covered_units / total_units) if total_units else 1.0,
            "coverage_ratio_by_estimated_latency": 1.0 if raw > 0.0 and not missing and not unavailable else 0.0,
            "exact_key_count": exact,
            "coarse_key_count": coarse,
            "missing_key_count": len(missing) + missing_count,
            "unavailable_key_count": len(unavailable) + unavailable_count,
            "coarse_keys_used": coarse_keys,
            "coverage_report": {
                "covered_unit_count": covered_units,
                "total_unit_count": total_units,
                "coverage_ratio_by_units": float(covered_units / total_units) if total_units else 1.0,
                "missing_key_count": len(missing) + missing_count,
                "unavailable_key_count": len(unavailable) + unavailable_count,
            },
            "source_by_component": {
                "T_compute_covered": "measured_lut",
                "T_boundary_cast": "measured_or_estimated",
                "T_boundary_qdq": "measured_or_estimated",
                "T_plugin_or_scatter": "measured_or_estimated",
                "T_grid_sample_or_geometry": "missing",
                "T_elementwise_merge": "missing",
                "T_memory_reformat": "estimated_overhead",
                "T_fixed_overhead": "estimated_overhead",
            },
        }
