"""Stage-2 BOPS computed from physical shapes and realized TRT precision."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, is_dataclass
from typing import Any, Mapping, Sequence

from quantization.tensorrt.layer_info import (
    has_canonical_identity,
    is_weighted_compute_layer,
    load_layer_info,
    precision_name,
)


PRECISION_BITS = {"FP32": 32, "FP16": 16, "INT8": 8}


def _payload(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return dict(value.to_dict())
    if is_dataclass(value):
        return dict(asdict(value))
    raise TypeError(f"unsupported_realized_bops_payload:{type(value).__name__}")


def _snapshot_rows(snapshot: Any) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    payload = _payload(snapshot)
    version = str(payload.get("snapshot_schema_version", ""))
    if version != "physical-structure-snapshot-v2":
        raise ValueError(f"physical_snapshot_schema_invalid:{version or 'missing'}")
    rows = {
        str(row.get("canonical_module_name") or row.get("module_path") or ""): dict(row)
        for row in payload.get("modules", [])
        if isinstance(row, Mapping)
        and str(row.get("canonical_module_name") or row.get("module_path") or "")
    }
    return payload, rows


def engine_realized_precision_profile(
    layer_info: Any,
    canonical_mapping: Any,
) -> dict[str, Any]:
    """Resolve one actual TRT compute precision per canonical entry."""

    layers = load_layer_info(layer_info)
    profile: dict[str, str] = {}
    issues: list[str] = []
    records: list[dict[str, Any]] = []
    for entry in canonical_mapping.entries:
        matches = [
            row
            for row in layers
            if has_canonical_identity(row, entry.canonical_node_name)
            and is_weighted_compute_layer(row)
        ]
        if len(matches) != 1:
            issues.append(
                f"canonical_engine_compute_layer_match_count:{entry.module_path}:{len(matches)}"
            )
            continue
        precision = precision_name(matches[0]).upper()
        if precision not in PRECISION_BITS:
            issues.append(f"canonical_engine_precision_unresolved:{entry.module_path}")
            continue
        profile[str(entry.module_path)] = precision
        records.append(
            {
                "module_path": str(entry.module_path),
                "canonical_node_name": str(entry.canonical_node_name),
                "requested_precision": str(entry.requested_precision).upper(),
                "legalized_precision": str(entry.realized_request_precision).upper(),
                "realized_precision": precision,
            }
        )
    counts = Counter(profile.values())
    return {
        "passed": not issues,
        "status": "passed" if not issues else "precision_realization_failure",
        "realized_precision_profile": dict(sorted(profile.items())),
        "precision_counts": {name: int(counts.get(name, 0)) for name in PRECISION_BITS},
        "records": records,
        "issues": issues,
    }


def _validate_physical_shape(shape: Any, snapshot_row: Mapping[str, Any]) -> None:
    module_type = str(getattr(shape, "module_type", ""))
    if module_type in {"Conv2d", "ConvTranspose2d"}:
        expected = (
            int(snapshot_row.get("in_channels") or 0),
            int(snapshot_row.get("out_channels") or 0),
            int(snapshot_row.get("groups") or 1),
        )
        actual = (
            int(getattr(shape, "c_in", 0) or 0),
            int(getattr(shape, "c_out", 0) or 0),
            int(getattr(shape, "groups", 1) or 1),
        )
    elif module_type == "Linear":
        expected = (
            int(snapshot_row.get("in_features") or 0),
            int(snapshot_row.get("out_features") or 0),
            1,
        )
        actual = (
            int(getattr(shape, "c_in", 0) or 0),
            int(getattr(shape, "c_out", 0) or 0),
            1,
        )
    else:
        return
    if expected != actual:
        raise ValueError(
            f"physical_runtime_shape_mismatch:{shape.module_path}:{actual}!={expected}"
        )


def compute_realized_bops(
    *,
    physical_runtime_shapes: Sequence[Any],
    baseline_runtime_shapes: Sequence[Any],
    realized_precision_profile: Mapping[str, str],
    physical_snapshot: Any,
    baseline_snapshot: Any,
    target_retention: float | None,
    tolerance: float,
) -> dict[str, Any]:
    """Compute final BOPS and enforce the configured fixed budget interval."""

    physical_payload, physical_rows = _snapshot_rows(physical_snapshot)
    baseline_payload, baseline_rows = _snapshot_rows(baseline_snapshot)
    profile = {str(key): str(value).upper() for key, value in realized_precision_profile.items()}
    breakdown: list[dict[str, Any]] = []
    realized_bops = 0.0
    seen_calls: set[tuple[str, int]] = set()
    runtime_modules: set[str] = set()
    for shape in physical_runtime_shapes:
        key = (str(shape.module_path), int(shape.call_index))
        if key in seen_calls:
            continue
        seen_calls.add(key)
        module_path = str(shape.module_path)
        runtime_modules.add(module_path)
        precision = profile.get(module_path, "")
        if precision not in PRECISION_BITS:
            raise ValueError(f"realized_precision_missing:{module_path}")
        snapshot_row = physical_rows.get(module_path)
        if snapshot_row is None:
            raise ValueError(f"physical_snapshot_module_missing:{module_path}")
        _validate_physical_shape(shape, snapshot_row)
        weight_bits = PRECISION_BITS[precision]
        activation_bits = PRECISION_BITS[precision]
        macs = float(shape.macs)
        bops = macs * weight_bits * activation_bits
        realized_bops += bops
        breakdown.append(
            {
                "module_path": module_path,
                "call_index": int(shape.call_index),
                "MACs": macs,
                "C_in": int(shape.c_in or 0),
                "C_out": int(shape.c_out or 0),
                "groups": int(shape.groups or 1),
                "realized_precision": precision,
                "weight_bits": weight_bits,
                "activation_bits": activation_bits,
                "BOPS": bops,
            }
        )
    if not breakdown:
        raise ValueError("realized_bops_runtime_shapes_empty")

    baseline_calls: set[tuple[str, int]] = set()
    fp32_reference_macs = 0.0
    for shape in baseline_runtime_shapes:
        key = (str(shape.module_path), int(shape.call_index))
        if key in baseline_calls:
            continue
        baseline_calls.add(key)
        fp32_reference_macs += float(shape.macs)
    fp32_reference_bops = fp32_reference_macs * 32.0 * 32.0
    if fp32_reference_bops <= 0.0:
        raise ValueError("fp32_reference_bops_empty")

    physical_weight_bits = 0.0
    baseline_weight_bits = 0.0
    for module_path in runtime_modules:
        physical_weight_bits += float(physical_rows[module_path].get("parameter_count", 0)) * PRECISION_BITS[profile[module_path]]
        baseline_row = baseline_rows.get(module_path)
        if baseline_row is None:
            raise ValueError(f"baseline_snapshot_module_missing:{module_path}")
        baseline_weight_bits += float(baseline_row.get("parameter_count", 0)) * 32.0
    if baseline_weight_bits <= 0.0:
        raise ValueError("baseline_weight_storage_empty")

    retention = realized_bops / fp32_reference_bops
    physical_macs = sum(float(row["MACs"]) for row in breakdown)
    int8_macs = sum(
        float(row["MACs"])
        for row in breakdown
        if str(row["realized_precision"]) == "INT8"
    )
    r_mac = physical_macs / fp32_reference_macs
    lower = None if target_retention is None else float(target_retention) - float(tolerance)
    upper = None if target_retention is None else float(target_retention) + float(tolerance)
    passed = target_retention is None or (float(lower) <= retention <= float(upper))
    physical_params = int(physical_payload.get("parameter_count", 0))
    baseline_params = int(baseline_payload.get("parameter_count", 0))
    return {
        "passed": passed,
        "status": "passed" if passed else "realized_BOPS_out_of_budget",
        "target_bops_retention": None if target_retention is None else float(target_retention),
        "tolerance": float(tolerance),
        "legal_interval": None if target_retention is None else [lower, upper],
        "realized_bops": realized_bops,
        "fp32_reference_bops": fp32_reference_bops,
        "bops_retention": retention,
        "R_MAC": r_mac,
        "physical_macs": physical_macs,
        "fp32_reference_macs": fp32_reference_macs,
        "int8_macs": int8_macs,
        "int8_macs_share_full": int8_macs / fp32_reference_macs,
        "int8_macs_share_physical": int8_macs / max(physical_macs, 1.0),
        "physical_params": physical_params,
        "baseline_params": baseline_params,
        "parameter_retention": physical_params / max(baseline_params, 1),
        "weight_storage_retention": physical_weight_bits / baseline_weight_bits,
        "realized_precision_counts": {
            name: sum(1 for module in runtime_modules if profile[module] == name)
            for name in PRECISION_BITS
        },
        "breakdown": breakdown,
    }
