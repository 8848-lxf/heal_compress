from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .schema import precision_to_profile


CANONICAL_PRECISIONS = {"FP32", "FP16", "INT8"}
PRECISION_PROFILE_TO_CANONICAL = {
    "TRT_FP32": "FP32",
    "TRT_FP16": "FP16",
    "TRT_INT8_QDQ": "INT8",
}


@dataclass(frozen=True)
class LayerPrecisionAssignment:
    unit_id: str
    precision: str
    precision_profile: str
    weight_precision: str
    activation_precision: str
    compute_precision: str
    matched_override: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "precision": self.precision,
            "precision_profile": self.precision_profile,
            "weight_precision": self.weight_precision,
            "activation_precision": self.activation_precision,
            "compute_precision": self.compute_precision,
            "matched_override": self.matched_override,
        }


def normalize_precision_label(value: Any) -> str:
    label = str(value).upper()
    if label in CANONICAL_PRECISIONS:
        return label
    if label in PRECISION_PROFILE_TO_CANONICAL:
        return PRECISION_PROFILE_TO_CANONICAL[label]
    raise ValueError(
        "unsupported single layer precision_profile "
        f"{value!r}; expected one of FP32/FP16/INT8 and do not use weight/activation cartesian labels"
    )


def profile_for_precision(precision: str) -> str:
    return precision_to_profile(normalize_precision_label(precision))


def assignment_for_unit(unit_id: str, precision: Any, *, matched_override: str | None = None) -> LayerPrecisionAssignment:
    canonical = normalize_precision_label(precision)
    profile = profile_for_precision(canonical)
    return LayerPrecisionAssignment(
        unit_id=str(unit_id),
        precision=canonical,
        precision_profile=profile,
        weight_precision=canonical,
        activation_precision=canonical,
        compute_precision=canonical,
        matched_override=matched_override,
    )


def split_precision_config(precision_config: dict[str, Any] | None) -> tuple[str, dict[str, Any]]:
    data = dict(precision_config or {})
    default = normalize_precision_label(data.get("default", "FP16"))
    overrides: dict[str, Any] = {}
    nested = data.get("overrides")
    if isinstance(nested, dict):
        overrides.update(nested)
    for key, value in data.items():
        if key in {"default", "overrides"}:
            continue
        overrides[str(key)] = value
    return default, overrides


def _unit_id(data: dict[str, Any]) -> str:
    return str(data.get("unit_id") or data.get("block_name") or data.get("module_name") or "")


def _unit_match_candidates(unit: dict[str, Any]) -> list[str]:
    unit_id = _unit_id(unit)
    module = str(unit.get("module_name") or "")
    block = str(unit.get("block_name") or "")
    candidates = [unit_id]
    if module and block:
        candidates.append(f"{module}.{block}")
    if module:
        candidates.append(module)
    if block:
        candidates.append(block)
    return [item for item in candidates if item]


def _match_override(unit: dict[str, Any], overrides: dict[str, Any]) -> tuple[str | None, Any | None]:
    if not overrides:
        return None, None
    unit_id = _unit_id(unit)
    candidates = set(_unit_match_candidates(unit))
    matches: list[tuple[int, str, Any]] = []
    for key, value in overrides.items():
        key_text = str(key)
        if key_text in candidates or (unit_id and unit_id.startswith(key_text + ".")):
            matches.append((len(key_text), key_text, value))
    if not matches:
        return None, None
    _length, key, value = sorted(matches, key=lambda item: (item[0], item[1]))[-1]
    return key, value


def resolve_layer_precision_config(
    precision_config: dict[str, Any] | None,
    deployment_units: list[dict[str, Any]],
) -> dict[str, LayerPrecisionAssignment]:
    default, overrides = split_precision_config(precision_config)
    assignments: dict[str, LayerPrecisionAssignment] = {}
    for unit in deployment_units:
        unit_id = _unit_id(unit)
        matched_key, override_value = _match_override(unit, overrides)
        precision = override_value if override_value is not None else default
        assignments[unit_id] = assignment_for_unit(unit_id, precision, matched_override=matched_key)
    return assignments


def apply_precision_config_to_units(
    deployment_units: list[dict[str, Any]],
    precision_config: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    assignments = resolve_layer_precision_config(precision_config, deployment_units)
    updated: list[dict[str, Any]] = []
    for unit in deployment_units:
        item = dict(unit)
        unit_id = _unit_id(item)
        assignment = assignments[unit_id]
        item.update(
            {
                "precision": assignment.precision,
                "precision_profile": assignment.precision_profile,
                "weight_precision": assignment.weight_precision,
                "activation_precision": assignment.activation_precision,
                "compute_precision": assignment.compute_precision,
            }
        )
        updated.append(item)
    return updated


def normalized_precision_set(precision_config: dict[str, Any] | None) -> set[str]:
    default, overrides = split_precision_config(precision_config)
    values = {default}
    values.update(normalize_precision_label(value) for value in overrides.values())
    return values
