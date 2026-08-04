"""Portable configuration loading and fail-closed protocol validation."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping

from .families import FamilySpec, get_family


PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PLACEHOLDER = re.compile(r"\$\{([A-Z][A-Z0-9_]*)\}")


@dataclass(frozen=True)
class ResolvedSearchConfig:
    source: Path
    root: Path
    family: FamilySpec
    payload: dict[str, Any]
    unresolved_placeholders: tuple[str, ...]


def _read_mapping(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
    else:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("PyYAML is required to load YAML search configs") from exc
        value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, Mapping):
        raise TypeError(f"search_config_must_be_mapping:{path}")
    return dict(value)


def _expand(value: Any, environment: Mapping[str, str], unresolved: set[str]) -> Any:
    if isinstance(value, str):
        def replace(match: re.Match[str]) -> str:
            name = match.group(1)
            replacement = environment.get(name)
            if replacement is None:
                unresolved.add(name)
                return match.group(0)
            return replacement

        return _PLACEHOLDER.sub(replace, value)
    if isinstance(value, list):
        return [_expand(item, environment, unresolved) for item in value]
    if isinstance(value, tuple):
        return tuple(_expand(item, environment, unresolved) for item in value)
    if isinstance(value, Mapping):
        return {
            str(key): _expand(item, environment, unresolved)
            for key, item in value.items()
        }
    return value


def _set_path(payload: dict[str, Any], dotted_key: str, value: Any) -> None:
    current = payload
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        child = current.setdefault(part, {})
        if not isinstance(child, dict):
            raise TypeError(f"config_override_parent_not_mapping:{dotted_key}")
        current = child
    current[parts[-1]] = value


def _validate_protocol(payload: dict[str, Any], *, allow_unresolved: bool) -> FamilySpec:
    model = dict(payload.get("model", {}) or {})
    family = get_family(str(model.get("family_id", "lidar_pyramid")))
    search = dict(payload.get("search", {}) or {})
    method = str(search.get("method", "ga")).lower()
    if method not in {"ga", "greedy"}:
        raise ValueError(f"unsupported_search_method:{method}")

    proxy = dict(payload.get("proxy", {}) or {})
    activation_enabled = bool(
        proxy.get(
            "include_activation_taylor",
            proxy.get("activation_taylor_included", False),
        )
    )
    proxy["include_activation_taylor"] = activation_enabled
    proxy["activation_taylor_included"] = activation_enabled
    proxy["objective_mode"] = (
        "joint_weight_activation_taylor_hard_bops"
        if activation_enabled
        else "joint_weight_taylor_hard_bops"
    )
    payload["proxy"] = proxy

    search.setdefault("show_progress", True)
    search.setdefault("protocol", "strict_stage12_v3")
    search.setdefault("budget_recovery_beam_width", 8)
    search.setdefault("budget_recovery_seed_pool_size", 32)
    search.setdefault("budget_recovery_max_depth", 64)
    if int(search["budget_recovery_beam_width"]) <= 0:
        raise ValueError("budget_recovery_beam_width_must_be_positive")
    if int(search["budget_recovery_seed_pool_size"]) < int(
        search["budget_recovery_beam_width"]
    ):
        raise ValueError("budget_recovery_seed_pool_smaller_than_beam")
    if int(search["budget_recovery_max_depth"]) <= 0:
        raise ValueError("budget_recovery_max_depth_must_be_positive")
    if method == "ga" and str(search["protocol"]) == "strict_stage12_v3":
        expected = {
            "initial_population_size": 64,
            "population_size": 64,
            "offspring_size": 64,
            "topk_stage2": 5,
        }
        for key, required in expected.items():
            if int(search.get(key, required)) != required:
                raise ValueError(f"strict_stage12_v3_{key}_must_equal_{required}")
        if int(search.get("generations_per_round", 10)) not in {5, 10}:
            raise ValueError("strict_stage12_v3_generations_must_equal_5_or_10")
    payload["search"] = search

    stage2 = dict(payload.get("stage2", {}) or {})
    gate = dict(stage2.get("greedy_anchor_accuracy_gate", {}) or {})
    gate.setdefault("enabled", method == "ga")
    gate.setdefault("tolerance", 0.005)
    tolerance = float(gate["tolerance"])
    if tolerance < 0.0 or tolerance > 0.005:
        raise ValueError(
            "greedy_anchor_accuracy_tolerance_must_be_in_closed_interval_0_0_005"
        )
    integrated_anchor = bool(search.get("integrated_greedy_anchor", False))
    if (
        gate["enabled"]
        and method == "ga"
        and not integrated_anchor
        and not gate.get("anchor_manifest")
    ):
        if not allow_unresolved:
            raise ValueError("greedy_anchor_manifest_required_for_formal_ga")
    stage2["greedy_anchor_accuracy_gate"] = gate
    payload["stage2"] = stage2

    if family.family_id == "heal_lidar_v2xvit" and not allow_unresolved:
        if not stage2.get("evaluation_manifest"):
            raise ValueError("v2xvit_evaluation_manifest_required_for_formal_search")

    output = dict(payload.get("output", {}) or {})
    output.setdefault("best_engine_dir", "best_engines")
    output.setdefault("best_engine_publish_mode", "hardlink")
    payload["output"] = output
    return family


def load_search_config(
    path: str | Path,
    *,
    overrides: Mapping[str, Any] | None = None,
    environment: Mapping[str, str] | None = None,
    allow_unresolved: bool = False,
) -> ResolvedSearchConfig:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"search_config_missing:{source}")
    payload = deepcopy(_read_mapping(source))
    for key, value in (overrides or {}).items():
        if value is not None:
            _set_path(payload, str(key), value)
    unresolved: set[str] = set()
    payload = _expand(payload, environment or os.environ, unresolved)
    family = _validate_protocol(payload, allow_unresolved=allow_unresolved)
    if unresolved and not allow_unresolved:
        raise ValueError("unresolved_config_placeholders:" + ",".join(sorted(unresolved)))
    return ResolvedSearchConfig(
        source=source,
        root=PROJECT_ROOT,
        family=family,
        payload=payload,
        unresolved_placeholders=tuple(sorted(unresolved)),
    )


__all__ = ["PROJECT_ROOT", "ResolvedSearchConfig", "load_search_config"]
