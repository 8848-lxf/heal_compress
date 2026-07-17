"""Registry for isolated HEAL model-family adapters."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

import torch.nn as nn

from .contracts import ModelFamilyAudit


class ModelFamilyProvider(Protocol):
    family_id: str

    def matches(self, config: Mapping[str, Any]) -> bool: ...

    def audit(self, model: nn.Module, config: Mapping[str, Any]) -> ModelFamilyAudit: ...


_PROVIDERS: dict[str, ModelFamilyProvider] = {}


def register_model_family(provider: ModelFamilyProvider) -> None:
    family_id = str(provider.family_id)
    existing = _PROVIDERS.get(family_id)
    if existing is not None and type(existing) is not type(provider):
        raise RuntimeError(f"duplicate_model_family_provider:{family_id}")
    _PROVIDERS[family_id] = provider


def registered_model_families() -> tuple[str, ...]:
    return tuple(sorted(_PROVIDERS))


def get_model_family(family_id: str) -> ModelFamilyProvider:
    try:
        return _PROVIDERS[str(family_id)]
    except KeyError as exc:
        raise KeyError(f"unknown_model_family:{family_id}") from exc


def detect_model_family(config: Mapping[str, Any]) -> ModelFamilyProvider:
    matches = [provider for provider in _PROVIDERS.values() if provider.matches(config)]
    if not matches:
        raise RuntimeError("no_model_family_provider_matches_config")
    if len(matches) > 1:
        raise RuntimeError(
            "ambiguous_model_family_provider:" + ",".join(sorted(row.family_id for row in matches))
        )
    return matches[0]
