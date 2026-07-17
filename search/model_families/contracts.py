"""Stable contracts shared by model-family search recipes."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Protocol


class SearchRunner(Protocol):
    """Minimal runner surface consumed by the search CLI."""

    def run(self, **kwargs: Any) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class ModelFamilyCapabilityManifest:
    """Immutable identity for a model-family deployment capability."""

    model_family: str
    recipe_version: str
    checkpoint_sha256: str
    config_sha256: str
    capabilities: Mapping[str, Any]

    @property
    def canonical_json(self) -> str:
        return json.dumps(
            {
                "capabilities": self.capabilities,
                "checkpoint_sha256": self.checkpoint_sha256,
                "config_sha256": self.config_sha256,
                "model_family": self.model_family,
                "recipe_version": self.recipe_version,
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json.encode("utf-8")).hexdigest()

