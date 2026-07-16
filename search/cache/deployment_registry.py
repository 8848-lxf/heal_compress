"""Protocol-aware deployment and evaluation identity registry."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Mapping

from ..hashing import canonical_json_hash


_DEPLOYMENT_FIELDS = (
    "physical_hash",
    "precision_hash",
    "calibration_signature",
    "build_signature",
)


def _required_text(payload: Mapping[str, Any], fields: tuple[str, ...], label: str) -> None:
    missing = [field for field in fields if not str(payload.get(field, ""))]
    if missing:
        raise ValueError(f"{label}_missing:{','.join(missing)}")


def deployment_identity(payload: Mapping[str, Any]) -> str:
    """Return the engine-reuse identity for one physical precision deployment."""

    _required_text(payload, _DEPLOYMENT_FIELDS, "deployment_identity")
    return canonical_json_hash(
        {field: payload[field] for field in _DEPLOYMENT_FIELDS}
    )


def evaluation_identity(
    deployment_id: str,
    protocol: str,
    manifest_hash: str,
    config_hash: str,
) -> str:
    """Return an evaluation identity that cannot collide across protocols."""

    payload = {
        "deployment_identity": str(deployment_id),
        "protocol": str(protocol),
        "manifest_hash": str(manifest_hash),
        "config_hash": str(config_hash),
    }
    _required_text(
        payload,
        ("deployment_identity", "protocol", "manifest_hash", "config_hash"),
        "evaluation_identity",
    )
    return canonical_json_hash(payload)


class DeploymentRegistry:
    """Append deployment/evaluation artifacts while retaining all lineage uses."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._rows: dict[tuple[str, str], dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.is_file():
            return
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row["kind"]), str(row["identity"]))
            current = self._rows.setdefault(
                key,
                {
                    "kind": key[0],
                    "identity": key[1],
                    "payload": {},
                    "lineage": [],
                },
            )
            current["payload"] = dict(row.get("payload", {}))
            lineage = row.get("lineage")
            if lineage is not None:
                current["lineage"].append(dict(lineage))

    def record(
        self,
        *,
        kind: str,
        identity: str,
        payload: Mapping[str, Any],
        lineage: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        kind_text = str(kind)
        identity_text = str(identity)
        if not kind_text or not identity_text:
            raise ValueError("deployment_registry_kind_and_identity_required")
        normalized_lineage = (
            dict(sorted((str(key), value) for key, value in lineage.items()))
            if lineage is not None
            else None
        )
        record = {
            "kind": kind_text,
            "identity": identity_text,
            "payload": dict(payload),
            "lineage": normalized_lineage,
            "recorded_epoch_seconds": time.time(),
        }
        serialized = json.dumps(record, sort_keys=True, ensure_ascii=True) + "\n"
        descriptor = os.open(
            self.path,
            os.O_APPEND | os.O_CREAT | os.O_WRONLY,
            0o644,
        )
        try:
            os.write(descriptor, serialized.encode("utf-8"))
        finally:
            os.close(descriptor)
        key = (kind_text, identity_text)
        current = self._rows.setdefault(
            key,
            {
                "kind": kind_text,
                "identity": identity_text,
                "payload": {},
                "lineage": [],
            },
        )
        current["payload"] = dict(payload)
        if normalized_lineage is not None:
            current["lineage"].append(normalized_lineage)
        return self.get(kind_text, identity_text) or {}

    def get(self, kind: str, identity: str) -> dict[str, Any] | None:
        row = self._rows.get((str(kind), str(identity)))
        if row is None:
            return None
        return {
            "kind": row["kind"],
            "identity": row["identity"],
            "payload": dict(row["payload"]),
            "lineage": [dict(item) for item in row["lineage"]],
        }
