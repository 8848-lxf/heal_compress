from __future__ import annotations

from pathlib import Path

import pytest


def _deployment_payload() -> dict[str, str]:
    return {
        "physical_hash": "physical",
        "precision_hash": "precision",
        "calibration_signature": "calibration",
        "build_signature": "build",
    }


def test_same_deployment_reuses_engine_but_not_evaluation_protocol() -> None:
    from search.cache.deployment_registry import (
        deployment_identity,
        evaluation_identity,
    )

    deployment = deployment_identity(_deployment_payload())
    eval_500 = evaluation_identity(deployment, "evaluate_500", "m500", "cfg")
    full = evaluation_identity(deployment, "full_validation", "m1789", "cfg")

    assert eval_500 != full
    assert eval_500 == evaluation_identity(
        deployment, "evaluate_500", "m500", "cfg"
    )


def test_deployment_identity_fails_closed_on_missing_lineage() -> None:
    from search.cache.deployment_registry import deployment_identity

    payload = _deployment_payload()
    payload["build_signature"] = ""

    with pytest.raises(ValueError, match="deployment_identity_missing:build_signature"):
        deployment_identity(payload)


def test_registry_preserves_all_lineage_references(tmp_path: Path) -> None:
    from search.cache.deployment_registry import (
        DeploymentRegistry,
        deployment_identity,
    )

    path = tmp_path / "registry.jsonl"
    identity = deployment_identity(_deployment_payload())
    registry = DeploymentRegistry(path)
    registry.record(
        kind="deployment",
        identity=identity,
        payload={"engine_path": "/engine.plan"},
        lineage={"budget": 0.21, "seed": 0, "generation": 1},
    )
    registry.record(
        kind="deployment",
        identity=identity,
        payload={"engine_path": "/engine.plan"},
        lineage={"budget": 0.20, "seed": 2, "generation": 7},
    )

    reloaded = DeploymentRegistry(path)
    row = reloaded.get("deployment", identity)

    assert row is not None
    assert row["payload"]["engine_path"] == "/engine.plan"
    assert row["lineage"] == [
        {"budget": 0.21, "generation": 1, "seed": 0},
        {"budget": 0.20, "generation": 7, "seed": 2},
    ]
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2
