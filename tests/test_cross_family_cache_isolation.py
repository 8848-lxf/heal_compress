from __future__ import annotations


def test_pyramid_artifact_cannot_satisfy_cobevt_cache_key() -> None:
    from search.model_families.lidar_cobevt.deployment_recipe import (
        model_family_deployment_identity,
    )

    common = {
        "physical_hash": "physical",
        "precision_hash": "precision",
        "calibration_signature": "calibration",
        "build_signature": "build",
    }
    pyramid = model_family_deployment_identity(
        model_family="lidar_pyramid", **common
    )
    cobevt = model_family_deployment_identity(
        model_family="lidar_cobevt", **common
    )

    assert pyramid != cobevt


def test_fixed_k_and_recipe_version_are_part_of_cobevt_identity() -> None:
    from search.model_families.lidar_cobevt.deployment_recipe import (
        model_family_deployment_identity,
    )

    common = {
        "model_family": "lidar_cobevt",
        "physical_hash": "physical",
        "precision_hash": "precision",
        "calibration_signature": "calibration",
        "build_signature": "build",
        "recipe_version": "v1",
    }
    fixed64 = model_family_deployment_identity(fixed_k=64, **common)
    fixed128 = model_family_deployment_identity(fixed_k=128, **common)

    assert fixed64 != fixed128

