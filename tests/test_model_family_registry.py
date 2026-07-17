from __future__ import annotations

import hashlib

import pytest


def test_missing_family_defaults_to_existing_pyramid_runner(tmp_path):
    from search.model_families.registry import create_family_runner, model_family_name

    seen = {}

    class ExistingRunner:
        def __init__(self, **kwargs):
            seen.update(kwargs)

    config = {"model": {}}
    runner = create_family_runner(
        config=config,
        checkpoint="model.pth",
        output_root=tmp_path,
        pyramid_runner_cls=ExistingRunner,
    )

    assert isinstance(runner, ExistingRunner)
    assert model_family_name(config) == "lidar_pyramid"
    assert seen == {
        "config": config,
        "checkpoint": "model.pth",
        "output_root": tmp_path,
        "resume": None,
    }


def test_cobevt_dispatch_uses_only_cobevt_loader(monkeypatch, tmp_path):
    from search.model_families import registry

    calls = []

    class CobevtRunner:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class PyramidRunner:
        def __init__(self, **kwargs):
            raise AssertionError("pyramid runner must not be constructed")

    monkeypatch.setattr(
        registry,
        "_load_cobevt_runner",
        lambda: calls.append("cobevt") or CobevtRunner,
    )
    runner = registry.create_family_runner(
        config={"model": {"family": "lidar_cobevt"}},
        checkpoint="model.pth",
        output_root=tmp_path,
        pyramid_runner_cls=PyramidRunner,
    )

    assert isinstance(runner, CobevtRunner)
    assert calls == ["cobevt"]


def test_unknown_model_family_fails_closed(tmp_path):
    from search.model_families.registry import create_family_runner

    with pytest.raises(RuntimeError, match="unsupported_model_family:camera_only"):
        create_family_runner(
            config={"model": {"family": "camera_only"}},
            checkpoint="model.pth",
            output_root=tmp_path,
            pyramid_runner_cls=object,
        )


def test_capability_manifest_hash_is_stable_and_family_specific():
    from search.model_families.contracts import ModelFamilyCapabilityManifest

    common = {
        "recipe_version": "1",
        "checkpoint_sha256": "a" * 64,
        "config_sha256": "b" * 64,
        "capabilities": {"typed_onnx": True, "strongly_typed": True},
    }
    pyramid = ModelFamilyCapabilityManifest(
        model_family="lidar_pyramid", **common
    )
    cobevt = ModelFamilyCapabilityManifest(model_family="lidar_cobevt", **common)

    assert pyramid.sha256 == hashlib.sha256(pyramid.canonical_json.encode()).hexdigest()
    assert pyramid.sha256 != cobevt.sha256

