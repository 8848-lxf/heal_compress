from __future__ import annotations


def test_default_family_never_loads_cobevt(monkeypatch, tmp_path):
    from search.model_families import registry

    class ExistingPyramidRunner:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(
        registry,
        "_load_cobevt_runner",
        lambda: (_ for _ in ()).throw(
            AssertionError("default pyramid dispatch imported CoBEVT")
        ),
    )

    runner = registry.create_family_runner(
        config={"model": {}},
        checkpoint="model.pth",
        output_root=tmp_path,
        pyramid_runner_cls=ExistingPyramidRunner,
    )

    assert isinstance(runner, ExistingPyramidRunner)


def test_explicit_pyramid_alias_resolves_to_existing_runner(tmp_path):
    from search.model_families.registry import create_family_runner

    class ExistingPyramidRunner:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    runner = create_family_runner(
        config={"model": {"family": "lidar_pyramid"}},
        checkpoint="model.pth",
        output_root=tmp_path,
        resume=tmp_path / "resume",
        pyramid_runner_cls=ExistingPyramidRunner,
    )

    assert isinstance(runner, ExistingPyramidRunner)
    assert runner.kwargs["resume"] == tmp_path / "resume"

