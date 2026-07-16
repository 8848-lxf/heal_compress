from __future__ import annotations

import json


def test_exact_existing_engine_requires_qdq_builder_plugin_and_engine_hashes(tmp_path) -> None:
    from quantization.config import TensorRTBuildConfig
    from search.stage2.lidar_pyramid_real_evaluator import (
        _file_hash,
        _load_exact_existing_engine_build,
        _plain,
    )

    output = tmp_path / "deployment"
    output.mkdir()
    qdq = output / "qdq.onnx"
    engine = output / "engine.plan"
    plugin = tmp_path / "plugin.so"
    qdq.write_bytes(b"exact-qdq")
    engine.write_bytes(b"exact-engine")
    plugin.write_bytes(b"exact-plugin")
    (output / "engine_layer_info.json").write_text('{"Layers": []}', encoding="utf-8")
    config = TensorRTBuildConfig(
        trtexec_path=tmp_path / "trtexec",
        plugin_path=plugin,
        strongly_typed=True,
    )
    (output / "engine_manifest.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "engine_hash": _file_hash(engine),
                "engine_structure_validation": {"passed": True},
                "precision_realization_validation": {"passed": True},
            }
        ),
        encoding="utf-8",
    )
    (output / "engine_build_environment_manifest.json").write_text(
        json.dumps(
            {
                "qdq_onnx_sha256": _file_hash(qdq),
                "builder_config": _plain(config),
                "TensorRT_root": str(tmp_path / "TensorRT"),
                "plugin_sha256": _file_hash(plugin),
            }
        ),
        encoding="utf-8",
    )

    result, issues = _load_exact_existing_engine_build(
        output,
        qdq_onnx=qdq,
        build_config=config,
        tensorrt_root=tmp_path / "TensorRT",
    )

    assert issues == []
    assert result is not None
    assert result["engine_rebuilt"] is False
    assert result["engine_hash"] == _file_hash(engine)

    qdq.write_bytes(b"changed-qdq")
    result, issues = _load_exact_existing_engine_build(
        output,
        qdq_onnx=qdq,
        build_config=config,
        tensorrt_root=tmp_path / "TensorRT",
    )
    assert result is None
    assert issues == ["qdq_onnx_hash_mismatch"]


def test_exact_engine_cache_link_never_overwrites_destination(tmp_path) -> None:
    from search.stage2.lidar_pyramid_real_evaluator import (
        _materialize_exact_engine_cache_link,
    )

    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    names = (
        "engine.plan",
        "engine_layer_info.json",
        "engine_manifest.json",
        "engine_build_environment_manifest.json",
    )
    for name in names:
        (source / name).write_bytes(f"source:{name}".encode())

    result = _materialize_exact_engine_cache_link(source, destination)

    assert result["status"] == "ok"
    assert result["engine_rebuilt"] is False
    assert all((destination / name).read_bytes() == (source / name).read_bytes() for name in names)

    (destination / "engine.plan").write_bytes(b"preserve-this")
    rejected = _materialize_exact_engine_cache_link(source, destination)
    assert rejected["status"] == "not_materialized"
    assert "engine.plan" in rejected["occupied_destination_files"]
    assert (destination / "engine.plan").read_bytes() == b"preserve-this"
