from __future__ import annotations

import json
from pathlib import Path

import pytest


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _source_experiment(root: Path) -> None:
    smoke = {
        "split": "val",
        "frame_ids": ["s0", "s1"],
        "warmup_frame_ids": ["w0", "w1"],
        "evaluation_frame_ids": ["s0", "s1"],
        "warmup_frames": 2,
        "num_frames": 2,
        "reset_after_warmup": True,
        "evaluation_offset": 2,
        "manifest_hash": "smoke-hash",
    }
    fixed = {
        "split": "val",
        "frame_ids": [f"f{index}" for index in range(60)],
        "warmup_frame_ids": ["w0", "w1"],
        "evaluation_frame_ids": [f"f{index}" for index in range(60)],
        "warmup_frames": 2,
        "num_frames": 60,
        "reset_after_warmup": True,
        "evaluation_offset": 2,
        "manifest_hash": "fixed-hash",
    }
    _write_json(root / "manifests/smoke10_manifest.json", smoke)
    _write_json(root / "manifests/fixed500_manifest.json", fixed)
    _write_json(root / "candidate_masks/baseline_d32.json", {"mask": "identity"})
    _write_json(
        root / "experiment_config.json",
        {
            "fixed_k": 29696,
            "fixed_k_validated": True,
            "fixed_k_contract": {
                "deployment_topology": "single_engine_fixed_k",
                "fixed_k": 29696,
                "overflow_count": 0,
                "validated_scope": "full_validation",
            },
            "candidates": [
                {
                    "candidate_id": "baseline_d32",
                    "d_qk": 32,
                    "d_v": 32,
                    "embed_dim": 256,
                    "experiment": "baseline",
                    "heads": 8,
                    "mask_path": str(root / "candidate_masks/baseline_d32.json"),
                    "variant": "unpruned_explicit_projection",
                },
                {"candidate_id": "qk_only_d24"},
            ],
            "manifests": {
                "smoke10": {
                    "path": str(root / "manifests/smoke10_manifest.json"),
                    "manifest_hash": "smoke-hash",
                },
                "fixed500": {
                    "path": str(root / "manifests/fixed500_manifest.json"),
                    "manifest_hash": "fixed-hash",
                },
            },
        },
    )


def test_prepare_boundary_audit_isolates_baseline_and_creates_fixed50(tmp_path: Path):
    from search.orchestration.lidar_cobevt_attention_precision_audit import (
        prepare_boundary_audit_run,
    )

    source = tmp_path / "source"
    output = tmp_path / "output"
    _source_experiment(source)

    result = prepare_boundary_audit_run(
        source_output=source,
        output_dir=output,
        checkpoint_hash="checkpoint-hash",
        config_hash="config-hash",
        plugin_hash="plugin-hash",
        code_commit="code-commit",
    )

    experiment = json.loads((output / "experiment_config.json").read_text())
    assert result["fixed_k"] == 29696
    assert result["deployment_topology"] == "single_engine_fixed_k"
    assert [row["candidate_id"] for row in experiment["candidates"]] == [
        "baseline_d32"
    ]
    assert Path(experiment["candidates"][0]["mask_path"]).parent == (
        output / "candidate_masks"
    )
    fixed50 = json.loads((output / "manifests/fixed50_manifest.json").read_text())
    assert fixed50["evaluation_frame_ids"] == [f"f{index}" for index in range(50)]
    assert fixed50["warmup_frame_ids"] == ["w0", "w1"]
    assert fixed50["num_frames"] == 50
    assert len(list((output / "profiles").iterdir())) == 12
    assert not list(output.rglob("*.plan"))


def test_prepare_boundary_audit_rejects_nonempty_destination(tmp_path: Path):
    from search.orchestration.lidar_cobevt_attention_precision_audit import (
        prepare_boundary_audit_run,
    )

    source = tmp_path / "source"
    output = tmp_path / "output"
    _source_experiment(source)
    output.mkdir()
    (output / "foreign.txt").write_text("do not overwrite", encoding="utf-8")

    with pytest.raises(RuntimeError, match="boundary_audit_output_not_empty"):
        prepare_boundary_audit_run(
            source_output=source,
            output_dir=output,
            checkpoint_hash="checkpoint-hash",
            config_hash="config-hash",
            plugin_hash="plugin-hash",
            code_commit="code-commit",
        )


@pytest.mark.parametrize(
    ("delta_map", "expected"),
    [
        (-0.050000, "catastrophic"),
        (-0.049999, "ambiguous"),
        (-0.010001, "ambiguous"),
        (-0.010000, "safe"),
        (0.001, "safe"),
    ],
)
def test_smoke_delta_classification_uses_exact_protocol_thresholds(
    delta_map: float, expected: str
):
    from search.orchestration.lidar_cobevt_attention_precision_audit import (
        classify_smoke_delta,
    )

    assert classify_smoke_delta(delta_map) == expected


def test_freshness_signature_changes_with_every_owned_identity():
    from search.orchestration.lidar_cobevt_attention_precision_audit import (
        boundary_freshness_signature,
    )

    fields = {
        "code_commit": "code",
        "checkpoint_hash": "checkpoint",
        "config_hash": "config",
        "plugin_hash": "plugin",
        "profile_hash": "profile",
        "fixed_k": 29696,
        "smoke_manifest_hash": "smoke",
        "fixed500_manifest_hash": "fixed500",
        "tensorrt_version": "10.9",
        "cuda_version": "11.8",
    }
    baseline = boundary_freshness_signature(**fields)
    assert baseline == boundary_freshness_signature(**fields)
    for key, value in fields.items():
        changed = dict(fields)
        changed[key] = value + "-other" if isinstance(value, str) else value + 256
        assert boundary_freshness_signature(**changed) != baseline
    assert len(baseline) == 64
    assert set(baseline) <= set("0123456789abcdef")


def test_diagnostic_builder_command_preserves_production_strong_typing(tmp_path: Path):
    from search.orchestration.lidar_cobevt_attention_precision_audit import (
        diagnostic_builder_command,
    )

    production = [
        "/trt/bin/trtexec",
        "--onnx=/run/typed_parser.onnx",
        "--saveEngine=/run/engine.plan",
        "--profilingVerbosity=detailed",
        "--exportLayerInfo=/run/engine_layer_info.json",
        "--skipInference",
        "--noTF32",
        "--stronglyTyped",
        "--staticPlugins=/run/scatter.so",
    ]

    command = diagnostic_builder_command(
        production,
        onnx_path=tmp_path / "diagnostic.onnx",
        engine_path=tmp_path / "diagnostic.plan",
        layer_info_path=tmp_path / "diagnostic_layer_info.json",
    )

    assert f"--onnx={tmp_path / 'diagnostic.onnx'}" in command
    assert f"--saveEngine={tmp_path / 'diagnostic.plan'}" in command
    assert f"--exportLayerInfo={tmp_path / 'diagnostic_layer_info.json'}" in command
    assert "--stronglyTyped" in command
    assert "--noTF32" in command
    assert "--staticPlugins=/run/scatter.so" in command
    assert "--skipInference" in command
    assert not any(item == "--fp16" or item == "--int8" for item in command)


def test_diagnostic_builder_command_rejects_nonproduction_precision_protocol(
    tmp_path: Path,
):
    from search.orchestration.lidar_cobevt_attention_precision_audit import (
        diagnostic_builder_command,
    )

    with pytest.raises(ValueError, match="diagnostic_builder_missing_required_flag"):
        diagnostic_builder_command(
            ["trtexec", "--onnx=a", "--saveEngine=b", "--noTF32"],
            onnx_path=tmp_path / "diagnostic.onnx",
            engine_path=tmp_path / "diagnostic.plan",
            layer_info_path=tmp_path / "diagnostic.json",
        )


def test_build_attention_diagnostic_engine_records_parity_only_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import subprocess

    from search.orchestration.lidar_cobevt_attention_precision_audit import (
        build_attention_diagnostic_engine,
    )

    diagnostic_onnx = tmp_path / "diagnostic.onnx"
    diagnostic_onnx.write_bytes(b"diagnostic-onnx")
    production = tmp_path / "production_build.json"
    _write_json(
        production,
        {
            "status": "ok",
            "builder_command": {
                "command": [
                    "/trt/bin/trtexec",
                    "--onnx=/run/typed_parser.onnx",
                    "--saveEngine=/run/engine.plan",
                    "--exportLayerInfo=/run/engine_layer_info.json",
                    "--skipInference",
                    "--noTF32",
                    "--stronglyTyped",
                    "--staticPlugins=/run/scatter.so",
                ]
            },
        },
    )

    calls = []

    def fake_run(command, **kwargs):
        calls.append(list(command))
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "2"
        engine = Path(next(value.split("=", 1)[1] for value in command if value.startswith("--saveEngine=")))
        layers = Path(next(value.split("=", 1)[1] for value in command if value.startswith("--exportLayerInfo=")))
        engine.write_bytes(b"diagnostic-engine")
        layers.write_text('{"Layers": []}', encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="built")

    monkeypatch.setattr(subprocess, "run", fake_run)

    report = build_attention_diagnostic_engine(
        production_build_report=production,
        diagnostic_onnx=diagnostic_onnx,
        output_dir=tmp_path / "diagnostic_engine",
        physical_gpu=2,
    )

    assert report["status"] == "ok"
    assert report["diagnostic_latency_invalid"] is True
    assert report["strongly_typed"] is True
    assert report["engine_sha256"]
    assert report["onnx_sha256"]
    assert Path(report["engine_path"]).is_file()
    assert (tmp_path / "diagnostic_engine/diagnostic_engine_build_report.json").is_file()

    reused = build_attention_diagnostic_engine(
        production_build_report=production,
        diagnostic_onnx=diagnostic_onnx,
        output_dir=tmp_path / "diagnostic_engine",
        physical_gpu=2,
    )

    assert len(calls) == 1
    assert reused["reused_existing_diagnostic_engine"] is True


def test_attention_parity_request_maps_physical_gpu_to_isolated_logical_device():
    from search.orchestration.lidar_cobevt_attention_precision_audit import (
        build_attention_parity_request,
    )

    request = build_attention_parity_request(
        reference_engine_path="/run/a0.plan",
        candidate_engine_path="/run/a1.plan",
        model_config="/run/config.yaml",
        heal_root="/run/HEAL",
        plugin_path="/run/scatter.so",
        eval_manifest_path="/run/smoke10.json",
        output_path="/run/parity.json",
        output_specs=[
            {"block_id": "layers.0.window", "role": "q_projection", "tensor_name": "q"}
        ],
        physical_gpu=2,
        profile_name="A1_qkv_projection_fp16_core_fp32",
    )

    assert request["cuda_visible_devices"] == "2"
    assert request["device"] == "cuda:0"
    assert request["physical_device"] == "cuda:2"
    assert request["fixed_k"] == 29696
    assert request["num_frames"] == 10
    assert request["num_workers"] == 8
    assert request["diagnostic_latency_invalid"] is True


def test_run_manifest_profile_registration_is_idempotent(tmp_path: Path):
    from search.orchestration.lidar_cobevt_attention_precision_audit import (
        register_run_profile,
    )

    manifest = tmp_path / "run_manifest.json"
    _write_json(manifest, {"profiles": ["A0_strict_fp32_reference"]})

    register_run_profile(manifest, "M1_projection_fp16_core_fp32")
    register_run_profile(manifest, "M1_projection_fp16_core_fp32")

    assert json.loads(manifest.read_text())["profiles"] == [
        "A0_strict_fp32_reference",
        "M1_projection_fp16_core_fp32",
    ]
