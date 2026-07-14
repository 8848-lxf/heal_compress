from __future__ import annotations

from pathlib import Path

import pytest

from quantization.config import TensorRTBuildConfig
from quantization.exceptions import TensorRTConfigurationError
from quantization.tensorrt.command import build_trt_command
from quantization.types import CanonicalPrecisionEntry, CanonicalPrecisionMappingResult
from search.baselines.original_engines import make_baseline_trt_build_config


def _mapping() -> CanonicalPrecisionMappingResult:
    return CanonicalPrecisionMappingResult(
        entries=[
            CanonicalPrecisionEntry(
                module_path="stem",
                canonical_node_name="__canonical__stem__Conv__call00000",
                precision_group="pg_stem",
                requested_precision="int8",
                realized_request_precision="int8",
                weight_initializer="stem.weight",
                onnx_op_type="Conv",
            )
        ]
    )


def _config(**overrides: object) -> TensorRTBuildConfig:
    payload: dict[str, object] = {
        "strongly_typed": True,
        "production_mode": True,
        "plugin_boundary_dtype": "fp16",
        "enable_fp16": False,
        "enable_int8": False,
        "precision_constraints": "none",
    }
    payload.update(overrides)
    return TensorRTBuildConfig.from_dict(payload)


def test_production_strongly_typed_command_has_no_weak_precision_flags(
    tmp_path: Path,
) -> None:
    config = _config(
        trtexec_path=Path("/opt/tensorrt/bin/trtexec"),
        plugin_path=tmp_path / "scatter.so",
    )

    result = build_trt_command(
        tmp_path / "typed.onnx",
        tmp_path / "engine.plan",
        _mapping(),
        config=config,
    )

    command = result.command
    assert "--stronglyTyped" in command
    assert result.policy_version == "trt-strongly-typed-explicit-qdq-cast-v1"
    forbidden = (
        "--fp16",
        "--int8",
        "--precisionConstraints",
        "--layerPrecisions",
        "--layerOutputTypes",
    )
    assert not any(part.startswith(forbidden) for part in command)


def test_production_builder_rejects_weakly_typed_fallback(tmp_path: Path) -> None:
    config = _config(strongly_typed=False)
    with pytest.raises(TensorRTConfigurationError, match="production_requires_strongly_typed"):
        build_trt_command(
            tmp_path / "typed.onnx",
            tmp_path / "engine.plan",
            _mapping(),
            config=config,
        )


@pytest.mark.parametrize("enabled", ("fp16", "int8"))
def test_strongly_typed_builder_rejects_implicit_precision_flags(
    tmp_path: Path, enabled: str
) -> None:
    config = _config(
        enable_fp16=enabled == "fp16",
        enable_int8=enabled == "int8",
    )
    with pytest.raises(
        TensorRTConfigurationError,
        match="strongly_typed_forbids_implicit_precision_flags",
    ):
        build_trt_command(
            tmp_path / "typed.onnx",
            tmp_path / "engine.plan",
            _mapping(),
            config=config,
        )


def test_strongly_typed_builder_rejects_precision_constraints(
    tmp_path: Path,
) -> None:
    config = _config(plugin_boundary_dtype="fp32", precision_constraints="obey")
    with pytest.raises(
        TensorRTConfigurationError,
        match="strongly_typed_forbids_precision_constraints",
    ):
        build_trt_command(
            tmp_path / "typed.onnx",
            tmp_path / "engine.plan",
            _mapping(),
            config=config,
        )


def test_strongly_typed_production_requires_floating_plugin_boundary(
    tmp_path: Path,
) -> None:
    config = _config(plugin_boundary_dtype="int8")
    with pytest.raises(
        TensorRTConfigurationError,
        match="strongly_typed_plugin_boundary_must_be_fp16_or_fp32",
    ):
        build_trt_command(
            tmp_path / "typed.onnx",
            tmp_path / "engine.plan",
            _mapping(),
            config=config,
        )


@pytest.mark.parametrize("baseline", ("strict_fp16", "matched_legacy_int8"))
def test_formal_baselines_use_strongly_typed_production_builder(
    tmp_path: Path, baseline: str
) -> None:
    config = make_baseline_trt_build_config(
        baseline,
        trtexec_path=Path("/opt/tensorrt/bin/trtexec"),
        plugin_path=tmp_path / "scatter.so",
        plugin_boundary_dtype="fp16",
        shape_profiles={},
    )

    assert config.strongly_typed is True
    assert config.production_mode is True
    assert config.plugin_boundary_dtype == "fp16"
    assert config.enable_fp16 is False
    assert config.enable_int8 is False
    assert config.precision_constraints == "none"


def test_search_context_rejects_unselected_plugin_boundary_before_gpu_init(
    tmp_path: Path,
) -> None:
    from search.integration.lidar_pyramid_context import build_lidar_pyramid_context

    with pytest.raises(
        RuntimeError,
        match="strongly_typed_plugin_boundary_must_be_fp16_or_fp32",
    ):
        build_lidar_pyramid_context(
            checkpoint_path=tmp_path / "missing.pth",
            model_config_path=tmp_path / "missing.yaml",
            output_dir=tmp_path / "run",
            plugin_boundary_dtype="readiness_selection_required",
        )


def test_worker_builder_contract_audit_rejects_forbidden_production_flags() -> None:
    from search.stage2.trt_build_worker import _builder_contract_audit

    config = _config()
    passed = _builder_contract_audit(config, ["trtexec", "--stronglyTyped"])
    assert passed["passed"] is True
    assert passed["strongly_typed"] is True
    assert passed["forbidden_options"] == []

    failed = _builder_contract_audit(
        config,
        ["trtexec", "--stronglyTyped", "--fp16", "--precisionConstraints=obey"],
    )
    assert failed["passed"] is False
    assert failed["forbidden_options"] == ["--fp16", "--precisionConstraints=obey"]
