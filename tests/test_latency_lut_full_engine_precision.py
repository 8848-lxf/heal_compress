from __future__ import annotations

import json
from pathlib import Path

from opencood.tools.compression.latency_lut.channel_resolver import ChannelResolver
from opencood.tools.compression.latency_lut.full_engine_precision import (
    apply_precision_config_to_units,
    normalize_precision_label,
    resolve_layer_precision_config,
)
from tools.latency_lut.run_full_engine_candidate_benchmark import parse_args, run
from tools.latency_lut import run_full_engine_candidate_benchmark as runner


def test_layer_precision_config_is_three_choice_and_maps_weight_activation_together():
    units = [
        {"unit_id": "backbone.stage1", "module_name": "backbone", "block_name": "stage1"},
        {"unit_id": "backbone.stage2", "module_name": "backbone", "block_name": "stage2"},
        {"unit_id": "detection_head.cls", "module_name": "detection_head", "block_name": "cls"},
    ]
    assignments = resolve_layer_precision_config(
        {
            "default": "FP16",
            "overrides": {
                "backbone.stage2": "INT8",
                "detection_head": "FP32",
            },
        },
        units,
    )

    assert assignments["backbone.stage1"].precision_profile == "TRT_FP16"
    assert assignments["backbone.stage1"].weight_precision == "FP16"
    assert assignments["backbone.stage1"].activation_precision == "FP16"
    assert assignments["backbone.stage2"].precision_profile == "TRT_INT8_QDQ"
    assert assignments["backbone.stage2"].weight_precision == "INT8"
    assert assignments["backbone.stage2"].activation_precision == "INT8"
    assert assignments["detection_head.cls"].precision_profile == "TRT_FP32"
    assert assignments["detection_head.cls"].weight_precision == "FP32"
    assert assignments["detection_head.cls"].activation_precision == "FP32"


def test_layer_precision_config_rejects_weight_activation_cartesian_labels():
    for label in ["W8A16", "W8A32", "W16A8", "weight_int8_activation_fp16"]:
        try:
            normalize_precision_label(label)
        except ValueError as exc:
            assert "single layer precision_profile" in str(exc)
        else:
            raise AssertionError(f"{label} must not be accepted")


def test_channel_resolver_applies_nested_precision_overrides_to_units():
    units = [
        {
            "unit_id": "backbone.stage1",
            "module_name": "backbone",
            "block_name": "stage1",
            "block_type": "conv_block",
            "precision": "FP16",
        },
        {
            "unit_id": "detection_head.cls",
            "module_name": "detection_head",
            "block_name": "cls",
            "block_type": "head_branch",
            "precision": "FP16",
        },
    ]

    resolved = ChannelResolver().resolve(
        {
            "deploy_mode": "single_engine_maxK",
            "fixed_K": 29696,
            "precision_config": {"default": "FP16", "overrides": {"detection_head": "FP32"}},
            "units": units,
        }
    )

    assert resolved[0].precision == "FP16"
    assert resolved[1].precision == "FP32"


def test_apply_precision_config_to_units_adds_consistent_precision_fields():
    units = [{"unit_id": "shrink.compression", "module_name": "shrink", "block_name": "compression"}]
    updated = apply_precision_config_to_units(units, {"default": "INT8"})

    assert updated[0]["precision"] == "INT8"
    assert updated[0]["precision_profile"] == "TRT_INT8_QDQ"
    assert updated[0]["weight_precision"] == "INT8"
    assert updated[0]["activation_precision"] == "INT8"
    assert updated[0]["compute_precision"] == "INT8"


def test_full_engine_runner_uses_fp16_fp32_mixed_builder_when_available(tmp_path: Path, monkeypatch):
    candidate = tmp_path / "baseline_like_fp16_head_fp32.json"
    output = tmp_path / "baseline_like_fp16_head_fp32.full_engine_result.json"
    config = tmp_path / "config.yaml"
    checkpoint = tmp_path / "model.pth"
    plugin = tmp_path / "libpointpillar_scatter_trt.so"
    config.write_text("name: test\n", encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint")
    plugin.write_bytes(b"plugin")
    candidate.write_text(
        json.dumps(
            {
                "candidate_id": "baseline_like_fp16_head_fp32",
                "deploy_mode": "single_engine_maxK",
                "fixed_K": 29696,
                "pruning": {"enabled": False},
                "precision_config": {
                    "default": "FP16",
                    "overrides": {"detection_head": "FP32"},
                },
            }
        ),
        encoding="utf-8",
    )

    def fake_export(ctx):
        ctx.onnx_path.parent.mkdir(parents=True, exist_ok=True)
        ctx.onnx_path.write_bytes(b"onnx")
        return {"success": True}

    def fake_build(ctx, candidate_payload):
        ctx.engine_path.parent.mkdir(parents=True, exist_ok=True)
        ctx.engine_path.write_bytes(b"engine")
        return {
            "success": True,
            "build_success": True,
            "build_report": str(tmp_path / "mixed_build_report.json"),
            "precision_verification": {
                "success": True,
                "observed_fp32_layers": 3,
                "observed_fp16_layers": 10,
                "observed_int8_layers": 0,
                "observed_by_unit": {
                    "detection_head": [
                        {"name": "/cls_head/Conv", "precision": "FP32"},
                        {"name": "/reg_head/Conv", "precision": "FP32"},
                        {"name": "/dir_head/Conv", "precision": "FP32"},
                    ]
                },
            },
        }

    monkeypatch.setattr(runner, "_apply_runtime_env", lambda ctx: None)
    monkeypatch.setattr(runner, "_export_single_engine_onnx", fake_export)
    monkeypatch.setattr(
        runner,
        "_prepare_route2_mixed_onnx",
        lambda ctx, candidate_payload: {"success": True, "status": "route2_fp_rewrite_ready"},
    )
    monkeypatch.setattr(runner, "_build_mixed_single_engine", fake_build)
    monkeypatch.setattr(
        runner,
        "_evaluate_single_engine_subset",
        lambda ctx: {
            "success": True,
            "actual_frames": 5,
            "mAP": 0.8,
            "AP@0.70": 0.7,
            "forward_ms": {"p50": 3.0, "p90": 3.2, "p95": 3.3, "mean": 3.1, "std": 0.2},
        },
    )

    result = run(
        parse_args(
            [
                "--candidate",
                str(candidate),
                "--output",
                str(output),
                "--deploy-mode",
                "single_engine_maxK",
                "--fixed-k",
                "29696",
                "--config",
                str(config),
                "--checkpoint",
                str(checkpoint),
                "--plugin",
                str(plugin),
            ]
        )
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert result["success"] is True
    assert payload["precision_profile"] == "MIXED_FP16_FP32"
    assert payload["mixed_precision"] is True
    assert payload["precision_verification"]["success"] is True
    assert payload["precision_verification"]["observed_fp32_layers"] == 3
    assert payload["precision_verification"]["observed_fp16_layers"] == 10
    assert payload["T_real_p50"] == 3.0
    assert payload["mAP"] == 0.8


def test_full_engine_runner_no_longer_requires_prebuilt_int8_qdq_onnx(tmp_path: Path):
    candidate = tmp_path / "baseline_like_fp16_shrink_int8_head_fp16.json"
    output = tmp_path / "baseline_like_fp16_shrink_int8_head_fp16.full_engine_result.json"
    candidate.write_text(
        json.dumps(
            {
                "candidate_id": "baseline_like_fp16_shrink_int8_head_fp16",
                "deploy_mode": "single_engine_maxK",
                "fixed_K": 29696,
                "pruning": {"enabled": False},
                "precision_config": {
                    "default": "FP16",
                    "overrides": {"shrink": "INT8"},
                },
            }
        ),
        encoding="utf-8",
    )

    result = run(
        parse_args(
            [
                "--candidate",
                str(candidate),
                "--output",
                str(output),
                "--deploy-mode",
                "single_engine_maxK",
                "--fixed-k",
                "29696",
            ]
        )
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert result["success"] is False
    assert payload["status"] != "int8_mixed_full_engine_qdq_path_missing"
    assert payload["failed_stage"] != "precision_config"
    assert "T_real_p50" not in payload


def test_pruning_importance_and_scope_labels_map_to_legacy_pruner_modes():
    assert runner._importance_mode("l1") == "l1_norm"
    assert runner._importance_mode("l2") == "l2_norm"
    assert runner._importance_mode("taylor_first_order") == "first_order_taylor"
    assert runner._importance_mode("fisher_second_order") == "second_order_fisher"
    assert runner._selection_mode({"scope": "local", "local_scope": "prune_domain"}) == "local_scope"
    assert runner._selection_mode({"scope": "global"}) == "constrained_global"
