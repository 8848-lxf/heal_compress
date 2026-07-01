from __future__ import annotations

import importlib
from pathlib import Path

import yaml


def test_quantization_formal_imports_and_default_config():
    defaults = importlib.import_module("quantization.utils.paths")
    config_path = Path("quantization/configs/lidar_pyramid_single_engine_maxk_fixedK29696.yaml")

    assert defaults.DEFAULT_STRATEGY == "single_engine_maxK"
    assert defaults.DEFAULT_FIXED_K == 29696
    assert defaults.DEFAULT_PRECISION == "fp16"
    assert config_path.is_file()

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["deployment"]["strategy"] == "single_engine_maxK"
    assert config["deployment"]["fixed_K"] == 29696
    assert config["deployment"]["precision"] == "fp16"
    assert config["deployment"]["int8"]["default_recommended"] is False


def test_tracer_formal_imports_and_coupled_group_schema():
    schema = importlib.import_module("tracer.coupled_channel_groups")

    group = schema.normalize_coupled_group(
        {
            "group_id": "group::conv1",
            "source_modules": ["conv1"],
            "dependent_modules": ["conv2"],
            "channel_indices": [0, 1, 2],
        }
    )

    assert group["group_id"] == "group::conv1"
    assert group["group_type"] == "conv_block"
    assert group["source_modules"] == ["conv1"]
    assert group["channel_indices"] == [0, 1, 2]
    assert group["is_prunable"] is True


def test_pruning_formal_imports_and_prune_plan_schema():
    plan_mod = importlib.import_module("pruning.planner.physical_prune_plan")

    plan = plan_mod.build_plan_from_coupled_groups(
        {
            "groups": [
                {
                    "group_id": "group::conv1",
                    "source_modules": ["conv1"],
                    "dependent_modules": ["conv2"],
                    "channel_indices": list(range(8)),
                    "is_prunable": True,
                }
            ]
        },
        target_prune_ratio=0.25,
        min_keep_ratio=0.5,
        importance="l1",
    )

    assert plan["schema_version"] == 1
    assert plan["target_prune_ratio"] == 0.25
    assert plan["min_keep_ratio"] == 0.5
    assert plan["prune_plan"][0]["group_id"] == "group::conv1"
    assert plan["prune_plan"][0]["keep_count"] == 6
    assert plan["legality"]["legal"] is True


def test_old_quant_deploy_wrappers_import_compatibly():
    modules = [
        "tests.quant_deploy.export_dynamic_single_engine_maxk_onnx",
        "tests.quant_deploy.build_dynamic_single_engine_maxk_trt_engine",
        "tests.quant_deploy.dump_train_calibration_npz_for_all_strategies",
        "tests.quant_deploy.select_idle_gpu",
    ]
    for name in modules:
        mod = importlib.import_module(name)
        assert hasattr(mod, "main") or hasattr(mod, "parse_args")
