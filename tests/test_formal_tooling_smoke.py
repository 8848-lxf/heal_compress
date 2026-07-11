from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import yaml

ROOT = Path(__file__).resolve().parents[1]
UNIAD = ROOT.parent
for path in (UNIAD, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


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


def test_pruning_formal_plan_keeps_protected_groups_full_width():
    plan_mod = importlib.import_module("pruning.planner.physical_prune_plan")

    plan = plan_mod.build_plan_from_coupled_groups(
        {
            "groups": [
                {
                    "group_id": "group::reg_head",
                    "source_modules": ["reg_head"],
                    "channel_indices": list(range(14)),
                    "is_prunable": False,
                    "is_protected": True,
                }
            ]
        },
        target_prune_ratio=0.125,
        min_keep_ratio=0.875,
        importance="l1",
    )

    row = plan["prune_plan"][0]
    assert row["is_protected"] is True
    assert row["is_prunable"] is False
    assert row["keep_count"] == 14
    assert row["prune_count"] == 0
    assert row["prune_indices"] == []
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


def test_formal_trt_evaluator_parses_benchmark_ap_thresholds():
    evaluator = importlib.import_module("trt_runtime.heal_trt_evaluator")

    assert evaluator.parse_ap_thresholds("0.03,0.3,0.5,0.7") == (0.03, 0.30, 0.50, 0.70)
    assert evaluator.threshold_key(0.03) == "AP@0.03"


def test_formal_single_engine_build_wrapper_delegates_to_formal_api(monkeypatch, tmp_path):
    build_mod = importlib.import_module("quantization.build.build_single_engine_maxk_engine")
    calls = {}
    assert not hasattr(build_mod, "load_quant_deploy_module")
    monkeypatch.setattr(build_mod, "find_trtexec", lambda *_args: {"trtexec_found": True, "trtexec_path": "/trtexec"})

    class Result:
        success = True

        def to_dict(self):
            return {"success": True, "elapsed_seconds": 1.0, "failure_reason": ""}

    def fake_build(onnx, engine, mapping, **kwargs):
        calls.update({"onnx": onnx, "engine": engine, "mapping": mapping, "kwargs": kwargs})
        Path(engine).write_bytes(b"engine")
        return Result()

    monkeypatch.setattr(build_mod, "build_trt_engine", fake_build)
    (tmp_path / "model.onnx").write_bytes(b"onnx")
    (tmp_path / "plugin.so").write_bytes(b"plugin")
    mapping = {
        "entries": [{
            "module_path": "stem", "canonical_node_name": "__canonical__stem__Conv__call00000",
            "precision_group": "pg::stem", "requested_precision": "fp16", "realized_request_precision": "fp16"
        }]
    }
    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_text(__import__("json").dumps(mapping), encoding="utf-8")
    args = SimpleNamespace(
        onnx=str(tmp_path / "model.onnx"),
        precision="fp16",
        fixed_k=29696,
        trt_root="/trt",
        trtexec_path="/trtexec",
        plugin=str(tmp_path / "plugin.so"),
        output_dir=str(tmp_path / "artifacts" / "engines" / "fixedK29696" / "dynamic_agent_single_engine_maxK" / "fp16"),
        calibration_frames=200,
        profile_calibration_frames=200,
        timeout=7,
        rebuild=False,
        force_recalibrate=False,
        precision_mapping=str(mapping_path),
        skip_existing=False,
        calibration_npz_dir=None,
        calibration_cache=None,
    )

    with __import__("pytest").warns(DeprecationWarning):
        report = build_mod.build_single_engine_maxk_engine(args)

    assert report["formal_tool"] == "quantization.build.build_single_engine_maxk_engine"
    assert report["success"] is True
    assert calls["mapping"].entries[0].canonical_node_name == "__canonical__stem__Conv__call00000"
    assert calls["kwargs"]["config"].precision_constraints == "obey"
