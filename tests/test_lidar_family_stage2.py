from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _write_passed_audits(root: Path) -> None:
    for name in (
        "physical_validation.json",
        "pruning_quantization_group_audit.json",
        "production_qdq_boundary_audit.json",
        "merge_precision_realization.json",
        "engine_structure_validation.json",
        "precision_realization_validation.json",
    ):
        (root / name).write_text(
            json.dumps({"status": "ok", "passed": True, "merges": []}),
            encoding="utf-8",
        )
    (root / "typed_graph_report.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "strongly_typed": True,
                "unresolved_tensor_dtype_count": 0,
                "plugin_qdq_count": 0,
            }
        ),
        encoding="utf-8",
    )


def test_deployment_audit_does_not_require_pyramid_concat_for_disco(
    tmp_path: Path,
) -> None:
    from search.integration.lidar_family_registry import get_lidar_family_spec
    from search.stage2.lidar_pyramid_real_evaluator import (
        LidarPyramidRealEvaluator,
    )

    _write_passed_audits(tmp_path)
    evaluator = object.__new__(LidarPyramidRealEvaluator)
    evaluator.context = SimpleNamespace(
        family_spec=get_lidar_family_spec("lidar_disco")
    )

    report = evaluator._deployment_audit_summary(tmp_path)

    assert report["passed"] is True
    assert report["required_named_merge_contracts"] == []
    assert not any("Concat_9" in reason for reason in report["failure_reasons"])


def test_deployment_audit_retains_concat9_gate_for_pyramid(tmp_path: Path) -> None:
    from search.integration.lidar_family_registry import get_lidar_family_spec
    from search.stage2.lidar_pyramid_real_evaluator import (
        LidarPyramidRealEvaluator,
    )

    _write_passed_audits(tmp_path)
    evaluator = object.__new__(LidarPyramidRealEvaluator)
    evaluator.context = SimpleNamespace(
        family_spec=get_lidar_family_spec("lidar_pyramid")
    )

    report = evaluator._deployment_audit_summary(tmp_path)

    assert report["passed"] is False
    assert report["required_named_merge_contracts"] == ["/Concat_9"]
    assert "named_merge_contract_match_count:/Concat_9:0" in report["failure_reasons"]


def test_family_evaluator_builds_exporter_from_context_spec(monkeypatch) -> None:
    from search.integration.lidar_family_registry import get_lidar_family_spec
    from search.stage2.lidar_pyramid_real_evaluator import (
        LidarPyramidRealEvaluator,
    )

    captured = {}

    def fake_factory(model, **kwargs):
        captured["model"] = model
        captured.update(kwargs)
        return "wrapper"

    monkeypatch.setattr(
        "search.stage2.lidar_pyramid_real_evaluator.build_family_trt_export_module",
        fake_factory,
    )
    evaluator = object.__new__(LidarPyramidRealEvaluator)
    evaluator.context = SimpleNamespace(
        family_spec=get_lidar_family_spec("lidar_disco")
    )
    model = nn.Identity()

    assert evaluator._build_export_wrapper(model, fixed_k=29696) == "wrapper"
    assert captured["family"].name == "lidar_disco"
    assert captured["output_names"] == ("cls_preds", "reg_preds", "dir_preds")
    assert captured["fixed_k"] == 29696


def test_family_real_evaluator_preserves_production_base_class() -> None:
    from search.stage2.lidar_family_real_evaluator import LidarFamilyRealEvaluator
    from search.stage2.lidar_pyramid_real_evaluator import (
        LidarPyramidRealEvaluator,
    )

    assert issubclass(LidarFamilyRealEvaluator, LidarPyramidRealEvaluator)


def test_evaluation_request_records_family_and_repository_root(
    tmp_path: Path, monkeypatch
) -> None:
    from search.integration import evaluation_provider as module

    monkeypatch.setattr(module, "modelopt_subprocess_env", lambda **_kwargs: {})
    monkeypatch.setattr(module, "modelopt_python_command", lambda _env: ["python"])
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=1, stdout="stopped"),
    )

    module.evaluate_engine_modelopt(
        engine_path="engine.plan",
        checkpoint="checkpoint.pth",
        model_config="config.yaml",
        model_family="lidar_disco",
        repository_root=Path(__file__).resolve().parents[1],
        heal_root="/home/lixingfeng/UniAD_examine/HEAL",
        device="cuda:7",
        output_dir=tmp_path,
        tensorrt_root="/tmp/tensorrt",
        plugin_path=None,
        num_frames=1,
        warmup_frames=0,
    )
    request = json.loads(
        (tmp_path / "evaluation_request.json").read_text(encoding="utf-8")
    )

    assert request["model_family"] == "lidar_disco"
    assert request["repository_root"] == str(Path(__file__).resolve().parents[1])

