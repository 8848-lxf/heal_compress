from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _capability(module_path: str, *, int8: bool = True):
    from search.model_family.contracts import WeightedOpCapability

    allowed = ("FP32", "FP16", "INT8") if int8 else ("FP32", "FP16")
    return WeightedOpCapability(
        canonical_id=f"module::{module_path}",
        module_path=module_path,
        op_type="Conv2d",
        source_kind="module",
        weight_shape=(4, 4, 1, 1),
        allowed_precisions=allowed,
        potential_precisions=("FP32", "FP16", "INT8"),
        default_precision="FP16",
        weight_granularity="per_output_channel",
        weight_axis=0,
        input_scale_owner="canonical_weighted_input_tensor",
        output_boundary="post_relu",
        production_enabled=int8,
        gate_reason="" if int8 else "protected_test_boundary",
    )


def _audit(family_id: str, capabilities) -> object:
    from search.model_family.contracts import ModelFamilyAudit

    return ModelFamilyAudit(
        schema_version="test-v1",
        family_id=family_id,
        model_type="HeterModelBaseline",
        parameter_count=1,
        weighted_ops=tuple(capabilities),
        pruning_domains=(),
        merge_boundaries=(),
        deployment_operators=(),
        plugin_requirements=(),
        input_contract={"fixed_k": 4, "max_agents": 2},
        blockers=(),
    )


def _origin(module_paths):
    from quantization.types import CanonicalMappingEntry, OnnxOriginMapResult

    return OnnxOriginMapResult(entries=[
        CanonicalMappingEntry(
            module_path=module_path,
            module_type="Conv2d",
            call_index=index,
            onnx_op_type="Conv",
            original_node_name=f"/Conv_{index}",
            canonical_node_name=f"__canonical__{index}",
            weight_initializer=f"weight_{index}",
            graph_index=index,
            groups=1,
            weight_shape=(4, 4, 1, 1),
        )
        for index, module_path in enumerate(module_paths)
    ])


def _write_island_onnx(path: Path, family_id: str, *, include_weighted: bool = False) -> None:
    import numpy as np
    from onnx import TensorProto, helper, numpy_helper
    import onnx

    nodes = []
    inputs = [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 2, 2])]
    initializers = []
    if include_weighted:
        initializers.append(numpy_helper.from_array(np.ones((1, 1, 1, 1), dtype=np.float32), name="weight_0"))
        nodes.append(helper.make_node("Conv", ["x", "weight_0"], ["conv_out"], name="__canonical__0"))
    nodes.append(helper.make_node("Concat", ["x", "x", "x"], ["bev_concat"], name="/Concat", axis=1))
    if family_id == "heal_lidar_fcooper":
        nodes.extend((
            helper.make_node("GridSample", ["x", "grid"], ["grid_out"], name="/GridSample"),
            helper.make_node("Mul", ["grid_out", "x"], ["mul_out"], name="/Mul_6"),
            helper.make_node("Where", ["condition", "mul_out", "x"], ["where_out"], name="/Where_2"),
            helper.make_node("ReduceMax", ["where_out"], ["fusion_out"], name="/ReduceMax"),
        ))
    else:
        nodes.extend((
            helper.make_node("GridSample", ["x", "grid"], ["grid_out"], name="/GridSample"),
            helper.make_node("Mul", ["grid_out", "x"], ["mul_out"], name="/Mul_6"),
            helper.make_node("Concat", ["mul_out", "x"], ["concat_out"], name="/Concat_5", axis=1),
            helper.make_node("Where", ["condition", "concat_out", "x"], ["where_out"], name="/Where_3"),
            helper.make_node("Softmax", ["where_out"], ["softmax_out"], name="/Softmax", axis=0),
            helper.make_node("Expand", ["softmax_out", "shape"], ["expand_out"], name="/Expand_2"),
            helper.make_node("Mul", ["expand_out", "mul_out"], ["weighted_out"], name="/Mul_9"),
            helper.make_node("ReduceSum", ["weighted_out"], ["fusion_out"], name="/ReduceSum"),
        ))
    if include_weighted:
        nodes.append(helper.make_node(
            "PointPillarScatterTRT", ["x"], ["plugin_out"], name="/scatter", domain="trt"
        ))
    output_name = "conv_out" if include_weighted else "fusion_out"
    graph = helper.make_graph(
        nodes,
        "fusion-island",
        inputs,
        [helper.make_tensor_value_info(output_name, TensorProto.FLOAT, None)],
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("trt", 1)],
    )
    onnx.save(model, path)


def test_custom_scatter_checker_does_not_hide_unrelated_onnx_errors() -> None:
    import onnx
    from onnx import TensorProto, helper

    from search.model_family.heal_lidar_deployment import _check_onnx_with_scatter_plugin

    plugin = helper.make_node(
        "PointPillarScatterTRT", ["x"], ["scatter"], name="scatter", domain="trt"
    )
    valid = helper.make_model(
        helper.make_graph(
            [plugin],
            "valid",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
            [helper.make_tensor_value_info("scatter", TensorProto.FLOAT, [1])],
        ),
        opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("trt", 1)],
    )
    _check_onnx_with_scatter_plugin(valid)

    malformed = onnx.ModelProto()
    malformed.ParseFromString(valid.SerializeToString())
    malformed.graph.node.extend([
        helper.make_node("Add", ["missing", "scatter"], ["bad"], name="bad_add")
    ])
    with pytest.raises(Exception):
        _check_onnx_with_scatter_plugin(malformed)


def test_baseline_onnx_mapping_requires_full_weighted_capability_coverage(tmp_path: Path) -> None:
    from search.model_family import build_heal_lidar_baseline_onnx_mapping

    path = tmp_path / "model.onnx"
    _write_island_onnx(path, "heal_lidar_fcooper")
    audit = _audit("heal_lidar_fcooper", [_capability("backbone.conv"), _capability("cls_head", int8=False)])
    origin = _origin(["backbone.conv", "cls_head"])

    mapping = build_heal_lidar_baseline_onnx_mapping(path, audit, origin)

    assert mapping.metadata["realized_graph_mapping_complete"] is True
    assert mapping.metadata["active_weighted_capability_count"] == 2
    with pytest.raises(RuntimeError, match="onnx_mapping_incomplete"):
        build_heal_lidar_baseline_onnx_mapping(path, audit, _origin(["backbone.conv"]))


def test_fcooper_precision_mapping_keeps_int8_compute_output_and_max_island_fp16(tmp_path: Path) -> None:
    from search.model_family import build_heal_lidar_baseline_precision_mapping

    path = tmp_path / "fcooper.onnx"
    _write_island_onnx(path, "heal_lidar_fcooper")
    audit = _audit(
        "heal_lidar_fcooper",
        [_capability("backbone.conv"), _capability("cls_head", int8=False)],
    )
    origin = _origin(["backbone.conv", "cls_head"])

    mapping, island = build_heal_lidar_baseline_precision_mapping(
        origin,
        {"backbone.conv": "int8", "cls_head": "fp16"},
        audit=audit,
        canonical_onnx_path=path,
        profile_id="mixed",
    )
    by_module = {row.module_path: row for row in mapping.entries}

    assert by_module["backbone.conv"].realized_request_precision == "int8"
    assert by_module["backbone.conv"].realized_output_precision == "fp16"
    assert mapping.auxiliary_layer_precisions["/ReduceMax"] == "fp16"
    assert island["weighted_fusion_int8_forbidden"] is True


def test_precision_mapping_rejects_origin_modules_missing_from_audit(tmp_path: Path) -> None:
    from search.model_family import build_heal_lidar_baseline_precision_mapping

    path = tmp_path / "fcooper.onnx"
    _write_island_onnx(path, "heal_lidar_fcooper")
    audit = _audit("heal_lidar_fcooper", [_capability("backbone.conv")])
    origin = _origin(["backbone.conv", "unknown.conv"])

    with pytest.raises(RuntimeError, match="origin_modules_unaudited"):
        build_heal_lidar_baseline_precision_mapping(
            origin,
            {"backbone.conv": "fp16", "unknown.conv": "fp16"},
            audit=audit,
            canonical_onnx_path=path,
            profile_id="invalid",
        )


def test_disconet_precision_mapping_forbids_pixel_weight_int8(tmp_path: Path) -> None:
    from search.model_family import build_heal_lidar_baseline_precision_mapping

    path = tmp_path / "disco.onnx"
    _write_island_onnx(path, "heal_lidar_disco")
    fusion = "fusion_net.pixel_weight_layer.conv1_1"
    audit = _audit("heal_lidar_disco", [_capability("backbone.conv"), _capability(fusion, int8=False)])
    origin = _origin(["backbone.conv", fusion])

    with pytest.raises(RuntimeError, match="capability_violation"):
        build_heal_lidar_baseline_precision_mapping(
            origin,
            {"backbone.conv": "int8", fusion: "int8"},
            audit=audit,
            canonical_onnx_path=path,
            profile_id="illegal",
        )
    mapping, island = build_heal_lidar_baseline_precision_mapping(
        origin,
        {"backbone.conv": "int8", fusion: "fp16"},
        audit=audit,
        canonical_onnx_path=path,
        profile_id="legal",
    )

    assert mapping.auxiliary_layer_precisions["/Softmax"] == "fp16"
    assert mapping.auxiliary_layer_precisions["/ReduceSum"] == "fp16"
    assert island["weighted_fusion_modules"] == [fusion]


def test_quantization_groups_expose_int8_only_for_audited_modules() -> None:
    from search.model_family import build_heal_lidar_baseline_quantization_groups

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = nn.Conv2d(4, 4, 1)
            self.head = nn.Conv2d(4, 2, 1)

    model = Model()
    audit = _audit(
        "heal_lidar_fcooper",
        [_capability("backbone"), _capability("head", int8=False)],
    )
    groups = build_heal_lidar_baseline_quantization_groups(model, audit)
    by_path = {row.module_paths[0]: row for row in groups}

    assert by_path["backbone"].allowed_precisions == ("FP32", "FP16", "INT8")
    assert by_path["backbone"].protected is False
    assert by_path["head"].allowed_precisions == ("FP32", "FP16")
    assert by_path["head"].protected is True


@pytest.mark.parametrize(
    "family_id",
    ("heal_lidar_fcooper", "heal_lidar_disco"),
)
def test_trt_fusion_island_audit_requires_fp16_and_rejects_int8(family_id: str) -> None:
    from search.model_family import validate_heal_lidar_fusion_island_realization

    names = (
        ("/GridSample", "/Mul_6", "/Where_2", "/ReduceMax")
        if family_id == "heal_lidar_fcooper"
        else ("/GridSample", "/Mul_6", "/Concat_5", "/Where_3", "/Softmax", "/Expand_2", "/Mul_9", "/ReduceSum")
    )
    metadata = "\x1f".join(f"[ONNX Layer: {name}]" for name in names)
    fp16_rows = [{
        "Name": "fused-fusion-island",
        "LayerType": "kgen",
        "Inputs": [{"Format/Datatype": "Half"}],
        "Outputs": [{"Format/Datatype": "Half"}],
        "Metadata": metadata,
    }]
    int8_rows = [{**fp16_rows[0], "Inputs": [{"Format/Datatype": "Int8"}]}]

    assert validate_heal_lidar_fusion_island_realization(fp16_rows, family=family_id)["passed"] is True
    failed = validate_heal_lidar_fusion_island_realization(int8_rows, family=family_id)
    assert failed["passed"] is False
    assert any("realized_int8" in issue for issue in failed["issues"])


def test_wrapper_parity_rejects_missing_outputs() -> None:
    from search.model_family.heal_lidar_deployment import _parity

    reference = {
        "cls_preds": torch.ones(1),
        "reg_preds": torch.ones(1),
        "dir_preds": torch.ones(1),
    }

    result = _parity(reference, (torch.ones(1),), ("cls_preds", "reg_preds", "dir_preds"))

    assert result["passed"] is False
    assert "output_contract_mismatch" in result["failure_reason"]


def test_baseline_evaluator_rejects_inconsistent_family_model_pair(tmp_path: Path) -> None:
    from search.stage2.heal_lidar_baseline_real_evaluator import HealLidarBaselineEvaluationConfig

    with pytest.raises(ValueError, match="family_model_mismatch"):
        HealLidarBaselineEvaluationConfig(
            family_id="heal_lidar_fcooper",
            model_name="lidar_disco",
            model_config_path=tmp_path / "config.yaml",
            checkpoint_path=tmp_path / "checkpoint.pth",
            heal_root=tmp_path / "HEAL",
            tensorrt_root=tmp_path / "TensorRT",
            plugin_path=tmp_path / "plugin.so",
            eval_manifest_path=tmp_path / "manifest.json",
            physical_gpu_id=0,
        )


def test_baseline_real_evaluator_reuses_engine_and_routes_six_input_contract(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from search.stage2.heal_lidar_baseline_real_evaluator import (
        HealLidarBaselineEvaluationConfig,
        HealLidarBaselineRealEvaluator,
    )
    from search.stage2 import heal_lidar_baseline_real_evaluator as evaluator_module

    config_path = tmp_path / "config.yaml"
    checkpoint = tmp_path / "checkpoint.pth"
    plugin = tmp_path / "plugin.so"
    manifest = tmp_path / "eval_manifest.json"
    engine = tmp_path / "candidate.plan"
    heal_root = tmp_path / "HEAL"
    trt_root = tmp_path / "TensorRT"
    heal_root.mkdir()
    trt_root.mkdir()
    for path, content in (
        (config_path, "model: test\n"),
        (checkpoint, "checkpoint"),
        (plugin, "plugin"),
        (manifest, "{}"),
        (engine, "engine"),
    ):
        path.write_text(content, encoding="utf-8")
    calls = {}

    def fake_evaluate(**kwargs):
        calls.update(kwargs)
        return {"status": "ok", "mAP": 0.5, "evaluated_frames": 2}

    monkeypatch.setattr(evaluator_module, "evaluate_v2xvit_engine_modelopt", fake_evaluate)
    evaluator = HealLidarBaselineRealEvaluator(HealLidarBaselineEvaluationConfig(
        family_id="heal_lidar_fcooper",
        model_name="lidar_fcooper",
        model_config_path=config_path,
        checkpoint_path=checkpoint,
        heal_root=heal_root,
        tensorrt_root=trt_root,
        plugin_path=plugin,
        eval_manifest_path=manifest,
        physical_gpu_id=3,
        fixed_k=29696,
        num_frames=2,
        warmup_frames=1,
        latency_rounds=1,
        dataloader_num_workers=0,
    ))

    result = evaluator.evaluate_existing_engine(engine, output_dir=tmp_path / "evaluation")

    assert result["status"] == "ok"
    assert result["engine_rebuilt"] is False
    assert calls["input_contract"] == "heal_lidar_baseline_fixed_k"
    assert calls["fixed_k"] == 29696
    assert calls["max_agents"] == 2
    assert calls["physical_gpu_id"] == 3
    assert (tmp_path / "evaluation/evaluation_acceptance.json").is_file()


def test_explicit_qdq_preserves_fcooper_functional_island(tmp_path: Path) -> None:
    from search.model_family import (
        build_heal_lidar_baseline_precision_mapping,
        insert_heal_lidar_baseline_explicit_qdq,
    )

    source = tmp_path / "source.onnx"
    output = tmp_path / "qdq.onnx"
    _write_island_onnx(source, "heal_lidar_fcooper", include_weighted=True)
    audit = _audit("heal_lidar_fcooper", [_capability("backbone.conv")])
    origin = _origin(["backbone.conv"])
    mapping, _island = build_heal_lidar_baseline_precision_mapping(
        origin,
        {"backbone.conv": "int8"},
        audit=audit,
        canonical_onnx_path=source,
        profile_id="int8",
    )
    scales = {
        "backbone.conv": {
            "activation_input_scale": 0.1,
            "weight_scale": 0.01,
            "activation_output_scale": 0.2,
            "insert_activation_output_qdq": False,
        }
    }

    result, audit_result = insert_heal_lidar_baseline_explicit_qdq(
        source,
        output,
        mapping,
        family="heal_lidar_fcooper",
        scales=scales,
    )

    assert result.inserted_layer_count == 1
    assert audit_result["passed"] is True
    assert output.is_file()
