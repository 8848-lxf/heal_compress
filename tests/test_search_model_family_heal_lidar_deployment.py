from __future__ import annotations

import json
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


def test_prune_only_auxiliary_fp32_contract_is_propagated_and_realized(
    tmp_path: Path,
) -> None:
    from search.model_family import (
        build_heal_lidar_baseline_precision_mapping,
        insert_heal_lidar_baseline_explicit_qdq,
        validate_heal_lidar_precision_realization,
    )

    path = tmp_path / "fcooper.onnx"
    qdq_path = tmp_path / "fcooper_fp32_qdq.onnx"
    _write_island_onnx(path, "heal_lidar_fcooper", include_weighted=True)
    audit = _audit("heal_lidar_fcooper", [_capability("backbone.conv")])
    origin = _origin(["backbone.conv"])
    prune_only, contract = build_heal_lidar_baseline_precision_mapping(
        origin,
        {"backbone.conv": "fp32"},
        audit=audit,
        canonical_onnx_path=path,
        profile_id="prune-only",
        auxiliary_precision="fp32",
    )
    normal, _ = build_heal_lidar_baseline_precision_mapping(
        origin,
        {"backbone.conv": "fp32"},
        audit=audit,
        canonical_onnx_path=path,
        profile_id="normal",
    )

    assert set(prune_only.auxiliary_layer_precisions.values()) == {"fp32"}
    assert set(prune_only.auxiliary_layer_output_types.values()) == {"fp32"}
    assert set(normal.auxiliary_layer_precisions.values()) == {"fp16"}
    assert contract["auxiliary_precision"] == "fp32"

    weighted = {
        "Name": "weighted",
        "LayerType": "Convolution",
        "Inputs": [{"Format/Datatype": "Float"}],
        "Outputs": [{"Format/Datatype": "Float"}],
        "Metadata": "[ONNX Layer: __canonical__0]",
    }
    auxiliary_metadata = "\x1f".join(
        f"[ONNX Layer: {name}]"
        for name in sorted(prune_only.auxiliary_layer_precisions)
    )
    auxiliary = {
        "Name": "fused-auxiliary",
        "LayerType": "kgen",
        "Inputs": [{"Format/Datatype": "Float"}],
        "Outputs": [{"Format/Datatype": "Float"}],
        "Metadata": auxiliary_metadata,
    }
    rows = [weighted, auxiliary]

    accepted = validate_heal_lidar_precision_realization(
        rows, prune_only, family="heal_lidar_fcooper"
    )
    rejected = validate_heal_lidar_precision_realization(
        rows, normal, family="heal_lidar_fcooper"
    )
    assert accepted["passed"] is True
    assert accepted["fusion_island"]["required_precision"] == "fp32"
    assert rejected["passed"] is False
    assert any(
        "not_fp16" in issue for issue in rejected["fusion_island"]["issues"]
    )

    _result, qdq_audit = insert_heal_lidar_baseline_explicit_qdq(
        path,
        qdq_path,
        prune_only,
        family="heal_lidar_fcooper",
        scales={},
    )
    assert qdq_audit["passed"] is True
    assert qdq_audit["auxiliary_precision"] == "fp32"
    assert set(qdq_audit["inserted_auxiliary_precisions"].values()) == {"fp32"}
    assert set(qdq_audit["inserted_auxiliary_output_types"].values()) == {"fp32"}


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


def test_runtime_graph_policy_allows_disconet_weighted_fusion_int8_without_named_rules(
    tmp_path: Path,
) -> None:
    from search.model_family import build_heal_lidar_baseline_precision_mapping

    path = tmp_path / "disco.onnx"
    _write_island_onnx(path, "heal_lidar_disco")
    fusion = "fusion_net.pixel_weight_layer.conv1_1"
    audit = _audit("heal_lidar_disco", [_capability(fusion, int8=False)])

    mapping, island = build_heal_lidar_baseline_precision_mapping(
        _origin([fusion]),
        {fusion: "int8"},
        audit=audit,
        canonical_onnx_path=path,
        profile_id="runtime-graph",
        precision_policy="heal_runtime_graph_v1",
    )

    assert mapping.entries[0].requested_precision == "int8"
    assert mapping.entries[0].protected_precision == ""
    assert "/Softmax" not in mapping.auxiliary_layer_precisions
    assert island["family_audit_used_for_precision_protection"] is False
    assert island["family_named_node_rules_used"] is False


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


def test_wrapper_parity_allows_sparse_fixed_k_roundoff_but_rejects_systemic_drift() -> None:
    from search.model_family.heal_lidar_deployment import _parity

    expected = torch.zeros(1, 1, 10, 10)
    sparse = expected.clone()
    sparse[..., 0, 0] = 3.5e-3
    systemic = expected + 1.0e-3
    reference = {name: expected for name in ("cls", "reg", "dir")}

    accepted = _parity(reference, (sparse, sparse, sparse), ("cls", "reg", "dir"))
    rejected = _parity(
        reference,
        (systemic, systemic, systemic),
        ("cls", "reg", "dir"),
    )

    assert accepted["passed"] is True
    assert accepted["outputs"]["cls"]["max_abs_tolerance"] == 5.0e-3
    assert rejected["passed"] is False
    assert rejected["outputs"]["cls"]["mean_abs"] > 5.0e-5


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
            eval_manifest_path=tmp_path / "manifest.json",
            physical_gpu_id=0,
        )


def test_baseline_real_evaluator_reuses_engine_and_routes_post_scatter_contract(
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
        eval_manifest_path=manifest,
        physical_gpu_id=3,
        num_frames=2,
        warmup_frames=1,
        latency_rounds=1,
        dataloader_num_workers=0,
    ))

    result = evaluator.evaluate_existing_engine(engine, output_dir=tmp_path / "evaluation")

    assert result["status"] == "ok"
    assert result["engine_rebuilt"] is False
    assert calls["input_contract"] == "heal_post_scatter_dynamic_frontend_v1"
    assert calls["fixed_k"] is None
    assert calls["checkpoint_path"] == checkpoint
    assert calls["max_agents"] == 2
    assert calls["physical_gpu_id"] == 3
    assert (tmp_path / "evaluation/evaluation_acceptance.json").is_file()


def test_runtime_graph_engine_acceptance_does_not_use_family_node_audit(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from search.stage2 import heal_lidar_baseline_real_evaluator as evaluator_module
    from search.stage2.heal_lidar_baseline_real_evaluator import (
        HealLidarBaselineEvaluationConfig,
        HealLidarBaselineRealEvaluator,
    )

    config_path = tmp_path / "config.yaml"
    checkpoint = tmp_path / "checkpoint.pth"
    plugin = tmp_path / "plugin.so"
    manifest = tmp_path / "eval_manifest.json"
    heal_root = tmp_path / "HEAL"
    trt_root = tmp_path / "TensorRT"
    heal_root.mkdir()
    trt_root.mkdir()
    for path in (config_path, checkpoint, plugin, manifest):
        path.write_text("test", encoding="utf-8")

    weighted = {
        "schema_version": "precision-realization-v1",
        "passed": True,
        "requested_int8_count": 1,
        "realized_int8_count": 1,
        "realized_fp16_count": 0,
        "mismatches": [],
        "hidden_cast_count": 0,
        "reformat_count": 0,
        "boundary_count": 1,
        "unresolved_layer_count": 0,
    }
    structure = {"schema_version": "engine-structure-validation-v1", "passed": True}

    def fake_build_engine_modelopt(**kwargs):
        engine = Path(kwargs["engine_path"])
        engine.parent.mkdir(parents=True, exist_ok=True)
        engine.write_bytes(b"engine")
        return {
            "status": "ok",
            "precision_realization_validation": weighted,
            "engine_structure_validation": structure,
        }

    monkeypatch.setattr(
        evaluator_module,
        "build_engine_modelopt",
        fake_build_engine_modelopt,
    )
    monkeypatch.setattr(
        evaluator_module,
        "audit_post_scatter_onnx",
        lambda _path: {
            "passed": True,
            "issues": [],
            "input_names": [
                "spatial_features",
                "pairwise_t_matrix",
                "agent_mask",
            ],
        },
    )
    monkeypatch.setattr(
        evaluator_module,
        "validate_heal_lidar_precision_realization",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("family-specific node audit must not run")
        ),
    )
    evaluator = HealLidarBaselineRealEvaluator(
        HealLidarBaselineEvaluationConfig(
            family_id="heal_lidar_disco",
            model_name="lidar_disco",
            model_config_path=config_path,
            checkpoint_path=checkpoint,
            heal_root=heal_root,
            tensorrt_root=trt_root,
            eval_manifest_path=manifest,
            physical_gpu_id=5,
            num_frames=1,
            warmup_frames=1,
            latency_rounds=1,
            search_space_policy="heal_runtime_graph_v1",
        )
    )

    result = evaluator.build_engine(
        nn.Conv2d(1, 1, 1),
        {"qdq_onnx_path": tmp_path / "candidate.onnx", "mapping": object()},
        output_dir=tmp_path / "deployment",
    )

    assert result["precision_acceptance"]["passed"] is True
    assert result["precision_acceptance"]["weighted_precision"] == weighted
    assert (
        result["precision_acceptance"]["fusion_island"]["manual_family_node_audit_used"]
        is False
    )


def test_baseline_candidate_result_publishes_generic_stage2_score(tmp_path: Path) -> None:
    from search.stage2.heal_lidar_baseline_real_evaluator import (
        HealLidarBaselineCandidateEvaluator,
    )

    evaluator = object.__new__(HealLidarBaselineCandidateEvaluator)
    result = {"status": "ok", "F2": 0.25, "candidate_hash": "candidate"}

    evaluator._write_candidate_result(tmp_path, result)

    assert json.loads((tmp_path / "candidate_stage2_result.json").read_text()) == result
    assert json.loads((tmp_path / "stage2_score.json").read_text()) == result


def test_baseline_candidate_build_only_api_does_not_evaluate(
    tmp_path: Path, monkeypatch
) -> None:
    from types import SimpleNamespace

    from search.candidate import CandidatePhenotype, PrecisionDecision
    from search.stage2.heal_lidar_baseline_real_evaluator import (
        HealLidarBaselineCandidateEvaluator,
    )
    from search.stage2 import heal_lidar_baseline_real_evaluator as evaluator_module

    model = nn.Conv2d(1, 1, 1)

    class Provider:
        @staticmethod
        def audit(*_args, **_kwargs):
            return object()

    context = SimpleNamespace(
        model=model,
        model_bundle=SimpleNamespace(provider=Provider(), config={}),
        trace_example_inputs={},
        family_id="heal_lidar_fcooper",
    )
    evaluator = object.__new__(HealLidarBaselineCandidateEvaluator)
    evaluator.context = context

    def fake_materialize(_phenotype, output_dir):
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "pruned_checkpoint.pth").write_bytes(b"checkpoint")
        return {"model": model}

    evaluator._materialize = fake_materialize
    evaluator._calibration_scales = lambda **_kwargs: ({}, {"frame_count": 0})

    source_onnx = tmp_path / "source.onnx"
    source_onnx.write_bytes(b"onnx")
    export = SimpleNamespace(
        export=SimpleNamespace(origin_map=object(), onnx_path=source_onnx)
    )

    class Mapping:
        entries = [SimpleNamespace(requested_precision="int8")]

        @staticmethod
        def to_dict():
            return {"entries": [{"requested_precision": "int8"}]}

    mapping = Mapping()
    qdq_path = tmp_path / "qdq.onnx"

    class QDQ:
        output_onnx = str(qdq_path)

        @staticmethod
        def to_dict():
            return {"output_onnx": str(qdq_path)}

    class RealEvaluator:
        @staticmethod
        def export_candidate(*_args, **_kwargs):
            return export

        @staticmethod
        def build_engine(_model, _qdq, *, output_dir):
            output_dir.mkdir(parents=True, exist_ok=True)
            engine = output_dir / "candidate.plan"
            engine.write_bytes(b"engine")
            acceptance = {
                "passed": True,
                "weighted_precision": {"realized_int8_count": 1},
                "fusion_island": {"passed": True},
            }
            (output_dir / "precision_realization_acceptance.json").write_text(
                json.dumps(acceptance), encoding="utf-8"
            )
            return {
                "engine_path": engine,
                "engine_sha256": "engine-hash",
                "precision_acceptance": acceptance,
            }

        @staticmethod
        def evaluate_existing_engine(*_args, **_kwargs):
            raise AssertionError("build-only API must not evaluate")

    evaluator._real_evaluator = lambda **_kwargs: RealEvaluator()
    mapping_call = {}

    def fake_mapping(*_args, **kwargs):
        mapping_call.update(kwargs)
        return mapping, {"policy": "fp32_fusion"}

    monkeypatch.setattr(
        evaluator_module,
        "build_heal_lidar_baseline_precision_mapping",
        fake_mapping,
    )

    def fake_insert(*_args, **_kwargs):
        qdq_path.write_bytes(b"qdq")
        return QDQ(), {"passed": True}

    monkeypatch.setattr(
        evaluator_module, "insert_heal_lidar_baseline_explicit_qdq", fake_insert
    )
    phenotype = CandidatePhenotype(
        pruned_unit_ids=[],
        precision_profile={"conv": PrecisionDecision("INT8", "INT8", "")},
        pruning_policy_version="test",
        precision_policy_version="test",
        metadata={"heal_lidar_auxiliary_precision": "FP32"},
    )

    result = evaluator.build_candidate_artifacts(
        phenotype,
        output_dir=tmp_path / "candidate",
        candidate_hash="candidate",
    )

    assert result["status"] == "ok"
    assert result["engine_built_this_call"] is True
    assert result["evaluation_invoked"] is False
    assert result["auxiliary_precision"] == "fp32"
    assert mapping_call["auxiliary_precision"] == "fp32"
    assert (tmp_path / "candidate/candidate_build_result.json").is_file()
    assert not (tmp_path / "candidate/evaluation").exists()


def test_heal_lidar_onnx_export_critical_section_is_serialized() -> None:
    import threading
    import time

    from search.model_family.heal_lidar_deployment import (
        _serialized_heal_lidar_onnx_export,
    )

    first_entered = threading.Event()
    release_first = threading.Event()
    second_entered = threading.Event()

    def first() -> None:
        with _serialized_heal_lidar_onnx_export():
            first_entered.set()
            assert release_first.wait(timeout=2.0)

    def second() -> None:
        assert first_entered.wait(timeout=2.0)
        with _serialized_heal_lidar_onnx_export():
            second_entered.set()

    first_thread = threading.Thread(target=first)
    second_thread = threading.Thread(target=second)
    first_thread.start()
    second_thread.start()
    assert first_entered.wait(timeout=2.0)
    time.sleep(0.05)
    assert not second_entered.is_set()
    release_first.set()
    first_thread.join(timeout=2.0)
    second_thread.join(timeout=2.0)
    assert second_entered.is_set()


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
