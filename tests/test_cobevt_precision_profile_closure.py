from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest


def _group(group_id: str, path: str, ordering: int):
    from search.quantization_space.types import QuantizationSearchGroup

    return QuantizationSearchGroup(
        group_id=group_id,
        module_paths=(path,),
        canonical_node_ids=(),
        allowed_precisions=("FP32", "FP16", "INT8"),
        protected=False,
        protection_reason="",
        ordering=ordering,
        parameter_count=1,
        baseline_macs=1.0,
    )


def test_cobevt_unified_space_keeps_nonprunable_runtime_weighted_groups() -> None:
    from search.adapters.transformer_models import (
        build_unified_transformer_search_space,
    )

    base = (
        _group("runtime::backbone", "backbone.block", 0),
        _group("runtime::head", "cls_head", 1),
        _group("runtime::qkv", "fusion.attn.to_qkv", 2),
    )
    transformer = _group(
        "transformer::qkv", "fusion.attn.to_qkv", len(base)
    )
    components = SimpleNamespace(
        transformer_domains=(),
        precision_units=(
            SimpleNamespace(module_paths=("fusion.attn.to_qkv",)),
        ),
        quantization_groups=(transformer,),
    )
    space = build_unified_transformer_search_space(
        components,
        cnn_quantization_groups=base,
    )
    by_id = {row.group_id: row for row in space.quantization_groups}
    assert "runtime::backbone" in by_id
    assert "runtime::head" in by_id
    assert "runtime::qkv" not in by_id
    assert "transformer::qkv" in by_id
    owners = {
        path
        for group in space.quantization_groups
        for path in group.module_paths
    }
    assert owners == {"backbone.block", "cls_head", "fusion.attn.to_qkv"}


def test_cobevt_formal_prepare_passes_complete_runtime_precision_inventory() -> None:
    from search.ga.transformer_stage12_v3 import prepare_cobevt_search

    source = inspect.getsource(prepare_cobevt_search)
    assert (
        "base_quantization_groups=context.search_space.quantization_groups"
        in source
    )
    assert "unified_functional_precision_paths" in source


def test_greedy_records_activation_taylor_pruning_counterfactual_without_using_it() -> None:
    from search.ga.cnn_stage12_v3 import greedy_anchors

    source = inspect.getsource(greedy_anchors)
    assert "trace, captured = run_path(1.0)" in source
    assert "counterfactual_trace, counterfactual_captured = run_path(0.0)" in source
    assert "counterfactual_used_for_winner_selection\": False" in source
    assert "activation_taylor_systematically_pushes_toward_pruning" in source


def test_weighted_and_functional_profiles_partition_exhaustively() -> None:
    from search.model_family.heal_lidar_deployment import (
        partition_heal_lidar_precision_profile,
    )

    weighted, functional = partition_heal_lidar_precision_profile(
        ("backbone", "head"),
        {
            "backbone": "FP16",
            "head": "FP32",
            "attention::__qk_matmul__": "FP32",
        },
        functional_precision_paths=("attention::__qk_matmul__",),
    )
    assert weighted == {"backbone": "fp16", "head": "fp32"}
    assert functional == {"attention::__qk_matmul__": "fp32"}

    with pytest.raises(RuntimeError, match="unknown=\\['invented'\\]"):
        partition_heal_lidar_precision_profile(
            ("backbone",),
            {"backbone": "FP16", "invented": "FP32"},
        )
    with pytest.raises(RuntimeError, match="missing_functional"):
        partition_heal_lidar_precision_profile(
            ("backbone",),
            {"backbone": "FP16"},
            functional_precision_paths=("attention::__av_matmul__",),
        )


def _fp32(name: str, shape: list[int]):
    from onnx import TensorProto, helper

    return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)


def _fp16(name: str, shape: list[int]):
    from onnx import TensorProto, helper

    return helper.make_tensor_value_info(name, TensorProto.FLOAT16, shape)


def _write_cobevt_functional_onnx(path) -> None:
    from onnx import helper, numpy_helper, save

    inputs = [
        _fp32("x", [1, 2, 4]),
        _fp32("ln_scale", [4]),
        _fp32("ln_bias", [4]),
        _fp16("half_a", [1, 2, 4]),
        _fp16("half_b", [1, 2, 4]),
    ]
    initializers = [
        numpy_helper.from_array(np.eye(4, dtype=np.float32), name=name)
        for name in ("q.weight", "k.weight", "v.weight", "ffn2.weight")
    ]
    prefix = "/layers.0/window_attention/fn"
    nodes = [
        helper.make_node("MatMul", ["x", "q.weight"], ["q"], name="q_canonical"),
        helper.make_node("MatMul", ["x", "k.weight"], ["k"], name="k_canonical"),
        helper.make_node("MatMul", ["x", "v.weight"], ["v"], name="v_canonical"),
        helper.make_node("Transpose", ["k"], ["kt"], name="kt", perm=[0, 2, 1]),
        helper.make_node("MatMul", ["q", "kt"], ["score"], name=f"{prefix}/Einsum"),
        helper.make_node("Softmax", ["score"], ["prob"], name=f"{prefix}/attend/attend.0/Softmax", axis=-1),
        helper.make_node("MatMul", ["prob", "v"], ["av"], name=f"{prefix}/Einsum_1"),
        helper.make_node(
            "LayerNormalization", ["x", "ln_scale", "ln_bias"], ["ln"],
            name="/layers.0/window_attention/norm/LayerNormalization", axis=-1,
        ),
        helper.make_node(
            "Add", ["half_a", "half_b"], ["attention_residual"],
            name="/layers.0/window_attention/Add",
        ),
        helper.make_node(
            "MatMul", ["x", "ffn2.weight"], ["ffn"], name="ffn2_canonical"
        ),
        helper.make_node(
            "Add", ["half_a", "half_b"], ["output"],
            name="/layers.0/window_ffd/Add",
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "cobevt-functional",
        inputs,
        [_fp16("output", [1, 2, 4])],
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    save(model, path)


def _unit(unit_id: str, role: str, state: str, owner: str, **metadata):
    return SimpleNamespace(
        unit_id=unit_id,
        role=role,
        default_state=state,
        activation_only=role != "ffn2",
        metadata={"functional_owner": owner, **metadata},
    )


def test_cobevt_functional_onnx_mapping_closes_attention_and_residuals(tmp_path) -> None:
    from search.stage2.v2xvit_functional_precision import (
        build_v2xvit_functional_onnx_mapping,
    )

    path = tmp_path / "cobevt.onnx"
    _write_cobevt_functional_onnx(path)
    attention_path = "fusion_net.layers.0.window_attention.fn"
    ffn_path = "fusion_net.layers.0.window_ffd.fn"
    units = (
        _unit(f"transformer_precision::{attention_path}::qk_matmul", "qk_matmul", "A32", attention_path),
        _unit(f"transformer_precision::{attention_path}::softmax", "softmax", "A32", attention_path),
        _unit(f"transformer_precision::{attention_path}::av", "av_matmul", "A32", attention_path),
        _unit("attention-residual", "residual_add", "A16", attention_path, boundary_kind="attention_residual_add"),
        _unit("attention-layernorm", "layernorm", "A32", "fusion_net.layers.0.window_attention.norm"),
        _unit(f"transformer_precision::{ffn_path}::ffn2", "ffn2", "W32A32", ffn_path),
        _unit("ffn-residual", "residual_add", "A16", ffn_path, boundary_kind="ffn_residual_add"),
    )
    attention = SimpleNamespace(
        module_path=attention_path,
        q_projection_paths=("attn.q",),
        k_projection_paths=("attn.k",),
        v_projection_paths=("attn.v",),
    )
    ffn = SimpleNamespace(
        module_path=ffn_path,
        ffn_type="standard",
        second_projection_path="ffn.fc2",
        down_projection_path="",
    )
    origin = SimpleNamespace(entries=(
        SimpleNamespace(module_path="attn.q", canonical_node_name="q_canonical"),
        SimpleNamespace(module_path="attn.k", canonical_node_name="k_canonical"),
        SimpleNamespace(module_path="attn.v", canonical_node_name="v_canonical"),
        SimpleNamespace(module_path="ffn.fc2", canonical_node_name="ffn2_canonical"),
    ))
    report = build_v2xvit_functional_onnx_mapping(
        path,
        origin_map=origin,
        precision_units=units,
        attention_instances=(attention,),
        ffn_instances=(ffn,),
        requested_states={unit.unit_id: unit.default_state for unit in units},
    )
    assert report["passed"]
    assert report["missing_unit_ids"] == []
    by_unit = {row["unit_id"]: row for row in report["rows"]}
    assert by_unit["attention-residual"]["onnx_node"] == "/layers.0/window_attention/Add"
    assert by_unit["ffn-residual"]["onnx_node"] == "/layers.0/window_ffd/Add"
