from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class RuntimeMergeToy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Conv2d(3, 4, 1)
        self.branch_a = nn.Conv2d(4, 4, 1)
        self.branch_b = nn.Conv2d(4, 4, 1)
        self.shared = nn.Conv2d(4, 4, 1)
        self.fuse = nn.Conv2d(8, 4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The tensor-dependent Python branch rejects symbolic FX tracing and
        # proves this test consumes the real runtime tensor-flow inventory.
        if bool((x.sum() > -1.0).item()):
            x = self.stem(x)
        merged = self.branch_a(x) + self.branch_b(x)
        a = self.shared(merged)
        b = self.shared(x)
        return self.fuse(torch.cat([a + b, x], dim=1))


def test_runtime_precision_relations_cover_every_weighted_module_once() -> None:
    from tracer.api import trace_model
    from tracer.config import TraceConfig
    from tracer.precision_coupling_tracer import build_runtime_precision_coupling

    model = RuntimeMergeToy().eval()
    trace = trace_model(
        model,
        torch.ones(1, 3, 4, 4),
        config=TraceConfig(fail_on_fx_trace_error=False),
    )
    assert trace.config["realized_backend"] == "runtime_tensor_flow"

    result = build_runtime_precision_coupling(model, trace)

    members = [name for group in result.groups for name in group.member_modules]
    assert sorted(members) == result.weighted_modules
    assert len(members) == len(set(members))
    assert result.weighted_module_call_counts["shared"] == 2
    assert sum("shared" in group.member_modules for group in result.groups) == 1
    assert {relation.relation_kind for relation in result.relations} >= {
        "residual_add",
        "concat",
    }
    assert all(not relation.force_same_precision for relation in result.relations)
    assert all(relation.merge_precision == "derived_per_candidate" for relation in result.relations)
    assert all(group.allowed_precisions == ["fp32", "fp16", "int8"] for group in result.groups)


def test_runtime_precision_relation_survives_functional_softmax() -> None:
    from tracer.api import trace_model
    from tracer.config import TraceConfig
    from tracer.precision_coupling_tracer import build_runtime_precision_coupling

    class SoftmaxFusion(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.feature = nn.Conv2d(3, 4, 1)
            self.weight = nn.Conv2d(3, 4, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if bool((x.sum() > -1.0).item()):
                feature = self.feature(x)
            return feature * F.softmax(self.weight(x), dim=1)

    model = SoftmaxFusion().eval()
    trace = trace_model(
        model,
        torch.ones(1, 3, 2, 2),
        config=TraceConfig(fail_on_fx_trace_error=False),
    )
    result = build_runtime_precision_coupling(model, trace)
    multiply = next(
        relation
        for relation in result.relations
        if relation.relation_kind == "elementwise_multiply"
    )
    assert multiply.member_modules == ["feature", "weight"]


def _adaptive_merge_fixture(left: str, right: str):
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    from quantization.types import CanonicalPrecisionEntry, CanonicalPrecisionMappingResult

    weights = [
        numpy_helper.from_array(np.ones((4, 4, 1, 1), dtype=np.float32), name="wa"),
        numpy_helper.from_array(np.ones((4, 4, 1, 1), dtype=np.float32), name="wb"),
        numpy_helper.from_array(np.ones((4, 4, 1, 1), dtype=np.float32), name="wf"),
    ]
    graph = helper.make_graph(
        [
            helper.make_node("Conv", ["x", "wa"], ["a"], name="conv_a"),
            helper.make_node("Conv", ["x", "wb"], ["b"], name="conv_b"),
            helper.make_node("Add", ["a", "b"], ["merged"], name="merge_add"),
            helper.make_node("Conv", ["merged", "wf"], ["y"], name="conv_fuse"),
        ],
        "adaptive_merge",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4, 2, 2])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4, 2, 2])],
        weights,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.checker.check_model(model)
    mapping = CanonicalPrecisionMappingResult(
        entries=[
            CanonicalPrecisionEntry("a", "conv_a", "pg_a", left, left, weight_initializer="wa", onnx_op_type="Conv"),
            CanonicalPrecisionEntry("b", "conv_b", "pg_b", right, right, weight_initializer="wb", onnx_op_type="Conv"),
            CanonicalPrecisionEntry("fuse", "conv_fuse", "pg_f", "fp16", "fp16", weight_initializer="wf", onnx_op_type="Conv"),
        ]
    )
    return model, mapping


def test_adaptive_merge_keeps_equal_precision_and_promotes_mixed_branches() -> None:
    from quantization.precision.merge_contract import apply_adaptive_merge_output_contract

    cases = [
        ("int8", "int8", "int8"),
        ("fp16", "fp16", "fp16"),
        ("int8", "fp16", "fp16"),
        ("int8", "fp32", "fp32"),
    ]
    for left, right, expected in cases:
        model, mapping = _adaptive_merge_fixture(left, right)
        resolved, report = apply_adaptive_merge_output_contract(model, mapping)
        merge = report["merges"][0]
        assert merge["derived_merge_precision"] == expected
        assert merge["all_branch_precisions_equal"] == (left == right)
        by_module = {row.module_path: row for row in resolved.entries}
        assert by_module["a"].realized_output_precision == left
        assert by_module["b"].realized_output_precision == right
        assert resolved.auxiliary_layer_precisions["merge_add"] == expected


def test_adaptive_int8_merge_qdq_keeps_dequantized_inputs_and_no_float_merge_cast(tmp_path) -> None:
    import onnx

    from quantization.config import QDQConfig
    from quantization.precision.merge_contract import apply_adaptive_merge_output_contract
    from quantization.precision.qdq_inserter import insert_explicit_qdq

    model, mapping = _adaptive_merge_fixture("int8", "int8")
    resolved, _report = apply_adaptive_merge_output_contract(model, mapping)
    input_path = tmp_path / "input.onnx"
    output_path = tmp_path / "qdq.onnx"
    onnx.save(model, input_path)
    scales = {
        "a": {"activation_input_scale": 0.1, "weight_scale": 0.1, "activation_output_scale": 0.2},
        "b": {"activation_input_scale": 0.1, "weight_scale": 0.1, "activation_output_scale": 0.2},
    }
    result = insert_explicit_qdq(
        input_path,
        output_path,
        resolved,
        scales=scales,
        config=QDQConfig(
            allowed_precisions=("fp32", "fp16", "int8"),
            merge_policy="adaptive_upcast_merge",
        ),
    )

    qdq_model = onnx.load(output_path)
    merge = next(node for node in qdq_model.graph.node if node.name == "merge_add")
    producers = {str(output): node for node in qdq_model.graph.node for output in node.output}
    assert all(producers[str(value)].op_type == "DequantizeLinear" for value in merge.input)
    assert result.calibration_metadata["adaptive_merge_cast_records"] == []
    audit = next(
        row
        for row in result.calibration_metadata["merge_quantization_audit"]
        if row["merge_op_name"] == "merge_add"
    )
    assert audit["derived_merge_precision"] == "int8"


def test_adaptive_mixed_merge_promotes_only_at_merge_edges(tmp_path) -> None:
    import onnx

    from quantization.config import QDQConfig
    from quantization.precision.merge_contract import apply_adaptive_merge_output_contract
    from quantization.precision.qdq_inserter import insert_explicit_qdq

    model, mapping = _adaptive_merge_fixture("int8", "fp16")
    resolved, report = apply_adaptive_merge_output_contract(model, mapping)
    by_module = {row.module_path: row for row in resolved.entries}
    assert by_module["a"].realized_output_precision == "int8"
    assert by_module["b"].realized_output_precision == "fp16"
    assert report["merges"][0]["derived_merge_precision"] == "fp16"

    input_path = tmp_path / "mixed_input.onnx"
    output_path = tmp_path / "mixed_qdq.onnx"
    onnx.save(model, input_path)
    result = insert_explicit_qdq(
        input_path,
        output_path,
        resolved,
        scales={
            "a": {
                "activation_input_scale": 0.1,
                "weight_scale": 0.1,
                "activation_output_scale": 0.2,
            }
        },
        config=QDQConfig(
            allowed_precisions=("fp32", "fp16", "int8"),
            merge_policy="adaptive_upcast_merge",
        ),
    )
    casts = result.calibration_metadata["adaptive_merge_cast_records"]
    assert len([row for row in casts if row["merge_op_name"] == "merge_add"]) == 2
    assert all(row["cast_dtype"] == "FP16" for row in casts)
