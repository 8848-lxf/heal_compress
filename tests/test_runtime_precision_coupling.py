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


def test_merge_audit_is_linear_on_reconvergent_dag(tmp_path) -> None:
    """A diamond-heavy ONNX graph must not trigger per-merge graph walks."""

    import time

    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    from quantization.config import QDQConfig
    from quantization.precision.qdq_inserter import insert_explicit_qdq
    from quantization.types import (
        CanonicalPrecisionEntry,
        CanonicalPrecisionMappingResult,
    )

    depth = 26
    nodes = [
        helper.make_node("Conv", ["x", "up.weight"], ["level_0"], name="weighted_upstream")
    ]
    current = "level_0"
    for level in range(1, depth + 1):
        left = f"level_{level}_left"
        right = f"level_{level}_right"
        output = f"level_{level}"
        nodes.extend(
            [
                helper.make_node("Identity", [current], [left], name=f"identity_{level}_left"),
                helper.make_node("Identity", [current], [right], name=f"identity_{level}_right"),
                helper.make_node("Add", [left, right], [output], name=f"diamond_add_{level}"),
            ]
        )
        current = output
    nodes.append(
        helper.make_node(
            "Conv", [current, "down.weight"], ["y"], name="weighted_downstream"
        )
    )
    graph = helper.make_graph(
        nodes,
        "reconvergent_merge_audit",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4, 2, 2])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4, 2, 2])],
        [
            numpy_helper.from_array(
                np.ones((4, 4, 1, 1), dtype=np.float32), name="up.weight"
            ),
            numpy_helper.from_array(
                np.ones((4, 4, 1, 1), dtype=np.float32), name="down.weight"
            ),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.checker.check_model(model)
    source = tmp_path / "reconvergent.onnx"
    destination = tmp_path / "reconvergent_qdq.onnx"
    onnx.save(model, source)
    mapping = CanonicalPrecisionMappingResult(
        entries=[
            CanonicalPrecisionEntry(
                "up",
                "weighted_upstream",
                "up_group",
                "fp32",
                "fp32",
                weight_initializer="up.weight",
                onnx_op_type="Conv",
            ),
            CanonicalPrecisionEntry(
                "down",
                "weighted_downstream",
                "down_group",
                "fp32",
                "fp32",
                weight_initializer="down.weight",
                onnx_op_type="Conv",
            ),
        ]
    )

    started = time.monotonic()
    result = insert_explicit_qdq(
        source,
        destination,
        mapping,
        scales={},
        config=QDQConfig(
            allowed_precisions=("fp32", "fp16"),
            require_calibration_scales=False,
            insert_activation_output_qdq=False,
            explicit_fp16_compute_casts=False,
            explicit_fp32_compute_casts=False,
        ),
    )
    elapsed = time.monotonic() - started

    audit = result.calibration_metadata["merge_quantization_audit"]
    assert len(audit) == depth
    assert elapsed < 5.0
    assert all(
        {
            producer["canonical_node"]
            for branch in row["input_branches"]
            for producer in branch["nearest_weighted_producers"]
        }
        == {"weighted_upstream"}
        for row in audit
    )
    assert all(
        {consumer["canonical_node"] for consumer in row["downstream_weighted_layers"]}
        == {"weighted_downstream"}
        for row in audit
    )


def _int64_shape_where_fixture():
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    from quantization.types import CanonicalPrecisionEntry, CanonicalPrecisionMappingResult

    graph = helper.make_graph(
        [
            helper.make_node("Conv", ["x", "weight"], ["feature"], name="conv"),
            helper.make_node("Shape", ["feature"], ["shape"], name="shape"),
            helper.make_node("Equal", ["shape", "shape"], ["condition"], name="shape_equal"),
            helper.make_node(
                "Where",
                ["condition", "shape", "shape"],
                ["selected_shape"],
                name="shape_where",
            ),
            helper.make_node("Expand", ["feature", "selected_shape"], ["y"], name="expand"),
        ],
        "int64_shape_where",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4, 2, 2])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4, 2, 2])],
        [
            numpy_helper.from_array(
                np.ones((4, 4, 1, 1), dtype=np.float32),
                name="weight",
            )
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.checker.check_model(model)
    mapping = CanonicalPrecisionMappingResult(
        entries=[
            CanonicalPrecisionEntry(
                "conv",
                "conv",
                "pg_conv",
                "fp32",
                "fp32",
                weight_initializer="weight",
                onnx_op_type="Conv",
            )
        ]
    )
    return model, mapping


def test_adaptive_merge_excludes_int64_where_shape_subgraph(tmp_path) -> None:
    import onnx

    from quantization.config import QDQConfig
    from quantization.precision.merge_contract import apply_adaptive_merge_output_contract
    from quantization.precision.qdq_inserter import insert_explicit_qdq

    model, mapping = _int64_shape_where_fixture()
    resolved, report = apply_adaptive_merge_output_contract(model, mapping)

    assert report["resolved_merge_count"] == 0
    excluded = next(
        row for row in report["excluded_merges"] if row["merge_op_name"] == "shape_where"
    )
    assert excluded["dtype_audit"]["data_input_dtypes"] == ["INT64", "INT64"]
    assert excluded["dtype_audit"]["output_dtypes"] == ["INT64"]
    assert "shape_where" not in resolved.auxiliary_layer_precisions

    input_path = tmp_path / "shape_input.onnx"
    output_path = tmp_path / "shape_qdq.onnx"
    onnx.save(model, input_path)
    result = insert_explicit_qdq(
        input_path,
        output_path,
        resolved,
        scales={},
        config=QDQConfig(
            allowed_precisions=("fp32", "fp16", "int8"),
            merge_policy="adaptive_upcast_merge",
        ),
    )
    qdq_model = onnx.load(output_path)
    onnx.checker.check_model(qdq_model)
    shape_where = next(node for node in qdq_model.graph.node if node.name == "shape_where")
    assert list(shape_where.input) == ["condition", "shape", "shape"]
    assert result.calibration_metadata["adaptive_merge_cast_records"] == []


def test_adaptive_merge_requires_runtime_relation_kind_to_match_onnx_op(tmp_path) -> None:
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    from quantization.precision.merge_contract import apply_adaptive_merge_output_contract
    from quantization.types import CanonicalPrecisionEntry, CanonicalPrecisionMappingResult

    graph = helper.make_graph(
        [
            helper.make_node("Conv", ["x", "wa"], ["a"], name="conv_a"),
            helper.make_node("Conv", ["x", "wb"], ["b"], name="conv_b"),
            helper.make_node("Where", ["condition", "a", "b"], ["selected"], name="float_where"),
            helper.make_node("Mul", ["a", "b"], ["y"], name="activation_mul"),
        ],
        "runtime_relation_kind",
        [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4, 2, 2]),
            helper.make_tensor_value_info("condition", TensorProto.BOOL, [1, 4, 2, 2]),
        ],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4, 2, 2])],
        [
            numpy_helper.from_array(np.ones((4, 4, 1, 1), dtype=np.float32), name="wa"),
            numpy_helper.from_array(np.ones((4, 4, 1, 1), dtype=np.float32), name="wb"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.checker.check_model(model)
    mapping = CanonicalPrecisionMappingResult(
        entries=[
            CanonicalPrecisionEntry("a", "conv_a", "pg_a", "fp16", "fp16", weight_initializer="wa", onnx_op_type="Conv"),
            CanonicalPrecisionEntry("b", "conv_b", "pg_b", "fp16", "fp16", weight_initializer="wb", onnx_op_type="Conv"),
        ]
    )

    resolved, report = apply_adaptive_merge_output_contract(
        model,
        mapping,
        runtime_relations=[
            {
                "relation_id": "runtime_mul",
                "relation_kind": "elementwise_multiply",
                "member_modules": ["a", "b"],
            }
        ],
    )

    assert [row["merge_op_name"] for row in report["merges"]] == ["activation_mul"]
    assert report["runtime_relation_matches"][0]["canonical_merge_nodes"] == ["activation_mul"]
    assert "activation_mul" in resolved.auxiliary_layer_precisions
    assert "float_where" not in resolved.auxiliary_layer_precisions
    excluded = next(
        row for row in report["excluded_merges"] if row["merge_op_name"] == "float_where"
    )
    assert excluded["exclusion_reason"] == "no_compatible_runtime_relation"

    from quantization.config import QDQConfig
    from quantization.precision.qdq_inserter import insert_explicit_qdq

    input_path = tmp_path / "runtime_relation_input.onnx"
    output_path = tmp_path / "runtime_relation_qdq.onnx"
    onnx.save(model, input_path)
    result = insert_explicit_qdq(
        input_path,
        output_path,
        resolved,
        scales={},
        config=QDQConfig(
            allowed_precisions=("fp32", "fp16", "int8"),
            merge_policy="adaptive_upcast_merge",
        ),
    )
    assert [
        row["merge_op_name"]
        for row in result.calibration_metadata["merge_quantization_audit"]
    ] == ["activation_mul"]
