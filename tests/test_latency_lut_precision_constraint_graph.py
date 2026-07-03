from __future__ import annotations

from opencood.tools.compression.latency_lut.precision_constraint_graph import (
    CAST_BOUNDARY_ALLOWED,
    MUST_SAME_DTYPE_BEFORE_OP,
    PrecisionConstraintGraph,
    PrecisionConstraintNode,
)
from opencood.tools.compression.latency_lut.precision_resolver import resolve_precision_constraints


def test_resolver_auto_promotes_fp16_fp32_add_region():
    graph = PrecisionConstraintGraph()
    graph.add_node(PrecisionConstraintNode("branch_a", "precision_region", precision="FP16"))
    graph.add_node(PrecisionConstraintNode("branch_b", "precision_region", precision="FP32"))
    graph.add_edge("branch_a", "branch_b", MUST_SAME_DTYPE_BEFORE_OP, reason="residual_add_requires_same_dtype")

    result = resolve_precision_constraints(
        graph,
        {
            "candidate_id": "toy",
            "precision_config": {
                "default": "FP16",
                "overrides": {"branch_b": "FP32"},
            },
        },
    )

    assert result["success"] is True
    assert result["resolved_precision_config"]["branch_a"] == "FP32"
    assert result["resolved_precision_config"]["branch_b"] == "FP32"
    assert result["auto_promoted_precision_regions"]
    assert result["auto_promoted_precision_regions"][0]["reason"] == "residual_add_requires_same_dtype"


def test_resolver_rejects_int8_merge_region_without_int8_add_support():
    graph = PrecisionConstraintGraph()
    graph.add_node(PrecisionConstraintNode("int8_branch", "precision_region", precision="INT8"))
    graph.add_node(PrecisionConstraintNode("fp16_branch", "precision_region", precision="FP16"))
    graph.add_edge("int8_branch", "fp16_branch", MUST_SAME_DTYPE_BEFORE_OP, reason="residual_add_requires_same_dtype")

    result = resolve_precision_constraints(
        graph,
        {
            "candidate_id": "toy_int8",
            "precision_config": {
                "default": "FP16",
                "overrides": {"int8_branch": "INT8"},
            },
        },
    )

    assert result["success"] is False
    assert result["status"] == "int8_residual_merge_not_supported"
    assert result["unsupported_int8_regions"]


def test_precision_constraint_graph_serializes_boundary_edges():
    graph = PrecisionConstraintGraph()
    graph.add_node(PrecisionConstraintNode("backbone.stage1", "deployment_unit"))
    graph.add_node(PrecisionConstraintNode("backbone.stage2", "deployment_unit"))
    graph.add_edge("backbone.stage1", "backbone.stage2", CAST_BOUNDARY_ALLOWED, reason="backbone_stage_boundary")

    payload = graph.to_dict()

    assert payload["nodes"][0]["node_id"] == "backbone.stage1"
    assert payload["edges"][0]["edge_type"] == CAST_BOUNDARY_ALLOWED
