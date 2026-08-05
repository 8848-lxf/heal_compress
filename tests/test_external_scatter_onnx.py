from pathlib import Path

import pytest

onnx = pytest.importorskip("onnx")
from onnx import TensorProto, helper

from carla_integration.external_scatter_onnx import externalize_scatter


def _toy_model(path: Path) -> None:
    inputs = [
        helper.make_tensor_value_info("voxel_features", TensorProto.FLOAT, [7, 2]),
        helper.make_tensor_value_info("voxel_coords", TensorProto.INT32, [7, 4]),
        helper.make_tensor_value_info("valid_voxel_mask", TensorProto.BOOL, [7]),
        helper.make_tensor_value_info(
            "pairwise_t_matrix", TensorProto.FLOAT, [1, "N", "N", 4, 4]
        ),
    ]
    output = helper.make_tensor_value_info("prediction", TensorProto.FLOAT, ["N", 1])
    nodes = [
        helper.make_node("Relu", ["voxel_features"], ["pillars"]),
        helper.make_node(
            "PointPillarScatterTRT",
            ["pillars", "voxel_coords", "valid_voxel_mask", "pairwise_t_matrix"],
            ["dense_bev"],
            domain="trt",
        ),
        helper.make_node("ReduceMean", ["dense_bev"], ["bev_mean"], axes=[1, 2, 3]),
        helper.make_node("ReduceMean", ["pairwise_t_matrix"], ["pairwise_mean"]),
        helper.make_node("Add", ["bev_mean", "pairwise_mean"], ["prediction"]),
    ]
    graph = helper.make_graph(nodes, "toy", inputs, [output])
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 11), helper.make_opsetid("trt", 1)],
    )
    onnx.save(model, str(path))


def test_externalize_scatter_removes_fixed_k_frontend(tmp_path: Path):
    source = tmp_path / "source.onnx"
    destination = tmp_path / "external.onnx"
    _toy_model(source)

    report = externalize_scatter(source, destination, channels=8, height=4, width=6)
    model = onnx.load(str(destination))

    assert [value.name for value in model.graph.input] == [
        "spatial_features",
        "pairwise_t_matrix",
    ]
    assert all(node.op_type != "PointPillarScatterTRT" for node in model.graph.node)
    assert all(node.op_type != "Relu" for node in model.graph.node)
    assert report["scatter_nodes_after_rewrite"] == 0
    dims = model.graph.input[0].type.tensor_type.shape.dim
    assert dims[0].dim_param == "num_agents"
    assert [value.dim_value for value in dims[1:]] == [8, 4, 6]
