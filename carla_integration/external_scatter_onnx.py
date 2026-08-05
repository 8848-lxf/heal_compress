"""Move the PointPillar frontend outside an exported HEAL ONNX graph."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set


SCATTER_OP = "PointPillarScatterTRT"
SPATIAL_FEATURES = "spatial_features"
PAIRWISE = "pairwise_t_matrix"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _required_node_indices(
    nodes: Sequence[Any], outputs: Iterable[str], boundaries: Set[str]
) -> Set[int]:
    producer: Dict[str, int] = {}
    for index, node in enumerate(nodes):
        for name in node.output:
            if name:
                producer[name] = index

    required: Set[int] = set()
    pending = list(outputs)
    while pending:
        value = pending.pop()
        if not value or value in boundaries:
            continue
        index = producer.get(value)
        if index is None or index in required:
            continue
        required.add(index)
        pending.extend(name for name in nodes[index].input if name)
    return required


def _rename_node_inputs(nodes: Iterable[Any], old: str, new: str) -> None:
    for node in nodes:
        for index, name in enumerate(node.input):
            if name == old:
                node.input[index] = new


def externalize_scatter(
    source: Path,
    destination: Path,
    *,
    channels: int = 64,
    height: int = 256,
    width: int = 512,
) -> Mapping[str, Any]:
    """Replace the embedded scatter/PFN frontend with a dense BEV input."""
    import onnx
    from onnx import TensorProto, helper

    source = source.expanduser().resolve()
    destination = destination.expanduser().resolve()
    model = onnx.load(str(source), load_external_data=True)
    scatter_nodes = [node for node in model.graph.node if node.op_type == SCATTER_OP]
    if len(scatter_nodes) != 1:
        raise ValueError(
            f"expected exactly one {SCATTER_OP} node, found {len(scatter_nodes)}"
        )
    scatter_output = str(scatter_nodes[0].output[0])
    graph_outputs = [str(value.name) for value in model.graph.output]
    required = _required_node_indices(
        model.graph.node,
        graph_outputs,
        boundaries={scatter_output, PAIRWISE},
    )
    kept_nodes = [
        node for index, node in enumerate(model.graph.node) if index in required
    ]
    if any(node.op_type == SCATTER_OP for node in kept_nodes):
        raise RuntimeError("scatter node remained reachable after boundary rewrite")
    _rename_node_inputs(kept_nodes, scatter_output, SPATIAL_FEATURES)

    pairwise_inputs = [value for value in model.graph.input if value.name == PAIRWISE]
    if len(pairwise_inputs) != 1:
        raise ValueError(f"expected exactly one {PAIRWISE} graph input")
    spatial_input = helper.make_tensor_value_info(
        SPATIAL_FEATURES,
        TensorProto.FLOAT,
        ["num_agents", int(channels), int(height), int(width)],
    )

    referenced = {
        name for node in kept_nodes for name in node.input if name
    }
    initializers = [
        initializer
        for initializer in model.graph.initializer
        if initializer.name in referenced
    ]
    initializer_names = {initializer.name for initializer in initializers}
    unresolved = sorted(
        referenced
        - initializer_names
        - {SPATIAL_FEATURES, PAIRWISE}
        - {name for node in kept_nodes for name in node.output if name}
    )
    if unresolved:
        raise RuntimeError(f"rewritten graph has unresolved values: {unresolved[:10]}")

    retained_names = referenced | {
        name for node in kept_nodes for name in node.output if name
    }
    value_info = [
        value
        for value in model.graph.value_info
        if value.name in retained_names and value.name != scatter_output
    ]
    graph = helper.make_graph(
        kept_nodes,
        model.graph.name + "_external_scatter",
        [spatial_input, pairwise_inputs[0]],
        list(model.graph.output),
        initializer=initializers,
        value_info=value_info,
    )
    rewritten = helper.make_model(
        graph,
        opset_imports=list(model.opset_import),
        producer_name="heal-carla-external-scatter",
    )
    rewritten.ir_version = model.ir_version
    rewritten.model_version = model.model_version
    rewritten.domain = model.domain
    rewritten.doc_string = model.doc_string
    for item in model.metadata_props:
        entry = rewritten.metadata_props.add()
        entry.key = item.key
        entry.value = item.value
    metadata = rewritten.metadata_props.add()
    metadata.key = "heal.point_frontend"
    metadata.value = "dynamic_voxel_pfn_scatter_outside_tensorrt"
    onnx.checker.check_model(rewritten)

    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(rewritten, str(destination))
    report = {
        "schema_version": "heal-external-scatter-onnx-v1",
        "source": str(source),
        "source_sha256": _sha256(source),
        "output": str(destination),
        "output_sha256": _sha256(destination),
        "removed_node_count": len(model.graph.node) - len(kept_nodes),
        "retained_node_count": len(kept_nodes),
        "inputs": {
            SPATIAL_FEATURES: ["num_agents", channels, height, width],
            PAIRWISE: [1, "num_agents", "num_agents", 4, 4],
        },
        "outputs": graph_outputs,
        "removed_fixed_k_inputs": sorted(
            value.name for value in model.graph.input if value.name != PAIRWISE
        ),
        "scatter_nodes_after_rewrite": sum(
            node.op_type == SCATTER_OP for node in kept_nodes
        ),
    }
    report_path = destination.with_suffix(destination.suffix + ".json")
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=512)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    report = externalize_scatter(
        arguments.source,
        arguments.destination,
        channels=arguments.channels,
        height=arguments.height,
        width=arguments.width,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
