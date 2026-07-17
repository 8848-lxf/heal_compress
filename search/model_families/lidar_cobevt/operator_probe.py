"""ONNX and TensorRT capability probes for the CoBEVT family recipe."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import onnx

REGISTERED_CUSTOM_OPS = {"trt::PointPillarScatterTRT"}
FORBIDDEN_WEAK_FLAGS = (
    "--fp16",
    "--int8",
    "--precisionConstraints",
    "--layerPrecisions",
    "--layerOutputTypes",
)


@dataclass(frozen=True)
class OperatorCapabilityReport:
    onnx_path: str
    node_count: int
    op_counts: dict[str, int]
    registered_custom_ops: tuple[str, ...]
    unregistered_custom_ops: tuple[str, ...]
    weak_precision_fallback_requested: bool
    passed: bool


def audit_onnx_operators(path: str | Path) -> OperatorCapabilityReport:
    source = Path(path)
    model = onnx.load(str(source))
    counts: Counter[str] = Counter()
    registered: set[str] = set()
    unregistered: set[str] = set()
    for node in model.graph.node:
        domain = str(node.domain)
        qualified = f"{domain}::{node.op_type}" if domain else str(node.op_type)
        counts[qualified] += 1
        if domain and domain not in {"ai.onnx"}:
            if qualified in REGISTERED_CUSTOM_OPS:
                registered.add(qualified)
            else:
                unregistered.add(qualified)
        elif not domain and node.op_type == "PointPillarScatterTRT":
            registered.add("trt::PointPillarScatterTRT")
    weak_fallback = any(
        str(prop.key).startswith("weak_precision")
        and str(prop.value).strip().lower() in {"1", "true", "yes"}
        for prop in model.metadata_props
    )
    return OperatorCapabilityReport(
        onnx_path=str(source),
        node_count=len(model.graph.node),
        op_counts=dict(sorted(counts.items())),
        registered_custom_ops=tuple(sorted(registered)),
        unregistered_custom_ops=tuple(sorted(unregistered)),
        weak_precision_fallback_requested=weak_fallback,
        passed=not unregistered and not weak_fallback,
    )


def strongly_typed_probe_command(
    *,
    trtexec_path: Path,
    onnx_path: Path,
    engine_path: Path,
    layer_info_path: Path,
    plugin_path: Path,
    shapes: Mapping[str, str],
) -> list[str]:
    shape_spec = ",".join(f"{name}:{value}" for name, value in sorted(shapes.items()))
    command = [
        str(trtexec_path),
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--exportLayerInfo={layer_info_path}",
        "--profilingVerbosity=detailed",
        "--memPoolSize=workspace:512",
        "--skipInference",
        "--noTF32",
        "--stronglyTyped",
        f"--staticPlugins={plugin_path}",
    ]
    if shape_spec:
        command.extend(
            (
                f"--minShapes={shape_spec}",
                f"--optShapes={shape_spec}",
                f"--maxShapes={shape_spec}",
            )
        )
    if any(token.startswith(FORBIDDEN_WEAK_FLAGS) for token in command):
        raise RuntimeError("weak_precision_flag_in_strongly_typed_probe")
    return command


def make_scatter_parser_compatible(
    source: str | Path, destination: str | Path
) -> dict[str, Any]:
    input_path = Path(source)
    output_path = Path(destination)
    model = onnx.load(str(input_path))
    changed: list[str] = []
    for node in model.graph.node:
        if node.op_type == "PointPillarScatterTRT" and node.domain == "trt":
            changed.append(str(node.name))
            node.domain = ""
    keep = [opset for opset in model.opset_import if opset.domain != "trt"]
    del model.opset_import[:]
    model.opset_import.extend(keep)
    onnx.save(model, str(output_path))
    return {
        "source": str(input_path),
        "destination": str(output_path),
        "changed_nodes": changed,
    }
