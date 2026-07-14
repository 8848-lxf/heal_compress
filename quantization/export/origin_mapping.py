"""PyTorch call to weighted ONNX node origin mapping."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..artifacts.io import atomic_write_json, file_sha256
from ..config import CanonicalNamingConfig
from ..exceptions import AmbiguousCanonicalMappingError, CanonicalMappingError
from ..types import CanonicalFunctionalComputeGroup, CanonicalMappingEntry, OnnxOriginMapResult
from .canonical_naming import canonical_node_name
from .origin_trace import build_weight_trace_index, onnx_attribute, trace_compute_node_weight


_WEIGHTED_REQUIRED_OPS = frozenset({"Conv", "ConvTranspose", "Gemm"})
_WEIGHTED_OPS = frozenset({"Conv", "ConvTranspose", "Gemm", "MatMul"})


def _downstream_contains_op(
    node: Any,
    consumers: Mapping[str, Sequence[Any]],
    op_type: str,
    *,
    max_depth: int = 4,
) -> bool:
    queue = [(str(output), 0) for output in node.output]
    seen: set[str] = set()
    while queue:
        tensor, depth = queue.pop(0)
        if tensor in seen or depth > max_depth:
            continue
        seen.add(tensor)
        for consumer in consumers.get(tensor, ()):
            if str(consumer.op_type) == op_type:
                return True
            queue.extend((str(output), depth + 1) for output in consumer.output)
    return False


def _functional_compute_groups(index: Mapping[str, Any], rows: Sequence[tuple[int, Any]]) -> list[CanonicalFunctionalComputeGroup]:
    """Canonicalize parameter-free functional MatMul families.

    The HEAL signal-maxK wrapper emits one affine-grid BMM per pyramid level.
    TensorRT fuses the three nodes into one GEMM row, so they intentionally
    share one protected compute identity instead of becoming three fake
    weighted precision genes.
    """

    consumers: dict[str, list[Any]] = defaultdict(list)
    for node in index["nodes"]:
        for input_name in node.input:
            consumers[str(input_name)].append(node)
    affine_grid = [
        (graph_index, node)
        for graph_index, node in rows
        if _downstream_contains_op(node, consumers, "GridSample")
    ]
    remaining = [(graph_index, node) for graph_index, node in rows if (graph_index, node) not in affine_grid]
    result: list[CanonicalFunctionalComputeGroup] = []
    if affine_grid:
        result.append(
            CanonicalFunctionalComputeGroup(
                module_path="pyramid_backbone.functional_affine_grid_matmul",
                module_type="functional_bmm",
                canonical_node_name=canonical_node_name(
                    "pyramid_backbone.functional_affine_grid_matmul",
                    "MatMulGroup",
                    0,
                ),
                original_node_names=tuple(str(node.name) for _, node in affine_grid),
                graph_indices=tuple(int(graph_index) for graph_index, _ in affine_grid),
                input_tensors=tuple(tuple(str(value) for value in node.input) for _, node in affine_grid),
                output_tensors=tuple(tuple(str(value) for value in node.output) for _, node in affine_grid),
                source_call="quantization.export.heal_lidar_pyramid._warp:torch.bmm",
                protection_reason="parameter_free_affine_grid_matmul_legacy_realized_fp16",
            )
        )
    for ordinal, (graph_index, node) in enumerate(remaining, start=len(result)):
        result.append(
            CanonicalFunctionalComputeGroup(
                module_path=f"functional_matmul.graph_{graph_index}",
                module_type="functional_matmul",
                canonical_node_name=canonical_node_name(f"functional_matmul.graph_{graph_index}", "MatMul", ordinal),
                original_node_names=(str(node.name),),
                graph_indices=(int(graph_index),),
                input_tensors=(tuple(str(value) for value in node.input),),
                output_tensors=(tuple(str(value) for value in node.output),),
                source_call="unresolved_parameter_free_functional_matmul",
                protection_reason="parameter_free_functional_matmul_requires_explicit_source_audit",
            )
        )
    return result


def _expected_ops(call: Mapping[str, Any]) -> set[str]:
    explicit = str(call.get("mapped_onnx_op_type") or call.get("onnx_op_type") or "")
    if explicit == "MatMul":
        return {"MatMul", "Gemm"}
    if explicit:
        return {explicit}
    module_type = str(call.get("module_type", ""))
    if "ConvTranspose" in module_type:
        return {"ConvTranspose"}
    if "Conv" in module_type:
        return {"Conv"}
    if "Linear" in module_type:
        return {"Gemm", "MatMul"}
    return set(_WEIGHTED_OPS)


def _call_value(call: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in call and call[name] not in (None, ""):
            return call[name]
    return default


def _node_row(index: Mapping[str, Any], node: Any, graph_index: int) -> dict[str, Any]:
    trace = trace_compute_node_weight(index, node)
    shape = tuple(int(value) for value in trace.get("root_initializer_shape", ()))
    groups = int(onnx_attribute(node, "group", 1) or 1)
    input_per_group = shape[1] if str(node.op_type) == "Conv" and len(shape) >= 2 else None
    output_per_group = shape[1] if str(node.op_type) == "ConvTranspose" and len(shape) >= 2 else None
    if str(node.op_type) == "Conv" and len(shape) >= 2:
        output_per_group = shape[0] // groups if shape[0] % groups == 0 else None
    if str(node.op_type) == "ConvTranspose" and len(shape) >= 2:
        input_per_group = shape[0] // groups if shape[0] % groups == 0 else None
    return {
        "node": node,
        "graph_index": int(graph_index),
        "op_type": str(node.op_type),
        "original_name": str(node.name),
        "root_initializer": str(trace.get("root_initializer", "")),
        "weight_shape": shape,
        "groups": groups,
        "input_channels_per_group": input_per_group,
        "output_channels_per_group": output_per_group,
        "channels_per_group": min(value for value in (input_per_group, output_per_group) if value is not None)
        if input_per_group is not None or output_per_group is not None
        else None,
        "root_trace": trace,
    }


def _call_matches_node(call: Mapping[str, Any], row: Mapping[str, Any]) -> bool:
    if str(row["op_type"]) not in _expected_ops(call):
        return False
    initializer = str(_call_value(call, "weight_initializer", "onnx_weight_initializer", default="") or "")
    if initializer and initializer != str(row["root_initializer"]):
        return False
    groups = _call_value(call, "groups")
    if groups not in (None, "") and int(groups) != int(row["groups"]):
        return False
    call_shape = tuple(int(value) for value in (_call_value(call, "weight_shape", default=()) or ()))
    node_shape = tuple(row["weight_shape"])
    if call_shape and node_shape:
        linear = bool(_expected_ops(call) & {"Gemm", "MatMul"})
        if node_shape != call_shape and not (linear and node_shape == tuple(reversed(call_shape))):
            return False
    return True


def build_onnx_origin_map(
    onnx_path: str | Path,
    module_calls: Sequence[Mapping[str, Any] | Any],
    *,
    naming_config: CanonicalNamingConfig | None = None,
    report_path: str | Path | None = None,
) -> OnnxOriginMapResult:
    """Build a deterministic, repeated-call-aware weighted origin map.

    Functional MatMul nodes are recorded but never represented as weighted
    modules. Any unresolved Conv/ConvTranspose/Gemm, unmatched weighted node,
    or non-order-resolvable ambiguity raises ``CanonicalMappingError``.
    """

    index = build_weight_trace_index(onnx_path)
    calls = [dict(row) if isinstance(row, Mapping) else dict(vars(row)) for row in module_calls]
    normalized_calls: list[dict[str, Any]] = []
    for order, call in enumerate(calls):
        module_path = str(_call_value(call, "module_path", "canonical_module_name", default="") or "")
        if not module_path:
            raise CanonicalMappingError(f"weighted module call {order} has no module path")
        call_index = int(_call_value(call, "call_index", "module_call_index", default=order))
        normalized_calls.append({**call, "module_path": module_path, "call_index": call_index, "_order": order})
    if len({int(row["call_index"]) for row in normalized_calls}) != len(normalized_calls):
        raise CanonicalMappingError("weighted module call indices are not unique")

    node_rows: list[dict[str, Any]] = []
    functional_matmuls: list[str] = []
    functional_matmul_rows: list[tuple[int, Any]] = []
    unresolved: list[dict[str, Any]] = []
    for graph_index, node in enumerate(index["nodes"]):
        if str(node.op_type) not in _WEIGHTED_OPS:
            continue
        row = _node_row(index, node, graph_index)
        if not row["root_trace"].get("success"):
            if str(node.op_type) == "MatMul":
                functional_matmuls.append(str(node.name))
                functional_matmul_rows.append((graph_index, node))
                continue
            unresolved.append(
                {"node_name": str(node.name), "op_type": str(node.op_type), "failure_reason": "root_initializer_unresolved"}
            )
            continue
        node_rows.append(row)
    if unresolved:
        raise CanonicalMappingError(f"unresolved weighted ONNX nodes: {unresolved}")

    candidates_by_call: dict[int, tuple[int, ...]] = {}
    for call_idx, call in enumerate(normalized_calls):
        candidates = tuple(index_ for index_, row in enumerate(node_rows) if _call_matches_node(call, row))
        if not candidates:
            raise CanonicalMappingError(f"no weighted ONNX node matches module call {call['module_path']}#{call['call_index']}")
        candidates_by_call[call_idx] = candidates
    groups: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for call_idx, candidates in candidates_by_call.items():
        groups[candidates].append(call_idx)
    candidate_sets = list(groups)
    for left_index, left in enumerate(candidate_sets):
        for right in candidate_sets[left_index + 1 :]:
            if set(left) & set(right):
                raise AmbiguousCanonicalMappingError("overlapping weighted-node candidate sets cannot be distinguished")

    assignment: dict[int, int] = {}
    for candidates, call_indices in groups.items():
        if len(candidates) != len(call_indices):
            raise AmbiguousCanonicalMappingError(
                f"module calls and indistinguishable ONNX nodes differ: calls={len(call_indices)}, nodes={len(candidates)}"
            )
        ordered_calls = sorted(call_indices, key=lambda idx: (int(normalized_calls[idx]["call_index"]), int(normalized_calls[idx]["_order"])))
        ordered_nodes = sorted(candidates, key=lambda idx: int(node_rows[idx]["graph_index"]))
        assignment.update(dict(zip(ordered_calls, ordered_nodes)))
    if len(assignment) != len(normalized_calls) or len(set(assignment.values())) != len(node_rows):
        raise CanonicalMappingError("weighted module calls and ONNX nodes are not one-to-one")

    policy = naming_config or CanonicalNamingConfig()
    entries: list[CanonicalMappingEntry] = []
    used_names: set[str] = set()
    for call_idx in sorted(assignment, key=lambda idx: int(normalized_calls[idx]["call_index"])):
        call = normalized_calls[call_idx]
        row = node_rows[assignment[call_idx]]
        name = canonical_node_name(call["module_path"], row["op_type"], call["call_index"], config=policy)
        if name in used_names:
            name = canonical_node_name(
                call["module_path"], row["op_type"], call["call_index"], config=policy, collision_salt=str(row["graph_index"])
            )
        if name in used_names:
            raise CanonicalMappingError(f"canonical node name collision: {name}")
        used_names.add(name)
        entries.append(
            CanonicalMappingEntry(
                module_path=str(call["module_path"]),
                module_type=str(call.get("module_type", "")),
                call_index=int(call["call_index"]),
                onnx_op_type=str(row["op_type"]),
                original_node_name=str(row["original_name"]),
                canonical_node_name=name,
                weight_initializer=str(row["root_initializer"]),
                graph_index=int(row["graph_index"]),
                groups=int(row["groups"]),
                channels_per_group=row["channels_per_group"],
                input_channels_per_group=row["input_channels_per_group"],
                output_channels_per_group=row["output_channels_per_group"],
                weight_shape=tuple(row["weight_shape"]),
                root_trace=tuple(row["root_trace"].get("trace_chain", [])),
            )
        )
    result = OnnxOriginMapResult(
        entries=entries,
        source_onnx=str(Path(onnx_path)),
        unresolved_weighted_nodes=unresolved,
        functional_matmul_nodes=functional_matmuls,
        functional_compute_groups=_functional_compute_groups(index, functional_matmul_rows),
        naming_policy_version=policy.policy_version,
    )
    if report_path is not None:
        atomic_write_json(report_path, result.to_dict())
    return result


def apply_canonical_node_names(
    input_path: str | Path,
    origin_map: OnnxOriginMapResult,
    *,
    output_path: str | Path | None = None,
    allow_custom_ops: bool = False,
) -> "CanonicalRenameResult":
    """Rename weighted nodes using graph-index-anchored origin entries."""

    import os
    import tempfile

    import onnx

    from ..types import CanonicalRenameResult

    source = Path(input_path)
    destination = Path(output_path) if output_path is not None else source
    model = onnx.load(str(source))
    by_index = {entry.graph_index: entry for entry in origin_map.entries}
    if len(by_index) != len(origin_map.entries):
        raise CanonicalMappingError("origin map contains duplicate graph indices")
    functional_by_index: dict[int, tuple[Any, int]] = {}
    for group in origin_map.functional_compute_groups:
        for member_index, graph_index in enumerate(group.graph_indices):
            if graph_index in by_index or graph_index in functional_by_index:
                raise CanonicalMappingError(f"duplicate functional graph index: {graph_index}")
            functional_by_index[int(graph_index)] = (group, member_index)
    renamed: list[dict[str, str]] = []
    seen_names: set[str] = set()
    for graph_index, node in enumerate(model.graph.node):
        entry = by_index.get(graph_index)
        functional = functional_by_index.get(graph_index)
        if entry is None and functional is None:
            continue
        if functional is not None:
            group, member_index = functional
            expected = group.original_node_names[member_index]
            if str(node.op_type) != group.onnx_op_type or str(node.name) != expected:
                raise CanonicalMappingError(
                    f"functional origin map no longer matches ONNX graph at index {graph_index}: {node.name}/{node.op_type}"
                )
            original = str(node.name)
            node.name = f"{group.canonical_node_name}__member{member_index:02d}"
            renamed.append({"original_node_name": original, "canonical_node_name": str(node.name)})
            continue
        if str(node.op_type) != entry.onnx_op_type or str(node.name) != entry.original_node_name:
            raise CanonicalMappingError(
                f"origin map no longer matches ONNX graph at index {graph_index}: {node.name}/{node.op_type}"
            )
        if entry.canonical_node_name in seen_names:
            raise CanonicalMappingError(f"duplicate canonical node name: {entry.canonical_node_name}")
        seen_names.add(entry.canonical_node_name)
        original = str(node.name)
        node.name = entry.canonical_node_name
        renamed.append({"original_node_name": original, "canonical_node_name": entry.canonical_node_name})
    expected_renamed = len(origin_map.entries) + sum(len(group.graph_indices) for group in origin_map.functional_compute_groups)
    if len(renamed) != expected_renamed:
        raise CanonicalMappingError("not every origin-map entry was renamed")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".onnx", dir=destination.parent)
    os.close(descriptor)
    try:
        onnx.save(model, temporary)
        try:
            onnx.checker.check_model(onnx.load(temporary))
        except Exception:
            if not allow_custom_ops:
                raise
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return CanonicalRenameResult(
        input_onnx=str(source),
        output_onnx=str(destination),
        renamed_node_count=len(renamed),
        renamed_nodes=renamed,
        naming_policy_version=origin_map.naming_policy_version,
        output_sha256=file_sha256(destination),
    )
