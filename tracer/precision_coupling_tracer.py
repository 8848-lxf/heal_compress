"""Precision coupling tracing for mixed-precision TensorRT/QDQ workflows.

The legacy entry point in this module consumes an FX-style dictionary.  The
runtime entry point consumes the same typed :class:`TraceResult` used by the
channel-dependency tracer, so dynamic HEAL forwards do not silently degrade to
an isolated module list when symbolic tracing fails.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

import torch.nn as nn

from .hashing import stable_id


PRECISIONS = ("fp32", "fp16", "int8")
UNSUPPORTED_INT8_KEYWORDS = ("scatter", "bev_pool", "bevpool", "warp", "plugin", "voxel", "pillar_vfe")
HEAD_KEYWORDS = ("head", "cls", "reg", "dir", "obj")


@dataclass
class PrecisionGroup:
    precision_group_id: str
    member_modules: list[str]
    reason: str
    allowed_precisions: list[str]
    default_precision: str
    force_same_precision: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PrecisionRelation:
    """One runtime-observed relationship between weighted compute branches."""

    relation_id: str
    operation_id: str
    relation_kind: str
    member_modules: list[str]
    input_branch_modules: list[list[str]]
    force_same_precision: bool
    merge_precision: str
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RuntimePrecisionCouplingResult:
    """Complete precision genes and typed relations derived from a runtime graph."""

    groups: list[PrecisionGroup]
    relations: list[PrecisionRelation]
    weighted_modules: list[str]
    weighted_module_call_counts: dict[str, int]
    relationship_source: str = "runtime_tensor_flow"
    schema_version: str = "runtime-precision-coupling-v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "relationship_source": self.relationship_source,
            "weighted_modules": list(self.weighted_modules),
            "weighted_module_call_counts": dict(sorted(self.weighted_module_call_counts.items())),
            "groups": [group.to_dict() for group in self.groups],
            "relations": [relation.to_dict() for relation in self.relations],
        }


def _runtime_relation_kind(op_type: str) -> str:
    value = str(op_type).lower()
    if "cat" in value:
        return "concat"
    if "add" in value:
        return "residual_add"
    if "stack" in value:
        return "stack"
    if "mul" in value:
        return "elementwise_multiply"
    if "where" in value:
        return "conditional_select"
    if "matmul" in value or "bmm" in value:
        return "matrix_merge"
    return "runtime_multi_input"


def _runtime_merge_has_floating_output(operation: Any) -> bool:
    """Reject shape/index joins before they become precision relations.

    Runtime provenance can legitimately flow from a weighted activation into
    shape construction (for example ``torch.stack`` over values obtained from
    ``Tensor.shape``).  Such an integer stack is not an activation merge and
    therefore has no Q/DQ precision contract.  New runtime traces carry dtype
    metadata; traces created before that metadata existed retain the previous
    conservative behaviour and are resolved by the ONNX fail-closed pass.
    """

    metadata = dict(getattr(operation, "metadata", {}) or {})
    dtypes = [
        str(value).lower().removeprefix("torch.")
        for value in metadata.get("output_dtypes", []) or []
    ]
    if not dtypes:
        return True
    floating = {
        "float16",
        "float32",
        "float64",
        "bfloat16",
        "half",
        "float",
        "double",
    }
    return all(value in floating for value in dtypes)


def _runtime_weighted_producers(
    operation_id: str,
    *,
    operations: dict[str, Any],
    weighted: set[str],
    memo: dict[str, set[str]],
    visiting: set[str] | None = None,
) -> set[str]:
    """Return nearest upstream weighted modules without crossing weighted compute."""

    if operation_id in memo:
        return set(memo[operation_id])
    visiting = set(visiting or ())
    if operation_id in visiting:
        return set()
    visiting.add(operation_id)
    operation = operations.get(operation_id)
    if operation is None:
        memo[operation_id] = set()
        return set()
    module_path = str(getattr(operation, "module_path", "") or "")
    if module_path in weighted:
        memo[operation_id] = {module_path}
        return {module_path}
    result: set[str] = set()
    for parent in getattr(operation, "input_ids", []) or []:
        result.update(
            _runtime_weighted_producers(
                str(parent),
                operations=operations,
                weighted=weighted,
                memo=memo,
                visiting=visiting,
            )
        )
    memo[operation_id] = set(result)
    return result


def _shared_weight_components(model: nn.Module, weighted_modules: list[str]) -> list[list[str]]:
    """Group distinct module paths that own the same physical weight tensor."""

    modules = dict(model.named_modules())
    parent = {name: name for name in weighted_modules}

    def find(name: str) -> str:
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return
        first, second = sorted((left_root, right_root))
        parent[second] = first

    owners: dict[int, list[str]] = defaultdict(list)
    for name in weighted_modules:
        weight = getattr(modules.get(name), "weight", None)
        if weight is not None:
            owners[id(weight)].append(name)
    for names in owners.values():
        for name in names[1:]:
            union(names[0], name)
    components: dict[str, list[str]] = defaultdict(list)
    for name in weighted_modules:
        components[find(name)].append(name)
    return sorted((sorted(names) for names in components.values()), key=lambda names: tuple(names))


def build_runtime_precision_coupling(
    model: nn.Module,
    trace_result: Any,
) -> RuntimePrecisionCouplingResult:
    """Generate precision genes and relations from a typed runtime tensor-flow graph.

    Every runtime-covered weighted module appears in exactly one gene.  Multiple
    call sites of one physical module and modules sharing one physical weight
    tensor are strong same-precision constraints.  Functional Add/Concat/etc.
    relationships are retained as merge contracts, but do not collapse genes.
    Merge precision is a candidate-level derived value: equal branch output
    precisions are preserved and mixed branches are promoted according to
    ``INT8 < FP16 < FP32``.
    """
    inventory = list(getattr(trace_result, "module_inventory", []) or [])
    weighted_modules = sorted(
        str(row.module_path) for row in inventory if bool(getattr(row, "weighted", False))
    )
    if not weighted_modules:
        raise RuntimeError("runtime_precision_trace_contains_no_weighted_modules")
    weighted_set = set(weighted_modules)
    call_counts: dict[str, int] = {name: 0 for name in weighted_modules}
    for record in getattr(trace_result, "module_call_trace", []) or []:
        path = str(getattr(record, "module_path", ""))
        if path in call_counts:
            call_counts[path] += 1
    uncalled = sorted(name for name, count in call_counts.items() if count <= 0)
    if uncalled:
        raise RuntimeError(f"runtime_precision_weighted_modules_uncalled:{uncalled}")

    operations = {
        str(operation.op_id): operation
        for operation in getattr(trace_result, "op_inventory", []) or []
    }
    if not operations:
        raise RuntimeError("runtime_precision_trace_contains_no_operations")
    memo: dict[str, set[str]] = {}
    relations: list[PrecisionRelation] = []
    for operation_id in sorted(operations):
        operation = operations[operation_id]
        if str(getattr(operation, "op_kind", "")) != "call_function":
            continue
        input_ids = [str(value) for value in getattr(operation, "input_ids", []) or []]
        metadata = dict(getattr(operation, "metadata", {}) or {})
        observed_input_count = max(len(input_ids), int(metadata.get("num_inputs", 0) or 0))
        if observed_input_count < 2:
            continue
        branches = [
            sorted(
                _runtime_weighted_producers(
                    parent,
                    operations=operations,
                    weighted=weighted_set,
                    memo=memo,
                )
            )
            for parent in input_ids
        ]
        members = sorted({name for branch in branches for name in branch})
        if len(members) < 2:
            continue
        relation_kind = _runtime_relation_kind(str(getattr(operation, "op_type", "")))
        if relation_kind == "runtime_multi_input":
            continue
        if not _runtime_merge_has_floating_output(operation):
            continue
        relation_payload = {
            "operation_id": operation_id,
            "relation_kind": relation_kind,
            "members": members,
            "branches": branches,
        }
        relations.append(
            PrecisionRelation(
                relation_id=stable_id("precision-relation", relation_payload),
                operation_id=operation_id,
                relation_kind=relation_kind,
                member_modules=members,
                input_branch_modules=branches,
                force_same_precision=False,
                merge_precision="derived_per_candidate",
                reason="adaptive_common_precision_at_runtime_merge",
                metadata={
                    "runtime_op_type": str(getattr(operation, "op_type", "")),
                    "tracked_input_count": len(input_ids),
                    "observed_input_count": observed_input_count,
                    "module_scope": list(metadata.get("module_scope", []) or []),
                    "branch_compute_precision_independent": True,
                    "input_qdq_placement": "weighted input and weight",
                    "output_qdq_placement": "derived from candidate merge precision",
                    "downstream_requantization": "owned by the next weighted input",
                },
            )
        )

    relations_by_module: dict[str, list[PrecisionRelation]] = defaultdict(list)
    for relation in relations:
        for name in relation.member_modules:
            relations_by_module[name].append(relation)

    groups: list[PrecisionGroup] = []
    covered: set[str] = set()
    for component in _shared_weight_components(model, weighted_modules):
        component_relations = sorted(
            {
                relation.relation_id: relation
                for name in component
                for relation in relations_by_module.get(name, [])
            }.values(),
            key=lambda relation: relation.relation_id,
        )
        group_id = stable_id("runtime-precision-group", {"members": component})
        strong_reason = (
            "shared_physical_weight" if len(component) > 1 else "same_physical_module_all_call_sites"
        )
        group = PrecisionGroup(
            precision_group_id=group_id,
            member_modules=list(component),
            reason=strong_reason,
            allowed_precisions=list(PRECISIONS),
            default_precision="fp16",
            force_same_precision=True,
        )
        setattr(
            group,
            "deployment_contract",
            {
                "schema_version": "runtime-precision-deployment-contract-v1",
                "relationship_source": "runtime_tensor_flow",
                "member_layers": list(component),
                "member_call_counts": {name: call_counts[name] for name in component},
                "strong_coupling_reason": strong_reason,
                "merge_boundaries": [
                    {
                        "relation_id": relation.relation_id,
                        "operation_id": relation.operation_id,
                        "merge_kind": relation.relation_kind,
                        "member_layers": list(relation.member_modules),
                        "input_branch_layers": [list(branch) for branch in relation.input_branch_modules],
                        "branch_compute_precision_independent": True,
                        "force_same_precision": False,
                        "merge_policy": "adaptive_upcast_merge",
                        "input_qdq_placement": relation.metadata["input_qdq_placement"],
                        "output_qdq_placement": relation.metadata["output_qdq_placement"],
                        "merge_scale_policy": "equal_precision_kept_else_promote_INT8_to_FP16_to_FP32",
                    }
                    for relation in component_relations
                ],
                "merge_boundary_resolution": "resolved_from_runtime_tensor_flow",
                "merge_precision_resolution": "derived_per_candidate_from_realized_branch_outputs",
                "compute_precision_policy": "owned_by_quantization_gene",
                "output_precision_policy": (
                    "DERIVED_ADAPTIVE_MERGE" if component_relations else "same_as_compute"
                ),
                "insert_activation_output_qdq": "derived_per_candidate",
                "weight_granularity": "per_channel",
                "weight_axis_policy": {
                    "Conv": 0,
                    "ConvTranspose": 1,
                    "MatMul": 1,
                    "Gemm": "0 if transB else 1",
                },
            },
        )
        groups.append(group)
        covered.update(component)

    missing = sorted(weighted_set - covered)
    duplicated = sorted(
        name
        for name in weighted_modules
        if sum(name in group.member_modules for group in groups) != 1
    )
    if missing or duplicated:
        raise RuntimeError(
            f"runtime_precision_group_coverage_invalid:missing={missing}:duplicated={duplicated}"
        )
    return RuntimePrecisionCouplingResult(
        groups=groups,
        relations=relations,
        weighted_modules=weighted_modules,
        weighted_module_call_counts=call_counts,
    )


def _node_by_name(dependency_graph: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(node.get("name") or node.get("node_id")): node for node in dependency_graph.get("nodes", [])}


def _producer_modules(name: str, nodes: dict[str, dict[str, Any]], seen: set[str] | None = None) -> list[str]:
    seen = seen or set()
    if name in seen:
        return []
    seen.add(name)
    node = nodes.get(name)
    if not node:
        return []
    module = str(node.get("module_name") or "")
    if module:
        return [module]
    out: list[str] = []
    for parent in node.get("inputs", []) or []:
        out.extend(_producer_modules(str(parent), nodes, seen))
    return out


def _unique(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _module_supported_precisions(name: str, module: nn.Module, *, allow_head_int8: bool) -> tuple[list[str], str, str]:
    low = name.lower()
    if any(key in low for key in UNSUPPORTED_INT8_KEYWORDS):
        return ["fp32", "fp16"], "fp16", "unsupported_int8"
    if any(key in low for key in HEAD_KEYWORDS) and not allow_head_int8:
        return ["fp32", "fp16"], "fp16", "head_constraint"
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        return list(PRECISIONS), "fp16", "user_constraint"
    return ["fp32", "fp16"], "fp16", "unsupported_int8"


def build_precision_coupling_groups(
    model: nn.Module,
    dependency_graph: dict[str, Any],
    sample_batch: Any | None = None,
    *,
    allow_head_int8: bool = False,
) -> list[PrecisionGroup]:
    nodes = _node_by_name(dependency_graph)
    groups: list[PrecisionGroup] = []
    covered: set[str] = set()
    idx = 0
    for node in dependency_graph.get("nodes", []) or []:
        op_type = str(node.get("op_type", "")).lower()
        if "add" not in op_type and "concat" not in op_type:
            continue
        members = _unique(
            module
            for input_name in node.get("inputs", []) or []
            for module in _producer_modules(str(input_name), nodes)
        )
        if len(members) < 2:
            continue
        reason = "concat" if "concat" in op_type else "residual"
        group = PrecisionGroup(
            precision_group_id=f"pg_{idx:04d}",
            member_modules=members,
            reason=reason,
            allowed_precisions=list(PRECISIONS),
            default_precision="fp16",
            force_same_precision=False,
        )
        groups.append(group)
        covered.update(members)
        idx += 1

    for name, module in model.named_modules():
        if not name or name in covered:
            continue
        allowed, default, reason = _module_supported_precisions(name, module, allow_head_int8=allow_head_int8)
        if not isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear, nn.BatchNorm2d, nn.ReLU, nn.Identity)):
            continue
        groups.append(
            PrecisionGroup(
                precision_group_id=f"pg_{idx:04d}",
                member_modules=[name],
                reason=reason,
                allowed_precisions=allowed,
                default_precision=default,
                force_same_precision=True,
            )
        )
        idx += 1
    return groups


def precision_groups_to_json(groups: list[PrecisionGroup]) -> list[dict[str, Any]]:
    return [group.to_dict() for group in groups]
