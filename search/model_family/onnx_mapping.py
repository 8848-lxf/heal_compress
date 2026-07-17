"""Model-family ONNX mapping layered on the verified module origin mapper.

The generic quantization origin mapper already handles repeated Conv/Linear
module calls.  V2X-ViT additionally owns functional HGT relation parameters
consumed through Gather -> Einsum paths and inactive type-specialized module
branches.  This module accounts for those cases without changing the accepted
LiDAR-pyramid mapper.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from quantization.export.origin_mapping import build_onnx_origin_map

from .contracts import ModelFamilyAudit


_FUNCTIONAL_WEIGHT_PASSTHROUGH = frozenset(
    {
        "Cast",
        "Concat",
        "Equal",
        "Expand",
        "Gather",
        "Identity",
        "Reshape",
        "Slice",
        "Squeeze",
        "Transpose",
        "Unsqueeze",
        "Where",
    }
)


@dataclass(frozen=True)
class ModelFamilyOnnxWeightedEntry:
    canonical_id: str
    module_path: str
    source_kind: str
    mapping_status: str
    active: bool
    onnx_op_types: tuple[str, ...] = ()
    onnx_node_names: tuple[str, ...] = ()
    graph_indices: tuple[int, ...] = ()
    weight_initializers: tuple[str, ...] = ()
    call_count: int = 0
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_id": self.canonical_id,
            "module_path": self.module_path,
            "source_kind": self.source_kind,
            "mapping_status": self.mapping_status,
            "active": self.active,
            "onnx_op_types": list(self.onnx_op_types),
            "onnx_node_names": list(self.onnx_node_names),
            "graph_indices": list(self.graph_indices),
            "weight_initializers": list(self.weight_initializers),
            "call_count": self.call_count,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ModelFamilyOnnxMapping:
    schema_version: str
    family_id: str
    source_onnx: str
    audit_hash: str
    weighted_entries: tuple[ModelFamilyOnnxWeightedEntry, ...]
    parameter_free_compute_groups: tuple[dict[str, Any], ...]
    activation_only_einsum_nodes: tuple[str, ...]
    unresolved: tuple[dict[str, Any], ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "family_id": self.family_id,
            "source_onnx": self.source_onnx,
            "audit_hash": self.audit_hash,
            "weighted_entries": [row.to_dict() for row in self.weighted_entries],
            "parameter_free_compute_groups": [dict(row) for row in self.parameter_free_compute_groups],
            "activation_only_einsum_nodes": list(self.activation_only_einsum_nodes),
            "unresolved": [dict(row) for row in self.unresolved],
            "metadata": dict(self.metadata),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        payload["mapping_hash"] = hashlib.sha256(encoded).hexdigest()
        return payload


def _upstream_initializers(
    tensor: str,
    *,
    initializers: set[str],
    producers: Mapping[str, Any],
    seen: set[str] | None = None,
) -> set[str]:
    visited = set() if seen is None else seen
    name = str(tensor)
    if name in visited:
        return set()
    visited.add(name)
    if name in initializers:
        return {name}
    producer = producers.get(name)
    if producer is None or str(producer.op_type) not in _FUNCTIONAL_WEIGHT_PASSTHROUGH:
        return set()
    roots: set[str] = set()
    for input_name in producer.input:
        roots.update(
            _upstream_initializers(
                str(input_name),
                initializers=initializers,
                producers=producers,
                seen=visited,
            )
        )
    return roots


def _inactive_reason(module_path: str) -> str:
    if any(f"{family}.1" in module_path for family in ("q_linears", "k_linears", "v_linears", "a_linears")):
        return "agent_type_1_not_realized_by_lidar_type0_export_contract"
    if module_path.endswith("prior_feed"):
        return "module_defined_but_not_called_by_HEAL_V2XTEncoder_forward"
    return "capability_present_but_not_realized_in_export_trace"


def build_v2xvit_onnx_mapping(
    onnx_path: str | Path,
    audit: ModelFamilyAudit,
    module_calls: Sequence[Mapping[str, Any] | Any],
) -> ModelFamilyOnnxMapping:
    """Map every static V2X-ViT weighted capability to active/inactive ONNX truth."""

    if audit.family_id != "heal_lidar_v2xvit":
        raise ValueError(f"v2xvit_mapping_family_mismatch:{audit.family_id}")
    import onnx

    source = Path(onnx_path)
    model = onnx.load(str(source), load_external_data=False)
    nodes = list(model.graph.node)
    initializers = {str(item.name) for item in model.graph.initializer}
    producers = {str(output): node for node in nodes for output in node.output}
    base = build_onnx_origin_map(source, module_calls)
    origins_by_module: dict[str, list[Any]] = defaultdict(list)
    for row in base.entries:
        origins_by_module[str(row.module_path)].append(row)

    functional_by_initializer: dict[str, list[tuple[int, Any]]] = defaultdict(list)
    activation_only_einsum: list[str] = []
    for graph_index, node in enumerate(nodes):
        if str(node.op_type) != "Einsum":
            continue
        relation_roots: set[str] = set()
        for input_name in node.input:
            relation_roots.update(
                root
                for root in _upstream_initializers(
                    str(input_name),
                    initializers=initializers,
                    producers=producers,
                )
                if root.endswith("relation_att") or root.endswith("relation_msg")
            )
        if relation_roots:
            for root in relation_roots:
                functional_by_initializer[root.removeprefix("model.")].append((graph_index, node))
        else:
            activation_only_einsum.append(str(node.name))

    entries: list[ModelFamilyOnnxWeightedEntry] = []
    unresolved: list[dict[str, Any]] = []
    for capability in sorted(audit.weighted_ops, key=lambda row: row.canonical_id):
        if capability.source_kind == "module":
            origins = sorted(origins_by_module.get(capability.module_path, ()), key=lambda row: row.call_index)
            if origins:
                entries.append(
                    ModelFamilyOnnxWeightedEntry(
                        canonical_id=capability.canonical_id,
                        module_path=capability.module_path,
                        source_kind=capability.source_kind,
                        mapping_status="active_mapped",
                        active=True,
                        onnx_op_types=tuple(str(row.onnx_op_type) for row in origins),
                        onnx_node_names=tuple(str(row.original_node_name) for row in origins),
                        graph_indices=tuple(int(row.graph_index) for row in origins),
                        weight_initializers=tuple(sorted({str(row.weight_initializer) for row in origins})),
                        call_count=len(origins),
                    )
                )
            else:
                reason = _inactive_reason(capability.module_path)
                status = "inactive_by_export_specialization"
                if reason == "capability_present_but_not_realized_in_export_trace":
                    status = "unresolved_active_module"
                    unresolved.append(
                        {
                            "canonical_id": capability.canonical_id,
                            "module_path": capability.module_path,
                            "reason": reason,
                        }
                    )
                entries.append(
                    ModelFamilyOnnxWeightedEntry(
                        canonical_id=capability.canonical_id,
                        module_path=capability.module_path,
                        source_kind=capability.source_kind,
                        mapping_status=status,
                        active=False,
                        reason=reason,
                    )
                )
            continue

        parameter_name = str(capability.metadata.get("parameter_name", ""))
        matches = sorted(functional_by_initializer.get(parameter_name, ()), key=lambda row: row[0])
        initializer = f"model.{parameter_name}"
        if matches and initializer in initializers:
            entries.append(
                ModelFamilyOnnxWeightedEntry(
                    canonical_id=capability.canonical_id,
                    module_path=capability.module_path,
                    source_kind=capability.source_kind,
                    mapping_status="active_functional_weight_mapped",
                    active=True,
                    onnx_op_types=tuple(str(node.op_type) for _, node in matches),
                    onnx_node_names=tuple(str(node.name) for _, node in matches),
                    graph_indices=tuple(int(index) for index, _ in matches),
                    weight_initializers=(initializer,),
                    call_count=len(matches),
                )
            )
        else:
            unresolved.append(
                {
                    "canonical_id": capability.canonical_id,
                    "module_path": capability.module_path,
                    "parameter_name": parameter_name,
                    "reason": "functional_weight_initializer_or_einsum_unresolved",
                }
            )
            entries.append(
                ModelFamilyOnnxWeightedEntry(
                    canonical_id=capability.canonical_id,
                    module_path=capability.module_path,
                    source_kind=capability.source_kind,
                    mapping_status="unresolved_functional_weight",
                    active=False,
                    reason="functional_weight_initializer_or_einsum_unresolved",
                )
            )

    grid_group = {
        "canonical_id": "functional::heal_lidar_v2xvit.export_grid_sample_affine_matmul",
        "mapping_status": "mapped_but_not_a_precision_gene",
        "op_type": "MatMul",
        "node_names": list(base.functional_matmul_nodes),
        "graph_indices": [
            int(index)
            for group in base.functional_compute_groups
            for index in group.graph_indices
        ],
        "source_call": "search.model_family.export.heal_v2xvit._warp_exportable",
        "precision_policy": "FP32_or_explicit_FP16_grid_generation_island",
        "reason": "parameter_free_affine_grid_generation",
    }
    active_entries = [row for row in entries if row.active]
    inactive_entries = [row for row in entries if not row.active]
    return ModelFamilyOnnxMapping(
        schema_version="heal-v2xvit-onnx-mapping-v1",
        family_id=audit.family_id,
        source_onnx=str(source.resolve()),
        audit_hash=audit.to_dict()["audit_hash"],
        weighted_entries=tuple(entries),
        parameter_free_compute_groups=(grid_group,) if base.functional_matmul_nodes else (),
        activation_only_einsum_nodes=tuple(activation_only_einsum),
        unresolved=tuple(unresolved),
        metadata={
            "static_weighted_capability_count": len(audit.weighted_ops),
            "active_weighted_capability_count": len(active_entries),
            "inactive_weighted_capability_count": len(inactive_entries),
            "active_module_weighted_capability_count": sum(
                row.active and row.source_kind == "module" for row in entries
            ),
            "active_functional_weighted_capability_count": sum(
                row.active and row.source_kind == "functional_parameter" for row in entries
            ),
            "active_weighted_compute_node_count": sum(row.call_count for row in active_entries),
            "parameter_free_grid_matmul_node_count": len(base.functional_matmul_nodes),
            "activation_only_einsum_node_count": len(activation_only_einsum),
            "realized_graph_mapping_complete": not unresolved,
            "full_static_branch_coverage": not inactive_entries,
            "type_specialization_policy": "DAIR_LiDAROnly_type0_until_real_manifest_proves_other_types",
        },
    )


__all__ = [
    "ModelFamilyOnnxMapping",
    "ModelFamilyOnnxWeightedEntry",
    "build_v2xvit_onnx_mapping",
]
