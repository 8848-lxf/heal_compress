"""Canonical precision capability and profile rules for LiDAR CoBEVT."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from quantization.precision.canonical_mapping import (
    build_canonical_precision_mapping,
)
from quantization.types import (
    CanonicalPrecisionMappingResult,
    OnnxOriginMapResult,
    PrecisionAssignment,
    PrecisionProfileResult,
    stable_json_hash,
)


def _onnx_tensor_types(model: Any) -> dict[str, int]:
    types = {
        str(initializer.name): int(initializer.data_type)
        for initializer in model.graph.initializer
    }
    for value in [*model.graph.input, *model.graph.value_info, *model.graph.output]:
        tensor_type = value.type.tensor_type
        if tensor_type.elem_type:
            types[str(value.name)] = int(tensor_type.elem_type)
    changed = True
    while changed:
        changed = False
        for node in model.graph.node:
            output_type = None
            if str(node.op_type) == "Cast":
                output_type = next(
                    (
                        int(attribute.i)
                        for attribute in node.attribute
                        if str(attribute.name) == "to"
                    ),
                    None,
                )
            elif str(node.op_type) in {
                "Einsum",
                "Identity",
                "Flatten",
                "Relu",
                "Reshape",
                "Softmax",
                "Squeeze",
                "Transpose",
                "Unsqueeze",
            } and node.input:
                output_type = types.get(str(node.input[0]))
            if output_type is None:
                continue
            for output in node.output:
                if types.get(str(output)) != int(output_type):
                    types[str(output)] = int(output_type)
                    changed = True
    return types


def _scatter_qdq_count(model: Any) -> int:
    producers = {
        str(output): node for node in model.graph.node for output in node.output
    }
    consumers: dict[str, list[Any]] = {}
    for node in model.graph.node:
        for value in node.input:
            consumers.setdefault(str(value), []).append(node)
    adjacent = []
    for plugin in (
        node
        for node in model.graph.node
        if str(node.op_type) == "PointPillarScatterTRT"
    ):
        adjacent.extend(
            producer
            for value in plugin.input
            if (producer := producers.get(str(value))) is not None
        )
        for value in plugin.output:
            adjacent.extend(consumers.get(str(value), ()))
    return sum(
        str(node.op_type) in {"QuantizeLinear", "DequantizeLinear"}
        for node in adjacent
    )


def apply_cobevt_auxiliary_typed_contract(
    input_onnx: str | Path,
    output_onnx: str | Path,
) -> dict[str, Any]:
    """Close CoBEVT-only auxiliary dtype gaps after canonical graph typing."""

    import onnx
    from onnx import TensorProto, helper

    model = onnx.load(str(input_onnx))
    try:
        model = onnx.shape_inference.infer_shapes(
            model, strict_mode=False, data_prop=True
        )
    except Exception:
        pass
    types = _onnx_tensor_types(model)
    existing_names = {
        str(node.name) for node in model.graph.node if str(node.name)
    } | {str(output) for node in model.graph.node for output in node.output}
    rewritten = []
    layernorm_records = []
    elementwise_records = []
    where_records = []
    concat_records = []
    layernorm_count = 0
    concat_count = 0
    for node in model.graph.node:
        op_type = str(node.op_type)
        cast_inputs: tuple[int, ...] = ()
        target_input_index = 0
        forced_activation_type: int | None = None
        cast_records = layernorm_records
        if op_type == "LayerNormalization":
            layernorm_count += 1
            cast_inputs = (1, 2)
        elif op_type in {"Add", "Sub", "Mul", "Div", "Pow", "Min", "Max"}:
            cast_inputs = tuple(range(1, len(node.input)))
            cast_records = elementwise_records
        elif op_type == "Where" and len(node.input) >= 3:
            target_input_index = 2
            cast_inputs = (1,)
            cast_records = where_records
        elif op_type == "Concat":
            concat_count += 1
            input_types = [types.get(str(value)) for value in node.input]
            floating = {int(TensorProto.FLOAT), int(TensorProto.FLOAT16)}
            if (
                input_types
                and all(value in floating for value in input_types)
                and len(set(input_types)) > 1
            ):
                forced_activation_type = (
                    int(TensorProto.FLOAT16)
                    if int(TensorProto.FLOAT16) in input_types
                    else int(TensorProto.FLOAT)
                )
                cast_inputs = tuple(
                    index
                    for index, value in enumerate(input_types)
                    if int(value) != forced_activation_type
                )
                cast_records = concat_records
        if not cast_inputs:
            rewritten.append(node)
            continue
        if forced_activation_type is None and str(node.input[target_input_index]) not in types:
            raise RuntimeError(
                f"cobevt_{op_type.lower()}_activation_dtype_unresolved:{node.name}"
            )
        activation_type = int(
            forced_activation_type
            if forced_activation_type is not None
            else types[str(node.input[target_input_index])]
        )
        for input_index in cast_inputs:
            if input_index >= len(node.input) or not str(node.input[input_index]):
                continue
            source = str(node.input[input_index])
            source_type = types.get(source)
            if source_type is None:
                raise RuntimeError(
                    f"cobevt_layernorm_parameter_dtype_unresolved:{node.name}:{input_index}"
                )
            if int(source_type) == activation_type:
                continue
            digest = hashlib.sha256(
                f"{node.name}\0{input_index}\0{activation_type}".encode("utf-8")
            ).hexdigest()[:12]
            cast_name = f"__cobevt_typed__{digest}__Cast"
            suffix = 0
            while cast_name in existing_names:
                suffix += 1
                cast_name = f"__cobevt_typed__{digest}_{suffix}__Cast"
            existing_names.add(cast_name)
            rewritten.append(
                helper.make_node(
                    "Cast",
                    [source],
                    [cast_name],
                    name=cast_name,
                    to=activation_type,
                )
            )
            node.input[input_index] = cast_name
            types[cast_name] = activation_type
            cast_records.append(
                {
                    "cast_node": cast_name,
                    "input_index": input_index,
                    "layernorm_node": str(node.name),
                    "source_tensor": source,
                    "source_type": int(source_type),
                    "target_type": activation_type,
                }
            )
        rewritten.append(node)
        for output in node.output:
            types[str(output)] = activation_type
    del model.graph.node[:]
    model.graph.node.extend(rewritten)
    scatter_qdq_count = _scatter_qdq_count(model)
    if scatter_qdq_count:
        raise RuntimeError(f"cobevt_scatter_qdq_forbidden:{scatter_qdq_count}")
    destination = Path(output_onnx)
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(destination))
    return {
        "input_onnx": str(input_onnx),
        "output_onnx": str(destination),
        "layernorm_count": layernorm_count,
        "layernorm_parameter_cast_count": len(layernorm_records),
        "layernorm_parameter_cast_records": layernorm_records,
        "elementwise_input_cast_count": len(elementwise_records),
        "elementwise_input_cast_records": elementwise_records,
        "where_input_cast_count": len(where_records),
        "where_input_cast_records": where_records,
        "concat_node_count": concat_count,
        "concat_input_cast_count": len(concat_records),
        "concat_input_cast_records": concat_records,
        "concat_contract_dtype": "FP16" if concat_records else "",
        "scatter_qdq_count": scatter_qdq_count,
        "policy_version": "lidar-cobevt-auxiliary-typed-contract-v1",
    }


def _is_prediction_head(module_path: str) -> bool:
    return str(module_path) in {"cls_head", "reg_head", "dir_head"}


@dataclass(frozen=True)
class CobevtPrecisionCapabilityEntry:
    module_path: str
    canonical_node_name: str
    onnx_op_type: str
    precision_group: str
    actions: tuple[str, ...]
    is_precision_gene: bool
    protection_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "actions": list(self.actions),
            "canonical_node_name": self.canonical_node_name,
            "is_precision_gene": self.is_precision_gene,
            "module_path": self.module_path,
            "onnx_op_type": self.onnx_op_type,
            "precision_group": self.precision_group,
            "protection_reason": self.protection_reason,
        }


@dataclass(frozen=True)
class CobevtPrecisionCapabilityManifest:
    weighted_entries: tuple[CobevtPrecisionCapabilityEntry, ...]
    functional_entries: tuple[CobevtPrecisionCapabilityEntry, ...]
    duplicate_precision_group_count: int
    scatter_is_precision_gene: bool
    recipe_version: str
    capability_hash: str

    def to_dict(self) -> dict:
        return {
            "capability_hash": self.capability_hash,
            "duplicate_precision_group_count": self.duplicate_precision_group_count,
            "functional_entries": [row.to_dict() for row in self.functional_entries],
            "recipe_version": self.recipe_version,
            "scatter_is_precision_gene": self.scatter_is_precision_gene,
            "weighted_entries": [row.to_dict() for row in self.weighted_entries],
        }


class CobevtQuantizationRecipe:
    RECIPE_VERSION = "lidar-cobevt-canonical-precision-v1"

    def __init__(
        self, *, verified_int8_modules: Iterable[str] | None = None
    ) -> None:
        self.verified_int8_modules = {
            str(value) for value in (verified_int8_modules or ())
        }

    def build_capability(
        self, origin_map: OnnxOriginMapResult
    ) -> CobevtPrecisionCapabilityManifest:
        weighted = []
        for origin in sorted(origin_map.entries, key=lambda row: row.module_path):
            actions = ["FP32", "FP16"]
            protected_reason = ""
            if _is_prediction_head(origin.module_path):
                protected_reason = "raw_detection_output_not_int8"
            elif origin.module_path in self.verified_int8_modules:
                actions.append("INT8")
            weighted.append(
                CobevtPrecisionCapabilityEntry(
                    module_path=origin.module_path,
                    canonical_node_name=origin.canonical_node_name,
                    onnx_op_type=origin.onnx_op_type,
                    precision_group=f"cobevt_pg::{origin.module_path}",
                    actions=tuple(actions),
                    is_precision_gene=True,
                    protection_reason=protected_reason,
                )
            )
        functional = tuple(
            CobevtPrecisionCapabilityEntry(
                module_path=group.module_path,
                canonical_node_name=group.canonical_node_name,
                onnx_op_type=group.onnx_op_type,
                precision_group=f"cobevt_protected::{group.module_path}",
                actions=("FP16",),
                is_precision_gene=False,
                protection_reason=group.protection_reason,
            )
            for group in sorted(
                origin_map.functional_compute_groups,
                key=lambda row: row.module_path,
            )
        )
        groups = [row.precision_group for row in weighted]
        duplicates = len(groups) - len(set(groups))
        payload = {
            "functional": [row.to_dict() for row in functional],
            "recipe_version": self.RECIPE_VERSION,
            "scatter_is_precision_gene": False,
            "weighted": [row.to_dict() for row in weighted],
        }
        return CobevtPrecisionCapabilityManifest(
            weighted_entries=tuple(weighted),
            functional_entries=functional,
            duplicate_precision_group_count=duplicates,
            scatter_is_precision_gene=False,
            recipe_version=self.RECIPE_VERSION,
            capability_hash=stable_json_hash(payload),
        )

    def build_profile(
        self,
        capability: CobevtPrecisionCapabilityManifest,
        requested: Mapping[str, str],
        *,
        profile_id: str,
    ) -> PrecisionProfileResult:
        expected = {row.module_path for row in capability.weighted_entries}
        actual = {str(name) for name in requested}
        if expected != actual:
            raise RuntimeError(
                f"precision_profile_module_set_mismatch:missing={sorted(expected-actual)},unknown={sorted(actual-expected)}"
            )
        assignments = []
        for ordering, row in enumerate(capability.weighted_entries):
            precision = str(requested[row.module_path]).strip().upper()
            if precision not in row.actions:
                raise RuntimeError(
                    f"precision_action_not_deployable:{row.module_path}:{precision}"
                )
            assignments.append(
                PrecisionAssignment(
                    module_path=row.module_path,
                    precision_group=row.precision_group,
                    requested_precision=precision.lower(),
                    protected_precision=(
                        precision.lower() if len(row.actions) == 1 else ""
                    ),
                    ordering=ordering,
                )
            )
        return PrecisionProfileResult(
            profile_id=str(profile_id),
            assignments=assignments,
            requested_int8_count=sum(
                row.requested_precision == "int8" for row in assignments
            ),
            requested_int8_ratio=(
                sum(row.requested_precision == "int8" for row in assignments)
                / max(len(assignments), 1)
            ),
            policy_version=self.RECIPE_VERSION,
        )

    def build_mapping(
        self,
        origin_map: OnnxOriginMapResult,
        profile: PrecisionProfileResult,
    ) -> CanonicalPrecisionMappingResult:
        mapping = build_canonical_precision_mapping(origin_map, profile)
        fallback = [
            row
            for row in mapping.entries
            if row.requested_precision != row.realized_request_precision
        ]
        if fallback:
            raise RuntimeError(
                "cobevt_precision_legalization_changed_request:"
                + ",".join(row.module_path for row in fallback)
            )
        return mapping
