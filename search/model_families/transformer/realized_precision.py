"""Requested/realized precision audit driven by EngineInspector evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

from .accumulator_contract import infer_accumulator_from_layer


def normalize_trt_dtype(value: str) -> str:
    token = str(value).lower()
    if "fp8" in token or "e4m3" in token or "e5m2" in token:
        return "FP8"
    if "bfloat16" in token or "bf16" in token:
        return "BF16"
    if "int8" in token:
        return "INT8"
    if "int32" in token:
        return "INT32"
    if "half" in token or "fp16" in token or "float16" in token:
        return "FP16"
    if "float" in token or "fp32" in token:
        return "FP32"
    if "bool" in token:
        return "BOOL"
    return "unknown"


def _tensor_dtypes(layer: Mapping[str, Any], field: str) -> tuple[str, ...]:
    return tuple(
        normalize_trt_dtype(str(row.get("Format/Datatype", "")))
        for row in layer.get(field, ())
    )


@dataclass(frozen=True)
class RealizedPrecisionRecord:
    model: str
    profile: str
    block: str
    role: str
    onnx_node: str
    requested_precision: str
    realized_precision: str
    requested_accumulator: str
    realized_accumulator: str
    input_dtype: tuple[str, ...]
    output_dtype: tuple[str, ...]
    tensorrt_layer_name: str
    tensorrt_layer_type: str
    tactic: str
    fusion_kind: str
    cast: bool
    reformat: bool
    qdq: bool
    evidence_source: str
    confidence: str
    conflict: str
    fallback: bool
    implementation_layer_count: int
    implementation_precision_set: tuple[str, ...]
    logical_onnx_input_dtype: tuple[str, ...] = ()
    logical_onnx_output_dtype: tuple[str, ...] = ()
    realization_basis: str = "engine_inspector"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["input_dtype"] = list(self.input_dtype)
        payload["output_dtype"] = list(self.output_dtype)
        payload["implementation_precision_set"] = list(self.implementation_precision_set)
        payload["logical_onnx_input_dtype"] = list(self.logical_onnx_input_dtype)
        payload["logical_onnx_output_dtype"] = list(self.logical_onnx_output_dtype)
        return payload


def load_engine_layers(path: str | Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("Layers"), list):
        raise ValueError("invalid_engine_inspector_layer_info")
    return list(payload["Layers"]), list(payload.get("Bindings", ()))


def _candidate_layers(
    layers: Iterable[Mapping[str, Any]],
    *,
    node_name: str,
    tensor_names: Iterable[str] = (),
) -> list[Mapping[str, Any]]:
    node = str(node_name)
    tensors = {str(value) for value in tensor_names if str(value)}
    named = [
        row
        for row in layers
        if node.startswith("__canonical__") and node in str(row.get("Name", ""))
    ]
    metadata = [
        row
        for row in layers
        if node
        and node
        in {
            value.strip()
            for value in re.findall(
                r"\[ONNX Layer:\s*([^\]]+)\]", str(row.get("Metadata", ""))
            )
        }
    ]
    exact = named or metadata
    if exact:
        compute = [
            row
            for row in exact
            if str(row.get("LayerType", "")).lower()
            not in {"constant", "noop", "reformat", "signal", "wait"}
        ]
        return compute or exact
    tensor_matches = [
        row
        for row in layers
        if tensors
        and tensors
        & {
            str(value.get("Name", ""))
            for value in [*row.get("Inputs", ()), *row.get("Outputs", ())]
        }
    ]
    compute = [
        row
        for row in tensor_matches
        if str(row.get("LayerType", "")).lower()
        not in {"constant", "noop", "reformat", "signal", "wait"}
    ]
    return compute or tensor_matches


def _realized_boundary_precision(
    candidates: list[Mapping[str, Any]], requested_precision: str
) -> tuple[str, tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    first_inputs = _tensor_dtypes(candidates[0], "Inputs")
    last_outputs = _tensor_dtypes(candidates[-1], "Outputs")
    implementation = tuple(
        sorted(
            {
                dtype
                for layer in candidates
                for field in ("Inputs", "Outputs")
                for dtype in _tensor_dtypes(layer, field)
                if dtype in {"FP32", "FP16", "BF16", "FP8", "INT8"}
            }
        )
    )
    ingress = {value for value in first_inputs if value in implementation}
    egress = {value for value in last_outputs if value in implementation}
    if requested_precision in ingress and requested_precision in egress:
        realized = requested_precision
    elif requested_precision in implementation and (
        requested_precision in ingress or requested_precision in egress
    ):
        # TensorRT can fuse an explicit boundary Cast into the operator kernel.
        # The externally visible engine tensor then reflects the neighboring
        # role even though the requested arithmetic dtype is present inside the
        # fused implementation.  Keep this distinct from accumulator evidence.
        realized = requested_precision
    elif requested_precision in implementation:
        # Both the pre-op and post-op Cast can be fused into one implementation
        # region (LayerNorm is the common case).  Exact ONNX Metadata matching
        # plus a directly observed internal dtype is still realized evidence;
        # it says nothing about the accumulator, which remains separately
        # graded below.
        realized = requested_precision
    elif len(ingress) == 1 and ingress == egress:
        realized = next(iter(ingress))
    elif len(ingress) == 1 and not egress:
        realized = next(iter(ingress))
    elif len(egress) == 1 and not ingress:
        realized = next(iter(egress))
    else:
        realized = "unknown"
    return realized, first_inputs, last_outputs, implementation


def audit_realized_precision(
    *,
    model: str,
    profile: str,
    requested_rows: Iterable[Mapping[str, Any]],
    layer_info_path: str | Path,
    typed_onnx_path: str | Path | None = None,
    strongly_typed: bool = False,
) -> list[RealizedPrecisionRecord]:
    layers, _bindings = load_engine_layers(layer_info_path)
    logical_types: dict[str, str] = {}
    logical_nodes: dict[str, Any] = {}
    if typed_onnx_path is not None:
        import onnx

        graph = onnx.load(str(typed_onnx_path), load_external_data=False)
        logical_nodes = {str(node.name): node for node in graph.graph.node}
        for value in [*graph.graph.input, *graph.graph.value_info, *graph.graph.output]:
            logical_types[str(value.name)] = normalize_trt_dtype(
                onnx.TensorProto.DataType.Name(int(value.type.tensor_type.elem_type))
            )
        for value in graph.graph.initializer:
            logical_types[str(value.name)] = normalize_trt_dtype(
                onnx.TensorProto.DataType.Name(int(value.data_type))
            )
    records: list[RealizedPrecisionRecord] = []
    for requested in requested_rows:
        candidates = _candidate_layers(
            layers,
            node_name=str(requested.get("onnx_node", "")),
            tensor_names=requested.get("tensor_names", ()),
        )
        requested_precision = str(requested.get("requested_precision", "unknown")).upper()
        requested_accumulator = str(requested.get("requested_accumulator", "unknown")).upper()
        logical_node = logical_nodes.get(str(requested.get("onnx_node", "")))
        logical_inputs = tuple(
            logical_types.get(str(value), "unknown")
            for value in getattr(logical_node, "input", ())
        )
        logical_outputs = tuple(
            logical_types.get(str(value), "unknown")
            for value in getattr(logical_node, "output", ())
        )
        if not candidates:
            records.append(
                RealizedPrecisionRecord(
                    model=str(model),
                    profile=str(profile),
                    block=str(requested.get("block", "")),
                    role=str(requested.get("role", "")),
                    onnx_node=str(requested.get("onnx_node", "")),
                    requested_precision=requested_precision,
                    realized_precision="unknown",
                    requested_accumulator=requested_accumulator,
                    realized_accumulator="unknown",
                    input_dtype=(),
                    output_dtype=(),
                    tensorrt_layer_name="",
                    tensorrt_layer_type="",
                    tactic="",
                    fusion_kind="unresolved",
                    cast=False,
                    reformat=False,
                    qdq=False,
                    evidence_source="engine_inspector_no_unique_match",
                    confidence="missing",
                    conflict="engine_layer_match_count_0",
                    fallback=True,
                    implementation_layer_count=0,
                    implementation_precision_set=(),
                    logical_onnx_input_dtype=logical_inputs,
                    logical_onnx_output_dtype=logical_outputs,
                    realization_basis="unresolved_engine_metadata",
                )
            )
            continue
        realized, inputs, outputs, implementation = _realized_boundary_precision(
            candidates, requested_precision
        )
        accumulator_rows = [infer_accumulator_from_layer(layer) for layer in candidates]
        known_accumulators = {row[0] for row in accumulator_rows if row[0] != "unknown"}
        accumulator = next(iter(known_accumulators)) if len(known_accumulators) == 1 else "unknown"
        accumulator_level = (
            "A" if any(row[1] == "A" for row in accumulator_rows)
            else "B" if any(row[1] == "B" for row in accumulator_rows)
            else "C"
        )
        accumulator_source = ";".join(dict.fromkeys(row[2] for row in accumulator_rows))
        name = " || ".join(str(layer.get("Name", "")) for layer in candidates)
        layer_type = " || ".join(str(layer.get("LayerType", "")) for layer in candidates)
        tactic = " || ".join(
            str(layer.get("TacticName", "")) for layer in candidates
        )
        metadata = " || ".join(str(layer.get("Metadata", "")) for layer in candidates)
        searchable = f"{name} {layer_type} {metadata}".lower()
        logical_floating = {
            value
            for value in (*logical_inputs, *logical_outputs)
            if value in {"FP32", "FP16", "BF16", "FP8", "INT8"}
        }
        realization_basis = "engine_inspector_direct_dtype"
        if (
            strongly_typed
            and requested_precision in logical_floating
        ):
            # TensorRT strongly typed mode cannot choose a different logical
            # tensor dtype.  Exact EngineInspector Metadata proves that the
            # fused kernel realizes this ONNX region even when its public
            # ingress/egress has absorbed adjacent Cast nodes and therefore
            # does not expose the internal logical tensor in layer-info.
            realized = requested_precision
            realization_basis = "strongly_typed_onnx_contract_plus_engine_metadata"
        conflict = (
            ""
            if realized == requested_precision
            else f"requested_{requested_precision}_realized_{realized}"
        )
        role = str(requested.get("role", ""))
        if (
            not conflict
            and role in {"qk_matmul", "av_matmul"}
            and requested_accumulator != "UNKNOWN"
            and accumulator != requested_accumulator
        ):
            conflict = (
                f"requested_accumulator_{requested_accumulator}_"
                f"realized_{accumulator}"
            )
        records.append(
            RealizedPrecisionRecord(
                model=str(model),
                profile=str(profile),
                block=str(requested.get("block", "")),
                role=role,
                onnx_node=str(requested.get("onnx_node", "")),
                requested_precision=requested_precision,
                realized_precision=realized,
                requested_accumulator=requested_accumulator,
                realized_accumulator=accumulator,
                input_dtype=inputs,
                output_dtype=outputs,
                tensorrt_layer_name=name,
                tensorrt_layer_type=layer_type,
                tactic=tactic,
                fusion_kind=(
                    "fused_or_decomposed"
                    if len(candidates) > 1 or "\x1f" in metadata or "+" in name
                    else "primitive"
                ),
                cast="cast" in searchable,
                reformat="reformat" in searchable or layer_type.lower() == "noop",
                qdq=any(token in searchable for token in ("quantize", "dequantize", "qdq")),
                evidence_source=(
                    f"{realization_basis};EngineInspector;"
                    f"accumulator_level_{accumulator_level};{accumulator_source}"
                ),
                confidence=(
                    "high"
                    if str(requested.get("onnx_node", "")) in f"{name} {metadata}"
                    else "medium"
                ),
                conflict=conflict,
                fallback=bool(conflict),
                implementation_layer_count=len(candidates),
                implementation_precision_set=implementation,
                logical_onnx_input_dtype=logical_inputs,
                logical_onnx_output_dtype=logical_outputs,
                realization_basis=realization_basis,
            )
        )
    return records


def assert_no_precision_conflicts(records: Iterable[RealizedPrecisionRecord]) -> None:
    conflicts = [row for row in records if row.conflict or row.fallback]
    if conflicts:
        raise RuntimeError(
            "requested_realized_precision_conflict:"
            + ",".join(f"{row.role}:{row.conflict}" for row in conflicts[:16])
        )


__all__ = [
    "RealizedPrecisionRecord",
    "assert_no_precision_conflicts",
    "audit_realized_precision",
    "load_engine_layers",
    "normalize_trt_dtype",
]
