"""Fail-closed Transformer precision search and realization contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from typing import Any, Mapping, Sequence

import torch.nn as nn

from ..pruning_space.transformer_domains import AttentionInstanceSpec, FFNInstanceSpec
from .types import QuantizationSearchGroup


SEARCH_PRECISION_STATES = ("W32A32", "W16A16", "W8A8")
ACTIVATION_PRECISION_STATES = ("A32", "A16", "A8")
INTERNAL_BY_SEARCH_STATE = {
    "W32A32": "FP32",
    "W16A16": "FP16",
    "W8A8": "INT8",
    "A32": "FP32",
    "A16": "FP16",
    "A8": "INT8",
}
SEARCH_STATE_BY_INTERNAL = {"FP32": "W32A32", "FP16": "W16A16", "INT8": "W8A8"}


def internal_precision_state(value: str) -> str:
    text = str(value).upper()
    if text in INTERNAL_BY_SEARCH_STATE:
        return INTERNAL_BY_SEARCH_STATE[text]
    if text in SEARCH_STATE_BY_INTERNAL:
        return text
    raise ValueError(f"transformer_precision_state_invalid:{value}")


@dataclass(frozen=True)
class TransformerPrecisionUnit:
    unit_id: str
    model: str
    family: str
    module_paths: tuple[str, ...]
    role: str
    allowed_states: tuple[str, ...]
    default_state: str
    activation_only: bool
    protected: bool
    protection_reason: str = ""
    ordering: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        legal = ACTIVATION_PRECISION_STATES if self.activation_only else SEARCH_PRECISION_STATES
        states = tuple(str(value).upper() for value in self.allowed_states)
        if not states or any(value not in legal for value in states):
            raise ValueError(f"transformer_precision_unit_states_invalid:{self.unit_id}:{states}")
        if str(self.default_state).upper() not in states:
            raise ValueError(f"transformer_precision_default_not_allowed:{self.unit_id}")
        if self.role == "qk_matmul" and states != ("A32",):
            raise ValueError("qk_matmul_must_be_fixed_a32")
        if self.role == "layernorm" and states != ("A32",):
            raise ValueError("layernorm_must_be_fixed_a32")
        if self.role == "residual_add" and "A8" in states:
            raise ValueError("residual_add_int8_forbidden_without_engine_audit")
        object.__setattr__(self, "allowed_states", states)
        object.__setattr__(self, "default_state", str(self.default_state).upper())
        object.__setattr__(self, "module_paths", tuple(str(value) for value in self.module_paths))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def allowed_internal_precisions(self) -> tuple[str, ...]:
        return tuple(INTERNAL_BY_SEARCH_STATE[value] for value in self.allowed_states)

    def to_search_group(self) -> QuantizationSearchGroup:
        return QuantizationSearchGroup(
            group_id=self.unit_id,
            module_paths=self.module_paths,
            canonical_node_ids=tuple(
                str(value) for value in self.metadata.get("canonical_node_ids", self.module_paths)
            ),
            allowed_precisions=self.allowed_internal_precisions,
            protected=self.protected,
            protection_reason=self.protection_reason,
            ordering=self.ordering,
            parameter_count=int(self.metadata.get("parameter_count", 0)),
            baseline_macs=float(self.metadata.get("baseline_macs", 0.0)),
            metadata={
                **self.metadata,
                "transformer_role": self.role,
                "external_allowed_states": list(self.allowed_states),
                "external_default_state": self.default_state,
                "activation_only": self.activation_only,
                "weight_activation_bound": not self.activation_only,
                "default_precision": INTERNAL_BY_SEARCH_STATE[self.default_state],
            },
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["allowed_internal_precisions"] = list(self.allowed_internal_precisions)
        return value


def _parameter_count(model: nn.Module | None, paths: Sequence[str]) -> int:
    if model is None:
        return 0
    total = 0
    seen: set[int] = set()
    for path in paths:
        try:
            module = model.get_submodule(path)
        except AttributeError:
            continue
        for parameter in module.parameters(recurse=False):
            if id(parameter) not in seen:
                seen.add(id(parameter))
                total += int(parameter.numel())
    return total


def build_transformer_precision_units(
    attention: Sequence[AttentionInstanceSpec],
    ffn: Sequence[FFNInstanceSpec],
    *,
    model: nn.Module | None = None,
    layernorm_paths: Sequence[str] = (),
    include_residual_boundaries: bool = True,
) -> tuple[TransformerPrecisionUnit, ...]:
    """Emit deployable units without assigning one module to two genes."""

    units: list[TransformerPrecisionUnit] = []
    residual_boundaries: set[str] = set()
    window_merge_boundaries: set[str] = set()

    def add(
        *,
        unit_id: str,
        model_name: str,
        family: str,
        paths: Sequence[str],
        role: str,
        states: tuple[str, ...],
        default: str,
        activation_only: bool,
        protected: bool = False,
        reason: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        path_tuple = tuple(dict.fromkeys(str(value) for value in paths))
        units.append(TransformerPrecisionUnit(
            unit_id=unit_id,
            model=model_name,
            family=family,
            module_paths=path_tuple,
            role=role,
            allowed_states=states,
            default_state=default,
            activation_only=activation_only,
            protected=protected,
            protection_reason=reason,
            ordering=len(units),
            metadata={
                **dict(metadata or {}),
                "parameter_count": _parameter_count(model, path_tuple),
            },
        ))

    for spec in attention:
        prefix = f"transformer_precision::{spec.module_path}"
        if spec.qkv_layout == "fused_qkv":
            fused = tuple(dict.fromkeys(
                spec.q_projection_paths + spec.k_projection_paths + spec.v_projection_paths
            ))
            add(
                unit_id=f"{prefix}::fused_qkv_projection",
                model_name=spec.model,
                family=spec.family,
                paths=fused,
                role="fused_qkv_projection",
                states=SEARCH_PRECISION_STATES,
                default="W32A32",
                activation_only=False,
                metadata={
                    "component_roles": ["q_projection", "k_projection", "v_projection"],
                    "shared_input_smoothing_boundary": True,
                },
            )
        else:
            add(
                unit_id=f"{prefix}::qk_projection",
                model_name=spec.model,
                family=spec.family,
                paths=spec.q_projection_paths + spec.k_projection_paths,
                role="qk_projection",
                states=SEARCH_PRECISION_STATES,
                default="W32A32",
                activation_only=False,
                metadata={"dq_cast_to_fp32_before_qk": True},
            )
            add(
                unit_id=f"{prefix}::v_projection",
                model_name=spec.model,
                family=spec.family,
                paths=spec.v_projection_paths,
                role="v_projection",
                states=SEARCH_PRECISION_STATES,
                default="W32A32",
                activation_only=False,
            )
        qk_functional_paths = [f"{spec.module_path}::__qk_matmul__"]
        if spec.adapter == "v2xvit_hgt" and spec.metadata.get("relation_att_path"):
            qk_functional_paths.append(str(spec.metadata["relation_att_path"]))
        add(
            unit_id=f"{prefix}::qk_matmul",
            model_name=spec.model,
            family=spec.family,
            paths=tuple(qk_functional_paths),
            role="qk_matmul",
            states=("A32",),
            default="A32",
            activation_only=True,
            protected=True,
            reason="operands_accumulation_and_output_fixed_fp32",
            metadata={
                "operand_precision": "FP32",
                "compute_precision": "FP32",
                "accumulator_precision": "FP32",
                "output_precision": "FP32",
                "dq_cast_required": True,
                "functional_owner": spec.module_path,
                "functional_op": "einsum",
                "functional_call_index": 0,
            },
        )
        av_functional_paths = [f"{spec.module_path}::__av_matmul__"]
        if spec.adapter == "v2xvit_hgt" and spec.metadata.get("relation_msg_path"):
            av_functional_paths.append(str(spec.metadata["relation_msg_path"]))
        add(
            unit_id=f"{prefix}::softmax",
            model_name=spec.model,
            family=spec.family,
            paths=spec.softmax_paths or (f"{spec.module_path}::__softmax_output__",),
            role="softmax",
            states=("A32",),
            default="A32",
            activation_only=True,
            protected=True,
            reason="deployment_closed_contract_floating_softmax_fp32_output",
            metadata={
                "A8_semantics": "floating_softmax_then_qdq_int8_output",
                "native_int8_compute_label": "INT8_PLUGIN",
                "functional_owner": spec.module_path,
                "functional_op": "softmax",
                "functional_call_index": 0,
            },
        )
        add(
            unit_id=f"{prefix}::av",
            model_name=spec.model,
            family=spec.family,
            paths=tuple(av_functional_paths),
            role="av_matmul",
            states=("A32",),
            default="A32",
            activation_only=True,
            protected=True,
            reason="deployment_closed_contract_av_fp32",
            metadata={
                "functional_owner": spec.module_path,
                "functional_op": "einsum",
                "functional_call_index": 2 if spec.adapter == "v2xvit_hgt" else 1,
            },
        )
        add(
            unit_id=f"{prefix}::output_projection",
            model_name=spec.model,
            family=spec.family,
            paths=spec.output_projection_paths,
            role="output_projection",
            states=SEARCH_PRECISION_STATES,
            default="W32A32",
            activation_only=False,
        )
        if include_residual_boundaries and spec.block_path not in residual_boundaries:
            residual_boundaries.add(spec.block_path)
            add(
                unit_id=f"transformer_precision::{spec.block_path}::attention_residual_add",
                model_name=spec.model,
                family=spec.family,
                paths=(f"{spec.block_path}::__attention_residual_add__",),
                role="residual_add",
                states=("A16",),
                default="A16",
                activation_only=True,
                protected=True,
                reason="int8_residual_add_not_open_without_realized_engine_audit",
                metadata={
                    "functional_owner": spec.module_path,
                    "attention_adapter": spec.adapter,
                    "boundary_kind": "attention_residual_add",
                },
            )
        if (
            include_residual_boundaries
            and spec.adapter == "v2xvit_window"
            and spec.block_path not in window_merge_boundaries
        ):
            window_merge_boundaries.add(spec.block_path)
            add(
                unit_id=f"transformer_precision::{spec.block_path}::window_merge",
                model_name=spec.model,
                family=spec.family,
                paths=(f"{spec.block_path}::__window_family_merge__",),
                role="attention_merge",
                states=("A16",),
                default="A16",
                activation_only=True,
                protected=True,
                reason="window_family_merge_fixed_fp16_deployment_contract",
                metadata={
                    "functional_owner": spec.block_path,
                    "attention_adapter": spec.adapter,
                    "boundary_kind": "window_family_merge",
                },
            )

    for spec in ffn:
        prefix = f"transformer_precision::{spec.module_path}"
        if spec.ffn_type == "gated":
            first_paths = (spec.gate_projection_path, spec.up_projection_path)
            first_role = "gated_ffn1"
        else:
            first_paths = (spec.first_projection_path,)
            first_role = "ffn1"
        add(
            unit_id=f"{prefix}::ffn1",
            model_name=spec.model,
            family=spec.family,
            paths=first_paths,
            role=first_role,
            states=SEARCH_PRECISION_STATES,
            default="W32A32",
            activation_only=False,
        )
        add(
            unit_id=f"{prefix}::ffn2",
            model_name=spec.model,
            family=spec.family,
            paths=((spec.down_projection_path,) if spec.ffn_type == "gated" else (spec.second_projection_path,)),
            role="ffn2",
            states=SEARCH_PRECISION_STATES,
            default="W32A32",
            activation_only=False,
        )

    model_name = attention[0].model if attention else ffn[0].model if ffn else ""
    for path in layernorm_paths:
        add(
            unit_id=f"transformer_precision::{path}::layernorm",
            model_name=model_name,
            family="transformer_layernorm",
            paths=(path,),
            role="layernorm",
            states=("A32",),
            default="A32",
            activation_only=True,
            protected=True,
            reason="layernorm_initial_contract_fp32",
            metadata={
                "functional_owner": path,
                "boundary_kind": "layernorm",
            },
        )

    if include_residual_boundaries:
        for spec in ffn:
            add(
                unit_id=f"transformer_precision::{spec.block_path}::ffn_residual_add",
                model_name=spec.model,
                family=spec.family,
                paths=(f"{spec.block_path}::__ffn_residual_add__",),
                role="residual_add",
                states=("A16",),
                default="A16",
                activation_only=True,
                protected=True,
                reason="ffn_residual_add_fixed_fp16_deployment_contract",
                metadata={
                    "functional_owner": spec.module_path,
                    "boundary_kind": "ffn_residual_add",
                },
            )

    module_owners: dict[str, str] = {}
    for unit in units:
        for path in unit.module_paths:
            if "::__" in path:
                continue
            previous = module_owners.setdefault(path, unit.unit_id)
            if previous != unit.unit_id:
                raise RuntimeError(f"precision_module_in_multiple_units:{path}:{previous}:{unit.unit_id}")
    return tuple(units)


def build_transformer_quantization_groups(
    units: Sequence[TransformerPrecisionUnit],
) -> tuple[QuantizationSearchGroup, ...]:
    return tuple(unit.to_search_group() for unit in sorted(units, key=lambda value: value.ordering))


def validate_external_precision_profile(
    requested: Mapping[str, str],
    units: Sequence[TransformerPrecisionUnit],
) -> dict[str, str]:
    by_id = {unit.unit_id: unit for unit in units}
    unknown = sorted(set(requested) - set(by_id))
    if unknown:
        raise ValueError(f"unknown_transformer_precision_units:{unknown}")
    legalized: dict[str, str] = {}
    for unit in units:
        value = str(requested.get(unit.unit_id, unit.default_state)).upper()
        if value not in unit.allowed_states:
            raise ValueError(f"transformer_precision_request_illegal:{unit.unit_id}:{value}")
        legalized[unit.unit_id] = value
    return legalized


def expected_softmax_realization(state: str, *, native_int8_plugin: bool = False) -> dict[str, Any]:
    value = str(state).upper()
    if value not in ACTIVATION_PRECISION_STATES:
        raise ValueError(f"softmax_state_invalid:{state}")
    if value == "A8" and native_int8_plugin:
        return {
            "softmax_compute": "INT8_PLUGIN",
            "softmax_input": "INT8",
            "softmax_output": "INT8",
            "qdq": False,
            "semantics": "native_int8_plugin",
        }
    if value == "A8":
        return {
            "softmax_compute": "FP32",
            "softmax_input": "FP32",
            "softmax_output": "INT8",
            "qdq": True,
            "semantics": "floating_softmax_quantized_output",
        }
    precision = INTERNAL_BY_SEARCH_STATE[value]
    return {
        "softmax_compute": precision,
        "softmax_input": precision,
        "softmax_output": precision,
        "qdq": False,
        "semantics": "floating_softmax",
    }


@dataclass(frozen=True)
class RealizedTransformerPrecision:
    unit_id: str
    role: str
    requested_state: str
    realized_weight: str
    realized_activation: str
    compute_precision: str
    accumulator_precision: str
    output_precision: str
    qdq: bool
    cast: bool
    reformat: bool
    fusion: str = ""
    fallback: bool = False
    tactic: str = ""
    native_int8_plugin: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def assert_transformer_precision_realized(
    requested: Mapping[str, str],
    units: Sequence[TransformerPrecisionUnit],
    realized: Sequence[RealizedTransformerPrecision],
) -> None:
    legalized = validate_external_precision_profile(requested, units)
    by_id = {row.unit_id: row for row in realized}
    issues: list[str] = []
    for unit in units:
        row = by_id.get(unit.unit_id)
        if row is None:
            issues.append(f"missing_unit_mapping:{unit.unit_id}")
            continue
        state = legalized[unit.unit_id]
        if str(row.requested_state).upper() != state:
            issues.append(f"requested_state_evidence_mismatch:{unit.unit_id}")
        if row.fallback:
            issues.append(f"precision_fallback:{unit.unit_id}")
        if unit.role == "qk_matmul":
            if not all(
                value == "FP32"
                for value in (
                    row.realized_activation,
                    row.compute_precision,
                    row.accumulator_precision,
                    row.output_precision,
                )
            ):
                issues.append(f"qk_fp32_contract_conflict:{unit.unit_id}")
        elif unit.role == "layernorm" and row.compute_precision != "FP32":
            issues.append(f"layernorm_fp32_contract_conflict:{unit.unit_id}")
        elif unit.role == "softmax":
            expected = expected_softmax_realization(state, native_int8_plugin=row.native_int8_plugin)
            if row.compute_precision != expected["softmax_compute"]:
                issues.append(f"softmax_compute_conflict:{unit.unit_id}")
            if row.output_precision != expected["softmax_output"] or bool(row.qdq) != bool(expected["qdq"]):
                issues.append(f"softmax_output_boundary_conflict:{unit.unit_id}")
        elif unit.activation_only:
            expected = INTERNAL_BY_SEARCH_STATE[state]
            if row.output_precision != expected:
                issues.append(f"activation_precision_conflict:{unit.unit_id}:{expected}:{row.output_precision}")
        else:
            expected = INTERNAL_BY_SEARCH_STATE[state]
            if row.realized_weight != expected or row.realized_activation != expected:
                issues.append(f"weight_activation_precision_conflict:{unit.unit_id}:{expected}")
    extra = sorted(set(by_id) - {unit.unit_id for unit in units})
    if extra:
        issues.append(f"unexpected_realized_precision_units:{extra}")
    if issues:
        raise RuntimeError("requested_realized_precision_conflict:" + ";".join(issues))


def precision_manifest_hash(units: Sequence[TransformerPrecisionUnit]) -> str:
    payload = [unit.to_dict() for unit in sorted(units, key=lambda value: value.ordering)]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


__all__ = [
    "ACTIVATION_PRECISION_STATES",
    "INTERNAL_BY_SEARCH_STATE",
    "RealizedTransformerPrecision",
    "SEARCH_PRECISION_STATES",
    "TransformerPrecisionUnit",
    "assert_transformer_precision_realized",
    "build_transformer_precision_units",
    "build_transformer_quantization_groups",
    "expected_softmax_realization",
    "internal_precision_state",
    "precision_manifest_hash",
    "validate_external_precision_profile",
]
