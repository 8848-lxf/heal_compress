"""Evidence grading for TensorRT/CUTLASS attention accumulation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any, Mapping


@dataclass(frozen=True)
class AccumulatorEvidence:
    operator: str
    requested_operand: str
    requested_accumulator: str
    realized_operand: str
    realized_accumulator: str
    evidence_level: str
    evidence_source: str
    searchable: bool
    conflict: str = ""

    def __post_init__(self) -> None:
        if self.evidence_level not in {"A", "B", "C"}:
            raise ValueError(f"invalid_accumulator_evidence_level:{self.evidence_level}")
        if self.searchable and self.evidence_level != "A":
            raise ValueError("only_level_a_accumulator_profile_is_searchable")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def infer_accumulator_from_layer(layer: Mapping[str, Any]) -> tuple[str, str, str]:
    """Return accumulator, level and source; never infer it from output dtype."""

    searchable = " ".join(
        str(layer.get(key, "")) for key in ("Name", "LayerType", "TacticName", "Metadata")
    ).lower()
    tactic = str(layer.get("TacticName", ""))
    if re.search(r"(?:f16|half).*(?:f32|float).*acc", searchable) or "f16f16_f32" in searchable:
        return "FP32", "A", f"TensorRT tactic specialization:{tactic}"
    if re.search(r"(?:bf16).*(?:f32|float).*acc", searchable) or "bf16bf16_f32" in searchable:
        return "FP32", "A", f"TensorRT tactic specialization:{tactic}"
    if (
        ("int8" in searchable and "int32" in searchable)
        or "i8i8_i32" in searchable
    ):
        return "INT32", "A", f"TensorRT tactic specialization:{tactic}"
    if "f32f32" in searchable and ("_f32" in searchable or "float" in searchable):
        return "FP32", "A", f"TensorRT tactic specialization:{tactic}"
    if tactic:
        return "unknown", "B", f"tactic present without accumulator metadata:{tactic}"
    return "unknown", "C", "EngineInspector exposes no accumulator metadata"


def build_accumulator_evidence(
    *,
    operator: str,
    requested_operand: str,
    requested_accumulator: str,
    realized_operand: str,
    layer: Mapping[str, Any],
) -> AccumulatorEvidence:
    accumulator, level, source = infer_accumulator_from_layer(layer)
    conflict = ""
    if accumulator != "unknown" and requested_accumulator != accumulator:
        conflict = f"requested_{requested_accumulator}_realized_{accumulator}"
    return AccumulatorEvidence(
        operator=str(operator),
        requested_operand=str(requested_operand),
        requested_accumulator=str(requested_accumulator),
        realized_operand=str(realized_operand),
        realized_accumulator=accumulator,
        evidence_level=level,
        evidence_source=source,
        searchable=level == "A" and not conflict,
        conflict=conflict,
    )


__all__ = [
    "AccumulatorEvidence",
    "build_accumulator_evidence",
    "infer_accumulator_from_layer",
]
