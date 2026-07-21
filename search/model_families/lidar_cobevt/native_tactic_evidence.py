"""Parse TensorRT editable-cache profiling logs without guessing precision."""

from __future__ import annotations

import re
from typing import Any, Iterable


_AUTOTUNE = re.compile(r"Autotuning op\s+(.+?)\(key:\s*(0x[0-9a-fA-F]+)\)")
_TACTIC = re.compile(
    r"^\s*(\d+),\s*([^,]+),\s*([^,]+),\s*([^,]+),\s*(.+?),\s*(0x[0-9a-fA-F]+),"
)
_SELECTED = re.compile(r"selected tactic.*?:\s*(0x[0-9a-fA-F]+)", re.IGNORECASE)


def parse_editable_timing_log(lines: Iterable[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    in_table = False
    for raw in lines:
        line = str(raw)
        match = _AUTOTUNE.search(line)
        if match:
            current = {
                "op": match.group(1).strip(),
                "key": match.group(2),
                "available_tactics": [],
                "selected_tactic": None,
            }
            records.append(current)
            in_table = False
            continue
        if current is None:
            continue
        if "tactic_id, cost(in ms)" in line:
            in_table = True
            continue
        if in_table:
            tactic = _TACTIC.search(line.replace("(foreignNode)", ""))
            if tactic:
                current["available_tactics"].append(
                    {
                        "tactic_id": int(tactic.group(1)),
                        "cost_ms": float(tactic.group(2)),
                        "relative_cost": float(tactic.group(3)),
                        "prediction_correlation": float(tactic.group(4)),
                        "kernel_name": tactic.group(5).strip(),
                        "tactic_hash": tactic.group(6),
                        "kernel_evidence": classify_tactic_kernel(tactic.group(5).strip()),
                    }
                )
                continue
        selected = _SELECTED.search(line)
        if selected:
            current["selected_tactic"] = selected.group(1)
            in_table = False
    return records


def classify_tactic_kernel(kernel_name: str) -> dict[str, Any]:
    kernel = str(kernel_name)
    lower = kernel.lower()
    patterns = (
        ("f16f16_f16f32_f32", "F16A32O32"),
        ("f16f16_f16f32_f16", "F16A32O16"),
        ("f16f16_f16f16_f16", "F16A16O16"),
        ("f16f16_f16f16_f32", "F16A16O32"),
        ("f32f32_f32f32_f32", "F32A32O32"),
        ("bf16bf16_bf16f32_bf16", "BF16A32OBF16"),
        ("bf16bf16_bf16f32_f32", "BF16A32O32"),
    )
    for marker, phenotype in patterns:
        if marker in lower:
            return {
                "phenotype": phenotype,
                "compute_phenotype": phenotype.split("O", 1)[0],
                "evidence_level": "LEVEL_A_DIRECT",
                "accumulator_precision": "FP32" if "A32" in phenotype else "FP16",
                "output_precision": (
                    "FP32" if phenotype.endswith("O32") else "FP16"
                ),
                "kernel_name": kernel,
            }
    return {
        "phenotype": "UNKNOWN_ACCUM",
        "compute_phenotype": "UNKNOWN_ACCUM",
        "evidence_level": "LEVEL_C_UNKNOWN",
        "accumulator_precision": "unknown",
        "output_precision": "unknown",
        "kernel_name": kernel,
    }


def realize_output_phenotype(kernel_evidence: dict[str, Any], output_precision: str) -> str:
    """Combine kernel compute evidence with the actual Inspector layer output dtype."""

    compute = str(kernel_evidence.get("compute_phenotype", "UNKNOWN_ACCUM"))
    output = str(output_precision).upper()
    if compute == "F16A32":
        return "F16A32O32" if output == "FP32" else "F16A32O16" if output == "FP16" else "F16A32O_UNKNOWN"
    if compute == "F16A16":
        return "F16A16O16" if output == "FP16" else "F16A16O32" if output == "FP32" else "F16A16O_UNKNOWN"
    return str(kernel_evidence.get("phenotype", "UNKNOWN_ACCUM"))


def classify_numerical_boundary(
    *, finite: bool, reference_norm: float, output_zero_ratio: float
) -> str:
    if not finite:
        return "NUMERICAL_UNSAFE_NONFINITE"
    if float(reference_norm) > 1.0e-8 and float(output_zero_ratio) >= 0.999:
        return "NUMERICAL_UNSAFE_OUTPUT_UNDERFLOW"
    return "NUMERICAL_SAFE"


def summarize_tactic_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        for tactic in record.get("available_tactics", []):
            evidence = tactic["kernel_evidence"]
            rows.append(
                {
                    "op": record["op"],
                    "key": record["key"],
                    "selected_tactic": record.get("selected_tactic"),
                    **{key: value for key, value in tactic.items() if key != "kernel_evidence"},
                    **evidence,
                }
            )
    return rows


def select_f16a32_tactic(record: dict[str, Any]) -> dict[str, Any] | None:
    """Select an explicitly encoded F16A32 kernel, preferring O32 output."""

    candidates = [
        tactic
        for tactic in record.get("available_tactics", [])
        if tactic.get("kernel_evidence", {}).get("phenotype") in {"F16A32O32", "F16A32O16"}
    ]
    candidates.sort(
        key=lambda tactic: (
            tactic["kernel_evidence"]["phenotype"] != "F16A32O32",
            float(tactic.get("cost_ms", float("inf"))),
            str(tactic.get("tactic_hash", "")),
        )
    )
    return candidates[0] if candidates else None


__all__ = [
    "classify_tactic_kernel",
    "classify_numerical_boundary",
    "parse_editable_timing_log",
    "realize_output_phenotype",
    "select_f16a32_tactic",
    "summarize_tactic_records",
]
