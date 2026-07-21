"""Build the fail-closed QK/AV operand and accumulator evidence matrix.

This phase deliberately does not infer an accumulator from an engine output
dtype.  It links every requested Q0--Q5/A0--A5 experiment to a real engine
inspection row where one exists and records unsupported native contracts as
unsupported instead of silently substituting a projection-only experiment.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


MODELS = ("lidar_cobevt", "lidar_v2xvit")


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    values = [dict(row) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({str(key) for row in values for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(values)


EXPERIMENTS = (
    # experiment, op, requested operand, requested accumulator, source section/profile
    ("Q0_F32A32", "qk_matmul", "FP32", "FP32", "baselines", "B1_TRT_ATTN_FP32"),
    ("Q1_F16_DEFAULT", "qk_matmul", "FP16", "unknown", "precision_sensitivity", "P2_QK_FP16"),
    ("Q2_F16A32", "qk_matmul", "FP16", "FP32", "", ""),
    ("Q3_BF16A32", "qk_matmul", "BF16", "FP32", "bf16", "H2_QKV_BF16_QK_BF16A32"),
    ("Q4_I8A32I", "qk_matmul", "INT8", "INT32", "", ""),
    ("Q5_FP8", "qk_matmul", "FP8", "unknown", "", ""),
    ("A0_F32A32", "av_matmul", "FP32", "FP32", "baselines", "B1_TRT_ATTN_FP32"),
    ("A1_F16_DEFAULT", "av_matmul", "FP16", "unknown", "precision_sensitivity", "P4_AV_FP16"),
    ("A2_F16A32", "av_matmul", "FP16", "FP32", "", ""),
    ("A3_BF16A32", "av_matmul", "BF16", "FP32", "", ""),
    ("A4_I8A32I", "av_matmul", "INT8", "INT32", "", ""),
    ("A5_FP8", "av_matmul", "FP8", "unknown", "", ""),
)


def _engine_rows(root: Path, model: str, section: str, profile: str, role: str) -> list[dict[str, str]]:
    return [
        row
        for row in _read_csv(root / section / model / profile / "requested_realized.csv")
        if row.get("role") == role
    ]


def _unsupported_reason(experiment: str) -> str:
    if experiment in {"Q2_F16A32", "A2_F16A32"}:
        return "TensorRT_10.9_network_contract_has_no_exact_separate_FP32_accumulator_control"
    if experiment in {"Q3_BF16A32", "A3_BF16A32"}:
        return "no_Level_A_BF16_operand_FP32_accumulator_realization"
    if experiment in {"Q4_I8A32I", "A4_I8A32I"}:
        return "native_INT8_attention_core_not_implemented_and_projection_DQ_is_not_native_INT8_matmul"
    return "native_FP8_attention_core_not_implemented_and_FP8_projection_DQ_is_not_native_FP8_matmul"


def build_matrix(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        for experiment, role, operand, accumulator, section, profile in EXPERIMENTS:
            evidence = _engine_rows(root, model, section, profile, role) if section else []
            result = _read_json(root / section / model / profile / "baseline_result.json") if section else {}
            if evidence:
                realized_operands = sorted({row.get("realized_precision", "unknown") for row in evidence})
                realized_accumulators = sorted({row.get("realized_accumulator", "unknown") for row in evidence})
                levels = sorted(
                    {
                        "A" if "accumulator_level_A" in row.get("evidence_source", "")
                        else "B" if "accumulator_level_B" in row.get("evidence_source", "")
                        else "C"
                        for row in evidence
                    }
                )
                conflicts = sorted({row.get("conflict", "") for row in evidence if row.get("conflict", "")})
                exact = (
                    not conflicts
                    and realized_operands == [operand]
                    and (
                        accumulator == "unknown"
                        or realized_accumulators == [accumulator]
                    )
                )
                level_a = bool(levels) and levels == ["A"]
                status = "realized_exact" if exact else "precision_conflict"
                reason = "" if exact else ";".join(conflicts) or "requested_realized_mismatch"
            else:
                realized_operands = []
                realized_accumulators = []
                levels = []
                conflicts = []
                exact = False
                level_a = False
                status = "unsupported"
                reason = _unsupported_reason(experiment)
            searchable = bool(exact and level_a and result.get("status") == "ok")
            rows.append(
                {
                    "model": model,
                    "experiment": experiment,
                    "operator": "QK" if role == "qk_matmul" else "AV",
                    "requested_operand": operand,
                    "requested_accumulator": accumulator,
                    "source_section": section,
                    "source_profile": profile,
                    "engine_status": result.get("status", "not_built"),
                    "engine_count": len(evidence),
                    "realized_operands": "|".join(realized_operands),
                    "realized_accumulators": "|".join(realized_accumulators),
                    "evidence_levels": "|".join(levels),
                    "requested_realized_exact": exact,
                    "searchable": searchable,
                    "status": status,
                    "reason": reason,
                    "projection_only_substitute_forbidden": experiment in {
                        "Q4_I8A32I", "Q5_FP8", "A4_I8A32I", "A5_FP8"
                    },
                }
            )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args(argv)
    root = Path(args.output_root).resolve()
    rows = build_matrix(root)
    destination = root / "accumulator"
    _write_csv(destination / "accumulator_matrix.csv", rows)
    _write_json(destination / "accumulator_matrix.json", rows)
    summary = {
        "schema_version": "h800-transformer-accumulator-matrix-v1",
        "rows": len(rows),
        "level_a_searchable": [
            f"{row['model']}:{row['experiment']}" for row in rows if row["searchable"]
        ],
        "unknown_not_searchable": [
            f"{row['model']}:{row['experiment']}" for row in rows
            if row["requested_realized_exact"] and not row["searchable"]
        ],
        "unsupported": [
            f"{row['model']}:{row['experiment']}" for row in rows if row["status"] == "unsupported"
        ],
        "output_dtype_is_accumulator_evidence": False,
        "projection_qdq_is_native_attention_core": False,
    }
    _write_json(destination / "accumulator_conclusion.json", summary)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
