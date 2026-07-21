"""Evidence writers for the bounded CoBEVT SmoothQuant deployment audit."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def summarize_precision_inventory(
    profile_id: str, rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Normalize requested/realized evidence without inferring from profile names."""

    normalized: list[dict[str, Any]] = []
    for source in rows:
        role = str(source.get("role", ""))
        requested = str(source.get("requested_dtype", "UNKNOWN")).upper()
        tactic = str(source.get("tactic_precision", "UNKNOWN")).upper()
        inputs = [str(value).upper() for value in source.get("realized_input_dtypes", [])]
        outputs = [str(value).upper() for value in source.get("realized_output_dtypes", [])]
        realized = tactic if tactic in {"INT8", "FP16", "FP32"} else "UNKNOWN"
        accumulator = "UNKNOWN"
        tactics = [str(value) for value in source.get("tactic_names", [])]
        if role == "qk_matmul" and realized == "FP32" and any(
            "f32f32" in value.lower() and "f32" in value.lower() for value in tactics
        ):
            accumulator = "FP32"
        normalized.append(
            {
                "profile_id": str(profile_id),
                "block_id": str(source.get("block_id", "")),
                "role": role,
                "requested_precision": requested,
                "realized_precision": realized,
                "realized_input_dtypes": "|".join(inputs),
                "realized_output_dtypes": "|".join(outputs),
                "qk_accumulator_precision": accumulator,
                "requested_realized_match": requested == realized,
                "tactic": "|".join(tactics),
                "fusion_kind": "primitive_gemm" if role in {"qk_matmul", "av_matmul"} else "weighted_gemm" if role.endswith("projection") else "other",
                "complete_fused_mha": False,
                "classification_conflict": False,
            }
        )
    return normalized


def build_profile_realization_row(
    profile_id: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    status: str,
    engine_sha256: str,
) -> dict[str, Any]:
    """Gate only explicitly requested INT8 projections plus the FP32 QK core."""

    selected = [
        row
        for row in rows
        if str(row.get("role", "")).endswith("projection")
        and str(row.get("requested_precision", "")).upper() == "INT8"
    ]
    qk = [row for row in rows if row.get("role") == "qk_matmul"]
    return {
        "profile_id": str(profile_id),
        "status": str(status),
        "engine_sha256": str(engine_sha256),
        "selected_projection_count": len(selected),
        "realized_int8_projection_count": sum(
            row.get("realized_precision") == "INT8" for row in selected
        ),
        "qk_fp32_count": sum(
            row.get("realized_precision") == "FP32"
            and row.get("qk_accumulator_precision") == "FP32"
            for row in qk
        ),
        "qk_count": len(qk),
        "requested_realized_match": bool(selected)
        and bool(qk)
        and all(row.get("requested_realized_match") is True for row in selected + qk),
        "complete_fused_mha": False,
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_smoothquant_precision_evidence(
    profile_directories: Mapping[str, Path], destination: Path
) -> dict[str, Any]:
    """Write the compact precision, Q/DQ, tactic and build evidence tables."""

    import onnx

    precision_rows: list[dict[str, Any]] = []
    qdq_rows: list[dict[str, Any]] = []
    smoothing_rows: list[dict[str, Any]] = []
    build_rows: list[dict[str, Any]] = []
    for profile_id, directory in sorted(profile_directories.items()):
        inventory_path = directory / "attention_precision_inventory.json"
        build_path = directory / "build_report.json"
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        build = json.loads(build_path.read_text(encoding="utf-8"))
        normalized = summarize_precision_inventory(profile_id, inventory)
        precision_rows.extend(normalized)
        engine_text = json.dumps(
            json.loads((directory / "engine_layer_info.json").read_text(encoding="utf-8"))
        )
        model = onnx.load(str(directory / "typed.onnx"), load_external_data=False)
        for node in model.graph.node:
            if node.op_type in {"QuantizeLinear", "DequantizeLinear"}:
                qdq_rows.append(
                    {
                        "profile_id": profile_id,
                        "node_name": str(node.name),
                        "op_type": str(node.op_type),
                        "inputs": "|".join(str(value) for value in node.input),
                        "outputs": "|".join(str(value) for value in node.output),
                    }
                )
            names = tuple(str(value) for value in (*node.input, *node.output))
            if "pre_quant_scale" in str(node.name) or any(
                "pre_quant_scale" in value for value in names
            ):
                smoothing_rows.append(
                    {
                        "profile_id": profile_id,
                        "node_name": str(node.name),
                        "op_type": str(node.op_type),
                        "runtime_layer_detected": str(node.name) in engine_text,
                    }
                )
        build_rows.append(
            build_profile_realization_row(
                profile_id,
                normalized,
                status=str(build.get("status", "unknown")),
                engine_sha256=str(build.get("engine_sha256", "")),
            )
        )
    destination.mkdir(parents=True, exist_ok=True)
    _write_csv(destination / "requested_realized_precision.csv", precision_rows)
    _write_json(destination / "requested_realized_precision.json", precision_rows)
    _write_csv(destination / "fusion_tactic_inventory.csv", precision_rows)
    _write_csv(destination / "quantization_boundary_inventory.csv", precision_rows)
    _write_csv(destination / "qdq_inventory.csv", qdq_rows)
    _write_csv(destination / "smoothing_fold_inventory.csv", smoothing_rows)
    _write_csv(destination / "build_matrix.csv", build_rows)
    _write_json(destination / "build_matrix.json", build_rows)
    return {
        "profiles": len(build_rows),
        "precision_rows": len(precision_rows),
        "qdq_rows": len(qdq_rows),
        "smoothing_rows": len(smoothing_rows),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", required=True)
    parser.add_argument("--profile", action="append", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    profiles: dict[str, Path] = {}
    for value in args.profile:
        profile_id, separator, path = str(value).partition("=")
        if not separator or not profile_id or not path:
            raise ValueError(f"invalid_profile_directory:{value}")
        profiles[profile_id] = Path(path)
    result = write_smoothquant_precision_evidence(profiles, Path(args.destination))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
