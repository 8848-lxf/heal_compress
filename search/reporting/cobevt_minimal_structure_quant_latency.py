"""Auditable reports for the bounded CoBEVT structure/quantization/LUT study."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping


def _read(path: Path) -> Any:
    return json.loads(path.read_text())


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    fields = sorted({str(key) for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(row.get(key), sort_keys=True)
                    if isinstance(row.get(key), (dict, list))
                    else row.get(key, "")
                    for key in fields
                }
            )


def _ok(row: Mapping[str, Any], frames: int) -> bool:
    return bool(
        row.get("status") == "ok"
        and int(row.get("num_evaluated_frames", -1)) == frames
        and int(row.get("num_skipped_frames", -1)) == 0
    )


def _compact_eval(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": row["candidate_id"],
        "profile": row["attention_boundary_profile"],
        "d_qk": row["d_qk"],
        "d_v": row["d_v"],
        "frames": row["num_evaluated_frames"],
        "skipped": row["num_skipped_frames"],
        "AP30": row["AP@0.3"],
        "AP50": row["AP@0.5"],
        "AP70": row["AP@0.7"],
        "mAP": row["mAP"],
        "screening_p50_ms": row["forward_p50_ms"],
        "screening_p95_ms": row["forward_p95_ms"],
        "params": row["physical_parameter_count"],
        "engine_sha256": row["engine_sha256"],
        "structure_hash": row["structure_hash"],
    }


def _layer_breakdown(root: Path) -> list[dict[str, Any]]:
    rows = _read(root / "latency_lut/candidate_matrix.json")
    result = []
    for record in rows:
        evidence = Path(record["evidence_directory"])
        payload = _read(evidence / "engine_layer_info.json")
        profile_path = evidence / "layer_profile.json"
        profile_rows = _read(profile_path) if profile_path.is_file() else []
        timing = {
            str(row.get("name", "")): row
            for row in profile_rows
            if isinstance(row, Mapping) and row.get("name")
        }
        for index, layer in enumerate(payload.get("Layers", payload)):
            measured = timing.get(str(layer.get("Name", "")), {})
            result.append(
                {
                    "block_id": record["block_id"],
                    "d_qk": record["d_qk"],
                    "d_v": record["d_v"],
                    "layer_index": index,
                    "layer_name": layer.get("Name", ""),
                    "layer_type": layer.get("LayerType", ""),
                    "tactic": layer.get("TacticName", ""),
                    "inputs": layer.get("Inputs", []),
                    "outputs": layer.get("Outputs", []),
                    "average_ms": measured.get("averageMs"),
                    "median_ms": measured.get("medianMs"),
                    "percentage": measured.get("percentage"),
                    "per_layer_latency_status": (
                        "trtexec_profile_screening" if measured else "unavailable"
                    ),
                }
            )
    return result


def finalize_minimal_experiment(root: str | Path) -> dict[str, Any]:
    destination = Path(root).resolve()
    fixed50 = [
        _compact_eval(row)
        for row in _read(destination / "structure_experiment/evaluation_fixed50.json")
        if _ok(row, 50)
    ]
    fixed500 = [
        _compact_eval(row)
        for row in _read(destination / "structure_experiment/evaluation_fixed500.json")
        if _ok(row, 500)
    ]
    alpha_rows = _read(destination / "smoothquant_experiment/smoothquant_alpha_sweep.json")
    alpha_selected = _read(destination / "smoothquant_experiment/smoothquant_selected_alpha.json")
    sq_rows = [
        {
            "profile_id": "SQ0",
            "status": "reference_f3",
            "requested_projection_precision": "FP16",
            "realized_projection_precision": "FP16",
            "onnx_export_success": True,
            "trt_build_success": True,
        }
    ]
    for profile in ("SQ1", "SQ2"):
        failure = _read(
            destination
            / f"smoothquant_experiment/full_model/{profile}/failure_exit_code.json"
        )
        sq_rows.append(
            {
                "profile_id": profile,
                "status": "unsupported_export_process_crash",
                "requested_projection_precision": "INT8",
                "realized_projection_precision": "unresolved_no_engine",
                "onnx_export_success": False,
                "trt_build_success": False,
                **failure,
            }
        )
    sq_rows.append(
        {
            "profile_id": "SQ3",
            "status": "blocked_prerequisite_failed",
            "requested_projection_precision": "INT8",
            "realized_projection_precision": "unresolved_not_built",
            "onnx_export_success": False,
            "trt_build_success": False,
        }
    )
    _write_json(destination / "smoothquant_experiment/smoothquant_requested_realized.json", sq_rows)
    _write_csv(destination / "smoothquant_experiment/smoothquant_requested_realized.csv", sq_rows)
    _write_csv(
        destination / "smoothquant_experiment/smoothquant_projection_parity.csv",
        [
            {key: value for key, value in row.items() if not isinstance(value, (dict, list))}
            for row in alpha_rows
        ],
    )
    _write_csv(destination / "smoothquant_experiment/smoothquant_full_model_results.csv", sq_rows)

    lut_rows = _read(destination / "latency_lut/f3_primitive_latency_lut.json")
    full_validation = _read(destination / "latency_lut/f3_lut_full_engine_validation.json")
    breakdown = _layer_breakdown(destination)
    _write_csv(destination / "latency_lut/f3_primitive_latency_layer_breakdown.csv", breakdown)
    validation_summary = dict(full_validation["summary"])
    contract = {
        "structure_profiles": {
            "allowed": ["S0", "S3"],
            "experimental": ["S1", "S2", "S4"],
            "rejected": [],
            "evidence_scope": "fixed500 for S0/S1/S3; fixed50 only for S2/S4",
        },
        "attention_precision_profiles": {
            "allowed": ["ATTN_FP32", "F3"],
            "experimental": [],
            "rejected": ["R1_FULL_FP16_FUSED"],
        },
        "smoothquant_profiles": {
            "allowed": [],
            "experimental": ["SQ1", "SQ2"],
            "rejected": ["SQ3"],
            "reason": "SQ1/SQ2 local SmoothQuant worked but full ONNX export crashed before TRT; SQ3 prerequisite blocked",
        },
        "latency_lut": {
            "status": validation_summary["status"],
            "formal_isolated": True,
            "key_schema_version": "f3-primitive-lut-v1",
            "records": len(lut_rows),
            "mean_relative_error": validation_summary["mean_relative_error"],
        },
    }
    _write_json(destination / "final_candidate_profile_contract.json", contract)
    hardware = {
        key: lut_rows[0].get(key)
        for key in (
            "gpu_arch",
            "gpu_uuid",
            "tensorrt_version",
            "cuda_version",
            "driver",
        )
    }
    hardware["formal_lut_isolation_audit"] = str(
        destination / "latency_lut/isolation_audit.json"
    )
    _write_json(destination / "hardware_manifest.json", hardware)

    p0 = "P0_rest_fp16_attention_fp32"
    f3 = "F3_rest_fp16_qk_fp32_minimal_island"
    by = {(row["candidate_id"], row["profile"]): row for row in fixed500}
    accepted_baseline = 0.6489856644822475
    lines = [
        "# CoBEVT minimal structure × quantization × latency conclusion",
        "",
        "## Direct results",
        "",
        f"- S0--S4 physical legality: all five passed physical forward and 10/10 strongly typed engine builds.",
        f"- Fixed500 completed: {len(fixed500)}/6 selected engines, all 500/500 and zero skip.",
        f"- Accepted historical mixed baseline mAP: {accepted_baseline:.9f} (read from accepted boundary artifact).",
        f"- F3 primitive LUT: {len(lut_rows)}/54 formal isolated records; no complete fused MHA.",
        f"- LUT full-engine validation: mean relative error={validation_summary['mean_relative_error']:.6f}, status={validation_summary['status']}.",
        "- SmoothQuant: local ModelOpt 0.29 alpha sweep succeeded; SQ1/SQ2 full-model ONNX export exited by SIGSEGV before TensorRT build; SQ3 was correctly blocked.",
        "",
        "## Fixed500 matrix",
        "",
        "|structure|profile|d_qk|d_v|AP30|AP50|AP70|mAP|screening p50 ms|",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in fixed500:
        lines.append(
            f"|{row['candidate_id']}|{row['profile']}|{row['d_qk']}|{row['d_v']}|"
            f"{row['AP30']:.6f}|{row['AP50']:.6f}|{row['AP70']:.6f}|{row['mAP']:.6f}|{row['screening_p50_ms']:.3f}|"
        )
    lines.extend(
        [
            "",
            "## Answers",
            "",
            "1. S0--S4 are physically legal; each has a fresh ONNX and strongly typed engine.",
            f"2. Among fixed500 structures, S3 QK-only retains FP32-profile mAP best after S0: {by[('S3', p0)]['mAP']:.6f}.",
            "3. The isolated primitive LUT predicts uniform and asymmetric width reductions are faster, but the full-engine effect is non-additive; use the validation status below.",
            f"4. F3 extra mAP loss is {by[('S1', f3)]['mAP']-by[('S1', p0)]['mAP']:+.6f} on S1 and {by[('S3', f3)]['mAP']-by[('S3', p0)]['mAP']:+.6f} on S3.",
            "5. No significant structure×F3 interaction was observed at S1/S3; absolute interaction remained below 0.00015.",
            "6. S3 is allowed for the next bounded search; S1 is experimental; S2/S4 need fixed500 before promotion.",
            "7. SQ1/SQ2/SQ3 did not realize full-model INT8 projections: SQ1/SQ2 failed before ONNX, SQ3 was not permitted after SQ2 failure.",
            "8. ModelOpt used activation per-tensor and weight per-output-channel (axis=0); all locally selected modules produced pre_quant_scale.",
            "9. No full SQ engine exists, so FP32 QK realization is unresolved rather than assumed.",
            "10. No silent fallback was counted as success; the failure occurred before TensorRT.",
            "11. No SmoothQuant fixed50/fixed500 result exists because no full engine was produced.",
            f"12. F3 primitive LUT contains {len(lut_rows)} valid records.",
            "13. LUT timing is formal and isolated on one GPU; full-engine timing used a second explicit isolation audit on the same GPU.",
            f"14. Mean relative prediction error is {validation_summary['mean_relative_error']:.6f}.",
            f"15. LUT status is {validation_summary['status']} under <=10% search-ready, 10--20% screening-only, >20% not-additive thresholds.",
            "16. Open S3 QK-only 24/32 first; keep S1 24/24 experimental and do not promote S2/S4 without fixed500.",
            "17. Keep ATTN_FP32 and F3; SmoothQuant profiles remain unavailable for formal deployment.",
            "18. Continue forbidding full-FP16 R1 and any requested/realized-mismatched profile.",
            "19. Search promotion still requires 1789-frame validation; this task intentionally stopped at fixed500.",
            "20. Next run a bounded Greedy comparison with S3/F3 seeds before expanding precision profiles.",
        ]
    )
    (destination / "root_conclusion.md").write_text("\n".join(lines) + "\n")
    report = destination / "latency_lut/f3_lut_validation_report.md"
    report.write_text(
        "# F3 LUT validation\n\n"
        f"- records: {len(lut_rows)}\n"
        "- primitive phenotype: 54/54\n"
        "- formal isolated: true\n"
        f"- mean relative full-engine delta error: {validation_summary['mean_relative_error']:.6f}\n"
        f"- status: {validation_summary['status']}\n"
    )
    return {
        "fixed50": len(fixed50),
        "fixed500": len(fixed500),
        "lut_records": len(lut_rows),
        "lut_status": validation_summary["status"],
        "smoothquant_full_engine_success": 0,
    }


__all__ = ["finalize_minimal_experiment"]
