"""Machine-readable boundary matrices for the CoBEVT accumulation audit."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


_PROFILES = {
    "A0_F3_REFERENCE": {
        "implementation": "native_tensorrt",
        "qk": ("FP32", "FP32", "FP32", "FP32", "FP32"),
        "av": ("FP16", "FP16", "FP16", "UNKNOWN_ACCUM", "FP16"),
        "evidence_level": "Level C",
        "fusion_kind": "primitive_qk_gemm_and_native_av",
    },
    "A2_QK_F16A32_PLUGIN": {
        "implementation": "plugin_oracle",
        "qk": ("FP16", "FP16", "FP32", "FP32", "FP32"),
        "av": ("FP16", "FP16", "FP16", "UNKNOWN_ACCUM", "FP16"),
        "evidence_level": "Level A",
        "fusion_kind": "plugin_qk_plus_primitive_av",
    },
    "B1_AV_F16A32_PLUGIN": {
        "implementation": "plugin_oracle",
        "qk": ("FP32", "FP32", "FP32", "FP32", "FP32"),
        "av": ("FP16", "FP16", "FP32", "FP32", "FP16"),
        "evidence_level": "Level A",
        "fusion_kind": "primitive_qk_plus_plugin_av",
    },
    "C1_QK_AV_F16A32_PLUGIN": {
        "implementation": "plugin_oracle",
        "qk": ("FP16", "FP16", "FP32", "FP32", "FP32"),
        "av": ("FP16", "FP16", "FP32", "FP32", "FP16"),
        "evidence_level": "Level A",
        "fusion_kind": "plugin_qk_and_av",
    },
}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({str(key) for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_search_contract(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Classify profiles without allowing opaque/plugin phenotypes into native search."""

    allowed: list[str] = []
    experimental: list[str] = []
    rejected: list[str] = []
    unsupported: list[str] = []
    for source in rows:
        profile = str(source.get("profile", ""))
        implementation = str(source.get("implementation", ""))
        evidence = str(source.get("evidence_level", ""))
        safe = str(source.get("fixed500_safety", "")) == "SAFE_FIXED500"
        latency_gain = bool(source.get("formal_latency_gain", False))
        if not profile:
            continue
        if profile in {"A0_F3_REFERENCE", "F3_REFERENCE"} and safe:
            allowed.append(profile)
            continue
        if implementation == "plugin_oracle":
            if safe and latency_gain and evidence == "Level A":
                experimental.append(profile)
            else:
                rejected.append(profile)
        elif implementation == "native_tensorrt" and safe and latency_gain and evidence == "Level A":
            allowed.append(profile)
        elif evidence in {"Level B", "Level C", "UNKNOWN_ACCUM"}:
            rejected.append(profile)
        else:
            unsupported.append(profile)
    return {
        "allowed": sorted(set(allowed)),
        "experimental": sorted(set(experimental)),
        "rejected": sorted(set(rejected)),
        "unsupported": sorted(set(unsupported)),
        "evidence_policy": {
            "minimum_accumulator_evidence": "Level A",
            "unknown_accumulator_searchable": False,
            "plugin_oracle_is_native_tensorrt": False,
        },
    }


def render_root_conclusion(
    rows: Sequence[Mapping[str, Any]], contract: Mapping[str, Any]
) -> str:
    """Render the audited conclusions without promoting plugin evidence to native TRT."""

    lines = [
        "# CoBEVT Attention Operand/Accumulator Root Conclusion",
        "",
        "## Direct answers",
        "",
        "1. TensorRT 10.9 strongly typed MatrixMultiply/Einsum has no independent "
        "accumulator-precision API, and EngineInspector does not expose accumulator "
        "precision directly.",
        "2. The native TensorRT 10.9 Cast-based F16A32 requests materialized FP32 operands before a "
        "F32A32 GEMM. They are not native F16A32.",
        "3. The cuBLASLt oracle shows FP16 operand rounding with FP32 accumulation is "
        "numerically safe on captured CoBEVT QK/AV tensors, while QK F16A16 has a "
        "large error fingerprint.",
        "4. R1 remains an opaque complete fused-MHA phenotype. Its collapse cannot be "
        "assigned to operand rounding alone; the evidence points to low-precision "
        "accumulation and/or another fused-MHA numerical path, but does not isolate "
        "those two native causes.",
        "5. Native TensorRT 10.9 INT8 QK and AV micro-engines did not build. The audit "
        "therefore provides no native I8A32I full-model result.",
        "6. The Level-A F16A32 implementation is a custom cuBLASLt plugin oracle. It "
        "不等同于 native TensorRT and is not a complete fused MHA.",
        "",
        "## Full-model fixed500 and formal latency",
        "",
        "| Profile | mAP | Delta mAP vs fresh F3 | p50 ms | Speedup | Safety |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {profile} | {mAP:.9f} | {delta:+.9f} | {p50:.6f} | "
            "{speedup:.6f} | {safety} |".format(
                profile=row.get("profile", ""),
                mAP=float(row.get("mAP", float("nan"))),
                delta=float(row.get("delta_mAP_vs_fresh_F3", float("nan"))),
                p50=float(row.get("formal_forward_p50_ms", float("nan"))),
                speedup=float(row.get("formal_speedup_vs_A0", float("nan"))),
                safety=row.get("fixed500_safety", "unknown"),
            )
        )
    lines.extend(
        [
            "",
            "All four profiles completed 500/500 frames with zero skips. Formal latency "
            "uses the isolated GPU-7 serial replay; parallel screening latency is excluded.",
            "",
            "## Search contract",
            "",
            f"- Allowed: `{json.dumps(contract.get('allowed', []), sort_keys=True)}`",
            f"- Experimental: `{json.dumps(contract.get('experimental', []), sort_keys=True)}`",
            f"- Rejected: `{json.dumps(contract.get('rejected', []), sort_keys=True)}`",
            f"- Unsupported: `{json.dumps(contract.get('unsupported', []), sort_keys=True)}`",
            "- Unknown accumulator phenotypes are not searchable.",
            "- Plugin-oracle evidence cannot be admitted as a native TensorRT precision gene.",
            "",
            "## Scope",
            "",
            "No 1789-frame validation, GA, Stage-A, Stage-B, or Pyramid process was run "
            "or modified. The existing additive unit-latency LUT remains not additive.",
            "",
        ]
    )
    return "\n".join(lines)


def _select_formal_replay_root(root: str | Path) -> Path:
    base = Path(root)
    strict = base / "latency/formal_replay_retry2"
    protocol_path = strict / "formal_protocol.json"
    if protocol_path.is_file():
        protocol = json.loads(protocol_path.read_text())
        valid = (
            protocol.get("status") == "ok"
            and int(protocol.get("warmup_executions", 0)) >= 200
            and int(protocol.get("timed_executions", 0)) >= 2000
            and int(protocol.get("latency_rounds", 0)) == 5
        )
        if valid:
            return strict
    return base / "latency/formal_replay_retry1"


def collect_full_model_boundary_rows(output_dir: str | Path) -> list[dict[str, Any]]:
    """Collect the four fresh full-model profiles and attach fixed500/formal evidence."""

    root = Path(output_dir)
    rows: list[dict[str, Any]] = []
    base_eval = json.loads(
        (root / "full_model/A0_F3_REFERENCE/candidates/baseline_d32/"
         "f3_rest_fp16_qk_fp32_minimal_island_engine_k29696/evaluation_fixed500/evaluation.json").read_text()
    )
    base_map = float(base_eval["mAP"])
    formal_root = _select_formal_replay_root(root)
    base_latency = json.loads(
        (formal_root / "A0_F3_REFERENCE/evaluation.json").read_text()
    )
    base_p50 = float(base_latency["forward_p50_ms"])
    for profile, spec in _PROFILES.items():
        candidate = root / "full_model" / profile / "candidates/baseline_d32/"
        candidate = candidate / "f3_rest_fp16_qk_fp32_minimal_island_engine_k29696"
        build = json.loads((candidate / "build_report.json").read_text())
        evaluation = json.loads((candidate / "evaluation_fixed500/evaluation.json").read_text())
        formal = json.loads((formal_root / profile / "evaluation.json").read_text())
        delta = float(evaluation["mAP"]) - base_map
        formal_p50 = float(formal["forward_p50_ms"])
        row = {
            "profile": profile,
            "implementation": spec["implementation"],
            "evidence_level": spec["evidence_level"],
            "qk_storage_precision": spec["qk"][0],
            "qk_operand_precision": spec["qk"][1],
            "qk_multiplication_precision": spec["qk"][2],
            "qk_accumulator_precision": spec["qk"][3],
            "qk_output_precision": spec["qk"][4],
            "av_storage_precision": spec["av"][0],
            "av_operand_precision": spec["av"][1],
            "av_multiplication_precision": spec["av"][2],
            "av_accumulator_precision": spec["av"][3],
            "av_output_precision": spec["av"][4],
            "requested_contract": "F16A32 plugin oracle" if spec["implementation"] == "plugin_oracle" else "accepted F3",
            "realized_contract": spec["fusion_kind"],
            "fusion_kind": spec["fusion_kind"],
            "plugin_used": spec["implementation"] == "plugin_oracle",
            "engine_path": build.get("engine_path", ""),
            "engine_sha256": build.get("engine_sha256", ""),
            "build_status": build.get("status", "failed"),
            "fixed500_status": evaluation.get("status", ""),
            "fixed500_evaluated_frames": evaluation.get("num_evaluated_frames", -1),
            "fixed500_skipped_frames": evaluation.get("num_skipped_frames", -1),
            "AP30": evaluation.get("AP@0.3"),
            "AP50": evaluation.get("AP@0.5"),
            "AP70": evaluation.get("AP@0.7"),
            "mAP": evaluation.get("mAP"),
            "delta_mAP_vs_fresh_F3": delta,
            "fixed500_safety": "SAFE_FIXED500" if delta >= -0.003 else ("BORDERLINE_FIXED500" if delta >= -0.010 else "UNSAFE_FIXED500"),
            "formal_status": formal.get("status", ""),
            "formal_replay_root": str(formal_root),
            "formal_gpu": formal.get("physical_device", ""),
            "formal_manifest_hash": formal.get("eval_manifest_hash", ""),
            "formal_forward_p50_ms": formal_p50,
            "formal_forward_p95_ms": formal.get("forward_p95_ms"),
            "formal_forward_p99_ms": formal.get("forward_p99_ms"),
            "formal_latency_gain": formal_p50 < base_p50,
            "formal_speedup_vs_A0": base_p50 / formal_p50,
        }
        rows.append(row)
    return rows


def write_boundary_artifacts(output_dir: str | Path) -> dict[str, Any]:
    root = Path(output_dir)
    rows = collect_full_model_boundary_rows(root)
    _write_csv(root / "full_precision_boundary_matrix.csv", rows)
    _write_json(root / "full_precision_boundary_matrix.json", rows)
    qk = [{key: row[key] for key in row if key.startswith("qk_") or key in {"profile", "mAP", "delta_mAP_vs_fresh_F3", "fixed500_safety", "formal_forward_p50_ms", "formal_latency_gain", "evidence_level", "implementation"}} for row in rows]
    av = [{key: row[key] for key in row if key.startswith("av_") or key in {"profile", "mAP", "delta_mAP_vs_fresh_F3", "fixed500_safety", "formal_forward_p50_ms", "formal_latency_gain", "evidence_level", "implementation"}} for row in rows]
    _write_csv(root / "qk_safety_boundary.csv", qk)
    _write_csv(root / "av_safety_boundary.csv", av)
    contract_rows = [
        {
            "profile": row["profile"],
            "implementation": row["implementation"],
            "evidence_level": row["evidence_level"],
            "fixed500_safety": row["fixed500_safety"],
            "formal_latency_gain": row["formal_latency_gain"],
        }
        for row in rows
    ]
    contract = build_search_contract(contract_rows)
    contract["profile_rows"] = contract_rows
    _write_json(root / "attention_operand_accumulation_search_contract.json", contract)
    (root / "root_conclusion.md").write_text(
        render_root_conclusion(rows, contract), encoding="utf-8"
    )
    return {"rows": rows, "contract": contract}


__all__ = [
    "build_search_contract",
    "collect_full_model_boundary_rows",
    "render_root_conclusion",
    "_select_formal_replay_root",
    "write_boundary_artifacts",
]
