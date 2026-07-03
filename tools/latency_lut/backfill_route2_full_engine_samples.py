from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from opencood.tools.compression.latency_lut.latency_proxy import LatencyProxy
from opencood.tools.compression.latency_lut.lut_database import LatencyLUTDatabase
from tools.latency_lut.build_full_engine_calibration_samples import (
    _select_unit_records,
    _stable_hash,
    _success_records,
    _unit_from_record,
)


DEFAULT_CANDIDATES = [
    "outputs/latency_lut/full_engine_candidates/baseline_like_fp16_shrink_int8_head_fp16_route2.result.json",
    "outputs/latency_lut/full_engine_candidates/baseline_like_fp16_head_fp32_shrink_int8_route2.result.json",
]


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _candidate_overrides(result: dict[str, Any]) -> dict[str, str]:
    precision = result.get("requested_precision_profile") or result.get("precision_config") or {}
    overrides = precision.get("overrides") if isinstance(precision, dict) else {}
    return {str(key): str(value).upper() for key, value in dict(overrides or {}).items()}


def _proxy_candidate(result: dict[str, Any], lut_path: str | Path) -> dict[str, Any]:
    records = _success_records(lut_path)
    overrides = _candidate_overrides(result)
    selected = _select_unit_records(
        records,
        default_precision=str((result.get("requested_precision_profile") or {}).get("default", "FP16")),
        keep_ratio=1.0,
        precision_overrides=overrides,
    )
    units = [_unit_from_record(record) for record in selected]
    precision_config = {"default": str((result.get("requested_precision_profile") or {}).get("default", "FP16"))}
    precision_config.update({unit["unit_id"]: unit["precision"] for unit in units})
    channel_config = {
        unit["unit_id"]: {"C_in": unit["C_in"], "C_mid": unit["C_mid"], "C_out": unit["C_out"]}
        for unit in units
    }
    return {
        "candidate_id": result["candidate_id"],
        "deploy_mode": result.get("deploy_mode", "single_engine_maxK"),
        "fixed_K": int(result.get("fixed_K", 29696)),
        "units": units,
        "precision_config": precision_config,
        "channel_config": channel_config,
        "channel_config_hash": _stable_hash(channel_config),
        "quant_config_hash": _stable_hash(precision_config),
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _sample_from_result(result: dict[str, Any], estimate: Any, proxy_candidate: dict[str, Any]) -> tuple[dict[str, Any] | None, list[str]]:
    reasons: list[str] = []
    payload = estimate.to_dict()
    qdq_report = _load_json(result["qdq_report"]) if result.get("qdq_report") and Path(result["qdq_report"]).is_file() else {}
    precision_failures = result.get("precision_verification_failures") or []
    if result.get("T_real_p50") is None:
        reasons.append("T_real_p50_missing")
    if result.get("T_real_p90") is None:
        reasons.append("T_real_p90_missing")
    if estimate.latency_lut_raw_ms is None or float(estimate.latency_lut_raw_ms) <= 0.0:
        reasons.append("T_lut_raw_missing_or_non_positive")
    if payload.get("missing_keys"):
        reasons.append("missing_keys_non_empty")
    if payload.get("unavailable_keys"):
        reasons.append("unavailable_keys_non_empty")
    if payload.get("unsupported_precision_regions"):
        reasons.append("unsupported_precision_regions_non_empty")
    if precision_failures:
        reasons.append("precision_verification_failures_non_empty")
    if reasons:
        return None, reasons
    sample = {
        "candidate_id": result["candidate_id"],
        "deploy_mode": result.get("deploy_mode", "single_engine_maxK"),
        "fixed_K": int(result.get("fixed_K", 29696)),
        "mixed_precision_backend": "strongly_typed_route2",
        "graph_explicit_precision": True,
        "uses_qdq": bool(result.get("uses_qdq")),
        "requested_precision_profile": result.get("requested_precision_profile") or {},
        "resolved_precision_profile": result.get("resolved_precision_profile") or result.get("requested_precision_profile") or {},
        "channel_config": proxy_candidate["channel_config"],
        "precision_config": proxy_candidate["precision_config"],
        "channel_config_hash": proxy_candidate["channel_config_hash"],
        "quant_config_hash": proxy_candidate["quant_config_hash"],
        "T_lut_raw": float(estimate.latency_lut_raw_ms),
        "T_lut_pred": float(estimate.latency_lut_raw_ms),
        "predicted_lut_ms": float(estimate.latency_lut_raw_ms),
        "T_lut_pred_before_calibration": float(estimate.latency_lut_raw_ms),
        "latency_proxy_ms": float(estimate.latency_ms),
        "T_real_p50": float(result["T_real_p50"]),
        "T_real_p90": float(result["T_real_p90"]),
        "T_real_p95": result.get("T_real_p95"),
        "T_real_mean": result.get("T_real_mean"),
        "real_engine_p50_ms": float(result["T_real_p50"]),
        "real_engine_p90_ms": float(result["T_real_p90"]),
        "real_engine_p95_ms": result.get("T_real_p95"),
        "real_engine_mean_ms": result.get("T_real_mean"),
        "mAP": result.get("mAP"),
        "AP_0.70": result.get("AP_0.70"),
        "AP_0_70": result.get("AP_0.70"),
        "num_val_frames": result.get("num_val_frames"),
        "observed_fp32_layers": int(result.get("observed_fp32_layers") or 0),
        "observed_fp16_layers": int(result.get("observed_fp16_layers") or 0),
        "observed_int8_layers": int(result.get("observed_int8_layers") or 0),
        "num_cast_inserted": int(qdq_report.get("num_cast_inserted") or result.get("num_cast_inserted") or 0),
        "num_qdq_nodes_inserted": int(qdq_report.get("num_qdq_nodes_inserted") or result.get("num_qdq_nodes_inserted") or 0),
        "num_precision_switches": int(estimate.calibration_features.get("num_precision_switch", 0)),
        "num_add_fixed": int(qdq_report.get("num_add_fixed") or result.get("num_add_fixed") or 0),
        "num_concat_fixed": int(qdq_report.get("num_concat_fixed") or result.get("num_concat_fixed") or 0),
        "num_grid_sample_fixed": int(qdq_report.get("num_grid_sample_fixed") or result.get("num_grid_sample_fixed") or 0),
        "missing_keys": payload.get("missing_keys", []),
        "unavailable_keys": payload.get("unavailable_keys", []),
        "unsupported_precision_regions": payload.get("unsupported_precision_regions", []),
        "precision_verification_failures": precision_failures,
        "fallback_layers": result.get("fallback_layers") or [],
        "pruning_keep_ratio": None,
        "features": {
            **estimate.calibration_features,
            "num_cast_inserted": int(qdq_report.get("num_cast_inserted") or 0),
            "num_qdq_nodes_inserted": int(qdq_report.get("num_qdq_nodes_inserted") or 0),
            "num_fp32_layers": int(result.get("observed_fp32_layers") or 0),
            "num_fp16_layers": int(result.get("observed_fp16_layers") or 0),
            "num_int8_layers": int(result.get("observed_int8_layers") or 0),
            "num_add_fixed": int(qdq_report.get("num_add_fixed") or 0),
            "num_concat_fixed": int(qdq_report.get("num_concat_fixed") or 0),
            "num_grid_sample_fixed": int(qdq_report.get("num_grid_sample_fixed") or 0),
            "num_auto_promoted_regions": 0,
            "pruning_keep_ratio": None,
        },
        "engine_hash": result.get("engine_hash"),
        "onnx_hash": result.get("onnx_hash"),
        "engine_path": result.get("engine_path"),
        "onnx_path": result.get("onnx_path"),
        "status": "success",
    }
    return sample, []


def backfill(args: argparse.Namespace) -> dict[str, Any]:
    db = LatencyLUTDatabase.from_jsonl(args.lut)
    proxy = LatencyProxy(db, kappa=0.0)
    output = Path(args.output)
    existing = _read_jsonl(output)
    by_id = {str(row.get("candidate_id")): row for row in existing}
    appended: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for result_path in [Path(p) for p in args.results]:
        result = _load_json(result_path)
        proxy_candidate = _proxy_candidate(result, args.lut)
        estimate = proxy.estimate(proxy_candidate)
        sample, reasons = _sample_from_result(result, estimate, proxy_candidate)
        if sample is None:
            skipped.append(
                {
                    "candidate_id": result.get("candidate_id"),
                    "result_path": str(result_path),
                    "reasons": reasons,
                    "T_lut_raw": estimate.latency_lut_raw_ms,
                    "missing_keys": estimate.to_dict().get("missing_keys", []),
                    "unavailable_keys": estimate.to_dict().get("unavailable_keys", []),
                }
            )
            continue
        by_id[sample["candidate_id"]] = sample
        appended.append(
            {
                "candidate_id": sample["candidate_id"],
                "T_lut_raw": sample["T_lut_raw"],
                "T_real_p50": sample["T_real_p50"],
                "absolute_error_ms": abs(sample["T_lut_raw"] - sample["T_real_p50"]),
                "relative_error": abs(sample["T_lut_raw"] - sample["T_real_p50"]) / sample["T_real_p50"],
            }
        )
    _write_jsonl(output, list(by_id.values()))
    report = {
        "status": "success" if not skipped else "partial_success",
        "lut": str(args.lut),
        "output": str(output),
        "num_appended_or_replaced": len(appended),
        "num_skipped": len(skipped),
        "appended_or_replaced": appended,
        "skipped": skipped,
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Full Engine Sample Backfill Report", ""]
    lines.append(f"- status: {report['status']}")
    lines.append(f"- samples appended/replaced: {len(appended)}")
    lines.append(f"- samples skipped: {len(skipped)}")
    lines.append("")
    for row in appended:
        lines.append(
            f"- {row['candidate_id']}: T_lut_raw={row['T_lut_raw']:.6f} ms, "
            f"T_real_p50={row['T_real_p50']:.6f} ms, abs_error={row['absolute_error_ms']:.6f} ms, "
            f"rel_error={row['relative_error']:.4f}"
        )
    if skipped:
        lines.append("")
        lines.append("## Skipped")
        for row in skipped:
            lines.append(f"- {row['candidate_id']}: {', '.join(row['reasons'])}")
    Path(args.report).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lut", default="outputs/latency_lut/lut_records.jsonl")
    parser.add_argument("--output", default="outputs/latency_lut/full_engine_samples.jsonl")
    parser.add_argument("--report", default="outputs/latency_lut/full_engine_sample_backfill_report.md")
    parser.add_argument("--results", nargs="*", default=DEFAULT_CANDIDATES)
    return parser.parse_args()


def main() -> int:
    report = backfill(parse_args())
    return 0 if report.get("num_skipped", 0) == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
