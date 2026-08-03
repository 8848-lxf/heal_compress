from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from opencood.tools.compression.latency_lut.schema import DEPLOY_MODE, FIXED_K, LatencyLUTKey
from opencood.tools.compression.latency_lut.tensorRT_benchmark import SUCCESS_STATUSES, benchmark_key


PROFILE_BY_PRECISION = {"FP32": "TRT_FP32", "FP16": "TRT_FP16", "INT8_QDQ": "TRT_INT8_QDQ"}
WEIGHT_BY_PRECISION = {"FP32": "FP32", "FP16": "FP16", "INT8_QDQ": "INT8"}


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.is_file():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _module_block(row: dict[str, Any]) -> tuple[str, str]:
    unit_type = str(row.get("unit_type") or "")
    stage = str(row.get("parent_stage") or "")
    uid = str(row.get("unit_id") or "")
    if unit_type == "gemm":
        return "pfn", "pfn_block"
    if unit_type == "plugin":
        return "scatter", "plugin"
    if "shrink" in stage or "shrink" in uid:
        return "shrink", "compression_1x1"
    if "fusion" in stage or "fusion" in uid:
        return "pyramid_fusion", "fusion_block"
    if "head" in stage or "cls_head" in uid or "reg_head" in uid or "dir_head" in uid:
        return "detection_head", "head_branch"
    return "backbone", "conv_block"


def _key_from_row(row: dict[str, Any]) -> LatencyLUTKey:
    precision = str(row["precision"])
    module, block_type = _module_block(row)
    weight = WEIGHT_BY_PRECISION[precision]
    kernel = int(row.get("K") or 1)
    metadata = {
        "atomic_unit_id": row["unit_id"],
        "parent_stage": row.get("parent_stage"),
        "parent_block": row.get("parent_block"),
        "shape_signature": row.get("shape_signature"),
        "channel_keep_ratio": row.get("channel_keep_ratio"),
    }
    if row.get("scale_source") is not None:
        metadata["scale_source"] = row.get("scale_source")
    if row.get("activation_scale") is not None:
        metadata["activation_scale"] = row.get("activation_scale")
    if row.get("weight_scale") is not None:
        metadata["weight_scale"] = row.get("weight_scale")
    return LatencyLUTKey(
        deploy_mode=DEPLOY_MODE,
        fixed_K=FIXED_K,
        module_name=module,
        block_name=str(row["unit_id"]),
        block_type=block_type,
        H=row.get("H"),
        W=row.get("W"),
        C_in=row.get("C_in_aligned8") or row.get("C_in"),
        C_mid=row.get("C_mid_aligned8") or row.get("C_mid"),
        C_out=row.get("C_out_aligned8") or row.get("C_out"),
        kernel_size=kernel,
        stride=row.get("stride") or 1,
        padding=kernel // 2 if kernel > 1 else 0,
        groups=row.get("groups") or 1,
        precision_profile=PROFILE_BY_PRECISION[precision],
        weight_precision=weight,
        activation_precision=weight,
        compute_precision=weight,
        plugin_flag=block_type == "plugin",
        plugin_name="PointPillarScatterTRT" if block_type == "plugin" else None,
        metadata=metadata,
    )


def _measurement_from_record(row: dict[str, Any], record: Any, *, warmup: int, repeat: int) -> dict[str, Any]:
    return {
        "lut_key": row["lut_key"],
        "unit_id": row["unit_id"],
        "unit_type": row["unit_type"],
        "parent_stage": row.get("parent_stage"),
        "parent_block": row.get("parent_block"),
        "precision": row["precision"],
        "H": row.get("H"),
        "W": row.get("W"),
        "C_in": row.get("C_in"),
        "C_out": row.get("C_out"),
        "C_in_aligned8": row.get("C_in_aligned8"),
        "C_out_aligned8": row.get("C_out_aligned8"),
        "channel_keep_ratio": row.get("channel_keep_ratio"),
        "benchmark_backend": "trt_subgraph",
        "engine_mode": "subgraph_engine",
        "warmup": int(warmup),
        "repeat": int(repeat),
        "latency_p50_ms": float(record.latency_p50_ms),
        "latency_p90_ms": float(record.latency_p90_ms),
        "latency_mean_ms": float(record.latency_mean_ms),
        "latency_std_ms": float(record.latency_std_ms),
        "source": "measured",
        "valid": record.status in SUCCESS_STATUSES,
        "invalid_reason": "" if record.status in SUCCESS_STATUSES else (record.error_message or record.status),
        "status": record.status,
        "key_hash": record.key_hash,
        "engine_hash": record.engine_hash,
        "onnx_hash": record.onnx_hash,
    }


def _bucket_measurements(rows: list[dict[str, Any]], buckets_path: str | Path, include_estimated: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    path = Path(buckets_path)
    if not path.is_file():
        return [], [{"unit_id": row.get("unit_id"), "reason": "bucket_file_missing"} for row in rows]
    raw = json.loads(path.read_text(encoding="utf-8"))
    buckets = raw.get("buckets") if isinstance(raw, dict) else raw
    buckets = list(buckets or [])
    measured: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    for row in rows:
        unit_type = str(row.get("unit_type") or "")
        precision = str(row.get("precision") or "")
        matches = [
            bucket for bucket in buckets
            if unit_type in str(bucket.get("component_type") or bucket.get("bucket_id") or "")
            or str(bucket.get("component_type") or "") in unit_type
        ]
        if precision != "INT8_QDQ":
            matches = [bucket for bucket in matches if "INT8" not in str(bucket.get("precision_transition") or "")] or matches
        if not matches:
            invalid.append({"unit_id": row.get("unit_id"), "precision": precision, "reason": "matching_bucket_missing"})
            continue
        bucket = matches[0]
        source = str(bucket.get("source") or "")
        if source == "estimated" and not include_estimated:
            invalid.append({"unit_id": row.get("unit_id"), "precision": precision, "reason": "estimated_bucket_not_written_as_measurement"})
            continue
        measured.append(
            {
                "lut_key": row["lut_key"],
                "unit_id": row["unit_id"],
                "unit_type": unit_type,
                "parent_stage": row.get("parent_stage"),
                "parent_block": row.get("parent_block"),
                "precision": precision,
                "H": row.get("H"),
                "W": row.get("W"),
                "C_in": row.get("C_in"),
                "C_out": row.get("C_out"),
                "C_in_aligned8": row.get("C_in_aligned8"),
                "C_out_aligned8": row.get("C_out_aligned8"),
                "channel_keep_ratio": row.get("channel_keep_ratio"),
                "benchmark_backend": "microbenchmark" if source == "measured" else "measured_bucket",
                "engine_mode": "bucket",
                "warmup": None,
                "repeat": None,
                "latency_p50_ms": float(bucket.get("latency_p50_ms") or bucket.get("p50_ms") or bucket.get("latency_ms") or 0.0),
                "latency_p90_ms": float(bucket.get("latency_p90_ms") or bucket.get("p90_ms") or bucket.get("latency_ms") or 0.0),
                "latency_mean_ms": float(bucket.get("latency_ms") or bucket.get("latency_p50_ms") or bucket.get("p50_ms") or 0.0),
                "latency_std_ms": float(bucket.get("uncertainty_ms") or 0.0),
                "source": source if source else "coarse",
                "valid": source in {"measured", "coarse"} or bool(include_estimated),
                "invalid_reason": "",
                "status": "success" if source == "measured" else source,
                "bucket_id": bucket.get("bucket_id"),
            }
        )
    return measured, invalid


def run(args: argparse.Namespace) -> dict[str, Any]:
    data = json.loads(Path(args.key_space).read_text(encoding="utf-8"))
    all_rows = list(data.get("keys") or data)
    existing = _read_jsonl(args.output) if args.resume else []
    done = {row.get("lut_key") for row in existing if row.get("valid")}
    rows = [
        row for row in all_rows
        if (not args.precision_filter or row.get("precision") in set(args.precision_filter))
        and (not args.unit_type_filter or row.get("unit_type") in set(args.unit_type_filter))
        and (not args.stage_filter or row.get("parent_stage") in set(args.stage_filter))
        and (not args.resume or row.get("lut_key") not in done)
    ]
    if args.limit is not None:
        rows = rows[: int(args.limit)]
    records: list[dict[str, Any]] = list(existing) if args.resume else []
    failures: list[dict[str, Any]] = []
    subgraph_rows = [row for row in rows if row.get("expected_source") == "subgraph_engine"]
    bucket_rows = [row for row in rows if row.get("expected_source") != "subgraph_engine"]
    for index, row in enumerate(subgraph_rows, start=1):
        print(f"[lut-v3] {index}/{len(subgraph_rows)} {row['unit_id']} {row['precision']} {row['channel_keep_ratio']}")
        key = _key_from_row(row)
        record = benchmark_key(
            key,
            output_dir=args.work_dir,
            onnx_dir=args.onnx_dir,
            engine_dir=args.engine_dir,
            warmup=int(args.warmup),
            repeat=int(args.repeat),
            dry_run=bool(args.dry_run),
            trtexec_path=args.trtexec,
            device=args.device,
            iterations=args.iterations,
            min_repeat_ms=args.min_repeat_ms,
            plugin_path=args.plugin,
        )
        measurement = _measurement_from_record(row, record, warmup=args.warmup, repeat=args.repeat)
        records.append(measurement)
        if not measurement["valid"]:
            failures.append({"unit_id": row["unit_id"], "precision": row["precision"], "reason": measurement["invalid_reason"]})
        if args.incremental_write:
            _write_jsonl(args.output, records)
            Path(args.failed_output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.failed_output).write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
    bucket_measurements, bucket_failures = _bucket_measurements(bucket_rows, args.buckets, bool(args.include_estimated_buckets))
    records.extend(bucket_measurements)
    failures.extend(bucket_failures)
    _write_jsonl(args.output, records)
    Path(args.failed_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.failed_output).write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
    stats = {
        "status": "success",
        "key_space": str(args.key_space),
        "output": str(args.output),
        "selected_this_run": len(rows),
        "subgraph_selected": len(subgraph_rows),
        "bucket_selected": len(bucket_rows),
        "records_total": len(records),
        "valid_records_total": sum(1 for row in records if row.get("valid")),
        "failed_or_invalid_this_run": len(failures),
        "by_precision": dict(Counter(row.get("precision") for row in records if row.get("valid"))),
        "by_unit_type": dict(Counter(row.get("unit_type") for row in records if row.get("valid"))),
        "by_source": dict(Counter(row.get("source") for row in records if row.get("valid"))),
        "dry_run": bool(args.dry_run),
        "resume": bool(args.resume),
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(
        "# LUT Measurements v3 Report\n\n"
        f"- selected this run: {stats['selected_this_run']}\n"
        f"- valid records total: {stats['valid_records_total']}\n"
        f"- failed/invalid this run: {stats['failed_or_invalid_this_run']}\n"
        f"- by precision: `{stats['by_precision']}`\n"
        f"- by unit type: `{stats['by_unit_type']}`\n"
        f"- by source: `{stats['by_source']}`\n"
        f"- dry_run: `{stats['dry_run']}`\n\n"
        "Estimated buckets are excluded by default and are not written as measured records.\n",
        encoding="utf-8",
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return stats


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--key-space", "--key_space", dest="key_space", default="outputs/latency_lut/lut_key_space_v3.json")
    parser.add_argument("--output", default="outputs/latency_lut/lut_measurements_v3.jsonl")
    parser.add_argument("--failed-output", "--failed_output", dest="failed_output", default="outputs/latency_lut/lut_measurements_v3.failed.json")
    parser.add_argument("--report", default="outputs/latency_lut/lut_measurements_v3_report.md")
    parser.add_argument("--buckets", default="outputs/latency_lut/non_compute_latency_buckets_v3.json")
    parser.add_argument("--work-dir", "--work_dir", dest="work_dir", default="outputs/latency_lut/v3_subgraphs")
    parser.add_argument("--onnx-dir", "--onnx_dir", dest="onnx_dir", default="outputs/latency_lut/v3_subgraphs_onnx")
    parser.add_argument("--engine-dir", "--engine_dir", dest="engine_dir", default="outputs/latency_lut/v3_subgraphs_engine")
    parser.add_argument("--trtexec", default="${TENSORRT_ROOT}/targets/x86_64-linux-gnu/bin/trtexec")
    parser.add_argument("--plugin", default=None)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--min-repeat-ms", "--min_repeat_ms", dest="min_repeat_ms", type=int, default=None)
    parser.add_argument("--precision-filter", "--precision_filter", dest="precision_filter", nargs="*", default=None)
    parser.add_argument("--unit-type-filter", "--unit_type_filter", dest="unit_type_filter", nargs="*", default=None)
    parser.add_argument("--stage-filter", "--stage_filter", dest="stage_filter", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", "--dry_run", dest="dry_run", action="store_true")
    parser.add_argument("--include-estimated-buckets", "--include_estimated_buckets", dest="include_estimated_buckets", action="store_true")
    parser.add_argument("--incremental-write", "--incremental_write", dest="incremental_write", action="store_true", default=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
