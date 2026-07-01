from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from opencood.tools.compression.latency_lut.schema import (
    LatencyLUTKey,
    LatencyRecord,
    read_key_jsonl,
    read_record_jsonl,
    write_jsonl,
    write_records_csv,
)
from opencood.tools.compression.latency_lut.tensorRT_benchmark import SUCCESS_STATUSES, benchmark_key


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark TensorRT latency LUT subgraphs.")
    parser.add_argument("--keys", default="outputs/latency_lut/keys.jsonl")
    parser.add_argument("--output", default="outputs/latency_lut/lut_records.jsonl")
    parser.add_argument("--failed-output", "--failed_output", dest="failed_output", default="outputs/latency_lut/failed_keys.jsonl")
    parser.add_argument("--backend", default="tensorrt")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeat", type=int, default=200)
    parser.add_argument("--dry-run", "--dry_run", dest="dry_run", action="store_true", help="Validate export paths without building TensorRT engines.")
    parser.add_argument("--no-dry-run", dest="dry_run", action="store_false", help="Build and benchmark TensorRT engines. This is the default.")
    parser.set_defaults(dry_run=False)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--module-filter", "--module_filter", dest="module_filter", nargs="*", default=None)
    parser.add_argument("--precision-filter", "--precision_filter", dest="precision_filter", nargs="*", default=None)
    parser.add_argument("--resume", action="store_true", help="Skip keys with existing success records in --output.")
    parser.add_argument("--work-dir", "--work_dir", dest="work_dir", default="outputs/latency_lut/subgraphs")
    parser.add_argument("--onnx-dir", "--onnx_dir", dest="onnx_dir", default="outputs/latency_lut/subgraphs_onnx")
    parser.add_argument("--engine-dir", "--engine_dir", dest="engine_dir", default="outputs/latency_lut/subgraphs_engine")
    parser.add_argument("--trtexec", default=None)
    parser.add_argument("--plugin", default=None, help="Optional PointPillarScatterTRT shared library for plugin LUT keys.")
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--min-repeat-ms", "--min_repeat_ms", dest="min_repeat_ms", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--allow-int8-fallback-to-fp16", "--allow_int8_fallback_to_fp16", dest="allow_int8_fallback_to_fp16", action="store_true")
    return parser.parse_args(argv)


def _module_match(key: LatencyLUTKey, filters: list[str] | None) -> bool:
    if not filters:
        return True
    haystack = " ".join([key.module_name, key.block_name, key.block_type]).lower()
    aliases = {
        "head": ["head", "detection_head", "head_branch"],
        "fusion": ["fusion", "pyramid_fusion", "fusion_block"],
        "shrink": ["shrink", "compression_1x1", "compression"],
        "backbone": ["backbone"],
        "plugin": ["plugin", "scatter", "pointpillarscattertrt"],
    }
    for raw_filter in filters:
        needle = str(raw_filter).lower()
        options = aliases.get(needle, [needle])
        if any(option in haystack for option in options):
            return True
    return False


def _precision_match(key: LatencyLUTKey, filters: list[str] | None) -> bool:
    if not filters:
        return True
    requested = {str(item).upper() for item in filters}
    return key.precision_profile.upper() in requested


def success_key_hashes(records_path: str | Path) -> set[str]:
    return {
        record.key_hash
        for record in read_record_jsonl(records_path)
        if record.status in SUCCESS_STATUSES
    }


def filter_keys(keys: list[LatencyLUTKey], args: argparse.Namespace, *, completed_hashes: set[str] | None = None) -> list[LatencyLUTKey]:
    selected = [
        key
        for key in keys
        if _module_match(key, getattr(args, "module_filter", None))
        and _precision_match(key, getattr(args, "precision_filter", None))
    ]
    if getattr(args, "resume", False):
        done = completed_hashes if completed_hashes is not None else success_key_hashes(args.output)
        selected = [key for key in selected if key.stable_hash() not in done]
    if getattr(args, "limit", None) is not None:
        selected = selected[: int(args.limit)]
    return selected


def _write_outputs(records: list[LatencyRecord], output: Path, failed_output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    failed_output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(records, output)
    write_records_csv(records, output.with_suffix(".csv"))
    failed_or_skipped = [record for record in records if record.status not in SUCCESS_STATUSES and record.status != "dry_run"]
    write_jsonl(failed_or_skipped, failed_output)


def _stats(records: list[LatencyRecord], selected: list[LatencyLUTKey], args: argparse.Namespace) -> dict:
    status_counts = Counter(record.status for record in records)
    precision_counts = Counter(record.key.precision_profile for record in records if record.status in SUCCESS_STATUSES)
    selected_precision_counts = Counter(key.precision_profile for key in selected)
    module_counts = Counter(record.key.module_name for record in records if record.status in SUCCESS_STATUSES)
    return {
        "keys_input": str(args.keys),
        "output": str(args.output),
        "failed_output": str(args.failed_output),
        "num_selected_this_run": len(selected),
        "num_records_written": len(records),
        "num_success": sum(status_counts[status] for status in SUCCESS_STATUSES),
        "num_failed": status_counts.get("failed", 0),
        "num_skipped": sum(count for status, count in status_counts.items() if status.startswith("skipped")),
        "num_dry_run": status_counts.get("dry_run", 0),
        "status_counts": dict(sorted(status_counts.items())),
        "success_by_precision": dict(sorted(precision_counts.items())),
        "selected_by_precision": dict(sorted(selected_precision_counts.items())),
        "success_by_module": dict(sorted(module_counts.items())),
        "dry_run": bool(args.dry_run),
        "resume": bool(args.resume),
    }


def run(args: argparse.Namespace) -> dict:
    keys = read_key_jsonl(args.keys)
    output = Path(args.output)
    failed_output = Path(args.failed_output)
    valid_key_hashes = {key.stable_hash() for key in keys}
    existing_records = [
        record
        for record in read_record_jsonl(output)
        if record.status != "dry_run" and record.key_hash in valid_key_hashes
    ] if args.resume else []
    existing_success = [record for record in existing_records if record.status in SUCCESS_STATUSES]
    selected = filter_keys(keys, args, completed_hashes={record.key_hash for record in existing_success})
    new_records: list[LatencyRecord] = []
    for index, key in enumerate(selected, start=1):
        print(
            f"[latency_lut] benchmark {index}/{len(selected)} "
            f"{key.module_name}/{key.block_name}/{key.block_type}/{key.precision_profile} "
            f"{key.stable_hash()[:12]}"
        )
        record = benchmark_key(
            key,
            output_dir=args.work_dir,
            onnx_dir=args.onnx_dir,
            engine_dir=args.engine_dir,
            warmup=int(args.warmup),
            repeat=int(args.repeat),
            backend=str(args.backend),
            dry_run=bool(args.dry_run),
            trtexec_path=args.trtexec,
            device=args.device,
            min_repeat_ms=args.min_repeat_ms,
            iterations=args.iterations,
            allow_int8_fallback_to_fp16=bool(args.allow_int8_fallback_to_fp16),
            plugin_path=args.plugin,
        )
        print(f"[latency_lut]   status={record.status} p50_ms={record.latency_p50_ms:.6f}")
        new_records.append(record)

    if args.resume:
        records_by_hash = {record.key_hash: record for record in existing_records}
        for record in new_records:
            records_by_hash[record.key_hash] = record
        records = list(records_by_hash.values())
    else:
        records = new_records
    _write_outputs(records, output, failed_output)
    stats = _stats(records, selected, args)
    output.with_suffix(".summary.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return stats


def main(argv: list[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
