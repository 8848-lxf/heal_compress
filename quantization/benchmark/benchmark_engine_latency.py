from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from quantization.utils.engine_io import file_info
from quantization.utils.logging import save_json, status_record
from quantization.utils.paths import DEFAULT_FIXED_K, DEFAULT_PRECISION, ensure_dir, validate_precision


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record formal engine latency benchmark metadata.")
    parser.add_argument("--engine", required=True)
    parser.add_argument("--precision", default=DEFAULT_PRECISION, choices=["fp32", "fp16", "int8"])
    parser.add_argument("--fixed-k", "--fixed_k", dest="fixed_k", type=int, default=DEFAULT_FIXED_K)
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", required=True)
    parser.add_argument("--note", default="Metadata-only command; runtime measurements require explicit evaluation inputs.")
    return parser.parse_args(argv)


def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    precision = validate_precision(args.precision)
    output_dir = ensure_dir(args.output_dir)
    report = status_record(
        success=Path(args.engine).is_file(),
        status="metadata_only" if Path(args.engine).is_file() else "missing_engine",
        formal_tool="quantization.benchmark.benchmark_engine_latency",
        strategy="single_engine_maxK",
        fixed_K=int(args.fixed_k),
        precision=precision,
        engine=file_info(args.engine),
        note=args.note,
    )
    save_json(report, output_dir / "formal_engine_latency_benchmark_report.json")
    return report


def main(argv: list[str] | None = None) -> int:
    report = benchmark(parse_args(argv))
    print(json.dumps({"success": report.get("success"), "status": report.get("status")}, indent=2))
    return 0 if report.get("success") else 2


if __name__ == "__main__":
    raise SystemExit(main())
