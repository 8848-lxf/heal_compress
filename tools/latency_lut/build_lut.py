from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.latency_lut.benchmark_subgraph import run as run_benchmark
from tools.latency_lut.benchmark_subgraph import parse_args as parse_benchmark_args
from tools.latency_lut.collect_keys import run as run_collect
from tools.latency_lut.collect_keys import parse_args as parse_collect_args


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect keys and benchmark latency LUT subgraphs.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", default="outputs/latency_lut")
    parser.add_argument("--backend", default="tensorrt")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeat", type=int, default=200)
    parser.add_argument("--dry-run", "--dry_run", dest="dry_run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.output_dir)
    keys_path = root / "keys.jsonl"
    records_path = root / "lut_records.jsonl"
    collect = run_collect(parse_collect_args(["--config", args.config, "--output", str(keys_path)] + (["--dry-run"] if args.dry_run else [])))
    bench_argv = [
        "--keys", str(keys_path),
        "--output", str(records_path),
        "--backend", args.backend,
        "--warmup", str(args.warmup),
        "--repeat", str(args.repeat),
    ]
    if args.dry_run:
        # collect dry-run does not write keys; run a real collect because
        # benchmark dry-run needs the key file but still performs no TRT work.
        run_collect(parse_collect_args(["--config", args.config, "--output", str(keys_path)]))
        bench_argv.append("--dry-run")
    if args.limit is not None:
        bench_argv.extend(["--limit", str(args.limit)])
    bench = run_benchmark(parse_benchmark_args(bench_argv))
    print(json.dumps({"collect": collect, "benchmark": bench}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
