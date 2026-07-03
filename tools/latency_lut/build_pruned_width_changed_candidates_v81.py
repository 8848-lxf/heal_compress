from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.latency_lut.build_pruned_width_changed_candidates_v8 import build as build_v8, parse_args as parse_v8_args


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = parse_v8_args(argv)
    if args.output == "outputs/latency_lut/pruned_width_changed_candidates_v8.json":
        args.output = "outputs/latency_lut/pruned_width_changed_candidates_v81.json"
    if args.report == "outputs/latency_lut/pruned_width_changed_candidates_v8_report.md":
        args.report = "outputs/latency_lut/pruned_width_changed_candidates_v81_report.md"
    return args


def main(argv: list[str] | None = None) -> int:
    build_v8(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
