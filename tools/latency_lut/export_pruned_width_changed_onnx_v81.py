from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.latency_lut.export_pruned_width_changed_onnx_v5 import parse_args as parse_v5_args, run as run_v5


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = parse_v5_args(argv)
    if args.candidates == "outputs/latency_lut/pruned_width_changed_candidates_v5.json":
        args.candidates = "outputs/latency_lut/pruned_width_changed_candidates_v81.json"
    if args.output_dir == "outputs/latency_lut/pruned_width_changed_onnx_v5":
        args.output_dir = "outputs/latency_lut/pruned_width_changed_onnx_v81"
    return args


def main(argv: list[str] | None = None) -> int:
    summary = run_v5(parse_args(argv))
    print(json.dumps({k: v for k, v in summary.items() if k != "reports"}, indent=2))
    return 0 if int(summary.get("width_changed_onnx_exported") or 0) else 2


if __name__ == "__main__":
    raise SystemExit(main())
