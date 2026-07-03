from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.latency_lut.decompose_pruned_candidate_lut_v5 import run as run_v5


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="outputs/latency_lut/pruned_width_changed_candidates_v5.json")
    parser.add_argument("--export-dir", "--export_dir", dest="export_dir", default="outputs/latency_lut/pruned_width_changed_onnx_v5")
    parser.add_argument("--profiles", default="outputs/latency_lut/pruned_precision_profiles_v6.json")
    parser.add_argument("--lut-v4", "--lut_v4", dest="lut_v4", default="outputs/latency_lut/layer_width_precision_lut_measurements_v4.jsonl")
    parser.add_argument("--lut-v5", "--lut_v5", dest="lut_v5", default="outputs/latency_lut/layer_width_precision_lut_measurements_v5.jsonl")
    parser.add_argument("--structural-buckets", "--structural_buckets", dest="structural_buckets", default="outputs/latency_lut/structural_op_latency_buckets_v4.json")
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", default="outputs/latency_lut/pruned_candidate_lut_decomposition_v6")
    parser.add_argument("--report", default="outputs/latency_lut/pruned_candidate_lut_decomposition_v6_report.md")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    run_v5(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
