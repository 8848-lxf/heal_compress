from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.latency_lut.benchmark_missing_lut_keys_for_pruned_candidates_v5 import run as run_v5


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decomposition-dir", "--decomposition_dir", dest="decomposition_dir", default="outputs/latency_lut/pruned_candidate_lut_decomposition_v6")
    parser.add_argument("--output", default="outputs/latency_lut/layer_width_precision_lut_measurements_v6.jsonl")
    parser.add_argument("--scale-cache", "--scale_cache", dest="scale_cache", default="outputs/latency_lut/activation_scale_cache_calib200.json")
    parser.add_argument("--work-dir", "--work_dir", dest="work_dir", default="outputs/latency_lut/layer_width_v6_missing_subgraphs")
    parser.add_argument("--onnx-dir", "--onnx_dir", dest="onnx_dir", default="outputs/latency_lut/layer_width_v6_missing_onnx")
    parser.add_argument("--engine-dir", "--engine_dir", dest="engine_dir", default="outputs/latency_lut/layer_width_v6_missing_engine")
    parser.add_argument("--trtexec", default="${TENSORRT_ROOT}/targets/x86_64-linux-gnu/bin/trtexec")
    parser.add_argument("--plugin", default=None)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=40)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--min-repeat-ms", "--min_repeat_ms", dest="min_repeat_ms", type=int, default=None)
    parser.add_argument("--report", default="outputs/latency_lut/missing_lut_key_benchmark_v6_report.md")
    parser.add_argument("--report-json", "--report_json", dest="report_json", default="outputs/latency_lut/missing_lut_key_benchmark_v6_report.json")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    run_v5(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
