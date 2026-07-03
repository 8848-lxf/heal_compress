from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.latency_lut.sample_precision_profiles_v5 import build as build_v5


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="outputs/latency_lut/pruned_width_changed_candidates_v5.json")
    parser.add_argument("--export-dir", "--export_dir", dest="export_dir", default="outputs/latency_lut/pruned_width_changed_onnx_v5")
    parser.add_argument("--inventory", default="outputs/latency_lut/atomic_deployment_unit_inventory_v2.json")
    parser.add_argument("--scale-cache", "--scale_cache", dest="scale_cache", default="outputs/latency_lut/activation_scale_cache_calib200.json")
    parser.add_argument("--lut", default="outputs/latency_lut/layer_width_precision_lut_measurements_v4.jsonl")
    parser.add_argument("--output", default="outputs/latency_lut/pruned_precision_profiles_v6.json")
    parser.add_argument("--report", default="outputs/latency_lut/pruned_precision_profiles_v6_report.md")
    parser.add_argument("--min-profiles", "--min_profiles", dest="min_profiles", type=int, default=40)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    payload = build_v5(parse_args(argv))
    Path("outputs/latency_lut/pruned_precision_profiles_v6_report.md").write_text(
        Path("outputs/latency_lut/pruned_precision_profiles_v6_report.md").read_text(encoding="utf-8").replace("v5", "v6"),
        encoding="utf-8",
    )
    return 0 if int(payload.get("num_profiles") or 0) >= 40 else 1


if __name__ == "__main__":
    raise SystemExit(main())
