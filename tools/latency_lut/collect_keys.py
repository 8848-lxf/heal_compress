from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from opencood.tools.compression.latency_lut.key_builder import (
    build_lut_keys_for_units,
    collect_key_stats,
    load_units_from_config,
)
from opencood.tools.compression.latency_lut.schema import DEPLOY_MODE, FIXED_K, write_jsonl


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect TensorRT latency LUT keys.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="outputs/latency_lut/keys.jsonl")
    parser.add_argument("--deploy-mode", "--deploy_mode", dest="deploy_mode", default=DEPLOY_MODE)
    parser.add_argument("--fixed-k", "--fixed_k", dest="fixed_k", type=int, default=FIXED_K)
    parser.add_argument("--dry-run", "--dry_run", dest="dry_run", action="store_true")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict:
    units, cfg = load_units_from_config(args.config)
    keys = build_lut_keys_for_units(
        units,
        keep_ratios=list(cfg.get("keep_ratios", [1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25])),
        min_keep_ratio=float(cfg.get("min_keep_ratio", 0.25)),
        channel_align=int(cfg.get("channel_align", 16)),
        precision_profiles=list(cfg.get("precision_profiles", ["TRT_FP32", "TRT_FP16", "TRT_INT8_QDQ"])),
        deploy_mode=args.deploy_mode,
        fixed_K=int(args.fixed_k),
        include_boundaries=bool(cfg.get("include_precision_boundaries", True)),
    )
    stats = collect_key_stats(keys, units)
    stats.update({"output": str(args.output), "dry_run": bool(args.dry_run)})
    if not args.dry_run:
        write_jsonl(keys, args.output)
        summary_path = Path(args.output).with_suffix(".summary.json")
        summary_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return stats


def main(argv: list[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
