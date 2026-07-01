from __future__ import annotations

import argparse
import json
from types import SimpleNamespace
from typing import Any

from quantization.utils.logging import save_json
from quantization.utils.paths import (
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIG,
    DEFAULT_FIXED_K,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_STRATEGY,
    ensure_dir,
    infer_output_root,
    load_quant_deploy_module,
    normalize_strategy,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dump train calibration NPZ for formal single_engine_maxK INT8 option.")
    parser.add_argument("--config", "--hypes-yaml", "--hypes_yaml", dest="config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--fixed-k", "--fixed_k", dest="fixed_k", type=int, default=DEFAULT_FIXED_K)
    parser.add_argument("--num-frames", "--num_frames", dest="num_frames", type=int, default=200)
    parser.add_argument("--split", default="train", choices=["train"])
    parser.add_argument("--strategy", default=DEFAULT_STRATEGY, choices=["single_engine_maxK", "dynamic_bucket"])
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", default=str(DEFAULT_OUTPUT_ROOT / "artifacts/calibration/train_calib_single_engine_maxK29696_200"))
    parser.add_argument("--heal-repo", "--heal_repo", dest="heal_repo", default="/home/lixingfeng/UniAD_examine/HEAL")
    parser.add_argument("--modality", default="m1")
    parser.add_argument("--max-scan-samples", "--max_scan_samples", dest="max_scan_samples", type=int, default=None)
    return parser.parse_args(argv)


def dump_calibration_npz(args: argparse.Namespace) -> dict[str, Any]:
    if args.split != "train":
        raise ValueError("formal INT8 calibration must use split=train")
    output_dir = ensure_dir(args.output_dir)
    output_root = infer_output_root(output_dir)
    strategy = normalize_strategy(args.strategy)
    legacy = load_quant_deploy_module("dump_train_calibration_npz_for_all_strategies")
    legacy_args = SimpleNamespace(
        output_root=str(output_root),
        calib_split="train",
        num_calib_frames=[int(args.num_frames)],
        fixed_K=int(args.fixed_k),
        ensure_agent_coverage="1,2",
        ensure_k_coverage="p50,p90,p95,p99,max",
        max_scan_samples=args.max_scan_samples,
        hypes_yaml=str(args.config),
        checkpoint=str(args.checkpoint),
        heal_repo=str(args.heal_repo),
        modality=str(args.modality),
        strategies=[strategy],
    )
    report = legacy.run(legacy_args)
    report.update(
        {
            "formal_tool": "quantization.calibrate.dump_train_calibration_npz",
            "strategy": strategy,
            "default_strategy": DEFAULT_STRATEGY,
            "fixed_K": int(args.fixed_k),
            "calibration_split": "train",
            "formal_output_dir": str(output_dir),
        }
    )
    save_json(report, output_dir / "formal_calibration_report.json")
    return report


def main(argv: list[str] | None = None) -> int:
    report = dump_calibration_npz(parse_args(argv))
    print(json.dumps({"success": report.get("success"), "fixed_K": report.get("fixed_K")}, indent=2))
    return 0 if report.get("success") else 2


if __name__ == "__main__":
    raise SystemExit(main())
