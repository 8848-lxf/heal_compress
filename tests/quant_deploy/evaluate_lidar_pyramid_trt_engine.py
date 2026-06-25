from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from quant_deploy_utils import ensure_quant_deploy_run_dirs, save_csv, save_json


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reserved evaluation entrypoint for lidar_pyramid TensorRT engines.")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--precision", choices=["fp32", "fp16", "int8"], required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dirs = ensure_quant_deploy_run_dirs(args.output_root)
    out_dir = dirs[f"evaluation_{args.precision}"]
    payload = {
        "precision": args.precision,
        "evaluation_status": "not_run",
        "reason": "current stage only benchmarks TensorRT engine forward latency",
        "ap_0_3": None,
        "ap_0_5": None,
        "ap_0_7": None,
        "map": None,
    }
    save_json(payload, out_dir / f"eval_metrics_{args.precision}.json")
    save_csv([payload], out_dir / f"eval_metrics_{args.precision}.csv")
    (out_dir / f"eval_log_{args.precision}.txt").write_text(payload["reason"] + "\n", encoding="utf-8")
    (out_dir / "pred_outputs").mkdir(parents=True, exist_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
