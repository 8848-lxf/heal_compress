#!/usr/bin/env python3
"""Build and fixed50-evaluate one all-keep AttFusion deployment candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from search.canonicalization import canonicalize_candidate
from search.ga.cnn_stage12_v3 import (
    MODEL_SPECS,
    baseline_genotype,
    create_real_evaluator,
    prepare_search,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--plugin", type=Path, required=True)
    parser.add_argument("--tensorrt-root", type=Path, required=True)
    args = parser.parse_args()
    prepared = prepare_search(
        MODEL_SPECS["attfusion"],
        output_root=args.output_root,
        physical_gpu=args.physical_gpu,
        plugin=args.plugin,
        tensorrt_root=args.tensorrt_root,
        taylor_samples=1,
    )
    candidate = baseline_genotype(prepared.space)
    evaluator = create_real_evaluator(prepared, output_root=args.output_root)
    result = evaluator.evaluate_candidate(
        canonicalize_candidate(candidate, prepared.space),
        output_dir=args.output_root / "stage2_all_keep_fp32",
        candidate_hash="attfusion_all_keep_fp32_preflight",
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str), flush=True)
    if result.get("status") != "ok":
        raise RuntimeError(
            f"attfusion_stage2_preflight_failed:{result.get('failure_reason', result)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
