#!/usr/bin/env python3
"""Evaluate one deployment-closed V2X-ViT TensorRT engine on a fixed manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.audit_heal_transformer_search_models import MODEL_SPECS
from search.model_family.evaluation import evaluate_v2xvit_engine_modelopt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--fixed-k", type=int, default=27904)
    parser.add_argument("--warmup-frames", type=int, default=0)
    parser.add_argument("--tensorrt-root", type=Path, default=Path("/home/lixingfeng/UniAD_examine/HEAL/prune_model/TensorRT-10.9_x86_cu118"))
    parser.add_argument("--plugin", type=Path, default=Path("/home/lixingfeng/UniAD_examine/heal_compress/quantization/plugins/pointpillar_scatter_trt/build/libpointpillar_scatter_trt.so"))
    args = parser.parse_args()
    result = evaluate_v2xvit_engine_modelopt(
        engine_path=args.engine,
        model_config=MODEL_SPECS["v2xvit"]["config"],
        heal_root="/home/lixingfeng/UniAD_examine/HEAL",
        output_dir=args.output_dir,
        tensorrt_root=args.tensorrt_root,
        plugin_path=args.plugin,
        eval_manifest_path=args.manifest,
        physical_gpu_id=int(args.physical_gpu),
        fixed_k=int(args.fixed_k),
        num_frames=int(args.frames),
        warmup_frames=int(args.warmup_frames),
        latency_rounds=0,
        dataloader_num_workers=8,
    )
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if str(result.get("status", "ok")) in {"ok", "completed"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
