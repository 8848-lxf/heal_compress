from __future__ import annotations

import argparse
import json
from pathlib import Path


def collect_scales(args: argparse.Namespace) -> dict:
    if not args.layer_mapping or not Path(args.layer_mapping).is_file():
        return {
            "success": False,
            "status": "activation_scale_unit_mapping_failed",
            "error": "full-graph PyTorch/ONNX node to deployment-unit mapping is required before collecting reliable activation scales",
            "scale_source": None,
            "num_frames": int(args.num_frames),
        }
    return {
        "success": False,
        "status": "activation_scale_collection_not_connected",
        "error": "mapping file exists, but module hook instrumentation for full HEAL lidar_pyramid deployment units is not connected in this tool yet",
        "scale_source": None,
        "num_frames": int(args.num_frames),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="${MODEL_ROOT}/lidar_pyramid/config.yaml")
    parser.add_argument("--checkpoint", default="${MODEL_ROOT}/lidar_pyramid/net_epoch_bestval_at17.pth")
    parser.add_argument("--num-frames", type=int, default=200)
    parser.add_argument("--layer-mapping", default=None)
    parser.add_argument("--output", default="outputs/latency_lut/full_graph_activation_scale_cache.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = collect_scales(args)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("success") else 2


if __name__ == "__main__":
    raise SystemExit(main())
