#!/usr/bin/env python3
"""Fully evaluate already accepted strict-FP32 engines without rebuilding."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any


REPO = Path(__file__).resolve().parents[1]
for path in (REPO.parent, REPO):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accepted-engine-root", type=Path, required=True)
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--eval-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, default=7)
    parser.add_argument("--heal-root", type=Path, default=Path("/home/lixingfeng/UniAD_examine/HEAL"))
    parser.add_argument("--tensorrt-root", type=Path, default=Path("/home/lixingfeng/UniAD_examine/TensorRT-10.9_x86_cu118"))
    parser.add_argument("--plugin", type=Path, default=REPO / "quantization/plugins/pointpillar_scatter_trt/build/libpointpillar_scatter_trt.so")
    parser.add_argument("--warmup-frames", type=int, default=200)
    parser.add_argument("--num-frames", type=int, default=1789)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--models", nargs="*", default=[])
    args = parser.parse_args()
    if os.environ.get("CONDA_DEFAULT_ENV") != "univ2x-opt":
        raise RuntimeError("existing_fp32_engine_evaluation_requires_univ2x_opt")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    selected = set(str(value) for value in args.models)
    sources = sorted(args.accepted_engine_root.glob("*/deployment_result.json"))
    if selected:
        sources = [path for path in sources if path.parent.name in selected]
    summary = []
    for result_path in sources:
        accepted = json.loads(result_path.read_text(encoding="utf-8"))
        model_name = str(accepted.get("model_name", result_path.parent.name))
        if accepted.get("status") != "ok":
            raise RuntimeError(f"source_engine_not_accepted:{model_name}")
        if not accepted.get("strongly_typed") or not accepted.get("no_tf32"):
            raise RuntimeError(f"source_engine_not_strict_fp32:{model_name}")
        if not accepted.get("precision_audit", {}).get("passed", False):
            raise RuntimeError(f"source_engine_precision_audit_failed:{model_name}")
        engine = result_path.parent / "strict_fp32.plan"
        if not engine.is_file() or _sha256(engine) != str(accepted.get("engine_sha256", "")):
            raise RuntimeError(f"source_engine_hash_mismatch:{model_name}")
        model_dir = args.models_root / model_name
        checkpoints = list(model_dir.glob("net_epoch_bestval_at*.pth"))
        if len(checkpoints) != 1:
            raise RuntimeError(f"source_checkpoint_not_unique:{model_name}")
        destination = output / model_name
        destination.mkdir(parents=True, exist_ok=False)
        if model_name == "lidar_pyramid":
            from search.integration.evaluation_provider import evaluate_engine_modelopt

            evaluation = evaluate_engine_modelopt(
                engine_path=engine,
                checkpoint=checkpoints[0],
                model_config=model_dir / "config.yaml",
                heal_root=args.heal_root,
                device=f"cuda:{int(args.physical_gpu)}",
                output_dir=destination / "evaluation",
                tensorrt_root=args.tensorrt_root,
                plugin_path=args.plugin,
                num_frames=int(args.num_frames),
                warmup_frames=int(args.warmup_frames),
                fixed_k=int(accepted["fixed_k"]),
                latency_rounds=1,
                eval_manifest_path=args.eval_manifest,
                dataloader_num_workers=int(args.workers),
            )
        else:
            from search.model_family.evaluation import evaluate_v2xvit_engine_modelopt

            evaluation = evaluate_v2xvit_engine_modelopt(
                engine_path=engine,
                model_config=model_dir / "config.yaml",
                heal_root=args.heal_root,
                output_dir=destination / "evaluation",
                tensorrt_root=args.tensorrt_root,
                plugin_path=args.plugin,
                eval_manifest_path=args.eval_manifest,
                physical_gpu_id=int(args.physical_gpu),
                fixed_k=int(accepted["fixed_k"]),
                max_agents=2,
                num_frames=int(args.num_frames),
                warmup_frames=int(args.warmup_frames),
                latency_rounds=1,
                dataloader_num_workers=int(args.workers),
                input_contract=str(accepted["input_contract"]),
            )
        _write_json(destination / "evaluation_acceptance.json", evaluation)
        provenance = {
            "model_name": model_name,
            "source_deployment_result": str(result_path.resolve()),
            "source_engine": str(engine.resolve()),
            "source_engine_sha256": _sha256(engine),
            "source_engine_rebuilt": False,
            "fixed_k": int(accepted["fixed_k"]),
            "input_contract": str(accepted["input_contract"]),
            "eval_manifest": str(args.eval_manifest.resolve()),
            "eval_manifest_sha256": _sha256(args.eval_manifest),
        }
        _write_json(destination / "reuse_provenance.json", provenance)
        summary.append(
            {
                "model_name": model_name,
                "status": evaluation.get("status"),
                "failure_reason": evaluation.get("failure_reason", ""),
                "AP@0.3": evaluation.get("AP@0.3"),
                "AP@0.5": evaluation.get("AP@0.5"),
                "AP@0.7": evaluation.get("AP@0.7"),
                "mAP": evaluation.get("mAP"),
                "forward_p50_ms": evaluation.get("forward_p50_ms"),
                "forward_p90_ms": evaluation.get("forward_p90_ms"),
                "forward_p99_ms": evaluation.get("forward_p99_ms"),
                "source_engine_sha256": provenance["source_engine_sha256"],
                "source_engine_rebuilt": False,
            }
        )
        _write_json(output / "strict_fp32_full_evaluation_summary.json", {"models": summary})
        if evaluation.get("status") != "ok":
            raise RuntimeError(f"strict_fp32_full_evaluation_failed:{model_name}:{evaluation.get('failure_reason', '')}")
    _write_json(
        output / "strict_fp32_full_evaluation_summary.json",
        {"models": summary, "all_passed": True, "source_engines_rebuilt": False},
    )
    print(json.dumps({"status": "ok", "models": len(summary), "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
