"""Evaluate accepted H800 Transformer TensorRT phenotypes on fixed manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from search.orchestration.lidar_transformer_h800_inventory import MODEL_SPECS


TRT_ROOT = Path("/home/lixingfeng/UniAD_examine/TensorRT-10.9_x86_cu118")
HEAL_ROOT = Path("/home/lixingfeng/UniAD_examine/HEAL")
PROTOCOL_FRAMES = {"smoke10": 10, "fixed50": 50, "fixed500": 500, "full1789": 1789}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def evaluate_profile(
    *,
    output_root: Path,
    model_name: str,
    profile: str,
    protocol: str,
    physical_gpu: int,
    plugin_path: Path,
    source_section: str = "baselines",
) -> dict[str, Any]:
    profile_dir = output_root / source_section / model_name / profile
    acceptance_path = profile_dir / "baseline_result.json"
    if not acceptance_path.is_file():
        raise RuntimeError(f"baseline_acceptance_missing:{model_name}:{profile}")
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    if acceptance.get("status") != "ok":
        raise RuntimeError(f"baseline_not_accepted:{model_name}:{profile}:{acceptance.get('status')}")
    if int(acceptance.get("requested_realized_conflict_count", -1)) != 0:
        raise RuntimeError(f"precision_conflict_blocks_evaluation:{model_name}:{profile}")
    engine = profile_dir / "engine.plan"
    if not engine.is_file() or not engine.stat().st_size:
        raise RuntimeError(f"engine_missing:{model_name}:{profile}")
    manifest = output_root / "evaluation" / "manifests" / model_name / f"{protocol}.json"
    frames = PROTOCOL_FRAMES[protocol]
    warmup = len(json.loads(manifest.read_text(encoding="utf-8"))["warmup_frame_ids"])
    destination = profile_dir / "evaluation" / protocol
    spec = MODEL_SPECS[model_name]
    if model_name == "lidar_cobevt":
        from search.integration.lidar_cobevt_evaluation_provider import (
            evaluate_cobevt_engine_modelopt,
        )

        result = evaluate_cobevt_engine_modelopt(
            engine_path=engine,
            checkpoint=spec["checkpoint"],
            model_config=spec["config"],
            heal_root=HEAL_ROOT,
            device=f"cuda:{physical_gpu}",
            output_dir=destination,
            tensorrt_root=TRT_ROOT,
            plugin_path=plugin_path,
            fixed_k=int(spec["fixed_k"]),
            num_frames=frames,
            warmup_frames=warmup,
            eval_manifest_path=manifest,
            num_workers=8,
            ap_iou_backend="gpu",
            latency_rounds=1,
        )
    else:
        from search.model_family.evaluation import evaluate_v2xvit_engine_modelopt

        result = evaluate_v2xvit_engine_modelopt(
            engine_path=engine,
            model_config=spec["config"],
            heal_root=HEAL_ROOT,
            output_dir=destination,
            tensorrt_root=TRT_ROOT,
            plugin_path=plugin_path,
            eval_manifest_path=manifest,
            physical_gpu_id=physical_gpu,
            fixed_k=int(spec["fixed_k"]),
            max_agents=2,
            num_frames=frames,
            warmup_frames=warmup,
            latency_rounds=1,
            dataloader_num_workers=8,
        )
    accepted = (
        result.get("status") == "ok"
        and int(result.get("num_evaluated_frames", -1)) == frames
        and int(result.get("num_skipped_frames", -1)) == 0
        and bool(result.get("reset_after_warmup", False))
    )
    summary = {
        "model": model_name,
        "profile": profile,
        "protocol": protocol,
        "physical_gpu": physical_gpu,
        "status": "ok" if accepted else "evaluation_failed",
        "evaluated": result.get("num_evaluated_frames", 0),
        "skipped": result.get("num_skipped_frames", 0),
        "AP@0.3": result.get("AP@0.3"),
        "AP@0.5": result.get("AP@0.5"),
        "AP@0.7": result.get("AP@0.7"),
        "mAP": result.get("mAP"),
        "forward_p50_ms": result.get("forward_p50_ms"),
        "eval_manifest_hash": result.get("eval_manifest_hash", ""),
        "failure_reason": result.get("failure_reason", ""),
    }
    _write_json(destination / "evaluation_acceptance.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--model", choices=tuple(MODEL_SPECS), required=True)
    parser.add_argument("--profiles", required=True)
    parser.add_argument("--protocol", choices=tuple(PROTOCOL_FRAMES), required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--plugin", required=True)
    parser.add_argument(
        "--source-section",
        choices=("baselines", "precision_sensitivity", "smoothquant", "smoothquant_sm90", "bf16", "fp8", "accumulator"),
        default="baselines",
    )
    args = parser.parse_args(argv)
    rows = [
        evaluate_profile(
            output_root=Path(args.output_root).resolve(),
            model_name=args.model,
            profile=profile.strip(),
            protocol=args.protocol,
            physical_gpu=args.physical_gpu,
            plugin_path=Path(args.plugin).resolve(),
            source_section=args.source_section,
        )
        for profile in args.profiles.split(",")
        if profile.strip()
    ]
    destination = (
        Path(args.output_root).resolve()
        / "evaluation"
        / f"{args.model}_{args.source_section}_{args.protocol}_matrix.json"
    )
    _write_json(destination, rows)
    print(json.dumps(rows, sort_keys=True))
    return 0 if all(row["status"] == "ok" for row in rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
