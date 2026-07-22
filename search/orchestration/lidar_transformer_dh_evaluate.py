"""Manifest-locked smoke10/fixed50/fixed500 evaluation of d_h engines."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from search.orchestration.lidar_transformer_h800_inventory import MODEL_SPECS


TRT_ROOT = Path("/home/lixingfeng/UniAD_examine/TensorRT-10.9_x86_cu118")
HEAL_ROOT = Path("/home/lixingfeng/UniAD_examine/HEAL")
FRAMES = {"smoke10": 10, "fixed50": 50, "fixed500": 500}


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluate(
    *, output_root: Path, model_name: str, family_id: str, d_h: int,
    profile: str, protocol: str, physical_gpu: int, plugin: Path,
) -> dict[str, Any]:
    directory = output_root / "engines" / model_name / family_id / f"dh_{int(d_h):03d}" / profile
    build = json.loads((directory / "baseline_result.json").read_text(encoding="utf-8"))
    if build.get("status") != "ok" or int(build.get("requested_realized_conflict_count", -1)) != 0:
        raise RuntimeError(f"dh_evaluation_build_not_exact:{build.get('status')}")
    engine = directory / "engine.plan"
    if not engine.is_file() or not engine.stat().st_size:
        raise RuntimeError("dh_evaluation_engine_missing")
    manifest = output_root / "evaluation" / "manifests" / model_name / f"{protocol}.json"
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    warmup = len(manifest_payload["warmup_frame_ids"])
    destination = directory / "evaluation" / protocol
    spec = MODEL_SPECS[model_name]
    if model_name == "lidar_cobevt":
        from search.integration.lidar_cobevt_evaluation_provider import evaluate_cobevt_engine_modelopt

        result = evaluate_cobevt_engine_modelopt(
            engine_path=engine,
            checkpoint=spec["checkpoint"],
            model_config=spec["config"],
            heal_root=HEAL_ROOT,
            device=f"cuda:{physical_gpu}",
            output_dir=destination,
            tensorrt_root=TRT_ROOT,
            plugin_path=plugin,
            fixed_k=int(spec["fixed_k"]),
            num_frames=FRAMES[protocol],
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
            plugin_path=plugin,
            eval_manifest_path=manifest,
            physical_gpu_id=physical_gpu,
            fixed_k=int(spec["fixed_k"]),
            max_agents=2,
            num_frames=FRAMES[protocol],
            warmup_frames=warmup,
            latency_rounds=1,
            dataloader_num_workers=8,
        )
    accepted = (
        result.get("status") == "ok"
        and int(result.get("num_evaluated_frames", -1)) == FRAMES[protocol]
        and int(result.get("num_skipped_frames", -1)) == 0
        and bool(result.get("reset_after_warmup", False))
    )
    structure = json.loads((output_root / "structures" / model_name / family_id / f"dh_{int(d_h):03d}" / "structure_result.json").read_text(encoding="utf-8"))
    summary = {
        "status": "ok" if accepted else "evaluation_failed",
        "model": model_name,
        "family_id": family_id,
        "d_h": int(d_h),
        "profile": profile,
        "protocol": protocol,
        "evaluated": int(result.get("num_evaluated_frames", 0)),
        "skipped": int(result.get("num_skipped_frames", 0)),
        "AP@0.3": result.get("AP@0.3"),
        "AP@0.5": result.get("AP@0.5"),
        "AP@0.7": result.get("AP@0.7"),
        "mAP": result.get("mAP"),
        "forward_p50_ms": result.get("forward_p50_ms"),
        "manifest_hash": str(manifest_payload["manifest_hash"]),
        "manifest_sha256": _sha256(manifest),
        "engine_sha256": _sha256(engine),
        "structure_hash": structure["structure_hash"],
        "scale_hash": build.get("scale_hash", "not_quantized"),
        "workers": 8,
        "ap_iou_backend": "gpu",
        "reset_after_warmup": result.get("reset_after_warmup"),
        "failure_reason": result.get("failure_reason", ""),
    }
    _write(destination / "evaluation_acceptance.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--model", choices=tuple(MODEL_SPECS), required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--dh", type=int, required=True)
    parser.add_argument("--profile", choices=("P32", "P16", "P8"), required=True)
    parser.add_argument("--protocol", choices=tuple(FRAMES), required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--plugin", required=True)
    args = parser.parse_args(argv)
    result = evaluate(
        output_root=Path(args.output_root).resolve(), model_name=args.model,
        family_id=args.family, d_h=args.dh, profile=args.profile,
        protocol=args.protocol, physical_gpu=args.physical_gpu,
        plugin=Path(args.plugin).resolve(),
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
