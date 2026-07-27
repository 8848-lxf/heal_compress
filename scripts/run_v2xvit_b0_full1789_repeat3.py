#!/usr/bin/env python3
"""Evaluate one strict V2X-ViT B0 engine three times on full1789."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from search.model_family.evaluation import evaluate_v2xvit_engine_modelopt


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(args: argparse.Namespace) -> int:
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    request = _read(args.request_json.resolve())
    manifest = _read(args.full_manifest.resolve())
    evaluation_ids = tuple(str(value) for value in manifest["evaluation_frame_ids"])
    warmup_ids = tuple(str(value) for value in manifest["warmup_frame_ids"])
    if len(evaluation_ids) != 1789 or len(set(evaluation_ids)) != 1789:
        raise RuntimeError("b0_full1789_manifest_invalid")
    if len(warmup_ids) < 200:
        raise RuntimeError("b0_full1789_warmup_invalid")
    engine = args.b0_engine.resolve()
    if not engine.is_file():
        raise RuntimeError(f"b0_engine_missing:{engine}")
    progress_path = root / "reports/progress.json"
    results: list[dict[str, Any]] = []
    for repeat in range(1, 4):
        destination = root / f"evaluation_full1789/repeat_{repeat}/B0"
        result_path = destination / "evaluation.json"
        if result_path.is_file():
            result = _read(result_path)
        else:
            result = evaluate_v2xvit_engine_modelopt(
                engine_path=engine,
                model_config=request["model_config"],
                heal_root=request["heal_root"],
                output_dir=destination,
                tensorrt_root=args.tensorrt_root.resolve(),
                plugin_path=request["plugin_path"],
                eval_manifest_path=args.full_manifest.resolve(),
                physical_gpu_id=int(args.physical_gpu),
                fixed_k=int(request["fixed_k"]),
                max_agents=int(request["max_agents"]),
                num_frames=1789,
                warmup_frames=200,
                latency_rounds=1,
                dataloader_num_workers=8,
            )
        if not (
            result.get("status") == "ok"
            and int(result.get("num_evaluated_frames", -1)) == 1789
            and int(result.get("num_skipped_frames", -1)) == 0
            and str(result.get("eval_manifest_hash"))
            == str(manifest.get("manifest_hash"))
        ):
            raise RuntimeError(f"b0_full1789_gate_failed:repeat={repeat}")
        row = {
            "repeat": repeat,
            "AP@0.3": float(result["AP@0.3"]),
            "AP@0.5": float(result["AP@0.5"]),
            "AP@0.7": float(result["AP@0.7"]),
            "mAP": float(result["mAP"]),
            "forward_p50_ms": float(result["forward_p50_ms"]),
            "evaluated": 1789,
            "skipped": 0,
        }
        previous = {int(value["repeat"]): value for value in results}
        previous[repeat] = row
        results = [previous[key] for key in sorted(previous)]
        _write(progress_path, {
            "status": "running",
            "completed_evaluations": len(results),
            "total_evaluations": 3,
            "results": results,
        })
        print(json.dumps({"event": "b0_full1789_complete", **row}, sort_keys=True), flush=True)
    summary: dict[str, Any] = {
        "status": "complete",
        "engine_path": str(engine),
        "engine_sha256": _sha256(engine),
        "manifest_path": str(args.full_manifest.resolve()),
        "manifest_hash": manifest.get("manifest_hash"),
        "repetitions": results,
        "evaluated_per_repeat": 1789,
        "skipped_total": 0,
    }
    for metric in ("AP@0.3", "AP@0.5", "AP@0.7", "mAP", "forward_p50_ms"):
        values = [float(row[metric]) for row in results]
        summary[f"{metric}_mean"] = statistics.fmean(values)
        summary[f"{metric}_std"] = statistics.stdev(values)
    _write(root / "reports/b0_full1789_repeat3.json", summary)
    _write(progress_path, {
        "status": "complete", "completed_evaluations": 3,
        "total_evaluations": 3,
    })
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--b0-engine", type=Path, required=True)
    parser.add_argument("--request-json", type=Path, required=True)
    parser.add_argument("--full-manifest", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument(
        "--tensorrt-root",
        type=Path,
        default=Path("/home/lixingfeng/UniAD_examine/TensorRT-10.9_x86_cu118"),
    )
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
