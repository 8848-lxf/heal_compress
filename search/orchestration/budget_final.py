"""Fixed-manifest reevaluation of per-generation Stage-2 winners."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Callable, Iterable

from ..integration.data_provider import load_split_frame_ids, write_eval_manifest
from ..integration.evaluation_provider import evaluate_engine_modelopt
from ..integration.runtime_environment import require_gpu_isolation
from ..stage2.objective import Stage2ObjectiveConfig, compute_stage2_score


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: (
                        json.dumps(row.get(field), sort_keys=True, default=str)
                        if isinstance(row.get(field), (dict, list, tuple))
                        else row.get(field, "")
                    )
                    for field in fields
                }
            )


def _require_complete_evaluation(
    evaluation: dict[str, Any],
    *,
    expected_frames: int,
    label: str,
) -> None:
    evaluated = int(evaluation.get("num_evaluated_frames", -1))
    skipped = int(evaluation.get("num_skipped_frames", -1))
    if (
        str(evaluation.get("status", "")) != "ok"
        or evaluated != int(expected_frames)
        or skipped != 0
    ):
        raise RuntimeError(
            "budget_final_evaluation_incomplete:"
            f"{label}:status={evaluation.get('status')}:"
            f"evaluated={evaluated}:skipped={skipped}"
        )


def _engine_path(winner: dict[str, Any]) -> Path:
    explicit = winner.get("engine_path")
    if explicit:
        path = Path(str(explicit))
    else:
        path = Path(str(winner.get("artifact_dir", ""))) / "engine.plan"
    if not path.is_file():
        raise RuntimeError(f"budget_final_engine_missing:{path}")
    return path


def _manifest_overlap(stage2_manifest_path: Path, final_ids: list[str]) -> dict[str, Any]:
    stage2_ids: list[str] = []
    if stage2_manifest_path.is_file():
        payload = json.loads(stage2_manifest_path.read_text(encoding="utf-8"))
        stage2_ids = [str(value) for value in payload.get("evaluation_frame_ids", [])]
    overlap = sorted(set(stage2_ids).intersection(final_ids))
    return {
        "stage2_evaluation_count": len(stage2_ids),
        "budget_final_evaluation_count": len(final_ids),
        "overlap_count": len(overlap),
        "overlap_ratio": len(overlap) / max(1, len(final_ids)),
        "overlap_frame_ids": overlap,
    }


def run_budget_final_evaluation(
    *,
    context: Any,
    run_dir: str | Path,
    generation_winners: Iterable[dict[str, Any]],
    config: dict[str, Any],
    budget: float,
    available_frame_ids: Iterable[str] | None = None,
    evaluate_fn: Callable[..., dict[str, Any]] = evaluate_engine_modelopt,
    gpu_isolation_fn: Callable[..., dict[str, Any]] = require_gpu_isolation,
) -> dict[str, Any]:
    """Evaluate all generation winners on one deterministic final manifest."""

    root = Path(run_dir)
    rows = [dict(row) for row in generation_winners]
    if not rows:
        raise RuntimeError("budget_final_requires_generation_winners")
    num_frames = int(config.get("num_frames", 500))
    warmup_frames = int(config.get("warmup_frames", 30))
    latency_rounds = int(config.get("latency_rounds", config.get("rounds", 1)))
    reset_after_warmup = bool(config.get("reset_after_warmup", True))
    evaluation_offset = int(config.get("evaluation_offset", 0))
    available = (
        [str(value) for value in available_frame_ids]
        if available_frame_ids is not None
        else load_split_frame_ids(
            context.model_bundle.adapter,
            context.model_config,
            split="val",
        )
    )
    manifest = write_eval_manifest(
        root / "manifests" / f"budget_final_{num_frames}.json",
        num_frames=num_frames,
        warmup_frames=warmup_frames,
        available_frame_ids=available,
        reset_after_warmup=reset_after_warmup,
        evaluation_offset=evaluation_offset,
    )
    manifest_payload = json.loads(manifest.path.read_text(encoding="utf-8"))
    final_ids = [str(value) for value in manifest_payload["evaluation_frame_ids"]]
    overlap = _manifest_overlap(Path(context.eval_manifest_path), final_ids)
    output_dir = root / "budget_final"

    def evaluate(engine_path: Path, destination: Path) -> dict[str, Any]:
        gpu_isolation_fn(
            context.physical_gpu_id,
            report_path=destination / "gpu_preflight.json",
        )
        result = evaluate_fn(
            engine_path=engine_path,
            checkpoint=context.checkpoint_path,
            model_config=context.model_config,
            heal_root="/home/lixingfeng/UniAD_examine/HEAL",
            device=context.runtime_device,
            output_dir=destination,
            tensorrt_root=context.tensorrt.tensorrt_root,
            plugin_path=context.tensorrt.plugin_path,
            num_frames=num_frames,
            warmup_frames=warmup_frames,
            latency_rounds=latency_rounds,
            conda_env=context.tensorrt.conda_env,
            eval_manifest_path=manifest.path,
        )
        gpu_isolation_fn(
            context.physical_gpu_id,
            report_path=destination / "gpu_postflight.json",
        )
        _require_complete_evaluation(
            result,
            expected_frames=num_frames,
            label=str(destination.name),
        )
        return result

    references: dict[str, dict[str, Any]] = {}
    for precision in ("strict_fp32", "strict_fp16"):
        engine = root / "baselines" / f"original_{precision}" / "engine.plan"
        if not engine.is_file():
            raise RuntimeError(f"budget_final_reference_engine_missing:{precision}:{engine}")
        references[precision] = evaluate(
            engine,
            output_dir / "references" / precision,
        )
    objective = Stage2ObjectiveConfig(
        eta_map=float(config.get("eta_ap", config.get("eta_map", 0.8))),
        eta_latency=float(config.get("eta_latency", 0.2)),
        latency_metric=str(config.get("latency_metric", "forward_p50_ms")),
        tau_ap=config.get("tau_ap"),
        max_map_drop=config.get("max_map_drop"),
    )
    baseline = {
        "mAP": references["strict_fp32"]["mAP"],
        objective.latency_metric: references["strict_fp16"][objective.latency_metric],
    }
    evaluated: list[dict[str, Any]] = []
    by_identity: dict[tuple[str, str], dict[str, Any]] = {}
    for winner in rows:
        generation = int(winner.get("generation", 0))
        identity = (
            str(winner.get("physical_hash", "")),
            str(winner.get("deployment_hash", "")),
        )
        if not all(identity):
            raise RuntimeError(f"budget_final_deployment_identity_missing:generation={generation}")
        reused = by_identity.get(identity)
        if reused is None:
            evaluation = evaluate(
                _engine_path(winner),
                output_dir / "candidates" / f"generation_{generation:03d}",
            )
            scored = compute_stage2_score(evaluation, baseline=baseline, config=objective)
            if not math.isfinite(float(scored.get("F2", float("inf")))):
                raise RuntimeError(f"budget_final_stage2_score_invalid:generation={generation}")
            forward_p50 = float(evaluation.get("forward_p50_ms", 0.0) or 0.0)
            row = {
                **winner,
                **evaluation,
                **scored,
                "budget": float(budget),
                "generation": generation,
                "evaluated": int(evaluation["num_evaluated_frames"]),
                "skipped": int(evaluation["num_skipped_frames"]),
                "FPS": 1000.0 / forward_p50 if forward_p50 > 0.0 else 0.0,
                "manifest_hash": manifest.manifest_hash,
            }
            by_identity[identity] = dict(row)
        else:
            row = {
                **reused,
                "generation": generation,
                "candidate_hash": winner.get("candidate_hash", ""),
                "reused_from_generation": int(reused["generation"]),
            }
        evaluated.append(row)
    winner = min(
        evaluated,
        key=lambda row: (float(row["F2"]), int(row["generation"]), str(row.get("candidate_hash", ""))),
    )
    budget_tag = int(round(float(budget) * 100.0))
    report = {
        "status": "ok",
        "budget": float(budget),
        "manifest": manifest.to_dict(),
        "manifest_overlap": overlap,
        "references": references,
        "candidate_count": len(evaluated),
        "unique_deployment_count": len(by_identity),
        "candidates": evaluated,
        "winner": winner,
    }
    _write_json(output_dir / "budget_final_summary.json", report)
    _write_json(root / f"budget_{budget_tag:03d}_winner.json", winner)
    _write_csv(root / f"budget_final_{num_frames}frames.csv", evaluated)
    return report
