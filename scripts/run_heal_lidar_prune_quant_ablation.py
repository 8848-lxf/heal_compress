#!/usr/bin/env python3
"""Prepare and build F-Cooper/DiscoNet P/Q ablation engines.

Engine construction and evaluation are intentionally separate.  This script
only prepares immutable phenotype specifications or builds one unique P-only /
Q-only deployment.  Accepted joint engines are referenced read-only.  A later
evaluation-only runner can therefore schedule fresh full-validation work without
ever invoking an engine builder on the same GPU.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from search.ablation.heal_lidar_prune_quant import (  # noqa: E402
    build_family_ablation_matrix,
    collect_authoritative_family_candidates,
)
from search.candidate import CandidatePhenotype  # noqa: E402
from search.integration.heal_lidar_baseline_context import (  # noqa: E402
    build_heal_lidar_baseline_context,
)
from search.stage2.heal_lidar_baseline_real_evaluator import (  # noqa: E402
    HealLidarBaselineCandidateEvaluator,
)


def _read_json(path: str | Path) -> dict[str, Any]:
    return dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _write_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    temporary.replace(destination)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_config(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"heal_lidar_ablation_config_invalid:{path}")
    return dict(payload)


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _prepare(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = args.run_dir.resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError(f"heal_lidar_ablation_run_dir_not_empty:{run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    config = _load_config(args.config)
    family_id = str(dict(config.get("model") or {}).get("family_id", ""))
    sources = collect_authoritative_family_candidates(
        family_id=family_id,
        ga_root=args.ga_root,
        greedy_root=args.greedy_root,
        repository_root=REPO_ROOT,
        tolerance=float(args.bops_tolerance),
    )
    matrix = build_family_ablation_matrix(sources)
    serialized_rows = []
    for row in matrix:
        payload = dict(row)
        phenotype = payload.pop("phenotype")
        phenotype_path = run_dir / "candidate_specs" / f"{row['row_id']}.json"
        _write_json(phenotype_path, phenotype)
        signature = str(row["deployment_config_signature"])
        if row["variant"] == "prune_quant":
            artifact_dir = Path(str(row["source_artifact_dir"])).resolve()
        else:
            artifact_dir = (run_dir / "unique_deployments" / signature).resolve()
        payload.update(
            {
                "phenotype_path": str(phenotype_path.resolve()),
                "artifact_dir": str(artifact_dir),
                "evaluation_output_policy": "fresh_evaluation_only_directory",
            }
        )
        serialized_rows.append(payload)
    manifest = {
        "schema_version": "heal-lidar-family-prune-quant-ablation-v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "run_dir": str(run_dir),
        "family_id": family_id,
        "config_path": str(args.config.resolve()),
        "config_sha256": _sha256(args.config),
        "ga_root": str(args.ga_root.resolve()),
        "greedy_root": str(args.greedy_root.resolve()),
        "protocol": {
            "variants": ["prune_quant", "prune_only", "quant_only"],
            "prune_quant": "reuse_exact_accepted_joint_engine_read_only",
            "prune_only": "exact_source_pruning_plus_strict_fp32",
            "quant_only": "original_all_keep_plus_exact_source_precision_contract",
            "engine_dedup_key": "deployment_config_signature",
            "fresh_calibration_for_new_int8_deployments": True,
            "engine_build_and_evaluation_separated": True,
            "joint_engines_rebuilt": False,
            "all_rows_require_fresh_evaluation": True,
        },
        "logical_row_count": len(serialized_rows),
        "unique_new_engine_build_count": sum(
            bool(row["requires_engine_build"]) for row in serialized_rows
        ),
        "rows": serialized_rows,
    }
    _write_json(run_dir / "ablation_manifest.json", manifest)
    return manifest


def _build_context(
    args: argparse.Namespace,
    *,
    config: dict[str, Any],
    row_id: str,
) -> Any:
    model = dict(config.get("model") or {})
    runtime = dict(config.get("runtime") or {})
    proxy = dict(config.get("proxy") or {})
    pruning = dict(config.get("pruning") or {})
    full = dict(config.get("full_validation") or {})
    return build_heal_lidar_baseline_context(
        family_id=str(model["family_id"]),
        checkpoint_path=model["checkpoint"],
        model_config_path=model["config"],
        output_dir=args.run_dir / "build_contexts" / row_id,
        heal_root=runtime["heal_root"],
        tensorrt_root=runtime["tensorrt_root"],
        plugin_path=None,
        gpu_id=str(int(args.gpu_id)),
        exclude_gpu_ids=[],
        tensorrt_env=str(runtime.get("tensorrt_env", "modelopt")),
        fisher_calibration_batches=int(proxy.get("fisher_calibration_batches", 8)),
        quant_calibration_batches=int(proxy.get("quant_calibration_batches", 200)),
        quant_calibration_npz_manifest=None,
        quant_activation_calibration_backend=str(
            proxy.get(
                "quant_activation_calibration_backend",
                "modelopt_histogram_entropy",
            )
        ),
        quant_calibration_force_rebuild=True,
        num_frames=int(full.get("num_frames", 1789)),
        warmup_frames=int(full.get("warmup_frames", 200)),
        reset_after_warmup=True,
        default_precision=str(dict(config.get("precision") or {}).get("default", "FP32")),
        max_agents=int(model.get("max_agents", 2)),
        minimum_retained_ratio=float(pruning.get("minimum_retained_ratio", 0.10)),
        dense_alignment=int(pruning.get("dense_channel_alignment", 4)),
    )


def _accepted_existing_owner_build(row: dict[str, Any]) -> dict[str, Any] | None:
    artifact_dir = Path(row["artifact_dir"])
    result_path = artifact_dir / "candidate_build_result.json"
    if result_path.is_file():
        existing = _read_json(result_path)
        engine = Path(str(existing.get("engine_path", "")))
        if (
            existing.get("status") == "ok"
            and engine.is_file()
            and _sha256(engine) == str(existing.get("engine_sha256", ""))
        ):
            return {**existing, "cache_hit": True, "engine_rebuilt": False}
        raise RuntimeError(f"heal_lidar_ablation_incomplete_build_exists:{artifact_dir}")
    if artifact_dir.exists() and any(artifact_dir.iterdir()):
        raise RuntimeError(f"heal_lidar_ablation_artifact_dir_not_empty:{artifact_dir}")
    return None


def _new_evaluator(
    args: argparse.Namespace,
    *,
    config: dict[str, Any],
    context: Any,
    runtime_id: str,
) -> HealLidarBaselineCandidateEvaluator:
    baselines = dict(config.get("baselines") or {})
    full = dict(config.get("full_validation") or {})
    return HealLidarBaselineCandidateEvaluator(
        context=context,
        run_dir=args.run_dir / "build_runtime" / runtime_id,
        baseline_engine_path=baselines["strict_fp32_engine"],
        num_frames=int(full.get("num_frames", 1789)),
        warmup_frames=int(full.get("warmup_frames", 200)),
        latency_rounds=int(full.get("latency_rounds", 3)),
        dataloader_num_workers=8,
    )


def _build_owner(
    row: dict[str, Any],
    *,
    evaluator: HealLidarBaselineCandidateEvaluator,
) -> dict[str, Any]:
    existing = _accepted_existing_owner_build(row)
    if existing is not None:
        return existing
    artifact_dir = Path(row["artifact_dir"])
    phenotype = CandidatePhenotype.from_dict(_read_json(row["phenotype_path"]))
    result = evaluator.build_candidate_artifacts(
        phenotype,
        output_dir=artifact_dir,
        candidate_hash=str(row["deployment_config_signature"]),
    )
    if bool(result.get("evaluation_invoked", True)):
        raise RuntimeError("heal_lidar_ablation_build_phase_invoked_evaluation")
    return {**result, "cache_hit": False, "engine_rebuilt": True}


def _build_one(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = args.run_dir / "ablation_manifest.json"
    manifest = _read_json(manifest_path)
    matches = [row for row in manifest["rows"] if row["row_id"] == args.row_id]
    if len(matches) != 1:
        raise RuntimeError(f"heal_lidar_ablation_row_not_unique:{args.row_id}")
    row = dict(matches[0])
    if row["variant"] == "prune_quant" or not bool(row["requires_engine_build"]):
        raise RuntimeError(f"heal_lidar_ablation_row_is_not_build_owner:{args.row_id}")
    existing = _accepted_existing_owner_build(row)
    if existing is not None:
        return existing
    config = _load_config(args.config)
    context = _build_context(args, config=config, row_id=args.row_id)
    evaluator = _new_evaluator(
        args, config=config, context=context, runtime_id=args.row_id
    )
    return _build_owner(row, evaluator=evaluator)


def _build_all(args: argparse.Namespace) -> dict[str, Any]:
    """Serially build every unique missing owner with one model/context load."""

    manifest = _read_json(args.run_dir / "ablation_manifest.json")
    owners = [
        dict(row)
        for row in manifest["rows"]
        if row["variant"] != "prune_quant" and bool(row["requires_engine_build"])
    ]
    if len({str(row["row_id"]) for row in owners}) != len(owners):
        raise RuntimeError("heal_lidar_ablation_duplicate_build_owner_row")
    progress_path = args.run_dir / f"build_all_gpu_{int(args.gpu_id)}_progress.json"
    records: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for row in owners:
        existing = _accepted_existing_owner_build(row)
        if existing is None:
            missing.append(row)
        else:
            records.append(
                {
                    "row_id": row["row_id"],
                    "status": "accepted_resume_hit",
                    "engine_path": existing["engine_path"],
                    "engine_sha256": existing["engine_sha256"],
                    "cache_hit": True,
                }
            )
    progress: dict[str, Any] = {
        "schema_version": "heal-lidar-family-ablation-build-all-progress-v1",
        "family_id": manifest["family_id"],
        "gpu_id": int(args.gpu_id),
        "same_gpu_serial": True,
        "evaluation_invoked": False,
        "owner_count": len(owners),
        "initial_resume_hit_count": len(records),
        "records": records,
        "status": "complete" if not missing else "building",
        "updated_at": datetime.now().astimezone().isoformat(),
    }
    _write_json(progress_path, progress)
    if not missing:
        return progress

    config = _load_config(args.config)
    context = _build_context(
        args,
        config=config,
        row_id=f"build_all_gpu_{int(args.gpu_id)}",
    )
    evaluator = _new_evaluator(
        args,
        config=config,
        context=context,
        runtime_id=f"build_all_gpu_{int(args.gpu_id)}",
    )
    for row in missing:
        started_at = datetime.now().astimezone().isoformat()
        try:
            result = _build_owner(row, evaluator=evaluator)
            record = {
                "row_id": row["row_id"],
                "status": "ok",
                "started_at": started_at,
                "finished_at": datetime.now().astimezone().isoformat(),
                "engine_path": result["engine_path"],
                "engine_sha256": result["engine_sha256"],
                "cache_hit": bool(result["cache_hit"]),
                "evaluation_invoked": bool(result.get("evaluation_invoked", True)),
            }
            if record["evaluation_invoked"]:
                raise RuntimeError("heal_lidar_ablation_build_all_invoked_evaluation")
            records.append(record)
        except Exception as exc:
            records.append(
                {
                    "row_id": row["row_id"],
                    "status": "failed",
                    "started_at": started_at,
                    "finished_at": datetime.now().astimezone().isoformat(),
                    "failure_reason": f"{type(exc).__name__}:{exc}",
                }
            )
            progress.update(
                {
                    "records": records,
                    "status": "failed",
                    "completed_owner_count": sum(
                        record["status"] in {"ok", "accepted_resume_hit"}
                        for record in records
                    ),
                    "updated_at": datetime.now().astimezone().isoformat(),
                }
            )
            _write_json(progress_path, progress)
            raise
        progress.update(
            {
                "records": records,
                "status": "building",
                "completed_owner_count": sum(
                    record["status"] in {"ok", "accepted_resume_hit"}
                    for record in records
                ),
                "updated_at": datetime.now().astimezone().isoformat(),
            }
        )
        _write_json(progress_path, progress)
    progress.update(
        {
            "status": "complete",
            "completed_owner_count": len(owners),
            "updated_at": datetime.now().astimezone().isoformat(),
        }
    )
    _write_json(progress_path, progress)
    return progress


def _finalize(args: argparse.Namespace) -> dict[str, Any]:
    manifest = _read_json(args.run_dir / "ablation_manifest.json")
    owner_results: dict[str, dict[str, Any]] = {}
    inventory: list[dict[str, Any]] = []
    for row in manifest["rows"]:
        if row["variant"] == "prune_quant":
            engine = Path(row["source_joint_engine_path"])
            digest = _sha256(engine)
            if digest != str(row["source_joint_engine_sha256"]):
                raise RuntimeError(
                    f"heal_lidar_joint_engine_changed:{row['row_id']}:{digest}"
                )
            resolved = {
                "engine_path": str(engine.resolve()),
                "engine_sha256": digest,
                "engine_origin": "accepted_source_joint_engine",
                "engine_built_by_row_id": None,
            }
        else:
            owner = str(row["build_owner_row_id"])
            if owner not in owner_results:
                owner_row = next(
                    candidate for candidate in manifest["rows"]
                    if candidate["row_id"] == owner
                )
                result = _read_json(
                    Path(owner_row["artifact_dir"]) / "candidate_build_result.json"
                )
                if result.get("status") != "ok" or result.get("evaluation_invoked") is not False:
                    raise RuntimeError(f"heal_lidar_ablation_owner_build_not_accepted:{owner}")
                engine = Path(result["engine_path"])
                if _sha256(engine) != str(result["engine_sha256"]):
                    raise RuntimeError(f"heal_lidar_ablation_owner_engine_changed:{owner}")
                owner_results[owner] = result
            result = owner_results[owner]
            resolved = {
                "engine_path": str(Path(result["engine_path"]).resolve()),
                "engine_sha256": result["engine_sha256"],
                "engine_origin": "fresh_ablation_build",
                "engine_built_by_row_id": owner,
            }
        inventory.append(
            {
                **row,
                **resolved,
                "requires_fresh_evaluation": True,
                "evaluation_completed": False,
            }
        )
    payload = {
        "schema_version": "heal-lidar-family-ablation-engine-inventory-v1",
        "family_id": manifest["family_id"],
        "logical_row_count": len(inventory),
        "unique_new_engine_build_count": len(owner_results),
        "joint_engine_build_count": 0,
        "evaluation_count": 0,
        "rows": inventory,
    }
    _write_json(args.run_dir / "engine_inventory.json", payload)
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--action",
        choices=("prepare", "build-one", "build-all", "finalize"),
        required=True,
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ga-root", type=Path)
    parser.add_argument("--greedy-root", type=Path)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--row-id", default="")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--bops-tolerance", type=float, default=0.005)
    args = parser.parse_args()
    args.config = args.config.resolve()
    args.run_dir = args.run_dir.resolve()
    if args.action == "prepare" and (args.ga_root is None or args.greedy_root is None):
        parser.error("prepare requires --ga-root and --greedy-root")
    if args.action == "build-one" and not args.row_id:
        parser.error("build-one requires --row-id")
    return args


def main() -> int:
    args = _parse_args()
    if args.action == "prepare":
        result = _prepare(args)
    elif args.action == "build-one":
        result = _build_one(args)
    elif args.action == "build-all":
        result = _build_all(args)
    else:
        result = _finalize(args)
    print(
        json.dumps(
            {
                "status": "ok",
                "action": args.action,
                "run_dir": str(args.run_dir),
                "row_count": len(result.get("rows", [])),
                "engine_path": result.get("engine_path"),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
