"""Resumable CoBEVT S0--S4 × precision/SmoothQuant/F3-LUT experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping
from zoneinfo import ZoneInfo

import torch

from search.model_families.lidar_cobevt.attention_dim_pruning import AttentionDimMask
from search.model_families.lidar_cobevt.attention_taylor import keep_indices_from_scores
from search.model_families.lidar_cobevt.minimal_structure_quant_latency import (
    cross_precision_contracts,
    mandatory_structure_profiles,
    structure_precision_interaction,
)
from search.orchestration.lidar_cobevt_attention_pruning import (
    AttentionCandidateSpec,
    candidate_engine_directory,
    materialize_candidate,
    run_export_build,
    run_structure_smoke,
    write_attention_masks,
)


DEFAULT_CHECKPOINT = Path(
    "/home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/"
    "LiDAROnly/lidar_cobevt/net_epoch_bestval_at19.pth"
)
DEFAULT_CONFIG = DEFAULT_CHECKPOINT.with_name("config.yaml")
DEFAULT_HEAL_ROOT = Path("/home/lixingfeng/UniAD_examine/HEAL")
DEFAULT_PLUGIN = Path(
    "/home/lixingfeng/UniAD_examine/heal_compress/quantization/plugins/"
    "pointpillar_scatter_trt/build/libpointpillar_scatter_trt.so"
)
DEFAULT_RANKING_SOURCE = Path(
    "/data/lxf/heal_data/outputs/cobevt_attention_dim_pruning_20260718_085644"
)
DEFAULT_BOUNDARY_SOURCE = Path(
    "/data/lxf/heal_data/outputs/cobevt_attention_fp16_boundary_audit_20260719_030621"
)
PROFILE_NAMES = (
    "P0_rest_fp16_attention_fp32",
    "F3_rest_fp16_qk_fp32_minimal_island",
)


def resolve_tensorrt_root_from_history(history_output: str | Path) -> Path:
    """Resolve the exact accepted TensorRT root; never guess a sibling path."""

    manifest_path = Path(history_output) / "run_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"accepted_tensorrt_manifest_missing:{manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    environment = payload.get("environment", {})
    root_value = environment.get("resolved_tensorrt_root") or environment.get(
        "tensorrt_root"
    )
    if not root_value:
        raise RuntimeError("accepted_tensorrt_root_missing")
    root = Path(str(root_value)).expanduser().resolve()
    trtexec = root / "targets/x86_64-linux-gnu/bin/trtexec"
    if not trtexec.is_file():
        raise RuntimeError(f"accepted_trtexec_missing:{trtexec}")
    return root


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[2],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({str(key) for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: json.dumps(row.get(field), sort_keys=True)
                    if isinstance(row.get(field), (dict, list, tuple))
                    else row.get(field, "")
                    for field in fields
                }
            )


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _manifest_record(path: Path) -> dict[str, Any]:
    payload = dict(_read(path))
    payload["path"] = str(path)
    return payload


def _mask_from_ranking(
    ranking: Mapping[str, Any], *, d_qk: int, d_v: int
) -> dict[str, AttentionDimMask]:
    masks: dict[str, AttentionDimMask] = {}
    for module_name, row in sorted(ranking.items()):
        qk_scores = row.get("qk_by_head")
        vo_scores = row.get("vo_by_head")
        if not qk_scores or not vo_scores:
            raise RuntimeError(f"attention_ranking_scores_missing:{module_name}")
        masks[str(module_name)] = AttentionDimMask(
            keep_indices_from_scores(qk_scores, keep_width=int(d_qk)),
            keep_indices_from_scores(vo_scores, keep_width=int(d_v)),
            original_d_qk=32,
            original_d_v=32,
        )
    if len(masks) != 6:
        raise RuntimeError(f"attention_ranking_block_count:{len(masks)}")
    return masks


def prepare_experiment(
    output_dir: Path,
    *,
    checkpoint: Path,
    config: Path,
    ranking_source: Path,
    boundary_source: Path,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError("minimal_experiment_output_not_fresh")
    output_dir.mkdir(parents=True, exist_ok=True)
    ranking_manifest = dict(
        _read(ranking_source / "attention_mean_gradients.manifest.json")
    )
    if _sha256(checkpoint) != str(ranking_manifest["checkpoint_hash"]):
        raise RuntimeError("ranking_checkpoint_hash_mismatch")
    if _sha256(config) != str(ranking_manifest["config_hash"]):
        raise RuntimeError("ranking_config_hash_mismatch")
    ranking = dict(_read(ranking_source / "qk_rankings.json"))
    if ranking != _read(ranking_source / "vo_rankings.json"):
        raise RuntimeError("qk_vo_ranking_artifact_divergence")

    structure_dir = output_dir / "structure_experiment"
    masks_dir = structure_dir / "candidate_masks"
    masks_dir.mkdir(parents=True)
    candidates: list[dict[str, Any]] = []
    for profile in mandatory_structure_profiles():
        masks = _mask_from_ranking(ranking, d_qk=profile.d_qk, d_v=profile.d_v)
        mask_path = masks_dir / f"{profile.profile_id}.json"
        write_attention_masks(mask_path, masks)
        candidates.append(
            {
                **AttentionCandidateSpec(
                    candidate_id=profile.profile_id,
                    experiment=profile.family,
                    variant=f"d_qk_{profile.d_qk}_d_v_{profile.d_v}",
                    d_qk=profile.d_qk,
                    d_v=profile.d_v,
                    embed_dim=profile.embed_dim,
                    heads=profile.heads,
                ).to_dict(),
                "mask_path": str(mask_path),
                "mask_sha256": _sha256(mask_path),
                "structure_profile_hash": profile.structure_hash,
            }
        )

    manifests_dir = output_dir / "manifests"
    manifests_dir.mkdir()
    manifest_records: dict[str, dict[str, Any]] = {}
    for name in ("smoke10", "fixed50", "fixed500"):
        source = boundary_source / "manifests" / f"{name}_manifest.json"
        destination = manifests_dir / source.name
        shutil.copy2(source, destination)
        if _sha256(source) != _sha256(destination):
            raise RuntimeError(f"manifest_copy_hash_mismatch:{name}")
        manifest_records[name] = _manifest_record(destination)

    fixed_k_contract = dict(_read(ranking_source / "fixed_k_contract.json"))
    if int(fixed_k_contract["fixed_k"]) != 29696:
        raise RuntimeError("accepted_fixed_k_not_29696")
    p0, f3 = cross_precision_contracts()
    experiment = {
        "candidates": candidates,
        "fixed_k": 29696,
        "fixed_k_contract": fixed_k_contract,
        "fixed_k_validated": True,
        "gradient_samples": int(ranking_manifest["sample_count"]),
        "gradient_statistics_manifest_hash": ranking_manifest["manifest_hash"],
        "manifests": manifest_records,
        "precision_profiles": [p0.to_dict(), f3.to_dict()],
        "protocol": {
            "ap_iou_backend": "gpu",
            "dataloader_workers": 8,
            "fixed_k": 29696,
            "no_tf32": True,
            "strongly_typed": True,
        },
    }
    _write_json(output_dir / "experiment_config.json", experiment)
    _write_json(structure_dir / "structure_profiles.json", candidates)
    _write_json(structure_dir / "attention_pruning_ranking.json", ranking)
    _write_json(structure_dir / "attention_retained_indices.json", {
        row["candidate_id"]: _read(Path(row["mask_path"])) for row in candidates
    })
    _write_json(structure_dir / "ranking_provenance.json", ranking_manifest)
    _write_json(output_dir / "checkpoint_manifest.json", {
        "path": str(checkpoint), "sha256": _sha256(checkpoint)
    })
    _write_json(output_dir / "dataset_manifest.json", {
        name: {
            "path": row["path"],
            "manifest_hash": row["manifest_hash"],
            "num_frames": row["num_frames"],
        }
        for name, row in manifest_records.items()
    })
    resolved_trt_root = resolve_tensorrt_root_from_history(boundary_source)
    run_manifest = {
        "branch": subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=Path(__file__).resolve().parents[2],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        ).stdout.strip(),
        "code_commit": _git_commit(),
        "config": str(config),
        "config_sha256": _sha256(config),
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "ranking_source": str(ranking_source),
        "ranking_source_manifest_hash": ranking_manifest["manifest_hash"],
        "boundary_source": str(boundary_source),
        "tensorrt_root": str(resolved_trt_root),
        "trtexec_sha256": _sha256(
            resolved_trt_root / "targets/x86_64-linux-gnu/bin/trtexec"
        ),
        "schema_version": "cobevt-minimal-structure-quant-latency-v1",
    }
    _write_json(output_dir / "run_manifest.json", run_manifest)
    return {"candidate_count": len(candidates), "output_dir": str(output_dir)}


def evaluate_a(
    output_dir: Path,
    *,
    checkpoint: Path,
    config: Path,
    heal_root: Path,
    trt_root: Path,
    plugin: Path,
    physical_gpu: int,
    frames: int,
    candidate_ids: Iterable[str] = (),
) -> dict[str, Any]:
    from search.integration.lidar_cobevt_evaluation_provider import (
        evaluate_cobevt_engine_modelopt,
    )

    experiment = dict(_read(output_dir / "experiment_config.json"))
    if frames not in {10, 50, 500}:
        raise ValueError("unsupported_minimal_evaluation_frame_count")
    selected = set(candidate_ids)
    manifest_name = {10: "smoke10", 50: "fixed50", 500: "fixed500"}[frames]
    manifest_path = Path(experiment["manifests"][manifest_name]["path"])
    rows: list[dict[str, Any]] = []
    for record in experiment["candidates"]:
        candidate_id = str(record["candidate_id"])
        if selected and candidate_id not in selected:
            continue
        for profile_name in PROFILE_NAMES:
            candidate_dir = candidate_engine_directory(
                output_dir,
                candidate_id,
                29696,
                precision="FP16",
                profile_name=profile_name,
            )
            build_path = candidate_dir / "build_report.json"
            row: dict[str, Any] = {
                "candidate_id": candidate_id,
                "attention_boundary_profile": profile_name,
                "d_qk": record["d_qk"],
                "d_v": record["d_v"],
                "frames": frames,
            }
            if not build_path.is_file():
                row.update(status="build_report_missing", failure_reason=str(build_path))
                rows.append(row)
                continue
            build = dict(_read(build_path))
            if build.get("status") != "ok":
                row.update(
                    status="build_failed", failure_reason=build.get("failure_reason", "")
                )
                rows.append(row)
                continue
            evaluation_dir = candidate_dir / f"evaluation_{manifest_name}"
            evaluation_path = evaluation_dir / "evaluation.json"
            if evaluation_path.is_file():
                evaluation = dict(_read(evaluation_path))
            else:
                evaluation = evaluate_cobevt_engine_modelopt(
                    engine_path=build["engine_path"],
                    checkpoint=checkpoint,
                    model_config=config,
                    heal_root=heal_root,
                    device=f"cuda:{physical_gpu}",
                    output_dir=evaluation_dir,
                    tensorrt_root=trt_root,
                    plugin_path=plugin,
                    fixed_k=29696,
                    num_frames=frames,
                    warmup_frames=20,
                    eval_manifest_path=manifest_path,
                    num_workers=8,
                    ap_iou_backend="gpu",
                    latency_rounds=1,
                )
            row.update(evaluation)
            row.update(
                engine_sha256=build.get("engine_sha256", ""),
                structure_hash=build.get("structure_hash", ""),
                physical_parameter_count=build.get("physical_parameter_count", 0),
            )
            rows.append(row)
    result_dir = output_dir / "structure_experiment"
    _write_json(result_dir / f"evaluation_{manifest_name}.json", rows)
    _write_csv(result_dir / f"evaluation_{manifest_name}.csv", rows)
    return {
        "attempted": len(rows),
        "successful": sum(
            row.get("status") == "ok"
            and int(row.get("num_evaluated_frames", -1)) == frames
            and int(row.get("num_skipped_frames", -1)) == 0
            for row in rows
        ),
    }


def select_fixed500_candidates(output_dir: Path) -> dict[str, Any]:
    rows = list(_read(output_dir / "structure_experiment/evaluation_fixed50.json"))
    valid = [
        row for row in rows
        if row.get("status") == "ok"
        and int(row.get("num_evaluated_frames", -1)) == 50
        and int(row.get("num_skipped_frames", -1)) == 0
    ]
    by_candidate: dict[str, list[dict[str, Any]]] = {}
    for row in valid:
        by_candidate.setdefault(str(row["candidate_id"]), []).append(row)
    if not all(len(by_candidate.get(candidate, ())) == 2 for candidate in ("S0",)):
        raise RuntimeError("fixed500_selection_baseline_pair_missing")
    def pair_score(candidate_id: str) -> tuple[float, str]:
        pair = by_candidate.get(candidate_id, [])
        return (min(float(row["mAP"]) for row in pair), candidate_id) if len(pair) == 2 else (-1.0, candidate_id)
    uniform = max(("S1", "S2"), key=pair_score)
    asymmetric = max(("S3", "S4"), key=pair_score)
    selected = ["S0", uniform, asymmetric]
    payload = {
        "candidate_ids": selected,
        "profile_count": 2,
        "engine_count": 6,
        "selection_rule": "baseline_pair_plus_best_min_map_uniform_and_asymmetric_pairs",
    }
    _write_json(output_dir / "structure_experiment/fixed500_selection.json", payload)
    return payload


def write_interactions(output_dir: Path) -> dict[str, Any]:
    rows = list(_read(output_dir / "structure_experiment/evaluation_fixed500.json"))
    lookup = {
        (str(row["candidate_id"]), str(row["attention_boundary_profile"])): row
        for row in rows
        if row.get("status") == "ok"
    }
    p0, f3 = PROFILE_NAMES
    s0p0 = float(lookup[("S0", p0)]["mAP"])
    s0f3 = float(lookup[("S0", f3)]["mAP"])
    results = []
    for candidate_id in sorted({key[0] for key in lookup}):
        if (candidate_id, p0) not in lookup or (candidate_id, f3) not in lookup:
            continue
        metrics = structure_precision_interaction(
            s0_p0_map=s0p0,
            s0_f3_map=s0f3,
            candidate_p0_map=float(lookup[(candidate_id, p0)]["mAP"]),
            candidate_f3_map=float(lookup[(candidate_id, f3)]["mAP"]),
        )
        results.append({"candidate_id": candidate_id, **metrics})
    root = output_dir / "structure_experiment"
    _write_json(root / "structure_precision_cross_matrix.json", rows)
    _write_csv(root / "structure_precision_cross_matrix.csv", rows)
    lines = ["# Structure × F3 interaction", ""]
    for row in results:
        lines.append(
            f"- {row['candidate_id']}: interaction={row['interaction']:+.9f}, "
            f"prune={row['delta_prune']:+.9f}, F3={row['delta_f3_candidate']:+.9f}"
        )
    (root / "structure_precision_interaction.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return {"interaction_rows": results}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        required=True,
        choices=(
            "prepare", "structure", "build-a", "evaluate-a", "select-fixed500",
            "write-interactions", "finalize",
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--heal-root", default=str(DEFAULT_HEAL_ROOT))
    parser.add_argument("--trt-root", default="")
    parser.add_argument("--plugin", default=str(DEFAULT_PLUGIN))
    parser.add_argument("--ranking-source", default=str(DEFAULT_RANKING_SOURCE))
    parser.add_argument("--boundary-source", default=str(DEFAULT_BOUNDARY_SOURCE))
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--physical-gpu", type=int, default=3)
    parser.add_argument("--frames", type=int, choices=(10, 50, 500), default=10)
    parser.add_argument("--candidate-id", action="append", default=[])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output_dir = Path(args.output_dir).expanduser().resolve()
    common = {
        "checkpoint": Path(args.checkpoint),
        "config": Path(args.config),
        "heal_root": Path(args.heal_root),
    }
    resolved_trt_root = (
        Path(args.trt_root).expanduser().resolve()
        if str(args.trt_root).strip()
        else (
            Path(_read(output_dir / "run_manifest.json")["tensorrt_root"])
            if (output_dir / "run_manifest.json").is_file()
            else resolve_tensorrt_root_from_history(Path(args.boundary_source))
        )
    )
    if args.phase == "prepare":
        result = prepare_experiment(
            output_dir,
            checkpoint=common["checkpoint"],
            config=common["config"],
            ranking_source=Path(args.ranking_source),
            boundary_source=Path(args.boundary_source),
        )
    elif args.phase == "structure":
        result = run_structure_smoke(
            output_dir=output_dir,
            device=torch.device(args.device),
            **common,
        )
    elif args.phase == "build-a":
        result = {}
        for profile_name in PROFILE_NAMES:
            result[profile_name] = run_export_build(
                output_dir=output_dir,
                device=torch.device(args.device),
                physical_gpu=int(args.physical_gpu),
                trt_root=resolved_trt_root,
                plugin_path=Path(args.plugin),
                attention_boundary_profile_name=profile_name,
                **common,
            )
    elif args.phase == "evaluate-a":
        result = evaluate_a(
            output_dir,
            trt_root=resolved_trt_root,
            plugin=Path(args.plugin),
            physical_gpu=int(args.physical_gpu),
            frames=int(args.frames),
            candidate_ids=args.candidate_id,
            **common,
        )
    elif args.phase == "select-fixed500":
        result = select_fixed500_candidates(output_dir)
    elif args.phase == "write-interactions":
        result = write_interactions(output_dir)
    else:
        from search.reporting.cobevt_minimal_structure_quant_latency import (
            finalize_minimal_experiment,
        )

        result = finalize_minimal_experiment(output_dir)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
