#!/usr/bin/env python3
"""Fresh full-model validation of explicit CoBEVT Attention accumulation contracts."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

from search.integration.lidar_cobevt_evaluation_provider import (
    evaluate_cobevt_engine_modelopt,
)
from search.model_families.lidar_cobevt.attention_plugin_rewrite import (
    rewrite_f3_attention_einsums,
)
from search.orchestration.lidar_cobevt_attention_precision_audit import (
    prepare_boundary_audit_run,
)
from search.orchestration.lidar_cobevt_attention_pruning import (
    candidate_engine_directory,
    run_export_build,
)


DEFAULT_SOURCE_OUTPUT = Path(
    "/data/lxf/heal_data/outputs/cobevt_attention_dim_pruning_20260718_085644"
)
DEFAULT_CHECKPOINT = Path(
    "/home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/"
    "LiDAROnly/lidar_cobevt/net_epoch_bestval_at19.pth"
)
DEFAULT_CONFIG = DEFAULT_CHECKPOINT.with_name("config.yaml")
DEFAULT_HEAL_ROOT = Path("/home/lixingfeng/UniAD_examine/HEAL")
DEFAULT_TRT_ROOT = Path(
    "/home/lixingfeng/UniAD_examine/HEAL/prune_model/TensorRT-10.9_x86_cu118"
)
DEFAULT_SCATTER_PLUGIN = Path(
    "/home/lixingfeng/UniAD_examine/heal_compress/quantization/plugins/"
    "pointpillar_scatter_trt/build/libpointpillar_scatter_trt.so"
)


@dataclass(frozen=True)
class FullModelProfile:
    profile_id: str
    plugin_families: tuple[str, ...]
    requires: tuple[str, ...] = ()


def full_model_profile_matrix() -> tuple[FullModelProfile, ...]:
    return (
        FullModelProfile("A0_F3_REFERENCE", ()),
        FullModelProfile("A2_QK_F16A32_PLUGIN", ("QK",)),
        FullModelProfile("B1_AV_F16A32_PLUGIN", ("AV",)),
        FullModelProfile(
            "C1_QK_AV_F16A32_PLUGIN",
            ("QK", "AV"),
            ("A2_QK_F16A32_PLUGIN", "B1_AV_F16A32_PLUGIN"),
        ),
    )


def profile_evaluation_is_allowed(
    profile_id: str, prerequisite_results: Mapping[str, bool]
) -> bool:
    profile = next(row for row in full_model_profile_matrix() if row.profile_id == profile_id)
    return all(bool(prerequisite_results.get(value, False)) for value in profile.requires)


def classify_fixed500_delta(delta_map: float) -> str:
    value = float(delta_map)
    if value >= -0.003:
        return "SAFE_FIXED500"
    if value >= -0.010:
        return "BORDERLINE_FIXED500"
    return "UNSAFE_FIXED500"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return completed.stdout.strip()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def _profile(profile_id: str) -> FullModelProfile:
    try:
        return next(row for row in full_model_profile_matrix() if row.profile_id == profile_id)
    except StopIteration as exc:
        raise ValueError(f"full_model_accumulation_profile_unknown:{profile_id}") from exc


def _run_dir(output_dir: Path, profile_id: str) -> Path:
    return output_dir / "full_model" / str(profile_id)


def prepare_profile(
    *,
    output_dir: Path,
    profile_id: str,
    source_output: Path,
    checkpoint: Path,
    config: Path,
    scatter_plugin: Path,
) -> dict[str, Any]:
    profile = _profile(profile_id)
    run_dir = _run_dir(output_dir, profile.profile_id)
    if (run_dir / "experiment_config.json").is_file():
        return {"profile_id": profile.profile_id, "run_dir": str(run_dir), "reused": True}
    return {
        "profile_id": profile.profile_id,
        **prepare_boundary_audit_run(
            source_output=source_output,
            output_dir=run_dir,
            checkpoint_hash=_sha256(checkpoint),
            config_hash=_sha256(config),
            plugin_hash=_sha256(scatter_plugin),
            code_commit=_commit(),
        ),
        "reused": False,
    }


def build_profile(
    *,
    output_dir: Path,
    profile_id: str,
    checkpoint: Path,
    config: Path,
    heal_root: Path,
    trt_root: Path,
    scatter_plugin: Path,
    mixed_plugin: Path,
    physical_gpu: int,
) -> dict[str, Any]:
    profile = _profile(profile_id)
    run_dir = _run_dir(output_dir, profile.profile_id)

    def transform(source: Path, destination: Path, *_args: Any) -> Mapping[str, Any]:
        return rewrite_f3_attention_einsums(
            source, destination, families=profile.plugin_families
        )

    result = run_export_build(
        output_dir=run_dir,
        checkpoint=checkpoint,
        config=config,
        heal_root=heal_root,
        device=torch.device(f"cuda:{int(physical_gpu)}"),
        physical_gpu=int(physical_gpu),
        trt_root=trt_root,
        plugin_path=scatter_plugin,
        additional_plugin_paths=(mixed_plugin,) if profile.plugin_families else (),
        engine_precision="FP32",
        candidate_ids=("baseline_d32",),
        attention_boundary_profile_name="F3_rest_fp16_qk_fp32_minimal_island",
        post_attention_boundary_transform=transform if profile.plugin_families else None,
    )
    build_dir = candidate_engine_directory(
        run_dir,
        "baseline_d32",
        29696,
        precision="FP32",
        profile_name="F3_rest_fp16_qk_fp32_minimal_island",
    )
    report_path = build_dir / "build_report.json"
    report = json.loads(report_path.read_text()) if report_path.is_file() else {}
    summary = {
        "profile_id": profile.profile_id,
        "plugin_families": list(profile.plugin_families),
        "run_dir": str(run_dir),
        "build_report": str(report_path),
        "status": str(report.get("status", "failed")),
        "failure_reason": str(report.get("failure_reason", "")),
        "engine_path": str(report.get("engine_path", "")),
        "engine_sha256": str(report.get("engine_sha256", "")),
        "result": result,
    }
    _write_json(run_dir / "profile_build_summary.json", summary)
    return summary


_PROTOCOLS = {"smoke10": (10, "smoke10"), "fixed50": (50, "fixed50"), "fixed500": (500, "fixed500")}


def evaluate_profile(
    *,
    output_dir: Path,
    profile_id: str,
    protocol: str,
    checkpoint: Path,
    config: Path,
    heal_root: Path,
    trt_root: Path,
    scatter_plugin: Path,
    mixed_plugin: Path,
    physical_gpu: int,
) -> dict[str, Any]:
    profile = _profile(profile_id)
    if protocol not in _PROTOCOLS:
        raise ValueError(f"full_model_accumulation_protocol_unknown:{protocol}")
    run_dir = _run_dir(output_dir, profile.profile_id)
    build_dir = candidate_engine_directory(
        run_dir, "baseline_d32", 29696, precision="FP32",
        profile_name="F3_rest_fp16_qk_fp32_minimal_island",
    )
    report = json.loads((build_dir / "build_report.json").read_text())
    if report.get("status") != "ok":
        raise RuntimeError(f"full_model_accumulation_build_not_ready:{profile_id}")
    experiment = json.loads((run_dir / "experiment_config.json").read_text())
    frames, manifest_key = _PROTOCOLS[protocol]
    evaluation = evaluate_cobevt_engine_modelopt(
        engine_path=report["engine_path"],
        checkpoint=checkpoint,
        model_config=config,
        heal_root=heal_root,
        device=f"cuda:{int(physical_gpu)}",
        output_dir=build_dir / f"evaluation_{protocol}",
        tensorrt_root=trt_root,
        plugin_path=scatter_plugin,
        additional_plugin_paths=(mixed_plugin,) if profile.plugin_families else (),
        fixed_k=29696,
        num_frames=frames,
        warmup_frames=20,
        eval_manifest_path=experiment["manifests"][manifest_key]["path"],
        num_workers=8,
        ap_iou_backend="gpu",
        latency_rounds=1,
    )
    evaluation["profile_id"] = profile.profile_id
    evaluation["protocol"] = protocol
    evaluation["evaluation_complete"] = (
        evaluation.get("status") == "ok"
        and int(evaluation.get("num_evaluated_frames", -1)) == frames
        and int(evaluation.get("num_skipped_frames", -1)) == 0
    )
    baseline_path = (
        _run_dir(output_dir, "A0_F3_REFERENCE")
        / "candidates/baseline_d32/f3_rest_fp16_qk_fp32_minimal_island_engine_k29696"
        / f"evaluation_{protocol}/evaluation.json"
    )
    if profile.profile_id != "A0_F3_REFERENCE" and baseline_path.is_file():
        baseline = json.loads(baseline_path.read_text())
        if "mAP" in evaluation and "mAP" in baseline:
            evaluation["delta_mAP_vs_fresh_F3"] = float(evaluation["mAP"]) - float(baseline["mAP"])
            if protocol == "fixed500":
                evaluation["fixed500_safety"] = classify_fixed500_delta(
                    evaluation["delta_mAP_vs_fresh_F3"]
                )
    _write_json(build_dir / f"evaluation_{protocol}/evaluation.json", evaluation)
    return evaluation


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("prepare", "build", "evaluate"), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--profile", required=True, choices=tuple(row.profile_id for row in full_model_profile_matrix()))
    parser.add_argument("--protocol", choices=tuple(_PROTOCOLS), default="smoke10")
    parser.add_argument("--source-output", default=str(DEFAULT_SOURCE_OUTPUT))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--heal-root", default=str(DEFAULT_HEAL_ROOT))
    parser.add_argument("--trt-root", default=str(DEFAULT_TRT_ROOT))
    parser.add_argument("--scatter-plugin", default=str(DEFAULT_SCATTER_PLUGIN))
    parser.add_argument("--mixed-plugin", required=True)
    parser.add_argument("--physical-gpu", type=int, default=7)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    common = {
        "output_dir": Path(args.output_dir).resolve(),
        "profile_id": args.profile,
        "checkpoint": Path(args.checkpoint).resolve(),
        "config": Path(args.config).resolve(),
        "scatter_plugin": Path(args.scatter_plugin).resolve(),
    }
    if args.phase == "prepare":
        result = prepare_profile(
            **common, source_output=Path(args.source_output).resolve()
        )
    elif args.phase == "build":
        result = build_profile(
            **common,
            heal_root=Path(args.heal_root).resolve(),
            trt_root=Path(args.trt_root).resolve(),
            mixed_plugin=Path(args.mixed_plugin).resolve(),
            physical_gpu=args.physical_gpu,
        )
    else:
        result = evaluate_profile(
            **common,
            protocol=args.protocol,
            heal_root=Path(args.heal_root).resolve(),
            trt_root=Path(args.trt_root).resolve(),
            mixed_plugin=Path(args.mixed_plugin).resolve(),
            physical_gpu=args.physical_gpu,
        )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
