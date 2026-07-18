"""Staged strongly typed FP16 boundary audit for CoBEVT Attention."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import torch

from search.integration.data_provider import write_eval_manifest
from search.model_families.lidar_cobevt.attention_precision_boundaries import (
    ATTENTION_BOUNDARY_PROFILE_NAMES,
    attention_boundary_profile,
    describe_existing_attention_boundaries,
)
from search.reporting.cobevt_attention_precision_inventory import (
    build_attention_precision_inventory,
    write_attention_precision_inventory,
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
DEFAULT_PLUGIN = Path(
    "/home/lixingfeng/UniAD_examine/heal_compress/quantization/plugins/"
    "pointpillar_scatter_trt/build/libpointpillar_scatter_trt.so"
)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def boundary_freshness_signature(
    *,
    code_commit: str,
    checkpoint_hash: str,
    config_hash: str,
    plugin_hash: str,
    profile_hash: str,
    fixed_k: int,
    smoke_manifest_hash: str,
    fixed500_manifest_hash: str,
    tensorrt_version: str,
    cuda_version: str,
) -> str:
    payload = {
        "code_commit": str(code_commit),
        "checkpoint_hash": str(checkpoint_hash),
        "config_hash": str(config_hash),
        "cuda_version": str(cuda_version),
        "fixed500_manifest_hash": str(fixed500_manifest_hash),
        "fixed_k": int(fixed_k),
        "plugin_hash": str(plugin_hash),
        "profile_hash": str(profile_hash),
        "smoke_manifest_hash": str(smoke_manifest_hash),
        "tensorrt_version": str(tensorrt_version),
    }
    missing = [key for key, value in payload.items() if value in {"", 0}]
    if missing:
        raise ValueError(f"boundary_freshness_signature_missing:{missing}")
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def classify_smoke_delta(delta_map: float) -> str:
    value = float(delta_map)
    if value <= -0.05:
        return "catastrophic"
    if value >= -0.01:
        return "safe"
    return "ambiguous"


def _manifest_record(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "frame_ids": list(payload.get("frame_ids", [])),
        "manifest_hash": str(payload["manifest_hash"]),
        "path": str(path),
        "split": str(payload.get("split", "val")),
    }


def prepare_boundary_audit_run(
    *,
    source_output: str | Path,
    output_dir: str | Path,
    checkpoint_hash: str,
    config_hash: str,
    plugin_hash: str,
    code_commit: str,
) -> dict[str, Any]:
    source = Path(source_output).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise RuntimeError(f"boundary_audit_output_not_empty:{destination}")
    destination.mkdir(parents=True, exist_ok=True)
    source_experiment_path = source / "experiment_config.json"
    if not source_experiment_path.is_file():
        raise FileNotFoundError(str(source_experiment_path))
    experiment = json.loads(source_experiment_path.read_text(encoding="utf-8"))
    contract = dict(experiment.get("fixed_k_contract", {}))
    if int(experiment.get("fixed_k", 0)) != 29696:
        raise RuntimeError("boundary_audit_requires_fixed_k_29696")
    if not bool(experiment.get("fixed_k_validated", False)):
        raise RuntimeError("boundary_audit_fixed_k_not_validated")
    if str(contract.get("validated_scope", "")) != "full_validation":
        raise RuntimeError("boundary_audit_fixed_k_not_full_validation_scanned")
    if int(contract.get("overflow_count", -1)) != 0:
        raise RuntimeError("boundary_audit_fixed_k_overflow_detected")
    if str(contract.get("deployment_topology", "single_engine_fixed_k")) != (
        "single_engine_fixed_k"
    ):
        raise RuntimeError("boundary_audit_requires_single_engine_fixed_k")
    candidates = [
        dict(row)
        for row in experiment.get("candidates", [])
        if str(row.get("candidate_id", "")) == "baseline_d32"
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"boundary_audit_baseline_candidate_count:{len(candidates)}")
    manifests = destination / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    source_smoke = Path(experiment["manifests"]["smoke10"]["path"])
    source_fixed = Path(experiment["manifests"]["fixed500"]["path"])
    smoke_path = manifests / "smoke10_manifest.json"
    fixed500_path = manifests / "fixed500_manifest.json"
    shutil.copy2(source_smoke, smoke_path)
    shutil.copy2(source_fixed, fixed500_path)
    fixed_payload = json.loads(fixed500_path.read_text(encoding="utf-8"))
    warmup_ids = [str(value) for value in fixed_payload["warmup_frame_ids"]]
    evaluation_ids = [str(value) for value in fixed_payload["evaluation_frame_ids"]]
    fixed50 = write_eval_manifest(
        manifests / "fixed50_manifest.json",
        num_frames=50,
        warmup_frames=len(warmup_ids),
        split=str(fixed_payload.get("split", "val")),
        available_frame_ids=[*warmup_ids, *evaluation_ids],
        reset_after_warmup=True,
        evaluation_offset=len(warmup_ids),
    )
    masks = destination / "candidate_masks"
    masks.mkdir(parents=True, exist_ok=True)
    source_mask = Path(candidates[0]["mask_path"])
    destination_mask = masks / "baseline_d32.json"
    shutil.copy2(source_mask, destination_mask)
    candidates[0]["mask_path"] = str(destination_mask)
    experiment["candidates"] = candidates
    experiment["manifests"] = {
        "smoke10": _manifest_record(smoke_path),
        "fixed50": fixed50.to_dict(),
        "fixed500": _manifest_record(fixed500_path),
    }
    experiment["deployment_policy"] = {
        "bucketed_engines": False,
        "deployment_topology": "single_engine_fixed_k",
        "fixed_k": 29696,
        "overflow_policy": "fail_closed",
        "overlimit_chunking": False,
    }
    _write_json(destination / "experiment_config.json", experiment)
    profile_root = destination / "profiles"
    for profile_name in ATTENTION_BOUNDARY_PROFILE_NAMES:
        (profile_root / profile_name).mkdir(parents=True, exist_ok=False)
    run_manifest = {
        "checkpoint_hash": str(checkpoint_hash),
        "code_commit": str(code_commit),
        "config_hash": str(config_hash),
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "deployment_policy": dict(experiment["deployment_policy"]),
        "fixed_k": 29696,
        "manifests": experiment["manifests"],
        "plugin_hash": str(plugin_hash),
        "profiles": list(ATTENTION_BOUNDARY_PROFILE_NAMES),
        "source_experiment_config_hash": _sha256(source_experiment_path),
        "source_output": str(source),
    }
    _write_json(destination / "run_manifest.json", run_manifest)
    import yaml

    (destination / "resolved_config.yaml").write_text(
        yaml.safe_dump(run_manifest, sort_keys=True), encoding="utf-8"
    )
    return {
        "deployment_topology": "single_engine_fixed_k",
        "fixed_k": 29696,
        "output_dir": str(destination),
        "profile_count": len(ATTENTION_BOUNDARY_PROFILE_NAMES),
        "run_manifest": str(destination / "run_manifest.json"),
    }


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError("boundary_audit_git_commit_unavailable")
    return completed.stdout.strip()


def _profile_directory(output_dir: Path, profile_name: str) -> Path:
    if str(profile_name) not in ATTENTION_BOUNDARY_PROFILE_NAMES:
        raise ValueError(f"unknown_attention_boundary_profile:{profile_name}")
    return output_dir / "profiles" / str(profile_name)


def run_static_precision_inventory(
    *, source_output: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    source = Path(source_output).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    profiles = {
        "legacy_strict_fp32": "fp32_engine_k29696",
        "legacy_attention_fp16": "attention_fp16_engine_k29696",
        "accepted_attention_fp32_rest_fp16": (
            "attention_fp32_rest_fp16_engine_k29696"
        ),
    }
    combined = []
    summaries = {}
    for label, directory_name in profiles.items():
        engine_dir = source / "candidates" / "baseline_d32" / directory_name
        build_path = engine_dir / "build_report.json"
        typed_path = engine_dir / "typed_aux.onnx"
        layer_info_path = engine_dir / "engine_layer_info.json"
        for path in (build_path, typed_path, layer_info_path):
            if not path.is_file():
                raise FileNotFoundError(str(path))
        build = json.loads(build_path.read_text(encoding="utf-8"))
        if str(build.get("status", "")) != "ok":
            raise RuntimeError(f"legacy_precision_inventory_build_invalid:{label}")
        entries = [
            SimpleNamespace(
                module_path=str(row["module_path"]),
                canonical_node_name=str(row["canonical_node_name"]),
            )
            for row in build["mapping"]["entries"]
        ]
        boundary = describe_existing_attention_boundaries(
            typed_path, entries, profile_name=label
        )
        rows = build_attention_precision_inventory(
            typed_path, boundary, layer_info_path
        )
        for row in rows:
            row["profile_name"] = label
            row["source_build_report"] = str(build_path)
            row["source_engine_sha256"] = str(build.get("engine_sha256", ""))
        profile_dir = destination / "static_inventory" / label
        summary = write_attention_precision_inventory(
            rows,
            profile_dir / "attention_precision_inventory.json",
            profile_dir / "attention_precision_inventory.md",
        )
        summary["role_dtype_counts"] = {
            f"{role}:{dtype}": sum(
                row["role"] == role and row["requested_dtype"] == dtype
                for row in rows
            )
            for role, dtype in sorted(
                {(row["role"], row["requested_dtype"]) for row in rows}
            )
        }
        summaries[label] = summary
        combined.extend(rows)
    write_attention_precision_inventory(
        combined,
        destination / "attention_precision_inventory.json",
        destination / "attention_precision_inventory.md",
    )
    result = {
        "profile_count": len(profiles),
        "profiles": summaries,
        "row_count": len(combined),
    }
    _write_json(destination / "static_inventory_summary.json", result)
    return result


def _load_run_manifest(output_dir: Path) -> dict[str, Any]:
    path = output_dir / "run_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(str(path))
    return json.loads(path.read_text(encoding="utf-8"))


def _profile_freshness(
    *, output_dir: Path, profile_name: str
) -> tuple[dict[str, Any], str]:
    run = _load_run_manifest(output_dir)
    profile = attention_boundary_profile(profile_name)
    signature = boundary_freshness_signature(
        code_commit=str(run["code_commit"]),
        checkpoint_hash=str(run["checkpoint_hash"]),
        config_hash=str(run["config_hash"]),
        plugin_hash=str(run["plugin_hash"]),
        profile_hash=profile.profile_hash,
        fixed_k=int(run["fixed_k"]),
        smoke_manifest_hash=str(run["manifests"]["smoke10"]["manifest_hash"]),
        fixed500_manifest_hash=str(run["manifests"]["fixed500"]["manifest_hash"]),
        tensorrt_version="10.9",
        cuda_version="11.8",
    )
    manifest = {
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "freshness_signature": signature,
        "profile": profile.to_dict(),
        "run_manifest": str(output_dir / "run_manifest.json"),
    }
    return manifest, signature


def run_boundary_build(
    *,
    output_dir: str | Path,
    profile_name: str,
    checkpoint: str | Path,
    config: str | Path,
    heal_root: str | Path,
    trt_root: str | Path,
    plugin_path: str | Path,
    device: torch.device,
    physical_gpu: int,
) -> dict[str, Any]:
    from search.orchestration.lidar_cobevt_attention_pruning import (
        candidate_engine_directory,
        run_export_build,
    )

    destination = Path(output_dir).expanduser().resolve()
    profile_dir = _profile_directory(destination, profile_name)
    profile_manifest, signature = _profile_freshness(
        output_dir=destination, profile_name=profile_name
    )
    _write_json(profile_dir / "requested_profile.json", profile_manifest)
    result = run_export_build(
        output_dir=destination,
        checkpoint=Path(checkpoint).expanduser().resolve(),
        config=Path(config).expanduser().resolve(),
        heal_root=Path(heal_root).expanduser().resolve(),
        device=device,
        physical_gpu=int(physical_gpu),
        trt_root=Path(trt_root).expanduser().resolve(),
        plugin_path=Path(plugin_path).expanduser().resolve(),
        engine_precision="FP32",
        candidate_ids=("baseline_d32",),
        attention_boundary_profile_name=profile_name,
    )
    engine_dir = candidate_engine_directory(
        destination,
        "baseline_d32",
        29696,
        precision="FP32",
        profile_name=profile_name,
    )
    build_path = engine_dir / "build_report.json"
    if not build_path.is_file():
        raise RuntimeError(f"boundary_build_report_missing:{profile_name}")
    build = json.loads(build_path.read_text(encoding="utf-8"))
    build["freshness_signature"] = signature
    build["profile_manifest_path"] = str(profile_dir / "requested_profile.json")
    _write_json(build_path, build)
    artifact_index = {
        "build_report": str(build_path),
        "engine_path": str(build.get("engine_path", "")),
        "engine_sha256": str(build.get("engine_sha256", "")),
        "freshness_signature": signature,
        "layer_info_path": str(build.get("layer_info_path", "")),
        "status": str(build.get("status", "")),
        "typed_onnx": str(engine_dir / "typed.onnx"),
        "typed_onnx_sha256": str(
            build.get("attention_boundary_report", {}).get(
                "output_onnx_sha256", ""
            )
        ),
    }
    _write_json(profile_dir / "artifact_index.json", artifact_index)
    return {**result, **artifact_index}


_PROTOCOLS = {
    "smoke10": (10, "smoke10"),
    "fixed50": (50, "fixed50"),
    "fixed500": (500, "fixed500"),
}


def run_boundary_evaluation(
    *,
    output_dir: str | Path,
    profile_name: str,
    protocol: str,
    checkpoint: str | Path,
    config: str | Path,
    heal_root: str | Path,
    trt_root: str | Path,
    plugin_path: str | Path,
    physical_gpu: int,
) -> dict[str, Any]:
    from search.integration.lidar_cobevt_evaluation_provider import (
        evaluate_cobevt_engine_modelopt,
    )

    if str(protocol) not in _PROTOCOLS:
        raise ValueError(f"unsupported_boundary_evaluation_protocol:{protocol}")
    destination = Path(output_dir).expanduser().resolve()
    profile_dir = _profile_directory(destination, profile_name)
    artifact_path = profile_dir / "artifact_index.json"
    if not artifact_path.is_file():
        raise FileNotFoundError(str(artifact_path))
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    requested_manifest, signature = _profile_freshness(
        output_dir=destination, profile_name=profile_name
    )
    if artifact.get("freshness_signature") != signature:
        raise RuntimeError(f"boundary_build_freshness_mismatch:{profile_name}")
    run = _load_run_manifest(destination)
    frame_count, manifest_key = _PROTOCOLS[str(protocol)]
    manifest = Path(run["manifests"][manifest_key]["path"])
    evaluation_dir = profile_dir / f"evaluation_{protocol}"
    result = evaluate_cobevt_engine_modelopt(
        engine_path=artifact["engine_path"],
        checkpoint=Path(checkpoint).expanduser().resolve(),
        model_config=Path(config).expanduser().resolve(),
        heal_root=Path(heal_root).expanduser().resolve(),
        device=f"cuda:{int(physical_gpu)}",
        output_dir=evaluation_dir,
        tensorrt_root=Path(trt_root).expanduser().resolve(),
        plugin_path=Path(plugin_path).expanduser().resolve(),
        fixed_k=29696,
        num_frames=frame_count,
        warmup_frames=20,
        eval_manifest_path=manifest,
        num_workers=8,
        ap_iou_backend="gpu",
        latency_rounds=1,
    )
    complete = (
        str(result.get("status", "")) == "ok"
        and int(result.get("num_evaluated_frames", -1)) == frame_count
        and int(result.get("num_skipped_frames", -1)) == 0
    )
    result.update(
        {
            "attention_boundary_profile": profile_name,
            "evaluation_complete": complete,
            "freshness_signature": signature,
            "manifest_hash": str(run["manifests"][manifest_key]["manifest_hash"]),
            "protocol": str(protocol),
        }
    )
    a0_path = (
        destination
        / "profiles/A0_strict_fp32_reference"
        / f"evaluation_{protocol}/evaluation.json"
    )
    if profile_name != "A0_strict_fp32_reference" and a0_path.is_file():
        a0 = json.loads(a0_path.read_text(encoding="utf-8"))
        if "mAP" in a0 and "mAP" in result:
            result["delta_mAP_vs_A0"] = float(result["mAP"]) - float(a0["mAP"])
            if protocol == "smoke10":
                result["smoke_classification"] = classify_smoke_delta(
                    result["delta_mAP_vs_A0"]
                )
    _write_json(evaluation_dir / "evaluation.json", result)
    _write_json(
        evaluation_dir / "evaluation_provenance.json",
        {
            **requested_manifest,
            "engine_sha256": artifact.get("engine_sha256", ""),
            "manifest_hash": run["manifests"][manifest_key]["manifest_hash"],
            "protocol": protocol,
        },
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase", choices=("prepare", "inventory", "build", "evaluate"), required=True
    )
    parser.add_argument("--source-output", default=str(DEFAULT_SOURCE_OUTPUT))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--heal-root", default=str(DEFAULT_HEAL_ROOT))
    parser.add_argument("--trt-root", default=str(DEFAULT_TRT_ROOT))
    parser.add_argument("--plugin", default=str(DEFAULT_PLUGIN))
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--physical-gpu", type=int, default=2)
    parser.add_argument("--profile", choices=ATTENTION_BOUNDARY_PROFILE_NAMES)
    parser.add_argument("--protocol", choices=tuple(_PROTOCOLS), default="smoke10")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    output_dir = Path(args.output_dir).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    config = Path(args.config).expanduser().resolve()
    plugin = Path(args.plugin).expanduser().resolve()
    if args.phase == "prepare":
        result = prepare_boundary_audit_run(
            source_output=args.source_output,
            output_dir=output_dir,
            checkpoint_hash=_sha256(checkpoint),
            config_hash=_sha256(config),
            plugin_hash=_sha256(plugin),
            code_commit=_git_commit(),
        )
    elif args.phase == "inventory":
        result = run_static_precision_inventory(
            source_output=args.source_output, output_dir=output_dir
        )
    elif args.phase == "build":
        if not args.profile:
            raise ValueError("boundary_build_profile_required")
        result = run_boundary_build(
            output_dir=output_dir,
            profile_name=args.profile,
            checkpoint=checkpoint,
            config=config,
            heal_root=args.heal_root,
            trt_root=args.trt_root,
            plugin_path=plugin,
            device=torch.device(args.device),
            physical_gpu=args.physical_gpu,
        )
    else:
        if not args.profile:
            raise ValueError("boundary_evaluation_profile_required")
        result = run_boundary_evaluation(
            output_dir=output_dir,
            profile_name=args.profile,
            protocol=args.protocol,
            checkpoint=checkpoint,
            config=config,
            heal_root=args.heal_root,
            trt_root=args.trt_root,
            plugin_path=plugin,
            physical_gpu=args.physical_gpu,
        )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


__all__ = [
    "boundary_freshness_signature",
    "classify_smoke_delta",
    "prepare_boundary_audit_run",
    "run_boundary_build",
    "run_boundary_evaluation",
    "run_static_precision_inventory",
]


if __name__ == "__main__":
    raise SystemExit(main())
