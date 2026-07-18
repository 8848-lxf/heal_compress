"""Resumable CoBEVT head-internal Attention pruning experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping
from zoneinfo import ZoneInfo

import torch

from quantization.precision.typed_graph import apply_strongly_typed_precision_contract
from quantization.tensorrt.precision_checker import validate_precision_realization
from quantization.types import (
    CanonicalFunctionalComputeGroup,
    CanonicalMappingEntry,
    OnnxOriginMapResult,
)
from search.integration.data_provider import EvaluationManifest, write_eval_manifest
from search.model_families.lidar_cobevt.attention_dim_pruning import (
    AttentionDimMask,
    PrunableCobevtAttention,
    materialize_attention_bottleneck,
    materialize_global_embedding_bottleneck,
    stratified_embedding_keep_indices,
    uniform_attention_masks,
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
REQUESTED_TRT_ROOT = Path("/home/lixingfeng/UniAD_examine/TensorRT-10.9_x86_cu118")
DEFAULT_PLUGIN = Path(
    "/home/lixingfeng/UniAD_examine/heal_compress/quantization/plugins/"
    "pointpillar_scatter_trt/build/libpointpillar_scatter_trt.so"
)
DEFAULT_FIXED_K = 25600


@dataclass(frozen=True)
class AttentionCandidateSpec:
    candidate_id: str
    experiment: str
    variant: str
    d_qk: int
    d_v: int
    embed_dim: int
    heads: int = 8

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "d_qk": int(self.d_qk),
            "d_v": int(self.d_v),
            "embed_dim": int(self.embed_dim),
            "experiment": self.experiment,
            "heads": int(self.heads),
            "variant": self.variant,
        }


def formal_candidate_specs() -> tuple[AttentionCandidateSpec, ...]:
    return (
        AttentionCandidateSpec(
            "baseline_d32", "baseline", "unpruned_explicit_projection", 32, 32, 256
        ),
        AttentionCandidateSpec("qk_only_d24", "qk_only", "d_qk_24", 24, 32, 256),
        AttentionCandidateSpec("qk_only_d16", "qk_only", "d_qk_16", 16, 32, 256),
        AttentionCandidateSpec("b1_uniform_d24", "b1", "internal_bottleneck_24", 24, 24, 256),
        AttentionCandidateSpec("b1_uniform_d16", "b1", "internal_bottleneck_16", 16, 16, 256),
        AttentionCandidateSpec("b2_global_d24", "b2", "global_embedding_24", 24, 24, 192),
    )


def write_attention_evaluation_manifests(
    destination: str | Path, available_frame_ids: Iterable[str]
) -> dict[str, Any]:
    root = Path(destination)
    ids = [str(value) for value in available_frame_ids]
    smoke = write_eval_manifest(
        root / "smoke10_manifest.json",
        num_frames=10,
        warmup_frames=20,
        split="val",
        available_frame_ids=ids,
        reset_after_warmup=True,
        evaluation_offset=20,
    )
    fixed = write_eval_manifest(
        root / "fixed500_manifest.json",
        num_frames=500,
        warmup_frames=20,
        split="val",
        available_frame_ids=ids,
        reset_after_warmup=True,
        evaluation_offset=20,
    )
    return {
        "smoke10": smoke,
        "fixed500": fixed,
        "protocol": {
            "ap_iou_backend": "gpu",
            "dataloader_workers": 8,
            "evaluation_frames": 500,
            "warmup_frames": 20,
        },
    }


def write_attention_masks(
    path: str | Path, masks: Mapping[str, AttentionDimMask]
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            {name: mask.to_dict() for name, mask in sorted(masks.items())},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def load_attention_masks(path: str | Path) -> dict[str, AttentionDimMask]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        str(name): AttentionDimMask(
            tuple(tuple(int(value) for value in row) for row in record["qk_keep_by_head"]),
            tuple(tuple(int(value) for value in row) for row in record["vo_keep_by_head"]),
            original_d_qk=int(record["original_d_qk"]),
            original_d_v=int(record["original_d_v"]),
        )
        for name, record in payload.items()
    }


def manifest_summary(rows: Mapping[str, EvaluationManifest]) -> dict[str, Any]:
    return {name: row.to_dict() for name, row in rows.items()}


def _write_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else ""


def _load_bundle(
    *, checkpoint: Path, config: Path, heal_root: Path, device: torch.device
) -> Any:
    from search.model_families.lidar_cobevt.model_capability import (
        CobevtModelCapability,
    )

    return CobevtModelCapability(checkpoint, config, heal_root).load(device=device)


def _masks_for_candidate(
    *,
    explicit_baseline: torch.nn.Module,
    gradients: Mapping[str, torch.Tensor],
    spec: AttentionCandidateSpec,
) -> tuple[dict[str, AttentionDimMask], dict[str, Any]]:
    from search.model_families.lidar_cobevt.attention_taylor import (
        attention_masks_from_mean_gradients,
    )

    return attention_masks_from_mean_gradients(
        explicit_baseline,
        gradients,
        d_qk=spec.d_qk,
        d_v=spec.d_v,
    )


def initialize_prepare_output(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    allowed = {
        "attention_mean_gradients.pt",
        "attention_mean_gradients.manifest.json",
        "candidate_masks",
    }
    unknown = sorted(path.name for path in output_dir.iterdir() if path.name not in allowed)
    if unknown:
        raise RuntimeError(f"prepare_output_unknown_existing_artifacts:{unknown}")
    masks = output_dir / "candidate_masks"
    masks.mkdir(exist_ok=True)
    if not masks.is_dir():
        raise RuntimeError("prepare_candidate_masks_path_not_directory")


def run_prepare(
    *,
    output_dir: Path,
    checkpoint: Path,
    config: Path,
    heal_root: Path,
    device: torch.device,
    gradient_samples: int,
) -> dict[str, Any]:
    from search.integration.data_provider import load_split_frame_ids
    from search.integration.runtime_environment import query_gpus
    from search.model_families.lidar_cobevt.attention_taylor import (
        collect_attention_mean_gradients,
    )

    initialize_prepare_output(output_dir)
    bundle = _load_bundle(
        checkpoint=checkpoint, config=config, heal_root=heal_root, device=device
    )
    stock_rows = [
        {
            "module": name,
            "embed_dim": int(module.to_qkv.in_features),
            "heads": int(module.heads),
            "d_qk": int(module.to_qkv.out_features // 3 // module.heads),
            "d_v": int(module.to_qkv.out_features // 3 // module.heads),
            "qkv_weight_shape": list(module.to_qkv.weight.shape),
            "out_weight_shape": list(module.to_out[0].weight.shape),
            "relative_position_bias_shape": list(
                module.relative_position_bias_table.weight.shape
            ),
        }
        for name, module in bundle.model.named_modules()
        if name.startswith("fusion_net")
        and module.__class__.__name__ == "Attention"
    ]
    identity_masks = uniform_attention_masks(bundle.model, d_qk=32, d_v=32)
    baseline_report = materialize_attention_bottleneck(bundle.model, identity_masks)
    if not baseline_report.passed:
        raise RuntimeError(f"explicit_attention_baseline_invalid:{baseline_report.issues}")
    code_commit = _git_commit()
    statistics = collect_attention_mean_gradients(
        model=bundle.model,
        adapter=bundle.adapter,
        model_config_path=config,
        device=device,
        cache_path=output_dir / "attention_mean_gradients.pt",
        num_samples=int(gradient_samples),
        checkpoint_hash=bundle.preflight.checkpoint_sha256,
        code_commit=code_commit,
    )
    candidate_rows = []
    ranking_audit = None
    for spec in formal_candidate_specs():
        masks, audit = _masks_for_candidate(
            explicit_baseline=bundle.model,
            gradients=statistics.gradients,
            spec=spec,
        )
        mask_path = output_dir / "candidate_masks" / f"{spec.candidate_id}.json"
        write_attention_masks(mask_path, masks)
        if ranking_audit is None:
            ranking_audit = audit
        candidate_rows.append(
            {
                **spec.to_dict(),
                "mask_path": str(mask_path),
                "mask_sha256": _sha256(mask_path),
            }
        )
    val_ids = load_split_frame_ids(bundle.adapter, config, split="val")
    manifests = write_attention_evaluation_manifests(output_dir / "manifests", val_ids)
    environment = {
        "branch": subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=Path(__file__).resolve().parents[2],
            text=True,
            stdout=subprocess.PIPE,
            check=False,
        ).stdout.strip(),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": bundle.preflight.checkpoint_sha256,
        "code_commit": code_commit,
        "config": str(config),
        "config_sha256": bundle.preflight.config_sha256,
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "device": str(device),
        "gpus": query_gpus(),
        "heal_root": str(heal_root),
        "plugin": str(DEFAULT_PLUGIN),
        "plugin_sha256": _sha256(DEFAULT_PLUGIN),
        "requested_tensorrt_root": str(REQUESTED_TRT_ROOT),
        "requested_tensorrt_root_exists": REQUESTED_TRT_ROOT.is_dir(),
        "resolved_tensorrt_root": str(DEFAULT_TRT_ROOT),
    }
    _write_json(output_dir / "environment.json", environment)
    _write_json(output_dir / "attention_inventory.json", stock_rows)
    _write_json(output_dir / "qk_rankings.json", ranking_audit or {})
    _write_json(output_dir / "vo_rankings.json", ranking_audit or {})
    _write_json(
        output_dir / "experiment_config.json",
        {
            "candidates": candidate_rows,
            "fixed_k": DEFAULT_FIXED_K,
            "gradient_samples": int(gradient_samples),
            "gradient_statistics_manifest_hash": statistics.manifest_hash,
            "gpu": str(device),
            "manifests": {
                "smoke10": manifests["smoke10"].to_dict(),
                "fixed500": manifests["fixed500"].to_dict(),
            },
            "protocol": manifests["protocol"],
        },
    )
    return {
        "output_dir": str(output_dir),
        "candidate_count": len(candidate_rows),
        "gradient_manifest_hash": statistics.manifest_hash,
        "fixed500_manifest_hash": manifests["fixed500"].manifest_hash,
    }


def _read_config(output_dir: Path) -> dict[str, Any]:
    return json.loads((output_dir / "experiment_config.json").read_text(encoding="utf-8"))


def _spec_from_record(record: Mapping[str, Any]) -> AttentionCandidateSpec:
    return AttentionCandidateSpec(
        candidate_id=str(record["candidate_id"]),
        experiment=str(record["experiment"]),
        variant=str(record["variant"]),
        d_qk=int(record["d_qk"]),
        d_v=int(record["d_v"]),
        embed_dim=int(record["embed_dim"]),
        heads=int(record["heads"]),
    )


def materialize_candidate(
    *,
    spec: AttentionCandidateSpec,
    mask_path: Path,
    checkpoint: Path,
    config: Path,
    heal_root: Path,
    device: torch.device,
) -> tuple[Any, Any]:
    bundle = _load_bundle(
        checkpoint=checkpoint, config=config, heal_root=heal_root, device=device
    )
    masks = load_attention_masks(mask_path)
    if spec.experiment == "b2":
        keep = stratified_embedding_keep_indices(
            heads=spec.heads,
            original_dim_per_head=32,
            keep_dim_per_head=spec.embed_dim // spec.heads,
        )
        report = materialize_global_embedding_bottleneck(
            bundle.model,
            masks=masks,
            embedding_keep_indices=keep,
        )
    else:
        report = materialize_attention_bottleneck(bundle.model, masks)
    if not report.passed:
        raise RuntimeError(f"candidate_physical_materialization_failed:{report.issues}")
    return bundle, report


def run_structure_smoke(
    *,
    output_dir: Path,
    checkpoint: Path,
    config: Path,
    heal_root: Path,
    device: torch.device,
) -> dict[str, Any]:
    from search.integration.data_provider import build_dataset_and_loader, move_batch_to_device

    experiment = _read_config(output_dir)
    first_bundle = _load_bundle(
        checkpoint=checkpoint, config=config, heal_root=heal_root, device=device
    )
    _dataset, loader = build_dataset_and_loader(
        first_bundle.adapter, config, split="val", num_workers=0, visualize=False
    )
    batch = move_batch_to_device(next(iter(loader)), device)
    del first_bundle
    rows = []
    for record in experiment["candidates"]:
        spec = _spec_from_record(record)
        destination = output_dir / "candidates" / spec.candidate_id
        destination.mkdir(parents=True, exist_ok=True)
        try:
            bundle, report = materialize_candidate(
                spec=spec,
                mask_path=Path(record["mask_path"]),
                checkpoint=checkpoint,
                config=config,
                heal_root=heal_root,
                device=device,
            )
            with torch.no_grad():
                outputs = bundle.adapter.forward_for_task(bundle.model, batch)
            finite = all(
                bool(torch.isfinite(value).all())
                for value in outputs.values()
                if torch.is_tensor(value)
            )
            output_shapes = {
                name: list(value.shape)
                for name, value in outputs.items()
                if torch.is_tensor(value)
            }
            status = "ok" if finite else "nonfinite_output"
            row = {
                **spec.to_dict(),
                "status": status,
                "structure_legal": bool(report.passed),
                "finite_output": finite,
                "output_shapes": output_shapes,
                "original_params": int(report.original_parameter_count),
                "params": int(report.physical_parameter_count),
                "param_reduction": 1.0
                - float(report.physical_parameter_count)
                / float(report.original_parameter_count),
                "structure_hash": str(report.structure_hash),
                "report": report,
            }
        except Exception as exc:  # noqa: BLE001
            row = {
                **spec.to_dict(),
                "status": "failed",
                "structure_legal": False,
                "failure_reason": f"{type(exc).__name__}: {exc}",
            }
        _write_json(destination / "shape_audit.json", row)
        rows.append(row)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    _write_json(output_dir / "shape_audit.json", rows)
    _write_csv(output_dir / "shape_audit.csv", rows)
    return {
        "candidate_count": len(rows),
        "passed_count": sum(row.get("status") == "ok" for row in rows),
    }


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({key for row in rows for key in row if key != "report"})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(row.get(key), sort_keys=True, default=str)
                        if isinstance(row.get(key), (dict, list, tuple))
                        else row.get(key, "")
                    )
                    for key in keys
                }
            )


def upsert_candidate_result(
    rows: Iterable[Mapping[str, Any]], record: Mapping[str, Any]
) -> list[dict[str, Any]]:
    candidate_id = str(record["candidate_id"])
    result = [dict(row) for row in rows]
    for index, row in enumerate(result):
        if str(row.get("candidate_id", "")) == candidate_id:
            result[index] = dict(record)
            return result
    result.append(dict(record))
    return result


def _latency_percentile(values: list[float], percentile: float) -> float:
    if not values:
        return float("nan")
    tensor = torch.as_tensor(values, dtype=torch.float64)
    return float(torch.quantile(tensor, float(percentile) / 100.0))


def cobevt_postprocess_mapping(
    outputs: Mapping[str, torch.Tensor],
) -> OrderedDict[str, Mapping[str, torch.Tensor]]:
    values: OrderedDict[str, Mapping[str, torch.Tensor]] = OrderedDict()
    values["ego"] = outputs
    return values


def evaluate_pytorch_candidate(
    *,
    bundle: Any,
    config: Path,
    manifest_path: Path,
    device: torch.device,
) -> dict[str, Any]:
    from torch.utils.data import DataLoader
    from opencood.data_utils.datasets import build_dataset
    from opencood.hypes_yaml import yaml_utils
    from opencood.utils import eval_utils
    from search.integration.data_provider import move_batch_to_device
    from search.integration.gpu_ap_iou import calculate_gpu_tp_fp_for_threshold

    hypes = yaml_utils.load_yaml(str(config))
    hypes = bundle.adapter._absolutize_dataset_paths(hypes)
    dataset = build_dataset(hypes, visualize=True, train=False)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        collate_fn=dataset.collate_batch_test,
        persistent_workers=True,
        prefetch_factor=2,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    warmup_ids = set(str(value) for value in manifest["warmup_frame_ids"])
    eval_ids = set(str(value) for value in manifest["evaluation_frame_ids"])
    split_ids = [str(value) for value in json.loads(Path(hypes["validate_dir"]).read_text())]
    stats = {
        threshold: {"tp": [], "fp": [], "gt": 0, "score": []}
        for threshold in (0.3, 0.5, 0.7)
    }
    evaluated: list[str] = []
    warmed: list[str] = []
    skips: Counter[str] = Counter()
    times: list[float] = []
    bundle.model.eval()
    for index, batch in enumerate(loader):
        if index >= len(split_ids):
            break
        frame_id = split_ids[index]
        role = "warmup" if frame_id in warmup_ids else "eval" if frame_id in eval_ids else ""
        if not role:
            continue
        try:
            batch = move_batch_to_device(batch, device)
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            with torch.no_grad():
                outputs = bundle.adapter.forward_for_task(bundle.model, batch)
            torch.cuda.synchronize(device)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            if not all(
                bool(torch.isfinite(value).all())
                for value in outputs.values()
                if torch.is_tensor(value)
            ):
                raise RuntimeError("nonfinite_output")
            values = cobevt_postprocess_mapping(outputs)
            pred_box, pred_score, gt_box = dataset.post_process(batch, values)
            if role == "warmup":
                warmed.append(frame_id)
                continue
            for threshold in (0.3, 0.5, 0.7):
                calculate_gpu_tp_fp_for_threshold(
                    pred_box,
                    pred_score,
                    gt_box,
                    stats,
                    threshold,
                    device,
                )
            times.append(elapsed_ms)
            evaluated.append(frame_id)
        except Exception as exc:  # noqa: BLE001
            skips[f"{type(exc).__name__}:{exc}"] += 1
            break
    ap = {}
    for threshold in (0.3, 0.5, 0.7):
        value, _, _ = eval_utils.calculate_ap(stats, threshold)
        ap[f"AP{int(threshold * 100):02d}"] = float(value)
    complete = (
        len(warmed) == len(warmup_ids)
        and len(evaluated) == len(eval_ids)
        and not skips
    )
    return {
        "status": "ok" if complete else "evaluation_failed",
        **ap,
        "mAP": sum(ap.values()) / len(ap),
        "frames_total": len(eval_ids),
        "frames_evaluated": len(evaluated),
        "frames_skipped": int(sum(skips.values())),
        "skip_reasons": dict(skips),
        "warmup_frames": len(warmed),
        "forward_p50_ms": _latency_percentile(times, 50),
        "forward_p90_ms": _latency_percentile(times, 90),
        "forward_p99_ms": _latency_percentile(times, 99),
        "latency_classification": "screening_shared_gpu",
        "manifest_hash": str(manifest["manifest_hash"]),
        "dataloader_workers": 8,
        "ap_iou_backend": "gpu",
    }


def run_pytorch500(
    *,
    output_dir: Path,
    checkpoint: Path,
    config: Path,
    heal_root: Path,
    device: torch.device,
) -> dict[str, Any]:
    experiment = _read_config(output_dir)
    manifest = Path(experiment["manifests"]["fixed500"]["path"])
    results_path = output_dir / "ap500_results.json"
    rows = (
        json.loads(results_path.read_text(encoding="utf-8"))
        if results_path.is_file()
        else []
    )
    completed = {str(row["candidate_id"]) for row in rows if row.get("status") == "ok"}
    for record in experiment["candidates"]:
        spec = _spec_from_record(record)
        if spec.candidate_id in completed:
            continue
        bundle, report = materialize_candidate(
            spec=spec,
            mask_path=Path(record["mask_path"]),
            checkpoint=checkpoint,
            config=config,
            heal_root=heal_root,
            device=device,
        )
        evaluation = evaluate_pytorch_candidate(
            bundle=bundle,
            config=config,
            manifest_path=manifest,
            device=device,
        )
        row = {
            **spec.to_dict(),
            **evaluation,
            "params": int(report.physical_parameter_count),
            "param_reduction": 1.0
            - float(report.physical_parameter_count)
            / float(report.original_parameter_count),
            "structure_hash": str(report.structure_hash),
        }
        rows = upsert_candidate_result(rows, row)
        _write_json(results_path, rows)
        _write_csv(output_dir / "ap500_results.csv", rows)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return {
        "candidate_count": len(rows),
        "successful_count": sum(row.get("status") == "ok" for row in rows),
    }


def _origin_map_from_dict(payload: Mapping[str, Any]) -> OnnxOriginMapResult:
    entries = []
    for row in payload.get("entries", []):
        record = dict(row)
        for key in ("weight_shape",):
            if key in record:
                record[key] = tuple(record[key])
        if "root_trace" in record:
            record["root_trace"] = tuple(record["root_trace"])
        entries.append(CanonicalMappingEntry(**record))
    functional = []
    for row in payload.get("functional_compute_groups", []):
        record = dict(row)
        for key in (
            "original_node_names",
            "graph_indices",
            "input_tensors",
            "output_tensors",
        ):
            if key in record:
                if key in {"input_tensors", "output_tensors"}:
                    record[key] = tuple(tuple(values) for values in record[key])
                else:
                    record[key] = tuple(record[key])
        functional.append(CanonicalFunctionalComputeGroup(**record))
    return OnnxOriginMapResult(
        entries=entries,
        source_onnx=str(payload.get("source_onnx", "")),
        unresolved_weighted_nodes=list(payload.get("unresolved_weighted_nodes", [])),
        functional_matmul_nodes=list(payload.get("functional_matmul_nodes", [])),
        functional_compute_groups=functional,
        naming_policy_version=str(payload.get("naming_policy_version", "")),
        schema_version=str(payload.get("schema_version", "onnx-origin-map-v2")),
        origin_map_hash=str(payload.get("origin_map_hash", "")),
    )


def _export_inputs(bundle: Any, config: Path, device: torch.device) -> dict[str, torch.Tensor]:
    from quantization.export.heal_lidar_cobevt import prepare_cobevt_maxk_inputs
    from search.integration.data_provider import build_dataset_and_loader, move_batch_to_device

    _dataset, loader = build_dataset_and_loader(
        bundle.adapter, config, split="val", num_workers=0, visualize=False
    )
    batch = move_batch_to_device(next(iter(loader)), device)
    ego = batch["ego"] if isinstance(batch, Mapping) and "ego" in batch else batch
    return prepare_cobevt_maxk_inputs(
        ego, fixed_k=DEFAULT_FIXED_K, max_cav=2
    )


def run_export_build(
    *,
    output_dir: Path,
    checkpoint: Path,
    config: Path,
    heal_root: Path,
    device: torch.device,
    physical_gpu: int,
    trt_root: Path,
    plugin_path: Path,
) -> dict[str, Any]:
    from search.integration.runtime_environment import discover_trt_environment
    from search.model_families.lidar_cobevt.deployment_recipe import CobevtDeploymentRecipe
    from search.model_families.lidar_cobevt.export_recipe import CobevtExportRecipe
    from search.model_families.lidar_cobevt.operator_probe import make_scatter_parser_compatible
    from search.model_families.lidar_cobevt.quantization_recipe import (
        CobevtQuantizationRecipe,
        apply_cobevt_auxiliary_typed_contract,
    )

    experiment = _read_config(output_dir)
    rows = []
    for record in experiment["candidates"]:
        spec = _spec_from_record(record)
        destination = output_dir / "candidates" / spec.candidate_id / "fp16_engine"
        report_path = destination / "build_report.json"
        if report_path.is_file():
            rows.append(json.loads(report_path.read_text(encoding="utf-8")))
            continue
        destination.mkdir(parents=True, exist_ok=True)
        try:
            bundle, physical = materialize_candidate(
                spec=spec,
                mask_path=Path(record["mask_path"]),
                checkpoint=checkpoint,
                config=config,
                heal_root=heal_root,
                device=device,
            )
            inputs = _export_inputs(bundle, config, device)
            source = destination / "source.onnx"
            export = CobevtExportRecipe(
                fixed_k=DEFAULT_FIXED_K, max_cav=2
            ).export(bundle.model, inputs, source)
            origin = _origin_map_from_dict(export.origin_map)
            quant = CobevtQuantizationRecipe()
            capability = quant.build_capability(origin)
            requested = {
                row.module_path: "FP16" for row in capability.weighted_entries
            }
            profile = quant.build_profile(
                capability, requested, profile_id=f"{spec.candidate_id}_strict_fp16"
            )
            mapping = quant.build_mapping(origin, profile)
            typed = destination / "typed.onnx"
            typed_report = apply_strongly_typed_precision_contract(
                source, typed, mapping, plugin_boundary="FP32"
            )
            typed_aux = destination / "typed_aux.onnx"
            auxiliary = apply_cobevt_auxiliary_typed_contract(typed, typed_aux)
            parser = destination / "typed_parser.onnx"
            parser_report = make_scatter_parser_compatible(typed_aux, parser)
            environment = discover_trt_environment(
                trt_root, plugin_path=plugin_path, conda_env="modelopt"
            )
            engine = destination / "engine.plan"
            layer_info = destination / "engine_layer_info.json"
            deployment = CobevtDeploymentRecipe(
                tensorrt_root=trt_root,
                trtexec_path=environment.trtexec_path,
                plugin_path=plugin_path,
                fixed_k=DEFAULT_FIXED_K,
            )
            command = deployment.builder_command(
                onnx_path=parser,
                engine_path=engine,
                mapping=mapping,
                layer_info_path=layer_info,
            )
            env = dict(environment.env)
            env["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
            modelopt_lib = Path("/home/lixingfeng/anaconda3/envs/modelopt/lib")
            env["LD_LIBRARY_PATH"] = ":".join(
                (str(modelopt_lib), env.get("LD_LIBRARY_PATH", ""))
            )
            started = time.monotonic()
            completed = subprocess.run(
                command.command,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                check=False,
                timeout=3600,
            )
            (destination / "trtexec.log").write_text(
                completed.stdout or "", encoding="utf-8"
            )
            if completed.returncode or not engine.is_file() or not layer_info.is_file():
                raise RuntimeError(f"trtexec_build_failed_rc_{completed.returncode}")
            realization = validate_precision_realization(layer_info, mapping)
            if not realization.passed:
                raise RuntimeError(f"precision_realization_failed:{realization.mismatches}")
            result = {
                **spec.to_dict(),
                "status": "ok",
                "build_elapsed_seconds": time.monotonic() - started,
                "builder_command": command.to_dict(),
                "engine_path": str(engine),
                "engine_sha256": _sha256(engine),
                "engine_size_bytes": engine.stat().st_size,
                "export": export.to_dict(),
                "layer_info_path": str(layer_info),
                "mapping": mapping.to_dict(),
                "physical_parameter_count": int(physical.physical_parameter_count),
                "precision_realization": realization.to_dict(),
                "profile": profile.to_dict(),
                "structure_hash": str(physical.structure_hash),
                "typed_report": typed_report,
                "auxiliary_typed_report": auxiliary,
                "parser_report": parser_report,
            }
        except Exception as exc:  # noqa: BLE001
            result = {
                **spec.to_dict(),
                "status": "failed",
                "failure_reason": f"{type(exc).__name__}: {exc}",
            }
        _write_json(report_path, result)
        rows.append(result)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    _write_json(output_dir / "full_engine_build_results.json", rows)
    _write_csv(output_dir / "full_engine_build_results.csv", rows)
    return {
        "candidate_count": len(rows),
        "successful_count": sum(row.get("status") == "ok" for row in rows),
    }


def run_trt500(
    *,
    output_dir: Path,
    checkpoint: Path,
    config: Path,
    heal_root: Path,
    physical_gpu: int,
    trt_root: Path,
    plugin_path: Path,
) -> dict[str, Any]:
    from search.integration.lidar_cobevt_evaluation_provider import (
        evaluate_cobevt_engine_modelopt,
    )

    experiment = _read_config(output_dir)
    manifest = Path(experiment["manifests"]["fixed500"]["path"])
    rows = []
    for record in experiment["candidates"]:
        spec = _spec_from_record(record)
        candidate = output_dir / "candidates" / spec.candidate_id / "fp16_engine"
        build_report = candidate / "build_report.json"
        if not build_report.is_file():
            continue
        build = json.loads(build_report.read_text(encoding="utf-8"))
        if build.get("status") != "ok":
            rows.append(
                {
                    **spec.to_dict(),
                    "status": "build_failed",
                    "failure_reason": build.get("failure_reason", ""),
                }
            )
            continue
        evaluation_dir = candidate / "evaluation_fixed500"
        result_path = evaluation_dir / "evaluation.json"
        if result_path.is_file():
            evaluation = json.loads(result_path.read_text(encoding="utf-8"))
        else:
            evaluation = evaluate_cobevt_engine_modelopt(
                engine_path=build["engine_path"],
                checkpoint=checkpoint,
                model_config=config,
                heal_root=heal_root,
                device=f"cuda:{physical_gpu}",
                output_dir=evaluation_dir,
                tensorrt_root=trt_root,
                plugin_path=plugin_path,
                fixed_k=DEFAULT_FIXED_K,
                num_frames=500,
                warmup_frames=20,
                eval_manifest_path=manifest,
                num_workers=8,
                ap_iou_backend="gpu",
                latency_rounds=1,
            )
        row = {
            **spec.to_dict(),
            **evaluation,
            "structure_hash": build.get("structure_hash", ""),
            "engine_sha256": build.get("engine_sha256", ""),
            "params": build.get("physical_parameter_count", 0),
            "latency_classification": "screening_shared_gpu",
        }
        rows.append(row)
        _write_json(output_dir / "trt_ap500_results.json", rows)
        _write_csv(output_dir / "trt_ap500_results.csv", rows)
    return {
        "candidate_count": len(rows),
        "successful_count": sum(row.get("status") == "ok" for row in rows),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=("prepare", "structure", "pytorch500", "export-build", "trt500"),
        required=True,
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--heal-root", default=str(DEFAULT_HEAL_ROOT))
    parser.add_argument("--trt-root", default=str(DEFAULT_TRT_ROOT))
    parser.add_argument("--plugin", default=str(DEFAULT_PLUGIN))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--physical-gpu", type=int, default=2)
    parser.add_argument("--gradient-samples", type=int, default=50)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    common = {
        "output_dir": Path(args.output_dir).expanduser().resolve(),
        "checkpoint": Path(args.checkpoint).expanduser().resolve(),
        "config": Path(args.config).expanduser().resolve(),
        "heal_root": Path(args.heal_root).expanduser().resolve(),
    }
    device = torch.device(args.device)
    if args.phase == "prepare":
        result = run_prepare(
            **common,
            device=device,
            gradient_samples=int(args.gradient_samples),
        )
    elif args.phase == "structure":
        result = run_structure_smoke(**common, device=device)
    elif args.phase == "pytorch500":
        result = run_pytorch500(**common, device=device)
    elif args.phase == "export-build":
        result = run_export_build(
            **common,
            device=device,
            physical_gpu=int(args.physical_gpu),
            trt_root=Path(args.trt_root).expanduser().resolve(),
            plugin_path=Path(args.plugin).expanduser().resolve(),
        )
    else:
        result = run_trt500(
            **common,
            physical_gpu=int(args.physical_gpu),
            trt_root=Path(args.trt_root).expanduser().resolve(),
            plugin_path=Path(args.plugin).expanduser().resolve(),
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


__all__ = [
    "AttentionCandidateSpec",
    "formal_candidate_specs",
    "load_attention_masks",
    "manifest_summary",
    "write_attention_evaluation_manifests",
    "write_attention_masks",
]


if __name__ == "__main__":
    raise SystemExit(main())
