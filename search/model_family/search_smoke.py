"""Real-data helpers for bounded V2X-ViT GA/greedy framework smoke tests."""

from __future__ import annotations

import random
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from ..candidate import CandidatePhenotype
from ..hashing import canonical_json_hash
from ..integration.data_provider import move_batch_to_device
from ..proxy.fisher_proxy import FisherStatistics
from ..pruning_space.local_domains import LocalPruningDomain
from .pruning import (
    apply_v2xvit_weight_fake_quantization,
    materialize_v2xvit_ffn_pruning,
)


def _set_sample_rng(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    torch.manual_seed(int(seed))


def load_v2xvit_manifest_batches(
    bundle: Any,
    manifest: Mapping[str, Any],
    *,
    sample_count: int,
    device: torch.device,
    prefer_two_agents: bool = True,
) -> tuple[list[Any], list[dict[str, Any]]]:
    from opencood.data_utils.datasets import build_dataset
    from opencood.hypes_yaml import yaml_utils

    hypes = yaml_utils.load_yaml(str(bundle.config_path))
    hypes = bundle.adapter._absolutize_dataset_paths(hypes)
    dataset = build_dataset(hypes, visualize=False, train=True)
    samples = [dict(row) for row in manifest["samples"]]
    if prefer_two_agents:
        samples = [row for row in samples if int(row["record_len"]) == 2] + [
            row for row in samples if int(row["record_len"]) != 2
        ]
    selected = samples[: int(sample_count)]
    if len(selected) != int(sample_count):
        raise RuntimeError(
            f"v2xvit_manifest_insufficient_calibration_samples:{len(selected)}<{sample_count}"
        )
    batches = []
    evidence = []
    for row in selected:
        _set_sample_rng(int(row["sample_seed"]))
        item = dataset[int(row["dataset_index"])]
        batch = dataset.collate_batch_train([item])
        if batch is None:
            raise RuntimeError(
                f"v2xvit_manifest_calibration_batch_none:{row['dataset_index']}"
            )
        observed_k = int(batch["ego"]["inputs_m1"]["voxel_features"].shape[0])
        if observed_k != int(row["voxel_count"]):
            raise RuntimeError(
                f"v2xvit_manifest_calibration_k_mismatch:{row['dataset_index']}:"
                f"{observed_k}!={row['voxel_count']}"
            )
        batches.append(move_batch_to_device(batch, device))
        evidence.append(
            {
                "ordinal": int(row["ordinal"]),
                "dataset_index": int(row["dataset_index"]),
                "vehicle_frame_id": str(row["vehicle_frame_id"]),
                "record_len": int(row["record_len"]),
                "voxel_count": observed_k,
                "sample_seed": int(row["sample_seed"]),
            }
        )
    return batches, evidence


def collect_v2xvit_manifest_fisher_statistics(
    bundle: Any,
    manifest: Mapping[str, Any],
    batches: Sequence[Any],
    sample_evidence: Sequence[Mapping[str, Any]],
) -> tuple[FisherStatistics, dict[str, Any]]:
    if not batches:
        raise RuntimeError("v2xvit_fisher_requires_manifest_batches")
    gradients: dict[str, torch.Tensor] = {}
    fisher: dict[str, torch.Tensor] = {}
    losses: list[float] = []
    model = bundle.model
    model.train(False)
    for batch in batches:
        model.zero_grad(set_to_none=True)
        outputs = bundle.adapter.forward_for_task(model, batch)
        loss = bundle.adapter.compute_task_loss(outputs, batch)
        if not torch.isfinite(loss):
            raise RuntimeError("v2xvit_fisher_nonfinite_task_loss")
        loss.backward()
        losses.append(float(loss.detach().cpu()))
        for name, parameter in model.named_parameters():
            if parameter.grad is None:
                continue
            gradient = parameter.grad.detach()
            gradients.setdefault(name, torch.zeros_like(gradient))
            fisher.setdefault(name, torch.zeros_like(gradient))
            gradients[name] += gradient
            fisher[name] += gradient.square()
    count = float(len(batches))
    gradients_cpu = {
        name: (value / count).detach().cpu() for name, value in gradients.items()
    }
    fisher_cpu = {
        name: (value / count).detach().cpu() for name, value in fisher.items()
    }
    model.zero_grad(set_to_none=True)
    identity = {
        "schema_version": "v2xvit-real-manifest-fisher-smoke-v1",
        "checkpoint_hash": bundle.checkpoint_hash,
        "calibration_manifest_hash": manifest["manifest_hash"],
        "samples": [dict(row) for row in sample_evidence],
        "task_loss": "HEAL configured detection task loss",
        "formula": "mean_gradient_and_empirical_fisher_E_gradient_squared",
    }
    manifest_hash = canonical_json_hash(identity)
    statistics = FisherStatistics(
        gradients=gradients_cpu,
        fisher_diag=fisher_cpu,
        manifest_hash=manifest_hash,
        statistics_version="v2xvit-real-manifest-fisher-smoke-v1",
    )
    report = {
        **identity,
        "manifest_hash": manifest_hash,
        "batch_count": len(batches),
        "losses": losses,
        "gradient_parameter_count": len(gradients_cpu),
        "fisher_parameter_count": len(fisher_cpu),
        "all_gradients_finite": all(
            bool(torch.isfinite(value).all()) for value in gradients_cpu.values()
        ),
        "all_fisher_finite": all(
            bool(torch.isfinite(value).all()) for value in fisher_cpu.values()
        ),
        "statistics_tensors_persisted": False,
    }
    return statistics, report


def _output_tensors(outputs: Any) -> dict[str, torch.Tensor]:
    if not isinstance(outputs, dict):
        raise RuntimeError("v2xvit_stage2_smoke_output_not_mapping")
    rows = {
        key: value
        for key, value in outputs.items()
        if key in {"cls_preds", "reg_preds", "dir_preds"}
        and torch.is_tensor(value)
    }
    if set(rows) != {"cls_preds", "reg_preds", "dir_preds"}:
        raise RuntimeError(f"v2xvit_stage2_smoke_output_contract:{sorted(rows)}")
    return rows


def _parity(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, Any]:
    ref = reference.detach().float().reshape(-1)
    value = actual.detach().float().reshape(-1)
    difference = (ref - value).abs()
    denominator = ref.norm() * value.norm()
    cosine = float((ref @ value / denominator.clamp_min(1.0e-12)).cpu())
    return {
        "shape": list(actual.shape),
        "finite": bool(torch.isfinite(actual).all()),
        "max_abs": float(difference.max().cpu()),
        "mean_abs": float(difference.mean().cpu()),
        "cosine": cosine,
    }


def run_v2xvit_physical_stage2_smoke(
    bundle: Any,
    phenotype: CandidatePhenotype,
    domains: Sequence[LocalPruningDomain],
    batch: Any,
    *,
    latency_rounds: int = 3,
) -> dict[str, Any]:
    with torch.no_grad():
        reference = _output_tensors(bundle.adapter.forward_for_task(bundle.model, batch))
    physical = materialize_v2xvit_ffn_pruning(bundle.model, phenotype, domains)
    reload_target = materialize_v2xvit_ffn_pruning(bundle.model, phenotype, domains)
    reload_target.model.load_state_dict(physical.model.state_dict(), strict=True)
    fake_quant = apply_v2xvit_weight_fake_quantization(physical.model, phenotype)
    physical.model.eval()
    with torch.no_grad():
        outputs = _output_tensors(bundle.adapter.forward_for_task(physical.model, batch))
        # One warmup, then bounded CUDA-event timing. This is only a smoke
        # measurement and is never compared with production TensorRT latency.
        bundle.adapter.forward_for_task(physical.model, batch)
        device = next(physical.model.parameters()).device
        if device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(max(1, int(latency_rounds))):
                bundle.adapter.forward_for_task(physical.model, batch)
            end.record()
            torch.cuda.synchronize(device)
            mean_ms = float(start.elapsed_time(end) / max(1, int(latency_rounds)))
        else:
            started = time.perf_counter()
            for _ in range(max(1, int(latency_rounds))):
                bundle.adapter.forward_for_task(physical.model, batch)
            mean_ms = (time.perf_counter() - started) * 1000.0 / max(
                1, int(latency_rounds)
            )
    parity = {key: _parity(reference[key], outputs[key]) for key in sorted(outputs)}
    passed = (
        all(bool(row["finite"]) for row in parity.values())
        and physical.parameter_count_after <= physical.parameter_count_before
    )
    return {
        "passed": passed,
        "backend": "real_pytorch_physical_model_plus_weight_fake_quant_smoke",
        "explicit_qdq_applied": False,
        "tensorrt_engine_built": False,
        "precision_realization_claimed": False,
        "checkpoint_strict_reload": True,
        "snapshot": physical.snapshot,
        "snapshot_hash": physical.snapshot_hash,
        "parameter_count_before": physical.parameter_count_before,
        "parameter_count_after": physical.parameter_count_after,
        "parameter_reduction": physical.parameter_count_before
        - physical.parameter_count_after,
        "real_forward": True,
        "output_parity": parity,
        "weight_fake_quantization": fake_quant,
        "latency_smoke_mean_ms": mean_ms,
        "latency_rounds": max(1, int(latency_rounds)),
    }


__all__ = [
    "collect_v2xvit_manifest_fisher_statistics",
    "load_v2xvit_manifest_batches",
    "run_v2xvit_physical_stage2_smoke",
]
