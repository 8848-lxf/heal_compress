"""Real Fisher and Q/DQ calibration providers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from ..hashing import canonical_json_hash
from ..proxy.fisher_proxy import FisherStatistics
from .data_provider import build_dataset_and_loader, iter_limited, move_batch_to_device


def _tensor_dict_to_cpu(rows: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in rows.items()}


def collect_or_load_fisher_statistics(
    *,
    model: torch.nn.Module,
    adapter: Any,
    model_config_path: str | Path,
    device: torch.device,
    cache_path: str | Path,
    num_batches: int,
) -> FisherStatistics:
    path = Path(cache_path)
    if path.is_file():
        payload = torch.load(path, map_location="cpu")
        return FisherStatistics(
            gradients={key: value for key, value in payload["gradients"].items()},
            fisher_diag={key: value for key, value in payload["fisher_diag"].items()},
            manifest_hash=str(payload.get("manifest_hash", "")),
            statistics_version=str(payload.get("statistics_version", "fisher-diagonal-v1")),
        )
    if int(num_batches) <= 0:
        raise RuntimeError("fisher_statistics_missing:num_batches")
    _dataset, loader = build_dataset_and_loader(adapter, model_config_path, split="train", num_workers=0, visualize=False)
    batches = iter_limited(loader, int(num_batches))
    if not batches:
        raise RuntimeError("fisher_statistics_missing:no_calibration_batches")
    gradients: dict[str, torch.Tensor] = {}
    fisher: dict[str, torch.Tensor] = {}
    model.train(False)
    for batch in batches:
        batch = move_batch_to_device(batch, device)
        model.zero_grad(set_to_none=True)
        output = adapter.forward_for_task(model, batch)
        loss = adapter.compute_task_loss(output, batch)
        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            grad = param.grad.detach()
            gradients.setdefault(name, torch.zeros_like(param.detach(), device=grad.device))
            fisher.setdefault(name, torch.zeros_like(param.detach(), device=grad.device))
            gradients[name] += grad
            fisher[name] += grad.pow(2)
    count = float(len(batches))
    for name in list(gradients):
        gradients[name] = gradients[name] / count
        fisher[name] = fisher[name] / count
    manifest_hash = canonical_json_hash({"split": "train", "num_batches": int(num_batches), "model_config": str(model_config_path)})
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "gradients": _tensor_dict_to_cpu(gradients),
            "fisher_diag": _tensor_dict_to_cpu(fisher),
            "manifest_hash": manifest_hash,
            "statistics_version": "fisher-diagonal-v1",
        },
        path,
    )
    return FisherStatistics(_tensor_dict_to_cpu(gradients), _tensor_dict_to_cpu(fisher), manifest_hash=manifest_hash)


def weight_only_calibration_scales(module_paths: list[str], model: torch.nn.Module) -> dict[str, dict[str, float]]:
    """Fallback scale payload with real per-module weight scales and conservative activation scales."""

    modules = dict(model.named_modules())
    scales: dict[str, dict[str, float]] = {}
    for name in module_paths:
        module = modules.get(name)
        weight = getattr(module, "weight", None)
        if weight is None:
            continue
        weight_amax = float(weight.detach().abs().amax().item())
        scale = max(weight_amax / 127.0, 1.0e-8)
        scales[name] = {
            "activation_input_scale": 1.0,
            "weight_scale": scale,
            "activation_output_scale": 1.0,
            "activation_source": "fallback_static_unit_scale",
            "weight_source": "actual_pruned_weight_absmax_div127",
        }
    return scales


def collect_or_load_qdq_calibration_scales(
    *,
    model: torch.nn.Module,
    adapter: Any,
    model_config_path: str | Path,
    module_paths: list[str],
    device: torch.device,
    cache_path: str | Path,
    num_batches: int,
) -> dict[str, dict[str, float]]:
    path = Path(cache_path)
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        return dict(payload["scales"])
    if int(num_batches) <= 0:
        raise RuntimeError("calibration_scales_missing:num_batches")
    try:
        from quantization.api import collect_calibration_scales
        from quantization.config import CalibrationConfig
    except ImportError:
        from heal_compress.quantization.api import collect_calibration_scales
        from heal_compress.quantization.config import CalibrationConfig
    _dataset, loader = build_dataset_and_loader(adapter, model_config_path, split="train", num_workers=0, visualize=False)
    batches = [move_batch_to_device(batch, device) for batch in iter_limited(loader, int(num_batches))]
    if not batches:
        raise RuntimeError("calibration_scales_missing:no_calibration_batches")

    def forward_fn(inner_model: torch.nn.Module, batch: Any) -> Any:
        return adapter.forward_for_task(inner_model, batch)

    result = collect_calibration_scales(
        model,
        batches,
        module_paths=module_paths,
        forward_fn=forward_fn,
        config=CalibrationConfig(frame_count=len(batches), require_observed_scales=True),
    )
    scales = result.scales()
    save_calibration_scales(
        path,
        scales,
        {
            "frame_count": len(batches),
            "module_count": len(module_paths),
            "manifest_hash": canonical_json_hash({"split": "train", "frames": len(batches), "modules": sorted(module_paths)}),
            "source": "quantization.collect_calibration_scales",
        },
    )
    return scales


def save_calibration_scales(path: str | Path, scales: dict[str, Any], metadata: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({"metadata": metadata, "scales": scales}, indent=2, sort_keys=True), encoding="utf-8")
