"""Model artifact helpers for v10.8 physical pruning experiments."""

from __future__ import annotations

import copy
import traceback
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn


def save_v108_model_artifacts(
    *,
    model: nn.Module,
    models_dir: Path,
    manifest: Mapping[str, Any],
    model_config: str,
    checkpoint_source: str,
) -> dict[str, Path]:
    models_dir.mkdir(parents=True, exist_ok=True)
    model_cpu = copy.deepcopy(model).cpu().eval()
    model_object_path = models_dir / "pruned_model_object.pth"
    state_manifest_path = models_dir / "pruned_state_dict_with_manifest.pth"
    torch.save(
        {
            "model_object": model_cpu,
            "config": str(model_config),
            "checkpoint_source": str(checkpoint_source),
            "architecture_changed": True,
            "manifest": dict(manifest),
        },
        model_object_path,
    )
    torch.save(
        {
            "state_dict": model_cpu.state_dict(),
            "config": str(model_config),
            "checkpoint_source": str(checkpoint_source),
            "architecture_manifest": dict(manifest),
            "requires_architecture_patch": True,
        },
        state_manifest_path,
    )
    return {"model_object": model_object_path, "state_dict_manifest": state_manifest_path}


def _torch_load(path: Path, *, map_location: torch.device | str) -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_v108_model_object(path: Path, *, device: torch.device) -> nn.Module:
    payload = _torch_load(path, map_location=device)
    model = payload.get("model_object") if isinstance(payload, dict) else None
    if not isinstance(model, nn.Module):
        raise TypeError("v10.8 model artifact does not contain an nn.Module model_object")
    return model.to(device).eval()


def smoke_v108_model(model: nn.Module, sample: Any, *, forward_fn: Any | None = None) -> dict[str, Any]:
    try:
        with torch.no_grad():
            output = forward_fn(model, sample) if forward_fn is not None else model(sample)
        return {
            "reload_forward_smoke_passed": True,
            "output_finite": _output_finite(output),
            "failure_reason": "",
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "reload_forward_smoke_passed": False,
            "output_finite": False,
            "failure_reason": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }


def _output_finite(value: Any) -> bool:
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all().item()) if value.is_floating_point() else True
    if isinstance(value, Mapping):
        return all(_output_finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(_output_finite(v) for v in value)
    return True
