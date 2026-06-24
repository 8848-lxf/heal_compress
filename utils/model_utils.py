"""Model loading, inspection, GPU selection, and helper utilities for HEAL models."""

from __future__ import annotations

import importlib.util
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# GPUs excluded by default (can be overridden by explicit --device)
_DEFAULT_EXCLUDED_GPUS = {5, 6, 7}


def auto_select_gpu(exclude: set[int] | None = None) -> str:
    """Select the GPU with the most free memory, excluding specified indices.

    Excludes GPUs 5, 6, 7 by default unless overridden.

    Args:
        exclude: Set of GPU indices to skip. Defaults to {5, 6, 7}.

    Returns:
        Device string like 'cuda:0'. Falls back to 'cpu' if no GPU available.
    """
    exclude = exclude if exclude is not None else _DEFAULT_EXCLUDED_GPUS
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            return "cpu"
        candidates = []
        for line in result.stdout.strip().split("\n"):
            parts = line.strip().split(",")
            if len(parts) != 2:
                continue
            idx = int(parts[0].strip())
            free_mb = int(parts[1].strip())
            if idx in exclude:
                continue
            candidates.append((idx, free_mb))
        if not candidates:
            return "cpu"
        candidates.sort(key=lambda x: -x[1])
        best_idx = candidates[0][0]
        logger.info(
            f"Auto-selected GPU {best_idx} "
            f"(free={candidates[0][1]} MB, excluded={sorted(exclude)})"
        )
        return f"cuda:{best_idx}"
    except Exception as exc:
        logger.warning(f"GPU auto-select via nvidia-smi failed: {exc}, trying torch.cuda fallback")
        # Fallback: use torch.cuda to find a GPU with the most free memory
        try:
            import torch as _torch
            if not _torch.cuda.is_available():
                return "cpu"
            best_idx, best_free = -1, -1
            for i in range(_torch.cuda.device_count()):
                if i in exclude:
                    continue
                free, total = _torch.cuda.mem_get_info(i)
                if free > best_free:
                    best_free = free
                    best_idx = i
            if best_idx >= 0:
                logger.info(f"Auto-selected GPU {best_idx} via torch.cuda (free={best_free // (1024*1024)} MB)")
                return f"cuda:{best_idx}"
        except Exception as exc2:
            logger.warning(f"torch.cuda fallback also failed: {exc2}")
        return "cpu"


def resolve_device(device: str | None, exclude_gpus: set[int] | None = None) -> str:
    """Resolve device string: use explicit if given, else auto-select GPU.

    Args:
        device: Explicit device string (e.g. 'cuda:0') or None for auto.
        exclude_gpus: GPUs to exclude during auto-selection.

    Returns:
        Device string.
    """
    if device is not None and device != "auto":
        return device
    return auto_select_gpu(exclude=exclude_gpus)


def import_export_module(source_script: str | Path) -> Any:
    """Dynamically import the HEAL export_dynamic_onnx.py module.

    Args:
        source_script: Absolute path to export_dynamic_onnx.py.

    Returns:
        The imported module object with functions like install_plugin_patches,
        load_model_and_hypes, DynamicAgentExportWrapper, etc.

    Raises:
        FileNotFoundError: If the script is not found.
    """
    source_script = Path(source_script).expanduser().resolve()
    if not source_script.is_file():
        raise FileNotFoundError(f"export script not found: {source_script}")
    if str(source_script.parent) not in sys.path:
        sys.path.insert(0, str(source_script.parent))
    spec = importlib.util.spec_from_file_location(
        "heal_dynamic_export", str(source_script)
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_model(
    checkpoint: str,
    config: str | None = None,
    device: str | torch.device = "cpu",
    export_script: str | None = None,
) -> tuple[nn.Module, Any]:
    """Load a HEAL model from checkpoint using the export script's loader.

    Falls back to direct torch.load if the export script is unavailable.

    Args:
        checkpoint: Path to model checkpoint (.pth).
        config: Optional HEAL YAML config path.
        device: Target device.
        export_script: Path to export_dynamic_onnx.py.

    Returns:
        Tuple of (model, hypes_dict_or_None).
    """
    device = torch.device(device)
    if export_script:
        mod = import_export_module(export_script)
        model, hypes = mod.load_model_and_hypes(Path(checkpoint), config, device)
        return model, hypes

    obj = torch.load(str(checkpoint), map_location=device)
    if isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], nn.Module):
        return obj["model"].to(device).eval(), obj.get("hypes")
    if isinstance(obj, nn.Module):
        return obj.to(device).eval(), None
    raise RuntimeError(
        "Cannot load model: checkpoint is a state_dict. Please provide --config and --export-script."
    )


def load_calibration_npz(
    npz_path: str,
    device: str | torch.device = "cpu",
    require_num_agents: int = 0,
) -> tuple[torch.Tensor, ...]:
    """Load calibration data from a .npz file in HEAL camera format.

    The npz must contain: imgs, rots, trans, intrins, post_rots, post_trans,
    pairwise_t_matrix.

    Args:
        npz_path: Path to the .npz file.
        device: Target device for tensors.
        require_num_agents: If >0, verify the npz has this many agents.

    Returns:
        Tuple of (imgs, rots, trans, intrins, post_rots, post_trans,
        pairwise_t_matrix).

    Raises:
        FileNotFoundError: If the npz file is not found.
        KeyError: If required keys are missing.
    """
    npz_path = Path(npz_path)
    if not npz_path.is_file():
        raise FileNotFoundError(f"calibration npz not found: {npz_path}")

    with np.load(str(npz_path)) as data:
        npz = {key: data[key] for key in data.files}

    def _tensor(key: str) -> torch.Tensor:
        if key not in npz:
            raise KeyError(f"npz missing required key '{key}'")
        return torch.from_numpy(np.asarray(npz[key])).to(
            device=torch.device(device), dtype=torch.float32
        )

    def _matrix(key: str, inv_key: str) -> torch.Tensor:
        if key in npz:
            return _tensor(key)
        if inv_key not in npz:
            raise KeyError(f"npz must contain '{key}' or '{inv_key}'")
        inv = np.asarray(npz[inv_key]).astype(np.float32)
        mat = np.linalg.inv(inv.reshape(-1, inv.shape[-2], inv.shape[-1])).reshape(
            inv.shape
        )
        return torch.from_numpy(mat).to(device=torch.device(device), dtype=torch.float32)

    imgs = _tensor("imgs")
    total_agents = int(imgs.shape[0])
    if require_num_agents and total_agents != require_num_agents:
        raise RuntimeError(
            f"Expected {require_num_agents} agents, npz has {total_agents}"
        )

    rots = _tensor("rots")
    trans = _tensor("trans")
    intrins = _matrix("intrins", "intrins_inv")
    post_rots = _matrix("post_rots", "post_rots_inv")
    post_trans = _tensor("post_trans")
    pairwise = _tensor("pairwise_t_matrix")

    if pairwise.dim() == 5 and pairwise.shape[0] == 1:
        pairwise = pairwise[:, :total_agents, :total_agents, :, :]

    return imgs, rots, trans, intrins, post_rots, post_trans, pairwise


def get_protected_layer_names(model: nn.Module) -> list[str]:
    """Identify layers that must NOT be pruned in a HEAL model.

    Protected layers include:
    - cls_head, reg_head, dir_head final output layers
    - LSS frustum/camC/D-related convolutions
    - Pyramid backbone cat output dimension
    - Camera intrinsic interface layers

    Args:
        model: The HEAL model.

    Returns:
        List of protected layer names.
    """
    protected_keywords = (
        "cls_head", "reg_head", "dir_head",
        "frustum", "camC",
    )
    protected = []
    for name, _ in model.named_modules():
        low = name.lower()
        if any(k.lower() in low for k in protected_keywords):
            protected.append(name)
    return protected


def detect_modality(model: nn.Module) -> str | None:
    """Detect the primary modality of a HEAL model.

    Args:
        model: The HEAL model.

    Returns:
        Modality name string (e.g. 'm1', 'm2') or None if not detected.
    """
    for modality_name in getattr(model, "modality_name_list", []):
        if getattr(model, "sensor_type_dict", {}).get(modality_name) == "camera":
            return modality_name
    for name in getattr(model, "modality_name_list", []):
        if hasattr(model, f"encoder_{name}"):
            return name
    return None
