from __future__ import annotations

from pathlib import Path
from typing import Any

from .dummy_input_builder import build_dataset_dummy_input
def load_lidar_pyramid_model(
    config: str,
    checkpoint: str,
    heal_root: str,
    device: str = "cpu",
) -> tuple[Any, Any]:
    """Deprecated HEAL loader with caller-supplied repository paths."""

    import torch
    try:
        from ..adapters.heal_lidar_adapter import HEALLiDARAdapter
    except ImportError:
        from adapters.heal_lidar_adapter import HEALLiDARAdapter

    if not Path(config).is_file():
        raise FileNotFoundError(f"config not found: {config}")
    if not Path(checkpoint).is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    adapter = HEALLiDARAdapter(heal_repo=heal_root, config={"model": {"hypes_yaml": str(config)}})
    model = adapter.build_model(str(config), str(checkpoint)).to(torch.device(device)).eval()
    return model, adapter


def trace_lidar_pyramid(
    *,
    config: str,
    checkpoint: str,
    heal_root: str,
    device: str = "cpu",
    split: str = "train",
) -> tuple[dict[str, Any], Any, dict[str, Any]]:
    from .generic_tracer import trace_model

    model, adapter = load_lidar_pyramid_model(config, checkpoint, heal_root=heal_root, device=device)
    sample, meta = build_dataset_dummy_input(adapter, model, config, split=split)
    trace = trace_model(model, sample, forward_fn=adapter.forward_for_task)
    return trace, model, meta
