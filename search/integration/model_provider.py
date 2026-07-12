"""Real HEAL lidar_pyramid model and trace provider."""

from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


DEFAULT_HEAL_ROOT = Path("/home/lixingfeng/UniAD_examine/HEAL")
DEFAULT_CONFIG = Path("/home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_pyramid/config.yaml")


@dataclass
class LidarPyramidModelBundle:
    model: torch.nn.Module
    adapter: Any
    model_config_path: Path
    checkpoint_path: Path
    checkpoint_hash: str
    trace_example_inputs: Any
    trace_result: Any


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_lidar_pyramid_model(
    *,
    checkpoint_path: str | Path,
    model_config_path: str | Path | None = None,
    heal_root: str | Path = DEFAULT_HEAL_ROOT,
    device: str = "cuda:0",
    trace: bool = True,
) -> LidarPyramidModelBundle:
    root = Path(heal_root).expanduser().resolve()
    if str(root.parent) not in sys.path:
        sys.path.insert(0, str(root.parent))
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if str(Path(__file__).resolve().parents[3]) not in sys.path:
        sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    try:
        from heal_compress.adapters.heal_lidar_adapter import HEALLiDARAdapter
        from heal_compress.tracer.api import trace_model
        from heal_compress.tracer.config import TraceConfig
    except ImportError:
        from adapters.heal_lidar_adapter import HEALLiDARAdapter
        from tracer.api import trace_model
        from tracer.config import TraceConfig

    ckpt = Path(checkpoint_path).expanduser().resolve()
    config = Path(model_config_path or DEFAULT_CONFIG).expanduser().resolve()
    if not ckpt.is_file():
        raise RuntimeError(f"checkpoint_missing:{ckpt}")
    if not config.is_file():
        raise RuntimeError(f"missing_model_config:{config}")
    torch_device = torch.device(device)
    if torch_device.type == "cuda":
        torch.cuda.set_device(torch_device)
    adapter = HEALLiDARAdapter(heal_repo=str(root), config={"model": {"hypes_yaml": str(config)}})
    model = adapter.build_model(str(config), str(ckpt)).to(torch_device).eval()
    example = adapter.build_synthetic_batch(model)
    trace_result = None
    if trace:
        trace_result = trace_model(
            model,
            example,
            config=TraceConfig(fail_on_fx_trace_error=False),
            forward_fn=adapter.forward_for_task,
        )
    return LidarPyramidModelBundle(
        model=model,
        adapter=adapter,
        model_config_path=config,
        checkpoint_path=ckpt,
        checkpoint_hash=sha256_file(ckpt),
        trace_example_inputs=example,
        trace_result=trace_result,
    )
