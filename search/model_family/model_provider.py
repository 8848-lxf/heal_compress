"""Strict HEAL model loading shared by new model-family adapters."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import sys
from typing import Any

import torch

from .contracts import ModelFamilyAudit
from .registry import ModelFamilyProvider, detect_model_family, get_model_family


@dataclass
class HealModelFamilyBundle:
    model: torch.nn.Module
    adapter: Any
    provider: ModelFamilyProvider
    audit: ModelFamilyAudit
    config: dict[str, Any]
    config_path: Path
    checkpoint_path: Path
    config_hash: str
    checkpoint_hash: str
    state_dict_tensor_count: int
    example_batch: Any | None = None
    smoke_outputs: Any | None = None
    weighted_modules_total: int = 0
    weighted_modules_called: tuple[str, ...] = ()
    uncalled_weighted_modules: tuple[str, ...] = ()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_heal_model_family(
    *,
    config_path: str | Path,
    checkpoint_path: str | Path,
    heal_root: str | Path,
    device: str = "cpu",
    family_id: str = "auto",
    forward_smoke: bool = False,
) -> HealModelFamilyBundle:
    """Load one HEAL model with strict state compatibility and audit it."""

    config_file = Path(config_path).expanduser().resolve()
    checkpoint_file = Path(checkpoint_path).expanduser().resolve()
    root = Path(heal_root).expanduser().resolve()
    for required, label in (
        (config_file, "model_config"),
        (checkpoint_file, "checkpoint"),
    ):
        if not required.is_file():
            raise RuntimeError(f"missing_{label}:{required}")
    if not root.is_dir():
        raise RuntimeError(f"missing_heal_root:{root}")
    for path in (str(root.parent), str(root), str(Path(__file__).resolve().parents[3])):
        if path not in sys.path:
            sys.path.insert(0, path)

    from opencood.hypes_yaml import yaml_utils
    try:
        from heal_compress.adapters.heal_lidar_adapter import HEALLiDARAdapter
    except ImportError:
        from adapters.heal_lidar_adapter import HEALLiDARAdapter

    config = yaml_utils.load_yaml(str(config_file))
    provider = detect_model_family(config) if family_id == "auto" else get_model_family(family_id)
    if not provider.matches(config):
        raise RuntimeError(f"model_family_config_mismatch:{provider.family_id}")
    adapter = HEALLiDARAdapter(
        heal_repo=str(root),
        config={"model": {"hypes_yaml": str(config_file)}},
    )
    model = adapter.build_model(str(config_file), checkpoint=None)
    raw = torch.load(checkpoint_file, map_location="cpu")
    state = raw.get("model", raw)
    if not isinstance(state, dict):
        raise RuntimeError("checkpoint_state_dict_missing")
    model.load_state_dict(state, strict=True)
    torch_device = torch.device(device)
    if torch_device.type == "cuda":
        torch.cuda.set_device(torch_device)
    model = model.to(torch_device).eval()
    example = None
    outputs = None
    weighted_types = (
        torch.nn.Conv1d,
        torch.nn.Conv2d,
        torch.nn.Conv3d,
        torch.nn.ConvTranspose1d,
        torch.nn.ConvTranspose2d,
        torch.nn.ConvTranspose3d,
        torch.nn.Linear,
    )
    weighted_paths = tuple(
        name for name, module in model.named_modules() if name and isinstance(module, weighted_types)
    )
    called: set[str] = set()
    if forward_smoke:
        example = adapter.build_synthetic_batch(model)
        hooks = [
            module.register_forward_hook(
                lambda _module, _inputs, _outputs, module_path=name: called.add(module_path)
            )
            for name, module in model.named_modules()
            if name and isinstance(module, weighted_types)
        ]
        try:
            with torch.no_grad():
                outputs = adapter.forward_for_task(model, example)
        finally:
            for hook in hooks:
                hook.remove()
    return HealModelFamilyBundle(
        model=model,
        adapter=adapter,
        provider=provider,
        audit=provider.audit(model, config),
        config=dict(config),
        config_path=config_file,
        checkpoint_path=checkpoint_file,
        config_hash=sha256_file(config_file),
        checkpoint_hash=sha256_file(checkpoint_file),
        state_dict_tensor_count=len(state),
        example_batch=example,
        smoke_outputs=outputs,
        weighted_modules_total=len(weighted_paths),
        weighted_modules_called=tuple(sorted(called)),
        uncalled_weighted_modules=tuple(sorted(set(weighted_paths) - called)),
    )
