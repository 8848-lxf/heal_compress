"""Real HEAL LiDAR model-family and trace provider."""

from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from .lidar_family import HEALLidarFamilySpec
from .lidar_family_registry import get_lidar_family_spec


DEFAULT_HEAL_ROOT = Path("/home/lixingfeng/UniAD_examine/HEAL")
DEFAULT_CONFIG = Path(
    "/home/lixingfeng/UniAD_examine/Auto_Search/original_models/"
    "dairv2s/LiDAROnly/lidar_pyramid/config.yaml"
)


@dataclass
class HEALLidarModelBundle:
    model: torch.nn.Module
    adapter: Any
    family_spec: HEALLidarFamilySpec
    model_config_path: Path
    checkpoint_path: Path
    checkpoint_hash: str
    checkpoint_load_audit: dict[str, Any]
    compatibility_audit: dict[str, Any]
    trace_example_inputs: Any
    trace_result: Any


LidarPyramidModelBundle = HEALLidarModelBundle


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prepare_import_paths(root: Path) -> None:
    repository_root = Path(__file__).resolve().parents[3]
    for path in (root.parent, root, repository_root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def _checkpoint_state(checkpoint: Path) -> Mapping[str, torch.Tensor]:
    payload = torch.load(checkpoint, map_location="cpu")
    state = payload.get("model", payload) if isinstance(payload, Mapping) else payload
    if not isinstance(state, Mapping) or not all(
        isinstance(key, str) for key in state
    ):
        raise RuntimeError(f"invalid_checkpoint_state_dict:{checkpoint}")
    return state


def _checkpoint_load_audit(
    model: torch.nn.Module,
    source_state: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    model_state = model.state_dict()
    model_keys = set(model_state)
    source_keys = set(source_state)
    missing = sorted(model_keys - source_keys)
    unexpected = sorted(source_keys - model_keys)
    parameter_keys = set(dict(model.named_parameters()))
    missing_parameters = sorted(parameter_keys.intersection(missing))
    unexpected_weighted = sorted(
        key
        for key in unexpected
        if key.endswith(".weight") or key.endswith(".bias")
    )
    shape_mismatches = []
    for key in sorted(model_keys.intersection(source_keys)):
        source = source_state[key]
        target = model_state[key]
        if not torch.is_tensor(source) or tuple(source.shape) != tuple(target.shape):
            shape_mismatches.append(
                {
                    "key": key,
                    "source_shape": tuple(getattr(source, "shape", ())),
                    "model_shape": tuple(target.shape),
                }
            )
    strict_weighted_pass = not (
        missing_parameters or unexpected_weighted or shape_mismatches
    )
    return {
        "source_state_entry_count": len(source_keys),
        "model_state_entry_count": len(model_keys),
        "missing_state_keys": missing,
        "unexpected_state_keys": unexpected,
        "missing_parameter_keys": missing_parameters,
        "unexpected_weighted_keys": unexpected_weighted,
        "shape_mismatches": shape_mismatches,
        "strict_weighted_pass": strict_weighted_pass,
        "strict_state_pass": not (missing or unexpected or shape_mismatches),
    }


def _install_family_compatibility(
    family: HEALLidarFamilySpec,
    *,
    heal_root: Path,
) -> dict[str, Any]:
    if family.compatibility_module is None:
        return {"status": "not_required", "family": family.name}
    if family.name == "lidar_disco":
        from .disconet_compat import install_disconet_compat_module

        return {
            "status": "installed_or_native",
            "family": family.name,
            **install_disconet_compat_module(heal_root),
        }
    raise RuntimeError(
        f"unsupported_family_compatibility_module:"
        f"{family.name}:{family.compatibility_module}"
    )


def load_heal_lidar_model(
    *,
    family: str | HEALLidarFamilySpec,
    checkpoint_path: str | Path | None = None,
    model_config_path: str | Path | None = None,
    heal_root: str | Path = DEFAULT_HEAL_ROOT,
    device: str = "cuda:0",
    trace: bool = True,
    strict_checkpoint: bool = True,
) -> HEALLidarModelBundle:
    spec = (
        get_lidar_family_spec(family) if isinstance(family, str) else family
    )
    root = Path(heal_root).expanduser().resolve()
    _prepare_import_paths(root)
    try:
        from heal_compress.adapters.heal_lidar_adapter import HEALLiDARAdapter
        from heal_compress.tracer.api import trace_model
        from heal_compress.tracer.config import TraceConfig
    except ImportError:
        from adapters.heal_lidar_adapter import HEALLiDARAdapter
        from tracer.api import trace_model
        from tracer.config import TraceConfig

    checkpoint_value = checkpoint_path or spec.default_checkpoint
    if checkpoint_value is None:
        raise RuntimeError(f"checkpoint_not_configured_for_family:{spec.name}")
    ckpt = Path(checkpoint_value).expanduser().resolve()
    config = Path(model_config_path or spec.default_config).expanduser().resolve()
    if not ckpt.is_file():
        raise RuntimeError(f"checkpoint_missing:{ckpt}")
    if not config.is_file():
        raise RuntimeError(f"missing_model_config:{config}")
    compatibility_audit = _install_family_compatibility(spec, heal_root=root)
    source_state = _checkpoint_state(ckpt)
    torch_device = torch.device(device)
    if torch_device.type == "cuda":
        torch.cuda.set_device(torch_device)
    adapter = HEALLiDARAdapter(
        heal_repo=str(root), config={"model": {"hypes_yaml": str(config)}}
    )
    model = adapter.build_model(str(config), str(ckpt)).to(torch_device).eval()
    load_audit = _checkpoint_load_audit(model, source_state)
    if strict_checkpoint and not load_audit["strict_weighted_pass"]:
        raise RuntimeError(f"checkpoint_weight_key_mismatch:{load_audit}")
    example = adapter.build_synthetic_batch(model)
    trace_result = None
    if trace:
        trace_result = trace_model(
            model,
            example,
            config=TraceConfig(fail_on_fx_trace_error=False),
            forward_fn=adapter.forward_for_task,
        )
    return HEALLidarModelBundle(
        model=model,
        adapter=adapter,
        family_spec=spec,
        model_config_path=config,
        checkpoint_path=ckpt,
        checkpoint_hash=sha256_file(ckpt),
        checkpoint_load_audit=load_audit,
        compatibility_audit=compatibility_audit,
        trace_example_inputs=example,
        trace_result=trace_result,
    )


def load_lidar_pyramid_model(
    *,
    checkpoint_path: str | Path,
    model_config_path: str | Path | None = None,
    heal_root: str | Path = DEFAULT_HEAL_ROOT,
    device: str = "cuda:0",
    trace: bool = True,
) -> LidarPyramidModelBundle:
    """Compatibility wrapper for the existing Pyramid context builder."""

    return load_heal_lidar_model(
        family="lidar_pyramid",
        checkpoint_path=checkpoint_path,
        model_config_path=model_config_path or DEFAULT_CONFIG,
        heal_root=heal_root,
        device=device,
        trace=trace,
        strict_checkpoint=True,
    )


__all__ = [
    "DEFAULT_CONFIG",
    "DEFAULT_HEAL_ROOT",
    "HEALLidarModelBundle",
    "LidarPyramidModelBundle",
    "load_heal_lidar_model",
    "load_lidar_pyramid_model",
    "sha256_file",
]
