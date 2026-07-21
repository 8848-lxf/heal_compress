"""Model and checkpoint capability for HEAL LiDAR CoBEVT."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn
import yaml

from adapters.heal_lidar_adapter import HEALLiDARAdapter

MODEL_FAMILY = "lidar_cobevt"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class CobevtModelPreflight:
    model_family: str
    checkpoint_path: str
    config_path: str
    heal_root: str
    checkpoint_sha256: str
    config_sha256: str
    model_core_method: str
    fusion_method: str


@dataclass(frozen=True)
class CheckpointLoadReport:
    checkpoint_sha256: str
    loaded_state_keys: int
    model_state_keys: int
    missing_keys: tuple[str, ...]
    missing_weight_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]

    @property
    def full_weight_coverage(self) -> bool:
        return not self.missing_weight_keys and not self.unexpected_keys


@dataclass(frozen=True)
class ScatterCapability:
    module_name: str
    module_type: str
    output_shape: tuple[int, int, int]
    allowed_boundary_dtypes: tuple[str, ...]
    quantization_gene: bool


@dataclass
class CobevtModelBundle:
    model: nn.Module
    adapter: HEALLiDARAdapter
    hypes: Mapping[str, Any]
    preflight: CobevtModelPreflight
    load_report: CheckpointLoadReport
    scatter: ScatterCapability


def validate_scatter_capability(model: nn.Module) -> ScatterCapability:
    encoder = getattr(model, "encoder_m1", None)
    scatter = getattr(encoder, "scatter", None)
    if scatter is None:
        raise RuntimeError("cobevt_pointpillar_scatter_missing")
    required = ("nx", "ny", "num_bev_features")
    missing = [name for name in required if not hasattr(scatter, name)]
    if missing:
        raise RuntimeError(f"cobevt_scatter_metadata_missing:{','.join(missing)}")
    return ScatterCapability(
        module_name="encoder_m1.scatter",
        module_type=scatter.__class__.__name__,
        output_shape=(
            int(scatter.num_bev_features),
            int(scatter.ny),
            int(scatter.nx),
        ),
        allowed_boundary_dtypes=("FP32", "FP16"),
        quantization_gene=False,
    )


class CobevtModelCapability:
    def __init__(
        self,
        checkpoint: str | Path,
        config: str | Path,
        heal_root: str | Path,
    ) -> None:
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        self.config = Path(config).expanduser().resolve()
        self.heal_root = Path(heal_root).expanduser().resolve()

    def _read_config(self) -> dict[str, Any]:
        if not self.config.is_file():
            raise FileNotFoundError(f"cobevt_config_missing:{self.config}")
        payload = yaml.safe_load(self.config.read_text(encoding="utf-8")) or {}
        if not isinstance(payload, dict):
            raise RuntimeError("cobevt_config_must_be_mapping")
        return payload

    def preflight(self) -> CobevtModelPreflight:
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"cobevt_checkpoint_missing:{self.checkpoint}")
        if not self.heal_root.is_dir():
            raise FileNotFoundError(f"heal_root_missing:{self.heal_root}")
        hypes = self._read_config()
        model = hypes.get("model", {})
        if not isinstance(model, Mapping):
            raise RuntimeError("cobevt_model_config_must_be_mapping")
        core_method = str(model.get("core_method", ""))
        if core_method != "heter_model_baseline":
            raise RuntimeError(f"model_core_not_supported:{core_method}")
        args = model.get("args", {})
        if not isinstance(args, Mapping):
            raise RuntimeError("cobevt_model_args_must_be_mapping")
        fusion_method = str(args.get("fusion_method", ""))
        if fusion_method != "cobevt":
            raise RuntimeError(f"model_fusion_not_cobevt:{fusion_method}")
        return CobevtModelPreflight(
            model_family=MODEL_FAMILY,
            checkpoint_path=str(self.checkpoint),
            config_path=str(self.config),
            heal_root=str(self.heal_root),
            checkpoint_sha256=_sha256(self.checkpoint),
            config_sha256=_sha256(self.config),
            model_core_method=core_method,
            fusion_method=fusion_method,
        )
    def load(self, *, device: str | torch.device = "cpu") -> CobevtModelBundle:
        preflight = self.preflight()
        adapter = HEALLiDARAdapter(
            heal_repo=str(self.heal_root),
            config={"model": {"hypes_yaml": str(self.config)}},
        )
        from opencood.hypes_yaml import yaml_utils
        from opencood.tools import train_utils

        hypes = yaml_utils.load_yaml(str(self.config))
        model = train_utils.create_model(hypes)
        state = torch.load(self.checkpoint, map_location="cpu")
        if isinstance(state, Mapping):
            state = state.get("model", state.get("state_dict", state))
        if not isinstance(state, Mapping):
            raise RuntimeError("cobevt_checkpoint_state_dict_missing")
        incompatible = model.load_state_dict(state, strict=False)
        parameter_names = {name for name, _ in model.named_parameters()}
        missing_keys = tuple(sorted(incompatible.missing_keys))
        report = CheckpointLoadReport(
            checkpoint_sha256=preflight.checkpoint_sha256,
            loaded_state_keys=len(state),
            model_state_keys=len(model.state_dict()),
            missing_keys=missing_keys,
            missing_weight_keys=tuple(
                name for name in missing_keys if name in parameter_names
            ),
            unexpected_keys=tuple(sorted(incompatible.unexpected_keys)),
        )
        model.to(device)
        model.eval()
        return CobevtModelBundle(
            model=model,
            adapter=adapter,
            hypes=hypes,
            preflight=preflight,
            load_report=report,
            scatter=validate_scatter_capability(model),
        )
