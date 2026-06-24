from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .base_adapter import BaseAdapter


class HEALLiDARAdapter(BaseAdapter):
    """Adapter for HEAL/OpenCOOD DAIR-V2X LiDAROnly models.

    The adapter imports HEAL lazily and never modifies the HEAL repository.
    """

    PROTECTED_KEYWORDS = (
        "voxel",
        "scatter",
        "pillar_vfe",
        "post",
        "nms",
        "anchor",
        "cls_head",
        "reg_head",
        "dir_head",
    )
    PRUNABLE_KEYWORDS = (
        "backbone",
        "shrink",
        "compression",
        "fusion",
        "fuse",
        "encoder",
    )

    def __init__(self, heal_repo: str | None = None, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.heal_repo = heal_repo or self.config.get("heal_repo")
        self._loss_hypes: dict[str, Any] | None = None
        self._criterion = None
        if self.heal_repo:
            repo = str(Path(self.heal_repo).expanduser().resolve())
            if repo not in sys.path:
                sys.path.insert(0, repo)

    def _resolve_heal_path(self, path: str | None) -> str | None:
        if not path:
            return None
        p = Path(path)
        if p.is_absolute() or not self.heal_repo:
            return str(p)
        return str(Path(self.heal_repo) / p)

    def _absolutize_dataset_paths(self, cfg: dict[str, Any]) -> dict[str, Any]:
        """Make HEAL dataset paths independent from the caller's cwd."""
        if not self.heal_repo:
            return cfg
        resolved = dict(cfg)
        for key in ("data_dir", "root_dir", "validate_dir", "test_dir"):
            value = resolved.get(key)
            if isinstance(value, str) and value and not Path(value).is_absolute():
                resolved[key] = str(Path(self.heal_repo) / value)
        return resolved

    def build_model(self, model_config: str, checkpoint: str | None = None) -> nn.Module:
        from opencood.hypes_yaml import yaml_utils
        from opencood.tools import train_utils

        cfg_path = self._resolve_heal_path(model_config)
        hypes = yaml_utils.load_yaml(cfg_path)
        model = train_utils.create_model(hypes)
        ckpt = self._resolve_heal_path(checkpoint)
        if ckpt and Path(ckpt).exists():
            if Path(ckpt).is_dir():
                _, model = train_utils.load_saved_model(ckpt, model)
            else:
                state = torch.load(ckpt, map_location="cpu")
                model.load_state_dict(state.get("model", state), strict=False)
        return model

    def build_dataset(self, data_config: str | dict[str, Any], split: str):
        from opencood.data_utils.datasets import build_dataset
        from opencood.hypes_yaml import yaml_utils

        if isinstance(data_config, dict) and "model" not in data_config and "train_params" not in data_config:
            hypes_yaml = data_config.get("hypes_yaml") or self.config.get("model", {}).get("hypes_yaml")
            cfg = yaml_utils.load_yaml(self._resolve_heal_path(hypes_yaml))
        else:
            cfg = data_config if isinstance(data_config, dict) else yaml_utils.load_yaml(self._resolve_heal_path(data_config))
        cfg = self._absolutize_dataset_paths(cfg)
        return build_dataset(cfg, visualize=False, train=(split == "train"))

    def get_calib_loader(self, calib_config: str | dict[str, Any]):
        from torch.utils.data import DataLoader

        cfg = calib_config if isinstance(calib_config, dict) else {}
        split = cfg.get("split", "val")
        dataset = self.build_dataset(cfg, split)
        return DataLoader(
            dataset,
            batch_size=int(cfg.get("batch_size", 1)),
            shuffle=(split == "train"),
            collate_fn=dataset.collate_batch_train,
            num_workers=int(cfg.get("num_workers", 0)),
        )

    def build_dummy_input(self, batch):
        if isinstance(batch, dict) and "ego" in batch:
            return batch["ego"]
        return batch

    def build_synthetic_batch(self, model: nn.Module):
        """Build a tiny HEAL LiDAR batch for structure tracing when data is absent.

        This is not a calibration sample. It only exercises the LiDAR encoder,
        BEV backbone, fusion path, and heads so the runtime dependency tracer can
        record the model topology without requiring DAIR-V2X files.
        """
        modality = getattr(model, "ego_modality", "m1")
        agent_num = 2
        voxels_per_agent = 24
        max_points = 32
        device = next(model.parameters(), torch.empty(0)).device
        encoder = getattr(model, f"encoder_{modality}", None)
        scatter = getattr(encoder, "scatter", None)
        nx = int(getattr(scatter, "nx", 16))
        ny = int(getattr(scatter, "ny", 16))

        coords = []
        for agent_idx in range(agent_num):
            for voxel_idx in range(voxels_per_agent):
                x = (voxel_idx * 7 + agent_idx) % max(nx, 1)
                y = (voxel_idx * 5 + agent_idx) % max(ny, 1)
                coords.append([agent_idx, 0, y, x])
        voxel_coords = torch.tensor(coords, dtype=torch.int32, device=device)
        voxel_features = torch.zeros(
            (agent_num * voxels_per_agent, max_points, 4),
            dtype=torch.float32,
            device=device,
        )
        voxel_num_points = torch.full(
            (agent_num * voxels_per_agent,),
            max_points,
            dtype=torch.int32,
            device=device,
        )
        for idx in range(voxel_features.shape[0]):
            voxel_features[idx, :, 0] = torch.linspace(-1.0, 1.0, max_points, device=device)
            voxel_features[idx, :, 1] = torch.linspace(1.0, -1.0, max_points, device=device)
            voxel_features[idx, :, 2] = 0.1 * (idx % 5)
            voxel_features[idx, :, 3] = 1.0

        pairwise_t_matrix = torch.eye(4, dtype=torch.float32, device=device).view(1, 1, 1, 4, 4)
        pairwise_t_matrix = pairwise_t_matrix.repeat(1, agent_num, agent_num, 1, 1)

        return {
            "agent_modality_list": [modality for _ in range(agent_num)],
            "record_len": torch.tensor([agent_num], dtype=torch.long, device=device),
            "pairwise_t_matrix": pairwise_t_matrix,
            f"inputs_{modality}": {
                "voxel_features": voxel_features,
                "voxel_coords": voxel_coords,
                "voxel_num_points": voxel_num_points,
            },
        }

    def forward_for_task(self, model: nn.Module, batch):
        if isinstance(batch, dict) and "ego" in batch:
            return model(batch["ego"])
        return model(batch)

    def extract_bev_feature(self, model: nn.Module, batch) -> torch.Tensor:
        outputs = self.forward_for_task(model, batch)
        if isinstance(outputs, dict):
            for key in ("fusion_bev", "bev_feature", "bev_feat", "spatial_features_2d", "feature"):
                if key in outputs and torch.is_tensor(outputs[key]):
                    return outputs[key]
        raise RuntimeError("无法从 HEAL 模型输出中自动识别 BEV 融合特征，请在 adapter 中注册特征 key。")

    def compute_task_loss(self, outputs, batch) -> torch.Tensor:
        criterion = self._get_criterion()
        ego_batch = batch.get("ego", batch) if isinstance(batch, dict) else batch
        if not isinstance(ego_batch, dict) or "label_dict" not in ego_batch:
            raise KeyError("HEAL task loss 需要 batch['ego']['label_dict'] 或 batch['label_dict']")
        loss = criterion(outputs, ego_batch["label_dict"])

        hypes = self._get_loss_hypes()
        train_params = hypes.get("train_params", {}) if isinstance(hypes, dict) else {}
        supervise_single = bool(train_params.get("supervise_single", False))
        single_weight = float(train_params.get("single_weight", 1.0))
        if supervise_single and "label_dict_single" in ego_batch:
            try:
                loss = loss + criterion(outputs, ego_batch["label_dict_single"], suffix="_single") * single_weight
            except TypeError:
                loss = loss + criterion(outputs, ego_batch["label_dict_single"]) * single_weight
        return loss

    def _get_loss_hypes(self) -> dict[str, Any]:
        if self._loss_hypes is not None:
            return self._loss_hypes
        from opencood.hypes_yaml import yaml_utils

        model_cfg = self.config.get("model", {}) if isinstance(self.config, dict) else {}
        hypes_yaml = model_cfg.get("hypes_yaml") or self.config.get("hypes_yaml")
        if not hypes_yaml:
            raise RuntimeError("无法构建 HEAL loss：adapter config 缺少 model.hypes_yaml")
        hypes = yaml_utils.load_yaml(self._resolve_heal_path(hypes_yaml))
        self._loss_hypes = hypes
        return hypes

    def _get_criterion(self):
        if self._criterion is not None:
            return self._criterion
        from opencood.tools import train_utils

        self._criterion = train_utils.create_loss(self._get_loss_hypes())
        return self._criterion

    def _module_names(self, model: nn.Module, types: tuple[type[nn.Module], ...]) -> list[str]:
        return [name for name, module in model.named_modules() if isinstance(module, types)]

    def get_quantizable_layers(self, model: nn.Module) -> list[str]:
        protected = set(self.get_protected_layers(model))
        return [name for name in self._module_names(model, (nn.Conv2d, nn.Linear)) if name not in protected]

    def get_prunable_layers(self, model: nn.Module) -> list[str]:
        protected = set(self.get_protected_layers(model))
        layers = []
        for name, module in model.named_modules():
            if not isinstance(module, (nn.Conv2d, nn.BatchNorm2d, nn.Linear)):
                continue
            if name in protected:
                continue
            if any(k in name.lower() for k in self.PRUNABLE_KEYWORDS):
                layers.append(name)
        return layers

    def get_protected_layers(self, model: nn.Module) -> list[str]:
        protected = []
        for name, _ in model.named_modules():
            low = name.lower()
            if any(k in low for k in self.PROTECTED_KEYWORDS):
                protected.append(name)
        return protected

    def export_forward(self, model: nn.Module, batch):
        return self.forward_for_task(model, batch)
