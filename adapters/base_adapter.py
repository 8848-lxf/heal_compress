from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch


class BaseAdapter(ABC):
    """Adapter boundary between this search framework and task-specific models."""

    @abstractmethod
    def build_model(self, model_config: str, checkpoint: str | None = None) -> torch.nn.Module:
        raise NotImplementedError

    @abstractmethod
    def build_dataset(self, data_config: str | dict[str, Any], split: str):
        raise NotImplementedError

    @abstractmethod
    def get_calib_loader(self, calib_config: str | dict[str, Any]):
        raise NotImplementedError

    @abstractmethod
    def build_dummy_input(self, batch):
        raise NotImplementedError

    @abstractmethod
    def forward_for_task(self, model: torch.nn.Module, batch):
        raise NotImplementedError

    @abstractmethod
    def extract_bev_feature(self, model: torch.nn.Module, batch) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def compute_task_loss(self, outputs, batch) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def get_quantizable_layers(self, model: torch.nn.Module) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def get_prunable_layers(self, model: torch.nn.Module) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def get_protected_layers(self, model: torch.nn.Module) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def export_forward(self, model: torch.nn.Module, batch):
        raise NotImplementedError
