"""Data and manifest providers for HEAL lidar_pyramid search."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import DataLoader


@dataclass
class EvaluationManifest:
    path: Path
    frame_ids: list[str]
    split: str
    manifest_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "frame_ids": list(self.frame_ids),
            "split": self.split,
            "manifest_hash": self.manifest_hash,
        }


def _hash_manifest(frame_ids: list[str], split: str) -> str:
    import hashlib

    payload = json.dumps({"split": split, "frame_ids": frame_ids}, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_dataset_and_loader(adapter: Any, model_config_path: str | Path, *, split: str, num_workers: int = 0, visualize: bool = True) -> tuple[Any, DataLoader]:
    from opencood.data_utils.datasets import build_dataset
    from opencood.hypes_yaml import yaml_utils

    hypes = yaml_utils.load_yaml(adapter._resolve_heal_path(str(model_config_path)))
    hypes = adapter._absolutize_dataset_paths(hypes)
    dataset = build_dataset(hypes, visualize=visualize, train=(split == "train"))
    collate = dataset.collate_batch_train if split == "train" else dataset.collate_batch_test
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate, num_workers=num_workers, pin_memory=False, drop_last=False)
    return dataset, loader


def write_eval_manifest(output_path: str | Path, *, num_frames: int, warmup_frames: int, split: str = "val") -> EvaluationManifest:
    total = int(num_frames) + int(warmup_frames)
    frame_ids = [str(index) for index in range(total)]
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest_hash = _hash_manifest(frame_ids, split)
    path.write_text(json.dumps({"split": split, "frame_ids": frame_ids, "manifest_hash": manifest_hash}, indent=2), encoding="utf-8")
    return EvaluationManifest(path, frame_ids, split, manifest_hash)


def iter_limited(loader: Iterable[Any], limit: int) -> list[Any]:
    rows = []
    for batch in loader:
        rows.append(batch)
        if len(rows) >= int(limit):
            break
    return rows


def move_batch_to_device(batch: Any, device: torch.device) -> Any:
    if torch.is_tensor(batch):
        return batch.to(device)
    if isinstance(batch, dict):
        return {key: move_batch_to_device(value, device) for key, value in batch.items()}
    if isinstance(batch, list):
        return [move_batch_to_device(value, device) for value in batch]
    if isinstance(batch, tuple):
        return tuple(move_batch_to_device(value, device) for value in batch)
    return batch
