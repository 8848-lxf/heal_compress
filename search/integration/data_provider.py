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


def _hash_manifest(
    frame_ids: list[str],
    split: str,
    *,
    warmup_frame_ids: list[str] | None = None,
    evaluation_frame_ids: list[str] | None = None,
    reset_after_warmup: bool = False,
) -> str:
    import hashlib

    payload = json.dumps(
        {
            "split": split,
            "frame_ids": frame_ids,
            "warmup_frame_ids": list(warmup_frame_ids or []),
            "evaluation_frame_ids": list(evaluation_frame_ids or frame_ids),
            "reset_after_warmup": bool(reset_after_warmup),
        },
        sort_keys=True,
    ).encode("utf-8")
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


def load_split_frame_ids(adapter: Any, model_config_path: str | Path, *, split: str = "val") -> list[str]:
    from opencood.hypes_yaml import yaml_utils

    hypes = yaml_utils.load_yaml(adapter._resolve_heal_path(str(model_config_path)))
    hypes = adapter._absolutize_dataset_paths(hypes)
    key = "root_dir" if split == "train" else ("test_dir" if split == "test" else "validate_dir")
    split_path = Path(str(hypes.get(key, ""))).expanduser()
    if not split_path.is_file():
        raise RuntimeError(f"dataset_split_manifest_missing:{split}:{split_path}")
    payload = json.loads(split_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise RuntimeError(f"dataset_split_manifest_not_list:{split}:{split_path}")
    frame_ids = [str(value) for value in payload]
    if len(frame_ids) != len(set(frame_ids)):
        raise RuntimeError(f"dataset_split_manifest_duplicate_ids:{split}:{split_path}")
    return frame_ids


def write_eval_manifest(
    output_path: str | Path,
    *,
    num_frames: int,
    warmup_frames: int,
    split: str = "val",
    available_frame_ids: Iterable[str] | None = None,
    reset_after_warmup: bool = False,
    evaluation_offset: int = 0,
) -> EvaluationManifest:
    offset = int(evaluation_offset)
    if offset < 0:
        raise ValueError("evaluation_offset_must_be_nonnegative")
    if offset and not reset_after_warmup:
        raise ValueError("evaluation_offset_requires_reset_after_warmup")
    total = int(num_frames) + int(warmup_frames)
    required = max(offset + int(num_frames), int(warmup_frames)) if reset_after_warmup else total
    available = [str(value) for value in available_frame_ids] if available_frame_ids is not None else [str(index) for index in range(required)]
    if len(available) < required:
        raise RuntimeError(f"insufficient_manifest_frames:{len(available)}<{required}")
    if reset_after_warmup:
        warmup_ids = available[: int(warmup_frames)]
        evaluation_ids = available[offset : offset + int(num_frames)]
        frame_ids = list(evaluation_ids)
    else:
        frame_ids = available[:total]
        warmup_ids = frame_ids[: int(warmup_frames)]
        evaluation_ids = frame_ids[int(warmup_frames) :]
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest_hash = _hash_manifest(
        frame_ids,
        split,
        warmup_frame_ids=warmup_ids,
        evaluation_frame_ids=evaluation_ids,
        reset_after_warmup=reset_after_warmup,
    )
    path.write_text(
        json.dumps(
            {
                "split": split,
                "frame_ids": frame_ids,
                "warmup_frame_ids": warmup_ids,
                "evaluation_frame_ids": evaluation_ids,
                "warmup_frames": len(warmup_ids),
                "num_frames": len(evaluation_ids),
                "reset_after_warmup": bool(reset_after_warmup),
                "evaluation_offset": offset,
                "manifest_hash": manifest_hash,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
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
