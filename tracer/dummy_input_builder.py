from __future__ import annotations

from pathlib import Path
from typing import Any

from .utils import add_repo_parent_to_sys_path


def build_dataset_dummy_input(adapter: Any, model: Any, config: str, *, split: str = "train") -> tuple[Any, dict[str, Any]]:
    """Build a tracing sample from the real dataset when available.

    Falls back to the adapter synthetic batch only when dataset construction is
    unavailable, and records that fallback in metadata.
    """
    add_repo_parent_to_sys_path()
    try:
        from opencood.data_utils.datasets import build_dataset
        from opencood.hypes_yaml import yaml_utils

        hypes = yaml_utils.load_yaml(adapter._resolve_heal_path(config))
        hypes = adapter._absolutize_dataset_paths(hypes)
        dataset = build_dataset(hypes, visualize=True, train=(split == "train"))
        collate = getattr(dataset, "collate_batch_train", None) if split == "train" else getattr(dataset, "collate_batch_test", None)
        collate = collate or getattr(dataset, "collate_batch_test", None) or getattr(dataset, "collate_batch_train")
        sample = collate([dataset[0]])
        return sample, {"dummy_input_source": "dataset", "split": split, "dataset_len": len(dataset), "sample_index": 0}
    except Exception as exc:
        sample = adapter.build_synthetic_batch(model)
        return sample, {"dummy_input_source": "adapter_synthetic_fallback", "split": split, "fallback_reason": str(exc)}
