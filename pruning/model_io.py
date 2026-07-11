"""Formal HEAL model loading and structure snapshot helpers.

This module intentionally mirrors the runtime helpers that were previously
kept in test scripts. Production tools should import from here instead of
``tests``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from collections.abc import Callable, Mapping
from typing import Any

import torch
import torch.nn as nn

from .exceptions import ModelLoadError, PhysicalStructureMismatchError
from .grouped_conv import grouped_conv_alignment_merge_factor, grouped_conv_pruning_fn, merge_grouped_conv_groups
from .types import ModelLoadResult, ModelProvenance


# Compatibility names now come only from caller-controlled environment. New
# code should pass paths through ProjectPaths/load_model explicitly.
DEFAULT_CHECKPOINT = os.environ.get("HEAL_CHECKPOINT")
DEFAULT_CONFIG = os.environ.get("HEAL_MODEL_CONFIG")
DEFAULT_HEAL_ROOT = os.environ.get("HEAL_REPOSITORY")

STRUCTURE_ATTRS = (
    "in_channels",
    "out_channels",
    "in_features",
    "out_features",
    "num_features",
    "groups",
    "normalized_shape",
)
HEAD_PROTECTED_KEYWORDS = ("cls_head", "reg_head", "dir_head")
FIXED_SHAPE_PROTECTED_KEYWORDS = ("pillar_vfe", "pfn_layers", "scatter", "voxel")


def setup_logger(output_dir: Path, name: str = "heal_structured_pruner") -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    fh = logging.FileHandler(output_dir / "prune_log.txt", mode="w", encoding="utf-8")
    sh.setFormatter(fmt)
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    logger.propagate = False
    return logger


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_config_payload(config: Mapping[str, Any] | str | Path | None) -> tuple[dict[str, Any], str]:
    if config is None:
        return {}, ""
    if isinstance(config, Mapping):
        return dict(config), ""
    path = Path(config)
    if not path.is_file():
        raise ModelLoadError(f"model config not found: {path}")
    text = path.read_text(encoding="utf-8")
    try:
        import yaml

        payload = yaml.safe_load(text)
    except ImportError:
        payload = json.loads(text)
    if payload is None:
        payload = {}
    if not isinstance(payload, Mapping):
        raise ModelLoadError("model config root must be a mapping")
    return dict(payload), str(path)


def _extract_state_dict(payload: Any, state_dict_key: str | None) -> Mapping[str, torch.Tensor]:
    if state_dict_key:
        if not isinstance(payload, Mapping) or state_dict_key not in payload:
            raise ModelLoadError(f"checkpoint lacks requested state_dict key: {state_dict_key}")
        payload = payload[state_dict_key]
    elif isinstance(payload, Mapping):
        for key in ("state_dict", "model_state_dict"):
            candidate = payload.get(key)
            if isinstance(candidate, Mapping):
                payload = candidate
                break
    if not isinstance(payload, Mapping) or not all(torch.is_tensor(value) for value in payload.values()):
        raise ModelLoadError("checkpoint does not contain a tensor-only state_dict")
    return payload


def load_model(
    model_factory: Callable[[Mapping[str, Any]], nn.Module],
    *,
    checkpoint_path: str | Path,
    model_config: Mapping[str, Any] | str | Path | None = None,
    device: str | torch.device = "cpu",
    strict_state_dict: bool = True,
    training: bool = False,
    state_dict_key: str | None = None,
) -> ModelLoadResult:
    """Construct and load a model without hidden paths or global state.

    ``model_factory`` receives a plain serializable config mapping. The
    checkpoint is loaded in tensor-only mode and all overlapping shapes are
    checked before ``load_state_dict`` mutates the new model.
    """

    checkpoint = Path(checkpoint_path)
    if not checkpoint.is_file():
        raise ModelLoadError(f"checkpoint not found: {checkpoint}")
    config_payload, config_source = _load_config_payload(model_config)
    model = model_factory(config_payload)
    if not isinstance(model, nn.Module):
        raise ModelLoadError("model_factory did not return torch.nn.Module")
    try:
        checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise ModelLoadError("safe tensor-only checkpoint loading requires a newer PyTorch") from exc
    state_dict = _extract_state_dict(checkpoint_payload, state_dict_key)
    live_state = model.state_dict()
    mismatches = {
        key: (tuple(live_state[key].shape), tuple(value.shape))
        for key, value in state_dict.items()
        if key in live_state and tuple(live_state[key].shape) != tuple(value.shape)
    }
    if mismatches:
        raise PhysicalStructureMismatchError(f"checkpoint/model shape mismatches: {mismatches}")
    incompatible = model.load_state_dict(state_dict, strict=strict_state_dict)
    target_device = torch.device(device)
    model.to(target_device)
    model.train(bool(training))
    provenance = ModelProvenance(
        config_path=config_source,
        checkpoint_path=str(checkpoint),
        checkpoint_sha256=_sha256_file(checkpoint),
        device=str(target_device),
        training=bool(training),
        strict_state_dict=bool(strict_state_dict),
    )
    return ModelLoadResult(
        model=model,
        provenance=provenance,
        missing_keys=list(incompatible.missing_keys),
        unexpected_keys=list(incompatible.unexpected_keys),
    )


def load_heal_model(args: argparse.Namespace, device: torch.device, logger: logging.Logger) -> tuple[nn.Module, Any]:
    """Deprecated argparse wrapper retained for historical command entries."""

    try:
        from heal_compress.adapters.heal_lidar_adapter import HEALLiDARAdapter
    except ImportError:
        from adapters.heal_lidar_adapter import HEALLiDARAdapter

    checkpoint = Path(args.checkpoint)
    config = Path(args.model_config)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    if not config.is_file():
        raise FileNotFoundError(f"model config not found: {config}")
    adapter = HEALLiDARAdapter(heal_repo=args.heal_root, config={"model": {"hypes_yaml": str(config)}})
    logger.info("Loading HEAL model: config=%s checkpoint=%s", config, checkpoint)
    model = adapter.build_model(str(config), str(checkpoint)).to(device).eval()
    return model, adapter


def collect_module_structure(model: nn.Module) -> dict[str, dict[str, Any]]:
    snapshot: dict[str, dict[str, Any]] = {}
    for name, module in model.named_modules():
        if not name:
            continue
        attrs = {}
        for attr in STRUCTURE_ATTRS:
            if hasattr(module, attr):
                value = getattr(module, attr)
                if isinstance(value, torch.Size):
                    value = tuple(value)
                attrs[attr] = value
        params = {pname: tuple(param.shape) for pname, param in module.named_parameters(recurse=False)}
        buffers = {bname: tuple(buf.shape) for bname, buf in module.named_buffers(recurse=False)}
        snapshot[name] = {
            "module_type": module.__class__.__name__,
            "attrs": attrs,
            "params": params,
            "buffers": buffers,
        }
    return snapshot


def count_params(model: nn.Module) -> tuple[int, float]:
    params = sum(p.numel() for p in model.parameters())
    mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    return int(params), float(mb)


def build_protected_layers(
    model: nn.Module,
    *,
    adapter_protected: list[str],
    extra_prefixes: list[str] | tuple[str, ...],
) -> list[str]:
    protected = {
        name
        for name in adapter_protected
        if any(k in name.lower() for k in HEAD_PROTECTED_KEYWORDS + FIXED_SHAPE_PROTECTED_KEYWORDS)
    }
    prefixes = tuple(p for p in extra_prefixes if p)
    if prefixes:
        for name, _module in model.named_modules():
            if any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes):
                protected.add(name)
    return sorted(protected)


def apply_only_regular_grouped_conv_filter(groups: list[Any], enabled: bool) -> None:
    if not enabled:
        return
    for group in groups:
        has_regular_grouped = False
        for item in getattr(group, "items", []):
            module = getattr(item, "module", None)
            if isinstance(module, nn.Conv2d) and module.groups > 1 and not (module.groups == module.in_channels == module.out_channels):
                has_regular_grouped = True
                break
        if not has_regular_grouped:
            group.protected = True
            group.protected_reason = "only_prune_regular_grouped_conv_filter"


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


def build_importance_calibration_data(adapter: HEALLiDARAdapter, args: argparse.Namespace, logger: logging.Logger) -> list[Any] | None:
    if args.importance_mode not in {"first_order_taylor", "second_order_fisher"}:
        return None
    if int(args.num_calib_batches or 0) <= 0:
        raise RuntimeError(f"{args.importance_mode}_gradient_missing: --num-calib-batches must be > 0")
    loader = adapter.get_calib_loader(
        {
            "hypes_yaml": str(args.model_config),
            "split": "train",
            "batch_size": 1,
            "num_workers": 0,
        }
    )
    batches = []
    for batch in loader:
        batches.append(batch)
        if len(batches) >= int(args.num_calib_batches):
            break
    if not batches:
        raise RuntimeError(f"{args.importance_mode}_gradient_missing: no calibration batches were produced")
    logger.info("Loaded %d train calibration batches for %s importance", len(batches), args.importance_mode)
    return batches


def configure_grouped_conv_pruning_fns(groups: list[Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    mode = args.group_conv_selection_mode
    if mode == "true_group_block_pruning":
        mode = "remove_groups"
    if mode == "remove_groups" and not getattr(args, "allow_remove_groups", False):
        mode = "keep_groups"
    if mode not in {
        "independent_group_topk",
        "group_balanced_output_groups_fixed",
        "remove_groups",
        "flat_output_groups_fixed",
        "group_coarsening_zero_padded_reblock",
    }:
        mode = "keep_groups"
    operations: list[dict[str, Any]] = []
    for group in groups:
        for item in getattr(group, "items", []):
            module = getattr(item, "module", None)
            if not isinstance(module, nn.Conv2d) or module.groups <= 1:
                continue
            old_name = getattr(item.pruning_fn, "__name__", "")
            item.pruning_fn = grouped_conv_pruning_fn(mode)
            item.reason = f"grouped_conv:{mode}"
            operations.append(
                {
                    "scope_id": group.group_id,
                    "module_name": item.name,
                    "old_pruning_fn": old_name,
                    "new_pruning_fn": getattr(item.pruning_fn, "__name__", ""),
                    "group_conv_selection_mode": args.group_conv_selection_mode,
                    "allow_remove_groups": getattr(args, "allow_remove_groups", False),
                }
            )
    return operations


def group_conv_reports(model: nn.Module, align: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    issues = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and module.groups > 1:
            in_per = module.in_channels // module.groups if module.in_channels % module.groups == 0 else 0
            out_per = module.out_channels // module.groups if module.out_channels % module.groups == 0 else 0
            ok = bool(in_per and out_per and in_per % align == 0 and out_per % align == 0)
            if not ok:
                issues.append(
                    {
                        "layer": name,
                        "issue": "violates_group_inner_channel_align",
                        "in_per_group": in_per,
                        "out_per_group": out_per,
                        "align": align,
                    }
                )
            rows.append(
                {
                    "layer": name,
                    "in_channels": module.in_channels,
                    "out_channels": module.out_channels,
                    "groups": module.groups,
                    "in_channels_per_group": in_per,
                    "out_channels_per_group": out_per,
                    "aligned": ok,
                }
            )
    return rows, {"all_group_convs_aligned": not issues, "issues": issues, "align": align}


def normalize_grouped_convs_for_alignment(model: nn.Module, align: int) -> list[dict[str, Any]]:
    operations: list[dict[str, Any]] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Conv2d) or module.groups <= 1:
            continue
        factor = grouped_conv_alignment_merge_factor(module, align)
        if factor is None:
            continue
        op = merge_grouped_conv_groups(module, factor)
        op["layer"] = name
        op["reason"] = "pre_prune_group_alignment_normalization"
        operations.append(op)
    return operations


def write_model_structure(model: nn.Module, path: Path, changes: dict[str, str] | None = None) -> None:
    changes = changes or {}
    lines = [str(model), "", "Named modules:"]
    for name, module in model.named_modules():
        if not name:
            continue
        attrs = []
        for attr in STRUCTURE_ATTRS:
            if hasattr(module, attr):
                attrs.append(f"{attr}={getattr(module, attr)}")
        line = f"{name}: {module.__class__.__name__} {' '.join(attrs)}"
        if name in changes:
            line = f"* {line}  # changed: {changes[name]}"
        lines.append(line)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def json_dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, default=str)
