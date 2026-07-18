"""First-order task-loss Taylor rankings for CoBEVT Attention dimensions."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
import torch.nn as nn


@dataclass(frozen=True)
class AttentionTaylorScores:
    qk_by_head: tuple[tuple[float, ...], ...]
    vo_by_head: tuple[tuple[float, ...], ...]

    def to_dict(self) -> dict[str, list[list[float]]]:
        return {
            "qk_by_head": [list(row) for row in self.qk_by_head],
            "vo_by_head": [list(row) for row in self.vo_by_head],
        }


@dataclass(frozen=True)
class AttentionMeanGradientStatistics:
    gradients: dict[str, torch.Tensor]
    manifest: dict[str, Any]
    manifest_hash: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _required_gradient(
    gradients: Mapping[str, torch.Tensor], name: str, parameter: torch.Tensor
) -> torch.Tensor:
    if name not in gradients:
        raise ValueError(f"attention_taylor_gradient_missing:{name}")
    gradient = gradients[name]
    if tuple(gradient.shape) != tuple(parameter.shape):
        raise ValueError(f"attention_taylor_gradient_shape_mismatch:{name}")
    if not bool(torch.isfinite(gradient).all()):
        raise ValueError(f"attention_taylor_gradient_nonfinite:{name}")
    return gradient


def _element_score(parameter: torch.Tensor, gradient: torch.Tensor) -> torch.Tensor:
    weight = parameter.detach().to(device="cpu", dtype=torch.float64)
    mean_gradient = gradient.detach().to(device="cpu", dtype=torch.float64)
    return (weight * mean_gradient).abs()


def first_order_attention_scores(
    module: nn.Module, gradients: Mapping[str, torch.Tensor]
) -> AttentionTaylorScores:
    heads = int(module.heads)
    d_qk = int(module.d_qk)
    d_v = int(module.d_v)
    q = _element_score(
        module.q_proj.weight,
        _required_gradient(gradients, "q_proj.weight", module.q_proj.weight),
    ).sum(dim=1)
    k = _element_score(
        module.k_proj.weight,
        _required_gradient(gradients, "k_proj.weight", module.k_proj.weight),
    ).sum(dim=1)
    v = _element_score(
        module.v_proj.weight,
        _required_gradient(gradients, "v_proj.weight", module.v_proj.weight),
    ).sum(dim=1)
    out = _element_score(
        module.out_proj.weight,
        _required_gradient(gradients, "out_proj.weight", module.out_proj.weight),
    ).sum(dim=0)
    if module.q_proj.bias is not None:
        q = q + _element_score(
            module.q_proj.bias,
            _required_gradient(gradients, "q_proj.bias", module.q_proj.bias),
        )
        k = k + _element_score(
            module.k_proj.bias,
            _required_gradient(gradients, "k_proj.bias", module.k_proj.bias),
        )
        v = v + _element_score(
            module.v_proj.bias,
            _required_gradient(gradients, "v_proj.bias", module.v_proj.bias),
        )
    qk = (q + k).reshape(heads, d_qk)
    vo = (v + out).reshape(heads, d_v)
    if not bool(torch.isfinite(qk).all()) or not bool(torch.isfinite(vo).all()):
        raise ValueError("attention_taylor_score_nonfinite")
    return AttentionTaylorScores(
        qk_by_head=tuple(tuple(float(value) for value in row) for row in qk.tolist()),
        vo_by_head=tuple(tuple(float(value) for value in row) for row in vo.tolist()),
    )


def keep_indices_from_scores(
    scores_by_head: Iterable[Iterable[float]], *, keep_width: int
) -> tuple[tuple[int, ...], ...]:
    rows = tuple(tuple(float(value) for value in row) for row in scores_by_head)
    if not rows or keep_width <= 0:
        raise ValueError("attention_taylor_keep_width_invalid")
    result = []
    for row in rows:
        if keep_width > len(row):
            raise ValueError("attention_taylor_keep_width_exceeds_original")
        ranking = sorted(range(len(row)), key=lambda index: (row[index], index))
        pruned = set(ranking[: len(row) - int(keep_width)])
        result.append(tuple(index for index in range(len(row)) if index not in pruned))
    return tuple(result)


def attention_masks_from_mean_gradients(
    model: nn.Module,
    gradients: Mapping[str, torch.Tensor],
    *,
    d_qk: int,
    d_v: int,
    module_type: type[nn.Module] | None = None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    from .attention_dim_pruning import AttentionDimMask, PrunableCobevtAttention

    expected_type = PrunableCobevtAttention if module_type is None else module_type
    masks: dict[str, AttentionDimMask] = {}
    audit: dict[str, dict[str, Any]] = {}
    for name, module in model.named_modules():
        if not isinstance(module, expected_type):
            continue
        prefix = f"{name}." if name else ""
        local = {
            parameter_name[len(prefix) :]: value
            for parameter_name, value in gradients.items()
            if parameter_name.startswith(prefix)
        }
        scores = first_order_attention_scores(module, local)
        qk_keep = keep_indices_from_scores(scores.qk_by_head, keep_width=int(d_qk))
        vo_keep = keep_indices_from_scores(scores.vo_by_head, keep_width=int(d_v))
        masks[name] = AttentionDimMask(
            qk_keep,
            vo_keep,
            original_d_qk=int(module.d_qk),
            original_d_v=int(module.d_v),
        )
        qk_rankings = tuple(
            tuple(sorted(range(len(row)), key=lambda index: (row[index], index)))
            for row in scores.qk_by_head
        )
        vo_rankings = tuple(
            tuple(sorted(range(len(row)), key=lambda index: (row[index], index)))
            for row in scores.vo_by_head
        )
        audit[name] = {
            **scores.to_dict(),
            "d_qk": int(d_qk),
            "d_v": int(d_v),
            "qk_keep_by_head": [list(row) for row in qk_keep],
            "vo_keep_by_head": [list(row) for row in vo_keep],
            "qk_ranking_by_head": [list(row) for row in qk_rankings],
            "vo_ranking_by_head": [list(row) for row in vo_rankings],
        }
    if not masks:
        raise RuntimeError("attention_taylor_modules_missing")
    return masks, audit


def collect_attention_mean_gradients(
    *,
    model: nn.Module,
    adapter: Any,
    model_config_path: str | Path,
    device: torch.device,
    cache_path: str | Path,
    num_samples: int,
    checkpoint_hash: str,
    code_commit: str,
    seed: int = 20260718,
) -> AttentionMeanGradientStatistics:
    """Stream mean task-loss gradients for Attention parameters only."""

    from search.hashing import canonical_json_hash
    from search.integration.data_provider import (
        build_dataset_and_loader,
        load_split_frame_ids,
        move_batch_to_device,
    )
    from .attention_dim_pruning import PrunableCobevtAttention

    destination = Path(cache_path)
    if destination.is_file():
        payload = torch.load(destination, map_location="cpu")
        return AttentionMeanGradientStatistics(
            gradients=dict(payload["gradients"]),
            manifest=dict(payload["manifest"]),
            manifest_hash=str(payload["manifest_hash"]),
        )
    if int(num_samples) <= 0:
        raise ValueError("attention_gradient_sample_count_must_be_positive")
    prefixes = tuple(
        f"{name}."
        for name, module in model.named_modules()
        if isinstance(module, PrunableCobevtAttention)
    )
    if not prefixes:
        raise RuntimeError("explicit_cobevt_attention_modules_missing")
    selected = {
        name: parameter
        for name, parameter in model.named_parameters()
        if name.startswith(prefixes)
        and any(
            token in name
            for token in (
                ".q_proj.",
                ".k_proj.",
                ".v_proj.",
                ".out_proj.",
            )
        )
    }
    if not selected:
        raise RuntimeError("attention_gradient_parameters_missing")

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    try:
        _dataset, loader = build_dataset_and_loader(
            adapter,
            model_config_path,
            split="train",
            num_workers=0,
            visualize=False,
        )
        sums = {
            name: torch.zeros_like(parameter, dtype=torch.float64, device="cpu")
            for name, parameter in selected.items()
        }
        losses: list[float] = []
        count = 0
        model.train(False)
        for batch in loader:
            if count >= int(num_samples):
                break
            batch = move_batch_to_device(batch, device)
            model.zero_grad(set_to_none=True)
            output = adapter.forward_for_task(model, batch)
            loss = adapter.compute_task_loss(output, batch)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"attention_task_loss_nonfinite:sample={count}")
            loss.backward()
            for name, parameter in selected.items():
                if parameter.grad is None:
                    raise RuntimeError(
                        f"attention_mean_gradient_missing:sample={count}:{name}"
                    )
                gradient = parameter.grad.detach()
                if not bool(torch.isfinite(gradient).all()):
                    raise RuntimeError(
                        f"attention_mean_gradient_nonfinite:sample={count}:{name}"
                    )
                sums[name] += gradient.to(dtype=torch.float64, device="cpu")
            losses.append(float(loss.detach().cpu()))
            count += 1
        if count != int(num_samples):
            raise RuntimeError(
                f"attention_gradient_dataset_too_short:{count}<{int(num_samples)}"
            )
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)

    gradients = {
        name: (value / float(count)).to(dtype=torch.float32)
        for name, value in sums.items()
    }
    frame_ids = load_split_frame_ids(
        adapter, model_config_path, split="train"
    )[:count]
    config_path = Path(model_config_path).expanduser().resolve()
    frame_hash = canonical_json_hash(
        {"frame_ids": frame_ids, "seed": int(seed), "split": "train"}
    )
    parameter_elements = sum(int(value.numel()) for value in gradients.values())
    manifest = {
        "calibration_manifest_hash": frame_hash,
        "checkpoint_hash": str(checkpoint_hash),
        "code_commit": str(code_commit),
        "config_hash": _sha256(config_path),
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "dtype": "float32",
        "gradient_accumulation": "streaming_mean_g_attention_parameters_only",
        "loss_components": ["L_cls", "L_reg", "L_dir", "L_obj"],
        "loss_max": max(losses),
        "loss_mean": sum(losses) / len(losses),
        "loss_min": min(losses),
        "micro_batch_size": 1,
        "parameter_element_count": parameter_elements,
        "parameter_tensor_count": len(gradients),
        "sample_count": count,
        "seed": int(seed),
        "task_loss": "HEAL/OpenCOOD original criterion",
    }
    manifest_hash = canonical_json_hash(manifest)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "gradients": gradients,
            "manifest": manifest,
            "manifest_hash": manifest_hash,
        },
        destination,
    )
    destination.with_suffix(".manifest.json").write_text(
        json.dumps(
            {**manifest, "manifest_hash": manifest_hash},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return AttentionMeanGradientStatistics(gradients, manifest, manifest_hash)


__all__ = [
    "AttentionTaylorScores",
    "AttentionMeanGradientStatistics",
    "attention_masks_from_mean_gradients",
    "collect_attention_mean_gradients",
    "first_order_attention_scores",
    "keep_indices_from_scores",
]
