"""Structure-aware offline SmoothQuant selection for Transformer projections."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any, Iterable, Mapping

import torch
import torch.nn as nn


SMOOTHQUANT_ALPHA_GRID = (0.6, 0.7, 0.75, 0.8)


def _tensor_hash(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("utf-8"))
    digest.update(json.dumps(list(tensor.shape)).encode("utf-8"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def smoothquant_scale(
    activation_samples: torch.Tensor,
    weight: torch.Tensor,
    *,
    alpha: float,
    epsilon: float = 1.0e-8,
) -> torch.Tensor:
    """Compute per-input-channel ``max|X|^alpha/max|W|^(1-alpha)``."""

    value = float(alpha)
    if value not in SMOOTHQUANT_ALPHA_GRID:
        raise ValueError(f"smoothquant_alpha_not_in_frozen_grid:{value}")
    if activation_samples.ndim < 1 or weight.ndim != 2:
        raise ValueError("smoothquant_requires_activation_last_dim_and_linear_weight")
    if activation_samples.shape[-1] != weight.shape[1]:
        raise ValueError("smoothquant_input_channel_mismatch")
    reduce = tuple(range(activation_samples.ndim - 1))
    activation_max = activation_samples.detach().abs().amax(dim=reduce) if reduce else activation_samples.detach().abs()
    weight_max = weight.detach().abs().amax(dim=0)
    activation_max = activation_max.clamp_min(float(epsilon))
    weight_max = weight_max.clamp_min(float(epsilon))
    return activation_max.pow(value) / weight_max.pow(1.0 - value)


@dataclass(frozen=True)
class SmoothQuantRecord:
    model: str
    family: str
    unit_id: str
    module_paths: tuple[str, ...]
    alpha: float
    structure_hash: str
    calibration_manifest_hash: str
    activation_scale_hash: str
    weight_scale_hash: str
    fused_qkv_shared_input: bool
    separate_projection_extra_boundaries: bool
    selection_evidence_hash: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def make_smoothquant_record(
    *,
    model: str,
    family: str,
    unit_id: str,
    module_paths: Iterable[str],
    alpha: float,
    structure_hash: str,
    calibration_manifest_hash: str,
    scale: torch.Tensor,
    fused_qkv_shared_input: bool,
    selection_evidence: Mapping[str, Any],
) -> SmoothQuantRecord:
    if not structure_hash or not calibration_manifest_hash:
        raise ValueError("smoothquant_structure_or_calibration_provenance_missing")
    activation_scale = scale.detach().reciprocal()
    payload = dict(selection_evidence)
    return SmoothQuantRecord(
        model=str(model),
        family=str(family),
        unit_id=str(unit_id),
        module_paths=tuple(sorted({str(value) for value in module_paths})),
        alpha=float(alpha),
        structure_hash=str(structure_hash),
        calibration_manifest_hash=str(calibration_manifest_hash),
        activation_scale_hash=_tensor_hash(activation_scale),
        weight_scale_hash=_tensor_hash(scale),
        fused_qkv_shared_input=bool(fused_qkv_shared_input),
        separate_projection_extra_boundaries=not bool(fused_qkv_shared_input),
        selection_evidence_hash=hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
        ).hexdigest(),
    )


def smoothquant_transform(
    activation: torch.Tensor,
    linear: nn.Linear,
    scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Return mathematically equivalent unquantized activation/weight/bias."""

    value = scale.to(device=activation.device, dtype=activation.dtype)
    if value.ndim != 1 or value.numel() != linear.in_features:
        raise ValueError("smoothquant_scale_shape_mismatch")
    transformed_activation = activation / value
    transformed_weight = linear.weight * value.to(linear.weight.dtype).unsqueeze(0)
    transformed_bias = None if linear.bias is None else linear.bias
    return transformed_activation, transformed_weight, transformed_bias


def select_smoothquant_alpha(
    rows: Iterable[Mapping[str, Any]],
    *,
    anchor_weight: float = 1.0,
) -> dict[str, Any]:
    """Select offline from proxy loss plus a small real-anchor loss."""

    values = [dict(row) for row in rows]
    if not values:
        raise ValueError("smoothquant_alpha_evidence_empty")
    provenance = {
        (str(row.get("unit_id", "")), str(row.get("structure_hash", "")), str(row.get("calibration_manifest_hash", "")))
        for row in values
    }
    if len(provenance) != 1 or any(not all(key) for key in provenance):
        raise ValueError("smoothquant_alpha_evidence_provenance_conflict")
    for row in values:
        alpha = float(row.get("alpha", float("nan")))
        proxy = float(row.get("calibration_proxy", float("nan")))
        anchor = float(row.get("anchor_loss", float("nan")))
        if alpha not in SMOOTHQUANT_ALPHA_GRID or not all(math.isfinite(value) for value in (proxy, anchor)):
            raise ValueError("smoothquant_alpha_evidence_invalid")
        row["selection_score"] = proxy + float(anchor_weight) * anchor
    return min(values, key=lambda row: (float(row["selection_score"]), float(row["anchor_loss"]), float(row["alpha"])))


class SmoothQuantRegistry:
    """Frozen per-structure records; alpha is never a GA gene."""

    def __init__(self) -> None:
        self._records: dict[tuple[str, str, str, str, str], SmoothQuantRecord] = {}

    @staticmethod
    def _key(record: SmoothQuantRecord) -> tuple[str, str, str, str, str]:
        return (
            record.model,
            record.family,
            record.unit_id,
            record.structure_hash,
            record.calibration_manifest_hash,
        )

    def freeze(self, record: SmoothQuantRecord) -> None:
        key = self._key(record)
        previous = self._records.get(key)
        if previous is not None and previous != record:
            raise RuntimeError(f"smoothquant_frozen_record_conflict:{key}")
        self._records[key] = record

    def require(
        self,
        *,
        model: str,
        family: str,
        unit_id: str,
        structure_hash: str,
        calibration_manifest_hash: str,
    ) -> SmoothQuantRecord:
        key = (model, family, unit_id, structure_hash, calibration_manifest_hash)
        record = self._records.get(key)
        if record is None:
            related = [value for value in self._records if value[:3] == key[:3]]
            reason = "structure_or_calibration_mismatch" if related else "record_missing"
            raise RuntimeError(f"smoothquant_calibration_incompatible:{reason}:{key}")
        return record

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "smoothquant-frozen-registry-v1",
            "alpha_is_search_gene": False,
            "alpha_grid": list(SMOOTHQUANT_ALPHA_GRID),
            "records": [self._records[key].to_dict() for key in sorted(self._records)],
        }


__all__ = [
    "SMOOTHQUANT_ALPHA_GRID",
    "SmoothQuantRecord",
    "SmoothQuantRegistry",
    "make_smoothquant_record",
    "select_smoothquant_alpha",
    "smoothquant_scale",
    "smoothquant_transform",
]
