"""Fixed task-loss Taylor rankings for legal pruning-width domains."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping, Sequence

import torch

from ..proxy.fisher_proxy import FisherStatistics
from ..proxy.parameter_slice_resolver import ParameterSlice


@dataclass(frozen=True)
class AtomicTaylorScore:
    unit_id: str
    first_order: float
    second_order: float
    joint_second_order: float
    affected_parameter_count: int
    affected_element_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "first_order": self.first_order,
            "second_order": self.second_order,
            "joint_second_order": self.joint_second_order,
            "affected_parameter_count": self.affected_parameter_count,
            "affected_element_count": self.affected_element_count,
        }


def _stable_hash(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _removed_mask(parameter: torch.Tensor, rows: Sequence[ParameterSlice]) -> torch.Tensor:
    removed = torch.zeros_like(parameter, dtype=torch.bool)
    for row in rows:
        if not row.indices:
            continue
        indices = torch.as_tensor(row.indices, dtype=torch.long, device=parameter.device)
        selector = [slice(None)] * parameter.ndim
        selector[int(row.axis)] = indices
        removed[tuple(selector)] = True
    return removed


def score_atomic_units_for_fixed_ranking(
    model: Any,
    statistics: FisherStatistics,
    unit_to_parameter_slices: Mapping[str, Sequence[ParameterSlice]],
    *,
    strict: bool = True,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Score each atomic unit for pruning only, independent of precision genes.

    For a pruned weight, ``delta_w=-w``.  The fixed ranking score is

    ``sum(|g*delta_w| + 0.5*h*delta_w^2)`` with ``h=E[g^2]``.

    Multiple closure slices touching the same parameter are unioned before
    scoring so residual/dependency overlap is never double counted.
    """

    if not statistics.gradients or not statistics.fisher_diag:
        raise RuntimeError("domain_importance_fisher_statistics_missing")
    parameters = dict(model.named_parameters())
    records: list[AtomicTaylorScore] = []
    missing: dict[str, list[str]] = {}
    # Compute elementwise terms once per parameter.  The previous loop
    # recomputed two large tensor products for every atomic unit touching the
    # parameter and synchronized the GPU each time, leaving the GPU idle for
    # minutes on V2X-ViT.  Caching preserves the exact abs-before-reduction
    # formula while making subsequent unit reductions cheap indexed sums.
    term_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for unit_id in sorted(unit_to_parameter_slices):
        by_parameter: dict[str, list[ParameterSlice]] = {}
        for row in unit_to_parameter_slices[unit_id]:
            by_parameter.setdefault(str(row.parameter_name), []).append(row)
        first_total = 0.0
        second_total = 0.0
        affected_elements = 0
        affected_parameters = 0
        for parameter_name, rows in sorted(by_parameter.items()):
            parameter = parameters.get(parameter_name)
            gradient = statistics.absolute_gradients.get(parameter_name)
            if gradient is None:
                gradient = statistics.gradients.get(parameter_name)
            fisher = statistics.fisher_diag.get(parameter_name)
            if parameter is None:
                missing.setdefault(unit_id, []).append(f"parameter:{parameter_name}")
                continue
            if gradient is None or fisher is None:
                missing.setdefault(unit_id, []).append(f"statistics:{parameter_name}")
                continue
            # This ranking is a fixed pre-search statistic.  Materialize each
            # parameter's terms once on CPU so thousands of unit reductions do
            # not force a GPU synchronize per atomic unit.
            weight = parameter.detach()
            removed = _removed_mask(torch.empty_like(weight, device="cpu"), rows)
            count = int(removed.sum())
            if count <= 0:
                continue
            if not bool(torch.isfinite(gradient.detach()).all()) or not bool(torch.isfinite(fisher.detach()).all()):
                raise RuntimeError(f"domain_importance_nonfinite_statistics:{parameter_name}")
            cached = term_cache.get(parameter_name)
            if cached is None:
                weight_cpu = weight.detach().cpu()
                gradient_cpu = gradient.detach().cpu().to(dtype=weight_cpu.dtype)
                fisher_cpu = fisher.detach().cpu().to(dtype=weight_cpu.dtype)
                first_all = (gradient_cpu * weight_cpu).abs()
                second_all = 0.5 * (fisher_cpu * weight_cpu.square()).abs()
                if bool((first_all < 0).any()) or bool((second_all < 0).any()):
                    raise RuntimeError(f"domain_importance_negative_element_term:{parameter_name}")
                term_cache[parameter_name] = (first_all.detach(), second_all.detach())
                cached = term_cache[parameter_name]
            first_all, second_all = cached
            first_total += float(first_all[removed].sum().detach().cpu())
            second_total += float(second_all[removed].sum().detach().cpu())
            affected_parameters += 1
            affected_elements += count
        records.append(
            AtomicTaylorScore(
                unit_id=str(unit_id),
                first_order=first_total,
                second_order=second_total,
                joint_second_order=first_total + second_total,
                affected_parameter_count=affected_parameters,
                affected_element_count=affected_elements,
            )
        )
    if strict and missing:
        raise RuntimeError(f"domain_importance_missing_statistics:{missing}")
    score_map = {record.unit_id: record.joint_second_order for record in records}
    manifest_payload = {
        "formula": "sum_elementwise(abs(g*(-w)) + 0.5*abs(E[g^2]*(-w)^2))_then_parameter_unit_aggregation",
        "elementwise_abs_before_reduction": True,
        "ranking_is_pruning_only": True,
        "precision_gene_independent": True,
        "statistics_manifest_hash": statistics.manifest_hash,
        "statistics_version": statistics.statistics_version,
        "records": [record.to_dict() for record in records],
        "missing": missing,
    }
    return score_map, {
        **manifest_payload,
        "ranking_manifest_hash": _stable_hash(manifest_payload),
    }
