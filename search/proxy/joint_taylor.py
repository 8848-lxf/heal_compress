"""Unified pruning and mixed-precision Taylor task-loss proxy."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from ..candidate import CandidatePhenotype
from .fisher_proxy import FisherStatistics
from .parameter_slice_resolver import ParameterSlice


JOINT_TAYLOR_MODES = (
    "joint_taylor_first_order",
    "joint_taylor_second_order_fisher_diag",
    "conditional_joint_taylor_first_order",
    "conditional_joint_taylor_second_order_fisher_diag",
)


@dataclass(frozen=True)
class JointTaylorResult:
    first_order_sum: float
    second_order_fisher_sum: float
    total_importance: float
    unique_parameter_count: int
    duplicate_slice_count: int
    importance_mode: str
    normalization: str = "none"
    sqnr_main_objective_contribution: float = 0.0
    finite: bool = True
    failure_reason: str = ""


@dataclass(frozen=True)
class ConditionalGroupImportance:
    group_id: str
    first_order_sum: float
    second_order_fisher_sum: float
    total_importance: float
    involved_precision_groups: tuple[str, ...]
    unique_parameter_count: int
    duplicate_slice_count: int
    importance_mode: str
    finite: bool
    failure_reason: str = ""


def _pseudo_quantize(value: torch.Tensor, precision: str) -> torch.Tensor:
    normalized = str(precision).upper()
    if normalized == "FP32":
        return value.detach().clone()
    if normalized == "FP16":
        return value.detach().to(torch.float16).to(value.dtype)
    if normalized != "INT8":
        raise ValueError(f"unsupported precision: {precision}")
    detached = value.detach()
    if detached.ndim <= 1:
        amax = detached.abs().amax()
        scale = torch.clamp(amax / 127.0, min=torch.finfo(detached.dtype).tiny)
    else:
        reduce_dims = tuple(range(1, detached.ndim))
        amax = detached.abs().amax(dim=reduce_dims, keepdim=True)
        scale = torch.clamp(amax / 127.0, min=torch.finfo(detached.dtype).tiny)
    return torch.clamp(torch.round(detached / scale), -127, 127) * scale


def _slice_union_mask(
    parameter: torch.Tensor,
    slices: Sequence[ParameterSlice],
) -> torch.Tensor:
    pruned = torch.zeros_like(parameter, dtype=torch.bool)
    for row in slices:
        if int(row.axis) < 0 or int(row.axis) >= parameter.ndim:
            raise RuntimeError(f"parameter_slice_axis_out_of_bounds:{row.parameter_name}:{row.axis}")
        indices = tuple(sorted({int(value) for value in row.indices}))
        if not indices:
            continue
        if min(indices) < 0 or max(indices) >= int(parameter.shape[int(row.axis)]):
            raise RuntimeError(f"parameter_slice_index_out_of_bounds:{row.parameter_name}")
        selector: list[Any] = [slice(None)] * parameter.ndim
        selector[int(row.axis)] = torch.as_tensor(indices, dtype=torch.long, device=parameter.device)
        pruned[tuple(selector)] = True
    return pruned


def effective_parameter_delta(
    parameter: torch.Tensor,
    *,
    precision: str,
    pruned_slices: Sequence[ParameterSlice] = (),
) -> torch.Tensor:
    detached = parameter.detach()
    quantized = _pseudo_quantize(detached, precision)
    pruned = _slice_union_mask(detached, pruned_slices)
    effective = torch.where(pruned, torch.zeros_like(detached), quantized)
    return effective - detached


def _cost_terms(
    delta: torch.Tensor,
    gradient: torch.Tensor,
    fisher_diag: torch.Tensor | None,
    *,
    second_order: bool,
    element_mask: torch.Tensor | None = None,
) -> tuple[float, float, int]:
    target = delta if element_mask is None else delta[element_mask]
    grad = gradient.to(device=delta.device, dtype=delta.dtype)
    grad = grad if element_mask is None else grad[element_mask]
    first = float((grad * target).abs().to(torch.float64).sum().cpu())
    second = 0.0
    if second_order:
        if fisher_diag is None:
            raise RuntimeError("joint_taylor_fisher_missing")
        fisher = fisher_diag.to(device=delta.device, dtype=delta.dtype)
        fisher = fisher if element_mask is None else fisher[element_mask]
        second = 0.5 * float((fisher * target.square()).to(torch.float64).sum().cpu())
    return first, second, int(target.numel())


class JointTaylorProxy:
    """Evaluate ``M * Q_b(W) - W`` over each unique parameter element."""

    def __init__(
        self,
        model: Any,
        *,
        statistics: FisherStatistics,
        unit_to_parameter_slices: Mapping[str, Sequence[ParameterSlice]],
        mode: str = "joint_taylor_second_order_fisher_diag",
    ) -> None:
        if mode not in JOINT_TAYLOR_MODES:
            raise ValueError(f"unsupported joint Taylor mode: {mode}")
        self.model = model
        self.statistics = statistics
        self.unit_to_parameter_slices = {
            str(key): tuple(values) for key, values in unit_to_parameter_slices.items()
        }
        self.mode = str(mode)

    @property
    def second_order(self) -> bool:
        return "second_order" in self.mode

    def _candidate_slices(
        self, phenotype: CandidatePhenotype
    ) -> dict[str, list[ParameterSlice]]:
        rows: dict[str, list[ParameterSlice]] = defaultdict(list)
        seen: set[tuple[str, int, tuple[int, ...]]] = set()
        for unit_id in phenotype.pruned_unit_ids:
            for row in self.unit_to_parameter_slices.get(unit_id, ()):
                key = (row.parameter_name, int(row.axis), tuple(sorted(set(row.indices))))
                if key in seen:
                    continue
                seen.add(key)
                rows[row.parameter_name].append(row)
        return dict(rows)

    def evaluate(self, phenotype: CandidatePhenotype) -> JointTaylorResult:
        pruned_by_parameter = self._candidate_slices(phenotype)
        first_total = 0.0
        second_total = 0.0
        unique_count = 0
        for parameter_name, parameter in self.model.named_parameters():
            gradient = self.statistics.gradients.get(parameter_name)
            if gradient is None:
                continue
            module_path = parameter_name.rsplit(".", 1)[0]
            precision = phenotype.realized_precision_profile.get(module_path, "FP32")
            delta = effective_parameter_delta(
                parameter,
                precision=precision,
                pruned_slices=pruned_by_parameter.get(parameter_name, ()),
            )
            first, second, count = _cost_terms(
                delta,
                gradient,
                self.statistics.fisher_diag.get(parameter_name),
                second_order=self.second_order,
            )
            first_total += first
            second_total += second
            unique_count += count
        total = first_total + second_total
        finite = math.isfinite(total)
        return JointTaylorResult(
            first_order_sum=float(first_total),
            second_order_fisher_sum=float(second_total),
            total_importance=float(total),
            unique_parameter_count=int(unique_count),
            duplicate_slice_count=0,
            importance_mode=self.mode,
            finite=finite,
            failure_reason="" if finite else "nonfinite_joint_taylor_score",
        )

    def conditional_group_costs(
        self, phenotype: CandidatePhenotype
    ) -> dict[str, ConditionalGroupImportance]:
        parameters = dict(self.model.named_parameters())
        output: dict[str, ConditionalGroupImportance] = {}
        for group_id, raw_rows in sorted(self.unit_to_parameter_slices.items()):
            rows_by_parameter: dict[str, list[ParameterSlice]] = defaultdict(list)
            seen: set[tuple[str, int, tuple[int, ...]]] = set()
            involved: set[str] = set()
            for row in raw_rows:
                key = (row.parameter_name, int(row.axis), tuple(sorted(set(row.indices))))
                if key in seen:
                    continue
                seen.add(key)
                rows_by_parameter[row.parameter_name].append(row)
                involved.add(row.module_path)
            first_total = 0.0
            second_total = 0.0
            unique_count = 0
            failure = ""
            try:
                for parameter_name, rows in rows_by_parameter.items():
                    parameter = parameters.get(parameter_name)
                    gradient = self.statistics.gradients.get(parameter_name)
                    if parameter is None or gradient is None:
                        continue
                    module_path = parameter_name.rsplit(".", 1)[0]
                    precision = phenotype.realized_precision_profile.get(module_path, "FP32")
                    mask = _slice_union_mask(parameter.detach(), rows)
                    keep_delta = effective_parameter_delta(parameter, precision=precision)
                    prune_delta = -parameter.detach()
                    keep_first, keep_second, _ = _cost_terms(
                        keep_delta,
                        gradient,
                        self.statistics.fisher_diag.get(parameter_name),
                        second_order=self.second_order,
                        element_mask=mask,
                    )
                    prune_first, prune_second, count = _cost_terms(
                        prune_delta,
                        gradient,
                        self.statistics.fisher_diag.get(parameter_name),
                        second_order=self.second_order,
                        element_mask=mask,
                    )
                    first_total += prune_first - keep_first
                    second_total += prune_second - keep_second
                    unique_count += count
            except RuntimeError as exc:
                failure = str(exc)
            total = first_total + second_total
            finite = not failure and math.isfinite(total)
            output[group_id] = ConditionalGroupImportance(
                group_id=group_id,
                first_order_sum=float(first_total),
                second_order_fisher_sum=float(second_total),
                total_importance=float(total),
                involved_precision_groups=tuple(sorted(involved)),
                unique_parameter_count=int(unique_count),
                duplicate_slice_count=0,
                importance_mode=self.mode,
                finite=bool(finite),
                failure_reason=failure or ("" if finite else "nonfinite_conditional_importance"),
            )
        return output

