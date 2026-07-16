"""Vectorized proxy scoring tables for large-population search."""

from __future__ import annotations

import copy
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, fields
from typing import Any

import torch
import torch.nn as nn

from ..candidate import CandidatePhenotype
from ..canonicalization import SearchSpaceSpec
from ..stage1.proxy_evaluator import BatchProxyResult
from .batch_channel_resolver import BatchChannelResolver, BatchChannelTables
from .candidate_perturbation import pseudo_quantize_tensor
from .fisher_proxy import FisherStatistics
from .normalization import NormalizationStats
from .parameter_slice_resolver import ParameterSlice
from .runtime_shape_profiler import RuntimeLayerShape
from .size_proxy import BIT_WIDTHS


@dataclass(frozen=True)
class BatchProxyTables:
    prune_loss: torch.Tensor
    sqnr_by_group_precision: torch.Tensor
    size_by_group_precision: torch.Tensor
    bops_by_group_precision: torch.Tensor
    fp16_size_baseline: float
    fp16_bops_baseline: float


def _gather_group_values(table: torch.Tensor, choices: torch.Tensor) -> torch.Tensor:
    rows = torch.arange(table.shape[0], device=choices.device)
    return table.to(choices.device)[rows.unsqueeze(0), choices]


def score_candidates_batched(
    tables: BatchProxyTables,
    prune_choices: torch.Tensor,
    precision_choices: torch.Tensor,
    *,
    bops_target: float | None,
    device: str | torch.device = "cuda",
    chunk_size: int = 128,
) -> dict[str, Any]:
    """Score candidates in chunks; reduce chunk size outside this function on OOM."""

    target_device = torch.device(device)
    prune_choices = prune_choices.to(target_device)
    precision_choices = precision_choices.to(target_device)
    outputs: dict[str, list[torch.Tensor]] = {"F1": [], "L_fisher": [], "L_sqnr": [], "R_size_vs_fp16": [], "R_bops_vs_fp16": [], "bops_violation": []}
    prune_loss = tables.prune_loss.to(target_device)
    for start in range(0, int(prune_choices.shape[0]), int(chunk_size)):
        stop = min(start + int(chunk_size), int(prune_choices.shape[0]))
        pchunk = prune_choices[start:stop]
        qchunk = precision_choices[start:stop]
        fisher = (pchunk.float() * prune_loss.unsqueeze(0)).sum(dim=1)
        sqnr = _gather_group_values(tables.sqnr_by_group_precision, qchunk).sum(dim=1)
        size = _gather_group_values(tables.size_by_group_precision, qchunk).sum(dim=1) / max(float(tables.fp16_size_baseline), 1e-12)
        bops = _gather_group_values(tables.bops_by_group_precision, qchunk).sum(dim=1) / max(float(tables.fp16_bops_baseline), 1e-12)
        violation = torch.clamp(bops - float(bops_target), min=0.0) if bops_target is not None else torch.zeros_like(bops)
        score = fisher + sqnr + size + bops + 20.0 * violation.square()
        outputs["F1"].append(score.detach())
        outputs["L_fisher"].append(fisher.detach())
        outputs["L_sqnr"].append(sqnr.detach())
        outputs["R_size_vs_fp16"].append(size.detach())
        outputs["R_bops_vs_fp16"].append(bops.detach())
        outputs["bops_violation"].append(violation.detach())
    result = {key: torch.cat(values, dim=0).cpu() for key, values in outputs.items()}
    result["candidate_count"] = int(prune_choices.shape[0])
    return result


def score_candidates_cpu(
    tables: BatchProxyTables,
    prune_choices: torch.Tensor,
    precision_choices: torch.Tensor,
    *,
    bops_target: float | None,
) -> dict[str, Any]:
    return score_candidates_batched(
        tables,
        prune_choices,
        precision_choices,
        bops_target=bops_target,
        device="cpu",
        chunk_size=max(1, int(prune_choices.shape[0])),
    )


def _slice_cost(value: torch.Tensor, axis: int, indices: tuple[int, ...]) -> float:
    if not indices:
        return 0.0
    selector = [slice(None)] * value.ndim
    selector[int(axis)] = torch.as_tensor(indices, dtype=torch.long, device=value.device)
    return float(value[tuple(selector)].sum().detach().cpu())


def _slice_taylor_cost(
    parameter: torch.Tensor,
    gradient: torch.Tensor | None,
    fisher: torch.Tensor | None,
    axis: int,
    indices: tuple[int, ...],
) -> float:
    if gradient is None or fisher is None or not indices:
        return 0.0
    parameter_value = parameter.detach()
    gradient_value = gradient.detach().to(device=parameter_value.device, dtype=parameter_value.dtype)
    fisher_value = fisher.detach().to(device=parameter_value.device, dtype=parameter_value.dtype)
    contribution = (gradient_value * parameter_value).abs() + 0.5 * fisher_value * parameter_value.pow(2)
    return _slice_cost(contribution, axis, indices)


def _slice_element_count(parameter: torch.Tensor, axis: int, indices: tuple[int, ...]) -> float:
    if not indices or int(parameter.shape[int(axis)]) <= 0:
        return 0.0
    return float(parameter.numel() * len(set(indices)) / int(parameter.shape[int(axis)]))


def _precision_index(value: str) -> int:
    return {"FP32": 0, "FP16": 1, "INT8": 2}[str(value).upper()]


def _percentile(values: torch.Tensor, pct: float) -> float:
    if values.numel() == 0:
        return 0.0
    return float(torch.quantile(values.detach().float().cpu(), float(pct)).item())


@dataclass
class TorchBatchedProxyScorer:
    """GPU-resident Stage-1 proxy scorer for phenotype batches."""

    space: SearchSpaceSpec
    device: torch.device
    batch_size: int
    action_ids: tuple[str, ...]
    precision_gene_ids: tuple[str, ...]
    layer_ids: tuple[str, ...]
    layer_cin: tuple[int, ...]
    layer_cout: tuple[int, ...]
    layer_shape_groups: tuple[tuple[int, int, tuple[int, ...]], ...]
    sqnr_active_mask: torch.Tensor
    action_fisher_cost: torch.Tensor
    fisher_weight_matrix: torch.Tensor
    fisher_out_vector: torch.Tensor
    fisher_weight_total: torch.Tensor
    fisher_total: torch.Tensor
    quant_taylor_matrix: torch.Tensor
    sqnr_noise_table: torch.Tensor
    sqnr_signal_matrix: torch.Tensor
    sqnr_noise_matrix: torch.Tensor
    sqnr_signal_by_layer: torch.Tensor
    action_sqnr_noise_reduction: torch.Tensor
    action_sqnr_signal_reduction: torch.Tensor
    layer_to_group: torch.Tensor
    layer_has_group: torch.Tensor
    precision_bits: torch.Tensor
    fp16_size_baseline: torch.Tensor
    fp32_size_baseline: torch.Tensor
    channel_resolver: BatchChannelResolver
    normalization: NormalizationStats
    config: Any
    runtime_shape_count: int
    unit_channel_effects: dict[str, tuple[tuple[int, str, tuple[int, ...]], ...]]
    uses_explicit_candidate_masks: bool = False
    compute_legacy_sqnr_metrics: bool = True
    gpu_batch_count: int = 0
    _action_index_cache: dict[str, int] = field(
        default_factory=dict, init=False, repr=False
    )
    _unit_out_flat_indices: dict[str, tuple[int, ...]] = field(
        default_factory=dict, init=False, repr=False
    )
    _unit_in_flat_indices: dict[str, tuple[int, ...]] = field(
        default_factory=dict, init=False, repr=False
    )

    backend: str = "cuda_batched"

    def clone_to_device(self, device: str | torch.device) -> "TorchBatchedProxyScorer":
        """Replicate read-only scoring tables without rebuilding Fisher data."""

        target = torch.device(device)
        clone = copy.copy(self)
        for field in fields(self):
            value = getattr(self, field.name)
            if torch.is_tensor(value):
                setattr(clone, field.name, value.to(target))
        clone.device = target
        clone.channel_resolver = self.channel_resolver.clone_to(target)
        clone.gpu_batch_count = 0
        clone._action_index_cache = dict(self._action_index_cache)
        clone._unit_out_flat_indices = dict(self._unit_out_flat_indices)
        clone._unit_in_flat_indices = dict(self._unit_in_flat_indices)
        return clone

    def _prepare_encode_cache(self) -> None:
        if self._action_index_cache:
            return
        self._action_index_cache = {
            value: index for index, value in enumerate(self.action_ids)
        }
        max_channels = int(self.channel_resolver.max_channels)
        out_rows: dict[str, tuple[int, ...]] = {}
        in_rows: dict[str, tuple[int, ...]] = {}
        for unit_id, effects in self.unit_channel_effects.items():
            out_indices: list[int] = []
            in_indices: list[int] = []
            for layer, direction, indices in effects:
                flattened = [
                    int(layer) * max_channels + int(index) for index in indices
                ]
                if direction == "out":
                    out_indices.extend(flattened)
                else:
                    in_indices.extend(flattened)
            out_rows[str(unit_id)] = tuple(out_indices)
            in_rows[str(unit_id)] = tuple(in_indices)
        self._unit_out_flat_indices = out_rows
        self._unit_in_flat_indices = in_rows

    @classmethod
    def from_components(
        cls,
        *,
        model: nn.Module,
        space: SearchSpaceSpec,
        unit_to_parameter_slices: dict[str, list[ParameterSlice]],
        fisher_statistics: FisherStatistics,
        runtime_shapes: list[RuntimeLayerShape] | tuple[RuntimeLayerShape, ...],
        normalization: NormalizationStats,
        config: Any,
        device: str | torch.device,
        batch_size: int,
    ) -> "TorchBatchedProxyScorer":
        target = torch.device(device)
        if target.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("cuda_proxy_requested_but_unavailable")
        modules = dict(model.named_modules())
        params = dict(model.named_parameters())
        joint_objective_mode = (
            str(getattr(config, "objective_mode", ""))
            == "joint_weight_taylor_hard_bops"
        )
        action_ids = tuple(space.pruning_unit_ids)
        precision_layer_set = {str(name) for name in space.precision_layer_ids}
        prunable_layer_set = {
            str(row.module_path)
            for rows in unit_to_parameter_slices.values()
            for row in rows
        }
        prunable_parameter_names = {
            str(row.parameter_name)
            for rows in unit_to_parameter_slices.values()
            for row in rows
        }
        layer_ids = tuple(sorted(precision_layer_set | prunable_layer_set))
        layer_index = {name: idx for idx, name in enumerate(layer_ids)}
        sqnr_active_mask = torch.tensor([name in precision_layer_set for name in layer_ids], dtype=torch.float32)
        precision_gene_ids = tuple(space.precision_gene_ids)
        group_index = {name: idx for idx, name in enumerate(precision_gene_ids)}
        module_to_group = {}
        if space.quantization_groups:
            for group in space.quantization_groups:
                for module_path in group.module_paths:
                    module_to_group[str(module_path)] = group.group_id
        else:
            module_to_group = {name: name for name in layer_ids}
        layer_group_ids = [module_to_group.get(name, name) for name in layer_ids]
        layer_to_group = torch.tensor([group_index.get(group_id, 0) for group_id in layer_group_ids], dtype=torch.long)
        layer_has_group = torch.tensor([group_id in group_index for group_id in layer_group_ids], dtype=torch.bool)

        action_fisher = torch.zeros(len(action_ids), dtype=torch.float32)
        action_param_reduction = torch.zeros((len(action_ids), len(layer_ids)), dtype=torch.float32)
        action_cin_reduction = torch.zeros((len(action_ids), len(layer_ids)), dtype=torch.float32)
        action_cout_reduction = torch.zeros((len(action_ids), len(layer_ids)), dtype=torch.float32)
        action_sqnr_signal_reduction = torch.zeros((len(action_ids), len(layer_ids)), dtype=torch.float32)
        action_sqnr_noise_reduction = torch.zeros((len(action_ids), len(layer_ids), 3), dtype=torch.float32)
        for action_idx, action_id in enumerate(action_ids):
            seen: set[tuple[str, int, tuple[int, ...]]] = set()
            for row in unit_to_parameter_slices.get(action_id, []):
                parameter = params.get(row.parameter_name)
                if parameter is None:
                    continue
                key = (row.parameter_name, int(row.axis), tuple(row.indices))
                if key in seen:
                    continue
                seen.add(key)
                grad = fisher_statistics.gradients.get(row.parameter_name)
                fisher = fisher_statistics.fisher_diag.get(row.parameter_name)
                if not (joint_objective_mode and space.pruning_domains):
                    action_fisher[action_idx] += _slice_taylor_cost(
                        parameter, grad, fisher, row.axis, row.indices
                    )
                layer = layer_index.get(row.module_path)
                if layer is None:
                    continue
                action_param_reduction[action_idx, layer] += _slice_element_count(parameter, row.axis, row.indices)
                module = modules.get(row.module_path)
                if row.parameter_name.endswith(".weight") and not joint_objective_mode:
                    signal_tensor = parameter.detach().pow(2)
                    action_sqnr_signal_reduction[action_idx, layer] += _slice_cost(signal_tensor, row.axis, row.indices)
                    for precision, pidx in {"FP32": 0, "FP16": 1, "INT8": 2}.items():
                        q = pseudo_quantize_tensor(parameter.detach(), precision, module=module)
                        action_sqnr_noise_reduction[action_idx, layer, pidx] += _slice_cost((q - parameter.detach()).pow(2), row.axis, row.indices)
                if row.parameter_name.endswith(".weight") and module is not None:
                    if isinstance(module, nn.ConvTranspose2d):
                        if int(row.axis) == 1:
                            action_cout_reduction[action_idx, layer] += len(set(row.indices))
                        elif int(row.axis) == 0:
                            action_cin_reduction[action_idx, layer] += len(set(row.indices))
                    else:
                        if int(row.axis) == 0:
                            action_cout_reduction[action_idx, layer] += len(set(row.indices))
                        elif int(row.axis) == 1:
                            action_cin_reduction[action_idx, layer] += len(set(row.indices))

        base_params = []
        fp16_size_baseline = 0.0
        base_cin = []
        base_cout = []
        base_groups = []
        layer_kind = []
        kernel_elems = []
        bias_out_multiplier = []
        channel_param_multiplier = []
        for name in layer_ids:
            module = modules.get(name)
            weight = getattr(module, "weight", None)
            fp16_size_baseline += float(weight.numel() * 16) if weight is not None else 0.0
            base_params.append(float(sum(parameter.numel() for parameter in module.parameters(recurse=False))) if module is not None else 0.0)
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                base_cin.append(float(module.num_features))
                base_cout.append(float(module.num_features))
            else:
                base_cin.append(float(getattr(module, "in_channels", getattr(module, "in_features", 1)) or 1))
                base_cout.append(float(getattr(module, "out_channels", getattr(module, "out_features", 1)) or 1))
            base_groups.append(float(getattr(module, "groups", 1) or 1))
            if isinstance(module, nn.Conv2d):
                layer_kind.append(1.0)
                kernel_elems.append(float(module.kernel_size[0] * module.kernel_size[1]))
                bias_out_multiplier.append(float(module.bias is not None))
                channel_param_multiplier.append(0.0)
            elif isinstance(module, nn.ConvTranspose2d):
                layer_kind.append(2.0)
                kernel_elems.append(float(module.kernel_size[0] * module.kernel_size[1]))
                bias_out_multiplier.append(float(module.bias is not None))
                channel_param_multiplier.append(0.0)
            elif isinstance(module, nn.Linear):
                layer_kind.append(3.0)
                kernel_elems.append(1.0)
                bias_out_multiplier.append(float(module.bias is not None))
                channel_param_multiplier.append(0.0)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                layer_kind.append(4.0)
                kernel_elems.append(1.0)
                bias_out_multiplier.append(0.0)
                channel_param_multiplier.append(float(module.weight is not None) + float(module.bias is not None))
            else:
                layer_kind.append(0.0)
                kernel_elems.append(1.0)
                bias_out_multiplier.append(0.0)
                channel_param_multiplier.append(0.0)
        max_channels = max(1, int(max([*base_cin, *base_cout] or [1])))
        uses_explicit_candidate_masks = bool(space.pruning_domains)
        action_out_mask = None if uses_explicit_candidate_masks else torch.zeros(
            (len(action_ids), len(layer_ids), max_channels), dtype=torch.float32
        )
        action_in_mask = None if uses_explicit_candidate_masks else torch.zeros(
            (len(action_ids), len(layer_ids), max_channels), dtype=torch.float32
        )
        unit_channel_effects: dict[
            str, tuple[tuple[int, str, tuple[int, ...]], ...]
        ] = {}
        for action_idx, action_id in enumerate(action_ids):
            effects: list[tuple[int, str, tuple[int, ...]]] = []
            for row in unit_to_parameter_slices.get(action_id, []):
                layer = layer_index.get(row.module_path)
                if layer is None:
                    continue
                module = modules.get(row.module_path)
                indices = [int(index) for index in row.indices if 0 <= int(index) < max_channels]
                if not indices:
                    continue
                is_out_axis = False
                is_in_axis = False
                if row.parameter_name.endswith(".bias") or isinstance(module, nn.modules.batchnorm._BatchNorm):
                    is_out_axis = True
                elif isinstance(module, nn.ConvTranspose2d):
                    is_in_axis = int(row.axis) == 0
                    is_out_axis = int(row.axis) == 1
                else:
                    is_out_axis = int(row.axis) == 0
                    is_in_axis = int(row.axis) == 1
                if is_out_axis:
                    effects.append((layer, "out", tuple(indices)))
                    if action_out_mask is not None:
                        action_out_mask[action_idx, layer, indices] = 1.0
                if is_in_axis:
                    effects.append((layer, "in", tuple(indices)))
                    if action_in_mask is not None:
                        action_in_mask[action_idx, layer, indices] = 1.0
            unit_channel_effects[action_id] = tuple(effects)

        sqnr_signal = torch.zeros(len(layer_ids), dtype=torch.float32)
        sqnr_noise = torch.zeros((len(layer_ids), 3), dtype=torch.float32)
        fisher_weight_matrix = torch.zeros(
            (len(layer_ids), max_channels, max_channels), dtype=torch.float32
        )
        fisher_out_vector = torch.zeros(
            (len(layer_ids), max_channels), dtype=torch.float32
        )
        if joint_objective_mode:
            sqnr_signal_matrix = torch.empty(0, dtype=torch.float32)
            sqnr_noise_matrix = torch.empty(0, dtype=torch.float32)
        else:
            sqnr_signal_matrix = torch.zeros(
                (len(layer_ids), max_channels, max_channels), dtype=torch.float32
            )
            sqnr_noise_matrix = torch.zeros(
                (len(layer_ids), max_channels, max_channels, 3), dtype=torch.float32
            )
        quant_taylor_matrix = torch.zeros(
            (len(layer_ids), max_channels, max_channels, 3),
            dtype=torch.float32,
        )
        for name, idx in layer_index.items():
            module = modules.get(name)
            weight = getattr(module, "weight", None)
            if weight is None:
                continue
            w = weight.detach()
            signal = float(w.pow(2).sum().cpu())
            sqnr_signal[idx] = signal
            weight_grad = fisher_statistics.gradients.get(f"{name}.weight")
            weight_fisher = fisher_statistics.fisher_diag.get(f"{name}.weight")
            weight_grad_value = None
            weight_fisher_value = None
            weight_is_searchable = (
                name in precision_layer_set or f"{name}.weight" in prunable_parameter_names
            )
            if weight_is_searchable and weight_grad is not None and weight_fisher is not None:
                weight_grad_value = weight_grad.detach().to(device=w.device, dtype=w.dtype)
                weight_fisher_value = weight_fisher.detach().to(device=w.device, dtype=w.dtype)
                contribution = (weight_grad_value * w).abs() + 0.5 * weight_fisher_value * w.pow(2)
                if contribution.ndim >= 2:
                    reduced = contribution.reshape(contribution.shape[0], contribution.shape[1], -1).sum(dim=2)
                    if isinstance(module, nn.ConvTranspose2d):
                        reduced = reduced.transpose(0, 1)
                    out_stop = min(max_channels, int(reduced.shape[0]))
                    in_stop = min(max_channels, int(reduced.shape[1]))
                    fisher_weight_matrix[idx, :out_stop, :in_stop] = reduced[
                        :out_stop, :in_stop
                    ].detach().float().cpu()
                elif contribution.ndim == 1:
                    stop = min(max_channels, int(contribution.shape[0]))
                    fisher_out_vector[idx, :stop] += contribution[:stop].detach().float().cpu()
            bias = getattr(module, "bias", None)
            if bias is not None and f"{name}.bias" in prunable_parameter_names:
                grad = fisher_statistics.gradients.get(f"{name}.bias")
                fisher = fisher_statistics.fisher_diag.get(f"{name}.bias")
                if grad is not None and fisher is not None:
                    b = bias.detach()
                    grad_value = grad.detach().to(device=b.device, dtype=b.dtype)
                    fisher_value = fisher.detach().to(device=b.device, dtype=b.dtype)
                    contribution = (grad_value * b).abs() + 0.5 * fisher_value * b.pow(2)
                    stop = min(max_channels, int(contribution.shape[0]))
                    fisher_out_vector[idx, :stop] += contribution[:stop].detach().float().cpu()
            for precision, pidx in {"FP32": 0, "FP16": 1, "INT8": 2}.items():
                q = pseudo_quantize_tensor(w, precision, module=module)
                if not joint_objective_mode:
                    sqnr_noise[idx, pidx] = float((q - w).pow(2).sum().cpu())
                if w.ndim >= 2:
                    if not joint_objective_mode:
                        signal_reduced = w.pow(2).reshape(
                            w.shape[0], w.shape[1], -1
                        ).sum(dim=2)
                        noise_reduced = (q - w).pow(2).reshape(
                            w.shape[0], w.shape[1], -1
                        ).sum(dim=2)
                        if isinstance(module, nn.ConvTranspose2d):
                            signal_reduced = signal_reduced.transpose(0, 1)
                            noise_reduced = noise_reduced.transpose(0, 1)
                        out_stop = min(max_channels, int(signal_reduced.shape[0]))
                        in_stop = min(max_channels, int(signal_reduced.shape[1]))
                        if pidx == 0:
                            sqnr_signal_matrix[
                                idx, :out_stop, :in_stop
                            ] = signal_reduced[:out_stop, :in_stop].detach().cpu()
                        sqnr_noise_matrix[
                            idx, :out_stop, :in_stop, pidx
                        ] = noise_reduced[:out_stop, :in_stop].detach().cpu()
                    if (
                        name in precision_layer_set
                        and weight_grad_value is not None
                        and weight_fisher_value is not None
                    ):
                        delta = q - w
                        quant_contribution = (
                            (weight_grad_value * delta).abs()
                            + 0.5 * weight_fisher_value * delta.square()
                        )
                        quant_reduced = quant_contribution.reshape(
                            quant_contribution.shape[0],
                            quant_contribution.shape[1],
                            -1,
                        ).sum(dim=2)
                        if isinstance(module, nn.ConvTranspose2d):
                            quant_reduced = quant_reduced.transpose(0, 1)
                        out_stop = min(max_channels, int(quant_reduced.shape[0]))
                        in_stop = min(max_channels, int(quant_reduced.shape[1]))
                        quant_taylor_matrix[
                            idx,
                            :out_stop,
                            :in_stop,
                            pidx,
                        ] = quant_reduced[:out_stop, :in_stop].detach().float().cpu()
        fisher_weight_total = fisher_weight_matrix.sum(dim=(1, 2))
        fisher_total = fisher_weight_total.sum() + fisher_out_vector.sum()

        channel_tables = BatchChannelTables(
            base_cin=torch.tensor(base_cin, dtype=torch.float32),
            base_cout=torch.tensor(base_cout, dtype=torch.float32),
            base_groups=torch.tensor(base_groups, dtype=torch.float32),
            base_params=torch.tensor(base_params, dtype=torch.float32),
            action_cin_reduction=action_cin_reduction,
            action_cout_reduction=action_cout_reduction,
            action_param_reduction=action_param_reduction,
            runtime_shapes=tuple(runtime_shapes),
            layer_ids=layer_ids,
            layer_kind=torch.tensor(layer_kind, dtype=torch.float32),
            kernel_elems=torch.tensor(kernel_elems, dtype=torch.float32),
            bias_out_multiplier=torch.tensor(bias_out_multiplier, dtype=torch.float32),
            channel_param_multiplier=torch.tensor(channel_param_multiplier, dtype=torch.float32),
            action_out_mask=action_out_mask,
            action_in_mask=action_in_mask,
            max_channels=max_channels,
            constant_base_params=max(
                0.0,
                float(sum(parameter.numel() for parameter in model.parameters()))
                - float(sum(base_params)),
            ),
        )
        grouped_layer_indices: dict[tuple[int, int], list[int]] = {}
        for layer, (cout, cin) in enumerate(zip(base_cout, base_cin)):
            grouped_layer_indices.setdefault((int(cout), int(cin)), []).append(layer)
        return cls(
            space=space,
            device=target,
            batch_size=int(batch_size),
            action_ids=action_ids,
            precision_gene_ids=precision_gene_ids,
            layer_ids=layer_ids,
            layer_cin=tuple(int(value) for value in base_cin),
            layer_cout=tuple(int(value) for value in base_cout),
            layer_shape_groups=tuple(
                (cout, cin, tuple(indices))
                for (cout, cin), indices in sorted(grouped_layer_indices.items())
            ),
            sqnr_active_mask=sqnr_active_mask.to(target),
            action_fisher_cost=action_fisher.to(target),
            fisher_weight_matrix=fisher_weight_matrix.to(target),
            fisher_out_vector=fisher_out_vector.to(target),
            fisher_weight_total=fisher_weight_total.to(target),
            fisher_total=fisher_total.to(target).clamp_min(1.0e-12),
            quant_taylor_matrix=quant_taylor_matrix.to(target),
            sqnr_noise_table=sqnr_noise.to(target),
            sqnr_signal_matrix=sqnr_signal_matrix.to(target),
            sqnr_noise_matrix=sqnr_noise_matrix.to(target),
            sqnr_signal_by_layer=sqnr_signal.to(target),
            action_sqnr_noise_reduction=action_sqnr_noise_reduction.to(target),
            action_sqnr_signal_reduction=action_sqnr_signal_reduction.to(target),
            layer_to_group=layer_to_group.to(target),
            layer_has_group=layer_has_group.to(target),
            precision_bits=torch.tensor([32.0, 16.0, 8.0], dtype=torch.float32, device=target),
            fp16_size_baseline=torch.tensor(max(fp16_size_baseline, 1.0), dtype=torch.float32, device=target),
            fp32_size_baseline=torch.tensor(max(fp16_size_baseline * 2.0, 1.0), dtype=torch.float32, device=target),
            channel_resolver=BatchChannelResolver(channel_tables, device=target),
            normalization=normalization,
            config=config,
            runtime_shape_count=len(runtime_shapes),
            unit_channel_effects=unit_channel_effects,
            uses_explicit_candidate_masks=uses_explicit_candidate_masks,
            compute_legacy_sqnr_metrics=not joint_objective_mode,
        )

    def _bilinear_cost_by_layer(
        self,
        retained_out: torch.Tensor,
        matrix: torch.Tensor,
        retained_in: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate exact per-layer retained cost without max-channel padding work."""

        result = torch.zeros(
            (retained_out.shape[0], len(self.layer_ids)),
            dtype=matrix.dtype,
            device=matrix.device,
        )
        for cout, cin, layer_indices in self.layer_shape_groups:
            indices = torch.as_tensor(
                layer_indices,
                dtype=torch.long,
                device=matrix.device,
            )
            left = retained_out[:, indices, :cout]
            right = retained_in[:, indices, :cin]
            grouped_matrix = matrix[indices, :cout, :cin]
            # One GEMM batch per layer-shape group.  The equivalent three-input
            # einsum may materialize a BxGxOxI intermediate and scales very
            # poorly for 128/512-candidate populations.
            weighted = torch.bmm(
                left.permute(1, 0, 2).contiguous(),
                grouped_matrix,
            )
            values = (
                weighted * right.permute(1, 0, 2)
            ).sum(dim=2).transpose(0, 1)
            result[:, indices] = values
        return result

    def _encode(
        self,
        phenotypes: list[CandidatePhenotype],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        self._prepare_encode_cache()
        action_index = self._action_index_cache
        group_index = {value: idx for idx, value in enumerate(self.precision_gene_ids)}
        pruning = torch.zeros((len(phenotypes), len(self.action_ids)), dtype=torch.int16)
        precision = torch.full((len(phenotypes), len(self.precision_gene_ids)), 1, dtype=torch.int16)
        out_masks = (
            torch.zeros(
                (len(phenotypes), len(self.layer_ids), self.channel_resolver.max_channels),
                dtype=torch.bool,
            )
            if self.uses_explicit_candidate_masks
            else None
        )
        in_masks = torch.zeros_like(out_masks) if out_masks is not None else None
        out_masks_flat = (
            out_masks.view(len(phenotypes), -1) if out_masks is not None else None
        )
        in_masks_flat = (
            in_masks.view(len(phenotypes), -1) if in_masks is not None else None
        )
        for row_idx, phenotype in enumerate(phenotypes):
            pruned_ids = [str(action_id) for action_id in phenotype.pruned_unit_ids]
            action_indices = [
                action_index[action_id]
                for action_id in pruned_ids
                if action_id in action_index
            ]
            if action_indices:
                pruning[row_idx, action_indices] = 1
            if out_masks_flat is not None and in_masks_flat is not None:
                out_indices = [
                    index
                    for action_id in pruned_ids
                    for index in self._unit_out_flat_indices.get(action_id, ())
                ]
                in_indices = [
                    index
                    for action_id in pruned_ids
                    for index in self._unit_in_flat_indices.get(action_id, ())
                ]
                if out_indices:
                    out_masks_flat[row_idx, out_indices] = True
                if in_indices:
                    in_masks_flat[row_idx, in_indices] = True
            if self.space.quantization_groups:
                for group in self.space.quantization_groups:
                    first_module = group.module_paths[0]
                    decision = phenotype.precision_profile.get(first_module)
                    idx = group_index.get(group.group_id)
                    if idx is not None and decision is not None:
                        precision[row_idx, idx] = _precision_index(decision.realized_precision)
            else:
                for group_id, idx in group_index.items():
                    decision = phenotype.precision_profile.get(group_id)
                    if decision is not None:
                        precision[row_idx, idx] = _precision_index(decision.realized_precision)
        return (
            pruning.to(self.device, non_blocking=True),
            precision.to(self.device, non_blocking=True),
            out_masks.to(self.device, non_blocking=True) if out_masks is not None else None,
            in_masks.to(self.device, non_blocking=True) if in_masks is not None else None,
        )

    def evaluate_batch(self, phenotypes: list[CandidatePhenotype], *, generation: int, outer_round: int) -> BatchProxyResult:
        if not phenotypes:
            return BatchProxyResult(metrics=[], stats={"gpu_batch_count": 0, "proxy_backend": self.backend})
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        else:
            start = end = None
        pruning_all, precision_all, out_masks_all, in_masks_all = self._encode(phenotypes)
        metric_chunks: dict[str, list[torch.Tensor]] = {
            "L_joint_weight_taylor": [],
            "L_pruning_only_taylor": [],
            "L_retained_weight_quant_taylor": [],
            "L_fisher": [],
            "L_sqnr": [],
            "R_size": [],
            "R_size_vs_fp16": [],
            "R_parameter_retention": [],
            "R_bops": [],
            "R_bops_vs_fp16": [],
            "R_bops_vs_fp32": [],
            "P_bops": [],
            "bops_violation": [],
            "int8_macs_ratio": [],
            "F1": [],
        }
        batch_count = 0
        for start_idx in range(0, len(phenotypes), self.batch_size):
            stop_idx = min(start_idx + self.batch_size, len(phenotypes))
            pruning = pruning_all[start_idx:stop_idx].to(dtype=torch.float32)
            precision = precision_all[start_idx:stop_idx].to(dtype=torch.long)
            batch_count += 1
            if out_masks_all is not None and in_masks_all is not None:
                channel = self.channel_resolver.resolve_explicit_masks(
                    out_masks_all[start_idx:stop_idx],
                    in_masks_all[start_idx:stop_idx],
                    pruning_choice_tensor=pruning,
                )
            else:
                channel = self.channel_resolver.resolve(pruning)
            layer_precision = torch.where(
                self.layer_has_group.unsqueeze(0),
                precision[:, self.layer_to_group],
                torch.ones((precision.shape[0], len(self.layer_ids)), dtype=torch.long, device=self.device),
            )
            layer_bits = self.precision_bits[layer_precision]
            params_after = channel["params_after"]
            size_bits = (params_after * layer_bits).sum(dim=1)
            fp16_size = self.fp16_size_baseline.clamp_min(1.0)
            out_pruned = channel.get("out_pruned_mask")
            in_pruned = channel.get("in_pruned_mask")
            if out_pruned is not None and in_pruned is not None:
                retained_out = (~out_pruned).float() * self.channel_resolver.valid_out_mask.unsqueeze(0).float()
                retained_in = (~in_pruned).float() * self.channel_resolver.valid_in_mask.unsqueeze(0).float()
                retained_out_fisher = retained_out.to(dtype=self.fisher_weight_matrix.dtype)
                retained_in_fisher = retained_in.to(dtype=self.fisher_weight_matrix.dtype)
                retained_fisher_weight = self._bilinear_cost_by_layer(
                    retained_out_fisher,
                    self.fisher_weight_matrix,
                    retained_in_fisher,
                )
                fisher_weight = (self.fisher_weight_total.unsqueeze(0) - retained_fisher_weight).clamp_min(0.0)
                fisher_vector = (out_pruned.to(dtype=self.fisher_out_vector.dtype) * self.fisher_out_vector.unsqueeze(0)).sum(dim=2)
                pruning_taylor_raw = (fisher_weight + fisher_vector).sum(dim=1)
                pruning_taylor_raw = torch.where(
                    pruning.sum(dim=1) == 0.0,
                    torch.zeros_like(pruning_taylor_raw),
                    pruning_taylor_raw,
                )
                fisher = pruning_taylor_raw / self.fisher_total.clamp_min(1.0e-12)
                retained_quant_taylor_raw = torch.zeros_like(pruning_taylor_raw)
                for precision_idx in range(3):
                    if precision_idx == 0:
                        continue
                    candidate_has_precision = (layer_precision == precision_idx).to(
                        dtype=retained_quant_taylor_raw.dtype
                    )
                    candidate_quant_by_layer = self._bilinear_cost_by_layer(
                        retained_out_fisher,
                        self.quant_taylor_matrix[:, :, :, precision_idx],
                        retained_in_fisher,
                    )
                    retained_quant_taylor_raw += (
                        candidate_quant_by_layer * candidate_has_precision
                    ).sum(dim=1)
                joint_taylor = (
                    pruning_taylor_raw + retained_quant_taylor_raw
                ) / self.fisher_total.clamp_min(1.0e-12)
                retained_quant_taylor = (
                    retained_quant_taylor_raw / self.fisher_total.clamp_min(1.0e-12)
                )
                if self.compute_legacy_sqnr_metrics:
                    signal_after = torch.einsum(
                        "blo,loi,bli->bl",
                        retained_out,
                        self.sqnr_signal_matrix,
                        retained_in,
                    ).clamp_min(0.0)
                    noise_after = torch.zeros_like(signal_after)
                    for precision_idx in range(3):
                        candidate_noise = torch.einsum(
                            "blo,loi,bli->bl",
                            retained_out,
                            self.sqnr_noise_matrix[:, :, :, precision_idx],
                            retained_in,
                        ).clamp_min(0.0)
                        noise_after = torch.where(
                            layer_precision == precision_idx,
                            candidate_noise,
                            noise_after,
                        )
                    sqnr = (
                        (noise_after / (signal_after + 1.0e-12))
                        * self.sqnr_active_mask.unsqueeze(0)
                    ).sum(dim=1)
                else:
                    sqnr = torch.zeros_like(joint_taylor)
            else:
                fisher = pruning.matmul(self.action_fisher_cost) / self.action_fisher_cost.sum().clamp_min(1.0e-12)
                joint_taylor = fisher
                retained_quant_taylor = torch.zeros_like(fisher)
                if self.compute_legacy_sqnr_metrics:
                    sqnr_noise_total = torch.gather(
                        self.sqnr_noise_table.unsqueeze(0).expand(
                            layer_precision.shape[0], -1, -1
                        ),
                        2,
                        layer_precision.unsqueeze(-1),
                    ).squeeze(-1)
                    signal_reduction = torch.einsum(
                        "bd,dl->bl", pruning, self.action_sqnr_signal_reduction
                    )
                    noise_reduction_all = torch.einsum(
                        "bd,dlp->blp", pruning, self.action_sqnr_noise_reduction
                    )
                    noise_reduction = torch.gather(
                        noise_reduction_all,
                        2,
                        layer_precision.unsqueeze(-1),
                    ).squeeze(-1)
                    signal_after = (
                        self.sqnr_signal_by_layer.unsqueeze(0) - signal_reduction
                    ).clamp_min(0.0)
                    noise_after = (sqnr_noise_total - noise_reduction).clamp_min(0.0)
                    sqnr = (
                        (noise_after / (signal_after + 1.0e-12))
                        * self.sqnr_active_mask.unsqueeze(0)
                    ).sum(dim=1)
                else:
                    sqnr = torch.zeros_like(joint_taylor)
            macs_after = self.channel_resolver.macs_after(channel)
            shape_bits = layer_bits[:, self.channel_resolver.shape_layer_indices]
            bops = (macs_after * shape_bits * shape_bits).sum(dim=1)
            fp16_bops = self.channel_resolver.fp16_bops_baseline
            fp32_bops = self.channel_resolver.fp32_bops_baseline
            shape_precision = layer_precision[:, self.channel_resolver.shape_layer_indices]
            int8_macs = torch.where(shape_precision == 2, macs_after, torch.zeros_like(macs_after)).sum(dim=1)
            total_macs = macs_after.sum(dim=1).clamp_min(1.0)
            fp32_size = self.fp32_size_baseline.clamp_min(1.0)
            parameter_count_base = (
                self.channel_resolver.base_params.sum()
                + self.channel_resolver.constant_base_params
            ).clamp_min(1.0)
            parameter_count_after = (
                params_after.sum(dim=1)
                + self.channel_resolver.constant_base_params
            )
            parameter_retention = parameter_count_after / parameter_count_base
            r_size_fp16 = size_bits / fp16_size
            r_size = size_bits / fp32_size
            r_bops_fp16 = bops / fp16_bops.clamp_min(1.0)
            r_bops_fp32 = bops / fp32_bops.clamp_min(1.0)
            r_bops = r_bops_fp32
            target = getattr(self.config, "bops_threshold", None)
            if target is None:
                bops_violation = torch.zeros_like(r_bops)
            elif str(getattr(self.config, "bops_penalty_formula", "")) == "squared_relative_excess":
                bops_violation = torch.clamp(r_bops / max(float(target), 1.0e-12) - 1.0, min=0.0)
            else:
                bops_violation = torch.clamp(r_bops - float(target), min=0.0)
            p_bops = bops_violation.square()
            norm_fisher = fisher / max(abs(float(self.normalization.medians.get("L_fisher", 1.0) or 1.0)), 1.0e-12)
            norm_sqnr = sqnr / max(abs(float(self.normalization.medians.get("L_sqnr", 1.0) or 1.0)), 1.0e-12)
            if str(getattr(self.config, "objective_mode", "")) == "joint_weight_taylor_hard_bops":
                raw_score = joint_taylor + float(
                    getattr(self.config, "parameter_retention_tiebreak_epsilon", 0.0)
                ) * parameter_retention
                hard = str(getattr(self.config, "bops_constraint_mode", "")) in {
                    "hard_feasibility",
                    "feasibility_first",
                }
                if hard:
                    score = torch.where(
                        bops_violation > 0.0,
                        1.0e6 + bops_violation * 1.0e3 + raw_score,
                        raw_score,
                    )
                else:
                    score = raw_score
            else:
                raw_score = (
                    float(getattr(self.config, "alpha_fisher", 1.0)) * norm_fisher
                    + float(getattr(self.config, "beta_sqnr", 1.0)) * norm_sqnr
                    + float(getattr(self.config, "gamma_size", 1.0)) * r_size
                    + float(getattr(self.config, "delta_bops", 1.0)) * p_bops
                )
                score = raw_score
            metric_chunks["L_joint_weight_taylor"].append(joint_taylor.detach())
            metric_chunks["L_pruning_only_taylor"].append(fisher.detach())
            metric_chunks["L_retained_weight_quant_taylor"].append(retained_quant_taylor.detach())
            metric_chunks["L_fisher"].append(fisher.detach())
            metric_chunks["L_sqnr"].append(sqnr.detach())
            metric_chunks["R_size"].append(r_size.detach())
            metric_chunks["R_size_vs_fp16"].append(r_size_fp16.detach())
            metric_chunks["R_parameter_retention"].append(parameter_retention.detach())
            metric_chunks["R_bops"].append(r_bops.detach())
            metric_chunks["R_bops_vs_fp16"].append(r_bops_fp16.detach())
            metric_chunks["R_bops_vs_fp32"].append(r_bops_fp32.detach())
            metric_chunks["P_bops"].append(p_bops.detach())
            metric_chunks["bops_violation"].append(bops_violation.detach())
            metric_chunks["int8_macs_ratio"].append((int8_macs / total_macs).detach())
            metric_chunks["F1"].append(score.detach())
        if self.device.type == "cuda" and start is not None and end is not None:
            end.record()
            torch.cuda.synchronize(self.device)
            elapsed_ms = float(start.elapsed_time(end))
            peak_memory = int(torch.cuda.max_memory_allocated(self.device))
        else:
            elapsed_ms = 0.0
            peak_memory = 0
        metrics_cpu = {key: torch.cat(values, dim=0).cpu() for key, values in metric_chunks.items()}
        fp16_bops_value = float(self.channel_resolver.fp16_bops_baseline.detach().cpu())
        fp32_bops_value = float(self.channel_resolver.fp32_bops_baseline.detach().cpu())
        normalization_payload = self.normalization.to_dict()
        rows: list[dict[str, Any]] = []
        for idx in range(len(phenotypes)):
            r_size = float(metrics_cpu["R_size"][idx])
            r_bops = float(metrics_cpu["R_bops"][idx])
            score = float(metrics_cpu["F1"][idx])
            rows.append(
                {
                    "L_fisher": float(metrics_cpu["L_fisher"][idx]),
                    "L_sqnr": float(metrics_cpu["L_sqnr"][idx]),
                    "L_joint_weight_taylor": float(metrics_cpu["L_joint_weight_taylor"][idx]),
                    "L_pruning_only_taylor": float(metrics_cpu["L_pruning_only_taylor"][idx]),
                    "L_retained_weight_quant_taylor": float(
                        metrics_cpu["L_retained_weight_quant_taylor"][idx]
                    ),
                    "R_size": r_size,
                    "R_size_vs_fp16_deploy": float(metrics_cpu["R_size_vs_fp16"][idx]),
                    "R_size_vs_fp32": r_size,
                    "R_size_reference": "original_fp32",
                    "R_parameter_retention": float(
                        metrics_cpu["R_parameter_retention"][idx]
                    ),
                    "parameter_pruning_rate": 1.0
                    - float(metrics_cpu["R_parameter_retention"][idx]),
                    "parameter_count_base": float(
                        (
                            self.channel_resolver.base_params.sum()
                            + self.channel_resolver.constant_base_params
                        ).detach().cpu()
                    ),
                    "parameter_count_after": float(
                        metrics_cpu["R_parameter_retention"][idx]
                        * (
                            self.channel_resolver.base_params.sum()
                            + self.channel_resolver.constant_base_params
                        ).detach().cpu()
                    ),
                    "constant_untracked_parameter_count": float(
                        self.channel_resolver.constant_base_params.detach().cpu()
                    ),
                    "R_bops": r_bops,
                    "R_bops_vs_fp16_deploy": float(metrics_cpu["R_bops_vs_fp16"][idx]),
                    "R_bops_vs_fp32": float(metrics_cpu["R_bops_vs_fp32"][idx]),
                    "R_bops_reference": "original_fp32",
                    "BOPS_target": float(getattr(self.config, "bops_threshold", 0.0)) if getattr(self.config, "bops_threshold", None) is not None else None,
                    "P_bops": float(metrics_cpu["P_bops"][idx]),
                    "int8_macs_ratio": float(metrics_cpu["int8_macs_ratio"][idx]),
                    "bops_fp16_baseline": fp16_bops_value,
                    "bops_fp32_baseline": fp32_bops_value,
                    "constraint_penalty": 0.0,
                    "bops_violation": float(metrics_cpu["bops_violation"][idx]),
                    "proxy_score_raw": float(
                        metrics_cpu["L_joint_weight_taylor"][idx]
                        if str(getattr(self.config, "objective_mode", ""))
                        == "joint_weight_taylor_hard_bops"
                        else score
                    ),
                    "F1": score,
                    "legal": True,
                    "normalization": normalization_payload,
                    "objective_mode": str(
                        getattr(self.config, "objective_mode", "legacy_fisher_sqnr_size_bops")
                    ),
                    "activation_taylor_included": False,
                    "generation": generation,
                    "outer_round": outer_round,
                }
            )
        self.gpu_batch_count += batch_count
        return BatchProxyResult(
            metrics=rows,
            stats={
                "proxy_backend": self.backend if self.device.type == "cuda" else "torch_batched_cpu",
                "proxy_device": str(self.device),
                "gpu_batch_count": batch_count,
                "cuda_event_elapsed_ms": elapsed_ms,
                "gpu_peak_memory_bytes": peak_memory,
                "candidates_per_second": float(len(phenotypes) / (elapsed_ms / 1000.0)) if elapsed_ms > 0.0 else 0.0,
            },
        )


@dataclass
class MultiDeviceTorchBatchedProxyScorer:
    """Shard one population across identical proxy tables on multiple GPUs."""

    scorers: tuple[TorchBatchedProxyScorer, ...]
    backend: str = "cuda_batched"

    @property
    def devices(self) -> tuple[str, ...]:
        return tuple(str(scorer.device) for scorer in self.scorers)

    @property
    def config(self) -> Any:
        return self.scorers[0].config

    @config.setter
    def config(self, value: Any) -> None:
        for scorer in self.scorers:
            scorer.config = value

    def evaluate_batch(
        self,
        phenotypes: list[CandidatePhenotype],
        *,
        generation: int,
        outer_round: int,
    ) -> BatchProxyResult:
        if not phenotypes:
            return BatchProxyResult(
                metrics=[],
                stats={
                    "gpu_batch_count": 0,
                    "proxy_backend": self.backend,
                    "proxy_gpu_ids": list(self.devices),
                },
            )
        if not self.scorers:
            raise RuntimeError("multi_gpu_proxy_has_no_scorers")
        worker_count = min(len(self.scorers), len(phenotypes))
        base, remainder = divmod(len(phenotypes), worker_count)
        partitions: list[tuple[int, int, TorchBatchedProxyScorer]] = []
        start = 0
        for worker_index in range(worker_count):
            size = base + (1 if worker_index < remainder else 0)
            stop = start + size
            partitions.append((start, stop, self.scorers[worker_index]))
            start = stop
        started = time.perf_counter()

        def score_partition(row: tuple[int, int, TorchBatchedProxyScorer]) -> tuple[int, BatchProxyResult]:
            begin, end, scorer = row
            return begin, scorer.evaluate_batch(
                phenotypes[begin:end],
                generation=generation,
                outer_round=outer_round,
            )

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            results = list(executor.map(score_partition, partitions))
        results.sort(key=lambda row: row[0])
        wall_ms = (time.perf_counter() - started) * 1000.0
        metrics = [metric for _start, result in results for metric in result.metrics]
        device_stats = [dict(result.stats) for _start, result in results]
        return BatchProxyResult(
            metrics=metrics,
            stats={
                "proxy_backend": self.backend,
                "proxy_device": ",".join(self.devices[:worker_count]),
                "proxy_gpu_ids": list(self.devices[:worker_count]),
                "multi_gpu_worker_count": worker_count,
                "gpu_batch_count": sum(
                    int(row.get("gpu_batch_count", 0) or 0) for row in device_stats
                ),
                "cuda_event_elapsed_ms": max(
                    (float(row.get("cuda_event_elapsed_ms", 0.0) or 0.0) for row in device_stats),
                    default=0.0,
                ),
                "multi_gpu_wall_elapsed_ms": wall_ms,
                "gpu_peak_memory_bytes": sum(
                    int(row.get("gpu_peak_memory_bytes", 0) or 0) for row in device_stats
                ),
                "candidates_per_second": (
                    float(len(phenotypes) / (wall_ms / 1000.0)) if wall_ms > 0.0 else 0.0
                ),
                "per_device_stats": device_stats,
            },
        )
