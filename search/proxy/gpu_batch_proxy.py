"""Vectorized proxy scoring tables for large-population search."""

from __future__ import annotations

from dataclasses import dataclass
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
    sqnr_active_mask: torch.Tensor
    action_fisher_cost: torch.Tensor
    fisher_weight_matrix: torch.Tensor
    fisher_out_vector: torch.Tensor
    fisher_weight_total: torch.Tensor
    fisher_total: torch.Tensor
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
    gpu_batch_count: int = 0

    backend: str = "cuda_batched"

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
        action_ids = tuple(space.pruning_unit_ids)
        precision_layer_set = {str(name) for name in space.precision_layer_ids}
        weighted_layer_set = {
            str(name)
            for name, module in model.named_modules()
            if name and getattr(module, "weight", None) is not None
        }
        layer_ids = tuple(sorted(precision_layer_set | weighted_layer_set))
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
                action_fisher[action_idx] += _slice_taylor_cost(parameter, grad, fisher, row.axis, row.indices)
                layer = layer_index.get(row.module_path)
                if layer is None:
                    continue
                action_param_reduction[action_idx, layer] += _slice_element_count(parameter, row.axis, row.indices)
                module = modules.get(row.module_path)
                if row.parameter_name.endswith(".weight"):
                    signal_tensor = parameter.detach().pow(2)
                    action_sqnr_signal_reduction[action_idx, layer] += _slice_cost(signal_tensor, row.axis, row.indices)
                    for precision, pidx in {"FP32": 0, "FP16": 1, "INT8": 2}.items():
                        q = pseudo_quantize_tensor(parameter.detach(), precision)
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
        action_out_mask = torch.zeros((len(action_ids), len(layer_ids), max_channels), dtype=torch.float32)
        action_in_mask = torch.zeros((len(action_ids), len(layer_ids), max_channels), dtype=torch.float32)
        for action_idx, action_id in enumerate(action_ids):
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
                    action_out_mask[action_idx, layer, indices] = 1.0
                if is_in_axis:
                    action_in_mask[action_idx, layer, indices] = 1.0

        sqnr_signal = torch.zeros(len(layer_ids), dtype=torch.float32)
        sqnr_noise = torch.zeros((len(layer_ids), 3), dtype=torch.float32)
        fisher_weight_matrix = torch.zeros((len(layer_ids), max_channels, max_channels), dtype=torch.float64)
        fisher_out_vector = torch.zeros((len(layer_ids), max_channels), dtype=torch.float64)
        sqnr_signal_matrix = torch.zeros((len(layer_ids), max_channels, max_channels), dtype=torch.float32)
        sqnr_noise_matrix = torch.zeros((len(layer_ids), max_channels, max_channels, 3), dtype=torch.float32)
        for name, idx in layer_index.items():
            module = modules.get(name)
            weight = getattr(module, "weight", None)
            if weight is None:
                continue
            w = weight.detach()
            signal = float(w.pow(2).sum().cpu())
            sqnr_signal[idx] = signal
            grad = fisher_statistics.gradients.get(f"{name}.weight")
            fisher = fisher_statistics.fisher_diag.get(f"{name}.weight")
            if grad is not None and fisher is not None:
                grad_value = grad.detach().to(device=w.device, dtype=w.dtype)
                fisher_value = fisher.detach().to(device=w.device, dtype=w.dtype)
                contribution = (grad_value * w).abs() + 0.5 * fisher_value * w.pow(2)
                if contribution.ndim >= 2:
                    reduced = contribution.reshape(contribution.shape[0], contribution.shape[1], -1).sum(dim=2)
                    if isinstance(module, nn.ConvTranspose2d):
                        reduced = reduced.transpose(0, 1)
                    out_stop = min(max_channels, int(reduced.shape[0]))
                    in_stop = min(max_channels, int(reduced.shape[1]))
                    fisher_weight_matrix[idx, :out_stop, :in_stop] = reduced[:out_stop, :in_stop].detach().double().cpu()
                elif contribution.ndim == 1:
                    stop = min(max_channels, int(contribution.shape[0]))
                    fisher_out_vector[idx, :stop] += contribution[:stop].detach().double().cpu()
            bias = getattr(module, "bias", None)
            if bias is not None:
                grad = fisher_statistics.gradients.get(f"{name}.bias")
                fisher = fisher_statistics.fisher_diag.get(f"{name}.bias")
                if grad is not None and fisher is not None:
                    b = bias.detach()
                    grad_value = grad.detach().to(device=b.device, dtype=b.dtype)
                    fisher_value = fisher.detach().to(device=b.device, dtype=b.dtype)
                    contribution = (grad_value * b).abs() + 0.5 * fisher_value * b.pow(2)
                    stop = min(max_channels, int(contribution.shape[0]))
                    fisher_out_vector[idx, :stop] += contribution[:stop].detach().double().cpu()
            for precision, pidx in {"FP32": 0, "FP16": 1, "INT8": 2}.items():
                q = pseudo_quantize_tensor(w, precision)
                sqnr_noise[idx, pidx] = float((q - w).pow(2).sum().cpu())
                if w.ndim >= 2:
                    signal_reduced = w.pow(2).reshape(w.shape[0], w.shape[1], -1).sum(dim=2)
                    noise_reduced = (q - w).pow(2).reshape(w.shape[0], w.shape[1], -1).sum(dim=2)
                    if isinstance(module, nn.ConvTranspose2d):
                        signal_reduced = signal_reduced.transpose(0, 1)
                        noise_reduced = noise_reduced.transpose(0, 1)
                    out_stop = min(max_channels, int(signal_reduced.shape[0]))
                    in_stop = min(max_channels, int(signal_reduced.shape[1]))
                    if pidx == 0:
                        sqnr_signal_matrix[idx, :out_stop, :in_stop] = signal_reduced[:out_stop, :in_stop].detach().cpu()
                    sqnr_noise_matrix[idx, :out_stop, :in_stop, pidx] = noise_reduced[:out_stop, :in_stop].detach().cpu()
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
        )
        return cls(
            space=space,
            device=target,
            batch_size=int(batch_size),
            action_ids=action_ids,
            precision_gene_ids=precision_gene_ids,
            layer_ids=layer_ids,
            sqnr_active_mask=sqnr_active_mask.to(target),
            action_fisher_cost=action_fisher.to(target),
            fisher_weight_matrix=fisher_weight_matrix.to(target),
            fisher_out_vector=fisher_out_vector.to(target),
            fisher_weight_total=fisher_weight_total.to(target),
            fisher_total=fisher_total.to(target).clamp_min(1.0e-12),
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
        )

    def _encode(self, phenotypes: list[CandidatePhenotype]) -> tuple[torch.Tensor, torch.Tensor]:
        action_index = {value: idx for idx, value in enumerate(self.action_ids)}
        group_index = {value: idx for idx, value in enumerate(self.precision_gene_ids)}
        pruning = torch.zeros((len(phenotypes), len(self.action_ids)), dtype=torch.int16)
        precision = torch.full((len(phenotypes), len(self.precision_gene_ids)), 1, dtype=torch.int16)
        for row_idx, phenotype in enumerate(phenotypes):
            for action_id in phenotype.pruned_unit_ids:
                idx = action_index.get(str(action_id))
                if idx is not None:
                    pruning[row_idx, idx] = 1.0
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
        return pruning.to(self.device, non_blocking=True), precision.to(self.device, non_blocking=True)

    def evaluate_batch(self, phenotypes: list[CandidatePhenotype], *, generation: int, outer_round: int) -> BatchProxyResult:
        if not phenotypes:
            return BatchProxyResult(metrics=[], stats={"gpu_batch_count": 0, "proxy_backend": self.backend})
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        else:
            start = end = None
        pruning_all, precision_all = self._encode(phenotypes)
        metric_chunks: dict[str, list[torch.Tensor]] = {
            "L_fisher": [],
            "L_sqnr": [],
            "R_size": [],
            "R_size_vs_fp16": [],
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
                retained_fisher_weight = torch.einsum("blo,loi,bli->bl", retained_out_fisher, self.fisher_weight_matrix, retained_in_fisher)
                fisher_weight = (self.fisher_weight_total.unsqueeze(0) - retained_fisher_weight).clamp_min(0.0)
                fisher_vector = (out_pruned.to(dtype=self.fisher_out_vector.dtype) * self.fisher_out_vector.unsqueeze(0)).sum(dim=2)
                fisher = (fisher_weight + fisher_vector).sum(dim=1) / self.fisher_total.clamp_min(1.0e-12)
                signal_after = torch.einsum("blo,loi,bli->bl", retained_out, self.sqnr_signal_matrix, retained_in).clamp_min(0.0)
                noise_after = torch.zeros_like(signal_after)
                for precision_idx in range(3):
                    candidate_noise = torch.einsum(
                        "blo,loi,bli->bl",
                        retained_out,
                        self.sqnr_noise_matrix[:, :, :, precision_idx],
                        retained_in,
                    ).clamp_min(0.0)
                    noise_after = torch.where(layer_precision == precision_idx, candidate_noise, noise_after)
                sqnr = ((noise_after / (signal_after + 1.0e-12)) * self.sqnr_active_mask.unsqueeze(0)).sum(dim=1)
            else:
                fisher = pruning.matmul(self.action_fisher_cost) / self.action_fisher_cost.sum().clamp_min(1.0e-12)
                sqnr_noise_total = torch.gather(
                    self.sqnr_noise_table.unsqueeze(0).expand(layer_precision.shape[0], -1, -1),
                    2,
                    layer_precision.unsqueeze(-1),
                ).squeeze(-1)
                signal_reduction = torch.einsum("bd,dl->bl", pruning, self.action_sqnr_signal_reduction)
                noise_reduction_all = torch.einsum("bd,dlp->blp", pruning, self.action_sqnr_noise_reduction)
                noise_reduction = torch.gather(noise_reduction_all, 2, layer_precision.unsqueeze(-1)).squeeze(-1)
                signal_after = (self.sqnr_signal_by_layer.unsqueeze(0) - signal_reduction).clamp_min(0.0)
                noise_after = (sqnr_noise_total - noise_reduction).clamp_min(0.0)
                sqnr = ((noise_after / (signal_after + 1.0e-12)) * self.sqnr_active_mask.unsqueeze(0)).sum(dim=1)
            macs_after = self.channel_resolver.macs_after(channel)
            shape_bits = layer_bits[:, self.channel_resolver.shape_layer_indices]
            bops = (macs_after * shape_bits * shape_bits).sum(dim=1)
            fp16_bops = self.channel_resolver.fp16_bops_baseline
            fp32_bops = self.channel_resolver.fp32_bops_baseline
            shape_precision = layer_precision[:, self.channel_resolver.shape_layer_indices]
            int8_macs = torch.where(shape_precision == 2, macs_after, torch.zeros_like(macs_after)).sum(dim=1)
            total_macs = macs_after.sum(dim=1).clamp_min(1.0)
            fp32_size = self.fp32_size_baseline.clamp_min(1.0)
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
            score = (
                float(getattr(self.config, "alpha_fisher", 1.0)) * norm_fisher
                + float(getattr(self.config, "beta_sqnr", 1.0)) * norm_sqnr
                + float(getattr(self.config, "gamma_size", 1.0)) * r_size
                + float(getattr(self.config, "delta_bops", 1.0)) * p_bops
            )
            metric_chunks["L_fisher"].append(fisher.detach())
            metric_chunks["L_sqnr"].append(sqnr.detach())
            metric_chunks["R_size"].append(r_size.detach())
            metric_chunks["R_size_vs_fp16"].append(r_size_fp16.detach())
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
                    "R_size": r_size,
                    "R_size_vs_fp16_deploy": float(metrics_cpu["R_size_vs_fp16"][idx]),
                    "R_size_vs_fp32": r_size,
                    "R_size_reference": "original_fp32",
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
                    "proxy_score_raw": score,
                    "F1": score,
                    "legal": True,
                    "normalization": normalization_payload,
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
