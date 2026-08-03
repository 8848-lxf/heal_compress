"""Fixed conservative Taylor costs for V2X-ViT Greedy actions.

Statistics are collected once before search.  Structural actions use functional
gate coordinates while precision actions use the tensors that feed deployment
Q/DQ boundaries.  The Greedy loop only performs dictionary lookups and sums.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import re
from typing import Any, Callable, Mapping, Sequence

import torch
import torch.nn as nn

from ..candidate import CandidatePhenotype
from ..pruning_space.local_domains import LocalPruningDomain
from ..quantization_space.types import QuantizationSearchGroup
from .joint_weight_activation_taylor import (
    _BoundaryCapture,
    _stable_hash,
    JointOutputTaylorStatistics,
    TaylorDeploymentUnit,
    coalesce_deployment_units,
    collect_joint_output_taylor_statistics,
    taylor_units_from_transformer_precision,
)
from .candidate_perturbation import pseudo_quantize_activation


def activation_taylor_terms(
    value: torch.Tensor,
    gradient: torch.Tensor,
    delta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return elementwise conservative first- and Fisher-second-order terms."""

    if value.shape != gradient.shape or value.shape != delta.shape:
        raise ValueError(
            f"activation_taylor_shape_mismatch:{value.shape}:{gradient.shape}:{delta.shape}"
        )
    if not all(bool(torch.isfinite(item).all()) for item in (value, gradient, delta)):
        raise RuntimeError("activation_taylor_nonfinite")
    first = (gradient * delta).abs()
    second = 0.5 * (gradient.square() * delta.square()).abs()
    if not bool(torch.isfinite(first).all()) or not bool(torch.isfinite(second).all()):
        raise RuntimeError("activation_taylor_terms_nonfinite")
    return first, second


class StructuralGateTaylorProxy:
    """Score only newly removed functional coordinates of one width action."""

    def __init__(
        self,
        *,
        unit_terms: Mapping[str, Mapping[str, float]],
        domain_units: Mapping[str, Sequence[str]],
        mapping_rows: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        self.unit_terms = {
            str(unit): {
                "first": float(values["first"]),
                "second": float(values["second"]),
            }
            for unit, values in unit_terms.items()
        }
        self.domain_units = {
            str(domain): tuple(str(unit) for unit in units)
            for domain, units in domain_units.items()
        }
        expected = {unit for units in self.domain_units.values() for unit in units}
        missing = sorted(expected - set(self.unit_terms))
        if missing:
            raise RuntimeError(f"structural_gate_mapping_incomplete:{missing[:16]}")
        for unit, terms in self.unit_terms.items():
            if min(terms.values()) < 0.0:
                raise RuntimeError(f"structural_gate_negative_term:{unit}:{terms}")
        self.mapping_rows = [dict(row) for row in mapping_rows]

    def pruning_action_breakdown(
        self,
        current: CandidatePhenotype,
        successor: CandidatePhenotype,
    ) -> dict[str, Any]:
        newly_removed = sorted(
            set(successor.pruned_unit_ids) - set(current.pruned_unit_ids)
        )
        missing = sorted(set(newly_removed) - set(self.unit_terms))
        if missing:
            raise RuntimeError(f"structural_gate_action_mapping_missing:{missing}")
        first = sum(self.unit_terms[unit]["first"] for unit in newly_removed)
        second = sum(self.unit_terms[unit]["second"] for unit in newly_removed)
        total = first + second
        if total < 0.0 or not torch.isfinite(torch.tensor(total)):
            raise RuntimeError("structural_gate_action_score_invalid")
        return {
            "delta_J_struct": total,
            "delta_J_prune": total,
            "delta_J_WQ": 0.0,
            "delta_J_AQ": 0.0,
            "first_order_abs_sum": first,
            "second_order_abs_sum": second,
            "newly_removed_unit_ids": newly_removed,
            "newly_removed_functional_coordinate_count": len(newly_removed),
            "risk_refund": 0.0,
            "formula": "sum_newly_removed(abs(g*(-u))+0.5*abs(g^2*u^2))",
            "elementwise_abs_before_reduction": True,
        }


class ActivationActionTaylorProxy:
    """Cache activation Q/DQ transition costs for mutable precision genes."""

    def __init__(
        self,
        *,
        statistics: JointOutputTaylorStatistics | None = None,
        units: Sequence[TaylorDeploymentUnit],
        gene_to_unit_ids: Mapping[str, Sequence[str]],
        precomputed_transition_terms: Mapping[
            tuple[str, str, str], tuple[float, float, int]
        ]
        | None = None,
    ) -> None:
        self.statistics = statistics
        self.units = {unit.unit_id: unit for unit in units}
        self.gene_to_unit_ids = {
            str(gene): tuple(str(unit) for unit in unit_ids)
            for gene, unit_ids in gene_to_unit_ids.items()
        }
        expected = {unit for values in self.gene_to_unit_ids.values() for unit in values}
        missing_units = sorted(expected - set(self.units))
        missing_stats = (
            sorted(expected - set(statistics.baseline_outputs))
            if statistics is not None
            else []
        )
        if missing_units or missing_stats:
            raise RuntimeError(
                f"activation_taylor_mapping_incomplete:units={missing_units}:stats={missing_stats}"
            )
        self._cache: dict[tuple[str, str, str], tuple[float, float, int]] = {
            (str(unit_id), str(current), str(successor)): (
                float(values[0]),
                float(values[1]),
                int(values[2]),
            )
            for (unit_id, current, successor), values in dict(
                precomputed_transition_terms or {}
            ).items()
        }
        if statistics is None and not self._cache:
            raise ValueError("activation_taylor_requires_statistics_or_precomputed_terms")

    def _transition_terms(
        self, unit_id: str, current_precision: str, next_precision: str
    ) -> tuple[float, float, int]:
        key = (unit_id, str(current_precision), str(next_precision))
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        if self.statistics is None:
            raise RuntimeError(
                f"activation_taylor_precomputed_transition_missing:{unit_id}:"
                f"{current_precision}:{next_precision}"
            )
        unit = self.units[unit_id]
        first_total = 0.0
        second_total = 0.0
        count = 0
        outputs = self.statistics.baseline_outputs[unit_id]
        gradients = self.statistics.gradients[unit_id]
        if len(outputs) != len(gradients) or not outputs:
            raise RuntimeError(f"activation_taylor_sample_count_invalid:{unit_id}")
        for value, gradient in zip(outputs, gradients):
            current = pseudo_quantize_activation(
                value, current_precision, channel_axis=unit.activation_channel_axis
            )
            successor = pseudo_quantize_activation(
                value, next_precision, channel_axis=unit.activation_channel_axis
            )
            delta = successor - current
            first, second = activation_taylor_terms(value, gradient, delta)
            first_total += float(first.sum())
            second_total += float(second.sum())
            count += int(value.numel())
        cached = (first_total, second_total, count)
        self._cache[key] = cached
        return cached

    def quantization_action_breakdown(
        self,
        current: CandidatePhenotype,
        successor: CandidatePhenotype,
        *,
        changed_gene_id: str,
    ) -> dict[str, Any]:
        unit_ids = self.gene_to_unit_ids.get(str(changed_gene_id))
        if not unit_ids:
            raise RuntimeError(
                f"activation_taylor_precision_gene_mapping_missing:{changed_gene_id}"
            )
        first_total = 0.0
        second_total = 0.0
        element_count = 0
        transitions: list[dict[str, str]] = []
        for unit_id in unit_ids:
            unit = self.units[unit_id]
            owner = unit.precision_owner
            current_precision = current.realized_precision_profile.get(owner, "FP32")
            next_precision = successor.realized_precision_profile.get(owner, "FP32")
            if current_precision == next_precision:
                raise RuntimeError(
                    f"activation_taylor_precision_owner_unchanged:{changed_gene_id}:{owner}"
                )
            first, second, count = self._transition_terms(
                unit_id, current_precision, next_precision
            )
            first_total += first
            second_total += second
            element_count += count
            transitions.append(
                {
                    "unit_id": unit_id,
                    "precision_owner": owner,
                    "current": current_precision,
                    "next": next_precision,
                }
            )
        total = first_total + second_total
        if total < 0.0:
            raise RuntimeError("activation_taylor_action_negative")
        return {
            "delta_J_AQ": total,
            "first_order_abs_sum": first_total,
            "second_order_abs_sum": second_total,
            "activation_element_count": element_count,
            "transitions": transitions,
            "risk_refund": 0.0,
            "formula": "sum(abs(g*(DQ(Q(A_next))-A_current))+0.5*abs(g^2*delta_A^2))",
            "elementwise_abs_before_reduction": True,
            "cross_parameter_signed_cancellation": False,
            "cross_sample_signed_cancellation": False,
        }


def _first_tensor(value: Any) -> torch.Tensor | None:
    if torch.is_tensor(value):
        return value
    if isinstance(value, Mapping):
        for item in value.values():
            found = _first_tensor(item)
            if found is not None:
                return found
    if isinstance(value, (tuple, list)):
        for item in value:
            found = _first_tensor(item)
            if found is not None:
                return found
    return None


def _semantic_cnn_output_path(model: nn.Module, path: str) -> str:
    modules = dict(model.named_modules())
    if "." not in path:
        return path
    parent_path, leaf = path.rsplit(".", 1)
    parent = modules.get(parent_path)
    if not isinstance(parent, nn.Sequential) or not leaf.isdigit():
        return path
    selected = path
    for index in range(int(leaf) + 1, len(parent)):
        child = parent[index]
        if isinstance(child, (nn.modules.batchnorm._BatchNorm, nn.ReLU)):
            selected = f"{parent_path}.{index}"
            continue
        break
    return selected


def _coordinate_terms(
    value: torch.Tensor, gradient: torch.Tensor, *, axis: int
) -> tuple[torch.Tensor, torch.Tensor]:
    first, second = activation_taylor_terms(value, gradient, -value)
    resolved = axis if axis >= 0 else value.ndim + axis
    reduce = tuple(index for index in range(value.ndim) if index != resolved)
    if reduce:
        first = first.sum(dim=reduce)
        second = second.sum(dim=reduce)
    return first, second


def collect_structural_gate_statistics(
    model: nn.Module,
    domains: Sequence[LocalPruningDomain],
    batch: Any,
    *,
    forward_fn: Callable[[nn.Module, Any], Any],
    loss_fn: Callable[[Any, Any], torch.Tensor],
) -> tuple[StructuralGateTaylorProxy, dict[str, Any]]:
    """Collect sample-mean functional gate scores for tracer-defined units.

    A mapping is one model batch. A list/tuple is treated as a fixed calibration
    sequence. Coordinate vectors are reduced immediately after every backward,
    so activation graphs are never retained across samples.
    """

    calibration_batches = (
        tuple(batch) if isinstance(batch, (tuple, list)) else (batch,)
    )
    if not calibration_batches:
        raise ValueError("structural_gate_calibration_batches_empty")

    modules = dict(model.named_modules())
    path_roles: dict[str, list[tuple[LocalPruningDomain, str]]] = {}
    domain_units: dict[str, tuple[str, ...]] = {}
    for domain in domains:
        domain_units[domain.domain_id] = tuple(domain.ordered_unit_ids)
        if domain.domain_type in {"cnn_channel", "grouped_conv_channel"}:
            path = _semantic_cnn_output_path(model, domain.root_module_path)
            path_roles.setdefault(path, []).append((domain, "cnn_output"))
        elif domain.domain_type == "ffn_hidden":
            path = str(domain.metadata.get("activation_path", ""))
            if not path or path not in modules:
                raise RuntimeError(
                    f"structural_gate_ffn_activation_mapping_missing:{domain.domain_id}:{path}"
                )
            path_roles.setdefault(path, []).append((domain, "ffn_hidden"))
        elif domain.domain_type == "attention_dh":
            for member in domain.dependency_members:
                role = str(member.get("role", ""))
                if role not in {"q", "k", "v"}:
                    continue
                path = str(member.get("module_path", ""))
                if path not in modules:
                    raise RuntimeError(
                        f"structural_gate_attention_projection_missing:{domain.domain_id}:{path}"
                    )
                path_roles.setdefault(path, []).append((domain, role))
        else:
            raise RuntimeError(
                f"structural_gate_domain_type_unsupported:{domain.domain_id}:{domain.domain_type}"
            )

    domain_role_terms: dict[tuple[str, str], tuple[torch.Tensor, torch.Tensor]] = {}
    capture_summary: dict[tuple[str, str, str], dict[str, Any]] = {}
    losses: list[float] = []
    for sample_index, calibration_batch in enumerate(calibration_batches):
        captured: dict[str, list[torch.Tensor]] = {path: [] for path in path_roles}
        handles = []
        for path in sorted(path_roles):
            module = modules[path]

            def hook(
                _module: nn.Module,
                _inputs: tuple[Any, ...],
                output: Any,
                *,
                _path: str = path,
            ) -> Any:
                tensor = _first_tensor(output)
                if tensor is None:
                    raise RuntimeError(f"structural_gate_tensor_missing:{_path}")
                if tensor.requires_grad:
                    tensor.retain_grad()
                captured[_path].append(tensor)
                return output

            handles.append(module.register_forward_hook(hook))
        model.zero_grad(set_to_none=True)
        try:
            result = forward_fn(model, calibration_batch)
            loss = loss_fn(result, calibration_batch)
            if loss.ndim != 0 or not bool(torch.isfinite(loss)):
                raise RuntimeError("structural_gate_task_loss_invalid")
            loss.backward()
            losses.append(float(loss.detach().cpu()))

            for path, roles in path_roles.items():
                tensors = captured[path]
                if not tensors:
                    raise RuntimeError(f"structural_gate_capture_missing:{path}")
                unique_roles = {
                    (domain.domain_id, role) for domain, role in roles
                }
                for domain_id, role in sorted(unique_roles):
                    domain = next(
                        domain
                        for domain, candidate_role in roles
                        if domain.domain_id == domain_id and candidate_role == role
                    )
                    first_sum: torch.Tensor | None = None
                    second_sum: torch.Tensor | None = None
                    for tensor in tensors:
                        if tensor.grad is None:
                            raise RuntimeError(
                                f"structural_gate_gradient_missing:{path}:sample{sample_index}"
                            )
                        value = tensor
                        gradient = tensor.grad
                        if role == "cnn_output":
                            axis = 1 if value.ndim >= 2 else -1
                        else:
                            axis = -1
                            if domain.domain_type == "attention_dh":
                                total = int(domain.groups) * int(domain.original_width)
                                role_order = {
                                    candidate_role
                                    for candidate_domain, candidate_role in roles
                                    if candidate_domain.domain_id == domain_id
                                }
                                if (
                                    value.shape[-1] == 3 * total
                                    and role_order >= {"q", "k", "v"}
                                ):
                                    offset = {"q": 0, "k": 1, "v": 2}[role] * total
                                    value = value[..., offset : offset + total]
                                    gradient = gradient[..., offset : offset + total]
                                elif value.shape[-1] != total:
                                    raise RuntimeError(
                                        "structural_gate_attention_width_mismatch:"
                                        f"{domain_id}:{path}:{value.shape[-1]}:{total}:{role}"
                                    )
                        first, second = _coordinate_terms(value, gradient, axis=axis)
                        first_sum = first if first_sum is None else first_sum + first
                        second_sum = second if second_sum is None else second_sum + second
                    assert first_sum is not None and second_sum is not None
                    if domain.domain_type == "attention_dh":
                        first_sum = first_sum.reshape(
                            domain.groups, domain.original_width
                        )
                        second_sum = second_sum.reshape(
                            domain.groups, domain.original_width
                        )
                    key = (domain_id, role)
                    first_cpu = first_sum.detach().cpu()
                    second_cpu = second_sum.detach().cpu()
                    if key in domain_role_terms:
                        previous_first, previous_second = domain_role_terms[key]
                        domain_role_terms[key] = (
                            previous_first + first_cpu,
                            previous_second + second_cpu,
                        )
                    else:
                        domain_role_terms[key] = (first_cpu, second_cpu)
                    summary_key = (domain_id, role, path)
                    summary = capture_summary.setdefault(
                        summary_key,
                        {
                            "domain_id": domain_id,
                            "domain_type": domain.domain_type,
                            "semantic_root_tensor": path,
                            "gate_tensor": path,
                            "gate_role": role,
                            "capture_count": 0,
                            "sample_count": 0,
                            "shapes": [],
                        },
                    )
                    summary["capture_count"] += len(tensors)
                    summary["sample_count"] += 1
                    shape = list(tensors[0].shape)
                    if shape not in summary["shapes"]:
                        summary["shapes"].append(shape)
        finally:
            for handle in handles:
                handle.remove()
            model.zero_grad(set_to_none=True)
            captured.clear()

    sample_count = len(calibration_batches)
    domain_role_terms = {
        key: (first / sample_count, second / sample_count)
        for key, (first, second) in domain_role_terms.items()
    }
    capture_rows = list(capture_summary.values())

    unit_terms: dict[str, dict[str, float]] = {}
    mapping_rows: list[dict[str, Any]] = []
    attention_pattern = re.compile(r"::head(?P<head>\d+)::(?P<role>qk|vo)::(?P<local>\d+)$")
    for domain in domains:
        for unit_id in domain.ordered_unit_ids:
            if domain.domain_type in {"cnn_channel", "grouped_conv_channel"}:
                first, second = domain_role_terms[(domain.domain_id, "cnn_output")]
                indices = domain.unit_root_indices.get(unit_id)
                if not indices:
                    raise RuntimeError(f"structural_gate_cnn_unit_index_missing:{unit_id}")
                first_value = float(first[list(indices)].sum())
                second_value = float(second[list(indices)].sum())
                coordinate = list(indices)
                role = "cnn_output"
            elif domain.domain_type == "ffn_hidden":
                first, second = domain_role_terms[(domain.domain_id, "ffn_hidden")]
                index = int(unit_id.rsplit("::", 1)[1])
                first_value = float(first[index])
                second_value = float(second[index])
                coordinate = index
                role = "ffn_hidden"
            else:
                matched = attention_pattern.search(unit_id)
                if matched is None:
                    raise RuntimeError(f"structural_gate_attention_unit_parse:{unit_id}")
                head = int(matched.group("head"))
                local = int(matched.group("local"))
                semantic_role = matched.group("role")
                source_roles = ("q", "k") if semantic_role == "qk" else ("v",)
                first_value = 0.0
                second_value = 0.0
                for source_role in source_roles:
                    first, second = domain_role_terms[(domain.domain_id, source_role)]
                    first_value += float(first[head, local])
                    second_value += float(second[head, local])
                coordinate = {"head": head, "local": local}
                role = semantic_role
            unit_terms[unit_id] = {"first": first_value, "second": second_value}
            mapping_rows.append(
                {
                    "structural_domain": domain.domain_id,
                    "tracer_group_id": domain.scope_id,
                    "unit_id": unit_id,
                    "gate_role": role,
                    "coordinate": coordinate,
                    "physical_dependent_parameters": [
                        dict(member) for member in domain.dependency_members
                    ],
                    "gate_first_order": first_value,
                    "gate_second_order": second_value,
                    "gate_score": first_value + second_value,
                    "legacy_coupled_weight_taylor_used_for_fitness": False,
                }
            )
    model.zero_grad(set_to_none=True)
    proxy = StructuralGateTaylorProxy(
        unit_terms=unit_terms,
        domain_units=domain_units,
        mapping_rows=mapping_rows,
    )
    return proxy, {
        "schema_version": "v2xvit-functional-gate-taylor-v2",
        "task_losses": losses,
        "sample_count": sample_count,
        "sample_reduction": "mean_of_per_sample_elementwise_abs_sums",
        "elementwise_abs_before_reduction": True,
        "domain_count": len(domains),
        "unit_count": len(unit_terms),
        "capture_rows": capture_rows,
        "mapping_rows": mapping_rows,
    }


def build_activation_taylor_units(
    model: nn.Module,
    groups: Sequence[QuantizationSearchGroup],
    *,
    mutable_gene_ids: Sequence[str],
    transformer_precision_units: Sequence[Any],
) -> tuple[tuple[TaylorDeploymentUnit, ...], dict[str, tuple[str, ...]], list[dict[str, Any]]]:
    """Map mutable precision genes to their actual deployment Q/DQ tensors."""

    mutable = set(str(value) for value in mutable_gene_ids)
    precision_by_id = {str(unit.unit_id): unit for unit in transformer_precision_units}
    units: list[TaylorDeploymentUnit] = []
    gene_to_units: dict[str, list[str]] = {}
    mapping_rows: list[dict[str, Any]] = []
    for group in sorted(groups, key=lambda value: value.ordering):
        if group.group_id not in mutable:
            continue
        group_units: list[TaylorDeploymentUnit] = []
        if bool(group.metadata.get("activation_only", False)):
            precision_unit = precision_by_id.get(group.group_id)
            if precision_unit is None:
                raise RuntimeError(
                    f"activation_taylor_functional_contract_missing:{group.group_id}"
                )
            group_units.extend(
                replace(
                    unit,
                    metadata={**unit.metadata, "precision_gene_id": group.group_id},
                )
                for unit in taylor_units_from_transformer_precision(
                    model, (precision_unit,)
                )
            )
        else:
            for index, path in enumerate(group.module_paths):
                try:
                    model.get_submodule(path)
                except AttributeError as exc:
                    raise RuntimeError(
                        f"activation_taylor_weighted_input_missing:{group.group_id}:{path}"
                    ) from exc
                group_units.append(
                    TaylorDeploymentUnit(
                        unit_id=f"activation_input::{group.group_id}::path{index}",
                        module_path=path,
                        unit_type=str(group.metadata.get("transformer_role", "weighted_input")),
                        boundary="module_input",
                        precision_owner=path,
                        quantizer_id=f"activation_input_quantizer::{group.group_id}::{path}",
                        metadata={
                            **group.metadata,
                            "precision_gene_id": group.group_id,
                            "source_precision_path": path,
                        },
                    )
                )
        if not group_units:
            raise RuntimeError(f"activation_taylor_gene_has_no_boundary:{group.group_id}")
        units.extend(group_units)
        gene_to_units[group.group_id] = [unit.unit_id for unit in group_units]
        mapping_rows.extend(
            {
                "precision_gene_id": group.group_id,
                "unit_id": unit.unit_id,
                "module_path": unit.module_path,
                "boundary": unit.boundary,
                "functional_op": unit.functional_op,
                "precision_owner": unit.precision_owner,
                "quantizer_id": unit.quantizer_id,
            }
            for unit in group_units
        )
    return (
        tuple(units),
        {key: tuple(value) for key, value in gene_to_units.items()},
        mapping_rows,
    )


def collect_activation_action_statistics(
    model: nn.Module,
    batch: Any,
    *,
    forward_fn: Callable[[nn.Module, Any], Any],
    loss_fn: Callable[[Any, Any], torch.Tensor],
    units: Sequence[TaylorDeploymentUnit],
    gene_to_unit_ids: Mapping[str, Sequence[str]],
    calibration_manifest_hash: str,
) -> tuple[ActivationActionTaylorProxy, dict[str, Any]]:
    statistics = collect_joint_output_taylor_statistics(
        model,
        (batch,),
        forward_fn=forward_fn,
        loss_fn=loss_fn,
        units=units,
        calibration_manifest_hash=calibration_manifest_hash,
    )
    proxy = ActivationActionTaylorProxy(
        statistics=statistics,
        units=units,
        gene_to_unit_ids=gene_to_unit_ids,
    )
    return proxy, {
        **statistics.to_manifest(),
        "formula": "sum(abs(g*delta_A)+0.5*abs(g^2*delta_A^2))",
        "elementwise_abs_before_reduction": True,
        "joint_taylor_used_for_fitness": False,
        "cross_residual_used_for_fitness": False,
    }


def collect_streaming_activation_action_statistics(
    model: nn.Module,
    calibration_batches: Sequence[Any],
    *,
    forward_fn: Callable[[nn.Module, Any], Any],
    loss_fn: Callable[[Any, Any], torch.Tensor],
    units: Sequence[TaylorDeploymentUnit],
    gene_to_unit_ids: Mapping[str, Sequence[str]],
    precision_ladders: Mapping[str, Sequence[str]],
    calibration_manifest_hash: str,
) -> tuple[ActivationActionTaylorProxy, dict[str, Any]]:
    """Precompute adjacent Q/DQ action costs without persisting activations."""

    if not calibration_batches:
        raise ValueError("activation_taylor_calibration_batches_empty")
    if not calibration_manifest_hash:
        raise ValueError("activation_taylor_calibration_manifest_hash_missing")
    selected, aliases = coalesce_deployment_units(units)
    selected_by_quantizer = {unit.quantizer_id: unit for unit in selected}
    alias_to_selected: dict[str, str] = {}
    for quantizer_id, alias_ids in aliases.items():
        selected_id = selected_by_quantizer[quantizer_id].unit_id
        alias_to_selected.update({str(alias): selected_id for alias in alias_ids})
    normalized_gene_units: dict[str, tuple[str, ...]] = {}
    for gene_id, unit_ids in gene_to_unit_ids.items():
        normalized = tuple(
            dict.fromkeys(alias_to_selected.get(str(unit_id), str(unit_id)) for unit_id in unit_ids)
        )
        missing = sorted(set(normalized) - {unit.unit_id for unit in selected})
        if missing:
            raise RuntimeError(
                f"activation_taylor_streaming_mapping_missing:{gene_id}:{missing}"
            )
        normalized_gene_units[str(gene_id)] = normalized

    transitions: dict[str, tuple[tuple[str, str], ...]] = {}
    for gene_id in normalized_gene_units:
        ladder = tuple(str(value).upper() for value in precision_ladders[gene_id])
        if len(ladder) < 2:
            raise RuntimeError(f"activation_taylor_ladder_not_mutable:{gene_id}:{ladder}")
        transitions[gene_id] = tuple(zip(ladder[:-1], ladder[1:]))

    accumulators: dict[tuple[str, str, str], list[float | int]] = {}
    shapes: dict[str, list[list[int]]] = {unit.unit_id: [] for unit in selected}
    losses: list[float] = []
    model.train(False)
    for sample_index, batch in enumerate(calibration_batches):
        model.zero_grad(set_to_none=True)
        with _BoundaryCapture(
            model, selected, quantize=False, retain_grad=True
        ) as capture:
            result = forward_fn(model, batch)
            loss = loss_fn(result, batch)
            if loss.ndim != 0 or not bool(torch.isfinite(loss)):
                raise RuntimeError("activation_taylor_task_loss_invalid")
            loss.backward()
            capture.require_one_per_unit()
            losses.append(float(loss.detach().cpu()))
            for gene_id, unit_ids in normalized_gene_units.items():
                for current_precision, next_precision in transitions[gene_id]:
                    for unit_id in unit_ids:
                        unit = next(row for row in selected if row.unit_id == unit_id)
                        value = capture.outputs[unit_id][0]
                        gradient = value.grad
                        if gradient is None:
                            raise RuntimeError(
                                f"activation_taylor_gradient_missing:{unit_id}:sample{sample_index}"
                            )
                        shape = list(value.shape)
                        if shape not in shapes[unit_id]:
                            shapes[unit_id].append(shape)
                        current = pseudo_quantize_activation(
                            value,
                            current_precision,
                            channel_axis=unit.activation_channel_axis,
                        )
                        successor = pseudo_quantize_activation(
                            value,
                            next_precision,
                            channel_axis=unit.activation_channel_axis,
                        )
                        first, second = activation_taylor_terms(
                            value, gradient, successor - current
                        )
                        key = (unit_id, current_precision, next_precision)
                        row = accumulators.setdefault(key, [0.0, 0.0, 0])
                        row[0] = float(row[0]) + float(first.sum())
                        row[1] = float(row[1]) + float(second.sum())
                        row[2] = int(row[2]) + int(value.numel())
        model.zero_grad(set_to_none=True)

    sample_count = len(calibration_batches)
    precomputed = {
        key: (
            float(values[0]) / sample_count,
            float(values[1]) / sample_count,
            int(values[2]),
        )
        for key, values in accumulators.items()
    }
    proxy = ActivationActionTaylorProxy(
        statistics=None,
        units=selected,
        gene_to_unit_ids=normalized_gene_units,
        precomputed_transition_terms=precomputed,
    )
    transition_rows = [
        {
            "unit_id": unit_id,
            "current_precision": current,
            "next_precision": successor,
            "first_order_abs_sample_mean": values[0],
            "second_order_abs_sample_mean": values[1],
            "element_count_all_samples": values[2],
        }
        for (unit_id, current, successor), values in sorted(precomputed.items())
    ]
    return proxy, {
        "schema_version": "v2xvit-streaming-activation-action-taylor-v2",
        "calibration_manifest_hash": str(calibration_manifest_hash),
        "unit_manifest_hash": _stable_hash([unit.to_dict() for unit in selected]),
        "sample_count": sample_count,
        "task_losses": losses,
        "sample_reduction": "mean_of_per_sample_elementwise_abs_sums",
        "statistics_tensors_persisted": False,
        "units": shapes,
        "transitions": transition_rows,
        "formula": "mean_samples(sum(abs(g*delta_A)+0.5*abs(g^2*delta_A^2)))",
        "elementwise_abs_before_reduction": True,
        "joint_taylor_used_for_fitness": False,
        "cross_residual_used_for_fitness": False,
    }


__all__ = [
    "ActivationActionTaylorProxy",
    "StructuralGateTaylorProxy",
    "activation_taylor_terms",
    "build_activation_taylor_units",
    "collect_activation_action_statistics",
    "collect_streaming_activation_action_statistics",
    "collect_structural_gate_statistics",
]
