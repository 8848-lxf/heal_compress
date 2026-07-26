"""Cached functional-gate and activation Taylor costs for unified Greedy.

The collector is intentionally run once before the search loop.  Search actions
only read the resulting non-negative, elementwise-absolute caches; they never
execute a model or export a candidate.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn

from ..candidate import CandidatePhenotype
from ..canonicalization import SearchSpaceSpec
from .candidate_perturbation import pseudo_quantize_activation
from .joint_weight_activation_taylor import (
    TaylorDeploymentUnit,
    _BoundaryCapture,
    coalesce_deployment_units,
    taylor_units_from_transformer_precision,
)


def _score(value: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
    # Model activations/gradients are FP32, but squaring both operands before
    # multiplying can overflow FP32 even when the mathematical Taylor score is
    # finite.  Accumulate the audit/cache statistic in FP64; this does not alter
    # the model forward, backward, quantizer, or elementwise-abs semantics.
    value = value.detach().double()
    grad = grad.detach().double()
    if not bool(torch.isfinite(value).all()) or not bool(torch.isfinite(grad).all()):
        raise RuntimeError("gate_taylor_input_nonfinite")
    first = (grad * (-value)).abs()
    second = 0.5 * (grad.square() * value.square()).abs()
    result = first + second
    if not bool(torch.isfinite(result).all()) or bool((result < 0).any()):
        raise RuntimeError("gate_taylor_element_score_invalid")
    return result


@dataclass(frozen=True)
class GateDomainScores:
    domain_id: str
    unit_scores: Mapping[str, float]
    semantic_root_tensor: str
    gate_tensor: str
    physical_dependencies: tuple[str, ...]
    family: str
    sample_count: int = 1
    per_sample_unit_scores: tuple[Mapping[str, float], ...] = ()


class FunctionalGateTaylorProxy:
    """Structural action risk from functional output coordinates only."""

    def __init__(self, scores: Mapping[str, GateDomainScores]) -> None:
        self.scores = dict(scores)

    def pruning_action_breakdown(self, current: CandidatePhenotype, successor: CandidatePhenotype) -> dict[str, Any]:
        total = 0.0
        count = 0
        domains: dict[str, float] = {}
        for domain_id, row in self.scores.items():
            before = set(current.pruned_unit_ids)
            after = set(successor.pruned_unit_ids)
            units = sorted((after - before) & set(row.unit_scores))
            value = sum(float(row.unit_scores[u]) for u in units)
            total += value
            count += len(units)
            if units:
                domains[domain_id] = value
        if total < 0.0 or not torch.isfinite(torch.tensor(total)):
            raise RuntimeError("gate_taylor_action_negative")
        return {
            "delta_J_prune": float(total), "delta_J_WQ": 0.0,
            "first_order_abs_sum": float(total), "second_order_abs_sum": 0.0,
            "newly_pruned_parameter_count": int(count), "risk_refund": 0.0,
            "formula": "sum_newly_removed(abs(g_gate*u_gate)+0.5*abs(h_gate*u_gate^2))",
            "proxy": "functional_gate_output_taylor", "domain_breakdown": domains,
            "elementwise_abs_before_reduction": True,
        }


def rerank_domains_by_gate_scores(domains: Sequence[Any], scores: Mapping[str, GateDomainScores]) -> tuple[Any, ...]:
    """Replace only the fixed nested ranking; tracer dependencies stay intact."""
    result = []
    for domain in domains:
        row = scores.get(str(domain.domain_id))
        if row is None:
            raise RuntimeError(f"gate_mapping_missing_domain:{domain.domain_id}")
        values = {str(k): float(v) for k, v in row.unit_scores.items()}
        if domain.domain_type == "attention_dh":
            by_role_head: dict[tuple[str, int], list[str]] = {}
            for unit in domain.ordered_unit_ids:
                parts = str(unit).split("::")
                role = next((p for p in parts if p in {"qk", "vo"}), "qk")
                head = int(next(p[4:] for p in parts if p.startswith("head")))
                by_role_head.setdefault((role, head), []).append(str(unit))
            for key in by_role_head:
                by_role_head[key].sort(key=lambda u: (values.get(u, 0.0), u))
            ordered = []
            for role in ("qk", "vo"):
                for head in range(int(domain.groups)):
                    ordered.extend(by_role_head.get((role, head), []))
            # Keep per-width closure exactly as the tracer encoded it, while
            # changing the nested order inside each head.
            width_map = {int(domain.original_width): ()}
            for width in sorted(int(w) for w in domain.legal_widths if int(w) != int(domain.original_width)):
                count = int(domain.original_width) - int(width)
                selected = []
                for head in range(int(domain.groups)):
                    selected.extend(by_role_head.get(("qk", head), [])[:count])
                    selected.extend(by_role_head.get(("vo", head), [])[:count])
                width_map[width] = tuple(selected)
        else:
            ordered = sorted((str(u) for u in domain.ordered_unit_ids), key=lambda u: (values.get(u, 0.0), u))
            width_map = {int(domain.original_width): ()}
            for width in sorted(int(w) for w in domain.legal_widths if int(w) != int(domain.original_width)):
                target = len(domain.width_to_pruned_unit_ids.get(width, ()))
                width_map[width] = tuple(ordered[:target])
        result.append(replace(domain, ordered_unit_ids=tuple(ordered), width_to_pruned_unit_ids=width_map, unit_scores=values, ranking_method="functional_gate_output_taylor_abs_sum"))
    return tuple(result)


@dataclass(frozen=True)
class ActivationTaylorCache:
    group_to_units: Mapping[str, tuple[str, ...]]
    # (unit_id, current_internal, next_internal) -> score
    transitions: Mapping[tuple[str, str, str], float]
    mapping: tuple[dict[str, Any], ...]
    unit_to_owner: Mapping[str, str] = None
    sample_count: int = 1
    per_sample_transitions: tuple[Mapping[tuple[str, str, str], float], ...] = ()

    def action_breakdown(self, current: CandidatePhenotype, successor: CandidatePhenotype) -> dict[str, Any]:
        total = 0.0; first = 0.0; second = 0.0; changed: list[str] = []
        for group, units in self.group_to_units.items():
            before = {u: str(current.realized_precision_profile.get((self.unit_to_owner or {}).get(u, u), "FP32")) for u in units}
            after = {u: str(successor.realized_precision_profile.get((self.unit_to_owner or {}).get(u, u), "FP32")) for u in units}
            for unit in units:
                if before[unit] == after[unit]:
                    continue
                key = (unit, before[unit], after[unit])
                value = self.transitions.get(key)
                if value is None:
                    # Search contracts only permit adjacent transitions.  A
                    # missing cache is a hard error, never a zero fallback.
                    raise RuntimeError(f"activation_taylor_transition_missing:{key}")
                total += float(value); changed.append(group)
        return {"delta_J_AQ": float(total), "activation_taylor_units": sorted(set(changed)),
                "formula": "sum_elementwise(abs(g_A*delta_A)+0.5*abs(h_A*delta_A^2))",
                "elementwise_abs_before_reduction": True}


def _axis_reduce(score: torch.Tensor, axis: int) -> torch.Tensor:
    axis = axis if axis >= 0 else score.ndim + axis
    dims = tuple(i for i in range(score.ndim) if i != axis)
    return score.sum(dim=dims) if dims else score


def collect_functional_gate_scores(
    model: nn.Module,
    domains: Sequence[Any],
    *,
    forward_fn: Any,
    loss_fn: Any,
    batch: Any,
) -> tuple[dict[str, GateDomainScores], list[dict[str, Any]]]:
    """Collect one fixed task-loss gate statistic and map it to tracer units."""
    modules = dict(model.named_modules())
    def member_value(member: Any, key: str, default: Any = "") -> Any:
        return member.get(key, default) if isinstance(member, Mapping) else getattr(member, key, default)
    wanted: dict[str, list[tuple[Any, str]]] = {}
    for domain in domains:
        if domain.domain_type in {"cnn_channel", "grouped_conv_channel"}:
            path = domain.module_path
            try:
                from search.model_family.deployment import _semantic_output_module_path
                path = _semantic_output_module_path(model, path)
            except Exception:
                pass
            wanted.setdefault(path, []).append((domain, "cnn"))
        elif domain.domain_type == "ffn_hidden":
            members = list(domain.dependency_members)
            path = next((str(member_value(m, "module_path")) for m in members if str(member_value(m, "role")) in {"first", "gate", "up"}), "")
            if not path:
                raise RuntimeError(f"gate_mapping_missing:{domain.domain_id}")
            try:
                from search.model_family.deployment import _semantic_output_module_path
                path = _semantic_output_module_path(model, path)
            except Exception:
                pass
            wanted.setdefault(path, []).append((domain, "ffn"))
        elif domain.domain_type == "attention_dh":
            for member in domain.dependency_members:
                role = str(member_value(member, "role"))
                if role in {"q", "k", "v"}:
                    wanted.setdefault(str(member_value(member, "module_path")), []).append((domain, role))
        else:
            raise RuntimeError(f"gate_mapping_unsupported_domain:{domain.domain_id}:{domain.domain_type}")
    missing = sorted(path for path in wanted if path not in modules)
    if missing:
        raise RuntimeError(f"gate_mapping_module_missing:{missing}")
    captures: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {p: [] for p in wanted}
    handles = []
    def hook(path: str):
        def fn(_module: nn.Module, _inputs: Any, output: Any) -> Any:
            value = output[0] if isinstance(output, (tuple, list)) else output
            if not torch.is_tensor(value):
                raise RuntimeError(f"gate_mapping_output_not_tensor:{path}")
            if value.requires_grad:
                value.retain_grad()
            captures[path].append((value, value))
            return output
        return fn
    for path in wanted:
        handles.append(modules[path].register_forward_hook(hook(path)))
    try:
        model.zero_grad(set_to_none=True)
        result = forward_fn(model, batch)
        loss = loss_fn(result, batch)
        if loss.ndim != 0 or not torch.isfinite(loss):
            raise RuntimeError("gate_taylor_task_loss_invalid")
        # Replace the second tuple entry by the gradient after backward.
        loss.backward()
        scores: dict[str, dict[str, float]] = {str(d.domain_id): {} for d in domains}
        mapping: list[dict[str, Any]] = []
        for path, rows in wanted.items():
            for value, _ in rows and captures[path]:
                grad = value.grad
                if grad is None:
                    continue
                local = _score(value, grad)
                for domain, role in wanted[path]:
                    if domain.domain_type in {"cnn_channel", "grouped_conv_channel"}:
                        vector = _axis_reduce(local, 1 if local.ndim >= 2 else 0).reshape(-1)
                        for unit, indices in domain.unit_root_indices.items():
                            scores[domain.domain_id][str(unit)] = scores[domain.domain_id].get(str(unit), 0.0) + float(vector[list(indices)].sum())
                        semantic = path
                    elif domain.domain_type == "ffn_hidden":
                        vector = _axis_reduce(local, local.ndim - 1).reshape(-1)
                        for unit, indices in domain.unit_root_indices.items():
                            scores[domain.domain_id][str(unit)] = scores[domain.domain_id].get(str(unit), 0.0) + float(vector[list(indices)].sum())
                        semantic = path
                    else:
                        heads = int(domain.constraints.get("heads", 1)); width = int(domain.original_width)
                        vector = _axis_reduce(local, local.ndim - 1).reshape(-1)
                        if vector.numel() < heads * width:
                            raise RuntimeError(f"gate_mapping_attention_shape:{domain.domain_id}:{tuple(value.shape)}")
                        # fused QKV carries three contiguous blocks; separate
                        # projections carry one H*d_h block.
                        if vector.numel() >= 3 * heads * width:
                            vector = vector[: 3 * heads * width].reshape(3, heads, width)[{"q": 0, "k": 1, "v": 2}[role]]
                        else:
                            vector = vector[: heads * width].reshape(heads, width)
                        for unit in domain.ordered_unit_ids:
                            text = str(unit)
                            parts = text.split("::")
                            head = int(next(p[4:] for p in parts if p.startswith("head")))
                            local_index = int(parts[-1])
                            role_token = next((p for p in parts if p in {"qk", "vo"}), "qk")
                            if (role in {"q", "k"} and role_token != "qk") or (role == "v" and role_token != "vo"):
                                continue
                            # q/k are one functional coordinate; v is the VO coordinate.
                            scores[domain.domain_id][text] = scores[domain.domain_id].get(text, 0.0) + float(vector[head, local_index])
                        semantic = path + f"::{role}"
                    mapping.append({"domain_id": str(domain.domain_id), "tracer_group_id": str(domain.domain_id), "semantic_root_tensor": semantic, "gate_tensor": semantic, "physical_dependencies": [str(member_value(m, "module_path")) for m in domain.dependency_members], "family": str(domain.family)})
    finally:
        for handle in handles: handle.remove()
        model.zero_grad(set_to_none=True)
    out = {str(d.domain_id): GateDomainScores(str(d.domain_id), scores[str(d.domain_id)], str(d.module_path), str(d.module_path), tuple(str(member_value(m, "module_path")) for m in d.dependency_members), str(d.family)) for d in domains}
    return out, mapping


def build_activation_units(model: nn.Module, space: SearchSpaceSpec, transformer_units: Sequence[Any]) -> tuple[tuple[TaylorDeploymentUnit, ...], dict[str, tuple[str, ...]]]:
    # Weighted Conv/Linear activation precision is defined at the deployment
    # Q/DQ input and is materialized below as a module-input unit.  Retaining
    # the older transformer module-output observers as well would collect a
    # second, unused statistic for the same precision group.  Only functional
    # boundaries without a concrete module input (currently AV P/V) belong to
    # the transformer-specific inventory.
    functional_units = tuple(
        unit for unit in transformer_units
        if str(unit.boundary) in {"functional_input", "functional_output"}
    )
    rows: list[TaylorDeploymentUnit] = list(functional_units)
    group_to_units: dict[str, tuple[str, ...]] = {}
    for unit in functional_units:
        group = str(unit.metadata.get("precision_unit_id", ""))
        if group:
            group_to_units[group] = tuple([*group_to_units.get(group, ()), unit.unit_id])
    for group in space.quantization_groups:
        group_id = str(group.group_id)
        ids = []
        for index, path in enumerate(group.module_paths):
            try:
                model.get_submodule(str(path))
            except AttributeError:
                continue
            unit_id = f"activation::{group_id}::{index}"
            rows.append(TaylorDeploymentUnit(unit_id=unit_id, module_path=str(path), unit_type=str(group.metadata.get("transformer_role", "cnn")), boundary="module_input", precision_owner=str(path), quantizer_id=unit_id, metadata={"precision_group_id": group_id}))
            ids.append(unit_id)
        if ids: group_to_units[group_id] = tuple(ids)
    selected, _ = coalesce_deployment_units(rows)
    return selected, group_to_units


def collect_activation_taylor_cache(model: nn.Module, units: Sequence[TaylorDeploymentUnit], group_to_units: Mapping[str, tuple[str, ...]], *, forward_fn: Any, loss_fn: Any, batch: Any) -> ActivationTaylorCache:
    selected, _ = coalesce_deployment_units(units)
    transitions: dict[tuple[str, str, str], float] = {}
    with _BoundaryCapture(model, selected, quantize=False, retain_grad=True) as capture:
        model.zero_grad(set_to_none=True)
        loss = loss_fn(forward_fn(model, batch), batch)
        if loss.ndim != 0 or not torch.isfinite(loss): raise RuntimeError("activation_taylor_task_loss_invalid")
        loss.backward(); capture.require_one_per_unit()
        for unit in selected:
            value = capture.outputs[unit.unit_id][0]
            if value.grad is None: raise RuntimeError(f"activation_taylor_gradient_missing:{unit.unit_id}")
            base = value.detach()
            grad = value.grad.detach()
            for old, new in (("FP32", "FP16"), ("FP16", "INT8"), ("FP32", "INT8")):
                delta = pseudo_quantize_activation(base, new) - pseudo_quantize_activation(base, old)
                grad64 = grad.double()
                delta64 = delta.double()
                if not bool(torch.isfinite(base).all()):
                    raise RuntimeError(
                        f"activation_taylor_base_nonfinite:{unit.unit_id}:{old}->{new}"
                    )
                if not bool(torch.isfinite(grad64).all()):
                    raise RuntimeError(
                        f"activation_taylor_gradient_nonfinite:{unit.unit_id}:{old}->{new}"
                    )
                if not bool(torch.isfinite(delta64).all()):
                    raise RuntimeError(
                        f"activation_taylor_delta_nonfinite:{unit.unit_id}:{old}->{new}"
                    )
                first = (grad64 * delta64).abs()
                second = 0.5 * (grad64.square() * delta64.square()).abs()
                score = first + second
                if not bool(torch.isfinite(score).all()) or bool((score < 0).any()):
                    raise RuntimeError(
                        f"activation_taylor_element_invalid:{unit.unit_id}:{old}->{new}"
                    )
                transitions[(unit.unit_id, old, new)] = float(score.sum().detach().cpu())
    model.zero_grad(set_to_none=True)
    mapping = tuple({"unit_id": u.unit_id, "precision_group_id": u.metadata.get("precision_group_id"), "module_path": u.module_path, "boundary": u.boundary, "quantizer_id": u.quantizer_id} for u in selected)
    return ActivationTaylorCache(dict(group_to_units), transitions, mapping, {u.unit_id: u.precision_owner for u in selected})


def collect_functional_gate_scores_multi(
    model: nn.Module,
    domains: Sequence[Any],
    *,
    forward_fn: Any,
    loss_fn: Any,
    calibration_batches: Sequence[Any],
) -> tuple[dict[str, GateDomainScores], list[dict[str, Any]]]:
    """Collect gate scores independently per sample, then take their mean.

    Calling the single-sample collector separately is intentional: no signed
    gradient or activation is accumulated before the elementwise absolute
    contribution has been reduced for that sample.
    """

    batches = calibration_batches
    if len(batches) == 0:
        raise ValueError("gate_taylor_calibration_batches_empty")
    sample_rows: list[dict[str, GateDomainScores]] = []
    mapping: list[dict[str, Any]] = []
    for index, batch in enumerate(batches):
        rows, current_mapping = collect_functional_gate_scores(
            model,
            domains,
            forward_fn=forward_fn,
            loss_fn=loss_fn,
            batch=batch,
        )
        sample_rows.append(rows)
        if index == 0:
            mapping = current_mapping
    result: dict[str, GateDomainScores] = {}
    for domain in domains:
        domain_id = str(domain.domain_id)
        unit_ids = sorted(
            {
                unit_id
                for rows in sample_rows
                for unit_id in rows[domain_id].unit_scores
            }
        )
        per_sample = tuple(
            {
                unit_id: float(rows[domain_id].unit_scores.get(unit_id, 0.0))
                for unit_id in unit_ids
            }
            for rows in sample_rows
        )
        mean_scores = {
            unit_id: sum(row[unit_id] for row in per_sample) / len(per_sample)
            for unit_id in unit_ids
        }
        template = sample_rows[0][domain_id]
        result[domain_id] = GateDomainScores(
            domain_id=domain_id,
            unit_scores=mean_scores,
            semantic_root_tensor=template.semantic_root_tensor,
            gate_tensor=template.gate_tensor,
            physical_dependencies=template.physical_dependencies,
            family=template.family,
            sample_count=len(per_sample),
            per_sample_unit_scores=per_sample,
        )
    return result, mapping


def collect_activation_taylor_cache_multi(
    model: nn.Module,
    units: Sequence[TaylorDeploymentUnit],
    group_to_units: Mapping[str, tuple[str, ...]],
    *,
    forward_fn: Any,
    loss_fn: Any,
    calibration_batches: Sequence[Any],
) -> ActivationTaylorCache:
    """Collect Q/DQ-input Taylor transitions per sample and mean afterwards."""

    batches = calibration_batches
    if len(batches) == 0:
        raise ValueError("activation_taylor_calibration_batches_empty")
    rows = [
        collect_activation_taylor_cache(
            model,
            units,
            group_to_units,
            forward_fn=forward_fn,
            loss_fn=loss_fn,
            batch=batch,
        )
        for batch in batches
    ]
    keys = sorted({key for row in rows for key in row.transitions})
    per_sample = tuple(dict(row.transitions) for row in rows)
    transitions = {
        key: sum(float(row[key]) for row in per_sample) / len(per_sample)
        for key in keys
    }
    template = rows[0]
    return ActivationTaylorCache(
        group_to_units=dict(template.group_to_units),
        transitions=transitions,
        mapping=template.mapping,
        unit_to_owner=dict(template.unit_to_owner or {}),
        sample_count=len(per_sample),
        per_sample_transitions=per_sample,
    )
