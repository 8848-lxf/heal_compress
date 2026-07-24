"""Joint structural, weight and activation output-perturbation Taylor proxy."""

from __future__ import annotations

import copy
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass, field
import hashlib
import json
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..candidate import CandidatePhenotype
from .candidate_perturbation import (
    deployment_precision,
    parameter_slices_for_phenotype,
    pseudo_quantize_activation,
    pseudo_quantize_tensor,
    retained_mask_for_parameter,
)
from .parameter_slice_resolver import ParameterSlice


def _stable_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class TaylorDeploymentUnit:
    """One deployment output boundary scored exactly once."""

    unit_id: str
    module_path: str
    unit_type: str
    boundary: str
    precision_owner: str
    quantizer_id: str
    tensor_index: int = 0
    call_index: int = 0
    functional_op: str = ""
    has_weight: bool = True
    protected: bool = False
    protection_reason: str = ""
    activation_channel_axis: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.boundary not in {"module_input", "module_output", "functional_output"}:
            raise ValueError(f"taylor_boundary_invalid:{self.unit_id}:{self.boundary}")
        if self.boundary == "functional_output" and self.functional_op not in {
            "softmax",
            "einsum",
            "matmul",
            "bmm",
        }:
            raise ValueError(f"taylor_functional_op_unsupported:{self.unit_id}:{self.functional_op}")
        if not self.quantizer_id:
            raise ValueError(f"taylor_quantizer_id_missing:{self.unit_id}")
        object.__setattr__(self, "metadata", dict(self.metadata))

    def identity(self) -> tuple[Any, ...]:
        return (
            self.module_path,
            self.boundary,
            self.functional_op,
            int(self.tensor_index),
            int(self.call_index),
            self.precision_owner,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def coalesce_deployment_units(
    units: Sequence[TaylorDeploymentUnit],
) -> tuple[tuple[TaylorDeploymentUnit, ...], dict[str, list[str]]]:
    """Deduplicate identical quantizer boundaries; conflicts fail closed."""

    by_quantizer: dict[str, TaylorDeploymentUnit] = {}
    aliases: dict[str, list[str]] = {}
    for unit in units:
        previous = by_quantizer.get(unit.quantizer_id)
        if previous is None:
            by_quantizer[unit.quantizer_id] = unit
            aliases[unit.quantizer_id] = [unit.unit_id]
            continue
        if previous.identity() != unit.identity():
            raise ValueError(
                f"quantizer_id_boundary_conflict:{unit.quantizer_id}:"
                f"{previous.identity()}:{unit.identity()}"
            )
        aliases[unit.quantizer_id].append(unit.unit_id)
    ordered = tuple(sorted(by_quantizer.values(), key=lambda value: (value.module_path, value.call_index, value.unit_id)))
    return ordered, aliases


def taylor_units_from_transformer_precision(
    model: nn.Module,
    precision_units: Sequence[Any],
    *,
    active_module_paths: Sequence[str] | None = None,
) -> tuple[TaylorDeploymentUnit, ...]:
    """Map precision contracts to exact Stage-1 output perturbation boundaries.

    Protected QK, LayerNorm, and residual boundaries have no searched
    perturbation and therefore are audited by the precision contract rather
    than counted as zero-valued Taylor units. Every open unit must resolve to a
    module or an explicitly described functional boundary.
    """

    active = set(str(value) for value in (active_module_paths or ()))
    rows: list[TaylorDeploymentUnit] = []
    protected_roles = {"qk_matmul", "layernorm", "residual_add"}
    for precision_unit in sorted(precision_units, key=lambda value: value.ordering):
        role = str(precision_unit.role)
        if role in protected_roles:
            continue
        metadata = dict(precision_unit.metadata)
        for path_index, path in enumerate(precision_unit.module_paths):
            path = str(path)
            # ``active_module_paths`` comes from weighted/runtime shape hooks.
            # Softmax and FFN activation modules can execute without appearing
            # in that inventory.  Their owner precision unit has already been
            # active-instance filtered by the model adapter, so retain these
            # mandatory activation boundaries here.
            if (
                active
                and "::__" not in path
                and path not in active
                and role not in {"softmax", "ffn_activation"}
            ):
                continue
            boundary = "module_output"
            module_path = path
            functional_op = ""
            call_index = 0
            try:
                model.get_submodule(path)
            except AttributeError:
                owner = str(metadata.get("functional_owner", ""))
                operation = str(metadata.get("functional_op", ""))
                if not owner or operation not in {"softmax", "einsum", "matmul", "bmm"}:
                    raise RuntimeError(
                        f"transformer_taylor_boundary_missing_unit_mapping:"
                        f"{precision_unit.unit_id}:{path}"
                    )
                boundary = "functional_output"
                module_path = owner
                functional_op = operation
                call_index = int(metadata.get("functional_call_index", 0))
            rows.append(
                TaylorDeploymentUnit(
                    unit_id=(
                        str(precision_unit.unit_id)
                        if len(precision_unit.module_paths) == 1
                        else f"{precision_unit.unit_id}::path{path_index}"
                    ),
                    module_path=module_path,
                    unit_type=role,
                    boundary=boundary,
                    precision_owner=path,
                    quantizer_id=f"activation_quantizer::{precision_unit.unit_id}::{path}",
                    call_index=call_index,
                    functional_op=functional_op,
                    has_weight=not bool(precision_unit.activation_only),
                    protected=bool(precision_unit.protected),
                    protection_reason=str(precision_unit.protection_reason),
                    metadata={
                        **metadata,
                        "precision_unit_id": str(precision_unit.unit_id),
                        "precision_role": role,
                        "source_precision_path": path,
                    },
                )
            )
    selected, _aliases = coalesce_deployment_units(rows)
    return selected


def _tensor_from_value(value: Any, index: int) -> torch.Tensor:
    if torch.is_tensor(value):
        if int(index) != 0:
            raise IndexError(f"tensor_boundary_index_invalid:{index}")
        return value
    if isinstance(value, (tuple, list)):
        selected = value[int(index)]
        if not torch.is_tensor(selected):
            raise TypeError(f"boundary_value_not_tensor:{index}:{type(selected).__name__}")
        return selected
    raise TypeError(f"boundary_output_unsupported:{type(value).__name__}")


def _replace_tensor_in_value(value: Any, index: int, replacement: torch.Tensor) -> Any:
    if torch.is_tensor(value):
        if int(index) != 0:
            raise IndexError(f"tensor_boundary_index_invalid:{index}")
        return replacement
    if isinstance(value, tuple):
        rows = list(value)
        rows[int(index)] = replacement
        return tuple(rows)
    if isinstance(value, list):
        rows = list(value)
        rows[int(index)] = replacement
        return rows
    raise TypeError(f"boundary_output_unsupported:{type(value).__name__}")


class _BoundaryCapture(AbstractContextManager):
    def __init__(
        self,
        model: nn.Module,
        units: Sequence[TaylorDeploymentUnit],
        *,
        precision_profile: Mapping[str, str] | None = None,
        quantize: bool,
        retain_grad: bool,
    ) -> None:
        self.model = model
        self.units = tuple(units)
        self.precision_profile = {str(key): deployment_precision(value) for key, value in dict(precision_profile or {}).items()}
        self.quantize = bool(quantize)
        self.retain_grad = bool(retain_grad)
        self.outputs: dict[str, list[torch.Tensor]] = {unit.unit_id: [] for unit in self.units}
        self._handles: list[Any] = []
        self._patches: list[tuple[Any, str, Any]] = []
        self._global_patches: list[tuple[dict[str, Any], str, Any]] = []
        self._call_counts: dict[tuple[str, str], int] = {}
        self._owner_stack: list[str] = []
        self._functional_reentrant = False

    def _precision(self, unit: TaylorDeploymentUnit) -> str:
        value = self.precision_profile.get(unit.precision_owner, "FP32")
        if unit.protected and value != "FP32":
            raise RuntimeError(
                f"protected_activation_precision_conflict:{unit.unit_id}:{value}:"
                f"{unit.protection_reason}"
            )
        return value

    def _record(self, unit: TaylorDeploymentUnit, tensor: torch.Tensor) -> torch.Tensor:
        value = tensor
        if self.quantize:
            value = pseudo_quantize_activation(
                tensor,
                self._precision(unit),
                channel_axis=unit.activation_channel_axis,
            )
        if self.retain_grad and value.requires_grad:
            value.retain_grad()
        self.outputs[unit.unit_id].append(value)
        return value

    def _install_module_hooks(self) -> None:
        grouped: dict[tuple[str, str], list[TaylorDeploymentUnit]] = {}
        for unit in self.units:
            if unit.boundary != "functional_output":
                grouped.setdefault((unit.module_path, unit.boundary), []).append(unit)
        for (path, boundary), units in grouped.items():
            module = self.model if path in {"", "__root__"} else self.model.get_submodule(path)
            ordered = tuple(sorted(units, key=lambda value: (value.call_index, value.tensor_index, value.unit_id)))
            if boundary == "module_input":
                def pre_hook(_module: nn.Module, inputs: tuple[Any, ...], *, _path=path, _units=ordered):
                    key = (_path, "module_input")
                    call = self._call_counts.get(key, 0)
                    self._call_counts[key] = call + 1
                    result: Any = inputs
                    for unit in _units:
                        if unit.call_index != call:
                            continue
                        tensor = _tensor_from_value(result, unit.tensor_index)
                        result = _replace_tensor_in_value(result, unit.tensor_index, self._record(unit, tensor))
                    return result
                self._handles.append(module.register_forward_pre_hook(pre_hook))
            else:
                def post_hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any, *, _path=path, _units=ordered):
                    key = (_path, "module_output")
                    call = self._call_counts.get(key, 0)
                    self._call_counts[key] = call + 1
                    result = output
                    for unit in _units:
                        if unit.call_index != call:
                            continue
                        tensor = _tensor_from_value(result, unit.tensor_index)
                        result = _replace_tensor_in_value(result, unit.tensor_index, self._record(unit, tensor))
                    return result
                self._handles.append(module.register_forward_hook(post_hook))

    def _install_functional_hooks(self) -> None:
        units = [unit for unit in self.units if unit.boundary == "functional_output"]
        if not units:
            return
        by_owner: dict[str, list[TaylorDeploymentUnit]] = {}
        for unit in units:
            by_owner.setdefault(unit.module_path, []).append(unit)
        for path in by_owner:
            module = self.model if path in {"", "__root__"} else self.model.get_submodule(path)

            def push(_module: nn.Module, _inputs: tuple[Any, ...], *, _path=path) -> None:
                self._owner_stack.append(_path)

            def pop(_module: nn.Module, _inputs: tuple[Any, ...], output: Any, *, _path=path) -> Any:
                if self._owner_stack and self._owner_stack[-1] == _path:
                    self._owner_stack.pop()
                elif _path in self._owner_stack:
                    self._owner_stack.remove(_path)
                return output

            self._handles.append(module.register_forward_pre_hook(push))
            self._handles.append(module.register_forward_hook(pop))

        replacements: dict[int, Callable[..., Any]] = {}

        def patch(obj: Any, attr: str, operation: str) -> None:
            original = getattr(obj, attr)

            def wrapper(*args: Any, **kwargs: Any) -> Any:
                if self._functional_reentrant:
                    return original(*args, **kwargs)
                self._functional_reentrant = True
                try:
                    result = original(*args, **kwargs)
                finally:
                    self._functional_reentrant = False
                if not self._owner_stack:
                    return result
                owner = self._owner_stack[-1]
                key = (owner, f"functional_{operation}")
                call = self._call_counts.get(key, 0)
                self._call_counts[key] = call + 1
                value = result
                for unit in by_owner.get(owner, ()):
                    if unit.functional_op == operation and unit.call_index == call:
                        tensor = _tensor_from_value(value, unit.tensor_index)
                        value = _replace_tensor_in_value(value, unit.tensor_index, self._record(unit, tensor))
                return value

            self._patches.append((obj, attr, original))
            setattr(obj, attr, wrapper)
            replacements[id(original)] = wrapper

        patch(torch, "softmax", "softmax")
        patch(F, "softmax", "softmax")
        patch(torch.Tensor, "softmax", "softmax")
        patch(torch, "einsum", "einsum")
        patch(torch, "matmul", "matmul")
        patch(torch, "bmm", "bmm")
        patch(torch.Tensor, "matmul", "matmul")

        # Patch exact module-level aliases such as ``from torch import einsum``.
        namespaces: dict[int, dict[str, Any]] = {}
        for module in self.model.modules():
            function = getattr(module.forward, "__func__", module.forward)
            namespace = getattr(function, "__globals__", None)
            if isinstance(namespace, dict):
                namespaces[id(namespace)] = namespace
        for namespace in namespaces.values():
            for name, value in list(namespace.items()):
                replacement = replacements.get(id(value))
                if replacement is None:
                    continue
                self._global_patches.append((namespace, name, value))
                namespace[name] = replacement

    def __enter__(self) -> "_BoundaryCapture":
        self._install_module_hooks()
        self._install_functional_hooks()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        for handle in self._handles:
            handle.remove()
        for namespace, name, original in reversed(self._global_patches):
            namespace[name] = original
        for obj, attr, original in reversed(self._patches):
            setattr(obj, attr, original)
        self._handles = []
        self._patches = []
        self._global_patches = []

    def require_one_per_unit(self) -> None:
        missing = {unit_id: len(values) for unit_id, values in self.outputs.items() if len(values) != 1}
        if missing:
            raise RuntimeError(f"joint_output_boundary_capture_count:{missing}")


@dataclass
class JointOutputTaylorStatistics:
    baseline_outputs: dict[str, tuple[torch.Tensor, ...]]
    gradients: dict[str, tuple[torch.Tensor, ...]]
    calibration_manifest_hash: str
    unit_manifest_hash: str
    sample_count: int
    task_losses: tuple[float, ...] = ()
    statistics_version: str = "joint-output-taylor-v1"

    def to_manifest(self) -> dict[str, Any]:
        return {
            "calibration_manifest_hash": self.calibration_manifest_hash,
            "unit_manifest_hash": self.unit_manifest_hash,
            "sample_count": int(self.sample_count),
            "task_losses": list(self.task_losses),
            "statistics_version": self.statistics_version,
            "units": {
                unit_id: [list(value.shape) for value in values]
                for unit_id, values in sorted(self.baseline_outputs.items())
            },
        }


def collect_joint_output_taylor_statistics(
    model: nn.Module,
    calibration_batches: Sequence[Any],
    *,
    forward_fn: Callable[[nn.Module, Any], Any],
    loss_fn: Callable[[Any, Any], torch.Tensor],
    units: Sequence[TaylorDeploymentUnit],
    calibration_manifest_hash: str,
) -> JointOutputTaylorStatistics:
    """Collect ``y``, ``g=dL/dy`` and implicit ``h=g^2`` on one task loss."""

    if not calibration_manifest_hash:
        raise ValueError("joint_output_calibration_manifest_hash_missing")
    selected, _aliases = coalesce_deployment_units(units)
    if not calibration_batches:
        raise ValueError("joint_output_calibration_batches_empty")
    outputs: dict[str, list[torch.Tensor]] = {unit.unit_id: [] for unit in selected}
    gradients: dict[str, list[torch.Tensor]] = {unit.unit_id: [] for unit in selected}
    losses: list[float] = []
    model.train(False)
    for batch in calibration_batches:
        model.zero_grad(set_to_none=True)
        with _BoundaryCapture(model, selected, quantize=False, retain_grad=True) as capture:
            result = forward_fn(model, batch)
            loss = loss_fn(result, batch)
            if loss.ndim != 0 or not torch.isfinite(loss):
                raise RuntimeError("joint_output_task_loss_must_be_finite_scalar")
            loss.backward()
            capture.require_one_per_unit()
            for unit in selected:
                value = capture.outputs[unit.unit_id][0]
                if value.grad is None:
                    raise RuntimeError(f"joint_output_gradient_missing:{unit.unit_id}")
                outputs[unit.unit_id].append(value.detach().cpu())
                gradients[unit.unit_id].append(value.grad.detach().cpu())
            losses.append(float(loss.detach().cpu()))
    model.zero_grad(set_to_none=True)
    unit_manifest_hash = _stable_hash([unit.to_dict() for unit in selected])
    return JointOutputTaylorStatistics(
        baseline_outputs={key: tuple(values) for key, values in outputs.items()},
        gradients={key: tuple(values) for key, values in gradients.items()},
        calibration_manifest_hash=str(calibration_manifest_hash),
        unit_manifest_hash=unit_manifest_hash,
        sample_count=len(calibration_batches),
        task_losses=tuple(losses),
    )


@dataclass(frozen=True)
class CandidateOutputBundle:
    outputs: dict[str, tuple[torch.Tensor, ...]]
    calibration_manifest_hash: str
    audit: dict[str, Any] = field(default_factory=dict)


class ModelCandidateOutputProvider:
    """Replay a complete virtual candidate on the exact calibration batches."""

    def __init__(
        self,
        model: nn.Module,
        calibration_batches: Sequence[Any],
        *,
        forward_fn: Callable[[nn.Module, Any], Any],
        units: Sequence[TaylorDeploymentUnit],
        unit_to_parameter_slices: Mapping[str, Sequence[ParameterSlice]],
        calibration_manifest_hash: str,
        candidate_model_factory: Callable[[nn.Module], nn.Module] | None = None,
        apply_structural_perturbation: bool = True,
        quantize_weights: bool = True,
        quantize_activations: bool = True,
    ) -> None:
        if not calibration_manifest_hash:
            raise ValueError("candidate_output_calibration_manifest_hash_missing")
        self.model = model
        self.calibration_batches = tuple(calibration_batches)
        self.forward_fn = forward_fn
        self.units, self.quantizer_aliases = coalesce_deployment_units(units)
        self.unit_to_parameter_slices = {
            str(key): list(values) for key, values in unit_to_parameter_slices.items()
        }
        self.calibration_manifest_hash = str(calibration_manifest_hash)
        self.candidate_model_factory = candidate_model_factory or copy.deepcopy
        self.apply_structural_perturbation = bool(apply_structural_perturbation)
        self.quantize_weights = bool(quantize_weights)
        self.quantize_activations = bool(quantize_activations)

    def __call__(self, phenotype: CandidatePhenotype) -> CandidateOutputBundle:
        candidate = self.candidate_model_factory(self.model)
        candidate.train(False)
        modules = dict(candidate.named_modules())
        slices = (
            parameter_slices_for_phenotype(
                phenotype, self.unit_to_parameter_slices
            )
            if self.apply_structural_perturbation
            else {}
        )
        pruned_parameters = 0
        quantized_parameters = 0
        with torch.no_grad():
            for name, parameter in candidate.named_parameters():
                module_path = name.rsplit(".", 1)[0]
                rows = slices.get(name, [])
                retained = retained_mask_for_parameter(parameter.detach(), rows)
                precision = (
                    deployment_precision(
                        phenotype.realized_precision_profile.get(
                            module_path, "FP32"
                        )
                    )
                    if self.quantize_weights
                    else "FP32"
                )
                if (
                    self.quantize_weights
                    and name.endswith(".weight")
                    and module_path in phenotype.realized_precision_profile
                ):
                    quantized = pseudo_quantize_tensor(parameter.detach(), precision, module=modules.get(module_path))
                    if precision != "FP32":
                        quantized_parameters += 1
                else:
                    quantized = parameter.detach()
                effective = torch.where(retained, quantized, torch.zeros_like(parameter))
                if rows:
                    pruned_parameters += 1
                parameter.copy_(effective)
        outputs: dict[str, list[torch.Tensor]] = {unit.unit_id: [] for unit in self.units}
        with torch.no_grad():
            for batch in self.calibration_batches:
                with _BoundaryCapture(
                    candidate,
                    self.units,
                    precision_profile=phenotype.realized_precision_profile,
                    quantize=self.quantize_activations,
                    retain_grad=False,
                ) as capture:
                    self.forward_fn(candidate, batch)
                    capture.require_one_per_unit()
                    for unit in self.units:
                        outputs[unit.unit_id].append(capture.outputs[unit.unit_id][0].detach().cpu())
        return CandidateOutputBundle(
            outputs={key: tuple(values) for key, values in outputs.items()},
            calibration_manifest_hash=self.calibration_manifest_hash,
            audit={
                "candidate_replay": "complete_virtual_structure_weight_activation",
                "pruned_parameter_tensor_count": pruned_parameters,
                "quantized_weight_tensor_count": quantized_parameters,
                "activation_quantizer_count": len(self.units),
                "quantizer_aliases": self.quantizer_aliases,
                "mask_used_only_for_stage1_proxy": True,
                "structural_perturbation_applied": self.apply_structural_perturbation,
                "weight_quantization_applied": self.quantize_weights,
                "activation_quantization_applied": self.quantize_activations,
            },
        )


def score_joint_output_perturbation(
    statistics: JointOutputTaylorStatistics,
    perturbed_outputs: Mapping[str, Sequence[torch.Tensor]],
) -> dict[str, Any]:
    """Apply the unnormalized common-unit first-plus-second-order formula."""

    if statistics.sample_count <= 0:
        raise ValueError("joint_output_statistics_sample_count_invalid")
    missing = sorted(set(statistics.baseline_outputs) - set(perturbed_outputs))
    extra = sorted(set(perturbed_outputs) - set(statistics.baseline_outputs))
    if missing or extra:
        raise RuntimeError(f"joint_output_unit_inventory_mismatch:missing={missing}:extra={extra}")
    first_total = 0.0
    second_total = 0.0
    breakdown: list[dict[str, Any]] = []
    for unit_id in sorted(statistics.baseline_outputs):
        baseline_rows = statistics.baseline_outputs[unit_id]
        gradient_rows = statistics.gradients[unit_id]
        candidate_rows = tuple(perturbed_outputs[unit_id])
        if not (len(baseline_rows) == len(gradient_rows) == len(candidate_rows) == statistics.sample_count):
            raise RuntimeError(f"joint_output_sample_count_mismatch:{unit_id}")
        unit_first = 0.0
        unit_second = 0.0
        delta_l1 = 0.0
        for baseline, gradient, candidate in zip(baseline_rows, gradient_rows, candidate_rows):
            if tuple(baseline.shape) != tuple(candidate.shape):
                raise RuntimeError(
                    f"joint_output_realized_shape_conflict:{unit_id}:{tuple(baseline.shape)}:{tuple(candidate.shape)}"
                )
            base = baseline.float()
            grad = gradient.float()
            delta = candidate.float() - base
            unit_first += float((grad * delta).abs().sum())
            unit_second += 0.5 * float((grad.square() * delta.square()).sum())
            delta_l1 += float(delta.abs().sum())
        unit_first /= statistics.sample_count
        unit_second /= statistics.sample_count
        delta_l1 /= statistics.sample_count
        first_total += unit_first
        second_total += unit_second
        breakdown.append({
            "unit_id": unit_id,
            "first_order": unit_first,
            "second_order": unit_second,
            "joint_loss_increment": unit_first + unit_second,
            "mean_delta_l1": delta_l1,
        })
    return {
        "L_joint_weight_activation_taylor": first_total + second_total,
        "L_joint_weight_activation_taylor_raw": first_total + second_total,
        "L_joint_first_order": first_total,
        "L_joint_second_order": second_total,
        "unit_breakdown": breakdown,
        "sample_count": statistics.sample_count,
        "normalization_applied": False,
        "type_calibration": "identity",
        "formula": "mean_samples sum_units(abs(g*delta_y)+0.5*g^2*delta_y^2)",
        "structural_weight_activation_cross_terms_preserved": True,
    }


@dataclass(frozen=True)
class DomainPerturbationCacheKey:
    model: str
    domain_id: str
    structural_width: int
    precision_profile: tuple[tuple[str, str], ...]
    calibration_manifest_hash: str

    @property
    def digest(self) -> str:
        return _stable_hash(asdict(self))


class DomainPerturbationCache:
    """Level-1 action cache; deliberately distinct from a latency LUT."""

    def __init__(self) -> None:
        self._values: dict[str, dict[str, Any]] = {}
        self._keys: dict[str, DomainPerturbationCacheKey] = {}

    def get(self, key: DomainPerturbationCacheKey) -> dict[str, Any] | None:
        value = self._values.get(key.digest)
        return None if value is None else dict(value)

    def put(self, key: DomainPerturbationCacheKey, value: Mapping[str, Any]) -> None:
        digest = key.digest
        previous = self._keys.get(digest)
        if previous is not None and previous != key:
            raise RuntimeError(f"domain_perturbation_cache_hash_collision:{digest}")
        self._keys[digest] = key
        self._values[digest] = dict(value)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "domain-perturbation-cache-v1",
            "latency_lut": False,
            "entry_count": len(self._values),
            "entries": [
                {"cache_key": asdict(self._keys[digest]), "cache_key_hash": digest, "metrics": self._values[digest]}
                for digest in sorted(self._values)
            ],
        }


class JointWeightActivationTaylorProxy:
    """Candidate-level Level-2 replay on the exact calibration manifest."""

    def __init__(
        self,
        *,
        statistics: JointOutputTaylorStatistics,
        output_provider: Callable[[CandidatePhenotype], CandidateOutputBundle],
        units: Sequence[TaylorDeploymentUnit],
    ) -> None:
        self.statistics = statistics
        self.output_provider = output_provider
        self.units, self.quantizer_aliases = coalesce_deployment_units(units)
        unit_hash = _stable_hash([unit.to_dict() for unit in self.units])
        if unit_hash != statistics.unit_manifest_hash:
            raise RuntimeError(
                f"joint_output_unit_manifest_conflict:{unit_hash}:{statistics.unit_manifest_hash}"
            )

    def evaluate_breakdown(self, phenotype: CandidatePhenotype) -> dict[str, Any]:
        bundle = self.output_provider(phenotype)
        if bundle.calibration_manifest_hash != self.statistics.calibration_manifest_hash:
            raise RuntimeError(
                "joint_output_calibration_manifest_conflict:"
                f"{bundle.calibration_manifest_hash}:{self.statistics.calibration_manifest_hash}"
            )
        metrics = score_joint_output_perturbation(self.statistics, bundle.outputs)
        value = float(metrics["L_joint_weight_activation_taylor"])
        return {
            **metrics,
            # Compatibility alias while Greedy/GA reports migrate to the
            # explicit joint weight-activation name.
            "L_joint_weight_taylor": value,
            "L_joint_weight_taylor_raw": value,
            "activation_taylor_included": True,
            "softmax_activation_taylor_supported": True,
            "quantizer_id_unique_count": len(self.units),
            "quantizer_aliases": self.quantizer_aliases,
            "calibration_manifest_hash": self.statistics.calibration_manifest_hash,
            "candidate_proxy_refresh": True,
            "provider_audit": dict(bundle.audit),
        }

    def evaluate(self, phenotype: CandidatePhenotype) -> float:
        return float(self.evaluate_breakdown(phenotype)["L_joint_weight_activation_taylor"])


__all__ = [
    "CandidateOutputBundle",
    "DomainPerturbationCache",
    "DomainPerturbationCacheKey",
    "JointOutputTaylorStatistics",
    "JointWeightActivationTaylorProxy",
    "ModelCandidateOutputProvider",
    "TaylorDeploymentUnit",
    "coalesce_deployment_units",
    "collect_joint_output_taylor_statistics",
    "score_joint_output_perturbation",
    "taylor_units_from_transformer_precision",
]
