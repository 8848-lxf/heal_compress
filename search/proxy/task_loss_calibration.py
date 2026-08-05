"""Actual task-loss measurements for frozen Taylor-objective calibration."""

from __future__ import annotations

import copy
import hashlib
import json
import random
from typing import Any, Callable, Mapping, Sequence

import torch
import torch.nn as nn

from ..candidate import CandidateGenotype
from ..canonicalization import SearchSpaceSpec, canonicalize_candidate
from ..ga.strict_stage12_v3 import phenotype_identity, validate_genotype_schema
from .candidate_perturbation import (
    parameter_slices_for_phenotype,
    pseudo_quantize_tensor,
    retained_mask_for_parameter,
)
from .joint_weight_activation_taylor import _BoundaryCapture, TaylorDeploymentUnit
from .calibrated_taylor_objective import fit_frozen_taylor_objective


CALIBRATION_MODES = ("pruning", "weight_quant", "activation_quant", "mixed")


def _hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def generate_stratified_calibration_candidates(
    space: SearchSpaceSpec,
    baseline: CandidateGenotype,
    *,
    count_per_mode: int,
    seed: int,
    excluded_candidates: Sequence[tuple[str, str]] = (),
) -> list[tuple[str, CandidateGenotype]]:
    """Generate deterministic pure-pruning, pure-WQ, pure-AQ and mixed rows."""

    if int(count_per_mode) <= 0:
        raise ValueError("task_loss_calibration_count_per_mode_invalid")
    validate_genotype_schema(baseline, space)
    rng = random.Random(int(seed))
    domains = tuple(space.pruning_domains)
    groups = {
        str(group.group_id): group for group in space.quantization_groups
    }
    rows: list[tuple[str, CandidateGenotype]] = []
    seen: set[tuple[str, str]] = {
        (str(mode), str(identity)) for mode, identity in excluded_candidates
    }
    for mode in CALIBRATION_MODES:
        attempts = 0
        while sum(row_mode == mode for row_mode, _ in rows) < int(count_per_mode):
            attempts += 1
            if attempts > int(count_per_mode) * 1000:
                raise RuntimeError(f"task_loss_calibration_candidate_generation_failed:{mode}")
            widths = dict(baseline.pruning_width_genes)
            precision = dict(baseline.precision_genes)
            if mode in {"pruning", "mixed"}:
                selected_count = rng.randint(1, max(1, len(domains)))
                for domain in rng.sample(list(domains), k=selected_count):
                    legal = tuple(int(value) for value in domain.legal_widths)
                    if len(legal) > 1:
                        # Quartile-stratified states cover mild through severe widths.
                        quantile = rng.choice((0.15, 0.35, 0.60, 0.85))
                        index = min(len(legal) - 2, max(0, int(round(quantile * (len(legal) - 1)))))
                        widths[str(domain.domain_id)] = legal[index]
            if mode in {"weight_quant", "activation_quant", "mixed"}:
                loci = list(space.precision_gene_ids)
                selected_count = rng.randint(1, max(1, len(loci)))
                for locus in rng.sample(loci, k=selected_count):
                    allowed = tuple(str(value) for value in groups[locus].allowed_precisions)
                    alternatives = [value for value in allowed if value != precision[locus]]
                    if alternatives:
                        precision[locus] = rng.choice(alternatives)
            candidate = CandidateGenotype(
                pruning_width_genes=widths,
                precision_genes=precision,
                meta={"created_by": f"task_loss_calibration_{mode}"},
            )
            validate_genotype_schema(candidate, space)
            identity = phenotype_identity(candidate, space)["complete_phenotype_hash"]
            key = (mode, identity)
            if key in seen:
                continue
            seen.add(key)
            rows.append((mode, candidate))
    return rows


def _perturbed_model(
    model: nn.Module,
    phenotype: Any,
    unit_to_parameter_slices: Mapping[str, Sequence[Any]],
    *,
    apply_structure: bool,
    apply_weight_quantization: bool,
) -> nn.Module:
    candidate = copy.deepcopy(model).eval()
    modules = dict(candidate.named_modules())
    slices = (
        parameter_slices_for_phenotype(
            phenotype,
            {str(key): list(value) for key, value in unit_to_parameter_slices.items()},
        )
        if apply_structure
        else {}
    )
    with torch.no_grad():
        for name, parameter in candidate.named_parameters():
            module_path = name.rsplit(".", 1)[0]
            value = parameter.detach()
            if (
                apply_weight_quantization
                and name.endswith(".weight")
                and module_path in phenotype.realized_precision_profile
            ):
                value = pseudo_quantize_tensor(
                    value,
                    phenotype.realized_precision_profile[module_path],
                    module=modules.get(module_path),
                )
            parameter_rows = list(slices.get(name, ()))
            if parameter_rows:
                retained = retained_mask_for_parameter(value, parameter_rows)
                value = torch.where(retained, value, torch.zeros_like(value))
            parameter.copy_(value)
    return candidate


def mean_task_loss(
    model: nn.Module,
    batches: Sequence[Any],
    *,
    forward_fn: Callable[[nn.Module, Any], Any],
    loss_fn: Callable[[Any, Any], torch.Tensor],
    activation_units: Sequence[TaylorDeploymentUnit] = (),
    precision_profile: Mapping[str, str] | None = None,
    quantize_activations: bool = False,
) -> float:
    losses: list[float] = []
    with torch.no_grad():
        for batch in batches:
            if quantize_activations:
                with _BoundaryCapture(
                    model,
                    activation_units,
                    precision_profile=precision_profile,
                    quantize=True,
                    retain_grad=False,
                ) as capture:
                    output = forward_fn(model, batch)
                    capture.require_one_per_unit()
            else:
                output = forward_fn(model, batch)
            loss = loss_fn(output, batch)
            if loss.ndim != 0 or not bool(torch.isfinite(loss)):
                raise RuntimeError("task_loss_calibration_nonfinite_loss")
            losses.append(float(loss.detach().cpu()))
    if not losses:
        raise ValueError("task_loss_calibration_batches_empty")
    return float(sum(losses) / len(losses))


def measure_task_loss_candidates(
    model: nn.Module,
    space: SearchSpaceSpec,
    candidates: Sequence[tuple[str, CandidateGenotype]],
    batches: Sequence[Any],
    *,
    unit_to_parameter_slices: Mapping[str, Sequence[Any]],
    activation_units: Sequence[TaylorDeploymentUnit],
    forward_fn: Callable[[nn.Module, Any], Any],
    loss_fn: Callable[[Any, Any], torch.Tensor],
    stage1_evaluator: Callable[[CandidateGenotype], Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Measure real configured task loss with independently toggled WQ and AQ."""

    baseline_loss = mean_task_loss(
        model, batches, forward_fn=forward_fn, loss_fn=loss_fn
    )
    rows: list[dict[str, Any]] = []
    for mode, genotype in candidates:
        if mode not in CALIBRATION_MODES:
            raise ValueError(f"task_loss_calibration_mode_invalid:{mode}")
        phenotype = canonicalize_candidate(genotype, space)
        candidate_model = _perturbed_model(
            model,
            phenotype,
            unit_to_parameter_slices,
            apply_structure=mode in {"pruning", "mixed"},
            apply_weight_quantization=mode in {"weight_quant", "mixed"},
        )
        task_loss = mean_task_loss(
            candidate_model,
            batches,
            forward_fn=forward_fn,
            loss_fn=loss_fn,
            activation_units=activation_units,
            precision_profile=phenotype.realized_precision_profile,
            quantize_activations=mode in {"activation_quant", "mixed"},
        )
        proxy = dict(stage1_evaluator(genotype))
        identity = phenotype_identity(genotype, space)
        perturbation_hash = _hash(
            {
                "candidate_mode": mode,
                "complete_phenotype_hash": identity["complete_phenotype_hash"],
            }
        )
        # Pure WQ/AQ labels use the same genotype but expose only the component
        # that was actually perturbed in the real task-loss forward.
        j_struct = float(proxy["J_struct_gate"]) if mode in {"pruning", "mixed"} else 0.0
        j_wq = float(proxy["J_WQ"]) if mode in {"weight_quant", "mixed"} else 0.0
        j_aq = float(proxy["J_AQ"]) if mode in {"activation_quant", "mixed"} else 0.0
        rows.append(
            {
                "candidate_hash": perturbation_hash,
                "complete_phenotype_hash": identity["complete_phenotype_hash"],
                "candidate_mode": mode,
                "J_struct_gate": j_struct,
                "J_WQ": j_wq,
                "J_AQ": j_aq,
                "J_WQ_x_J_AQ": j_wq * j_aq,
                "baseline_task_loss": baseline_loss,
                "candidate_task_loss": task_loss,
                "delta_task_loss": task_loss - baseline_loss,
                "R_bops_vs_fp32": float(proxy["R_bops_vs_fp32"]),
                "genotype": genotype.to_dict(),
            }
        )
        del candidate_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    audit = {
        "schema_version": "actual-task-loss-perturbation-calibration-v1",
        "candidate_count": len(rows),
        "baseline_task_loss": baseline_loss,
        "candidate_modes": list(CALIBRATION_MODES),
        "structure_semantics": "dependency-closed_zero_mask_functionally_equivalent_to_channel_removal",
        "weight_semantics": "production_granularity_fake_quantization",
        "activation_semantics": "deployment_boundary_fake_quantization",
        "candidate_manifest_hash": _hash(rows),
    }
    return rows, audit


def fit_task_loss_calibration_for_search(
    model: nn.Module,
    space: SearchSpaceSpec,
    baseline: CandidateGenotype,
    fit_batches: Sequence[Any],
    validation_batches: Sequence[Any],
    *,
    unit_to_parameter_slices: Mapping[str, Sequence[Any]],
    activation_units: Sequence[TaylorDeploymentUnit],
    forward_fn: Callable[[nn.Module, Any], Any],
    loss_fn: Callable[[Any, Any], torch.Tensor],
    stage1_evaluator: Callable[[CandidateGenotype], Mapping[str, Any]],
    count_per_mode_fit: int = 4,
    count_per_mode_validation: int = 2,
    seed: int = 0,
) -> tuple[Any, dict[str, Any]]:
    fit_candidates = generate_stratified_calibration_candidates(
        space,
        baseline,
        count_per_mode=int(count_per_mode_fit),
        seed=int(seed),
    )
    validation_candidates = generate_stratified_calibration_candidates(
        space,
        baseline,
        count_per_mode=int(count_per_mode_validation),
        seed=int(seed) + 1_000_003,
        excluded_candidates=tuple(
            (mode, phenotype_identity(candidate, space)["complete_phenotype_hash"])
            for mode, candidate in fit_candidates
        ),
    )
    fit_rows, fit_audit = measure_task_loss_candidates(
        model,
        space,
        fit_candidates,
        fit_batches,
        unit_to_parameter_slices=unit_to_parameter_slices,
        activation_units=activation_units,
        forward_fn=forward_fn,
        loss_fn=loss_fn,
        stage1_evaluator=stage1_evaluator,
    )
    validation_rows, validation_audit = measure_task_loss_candidates(
        model,
        space,
        validation_candidates,
        validation_batches,
        unit_to_parameter_slices=unit_to_parameter_slices,
        activation_units=activation_units,
        forward_fn=forward_fn,
        loss_fn=loss_fn,
        stage1_evaluator=stage1_evaluator,
    )
    data_hash = _hash(
        {
            "fit_candidate_manifest_hash": fit_audit["candidate_manifest_hash"],
            "validation_candidate_manifest_hash": validation_audit[
                "candidate_manifest_hash"
            ],
            "fit_batch_count": len(fit_batches),
            "validation_batch_count": len(validation_batches),
        }
    )
    fit_hashes = {str(row["candidate_hash"]) for row in fit_rows}
    validation_hashes = {str(row["candidate_hash"]) for row in validation_rows}
    if fit_hashes & validation_hashes:
        raise RuntimeError("task_loss_calibration_fit_validation_candidates_overlap")
    calibration = fit_frozen_taylor_objective(
        fit_rows,
        validation_rows,
        calibration_data_hash=data_hash,
    )
    return calibration, {
        "schema_version": "search-task-loss-objective-calibration-v1",
        "fit": fit_audit,
        "validation": validation_audit,
        "fit_rows": fit_rows,
        "validation_rows": validation_rows,
        "frozen_calibration": calibration.to_dict(),
        "coefficients_frozen_before_greedy_and_ga": True,
        "fit_validation_candidate_sets_disjoint": not bool(
            set(calibration.fit_candidate_hashes)
            & set(calibration.validation_candidate_hashes)
        ),
    }


__all__ = [
    "CALIBRATION_MODES",
    "generate_stratified_calibration_candidates",
    "mean_task_loss",
    "measure_task_loss_candidates",
    "fit_task_loss_calibration_for_search",
]
