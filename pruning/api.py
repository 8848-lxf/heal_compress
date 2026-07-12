"""Stable public API for formal structured pruning."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable

import torch.nn as nn

from .artifacts.hashing import compute_physical_hashes
from .artifacts.snapshot import build_physical_structure_snapshot
from .config import AlignmentConfig, GroupedConvConfig, ImportanceConfig, ImportanceMode, PruningConfig, SelectionConfig
from .exceptions import PruningLegalityError
from .importance.aggregation import coupled_dependency_mean
from .importance.first_order_taylor import estimate_unit_parameter_cost, score_first_order_taylor
from .importance.norm import norm_parameter_slice
from .importance.normalization import normalize_scope_scores
from .importance.second_order_fisher import score_second_order_fisher
from .materialization.executor import materialize_pruning
from .materialization.legalizer import legalize_pruning_plan
from .materialization.planner import build_physical_pruning_plan, estimate_physical_parameter_count
from .materialization.replay import replay_pruning
from .model_io import load_model
from .selection.global_ranking import select_global_units
from .types import (
    AtomicPruneUnit,
    ImportanceResult,
    MaterializationResult,
    ModelLoadResult,
    PhysicalHashes,
    PhysicalPruningPlan,
    PhysicalStructureSnapshot,
    PhysicalValidationResult,
    SamplingPruningRequest,
)
from .validation.structure import validate_physical_model


def _importance_config(config: ImportanceConfig | PruningConfig | None) -> ImportanceConfig:
    if config is None:
        return ImportanceConfig()
    return config.importance if isinstance(config, PruningConfig) else config


def _norm_member_score(module: nn.Module, axis: str, indices: Sequence[int], order: int) -> float:
    parameter = getattr(module, "weight", None)
    if parameter is None:
        return float("inf")
    if isinstance(module, nn.modules.batchnorm._BatchNorm):
        return norm_parameter_slice(parameter, indices, 0, order=order)
    if isinstance(module, nn.ConvTranspose2d):
        parameter_axis = 1 if axis in {"out", "channel"} else 0
    else:
        parameter_axis = 0 if axis in {"out", "channel"} else 1
    return norm_parameter_slice(parameter, indices, parameter_axis, order=order)


def _score_norm_mode(
    model: nn.Module,
    units: Sequence[Any],
    *,
    config: ImportanceConfig,
    order: int,
) -> ImportanceResult:
    modules = dict(model.named_modules())
    raw: dict[str, float] = {}
    scope_units: dict[str, list[str]] = defaultdict(list)
    rows: list[dict[str, Any]] = []
    for unit in units:
        scores = [
            _norm_member_score(modules[member.module_path], member.axis, member.indices, order)
            for member in getattr(unit, "members", [])
            if member.module_path in modules
        ]
        stable_id = str(unit.stable_id)
        scope_id = str(unit.scope_id)
        raw[stable_id] = coupled_dependency_mean(scores)
        scope_units[scope_id].append(stable_id)
        rows.append({"stable_id": stable_id, "scope_id": scope_id, "member_scores": scores})
    normalized_scopes = normalize_scope_scores(
        {scope: [raw[stable_id] for stable_id in stable_ids] for scope, stable_ids in scope_units.items()},
        config=config.normalization,
    )
    normalized = {
        stable_id: score
        for scope, stable_ids in scope_units.items()
        for stable_id, score in zip(stable_ids, normalized_scopes[scope])
    }
    return ImportanceResult(
        mode=config.mode.value,
        normalization=config.normalization.strategy.value,
        aggregation=config.aggregation.value,
        raw_scores=raw,
        normalized_scores=normalized,
        unit_parameter_costs={str(unit.stable_id): estimate_unit_parameter_cost(model, unit) for unit in units},
        unit_scores=rows,
        implementation_version=f"{config.mode.value}-v1",
    )


def score_pruning_units(
    model: nn.Module,
    units: Sequence[Any],
    *,
    config: ImportanceConfig | PruningConfig | None = None,
    calibration_batches: int = 0,
    task_loss: str = "",
) -> ImportanceResult:
    """Score immutable coupled units without changing model structure.

    Gradients must already be populated by caller-controlled calibration for
    Taylor/Fisher modes. The formal default is normalized first-order Taylor.
    """

    cfg = _importance_config(config)
    if cfg.mode is ImportanceMode.FIRST_ORDER_TAYLOR:
        return score_first_order_taylor(
            model,
            units,
            config=cfg,
            calibration_batches=calibration_batches,
            task_loss=task_loss,
        )
    if cfg.mode is ImportanceMode.L1_NORM:
        return _score_norm_mode(model, units, config=cfg, order=1)
    if cfg.mode is ImportanceMode.L2_NORM:
        return _score_norm_mode(model, units, config=cfg, order=2)
    if cfg.mode is ImportanceMode.SECOND_ORDER_FISHER:
        return score_second_order_fisher(
            model,
            units,
            config=cfg,
            calibration_batches=calibration_batches,
            task_loss=task_loss,
        )
    raise ValueError(f"unsupported importance mode: {cfg.mode}")


def select_pruning_request(
    units: Sequence[Any],
    *,
    importance_result: ImportanceResult | None = None,
    channel_budget: int | None = None,
    parameter_budget: int | None = None,
    grouped_config: GroupedConvConfig | None = None,
    selection_config: SelectionConfig | None = None,
    alignment_config: AlignmentConfig | None = None,
) -> SamplingPruningRequest:
    """Globally rank normalized scores and return one non-mutating request.

    Trace-time atoms can be passed directly together with ``importance_result``;
    their score is the arithmetic mean of the recorded source coupled units.
    Missing source scores fail closed. Already scored pruning atoms remain
    accepted for compatibility and focused policy checks.
    """

    scored: list[AtomicPruneUnit] = []
    for unit in units:
        source_ids = [str(value) for value in getattr(unit, "source_coupled_unit_ids", ())]
        if importance_result is None:
            if not isinstance(unit, AtomicPruneUnit):
                raise PruningLegalityError(
                    "trace-time atomic units require an explicit ImportanceResult"
                )
            scored.append(unit)
            continue
        missing = [
            stable_id
            for stable_id in source_ids
            if stable_id not in importance_result.normalized_scores
            or stable_id not in importance_result.raw_scores
        ]
        if not source_ids or missing:
            raise PruningLegalityError(
                f"atomic unit {getattr(unit, 'stable_id', '<unknown>')} has missing importance scores: {missing}"
            )
        normalized_score = sum(
            float(importance_result.normalized_scores[stable_id]) for stable_id in source_ids
        ) / len(source_ids)
        raw_score = sum(float(importance_result.raw_scores[stable_id]) for stable_id in source_ids) / len(source_ids)
        scored.append(
            AtomicPruneUnit(
                scope_id=str(unit.scope_id),
                root_module_path=str(unit.root_module_path),
                root_axis=str(unit.root_axis),
                root_indices=list(unit.root_indices),
                source_coupled_unit_ids=source_ids,
                normalized_score=normalized_score,
                raw_score=raw_score,
                parameter_cost=(
                    sum(int(importance_result.unit_parameter_costs.get(stable_id, 0)) for stable_id in source_ids)
                    or int(getattr(unit, "parameter_cost", 0))
                ),
                channel_cost=int(getattr(unit, "channel_cost", max(len(unit.root_indices), 1))),
                protected=bool(getattr(unit, "protected", False)),
                protection_reason=str(getattr(unit, "protection_reason", "")),
                group_keep_map=dict(getattr(unit, "group_keep_map", {})),
                group_prune_map=dict(getattr(unit, "group_prune_map", {})),
                constraints=dict(getattr(unit, "constraints", {})),
                metadata={
                    **dict(getattr(unit, "metadata", {})),
                    "closure_members": [
                        member.to_dict() if hasattr(member, "to_dict") else dict(vars(member))
                        for member in getattr(unit, "members", ())
                    ],
                    "importance_mode": importance_result.mode,
                    "importance_normalization": importance_result.normalization,
                    "importance_implementation_version": importance_result.implementation_version,
                },
            )
        )

    return select_global_units(
        scored,
        channel_budget=channel_budget,
        parameter_budget=parameter_budget,
        grouped_config=grouped_config,
        selection_config=selection_config,
        alignment_config=alignment_config,
    )


__all__ = [
    "build_physical_pruning_plan",
    "build_physical_structure_snapshot",
    "compute_physical_hashes",
    "estimate_physical_parameter_count",
    "legalize_pruning_plan",
    "load_model",
    "materialize_pruning",
    "replay_pruning",
    "score_pruning_units",
    "select_pruning_request",
    "validate_physical_model",
]
