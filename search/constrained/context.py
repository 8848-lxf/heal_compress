"""Constrained context projection and INT8 sensitivity allowlist."""

from __future__ import annotations

import copy
from dataclasses import dataclass, is_dataclass, replace
from typing import Any, Sequence

import torch.nn as nn

from ..anchors.bops_021 import quantization_perturbation_metrics
from ..pruning_space.action_catalog import build_pruning_action_catalog


@dataclass(frozen=True)
class PrecisionSensitivityRecord:
    group_id: str
    module_paths: tuple[str, ...]
    canonical_macs: float
    taylor_fisher_perturbation: float
    sqnr_loss: float
    sensitivity_prior: float
    legal_int8: bool
    rejection_reason: str = ""
    saturation_ratio: float = 0.0
    sensitivity_prior_reasons: tuple[str, ...] = ()

    @property
    def ranking_score(self) -> float:
        loss = (
            float(self.taylor_fisher_perturbation) + float(self.sqnr_loss)
        ) * float(self.sensitivity_prior)
        return loss / max(float(self.canonical_macs), 1.0e-30)


def select_int8_allowlist(
    records: Sequence[PrecisionSensitivityRecord],
    *,
    priority_group_id: str,
    allowed_module_prefixes: Sequence[str],
    max_groups: int,
) -> dict[str, Any]:
    rows = list(records)
    total_macs = sum(max(0.0, float(row.canonical_macs)) for row in rows)
    if total_macs <= 0.0:
        raise ValueError("precision_sensitivity_canonical_MACs_empty")
    prefixes = tuple(str(value) for value in allowed_module_prefixes)
    eligible: list[PrecisionSensitivityRecord] = []
    payload: list[dict[str, Any]] = []
    for row in rows:
        module_allowed = bool(row.module_paths) and all(
            any(str(module) == prefix or str(module).startswith(prefix) for prefix in prefixes)
            for module in row.module_paths
        )
        reason = str(row.rejection_reason or "")
        legal = bool(row.legal_int8) and not reason
        if legal and not module_allowed:
            legal = False
            reason = "module_not_in_late_low_sensitivity_allowlist"
        if legal:
            eligible.append(row)
        payload.append(
            {
                "group_id": str(row.group_id),
                "module_paths": list(row.module_paths),
                "canonical_MAC": float(row.canonical_macs),
                "MAC_share": float(row.canonical_macs) / total_macs,
                "taylor_fisher_perturbation": float(row.taylor_fisher_perturbation),
                "SQNR_loss": float(row.sqnr_loss),
                "sensitivity_prior": float(row.sensitivity_prior),
                "ranking_score": float(row.ranking_score),
                "legal_INT8_status": legal,
                "rejection_reason": reason,
                "saturation_ratio": float(row.saturation_ratio),
                "sensitivity_prior_reasons": list(row.sensitivity_prior_reasons),
            }
        )
    priority = str(priority_group_id)
    ordered = sorted(
        eligible,
        key=lambda row: (
            0 if str(row.group_id) == priority else 1,
            float(row.ranking_score),
            str(row.group_id),
        ),
    )
    selected = ordered[: max(0, int(max_groups))]
    selected_ids = [str(row.group_id) for row in selected]
    if priority not in selected_ids:
        raise RuntimeError(f"priority_INT8_group_not_eligible:{priority}")
    selected_set = set(selected_ids)
    for row in payload:
        row["selected"] = row["group_id"] in selected_set
        if row["legal_INT8_status"] and not row["selected"]:
            row["rejection_reason"] = "ranked_below_allowlist_limit"
    return {
        "status": "passed",
        "priority_group_id": priority,
        "selected_group_ids": selected_ids,
        "selected_group_count": len(selected_ids),
        "max_groups": int(max_groups),
        "canonical_MAC_total": total_macs,
        "allowed_module_prefixes": list(prefixes),
        "groups": sorted(payload, key=lambda row: row["group_id"]),
    }


def _sensitivity_prior(module_paths: Sequence[str]) -> tuple[float, tuple[str, ...]]:
    text = " ".join(str(value).lower() for value in module_paths)
    rules = (
        (("pillar_vfe", "encoder_m1"), 8.0, "pillar_vfe_protection"),
        (("backbone_m1", "resnet.layer0"), 6.0, "early_backbone_protection"),
        (("resnet.layer1", "stage1"), 3.0, "stage1_sensitivity"),
        (("shrink",), 6.0, "shrink_sensitivity"),
        (("cls_head", "reg_head", "dir_head", "single_head"), 8.0, "head_protection"),
    )
    multiplier = 1.0
    reasons: list[str] = []
    for tokens, value, reason in rules:
        if any(token in text for token in tokens):
            multiplier = max(multiplier, value)
            reasons.append(reason)
    return multiplier, tuple(reasons)


def measure_precision_sensitivity(
    context: Any,
    *,
    fisher_statistics: Any,
    baseline_runtime_shapes: Sequence[Any],
) -> list[PrecisionSensitivityRecord]:
    """Measure FP16-to-INT8 loss for every canonical precision group."""

    modules = dict(context.model.named_modules())
    macs_by_module: dict[str, float] = {}
    counted: set[tuple[str, int]] = set()
    for shape in baseline_runtime_shapes:
        key = (str(shape.module_path), int(getattr(shape, "call_index", 0)))
        if key in counted:
            continue
        counted.add(key)
        macs_by_module[key[0]] = macs_by_module.get(key[0], 0.0) + float(
            getattr(shape, "macs", 0.0) or 0.0
        )
    raw_rows: list[dict[str, Any]] = []
    for group in context.search_space.quantization_groups:
        group_macs = sum(
            float(macs_by_module.get(str(module_path), 0.0))
            for module_path in group.module_paths
        )
        if group_macs <= 0.0:
            group_macs = float(getattr(group, "baseline_macs", 0.0) or 0.0)
        if group_macs <= 0.0:
            raise RuntimeError(
                f"constrained_quantization_group_MACs_missing:{group.group_id}"
            )
        legal = not bool(group.protected) and "INT8" in group.allowed_precisions
        rejection_reason = ""
        if bool(group.protected):
            rejection_reason = str(group.protection_reason or "protected_precision_group")
        elif "INT8" not in group.allowed_precisions:
            rejection_reason = "INT8_not_legal"
        taylor = 0.0
        sqnr = 0.0
        saturated_values = 0.0
        parameter_values = 0.0
        if legal:
            for module_path in group.module_paths:
                module = modules.get(str(module_path))
                weight = getattr(module, "weight", None)
                if weight is None:
                    legal = False
                    rejection_reason = f"weighted_parameter_missing:{module_path}"
                    break
                output_axis = 1 if isinstance(module, nn.ConvTranspose2d) else 0
                metric = quantization_perturbation_metrics(
                    weight,
                    gradient=fisher_statistics.gradients.get(
                        f"{module_path}.weight"
                    ),
                    fisher=fisher_statistics.fisher_diag.get(
                        f"{module_path}.weight"
                    ),
                    output_axis=output_axis,
                )
                count = float(weight.numel())
                taylor += float(metric["taylor_loss"])
                sqnr += float(metric["sqnr_loss"])
                saturated_values += float(metric["saturation_ratio"]) * count
                parameter_values += count
        prior, prior_reasons = _sensitivity_prior(group.module_paths)
        raw_rows.append(
            {
                "group": group,
                "canonical_macs": group_macs,
                "taylor": taylor,
                "sqnr": sqnr,
                "prior": prior,
                "prior_reasons": prior_reasons,
                "legal": legal,
                "rejection_reason": rejection_reason,
                "saturation": saturated_values / max(parameter_values, 1.0),
            }
        )
    taylor_total = sum(float(row["taylor"]) for row in raw_rows if row["legal"])
    sqnr_total = sum(float(row["sqnr"]) for row in raw_rows if row["legal"])
    return [
        PrecisionSensitivityRecord(
            group_id=str(row["group"].group_id),
            module_paths=tuple(str(value) for value in row["group"].module_paths),
            canonical_macs=float(row["canonical_macs"]),
            taylor_fisher_perturbation=(
                float(row["taylor"]) / max(taylor_total, 1.0e-30)
            ),
            sqnr_loss=float(row["sqnr"]) / max(sqnr_total, 1.0e-30),
            sensitivity_prior=float(row["prior"]),
            legal_int8=bool(row["legal"]),
            rejection_reason=str(row["rejection_reason"]),
            saturation_ratio=float(row["saturation"]),
            sensitivity_prior_reasons=tuple(row["prior_reasons"]),
        )
        for row in raw_rows
    ]


def _replace_context(context: Any, **changes: Any) -> Any:
    if is_dataclass(context):
        return replace(context, **changes)
    result = copy.copy(context)
    for key, value in changes.items():
        setattr(result, key, value)
    return result


def apply_constrained_pruning_context(
    context: Any,
    *,
    allowed_root_patterns: Sequence[str],
    grouped_conv_mode: str,
    grouped_conv_align: int,
    grouped_allowed_channels_per_group: Sequence[int],
    allowed_precision_values: Sequence[str] | None = None,
) -> tuple[Any, dict[str, Any]]:
    patterns = tuple(str(value) for value in allowed_root_patterns)
    if not patterns:
        raise ValueError("constrained_pruning_root_patterns_empty")
    source = list(getattr(getattr(context, "trace_result", None), "atomic_prune_units", []) or [])
    selected = []
    prohibited_tokens = (
        "pillar_vfe",
        "scatter",
        "cls_head",
        "reg_head",
        "dir_head",
        "single_head",
        "functional_affine_grid",
        "shrink_conv.layers.0.double_conv.2",
    )
    for unit in source:
        path = str(getattr(unit, "root_module_path", ""))
        lower = path.lower()
        constraints = dict(getattr(unit, "constraints", {}) or {})
        if not any(path == pattern or path.startswith(pattern + ".") for pattern in patterns):
            continue
        if bool(getattr(unit, "protected", False)) or any(token in lower for token in prohibited_tokens):
            continue
        if constraints.get("grouped_conv") and not constraints.get("depthwise"):
            continue
        if str(getattr(unit, "root_axis", "")) not in {"out", "channel"}:
            continue
        if not list(getattr(unit, "root_indices", []) or []):
            continue
        selected.append(unit)
    selected = sorted(
        selected,
        key=lambda unit: (
            str(getattr(unit, "root_module_path", "")),
            min(int(value) for value in getattr(unit, "root_indices", []) or [0]),
            str(getattr(unit, "stable_id", "")),
        ),
    )
    if not selected:
        raise RuntimeError(f"constrained_pruning_roots_unmatched:{patterns}")
    selected_by_id = {
        str(getattr(unit, "stable_id", "")): unit
        for unit in selected
    }
    metadata = {
        unit_id: {
            "scope_id": str(getattr(unit, "scope_id", "")),
            "root_module_path": str(getattr(unit, "root_module_path", "")),
            "root_axis": str(getattr(unit, "root_axis", "")),
            "root_indices": [int(value) for value in getattr(unit, "root_indices", []) or []],
            "constraints": dict(getattr(unit, "constraints", {}) or {}),
            "normalized_score": float(getattr(unit, "normalized_score", 0.0)),
        }
        for unit_id, unit in selected_by_id.items()
    }
    allowed_precision = {
        str(value).upper() for value in (allowed_precision_values or ())
    }
    quantization_groups = context.search_space.quantization_groups
    if allowed_precision:
        projected_groups = []
        for group in quantization_groups:
            projected = tuple(
                precision
                for precision in group.allowed_precisions
                if str(precision).upper() in allowed_precision
            )
            if group.protected:
                projected = tuple(
                    precision
                    for precision in projected
                    if str(precision).upper() == "FP16"
                ) or ("FP16",)
            if not projected:
                raise RuntimeError(
                    f"constrained_precision_group_has_no_allowed_value:{group.group_id}"
                )
            projected_groups.append(
                replace(group, allowed_precisions=projected)
            )
        quantization_groups = tuple(projected_groups)
    search_space = replace(
        context.search_space,
        pruning_unit_ids=list(selected_by_id),
        pruning_unit_metadata=metadata,
        protected_pruning_unit_ids=set(),
        quantization_groups=quantization_groups,
    )
    catalog = build_pruning_action_catalog(
        selected,
        grouped_conv_mode=str(grouped_conv_mode),
        grouped_conv_align=int(grouped_conv_align),
        grouped_allowed_channels_per_group=[
            int(value) for value in grouped_allowed_channels_per_group
        ],
    )
    report = {
        "status": "passed",
        "source": "existing_formal_trace_atomic_prune_units",
        "tracer_modified": False,
        "allowed_root_patterns": list(patterns),
        "source_atomic_unit_count": len(source),
        "selected_atomic_unit_count": len(selected),
        "selected_pruning_unit_ids": list(search_space.pruning_unit_ids),
        "selected_root_paths": sorted(
            {str(getattr(unit, "root_module_path", "")) for unit in selected}
        ),
        "grouped_conv_domains_selected": 0,
        "protected_policy": "only_explicit_late_low_sensitivity_roots_are_exposed",
        "allowed_precision_values": sorted(allowed_precision),
        "precision_group_ids_unchanged": [
            group.group_id for group in quantization_groups
        ]
        == [group.group_id for group in context.search_space.quantization_groups],
    }
    return (
        _replace_context(
            context,
            atomic_prune_units=selected,
            pruning_action_catalog=catalog,
            search_space=search_space,
        ),
        report,
    )
