"""Deterministic contracts for the BOPS-retention 0.21 anchor study.

This module deliberately contains no GA operators.  It constructs the three
controlled genotypes and provides fail-closed audits shared by the planner,
the production evaluator, and the report generator.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

import torch

from ..candidate import CandidateGenotype, normalize_precision
from ..canonicalization import SearchSpaceSpec
from ..hashing import canonical_json_hash


PRECISION_BITS = {"FP32": 32, "FP16": 16, "INT8": 8}


@dataclass(frozen=True)
class QuantizationSensitivity:
    group_id: str
    module_paths: tuple[str, ...]
    macs: float
    taylor_loss: float
    sqnr_loss: float
    prior_multiplier: float
    saturation_ratio: float

    @property
    def ranking_score(self) -> float:
        loss = (float(self.taylor_loss) + float(self.sqnr_loss)) * float(self.prior_multiplier)
        return loss / max(float(self.macs), 1.0e-12)

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "ranking_score": self.ranking_score}


def theoretical_bops_retention(precision: str, reference: str = "FP32") -> float:
    value = normalize_precision(precision)
    baseline = normalize_precision(reference, default="FP32")
    return (PRECISION_BITS[value] * PRECISION_BITS[value]) / (
        PRECISION_BITS[baseline] * PRECISION_BITS[baseline]
    )


def make_all_fp16_genotype(space: SearchSpaceSpec) -> CandidateGenotype:
    return CandidateGenotype(
        pruning_genes={unit_id: 1 for unit_id in space.pruning_unit_ids},
        precision_genes={group_id: "FP16" for group_id in space.precision_gene_ids},
        meta={"anchor": "fp16_baseline", "construction": "deterministic_all_keep_all_fp16"},
    )


def make_all_fp16_pruning_genotype(
    space: SearchSpaceSpec,
    *,
    pruned_unit_ids: Iterable[str],
) -> CandidateGenotype:
    selected = {str(value) for value in pruned_unit_ids}
    unknown = sorted(selected - set(space.pruning_unit_ids))
    if unknown:
        raise ValueError(f"unknown_pruning_unit_ids:{unknown}")
    return CandidateGenotype(
        pruning_genes={unit_id: 0 if unit_id in selected else 1 for unit_id in space.pruning_unit_ids},
        precision_genes={group_id: "FP16" for group_id in space.precision_gene_ids},
        meta={"anchor": "fp16_pruning", "construction": "deterministic_local_width_scan"},
    )


def make_mixed_no_prune_genotype(
    space: SearchSpaceSpec,
    *,
    int8_group_ids: Iterable[str],
) -> CandidateGenotype:
    selected = {str(value) for value in int8_group_ids}
    known = set(space.precision_gene_ids)
    unknown = sorted(selected - known)
    if unknown:
        raise ValueError(f"unknown_quantization_group_ids:{unknown}")
    group_by_id = {group.group_id: group for group in space.quantization_groups}
    illegal = sorted(
        group_id
        for group_id in selected
        if group_id in group_by_id
        and (group_by_id[group_id].protected or "INT8" not in group_by_id[group_id].allowed_precisions)
    )
    if illegal:
        raise ValueError(f"illegal_int8_group_ids:{illegal}")
    return CandidateGenotype(
        pruning_genes={unit_id: 1 for unit_id in space.pruning_unit_ids},
        precision_genes={group_id: "INT8" if group_id in selected else "FP16" for group_id in space.precision_gene_ids},
        meta={"anchor": "mixed_no_prune", "construction": "deterministic_mac_weighted_int8_prefix"},
    )


def validate_anchor_semantics(
    anchor: str,
    genotype: CandidateGenotype,
    space: SearchSpaceSpec,
) -> dict[str, Any]:
    name = str(anchor).strip().lower()
    pruning_values = {int(value) for value in genotype.pruning_genes.values()}
    precision_values = {normalize_precision(value) for value in genotype.precision_genes.values()}
    issues: list[str] = []
    if set(genotype.pruning_genes) != set(space.pruning_unit_ids):
        issues.append("pruning_gene_coverage_mismatch")
    if set(genotype.precision_genes) != set(space.precision_gene_ids):
        issues.append("precision_gene_coverage_mismatch")
    if name in {"fp16_baseline", "mixed_no_prune"} and pruning_values != {1}:
        issues.append("no_prune_anchor_contains_pruned_channel")
    if name in {"fp16_baseline", "fp16_pruning"} and precision_values != {"FP16"}:
        issues.append("all_fp16_anchor_contains_non_fp16_gene")
    if name == "fp16_pruning" and pruning_values == {1}:
        issues.append("fp16_pruning_anchor_is_all_keep")
    return {
        "passed": not issues,
        "status": "passed" if not issues else "anchor_semantics_failure",
        "anchor": name,
        "issues": issues,
        "pruned_unit_count": sum(int(value) == 0 for value in genotype.pruning_genes.values()),
        "precision_gene_counts": {
            precision: sum(normalize_precision(value) == precision for value in genotype.precision_genes.values())
            for precision in PRECISION_BITS
        },
    }


def _module_profile_from_genes(
    genes: Mapping[str, str],
    space: SearchSpaceSpec,
) -> dict[str, str]:
    if not space.quantization_groups:
        return {
            module: normalize_precision(genes.get(module, space.default_precision), default=space.default_precision)
            for module in space.precision_layer_ids
        }
    result: dict[str, str] = {}
    for group in space.quantization_groups:
        requested = normalize_precision(genes.get(group.group_id, space.default_precision), default=space.default_precision)
        for module in group.module_paths:
            result[module] = requested
    return dict(sorted(result.items()))


def precision_identity_audit(
    *,
    raw: CandidateGenotype,
    repaired: CandidateGenotype,
    requested_module_profile: Mapping[str, str],
    realized_module_profile: Mapping[str, str],
    space: SearchSpaceSpec,
) -> dict[str, Any]:
    profiles = {
        "raw_precision_genes_hash": _module_profile_from_genes(raw.precision_genes, space),
        "repaired_precision_genes_hash": _module_profile_from_genes(repaired.precision_genes, space),
        "requested_precision_profile_hash": {
            str(key): normalize_precision(value) for key, value in sorted(requested_module_profile.items())
        },
        "realized_precision_profile_hash": {
            str(key): normalize_precision(value) for key, value in sorted(realized_module_profile.items())
        },
    }
    hashes = {name: canonical_json_hash(profile) for name, profile in profiles.items()}
    raw_repaired_same = hashes["raw_precision_genes_hash"] == hashes["repaired_precision_genes_hash"]
    repaired_requested_same = hashes["repaired_precision_genes_hash"] == hashes["requested_precision_profile_hash"]
    requested_realized_same = hashes["requested_precision_profile_hash"] == hashes["realized_precision_profile_hash"]
    return {
        "passed": raw_repaired_same and repaired_requested_same and requested_realized_same,
        "raw_repair_changed_precision": not raw_repaired_same,
        "legalization_changed_precision": not repaired_requested_same,
        "precision_realization_changed": not requested_realized_same,
        "profile_hashes": hashes,
        "profiles": profiles,
    }


def validate_strongly_typed_profile(
    requested: Mapping[str, str],
    realized: Mapping[str, str],
    *,
    builder_flags: Mapping[str, Any],
) -> dict[str, Any]:
    requested_profile = {str(key): normalize_precision(value) for key, value in sorted(requested.items())}
    realized_profile = {str(key): normalize_precision(value) for key, value in sorted(realized.items())}
    issues: list[str] = []
    if not bool(builder_flags.get("strongly_typed", False)):
        issues.append("strongly_typed_disabled")
    if not bool(builder_flags.get("production_mode", False)):
        issues.append("production_mode_disabled")
    if requested_profile != realized_profile:
        issues.append("requested_realized_profile_mismatch")
    return {
        "passed": not issues,
        "status": "passed" if not issues else "precision_realization_failure",
        "issues": issues,
        "requested_profile_hash": canonical_json_hash(requested_profile),
        "realized_profile_hash": canonical_json_hash(realized_profile),
    }


def validate_plugin_gene_exclusion(
    space: SearchSpaceSpec,
    *,
    plugin_tokens: Sequence[str] = ("scatter", "pointpillar"),
) -> dict[str, Any]:
    tokens = tuple(str(value).lower() for value in plugin_tokens)
    values = [
        *space.precision_gene_ids,
        *space.precision_layer_ids,
        *(module for group in space.quantization_groups for module in group.module_paths),
    ]
    matches = sorted({value for value in values if any(token in str(value).lower() for token in tokens)})
    return {
        "passed": not matches,
        "status": "passed" if not matches else "plugin_quantization_gene_failure",
        "matching_gene_ids": matches,
    }


def realized_bops_gate(
    retention: float,
    *,
    target: float = 0.21,
    tolerance: float = 0.005,
) -> dict[str, Any]:
    lower = float(target) - float(tolerance)
    upper = float(target) + float(tolerance)
    passed = lower <= float(retention) <= upper
    return {
        "passed": passed,
        "status": "passed" if passed else "realized_BOPS_out_of_budget",
        "retention": float(retention),
        "target": float(target),
        "tolerance": float(tolerance),
        "legal_interval": [lower, upper],
    }


def decompose_bops_retention(
    *,
    raw_proxy: float,
    repaired_proxy: float,
    physical: float,
    realized: float,
) -> dict[str, float]:
    return {
        "raw_proxy_bops_retention": float(raw_proxy),
        "repaired_proxy_bops_retention": float(repaired_proxy),
        "physical_bops_retention": float(physical),
        "realized_bops_retention": float(realized),
        "delta_bops_channel_repair": float(repaired_proxy) - float(raw_proxy),
        "delta_bops_physical_materialization": float(physical) - float(repaired_proxy),
        "delta_bops_precision_realization": float(realized) - float(physical),
    }


def identify_low_damage_bops_path(
    strict_fp32: Mapping[str, Any],
    anchors: Mapping[str, Mapping[str, Any]],
    *,
    target: float = 0.21,
    tolerance: float = 0.005,
    expected_frames: int = 200,
) -> dict[str, Any]:
    """Classify observed Pareto points without creating an AP admission gate."""

    reference_keys = ("mAP", "AP@0.7", "forward_p50_ms")
    missing_reference = [key for key in reference_keys if strict_fp32.get(key) is None]
    if missing_reference:
        return {
            "identified": False,
            "status": "strict_fp32_metrics_missing",
            "missing_reference_metrics": missing_reference,
            "qualifying_anchors": [],
            "formal_ap_hard_gate_applied": False,
        }

    reference = {key: float(strict_fp32[key]) for key in reference_keys}
    observations: dict[str, Any] = {}
    qualifying: list[str] = []
    for name in sorted(anchors):
        row = anchors[name]
        required = (*reference_keys, "R_BOPS_realized")
        missing = [key for key in required if row.get(key) is None]
        issues: list[str] = []
        if not bool(row.get("passed", False)):
            issues.append("anchor_audit_failed")
        if int(row.get("evaluated", -1)) != int(expected_frames):
            issues.append("evaluated_frame_count_mismatch")
        if int(row.get("skipped", -1)) != 0:
            issues.append("evaluation_skip")
        if missing:
            issues.append("metrics_missing:" + ",".join(missing))

        deltas: dict[str, float] = {}
        bops_passed = False
        pareto_noninferior = False
        if not missing:
            deltas = {
                "delta_mAP_vs_strict_fp32": float(row["mAP"]) - reference["mAP"],
                "delta_AP07_vs_strict_fp32": float(row["AP@0.7"]) - reference["AP@0.7"],
                "delta_p50_ms_vs_strict_fp32": float(row["forward_p50_ms"])
                - reference["forward_p50_ms"],
            }
            bops_passed = realized_bops_gate(
                float(row["R_BOPS_realized"]),
                target=target,
                tolerance=tolerance,
            )["passed"]
            pareto_noninferior = (
                deltas["delta_mAP_vs_strict_fp32"] >= 0.0
                and deltas["delta_AP07_vs_strict_fp32"] >= 0.0
                and deltas["delta_p50_ms_vs_strict_fp32"] < 0.0
            )
        if not bops_passed:
            issues.append("realized_BOPS_out_of_budget")
        if not pareto_noninferior:
            issues.append("not_observed_pareto_noninferior")
        passed = not issues
        if passed:
            qualifying.append(str(name))
        observations[str(name)] = {
            "qualifies": passed,
            "issues": issues,
            **deltas,
        }

    return {
        "identified": bool(qualifying),
        "status": "observed_low_damage_path" if qualifying else "no_observed_low_damage_path",
        "basis": "relative_observed_mAP_AP07_p50_vs_same_run_strict_fp32",
        "qualifying_anchors": qualifying,
        "observations": observations,
        "formal_ap_hard_gate_applied": False,
    }


def select_mixed_precision_prefix(
    rows: Sequence[QuantizationSensitivity],
    *,
    target: float = 0.21,
    tolerance: float = 0.005,
    total_macs: float | None = None,
) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: (row.ranking_score, row.group_id))
    denominator_macs = (
        sum(float(row.macs) for row in ordered)
        if total_macs is None
        else float(total_macs)
    )
    if denominator_macs <= 0.0:
        raise ValueError("quantization_sensitivity_macs_empty")
    selected: list[QuantizationSensitivity] = []
    trace: list[dict[str, Any]] = []
    selected_macs = 0.0
    gate: dict[str, Any] | None = None
    for row in ordered:
        tentative_macs = selected_macs + float(row.macs)
        share = tentative_macs / denominator_macs
        retention = (
            theoretical_bops_retention("FP16") * (1.0 - share)
            + theoretical_bops_retention("INT8") * share
        )
        current = realized_bops_gate(retention, target=target, tolerance=tolerance)
        if retention < current["legal_interval"][0]:
            trace.append(
                {
                    "skipped_group_id": row.group_id,
                    "reason": "discrete_group_overshoots_lower_bops_bound",
                    "tentative_int8_macs_ratio": share,
                    "tentative_bops_retention": retention,
                }
            )
            continue
        selected.append(row)
        selected_macs = tentative_macs
        trace.append(
            {
                "added_group_id": row.group_id,
                "selected_group_count": len(selected),
                "int8_macs_ratio": share,
                "bops_retention": retention,
                "gate_passed": current["passed"],
            }
        )
        if current["passed"]:
            gate = current
            break
    if gate is None:
        gate = realized_bops_gate(0.25, target=target, tolerance=tolerance)
    share = sum(float(row.macs) for row in selected) / denominator_macs
    retention = (
        theoretical_bops_retention("FP16") * (1.0 - share)
        + theoretical_bops_retention("INT8") * share
    )
    return {
        **gate,
        "selected_group_ids": [row.group_id for row in selected],
        "selected_group_count": len(selected),
        "int8_macs_ratio": share,
        "bops_retention": retention,
        "ranking": [row.to_dict() for row in ordered],
        "selection_trace": trace,
    }


def quantization_perturbation_metrics(
    weight: torch.Tensor,
    *,
    gradient: torch.Tensor | None,
    fisher: torch.Tensor | None,
    output_axis: int,
) -> dict[str, float]:
    """Measure FP16-to-per-output-channel-INT8 perturbation for one weight."""

    value = weight.detach()
    fp16 = value.to(torch.float16).to(value.dtype)
    axis = int(output_axis)
    reduce_dims = tuple(index for index in range(value.ndim) if index != axis)
    amax = value.abs().amax(dim=reduce_dims, keepdim=True)
    scale = torch.where(amax > 0.0, amax / 127.0, torch.ones_like(amax))
    codes = torch.clamp(torch.round(value / scale), -127, 127)
    int8 = codes * scale
    delta = int8 - fp16
    taylor = 0.0
    if gradient is not None:
        grad = gradient.detach().to(device=value.device, dtype=value.dtype)
        taylor += float((grad * delta).abs().sum().cpu())
    if fisher is not None:
        diagonal = fisher.detach().to(device=value.device, dtype=value.dtype)
        taylor += 0.5 * float((diagonal * delta.pow(2)).sum().cpu())
    signal = float(fp16.pow(2).sum().cpu())
    noise = float(delta.pow(2).sum().cpu())
    return {
        "taylor_loss": taylor,
        "sqnr_loss": noise / max(signal, 1.0e-12),
        "saturation_ratio": float((codes.abs() >= 127.0).float().mean().cpu()),
    }


def validate_manifest_consistency(anchor_records: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    calibration = {
        str(name): str(record.get("calibration_manifest_hash", ""))
        for name, record in anchor_records.items()
    }
    validation = {
        str(name): str(record.get("validation_manifest_hash", ""))
        for name, record in anchor_records.items()
    }
    missing = sorted(
        name for name in anchor_records if not calibration[name] or not validation[name]
    )
    calibration_values = set(calibration.values()) - {""}
    validation_values = set(validation.values()) - {""}
    issues = []
    if missing:
        issues.append(f"manifest_hash_missing:{','.join(missing)}")
    if len(calibration_values) != 1:
        issues.append("calibration_manifest_hash_mismatch")
    if len(validation_values) != 1:
        issues.append("validation_manifest_hash_mismatch")
    return {
        "passed": not issues,
        "status": "passed" if not issues else "anchor_manifest_mismatch",
        "issues": issues,
        "calibration_manifest_hashes": calibration,
        "validation_manifest_hashes": validation,
    }
