"""Repair, precision legalization, and canonical phenotype construction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .candidate import CandidateGenotype, CandidatePhenotype, PrecisionDecision, normalize_precision
from .quantization_space.legalizer import legalize_group_precision_genes
from .quantization_space.types import QuantizationSearchGroup


@dataclass(frozen=True)
class SearchSpaceSpec:
    """Stable search-space metadata needed for repair and hashing."""

    pruning_unit_ids: list[str]
    precision_layer_ids: list[str]
    quantization_groups: tuple[QuantizationSearchGroup, ...] = ()
    pruning_unit_metadata: dict[str, dict[str, Any]] = field(default_factory=dict)
    protected_pruning_unit_ids: set[str] = field(default_factory=set)
    default_precision: str = "FP16"
    pruning_policy_version: str = "formal-plan-first-v1"
    precision_policy_version: str = "explicit-qdq-canonical-fp16-int8-v1"
    trace_snapshot_hash: str = ""
    calibration_manifest_hash: str = ""
    onnx_export_config_hash: str = ""
    tensorrt_version: str = ""
    gpu_compute_capability: str = ""
    builder_flags: dict[str, Any] = field(default_factory=dict)
    plugin_hashes: dict[str, str] = field(default_factory=dict)
    code_commit: str = ""
    structure_gene_type: str = "coupled_channel_keep_mask"
    legal_width_inventory: Any | None = field(default=None, repr=False, compare=False)
    fixed_width_decoder: Any | None = field(default=None, repr=False, compare=False)
    precision_action_space: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "pruning_unit_ids", sorted({str(value) for value in self.pruning_unit_ids}))
        object.__setattr__(self, "precision_layer_ids", sorted({str(value) for value in self.precision_layer_ids}))
        object.__setattr__(self, "quantization_groups", tuple(sorted(self.quantization_groups, key=lambda row: row.ordering)))
        object.__setattr__(self, "pruning_unit_metadata", {str(key): dict(value) for key, value in self.pruning_unit_metadata.items()})
        object.__setattr__(self, "protected_pruning_unit_ids", {str(value) for value in self.protected_pruning_unit_ids})
        object.__setattr__(self, "default_precision", normalize_precision(self.default_precision))
        object.__setattr__(self, "builder_flags", dict(self.builder_flags))
        object.__setattr__(self, "plugin_hashes", {str(k): str(v) for k, v in self.plugin_hashes.items()})
        object.__setattr__(self, "code_commit", str(self.code_commit))
        structure_gene_type = str(self.structure_gene_type)
        if structure_gene_type not in {"coupled_channel_keep_mask", "legal_keep_width", "legal_pruning_action"}:
            raise ValueError(f"unsupported_structure_gene_type:{structure_gene_type}")
        object.__setattr__(self, "structure_gene_type", structure_gene_type)
        action_space = {
            str(group_id): tuple(normalize_precision(value) for value in values)
            for group_id, values in sorted(self.precision_action_space.items())
        }
        object.__setattr__(self, "precision_action_space", action_space)
        if structure_gene_type == "legal_keep_width":
            if self.legal_width_inventory is None:
                raise ValueError("legal_width_inventory_missing")
            if self.fixed_width_decoder is None:
                raise ValueError("fixed_width_decoder_missing")
            if not action_space:
                raise ValueError("legal_width_precision_action_space_missing")

    @property
    def precision_gene_ids(self) -> list[str]:
        if self.quantization_groups:
            return [group.group_id for group in self.quantization_groups]
        return list(self.precision_layer_ids)

    @property
    def legal_width_domain_ids(self) -> list[str]:
        if self.structure_gene_type != "legal_keep_width":
            return []
        return list(self.legal_width_inventory.domain_ids)


def repair_genotype(genotype: CandidateGenotype, space: SearchSpaceSpec) -> CandidateGenotype:
    """Repair raw genes without collapsing unknowns into persistent identity."""

    pruning = {}
    for unit_id in space.pruning_unit_ids:
        pruning[unit_id] = 1 if unit_id in space.protected_pruning_unit_ids else int(genotype.pruning_genes.get(unit_id, 1))
        pruning[unit_id] = 1 if pruning[unit_id] else 0
    if space.quantization_groups:
        legalization = legalize_group_precision_genes(
            genotype.precision_genes,
            space.quantization_groups,
            default_precision=space.default_precision,
        )
        precision = dict(legalization.stage1_legalized_group_profile)
        meta = {
            **dict(genotype.meta),
            "requested_group_profile": dict(legalization.requested_group_profile),
            "stage1_legalized_group_profile": dict(legalization.stage1_legalized_group_profile),
            "precision_fallback_report": dict(legalization.fallback_report),
        }
    else:
        precision = {
            layer_id: normalize_precision(genotype.precision_genes.get(layer_id, space.default_precision), default=space.default_precision)
            for layer_id in space.precision_layer_ids
        }
        meta = dict(genotype.meta)
    return CandidateGenotype(pruning_genes=pruning, precision_genes=precision, meta=meta)


def canonicalize_candidate(
    genotype: CandidateGenotype,
    space: SearchSpaceSpec,
    *,
    realized_precision: Mapping[str, tuple[str, str] | str] | None = None,
) -> CandidatePhenotype:
    """Build the phenotype after repair and precision legality fallback."""

    repaired = repair_genotype(genotype, space)
    realized = dict(realized_precision or {})
    profile: dict[str, PrecisionDecision] = {}
    metadata: dict[str, Any] = {"repair_version": "search-repair-v1", **dict(repaired.meta)}
    if space.quantization_groups:
        legalization = legalize_group_precision_genes(
            repaired.precision_genes,
            space.quantization_groups,
            default_precision=space.default_precision,
        )
        group_profile = dict(legalization.stage1_legalized_group_profile)
        group_fallback = dict(legalization.fallback_report)
        for group in space.quantization_groups:
            raw = realized.get(group.group_id, group_profile[group.group_id])
            if isinstance(raw, tuple):
                realized_value, fallback_reason = raw
            else:
                realized_value = str(raw)
                fallback_reason = group_fallback.get(group.group_id, {}).get("fallback_reason", "")
                if normalize_precision(realized_value) != group_profile[group.group_id] and not fallback_reason:
                    fallback_reason = "precision_policy_fallback"
            for module_path in group.module_paths:
                profile[module_path] = PrecisionDecision(
                    legalization.requested_group_profile[group.group_id],
                    str(realized_value),
                    fallback_reason,
                )
        metadata.update(legalization.to_dict())
    else:
        for layer_id in space.precision_layer_ids:
            requested = repaired.precision_genes[layer_id]
            raw = realized.get(layer_id, requested)
            if isinstance(raw, tuple):
                realized_value, fallback_reason = raw
            else:
                realized_value = str(raw)
                fallback_reason = "" if normalize_precision(realized_value) == requested else "precision_policy_fallback"
            profile[layer_id] = PrecisionDecision(requested, str(realized_value), fallback_reason)
    return CandidatePhenotype(
        pruned_unit_ids=[unit_id for unit_id, keep in repaired.pruning_genes.items() if int(keep) == 0],
        precision_profile=profile,
        pruning_policy_version=space.pruning_policy_version,
        precision_policy_version=space.precision_policy_version,
        metadata=metadata,
    )


def canonicalize_legal_width_candidate(
    genotype: Any,
    space: SearchSpaceSpec,
    *,
    realized_precision: Mapping[str, tuple[str, str] | str] | None = None,
) -> CandidatePhenotype:
    """Decode a legal-width chromosome without invoking normal repair."""

    from .hashing import canonical_json_hash

    if space.structure_gene_type != "legal_keep_width":
        raise ValueError("canonicalize_legal_width_candidate_requires_legal_width_space")
    genotype.validate(space.legal_width_inventory, space.precision_action_space)
    decoded = space.fixed_width_decoder.decode(genotype.width_genes)
    realized = dict(realized_precision or {})
    profile: dict[str, PrecisionDecision] = {}
    metadata: dict[str, Any] = {
        **dict(genotype.meta),
        **decoded.to_dict(),
        "normal_candidate_repair_invoked": False,
        "structure_gene_type": "legal_keep_width",
        "precision_hash": genotype.precision_hash,
        "phenotype_hash": canonical_json_hash(
            {
                "structure_hash": decoded.structure_hash,
                "precision_hash": genotype.precision_hash,
            }
        ),
    }
    if space.quantization_groups:
        legalization = legalize_group_precision_genes(
            genotype.precision_genes,
            space.quantization_groups,
            default_precision=space.default_precision,
        )
        if legalization.fallback_report:
            raise RuntimeError(
                f"legal_width_precision_action_not_deployable:{legalization.fallback_report}"
            )
        for group in space.quantization_groups:
            requested = legalization.stage1_legalized_group_profile[group.group_id]
            raw = realized.get(group.group_id, requested)
            if isinstance(raw, tuple):
                realized_value, fallback_reason = raw
            else:
                realized_value = str(raw)
                fallback_reason = "" if normalize_precision(realized_value) == requested else "precision_realization_changed"
            for module_path in group.module_paths:
                profile[module_path] = PrecisionDecision(
                    requested, str(realized_value), fallback_reason
                )
        metadata.update(legalization.to_dict())
    else:
        for layer_id in space.precision_layer_ids:
            requested = genotype.precision_genes[layer_id]
            raw = realized.get(layer_id, requested)
            if isinstance(raw, tuple):
                realized_value, fallback_reason = raw
            else:
                realized_value = str(raw)
                fallback_reason = "" if normalize_precision(realized_value) == requested else "precision_realization_changed"
            profile[layer_id] = PrecisionDecision(
                requested, str(realized_value), fallback_reason
            )
    return CandidatePhenotype(
        pruned_unit_ids=list(decoded.pruned_unit_ids),
        precision_profile=profile,
        pruning_policy_version="legal-width-fixed-prune-only-taylor-v1",
        precision_policy_version=space.precision_policy_version,
        metadata=metadata,
    )
