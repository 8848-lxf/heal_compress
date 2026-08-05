"""Strict genotype validation and canonical phenotype construction.

Search operators are expected to be legal by construction.  The historical
``repair_genotype`` public name remains for compatibility, but a domain-width
or variable-precision value is never snapped to another search state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .candidate import CandidateGenotype, CandidatePhenotype, PrecisionDecision, normalize_precision
from .pruning_space.local_domains import expand_domain_width_genes, legalize_domain_width_genes
from .quantization_space.legalizer import legalize_group_precision_genes
from .quantization_space.types import QuantizationSearchGroup


@dataclass(frozen=True)
class SearchSpaceSpec:
    """Stable search-space metadata needed for repair and hashing."""

    pruning_unit_ids: list[str]
    precision_layer_ids: list[str]
    quantization_groups: tuple[QuantizationSearchGroup, ...] = ()
    pruning_domains: tuple[Any, ...] = ()
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

    def __post_init__(self) -> None:
        object.__setattr__(self, "pruning_unit_ids", sorted({str(value) for value in self.pruning_unit_ids}))
        object.__setattr__(self, "precision_layer_ids", sorted({str(value) for value in self.precision_layer_ids}))
        object.__setattr__(self, "quantization_groups", tuple(sorted(self.quantization_groups, key=lambda row: row.ordering)))
        object.__setattr__(
            self,
            "pruning_domains",
            tuple(sorted(self.pruning_domains, key=lambda row: str(getattr(row, "domain_id", "")))),
        )
        object.__setattr__(self, "pruning_unit_metadata", {str(key): dict(value) for key, value in self.pruning_unit_metadata.items()})
        object.__setattr__(self, "protected_pruning_unit_ids", {str(value) for value in self.protected_pruning_unit_ids})
        object.__setattr__(self, "default_precision", normalize_precision(self.default_precision))
        object.__setattr__(self, "builder_flags", dict(self.builder_flags))
        object.__setattr__(self, "plugin_hashes", {str(k): str(v) for k, v in self.plugin_hashes.items()})
        # Runtime-only memoization. Legal widths and frozen rankings make the
        # structural expansion a pure function of this tuple.
        object.__setattr__(self, "_width_expansion_cache", {})

    @property
    def precision_gene_ids(self) -> list[str]:
        if self.quantization_groups:
            # Protected and single-state groups are deployment constants, not
            # mutable loci.  They are materialized while constructing the
            # phenotype and therefore cannot be "repaired" after GA/Greedy
            # proposes an illegal QK/LayerNorm/residual precision.
            return [
                group.group_id
                for group in self.quantization_groups
                if not group.protected and len(set(group.allowed_precisions)) > 1
            ]
        return list(self.precision_layer_ids)

    @property
    def constant_precision_group_ids(self) -> list[str]:
        if not self.quantization_groups:
            return []
        variable = set(self.precision_gene_ids)
        return [
            group.group_id
            for group in self.quantization_groups
            if group.group_id not in variable
        ]

    @property
    def pruning_gene_ids(self) -> list[str]:
        if self.pruning_domains:
            return [
                str(domain.domain_id)
                for domain in self.pruning_domains
                if len(getattr(domain, "legal_widths", ())) > 1
            ]
        return list(self.pruning_unit_ids)


def repair_genotype(genotype: CandidateGenotype, space: SearchSpaceSpec) -> CandidateGenotype:
    """Validate and canonicalize a search genotype without changing its phenotype.

    Missing loci receive their documented baseline value.  Explicit unknown,
    protected, single-state, illegal-width, or disallowed precision loci fail
    closed instead of being rewritten.  Consequently any returned genotype is
    phenotype-equivalent to the submitted search decisions.
    """

    pruning = {}
    if not space.pruning_domains:
        for unit_id in space.pruning_unit_ids:
            pruning[unit_id] = 1 if unit_id in space.protected_pruning_unit_ids else int(genotype.pruning_genes.get(unit_id, 1))
            pruning[unit_id] = 1 if pruning[unit_id] else 0
    width_genes = (
        legalize_domain_width_genes(genotype.pruning_width_genes, space.pruning_domains)
        if space.pruning_domains
        else {}
    )
    if space.pruning_domains:
        # Atomic masks are derived only after width expansion. Keeping this raw
        # field empty prevents both a second coordinate and thousands of
        # redundant all-keep values from entering hash/GA bookkeeping.
        pruning = {}
    if space.quantization_groups:
        variable_ids = set(space.precision_gene_ids)
        explicit_ids = set(genotype.precision_genes)
        non_variable = sorted(explicit_ids - variable_ids)
        if non_variable:
            raise ValueError(
                "non_variable_precision_genes_must_not_enter_genotype:"
                f"{non_variable}"
            )
        legalization = legalize_group_precision_genes(
            genotype.precision_genes,
            space.quantization_groups,
            default_precision=space.default_precision,
        )
        precision = {
            group_id: legalization.stage1_legalized_group_profile[group_id]
            for group_id in space.precision_gene_ids
        }
        changed = {
            group_id: {
                "requested_precision": genotype.precision_genes.get(
                    group_id, space.default_precision
                ),
                "realized_precision": precision[group_id],
            }
            for group_id in space.precision_gene_ids
            if group_id in genotype.precision_genes
            and genotype.precision_genes[group_id] != precision[group_id]
        }
        if changed:
            raise ValueError(f"variable_precision_gene_requires_repair:{changed}")
        meta = {
            **dict(genotype.meta),
            "requested_group_profile": dict(legalization.requested_group_profile),
            "stage1_legalized_group_profile": dict(legalization.stage1_legalized_group_profile),
            "precision_fallback_report": dict(legalization.fallback_report),
            "constant_precision_group_profile": {
                group_id: legalization.stage1_legalized_group_profile[group_id]
                for group_id in space.constant_precision_group_ids
            },
        }
    else:
        precision = {
            layer_id: normalize_precision(genotype.precision_genes.get(layer_id, space.default_precision), default=space.default_precision)
            for layer_id in space.precision_layer_ids
        }
        meta = dict(genotype.meta)
    return CandidateGenotype(
        pruning_genes=pruning,
        precision_genes=precision,
        meta=meta,
        pruning_width_genes=width_genes,
    )


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
    if space.pruning_domains:
        width_key = tuple(
            (str(domain.domain_id), int(repaired.pruning_width_genes[domain.domain_id]))
            for domain in space.pruning_domains
        )
        cached_expansion = space._width_expansion_cache.get(width_key)
        if cached_expansion is None:
            expanded_ids, expanded_metadata = expand_domain_width_genes(
                repaired.pruning_width_genes,
                space.pruning_domains,
            )
            cached_expansion = (tuple(expanded_ids), expanded_metadata)
            space._width_expansion_cache[width_key] = cached_expansion
        pruned_unit_ids = list(cached_expansion[0])
        width_metadata = dict(cached_expansion[1])
        metadata.update(width_metadata)
        metadata["repair_version"] = "domain-width-legal-by-construction-v1"
    else:
        pruned_unit_ids = [
            unit_id for unit_id, keep in repaired.pruning_genes.items() if int(keep) == 0
        ]
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
        pruned_unit_ids=pruned_unit_ids,
        precision_profile=profile,
        pruning_policy_version=space.pruning_policy_version,
        precision_policy_version=space.precision_policy_version,
        metadata=metadata,
    )
