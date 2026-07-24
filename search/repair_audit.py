"""Evidence helpers that distinguish canonicalization from actual repair."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .candidate import CandidateGenotype
from .canonicalization import SearchSpaceSpec, repair_genotype
from .hashing import canonical_json_hash


def genotype_gene_payload(candidate: CandidateGenotype) -> dict[str, Any]:
    """Return only mutable search decisions; provenance metadata is excluded."""

    return {
        "pruning_genes": dict(sorted(candidate.pruning_genes.items())),
        "pruning_width_genes": dict(
            sorted(candidate.pruning_width_genes.items())
        ),
        "precision_genes": dict(sorted(candidate.precision_genes.items())),
    }


def genotype_gene_hash(candidate: CandidateGenotype) -> str:
    return canonical_json_hash(genotype_gene_payload(candidate))


@dataclass(frozen=True)
class GenotypeCanonicalizationAudit:
    raw_genotype: dict[str, Any]
    canonical_genotype: dict[str, Any]
    raw_genotype_hash: str
    canonical_genotype_hash: str
    canonicalization_count: int
    canonicalization_actions: tuple[dict[str, Any], ...] = ()
    structural_repair_count: int = 0
    attention_width_repair_count: int = 0
    ffn_width_repair_count: int = 0
    cnn_width_repair_count: int = 0
    grouped_conv_repair_count: int = 0
    dependency_repair_count: int = 0
    precision_repair_count: int = 0
    qk_precision_repair_count: int = 0
    budget_projection_count: int = 0
    budget_projection_steps: int = 0
    repair_actions: tuple[dict[str, Any], ...] = ()
    phenotype_changed_by_repair: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def canonicalize_genotype_with_audit(
    raw: CandidateGenotype,
    space: SearchSpaceSpec,
) -> tuple[CandidateGenotype, GenotypeCanonicalizationAudit]:
    """Strictly canonicalize a legal candidate and explain expression-only work.

    Any actual structural or precision change raises in ``repair_genotype``;
    therefore a returned audit is repair-free by construction.
    """

    canonical = repair_genotype(raw, space)
    actions: list[dict[str, Any]] = []
    if space.pruning_domains and raw.pruning_genes:
        actions.append(
            {
                "type": "canonicalization",
                "field": "pruning_genes",
                "action": "drop_redundant_atomic_mask_coordinate",
                "count": len(raw.pruning_genes),
            }
        )
    for domain in space.pruning_domains:
        if domain.domain_id not in raw.pruning_width_genes:
            actions.append(
                {
                    "type": "canonicalization",
                    "field": f"pruning_width_genes.{domain.domain_id}",
                    "action": "fill_all_keep_default",
                    "value": int(domain.original_width),
                }
            )
    for gene_id in space.precision_gene_ids:
        if gene_id not in raw.precision_genes:
            actions.append(
                {
                    "type": "canonicalization",
                    "field": f"precision_genes.{gene_id}",
                    "action": "fill_documented_variable_default",
                    "value": canonical.precision_genes[gene_id],
                }
            )
    constant_profile = canonical.meta.get("constant_precision_group_profile", {})
    for group_id, precision in sorted(constant_profile.items()):
        actions.append(
            {
                "type": "canonicalization",
                "field": f"constant_precision_group_profile.{group_id}",
                "action": "materialize_protected_or_single_state_contract",
                "value": precision,
            }
        )
    raw_payload = genotype_gene_payload(raw)
    canonical_payload = genotype_gene_payload(canonical)
    audit = GenotypeCanonicalizationAudit(
        raw_genotype=raw_payload,
        canonical_genotype=canonical_payload,
        raw_genotype_hash=canonical_json_hash(raw_payload),
        canonical_genotype_hash=canonical_json_hash(canonical_payload),
        canonicalization_count=len(actions),
        canonicalization_actions=tuple(actions),
    )
    return canonical, audit


__all__ = [
    "GenotypeCanonicalizationAudit",
    "canonicalize_genotype_with_audit",
    "genotype_gene_hash",
    "genotype_gene_payload",
]
