"""Independent legal-width and canonical precision chromosomes."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ..candidate import normalize_precision
from ..hashing import canonical_json_hash
from ..space.legal_width_inventory import LegalWidthInventory


@dataclass(frozen=True)
class LegalWidthGenotype:
    width_genes: dict[str, int]
    precision_genes: dict[str, str]
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "width_genes",
            {str(key): int(value) for key, value in sorted(self.width_genes.items())},
        )
        object.__setattr__(
            self,
            "precision_genes",
            {
                str(key): normalize_precision(value)
                for key, value in sorted(self.precision_genes.items())
            },
        )
        object.__setattr__(self, "meta", dict(self.meta))

    @property
    def width_vector_hash(self) -> str:
        return canonical_json_hash(self.width_genes)

    @property
    def precision_hash(self) -> str:
        return canonical_json_hash(self.precision_genes)

    @property
    def genotype_hash(self) -> str:
        return canonical_json_hash(
            {
                "encoding": "legal_keep_width",
                "width_vector_hash": self.width_vector_hash,
                "precision_hash": self.precision_hash,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "structure_gene_type": "legal_keep_width",
            "width_genes": dict(self.width_genes),
            "layer_bitwidth": dict(self.precision_genes),
            "precision_genes": dict(self.precision_genes),
            "genotype_hash": self.genotype_hash,
            "width_vector_hash": self.width_vector_hash,
            "precision_hash": self.precision_hash,
            "meta": dict(self.meta),
        }

    def validate(
        self,
        inventory: LegalWidthInventory,
        precision_actions: Mapping[str, Sequence[str]],
    ) -> None:
        expected_domains = set(inventory.domain_ids)
        actual_domains = set(self.width_genes)
        if actual_domains != expected_domains:
            missing = sorted(expected_domains - actual_domains)
            unknown = sorted(actual_domains - expected_domains)
            raise ValueError(f"width_gene_domain_mismatch:missing={missing}:unknown={unknown}")
        for domain in inventory.domains:
            index = int(self.width_genes[domain.domain_id])
            if not 0 <= index < len(domain.legal_keep_widths):
                raise ValueError(
                    f"width_gene_index_out_of_range:{domain.domain_id}:{index}"
                )
        expected_precision = set(str(key) for key in precision_actions)
        actual_precision = set(self.precision_genes)
        if expected_precision != actual_precision:
            raise ValueError("precision_gene_domain_mismatch")
        for group_id, actions in precision_actions.items():
            allowed = {normalize_precision(value) for value in actions}
            if self.precision_genes[str(group_id)] not in allowed:
                raise ValueError(f"precision_gene_action_illegal:{group_id}")


def random_legal_width_genotype(
    inventory: LegalWidthInventory,
    *,
    precision_actions: Mapping[str, Sequence[str]],
    rng: random.Random,
) -> LegalWidthGenotype:
    genotype = LegalWidthGenotype(
        width_genes={
            domain.domain_id: rng.randrange(len(domain.legal_keep_widths))
            for domain in inventory.domains
        },
        precision_genes={
            str(group_id): rng.choice(
                tuple(normalize_precision(value) for value in actions)
            )
            for group_id, actions in sorted(precision_actions.items())
        },
        meta={"created_by": "random_legal_width"},
    )
    genotype.validate(inventory, precision_actions)
    return genotype
