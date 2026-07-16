"""Enumerate adjacent legal compression actions without repair."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..encoding.legal_width_genotype import LegalWidthGenotype
from ..space.legal_width_inventory import LegalWidthInventory


_PRECISION_BITS = {"FP32": 32, "FP16": 16, "INT8": 8}


@dataclass(frozen=True)
class GreedyAction:
    kind: str
    gene_id: str
    from_value: int | str
    to_value: int | str

    def __post_init__(self) -> None:
        if self.kind not in {"width", "precision"}:
            raise ValueError(f"unsupported_greedy_action_kind:{self.kind}")
        if not str(self.gene_id):
            raise ValueError("greedy_action_gene_id_required")

    @property
    def action_id(self) -> str:
        return f"{self.kind}:{self.gene_id}:{self.from_value}->{self.to_value}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "kind": self.kind,
            "gene_id": self.gene_id,
            "from_value": self.from_value,
            "to_value": self.to_value,
        }


def _ordered_precision_actions(actions: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(dict.fromkeys(str(value).upper() for value in actions))
    unknown = sorted(set(normalized) - set(_PRECISION_BITS))
    if unknown:
        raise ValueError(f"unsupported_greedy_precision_actions:{unknown}")
    return tuple(sorted(normalized, key=lambda value: (-_PRECISION_BITS[value], value)))


def enumerate_legal_actions(
    genotype: LegalWidthGenotype,
    *,
    inventory: LegalWidthInventory,
    precision_actions: Mapping[str, Sequence[str]],
) -> list[tuple[GreedyAction, LegalWidthGenotype]]:
    """Return one-step legal width and precision compression successors."""

    genotype.validate(inventory, precision_actions)
    rows: list[tuple[GreedyAction, LegalWidthGenotype]] = []

    for domain_id in sorted(inventory.domain_ids):
        current = int(genotype.width_genes[domain_id])
        if current <= 0:
            continue
        action = GreedyAction("width", domain_id, current, current - 1)
        widths = dict(genotype.width_genes)
        widths[domain_id] = current - 1
        child = LegalWidthGenotype(
            width_genes=widths,
            precision_genes=genotype.precision_genes,
            meta={
                **genotype.meta,
                "created_by": "greedy_legal_action",
                "parent_genotype_hash": genotype.genotype_hash,
                "greedy_action_id": action.action_id,
            },
        )
        child.validate(inventory, precision_actions)
        rows.append((action, child))

    for group_id in sorted(precision_actions):
        ordered = _ordered_precision_actions(precision_actions[group_id])
        current = genotype.precision_genes[str(group_id)]
        try:
            index = ordered.index(current)
        except ValueError as exc:
            raise ValueError(
                f"greedy_precision_gene_not_deployable:{group_id}:{current}"
            ) from exc
        if index + 1 >= len(ordered):
            continue
        following = ordered[index + 1]
        action = GreedyAction(
            "precision", str(group_id), current, following
        )
        precisions = dict(genotype.precision_genes)
        precisions[str(group_id)] = following
        child = LegalWidthGenotype(
            width_genes=genotype.width_genes,
            precision_genes=precisions,
            meta={
                **genotype.meta,
                "created_by": "greedy_legal_action",
                "parent_genotype_hash": genotype.genotype_hash,
                "greedy_action_id": action.action_id,
            },
        )
        child.validate(inventory, precision_actions)
        rows.append((action, child))

    return sorted(rows, key=lambda row: row[0].action_id)
