"""Decode legal retained widths using a precision-independent Taylor ranking."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from ..hashing import canonical_json_hash
from ..proxy.fisher_proxy import FisherStatistics
from ..proxy.parameter_slice_resolver import ParameterSlice
from ..space.legal_width_inventory import LegalWidthInventory


def _slice_union_flat_indices(
    shape: Sequence[int], parameter_slices: Sequence[ParameterSlice]
) -> torch.Tensor:
    """Return sorted CPU flat indices without allocating full-size masks."""

    dimensions = tuple(int(value) for value in shape)
    rows: list[torch.Tensor] = []
    for parameter_slice in parameter_slices:
        axis = int(parameter_slice.axis)
        if not 0 <= axis < len(dimensions):
            continue
        axis_size = dimensions[axis]
        selected = sorted(
            {
                int(value)
                for value in parameter_slice.indices
                if 0 <= int(value) < axis_size
            }
        )
        if not selected:
            continue
        outer = math.prod(dimensions[:axis])
        inner = math.prod(dimensions[axis + 1 :])
        outer_offsets = (
            torch.arange(outer, dtype=torch.long) * axis_size * inner
        ).view(-1, 1, 1)
        axis_offsets = (
            torch.tensor(selected, dtype=torch.long) * inner
        ).view(1, -1, 1)
        inner_offsets = torch.arange(inner, dtype=torch.long).view(1, 1, -1)
        rows.append((outer_offsets + axis_offsets + inner_offsets).reshape(-1))
    if not rows:
        return torch.empty(0, dtype=torch.long)
    return torch.unique(torch.cat(rows), sorted=True)


@dataclass(frozen=True)
class CanonicalPruneRankingRow:
    domain_id: str
    physical_group_id: int
    atomic_unit_id: str
    first_order_score: float
    second_order_score: float
    rank: int
    parameter_element_count: int
    duplicate_parameter_element_count: int
    dependency_member_count: int
    checkpoint_hash: str
    fisher_manifest_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain_id": self.domain_id,
            "physical_group_id": self.physical_group_id,
            "atomic_unit_id": self.atomic_unit_id,
            "first_order_score": self.first_order_score,
            "second_order_score": self.second_order_score,
            "rank": self.rank,
            "parameter_element_count": self.parameter_element_count,
            "duplicate_parameter_element_count": self.duplicate_parameter_element_count,
            "dependency_member_count": self.dependency_member_count,
            "checkpoint_hash": self.checkpoint_hash,
            "fisher_manifest_hash": self.fisher_manifest_hash,
        }


@dataclass(frozen=True)
class CanonicalPruneRanking:
    rows: tuple[CanonicalPruneRankingRow, ...]
    ranking_mode: str
    ranking_hash: str
    manifest: dict[str, Any]

    def to_decoder_rows(self) -> list[dict[str, Any]]:
        return [row.to_dict() for row in self.rows]


def build_canonical_prune_ranking(
    model: Any,
    *,
    statistics: FisherStatistics,
    unit_to_parameter_slices: Mapping[str, Sequence[ParameterSlice]],
    inventory: LegalWidthInventory,
    checkpoint_hash: str,
    fisher_manifest_hash: str,
    ranking_mode: str = "prune_only_second_order_fisher",
) -> CanonicalPruneRanking:
    """Score pure removal once, assigning overlapping elements one owner."""

    if ranking_mode not in {
        "prune_only_first_order",
        "prune_only_second_order_fisher",
    }:
        raise ValueError(f"unsupported_prune_ranking_mode:{ranking_mode}")
    parameters = dict(model.named_parameters())
    claimed = {
        name: torch.zeros(int(parameter.numel()), dtype=torch.bool)
        for name, parameter in parameters.items()
    }
    parameter_values = {
        name: parameter.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
        for name, parameter in parameters.items()
    }
    gradient_values = {
        name: value.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
        for name, value in statistics.gradients.items()
    }
    fisher_values_by_name = {
        name: value.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
        for name, value in statistics.fisher_diag.items()
    }
    domain_by_unit = {
        unit_id: (domain, int(group))
        for domain in inventory.domains
        for group, unit_ids in domain.physical_groups.items()
        for unit_id in unit_ids
    }
    raw_rows: list[dict[str, Any]] = []
    for unit_id in sorted(inventory.unit_ids):
        slices = tuple(unit_to_parameter_slices.get(unit_id, ()))
        if not slices:
            raise RuntimeError(f"canonical_prune_ranking_slice_missing:{unit_id}")
        first_total = 0.0
        second_term = 0.0
        element_count = 0
        duplicate_count = 0
        by_parameter: dict[str, list[ParameterSlice]] = defaultdict(list)
        for row in slices:
            by_parameter[row.parameter_name].append(row)
        for parameter_name, parameter_slices in sorted(by_parameter.items()):
            parameter = parameters.get(parameter_name)
            gradient = statistics.gradients.get(parameter_name)
            fisher = statistics.fisher_diag.get(parameter_name)
            if parameter is None:
                raise RuntimeError(
                    f"canonical_prune_ranking_parameter_missing:{parameter_name}"
                )
            if gradient is None:
                raise RuntimeError(
                    f"canonical_prune_ranking_gradient_missing:{parameter_name}"
                )
            if ranking_mode == "prune_only_second_order_fisher" and fisher is None:
                raise RuntimeError(
                    f"canonical_prune_ranking_fisher_missing:{parameter_name}"
                )
            selected = _slice_union_flat_indices(
                parameter.shape, parameter_slices
            )
            already_claimed = claimed[parameter_name][selected]
            duplicate_count += int(already_claimed.sum().item())
            owned = selected[~already_claimed]
            claimed[parameter_name][selected] = True
            if owned.numel() == 0:
                continue
            weight = parameter_values[parameter_name][owned]
            grad = gradient_values[parameter_name][owned]
            first_total += float((grad * weight).abs().sum().cpu())
            if ranking_mode == "prune_only_second_order_fisher":
                fisher_values = fisher_values_by_name[parameter_name][owned]
                second_term += 0.5 * float(
                    (fisher_values * weight.square()).sum().cpu()
                )
            element_count += int(owned.numel())
        if not math.isfinite(first_total + second_term):
            raise RuntimeError(f"canonical_prune_ranking_nonfinite:{unit_id}")
        domain, physical_group = domain_by_unit[unit_id]
        raw_rows.append(
            {
                "domain_id": domain.domain_id,
                "physical_group_id": physical_group,
                "atomic_unit_id": unit_id,
                "first_order_score": first_total,
                "second_order_score": first_total + second_term,
                "parameter_element_count": element_count,
                "duplicate_parameter_element_count": duplicate_count,
                "dependency_member_count": len(slices),
                "checkpoint_hash": str(checkpoint_hash),
                "fisher_manifest_hash": str(fisher_manifest_hash),
            }
        )
    score_key = (
        "first_order_score"
        if ranking_mode == "prune_only_first_order"
        else "second_order_score"
    )
    output_rows: list[CanonicalPruneRankingRow] = []
    grouped_rows: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        grouped_rows[(row["domain_id"], row["physical_group_id"])].append(row)
    for key in sorted(grouped_rows):
        ordered = sorted(
            grouped_rows[key],
            key=lambda row: (float(row[score_key]), row["atomic_unit_id"]),
        )
        output_rows.extend(
            CanonicalPruneRankingRow(rank=rank, **row)
            for rank, row in enumerate(ordered)
        )
    manifest = {
        "ranking_mode": ranking_mode,
        "ranking_depends_on_precision": False,
        "source_model": "original_unquantized",
        "overlap_policy": "global_parameter_element_first_stable_owner",
        "checkpoint_hash": str(checkpoint_hash),
        "fisher_manifest_hash": str(fisher_manifest_hash),
        "width_space_hash": inventory.width_space_hash,
        "unit_count": len(output_rows),
        "owned_parameter_element_count": sum(
            row.parameter_element_count for row in output_rows
        ),
        "overlap_parameter_element_count": sum(
            row.duplicate_parameter_element_count for row in output_rows
        ),
    }
    ranking_hash = canonical_json_hash(
        {**manifest, "rows": [row.to_dict() for row in output_rows]}
    )
    manifest["ranking_hash"] = ranking_hash
    return CanonicalPruneRanking(
        rows=tuple(output_rows),
        ranking_mode=ranking_mode,
        ranking_hash=ranking_hash,
        manifest=manifest,
    )


@dataclass(frozen=True)
class DecodedWidthStructure:
    width_genes: dict[str, int]
    keep_widths: dict[str, int]
    pruned_unit_ids: tuple[str, ...]
    group_mask: dict[str, int]
    group_keep_map: dict[int, tuple[int, ...]]
    group_prune_map: dict[int, tuple[int, ...]]
    per_domain_group_keep_map: dict[str, dict[int, tuple[int, ...]]]
    per_domain_group_prune_map: dict[str, dict[int, tuple[int, ...]]]
    group_keep_map_by_scope: dict[str, dict[int, tuple[int, ...]]]
    group_prune_map_by_scope: dict[str, dict[int, tuple[int, ...]]]
    width_vector_hash: str
    structure_hash: str
    physical_plan_hash: str
    ranking_hash: str
    inventory_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "width_genes": dict(self.width_genes),
            "keep_widths": dict(self.keep_widths),
            "pruned_unit_ids": list(self.pruned_unit_ids),
            "group_mask": dict(self.group_mask),
            "per_domain_group_keep_map": {
                domain: {str(group): list(values) for group, values in sorted(rows.items())}
                for domain, rows in sorted(self.per_domain_group_keep_map.items())
            },
            "per_domain_group_prune_map": {
                domain: {str(group): list(values) for group, values in sorted(rows.items())}
                for domain, rows in sorted(self.per_domain_group_prune_map.items())
            },
            "group_keep_map_by_scope": {
                scope: {str(group): list(values) for group, values in sorted(rows.items())}
                for scope, rows in sorted(self.group_keep_map_by_scope.items())
            },
            "group_prune_map_by_scope": {
                scope: {str(group): list(values) for group, values in sorted(rows.items())}
                for scope, rows in sorted(self.group_prune_map_by_scope.items())
            },
            "width_vector_hash": self.width_vector_hash,
            "structure_hash": self.structure_hash,
            "physical_plan_hash": self.physical_plan_hash,
            "ranking_hash": self.ranking_hash,
            "inventory_hash": self.inventory_hash,
        }


class FixedTaylorWidthDecoder:
    """Map widths to one immutable nested mask without observing precision."""

    def __init__(
        self,
        inventory: LegalWidthInventory,
        ranking_rows: Sequence[Mapping[str, Any]],
        *,
        ranking_mode: str = "prune_only_second_order_fisher",
    ) -> None:
        if ranking_mode not in {
            "prune_only_first_order",
            "prune_only_second_order_fisher",
        }:
            raise ValueError(f"unsupported_prune_ranking_mode:{ranking_mode}")
        self.inventory = inventory
        self.ranking_mode = ranking_mode
        score_name = (
            "first_order_score"
            if ranking_mode == "prune_only_first_order"
            else "second_order_score"
        )
        ranking: dict[tuple[str, int], list[str]] = defaultdict(list)
        normalized_rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        for raw in ranking_rows:
            row = dict(raw)
            domain_id = str(row["domain_id"])
            physical_group = int(row.get("physical_group_id", 0))
            unit_id = str(row["atomic_unit_id"])
            if unit_id in seen:
                raise ValueError(f"ranking_atomic_unit_duplicate:{unit_id}")
            seen.add(unit_id)
            normalized_rows.append(
                {
                    **row,
                    "domain_id": domain_id,
                    "physical_group_id": physical_group,
                    "atomic_unit_id": unit_id,
                    "_score": float(row[score_name]),
                }
            )
        normalized_rows.sort(
            key=lambda row: (
                row["domain_id"],
                row["physical_group_id"],
                row["_score"],
                row["atomic_unit_id"],
            )
        )
        for row in normalized_rows:
            ranking[(row["domain_id"], row["physical_group_id"])].append(
                row["atomic_unit_id"]
            )
        expected = set(inventory.unit_ids)
        if seen != expected:
            raise ValueError(
                f"ranking_inventory_mismatch:missing={sorted(expected-seen)}:unknown={sorted(seen-expected)}"
            )
        for domain in inventory.domains:
            for group, expected_units in domain.physical_groups.items():
                actual = ranking.get((domain.domain_id, int(group)), [])
                if set(actual) != set(expected_units):
                    raise ValueError(
                        f"ranking_physical_group_incomplete:{domain.domain_id}:{group}"
                    )
        self._ranking = {key: tuple(values) for key, values in ranking.items()}
        self.ranking_hash = canonical_json_hash(
            {
                "ranking_mode": ranking_mode,
                "inventory_hash": inventory.width_space_hash,
                "rows": [
                    {key: value for key, value in row.items() if key != "_score"}
                    for row in normalized_rows
                ],
            }
        )

    def decode(self, width_genes: Mapping[str, int]) -> DecodedWidthStructure:
        genes = {str(key): int(value) for key, value in sorted(width_genes.items())}
        if set(genes) != set(self.inventory.domain_ids):
            raise ValueError("width_decoder_domain_mismatch")
        mask = {unit_id: 1 for unit_id in self.inventory.unit_ids}
        keep_widths: dict[str, int] = {}
        domain_keep_maps: dict[str, dict[int, tuple[int, ...]]] = {}
        domain_prune_maps: dict[str, dict[int, tuple[int, ...]]] = {}
        keep_maps_by_scope: dict[str, dict[int, tuple[int, ...]]] = {}
        prune_maps_by_scope: dict[str, dict[int, tuple[int, ...]]] = {}
        pruned: set[str] = set()
        for domain in self.inventory.domains:
            index = genes[domain.domain_id]
            if not 0 <= index < len(domain.legal_keep_widths):
                raise ValueError(
                    f"width_gene_index_out_of_range:{domain.domain_id}:{index}"
                )
            keep_width = int(domain.legal_keep_widths[index])
            keep_widths[domain.domain_id] = keep_width
            per_group_keep: dict[int, tuple[int, ...]] = {}
            per_group_prune: dict[int, tuple[int, ...]] = {}
            for physical_group, group_units in sorted(domain.physical_groups.items()):
                original = (
                    domain.per_group_original_width
                    if domain.domain_kind == "grouped"
                    else domain.original_width
                )
                prune_count = original - keep_width
                ordered = self._ranking[(domain.domain_id, int(physical_group))]
                selected = tuple(ordered[:prune_count])
                selected_set = set(selected)
                if selected_set.intersection(domain.protected_unit_ids):
                    raise RuntimeError(f"fixed_ranking_selected_protected_unit:{domain.domain_id}")
                pruned.update(selected_set)
                for unit_id in selected:
                    mask[unit_id] = 0
                pruned_local = tuple(
                    sorted(domain.unit_local_indices[unit_id] for unit_id in selected)
                )
                kept_local = tuple(
                    index
                    for index in range(original)
                    if index not in set(pruned_local)
                )
                per_group_prune[int(physical_group)] = pruned_local
                per_group_keep[int(physical_group)] = kept_local
            domain_prune_maps[domain.domain_id] = per_group_prune
            domain_keep_maps[domain.domain_id] = per_group_keep
            if domain.domain_kind == "grouped":
                if domain.scope_id in keep_maps_by_scope:
                    raise RuntimeError(
                        f"grouped_scope_has_multiple_width_domains:{domain.scope_id}"
                    )
                keep_maps_by_scope[domain.scope_id] = per_group_keep
                prune_maps_by_scope[domain.scope_id] = per_group_prune
        width_hash = canonical_json_hash(genes)
        structure_payload = {
            "inventory_hash": self.inventory.width_space_hash,
            "pruned_unit_ids": sorted(pruned),
        }
        structure_hash = canonical_json_hash(structure_payload)
        physical_plan_hash = canonical_json_hash(
            {
                **structure_payload,
                "per_domain_group_keep_map": domain_keep_maps,
                "per_domain_group_prune_map": domain_prune_maps,
            }
        )
        only_domain = self.inventory.domains[0].domain_id if len(self.inventory.domains) == 1 else ""
        return DecodedWidthStructure(
            width_genes=genes,
            keep_widths=keep_widths,
            pruned_unit_ids=tuple(sorted(pruned)),
            group_mask={key: mask[key] for key in sorted(mask)},
            group_keep_map=dict(domain_keep_maps.get(only_domain, {})),
            group_prune_map=dict(domain_prune_maps.get(only_domain, {})),
            per_domain_group_keep_map=domain_keep_maps,
            per_domain_group_prune_map=domain_prune_maps,
            group_keep_map_by_scope=keep_maps_by_scope,
            group_prune_map_by_scope=prune_maps_by_scope,
            width_vector_hash=width_hash,
            structure_hash=structure_hash,
            physical_plan_hash=physical_plan_hash,
            ranking_hash=self.ranking_hash,
            inventory_hash=self.inventory.width_space_hash,
        )
