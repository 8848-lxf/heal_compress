"""Prevalidated retained-width choices for local pruning domains."""

from __future__ import annotations

import math
import csv
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from ..hashing import canonical_json_hash


@dataclass(frozen=True)
class LegalWidthDomain:
    domain_id: str
    root_module: str
    root_axis: str
    scope_id: str
    domain_kind: str
    original_width: int
    legal_keep_widths: tuple[int, ...]
    minimum_width: int
    alignment: int
    group_count: int = 1
    per_group_original_width: int = 0
    protected: bool = False
    prunable: bool = True
    physical_replay_supported: bool = True
    exclusion_reason: str = ""
    unit_ids: tuple[str, ...] = ()
    protected_unit_ids: tuple[str, ...] = ()
    physical_groups: Mapping[int, tuple[str, ...]] = field(default_factory=dict)
    unit_local_indices: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.domain_kind not in {"dense", "grouped"}:
            raise ValueError(f"unsupported_legal_width_domain_kind:{self.domain_kind}")
        if not self.legal_keep_widths:
            raise ValueError(f"legal_width_domain_empty:{self.domain_id}")
        if tuple(sorted(set(self.legal_keep_widths))) != self.legal_keep_widths:
            raise ValueError(f"legal_widths_not_sorted_unique:{self.domain_id}")
        if self.domain_kind == "dense" and self.legal_keep_widths[-1] != self.original_width:
            raise ValueError(f"dense_original_width_missing:{self.domain_id}")
        if self.domain_kind == "grouped" and self.legal_keep_widths[-1] != self.per_group_original_width:
            raise ValueError(f"grouped_original_width_missing:{self.domain_id}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain_id": self.domain_id,
            "root_module": self.root_module,
            "root_axis": self.root_axis,
            "scope_id": self.scope_id,
            "domain_kind": self.domain_kind,
            "original_width": self.original_width,
            "legal_keep_widths": list(self.legal_keep_widths),
            "minimum_width": self.minimum_width,
            "alignment": self.alignment,
            "group_count": self.group_count,
            "per_group_original_width": self.per_group_original_width,
            "protected": self.protected,
            "prunable": self.prunable,
            "physical_replay_supported": self.physical_replay_supported,
            "exclusion_reason": self.exclusion_reason,
            "unit_ids": list(self.unit_ids),
            "protected_unit_ids": list(self.protected_unit_ids),
            "physical_groups": {
                str(group): list(unit_ids)
                for group, unit_ids in sorted(self.physical_groups.items())
            },
            "unit_local_indices": dict(sorted(self.unit_local_indices.items())),
        }


@dataclass(frozen=True)
class LegalWidthInventory:
    domains: tuple[LegalWidthDomain, ...]
    width_space_hash: str
    inventory_version: str = "legal-keep-width-v1"

    @property
    def domain_ids(self) -> tuple[str, ...]:
        return tuple(domain.domain_id for domain in self.domains)

    @property
    def domains_by_id(self) -> dict[str, LegalWidthDomain]:
        return {domain.domain_id: domain for domain in self.domains}

    @property
    def unit_ids(self) -> tuple[str, ...]:
        return tuple(
            unit_id
            for domain in self.domains
            for unit_id in domain.unit_ids
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "inventory_version": self.inventory_version,
            "width_space_hash": self.width_space_hash,
            "domains": [domain.to_dict() for domain in self.domains],
        }


@dataclass(frozen=True)
class PreparedLegalWidthSearchSpace:
    search_space: Any
    inventory: LegalWidthInventory
    first_order_ranking: Any
    second_order_ranking: Any
    decoder: Any


def _precision_action_space(
    search_space: Any,
    requested_actions: Sequence[str],
) -> dict[str, tuple[str, ...]]:
    from ..candidate import normalize_precision

    requested = tuple(
        dict.fromkeys(normalize_precision(value) for value in requested_actions)
    )
    result: dict[str, tuple[str, ...]] = {}
    if search_space.quantization_groups:
        for group in search_space.quantization_groups:
            if group.protected:
                default = normalize_precision(
                    group.metadata.get("default_precision", search_space.default_precision)
                )
                actions = (default,)
            else:
                actions = tuple(
                    value for value in requested if value in group.allowed_precisions
                )
            if not actions:
                raise RuntimeError(
                    f"precision_group_has_no_deployable_action:{group.group_id}"
                )
            result[group.group_id] = actions
        return result
    for layer_id in search_space.precision_layer_ids:
        if not requested:
            raise RuntimeError(f"precision_layer_has_no_deployable_action:{layer_id}")
        result[layer_id] = requested
    return result


def prepare_legal_width_search_space(
    search_space: Any,
    *,
    model: Any,
    units: Sequence[Any],
    statistics: Any,
    unit_to_parameter_slices: Mapping[str, Sequence[Any]],
    checkpoint_hash: str,
    fisher_manifest_hash: str,
    requested_precision_actions: Sequence[str],
    minimum_retained_ratio: float = 0.10,
    minimum_retained_channels: int = 1,
    dense_alignment: int = 4,
    grouped_allowed_channels_per_group: Sequence[int] = (4, 8, 16, 32, 64, 128, 256, 512),
    per_domain_max_prune_rate: float = 0.80,
    allowlisted_domain_ids: set[str] | frozenset[str] | None = None,
    unsupported_domain_ids: set[str] | frozenset[str] = frozenset(),
) -> PreparedLegalWidthSearchSpace:
    """Bind a legal inventory and immutable rankings to an existing space."""

    from dataclasses import replace

    from ..decoding.fixed_taylor_width_decoder import (
        FixedTaylorWidthDecoder,
        build_canonical_prune_ranking,
    )

    inventory = build_legal_width_inventory(
        units,
        minimum_retained_ratio=minimum_retained_ratio,
        minimum_retained_channels=minimum_retained_channels,
        dense_alignment=dense_alignment,
        grouped_allowed_channels_per_group=grouped_allowed_channels_per_group,
        per_domain_max_prune_rate=per_domain_max_prune_rate,
        allowlisted_domain_ids=allowlisted_domain_ids,
        unsupported_domain_ids=unsupported_domain_ids,
    )
    first = build_canonical_prune_ranking(
        model,
        statistics=statistics,
        unit_to_parameter_slices=unit_to_parameter_slices,
        inventory=inventory,
        checkpoint_hash=checkpoint_hash,
        fisher_manifest_hash=fisher_manifest_hash,
        ranking_mode="prune_only_first_order",
    )
    second = build_canonical_prune_ranking(
        model,
        statistics=statistics,
        unit_to_parameter_slices=unit_to_parameter_slices,
        inventory=inventory,
        checkpoint_hash=checkpoint_hash,
        fisher_manifest_hash=fisher_manifest_hash,
        ranking_mode="prune_only_second_order_fisher",
    )
    decoder = FixedTaylorWidthDecoder(
        inventory,
        second.to_decoder_rows(),
        ranking_mode="prune_only_second_order_fisher",
    )
    actions = _precision_action_space(search_space, requested_precision_actions)
    prepared_space = replace(
        search_space,
        pruning_unit_ids=list(inventory.unit_ids),
        structure_gene_type="legal_keep_width",
        legal_width_inventory=inventory,
        fixed_width_decoder=decoder,
        precision_action_space=actions,
        pruning_policy_version="legal-width-fixed-prune-only-taylor-v1",
    )
    return PreparedLegalWidthSearchSpace(
        search_space=prepared_space,
        inventory=inventory,
        first_order_ranking=first,
        second_order_ranking=second,
        decoder=decoder,
    )


def write_legal_width_search_space_artifacts(
    prepared: PreparedLegalWidthSearchSpace,
    output_dir: str | Any,
) -> dict[str, str]:
    """Persist lightweight inventory and canonical ranking evidence."""

    import json
    from pathlib import Path

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    inventory_json = destination / "legal_width_inventory.json"
    inventory_csv = destination / "legal_width_inventory.csv"
    first_csv = destination / "canonical_prune_ranking_first_order.csv"
    second_csv = destination / "canonical_prune_ranking_second_order.csv"
    ranking_manifest = destination / "canonical_ranking_manifest.json"
    width_hash = destination / "width_space_hash.json"
    inventory_json.write_text(
        json.dumps(prepared.inventory.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    inventory_fields = [
        "domain_id",
        "root_module",
        "root_axis",
        "scope_id",
        "domain_kind",
        "original_width",
        "legal_keep_widths",
        "minimum_width",
        "alignment",
        "group_count",
        "per_group_original_width",
        "protected",
        "prunable",
        "physical_replay_supported",
        "exclusion_reason",
    ]
    with inventory_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=inventory_fields)
        writer.writeheader()
        for domain in prepared.inventory.domains:
            row = domain.to_dict()
            writer.writerow(
                {
                    key: (
                        json.dumps(row[key], sort_keys=True)
                        if isinstance(row[key], (list, dict))
                        else row[key]
                    )
                    for key in inventory_fields
                }
            )
    ranking_fields = list(prepared.first_order_ranking.rows[0].to_dict()) if prepared.first_order_ranking.rows else []
    for path, ranking in (
        (first_csv, prepared.first_order_ranking),
        (second_csv, prepared.second_order_ranking),
    ):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=ranking_fields)
            writer.writeheader()
            writer.writerows(row.to_dict() for row in ranking.rows)
    ranking_manifest.write_text(
        json.dumps(
            {
                "first_order": prepared.first_order_ranking.manifest,
                "second_order": prepared.second_order_ranking.manifest,
                "decoder_ranking_hash": prepared.decoder.ranking_hash,
                "precision_action_space": prepared.search_space.precision_action_space,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    width_hash.write_text(
        json.dumps(
            {"width_space_hash": prepared.inventory.width_space_hash},
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return {
        "inventory_json": str(inventory_json),
        "inventory_csv": str(inventory_csv),
        "first_order_ranking_csv": str(first_csv),
        "second_order_ranking_csv": str(second_csv),
        "ranking_manifest": str(ranking_manifest),
        "width_space_hash": str(width_hash),
    }


def _domain_key(unit: Any) -> tuple[str, str, str]:
    return (
        str(getattr(unit, "root_module_path", "")),
        str(getattr(unit, "root_axis", "")),
        str(getattr(unit, "scope_id", "")),
    )


def _stable_id(unit: Any) -> str:
    return str(getattr(unit, "stable_id"))


def _root_index(unit: Any) -> int:
    indices = [int(value) for value in getattr(unit, "root_indices", ())]
    if len(indices) != 1:
        raise ValueError(f"legal_width_requires_atomic_root_index:{_stable_id(unit)}")
    return indices[0]


def _minimum_keep(
    width: int,
    *,
    minimum_retained_ratio: float,
    minimum_retained_channels: int,
    per_domain_max_prune_rate: float,
    protected_count: int = 0,
) -> int:
    return max(
        1,
        int(minimum_retained_channels),
        int(protected_count),
        int(math.ceil(width * float(minimum_retained_ratio) - 1.0e-12)),
        int(math.ceil(width * (1.0 - float(per_domain_max_prune_rate)) - 1.0e-12)),
    )


def _dense_widths(width: int, *, minimum: int, alignment: int) -> tuple[int, ...]:
    values = {int(width)}
    values.update(
        retained
        for retained in range(int(alignment), int(width) + 1, int(alignment))
        if retained >= int(minimum)
    )
    return tuple(sorted(values))


def _grouped_widths(
    width: int,
    *,
    minimum: int,
    allowed: Iterable[int],
) -> tuple[int, ...]:
    values = {
        int(value)
        for value in allowed
        if int(minimum) <= int(value) <= int(width)
    }
    values.add(int(width))
    return tuple(sorted(values))


def build_legal_width_inventory(
    units: Sequence[Any],
    *,
    minimum_retained_ratio: float = 0.10,
    minimum_retained_channels: int = 1,
    dense_alignment: int = 4,
    grouped_allowed_channels_per_group: Sequence[int] = (4, 8, 16, 32, 64, 128, 256, 512),
    per_domain_max_prune_rate: float = 0.80,
    allowlisted_domain_ids: set[str] | frozenset[str] | None = None,
    unsupported_domain_ids: set[str] | frozenset[str] = frozenset(),
) -> LegalWidthInventory:
    """Enumerate only widths already legal under physical constraints."""

    if dense_alignment <= 0:
        raise ValueError("dense_alignment_must_be_positive")
    if not 0.0 <= float(per_domain_max_prune_rate) <= 1.0:
        raise ValueError("per_domain_max_prune_rate_out_of_range")
    grouped: dict[tuple[str, str, str], list[Any]] = defaultdict(list)
    for unit in units:
        key = _domain_key(unit)
        if not key[0] or not key[1]:
            raise ValueError(f"legal_width_unit_missing_domain:{_stable_id(unit)}")
        grouped[key].append(unit)

    domains: list[LegalWidthDomain] = []
    allowlist = None if allowlisted_domain_ids is None else set(allowlisted_domain_ids)
    unsupported = set(unsupported_domain_ids)
    for (root, axis, scope), rows in sorted(grouped.items()):
        domain_id = f"{root}::{axis}::{scope}"
        by_index = {_root_index(unit): unit for unit in rows}
        if len(by_index) != len(rows):
            raise ValueError(f"duplicate_atomic_root_index:{domain_id}")
        original_width = max(by_index) + 1
        all_indices_present = set(by_index) == set(range(original_width))
        constraints = dict(getattr(rows[0], "constraints", {}) or {})
        is_grouped = bool(constraints.get("grouped_conv")) and not bool(constraints.get("depthwise"))
        protected_ids = tuple(sorted(_stable_id(row) for row in rows if bool(getattr(row, "protected", False))))
        allowlisted = allowlist is None or domain_id in allowlist
        replay_supported = domain_id not in unsupported and all_indices_present
        forced_exclusion = ""
        if not allowlisted:
            forced_exclusion = "domain_not_allowlisted"
        elif not replay_supported:
            forced_exclusion = "physical_replay_unsupported"

        if is_grouped:
            group_count = int(constraints.get("groups") or 0)
            per_group_width = int(constraints.get("channels_per_group") or 0)
            if group_count <= 0 or per_group_width <= 0:
                raise ValueError(f"invalid_grouped_width_metadata:{domain_id}")
            if original_width != group_count * per_group_width:
                raise ValueError(f"incomplete_grouped_domain:{domain_id}")
            physical_groups: dict[int, tuple[str, ...]] = {}
            local_indices: dict[str, int] = {}
            protected_per_group: dict[int, int] = defaultdict(int)
            for physical_group in range(group_count):
                group_units: list[str] = []
                for local_index in range(per_group_width):
                    absolute = physical_group * per_group_width + local_index
                    unit = by_index[absolute]
                    unit_id = _stable_id(unit)
                    group_units.append(unit_id)
                    local_indices[unit_id] = local_index
                    if bool(getattr(unit, "protected", False)):
                        protected_per_group[physical_group] += 1
                physical_groups[physical_group] = tuple(group_units)
            minimum = _minimum_keep(
                per_group_width,
                minimum_retained_ratio=minimum_retained_ratio,
                minimum_retained_channels=minimum_retained_channels,
                per_domain_max_prune_rate=per_domain_max_prune_rate,
                protected_count=max(protected_per_group.values(), default=0),
            )
            widths = _grouped_widths(
                per_group_width,
                minimum=minimum,
                allowed=grouped_allowed_channels_per_group,
            )
            fully_protected = len(protected_ids) == original_width
            prunable = not fully_protected and not forced_exclusion and len(widths) > 1
            reason = (
                str(getattr(rows[0], "protection_reason", "")) or "protected_domain"
                if fully_protected
                else forced_exclusion
            )
            if not prunable:
                widths = (per_group_width,)
            domains.append(
                LegalWidthDomain(
                    domain_id=domain_id,
                    root_module=root,
                    root_axis=axis,
                    scope_id=scope,
                    domain_kind="grouped",
                    original_width=original_width,
                    legal_keep_widths=widths,
                    minimum_width=minimum,
                    alignment=1,
                    group_count=group_count,
                    per_group_original_width=per_group_width,
                    protected=fully_protected,
                    prunable=prunable,
                    physical_replay_supported=replay_supported,
                    exclusion_reason=reason,
                    unit_ids=tuple(_stable_id(by_index[index]) for index in range(original_width)),
                    protected_unit_ids=protected_ids,
                    physical_groups=physical_groups,
                    unit_local_indices=local_indices,
                )
            )
            continue

        minimum = _minimum_keep(
            original_width,
            minimum_retained_ratio=minimum_retained_ratio,
            minimum_retained_channels=minimum_retained_channels,
            per_domain_max_prune_rate=per_domain_max_prune_rate,
            protected_count=len(protected_ids),
        )
        widths = _dense_widths(original_width, minimum=minimum, alignment=dense_alignment)
        fully_protected = len(protected_ids) == original_width
        prunable = not fully_protected and not forced_exclusion and len(widths) > 1
        reason = (
            str(getattr(rows[0], "protection_reason", "")) or "protected_domain"
            if fully_protected
            else forced_exclusion
        )
        if not prunable:
            widths = (original_width,)
        domains.append(
            LegalWidthDomain(
                domain_id=domain_id,
                root_module=root,
                root_axis=axis,
                scope_id=scope,
                domain_kind="dense",
                original_width=original_width,
                legal_keep_widths=widths,
                minimum_width=minimum,
                alignment=int(dense_alignment),
                group_count=1,
                per_group_original_width=original_width,
                protected=fully_protected,
                prunable=prunable,
                physical_replay_supported=replay_supported,
                exclusion_reason=reason,
                unit_ids=tuple(_stable_id(by_index[index]) for index in range(original_width)),
                protected_unit_ids=protected_ids,
                physical_groups={0: tuple(_stable_id(by_index[index]) for index in range(original_width))},
                unit_local_indices={_stable_id(by_index[index]): index for index in range(original_width)},
            )
        )

    payload = [domain.to_dict() for domain in domains]
    return LegalWidthInventory(
        domains=tuple(domains),
        width_space_hash=canonical_json_hash(
            {"inventory_version": "legal-keep-width-v1", "domains": payload}
        ),
    )
