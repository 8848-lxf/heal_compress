"""Legal local-domain width genes backed by fixed atomic-unit rankings."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Any, Mapping, Sequence


SAFE_GROUPED_CHANNELS_PER_GROUP = (4, 8, 16, 32, 64, 128, 256, 512)


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class LocalPruningDomain:
    """One root-local pruning domain and every legal retained-width action.

    ``ordered_unit_ids`` is a fixed low-to-high task-loss Taylor ranking.  A
    width gene never carries a free channel mask: it selects one precomputed
    prefix of that ranking.  Grouped convolutions keep a separate ranking per
    original group and prune the same count from every group.
    """

    domain_id: str
    root_module_path: str
    root_axis: str
    scope_id: str
    kind: str
    original_width: int
    total_original_width: int
    ordered_unit_ids: tuple[str, ...]
    legal_widths: tuple[int, ...]
    width_to_pruned_unit_ids: dict[int, tuple[str, ...]] = field(default_factory=dict)
    unit_root_indices: dict[str, tuple[int, ...]] = field(default_factory=dict)
    ordered_unit_ids_by_group: dict[int, tuple[str, ...]] = field(default_factory=dict)
    group_local_indices: dict[int, dict[str, int]] = field(default_factory=dict)
    group_keep_maps: dict[int, dict[int, list[int]]] = field(default_factory=dict)
    group_prune_maps: dict[int, dict[int, list[int]]] = field(default_factory=dict)
    groups: int = 1
    ranking_method: str = "second_order_fisher_taylor"
    ranking_hash: str = ""
    unit_scores: dict[str, float] = field(default_factory=dict)
    constraints: dict[str, Any] = field(default_factory=dict)
    domain_type: str = ""
    model: str = ""
    module_path: str = ""
    family: str = ""
    block_path: str = ""
    dependency_members: tuple[dict[str, Any], ...] = ()
    ranking_groups: dict[str, Any] = field(default_factory=dict)
    latency_mapping: dict[str, Any] = field(default_factory=dict)
    precision_units: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        legacy_types = {
            "dense": "cnn_channel",
            "regular_grouped": "grouped_conv_channel",
        }
        resolved_type = str(self.domain_type or legacy_types.get(self.kind, self.kind))
        object.__setattr__(self, "domain_type", resolved_type)
        object.__setattr__(self, "model", str(self.model or ""))
        object.__setattr__(self, "module_path", str(self.module_path or self.root_module_path))
        object.__setattr__(self, "family", str(self.family or ""))
        object.__setattr__(self, "block_path", str(self.block_path or ""))
        object.__setattr__(
            self,
            "dependency_members",
            tuple(dict(value) for value in self.dependency_members),
        )
        object.__setattr__(self, "ranking_groups", dict(self.ranking_groups))
        object.__setattr__(self, "latency_mapping", dict(self.latency_mapping))
        object.__setattr__(self, "precision_units", tuple(str(value) for value in self.precision_units))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def width_semantics(self) -> str:
        if self.domain_type == "grouped_conv_channel":
            return "retained_channels_per_group"
        if self.domain_type == "attention_dh":
            return "retained_dimension_per_head"
        if self.domain_type == "ffn_hidden":
            return "retained_ffn_hidden_units"
        return "retained_output_channels"

    @property
    def current_width(self) -> int:
        """All-keep width; a candidate owns the mutable current state."""

        return int(self.original_width)

    @property
    def root_module(self) -> str:
        """Serializable root-module identity used by domain adapters."""

        return self.root_module_path

    def repair_width(self, retained_width: int) -> int:
        """Map a diagnostic raw width to the nearest legal deployment width.

        Formal candidate legalization remains strict.  This helper is used by
        initialization/repair code that explicitly opts into nearest repair;
        ties prefer the wider shape to avoid accidental over-pruning.
        """

        value = int(retained_width)
        return int(min(self.legal_widths, key=lambda width: (abs(int(width) - value), -int(width))))

    def decode_width(self, retained_width: int) -> dict[str, Any]:
        """Decode one legal scalar width to its immutable physical keep sets."""

        width = int(retained_width)
        self.pruned_unit_ids_for_width(width)
        if self.domain_type == "attention_dh":
            qk = self.ranking_groups.get("qk_low_to_high_by_head") or ()
            vo = self.ranking_groups.get("vo_low_to_high_by_head") or ()
            qk_keep = [sorted(int(value) for value in row[-width:]) for row in qk]
            vo_keep = [sorted(int(value) for value in row[-width:]) for row in vo]
            return {
                "domain_id": self.domain_id,
                "domain_type": self.domain_type,
                "module_path": self.module_path,
                "target_d_h": width,
                "heads": int(self.constraints.get("heads", len(qk_keep))),
                "qk_keep_by_head": qk_keep,
                "vo_keep_by_head": vo_keep,
                "shared_qkvo_index": bool(self.constraints.get("shared_qkvo_index", False)),
            }
        if self.domain_type == "ffn_hidden":
            order = tuple(int(value) for value in self.ranking_groups.get("ffn_low_to_high", ()))
            return {
                "domain_id": self.domain_id,
                "domain_type": self.domain_type,
                "module_path": self.module_path,
                "target_d_ff": width,
                "keep_indices": sorted(order[-width:]),
                "ffn_type": str(self.constraints.get("ffn_type", "standard")),
            }
        return {
            "domain_id": self.domain_id,
            "domain_type": self.domain_type,
            "retained_width": width,
            "pruned_unit_ids": list(self.pruned_unit_ids_for_width(width)),
        }

    def pruned_unit_ids_for_width(self, retained_width: int) -> tuple[str, ...]:
        width = int(retained_width)
        if width not in self.legal_widths or width not in self.width_to_pruned_unit_ids:
            raise ValueError(f"illegal_domain_width:{self.domain_id}:{width}:{list(self.legal_widths)}")
        return tuple(self.width_to_pruned_unit_ids[width])

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain_id": self.domain_id,
            "root_module_path": self.root_module_path,
            "root_axis": self.root_axis,
            "scope_id": self.scope_id,
            "kind": self.kind,
            "domain_type": self.domain_type,
            "model": self.model,
            "module_path": self.module_path,
            "family": self.family,
            "block_path": self.block_path,
            "original_width": self.original_width,
            "total_original_width": self.total_original_width,
            "groups": self.groups,
            "width_semantics": self.width_semantics,
            "ordered_unit_ids": list(self.ordered_unit_ids),
            "ordered_unit_ids_by_group": {
                str(group): list(values) for group, values in sorted(self.ordered_unit_ids_by_group.items())
            },
            "unit_root_indices": {
                unit_id: list(values) for unit_id, values in sorted(self.unit_root_indices.items())
            },
            "legal_widths": list(self.legal_widths),
            "width_to_pruned_unit_ids": {
                str(width): list(values) for width, values in sorted(self.width_to_pruned_unit_ids.items())
            },
            "group_keep_maps": {
                str(width): {str(group): list(values) for group, values in sorted(mapping.items())}
                for width, mapping in sorted(self.group_keep_maps.items())
            },
            "group_prune_maps": {
                str(width): {str(group): list(values) for group, values in sorted(mapping.items())}
                for width, mapping in sorted(self.group_prune_maps.items())
            },
            "ranking_method": self.ranking_method,
            "ranking_hash": self.ranking_hash,
            "unit_scores": dict(sorted(self.unit_scores.items())),
            "constraints": dict(self.constraints),
            "dependency_members": [dict(value) for value in self.dependency_members],
            "ranking_groups": dict(self.ranking_groups),
            "latency_mapping": dict(self.latency_mapping),
            "precision_units": list(self.precision_units),
            "metadata": dict(self.metadata),
        }


def legal_dense_widths(
    *,
    original_width: int,
    minimum_retained_ratio: float,
    alignment: int = 4,
) -> tuple[int, ...]:
    """Return dense Conv/Linear retained widths aligned to four plus original."""

    width = int(original_width)
    align = int(alignment)
    if width <= 0 or align <= 0:
        return ()
    minimum = max(1, int(math.ceil(width * float(minimum_retained_ratio))))
    values = {width}
    values.update(retained for retained in range(align, width + 1, align) if retained >= minimum)
    return tuple(sorted(values))


def legal_grouped_widths(
    *,
    original_channels_per_group: int,
    minimum_retained_ratio: float,
    allowed_channels_per_group: Sequence[int] = SAFE_GROUPED_CHANNELS_PER_GROUP,
) -> tuple[int, ...]:
    """Return safe retained channels/group choices without changing groups."""

    width = int(original_channels_per_group)
    if width <= 0:
        return ()
    minimum = max(1, int(math.ceil(width * float(minimum_retained_ratio))))
    values = {
        int(value)
        for value in allowed_channels_per_group
        if minimum <= int(value) <= width
    }
    values.add(width)
    return tuple(sorted(values))


def _score_for_unit(
    unit: Any,
    importance_scores: Mapping[str, float] | None,
) -> float:
    unit_id = str(getattr(unit, "stable_id", ""))
    if importance_scores is not None and unit_id in importance_scores:
        return float(importance_scores[unit_id])
    return float(getattr(unit, "normalized_score", 0.0))


def _dense_domain(
    key: tuple[str, str, str],
    rows: list[Any],
    *,
    importance_scores: Mapping[str, float] | None,
    ranking_method: str,
    minimum_retained_ratio: float,
    dense_alignment: int,
) -> LocalPruningDomain | None:
    root, axis, scope_id = key
    scores = {str(getattr(row, "stable_id")): _score_for_unit(row, importance_scores) for row in rows}
    ordered_rows = sorted(
        rows,
        key=lambda row: (
            scores[str(getattr(row, "stable_id"))],
            min(getattr(row, "root_indices", [0]) or [0]),
            str(getattr(row, "stable_id", "")),
        ),
    )
    unit_indices = {
        str(getattr(row, "stable_id")): tuple(sorted({int(value) for value in getattr(row, "root_indices", []) or []}))
        for row in ordered_rows
    }
    original_width = max([value for values in unit_indices.values() for value in values] or [-1]) + 1
    aligned = set(
        legal_dense_widths(
            original_width=original_width,
            minimum_retained_ratio=minimum_retained_ratio,
            alignment=dense_alignment,
        )
    )
    width_to_pruned: dict[int, tuple[str, ...]] = {original_width: ()}
    removed_indices: set[int] = set()
    prefix: list[str] = []
    for row in ordered_rows:
        unit_id = str(getattr(row, "stable_id"))
        prefix.append(unit_id)
        removed_indices.update(unit_indices[unit_id])
        retained = original_width - len(removed_indices)
        if retained in aligned:
            width_to_pruned.setdefault(retained, tuple(prefix))
    legal = tuple(sorted(width_to_pruned))
    if original_width <= 0:
        return None
    ranking_payload = {
        "method": ranking_method,
        "domain": [root, axis, scope_id],
        "ordered": [str(getattr(row, "stable_id")) for row in ordered_rows],
        "scores": scores,
    }
    return LocalPruningDomain(
        domain_id=f"{root}::{axis}",
        root_module_path=root,
        root_axis=axis,
        scope_id=scope_id,
        kind="dense",
        original_width=original_width,
        total_original_width=original_width,
        ordered_unit_ids=tuple(str(getattr(row, "stable_id")) for row in ordered_rows),
        legal_widths=legal,
        width_to_pruned_unit_ids=width_to_pruned,
        unit_root_indices=unit_indices,
        groups=1,
        ranking_method=ranking_method,
        ranking_hash=_stable_hash(ranking_payload),
        unit_scores=scores,
        constraints=dict(getattr(rows[0], "constraints", {}) or {}),
    )


def _grouped_domain(
    key: tuple[str, str, str],
    rows: list[Any],
    *,
    importance_scores: Mapping[str, float] | None,
    ranking_method: str,
    minimum_retained_ratio: float,
    allowed_channels_per_group: Sequence[int],
) -> LocalPruningDomain | None:
    root, axis, scope_id = key
    constraints = dict(getattr(rows[0], "constraints", {}) or {})
    groups = int(constraints.get("groups") or 0)
    width = int(constraints.get("channels_per_group") or constraints.get("channels_per_group_before") or 0)
    if groups <= 1 or width <= 0:
        return None
    scores = {str(getattr(row, "stable_id")): _score_for_unit(row, importance_scores) for row in rows}
    unit_indices: dict[str, tuple[int, ...]] = {}
    by_group: dict[int, list[Any]] = {group: [] for group in range(groups)}
    local_indices: dict[int, dict[str, int]] = {group: {} for group in range(groups)}
    for row in rows:
        unit_id = str(getattr(row, "stable_id"))
        indices = tuple(sorted({int(value) for value in getattr(row, "root_indices", []) or []}))
        unit_indices[unit_id] = indices
        if len(indices) != 1:
            continue
        group, local = divmod(indices[0], width)
        if group not in by_group or not 0 <= local < width:
            continue
        by_group[group].append(row)
        local_indices[group][unit_id] = local
    ordered_by_group: dict[int, tuple[str, ...]] = {}
    for group in range(groups):
        ordered = sorted(
            by_group[group],
            key=lambda row: (
                scores[str(getattr(row, "stable_id"))],
                local_indices[group][str(getattr(row, "stable_id"))],
                str(getattr(row, "stable_id")),
            ),
        )
        ordered_by_group[group] = tuple(str(getattr(row, "stable_id")) for row in ordered)
    legal_candidates = legal_grouped_widths(
        original_channels_per_group=width,
        minimum_retained_ratio=minimum_retained_ratio,
        allowed_channels_per_group=allowed_channels_per_group,
    )
    width_to_pruned: dict[int, tuple[str, ...]] = {}
    keep_maps: dict[int, dict[int, list[int]]] = {}
    prune_maps: dict[int, dict[int, list[int]]] = {}
    for retained in legal_candidates:
        remove_count = width - int(retained)
        if any(len(ordered_by_group[group]) < remove_count for group in range(groups)):
            continue
        selected = [
            unit_id
            for group in range(groups)
            for unit_id in ordered_by_group[group][:remove_count]
        ]
        width_to_pruned[int(retained)] = tuple(selected)
        selected_set = set(selected)
        keep_maps[int(retained)] = {
            group: sorted(
                local
                for unit_id, local in local_indices[group].items()
                if unit_id not in selected_set
            )
            for group in range(groups)
        }
        prune_maps[int(retained)] = {
            group: sorted(
                local
                for unit_id, local in local_indices[group].items()
                if unit_id in selected_set
            )
            for group in range(groups)
        }
    legal = tuple(sorted(width_to_pruned))
    if not legal or width not in legal:
        return None
    flattened = tuple(unit_id for group in range(groups) for unit_id in ordered_by_group[group])
    ranking_payload = {
        "method": ranking_method,
        "domain": [root, axis, scope_id],
        "ordered_by_group": ordered_by_group,
        "scores": scores,
    }
    return LocalPruningDomain(
        domain_id=f"{root}::{axis}",
        root_module_path=root,
        root_axis=axis,
        scope_id=scope_id,
        kind="regular_grouped",
        original_width=width,
        total_original_width=groups * width,
        ordered_unit_ids=flattened,
        legal_widths=legal,
        width_to_pruned_unit_ids=width_to_pruned,
        unit_root_indices=unit_indices,
        ordered_unit_ids_by_group=ordered_by_group,
        group_local_indices=local_indices,
        group_keep_maps=keep_maps,
        group_prune_maps=prune_maps,
        groups=groups,
        ranking_method=ranking_method,
        ranking_hash=_stable_hash(ranking_payload),
        unit_scores=scores,
        constraints=constraints,
    )


def build_local_pruning_domains(
    units: Sequence[Any],
    *,
    importance_scores: Mapping[str, float] | None = None,
    ranking_method: str = "second_order_fisher_taylor",
    minimum_retained_ratio: float = 0.10,
    dense_alignment: int = 4,
    grouped_allowed_channels_per_group: Sequence[int] = SAFE_GROUPED_CHANNELS_PER_GROUP,
) -> list[LocalPruningDomain]:
    """Build all legal root-local width genes from formal atomic units."""

    grouped_by_root: dict[tuple[str, str], list[Any]] = defaultdict(list)
    seen_units: set[str] = set()
    for unit in units:
        if bool(getattr(unit, "protected", False)):
            continue
        unit_id = str(getattr(unit, "stable_id", ""))
        if not unit_id:
            continue
        if unit_id in seen_units:
            raise RuntimeError(f"duplicate_atomic_unit_in_domain_builder:{unit_id}")
        seen_units.add(unit_id)
        key = (
            str(getattr(unit, "root_module_path", "")),
            str(getattr(unit, "root_axis", "")),
        )
        if key[0] and key[1]:
            grouped_by_root[key].append(unit)
    domains: list[LocalPruningDomain] = []
    for (root, axis), rows in sorted(grouped_by_root.items()):
        scopes = sorted({str(getattr(row, "scope_id", "")) for row in rows})
        scope_id = scopes[0] if len(scopes) == 1 else "|".join(scopes)
        key = (root, axis, scope_id)
        constraints = dict(getattr(rows[0], "constraints", {}) or {})
        if constraints.get("grouped_conv") and not constraints.get("depthwise"):
            domain = _grouped_domain(
                key,
                rows,
                importance_scores=importance_scores,
                ranking_method=ranking_method,
                minimum_retained_ratio=minimum_retained_ratio,
                allowed_channels_per_group=grouped_allowed_channels_per_group,
            )
        else:
            domain = _dense_domain(
                key,
                rows,
                importance_scores=importance_scores,
                ranking_method=ranking_method,
                minimum_retained_ratio=minimum_retained_ratio,
                dense_alignment=dense_alignment,
            )
        if domain is not None:
            domains.append(domain)
    unit_memberships: dict[str, str] = {}
    for domain in domains:
        for unit_id in domain.ordered_unit_ids:
            previous = unit_memberships.setdefault(unit_id, domain.domain_id)
            if previous != domain.domain_id:
                raise RuntimeError(f"atomic_unit_in_multiple_width_domains:{unit_id}:{previous}:{domain.domain_id}")
    return sorted(domains, key=lambda row: row.domain_id)


def legalize_domain_width_genes(
    width_genes: Mapping[str, int],
    domains: Sequence[LocalPruningDomain],
) -> dict[str, int]:
    """Validate width genes; missing genes mean all-keep, invalid widths fail."""

    by_id = {domain.domain_id: domain for domain in domains}
    unknown = sorted(set(str(key) for key in width_genes) - set(by_id))
    if unknown:
        raise ValueError(f"unknown_domain_width_genes:{unknown}")
    legalized: dict[str, int] = {}
    for domain in domains:
        width = int(width_genes.get(domain.domain_id, domain.original_width))
        if width not in domain.legal_widths:
            raise ValueError(
                f"illegal_domain_width_gene:{domain.domain_id}:{width}:{list(domain.legal_widths)}"
            )
        legalized[domain.domain_id] = width
    return legalized


def expand_domain_width_genes(
    width_genes: Mapping[str, int],
    domains: Sequence[LocalPruningDomain],
) -> tuple[list[str], dict[str, Any]]:
    """Expand legal widths to the exact immutable atomic-unit prune mask."""

    legalized = legalize_domain_width_genes(width_genes, domains)
    pruned: set[str] = set()
    group_keep_by_scope: dict[str, dict[int, list[int]]] = {}
    group_prune_by_scope: dict[str, dict[int, list[int]]] = {}
    domain_rows: dict[str, dict[str, Any]] = {}
    for domain in domains:
        width = legalized[domain.domain_id]
        selected = domain.pruned_unit_ids_for_width(width)
        overlap = pruned.intersection(selected)
        if overlap:
            raise RuntimeError(f"domain_width_expansion_overlap:{domain.domain_id}:{sorted(overlap)}")
        pruned.update(selected)
        if domain.kind == "regular_grouped":
            group_keep_by_scope[domain.scope_id] = {
                int(group): list(values)
                for group, values in domain.group_keep_maps[width].items()
            }
            group_prune_by_scope[domain.scope_id] = {
                int(group): list(values)
                for group, values in domain.group_prune_maps[width].items()
            }
        domain_rows[domain.domain_id] = {
            "domain_type": domain.domain_type,
            "model": domain.model,
            "module_path": domain.module_path,
            "family": domain.family,
            "retained_width": width,
            "original_width": domain.original_width,
            "width_semantics": domain.width_semantics,
            "pruned_unit_count": len(selected),
            "selection_commitment": _stable_hash(
                {
                    "domain_id": domain.domain_id,
                    "retained_width": width,
                    "ranking_hash": domain.ranking_hash,
                }
            ),
            "ranking_hash": domain.ranking_hash,
            "alignment_repair_applied": False,
            "decoded_width_state": domain.decode_width(width),
        }
    expansion_payload = {
        "policy_version": "legal-domain-width-fixed-ranking-v2",
        "domain_width_profile": legalized,
        "domains": domain_rows,
        "pruned_unit_count": len(pruned),
        "group_keep_map_by_scope": group_keep_by_scope,
        "group_prune_map_by_scope": group_prune_by_scope,
    }
    expansion_payload["domain_width_expansion_hash"] = _stable_hash(
        {
            "policy_version": expansion_payload["policy_version"],
            "domain_width_profile": legalized,
            "domain_selection_commitments": {
                domain_id: row["selection_commitment"]
                for domain_id, row in sorted(domain_rows.items())
            },
            "group_keep_map_by_scope": group_keep_by_scope,
            "group_prune_map_by_scope": group_prune_by_scope,
        }
    )
    return sorted(pruned), expansion_payload
