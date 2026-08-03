"""One physical materializer for CNN and Transformer width domains.

The search codec exposes one scalar width per domain.  This module is the
matching Stage-2 decoder: it delegates CNN dependency closures to the existing
plan-first pruner, then rewrites Attention/FFN widths on that physical model.
No search-loop branch depends on the domain type.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import hashlib
import json
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import torch.nn as nn

from ..candidate import CandidatePhenotype
from .local_domains import (
    LocalPruningDomain,
    expand_domain_width_genes,
    legalize_domain_width_genes,
)
from .transformer_physical_pruner import (
    TransformerPhysicalPruneReport,
    materialize_transformer_widths,
    state_dict_shape_hash,
)

if TYPE_CHECKING:
    from ..adapters.pruning_adapter import FormalPruningAdapter


CNN_DOMAIN_TYPES = {"cnn_channel", "grouped_conv_channel"}
TRANSFORMER_DOMAIN_TYPES = {"attention_dh", "ffn_hidden"}


def _hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), default=str
        ).encode("utf-8")
    ).hexdigest()


def _parameter_count(model: nn.Module) -> int:
    return sum(int(parameter.numel()) for parameter in model.parameters())


def _cnn_realized_width(model: nn.Module, domain: LocalPruningDomain) -> int:
    module = model.get_submodule(domain.root_module_path)
    if domain.root_axis in {"out", "channel"}:
        realized = int(
            getattr(module, "out_channels", getattr(module, "out_features", 0))
        )
    elif domain.root_axis == "in":
        realized = int(
            getattr(module, "in_channels", getattr(module, "in_features", 0))
        )
    else:
        raise RuntimeError(
            f"unified_physical_cnn_axis_unsupported:{domain.domain_id}:"
            f"{domain.root_axis}"
        )
    if domain.domain_type == "grouped_conv_channel":
        groups = int(domain.groups)
        if groups <= 0 or realized % groups:
            raise RuntimeError(
                f"unified_physical_grouped_width_invalid:{domain.domain_id}:"
                f"{realized}:{groups}"
            )
        realized //= groups
    return realized


@dataclass(frozen=True)
class UnifiedPhysicalPruneReport:
    model: str
    requested_widths: dict[str, int]
    realized_widths: dict[str, int]
    original_parameter_count: int
    physical_parameter_count: int
    cnn_domain_count: int
    transformer_domain_count: int
    cnn_plan_entry_count: int
    cnn_validation_passed: bool
    transformer_report: dict[str, Any]
    state_dict_shape_hash: str
    structure_hash: str
    mask_only: bool
    hidden_padding: bool
    passed: bool
    issues: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class UnifiedPhysicalPruneResult:
    model: nn.Module
    report: UnifiedPhysicalPruneReport
    cnn_request: Any | None = None
    cnn_plan: Any | None = None
    cnn_ledger: Any | None = None
    cnn_snapshot: Any | None = None
    cnn_hashes: Any | None = None
    cnn_validation: Any | None = None


def materialize_unified_widths(
    model: nn.Module,
    atomic_units: Sequence[Any],
    domains: Sequence[LocalPruningDomain],
    width_genes: Mapping[str, int],
    *,
    model_name: str = "",
    formal_pruning_adapter: "FormalPruningAdapter | None" = None,
) -> UnifiedPhysicalPruneResult:
    """Physically apply every CNN/Attention/FFN domain exactly once."""

    supported = CNN_DOMAIN_TYPES | TRANSFORMER_DOMAIN_TYPES
    unsupported = sorted(
        {
            domain.domain_type
            for domain in domains
            if domain.domain_type not in supported
        }
    )
    if unsupported:
        raise RuntimeError(f"unified_physical_domain_type_unsupported:{unsupported}")
    ordered_domains = tuple(domains)
    requested = legalize_domain_width_genes(width_genes, ordered_domains)
    cnn_domains = tuple(
        domain for domain in ordered_domains if domain.domain_type in CNN_DOMAIN_TYPES
    )
    transformer_domains = tuple(
        domain
        for domain in ordered_domains
        if domain.domain_type in TRANSFORMER_DOMAIN_TYPES
    )
    original_count = _parameter_count(model)
    issues: list[str] = []
    cnn_request = cnn_plan = cnn_ledger = cnn_snapshot = cnn_hashes = cnn_validation = None

    changed_cnn = tuple(
        domain
        for domain in cnn_domains
        if requested[domain.domain_id] != int(domain.original_width)
    )
    if changed_cnn:
        cnn_widths = {
            domain.domain_id: requested[domain.domain_id] for domain in cnn_domains
        }
        pruned_unit_ids, metadata = expand_domain_width_genes(
            cnn_widths, cnn_domains
        )
        phenotype = CandidatePhenotype(
            pruned_unit_ids=pruned_unit_ids,
            pruning_policy_version="formal-plan-first-cnn-domain-width-v1",
            metadata=metadata,
        )
        atomic_by_id = {
            str(getattr(unit, "stable_id")): unit for unit in atomic_units
        }
        missing = sorted(set(pruned_unit_ids) - set(atomic_by_id))
        if missing:
            raise RuntimeError(
                f"unified_physical_cnn_atomic_units_missing:{missing[:16]}:"
                f"count={len(missing)}"
            )
        if formal_pruning_adapter is None:
            # Lazy import avoids a canonicalization -> pruning_space ->
            # adapters -> canonicalization initialization cycle.
            from ..adapters.pruning_adapter import FormalPruningAdapter

            formal = FormalPruningAdapter()
        else:
            formal = formal_pruning_adapter
        cnn_request = formal.request_from_phenotype(
            phenotype, tuple(atomic_by_id.values())
        )
        materialized = formal.materialize_from_request(model, cnn_request)
        physical = materialized["model"]
        cnn_plan = materialized["plan"]
        cnn_ledger = materialized["ledger"]
        cnn_snapshot = materialized["snapshot"]
        cnn_hashes = materialized["hashes"]
        cnn_validation = materialized["validation"]
        if not bool(getattr(cnn_validation, "passed", False)):
            issues.extend(
                f"cnn_physical_validation:{value}"
                for value in getattr(cnn_validation, "issues", ())
            )
    else:
        physical = copy.deepcopy(model)

    transformer_widths = {
        domain.domain_id: requested[domain.domain_id]
        for domain in transformer_domains
    }
    if transformer_domains:
        transformer_report: TransformerPhysicalPruneReport = (
            materialize_transformer_widths(
                physical,
                transformer_domains,
                transformer_widths,
                model_name=model_name,
            )
        )
        issues.extend(transformer_report.issues)
        transformer_payload = transformer_report.to_dict()
    else:
        transformer_payload = {
            "schema_version": "no-transformer-domains-v1",
            "passed": True,
            "requested_widths": {},
            "realized_widths": {},
            "operations": [],
        }

    realized = {
        domain.domain_id: _cnn_realized_width(physical, domain)
        for domain in cnn_domains
    }
    realized.update(dict(transformer_payload.get("realized_widths", {})))
    issues.extend(
        f"requested_realized_width_conflict:{domain_id}:"
        f"{requested[domain_id]}:{realized.get(domain_id)}"
        for domain_id in requested
        if requested[domain_id] != realized.get(domain_id)
    )
    physical_count = _parameter_count(physical)
    changed = any(
        requested[domain.domain_id] < int(domain.original_width)
        for domain in ordered_domains
    )
    if changed and physical_count >= original_count:
        issues.append("unified_physical_parameter_count_did_not_decrease")
    shape_hash = state_dict_shape_hash(physical)
    structure_payload = {
        "recipe": "cnn-transformer-unified-physical-width-v1",
        "model": str(model_name),
        "requested_widths": requested,
        "realized_widths": realized,
        "cnn_hashes": str(cnn_hashes),
        "transformer_structure_hash": transformer_payload.get("structure_hash", ""),
        "state_dict_shape_hash": shape_hash,
    }
    report = UnifiedPhysicalPruneReport(
        model=str(model_name),
        requested_widths=dict(requested),
        realized_widths=realized,
        original_parameter_count=original_count,
        physical_parameter_count=physical_count,
        cnn_domain_count=len(cnn_domains),
        transformer_domain_count=len(transformer_domains),
        cnn_plan_entry_count=len(getattr(cnn_plan, "entries", ())),
        cnn_validation_passed=(
            bool(getattr(cnn_validation, "passed", False))
            if changed_cnn
            else True
        ),
        transformer_report=transformer_payload,
        state_dict_shape_hash=shape_hash,
        structure_hash=_hash(structure_payload),
        mask_only=False,
        hidden_padding=False,
        passed=not issues,
        issues=tuple(issues),
    )
    return UnifiedPhysicalPruneResult(
        model=physical,
        report=report,
        cnn_request=cnn_request,
        cnn_plan=cnn_plan,
        cnn_ledger=cnn_ledger,
        cnn_snapshot=cnn_snapshot,
        cnn_hashes=cnn_hashes,
        cnn_validation=cnn_validation,
    )


__all__ = [
    "CNN_DOMAIN_TYPES",
    "TRANSFORMER_DOMAIN_TYPES",
    "UnifiedPhysicalPruneReport",
    "UnifiedPhysicalPruneResult",
    "materialize_unified_widths",
]
