"""Physical Transformer-FFN hidden-width pruning for HEAL V2X-ViT."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn as nn

from ...candidate import CandidatePhenotype
from ...hashing import canonical_json_hash
from ...proxy.candidate_perturbation import pseudo_quantize_tensor
from ...pruning_space.local_domains import LocalPruningDomain
from ...pruning_space.unified_physical_pruner import materialize_unified_widths


@dataclass
class V2XViTPhysicalPruningResult:
    model: nn.Module
    snapshot: dict[str, Any]
    snapshot_hash: str
    parameter_count_before: int
    parameter_count_after: int


def _replace_submodule(model: nn.Module, path: str, replacement: nn.Module) -> None:
    parent_path, leaf = str(path).rsplit(".", 1)
    parent = model.get_submodule(parent_path)
    if leaf.isdigit() and isinstance(parent, (nn.Sequential, nn.ModuleList)):
        parent[int(leaf)] = replacement
    else:
        setattr(parent, leaf, replacement)


def _linear_like(
    source: nn.Linear,
    *,
    in_features: int,
    out_features: int,
) -> nn.Linear:
    return nn.Linear(
        int(in_features),
        int(out_features),
        bias=source.bias is not None,
        device=source.weight.device,
        dtype=source.weight.dtype,
    )


def materialize_v2xvit_ffn_pruning(
    model: nn.Module,
    phenotype: CandidatePhenotype,
    domains: Sequence[LocalPruningDomain],
) -> V2XViTPhysicalPruningResult:
    """Materialize exactly the immutable FFN domain-width phenotype."""

    domain_rows = {
        str(row.domain_id): row
        for row in domains
        if row.constraints.get("v2xvit_domain_kind")
        == "transformer_ffn_hidden_width"
    }
    if not domain_rows:
        raise RuntimeError("v2xvit_ffn_materializer_has_no_domains")
    known_units = {
        unit_id for domain in domain_rows.values() for unit_id in domain.ordered_unit_ids
    }
    unknown = sorted(set(phenotype.pruned_unit_ids) - known_units)
    if unknown:
        raise RuntimeError(f"v2xvit_ffn_materializer_unknown_units:{unknown[:8]}")
    width_profile = {
        str(key): int(value)
        for key, value in dict(
            phenotype.metadata.get("domain_width_profile") or {}
        ).items()
    }
    extra_domains = sorted(set(width_profile) - set(domain_rows))
    if extra_domains:
        raise RuntimeError(f"v2xvit_ffn_materializer_unknown_domains:{extra_domains}")

    physical = copy.deepcopy(model)
    before = sum(int(parameter.numel()) for parameter in physical.parameters())
    snapshots: list[dict[str, Any]] = []
    selected_global: set[str] = set()
    for domain_id, domain in sorted(domain_rows.items()):
        retained_width = int(width_profile.get(domain_id, domain.original_width))
        if retained_width not in domain.legal_widths:
            raise RuntimeError(
                f"v2xvit_ffn_materializer_illegal_width:{domain_id}:{retained_width}"
            )
        expected_units = set(domain.pruned_unit_ids_for_width(retained_width))
        observed_units = set(phenotype.pruned_unit_ids).intersection(
            domain.ordered_unit_ids
        )
        if observed_units != expected_units:
            raise RuntimeError(f"v2xvit_ffn_materializer_mask_width_mismatch:{domain_id}")
        selected_global.update(expected_units)
        pruned_indices = sorted(
            {
                index
                for unit_id in expected_units
                for index in domain.unit_root_indices[unit_id]
            }
        )
        keep_indices = [
            index
            for index in range(int(domain.original_width))
            if index not in set(pruned_indices)
        ]
        if len(keep_indices) != retained_width:
            raise RuntimeError(f"v2xvit_ffn_materializer_keep_count_mismatch:{domain_id}")

        first_path = str(domain.constraints["first_linear"])
        second_path = str(domain.constraints["second_linear"])
        first = physical.get_submodule(first_path)
        second = physical.get_submodule(second_path)
        if not isinstance(first, nn.Linear) or not isinstance(second, nn.Linear):
            raise RuntimeError(f"v2xvit_ffn_materializer_linear_missing:{domain_id}")
        index = torch.as_tensor(keep_indices, dtype=torch.long, device=first.weight.device)
        new_first = _linear_like(
            first,
            in_features=first.in_features,
            out_features=retained_width,
        )
        new_second = _linear_like(
            second,
            in_features=retained_width,
            out_features=second.out_features,
        )
        with torch.no_grad():
            new_first.weight.copy_(first.weight.index_select(0, index))
            if first.bias is not None and new_first.bias is not None:
                new_first.bias.copy_(first.bias.index_select(0, index))
            new_second.weight.copy_(second.weight.index_select(1, index))
            if second.bias is not None and new_second.bias is not None:
                new_second.bias.copy_(second.bias)
        _replace_submodule(physical, first_path, new_first)
        _replace_submodule(physical, second_path, new_second)
        snapshots.append(
            {
                "domain_id": domain_id,
                "first_linear": first_path,
                "second_linear": second_path,
                "original_width": int(domain.original_width),
                "retained_width": retained_width,
                "keep_indices": keep_indices,
                "prune_indices": pruned_indices,
                "pruned_unit_ids": sorted(expected_units),
                "ranking_hash": domain.ranking_hash,
            }
        )
    if selected_global != set(phenotype.pruned_unit_ids):
        raise RuntimeError("v2xvit_ffn_materializer_global_mask_mismatch")
    after = sum(int(parameter.numel()) for parameter in physical.parameters())
    if phenotype.pruned_unit_ids and after >= before:
        raise RuntimeError(f"v2xvit_ffn_parameter_count_not_reduced:{before}:{after}")
    snapshot = {
        "schema_version": "v2xvit-ffn-physical-pruning-v1",
        "pruned_unit_ids": list(phenotype.pruned_unit_ids),
        "domain_width_profile": width_profile,
        "domains": snapshots,
        "parameter_count_before": before,
        "parameter_count_after": after,
        "parameter_reduction": before - after,
    }
    return V2XViTPhysicalPruningResult(
        model=physical,
        snapshot=snapshot,
        snapshot_hash=canonical_json_hash(snapshot),
        parameter_count_before=before,
        parameter_count_after=after,
    )


def apply_v2xvit_weight_fake_quantization(
    model: nn.Module,
    phenotype: CandidatePhenotype,
) -> dict[str, Any]:
    """Apply the Stage-1 weight perturbation for a real-forward smoke only."""

    modules = dict(model.named_modules())
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for module_path, precision in sorted(
            phenotype.realized_precision_profile.items()
        ):
            module = modules.get(module_path)
            weight = getattr(module, "weight", None)
            if weight is None:
                raise RuntimeError(
                    f"v2xvit_fake_quant_module_weight_missing:{module_path}"
                )
            before = weight.detach().clone()
            quantized = pseudo_quantize_tensor(
                weight.detach(), precision, module=module
            )
            weight.copy_(quantized)
            rows.append(
                {
                    "module_path": module_path,
                    "precision": precision,
                    "weight_shape": list(weight.shape),
                    "max_abs_delta": float(
                        (quantized - before).abs().max().detach().cpu()
                    ),
                    "semantics": "weight_fake_quant_only_not_activation_qdq",
                }
            )
    return {
        "schema_version": "v2xvit-weight-fake-quant-smoke-v1",
        "layer_count": len(rows),
        "layers": rows,
        "explicit_qdq_applied": False,
        "tensorrt_precision_realized": False,
    }


def materialize_v2xvit_unified_pruning(
    model: nn.Module,
    phenotype: CandidatePhenotype,
    domains: Sequence[LocalPruningDomain],
    cnn_atomic_units: Sequence[Any],
) -> V2XViTPhysicalPruningResult:
    """Materialize CNN, attention ``d_h`` and FFN ``d_ff`` width genes."""

    width_profile = {
        str(key): int(value)
        for key, value in dict(
            phenotype.metadata.get("domain_width_profile") or {}
        ).items()
    }
    expected = {str(domain.domain_id) for domain in domains}
    if set(width_profile) != expected:
        raise RuntimeError(
            "v2xvit_unified_width_profile_mismatch:"
            f"missing={sorted(expected-set(width_profile))}:"
            f"extra={sorted(set(width_profile)-expected)}"
        )
    result = materialize_unified_widths(
        model,
        cnn_atomic_units,
        domains,
        width_profile,
        model_name="heal_lidar_v2xvit",
    )
    if not result.report.passed:
        raise RuntimeError(
            f"v2xvit_unified_physical_pruning_failed:{result.report.issues}"
        )
    snapshot = {
        "schema_version": "v2xvit-unified-physical-pruning-v1",
        "pruned_unit_ids": list(phenotype.pruned_unit_ids),
        "domain_width_profile": width_profile,
        "report": result.report.to_dict(),
    }
    return V2XViTPhysicalPruningResult(
        model=result.model,
        snapshot=snapshot,
        snapshot_hash=canonical_json_hash(snapshot),
        parameter_count_before=result.report.original_parameter_count,
        parameter_count_after=result.report.physical_parameter_count,
    )


__all__ = [
    "V2XViTPhysicalPruningResult",
    "apply_v2xvit_weight_fake_quantization",
    "materialize_v2xvit_ffn_pruning",
    "materialize_v2xvit_unified_pruning",
]
