from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pruning.types import AtomicPruneUnit


def _dense_units(width: int) -> list[AtomicPruneUnit]:
    return [
        AtomicPruneUnit(
            scope_id="dense_scope",
            root_module_path="backbone.conv",
            root_axis="out",
            root_indices=[index],
            source_coupled_unit_ids=[f"coupled_{index}"],
            normalized_score=float(index),
            _stable_id=f"dense_{index}",
        )
        for index in range(width)
    ]


def test_inventory_contains_only_prevalidated_dense_widths() -> None:
    from search.space.legal_width_inventory import build_legal_width_inventory

    inventory = build_legal_width_inventory(
        _dense_units(18),
        minimum_retained_ratio=0.10,
        minimum_retained_channels=4,
        dense_alignment=4,
        per_domain_max_prune_rate=0.80,
    )

    domain = inventory.domains_by_id["backbone.conv::out::dense_scope"]
    assert domain.domain_kind == "dense"
    assert domain.original_width == 18
    assert domain.legal_keep_widths == (4, 8, 12, 16, 18)
    assert all(
        width == 18 or width % 4 == 0 for width in domain.legal_keep_widths
    )
    assert max((18 - width) / 18 for width in domain.legal_keep_widths) <= 0.80
    assert inventory.width_space_hash


def test_protected_domain_has_only_original_width() -> None:
    from search.space.legal_width_inventory import build_legal_width_inventory

    units = _dense_units(8)
    for unit in units:
        unit.protected = True
        unit.protection_reason = "head_contract"
    inventory = build_legal_width_inventory(units)
    domain = next(iter(inventory.domains))

    assert domain.protected is True
    assert domain.prunable is False
    assert domain.legal_keep_widths == (8,)
    assert domain.exclusion_reason == "head_contract"


def test_prepare_search_space_filters_precision_actions_by_deployment_capability() -> None:
    import torch

    from search.canonicalization import SearchSpaceSpec
    from search.proxy.fisher_proxy import FisherStatistics
    from search.proxy.parameter_slice_resolver import ParameterSlice
    from search.quantization_space.types import QuantizationSearchGroup
    from search.space.legal_width_inventory import prepare_legal_width_search_space

    model = torch.nn.Sequential()
    model.add_module("conv", torch.nn.Conv2d(2, 2, 1, bias=False))
    units = _dense_units(2)
    units[0].root_module_path = "conv"
    units[1].root_module_path = "conv"
    quant_groups = (
        QuantizationSearchGroup(
            "pg0", ("conv",), (), ("FP32", "FP16", "INT8"), False, "", 0, 4, 4.0
        ),
        QuantizationSearchGroup(
            "pg_protected", ("head",), (), ("FP32", "FP16"), True,
            "protected", 1, 1, 1.0, {"default_precision": "FP16"},
        ),
    )
    base = SearchSpaceSpec(
        pruning_unit_ids=[unit.stable_id for unit in units],
        precision_layer_ids=["conv", "head"],
        quantization_groups=quant_groups,
    )
    statistics = FisherStatistics(
        gradients={"conv.weight": torch.ones_like(model.conv.weight)},
        fisher_diag={"conv.weight": torch.ones_like(model.conv.weight)},
    )
    slices = {
        unit.stable_id: [
            ParameterSlice(
                "conv.weight", "conv", 0, tuple(unit.root_indices), "root_out"
            )
        ]
        for unit in units
    }

    prepared = prepare_legal_width_search_space(
        base,
        model=model,
        units=units,
        statistics=statistics,
        unit_to_parameter_slices=slices,
        checkpoint_hash="checkpoint",
        fisher_manifest_hash="fisher",
        requested_precision_actions=("FP16", "INT8"),
        dense_alignment=1,
        per_domain_max_prune_rate=0.5,
    )

    assert prepared.search_space.structure_gene_type == "legal_keep_width"
    assert prepared.search_space.precision_action_space == {
        "pg0": ("FP16", "INT8"),
        "pg_protected": ("FP16",),
    }
    assert prepared.second_order_ranking.manifest["ranking_depends_on_precision"] is False
    assert prepared.decoder.ranking_hash
