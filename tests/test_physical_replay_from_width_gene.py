from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pruning.types import AtomicPruneUnit


def test_physical_replay_uses_decoded_mask_without_reranking() -> None:
    from pruning.api import estimate_physical_parameter_count
    from search.adapters.pruning_adapter import FormalPruningAdapter
    from search.candidate import CandidatePhenotype
    from search.decoding.fixed_taylor_width_decoder import FixedTaylorWidthDecoder
    from search.space.legal_width_inventory import build_legal_width_inventory
    from search.stage2.physical_validation import validate_repaired_physical_plan

    model = torch.nn.Sequential()
    model.add_module("conv0", torch.nn.Conv2d(4, 8, 1, bias=True))
    model.add_module("conv1", torch.nn.Conv2d(8, 2, 1, bias=False))
    units = []
    for index in range(8):
        unit = AtomicPruneUnit(
            "scope", "conv0", "out", [index], [f"c{index}"], float(index),
            _stable_id=f"u{index}",
        )
        unit.members = [
            SimpleNamespace(
                module_path="conv0",
                axis="out",
                indices=[index],
                dependency_type="root_output",
            ),
            SimpleNamespace(
                module_path="conv1",
                axis="in",
                indices=[index],
                dependency_type="successor_input",
            ),
        ]
        units.append(unit)
    inventory = build_legal_width_inventory(units, dense_alignment=4)
    domain_id = inventory.domain_ids[0]
    decoder = FixedTaylorWidthDecoder(
        inventory,
        [
            {
                "domain_id": domain_id,
                "physical_group_id": 0,
                "atomic_unit_id": f"u{index}",
                "first_order_score": float(index),
                "second_order_score": float(index),
            }
            for index in range(8)
        ],
    )
    decoded = decoder.decode({domain_id: 0})
    phenotype = CandidatePhenotype(
        pruned_unit_ids=list(decoded.pruned_unit_ids),
        metadata=decoded.to_dict(),
    )
    # Changing source scores after decode must not affect export selection.
    for unit in units:
        unit.normalized_score = -float(unit.root_indices[0])
    adapter = FormalPruningAdapter()
    request = adapter.request_from_phenotype(phenotype, units)
    result = adapter.materialize_from_request(model, request)
    predicted = estimate_physical_parameter_count(model, result["plan"])
    physical = sum(parameter.numel() for parameter in result["model"].parameters())
    plan_audit = validate_repaired_physical_plan(
        phenotype, request, result["plan"]
    )

    assert request.selected_atomic_unit_ids == list(decoded.pruned_unit_ids)
    assert set(request.selected_atomic_unit_ids) == {"u0", "u1", "u2", "u3"}
    assert result["model"].conv0.out_channels == 4
    assert result["model"].conv1.in_channels == 4
    assert predicted == physical
    assert plan_audit["passed"] is True
    assert plan_audit["repaired_mask_to_physical_plan_verified"] is True

