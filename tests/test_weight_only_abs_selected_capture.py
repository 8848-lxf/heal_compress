from __future__ import annotations


def _space():
    from search.canonicalization import SearchSpaceSpec
    from search.pruning_space.local_domains import LocalPruningDomain

    domains = []
    for index in range(2):
        domain_id = f"d{index}"
        unit = f"{domain_id}:0"
        domains.append(LocalPruningDomain(
            domain_id=domain_id,
            root_module_path=domain_id,
            root_axis="out",
            scope_id=domain_id,
            kind="dense",
            original_width=8,
            total_original_width=8,
            ordered_unit_ids=(unit,),
            legal_widths=(4, 8),
            width_to_pruned_unit_ids={4: (unit,), 8: ()},
            unit_root_indices={unit: (0,)},
            domain_type="cnn_channel",
        ))
    return SearchSpaceSpec(
        pruning_unit_ids=[], precision_layer_ids=[], pruning_domains=tuple(domains),
        quantization_groups=(), default_precision="FP32",
    )


class _StructureProxy:
    def pruning_action_breakdown(self, before, after):
        added = set(after.pruned_unit_ids) - set(before.pruned_unit_ids)
        # d0 is selected first by utility even though the unselected d1
        # neighbor is exactly in the requested budget band.
        score = 0.0 if "d0:0" in added else 10.0
        return {
            "delta_J_prune": score,
            "delta_J_WQ": 0.0,
            "first_order_abs_sum": score,
            "second_order_abs_sum": 0.0,
            "newly_pruned_parameter_count": 1,
        }


class _WeightProxy:
    def weight_quantization_action_breakdown(self, _before, _after):
        raise AssertionError("no precision locus in this fixture")


def test_budget_capture_uses_selected_trajectory_not_unselected_neighbor() -> None:
    from search.greedy.weight_only_abs import run_weight_only_abs_greedy

    space = _space()

    def bops(phenotype):
        pruned = set(phenotype.pruned_unit_ids)
        total = 100.0
        if "d0:0" in pruned:
            total -= 50.0
        if "d1:0" in pruned:
            total -= 30.0
        return {"bops_total": total, "R_bops_vs_fp32": total / 100.0}

    result = run_weight_only_abs_greedy(
        space,
        weight_proxy=_WeightProxy(),
        structure_proxy=_StructureProxy(),
        bops_evaluator=bops,
        size_evaluator=lambda _phenotype: {
            "size_bits_total": 320.0,
            "R_parameter_retention": 1.0,
            "R_size_vs_fp32": 1.0,
        },
        target=0.70,
        tolerance_abs=1.0e-9,
        maximum_steps=None,
    )
    assert result["budget_reached"] is False
    assert result["captured_targets"] == []
    in_band_neighbors = [
        row for row in result["trace"]
        if abs(row["current_retention"] - 0.70) <= 1.0e-9
    ]
    assert len(in_band_neighbors) == 1
    assert in_band_neighbors[0]["selected"] is False
