from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_slice_union_counts_overlapping_dependency_closures_once() -> None:
    from search.audits.prune_rate_reachability import parameter_slice_union

    rows = [
        {"unit_id": "u0", "parameter_name": "conv.weight", "axis": 0, "indices": [0]},
        {"unit_id": "u1", "parameter_name": "conv.weight", "axis": 1, "indices": [0]},
        {"unit_id": "u2", "parameter_name": "conv.weight", "axis": 0, "indices": [0]},
    ]

    result = parameter_slice_union({"conv.weight": (2, 2)}, rows)

    assert result["raw_element_sum"] == 6
    assert result["global_union_element_count"] == 3
    assert result["duplicate_or_overlap_element_count"] == 3


def test_dependency_closure_counts_successor_input_and_root_output() -> None:
    from search.audits.prune_rate_reachability import parameter_slice_union

    rows = [
        {"unit_id": "u", "parameter_name": "root.weight", "axis": 0, "indices": [0]},
        {"unit_id": "u", "parameter_name": "next.weight", "axis": 1, "indices": [0]},
    ]
    result = parameter_slice_union(
        {"root.weight": (2, 3), "next.weight": (5, 2)}, rows
    )

    assert result["global_union_by_parameter"]["root.weight"] == 3
    assert result["global_union_by_parameter"]["next.weight"] == 5
    assert result["global_union_element_count"] == 8


def test_grouped_virtual_shape_expands_local_input_removals_across_groups() -> None:
    import torch.nn as nn

    from search.candidate import CandidatePhenotype
    from search.proxy.parameter_slice_resolver import ParameterSlice
    from search.proxy.virtual_shape_resolver import resolve_virtual_shapes

    model = nn.Module()
    model.grouped = nn.Conv2d(512, 512, 3, groups=32, bias=False)
    slices = {
        "u": [
            ParameterSlice(
                "grouped.weight", "grouped", 0, tuple(range(384)), "prune_out"
            ),
            ParameterSlice(
                # Independent physical groups may prune different local
                # positions, so their aggregate local-index union can be full.
                "grouped.weight", "grouped", 1, tuple(range(16)), "prune_in_local"
            ),
        ]
    }

    shape = resolve_virtual_shapes(
        model,
        CandidatePhenotype(pruned_unit_ids=["u"]),
        slices,
    )["grouped"]

    assert shape.c_out_after == 128
    assert shape.c_in_after == 128
    assert shape.parameter_count_after == 128 * 4 * 3 * 3
