from __future__ import annotations

from tools.latency_lut.fixed_width_boundary_registry_v84 import (
    default_fixed_width_boundaries,
    protected_prefixes_for_fixed_width_boundaries,
)


def test_pfn_to_pointpillar_scatter_boundary_is_explicit_and_narrow():
    boundaries = default_fixed_width_boundaries()
    boundary = boundaries[0]

    assert boundary["boundary_id"] == "pfn_encoder_to_pointpillar_scatter"
    assert boundary["protected_reason"] == "fixed_width_boundary:pfn_to_pointpillar_scatter"
    assert boundary["fixed_channel"] == 64
    prefixes = protected_prefixes_for_fixed_width_boundaries(boundaries)
    assert "encoder_m1" not in prefixes
    assert any("pfn_layers" in prefix for prefix in prefixes)

