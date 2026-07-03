from __future__ import annotations

from opencood.tools.compression.root_node_local_pruner import (
    CoupledChannelUnitV8,
    RootNodeLocalPruningDomain,
    build_root_node_local_domains,
    select_units_by_domain_local_ratio,
)


def _unit(root: str, idx: int, score: float) -> CoupledChannelUnitV8:
    return CoupledChannelUnitV8(
        unit_id=f"{root}::ch{idx}",
        root_node=root,
        root_module=root,
        root_axis="out_channels",
        root_channel_index=idx,
        members=[],
        dependency_types=[],
        importance=score,
    )


def test_same_root_node_units_enter_same_domain_and_roots_do_not_mix():
    units = [_unit("root_a", i, float(i)) for i in range(4)] + [_unit("root_b", i, float(i)) for i in range(2)]

    domains = build_root_node_local_domains(units)

    by_root = {domain.root_node: domain for domain in domains}
    assert set(by_root) == {"root_a", "root_b"}
    assert by_root["root_a"].unit_ids == ["root_a::ch0", "root_a::ch1", "root_a::ch2", "root_a::ch3"]
    assert by_root["root_b"].unit_ids == ["root_b::ch0", "root_b::ch1"]
    assert all(domain.cross_domain_ranking is False for domain in domains)


def test_domain_local_selector_keeps_topk_per_domain_without_cross_domain_comparison():
    units = [_unit("root_a", i, score) for i, score in enumerate([0.1, 0.2, 0.9, 1.0])] + [
        _unit("root_b", i, score) for i, score in enumerate([100.0, 101.0, 102.0, 103.0])
    ]
    domains = build_root_node_local_domains(units)

    summary = select_units_by_domain_local_ratio(domains, units, keep_ratio=0.5, align=1, min_keep_ratio=0.0)

    by_domain = {row["root_node"]: row for row in summary["domains"]}
    assert by_domain["root_a"]["kept_unit_ids"] == ["root_a::ch2", "root_a::ch3"]
    assert by_domain["root_a"]["pruned_unit_ids"] == ["root_a::ch0", "root_a::ch1"]
    assert by_domain["root_b"]["kept_unit_ids"] == ["root_b::ch2", "root_b::ch3"]
    assert by_domain["root_b"]["pruned_unit_ids"] == ["root_b::ch0", "root_b::ch1"]
    assert summary["global_ranking"] is False
    assert summary["module_stage_based_domain"] is False


def test_different_keep_ratios_produce_different_pruned_unit_counts():
    units = [_unit("root", i, float(i)) for i in range(8)]
    domains = build_root_node_local_domains(units)

    keep875 = select_units_by_domain_local_ratio(domains, units, keep_ratio=0.875, align=1, min_keep_ratio=0.0)
    keep625 = select_units_by_domain_local_ratio(domains, units, keep_ratio=0.625, align=1, min_keep_ratio=0.0)

    assert keep875["domains"][0]["actual_num_prune"] == 1
    assert keep625["domains"][0]["actual_num_prune"] == 3
