from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _phenotype():
    from search.candidate import CandidatePhenotype, PrecisionDecision

    return CandidatePhenotype(
        pruned_unit_ids=["u1", "u2"],
        precision_profile={
            "a": PrecisionDecision("INT8", "INT8"),
            "b": PrecisionDecision("FP16", "FP16"),
        },
        metadata={
            "domain_width_profile": {"d": 4},
            "domain_width_expansion_hash": "widths",
            "domains": {"d": {}},
            "group_keep_map_by_scope": {"s": {0: [0, 1]}},
            "group_prune_map_by_scope": {"s": {0: [2, 3]}},
            "requested_group_profile": {"g0": "INT8", "g1": "FP16"},
            "stage1_legalized_group_profile": {"g0": "INT8", "g1": "FP16"},
            "quantization_group_contracts": {"g0": {"member_layers": ["a"]}, "g1": {"member_layers": ["b"]}},
        },
    )


def test_prune_only_preserves_mask_and_forces_strict_fp32():
    from search.ablation.lidar_pyramid_prune_quant import build_ablation_phenotype

    source = _phenotype()
    result = build_ablation_phenotype(source, "prune_only")
    assert result.pruned_unit_ids == source.pruned_unit_ids
    assert set(result.realized_precision_profile.values()) == {"FP32"}
    assert set(result.metadata["stage1_legalized_group_profile"].values()) == {"FP32"}
    assert result.metadata["group_keep_map_by_scope"] == source.metadata["group_keep_map_by_scope"]


def test_quant_only_is_all_keep_and_preserves_precision_contract():
    from search.ablation.lidar_pyramid_prune_quant import build_ablation_phenotype

    source = _phenotype()
    result = build_ablation_phenotype(source, "quant_only")
    assert result.pruned_unit_ids == []
    assert result.realized_precision_profile == source.realized_precision_profile
    assert "domain_width_profile" not in result.metadata
    assert "group_keep_map_by_scope" not in result.metadata
    assert result.metadata["quantization_group_contracts"] == source.metadata["quantization_group_contracts"]


def test_latest_pq_runner_compacts_only_derived_contract_history(tmp_path):
    from scripts.run_pyramid_latest_pq_decomposition_repeat5 import (
        _load_compact_source_phenotype,
    )
    from search.integration.heal_lidar_family_fair_evaluation import sha256_file

    payload = _phenotype().to_dict()
    payload["metadata"]["precision_group_expansion"] = {
        "g0": ["a"],
        "g1": ["b"],
    }
    payload["metadata"]["quantization_group_contracts"] = {
        "g0": {"member_layers": ["a"], "merge_boundaries": [{"id": 1}] * 9},
        "g1": {"member_layers": ["b"], "merge_boundaries": [{"id": 2}] * 7},
    }
    path = tmp_path / "phenotype.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    source = {
        "source_candidate_hash": "source",
        "source_phenotype_path": str(path),
        "source_phenotype_sha256": sha256_file(path),
        "source_phenotype_size_bytes": path.stat().st_size,
    }
    groups = [
        SimpleNamespace(
            group_id="g0",
            module_paths=("a",),
            allowed_precisions=("FP32", "FP16", "INT8"),
            metadata={"merge_boundaries": [{"id": "clean"}]},
        ),
        SimpleNamespace(
            group_id="g1",
            module_paths=("b",),
            allowed_precisions=("FP32", "FP16"),
            metadata={"merge_boundaries": []},
        ),
    ]
    compact, audit = _load_compact_source_phenotype(
        source,
        quantization_groups=groups,
    )
    assert compact.pruned_unit_ids == payload["pruned_unit_ids"]
    assert compact.realized_precision_profile == payload["realized_precision_profile"]
    assert audit["source_merge_boundary_count"] == 16
    assert audit["clean_merge_boundary_count"] == 1
    assert audit["structure_and_precision_preserved"] is True
    assert compact.metadata["quantization_group_contracts"]["g0"][
        "merge_boundaries"
    ] == [{"id": "clean"}]


def test_repeat5_runners_accept_only_explicit_completed_budget_subsets():
    from scripts.run_pyramid_greedy_ga_full1789_repeat5 import (
        _parse_budget_labels as parse_joint_labels,
    )
    from scripts.run_pyramid_latest_pq_decomposition_repeat5 import (
        _parse_budget_labels as parse_control_labels,
    )

    assert parse_joint_labels("005") == ("005",)
    assert parse_joint_labels("030,025") == ("030", "025")
    assert parse_control_labels("005") == ("005",)
    assert parse_control_labels("030,025") == ("030", "025")

    import pytest

    for parser in (parse_joint_labels, parse_control_labels):
        with pytest.raises(ValueError):
            parser("")
        with pytest.raises(ValueError):
            parser("005,005")
        with pytest.raises(ValueError):
            parser("075")


def test_greedy_budget_replay_uses_lowest_taylor_feasible_state(tmp_path):
    from search.ablation.lidar_pyramid_prune_quant import replay_greedy_budget_candidate
    from search.candidate import CandidateGenotype, CandidatePhenotype, PrecisionDecision

    initial = CandidateGenotype(
        pruning_width_genes={"d": 8},
        precision_genes={"g": "FP32", "h": "FP32"},
    )
    one = CandidateGenotype(
        pruning_width_genes={"d": 8},
        precision_genes={"g": "FP16", "h": "FP32"},
    )
    two = CandidateGenotype(
        pruning_width_genes={"d": 8},
        precision_genes={"g": "FP16", "h": "FP16"},
    )
    def identity(candidate):
        value = {"pruning_width_genes": candidate.pruning_width_genes, "precision_genes": candidate.precision_genes}
        return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    phenotype = CandidatePhenotype(
        precision_profile={"a": PrecisionDecision("FP16", "FP16")}
    ).to_dict()
    payload = {
        "initial_candidate": initial.to_dict(),
        "steps": [
            {"step_index": 1, "action_kind": "precision", "action_gene_id": "g", "previous_value": "FP32", "selected_value": "FP16", "bops_after": 0.304, "loss_after": 0.1, "candidate_hash": identity(one), "metrics": {"R_bops_vs_fp32": 0.304, "L_joint_weight_taylor": 0.1, "phenotype": phenotype}},
            {"step_index": 2, "action_kind": "precision", "action_gene_id": "h", "previous_value": "FP32", "selected_value": "FP16", "bops_after": 0.300, "loss_after": 0.2, "candidate_hash": identity(two), "metrics": {"R_bops_vs_fp32": 0.300, "L_joint_weight_taylor": 0.2, "phenotype": phenotype}},
        ],
    }
    path = tmp_path / "greedy_path.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    result = replay_greedy_budget_candidate(path, target=0.30, tolerance=0.005)
    assert result["selected_step_index"] == 1
    assert result["actual_bops"] == 0.304
