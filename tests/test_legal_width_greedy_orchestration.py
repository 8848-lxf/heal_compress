from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pruning.types import AtomicPruneUnit


def _context_and_proxy():
    from search.canonicalization import (
        SearchSpaceSpec,
        canonicalize_legal_width_candidate,
    )
    from search.decoding.fixed_taylor_width_decoder import FixedTaylorWidthDecoder
    from search.hashing import candidate_hash
    from search.space.legal_width_inventory import build_legal_width_inventory

    units = [
        AtomicPruneUnit(
            "scope",
            "conv",
            "out",
            [index],
            [f"c{index}"],
            float(index),
            _stable_id=f"u{index}",
        )
        for index in range(100)
    ]
    inventory = build_legal_width_inventory(
        units,
        minimum_retained_ratio=0.05,
        minimum_retained_channels=5,
        dense_alignment=1,
        per_domain_max_prune_rate=0.95,
    )
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
            for index in range(100)
        ],
    )
    space = SearchSpaceSpec(
        pruning_unit_ids=list(inventory.unit_ids),
        precision_layer_ids=["pg"],
        default_precision="FP32",
        structure_gene_type="legal_keep_width",
        legal_width_inventory=inventory,
        fixed_width_decoder=decoder,
        precision_action_space={"pg": ("FP32",)},
    )
    context = SimpleNamespace(
        search_space=space,
        code_commit="test-commit",
        checkpoint_hash="checkpoint",
    )

    class Proxy:
        def __init__(self) -> None:
            self.call_count = 0
            self.batch_call_count = 0

        def evaluate(self, genotype, *, generation=0, outer_round=0):
            del generation, outer_round
            self.call_count += 1
            keep = inventory.domains[0].legal_keep_widths[
                genotype.width_genes[domain_id]
            ]
            phenotype = canonicalize_legal_width_candidate(genotype, space)
            return {
                "R_BOPS": 0.4 * keep / 100.0,
                "R_bops_vs_fp32": 0.4 * keep / 100.0,
                "L_joint_raw": (100 - keep) * 0.01,
                "L_joint_first_order": (100 - keep) * 0.006,
                "L_joint_second_order": (100 - keep) * 0.004,
                "R_prune": (100 - keep) / 100.0,
                "F1": (100 - keep) * 0.01,
                "candidate_hash": candidate_hash(phenotype, space),
                "phenotype": phenotype.to_dict(),
            }

        def evaluate_batch(self, genotypes, *, generation=0, outer_round=0):
            self.batch_call_count += 1
            return SimpleNamespace(
                metrics=[
                    self.evaluate(
                        genotype,
                        generation=generation,
                        outer_round=outer_round,
                    )
                    for genotype in genotypes
                ]
            )

    return context, Proxy()


def test_six_budget_greedy_writes_endpoints_and_fixed_scale(tmp_path: Path) -> None:
    from search.orchestration.legal_width_greedy import run_six_budget_greedy

    context, proxy = _context_and_proxy()
    result = run_six_budget_greedy(
        context=context,
        proxy=proxy,
        run_dir=tmp_path,
        config={
            "targets": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
            "primary_tolerance": 0.005,
            "expanded_tolerance": 0.0075,
            "frontier_size": 8,
            "max_expansions": 4096,
            "anchor_scale_rows": [
                {"phenotype_hash": "anchor", "L_joint_raw": 2.0}
            ],
        },
    )

    assert result["target_count"] == 6
    assert result["feasible_budget_count"] == 6
    assert len(result["endpoint_records"]) == 6
    assert (tmp_path / "greedy" / "greedy_summary.json").is_file()
    assert (tmp_path / "greedy" / "budget_005_trace.json").is_file()
    assert (tmp_path / "greedy" / "budget_030_endpoint.json").is_file()
    scale_path = tmp_path / "joint_loss_scale.json"
    assert scale_path.stat().st_mode & 0o222 == 0
    assert result["joint_loss_scale"]["mapping"] == "linear_fixed_scale"
    assert result["joint_loss_scale"]["member_count"] > 6
    assert all(
        row["normal_candidate_repair_invoked"] is False
        for row in result["path_states"]
    )
    assert proxy.call_count <= 96
    assert proxy.batch_call_count > 0
    assert all(
        abs(row["actual_bops"] - row["target_bops"]) <= 0.005 + 1.0e-12
        for row in result["endpoints"]
    )


def test_greedy_config_contract_is_raw_weight_only_and_has_no_ap_gate() -> None:
    import yaml

    path = (
        Path(__file__).resolve().parents[1]
        / "search/configs/lidar_pyramid_4090_greedy_six_budget.yaml"
    )
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert payload["structure_gene"]["type"] == "legal_keep_width"
    assert payload["structure_gene"]["ranking_depends_on_precision"] is False
    assert payload["candidate_proxy"]["mode"] == (
        "joint_pruning_quantization_second_order_fisher"
    )
    assert payload["proxy"]["task_score_mapping"] == "raw_joint_loss"
    assert payload["proxy"]["include_activation_taylor"] is False
    assert payload["proxy"]["sqnr_main_objective_weight"] == 0.0
    assert payload["proxy"]["bops_reference"] == "original_strict_fp32"
    assert payload["greedy"]["targets"] == [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    assert payload["greedy"]["frontier_size"] == 1
    assert payload["greedy"]["primary_tolerance"] == pytest.approx(0.005)
    assert payload["greedy"]["expanded_tolerance"] == pytest.approx(0.0075)
    assert "min_map" not in payload.get("stage2", {})
    assert "min_ap07" not in payload.get("stage2", {})
    assert "max_map_drop" not in payload.get("stage2", {})


def test_joint_proxy_scale_loader_routes_explicit_mappings(tmp_path: Path) -> None:
    from search.orchestration.lidar_pyramid_search import _load_joint_proxy_scale
    from search.proxy.joint_loss_scale import (
        calibrate_joint_loss_scale,
        write_joint_loss_scale,
    )

    path = write_joint_loss_scale(
        tmp_path / "joint_loss_scale.json",
        calibrate_joint_loss_scale(
            [{"phenotype_hash": "p", "L_joint_raw": 1.0}],
            code_commit="abc",
        ),
    )

    assert _load_joint_proxy_scale(
        {
            "proxy_mode": "joint_taylor_second_order_fisher_diag",
            "task_score_mapping": "raw_joint_loss",
        }
    ) == {}
    loaded = _load_joint_proxy_scale(
        {
            "proxy_mode": "joint_taylor_second_order_fisher_diag",
            "task_score_mapping": "linear_fixed_scale",
            "joint_loss_scale_path": str(path),
        }
    )
    assert loaded["mapping"] == "linear_fixed_scale"
    assert loaded["value"] == pytest.approx(1.0)
    with pytest.raises(RuntimeError, match="task_score_mapping_required"):
        _load_joint_proxy_scale(
            {"proxy_mode": "joint_taylor_second_order_fisher_diag"}
        )


def test_cli_accepts_greedy_only_and_joint_loss_scale() -> None:
    from search.cli import parse_args

    args = parse_args(
        [
            "--config",
            "config.yaml",
            "--greedy-only",
            "--joint-loss-scale",
            "scale.json",
        ]
    )

    assert args.greedy_only is True
    assert args.joint_loss_scale == "scale.json"
