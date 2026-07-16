from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pruning.types import AtomicPruneUnit


def test_multi_seed_stage1_runs_full_generations_without_normal_repair(tmp_path: Path) -> None:
    from search.canonicalization import SearchSpaceSpec
    from search.decoding.fixed_taylor_width_decoder import FixedTaylorWidthDecoder
    from search.orchestration.legal_width_joint_ga import run_legal_width_stage1_seeds
    from search.space.legal_width_inventory import build_legal_width_inventory
    from search.stage1.proxy_evaluator import BatchProxyResult

    units = [
        AtomicPruneUnit(
            f"scope{domain}",
            f"conv{domain}",
            "out",
            [index],
            [f"c{domain}_{index}"],
            float(index),
            _stable_id=f"u{domain}_{index}",
        )
        for domain in range(3)
        for index in range(8)
    ]
    inventory = build_legal_width_inventory(units, dense_alignment=2)
    ranking = [
        {
            "domain_id": domain.domain_id,
            "physical_group_id": 0,
            "atomic_unit_id": unit_id,
            "first_order_score": float(index),
            "second_order_score": float(index),
        }
        for domain in inventory.domains
        for index, unit_id in enumerate(domain.unit_ids)
    ]
    decoder = FixedTaylorWidthDecoder(inventory, ranking)
    space = SearchSpaceSpec(
        pruning_unit_ids=list(inventory.unit_ids),
        precision_layer_ids=["pg0", "pg1"],
        structure_gene_type="legal_keep_width",
        legal_width_inventory=inventory,
        fixed_width_decoder=decoder,
        precision_action_space={
            "pg0": ("FP16", "INT8"),
            "pg1": ("FP16", "INT8"),
        },
    )

    class Proxy:
        def evaluate_batch(self, genotypes, *, generation, outer_round):
            rows = []
            for genotype in genotypes:
                width_sum = sum(genotype.width_genes.values())
                int8 = sum(value == "INT8" for value in genotype.precision_genes.values())
                bops = 0.205 + 0.001 * ((width_sum + int8) % 10)
                rows.append(
                    {
                        "F1": -0.8 + 0.001 * width_sum,
                        "J1": 0.8 - 0.001 * width_sum,
                        "S_task": 0.9,
                        "R_prune": 0.1,
                        "R_bops_vs_fp32": bops,
                        "R_MAC": 0.8,
                        "L_joint_raw": 0.1,
                        "L_joint_first_order": 0.08,
                        "L_joint_second_order": 0.02,
                    }
                )
            return BatchProxyResult(rows, {"generation": generation, "seed": outer_round})

    result = run_legal_width_stage1_seeds(
        context=SimpleNamespace(search_space=space),
        proxy=Proxy(),
        run_dir=tmp_path,
        search_config={
            "population_size": 8,
            "initial_population_size": 8,
            "offspring_size": 8,
            "generations": 3,
            "independent_seeds": 2,
            "seed": 17,
            "target_bops_retention": 0.21,
            "bops_tolerance": 0.005,
        },
    )

    assert result["ga_seeds"] == 2
    assert result["ga_generations"] == 3
    assert result["total_proxy_evaluations"] == 48
    assert len(result["generation_summaries"]) == 6
    assert result["repair_report"]["normal_candidate_count"] == 48
    assert result["repair_report"]["repair_invocation_count"] == 0
    assert result["archive_summary"]["feasible_phenotype_count"] > 0
    assert (tmp_path / "stage1_legal_width_summary.json").is_file()


def test_legal_width_anchor_path_uses_only_adjacent_legal_states() -> None:
    from search.anchors.joint_taylor_sweep import plan_legal_width_anchor_structures
    from search.decoding.fixed_taylor_width_decoder import FixedTaylorWidthDecoder
    from search.space.legal_width_inventory import build_legal_width_inventory

    units = [
        AtomicPruneUnit(
            f"scope{domain}", f"conv{domain}", "out", [index],
            [f"c{domain}_{index}"], float(index), _stable_id=f"u{domain}_{index}",
        )
        for domain in range(2)
        for index in range(8)
    ]
    inventory = build_legal_width_inventory(units, dense_alignment=4)
    ranking = [
        {
            "domain_id": domain.domain_id,
            "physical_group_id": 0,
            "atomic_unit_id": unit_id,
            "first_order_score": float(rank + domain_index * 100),
            "second_order_score": float(rank + domain_index * 100),
        }
        for domain_index, domain in enumerate(inventory.domains)
        for rank, unit_id in enumerate(domain.unit_ids)
    ]
    decoder = FixedTaylorWidthDecoder(inventory, ranking)

    plan = plan_legal_width_anchor_structures(
        inventory=inventory,
        decoder=decoder,
        ranking_rows=ranking,
        ranking_mode="prune_only_second_order_fisher",
        requested_prune_rates=(0.0, 0.25, 0.5),
        original_params=160,
        parameter_count_fn=lambda mask: 160 - 10 * sum(
            int(keep) == 0 for keep in mask.values()
        ),
    )

    assert len(plan.structures) == 3
    assert all(
        row.repair_metadata["repair_invoked"] is False for row in plan.structures
    )
    assert all(
        row.repair_metadata["ranking_mode"] == "prune_only_second_order_fisher"
        for row in plan.structures
    )
    assert [row.requested_prune_rate for row in plan.structures] == [0.0, 0.25, 0.5]
    assert plan.maximum_realized_prune_rate == 0.5


def test_archive_first_stage2_backfills_until_minimum_success(tmp_path: Path) -> None:
    from search.archive.feasible_pareto_archive import FeasibleParetoArchive
    from search.candidate import CandidatePhenotype, PrecisionDecision
    from search.orchestration.legal_width_stage2 import (
        run_legal_width_stage2_screening,
    )
    from search.stage1.topk_selector import ProxyCandidateRecord

    archive = FeasibleParetoArchive()
    records = {}
    for index in range(7):
        phenotype = CandidatePhenotype(
            pruned_unit_ids=[f"u{index}"],
            precision_profile={
                "conv": PrecisionDecision("FP16", "FP16")
            },
            metadata={
                "phenotype_hash": f"p{index}",
                "structure_hash": f"structure{index}",
                "precision_hash": "fp16",
            },
        )
        genotype = SimpleNamespace(
            to_dict=lambda index=index: {"candidate": index},
            meta={"seed_family": "ga"},
        )
        record = ProxyCandidateRecord(
            f"candidate{index}", genotype, phenotype, float(index), {"J1": 1.0}
        )
        records[f"p{index}"] = record
        archive.add(
            {
                "phenotype_hash": f"p{index}",
                "structure_hash": f"structure{index}",
                "precision_hash": "fp16",
                "R_BOPS": 0.20 + index * 0.001,
                "S_task": 0.9,
                "R_param": 0.9,
                "latency_proxy_value": 0.8,
                "structure_legal": True,
                "precision_legal": True,
                "finite_joint_proxy": True,
                "missing_mapping": 0,
            },
            active_budget=0.215,
        )

    class Pool:
        parallelism = 4

        def __init__(self):
            self.tasks = []

        def map_tasks(self, tasks):
            self.tasks.extend(tasks)
            return [
                {
                    "candidate_hash": task["candidate_hash"],
                    "status": (
                        "engine_build_failed"
                        if task["candidate_hash"] == "candidate0"
                        else "ok"
                    ),
                    "worker_gpu_id": 4 + (len(self.tasks) - 1) % 4,
                    "precision_identity_passed": True,
                }
                for task in tasks
            ]

    pool = Pool()
    result = run_legal_width_stage2_screening(
        archive=archive,
        records_by_phenotype_hash=records,
        stage2_pool=pool,
        run_dir=tmp_path,
        minimum_successful_candidates=5,
        maximum_attempts=7,
        smoke_frames=0,
        smoke_warmup_frames=0,
    )

    assert result["attempted_count"] == 7
    assert result["successful_count"] == 6
    assert result["minimum_success_reached"] is True
    assert result["failure_reason_histogram"] == {"engine_build_failed": 1}
    assert all(task["raw_precision_gene_hash"] for task in pool.tasks)
    assert all(
        task["raw_precision_gene_hash"] == task["repaired_precision_gene_hash"]
        for task in pool.tasks
    )
    assert (tmp_path / "stage2_screening_results.json").is_file()
