#!/usr/bin/env python3
"""Run six-budget V2X-ViT GA with the strict Stage-1/Stage-2/V1--V3 runner.

The six exact Greedy winners are immutable V1 anchors.  Every budget runs one
seed (zero), generation 0 is initialization, and generations 1--10 are the ten
formal evolution generations.  Stage-2 materialization/calibration/export/TRT
and fixed50 use the already deployment-closed V2X-ViT implementation.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]

from scripts.audit_heal_transformer_search_models import _load
from scripts.run_v2xvit_formal_ga_gen10 import (
    RealStage2Evaluator,
    atomic_write,
    best_real_candidate,
    stage2_payload,
)
from scripts.run_v2xvit_greedy005_full import (
    _baseline_candidate,
    _build_full_space,
    _formal_space,
)
from scripts.run_v2xvit_greedy005_weight_only_abs import _type_coverage_forward
from scripts.run_v2xvit_six_budget_proxy import FrozenTrainPrefix
from scripts.smoke_transformer_unified_search import _multi_agent_validation_batch
from search.candidate import CandidateGenotype
from search.canonicalization import canonicalize_candidate
from search.ga.cnn_stage12_v3 import build_initial_population
from search.ga.stage12_v3 import (
    StrictGAConfig,
    StrictStage12V3Runner,
    UnifiedTaylorStage1Evaluator,
    phenotype_identity,
    validate_genotype_schema,
)
from search.model_family.calibration_manifest import load_v2xvit_train_manifest
from search.proxy.conservative_gate_activation_taylor import (
    FunctionalGateTaylorProxy,
    build_activation_units,
    collect_activation_taylor_cache_multi,
    collect_functional_gate_scores_multi,
    rerank_domains_by_gate_scores,
)
from search.proxy.joint_weight_activation_taylor import (
    taylor_units_from_transformer_precision,
)
from search.proxy.joint_weight_taylor import JointWeightTaylorProxy


TARGETS = (0.30, 0.25, 0.20, 0.15, 0.10, 0.05)
TOLERANCE = 0.005
SEED = 0
GENERATIONS = 10
POPULATION = 64
OFFSPRING = 64
STAGE2_QUOTA = 5


def _label(target: float) -> str:
    return f"{int(round(float(target) * 100)):03d}"


def _parse_targets(value: str) -> tuple[float, ...]:
    targets = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not targets:
        raise argparse.ArgumentTypeError("at least one target budget is required")
    if len(set(targets)) != len(targets):
        raise argparse.ArgumentTypeError("target budgets must be unique")
    unsupported = tuple(target for target in targets if target not in TARGETS)
    if unsupported:
        raise argparse.ArgumentTypeError(f"unsupported target budgets: {unsupported}")
    return targets


def _shard_suffix(shard_id: str) -> str:
    value = str(shard_id).strip()
    if not value:
        return ""
    if not all(character.isalnum() or character in "-_" for character in value):
        raise RuntimeError(f"invalid_shard_id:{value}")
    return f"_{value}"


def _load_winner(root: Path, target: float) -> CandidateGenotype:
    path = root / f"greedy/budget_{_label(target)}/exact_winner.json"
    if not path.is_file():
        raise RuntimeError(f"v2xvit_exact_greedy_winner_missing:{target}:{path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("budget_reached"):
        raise RuntimeError(f"v2xvit_exact_greedy_budget_not_reached:{target}")
    return CandidateGenotype.from_dict(payload["genotype"])


def _fixed_request(fixed_k: int) -> dict[str, Any]:
    from scripts.audit_heal_transformer_search_models import MODEL_SPECS

    return {
        "model_config": str(MODEL_SPECS["v2xvit"]["config"]),
        "heal_root": "/home/lixingfeng/UniAD_examine/HEAL",
        "fixed_k": int(fixed_k),
        "max_agents": 2,
        "input_contract": "heal_v2xvit_fixed_k",
    }


def run(args: argparse.Namespace) -> int:
    root = args.output_root.resolve()
    targets = tuple(args.targets)
    shard_suffix = _shard_suffix(args.shard_id)
    if int(args.seed) != SEED:
        raise RuntimeError("v2xvit_six_budget_ga_requires_seed_zero")
    if int(args.generations) != GENERATIONS:
        raise RuntimeError("v2xvit_six_budget_ga_requires_exactly_ten_generations")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"v2xvit_six_budget_ga_requires_one_visible_gpu:{torch.cuda.device_count()}"
        )
    required = (
        root / "reports/six_budget_greedy_winners.json",
        root / "reports/taylor_sample_convergence.json",
        root / "reports/input_provenance.json",
        args.fixed50_manifest,
        args.plugin,
    )
    missing = [str(path) for path in required if not Path(path).is_file()]
    if missing:
        raise RuntimeError(f"v2xvit_six_budget_ga_required_artifacts_missing:{missing}")
    convergence = json.loads(
        (root / "reports/taylor_sample_convergence.json").read_text(encoding="utf-8")
    )
    if not convergence.get("passed"):
        raise RuntimeError("v2xvit_six_budget_ga_taylor_convergence_not_passed")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    train_manifest_path = (
        REPO / "search/model_family/manifests/heal_lidar_v2xvit_train200_fixed_k.json"
    )
    train200 = load_v2xvit_train_manifest(train_manifest_path)
    fixed_k = int(train200["fixed_k_contract"]["value"])
    provenance = json.loads(
        (root / "reports/input_provenance.json").read_text(encoding="utf-8")
    )
    calibration_hash = str(provenance["taylor_manifest_hash"])
    model, adapter, hypes, _ = _load("v2xvit", device)
    representative, _, _ = _multi_agent_validation_batch(adapter, hypes, device)
    train32 = FrozenTrainPrefix(
        adapter=adapter,
        hypes=hypes,
        device=device,
        manifest=train200,
        count=32,
    )
    identity = _build_full_space(model, adapter, hypes, representative)
    print("[v2xvit-formal-ga] rebuild frozen train32 Fisher", flush=True)
    formal = _formal_space(
        model,
        adapter,
        hypes,
        representative,
        identity,
        calibration_hash,
        fisher_forward_fn=lambda current_model, batch: _type_coverage_forward(
            adapter, current_model, batch
        ),
        fisher_batches=train32,
    )
    print("[v2xvit-formal-ga] rebuild functional gate cache", flush=True)
    gate32, gate_mapping = collect_functional_gate_scores_multi(
        model,
        formal["space"].pruning_domains,
        forward_fn=lambda current_model, batch: _type_coverage_forward(
            adapter, current_model, batch
        ),
        loss_fn=adapter.compute_task_loss,
        calibration_batches=train32,
    )
    domains = rerank_domains_by_gate_scores(formal["space"].pruning_domains, gate32)
    space = replace(
        formal["space"],
        pruning_domains=domains,
        pruning_unit_ids=[unit for domain in domains for unit in domain.ordered_unit_ids],
    )
    baseline = _baseline_candidate(space)
    transformer_units = taylor_units_from_transformer_precision(
        model,
        formal["components"].precision_units,
        active_module_paths=formal["active_paths"],
    )
    activation_units, group_to_units = build_activation_units(
        model, space, transformer_units
    )
    print("[v2xvit-formal-ga] rebuild activation Q/DQ cache", flush=True)
    activation32 = collect_activation_taylor_cache_multi(
        model,
        activation_units,
        group_to_units,
        forward_fn=lambda current_model, batch: _type_coverage_forward(
            adapter, current_model, batch
        ),
        loss_fn=adapter.compute_task_loss,
        calibration_batches=train32,
    )
    weight32 = JointWeightTaylorProxy(
        model,
        statistics=formal["fisher"],
        unit_to_parameter_slices=formal["slices"],
        strict=True,
    )
    qkv_paths = tuple(
        path
        for spec in formal["components"].attention_instances
        for path in (spec.q_projection_paths + spec.k_projection_paths)
    )
    atomic_write(
        root / f"ga/formal_cache_audit{shard_suffix}.json",
        {
            "sample_count": 32,
            "calibration_hash": calibration_hash,
            "gate_domain_count": len(gate32),
            "gate_mapping_count": len(gate_mapping),
            "activation_observer_count": len(activation32.mapping),
            "fisher_formula": "E[g^2]",
            "search_loop_forward_calls": 0,
            "search_loop_backward_calls": 0,
            "search_loop_exports": 0,
            "search_loop_trt_builds": 0,
        },
    )
    atomic_write(
        root / f"reports/formal_ga_config{shard_suffix}.json",
        {
            "model": "v2xvit",
            "targets": list(targets),
            "tolerance_abs": TOLERANCE,
            "seed_count": 1,
            "executed_seeds": [0],
            "formal_generations": GENERATIONS,
            "generation_zero_counted": False,
            "formal_generation_ids": list(range(1, 11)),
            "population_size": POPULATION,
            "offspring_size": OFFSPRING,
            "survivor_size": POPULATION,
            "stage2_new_candidate_quota": STAGE2_QUOTA,
            "framework": "StrictStage12V3Runner",
            "full1789": False,
        },
    )

    request = _fixed_request(fixed_k)
    results: dict[str, Any] = {}
    failures: list[dict[str, Any]] = []
    for target in targets:
        label = _label(target)
        try:
            anchor = _load_winner(root, target)
            validate_genotype_schema(anchor, space)
            anchor_phenotype = canonicalize_candidate(anchor, space)
            source = json.loads(
                (root / f"greedy/budget_{label}/exact_winner.json").read_text(
                    encoding="utf-8"
                )
            )
            if anchor_phenotype.to_dict() != source["phenotype"]:
                raise RuntimeError(
                    f"v2xvit_greedy_anchor_phenotype_drift:budget_{label}"
                )
            stage1 = UnifiedTaylorStage1Evaluator(
                space,
                baseline=baseline,
                structure_proxy=FunctionalGateTaylorProxy(gate32),
                weight_proxy=weight32,
                activation_cache=activation32,
                bops_evaluator=formal["bops"].evaluate_breakdown,
                size_evaluator=formal["size"].evaluate_breakdown,
                target=target,
                tolerance_abs=TOLERANCE,
                enforce_bops_hard_gate=True,
            )
            initial = build_initial_population(
                anchor,
                space=space,
                evaluator=stage1,
                seed=SEED + int(round(target * 1000)),
                size=POPULATION,
            )
            real = RealStage2Evaluator(
                root=root,
                label=label,
                seed=SEED,
                model=model,
                adapter=adapter,
                hypes=hypes,
                representative=representative,
                identity=identity,
                space=space,
                qkv_paths=qkv_paths,
                request=request,
                fixed50_manifest=args.fixed50_manifest,
                plugin=args.plugin,
                tensorrt_root=args.tensorrt_root,
                physical_gpu=args.physical_gpu,
                stage2_gpus=(),
            )
            greedy = real(anchor, 0)
            if not greedy.deployable:
                raise RuntimeError(
                    f"v2xvit_exact_greedy_anchor_not_deployable:budget_{label}"
                )
            config = StrictGAConfig(
                target_bops_retention=target,
                tolerance_abs=TOLERANCE,
                population_size=POPULATION,
                offspring_size=OFFSPRING,
                generations=GENERATIONS,
                stage2_new_candidate_quota=STAGE2_QUOTA,
                random_seed=SEED,
                generation_contract="formal_gen10",
            )
            runner = StrictStage12V3Runner(
                space,
                config,
                stage1_evaluator=stage1,
                stage2_evaluator=real,
            )
            budget_root = root / f"ga/budget_{label}/seed_0"

            def callback(record: Mapping[str, Any]) -> None:
                generation = int(record["generation"])
                atomic_write(
                    budget_root
                    / f"generation_{generation:02d}/generation_summary.json",
                    dict(record),
                )
                print(
                    json.dumps(
                        {
                            "event": "v2xvit_formal_ga_generation",
                            "budget": target,
                            "generation": generation,
                            "stage2_new_candidate_count": record.get(
                                "stage2_new_candidate_count", 0
                            ),
                            "generation_winner_hash": record.get(
                                "generation_winner_hash"
                            ),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

            outcome = runner.run(
                initial,
                greedy_anchor=greedy,
                generation_callback=callback,
            )
            final = best_real_candidate(list(outcome["evaluated"].values()), greedy)
            row = {
                "budget": target,
                "seed": 0,
                "completed_evolution_generations": outcome[
                    "completed_evolution_generations"
                ],
                "generation_zero_counted": outcome["generation_zero_counted"],
                "termination_reason": outcome["termination_reason"],
                "greedy_anchor": stage2_payload(greedy),
                "global_anchors": [
                    stage2_payload(item) for item in outcome["anchors"].unique()
                ],
                "final_winner": stage2_payload(final),
                "ga_improved_greedy": (
                    final.complete_phenotype_hash
                    != greedy.complete_phenotype_hash
                ),
                "stage2_real_evaluation_count": len(outcome["evaluated"]) - 1,
                "repair_counts": outcome["formal_ga_repair_counts"],
            }
            atomic_write(budget_root / "budget_summary.json", row)
            results[label] = row
        except Exception as exc:
            failure = {
                "budget": target,
                "status": "formal_budget_failed",
                "failure": f"{type(exc).__name__}:{exc}",
            }
            failures.append(failure)
            atomic_write(root / f"ga/budget_{label}/failure.json", failure)
            print(json.dumps(failure, sort_keys=True), flush=True)

    atomic_write(
        root / f"reports/formal_ga_results{shard_suffix}.json",
        {
            "model": "v2xvit",
            "framework": "StrictStage12V3Runner",
            "seed_count": 1,
            "executed_seeds": [0],
            "targets": list(targets),
            "shard_id": str(args.shard_id),
            "formal_generations": 10,
            "generation_zero_counted": False,
            "population_size": 64,
            "offspring_size": 64,
            "stage2_new_candidate_quota": 5,
            "results": results,
            "failures": failures,
            "full1789_executed": False,
        },
    )
    return 0 if not failures else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument(
        "--targets",
        type=_parse_targets,
        default=TARGETS,
        help="Comma-separated subset of the frozen six budgets.",
    )
    parser.add_argument(
        "--shard-id",
        default="",
        help="Unique suffix for shard-level cache/config/result reports.",
    )
    parser.add_argument("--fixed50-manifest", type=Path, required=True)
    parser.add_argument("--plugin", type=Path, required=True)
    parser.add_argument(
        "--tensorrt-root",
        type=Path,
        default=Path("/home/lixingfeng/UniAD_examine/TensorRT-10.9_x86_cu118"),
    )
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
