#!/usr/bin/env python3
"""Run six-budget V2X-ViT GA with the strict Stage-1/Stage-2/V1--V3 runner.

The six exact Greedy winners are immutable V1 anchors. Every budget runs one
seed (zero), generation 0 is initialization, and generations 1--5 are the five
formal evolution generations. Every real Stage-2 candidate uses materialization,
fresh calibration, export, TensorRT, and the same frozen 500-frame validation
manifest before it can influence selection.  Top-5 candidates use a 300-frame
screen, while the one selected winner from every generation is re-evaluated on
500 frames before the unique budget winner is chosen.
"""

from __future__ import annotations

import argparse
import json
import math
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
from search.ga.anchor_constrained_space import constrain_domains_to_frozen_anchors
from search.ga.frozen_domain_manifest import (
    build_manifest,
    load_manifest,
    validate_against_replay,
    write_manifest,
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
GENERATIONS = 5
POPULATION = 64
OFFSPRING = 64
STAGE2_QUOTA = 5
STAGE2_EVALUATION_FRAMES = 300
STAGE2_EVALUATION_WARMUP_FRAMES = 100
STAGE2_EVALUATION_PROTOCOL = "top5_fixed300_warmup100_screening"
GENERATION_WINNER_EVALUATION_FRAMES = 500
GENERATION_WINNER_EVALUATION_WARMUP_FRAMES = 200
GENERATION_WINNER_EVALUATION_PROTOCOL = "generation_winner_fixed500_warmup200"


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
    source_root = args.output_root.resolve()
    root = (args.run_root or args.output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    targets = tuple(args.targets)
    activation_taylor_fitness_weight = float(
        args.activation_taylor_fitness_weight
    )
    if (
        not math.isfinite(activation_taylor_fitness_weight)
        or activation_taylor_fitness_weight < 0.0
    ):
        raise RuntimeError("activation_taylor_fitness_weight_invalid")
    shard_suffix = _shard_suffix(args.shard_id)
    if int(args.seed) != SEED:
        raise RuntimeError("v2xvit_six_budget_ga_requires_seed_zero")
    if int(args.generations) != GENERATIONS:
        raise RuntimeError("v2xvit_six_budget_ga_requires_exactly_five_generations")
    if args.stage2_gpus:
        raise RuntimeError("v2xvit_campaign_forbids_cross_gpu_stage2")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible != str(args.physical_gpu):
        raise RuntimeError(
            "v2xvit_campaign_cuda_visible_devices_mismatch:"
            f"{visible}!={args.physical_gpu}"
        )
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"v2xvit_six_budget_ga_requires_one_visible_gpu:{torch.cuda.device_count()}"
        )
    required = (
        source_root / "reports/six_budget_greedy_winners.json",
        source_root / "reports/taylor_sample_convergence.json",
        source_root / "reports/input_provenance.json",
        args.stage2_manifest,
        args.plugin,
    )
    missing = [str(path) for path in required if not Path(path).is_file()]
    if missing:
        raise RuntimeError(f"v2xvit_six_budget_ga_required_artifacts_missing:{missing}")
    convergence = json.loads(
        (source_root / "reports/taylor_sample_convergence.json").read_text(encoding="utf-8")
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
        (source_root / "reports/input_provenance.json").read_text(encoding="utf-8")
    )
    source_activation_weight = float(
        provenance.get("activation_taylor_fitness_weight", 1.0)
    )
    if source_activation_weight != activation_taylor_fitness_weight:
        raise RuntimeError(
            "greedy_ga_activation_taylor_weight_mismatch:"
            f"{source_activation_weight}!={activation_taylor_fitness_weight}"
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
    replay_domains = rerank_domains_by_gate_scores(
        formal["space"].pruning_domains, gate32
    )

    # The first process that prepares this run freezes the complete ranking and
    # every legal width mask.  Subsequent shard processes must load this exact
    # table; rebuilding gate rankings in each process is the phenotype-drift
    # bug this runner is designed to prevent.
    anchor_records = []
    for anchor_target in targets:
        anchor_label = _label(anchor_target)
        anchor_path = source_root / f"greedy/budget_{anchor_label}/exact_winner.json"
        anchor_payload = json.loads(anchor_path.read_text(encoding="utf-8"))
        anchor_records.append({
            "candidate_hash": str(anchor_payload.get("candidate_hash", "")),
            "physical_phenotype_hash": str(
                anchor_payload.get("physical_phenotype_hash", "")
            ),
            "genotype": anchor_payload["genotype"],
            "phenotype": anchor_payload["phenotype"],
        })
    if args.prepare_frozen_domain_manifest:
        constrained = constrain_domains_to_frozen_anchors(
            replay_domains, [record["phenotype"] for record in anchor_records]
        )
        manifest = build_manifest(
            constrained,
            anchor_phenotypes=anchor_records,
            source_root=str(source_root),
            calibration_hash=calibration_hash,
            trace_hash=str(formal["space"].trace_snapshot_hash),
        )
        manifest_path = args.frozen_domain_manifest.resolve()
        write_manifest(manifest_path, manifest)
        # Round-trip and revalidate before any search process is allowed to
        # proceed.  This also catches incomplete grouped-domain serialization.
        _, restored = load_manifest(manifest_path)
        validate_against_replay(restored, replay_domains)
        restored_space = replace(
            formal["space"],
            pruning_domains=restored,
            pruning_unit_ids=[
                unit for domain in restored for unit in domain.ordered_unit_ids
            ],
        )
        for record in anchor_records:
            actual = canonicalize_candidate(
                CandidateGenotype.from_dict(record["genotype"]), restored_space
            ).to_dict()
            if actual != record["phenotype"]:
                raise RuntimeError(
                    "frozen_domain_manifest_does_not_restore_all_exact_anchors:"
                    f"{record['candidate_hash']}"
                )
        atomic_write(
            root / "reports/frozen_domain_manifest_preparation.json",
            {
                "status": "ok",
                "manifest": str(manifest_path),
                "manifest_hash": manifest["manifest_hash"],
                "domain_count": manifest["domain_count"],
                "anchor_count": len(anchor_records),
                "source_root": str(source_root),
                "search_not_started": True,
            },
        )
        return 0
    if args.frozen_domain_manifest is None:
        raise RuntimeError("v2xvit_six_budget_ga_frozen_domain_manifest_required")
    manifest_meta, frozen_domains = load_manifest(
        args.frozen_domain_manifest.resolve()
    )
    if str(manifest_meta.get("source_root")) != str(source_root):
        raise RuntimeError("frozen_domain_manifest_source_root_mismatch")
    if str(manifest_meta.get("calibration_hash")) != calibration_hash:
        raise RuntimeError("frozen_domain_manifest_calibration_hash_mismatch")
    validate_against_replay(frozen_domains, replay_domains)
    domains = frozen_domains
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
            "activation_taylor_fitness_weight": (
                activation_taylor_fitness_weight
            ),
            "activation_taylor_used_for_fitness": bool(
                activation_taylor_fitness_weight != 0.0
            ),
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
            "formal_generation_ids": list(range(1, GENERATIONS + 1)),
            "population_size": POPULATION,
            "offspring_size": OFFSPRING,
            "survivor_size": POPULATION,
            "stage2_new_candidate_quota": STAGE2_QUOTA,
            "stage2_evaluation_protocol": STAGE2_EVALUATION_PROTOCOL,
            "stage2_evaluation_frames": STAGE2_EVALUATION_FRAMES,
            "stage2_evaluation_warmup_frames": STAGE2_EVALUATION_WARMUP_FRAMES,
            "generation_winner_evaluation_protocol": (
                GENERATION_WINNER_EVALUATION_PROTOCOL
            ),
            "generation_winner_evaluation_frames": (
                GENERATION_WINNER_EVALUATION_FRAMES
            ),
            "generation_winner_evaluation_warmup_frames": (
                GENERATION_WINNER_EVALUATION_WARMUP_FRAMES
            ),
            "stage2_physical_gpus": list(args.stage2_gpus),
            "framework": "StrictStage12V3Runner",
            "stage1_proxy": (
                "J_struct_gate + J_WQ + "
                f"{activation_taylor_fitness_weight:g} * J_AQ"
            ),
            "activation_taylor_fitness_weight": (
                activation_taylor_fitness_weight
            ),
            "activation_taylor_used_for_fitness": bool(
                activation_taylor_fitness_weight != 0.0
            ),
            "activation_quantization_used_in_deployment": True,
            "artifact_source_root": str(source_root),
            "isolated_run_root": str(root),
            "frozen_domain_manifest": str(args.frozen_domain_manifest.resolve()),
            "frozen_domain_manifest_hash": manifest_meta["manifest_hash"],
            "frozen_domain_table_hash": manifest_meta["domain_table_hash"],
            "full1789": False,
        },
    )

    request = _fixed_request(fixed_k)
    results: dict[str, Any] = {}
    failures: list[dict[str, Any]] = []
    for target in targets:
        label = _label(target)
        try:
            anchor = _load_winner(source_root, target)
            validate_genotype_schema(anchor, space)
            anchor_phenotype = canonicalize_candidate(anchor, space)
            source = json.loads(
                (source_root / f"greedy/budget_{label}/exact_winner.json").read_text(
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
                activation_taylor_fitness_weight=(
                    activation_taylor_fitness_weight
                ),
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
                evaluation_manifest=args.stage2_manifest,
                evaluation_frames=STAGE2_EVALUATION_FRAMES,
                evaluation_warmup_frames=STAGE2_EVALUATION_WARMUP_FRAMES,
                evaluation_protocol=STAGE2_EVALUATION_PROTOCOL,
                plugin=args.plugin,
                tensorrt_root=args.tensorrt_root,
                physical_gpu=args.physical_gpu,
                stage2_gpus=tuple(args.stage2_gpus),
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
                generation_contract="formal_gen5",
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
            # The 300-frame Top-5 screen drives online V1/V2/V3 feedback.  The
            # authoritative budget winner is selected only from the Greedy
            # anchor and the unique per-generation winners after 500-frame
            # validation with a 200-frame warmup.
            greedy_validated = real.validate_generation_winner(
                greedy,
                0,
                evaluation_frames=GENERATION_WINNER_EVALUATION_FRAMES,
                evaluation_warmup_frames=(
                    GENERATION_WINNER_EVALUATION_WARMUP_FRAMES
                ),
                evaluation_protocol=GENERATION_WINNER_EVALUATION_PROTOCOL,
            )
            if not greedy_validated.deployable:
                raise RuntimeError(
                    f"v2xvit_greedy_anchor_fixed500_failed:budget_{label}"
                )
            generation_winner_generations: dict[str, int] = {}
            for record in outcome["history"]:
                winner_hash = record.get("generation_winner_hash")
                if winner_hash:
                    generation_winner_generations.setdefault(
                        str(winner_hash), int(record["generation"])
                    )
            generation_winner_validations = []
            for winner_hash, generation in generation_winner_generations.items():
                screening = outcome["evaluated"].get(winner_hash)
                if screening is None:
                    raise RuntimeError(
                        "generation_winner_screening_result_missing:"
                        f"{winner_hash}"
                    )
                generation_winner_validations.append(
                    real.validate_generation_winner(
                        screening,
                        generation,
                        evaluation_frames=GENERATION_WINNER_EVALUATION_FRAMES,
                        evaluation_warmup_frames=(
                            GENERATION_WINNER_EVALUATION_WARMUP_FRAMES
                        ),
                        evaluation_protocol=(
                            GENERATION_WINNER_EVALUATION_PROTOCOL
                        ),
                    )
                )
            final = best_real_candidate(
                [greedy_validated, *generation_winner_validations],
                greedy_validated,
            )
            row = {
                "budget": target,
                "seed": 0,
                "completed_evolution_generations": outcome[
                    "completed_evolution_generations"
                ],
                "generation_zero_counted": outcome["generation_zero_counted"],
                "termination_reason": outcome["termination_reason"],
                "greedy_anchor_screening": stage2_payload(greedy),
                "greedy_anchor": stage2_payload(greedy_validated),
                "global_anchors": [
                    stage2_payload(item) for item in outcome["anchors"].unique()
                ],
                "final_winner": stage2_payload(final),
                "ga_improved_greedy": (
                    final.complete_phenotype_hash
                    != greedy.complete_phenotype_hash
                ),
                "stage2_real_evaluation_count": len(outcome["evaluated"]) - 1,
                "stage2_top5_screening_frames": STAGE2_EVALUATION_FRAMES,
                "stage2_top5_screening_warmup_frames": (
                    STAGE2_EVALUATION_WARMUP_FRAMES
                ),
                "generation_winner_validation_frames": (
                    GENERATION_WINNER_EVALUATION_FRAMES
                ),
                "generation_winner_validation_warmup_frames": (
                    GENERATION_WINNER_EVALUATION_WARMUP_FRAMES
                ),
                "generation_winner_validation_count": len(
                    generation_winner_validations
                ),
                "generation_winner_validations": [
                    stage2_payload(item) for item in generation_winner_validations
                ],
                "repair_counts": outcome["formal_ga_repair_counts"],
                "activation_taylor_fitness_weight": (
                    activation_taylor_fitness_weight
                ),
                "activation_taylor_used_for_fitness": bool(
                    activation_taylor_fitness_weight != 0.0
                ),
                "activation_quantization_used_in_deployment": True,
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
            "formal_generations": GENERATIONS,
            "generation_zero_counted": False,
            "population_size": 64,
            "offspring_size": 64,
            "stage2_new_candidate_quota": 5,
            "stage2_evaluation_protocol": STAGE2_EVALUATION_PROTOCOL,
            "stage2_evaluation_frames": STAGE2_EVALUATION_FRAMES,
            "stage2_evaluation_warmup_frames": (
                STAGE2_EVALUATION_WARMUP_FRAMES
            ),
            "generation_winner_evaluation_frames": (
                GENERATION_WINNER_EVALUATION_FRAMES
            ),
            "generation_winner_evaluation_warmup_frames": (
                GENERATION_WINNER_EVALUATION_WARMUP_FRAMES
            ),
            "stage2_physical_gpus": list(args.stage2_gpus),
            "results": results,
            "failures": failures,
            "full1789_executed": False,
            "activation_taylor_fitness_weight": (
                activation_taylor_fitness_weight
            ),
            "activation_taylor_used_for_fitness": bool(
                activation_taylor_fitness_weight != 0.0
            ),
            "activation_quantization_used_in_deployment": True,
        },
    )
    return 0 if not failures else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=None,
        help="Isolated destination for repaired GA artifacts; output-root is read-only source.",
    )
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--generations", type=int, default=5)
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
    parser.add_argument(
        "--frozen-domain-manifest",
        type=Path,
        required=True,
        help="Canonical full-domain table shared by every repaired controller/worker.",
    )
    parser.add_argument(
        "--prepare-frozen-domain-manifest",
        action="store_true",
        help="Prepare and validate the manifest, then exit before search.",
    )
    parser.add_argument(
        "--stage2-manifest",
        type=Path,
        required=True,
        help="Fixed500 manifest used by every real Stage-2 candidate.",
    )
    parser.add_argument(
        "--stage2-gpus",
        type=lambda value: tuple(
            int(item.strip()) for item in value.split(",") if item.strip()
        ),
        default=(),
        help="Comma-separated physical GPUs for parallel Stage-2 candidates.",
    )
    parser.add_argument("--plugin", type=Path, required=True)
    parser.add_argument(
        "--tensorrt-root",
        type=Path,
        default=Path("/home/lixingfeng/UniAD_examine/TensorRT-10.9_x86_cu118"),
    )
    parser.add_argument(
        "--activation-taylor-fitness-weight",
        type=float,
        default=0.0,
        help=(
            "Stage-1 coefficient for raw J_AQ. This campaign defaults to zero; "
            "activation quantization remains enabled in deployment."
        ),
    )
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
