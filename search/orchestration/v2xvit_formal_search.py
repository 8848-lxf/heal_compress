"""Deployment-closed Greedy-anchor and strict GA search for HEAL V2X-ViT."""

from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Any, Mapping

from ..cache.proxy_cache import ProxyCache
from ..candidate import CandidateGenotype
from ..canonicalization import SearchSpaceSpec, canonicalize_candidate
from ..ga.strict_stage12_v3 import (
    Stage2Result,
    UnifiedTaylorStage1Evaluator,
    adjacent_mutation,
    phenotype_identity,
    score_stage2,
)
from ..greedy import GreedyBudgetSearch, GreedySearchConfig
from ..hashing import canonical_json_hash, search_hash
from ..model_family.calibration_manifest import load_v2xvit_train_manifest
from ..model_family.model_provider import load_heal_model_family
from ..model_family.search_smoke import (
    collect_v2xvit_manifest_fisher_statistics,
    load_v2xvit_manifest_batches,
)
from ..model_family.search_space import (
    build_ranked_v2xvit_ffn_domains,
    build_v2xvit_ffn_atomic_units,
    build_v2xvit_quantization_groups,
)
from ..proxy.bops_proxy import BOPSProxy
from ..proxy.gpu_batch_proxy import TorchBatchedProxyScorer
from ..proxy.joint_weight_taylor import JointWeightTaylorProxy
from ..proxy.normalization import NormalizationStats
from ..proxy.objective import ProxyObjective, ProxyObjectiveConfig
from ..proxy.parameter_slice_resolver import build_unit_parameter_slices
from ..proxy.runtime_shape_profiler import profile_runtime_layer_shapes
from ..proxy.size_proxy import SizeProxy
from ..pruning_space.domain_importance import score_atomic_units_for_fixed_ranking
from ..stage1.proxy_evaluator import Stage1ProxyEvaluator
from ..unified.config import PROJECT_ROOT, ResolvedSearchConfig
from ..unified.formal import run_strict_formal_ga


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _resolve(value: str | Path, *, label: str, file: bool = True) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    if file and not path.is_file():
        raise RuntimeError(f"v2xvit_formal_required_file_missing:{label}:{path}")
    if not file and not path.is_dir():
        raise RuntimeError(f"v2xvit_formal_required_directory_missing:{label}:{path}")
    return path


def _stage2_payload(result: Stage2Result) -> dict[str, Any]:
    return {
        "complete_phenotype_hash": result.complete_phenotype_hash,
        "genotype": result.genotype.to_dict(),
        "status": result.status,
        "mAP": result.map,
        "forward_p50_ms": result.p50_ms,
        "requested_realized_exact": result.requested_realized_exact,
        "evaluated": result.evaluated,
        "skipped": result.skipped,
        **dict(result.metadata),
    }


def _history_payload(history: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in history:
        row = dict(source)
        stage2_rows = []
        for scored in row.get("stage2_rows", []):
            value = dict(scored)
            result = value.get("result")
            if isinstance(result, Stage2Result):
                value["result"] = _stage2_payload(result)
            stage2_rows.append(value)
        if "stage2_rows" in row:
            row["stage2_rows"] = stage2_rows
        rows.append(row)
    return rows


class V2XViTFormalSearch:
    """Run the complete public V2X-ViT search without private path defaults."""

    def __init__(
        self,
        *,
        config: ResolvedSearchConfig,
        output_root: str | Path,
    ) -> None:
        self.config = config
        self.payload = config.payload
        self.output_root = Path(output_root).expanduser().resolve()

    def _build_search_state(self, run_dir: Path) -> dict[str, Any]:
        import torch

        model_cfg = dict(self.payload.get("model", {}) or {})
        runtime = dict(self.payload.get("runtime", {}) or {})
        proxy_cfg = dict(self.payload.get("proxy", {}) or {})
        checkpoint = _resolve(model_cfg["checkpoint"], label="checkpoint")
        config_path = _resolve(model_cfg["config"], label="model_config")
        heal_root = _resolve(runtime["heal_root"], label="heal_root", file=False)
        train_manifest_path = _resolve(
            proxy_cfg["quant_calibration_npz_manifest"],
            label="train_calibration_manifest",
        )
        device_name = str(runtime.get("device", "cuda:0"))
        device = torch.device(device_name)
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("v2xvit_formal_search_requires_cuda")
        torch.cuda.set_device(device)

        manifest = load_v2xvit_train_manifest(train_manifest_path)
        configured_fixed_k = int(model_cfg.get("fixed_k", 0) or 0)
        manifest_fixed_k = int(manifest["fixed_k_contract"]["value"])
        if configured_fixed_k and configured_fixed_k != manifest_fixed_k:
            raise RuntimeError(
                "v2xvit_fixed_k_config_manifest_mismatch:"
                f"{configured_fixed_k}!={manifest_fixed_k}"
            )
        bundle = load_heal_model_family(
            config_path=config_path,
            checkpoint_path=checkpoint,
            heal_root=heal_root,
            device=device_name,
            family_id="heal_lidar_v2xvit",
            forward_smoke=False,
        )
        fisher_samples = int(proxy_cfg.get("fisher_calibration_batches", 8))
        batches, evidence = load_v2xvit_manifest_batches(
            bundle,
            manifest,
            sample_count=fisher_samples,
            device=device,
        )
        statistics, fisher_report = collect_v2xvit_manifest_fisher_statistics(
            bundle, manifest, batches, evidence
        )
        _write_json(run_dir / "manifests/fisher_statistics.json", fisher_report)

        runtime_profile = profile_runtime_layer_shapes(
            bundle.model,
            batches[0],
            forward_fn=bundle.adapter.forward_for_task,
        )
        active_paths = sorted({row.module_path for row in runtime_profile.shapes})
        atomic_units, _ = build_v2xvit_ffn_atomic_units(bundle.model, bundle.audit)
        unit_slices = build_unit_parameter_slices(bundle.model, atomic_units)
        importance, ranking = score_atomic_units_for_fixed_ranking(
            bundle.model, statistics, unit_slices, strict=True
        )
        domains = build_ranked_v2xvit_ffn_domains(atomic_units, importance)
        quantization_groups = build_v2xvit_quantization_groups(
            bundle.model,
            bundle.audit,
            active_module_paths=active_paths,
        )
        capability = torch.cuda.get_device_capability(device)
        space = SearchSpaceSpec(
            pruning_unit_ids=[row.stable_id for row in atomic_units],
            precision_layer_ids=active_paths,
            quantization_groups=tuple(quantization_groups),
            pruning_domains=tuple(domains),
            default_precision="FP32",
            pruning_policy_version="v2xvit-ffn-legal-domain-width-fixed-ranking-v1",
            precision_policy_version="v2xvit-deployment-closed-v1",
            trace_snapshot_hash=bundle.audit.to_dict()["audit_hash"],
            calibration_manifest_hash=str(manifest["manifest_hash"]),
            onnx_export_config_hash=canonical_json_hash(
                {
                    "fixed_k": manifest_fixed_k,
                    "max_agents": int(model_cfg.get("max_agents", 2)),
                    "explicit_qdq": True,
                }
            ),
            tensorrt_version="runtime_verified",
            gpu_compute_capability=f"{capability[0]}.{capability[1]}",
            builder_flags={
                "strongly_typed": True,
                "explicit_qdq": True,
                "scatter_plugin": True,
            },
        )
        _write_json(
            run_dir / "manifests/search_space.json",
            {
                "family_id": "heal_lidar_v2xvit",
                "fixed_k": manifest_fixed_k,
                "domains": [row.to_dict() for row in domains],
                "quantization_groups": [row.to_dict() for row in quantization_groups],
                "ranking": ranking,
            },
        )

        normalization = NormalizationStats()
        size = SizeProxy(model=bundle.model, unit_to_parameter_slices=unit_slices)
        bops = BOPSProxy(
            model=bundle.model,
            unit_to_parameter_slices=unit_slices,
            runtime_shapes=runtime_profile.shapes,
        )
        joint = JointWeightTaylorProxy(
            bundle.model,
            statistics=statistics,
            unit_to_parameter_slices=unit_slices,
            strict=True,
        )
        objective_config = ProxyObjectiveConfig(
            objective_mode="joint_weight_taylor_hard_bops",
            bops_threshold=min(float(value) for value in self.payload["search"]["bops_targets"]),
            bops_constraint_mode="hard_band_feasibility",
            bops_tolerance_abs=float(proxy_cfg.get("bops_tolerance_abs", 0.005)),
            parameter_retention_tiebreak_epsilon=0.0,
        )
        objective = ProxyObjective(
            size=size,
            bops=bops,
            joint_weight_taylor=joint,
            normalization=normalization,
            config=objective_config,
        )
        scorer = TorchBatchedProxyScorer.from_components(
            model=bundle.model,
            space=space,
            unit_to_parameter_slices=unit_slices,
            fisher_statistics=statistics,
            runtime_shapes=runtime_profile.shapes,
            normalization=normalization,
            config=objective_config,
            device=device,
            batch_size=int(proxy_cfg.get("batch_size", 128)),
        )
        greedy_proxy = Stage1ProxyEvaluator(
            space,
            objective=objective,
            cache=ProxyCache(run_dir / "archives/proxy_archive.jsonl"),
            cache_key_fn=lambda phenotype, _space: search_hash(
                phenotype,
                trace_hash=space.trace_snapshot_hash,
                proxy_version="v2xvit-public-formal-joint-weight-taylor-v1",
                calibration_statistics_version=(
                    statistics.statistics_version + ":" + statistics.manifest_hash
                ),
            ),
            batch_scorer=scorer,
            proxy_backend="cuda_batched",
            proxy_device=device_name,
            proxy_batch_size=int(proxy_cfg.get("batch_size", 128)),
        )
        return {
            "bundle": bundle,
            "batches": batches,
            "bops": bops,
            "checkpoint": checkpoint,
            "config_path": config_path,
            "domains": domains,
            "greedy_proxy": greedy_proxy,
            "heal_root": heal_root,
            "joint": joint,
            "manifest": manifest,
            "manifest_path": train_manifest_path,
            "size": size,
            "space": space,
        }

    def _deploy(
        self,
        state: Mapping[str, Any],
        genotype: CandidateGenotype,
        *,
        target: float,
        generation: int,
        run_dir: Path,
    ) -> Stage2Result:
        from quantization.types import stable_json_hash
        from ..model_family.deployment import build_physical_structure_snapshot_v2
        from ..model_family.evaluation import evaluate_v2xvit_engine_modelopt
        from ..model_family.pruning import materialize_v2xvit_ffn_pruning
        from ..stage2.v2xvit_candidate_deployer import export_build_candidate

        runtime = dict(self.payload.get("runtime", {}) or {})
        stage2 = dict(self.payload.get("stage2", {}) or {})
        model_cfg = dict(self.payload.get("model", {}) or {})
        space = state["space"]
        phenotype = canonicalize_candidate(genotype, space)
        identity = phenotype_identity(genotype, space)
        candidate_id = identity["complete_phenotype_hash"]
        candidate_dir = (
            run_dir
            / "stage2"
            / f"budget_{int(round(target * 1000)):04d}"
            / f"generation_{generation:02d}"
            / candidate_id[:16]
        )
        try:
            physical = materialize_v2xvit_ffn_pruning(
                state["bundle"].model, phenotype, state["domains"]
            )
            snapshot = build_physical_structure_snapshot_v2(physical.model)
            shapes = {
                name: list(value.shape)
                for name, value in sorted(physical.model.state_dict().items())
            }
            physical_report = {
                "structure_hash": physical.snapshot_hash,
                "state_dict_shape_hash": stable_json_hash(shapes),
                "snapshot_hash": snapshot["snapshot_hash"],
            }
            build = export_build_candidate(
                candidate_dir=candidate_dir,
                model=physical.model,
                adapter=state["bundle"].adapter,
                hypes={},
                real_batch=state["batches"][0],
                module_precision_profile=phenotype.realized_precision_profile,
                candidate_identity={
                    "profile_id": candidate_id,
                    "config_path": str(state["config_path"]),
                    "target_bops_retention": target,
                },
                train200_manifest=state["manifest"],
                train200_manifest_path=state["manifest_path"],
                checkpoint_path=state["checkpoint"],
                physical_report=physical_report,
                fixed_k=int(state["manifest"]["fixed_k_contract"]["value"]),
                physical_gpu=int(runtime.get("physical_gpu", 0)),
                tensorrt_root=_resolve(
                    runtime["tensorrt_root"], label="tensorrt_root", file=False
                ),
                plugin_path=_resolve(runtime["plugin_path"], label="scatter_plugin"),
            )
            if build.get("status") != "ok":
                raise RuntimeError(
                    f"v2xvit_stage2_engine_rejected:{build.get('failure_reason', build.get('status'))}"
                )
            evaluation = evaluate_v2xvit_engine_modelopt(
                engine_path=build["engine_path"],
                model_config=state["config_path"],
                heal_root=state["heal_root"],
                output_dir=candidate_dir / "evaluation",
                tensorrt_root=_resolve(
                    runtime["tensorrt_root"], label="tensorrt_root", file=False
                ),
                plugin_path=_resolve(runtime["plugin_path"], label="scatter_plugin"),
                eval_manifest_path=_resolve(
                    stage2["evaluation_manifest"], label="evaluation_manifest"
                ),
                physical_gpu_id=int(runtime.get("physical_gpu", 0)),
                fixed_k=int(state["manifest"]["fixed_k_contract"]["value"]),
                max_agents=int(model_cfg.get("max_agents", 2)),
                num_frames=int(stage2.get("num_frames", 500)),
                warmup_frames=int(stage2.get("warmup_frames", 200)),
                latency_rounds=int(stage2.get("latency_rounds", 3)),
                dataloader_num_workers=int(stage2.get("dataloader_num_workers", 8)),
            )
            if evaluation.get("status") != "ok":
                raise RuntimeError(
                    f"v2xvit_stage2_evaluation_rejected:{evaluation.get('failure_reason', evaluation.get('status'))}"
                )
            result = Stage2Result(
                complete_phenotype_hash=candidate_id,
                genotype=genotype,
                status="ok",
                map=float(evaluation["mAP"]),
                p50_ms=float(evaluation["forward_p50_ms"]),
                requested_realized_exact=bool(build["requested_realized_exact"]),
                evaluated=int(evaluation["num_evaluated_frames"]),
                skipped=int(evaluation["num_skipped_frames"]),
                metadata={
                    "engine_path": str(build["engine_path"]),
                    "candidate_dir": str(candidate_dir),
                    "target_bops_retention": target,
                },
            )
        except Exception as exc:
            result = Stage2Result(
                complete_phenotype_hash=candidate_id,
                genotype=genotype,
                status="failed",
                metadata={
                    "candidate_dir": str(candidate_dir),
                    "failure_reason": f"{type(exc).__name__}:{exc}",
                    "target_bops_retention": target,
                },
            )
        _write_json(candidate_dir / "formal_stage2_result.json", _stage2_payload(result))
        return result

    def _full_validate(
        self,
        state: Mapping[str, Any],
        winner: Stage2Result,
        *,
        target: float,
        run_dir: Path,
    ) -> dict[str, Any]:
        from ..model_family.evaluation import evaluate_v2xvit_engine_modelopt

        runtime = dict(self.payload.get("runtime", {}) or {})
        stage2 = dict(self.payload.get("stage2", {}) or {})
        full = dict(self.payload.get("full_validation", {}) or {})
        model_cfg = dict(self.payload.get("model", {}) or {})
        engine_path = str(winner.metadata.get("engine_path", ""))
        if not engine_path:
            raise RuntimeError("v2xvit_full_validation_winner_engine_missing")
        manifest_value = full.get("evaluation_manifest") or stage2.get(
            "evaluation_manifest"
        )
        evaluation = evaluate_v2xvit_engine_modelopt(
            engine_path=engine_path,
            model_config=state["config_path"],
            heal_root=state["heal_root"],
            output_dir=(
                run_dir
                / "full_validation"
                / f"budget_{int(round(target * 1000)):04d}"
            ),
            tensorrt_root=_resolve(
                runtime["tensorrt_root"], label="tensorrt_root", file=False
            ),
            plugin_path=_resolve(runtime["plugin_path"], label="scatter_plugin"),
            eval_manifest_path=_resolve(
                manifest_value, label="full_evaluation_manifest"
            ),
            physical_gpu_id=int(runtime.get("physical_gpu", 0)),
            fixed_k=int(state["manifest"]["fixed_k_contract"]["value"]),
            max_agents=int(model_cfg.get("max_agents", 2)),
            num_frames=int(full.get("num_frames", 1789)),
            warmup_frames=int(full.get("warmup_frames", 200)),
            latency_rounds=int(full.get("latency_rounds", 3)),
            dataloader_num_workers=int(full.get("dataloader_num_workers", 8)),
        )
        if evaluation.get("status") != "ok":
            raise RuntimeError(
                "v2xvit_full_validation_failed:"
                f"{evaluation.get('failure_reason', evaluation.get('status'))}"
            )
        return dict(evaluation)

    @staticmethod
    def _initial_population(
        *,
        anchor: CandidateGenotype,
        space: SearchSpaceSpec,
        evaluator: Any,
        population_size: int,
        seed: int,
    ) -> list[CandidateGenotype]:
        rng = random.Random(seed)
        feasible: dict[str, CandidateGenotype] = {}
        frontier = [anchor]
        near: list[CandidateGenotype] = [anchor]
        anchor_hash = phenotype_identity(anchor, space)["complete_phenotype_hash"]
        feasible[anchor_hash] = anchor
        attempts = 0
        maximum_attempts = population_size * 5000
        while len(feasible) < population_size and attempts < maximum_attempts:
            attempts += 1
            parent = rng.choice(frontier if frontier and rng.random() < 0.8 else near)
            child = adjacent_mutation(parent, space, rng)
            metrics = dict(evaluator(child))
            identity = str(metrics["complete_phenotype_hash"])
            if bool(metrics["bops_feasible"]):
                feasible.setdefault(identity, child)
                frontier.append(child)
            elif float(metrics["bops_deviation"]) <= 0.025:
                near.append(child)
            if len(frontier) > population_size * 8:
                frontier = frontier[-population_size * 8 :]
            if len(near) > population_size * 16:
                near = near[-population_size * 16 :]
        if len(feasible) != population_size:
            raise RuntimeError(
                "v2xvit_formal_initial_population_incomplete:"
                f"{len(feasible)}/{population_size}:attempts={attempts}"
            )
        return list(feasible.values())

    def run(self) -> dict[str, Any]:
        search = dict(self.payload.get("search", {}) or {})
        proxy = dict(self.payload.get("proxy", {}) or {})
        experiment = str(
            dict(self.payload.get("output", {}) or {}).get(
                "experiment_name", "lidar_v2xvit_unified_ga"
            )
        )
        run_dir = self.output_root / experiment
        run_dir.mkdir(parents=True, exist_ok=False)
        state = self._build_search_state(run_dir)
        space = state["space"]
        targets = tuple(float(value) for value in search["bops_targets"])
        greedy = GreedyBudgetSearch(
            space,
            config=GreedySearchConfig(
                bops_targets=targets,
                bops_tolerance_abs=float(proxy.get("bops_tolerance_abs", 0.005)),
                budget_recovery_beam_width=int(search.get("budget_recovery_beam_width", 8)),
                budget_recovery_seed_pool_size=int(
                    search.get("budget_recovery_seed_pool_size", 32)
                ),
                budget_recovery_max_depth=int(search.get("budget_recovery_max_depth", 64)),
            ),
        ).run(
            lambda candidates, step: state["greedy_proxy"].evaluate_batch(
                candidates, generation=step, outer_round=0
            ).metrics
        )
        _write_json(run_dir / "greedy/search_result.json", greedy.to_dict())
        missing = [target for target in targets if target not in greedy.budget_candidates]
        if missing:
            raise RuntimeError(f"v2xvit_greedy_budget_unreachable:{missing}")

        anchors = {
            target: self._deploy(
                state,
                greedy.budget_candidates[target],
                target=target,
                generation=0,
                run_dir=run_dir,
            )
            for target in targets
        }
        if not all(anchor.deployable for anchor in anchors.values()):
            failures = {
                target: _stage2_payload(anchor)
                for target, anchor in anchors.items()
                if not anchor.deployable
            }
            raise RuntimeError(f"v2xvit_greedy_anchor_deployment_failed:{failures}")
        anchor_manifest = {
            "schema_version": "heal-unified-greedy-anchor-v1",
            "family_id": "heal_lidar_v2xvit",
            "anchors": {
                f"{target:.6f}": _stage2_payload(anchor)
                for target, anchor in anchors.items()
            },
        }
        _write_json(run_dir / "greedy/anchor_manifest.json", anchor_manifest)
        if str(search.get("method", "ga")) == "greedy":
            primary = anchors[min(targets, key=lambda value: abs(value - 0.10))]
            full_validation = {
                f"{target:.6f}": self._full_validate(
                    state, anchor, target=target, run_dir=run_dir
                )
                for target, anchor in anchors.items()
            }
            return {
                "status": "ok",
                "family_id": "heal_lidar_v2xvit",
                "method": "greedy",
                "run_dir": str(run_dir),
                "anchors": anchor_manifest["anchors"],
                "full_validation": full_validation,
                "best": _stage2_payload(primary),
            }

        results: dict[str, Any] = {}
        target_winners: dict[float, Stage2Result] = {}
        for ordinal, target in enumerate(targets):
            baseline = CandidateGenotype(
                pruning_width_genes={
                    domain.domain_id: int(domain.original_width)
                    for domain in space.pruning_domains
                },
                precision_genes={
                    group.group_id: str(group.allowed_precisions[0])
                    for group in space.quantization_groups
                    if group.group_id in space.precision_gene_ids
                },
                meta={"created_by": "v2xvit_formal_baseline"},
            )
            stage1 = UnifiedTaylorStage1Evaluator(
                space,
                baseline=baseline,
                structure_proxy=state["joint"],
                weight_proxy=state["joint"],
                activation_cache=None,
                bops_evaluator=state["bops"].evaluate_breakdown,
                size_evaluator=state["size"].evaluate_breakdown,
                target=target,
                tolerance_abs=float(proxy.get("bops_tolerance_abs", 0.005)),
                enforce_bops_hard_gate=True,
                include_activation_taylor=bool(proxy.get("include_activation_taylor", False)),
            )
            initial = self._initial_population(
                anchor=anchors[target].genotype,
                space=space,
                evaluator=stage1,
                population_size=int(search.get("population_size", 64)),
                seed=int(search.get("seed", 42)) + ordinal,
            )
            formal = run_strict_formal_ga(
                self.config,
                space=space,
                initial_population=initial,
                greedy_anchor=anchors[target],
                stage1_evaluator=stage1,
                stage2_evaluator=lambda candidate, generation, current=target: self._deploy(
                    state,
                    candidate,
                    target=current,
                    generation=generation,
                    run_dir=run_dir,
                ),
                target=target,
            )
            candidates = list(formal["evaluated"].values())
            winner = min(
                candidates,
                key=lambda row: (
                    float(
                        score_stage2(
                            row,
                            greedy_map=float(anchors[target].map),
                            greedy_p50_ms=float(anchors[target].p50_ms),
                            accuracy_tolerance=float(
                                self.payload["stage2"]["greedy_anchor_accuracy_gate"][
                                    "tolerance"
                                ]
                            ),
                        )["F_S2"]
                    ),
                    row.complete_phenotype_hash,
                ),
            )
            target_winners[target] = winner
            full_validation = self._full_validate(
                state, winner, target=target, run_dir=run_dir
            )
            results[f"{target:.6f}"] = {
                "completed_evolution_generations": formal[
                    "completed_evolution_generations"
                ],
                "termination_reason": formal["termination_reason"],
                "history": _history_payload(formal["history"]),
                "winner": _stage2_payload(winner),
                "full_validation": full_validation,
            }
            _write_json(
                run_dir / f"ga/budget_{int(round(target * 1000)):04d}/result.json",
                results[f"{target:.6f}"],
            )
        primary = target_winners[min(targets, key=lambda value: abs(value - 0.10))]
        summary = {
            "status": "ok",
            "family_id": "heal_lidar_v2xvit",
            "formal_protocol": "strict_stage12_v3",
            "run_dir": str(run_dir),
            "targets": results,
            "best": _stage2_payload(primary),
        }
        _write_json(run_dir / "formal_search_summary.json", summary)
        return summary


__all__ = ["V2XViTFormalSearch"]
