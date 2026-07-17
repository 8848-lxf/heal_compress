"""Real lidar_pyramid two-stage GA orchestration."""

from __future__ import annotations

import csv
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

from ..cache.proxy_cache import ProxyCache
from ..candidate import CandidateGenotype, CandidatePhenotype
from ..canonicalization import canonicalize_candidate
from ..ga.engine import GAConfig, GeneticSearchEngine
from ..greedy import GreedyBudgetSearch, GreedySearchConfig
from ..hashing import candidate_hash, canonical_json_hash, search_hash
from ..integration.calibration_provider import collect_or_load_fisher_statistics
from ..integration.lidar_pyramid_context import build_lidar_pyramid_context
from ..integration.model_provider import load_lidar_pyramid_model
from ..integration.runtime_environment import GPUSelection
from ..proxy.bops_proxy import BOPSProxy
from ..proxy.fisher_proxy import FisherTaylorProxy
from ..proxy.joint_weight_taylor import JointWeightTaylorProxy
from ..proxy.normalization import NormalizationStats, build_normalization_stats
from ..proxy.objective import ProxyObjective, ProxyObjectiveConfig, bops_soft_penalty, bops_target_for_generation, bops_target_for_outer_round
from ..proxy.gpu_batch_proxy import (
    MultiDeviceTorchBatchedProxyScorer,
    TorchBatchedProxyScorer,
)
from ..integration.runtime_environment import query_gpus
from ..proxy.parameter_slice_resolver import build_unit_parameter_slices
from ..proxy.runtime_shape_profiler import profile_runtime_layer_shapes
from ..proxy.size_proxy import SizeProxy
from ..proxy.sqnr_proxy import SQNRProxy
from ..pruning_space.mask_repair import GroupedDomainSpec, RepairPolicy, dense_floor_repair, grouped_equal_count_floor_repair
from ..pruning_space.domain_importance import score_atomic_units_for_fixed_ranking
from ..pruning_space.local_domains import build_local_pruning_domains
from ..stage1.proxy_evaluator import Stage1ProxyEvaluator
from ..stage1.repair_selection import select_repaired_stage2_topk
from ..stage1.topk_selector import ProxyCandidateRecord, TopKConfig, select_stage1_topk
from ..stage2.lidar_pyramid_real_evaluator import LidarPyramidRealEvaluator
from ..stage2.objective import Stage2ObjectiveConfig
from ..stage2.repaired_topk_manifest import write_repaired_topk_manifest
from ..stage2.round_results import write_round_stage2_results


def _write_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{time.time_ns()}.tmp"
    )
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    temporary.replace(destination)


def _append_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def _load_candidate(path: str | Path) -> CandidateGenotype | CandidatePhenotype:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "pruned_unit_ids" in payload or "precision_profile" in payload:
        return CandidatePhenotype.from_dict(payload)
    return CandidateGenotype.from_dict(payload)


def _proxy_device_from_config(proxy_cfg: dict[str, Any], context: Any) -> str:
    requested = str(proxy_cfg.get("device", "cpu")).lower()
    if requested == "auto":
        return str(context.runtime_device)
    if requested == "cuda":
        return str(context.runtime_device)
    return str(proxy_cfg.get("device", "cpu"))


def _select_idle_gpu_pool(
    runtime: dict[str, Any],
    *,
    role: str,
    primary_gpu_id: int,
) -> tuple[list[int], list[dict[str, Any]]]:
    report = query_gpus()
    configured = runtime.get(f"{role}_gpu_ids", runtime.get("parallel_gpu_ids"))
    requested = (
        [int(value) for value in configured]
        if configured not in (None, "", "auto")
        else [int(primary_gpu_id)]
    )
    minimum_free = int(runtime.get("parallel_gpu_min_free_mib", 60_000))
    maximum_utilization = int(runtime.get("parallel_gpu_max_utilization_pct", 10))
    by_id = {int(row["index"]): row for row in report}
    selected = [
        gpu_id
        for gpu_id in requested
        if gpu_id in by_id
        and int(by_id[gpu_id]["memory_free_mib"]) >= minimum_free
        and int(by_id[gpu_id]["utilization_gpu_pct"]) <= maximum_utilization
    ]
    if int(primary_gpu_id) in requested and int(primary_gpu_id) not in selected:
        primary = by_id.get(int(primary_gpu_id), {})
        # The main model itself consumes memory on the selected GPU before this
        # audit. Retain it when no unrelated high-utilization task is present.
        if primary and int(primary.get("utilization_gpu_pct", 100)) <= maximum_utilization:
            selected.append(int(primary_gpu_id))
    if not selected:
        selected = [int(primary_gpu_id)]
    maximum_workers = int(runtime.get(f"{role}_max_workers", len(selected)))
    return selected[: max(1, maximum_workers)], report


def _require_gpu_proxy_if_needed(*, proxy_cfg: dict[str, Any], search_cfg: dict[str, Any], actual_backend: str) -> None:
    requested = str(proxy_cfg.get("device", "cpu")).lower()
    initial = int(search_cfg.get("initial_population_size", search_cfg.get("population_size", 0)) or 0)
    if requested != "cpu" and initial >= 512 and actual_backend != "cuda_batched":
        raise RuntimeError("gpu_proxy_required_but_not_active")


def _shared_eval_manifest_protocol(
    stage2_cfg: dict[str, Any],
    full_validation_cfg: dict[str, Any],
) -> tuple[int, int, bool]:
    """Create one manifest large enough for Stage-2 and final validation."""

    return (
        max(
            int(stage2_cfg.get("num_frames", 5)),
            int(full_validation_cfg.get("num_frames", 0) or 0),
        ),
        max(
            int(stage2_cfg.get("warmup_frames", 10)),
            int(full_validation_cfg.get("warmup_frames", 0) or 0),
        ),
        bool(
            stage2_cfg.get("reset_after_warmup", False)
            or full_validation_cfg.get("reset_after_warmup", False)
        ),
    )


class LidarPyramidTwoStageSearch:
    def __init__(self, *, config: dict[str, Any], checkpoint: str | Path, output_root: str | Path, resume: str | Path | None = None) -> None:
        self.config = config
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        self.output_root = Path(output_root)
        self.resume = Path(resume).expanduser().resolve() if resume else None

    def _run_dir(self) -> Path:
        if self.resume is not None:
            if not self.resume.is_dir():
                raise RuntimeError(f"resume_run_dir_missing:{self.resume}")
            return self.resume
        stamp = time.strftime("%Y%m%d_%H%M%S")
        experiment = str(self.config.get("output", {}).get("experiment_name", "two_stage_joint_search"))
        candidate = self.output_root / f"{experiment}_{stamp}"
        suffix = 0
        while candidate.exists():
            suffix += 1
            candidate = self.output_root / f"{experiment}_{stamp}_{suffix:02d}"
        candidate.mkdir(parents=True, exist_ok=False)
        return candidate

    def run(
        self,
        *,
        stage1_only: bool = False,
        stage2_only: bool = False,
        baseline_only: bool = False,
        candidate_config: str | Path | list[str] | list[Path] | None = None,
    ) -> dict[str, Any]:
        run_dir = self._run_dir()
        (run_dir / "archives").mkdir(parents=True, exist_ok=True)
        (run_dir / "baseline").mkdir(parents=True, exist_ok=True)
        manifest_path = run_dir / "run_manifest.json"
        initial_manifest: dict[str, Any] = {}
        if self.resume is not None and manifest_path.is_file():
            initial_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        initial_manifest.update(
            {
                "checkpoint": str(self.checkpoint),
                "resume": str(self.resume or ""),
                "stage1_only": stage1_only,
                "stage2_only": stage2_only,
                "baseline_only": baseline_only,
            }
        )
        _write_json(manifest_path, initial_manifest)
        runtime = dict(self.config.get("runtime", {}))
        search_cfg = dict(self.config.get("search", {}))
        pruning_cfg = dict(self.config.get("pruning", {}))
        proxy_cfg = dict(self.config.get("proxy", self.config.get("proxy_objective", {})))
        stage2_cfg = dict(self.config.get("stage2") or self.config.get("stage2_smoke") or self.config.get("evaluation", {}))
        full_validation_cfg = dict(self.config.get("full_validation", {}) or {})
        model_cfg = dict(self.config.get("model", {}))
        (
            manifest_num_frames,
            manifest_warmup_frames,
            manifest_reset_after_warmup,
        ) = _shared_eval_manifest_protocol(
            stage2_cfg,
            full_validation_cfg,
        )
        context = build_lidar_pyramid_context(
            checkpoint_path=self.checkpoint,
            output_dir=run_dir,
            model_config_path=model_cfg.get("config") or model_cfg.get("hypes_yaml"),
            heal_root=runtime.get("heal_root", "/home/lixingfeng/UniAD_examine/HEAL"),
            tensorrt_root=runtime.get("tensorrt_root", "/home/lixingfeng/UniAD_examine/HEAL/prune_model/TensorRT-10.9_x86_cu118"),
            plugin_path=runtime.get("plugin_path"),
            gpu_id=str(runtime.get("gpu_id", "auto")),
            exclude_gpu_ids=[int(v) for v in runtime.get("exclude_gpu_ids", [5, 6, 7])],
            tensorrt_env=str(runtime.get("tensorrt_env", "modelopt")),
            fisher_calibration_batches=int(proxy_cfg.get("fisher_calibration_batches", 8)),
            quant_calibration_batches=int(proxy_cfg.get("quant_calibration_batches", 16)),
            quant_calibration_npz_manifest=proxy_cfg.get("quant_calibration_npz_manifest"),
            quant_activation_calibration_backend=str(
                proxy_cfg.get("quant_activation_calibration_backend", "modelopt_histogram_entropy")
            ),
            quant_activation_calibration_cache_path=proxy_cfg.get(
                "quant_activation_calibration_cache_path"
            ),
            quant_calibration_force_rebuild=bool(proxy_cfg.get("quant_calibration_force_rebuild", False)),
            num_frames=manifest_num_frames,
            warmup_frames=manifest_warmup_frames,
            reset_after_warmup=manifest_reset_after_warmup,
            default_precision=str(self.config.get("precision", {}).get("default", "FP16")),
            max_pruning_units=int(search_cfg.get("max_pruning_units", 96)),
            grouped_conv_mode=str(
                pruning_cfg.get("grouped_conv_mode")
                or dict(pruning_cfg.get("grouped_conv", {}) or {}).get("position_mode", "shared_local_mean")
            ),
            grouped_conv_align=int(
                pruning_cfg.get("grouped_conv_align")
                or dict(pruning_cfg.get("grouped_conv", {}) or {}).get("default_channels_per_group", 8)
            ),
            grouped_allowed_channels_per_group=[
                int(value)
                for value in dict(pruning_cfg.get("grouped_conv", {}) or {}).get(
                    "allowed_channels_per_group",
                    [4, 8, 16, 32, 64, 128, 256, 512],
                )
            ],
            pruning_gene_type=str(pruning_cfg.get("gene_type", pruning_cfg.get("search_variable", "legal_pruning_action"))),
        )
        _write_json(run_dir / "environment.json", {"gpu": context.gpu_selection.to_dict(), "tensorrt": context.tensorrt.to_dict()})
        stage1_gpu_ids, stage1_gpu_report = _select_idle_gpu_pool(
            runtime,
            role="stage1",
            primary_gpu_id=context.physical_gpu_id,
        )
        stage2_gpu_ids, stage2_gpu_report = _select_idle_gpu_pool(
            runtime,
            role="stage2",
            primary_gpu_id=context.physical_gpu_id,
        )
        _write_json(
            run_dir / "gpu_parallelism_manifest.json",
            {
                "selection_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "selection_policy": {
                    "minimum_free_memory_mib": int(
                        runtime.get("parallel_gpu_min_free_mib", 60_000)
                    ),
                    "maximum_utilization_pct": int(
                        runtime.get("parallel_gpu_max_utilization_pct", 10)
                    ),
                    "candidate_gpu_ids": runtime.get("parallel_gpu_ids", []),
                },
                "stage1_gpu_ids": stage1_gpu_ids,
                "stage2_gpu_ids": stage2_gpu_ids,
                "stage1_gpu_snapshot": stage1_gpu_report,
                "stage2_gpu_snapshot": stage2_gpu_report,
                "stage2_worker_policy": "one_candidate_per_gpu_sequential_queue",
                "final_latency_policy": "serial_no_cross_gpu_concurrency",
            },
        )
        self._stage2_gpu_ids = list(stage2_gpu_ids)
        self._runtime_config = dict(runtime)
        if baseline_only:
            real_evaluator = LidarPyramidRealEvaluator(
                context=context,
                run_dir=run_dir,
                num_frames=int(stage2_cfg.get("num_frames", 5)),
                warmup_frames=int(stage2_cfg.get("warmup_frames", 10)),
                latency_rounds=int(stage2_cfg.get("latency_rounds", stage2_cfg.get("rounds", 1))),
                stage2_config=Stage2ObjectiveConfig(
                    eta_map=float(stage2_cfg.get("eta_ap", stage2_cfg.get("eta_map", 1.0))),
                    eta_latency=float(stage2_cfg.get("eta_latency", 1.0)),
                    latency_metric=str(stage2_cfg.get("latency_metric", "forward_mean_ms")),
                    accuracy_reference=str(stage2_cfg.get("accuracy_reference", "original_strict_fp32")),
                    latency_reference=str(stage2_cfg.get("latency_reference", "original_strict_fp32")),
                    tau_ap=stage2_cfg.get("tau_ap"),
                    max_map_drop=stage2_cfg.get("max_map_drop"),
                ),
            )
            baseline_cfg = dict(self.config.get("baselines", {}) or {})
            precisions = [
                str(value)
                for value in baseline_cfg.get(
                    "precisions",
                    ["strict_fp16", "trusted_explicit_qdq_int8"],
                )
            ]
            rows = real_evaluator.evaluate_original_baselines(precisions)
            return {
                "run_dir": str(run_dir),
                "selected_gpu": context.physical_gpu_id,
                "baseline_only": True,
                "baselines": rows,
            }
        raw_unit_slices = build_unit_parameter_slices(context.model, context.atomic_prune_units)
        pruning_gene_type = str(pruning_cfg.get("gene_type", pruning_cfg.get("search_variable", "legal_pruning_action")))
        domain_width_mode = pruning_gene_type in {
            "legal_domain_width",
            "domain_width",
            "coupled_domain_width",
        }
        unit_slices = (
            raw_unit_slices
            if domain_width_mode or pruning_gene_type == "coupled_channel_keep_mask"
            else self._action_slices(context, raw_unit_slices)
        )
        runtime_shapes = profile_runtime_layer_shapes(
            context.model,
            context.trace_example_inputs,
            forward_fn=context.model_bundle.adapter.forward_for_task,
        )
        _write_json(run_dir / "runtime_layer_shapes.json", runtime_shapes.to_dict())
        fisher_stats = collect_or_load_fisher_statistics(
            model=context.model,
            adapter=context.model_bundle.adapter,
            model_config_path=context.model_config,
            device=__import__("torch").device(context.runtime_device),
            cache_path=run_dir / "archives" / "fisher_statistics.pt",
            num_batches=context.fisher_calibration_batches,
        )
        _write_json(
            run_dir / "archives" / "fisher_statistics_manifest.json",
            {"manifest_hash": fisher_stats.manifest_hash, "statistics_version": fisher_stats.statistics_version, "path": str(run_dir / "archives" / "fisher_statistics.pt")},
        )
        if domain_width_mode:
            fixed_scores, ranking_manifest = score_atomic_units_for_fixed_ranking(
                context.model,
                fisher_stats,
                raw_unit_slices,
                strict=True,
            )
            grouped_cfg = dict(pruning_cfg.get("grouped_conv", {}) or {})
            dense_cfg = dict(pruning_cfg.get("dense", {}) or {})
            domains = build_local_pruning_domains(
                context.atomic_prune_units,
                importance_scores=fixed_scores,
                ranking_method="pruning_only_first_plus_second_order_fisher_taylor",
                minimum_retained_ratio=float(pruning_cfg.get("minimum_retained_ratio", 0.10)),
                dense_alignment=int(
                    dense_cfg.get("alignment", pruning_cfg.get("dense_channel_alignment", 4))
                ),
                grouped_allowed_channels_per_group=[
                    int(value)
                    for value in grouped_cfg.get(
                        "allowed_channels_per_group",
                        [4, 8, 16, 32, 64, 128, 256, 512],
                    )
                ],
            )
            if not any(len(domain.legal_widths) > 1 for domain in domains):
                raise RuntimeError("domain_width_search_has_no_nontrivial_legal_gene")
            context.search_space = replace(
                context.search_space,
                pruning_domains=tuple(domains),
                pruning_policy_version="legal-domain-width-fixed-ranking-v1",
            )
            _write_json(
                run_dir / "archives" / "fixed_pruning_taylor_ranking.json",
                ranking_manifest,
            )
        self._write_local_domains(context, raw_unit_slices, run_dir)
        raw_objective = self._objective(context, unit_slices, fisher_stats, None, runtime_shapes.shapes)
        normalization = self._build_normalization(context, raw_objective, run_dir)
        objective = self._objective(context, unit_slices, fisher_stats, normalization, runtime_shapes.shapes)
        proxy_cache = ProxyCache(run_dir / "archives" / "proxy_archive.jsonl")
        proxy_device = _proxy_device_from_config(proxy_cfg, context)
        proxy_batch_size = int(proxy_cfg.get("batch_size", proxy_cfg.get("proxy_batch_size", 128)))
        batch_scorer = None
        proxy_backend = "scalar_cpu"
        proxy_gpu_ids: list[int] = []
        if str(proxy_device).startswith("cuda"):
            primary_scorer = TorchBatchedProxyScorer.from_components(
                model=context.model,
                space=context.search_space,
                unit_to_parameter_slices=unit_slices,
                fisher_statistics=fisher_stats,
                runtime_shapes=runtime_shapes.shapes,
                normalization=normalization,
                config=objective.config,
                device=proxy_device,
                batch_size=proxy_batch_size,
            )
            proxy_gpu_ids = list(stage1_gpu_ids)
            scorers = tuple(
                primary_scorer
                if int(gpu_id) == int(context.physical_gpu_id)
                else primary_scorer.clone_to_device(f"cuda:{gpu_id}")
                for gpu_id in proxy_gpu_ids
            )
            batch_scorer = (
                MultiDeviceTorchBatchedProxyScorer(scorers=scorers)
                if len(scorers) > 1
                else primary_scorer
            )
            proxy_backend = "cuda_batched"
        elif str(proxy_cfg.get("device", "cpu")).lower() != "cpu":
            proxy_backend = "torch_batched_cpu"
        _require_gpu_proxy_if_needed(proxy_cfg=proxy_cfg, search_cfg=search_cfg, actual_backend=proxy_backend)
        proxy = Stage1ProxyEvaluator(
            context.search_space,
            objective=objective,
            cache=proxy_cache,
            cache_key_fn=lambda phenotype, _space: search_hash(
                phenotype,
                trace_hash=context.search_space.trace_snapshot_hash,
                proxy_version="real-fisher-sqnr-size-bops-v1",
                calibration_statistics_version=fisher_stats.statistics_version + ":" + fisher_stats.manifest_hash,
            ),
            batch_scorer=batch_scorer,
            proxy_backend=proxy_backend,
            proxy_device=proxy_device,
            proxy_batch_size=proxy_batch_size,
        )
        run_manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
        run_manifest.update(
            {
                "proxy_backend": proxy_backend,
                "proxy_device": proxy_device,
                "proxy_batch_size": proxy_batch_size,
                "proxy_gpu_ids": proxy_gpu_ids,
                "multi_gpu_proxy_worker_count": len(proxy_gpu_ids),
                "initial_candidate_count": int(search_cfg.get("initial_population_size", search_cfg.get("population_size", 0))),
                "unique_phenotype_count": 0,
                "gpu_batch_count": 0,
                "cache_miss_count": 0,
            }
        )
        _write_json(run_dir / "run_manifest.json", run_manifest)
        print(
            json.dumps(
                {
                    "event": "proxy_startup",
                    "proxy_backend": proxy_backend,
                    "proxy_device": proxy_device,
                    "proxy_batch_size": proxy_batch_size,
                    "proxy_gpu_ids": proxy_gpu_ids,
                    "multi_gpu_proxy_worker_count": len(proxy_gpu_ids),
                    "initial_candidate_count": int(search_cfg.get("initial_population_size", search_cfg.get("population_size", 0))),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        real_evaluator = LidarPyramidRealEvaluator(
            context=context,
            run_dir=run_dir,
            num_frames=int(stage2_cfg.get("num_frames", 5)),
            warmup_frames=int(stage2_cfg.get("warmup_frames", 10)),
            latency_rounds=int(stage2_cfg.get("latency_rounds", stage2_cfg.get("rounds", 1))),
            stage2_config=Stage2ObjectiveConfig(
                eta_map=float(stage2_cfg.get("eta_ap", stage2_cfg.get("eta_map", 1.0))),
                eta_latency=float(stage2_cfg.get("eta_latency", 1.0)),
                latency_metric=str(stage2_cfg.get("latency_metric", "forward_mean_ms")),
                accuracy_reference=str(stage2_cfg.get("accuracy_reference", "original_strict_fp32")),
                latency_reference=str(stage2_cfg.get("latency_reference", "original_strict_fp32")),
                tau_ap=stage2_cfg.get("tau_ap"),
                max_map_drop=stage2_cfg.get("max_map_drop"),
            ),
        )
        search_method = str(search_cfg.get("method", "ga")).strip().lower()
        baseline_cfg = dict(self.config.get("baselines", {}) or {})
        if baseline_cfg.get("build_before_search") and not stage2_only and search_method != "greedy":
            real_evaluator.evaluate_original_baselines(
                [str(value) for value in baseline_cfg.get("precisions", ["strict_fp32", "strict_fp16", "maximal_legal_int8"])]
            )
        if stage2_only:
            if candidate_config is None:
                raise RuntimeError("stage2_only_requires_candidate_config")
            paths = candidate_config if isinstance(candidate_config, list) else [candidate_config]
            results = []
            for path in paths:
                candidate = _load_candidate(path)
                phenotype = candidate if isinstance(candidate, CandidatePhenotype) else canonicalize_candidate(candidate, context.search_space)
                key = candidate_hash(phenotype, context.search_space)
                result = real_evaluator.evaluate_candidate(phenotype, output_dir=run_dir / "round_000" / "stage2" / key, candidate_hash=key)
                results.append(result)
            return {"run_dir": str(run_dir), "selected_gpu": context.physical_gpu_id, "stage2_only": True, "results": results, "result": results[0] if results else None}
        if search_method == "greedy":
            rows = self._run_greedy(
                context,
                proxy,
                run_dir,
                search_cfg,
                stage1_only=stage1_only,
            )
        elif search_method == "ga":
            rows = self._run_ga(
                context,
                proxy,
                real_evaluator,
                run_dir,
                search_cfg,
                stage1_only=stage1_only,
            )
        else:
            raise ValueError(f"unsupported_search_method:{search_method}")
        if self.resume is not None:
            manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
            _write_json(
                run_dir / "resume_cache_report.json",
                {
                    "proxy_cache_hit": int(manifest.get("cache_hit_count", 0) or 0) > 0,
                    "proxy_cache_miss_count": int(manifest.get("cache_miss_count", 0) or 0),
                    "gpu_batch_count": int(manifest.get("gpu_batch_count", 0) or 0),
                    "scalar_evaluate_call_count": int(manifest.get("scalar_evaluate_call_count", 0) or 0),
                    "physical_artifact_hit": None,
                    "onnx_hit": None,
                    "calibration_hit": None,
                    "qdq_onnx_hit": None,
                    "engine_hit": None,
                    "real_evaluation_hit": None,
                    "note": "stage1_only resume reports proxy cache; Stage-2 layered cache fields are populated during candidate evaluation.",
                },
            )
        return {"run_dir": str(run_dir), "selected_gpu": context.physical_gpu_id, "evaluated": len(rows), "best": min(rows, key=lambda row: float(row.get("F2", float("inf")))) if rows else None}

    def _full_validation_evaluator(
        self,
        context: Any,
        run_dir: Path,
    ) -> LidarPyramidRealEvaluator:
        full_cfg = dict(self.config.get("full_validation", {}) or {})
        stage2_cfg = dict(
            self.config.get("stage2")
            or self.config.get("stage2_smoke")
            or self.config.get("evaluation", {})
        )
        return LidarPyramidRealEvaluator(
            context=context,
            run_dir=run_dir / "full_validation",
            num_frames=int(full_cfg.get("num_frames", 1789)),
            warmup_frames=int(full_cfg.get("warmup_frames", 200)),
            latency_rounds=int(
                full_cfg.get(
                    "latency_rounds",
                    stage2_cfg.get("latency_rounds", stage2_cfg.get("rounds", 1)),
                )
            ),
            stage2_config=Stage2ObjectiveConfig(
                eta_map=float(stage2_cfg.get("eta_ap", stage2_cfg.get("eta_map", 1.0))),
                eta_latency=float(stage2_cfg.get("eta_latency", 1.0)),
                latency_metric=str(stage2_cfg.get("latency_metric", "forward_p50_ms")),
                accuracy_reference=str(stage2_cfg.get("accuracy_reference", "original_strict_fp32")),
                latency_reference=str(stage2_cfg.get("latency_reference", "original_strict_fp32")),
                tau_ap=stage2_cfg.get("tau_ap"),
                max_map_drop=stage2_cfg.get("max_map_drop"),
            ),
            engine_reuse_roots=[run_dir],
        )

    def _ga_stage2_evaluator_pool(
        self,
        context: Any,
        real_evaluator: LidarPyramidRealEvaluator,
        run_dir: Path,
    ) -> list[tuple[int, LidarPyramidRealEvaluator]]:
        cached = getattr(self, "_ga_stage2_worker_pool", None)
        if cached is not None:
            return list(cached)
        gpu_ids = list(
            getattr(self, "_stage2_gpu_ids", [context.physical_gpu_id])
        )
        reference = real_evaluator._stage2_reference_baseline()
        real_evaluator._reference_baseline_override = dict(reference)
        workers: list[tuple[int, LidarPyramidRealEvaluator]] = []
        worker_rows: list[dict[str, Any]] = []
        runtime = dict(getattr(self, "_runtime_config", {}) or {})
        gpu_report = query_gpus()
        for gpu_id in gpu_ids:
            if int(gpu_id) == int(context.physical_gpu_id):
                workers.append((int(gpu_id), real_evaluator))
                worker_rows.append(
                    {
                        "gpu_id": int(gpu_id),
                        "context_source": "primary_search_context",
                        "status": "ready",
                    }
                )
                continue
            try:
                bundle = load_lidar_pyramid_model(
                    checkpoint_path=context.checkpoint_path,
                    model_config_path=context.model_config,
                    heal_root=runtime.get(
                        "heal_root", "/home/lixingfeng/UniAD_examine/HEAL"
                    ),
                    device=f"cuda:{gpu_id}",
                    trace=False,
                )
                if bundle.checkpoint_hash != context.checkpoint_hash:
                    raise RuntimeError(
                        f"stage2_worker_checkpoint_hash_mismatch:{gpu_id}"
                    )
                bundle.trace_result = context.trace_result
                report_row = next(
                    (
                        row
                        for row in gpu_report
                        if int(row["index"]) == int(gpu_id)
                    ),
                    {},
                )
                worker_context = replace(
                    context,
                    model=bundle.model,
                    model_bundle=bundle,
                    trace_example_inputs=bundle.trace_example_inputs,
                    export_example_inputs=bundle.trace_example_inputs,
                    physical_gpu_id=int(gpu_id),
                    runtime_device=f"cuda:{gpu_id}",
                    gpu_selection=GPUSelection(
                        gpu_id_arg=str(gpu_id),
                        physical_gpu_id=int(gpu_id),
                        runtime_device=f"cuda:{gpu_id}",
                        excluded_gpu_ids=[],
                        gpu_report=gpu_report,
                    ),
                )
                worker = LidarPyramidRealEvaluator(
                    context=worker_context,
                    run_dir=run_dir / "stage2_workers" / f"gpu_{gpu_id}",
                    num_frames=real_evaluator.num_frames,
                    warmup_frames=real_evaluator.warmup_frames,
                    latency_rounds=real_evaluator.latency_rounds,
                    stage2_config=real_evaluator.objective_config,
                    artifact_cache=real_evaluator.artifacts,
                    real_cache=real_evaluator.real_cache,
                    reference_baseline=reference,
                )
                workers.append((int(gpu_id), worker))
                worker_rows.append(
                    {
                        "gpu_id": int(gpu_id),
                        "context_source": "checkpoint_reload_without_retrace",
                        "status": "ready",
                        "memory_free_mib_at_selection": report_row.get(
                            "memory_free_mib"
                        ),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                worker_rows.append(
                    {
                        "gpu_id": int(gpu_id),
                        "status": "rejected",
                        "reason": f"{type(exc).__name__}:{exc}",
                    }
                )
        if not workers:
            raise RuntimeError("no_stage2_gpu_worker_available")
        _write_json(
            run_dir / "stage2_workers" / "worker_pool_manifest.json",
            {
                "workers": worker_rows,
                "active_gpu_ids": [gpu_id for gpu_id, _worker in workers],
                "reference_baseline": {
                    "mAP": reference.get("mAP"),
                    real_evaluator.objective_config.latency_metric: reference.get(
                        real_evaluator.objective_config.latency_metric
                    ),
                },
                "scheduling": "one_candidate_per_gpu_sequential_queue",
            },
        )
        self._ga_stage2_worker_pool = list(workers)
        return workers

    def _evaluate_ga_stage2_selected_parallel(
        self,
        *,
        context: Any,
        real_evaluator: LidarPyramidRealEvaluator,
        run_dir: Path,
        round_dir: Path,
        selected: list[Any],
    ) -> list[dict[str, Any]]:
        workers = self._ga_stage2_evaluator_pool(
            context, real_evaluator, run_dir
        )
        result_cache: dict[str, dict[str, Any]] = getattr(
            self, "_ga_stage2_result_cache", {}
        )
        self._ga_stage2_result_cache = result_cache
        rows_by_index: dict[int, dict[str, Any]] = {}
        pending: list[tuple[int, Any, Path]] = []
        for index, item in enumerate(selected):
            record = item.record
            candidate_dir = round_dir / "stage2" / record.candidate_hash
            _write_json(candidate_dir / "genotype.json", record.genotype.to_dict())
            _write_json(
                candidate_dir / "repaired_genotype.json", record.genotype.to_dict()
            )
            cached = result_cache.get(record.candidate_hash)
            if cached is not None:
                rows_by_index[index] = {
                    "candidate_hash": record.candidate_hash,
                    "F1": record.F1,
                    **dict(cached),
                    "cross_round_deployment_cache_hit": True,
                }
                _write_json(
                    candidate_dir / "stage2_cache_hit.json",
                    {
                        "candidate_hash": record.candidate_hash,
                        "source_artifact_dir": cached.get("artifact_dir", ""),
                        "engine_rebuilt": False,
                        "evaluation_rerun": False,
                    },
                )
            else:
                pending.append((index, item, candidate_dir))
        queues: list[list[tuple[int, Any, Path]]] = [
            [] for _worker in workers
        ]
        for task_index, task in enumerate(pending):
            queues[task_index % len(workers)].append(task)

        def run_queue(
            worker_row: tuple[int, LidarPyramidRealEvaluator],
            queue: list[tuple[int, Any, Path]],
        ) -> list[tuple[int, dict[str, Any]]]:
            gpu_id, evaluator = worker_row
            torch_module = __import__("torch")
            torch_module.cuda.set_device(int(gpu_id))
            completed: list[tuple[int, dict[str, Any]]] = []
            for index, item, candidate_dir in queue:
                record = item.record
                try:
                    result = evaluator.evaluate_candidate(
                        record.phenotype,
                        output_dir=candidate_dir,
                        candidate_hash=record.candidate_hash,
                    )
                except Exception as exc:  # noqa: BLE001
                    result = {
                        "status": "stage2_worker_failed",
                        "failure_reason": f"{type(exc).__name__}:{exc}",
                        "F2": float("inf"),
                        "artifact_dir": str(candidate_dir),
                    }
                completed.append(
                    (
                        index,
                        {
                            "candidate_hash": record.candidate_hash,
                            "F1": record.F1,
                            "assigned_gpu_id": int(gpu_id),
                            **result,
                        },
                    )
                )
            return completed

        active = [
            (worker, queue)
            for worker, queue in zip(workers, queues)
            if queue
        ]
        if active:
            with ThreadPoolExecutor(max_workers=len(active)) as executor:
                futures = [
                    executor.submit(run_queue, worker, queue)
                    for worker, queue in active
                ]
                for future in futures:
                    for index, row in future.result():
                        rows_by_index[index] = row
                        result_cache[str(row["candidate_hash"])] = dict(row)
        rows = [rows_by_index[index] for index in range(len(selected))]
        _write_json(
            round_dir / "stage2_parallel_schedule.json",
            {
                "worker_gpu_ids": [gpu_id for gpu_id, _worker in workers],
                "candidate_assignments": [
                    {
                        "candidate_hash": row.get("candidate_hash"),
                        "assigned_gpu_id": row.get("assigned_gpu_id"),
                        "cross_round_deployment_cache_hit": row.get(
                            "cross_round_deployment_cache_hit", False
                        ),
                    }
                    for row in rows
                ],
                "per_gpu_execution": "sequential",
                "cross_gpu_execution": "parallel",
            },
        )
        return rows

    def _run_greedy(
        self,
        context: Any,
        proxy: Stage1ProxyEvaluator,
        run_dir: Path,
        search_cfg: dict[str, Any],
        *,
        stage1_only: bool,
    ) -> list[dict[str, Any]]:
        proxy_cfg = dict(self.config.get("proxy", self.config.get("proxy_objective", {})))
        targets = tuple(
            float(value)
            for value in (
                search_cfg.get("bops_targets")
                or proxy_cfg.get("bops_targets")
                or [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
            )
        )
        if getattr(proxy, "objective", None) is not None:
            proxy.objective.config = replace(
                proxy.objective.config,
                bops_threshold=None,
                bops_constraint_mode="hard_band_feasibility",
            )
            if getattr(proxy, "batch_scorer", None) is not None:
                proxy.batch_scorer.config = proxy.objective.config
        proxy.cache_key_fn = lambda phenotype, _space: search_hash(
            phenotype,
            trace_hash=context.search_space.trace_snapshot_hash,
            proxy_version="domain-width-joint-weight-taylor-greedy-v1",
            calibration_statistics_version=str(
                getattr(getattr(proxy, "objective", None), "statistics_version", "")
            ),
        )

        def evaluate_batch(
            candidates: list[CandidateGenotype],
            step: int,
        ) -> list[dict[str, Any]]:
            result = proxy.evaluate_batch(
                candidates,
                generation=int(step),
                outer_round=-1,
            )
            return list(result.metrics)

        greedy = GreedyBudgetSearch(
            context.search_space,
            config=GreedySearchConfig(
                bops_targets=targets,
                minimum_bops_reduction=float(
                    search_cfg.get("minimum_bops_reduction", 1.0e-12)
                ),
                maximum_steps=int(search_cfg.get("maximum_steps", 10000)),
                parameter_retention_tiebreak=bool(
                    search_cfg.get("parameter_retention_tiebreak", True)
                ),
                bops_tolerance_abs=float(
                    proxy_cfg.get("bops_tolerance_abs", 0.005)
                ),
            ),
        )
        result = greedy.run(evaluate_batch)
        greedy_dir = run_dir / "greedy"
        _write_json(greedy_dir / "greedy_path.json", result.to_dict())
        _write_json(
            greedy_dir / "budget_summary.json",
            {
                "targets": list(sorted(targets)),
                "reached": list(sorted(result.budget_candidates)),
                "unreachable": list(result.unreachable_targets),
                "termination_reason": result.termination_reason,
                "stage2_policy": "only_unique_final_candidate_per_budget_full_validation",
            },
        )
        budget_rows: list[dict[str, Any]] = []
        by_candidate_hash: dict[str, dict[str, Any]] = {}
        for target, genotype in sorted(result.budget_candidates.items()):
            phenotype = canonicalize_candidate(genotype, context.search_space)
            identity = candidate_hash(phenotype, context.search_space)
            by_candidate_hash.setdefault(
                identity,
                {"phenotype": phenotype, "budgets": [], "metrics": result.budget_metrics[target]},
            )["budgets"].append(float(target))
            _write_json(
                greedy_dir / "budgets" / f"bops_{target:.4f}" / "candidate.json",
                {
                    "target": target,
                    "candidate_hash": identity,
                    "genotype": genotype.to_dict(),
                    "phenotype": phenotype.to_dict(),
                    "proxy_metrics": result.budget_metrics[target],
                },
            )
        if stage1_only:
            return [
                {
                    "candidate_hash": identity,
                    "budgets": row["budgets"],
                    **dict(row["metrics"]),
                    "status": "stage1_only",
                }
                for identity, row in sorted(by_candidate_hash.items())
            ]
        full_evaluator = self._full_validation_evaluator(context, run_dir)
        for identity, row in sorted(by_candidate_hash.items()):
            candidate_dir = greedy_dir / "stage2_full" / identity
            evaluated = full_evaluator.evaluate_candidate(
                row["phenotype"],
                output_dir=candidate_dir,
                candidate_hash=identity,
            )
            metrics = dict(row["metrics"])
            r_bops = float(metrics.get("R_bops_vs_fp32", 0.0) or 0.0)
            r_size = float(metrics.get("R_size_vs_fp32", 0.0) or 0.0)
            parameter_base = float(metrics.get("parameter_count_base", 0.0) or 0.0)
            parameter_after = float(metrics.get("parameter_count_after", 0.0) or 0.0)
            latency_ratio = float(evaluated.get("R_latency_real", 0.0) or 0.0)
            budget_rows.append(
                {
                    "candidate_hash": identity,
                    "budgets": sorted(row["budgets"]),
                    "proxy_metrics": metrics,
                    "BOPS_compression_x": 1.0 / r_bops if r_bops > 0.0 else None,
                    "mixed_weight_compression_x": 1.0 / r_size if r_size > 0.0 else None,
                    "parameter_pruning_rate": (
                        1.0 - parameter_after / parameter_base
                        if parameter_base > 0.0
                        else None
                    ),
                    "parameter_compression_x": (
                        parameter_base / parameter_after
                        if parameter_base > 0.0 and parameter_after > 0.0
                        else None
                    ),
                    "measured_speedup_vs_FP32": (
                        1.0 / latency_ratio if latency_ratio > 0.0 else None
                    ),
                    **evaluated,
                }
            )
        successful = [
            row for row in budget_rows if str(row.get("status", "")) == "ok"
        ]
        best = (
            min(successful, key=lambda row: float(row.get("F2", float("inf"))))
            if successful
            else None
        )
        _write_json(
            greedy_dir / "full_validation_results.json",
            {
                "candidates": budget_rows,
                "best": best,
                "unique_candidate_count": len(by_candidate_hash),
                "engine_build_policy": "one_unique_candidate_per_budget_identity",
            },
        )
        if best is not None:
            _write_json(greedy_dir / "best_candidate.json", best)
        return budget_rows

    def _objective(self, context: Any, unit_slices: dict[str, Any], fisher_stats: Any, normalization: Any | None, runtime_shapes: Any | None = None) -> ProxyObjective:
        proxy_cfg = dict(self.config.get("proxy", self.config.get("proxy_objective", {})))
        objective_mode = str(
            proxy_cfg.get("objective_mode", "legacy_fisher_sqnr_size_bops")
        )
        return ProxyObjective(
            fisher=FisherTaylorProxy(context.model, statistics=fisher_stats, unit_to_parameter_names=unit_slices),
            sqnr=SQNRProxy(context.model, unit_to_parameter_slices=unit_slices),
            size=SizeProxy(context.model, unit_to_parameter_slices=unit_slices),
            bops=BOPSProxy(context.model, unit_to_parameter_slices=unit_slices, runtime_shapes=runtime_shapes),
            joint_weight_taylor=(
                JointWeightTaylorProxy(
                    context.model,
                    statistics=fisher_stats,
                    unit_to_parameter_slices=unit_slices,
                )
                if objective_mode == "joint_weight_taylor_hard_bops"
                else None
            ),
            normalization=normalization,
            config=ProxyObjectiveConfig(
                objective_mode=objective_mode,
                alpha_fisher=float(proxy_cfg.get("alpha_prune", proxy_cfg.get("alpha_fisher", 1.0))),
                beta_sqnr=float(proxy_cfg.get("beta_sqnr", 1.0)),
                gamma_size=float(proxy_cfg.get("gamma_size", 1.0)),
                delta_bops=float(proxy_cfg.get("delta_bops_penalty", proxy_cfg.get("delta_bops", 1.0))),
                size_threshold=proxy_cfg.get("size_threshold"),
                bops_threshold=proxy_cfg.get("bops_threshold", None),
                bops_constraint_mode=str(proxy_cfg.get("bops_constraint_mode", "weighted_penalty")),
                bops_tolerance_abs=float(proxy_cfg.get("bops_tolerance_abs", 0.0)),
                bops_penalty_formula=str(
                    dict(proxy_cfg.get("bops_soft_constraint", {}) or {}).get(
                        "formula",
                        proxy_cfg.get("bops_penalty_formula", "absolute_excess_squared"),
                    )
                ),
                lambda_bops=float(proxy_cfg.get("lambda_bops", 1.0)),
                parameter_retention_tiebreak_epsilon=float(
                    proxy_cfg.get("parameter_retention_tiebreak_epsilon", 0.0)
                ),
            ),
        )

    @staticmethod
    def _action_slices(context: Any, raw_unit_slices: dict[str, Any]) -> dict[str, Any]:
        catalog = getattr(context, "pruning_action_catalog", None)
        if catalog is None:
            return raw_unit_slices
        params = dict(context.model.named_parameters())
        result: dict[str, list[Any]] = {}
        for action in catalog.actions:
            rows = []
            for source_id in action.source_atomic_unit_ids:
                rows.extend(raw_unit_slices.get(str(source_id), []))
            result[action.action_id] = sanitize_action_parameter_slices(action, rows, params)
        return result

    def _write_local_domains(self, context: Any, unit_slices: dict[str, Any], run_dir: Path) -> None:
        units = list(getattr(context, "atomic_prune_units", []) or [])
        domains = list(context.search_space.pruning_domains) or build_local_pruning_domains(units)
        payload = []
        for domain in domains:
            row = domain.to_dict()
            row["keep_mask_coordinate_system"] = "original_channel_index"
            row["parameter_slice_mapping"] = {
                unit_id: [
                    {
                        "parameter_name": item.parameter_name,
                        "module_path": item.module_path,
                        "axis": item.axis,
                        "indices": list(item.indices),
                        "operation": item.operation,
                    }
                    for item in unit_slices.get(unit_id, [])
                ]
                for unit_id in domain.ordered_unit_ids
            }
            row["fixed_pruning_taylor_score"] = dict(domain.unit_scores)
            row["ranking_is_precision_gene_independent"] = True
            row["post_search_alignment_repair_required"] = False
            row["protected_units"] = [
                str(getattr(unit, "stable_id", ""))
                for unit in units
                if bool(getattr(unit, "protected", False)) and str(getattr(unit, "stable_id", "")) in set(domain.ordered_unit_ids)
            ]
            payload.append(row)
        _write_json(run_dir / "local_pruning_domains.json", payload)

    def _repair_raw_keep_mask(self, context: Any, genotype: CandidateGenotype) -> tuple[CandidateGenotype | None, dict[str, Any]]:
        pruning_cfg = dict(self.config.get("pruning", {}) or {})
        gene_type = str(pruning_cfg.get("gene_type", pruning_cfg.get("search_variable", "")))
        if gene_type in {"legal_domain_width", "domain_width", "coupled_domain_width"}:
            # canonicalize_candidate expands an already legal retained-width
            # choice to the immutable atomic mask. No 1->0 alignment repair or
            # ranking pass is allowed after search.
            return genotype, {
                "status": "ok",
                "repair_mode": "domain_width_legal_by_construction",
                "alignment_edits": 0,
                "reranking_applied": False,
            }
        if gene_type != "coupled_channel_keep_mask":
            return genotype, {"status": "ok", "repair_mode": "legal_action_passthrough"}
        mask = {unit_id: int(genotype.pruning_genes.get(unit_id, 1)) for unit_id in context.search_space.pruning_unit_ids}
        units_by_id = {str(getattr(unit, "stable_id", "")): unit for unit in getattr(context, "atomic_prune_units", []) or []}
        grouped_by_domain: dict[tuple[str, str, str], list[Any]] = {}
        for unit_id in context.search_space.pruning_unit_ids:
            unit = units_by_id.get(unit_id)
            if unit is None:
                continue
            key = (
                str(getattr(unit, "root_module_path", "")),
                str(getattr(unit, "root_axis", "")),
                str(getattr(unit, "scope_id", "")),
            )
            grouped_by_domain.setdefault(key, []).append(unit)
        dense_repairs = 0
        grouped_repairs = 0
        failures: dict[str, int] = {}
        group_keep_maps: dict[str, dict[int, list[int]]] = {}
        group_prune_maps: dict[str, dict[int, list[int]]] = {}
        group_keep_maps_by_scope: dict[str, dict[int, list[int]]] = {}
        group_prune_maps_by_scope: dict[str, dict[int, list[int]]] = {}
        dense_cfg = dict(pruning_cfg.get("dense", {}) or {})
        grouped_cfg = dict(pruning_cfg.get("grouped_conv", {}) or {})
        for key, units in sorted(grouped_by_domain.items()):
            constraints = dict(getattr(units[0], "constraints", {}) or {})
            if constraints.get("grouped_conv") and not constraints.get("depthwise"):
                groups = int(constraints.get("groups") or 0)
                width = int(constraints.get("channels_per_group") or constraints.get("channels_per_group_before") or 0)
                if groups <= 0 or width <= 0:
                    failures["invalid_grouped_metadata"] = failures.get("invalid_grouped_metadata", 0) + 1
                    continue
                by_group: dict[int, list[str]] = {group: [] for group in range(groups)}
                local_indices: dict[int, dict[str, int]] = {group: {} for group in range(groups)}
                ordered: dict[int, list[str]] = {group: [] for group in range(groups)}
                for unit in units:
                    unit_id = str(getattr(unit, "stable_id", ""))
                    root_indices = list(getattr(unit, "root_indices", []) or [])
                    if not root_indices:
                        continue
                    absolute = int(root_indices[0])
                    group, local = divmod(absolute, width)
                    if group not in by_group:
                        continue
                    by_group[group].append(unit_id)
                    local_indices[group][unit_id] = local
                    ordered[group].append(unit_id)
                for group in ordered:
                    ordered[group].sort(key=lambda item: (float(getattr(units_by_id[item], "normalized_score", 0.0)), local_indices[group][item], item))
                    by_group[group].sort(key=lambda item: (local_indices[group][item], item))
                result = grouped_equal_count_floor_repair(
                    mask,
                    GroupedDomainSpec(
                        groups={group: tuple(values) for group, values in by_group.items()},
                        ordered_low_to_high={group: tuple(values) for group, values in ordered.items()},
                        local_indices=local_indices,
                        allowed_channels_per_group=tuple(
                            int(value)
                            for value in grouped_cfg.get("allowed_channels_per_group", [4, 8, 16, 32, 64, 128, 256, 512])
                        ),
                    ),
                    RepairPolicy(),
                )
                if result.status != "ok":
                    failures[result.failure_reason or "grouped_repair_failed"] = failures.get(result.failure_reason or "grouped_repair_failed", 0) + 1
                    return None, {"status": "failed", "failure_reason": result.failure_reason, "failure_reasons": failures}
                grouped_repairs += int(bool(result.removed_unit_ids))
                mask.update(result.repaired_mask)
                domain_id = "::".join(key)
                group_keep_maps[domain_id] = result.group_keep_map
                group_prune_maps[domain_id] = result.group_prune_map
                scope_id = key[2]
                group_keep_maps_by_scope[scope_id] = result.group_keep_map
                group_prune_maps_by_scope[scope_id] = result.group_prune_map
            else:
                ordered_units = tuple(
                    str(getattr(unit, "stable_id", ""))
                    for unit in sorted(units, key=lambda row: (float(getattr(row, "normalized_score", 0.0)), min(getattr(row, "root_indices", [0]) or [0]), str(getattr(row, "stable_id", ""))))
                )
                result = dense_floor_repair(
                    {unit_id: mask.get(unit_id, 1) for unit_id in ordered_units},
                    ordered_low_to_high=ordered_units,
                    alignment=int(dense_cfg.get("alignment", pruning_cfg.get("dense_channel_alignment", 4))),
                    minimum_width=max(1, int(len(ordered_units) * float(pruning_cfg.get("minimum_retained_ratio", 0.10)))),
                )
                if result.status != "ok":
                    failures[result.failure_reason or "dense_repair_failed"] = failures.get(result.failure_reason or "dense_repair_failed", 0) + 1
                    return None, {"status": "failed", "failure_reason": result.failure_reason, "failure_reasons": failures}
                dense_repairs += int(bool(result.removed_unit_ids))
                mask.update(result.repaired_mask)
        repaired = CandidateGenotype(
            pruning_genes=mask,
            precision_genes=dict(genotype.precision_genes),
            meta={
                **dict(genotype.meta),
                "repair_mode": "mask_preserving_monotonic_floor",
                "keep_value": 1,
                "prune_value": 0,
                "dense_repair_count": dense_repairs,
                "grouped_repair_count": grouped_repairs,
                "group_keep_map": group_keep_maps,
                "group_prune_map": group_prune_maps,
                "group_keep_map_by_scope": group_keep_maps_by_scope,
                "group_prune_map_by_scope": group_prune_maps_by_scope,
            },
        )
        return repaired, {
            "status": "ok",
            "dense_repair_count": dense_repairs,
            "grouped_repair_count": grouped_repairs,
            "failure_reasons": failures,
        }

    def _build_normalization(self, context: Any, objective: ProxyObjective, run_dir: Path) -> Any:
        proxy_cfg = dict(self.config.get("proxy", self.config.get("proxy_objective", {})))
        strategy = str(proxy_cfg.get("term_normalization", "none")).strip().lower()
        if strategy in {"", "none", "identity"}:
            stats = NormalizationStats(version="none-v1")
            _write_json(
                run_dir / "archives" / "proxy_normalization.json",
                {**stats.to_dict(), "strategy": "none"},
            )
            return stats
        if strategy != "fixed_median":
            raise ValueError(f"unsupported_proxy_term_normalization:{strategy}")

        from ..ga.immigrants import random_immigrant

        rng = random.Random(int(self.config.get("search", {}).get("seed", 42)) + 999)
        rows = []
        for _ in range(8):
            phenotype = canonicalize_candidate(random_immigrant(context.search_space, rng), context.search_space)
            metrics = objective.evaluate(phenotype)
            rows.append({"L_fisher": float(metrics["L_fisher"]), "L_sqnr": float(metrics["L_sqnr"])})
        stats = build_normalization_stats(rows, ["L_fisher", "L_sqnr"])
        _write_json(
            run_dir / "archives" / "proxy_normalization.json",
            {**stats.to_dict(), "strategy": "fixed_median"},
        )
        return stats

    def _run_ga(self, context: Any, proxy: Stage1ProxyEvaluator, real_evaluator: LidarPyramidRealEvaluator, run_dir: Path, search_cfg: dict[str, Any], *, stage1_only: bool) -> list[dict[str, Any]]:
        evaluated_rows: list[dict[str, Any]] = []
        previous_elite: list[CandidateGenotype] = []
        previous_best: CandidateGenotype | None = None
        outer_rounds = int(search_cfg.get("outer_rounds", 1))
        proxy_backend = str(getattr(proxy, "proxy_backend", "scalar_cpu"))
        proxy_cfg = dict(self.config.get("proxy", self.config.get("proxy_objective", {})))
        configured_bops_targets = sorted(
            {
                float(value)
                for value in (
                    search_cfg.get("bops_targets")
                    or proxy_cfg.get("bops_targets")
                    or []
                )
            },
            reverse=True,
        )
        if configured_bops_targets:
            outer_rounds = len(configured_bops_targets)
        soft_schedule = dict(proxy_cfg.get("bops_soft_constraint", {}) or {})
        if not soft_schedule:
            soft_schedule = dict(proxy_cfg.get("bops_target_schedule", {}) or {})
        budget_seed_candidates: dict[float, list[CandidateGenotype]] = {}
        greedy_seed_path = run_dir / "ga_seed_greedy_path.json"
        use_greedy_seeds = bool(search_cfg.get("greedy_frontier_warm_start", True))
        if use_greedy_seeds and configured_bops_targets:
            if self.resume is not None and greedy_seed_path.is_file():
                seed_payload = json.loads(greedy_seed_path.read_text(encoding="utf-8"))
                nearest_payload = dict(
                    seed_payload.get("nearest_budget_candidates", {}) or {}
                )
                exact_payload = dict(seed_payload.get("budget_candidates", {}) or {})
                for target in configured_bops_targets:
                    rows = []
                    for source in (exact_payload, nearest_payload):
                        payload = source.get(f"{target:.6f}")
                        if payload:
                            rows.append(CandidateGenotype.from_dict(payload))
                    budget_seed_candidates[target] = rows
            else:
                proxy_objective = getattr(proxy, "objective", None)
                if proxy_objective is not None:
                    proxy.objective.config = replace(
                        proxy.objective.config,
                        bops_threshold=None,
                    )
                    if getattr(proxy, "batch_scorer", None) is not None:
                        proxy.batch_scorer.config = proxy.objective.config

                def evaluate_greedy_seed_batch(
                    candidates: list[CandidateGenotype], step: int
                ) -> list[dict[str, Any]]:
                    batch = proxy.evaluate_batch(
                        candidates,
                        generation=int(step),
                        outer_round=-2,
                    )
                    return list(batch.metrics)

                greedy_seed_search = GreedyBudgetSearch(
                    context.search_space,
                    config=GreedySearchConfig(
                        bops_targets=tuple(configured_bops_targets),
                        minimum_bops_reduction=float(
                            search_cfg.get("minimum_bops_reduction", 1.0e-12)
                        ),
                        maximum_steps=int(
                            search_cfg.get("greedy_seed_maximum_steps", 10000)
                        ),
                        parameter_retention_tiebreak=True,
                        bops_tolerance_abs=float(
                            proxy_cfg.get("bops_tolerance_abs", 0.005)
                        ),
                    ),
                )
                greedy_seed_result = greedy_seed_search.run(
                    evaluate_greedy_seed_batch
                )
                _write_json(greedy_seed_path, greedy_seed_result.to_dict())
                for target in configured_bops_targets:
                    rows = []
                    exact = greedy_seed_result.budget_candidates.get(target)
                    nearest = greedy_seed_result.nearest_budget_candidates.get(target)
                    if exact is not None:
                        rows.append(exact)
                    if nearest is not None and nearest not in rows:
                        rows.append(nearest)
                    budget_seed_candidates[target] = rows
        global_seen_raw_hashes = self._load_seen_raw_hashes(run_dir)
        for round_index in range(outer_rounds):
            round_dir = run_dir / f"round_{round_index:03d}"
            round_dir.mkdir(parents=True, exist_ok=True)
            topk_stage2 = int(search_cfg.get("topk_stage2", search_cfg.get("topk_real", 1)))
            round_bops_target = (
                configured_bops_targets[round_index]
                if configured_bops_targets
                else bops_target_for_outer_round(round_index, outer_rounds, soft_schedule)
                if soft_schedule
                else None
            )
            proxy_objective = getattr(proxy, "objective", None)
            if round_bops_target is not None and proxy_objective is not None:
                proxy.objective.config = replace(
                    proxy.objective.config,
                    bops_threshold=float(round_bops_target),
                    bops_constraint_mode=str(proxy_cfg.get("bops_constraint_mode", "weighted_penalty")),
                    bops_tolerance_abs=float(proxy_cfg.get("bops_tolerance_abs", 0.0)),
                )
                if getattr(proxy, "batch_scorer", None) is not None:
                    proxy.batch_scorer.config = proxy.objective.config
            objective_config = getattr(proxy_objective, "config", ProxyObjectiveConfig())
            normalization = getattr(proxy_objective, "normalization", None)
            joint_mode = objective_config.objective_mode == "joint_weight_taylor_hard_bops"
            objective_manifest = {
                "objective_mode": objective_config.objective_mode,
                "alpha": objective_config.alpha_fisher,
                "beta": objective_config.beta_sqnr,
                "gamma": objective_config.gamma_size,
                "delta": objective_config.delta_bops,
                "normalization_definitions": normalization.to_dict() if normalization is not None else {},
                "FP32_reference_hashes": {
                    "checkpoint_hash": getattr(context, "checkpoint_hash", ""),
                    "trace_hash": context.search_space.trace_snapshot_hash,
                },
                "T_BOPS": round_bops_target,
                "bops_penalty_formula": objective_config.bops_penalty_formula,
                "bops_constraint_mode": objective_config.bops_constraint_mode,
                "bops_tolerance_abs": objective_config.bops_tolerance_abs,
                "objective": (
                    "min normalized_joint_weight_taylor subject_to abs(R_BOPS_vs_original_FP32-target)<=tolerance"
                    if joint_mode
                    else "alpha*R_Fisher + beta*L_SQNR + gamma*R_Size_vs_FP32 + delta*P_BOPS"
                ),
                "joint_weight_delta": (
                    "pruned:-w; retained:Q_precision(w)-w"
                    if joint_mode
                    else "not_applicable"
                ),
                "activation_taylor_included": False if joint_mode else None,
                "parameter_retention_role": (
                    "report_only"
                    if joint_mode and objective_config.parameter_retention_tiebreak_epsilon == 0.0
                    else "epsilon_tiebreak"
                    if joint_mode
                    else "legacy_weighted_term"
                ),
                "ga_selection": {
                    "primary": "L_joint_weight_taylor",
                    "secondary_within_taylor_epsilon": "R_parameter_retention",
                    "taylor_relative_epsilon": float(
                        search_cfg.get("taylor_relative_epsilon", 0.05)
                    ),
                    "taylor_absolute_epsilon": float(
                        search_cfg.get("taylor_absolute_epsilon", 1.0e-8)
                    ),
                },
            }
            objective_path = round_dir / "stage1_objective_config.json"
            round_state_path = round_dir / "round_state.json"
            round_state = (
                json.loads(round_state_path.read_text(encoding="utf-8"))
                if round_state_path.is_file()
                else {}
            )
            resume_stage1_complete = bool(
                self.resume is not None
                and str(round_state.get("phase", ""))
                in {"stage1_complete", "stage2_running", "round_complete"}
                and (round_dir / "stage1_topk.json").is_file()
                and (round_dir / "repaired_top5_manifest.json").is_file()
            )
            if resume_stage1_complete:
                if not objective_path.is_file():
                    raise RuntimeError(
                        f"resume_stage1_objective_missing:{objective_path}"
                    )
                existing_objective = json.loads(
                    objective_path.read_text(encoding="utf-8")
                )
                if canonical_json_hash(existing_objective) != canonical_json_hash(
                    objective_manifest
                ):
                    raise RuntimeError(
                        f"resume_stage1_contract_mismatch:{round_dir}"
                    )
                resumed_selected = self._load_round_stage1_selections(
                    round_dir, context
                )
                if len(resumed_selected) != topk_stage2:
                    raise RuntimeError(
                        "resume_stage1_topk_incomplete:"
                        f"{len(resumed_selected)}!={topk_stage2}"
                    )
                _write_json(
                    round_dir / "round_state.json",
                    {
                        "phase": "stage1_complete",
                        "round_index": round_index,
                        "stage1_reused": True,
                        "objective_hash": canonical_json_hash(objective_manifest),
                    },
                )
                if not stage1_only:
                    _write_json(
                        round_dir / "round_state.json",
                        {
                            "phase": "stage2_running",
                            "round_index": round_index,
                            "stage1_reused": True,
                            "objective_hash": canonical_json_hash(
                                objective_manifest
                            ),
                        },
                    )
                    resumed_rows = self._evaluate_ga_stage2_selected_parallel(
                        context=context,
                        real_evaluator=real_evaluator,
                        run_dir=run_dir,
                        round_dir=round_dir,
                        selected=resumed_selected,
                    )
                    evaluated_rows.extend(resumed_rows)
                    write_round_stage2_results(run_dir, round_index=round_index)
                    _write_json(
                        round_dir / "round_state.json",
                        {
                            "phase": "round_complete",
                            "round_index": round_index,
                            "stage1_reused": True,
                            "evaluated": len(resumed_rows),
                            "objective_hash": canonical_json_hash(
                                objective_manifest
                            ),
                        },
                    )
                continue
            _write_json(objective_path, objective_manifest)
            objective_hash = canonical_json_hash(objective_manifest)
            proxy_semantics_manifest = dict(objective_manifest)
            if joint_mode:
                # The hard BOPS target changes feasibility/ranking, not the
                # expensive per-candidate joint Taylor/BOPS measurements.
                # Cache those measurements once across all budget rounds and
                # reapply the target in annotate_metrics below.
                proxy_semantics_manifest["T_BOPS"] = "applied_after_cached_proxy"
            proxy_semantics_hash = canonical_json_hash(proxy_semantics_manifest)
            statistics_source = (
                getattr(proxy_objective, "joint_weight_taylor", None)
                if joint_mode
                else getattr(proxy_objective, "fisher", None)
            )
            statistics = getattr(statistics_source, "statistics", None)
            statistics_identity = ":".join(
                [
                    str(getattr(statistics, "statistics_version", "")),
                    str(getattr(statistics, "manifest_hash", "")),
                ]
            )
            proxy.cache_key_fn = lambda phenotype, _space, _proxy_semantics_hash=proxy_semantics_hash, _statistics_identity=statistics_identity: search_hash(
                phenotype,
                trace_hash=context.search_space.trace_snapshot_hash,
                proxy_version=f"domain-width-joint-weight-taylor-v1:{_proxy_semantics_hash}",
                calibration_statistics_version=_statistics_identity,
            )
            ga = GeneticSearchEngine(
                context.search_space,
                GAConfig(
                    population_size=int(search_cfg.get("population_size", 8)),
                    initial_population_size=int(search_cfg.get("initial_population_size", search_cfg.get("population_size", 8))),
                    offspring_size=int(search_cfg.get("offspring_size", search_cfg.get("population_size", 8))),
                    num_generations=int(search_cfg.get("generations_per_round", 5)),
                    elite_ratio=float(search_cfg.get("elite_ratio", 0.10)),
                    crossover_rate=float(search_cfg.get("crossover_rate", 0.80)),
                    prune_mutation_rate=float(search_cfg.get("prune_mutation_rate", 0.03)),
                    precision_mutation_rate=float(search_cfg.get("precision_group_mutation_rate", search_cfg.get("precision_mutation_rate", 0.08))),
                    immigrant_ratio=float(search_cfg.get("immigrant_ratio", 0.08)),
                    stagnation_generations=int(search_cfg.get("stagnation_generations", 3)),
                    stagnation_immigrant_ratio=float(search_cfg.get("stagnation_immigrant_ratio", 0.25)),
                    preserve_evaluated_elites=bool(
                        search_cfg.get(
                            "preserve_evaluated_elites",
                            bool(context.search_space.pruning_domains),
                        )
                    ),
                    mutation_action_min=int(search_cfg.get("mutation_action_min", 1)),
                    mutation_action_max=int(search_cfg.get("mutation_action_max", 2)),
                    early_stop_patience=int(search_cfg.get("early_stop_patience", 0)),
                    minimum_generations=int(search_cfg.get("minimum_generations", 1)),
                    constraint_first_ranking=str(
                        proxy_cfg.get("bops_constraint_mode", "")
                    )
                    == "hard_band_feasibility",
                    taylor_relative_epsilon=float(
                        search_cfg.get("taylor_relative_epsilon", 0.05)
                    ),
                    taylor_absolute_epsilon=float(
                        search_cfg.get("taylor_absolute_epsilon", 1.0e-8)
                    ),
                    seeded_initial_population_ratio=float(
                        search_cfg.get("seeded_initial_population_ratio", 0.90)
                    ),
                    random_seed=int(search_cfg.get("seed", 42)) + round_index,
                ),
            )

            def annotate_metrics(genotype: CandidateGenotype, metrics: dict[str, Any], generation: int) -> dict[str, Any]:
                grouped_action_ids = {
                    action.action_id
                    for action in getattr(context.pruning_action_catalog, "actions", [])
                    if getattr(action, "kind", "") == "grouped_bundle"
                }
                metrics["grouped_action_count"] = sum(
                    1
                    for action_id, keep in genotype.pruning_genes.items()
                    if action_id in grouped_action_ids and int(keep) == 0
                )
                target = round_bops_target
                if target is None and proxy_cfg.get("bops_target_schedule"):
                    target = bops_target_for_generation(
                        generation,
                        int(search_cfg.get("generations_per_round", 5)),
                        proxy_cfg.get("bops_target_schedule"),
                    )
                if target is not None:
                    bops_value = float(metrics.get("R_bops_vs_fp32", metrics.get("R_bops", float("inf"))))
                    violation, p_bops = bops_soft_penalty(
                        bops_value,
                        float(target),
                        formula=getattr(getattr(proxy, "objective", None), "config", ProxyObjectiveConfig()).bops_penalty_formula,
                        constraint_mode=str(
                            proxy_cfg.get("bops_constraint_mode", "weighted_penalty")
                        ),
                        tolerance_abs=float(
                            proxy_cfg.get("bops_tolerance_abs", 0.0)
                        ),
                    )
                    metrics["BOPS_target"] = float(target)
                    metrics["bops_violation"] = float(violation)
                    metrics["bops_abs_delta"] = abs(bops_value - float(target))
                    metrics["bops_tolerance_abs"] = float(
                        proxy_cfg.get("bops_tolerance_abs", 0.0)
                    )
                    metrics["P_bops"] = float(p_bops)
                    metrics["bops_feasible"] = violation <= 0.0
                    if str(proxy_cfg.get("bops_constraint_mode", "weighted_penalty")) in {
                        "feasibility_first",
                        "hard_feasibility",
                    }:
                        raw = float(
                            metrics.get("proxy_score_raw", metrics.get("F1", 0.0))
                        )
                        metrics["F1"] = (
                            1.0e6 + violation * 1.0e3 + raw
                            if violation > 0
                            else raw
                        )
                return metrics

            def evaluate_genotype(genotype: CandidateGenotype, generation: int) -> dict[str, Any]:
                metrics = proxy.evaluate(genotype, generation=generation, outer_round=round_index)
                return annotate_metrics(genotype, metrics, generation)

            def evaluate_genotypes_batch(genotypes: list[CandidateGenotype], generation: int) -> Any:
                batch = proxy.evaluate_batch(genotypes, generation=generation, outer_round=round_index)
                batch.metrics = [annotate_metrics(genotype, metrics, generation) for genotype, metrics in zip(genotypes, batch.metrics)]
                return batch

            scored = ga.run(
                evaluate_genotype if proxy_backend == "scalar_cpu" else None,
                batch_evaluator=evaluate_genotypes_batch if proxy_backend != "scalar_cpu" else None,
                previous_elite=(
                    []
                    if bool(search_cfg.get("independent_budget_rounds", True))
                    else previous_elite
                ),
                previous_best=(
                    None
                    if bool(search_cfg.get("independent_budget_rounds", True))
                    else previous_best
                ),
                seed_candidates=(
                    budget_seed_candidates.get(float(round_bops_target), [])
                    if round_bops_target is not None
                    else []
                ),
                seen_candidate_keys=global_seen_raw_hashes,
                candidate_key_fn=lambda genotype: self._raw_genotype_hash(genotype, context),
            )
            global_seen_raw_hashes.update(self._raw_genotype_hash(genotype, context) for genotype, _score, _metrics in scored)
            _append_jsonl(
                run_dir / "seen_raw_genotypes.jsonl",
                [
                    {
                        "round": round_index,
                        "generation": int(metrics.get("generation", -1)),
                        "raw_genotype_hash": self._raw_genotype_hash(genotype, context),
                        "F1": score,
                    }
                    for genotype, score, metrics in scored
                ],
            )
            manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
            manifest.update(
                {
                    "cache_miss_count": proxy.cache_miss_count,
                    "cache_hit_count": proxy.cache_hit_count,
                    "gpu_batch_count": proxy.gpu_batch_count,
                    "unique_phenotype_count": getattr(proxy, "unique_phenotype_count", 0),
                    "current_unique_phenotype_count": getattr(proxy, "current_unique_phenotype_count", 0),
                    "scalar_evaluate_call_count": proxy.scalar_evaluate_call_count,
                    "batch_evaluate_call_count": proxy.batch_evaluate_call_count,
                    "cuda_event_elapsed_ms": getattr(proxy, "last_batch_stats", {}).get("cuda_event_elapsed_ms", 0.0),
                    "gpu_peak_memory_bytes": getattr(proxy, "last_batch_stats", {}).get("gpu_peak_memory_bytes", 0),
                    "candidates_per_second": getattr(proxy, "last_batch_stats", {}).get("candidates_per_second", 0.0),
                }
            )
            _write_json(run_dir / "run_manifest.json", manifest)
            records = self._records_from_scored(scored, context)
            for generation in range(int(search_cfg.get("generations_per_round", 5))):
                self._write_generation(round_dir / f"generation_{generation:03d}.csv", [row for row in scored if int(row[2].get("generation", -1)) == generation], context)
            self._write_stage1(round_dir / "stage1_scores.csv", records)
            use_repaired_topk = str(self.config.get("pruning", {}).get("gene_type", self.config.get("pruning", {}).get("search_variable", ""))) == "coupled_channel_keep_mask" or "topk_stage2" in search_cfg
            if use_repaired_topk:
                def repair_candidate(genotype: CandidateGenotype) -> tuple[CandidateGenotype | None, dict[str, Any]]:
                    return self._repair_raw_keep_mask(context, genotype)

                def rescore_one(phenotype: CandidatePhenotype) -> dict[str, Any]:
                    batch = proxy.evaluate_batch([phenotype], generation=int(search_cfg.get("generations_per_round", 5)), outer_round=round_index)
                    return annotate_metrics(CandidateGenotype({}, {}), batch.metrics[0], int(search_cfg.get("generations_per_round", 5)))

                def rescore_batch(phenotypes: list[CandidatePhenotype]) -> list[dict[str, Any]]:
                    batch = proxy.evaluate_batch(phenotypes, generation=int(search_cfg.get("generations_per_round", 5)), outer_round=round_index)
                    return [annotate_metrics(CandidateGenotype({}, {}), row, int(search_cfg.get("generations_per_round", 5))) for row in batch.metrics]

                stage2_pool = scored
                if str(proxy_cfg.get("bops_constraint_mode", "weighted_penalty")) in {
                    "hard_feasibility",
                    "feasibility_first",
                    "hard_band_feasibility",
                }:
                    stage2_pool = [
                        row for row in scored if bool(row[2].get("bops_feasible", False))
                    ]
                repaired_records, repair_report = select_repaired_stage2_topk(
                    stage2_pool,
                    space=context.search_space,
                    repair_fn=repair_candidate,
                    rescore_fn=rescore_one,
                    batch_rescore_fn=rescore_batch,
                    topk=topk_stage2,
                    repair_pool_size=int(search_cfg.get("repair_pool_size", max(50, topk_stage2 * 10))),
                    selection_policy=str(
                        search_cfg.get(
                            "stage2_topk_selection_policy",
                            "three_plus_two_diversity",
                        )
                    ),
                    exploitation_count=int(
                        search_cfg.get("stage2_exploitation_count", 3)
                    ),
                    diversity_count=int(
                        search_cfg.get("stage2_diversity_count", 2)
                    ),
                    taylor_relative_epsilon=float(
                        search_cfg.get("taylor_relative_epsilon", 0.05)
                    ),
                    taylor_absolute_epsilon=float(
                        search_cfg.get("taylor_absolute_epsilon", 1.0e-8)
                    ),
                    eligibility_fn=(
                        lambda metrics: bool(metrics.get("bops_feasible", False))
                    )
                    if str(proxy_cfg.get("bops_constraint_mode", ""))
                    == "hard_band_feasibility"
                    else None,
                )
                if len(repaired_records) < topk_stage2:
                    raise RuntimeError(
                        "insufficient_bops_band_candidates_for_stage2:"
                        f"{len(repaired_records)}<{topk_stage2}:"
                        f"target={round_bops_target}:"
                        f"tolerance={proxy_cfg.get('bops_tolerance_abs', 0.0)}"
                    )
                selected = [
                    type(
                        "Selection",
                        (),
                        {
                            "role": record.metrics.get(
                                "stage2_selection_role", "repaired"
                            ),
                            "record": record,
                        },
                    )
                    for record in repaired_records
                ]
                _write_json(round_dir / "repair_report.json", repair_report)
            else:
                selected = select_stage1_topk(
                    records,
                    real_eval_hashes=set(),
                    archive_genotypes=previous_elite,
                    config=TopKConfig(
                        topk_real=int(search_cfg.get("topk_real", 1)),
                        exploitation_count=int(search_cfg.get("exploitation_count", 1)),
                        diversity_count=int(search_cfg.get("diversity_count", 0)),
                        exploration_count=int(search_cfg.get("exploration_count", 0)),
                    ),
                )
                repair_report = {"repair_pool_size": 0, "repair_failed_count": 0, "duplicate_repaired_phenotype_count": 0}
            _write_json(round_dir / "stage1_topk.json", [{"role": item.role, "candidate_hash": item.record.candidate_hash, "F1": item.record.F1, "phenotype": item.record.phenotype.to_dict()} for item in selected])
            if use_repaired_topk:
                write_repaired_topk_manifest(run_dir, round_index=round_index)
            _append_jsonl(
                run_dir / "seen_repaired_phenotypes.jsonl",
                [
                    {
                        "round": round_index,
                        "repaired_phenotype_hash": item.record.candidate_hash,
                        "F1": item.record.F1,
                        "pruned_unit_count": len(item.record.phenotype.pruned_unit_ids),
                        "precision_profile_hash": canonical_json_hash(item.record.phenotype.realized_precision_profile),
                    }
                    for item in selected
                ],
            )
            manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
            manifest.update(
                {
                    "cache_miss_count": proxy.cache_miss_count,
                    "cache_hit_count": proxy.cache_hit_count,
                    "gpu_batch_count": proxy.gpu_batch_count,
                    "unique_phenotype_count": getattr(proxy, "unique_phenotype_count", 0),
                    "current_unique_phenotype_count": getattr(proxy, "current_unique_phenotype_count", 0),
                    "scalar_evaluate_call_count": proxy.scalar_evaluate_call_count,
                    "batch_evaluate_call_count": proxy.batch_evaluate_call_count,
                    "cuda_event_elapsed_ms": getattr(proxy, "last_batch_stats", {}).get("cuda_event_elapsed_ms", 0.0),
                    "gpu_peak_memory_bytes": getattr(proxy, "last_batch_stats", {}).get("gpu_peak_memory_bytes", 0),
                    "candidates_per_second": getattr(proxy, "last_batch_stats", {}).get("candidates_per_second", 0.0),
                    "last_round_bops_target": round_bops_target,
                    "topk_stage2": topk_stage2,
                }
            )
            _write_json(run_dir / "run_manifest.json", manifest)
            _write_json(
                round_dir / "round_state.json",
                {
                    "phase": "stage1_complete",
                    "round_index": round_index,
                    "stage1_reused": False,
                    "objective_hash": objective_hash,
                    "selected": len(selected),
                },
            )
            if not stage1_only:
                _write_json(
                    round_dir / "round_state.json",
                    {
                        "phase": "stage2_running",
                        "round_index": round_index,
                        "stage1_reused": False,
                        "objective_hash": objective_hash,
                        "selected": len(selected),
                    },
                )
                round_stage2_rows = self._evaluate_ga_stage2_selected_parallel(
                    context=context,
                    real_evaluator=real_evaluator,
                    run_dir=run_dir,
                    round_dir=round_dir,
                    selected=selected,
                )
                evaluated_rows.extend(round_stage2_rows)
                write_round_stage2_results(run_dir, round_index=round_index)
            if not bool(search_cfg.get("independent_budget_rounds", True)):
                previous_elite = [
                    record.genotype
                    for record in records[: max(1, min(5, len(records)))]
                ]
                previous_best = previous_elite[0] if previous_elite else None
            _write_json(round_dir / "round_summary.json", {"best_F1": records[0].F1 if records else None, "selected": len(selected), "evaluated": len(evaluated_rows)})
            if records:
                _write_json(round_dir / "best_candidate.json", {"candidate_hash": records[0].candidate_hash, "F1": records[0].F1, "phenotype": records[0].phenotype.to_dict()})
            if not stage1_only:
                _write_json(
                    round_dir / "round_state.json",
                    {
                        "phase": "round_complete",
                        "round_index": round_index,
                        "stage1_reused": False,
                        "objective_hash": objective_hash,
                        "selected": len(selected),
                    },
                )
        self._write_global(run_dir, evaluated_rows)
        final_cfg = dict(self.config.get("final_selection", {}) or {})
        if (
            not stage1_only
            and bool(final_cfg.get("reevaluate_round_winners_on_full_validation", True))
        ):
            self._full_validate_ga_round_winners(context, run_dir)
        return evaluated_rows

    def _full_validate_ga_round_winners(
        self,
        context: Any,
        run_dir: Path,
    ) -> list[dict[str, Any]]:
        winners: dict[str, dict[str, Any]] = {}
        for path in sorted(run_dir.glob("round_*/round_best_candidate.json")):
            row = json.loads(path.read_text(encoding="utf-8"))
            candidate_id = str(row.get("candidate_hash", ""))
            source = Path(str(row.get("artifact_dir", "")))
            if not candidate_id or not source.is_dir():
                raise RuntimeError(
                    f"round_winner_artifact_missing:{path}:{candidate_id}:{source}"
                )
            phenotype_path = source / "phenotype.json"
            if not phenotype_path.is_file():
                raise RuntimeError(
                    f"round_winner_phenotype_missing:{phenotype_path}"
                )
            entry = winners.setdefault(
                candidate_id,
                {
                    "candidate_hash": candidate_id,
                    "source_artifact_dir": str(source),
                    "phenotype": CandidatePhenotype.from_dict(
                        json.loads(phenotype_path.read_text(encoding="utf-8"))
                    ),
                    "compression_metrics": {
                        key: row.get(key)
                        for key in (
                            "BOPS_target",
                            "R_BOPS_vs_FP32",
                            "BOPS_abs_delta",
                            "BOPS_compression_x",
                            "R_Size_vs_FP32",
                            "mixed_weight_compression_x",
                            "parameter_count_base",
                            "parameter_count_after",
                            "parameter_pruning_rate",
                            "parameter_compression_x",
                        )
                    },
                    "rounds": [],
                },
            )
            entry["rounds"].append(path.parent.name)
        if not winners:
            return []
        evaluator = self._full_validation_evaluator(context, run_dir)
        rows: list[dict[str, Any]] = []
        for candidate_id, entry in sorted(winners.items()):
            result = evaluator.reevaluate_existing_candidate_engine(
                entry["phenotype"],
                source_artifact_dir=entry["source_artifact_dir"],
                output_dir=run_dir / "full_validation" / "candidates" / candidate_id,
                candidate_hash=candidate_id,
            )
            latency_ratio = float(result.get("R_latency_real", 0.0) or 0.0)
            rows.append(
                {
                    "rounds": sorted(entry["rounds"]),
                    **dict(entry["compression_metrics"]),
                    "measured_speedup_vs_FP32": (
                        1.0 / latency_ratio if latency_ratio > 0.0 else None
                    ),
                    **result,
                }
            )
        successful = [row for row in rows if str(row.get("status", "")) == "ok"]
        winner = (
            min(successful, key=lambda row: float(row.get("F2", float("inf"))))
            if successful
            else None
        )
        _write_json(
            run_dir / "final_full_validation_results.json",
            {
                "candidates": rows,
                "winner": winner,
                "candidate_engine_rebuild_count": 0,
                "deduplicated_round_winner_count": len(winners),
            },
        )
        if winner is not None:
            _write_json(run_dir / "final_winner.json", winner)
        return rows

    @staticmethod
    def _raw_genotype_hash(genotype: CandidateGenotype, context: Any) -> str:
        return canonical_json_hash(
            {
                "pruning_genes": genotype.pruning_genes,
                "pruning_width_genes": genotype.pruning_width_genes,
                "precision_genes": genotype.precision_genes,
                "trace_hash": context.search_space.trace_snapshot_hash,
                "search_space_version": "legal-domain-width-joint-weight-taylor-v1",
            }
        )

    @staticmethod
    def _load_seen_raw_hashes(run_dir: Path) -> set[str]:
        path = run_dir / "seen_raw_genotypes.jsonl"
        if not path.is_file():
            return set()
        seen: set[str] = set()
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                value = row.get("raw_genotype_hash")
                if value:
                    seen.add(str(value))
        return seen

    @staticmethod
    def _round_stage2_complete(round_dir: Path, topk_stage2: int) -> bool:
        results_path = round_dir / "stage2_top5_results.json"
        best_path = round_dir / "round_best_F1_F2.json"
        if not (results_path.is_file() and best_path.is_file()):
            return False
        try:
            results = json.loads(results_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return False
        rows = results.get("candidates", []) or []
        return len(rows) >= int(topk_stage2) and any(str(row.get("status")) == "ok" for row in rows)

    @staticmethod
    def _round_topk_as_genotypes(round_dir: Path, context: Any) -> list[CandidateGenotype]:
        path = round_dir / "stage1_topk.json"
        if not path.is_file():
            return []
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
        genotypes: list[CandidateGenotype] = []
        for row in rows:
            phenotype_payload = row.get("phenotype") or {}
            phenotype = CandidatePhenotype.from_dict(phenotype_payload)
            genotypes.append(
                LidarPyramidTwoStageSearch._genotype_from_phenotype(
                    phenotype, context
                )
            )
        return genotypes

    @staticmethod
    def _genotype_from_phenotype(
        phenotype: CandidatePhenotype, context: Any
    ) -> CandidateGenotype:
        pruned = set(phenotype.pruned_unit_ids)
        metadata = dict(phenotype.metadata or {})
        requested_groups = dict(
            metadata.get("requested_group_profile")
            or metadata.get("stage1_legalized_group_profile")
            or {}
        )
        requested_layers = phenotype.requested_precision_profile
        return CandidateGenotype(
            pruning_genes=(
                {}
                if context.search_space.pruning_domains
                else {
                    unit_id: (0 if unit_id in pruned else 1)
                    for unit_id in context.search_space.pruning_unit_ids
                }
            ),
            precision_genes={
                group_id: requested_groups.get(
                    group_id,
                    requested_layers.get(
                        group_id, context.search_space.default_precision
                    ),
                )
                for group_id in context.search_space.precision_gene_ids
            },
            pruning_width_genes={
                str(key): int(value)
                for key, value in dict(
                    metadata.get("domain_width_profile") or {}
                ).items()
            },
            meta={"created_by": "resume_stage1_topk"},
        )

    @staticmethod
    def _load_round_stage1_selections(
        round_dir: Path, context: Any
    ) -> list[Any]:
        path = round_dir / "stage1_topk.json"
        if not path.is_file():
            return []
        rows = json.loads(path.read_text(encoding="utf-8"))
        selections = []
        for row in rows:
            phenotype = CandidatePhenotype.from_dict(row.get("phenotype") or {})
            record = ProxyCandidateRecord(
                candidate_hash=str(row.get("candidate_hash", "")),
                genotype=LidarPyramidTwoStageSearch._genotype_from_phenotype(
                    phenotype, context
                ),
                phenotype=phenotype,
                F1=float(row.get("F1", float("inf"))),
                metrics={"resume_source": str(path)},
            )
            selections.append(
                type(
                    "ResumeSelection",
                    (),
                    {
                        "role": str(row.get("role", "repaired")),
                        "record": record,
                    },
                )
            )
        return selections

    @staticmethod
    def _records_from_scored(scored: list[tuple[CandidateGenotype, float, dict[str, Any]]], context: Any) -> list[ProxyCandidateRecord]:
        unique: dict[str, ProxyCandidateRecord] = {}
        for genotype, score, metrics in scored:
            phenotype = canonicalize_candidate(genotype, context.search_space)
            key = candidate_hash(phenotype, context.search_space)
            unique.setdefault(key, ProxyCandidateRecord(key, genotype, phenotype, float(score), metrics))
        return sorted(unique.values(), key=lambda row: row.F1)

    @staticmethod
    def _write_generation(path: Path, rows: list[tuple[CandidateGenotype, float, dict[str, Any]]], context: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["candidate_hash", "F1", "L_fisher", "L_sqnr", "R_size", "R_size_vs_fp32", "R_size_vs_fp16_deploy", "R_parameter_retention", "parameter_pruning_rate", "parameter_count_base", "parameter_count_after", "R_bops", "R_bops_vs_fp32", "R_bops_vs_fp16_deploy", "BOPS_target", "bops_abs_delta", "P_bops", "bops_feasible", "bops_violation", "int8_macs_ratio", "grouped_action_count", "pruned_units", "int8_layers", "cache_hit"])
            writer.writeheader()
            for genotype, score, metrics in rows:
                phenotype = canonicalize_candidate(genotype, context.search_space)
                writer.writerow(
                    {
                        "candidate_hash": candidate_hash(phenotype, context.search_space),
                        "F1": score,
                        "L_fisher": metrics.get("L_fisher"),
                        "L_sqnr": metrics.get("L_sqnr"),
                        "R_size": metrics.get("R_size"),
                        "R_size_vs_fp32": metrics.get("R_size_vs_fp32"),
                        "R_size_vs_fp16_deploy": metrics.get("R_size_vs_fp16_deploy"),
                        "R_parameter_retention": metrics.get("R_parameter_retention"),
                        "parameter_pruning_rate": metrics.get("parameter_pruning_rate"),
                        "parameter_count_base": metrics.get("parameter_count_base"),
                        "parameter_count_after": metrics.get("parameter_count_after"),
                        "R_bops": metrics.get("R_bops"),
                        "R_bops_vs_fp32": metrics.get("R_bops_vs_fp32"),
                        "R_bops_vs_fp16_deploy": metrics.get("R_bops_vs_fp16_deploy"),
                        "BOPS_target": metrics.get("BOPS_target"),
                        "bops_abs_delta": metrics.get("bops_abs_delta"),
                        "P_bops": metrics.get("P_bops"),
                        "bops_feasible": metrics.get("bops_feasible"),
                        "bops_violation": metrics.get("bops_violation"),
                        "int8_macs_ratio": metrics.get("int8_macs_ratio"),
                        "grouped_action_count": metrics.get("grouped_action_count"),
                        "pruned_units": len(phenotype.pruned_unit_ids),
                        "int8_layers": sum(value == "INT8" for value in phenotype.realized_precision_profile.values()),
                        "cache_hit": bool(metrics.get("cache_hit", False)),
                    }
                )

    @staticmethod
    def _write_stage1(path: Path, records: list[ProxyCandidateRecord]) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["candidate_hash", "F1", "L_fisher", "L_sqnr", "R_size", "R_size_vs_fp32", "R_size_vs_fp16_deploy", "R_parameter_retention", "parameter_pruning_rate", "parameter_count_base", "parameter_count_after", "R_bops", "R_bops_vs_fp32", "R_bops_vs_fp16_deploy", "BOPS_target", "bops_abs_delta", "P_bops", "bops_feasible", "bops_violation", "int8_macs_ratio", "grouped_action_count", "pruned_units", "int8_layers"])
            writer.writeheader()
            for record in records:
                writer.writerow(
                    {
                        "candidate_hash": record.candidate_hash,
                        "F1": record.F1,
                        "L_fisher": record.metrics.get("L_fisher"),
                        "L_sqnr": record.metrics.get("L_sqnr"),
                        "R_size": record.metrics.get("R_size"),
                        "R_size_vs_fp32": record.metrics.get("R_size_vs_fp32"),
                        "R_size_vs_fp16_deploy": record.metrics.get("R_size_vs_fp16_deploy"),
                        "R_parameter_retention": record.metrics.get("R_parameter_retention"),
                        "parameter_pruning_rate": record.metrics.get("parameter_pruning_rate"),
                        "parameter_count_base": record.metrics.get("parameter_count_base"),
                        "parameter_count_after": record.metrics.get("parameter_count_after"),
                        "R_bops": record.metrics.get("R_bops"),
                        "R_bops_vs_fp32": record.metrics.get("R_bops_vs_fp32"),
                        "R_bops_vs_fp16_deploy": record.metrics.get("R_bops_vs_fp16_deploy"),
                        "BOPS_target": record.metrics.get("BOPS_target"),
                        "bops_abs_delta": record.metrics.get("bops_abs_delta"),
                        "P_bops": record.metrics.get("P_bops"),
                        "bops_feasible": record.metrics.get("bops_feasible"),
                        "bops_violation": record.metrics.get("bops_violation"),
                        "int8_macs_ratio": record.metrics.get("int8_macs_ratio"),
                        "grouped_action_count": record.metrics.get("grouped_action_count"),
                        "pruned_units": len(record.phenotype.pruned_unit_ids),
                        "int8_layers": sum(value == "INT8" for value in record.phenotype.realized_precision_profile.values()),
                    }
                )

    @staticmethod
    def _write_global(run_dir: Path, rows: list[dict[str, Any]]) -> None:
        if not rows:
            _write_json(run_dir / "global_summary.json", {"evaluated": 0})
            (run_dir / "global_summary.csv").write_text("candidate_hash,F1,F2,status\n", encoding="utf-8")
            return
        best = min(rows, key=lambda row: float(row.get("F2", float("inf"))))
        _write_json(run_dir / "global_summary.json", {"evaluated": len(rows), "best": best})
        _write_json(run_dir / "global_best_candidate.json", best)
        fields = sorted({key for row in rows for key in row})
        with (run_dir / "global_summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
from ..pruning_space.action_codec import sanitize_action_parameter_slices
