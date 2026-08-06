"""Family-aware two-stage search for HEAL LiDAR F-Cooper and DiscoNet."""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from deploy.post_scatter import prepare_post_scatter_inputs

from ..cache.proxy_cache import ProxyCache
from ..candidate import CandidateGenotype, CandidatePhenotype
from ..canonicalization import canonicalize_candidate
from ..integration.calibration_provider import collect_or_load_fisher_statistics
from ..integration.heal_lidar_baseline_context import build_heal_lidar_baseline_context
from ..integration.runtime_environment import GPUSelection, query_gpus
from ..model_family.model_provider import load_heal_model_family
from ..proxy.gpu_batch_proxy import MultiDeviceTorchBatchedProxyScorer, TorchBatchedProxyScorer
from ..proxy.parameter_slice_resolver import build_unit_parameter_slices
from ..proxy.runtime_shape_profiler import profile_runtime_layer_shapes
from ..pruning_space.domain_importance import score_atomic_units_for_fixed_ranking
from ..pruning_space.local_domains import build_local_pruning_domains
from ..stage1.proxy_evaluator import Stage1ProxyEvaluator
from ..stage2.heal_lidar_baseline_real_evaluator import HealLidarBaselineCandidateEvaluator
from ..stage2.objective import Stage2ObjectiveConfig
from ..resource_monitor import SearchResourceMonitor
from .lidar_pyramid_search import (
    LidarPyramidTwoStageSearch,
    _load_candidate,
    _proxy_device_from_config,
    _require_gpu_proxy_if_needed,
    _shared_eval_manifest_protocol,
    _write_json,
)
from ..hashing import candidate_hash, search_hash


def _read_json_if_possible(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return dict(payload) if isinstance(payload, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def _write_search_cost_summary(
    run_dir: Path,
    *,
    method: str,
    proxy: Stage1ProxyEvaluator | None,
    resource_summary: dict[str, Any],
) -> dict[str, Any]:
    engine_timings = [
        _read_json_if_possible(path)
        for path in run_dir.rglob("engine_build_phase_timings.json")
    ]
    engine_phase_totals: dict[str, float] = {}
    for payload in engine_timings:
        for name, value in dict(payload.get("timings", {}) or {}).items():
            engine_phase_totals[str(name)] = (
                engine_phase_totals.get(str(name), 0.0) + float(value or 0.0)
            )
    evaluations: list[tuple[Path, dict[str, Any]]] = [
        (path, _read_json_if_possible(path))
        for path in run_dir.rglob("evaluation_acceptance.json")
    ]
    candidate_frames = 0
    reference_frames = 0
    candidate_evaluations = 0
    reference_evaluations = 0
    for path, payload in evaluations:
        identity = dict(payload.get("evaluator_identity", {}) or {})
        frames = int(identity.get("num_frames", payload.get("num_frames", 0)) or 0)
        if "reference" in path.parts or "baseline" in path.parts:
            reference_evaluations += 1
            reference_frames += frames
        else:
            candidate_evaluations += 1
            candidate_frames += frames
    greedy_payload = _read_json_if_possible(run_dir / "greedy/greedy_path.json")
    greedy_neighbors = [
        {
            "step": int(row.get("step_index", index + 1)),
            "neighbor_count": int(row.get("neighbor_count", 0)),
        }
        for index, row in enumerate(greedy_payload.get("steps", []) or [])
    ]
    ga_generation_rows = []
    for path in sorted(run_dir.rglob("ga_generation_resource_stats.json")):
        payload = _read_json_if_possible(path)
        for row in payload.get("generations", []) or []:
            ga_generation_rows.append(
                {
                    "round_index": int(payload.get("round_index", -1)),
                    "bops_target": payload.get("bops_target"),
                    **dict(row),
                }
            )
    proxy_peak_bytes = int(
        getattr(proxy, "last_batch_stats", {}).get("gpu_peak_memory_bytes", 0)
        if proxy is not None
        else 0
    )
    stage1_phases = {
        name: row
        for name, row in dict(resource_summary.get("phase_totals", {}) or {}).items()
        if str(name).startswith("stage1.")
    }
    stage2_phases = {
        name: row
        for name, row in dict(resource_summary.get("phase_totals", {}) or {}).items()
        if str(name).startswith("stage2.")
    }
    summary = {
        "schema_version": "heal-lidar-search-cost-summary-v1",
        "method": str(method),
        "stage1_elapsed_seconds": sum(
            float(row.get("elapsed_seconds", 0.0)) for row in stage1_phases.values()
        ),
        "stage1_allocated_gpu_hours": sum(
            float(row.get("allocated_gpu_hours", 0.0))
            for row in stage1_phases.values()
        ),
        "stage2_elapsed_gpu_seconds_sum": 3600.0
        * sum(
            float(row.get("allocated_gpu_hours", 0.0))
            for row in stage2_phases.values()
        ),
        "stage2_allocated_gpu_hours": sum(
            float(row.get("allocated_gpu_hours", 0.0))
            for row in stage2_phases.values()
        ),
        "per_phase": dict(resource_summary.get("phase_totals", {}) or {}),
        "proxy_evaluation_request_count": (
            int(proxy.cache_hit_count + proxy.cache_miss_count) if proxy is not None else 0
        ),
        "proxy_evaluation_count": int(proxy.cache_miss_count) if proxy is not None else 0,
        "unique_candidate_count": int(proxy.unique_phenotype_count) if proxy is not None else 0,
        "cache_hit_count": int(proxy.cache_hit_count) if proxy is not None else 0,
        "cache_miss_count": int(proxy.cache_miss_count) if proxy is not None else 0,
        "gpu_batch_count": int(proxy.gpu_batch_count) if proxy is not None else 0,
        "stage2_engine_build_attempt_count": len(engine_timings),
        "stage2_candidate_evaluation_count": candidate_evaluations,
        "stage2_reference_evaluation_count": reference_evaluations,
        "stage2_candidate_frames": candidate_frames,
        "stage2_reference_frames": reference_frames,
        "stage2_total_frames": candidate_frames + reference_frames,
        "peak_vram_mib": max(
            float(resource_summary.get("peak_memory_used_mib", 0.0) or 0.0),
            proxy_peak_bytes / (1024.0 * 1024.0),
        ),
        "engine_build_phase_totals_seconds": engine_phase_totals,
        "greedy_neighbors_per_step": greedy_neighbors,
        "ga_canonical_phenotypes_per_generation": ga_generation_rows,
        "resource_monitor": resource_summary,
    }
    _write_json(run_dir / "search_cost_summary.json", summary)
    return summary


class HealLidarBaselineTwoStageSearch(LidarPyramidTwoStageSearch):
    """Reuse the accepted GA/greedy semantics with family-scoped deployment."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        model_cfg = dict(self.config.get("model", {}) or {})
        self.family_id = str(model_cfg.get("family_id", ""))
        expected = {
            "heal_lidar_fcooper": "lidar_fcooper",
            "heal_lidar_disco": "lidar_disco",
        }
        if self.family_id not in expected:
            raise RuntimeError(f"unsupported_baseline_search_family:{self.family_id}")
        self.model_name = expected[self.family_id]

    @staticmethod
    def _strict_idle_gpu_pool(
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
        rejected = []
        selected = []
        for gpu_id in requested:
            row = by_id.get(gpu_id)
            reasons = []
            if row is None:
                reasons.append("gpu_missing")
            else:
                if int(row["memory_free_mib"]) < minimum_free:
                    reasons.append(f"free_mib<{minimum_free}")
                if int(row["utilization_gpu_pct"]) > maximum_utilization:
                    reasons.append(f"utilization>{maximum_utilization}")
            if reasons:
                rejected.append({"gpu_id": gpu_id, "reasons": reasons, "snapshot": row})
            else:
                selected.append(gpu_id)
        minimum_workers = int(runtime.get(f"{role}_minimum_workers", 1))
        if len(selected) < minimum_workers:
            raise RuntimeError(
                f"heal_lidar_{role}_gpu_pool_insufficient:selected={selected}:"
                f"minimum={minimum_workers}:rejected={rejected}"
            )
        maximum_workers = int(runtime.get(f"{role}_max_workers", len(selected)))
        return selected[:maximum_workers], report

    def _objective_config(self) -> Stage2ObjectiveConfig:
        stage2_cfg = dict(
            self.config.get("stage2")
            or self.config.get("stage2_smoke")
            or self.config.get("evaluation", {})
        )
        return Stage2ObjectiveConfig(
            eta_map=float(stage2_cfg.get("eta_ap", stage2_cfg.get("eta_map", 1.0))),
            eta_latency=float(stage2_cfg.get("eta_latency", 1.0)),
            latency_metric=str(stage2_cfg.get("latency_metric", "forward_p50_ms")),
            accuracy_reference=str(stage2_cfg.get("accuracy_reference", "original_strict_fp32")),
            latency_reference=str(stage2_cfg.get("latency_reference", "original_strict_fp32")),
            tau_ap=stage2_cfg.get("tau_ap"),
            max_map_drop=stage2_cfg.get("max_map_drop"),
        )

    def _baseline_engine(self) -> Path | None:
        baseline_cfg = dict(self.config.get("baselines", {}) or {})
        path = Path(str(baseline_cfg.get("strict_fp32_engine", ""))).expanduser().resolve()
        return path if path.is_file() else None

    def _candidate_evaluator(
        self,
        context: Any,
        run_dir: Path,
        *,
        num_frames: int,
        warmup_frames: int,
        latency_rounds: int,
        reference_baseline: dict[str, Any] | None = None,
    ) -> HealLidarBaselineCandidateEvaluator:
        return HealLidarBaselineCandidateEvaluator(
            context=context,
            run_dir=run_dir,
            baseline_engine_path=self._baseline_engine(),
            num_frames=num_frames,
            warmup_frames=warmup_frames,
            latency_rounds=latency_rounds,
            objective_config=self._objective_config(),
            dataloader_num_workers=int(
                dict(self.config.get("stage2", {}) or {}).get("dataloader_num_workers", 8)
            ),
            reference_baseline=reference_baseline,
        )

    def _full_validation_evaluator(self, context: Any, run_dir: Path) -> Any:
        full = dict(self.config.get("full_validation", {}) or {})
        return self._candidate_evaluator(
            context,
            run_dir / "full_validation",
            num_frames=int(full.get("num_frames", 1789)),
            warmup_frames=int(full.get("warmup_frames", 200)),
            latency_rounds=int(full.get("latency_rounds", 3)),
        )

    def _ga_stage2_evaluator_pool(
        self,
        context: Any,
        real_evaluator: HealLidarBaselineCandidateEvaluator,
        run_dir: Path,
        *,
        pool_kind: str = "screening_500",
    ) -> list[tuple[int, HealLidarBaselineCandidateEvaluator]]:
        if pool_kind not in {"screening_500", "full_validation"}:
            raise ValueError(f"unsupported_ga_evaluator_pool_kind:{pool_kind}")
        cache_attribute = (
            "_ga_stage2_worker_pool"
            if pool_kind == "screening_500"
            else "_ga_full_validation_worker_pool"
        )
        worker_dir_name = (
            "stage2_workers"
            if pool_kind == "screening_500"
            else "full_validation_workers"
        )
        cached = getattr(self, cache_attribute, None)
        if cached is not None:
            return list(cached)
        gpu_ids = list(getattr(self, "_stage2_gpu_ids", [context.physical_gpu_id]))
        reference = real_evaluator._stage2_reference_baseline()
        real_evaluator._reference_baseline_override = dict(reference)
        workers: list[tuple[int, HealLidarBaselineCandidateEvaluator]] = []
        rows = []
        runtime = dict(getattr(self, "_runtime_config", {}) or {})
        gpu_report = query_gpus()
        for gpu_id in gpu_ids:
            if int(gpu_id) == int(context.physical_gpu_id):
                workers.append((int(gpu_id), real_evaluator))
                rows.append({"gpu_id": int(gpu_id), "status": "ready", "source": "primary"})
                continue
            try:
                bundle = load_heal_model_family(
                    config_path=context.model_config,
                    checkpoint_path=context.checkpoint_path,
                    heal_root=runtime["heal_root"],
                    device=f"cuda:{gpu_id}",
                    family_id=context.family_id,
                    forward_smoke=True,
                )
                if bundle.checkpoint_hash != context.checkpoint_hash:
                    raise RuntimeError("checkpoint_hash_mismatch")
                export_inputs = prepare_post_scatter_inputs(
                    bundle.model,
                    bundle.example_batch,
                    max_agents=context.max_agents,
                    include_agent_mask=True,
                )
                worker_context = replace(
                    context,
                    model=bundle.model,
                    model_bundle=bundle,
                    trace_example_inputs=bundle.example_batch,
                    export_example_inputs=export_inputs,
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
                worker = self._candidate_evaluator(
                    worker_context,
                    run_dir / worker_dir_name / f"gpu_{gpu_id}",
                    num_frames=real_evaluator.num_frames,
                    warmup_frames=real_evaluator.warmup_frames,
                    latency_rounds=real_evaluator.latency_rounds,
                    reference_baseline=reference,
                )
                workers.append((int(gpu_id), worker))
                rows.append({"gpu_id": int(gpu_id), "status": "ready", "source": "strict_reload"})
            except Exception as exc:  # noqa: BLE001
                rows.append({
                    "gpu_id": int(gpu_id),
                    "status": "rejected",
                    "reason": f"{type(exc).__name__}:{exc}",
                })
        if len(workers) < int(runtime.get("stage2_minimum_workers", 1)):
            raise RuntimeError(f"baseline_stage2_worker_pool_insufficient:{rows}")
        _write_json(
            run_dir / worker_dir_name / "worker_pool_manifest.json",
            {
                "workers": rows,
                "active_gpu_ids": [gpu_id for gpu_id, _ in workers],
                "pool_kind": pool_kind,
            },
        )
        setattr(self, cache_attribute, list(workers))
        return workers

    def run(
        self,
        *,
        stage1_only: bool = False,
        stage2_only: bool = False,
        baseline_only: bool = False,
        candidate_config: str | Path | list[str] | list[Path] | None = None,
    ) -> dict[str, Any]:
        run_started = time.perf_counter()
        run_dir = self._run_dir()
        (run_dir / "archives").mkdir(parents=True, exist_ok=True)
        (run_dir / "baseline").mkdir(parents=True, exist_ok=True)
        runtime = dict(self.config.get("runtime", {}) or {})
        search_cfg = dict(self.config.get("search", {}) or {})
        proxy_cfg = dict(self.config.get("proxy", self.config.get("proxy_objective", {})) or {})
        stage2_cfg = dict(
            self.config.get("stage2")
            or self.config.get("stage2_smoke")
            or self.config.get("evaluation", {})
        )
        full_cfg = dict(self.config.get("full_validation", {}) or {})
        model_cfg = dict(self.config.get("model", {}) or {})
        manifest_frames, manifest_warmup, manifest_reset = _shared_eval_manifest_protocol(
            stage2_cfg, full_cfg
        )
        context = build_heal_lidar_baseline_context(
            family_id=self.family_id,
            checkpoint_path=self.checkpoint,
            model_config_path=model_cfg["config"],
            output_dir=run_dir,
            heal_root=runtime["heal_root"],
            tensorrt_root=runtime["tensorrt_root"],
            plugin_path=None,
            gpu_id=str(runtime.get("gpu_id", "auto")),
            exclude_gpu_ids=[int(value) for value in runtime.get("exclude_gpu_ids", [])],
            tensorrt_env=str(runtime.get("tensorrt_env", "modelopt")),
            fisher_calibration_batches=int(proxy_cfg.get("fisher_calibration_batches", 8)),
            quant_calibration_batches=int(proxy_cfg.get("quant_calibration_batches", 200)),
            quant_calibration_npz_manifest=proxy_cfg.get("quant_calibration_npz_manifest"),
            quant_activation_calibration_backend=str(
                proxy_cfg.get("quant_activation_calibration_backend", "modelopt_histogram_entropy")
            ),
            quant_activation_calibration_cache_path=proxy_cfg.get(
                "quant_activation_calibration_cache_path"
            ),
            quant_calibration_force_rebuild=bool(
                proxy_cfg.get("quant_calibration_force_rebuild", False)
            ),
            num_frames=manifest_frames,
            warmup_frames=manifest_warmup,
            reset_after_warmup=manifest_reset,
            default_precision=str(self.config.get("precision", {}).get("default", "FP16")),
            max_agents=int(model_cfg.get("max_agents", 2)),
            minimum_retained_ratio=float(
                dict(self.config.get("pruning", {}) or {}).get("minimum_retained_ratio", 0.10)
            ),
            dense_alignment=int(
                dict(self.config.get("pruning", {}) or {}).get("dense_channel_alignment", 4)
            ),
            require_quant_calibration_manifest=False,
            search_space_policy=str(
                model_cfg.get(
                    "search_space_policy",
                    "legacy_family_static_dependency_closure_v1",
                )
            ),
        )
        _write_json(
            run_dir / "run_manifest.json",
            {
                "family_id": self.family_id,
                "model_name": self.model_name,
                "checkpoint": str(self.checkpoint),
                "stage1_only": stage1_only,
                "stage2_only": stage2_only,
                "baseline_only": baseline_only,
                "resume": str(self.resume or ""),
            },
        )
        _write_json(
            run_dir / "environment.json",
            {"gpu": context.gpu_selection.to_dict(), "tensorrt": context.tensorrt.to_dict()},
        )
        stage1_gpu_ids, stage1_report = self._strict_idle_gpu_pool(
            runtime, role="stage1", primary_gpu_id=context.physical_gpu_id
        )
        if stage1_only:
            stage2_gpu_ids, stage2_report = [context.physical_gpu_id], stage1_report
        else:
            stage2_gpu_ids, stage2_report = self._strict_idle_gpu_pool(
                runtime, role="stage2", primary_gpu_id=context.physical_gpu_id
            )
        self._stage2_gpu_ids = stage2_gpu_ids
        self._stage1_gpu_ids = stage1_gpu_ids
        self._runtime_config = runtime
        _write_json(run_dir / "gpu_parallelism_manifest.json", {
            "stage1_gpu_ids": stage1_gpu_ids,
            "stage2_gpu_ids": stage2_gpu_ids,
            "stage1_gpu_snapshot": stage1_report,
            "stage2_gpu_snapshot": stage2_report,
            "minimum_free_memory_mib": int(runtime.get("parallel_gpu_min_free_mib", 60_000)),
            "maximum_utilization_pct": int(runtime.get("parallel_gpu_max_utilization_pct", 10)),
            "selection_policy": "fail_closed",
        })
        monitor = SearchResourceMonitor(
            run_dir / "resources",
            sorted(set(stage1_gpu_ids) | set(stage2_gpu_ids)),
            sample_interval_seconds=float(
                dict(self.config.get("resources", {}) or {}).get(
                    "gpu_sample_interval_seconds", 0.5
                )
            ),
        )
        self._resource_recorder = monitor
        monitor.record_completed_phase(
            "setup.context_and_runtime_trace",
            time.perf_counter() - run_started,
            [int(context.physical_gpu_id)],
            search_space_policy=str(context.search_space_policy),
        )
        if baseline_only:
            evaluator = self._candidate_evaluator(
                context,
                run_dir / "baseline_only",
                num_frames=int(stage2_cfg.get("num_frames", 500)),
                warmup_frames=int(stage2_cfg.get("warmup_frames", 200)),
                latency_rounds=int(stage2_cfg.get("latency_rounds", 3)),
            )
            with self._resource_phase(
                "stage2.reference_baseline",
                [int(context.physical_gpu_id)],
            ):
                baseline = evaluator._stage2_reference_baseline()
            resource_summary = monitor.close()
            cost = _write_search_cost_summary(
                run_dir,
                method="baseline_only",
                proxy=None,
                resource_summary=resource_summary,
            )
            return {
                "run_dir": str(run_dir),
                "baseline_only": True,
                "baseline": baseline,
                "search_cost_summary": cost,
            }
        if stage2_only:
            if candidate_config is None:
                raise RuntimeError("baseline_stage2_only_requires_candidate_config")
            paths = candidate_config if isinstance(candidate_config, list) else [candidate_config]
            evaluator = self._candidate_evaluator(
                context,
                run_dir / "stage2_only",
                num_frames=int(stage2_cfg.get("num_frames", 500)),
                warmup_frames=int(stage2_cfg.get("warmup_frames", 200)),
                latency_rounds=int(stage2_cfg.get("latency_rounds", 3)),
            )
            results = []
            for source in paths:
                loaded = _load_candidate(source)
                candidate = (
                    loaded
                    if isinstance(loaded, CandidatePhenotype)
                    else canonicalize_candidate(loaded, context.search_space)
                )
                identity = candidate_hash(candidate, context.search_space)
                with self._resource_phase(
                    "stage2.candidate_evaluation",
                    [int(context.physical_gpu_id)],
                    candidate_hash=identity,
                ):
                    results.append(evaluator.evaluate_candidate(
                        candidate,
                        output_dir=run_dir / "stage2_only" / identity,
                        candidate_hash=identity,
                    ))
            resource_summary = monitor.close()
            cost = _write_search_cost_summary(
                run_dir,
                method="stage2_only",
                proxy=None,
                resource_summary=resource_summary,
            )
            return {
                "run_dir": str(run_dir),
                "selected_gpu": context.physical_gpu_id,
                "stage2_only": True,
                "results": results,
                "result": results[0] if results else None,
                "search_cost_summary": cost,
            }

        preparation_started = time.perf_counter()
        raw_unit_slices = build_unit_parameter_slices(
            context.model, context.atomic_prune_units
        )
        runtime_shapes = profile_runtime_layer_shapes(
            context.model,
            context.trace_example_inputs,
            forward_fn=context.model_bundle.adapter.forward_for_task,
        )
        _write_json(run_dir / "runtime_layer_shapes.json", runtime_shapes.to_dict())
        fisher = collect_or_load_fisher_statistics(
            model=context.model,
            adapter=context.model_bundle.adapter,
            model_config_path=context.model_config,
            device=__import__("torch").device(context.runtime_device),
            cache_path=run_dir / "archives/fisher_statistics.pt",
            num_batches=context.fisher_calibration_batches,
        )
        fixed_scores, ranking = score_atomic_units_for_fixed_ranking(
            context.model,
            fisher,
            raw_unit_slices,
            strict=True,
        )
        domains = build_local_pruning_domains(
            context.atomic_prune_units,
            importance_scores=fixed_scores,
            ranking_method="pruning_only_first_plus_second_order_fisher_taylor",
            minimum_retained_ratio=float(
                dict(self.config.get("pruning", {}) or {}).get("minimum_retained_ratio", 0.10)
            ),
            dense_alignment=int(
                dict(self.config.get("pruning", {}) or {}).get("dense_channel_alignment", 4)
            ),
        )
        context.search_space = replace(
            context.search_space,
            pruning_domains=tuple(domains),
        )
        _write_json(run_dir / "archives/fixed_pruning_taylor_ranking.json", ranking)
        self._write_local_domains(context, raw_unit_slices, run_dir)
        raw_objective = self._objective(
            context, raw_unit_slices, fisher, None, runtime_shapes.shapes
        )
        normalization = self._build_normalization(context, raw_objective, run_dir)
        objective = self._objective(
            context, raw_unit_slices, fisher, normalization, runtime_shapes.shapes
        )
        proxy_device = _proxy_device_from_config(proxy_cfg, context)
        proxy_batch_size = int(proxy_cfg.get("batch_size", 128))
        batch_scorer = None
        proxy_backend = "scalar_cpu"
        if str(proxy_device).startswith("cuda"):
            primary = TorchBatchedProxyScorer.from_components(
                model=context.model,
                space=context.search_space,
                unit_to_parameter_slices=raw_unit_slices,
                fisher_statistics=fisher,
                runtime_shapes=runtime_shapes.shapes,
                normalization=normalization,
                config=objective.config,
                device=proxy_device,
                batch_size=proxy_batch_size,
            )
            scorers = tuple(
                primary if int(gpu_id) == int(context.physical_gpu_id)
                else primary.clone_to_device(f"cuda:{gpu_id}")
                for gpu_id in stage1_gpu_ids
            )
            batch_scorer = (
                MultiDeviceTorchBatchedProxyScorer(scorers=scorers)
                if len(scorers) > 1 else primary
            )
            proxy_backend = "cuda_batched"
        _require_gpu_proxy_if_needed(
            proxy_cfg=proxy_cfg,
            search_cfg=search_cfg,
            actual_backend=proxy_backend,
        )
        proxy = Stage1ProxyEvaluator(
            context.search_space,
            objective=objective,
            cache=ProxyCache(run_dir / "archives/proxy_archive.jsonl"),
            cache_key_fn=lambda phenotype, _space: search_hash(
                phenotype,
                trace_hash=context.search_space.trace_snapshot_hash,
                proxy_version="heal-lidar-baseline-fisher-sqnr-size-bops-v1",
                calibration_statistics_version=f"{fisher.statistics_version}:{fisher.manifest_hash}",
            ),
            batch_scorer=batch_scorer,
            proxy_backend=proxy_backend,
            proxy_device=proxy_device,
            proxy_batch_size=proxy_batch_size,
        )
        evaluator = self._candidate_evaluator(
            context,
            run_dir,
            num_frames=int(stage2_cfg.get("num_frames", 500)),
            warmup_frames=int(stage2_cfg.get("warmup_frames", 200)),
            latency_rounds=int(stage2_cfg.get("latency_rounds", 3)),
        )
        monitor.record_completed_phase(
            "stage1.preparation",
            time.perf_counter() - preparation_started,
            stage1_gpu_ids,
            fisher_calibration_batches=int(context.fisher_calibration_batches),
        )
        method = str(search_cfg.get("method", "ga")).lower()
        if method == "greedy":
            rows = self._run_greedy(
                context, proxy, run_dir, search_cfg, stage1_only=stage1_only
            )
        elif method == "ga":
            rows = self._run_ga(
                context,
                proxy,
                evaluator,
                run_dir,
                search_cfg,
                stage1_only=stage1_only,
            )
        else:
            raise RuntimeError(f"unsupported_baseline_search_method:{method}")
        monitor.record_event(
            "proxy_counters",
            cache_hit_count=int(proxy.cache_hit_count),
            cache_miss_count=int(proxy.cache_miss_count),
            unique_candidate_count=int(proxy.unique_phenotype_count),
            gpu_batch_count=int(proxy.gpu_batch_count),
            scalar_evaluate_call_count=int(proxy.scalar_evaluate_call_count),
            batch_evaluate_call_count=int(proxy.batch_evaluate_call_count),
        )
        resource_summary = monitor.close()
        cost = _write_search_cost_summary(
            run_dir,
            method=method,
            proxy=proxy,
            resource_summary=resource_summary,
        )
        return {
            "run_dir": str(run_dir),
            "selected_gpu": context.physical_gpu_id,
            "evaluated": len(rows),
            "best": min(rows, key=lambda row: float(row.get("F2", float("inf")))) if rows else None,
            "search_cost_summary": cost,
        }


__all__ = ["HealLidarBaselineTwoStageSearch"]
