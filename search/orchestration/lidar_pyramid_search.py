"""Real lidar_pyramid two-stage GA orchestration."""

from __future__ import annotations

import csv
import json
import math
import random
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from ..cache.proxy_cache import ProxyCache
from ..candidate import CandidateGenotype, CandidatePhenotype
from ..canonicalization import canonicalize_candidate
from ..ga.engine import GAConfig, GeneticSearchEngine
from ..hashing import candidate_hash, canonical_json_hash, search_hash
from ..integration.calibration_provider import collect_or_load_fisher_statistics
from ..integration.lidar_pyramid_context import build_lidar_pyramid_context
from ..integration.runtime_environment import require_gpu_isolation
from ..proxy.bops_proxy import BOPSProxy
from ..proxy.fisher_proxy import FisherTaylorProxy
from ..proxy.normalization import NormalizationStats, build_normalization_stats
from ..proxy.objective import ProxyObjective, ProxyObjectiveConfig, bops_soft_penalty, bops_target_for_generation, bops_target_for_outer_round
from ..proxy.gpu_batch_proxy import TorchBatchedProxyScorer
from ..proxy.parameter_slice_resolver import build_unit_parameter_slices
from ..proxy.runtime_shape_profiler import profile_runtime_layer_shapes
from ..proxy.size_proxy import SizeProxy
from ..proxy.sqnr_proxy import SQNRProxy
from ..pruning_space.mask_repair import GroupedDomainSpec, RepairPolicy, dense_floor_repair, grouped_equal_count_floor_repair
from ..stage1.proxy_evaluator import Stage1ProxyEvaluator
from ..stage1.repair_selection import select_repaired_stage2_topk
from ..stage1.topk_selector import ProxyCandidateRecord, TopKConfig, select_stage1_topk
from ..stage2.lidar_pyramid_real_evaluator import LidarPyramidRealEvaluator
from ..stage2.objective import Stage2ObjectiveConfig
from ..stage2.repaired_topk_manifest import write_repaired_topk_manifest
from ..stage2.round_results import write_round_stage2_results
from .budget_final import run_budget_final_evaluation
from .generation_stage2 import deploy_generation_with_backfill, fixed_bops_admission


def _write_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


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


def _require_gpu_proxy_if_needed(*, proxy_cfg: dict[str, Any], search_cfg: dict[str, Any], actual_backend: str) -> None:
    requested = str(proxy_cfg.get("device", "cpu")).lower()
    initial = int(search_cfg.get("initial_population_size", search_cfg.get("population_size", 0)) or 0)
    if requested != "cpu" and initial >= 512 and actual_backend != "cuda_batched":
        raise RuntimeError("gpu_proxy_required_but_not_active")


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
        model_cfg = dict(self.config.get("model", {}))
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
            num_frames=int(stage2_cfg.get("num_frames", 5)),
            warmup_frames=int(stage2_cfg.get("warmup_frames", 10)),
            reset_after_warmup=bool(stage2_cfg.get("reset_after_warmup", False)),
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
        if not stage1_only:
            require_gpu_isolation(
                context.physical_gpu_id,
                report_path=run_dir / "gpu_preflight.json",
            )
        if baseline_only:
            real_evaluator = LidarPyramidRealEvaluator(
                context=context,
                run_dir=run_dir,
                num_frames=int(stage2_cfg.get("num_frames", 5)),
                warmup_frames=int(stage2_cfg.get("warmup_frames", 10)),
                latency_rounds=int(stage2_cfg.get("latency_rounds", stage2_cfg.get("rounds", 1))),
                target_bops_retention=stage2_cfg.get("target_bops_retention"),
                bops_tolerance=float(stage2_cfg.get("bops_tolerance", stage2_cfg.get("tolerance", 0.005))),
                stage2_config=Stage2ObjectiveConfig(
                    eta_map=float(stage2_cfg.get("eta_ap", stage2_cfg.get("eta_map", 1.0))),
                    eta_latency=float(stage2_cfg.get("eta_latency", 1.0)),
                    latency_metric=str(stage2_cfg.get("latency_metric", "forward_mean_ms")),
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
        unit_slices = raw_unit_slices if pruning_gene_type == "coupled_channel_keep_mask" else self._action_slices(context, raw_unit_slices)
        self._write_local_domains(context, unit_slices, run_dir)
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
        raw_objective = self._objective(context, unit_slices, fisher_stats, None, runtime_shapes.shapes)
        normalization = self._build_normalization(context, raw_objective, run_dir)
        objective = self._objective(context, unit_slices, fisher_stats, normalization, runtime_shapes.shapes)
        proxy_cache = ProxyCache(run_dir / "archives" / "proxy_archive.jsonl")
        proxy_device = _proxy_device_from_config(proxy_cfg, context)
        proxy_batch_size = int(proxy_cfg.get("batch_size", proxy_cfg.get("proxy_batch_size", 128)))
        batch_scorer = None
        proxy_backend = "scalar_cpu"
        if str(proxy_device).startswith("cuda"):
            batch_scorer = TorchBatchedProxyScorer.from_components(
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
                code_commit=context.code_commit,
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
                "proxy_gpu_ids": [context.physical_gpu_id] if str(proxy_device).startswith("cuda") else [],
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
                    "proxy_gpu_ids": [context.physical_gpu_id] if str(proxy_device).startswith("cuda") else [],
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
            target_bops_retention=stage2_cfg.get("target_bops_retention"),
            bops_tolerance=float(stage2_cfg.get("bops_tolerance", stage2_cfg.get("tolerance", 0.005))),
            stage2_config=Stage2ObjectiveConfig(
                eta_map=float(stage2_cfg.get("eta_ap", stage2_cfg.get("eta_map", 1.0))),
                eta_latency=float(stage2_cfg.get("eta_latency", 1.0)),
                latency_metric=str(stage2_cfg.get("latency_metric", "forward_mean_ms")),
                tau_ap=stage2_cfg.get("tau_ap"),
                max_map_drop=stage2_cfg.get("max_map_drop"),
            ),
        )
        baseline_cfg = dict(self.config.get("baselines", {}) or {})
        if baseline_cfg.get("build_before_search") and not stage2_only:
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
        rows = self._run_ga(context, proxy, real_evaluator, run_dir, search_cfg, stage1_only=stage1_only)
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

    def _objective(self, context: Any, unit_slices: dict[str, Any], fisher_stats: Any, normalization: Any | None, runtime_shapes: Any | None = None) -> ProxyObjective:
        proxy_cfg = dict(self.config.get("proxy", self.config.get("proxy_objective", {})))
        return ProxyObjective(
            fisher=FisherTaylorProxy(context.model, statistics=fisher_stats, unit_to_parameter_names=unit_slices),
            sqnr=SQNRProxy(context.model, unit_to_parameter_slices=unit_slices),
            size=SizeProxy(context.model, unit_to_parameter_slices=unit_slices),
            bops=BOPSProxy(context.model, unit_to_parameter_slices=unit_slices, runtime_shapes=runtime_shapes),
            normalization=normalization,
            config=ProxyObjectiveConfig(
                alpha_fisher=float(proxy_cfg.get("alpha_prune", proxy_cfg.get("alpha_fisher", 1.0))),
                beta_sqnr=float(proxy_cfg.get("beta_sqnr", 1.0)),
                gamma_size=float(proxy_cfg.get("gamma_size", 1.0)),
                delta_bops=float(proxy_cfg.get("delta_bops_penalty", proxy_cfg.get("delta_bops", 1.0))),
                size_threshold=proxy_cfg.get("size_threshold"),
                bops_threshold=proxy_cfg.get("bops_threshold", None),
                bops_constraint_mode=str(proxy_cfg.get("bops_constraint_mode", "weighted_penalty")),
                bops_penalty_formula=str(
                    dict(proxy_cfg.get("bops_soft_constraint", {}) or {}).get(
                        "formula",
                        proxy_cfg.get("bops_penalty_formula", "absolute_excess_squared"),
                    )
                ),
                lambda_bops=float(proxy_cfg.get("lambda_bops", 1.0)),
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
        from ..pruning_space.local_domains import build_local_pruning_domains

        units = list(getattr(context, "atomic_prune_units", []) or [])
        domains = build_local_pruning_domains(units)
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
            row["first_order_taylor_score"] = {
                str(getattr(unit, "stable_id", "")): float(getattr(unit, "normalized_score", 0.0))
                for unit in units
                if str(getattr(unit, "stable_id", "")) in set(domain.ordered_unit_ids)
            }
            row["second_order_fisher_score"] = {}
            row["protected_units"] = [
                str(getattr(unit, "stable_id", ""))
                for unit in units
                if bool(getattr(unit, "protected", False)) and str(getattr(unit, "stable_id", "")) in set(domain.ordered_unit_ids)
            ]
            payload.append(row)
        _write_json(run_dir / "local_pruning_domains.json", payload)

    def _repair_raw_keep_mask(self, context: Any, genotype: CandidateGenotype) -> tuple[CandidateGenotype | None, dict[str, Any]]:
        pruning_cfg = dict(self.config.get("pruning", {}) or {})
        if str(pruning_cfg.get("gene_type", pruning_cfg.get("search_variable", ""))) != "coupled_channel_keep_mask":
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
        soft_schedule = dict(proxy_cfg.get("bops_soft_constraint", {}) or {})
        if not soft_schedule:
            soft_schedule = dict(proxy_cfg.get("bops_target_schedule", {}) or {})
        per_generation_stage2 = bool(search_cfg.get("per_generation_stage2", False))
        fixed_bops_target = search_cfg.get("target_bops_retention")
        fixed_bops_tolerance = float(search_cfg.get("bops_tolerance", 0.005))
        global_seen_raw_hashes = self._load_seen_raw_hashes(run_dir)
        for round_index in range(outer_rounds):
            round_dir = run_dir / f"round_{round_index:03d}"
            round_dir.mkdir(parents=True, exist_ok=True)
            topk_stage2 = int(search_cfg.get("topk_stage2", search_cfg.get("topk_real", 1)))
            if self.resume is not None and not per_generation_stage2 and not stage1_only and self._round_stage2_complete(round_dir, topk_stage2):
                round_results = json.loads((round_dir / "stage2_top5_results.json").read_text(encoding="utf-8"))
                evaluated_rows.extend(round_results.get("candidates", []) or [])
                previous_elite = self._round_topk_as_genotypes(round_dir, context)
                previous_best = previous_elite[0] if previous_elite else previous_best
                continue
            round_bops_target = (
                float(fixed_bops_target)
                if fixed_bops_target is not None
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
                )
                if getattr(proxy, "batch_scorer", None) is not None:
                    proxy.batch_scorer.config = proxy.objective.config
            objective_config = getattr(proxy_objective, "config", ProxyObjectiveConfig())
            normalization = getattr(proxy_objective, "normalization", None)
            objective_manifest = {
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
                "objective": "alpha*R_Fisher + beta*L_SQNR + gamma*R_Size_vs_FP32 + delta*P_BOPS",
            }
            _write_json(round_dir / "stage1_objective_config.json", objective_manifest)
            objective_hash = canonical_json_hash(objective_manifest)
            proxy.cache_key_fn = lambda phenotype, _space, _objective_hash=objective_hash: search_hash(
                phenotype,
                trace_hash=context.search_space.trace_snapshot_hash,
                proxy_version=f"real-fisher-sqnr-size-bops-v2:{_objective_hash}",
                calibration_statistics_version=f"{getattr(getattr(proxy, 'objective', None), 'fisher_version', '')}:{objective_hash}",
                code_commit=context.code_commit,
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
                    if fixed_bops_target is not None:
                        admission = fixed_bops_admission(
                            bops_value,
                            target=float(target),
                            tolerance=fixed_bops_tolerance,
                        )
                        violation = float(admission["violation"])
                        p_bops = violation * violation
                        metrics["BOPS_legal_interval"] = admission["legal_interval"]
                    else:
                        violation, p_bops = bops_soft_penalty(
                            bops_value,
                            float(target),
                            formula=getattr(getattr(proxy, "objective", None), "config", ProxyObjectiveConfig()).bops_penalty_formula,
                        )
                    metrics["BOPS_target"] = float(target)
                    metrics["bops_violation"] = float(violation)
                    metrics["P_bops"] = float(p_bops)
                    metrics["bops_feasible"] = violation <= 0.0
                    if (
                        fixed_bops_target is not None
                        or str(proxy_cfg.get("bops_constraint_mode", "weighted_penalty")) == "feasibility_first"
                    ) and violation > 0:
                        metrics["F1"] = 1.0e6 + violation * 1.0e3 + float(metrics.get("proxy_score_raw", metrics.get("F1", 0.0)))
                return metrics

            def evaluate_genotype(genotype: CandidateGenotype, generation: int) -> dict[str, Any]:
                metrics = proxy.evaluate(genotype, generation=generation, outer_round=round_index)
                return annotate_metrics(genotype, metrics, generation)

            def evaluate_genotypes_batch(genotypes: list[CandidateGenotype], generation: int) -> Any:
                batch = proxy.evaluate_batch(genotypes, generation=generation, outer_round=round_index)
                batch.metrics = [annotate_metrics(genotype, metrics, generation) for genotype, metrics in zip(genotypes, batch.metrics)]
                return batch

            generation_reports: list[dict[str, Any]] = []

            def on_generation(
                generation: int,
                generation_scored: list[tuple[CandidateGenotype, float, dict[str, Any]]],
            ) -> None:
                self._write_generation(
                    round_dir / f"generation_{generation + 1:03d}_stage1.csv",
                    generation_scored,
                    context,
                )
                if not per_generation_stage2 or stage1_only:
                    return
                eligible = [
                    row
                    for row in generation_scored
                    if math.isfinite(float(row[1]))
                    and bool(row[2].get("bops_feasible", fixed_bops_target is None))
                ]

                def repair_candidate(genotype: CandidateGenotype) -> tuple[CandidateGenotype | None, dict[str, Any]]:
                    return self._repair_raw_keep_mask(context, genotype)

                def rescore_batch(phenotypes: list[CandidatePhenotype]) -> list[dict[str, Any]]:
                    batch = proxy.evaluate_batch(
                        phenotypes,
                        generation=generation,
                        outer_round=round_index,
                    )
                    return [
                        annotate_metrics(CandidateGenotype({}, {}), row, generation)
                        for row in batch.metrics
                    ]

                ranked_records, repair_report = select_repaired_stage2_topk(
                    eligible,
                    space=context.search_space,
                    repair_fn=repair_candidate,
                    rescore_fn=lambda phenotype: rescore_batch([phenotype])[0],
                    batch_rescore_fn=rescore_batch,
                    topk=max(topk_stage2, len(eligible)),
                    repair_pool_size=max(topk_stage2, len(eligible)),
                )
                genuine_records = [
                    record
                    for record in ranked_records
                    if record.phenotype.pruned_unit_ids
                    or any(
                        precision == "INT8"
                        for precision in record.phenotype.realized_precision_profile.values()
                    )
                ]
                repair_report.update(
                    {
                        "generation": generation + 1,
                        "stage1_budget_eligible_count": len(eligible),
                        "genuinely_compressed_count": len(genuine_records),
                        "not_genuinely_compressed_count": len(ranked_records) - len(genuine_records),
                    }
                )
                _write_json(
                    round_dir / f"generation_{generation + 1:03d}_repair_report.json",
                    repair_report,
                )

                def deploy(record: ProxyCandidateRecord, candidate_dir: Path) -> dict[str, Any]:
                    _write_json(candidate_dir / "genotype.json", record.genotype.to_dict())
                    _write_json(candidate_dir / "repaired_genotype.json", record.genotype.to_dict())
                    _write_json(candidate_dir / "phenotype.json", record.phenotype.to_dict())
                    result = real_evaluator.evaluate_candidate(
                        record.phenotype,
                        output_dir=candidate_dir,
                        candidate_hash=record.candidate_hash,
                    )
                    evaluated_rows.append(
                        {
                            "generation": generation + 1,
                            "candidate_hash": record.candidate_hash,
                            "F1": record.F1,
                            **result,
                        }
                    )
                    return result

                generation_report = deploy_generation_with_backfill(
                    genuine_records,
                    generation_index=generation,
                    output_dir=round_dir,
                    deploy_fn=deploy,
                    topk=topk_stage2,
                )
                generation_reports.append(generation_report)

            scored = ga.run(
                evaluate_genotype if proxy_backend == "scalar_cpu" else None,
                batch_evaluator=evaluate_genotypes_batch if proxy_backend != "scalar_cpu" else None,
                previous_elite=previous_elite,
                previous_best=previous_best,
                seen_candidate_keys=global_seen_raw_hashes,
                candidate_key_fn=lambda genotype: self._raw_genotype_hash(genotype, context),
                generation_callback=on_generation,
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
            if not per_generation_stage2:
                for generation in range(int(search_cfg.get("generations_per_round", 5))):
                    self._write_generation(
                        round_dir / f"generation_{generation:03d}.csv",
                        [
                            row
                            for row in scored
                            if int(row[2].get("generation", -1)) == generation
                        ],
                        context,
                    )
            self._write_stage1(round_dir / "stage1_scores.csv", records)
            if per_generation_stage2:
                previous_elite = [record.genotype for record in records[: max(1, min(5, len(records)))]]
                previous_best = previous_elite[0] if previous_elite else None
                generation_winners = [report["winner"] for report in generation_reports]
                _write_json(
                    round_dir / "generation_winners.json",
                    generation_winners,
                )
                budget_final_report = None
                budget_final_cfg = dict(self.config.get("budget_final", {}) or {})
                if budget_final_cfg and not stage1_only:
                    stage2_policy = dict(
                        self.config.get("stage2")
                        or self.config.get("stage2_smoke")
                        or self.config.get("evaluation", {})
                    )
                    budget_final_report = run_budget_final_evaluation(
                        context=context,
                        run_dir=run_dir,
                        generation_winners=generation_winners,
                        config={**stage2_policy, **budget_final_cfg},
                        budget=float(round_bops_target),
                    )
                _write_json(
                    round_dir / "round_summary.json",
                    {
                        "best_F1": records[0].F1 if records else None,
                        "generation_count": len(generation_reports),
                        "generation_winner_count": len(generation_reports),
                        "evaluated": len(evaluated_rows),
                        "budget_final_status": (
                            budget_final_report.get("status")
                            if budget_final_report is not None
                            else "not_requested"
                        ),
                        "budget_winner": (
                            budget_final_report.get("winner")
                            if budget_final_report is not None
                            else None
                        ),
                    },
                )
                continue
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

                repaired_records, repair_report = select_repaired_stage2_topk(
                    scored,
                    space=context.search_space,
                    repair_fn=repair_candidate,
                    rescore_fn=rescore_one,
                    batch_rescore_fn=rescore_batch,
                    topk=topk_stage2,
                    repair_pool_size=int(search_cfg.get("repair_pool_size", max(50, topk_stage2 * 10))),
                )
                selected = [type("Selection", (), {"role": "repaired", "record": record}) for record in repaired_records]
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
            if not stage1_only:
                for item in selected:
                    record = item.record
                    candidate_dir = round_dir / "stage2" / record.candidate_hash
                    _write_json(candidate_dir / "genotype.json", record.genotype.to_dict())
                    _write_json(candidate_dir / "repaired_genotype.json", record.genotype.to_dict())
                    result = real_evaluator.evaluate_candidate(record.phenotype, output_dir=candidate_dir, candidate_hash=record.candidate_hash)
                    evaluated_rows.append({"candidate_hash": record.candidate_hash, "F1": record.F1, **result})
                write_round_stage2_results(run_dir, round_index=round_index)
            previous_elite = [record.genotype for record in records[: max(1, min(5, len(records)))]]
            previous_best = previous_elite[0] if previous_elite else None
            _write_json(round_dir / "round_summary.json", {"best_F1": records[0].F1 if records else None, "selected": len(selected), "evaluated": len(evaluated_rows)})
            if records:
                _write_json(round_dir / "best_candidate.json", {"candidate_hash": records[0].candidate_hash, "F1": records[0].F1, "phenotype": records[0].phenotype.to_dict()})
            if evaluated_rows:
                round_rows = [row for row in evaluated_rows if row.get("candidate_hash") in {item.record.candidate_hash for item in selected}]
                if round_rows:
                    winner = min(round_rows, key=lambda row: float(row.get("F2", float("inf"))))
                    _write_json(round_dir / "round_best_candidate.json", winner)
                    _write_json(round_dir / "round_best_F1_F2.json", {"candidate_hash": winner.get("candidate_hash"), "F1": winner.get("F1"), "F2": winner.get("F2")})
        self._write_global(run_dir, evaluated_rows)
        return evaluated_rows

    @staticmethod
    def _raw_genotype_hash(genotype: CandidateGenotype, context: Any) -> str:
        return canonical_json_hash(
            {
                "pruning_genes": genotype.pruning_genes,
                "precision_genes": genotype.precision_genes,
                "trace_hash": context.search_space.trace_snapshot_hash,
                "search_space_version": "coupled-mask-fp32-bops-v1",
                "code_commit": str(
                    getattr(context, "code_commit", context.search_space.code_commit)
                ),
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
            pruned = set(phenotype.pruned_unit_ids)
            metadata = dict(phenotype.metadata or {})
            requested_groups = dict(metadata.get("requested_group_profile") or metadata.get("stage1_legalized_group_profile") or {})
            genotypes.append(
                CandidateGenotype(
                    pruning_genes={unit_id: (0 if unit_id in pruned else 1) for unit_id in context.search_space.pruning_unit_ids},
                    precision_genes={
                        group_id: requested_groups.get(group_id, context.search_space.default_precision)
                        for group_id in context.search_space.precision_gene_ids
                    },
                )
            )
        return genotypes

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
            writer = csv.DictWriter(handle, fieldnames=["candidate_hash", "F1", "L_fisher", "L_sqnr", "R_size", "R_size_vs_fp32", "R_size_vs_fp16_deploy", "R_bops", "R_bops_vs_fp32", "R_bops_vs_fp16_deploy", "BOPS_target", "P_bops", "bops_feasible", "bops_violation", "int8_macs_ratio", "grouped_action_count", "pruned_units", "int8_layers", "cache_hit"])
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
                        "R_bops": metrics.get("R_bops"),
                        "R_bops_vs_fp32": metrics.get("R_bops_vs_fp32"),
                        "R_bops_vs_fp16_deploy": metrics.get("R_bops_vs_fp16_deploy"),
                        "BOPS_target": metrics.get("BOPS_target"),
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
            writer = csv.DictWriter(handle, fieldnames=["candidate_hash", "F1", "L_fisher", "L_sqnr", "R_size", "R_size_vs_fp32", "R_size_vs_fp16_deploy", "R_bops", "R_bops_vs_fp32", "R_bops_vs_fp16_deploy", "BOPS_target", "P_bops", "bops_feasible", "bops_violation", "int8_macs_ratio", "grouped_action_count", "pruned_units", "int8_layers"])
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
                        "R_bops": record.metrics.get("R_bops"),
                        "R_bops_vs_fp32": record.metrics.get("R_bops_vs_fp32"),
                        "R_bops_vs_fp16_deploy": record.metrics.get("R_bops_vs_fp16_deploy"),
                        "BOPS_target": record.metrics.get("BOPS_target"),
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
