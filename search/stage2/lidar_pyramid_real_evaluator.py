"""Real lidar_pyramid Stage-2 evaluator."""

from __future__ import annotations

import csv
import json
import shutil
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch

from ..adapters.pruning_adapter import FormalPruningAdapter
from ..cache.artifact_cache import ArtifactCache
from ..cache.real_eval_cache import RealEvalCache
from ..candidate import CandidatePhenotype
from ..hashing import canonical_json_hash, deployment_hash, eval_hash, physical_hash
from ..integration.calibration_provider import collect_or_load_qdq_calibration_scales
from ..integration.evaluation_provider import evaluate_engine_modelopt
from ..integration.lidar_pyramid_context import LidarPyramidSearchContext
from ..integration.trt_compatible_export import build_search_trt_compatible_export_module, make_pointpillar_domain_compatible
from ..pruning_space.action_codec import selected_actions_from_genes
from ..pruning_space.grouped_bundle_adapter import request_from_pruning_actions
from ..baselines.original_engines import make_baseline_trt_build_config, validate_baseline_layer_precisions
from .candidate_artifacts import write_candidate_summary_artifacts
from .objective import Stage2ObjectiveConfig, compute_stage2_score
from .physical_validation import validate_repaired_physical_plan
from .mixed_precision_export import summarize_qdq_realization
from .trt_modelopt import build_engine_modelopt


def _plain(value: Any) -> Any:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _plain(value.to_dict())
    if is_dataclass(value):
        return _plain(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_plain(item) for item in value]
    return value


def _write_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(_plain(payload), indent=2, sort_keys=True), encoding="utf-8")


def _file_hash(path: str | Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _shape_profiles(fixed_k: int = 29696) -> dict[str, dict[str, tuple[int, ...]]]:
    return {
        "pairwise_t_matrix": {"min": (1, 1, 1, 4, 4), "opt": (1, 2, 2, 4, 4), "max": (1, 2, 2, 4, 4)},
        "valid_voxel_mask": {"min": (fixed_k,), "opt": (fixed_k,), "max": (fixed_k,)},
        "voxel_coords": {"min": (fixed_k, 4), "opt": (fixed_k, 4), "max": (fixed_k, 4)},
        "voxel_features": {"min": (fixed_k, 32, 4), "opt": (fixed_k, 32, 4), "max": (fixed_k, 32, 4)},
        "voxel_num_points": {"min": (fixed_k,), "opt": (fixed_k,), "max": (fixed_k,)},
    }


def _param_count(model: torch.nn.Module) -> int:
    return sum(int(param.numel()) for param in model.parameters())


def _physical_selection_key(phenotype: CandidatePhenotype, checkpoint_hash: str) -> str:
    metadata = dict(phenotype.metadata or {})
    return canonical_json_hash(
        {
            "checkpoint_hash": checkpoint_hash,
            "pruned_unit_ids": sorted(phenotype.pruned_unit_ids),
            "resolved_prune_unit_ids": metadata.get("resolved_prune_unit_ids", []),
            "resolved_prune_indices": metadata.get("resolved_prune_indices", {}),
            "resolved_prune_indices_by_scope": metadata.get("resolved_prune_indices_by_scope", {}),
            "group_keep_map_by_scope": metadata.get("group_keep_map_by_scope", {}),
            "group_prune_map_by_scope": metadata.get("group_prune_map_by_scope", {}),
            "pruning_policy_version": phenotype.pruning_policy_version,
        }
    )


def _tensorrt_cache_identity(tensorrt: Any) -> dict[str, Any]:
    payload = dict(tensorrt.to_dict() if hasattr(tensorrt, "to_dict") else tensorrt)
    payload.pop("env_hash", None)
    return payload


def _as_dict(value: Any) -> Any:
    return value.to_dict() if hasattr(value, "to_dict") else value


def _width_value(row: dict[str, Any], *names: str) -> int | None:
    for name in names:
        value = row.get(name)
        if value is not None:
            return int(value)
    return None


def _write_physical_widths_csv(path: str | Path, *, snapshot_payload: dict[str, Any], plan_payload: dict[str, Any]) -> None:
    entries_by_module: dict[str, dict[str, dict[str, Any]]] = {}
    for entry in plan_payload.get("entries", []) or []:
        module_path = str(entry.get("module_path", ""))
        axis = str(entry.get("axis", ""))
        if module_path and axis:
            entries_by_module.setdefault(module_path, {})[axis] = dict(entry)
    fields = [
        "module path",
        "op type",
        "original C_in",
        "original C_out",
        "pruned C_in",
        "pruned C_out",
        "groups",
        "channels per group before",
        "channels per group after",
        "alignment status",
    ]
    rows: list[dict[str, Any]] = []
    allowed_group_widths = {4, 8, 16, 32, 64, 128, 256, 512}
    for module in snapshot_payload.get("modules", []) or []:
        module_path = str(module.get("canonical_module_name", ""))
        op_type = str(module.get("module_type", ""))
        groups = int(module.get("groups") or 1)
        original_in = _width_value(module, "in_channels", "in_features", "num_features")
        original_out = _width_value(module, "out_channels", "out_features", "num_features")
        pruned_in = original_in
        pruned_out = original_out
        plan_axes = entries_by_module.get(module_path, {})
        in_entry = plan_axes.get("in")
        out_entry = plan_axes.get("out") or plan_axes.get("channel")
        if in_entry is not None:
            original_in = int(in_entry.get("original_axis_size", original_in or 0))
            pruned_in = len(in_entry.get("keep_indices", []) or [])
        if out_entry is not None:
            original_out = int(out_entry.get("original_axis_size", original_out or 0))
            pruned_out = len(out_entry.get("keep_indices", []) or [])
        before = ""
        after = ""
        status = "not_channel_pruned"
        if op_type in {"Conv2d", "ConvTranspose2d"}:
            if groups > 1 and original_out is not None and pruned_out is not None and original_out % groups == 0 and pruned_out % groups == 0:
                before_value = original_out // groups
                after_value = pruned_out // groups
                before = before_value
                after = after_value
                if after_value in allowed_group_widths:
                    status = "grouped_width_safe"
                else:
                    status = "grouped_width_not_in_safe_set"
            elif pruned_out is not None:
                status = "dense_width_aligned" if pruned_out % 4 == 0 else "dense_width_not_multiple_of_4"
        elif op_type in {"Linear", "BatchNorm1d", "BatchNorm2d"}:
            status = "not_conv_alignment_target"
        rows.append(
            {
                "module path": module_path,
                "op type": op_type,
                "original C_in": "" if original_in is None else original_in,
                "original C_out": "" if original_out is None else original_out,
                "pruned C_in": "" if pruned_in is None else pruned_in,
                "pruned C_out": "" if pruned_out is None else pruned_out,
                "groups": groups,
                "channels per group before": before,
                "channels per group after": after,
                "alignment status": status,
            }
        )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_physical_artifact_files(
    *,
    output_dir: str | Path,
    request: Any,
    plan: Any,
    ledger: Any,
    snapshot: Any,
    validation: Any,
    model: torch.nn.Module,
    checkpoint_hash: str,
    physical_hash_value: str,
    parameter_count_base: int,
    parameter_count_pruned: int,
    plan_validation: dict[str, Any],
) -> None:
    destination = Path(output_dir)
    request_payload = _as_dict(request)
    plan_payload = _as_dict(plan)
    ledger_payload = _as_dict(ledger)
    snapshot_payload = _as_dict(snapshot)
    validation_payload = _as_dict(validation)
    _write_json(destination / "pruning_request.json", request_payload)
    _write_json(destination / "sampling_pruning_request.json", request_payload)
    _write_json(destination / "physical_plan_validation.json", plan_validation)
    _write_json(destination / "physical_plan.json", plan_payload)
    _write_json(destination / "physical_pruning_plan.json", plan_payload)
    _write_json(destination / "legalized_plan.json", plan_payload)
    _write_json(destination / "materialization_ledger.json", ledger_payload)
    _write_json(destination / "materialization_report.json", ledger_payload)
    _write_json(destination / "physical_snapshot.json", snapshot_payload)
    _write_json(destination / "physical_structure_snapshot.json", snapshot_payload)
    _write_json(destination / "physical_validation.json", validation_payload)
    _write_physical_widths_csv(destination / "physical_widths.csv", snapshot_payload=snapshot_payload, plan_payload=plan_payload)
    checkpoint_payload = {"model": model.state_dict(), "checkpoint_hash": checkpoint_hash}
    torch.save(checkpoint_payload, destination / "pruned_state_dict.pth")
    torch.save(checkpoint_payload, destination / "pruned_checkpoint.pth")
    _write_json(
        destination / "physical_hash.json",
        {
            "physical_hash": physical_hash_value,
            "parameter_count_base": int(parameter_count_base),
            "parameter_count_pruned": int(parameter_count_pruned),
            "parameter_reduction": 1.0 - (int(parameter_count_pruned) / max(int(parameter_count_base), 1)),
        },
    )


class LidarPyramidRealEvaluator:
    """Evaluate one phenotype through the formal pruning and Q/DQ deployment chain."""

    def __init__(
        self,
        *,
        context: LidarPyramidSearchContext,
        run_dir: str | Path,
        num_frames: int,
        warmup_frames: int,
        latency_rounds: int,
        stage2_config: Stage2ObjectiveConfig | None = None,
        artifact_cache: ArtifactCache | None = None,
        real_cache: RealEvalCache | None = None,
    ) -> None:
        self.context = context
        self.run_dir = Path(run_dir)
        self.num_frames = int(num_frames)
        self.warmup_frames = int(warmup_frames)
        self.latency_rounds = int(latency_rounds)
        self.objective_config = stage2_config or Stage2ObjectiveConfig()
        archives = self.run_dir / "archives"
        self.artifacts = artifact_cache or ArtifactCache(archives / "artifact_index.jsonl")
        self.real_cache = real_cache or RealEvalCache(archives / "real_eval_archive.jsonl")
        self.pruning = FormalPruningAdapter()
        self._baseline: dict[str, Any] | None = None
        self._physical_memory: dict[str, dict[str, Any]] = {}

    def evaluate_baseline(self) -> dict[str, Any]:
        if self._baseline is not None:
            return dict(self._baseline)
        key = canonical_json_hash(
            {
                "kind": "baseline",
                "checkpoint_hash": self.context.checkpoint_hash,
                "eval_manifest_hash": self.context.eval_manifest_hash,
                "num_frames": self.num_frames,
                "warmup_frames": self.warmup_frames,
                "latency_rounds": self.latency_rounds,
                "gpu": self.context.physical_gpu_id,
                "tensorrt": _tensorrt_cache_identity(self.context.tensorrt),
            }
        )
        cached = self.real_cache.get(key)
        if cached is not None:
            cached["cache_hit"] = True
            self._baseline = cached
            return dict(cached)
        baseline_dir = self.run_dir / "baseline"
        phenotype = self._default_precision_phenotype([])
        raw = self._deploy_and_evaluate(
            phenotype=phenotype,
            output_dir=baseline_dir / "baseline_engine",
            candidate_label="baseline",
            pruned_unit_ids=[],
        )
        if raw.get("status") != "ok":
            raise RuntimeError(f"baseline_evaluation_failed:{raw.get('failure_reason', raw.get('status'))}")
        baseline_eval = dict(raw["evaluation"])
        baseline_eval["status"] = "ok"
        baseline_eval["cache_key"] = key
        _write_json(baseline_dir / "baseline_eval.json", baseline_eval)
        self._copy_latency(raw["evaluation"], baseline_dir / "baseline_latency.csv")
        self.real_cache.put(key, baseline_eval)
        self._baseline = baseline_eval
        return dict(baseline_eval)

    def evaluate_original_baseline(self, baseline_precision: str, *, full_validation: bool = False) -> dict[str, Any]:
        kind = str(baseline_precision).lower()
        baseline_dir = self.run_dir / "baselines" / f"original_{kind}"
        baseline_dir.mkdir(parents=True, exist_ok=True)
        key = canonical_json_hash(
            {
                "kind": "original_precision_baseline",
                "precision": kind,
                "checkpoint_hash": self.context.checkpoint_hash,
                "eval_manifest_hash": self.context.eval_manifest_hash,
                "num_frames": self.num_frames,
                "warmup_frames": self.warmup_frames,
                "latency_rounds": self.latency_rounds,
                "full_validation": bool(full_validation),
                "gpu": self.context.physical_gpu_id,
                "tensorrt": _tensorrt_cache_identity(self.context.tensorrt),
            }
        )
        cached = self.real_cache.get(key)
        if cached is not None:
            cached["cache_hit"] = True
            _write_json(baseline_dir / "baseline_cache_hit.json", {"cache_key": key, "engine_hash": cached.get("engine_hash", "")})
            return dict(cached)
        existing = self._load_existing_original_baseline(baseline_dir, kind, key)
        if existing is not None:
            self.real_cache.put(key, existing)
            _write_json(baseline_dir / "baseline_cache_hit.json", {"cache_key": key, "engine_hash": existing.get("engine_hash", ""), "cache_source": "existing_baseline_eval"})
            return existing
        phenotype = self._baseline_precision_phenotype(kind)
        raw = self._deploy_and_evaluate(
            phenotype=phenotype,
            output_dir=baseline_dir,
            candidate_label=f"original_{kind}",
            pruned_unit_ids=[],
            baseline_precision=kind,
        )
        if raw.get("status") != "ok":
            result = {
                "status": str(raw.get("status", "baseline_failed")),
                "failure_reason": str(raw.get("failure_reason", "")),
                "baseline_precision": kind,
                "cache_key": key,
            }
        else:
            result = {
                **raw["evaluation"],
                "baseline_precision": kind,
                "cache_key": key,
                "deployment_hash": raw.get("deployment_hash", ""),
                "eval_hash": raw.get("eval_hash", ""),
                "physical_hash": raw.get("physical_hash", ""),
                "engine_hash": raw.get("engine_hash", ""),
                "engine_path": raw.get("engine_path", ""),
                "precision_validation": raw.get("baseline_precision_validation", {}),
                "qdq_realization_summary": raw.get("qdq_realization_summary", {}),
                "status": "ok",
            }
        _write_json(baseline_dir / "baseline_eval.json", result)
        self.real_cache.put(key, result)
        return result

    def evaluate_original_baselines(self, precisions: list[str] | tuple[str, ...]) -> dict[str, Any]:
        rows = {str(precision): self.evaluate_original_baseline(str(precision), full_validation=True) for precision in precisions}
        table_dir = self.run_dir / "baselines"
        _write_json(table_dir / "full_validation_baselines.json", rows)
        self._write_baseline_csv(table_dir / "full_validation_baselines.csv", rows)
        self._write_baseline_markdown(table_dir / "full_validation_baselines.md", rows)
        return rows

    def evaluate_candidate(self, phenotype: CandidatePhenotype, *, output_dir: str | Path, candidate_hash: str) -> dict[str, Any]:
        baseline = self._stage2_reference_baseline()
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        cache_key = canonical_json_hash(
            {
                "candidate_hash": candidate_hash,
                "eval_manifest_hash": self.context.eval_manifest_hash,
                "num_frames": self.num_frames,
                "warmup_frames": self.warmup_frames,
                "latency_rounds": self.latency_rounds,
                "stage2_reference_policy": "strict_fp32_ap_strict_fp16_latency_v1",
                "objective_config": asdict(self.objective_config),
                "gpu": self.context.physical_gpu_id,
                "tensorrt": _tensorrt_cache_identity(self.context.tensorrt),
            }
        )
        cached = self.real_cache.get(cache_key)
        if cached is not None:
            cached["cache_hit"] = True
            if not (destination / "stage2_score.json").exists():
                _write_json(destination / "stage2_score.json", cached)
            write_candidate_summary_artifacts(
                destination,
                candidate_hash=candidate_hash,
                phenotype=phenotype,
                stage2_score=cached,
                objective_config=self.objective_config,
                stage1_manifest_record=self._stage1_manifest_record(candidate_hash),
                overwrite=False,
            )
            return dict(cached)
        _write_json(destination / "phenotype.json", phenotype.to_dict())
        raw = self._load_existing_deployment_evaluation(destination)
        if raw is None:
            raw = self._deploy_and_evaluate(
                phenotype=phenotype,
                output_dir=destination,
                candidate_label=candidate_hash,
                pruned_unit_ids=phenotype.pruned_unit_ids,
            )
        if raw.get("status") == "ok":
            scored = compute_stage2_score(raw["evaluation"], baseline=baseline, config=self.objective_config)
            result = {
                **raw["evaluation"],
                **scored,
                "candidate_hash": candidate_hash,
                "cache_key": cache_key,
                "deployment_hash": raw.get("deployment_hash", ""),
                "eval_hash": raw.get("eval_hash", ""),
                "physical_hash": raw.get("physical_hash", ""),
                "engine_hash": raw.get("engine_hash", ""),
                "status": "ok",
                "artifact_dir": str(destination),
            }
        else:
            result = {
                "candidate_hash": candidate_hash,
                "cache_key": cache_key,
                "status": str(raw.get("status", "evaluation_failed")),
                "failure_reason": str(raw.get("failure_reason", raw.get("status", ""))),
                "F2": float("inf"),
                "artifact_dir": str(destination),
            }
        _write_json(destination / "stage2_score.json", result)
        write_candidate_summary_artifacts(
            destination,
            candidate_hash=candidate_hash,
            phenotype=phenotype,
            stage2_score=result,
            objective_config=self.objective_config,
            stage1_manifest_record=self._stage1_manifest_record(candidate_hash),
        )
        self.real_cache.put(cache_key, result)
        return result

    def _stage2_reference_baseline(self) -> dict[str, Any]:
        accuracy = self.evaluate_original_baseline("strict_fp32", full_validation=False)
        latency = self.evaluate_original_baseline("strict_fp16", full_validation=False)
        if str(accuracy.get("status", "")) != "ok":
            raise RuntimeError(f"strict_fp32_accuracy_baseline_failed:{accuracy.get('failure_reason', accuracy.get('status'))}")
        if str(latency.get("status", "")) != "ok":
            raise RuntimeError(f"strict_fp16_latency_baseline_failed:{latency.get('failure_reason', latency.get('status'))}")
        metric = self.objective_config.latency_metric
        combined = {
            "status": "ok",
            "mAP": float(accuracy.get("mAP", accuracy.get("map", 0.0)) or 0.0),
            metric: float(latency.get(metric, 0.0) or 0.0),
            "accuracy_reference": "original_strict_fp32",
            "latency_reference": "original_strict_fp16",
            "strict_fp32": accuracy,
            "strict_fp16": latency,
        }
        _write_json(self.run_dir / "stage2_reference_baseline.json", combined)
        return combined

    @staticmethod
    def _load_existing_deployment_evaluation(output_dir: Path) -> dict[str, Any] | None:
        evaluation_path = output_dir / "evaluation.json"
        manifest_path = output_dir / "deployment_manifest.json"
        engine_path = output_dir / "engine.plan"
        if not (evaluation_path.is_file() and manifest_path.is_file() and engine_path.is_file()):
            return None
        try:
            evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        if str(evaluation.get("status", "")) != "ok":
            return None
        return {
            "status": "ok",
            "evaluation": evaluation,
            "physical_hash": manifest.get("physical_hash", ""),
            "deployment_hash": manifest.get("deployment_hash", ""),
            "eval_hash": manifest.get("eval_hash", ""),
            "engine_hash": manifest.get("engine_hash", _file_hash(engine_path)),
            "engine_path": str(engine_path),
            "baseline_precision_validation": {},
            "qdq_realization_summary": json.loads((output_dir / "qdq_realization_summary.json").read_text(encoding="utf-8")) if (output_dir / "qdq_realization_summary.json").is_file() else {},
        }

    @staticmethod
    def _load_existing_original_baseline(baseline_dir: Path, kind: str, cache_key: str) -> dict[str, Any] | None:
        baseline_eval_path = baseline_dir / "baseline_eval.json"
        engine_path = baseline_dir / "engine.plan"
        evaluation_path = baseline_dir / "evaluation.json"
        if not (baseline_eval_path.is_file() and engine_path.is_file() and evaluation_path.is_file()):
            return None
        try:
            result = json.loads(baseline_eval_path.read_text(encoding="utf-8"))
            evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        if str(result.get("status", "")) != "ok" or str(evaluation.get("status", "")) != "ok":
            return None
        recorded_kind = str(result.get("baseline_precision", kind)).lower()
        if recorded_kind and recorded_kind != str(kind).lower():
            return None
        loaded = dict(result)
        loaded["cache_key"] = cache_key
        loaded["cache_hit"] = True
        loaded["cache_source"] = "existing_baseline_eval"
        loaded.setdefault("baseline_precision", str(kind).lower())
        loaded.setdefault("engine_path", str(engine_path))
        loaded.setdefault("engine_hash", _file_hash(engine_path))
        return loaded

    def _stage1_manifest_record(self, candidate_hash: str) -> dict[str, Any]:
        for path in sorted(self.run_dir.glob("round_*/repaired_top5_manifest.json")):
            try:
                manifest = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            for row in manifest.get("candidates", []) or []:
                if str(row.get("repaired_phenotype_hash")) == str(candidate_hash):
                    return dict(row)
        return {}

    def _default_precision_phenotype(self, pruned_unit_ids: list[str]) -> CandidatePhenotype:
        from ..candidate import PrecisionDecision
        from ..quantization_space.legalizer import legalize_group_precision_genes

        if self.context.search_space.quantization_groups:
            genes = {
                group.group_id: self.context.search_space.default_precision
                for group in self.context.search_space.quantization_groups
            }
            legalization = legalize_group_precision_genes(
                genes,
                self.context.search_space.quantization_groups,
                default_precision=self.context.search_space.default_precision,
            )
            return CandidatePhenotype(
                pruned_unit_ids=pruned_unit_ids,
                precision_profile=legalization.expand_to_module_profile(),
                metadata=legalization.to_dict(),
            )
        return CandidatePhenotype(
            pruned_unit_ids=pruned_unit_ids,
            precision_profile={
                layer: PrecisionDecision("FP16", "FP16", "")
                for layer in self.context.search_space.precision_layer_ids
            },
        )

    def _baseline_precision_phenotype(self, baseline_precision: str) -> CandidatePhenotype:
        from ..candidate import PrecisionDecision
        from ..quantization_space.legalizer import legalize_group_precision_genes

        kind = str(baseline_precision).lower()
        if not self.context.search_space.quantization_groups:
            precision = "FP32" if kind == "strict_fp32" else "FP16"
            if kind in {"maximal_legal_int8", "pure_strict_int8"}:
                precision = "INT8"
            return CandidatePhenotype(
                pruned_unit_ids=[],
                precision_profile={
                    layer: PrecisionDecision(precision, precision, "")
                    for layer in self.context.search_space.precision_layer_ids
                },
                metadata={"baseline_precision": kind},
            )
        requested: dict[str, str] = {}
        for group in self.context.search_space.quantization_groups:
            if kind == "strict_fp32":
                requested[group.group_id] = "FP32"
            elif kind == "strict_fp16":
                requested[group.group_id] = "FP16"
            elif kind == "pure_strict_int8":
                requested[group.group_id] = "INT8"
            else:
                requested[group.group_id] = "INT8" if "INT8" in group.allowed_precisions and not group.protected else "FP16"
        legalization = legalize_group_precision_genes(
            requested,
            self.context.search_space.quantization_groups,
            default_precision=self.context.search_space.default_precision,
        )
        return CandidatePhenotype(
            pruned_unit_ids=[],
            precision_profile=legalization.expand_to_module_profile(),
            metadata={**legalization.to_dict(), "baseline_precision": kind},
        )

    def _deploy_and_evaluate(
        self,
        *,
        phenotype: CandidatePhenotype,
        output_dir: Path,
        candidate_label: str,
        pruned_unit_ids: list[str],
        baseline_precision: str | None = None,
    ) -> dict[str, Any]:
        try:
            physical = self._materialize_physical(phenotype, output_dir)
            qdq = self._export_qdq(phenotype, physical, output_dir)
            trt = self._build_engine(qdq, physical, output_dir, baseline_precision=baseline_precision)
            if trt.get("status") != "ok":
                return {"status": trt.get("status", "engine_build_failed"), "failure_reason": trt.get("failure_reason", trt.get("status", ""))}
            evaluation = self._evaluate_engine(trt["engine_path"], output_dir)
            if evaluation.get("status") != "ok":
                return {"status": "evaluation_failed", "failure_reason": evaluation.get("failure_reason", evaluation.get("status", "")), "evaluation": evaluation}
            calibration_scale_hash = canonical_json_hash(qdq.get("calibration_scales", {}))
            deploy_hash = deployment_hash(
                physical_hash_value=physical["physical_hash"],
                realized_precision_profile=qdq["realized_precision_profile"],
                calibration_scale_hash=calibration_scale_hash,
                onnx_export_config_hash=self.context.search_space.onnx_export_config_hash,
                tensorrt_version=self.context.search_space.tensorrt_version,
                gpu_compute_capability=self.context.search_space.gpu_compute_capability,
                builder_flags=self.context.search_space.builder_flags,
                optimization_profiles=_shape_profiles(),
                plugin_hashes=self.context.search_space.plugin_hashes,
            )
            eval_key = eval_hash(
                deployment_hash_value=deploy_hash,
                validation_manifest_hash=self.context.eval_manifest_hash,
                evaluation_config_hash=canonical_json_hash({"num_frames": self.num_frames, "warmup": self.warmup_frames, "rounds": self.latency_rounds}),
                postprocess_config={"source": "HEAL dataset.post_process"},
                warmup=self.warmup_frames,
                rounds=self.latency_rounds,
                latency_metric_definition=self.objective_config.latency_metric,
            )
            _write_json(
                output_dir / "deployment_manifest.json",
                {
                    "candidate_label": candidate_label,
                    "pruned_unit_ids": pruned_unit_ids,
                    "physical_hash": physical["physical_hash"],
                    "deployment_hash": deploy_hash,
                    "eval_hash": eval_key,
                    "engine_hash": trt.get("engine_hash", ""),
                },
            )
            return {
                "status": "ok",
                "evaluation": evaluation,
                "physical_hash": physical["physical_hash"],
                "deployment_hash": deploy_hash,
                "eval_hash": eval_key,
                "engine_hash": trt.get("engine_hash", ""),
                "engine_path": trt.get("engine_path", ""),
                "baseline_precision_validation": trt.get("baseline_precision_validation", {}),
                "qdq_realization_summary": qdq.get("qdq_realization_summary", {}),
            }
        except Exception as exc:  # noqa: BLE001
            return {"status": "evaluation_failed", "failure_reason": f"{type(exc).__name__}: {exc}"}

    def _materialize_physical(self, phenotype: CandidatePhenotype, output_dir: Path) -> dict[str, Any]:
        selection_key = _physical_selection_key(phenotype, self.context.checkpoint_hash)
        if selection_key in self._physical_memory:
            result = dict(self._physical_memory[selection_key])
            result["cache_hit"] = True
            _write_json(output_dir / "physical_cache_hit.json", {"selection_key": selection_key, "physical_hash": result["physical_hash"]})
            _write_physical_artifact_files(
                output_dir=output_dir,
                request=result["request"],
                plan=result["plan"],
                ledger=result["ledger"],
                snapshot=result["snapshot"],
                validation=result["validation"],
                model=result["model"],
                checkpoint_hash=self.context.checkpoint_hash,
                physical_hash_value=result["physical_hash"],
                parameter_count_base=int(result["parameter_count_base"]),
                parameter_count_pruned=int(result["parameter_count_pruned"]),
                plan_validation=dict(result.get("plan_validation") or {"passed": True, "cache_hit": True}),
            )
            return result
        action_ids = set(getattr(self.context.pruning_action_catalog, "action_ids", []) or []) if getattr(self.context, "pruning_action_catalog", None) is not None else set()
        if action_ids and set(phenotype.pruned_unit_ids).issubset(action_ids):
            genes = {action_id: (0 if action_id in set(phenotype.pruned_unit_ids) else 1) for action_id in self.context.search_space.pruning_unit_ids}
            actions = selected_actions_from_genes(genes, self.context.pruning_action_catalog.actions)
            request = request_from_pruning_actions(actions)
        else:
            request = self.pruning.request_from_phenotype(phenotype, self.context.atomic_prune_units)
        _write_json(output_dir / "pruning_request.json", request.to_dict())
        _write_json(output_dir / "sampling_pruning_request.json", request.to_dict())
        plan = self.pruning.build_plan_fn(self.context.model, request)
        legal_plan = self.pruning.legalize_plan_fn(self.context.model, plan)
        plan_validation = validate_repaired_physical_plan(phenotype, request, legal_plan)
        _write_json(output_dir / "physical_plan_validation.json", plan_validation)
        if not bool(plan_validation.get("passed", False)):
            raise RuntimeError("repaired_physical_plan_mismatch")
        materialized_result = self.pruning.materialize_fn(self.context.model, legal_plan, in_place=False, build_snapshot=True)
        snapshot = materialized_result.snapshot if getattr(materialized_result, "snapshot", None) is not None else self.pruning.snapshot_fn(materialized_result.model)
        hashes = self.pruning.hash_fn(snapshot)
        validation = self.pruning.validate_fn(
            materialized_result.model,
            expected_snapshot=snapshot,
            grouped_config=None,
            fixed_output_contracts=None,
            example_inputs=(self.context.trace_example_inputs,),
        )
        materialized = {
            "model": materialized_result.model,
            "plan": legal_plan,
            "ledger": materialized_result.ledger,
            "snapshot": snapshot,
            "hashes": hashes,
            "validation": validation,
        }
        if hasattr(validation, "passed") and not bool(validation.passed):
            raise RuntimeError("prune_failed:physical_validation")
        physical_key = physical_hash(
            legal_physical_plan=materialized["plan"],
            physical_snapshot=materialized["snapshot"],
            checkpoint_hash=self.context.checkpoint_hash,
            pruning_policy_version=phenotype.pruning_policy_version,
        )
        result = {
            **materialized,
            "request": request,
            "plan_validation": plan_validation,
            "physical_hash": physical_key,
            "parameter_count_base": _param_count(self.context.model),
            "parameter_count_pruned": _param_count(materialized["model"]),
        }
        _write_physical_artifact_files(
            output_dir=output_dir,
            request=request,
            plan=materialized["plan"],
            ledger=materialized["ledger"],
            snapshot=materialized["snapshot"],
            validation=validation,
            model=materialized["model"],
            checkpoint_hash=self.context.checkpoint_hash,
            physical_hash_value=physical_key,
            parameter_count_base=int(result["parameter_count_base"]),
            parameter_count_pruned=int(result["parameter_count_pruned"]),
            plan_validation=plan_validation,
        )
        self.artifacts.put_physical(physical_key, {"artifact_dir": str(output_dir), "pruned_state_dict": str(output_dir / "pruned_state_dict.pth")})
        self._physical_memory[selection_key] = result
        return result

    def _export_qdq(self, phenotype: CandidatePhenotype, physical: dict[str, Any], output_dir: Path) -> dict[str, Any]:
        try:
            from quantization.api import build_canonical_precision_mapping, export_pruned_signal_maxk_onnx, insert_explicit_qdq, prepare_signal_maxk_inputs
            from quantization.config import CalibrationConfig, CanonicalNamingConfig, OnnxExportConfig, QDQConfig
            from quantization.types import PrecisionAssignment, PrecisionProfileResult
        except ImportError:
            from heal_compress.quantization.api import build_canonical_precision_mapping, export_pruned_signal_maxk_onnx, insert_explicit_qdq, prepare_signal_maxk_inputs
            from heal_compress.quantization.config import CalibrationConfig, CanonicalNamingConfig, OnnxExportConfig, QDQConfig
            from heal_compress.quantization.types import PrecisionAssignment, PrecisionProfileResult

        qcfg = OnnxExportConfig(fixed_k=29696, min_agents=1, opt_agents=2, max_agents=2)
        wrapper = build_search_trt_compatible_export_module(
            physical["model"],
            output_names=qcfg.output_names,
            fixed_k=qcfg.fixed_k,
            modality="m1",
        ).to(self.context.runtime_device).eval()
        inputs = prepare_signal_maxk_inputs(self.context.trace_example_inputs, config=qcfg, modality="m1")
        export = export_pruned_signal_maxk_onnx(
            wrapper,
            inputs,
            output_dir / "exported.onnx",
            physical["snapshot"],
            config=qcfg,
            naming_config=CanonicalNamingConfig(),
            report_path=output_dir / "onnx_export_report.json",
        )
        origin_map = export.origin_map
        if origin_map is None:
            raise RuntimeError("onnx_export_failed:no_origin_map")
        if not (output_dir / "pruned_fp32.onnx").exists():
            shutil.copyfile(output_dir / "exported.onnx", output_dir / "pruned_fp32.onnx")
        _write_json(output_dir / "origin_map.json", origin_map.to_dict())
        assignments = []
        requested_profile = {}
        for order, origin in enumerate(sorted(origin_map.entries, key=lambda row: (row.call_index, row.graph_index))):
            decision = phenotype.precision_profile.get(origin.module_path)
            requested = (decision.requested_precision if decision is not None else self.context.search_space.default_precision).lower()
            module_to_group = dict(phenotype.metadata.get("module_to_precision_group") or {})
            if not module_to_group and self.context.search_space.quantization_groups:
                module_to_group = {
                    module_path: group.group_id
                    for group in self.context.search_space.quantization_groups
                    for module_path in group.module_paths
                }
            group_id = str(module_to_group.get(origin.module_path, ""))
            if not group_id:
                if self.context.search_space.quantization_groups:
                    raise RuntimeError(f"missing_quantization_group_member_mapping:{origin.module_path}")
                group_id = f"pg::{origin.module_path}"
            requested_profile[origin.module_path] = requested.upper()
            assignments.append(
                PrecisionAssignment(
                    module_path=origin.module_path,
                    precision_group=group_id,
                    requested_precision=requested,
                    ordering=order,
                )
            )
        profile = PrecisionProfileResult(
            profile_id="search_candidate",
            assignments=assignments,
            requested_int8_count=sum(row.requested_precision == "int8" for row in assignments),
            requested_int8_ratio=sum(row.requested_precision == "int8" for row in assignments) / max(len(assignments), 1),
            policy_version=phenotype.precision_policy_version,
        )
        qdq_config = QDQConfig(
            allowed_precisions=("fp32", "fp16", "int8"),
            grouped_conv_int8_allowed_channels_per_group=(4, 8, 16, 32, 64, 128, 256, 512),
        )
        mapping = build_canonical_precision_mapping(origin_map, profile, config=qdq_config)
        realized_profile = {row.module_path: str(row.realized_request_precision).upper() for row in mapping.entries}
        realized_group_profile: dict[str, str] = {}
        requested_group_profile: dict[str, str] = {}
        for row in mapping.entries:
            requested_group_profile.setdefault(row.precision_group, str(row.requested_precision).upper())
            previous = realized_group_profile.get(row.precision_group)
            realized = str(row.realized_request_precision).upper()
            if previous is None:
                realized_group_profile[row.precision_group] = realized
            elif previous != realized:
                realized_group_profile[row.precision_group] = "FP16"
        fallback_report = {
            row.module_path: row.fallback_reason
            for row in mapping.entries
            if row.requested_precision != row.realized_request_precision or row.fallback_reason
        }
        _write_json(output_dir / "requested_precision_profile.json", requested_profile)
        _write_json(output_dir / "requested_quantization_groups.json", requested_group_profile)
        _write_json(output_dir / "stage1_legalized_quantization_groups.json", phenotype.metadata.get("stage1_legalized_group_profile", requested_group_profile))
        _write_json(output_dir / "legalized_precision_profile.json", phenotype.metadata.get("stage1_legalized_group_profile", requested_group_profile))
        _write_json(output_dir / "realized_precision_profile.json", realized_profile)
        _write_json(output_dir / "stage2_realized_precision_profile.json", realized_profile)
        _write_json(output_dir / "stage2_realized_quantization_groups.json", realized_group_profile)
        _write_json(output_dir / "precision_group_expansion.json", phenotype.metadata.get("precision_group_expansion", {}))
        _write_json(output_dir / "precision_fallback_report.json", fallback_report)
        _write_json(output_dir / "canonical_layer_map.json", mapping.to_dict())
        int8_modules = sorted(row.module_path for row in mapping.entries if row.realized_request_precision == "int8")
        scales: dict[str, Any] = {}
        if int8_modules:
            scales = collect_or_load_qdq_calibration_scales(
                model=physical["model"],
                adapter=self.context.model_bundle.adapter,
                model_config_path=self.context.model_config,
                module_paths=int8_modules,
                device=torch.device(self.context.runtime_device),
                cache_path=self.run_dir / "archives" / "calibration" / f"{physical['physical_hash']}_{canonical_json_hash(int8_modules)}.json",
                num_batches=self.context.quant_calibration_batches,
            )
        _write_json(output_dir / "calibration_manifest.json", {"module_paths": int8_modules, "batches": self.context.quant_calibration_batches})
        _write_json(output_dir / "calibration_scales.json", scales)
        qdq = insert_explicit_qdq(
            export.onnx_path,
            output_dir / "qdq.onnx",
            mapping,
            scales=scales,
            config=qdq_config,
            calibration_metadata={
                "calibration_manifest_hash": canonical_json_hash({"module_paths": int8_modules, "batches": self.context.quant_calibration_batches}),
                "calibration_config": CalibrationConfig(frame_count=max(1, self.context.quant_calibration_batches)).to_dict(),
            },
        )
        if not (output_dir / "pruned_qdq.onnx").exists():
            shutil.copyfile(output_dir / "qdq.onnx", output_dir / "pruned_qdq.onnx")
        compatibility = make_pointpillar_domain_compatible(output_dir / "qdq.onnx", output_dir / "qdq_trt_compatible.onnx")
        _write_json(output_dir / "onnx_domain_compatibility_report.json", compatibility)
        _write_json(output_dir / "qdq_report.json", qdq.to_dict())
        group_macs = {group.group_id: float(group.baseline_macs) for group in self.context.search_space.quantization_groups}
        total_group_macs = sum(group_macs.values()) or 1.0
        int8_macs_ratio = sum(
            group_macs.get(group_id, 0.0)
            for group_id, precision in realized_group_profile.items()
            if str(precision).upper() == "INT8"
        ) / total_group_macs
        qdq_summary = summarize_qdq_realization(
            requested_group_profile=requested_group_profile,
            realized_group_profile=realized_group_profile,
            realized_canonical_profile=realized_profile,
            qdq_report=qdq.to_dict(),
            int8_macs_ratio=int8_macs_ratio,
        )
        _write_json(output_dir / "qdq_realization_summary.json", qdq_summary)
        return {
            "export": export,
            "origin_map": origin_map,
            "precision_profile": profile,
            "precision_mapping": mapping,
            "realized_precision_profile": realized_profile,
            "realized_group_profile": realized_group_profile,
            "fallback_report": fallback_report,
            "calibration_scales": scales,
            "qdq_realization_summary": qdq_summary,
            "qdq": qdq,
            "qdq_onnx": str(output_dir / "qdq.onnx"),
            "trt_build_onnx": str(output_dir / "qdq_trt_compatible.onnx"),
        }

    def _build_engine(self, qdq: dict[str, Any], physical: dict[str, Any], output_dir: Path, *, baseline_precision: str | None = None) -> dict[str, Any]:
        try:
            from quantization.config import TensorRTBuildConfig
        except ImportError:
            from heal_compress.quantization.config import TensorRTBuildConfig
        if baseline_precision:
            build_config = make_baseline_trt_build_config(
                baseline_precision,
                trtexec_path=self.context.tensorrt.trtexec_path,
                plugin_path=self.context.tensorrt.plugin_path,
                shape_profiles=_shape_profiles(),
            )
        else:
            build_config = TensorRTBuildConfig(
                trtexec_path=self.context.tensorrt.trtexec_path,
                plugin_path=self.context.tensorrt.plugin_path,
                shape_profiles=_shape_profiles(),
                enable_fp16=True,
                enable_int8=any(row.realized_request_precision == "int8" for row in qdq["precision_mapping"].entries),
                no_tf32=True,
                precision_constraints="obey",
                skip_inference=True,
                export_layer_info=True,
            )
        engine_path = output_dir / "engine.plan"
        result = build_engine_modelopt(
            qdq_onnx=qdq.get("trt_build_onnx", qdq["qdq_onnx"]),
            engine_path=engine_path,
            precision_mapping=qdq["precision_mapping"],
            build_config=build_config,
            physical_snapshot=physical["snapshot"],
            output_dir=output_dir,
            tensorrt_root=self.context.tensorrt.tensorrt_root,
            conda_env=self.context.tensorrt.conda_env,
            gpu_id=self.context.physical_gpu_id,
        )
        if engine_path.is_file():
            result["engine_path"] = str(engine_path)
            result["engine_hash"] = _file_hash(engine_path)
        _write_json(output_dir / "engine_manifest.json", result)
        if "engine_structure_validation" in result:
            _write_json(output_dir / "engine_structure_validation.json", result["engine_structure_validation"])
        if "precision_realization_validation" in result:
            _write_json(output_dir / "precision_realization_validation.json", result["precision_realization_validation"])
        if baseline_precision and (output_dir / "engine_layer_info.json").is_file():
            baseline_report = validate_baseline_layer_precisions(baseline_precision, output_dir / "engine_layer_info.json")
            result["baseline_precision_validation"] = baseline_report
            _write_json(output_dir / "baseline_precision_validation.json", baseline_report)
            if not baseline_report.get("passed", False):
                result["status"] = baseline_report.get("status", "baseline_precision_validation_failed")
                result["failure_reason"] = ",".join(baseline_report.get("issues", []))
        return result

    def _evaluate_engine(self, engine_path: str | Path, output_dir: Path) -> dict[str, Any]:
        result = evaluate_engine_modelopt(
            engine_path=engine_path,
            checkpoint=self.context.checkpoint_path,
            model_config=self.context.model_config,
            heal_root="/home/lixingfeng/UniAD_examine/HEAL",
            device=self.context.runtime_device,
            output_dir=output_dir,
            tensorrt_root=self.context.tensorrt.tensorrt_root,
            plugin_path=self.context.tensorrt.plugin_path,
            num_frames=self.num_frames,
            warmup_frames=self.warmup_frames,
            fixed_k=29696,
            latency_rounds=self.latency_rounds,
            conda_env=self.context.tensorrt.conda_env,
        )
        _write_json(output_dir / "evaluation.json", result)
        self._copy_latency(result, output_dir / "latency.csv")
        return result

    @staticmethod
    def _copy_latency(evaluation: dict[str, Any], path: str | Path) -> None:
        rows = list(evaluation.get("latency_rows") or [])
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not rows:
            destination.write_text("", encoding="utf-8")
            return
        fields = sorted({key for row in rows for key in row})
        with destination.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _write_baseline_csv(path: str | Path, rows: dict[str, dict[str, Any]]) -> None:
        fields = [
            "precision baseline",
            "engine status",
            "INT8 group count",
            "INT8 MACs ratio",
            "Q/DQ node count",
            "AP@0.3",
            "AP@0.5",
            "AP@0.7",
            "mAP",
            "forward_mean_ms",
            "forward_p50_ms",
            "forward_p90_ms",
            "forward_p95_ms",
            "forward_p99_ms",
            "evaluated frames",
            "skipped frames",
            "engine hash",
        ]
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for name, row in rows.items():
                qdq = dict(row.get("qdq_realization_summary") or {})
                writer.writerow(
                    {
                        "precision baseline": name,
                        "engine status": row.get("status"),
                        "INT8 group count": qdq.get("realized_int8_group_count", 0),
                        "INT8 MACs ratio": qdq.get("realized_int8_macs_ratio", 0.0),
                        "Q/DQ node count": int(qdq.get("QuantizeLinear_count", 0) or 0) + int(qdq.get("DequantizeLinear_count", 0) or 0),
                        "AP@0.3": row.get("AP@0.3"),
                        "AP@0.5": row.get("AP@0.5"),
                        "AP@0.7": row.get("AP@0.7"),
                        "mAP": row.get("mAP"),
                        "forward_mean_ms": row.get("forward_mean_ms"),
                        "forward_p50_ms": row.get("forward_p50_ms"),
                        "forward_p90_ms": row.get("forward_p90_ms"),
                        "forward_p95_ms": row.get("forward_p95_ms"),
                        "forward_p99_ms": row.get("forward_p99_ms"),
                        "evaluated frames": row.get("num_evaluated_frames"),
                        "skipped frames": row.get("num_skipped_frames"),
                        "engine hash": row.get("engine_hash"),
                    }
                )

    @staticmethod
    def _write_baseline_markdown(path: str | Path, rows: dict[str, dict[str, Any]]) -> None:
        lines = [
            "| baseline | status | mAP | p50 ms | p90 ms | p95 ms | evaluated | skipped | engine hash |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
        for name, row in rows.items():
            lines.append(
                "| {name} | {status} | {map} | {p50} | {p90} | {p95} | {evaluated} | {skipped} | {hash} |".format(
                    name=name,
                    status=row.get("status", ""),
                    map=row.get("mAP", ""),
                    p50=row.get("forward_p50_ms", ""),
                    p90=row.get("forward_p90_ms", ""),
                    p95=row.get("forward_p95_ms", ""),
                    evaluated=row.get("num_evaluated_frames", ""),
                    skipped=row.get("num_skipped_frames", ""),
                    hash=row.get("engine_hash", ""),
                )
            )
        Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
