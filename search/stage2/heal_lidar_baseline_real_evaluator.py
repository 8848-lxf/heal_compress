"""Real TensorRT deployment/evaluation for HEAL LiDAR F-Cooper and DiscoNet."""

from __future__ import annotations

import hashlib
import json
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch.nn as nn

from quantization.config import TensorRTBuildConfig
from quantization.types import stable_json_hash
from search.model_family.deployment import build_physical_structure_snapshot_v2
from search.model_family.evaluation import evaluate_v2xvit_engine_modelopt
from search.model_family.export.heal_lidar_baselines import HealLidarBaselineExportPolicy
from search.model_family.heal_lidar_deployment import (
    HEAL_LIDAR_BASELINE_INPUT_NAMES,
    build_heal_lidar_baseline_precision_mapping,
    export_heal_lidar_baseline_fixed_k_onnx,
    insert_heal_lidar_baseline_explicit_qdq,
    validate_heal_lidar_precision_realization,
)
from search.stage2.objective import Stage2ObjectiveConfig, compute_stage2_score
from search.stage2.trt_modelopt import build_engine_modelopt


@dataclass(frozen=True)
class HealLidarBaselineEvaluationConfig:
    family_id: str
    model_name: str
    model_config_path: Path
    checkpoint_path: Path
    heal_root: Path
    tensorrt_root: Path
    plugin_path: Path
    eval_manifest_path: Path
    physical_gpu_id: int
    fixed_k: int = 29696
    max_agents: int = 2
    num_frames: int = 500
    warmup_frames: int = 200
    latency_rounds: int = 3
    dataloader_num_workers: int = 8
    workspace_mib: int = 4096
    build_timeout_seconds: int = 3600
    conda_env: str = "modelopt"
    search_space_policy: str = "legacy_family_static_dependency_closure_v1"

    def __post_init__(self) -> None:
        if self.family_id not in {
            "heal_lidar_fcooper", "heal_lidar_disco", "heal_lidar_attfusion",
            "heal_lidar_cobevt",
        }:
            raise ValueError(f"unsupported_baseline_evaluator_family:{self.family_id}")
        expected_model = {
            "heal_lidar_fcooper": "lidar_fcooper",
            "heal_lidar_disco": "lidar_disco",
            "heal_lidar_attfusion": "lidar_attfuse",
            "heal_lidar_cobevt": "lidar_cobevt",
        }[self.family_id]
        if self.model_name != expected_model:
            raise ValueError(
                f"baseline_evaluator_family_model_mismatch:{self.family_id}:{self.model_name}:"
                f"expected={expected_model}"
            )
        if int(self.fixed_k) <= 0 or int(self.max_agents) != 2:
            raise ValueError("invalid_baseline_fixed_k_or_agent_contract")
        if min(int(self.num_frames), int(self.warmup_frames), int(self.latency_rounds)) <= 0:
            raise ValueError("invalid_baseline_evaluation_protocol")


class HealLidarBaselineRealEvaluator:
    """Compose family export/QDQ/TRT primitives without Pyramid assumptions."""

    pipeline_version = "heal-lidar-baseline-real-evaluator-v1"

    def __init__(self, config: HealLidarBaselineEvaluationConfig) -> None:
        self.config = config
        required = {
            "model_config": config.model_config_path,
            "checkpoint": config.checkpoint_path,
            "heal_root": config.heal_root,
            "tensorrt_root": config.tensorrt_root,
            "plugin": config.plugin_path,
            "evaluation_manifest": config.eval_manifest_path,
        }
        missing = [f"{name}:{path}" for name, path in required.items() if not Path(path).exists()]
        if missing:
            raise RuntimeError(f"heal_lidar_evaluator_required_artifact_missing:{missing}")

    @staticmethod
    def _write_json(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")

    @staticmethod
    def _sha256(path: str | Path) -> str:
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def identity(self) -> dict[str, Any]:
        payload = {
            "pipeline_version": self.pipeline_version,
            "family_id": self.config.family_id,
            "model_name": self.config.model_name,
            "model_config_path": str(self.config.model_config_path.resolve()),
            "model_config_sha256": self._sha256(self.config.model_config_path),
            "checkpoint_path": str(self.config.checkpoint_path.resolve()),
            "checkpoint_sha256": self._sha256(self.config.checkpoint_path),
            "plugin_path": str(self.config.plugin_path.resolve()),
            "plugin_sha256": self._sha256(self.config.plugin_path),
            "eval_manifest_path": str(self.config.eval_manifest_path.resolve()),
            "eval_manifest_sha256": self._sha256(self.config.eval_manifest_path),
            "fixed_k": int(self.config.fixed_k),
            "max_agents": int(self.config.max_agents),
            "input_contract": "heal_lidar_baseline_fixed_k",
            "num_frames": int(self.config.num_frames),
            "warmup_frames": int(self.config.warmup_frames),
            "latency_rounds": int(self.config.latency_rounds),
            "search_space_policy": str(self.config.search_space_policy),
        }
        payload["evaluator_identity_hash"] = stable_json_hash(payload)
        return payload

    def export_candidate(
        self,
        model: nn.Module,
        ego_batch: Mapping[str, Any],
        audit: Any,
        *,
        output_dir: str | Path,
    ) -> Any:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        artifact = export_heal_lidar_baseline_fixed_k_onnx(
            model,
            ego_batch,
            destination / "physical_fp32.onnx",
            audit=audit,
            policy=HealLidarBaselineExportPolicy(
                fixed_k=int(self.config.fixed_k),
                max_agents=int(self.config.max_agents),
            ),
        )
        self._write_json(destination / "onnx_export_acceptance.json", artifact.to_dict())
        return artifact

    def build_qdq(
        self,
        export_artifact: Any,
        audit: Any,
        module_precision_profile: Mapping[str, str],
        scales: Mapping[str, Any],
        *,
        output_dir: str | Path,
        profile_id: str,
        calibration_metadata: Mapping[str, Any] | None = None,
        auxiliary_precision: str = "fp16",
        runtime_precision_relations: Any = (),
        module_to_precision_group: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        origin = export_artifact.export.origin_map
        mapping, island = build_heal_lidar_baseline_precision_mapping(
            origin,
            module_precision_profile,
            audit=audit,
            canonical_onnx_path=export_artifact.export.onnx_path,
            profile_id=profile_id,
            auxiliary_precision=auxiliary_precision,
            precision_policy=self.config.search_space_policy,
            runtime_precision_relations=runtime_precision_relations,
            module_to_precision_group=module_to_precision_group,
        )
        qdq, qdq_island = insert_heal_lidar_baseline_explicit_qdq(
            export_artifact.export.onnx_path,
            destination / "explicit_qdq.onnx",
            mapping,
            family=self.config.family_id,
            scales=scales,
            calibration_metadata=calibration_metadata,
        )
        self._write_json(destination / "canonical_precision_mapping.json", mapping.to_dict())
        self._write_json(destination / "fusion_island_contract.json", island)
        self._write_json(destination / "qdq_insertion_acceptance.json", qdq.to_dict())
        self._write_json(destination / "qdq_fusion_island_acceptance.json", qdq_island)
        return {
            "mapping": mapping,
            "fusion_island": island,
            "qdq": qdq,
            "qdq_fusion_island": qdq_island,
            "qdq_onnx_path": Path(qdq.output_onnx),
        }

    def build_engine(
        self,
        model: nn.Module,
        qdq_artifact: Mapping[str, Any],
        *,
        output_dir: str | Path,
    ) -> dict[str, Any]:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        trtexec = self.config.tensorrt_root / "bin/trtexec"
        if not trtexec.is_file():
            trtexec = self.config.tensorrt_root / "targets/x86_64-linux-gnu/bin/trtexec"
        build_config = TensorRTBuildConfig(
            trtexec_path=trtexec,
            plugin_path=self.config.plugin_path,
            workspace_mib=int(self.config.workspace_mib),
            shape_profiles={},
            timeout_seconds=int(self.config.build_timeout_seconds),
            enable_fp16=True,
            enable_int8=True,
            no_tf32=True,
            skip_inference=True,
            export_layer_info=True,
            strongly_typed=True,
            policy_version="heal-lidar-baseline-explicit-qdq-strongly-typed-no-tf32-v1",
        )
        snapshot = build_physical_structure_snapshot_v2(
            model,
            model_family=self.config.family_id,
        )
        engine_path = destination / "candidate.plan"
        build = build_engine_modelopt(
            qdq_onnx=qdq_artifact["qdq_onnx_path"],
            engine_path=engine_path,
            precision_mapping=qdq_artifact["mapping"],
            build_config=build_config,
            physical_snapshot=snapshot,
            output_dir=destination / "engine_build",
            tensorrt_root=self.config.tensorrt_root,
            conda_env=self.config.conda_env,
            gpu_id=int(self.config.physical_gpu_id),
        )
        self._write_json(destination / "engine_build_acceptance.json", build)
        if build.get("status") != "ok":
            raise RuntimeError(
                f"heal_lidar_candidate_engine_build_failed:{build.get('failure_reason', build.get('status'))}"
            )
        layer_info = destination / "engine_build/engine_layer_info.json"
        if self.config.search_space_policy in {
            "heal_runtime_graph_v1",
            "pyramid_runtime_tracer_v1",
        }:
            # The generic runtime policy accepts TensorRT's structure and
            # weighted-op precision proofs without family-named node lists.
            weighted = dict(build.get("precision_realization_validation", {}) or {})
            structure = dict(build.get("engine_structure_validation", {}) or {})
            precision = {
                "schema_version": "heal-runtime-graph-precision-acceptance-v1",
                "passed": bool(weighted.get("passed", False) and structure.get("passed", False)),
                "weighted_precision": weighted,
                "engine_structure": structure,
                "fusion_island": {
                    "passed": True,
                    "validation_policy": "runtime_graph_generic_engine_acceptance",
                    "manual_family_node_audit_used": False,
                    "qdq_auxiliary_contract": "validated_before_engine_build",
                },
            }
        else:
            precision = validate_heal_lidar_precision_realization(
                layer_info,
                qdq_artifact["mapping"],
                family=self.config.family_id,
            )
        self._write_json(destination / "precision_realization_acceptance.json", precision)
        if not precision["passed"]:
            raise RuntimeError(f"heal_lidar_candidate_precision_realization_failed:{precision}")
        if not engine_path.is_file() or engine_path.stat().st_size <= 0:
            raise RuntimeError("heal_lidar_candidate_engine_missing")
        return {
            "engine_path": engine_path,
            "engine_sha256": self._sha256(engine_path),
            "build": build,
            "precision_acceptance": precision,
            "physical_snapshot": snapshot,
        }

    def evaluate_existing_engine(
        self,
        engine_path: str | Path,
        *,
        output_dir: str | Path,
        expected_engine_sha256: str = "",
    ) -> dict[str, Any]:
        engine = Path(engine_path).resolve()
        if not engine.is_file() or engine.stat().st_size <= 0:
            raise RuntimeError(f"heal_lidar_evaluation_engine_missing:{engine}")
        digest = self._sha256(engine)
        if expected_engine_sha256 and digest != str(expected_engine_sha256):
            raise RuntimeError(f"heal_lidar_evaluation_engine_hash_mismatch:{digest}")
        evaluation = evaluate_v2xvit_engine_modelopt(
            engine_path=engine,
            model_config=self.config.model_config_path,
            heal_root=self.config.heal_root,
            output_dir=Path(output_dir),
            tensorrt_root=self.config.tensorrt_root,
            plugin_path=self.config.plugin_path,
            eval_manifest_path=self.config.eval_manifest_path,
            physical_gpu_id=int(self.config.physical_gpu_id),
            fixed_k=int(self.config.fixed_k),
            max_agents=int(self.config.max_agents),
            num_frames=int(self.config.num_frames),
            warmup_frames=int(self.config.warmup_frames),
            latency_rounds=int(self.config.latency_rounds),
            dataloader_num_workers=int(self.config.dataloader_num_workers),
            input_contract="heal_lidar_baseline_fixed_k",
        )
        if evaluation.get("status") != "ok":
            raise RuntimeError(
                f"heal_lidar_engine_evaluation_failed:{evaluation.get('failure_reason', evaluation.get('status'))}"
            )
        result = {
            **dict(evaluation),
            "engine_path": str(engine),
            "engine_sha256": digest,
            "engine_rebuilt": False,
            "evaluator_identity": self.identity(),
        }
        self._write_json(Path(output_dir) / "evaluation_acceptance.json", result)
        return result

    def deploy_and_evaluate(
        self,
        model: nn.Module,
        ego_batch: Mapping[str, Any],
        audit: Any,
        module_precision_profile: Mapping[str, str],
        scales: Mapping[str, Any],
        *,
        output_dir: str | Path,
        profile_id: str,
        calibration_metadata: Mapping[str, Any] | None = None,
        auxiliary_precision: str = "fp16",
    ) -> dict[str, Any]:
        destination = Path(output_dir)
        export = self.export_candidate(model, ego_batch, audit, output_dir=destination / "export")
        qdq = self.build_qdq(
            export,
            audit,
            module_precision_profile,
            scales,
            output_dir=destination / "qdq",
            profile_id=profile_id,
            calibration_metadata=calibration_metadata,
            auxiliary_precision=auxiliary_precision,
        )
        engine = self.build_engine(model, qdq, output_dir=destination / "deployment")
        evaluation = self.evaluate_existing_engine(
            engine["engine_path"],
            output_dir=destination / "evaluation",
            expected_engine_sha256=engine["engine_sha256"],
        )
        result = {
            "status": "ok",
            "profile_id": profile_id,
            "export_acceptance": export.to_dict(),
            "qdq_output_sha256": qdq["qdq"].output_sha256,
            "engine_sha256": engine["engine_sha256"],
            "precision_acceptance": engine["precision_acceptance"],
            "evaluation": evaluation,
            "evaluator_identity": self.identity(),
        }
        result["deployment_evaluation_hash"] = stable_json_hash(result)
        self._write_json(destination / "candidate_acceptance.json", result)
        return result


class HealLidarBaselineCandidateEvaluator:
    """Materialize one phenotype, calibrate, deploy, and score it against FP32."""

    def __init__(
        self,
        *,
        context: Any,
        run_dir: str | Path,
        baseline_engine_path: str | Path,
        num_frames: int,
        warmup_frames: int,
        latency_rounds: int,
        objective_config: Stage2ObjectiveConfig | None = None,
        dataloader_num_workers: int = 8,
        reference_baseline: Mapping[str, Any] | None = None,
    ) -> None:
        self.context = context
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.baseline_engine_path = Path(baseline_engine_path).expanduser().resolve()
        if not self.baseline_engine_path.is_file():
            raise RuntimeError(
                f"heal_lidar_baseline_reference_engine_missing:{self.baseline_engine_path}"
            )
        self.num_frames = int(num_frames)
        self.warmup_frames = int(warmup_frames)
        self.latency_rounds = int(latency_rounds)
        self.objective_config = objective_config or Stage2ObjectiveConfig(
            latency_metric="forward_p50_ms"
        )
        self.dataloader_num_workers = int(dataloader_num_workers)
        self._reference_baseline_override = dict(reference_baseline or {})
        self.artifacts = None
        self.real_cache = None

    def _real_evaluator(self, *, output_dir: Path) -> HealLidarBaselineRealEvaluator:
        return HealLidarBaselineRealEvaluator(HealLidarBaselineEvaluationConfig(
            family_id=self.context.family_id,
            model_name=self.context.model_name,
            model_config_path=self.context.model_config,
            checkpoint_path=self.context.checkpoint_path,
            heal_root=Path(self.context.model_bundle.adapter.heal_repo),
            tensorrt_root=self.context.tensorrt.tensorrt_root,
            plugin_path=self.context.plugin_paths[0],
            eval_manifest_path=self.context.eval_manifest_path,
            physical_gpu_id=int(self.context.physical_gpu_id),
            fixed_k=int(self.context.fixed_k),
            max_agents=int(self.context.max_agents),
            num_frames=int(self.num_frames),
            warmup_frames=int(self.warmup_frames),
            latency_rounds=int(self.latency_rounds),
            dataloader_num_workers=int(self.dataloader_num_workers),
            conda_env=self.context.tensorrt.conda_env,
            search_space_policy=str(self.context.search_space_policy),
        ))

    def _stage2_reference_baseline(self) -> dict[str, Any]:
        if self._reference_baseline_override:
            return dict(self._reference_baseline_override)
        evaluator = self._real_evaluator(output_dir=self.run_dir / "reference")
        result = evaluator.evaluate_existing_engine(
            self.baseline_engine_path,
            output_dir=self.run_dir / "reference",
        )
        self._reference_baseline_override = dict(result)
        return dict(result)

    def _materialize(self, phenotype: Any, output_dir: Path) -> dict[str, Any]:
        unified_domains = tuple(
            getattr(self.context, "unified_pruning_domains", ()) or ()
        )
        if unified_domains:
            from search.pruning_space.unified_physical_pruner import (
                materialize_unified_widths,
            )

            width_profile = dict(
                phenotype.metadata.get("domain_width_profile", {}) or {}
            )
            if set(width_profile) != {
                str(domain.domain_id) for domain in unified_domains
            }:
                raise RuntimeError(
                    "heal_transformer_unified_width_profile_mismatch:"
                    f"observed={sorted(width_profile)}:"
                    f"expected={sorted(str(domain.domain_id) for domain in unified_domains)}"
                )
            result = materialize_unified_widths(
                self.context.model,
                tuple(getattr(self.context, "unified_atomic_units", ())),
                unified_domains,
                width_profile,
                model_name=str(
                    getattr(self.context, "unified_model_name", self.context.model_name)
                ),
            )
            self._write_json(output_dir / "phenotype.json", phenotype.to_dict())
            self._write_json(
                output_dir / "unified_physical_report.json", result.report.to_dict()
            )
            if not result.report.passed:
                raise RuntimeError(
                    f"heal_transformer_unified_physical_failed:{result.report.issues}"
                )
            __import__("torch").save(
                result.model.state_dict(), output_dir / "pruned_checkpoint.pth"
            )
            return {
                "model": result.model,
                "unified": result,
                "plan": result.cnn_plan,
                "ledger": result.cnn_ledger,
                "snapshot": result.cnn_snapshot,
            }

        from search.adapters.pruning_adapter import FormalPruningAdapter
        from search.model_family.heal_lidar_pruning import materialize_heal_lidar_baseline

        request = FormalPruningAdapter().request_from_phenotype(
            phenotype,
            self.context.atomic_prune_units,
        )
        result = materialize_heal_lidar_baseline(
            self.context.model,
            request,
            family=self.context.family_id,
            example_inputs=self.context.trace_example_inputs,
        )
        self._write_json(output_dir / "phenotype.json", phenotype.to_dict())
        self._write_json(output_dir / "sampling_request.json", request.to_dict())
        self._write_json(output_dir / "physical_plan.json", result["plan"].to_dict())
        self._write_json(output_dir / "physical_ledger.json", result["ledger"].to_dict())
        self._write_json(output_dir / "physical_snapshot.json", result["snapshot"].to_dict())
        __import__("torch").save(
            result["model"].state_dict(),
            output_dir / "pruned_checkpoint.pth",
        )
        return result

    def _write_candidate_result(self, destination: Path, result: Mapping[str, Any]) -> None:
        """Publish both family-native and generic Stage-2 score contracts."""

        self._write_json(destination / "candidate_stage2_result.json", result)
        self._write_json(destination / "stage2_score.json", result)

    @staticmethod
    def _write_json(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )

    def _calibration_scales(
        self,
        *,
        model: nn.Module,
        export_artifact: Any,
        mapping: Any,
        output_dir: Path,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        int8_modules = sorted({
            row.module_path
            for row in mapping.entries
            if row.realized_request_precision == "int8" and row.weight_initializer
        })
        if not int8_modules:
            return {}, {"reason": "candidate_has_no_int8_layers", "frame_count": 0}
        backend = str(self.context.quant_activation_calibration_backend)
        if backend == "tensorrt_entropy_calibration2":
            from search.integration.calibration_provider import (
                BASELINE_FIXED_K_CALIBRATION_INPUT_NAMES,
                build_tensorrt_entropy_calibration_cache_modelopt,
                qdq_scales_from_tensorrt_entropy_cache,
            )

            if self.context.quant_calibration_npz_manifest is None:
                raise RuntimeError("baseline_candidate_entropy_manifest_missing")
            cache_dir = output_dir / "entropy_calibration"
            existing = cache_dir / "calibration_result.json"
            entropy = build_tensorrt_entropy_calibration_cache_modelopt(
                onnx_path=export_artifact.export.onnx_path,
                calibration_npz_manifest=self.context.quant_calibration_npz_manifest,
                output_dir=cache_dir,
                tensorrt_root=self.context.tensorrt.tensorrt_root,
                plugin_path=self.context.plugin_paths[0],
                physical_gpu_id=int(self.context.physical_gpu_id),
                num_batches=int(self.context.quant_calibration_batches),
                fixed_k=int(self.context.fixed_k),
                input_names=BASELINE_FIXED_K_CALIBRATION_INPUT_NAMES,
                conda_env=self.context.tensorrt.conda_env,
                force_rebuild=not existing.is_file(),
            )
            scales, metadata = qdq_scales_from_tensorrt_entropy_cache(
                onnx_path=export_artifact.export.onnx_path,
                origin_map=export_artifact.export.origin_map,
                module_paths=int8_modules,
                cache_path=entropy["calibration_cache_path"],
                weight_granularity="per_channel",
            )
            return scales, {**metadata, "entropy_build": entropy}
        if backend == "external_tensorrt_entropy_cache_exact_match":
            from search.integration.calibration_provider import qdq_scales_from_tensorrt_entropy_cache

            if self.context.quant_activation_calibration_cache_path is None:
                raise RuntimeError("baseline_candidate_external_entropy_cache_missing")
            return qdq_scales_from_tensorrt_entropy_cache(
                onnx_path=export_artifact.export.onnx_path,
                origin_map=export_artifact.export.origin_map,
                module_paths=int8_modules,
                cache_path=self.context.quant_activation_calibration_cache_path,
                weight_granularity="per_channel",
            )
        from search.integration.calibration_provider import (
            BASELINE_FIXED_K_CALIBRATION_INPUT_NAMES,
            collect_or_load_qdq_calibration_scales,
        )

        scales = collect_or_load_qdq_calibration_scales(
            model=model,
            adapter=self.context.model_bundle.adapter,
            model_config_path=self.context.model_config,
            module_paths=int8_modules,
            device=__import__("torch").device(self.context.runtime_device),
            cache_path=output_dir / "modelopt_histogram_scales.json",
            num_batches=int(self.context.quant_calibration_batches),
            onnx_path=export_artifact.export.onnx_path,
            origin_map=export_artifact.export.origin_map,
            weight_granularity="per_channel",
            activation_calibration_method="entropy",
            fixed_k=int(self.context.fixed_k),
            calibration_npz_manifest=self.context.quant_calibration_npz_manifest,
            calibration_input_names=BASELINE_FIXED_K_CALIBRATION_INPUT_NAMES,
        )
        return scales, {
            "activation_calibration_method": "modelopt_histogram_entropy",
            "frame_count": int(self.context.quant_calibration_batches),
        }

    def evaluate_candidate(
        self,
        phenotype: Any,
        *,
        output_dir: str | Path,
        candidate_hash: str = "",
    ) -> dict[str, Any]:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        self._write_json(destination / "phenotype.json", phenotype.to_dict())
        try:
            deployment = self.build_candidate_artifacts(
                phenotype,
                output_dir=destination,
                candidate_hash=candidate_hash,
            )
            evaluator = self._real_evaluator(output_dir=destination)
            evaluation = evaluator.evaluate_existing_engine(
                deployment["engine_path"],
                output_dir=destination / "evaluation",
                expected_engine_sha256=deployment["engine_sha256"],
            )
            baseline = self._stage2_reference_baseline()
            score = compute_stage2_score(
                evaluation,
                baseline=baseline,
                config=self.objective_config,
            )
            result = {
                **deployment,
                "status": "ok",
                "candidate_hash": candidate_hash,
                "artifact_dir": str(destination),
                "evaluation_acceptance": evaluation.get("status") == "ok",
                **evaluation,
                **score,
                "evaluation_invoked": True,
            }
            self._write_candidate_result(destination, result)
            return result
        except Exception as exc:  # noqa: BLE001
            result = {
                "status": "evaluation_failed",
                "failure_reason": f"{type(exc).__name__}:{exc}",
                "failure_traceback": traceback.format_exc(),
                "F2": float("inf"),
                "candidate_hash": candidate_hash,
                "artifact_dir": str(destination),
            }
            self._write_candidate_result(destination, result)
            return result

    def build_candidate_artifacts(
        self,
        phenotype: Any,
        *,
        output_dir: str | Path,
        candidate_hash: str = "",
    ) -> dict[str, Any]:
        """Build physical/ONNX/QDQ/engine artifacts without evaluating them.

        This is the formal build-only boundary used by P/Q ablations.  It lets
        orchestration serialize engine construction independently from fresh
        full-validation workers and makes it impossible to mistake a short
        ``evaluate_candidate`` run for an engine-build phase.
        """

        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        self._write_json(destination / "phenotype.json", phenotype.to_dict())
        timings: dict[str, float] = {}
        total_started = time.perf_counter()

        def timed(name: str, function: Any) -> Any:
            started = time.perf_counter()
            try:
                return function()
            finally:
                timings[name] = time.perf_counter() - started

        try:
            physical = timed(
                "physical_materialization_seconds",
                lambda: self._materialize(phenotype, destination / "physical"),
            )
            model = physical["model"]
            audit = timed(
                "post_materialization_validation_seconds",
                lambda: self.context.model_bundle.provider.audit(
                    model,
                    self.context.model_bundle.config,
                    require_original_widths=False,
                ),
            )
            evaluator = self._real_evaluator(output_dir=destination)
            export = timed(
                "onnx_export_seconds",
                lambda: evaluator.export_candidate(
                    model,
                    self.context.trace_example_inputs,
                    audit,
                    output_dir=destination / "export",
                ),
            )
            qkv_paths = tuple(getattr(self.context, "unified_qkv_paths", ()) or ())
            qkv_nodes: list[str] = []
            onnx_attention = None
            if qkv_paths:
                from search.stage2.transformer_precision_export import (
                    audit_onnx_attention_fp32_contract,
                )

                qkv_nodes = sorted({
                    str(row.canonical_node_name)
                    for row in export.export.origin_map.entries
                    if any(
                        str(row.module_path) == path
                        or str(row.module_path).endswith(f".{path}")
                        for path in qkv_paths
                    )
                })
                if not qkv_nodes:
                    raise RuntimeError("heal_transformer_qkv_origin_mapping_missing")
                onnx_attention = audit_onnx_attention_fp32_contract(
                    export.export.onnx_path,
                    qkv_canonical_node_names=qkv_nodes,
                )
                self._write_json(
                    destination / "export/onnx_attention_fp32_audit.json",
                    onnx_attention,
                )
                if not onnx_attention.get("passed"):
                    raise RuntimeError(
                        f"heal_transformer_onnx_attention_contract_failed:{onnx_attention}"
                    )
            profile = {
                module_path: precision.lower()
                for module_path, precision in phenotype.realized_precision_profile.items()
            }
            auxiliary_precision = str(
                phenotype.metadata.get("heal_lidar_auxiliary_precision", "FP16")
            ).lower()
            module_to_precision_group = {
                module_path: group.group_id
                for group in getattr(
                    getattr(self.context, "search_space", None),
                    "quantization_groups",
                    (),
                )
                for module_path in group.module_paths
            }
            mapping, island = timed(
                "precision_mapping_seconds",
                lambda: build_heal_lidar_baseline_precision_mapping(
                    export.export.origin_map,
                    profile,
                    audit=audit,
                    canonical_onnx_path=export.export.onnx_path,
                    profile_id=candidate_hash or "candidate",
                    auxiliary_precision=auxiliary_precision,
                    precision_policy=str(getattr(
                        self.context,
                        "search_space_policy",
                        "legacy_family_static_dependency_closure_v1",
                    )),
                    runtime_precision_relations=list(getattr(
                        getattr(self.context, "precision_coupling_result", None),
                        "relations",
                        [],
                    )),
                    module_to_precision_group=module_to_precision_group,
                    functional_precision_paths=tuple(getattr(
                        self.context,
                        "unified_functional_precision_paths",
                        (),
                    )),
                ),
            )
            scales, calibration = timed(
                "activation_calibration_seconds",
                lambda: self._calibration_scales(
                    model=model,
                    export_artifact=export,
                    mapping=mapping,
                    output_dir=destination / "calibration",
                ),
            )
            qdq_result, qdq_island = timed(
                "qdq_insertion_seconds",
                lambda: insert_heal_lidar_baseline_explicit_qdq(
                    export.export.onnx_path,
                    destination / "qdq/explicit_qdq.onnx",
                    mapping,
                    family=self.context.family_id,
                    scales=scales,
                    calibration_metadata=calibration,
                ),
            )
            qdq = {
                "mapping": mapping,
                "fusion_island": island,
                "qdq": qdq_result,
                "qdq_fusion_island": qdq_island,
                "qdq_onnx_path": Path(qdq_result.output_onnx),
            }
            qdq_attention = None
            functional_onnx = None
            if qkv_paths:
                from scripts.run_v2xvit_greedy005_stage2 import (
                    _force_attention_fp32_contract,
                )
                from search.stage2.transformer_precision_export import (
                    audit_onnx_attention_fp32_contract,
                )

                pre_cast = audit_onnx_attention_fp32_contract(
                    qdq["qdq_onnx_path"],
                    qkv_canonical_node_names=qkv_nodes,
                )
                cast_report = _force_attention_fp32_contract(
                    qdq["qdq_onnx_path"], pre_cast, av_profile="AV32"
                )
                self._write_json(
                    destination / "qdq/qk_softmax_av_fp32_cast_report.json",
                    {"pre_cast": pre_cast, **cast_report},
                )
                qdq_attention = audit_onnx_attention_fp32_contract(
                    qdq["qdq_onnx_path"],
                    qkv_canonical_node_names=qkv_nodes,
                )
                self._write_json(
                    destination / "qdq/qdq_attention_fp32_audit.json",
                    qdq_attention,
                )
                if not qdq_attention.get("passed"):
                    raise RuntimeError(
                        "heal_transformer_qdq_attention_contract_failed:"
                        f"{qdq_attention}"
                    )
                from search.stage2.v2xvit_functional_precision import (
                    build_v2xvit_functional_onnx_mapping,
                    requested_states_from_phenotype,
                )

                precision_units = tuple(getattr(
                    self.context, "unified_precision_units", ()
                ))
                requested_states = requested_states_from_phenotype(
                    phenotype, precision_units
                )
                functional_onnx = build_v2xvit_functional_onnx_mapping(
                    qdq["qdq_onnx_path"],
                    origin_map=export.export.origin_map,
                    precision_units=precision_units,
                    attention_instances=tuple(getattr(
                        self.context, "unified_attention_instances", ()
                    )),
                    ffn_instances=tuple(getattr(
                        self.context, "unified_ffn_instances", ()
                    )),
                    requested_states=requested_states,
                )
                self._write_json(
                    destination / "qdq/functional_precision_onnx_audit.json",
                    functional_onnx,
                )
                if not functional_onnx.get("passed"):
                    raise RuntimeError(
                        "heal_transformer_functional_onnx_contract_failed:"
                        f"{functional_onnx}"
                    )
            self._write_json(
                destination / "qdq/canonical_precision_mapping.json", mapping.to_dict()
            )
            self._write_json(destination / "qdq/fusion_island_contract.json", island)
            self._write_json(
                destination / "qdq/qdq_insertion_acceptance.json", qdq_result.to_dict()
            )
            self._write_json(
                destination / "calibration/calibration_metadata.json", calibration
            )
            engine = timed(
                "tensorrt_engine_build_seconds",
                lambda: evaluator.build_engine(
                    model, qdq, output_dir=destination / "deployment"
                ),
            )
            trt_attention = None
            functional_trt = None
            if qkv_paths:
                from search.stage2.transformer_precision_export import (
                    audit_trt_attention_fp32_contract,
                )

                trt_attention = audit_trt_attention_fp32_contract(
                    destination / "deployment/engine_build/engine_layer_info.json",
                    qdq_attention,
                )
                self._write_json(
                    destination / "deployment/trt_attention_fp32_audit.json",
                    trt_attention,
                )
                if not trt_attention.get("passed"):
                    raise RuntimeError(
                        "heal_transformer_trt_attention_contract_failed:"
                        f"{trt_attention}"
                    )
                from search.stage2.v2xvit_functional_precision import (
                    audit_trt_v2xvit_functional_precision,
                )

                functional_trt = audit_trt_v2xvit_functional_precision(
                    destination / "deployment/engine_build/engine_layer_info.json",
                    functional_onnx,
                )
                self._write_json(
                    destination / "deployment/functional_precision_trt_audit.json",
                    functional_trt,
                )
                if not functional_trt.get("passed"):
                    raise RuntimeError(
                        "heal_transformer_functional_trt_contract_failed:"
                        f"{functional_trt}"
                    )
            timings["total_build_pipeline_seconds"] = (
                time.perf_counter() - total_started
            )
            self._write_json(
                destination / "engine_build_phase_timings.json",
                {
                    "schema_version": "heal-lidar-engine-build-phase-timings-v1",
                    "candidate_hash": candidate_hash,
                    "physical_gpu_id": int(getattr(self.context, "physical_gpu_id", 0)),
                    "timings": timings,
                    "engine_build_gpu_hours": timings.get(
                        "tensorrt_engine_build_seconds", 0.0
                    )
                    / 3600.0,
                    "total_pipeline_gpu_hours": timings[
                        "total_build_pipeline_seconds"
                    ]
                    / 3600.0,
                },
            )
            before = sum(
                int(parameter.numel()) for parameter in self.context.model.parameters()
            )
            after = sum(int(parameter.numel()) for parameter in model.parameters())
            result = {
                "schema_version": "heal-lidar-candidate-build-only-v1",
                "status": "ok",
                "candidate_hash": candidate_hash,
                "artifact_dir": str(destination.resolve()),
                "engine_path": str(Path(engine["engine_path"]).resolve()),
                "engine_sha256": engine["engine_sha256"],
                "engine_built_this_call": True,
                "evaluation_invoked": False,
                "physical_parameter_count_before": before,
                "physical_parameter_count_after": after,
                "physical_parameter_pruning_ratio": 1.0 - after / max(before, 1),
                "requested_int8_count": sum(
                    row.requested_precision == "int8" for row in mapping.entries
                ),
                "realized_int8_count": engine["precision_acceptance"][
                    "weighted_precision"
                ]["realized_int8_count"],
                "physical_acceptance": True,
                "qdq_acceptance": bool(qdq_island["passed"]),
                "engine_acceptance": True,
                "precision_acceptance": bool(engine["precision_acceptance"]["passed"]),
                "merge_acceptance": bool(
                    engine["precision_acceptance"]["fusion_island"]["passed"]
                ),
                "transformer_attention_fp32_acceptance": bool(
                    not qkv_paths
                    or (
                        onnx_attention
                        and qdq_attention
                        and trt_attention
                        and onnx_attention.get("passed")
                        and qdq_attention.get("passed")
                        and trt_attention.get("passed")
                    )
                ),
                "transformer_functional_precision_acceptance": bool(
                    not qkv_paths
                    or (
                        functional_onnx
                        and functional_trt
                        and functional_onnx.get("passed")
                        and functional_trt.get("passed")
                    )
                ),
                "auxiliary_precision": auxiliary_precision,
                "calibration_metadata_path": str(
                    (destination / "calibration/calibration_metadata.json").resolve()
                ),
                "precision_acceptance_path": str(
                    (
                        destination
                        / "deployment/precision_realization_acceptance.json"
                    ).resolve()
                ),
                "engine_build_phase_timings": dict(timings),
            }
            self._write_json(destination / "candidate_build_result.json", result)
            return result
        except Exception as exc:
            timings["total_build_pipeline_seconds"] = (
                time.perf_counter() - total_started
            )
            self._write_json(
                destination / "engine_build_phase_timings.json",
                {
                    "schema_version": "heal-lidar-engine-build-phase-timings-v1",
                    "candidate_hash": candidate_hash,
                    "physical_gpu_id": int(getattr(self.context, "physical_gpu_id", 0)),
                    "status": "failed",
                    "timings": timings,
                },
            )
            failure = {
                "schema_version": "heal-lidar-candidate-build-only-v1",
                "status": "build_failed",
                "candidate_hash": candidate_hash,
                "artifact_dir": str(destination.resolve()),
                "failure_reason": f"{type(exc).__name__}:{exc}",
                "failure_traceback": traceback.format_exc(),
                "engine_build_phase_timings": dict(timings),
                "evaluation_invoked": False,
            }
            self._write_json(destination / "candidate_build_result.json", failure)
            raise

    def reevaluate_existing_candidate_engine(
        self,
        phenotype: Any,
        *,
        source_artifact_dir: str | Path,
        output_dir: str | Path,
        candidate_hash: str = "",
    ) -> dict[str, Any]:
        source = Path(source_artifact_dir)
        engine = source / "deployment/candidate.plan"
        acceptance = source / "deployment/precision_realization_acceptance.json"
        if not engine.is_file() or not acceptance.is_file():
            raise RuntimeError(
                f"heal_lidar_existing_candidate_acceptance_missing:{engine}:{acceptance}"
            )
        precision = json.loads(acceptance.read_text(encoding="utf-8"))
        if not precision.get("passed", False):
            raise RuntimeError("heal_lidar_existing_candidate_precision_not_accepted")
        evaluator = self._real_evaluator(output_dir=Path(output_dir))
        evaluation = evaluator.evaluate_existing_engine(
            engine,
            output_dir=Path(output_dir),
        )
        score = compute_stage2_score(
            evaluation,
            baseline=self._stage2_reference_baseline(),
            config=self.objective_config,
        )
        result = {
            "status": "ok",
            "candidate_hash": candidate_hash,
            "artifact_dir": str(Path(output_dir)),
            "source_artifact_dir": str(source),
            "engine_rebuilt": False,
            "physical_artifact_rebuilt": False,
            "onnx_rebuilt": False,
            "calibration_rerun": False,
            "qdq_onnx_rebuilt": False,
            "precision_acceptance": True,
            "merge_acceptance": bool(precision.get("fusion_island", {}).get("passed", False)),
            "evaluation_acceptance": True,
            **evaluation,
            **score,
        }
        self._write_json(Path(output_dir) / "candidate_full_validation_result.json", result)
        return result


__all__ = [
    "HealLidarBaselineCandidateEvaluator",
    "HealLidarBaselineEvaluationConfig",
    "HealLidarBaselineRealEvaluator",
]
