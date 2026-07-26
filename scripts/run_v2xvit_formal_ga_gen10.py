#!/usr/bin/env python3
"""Run deployment-closed V2X-ViT GA: generation 0 plus exactly generations 1--10."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.audit_heal_transformer_search_models import MODEL_SPECS, _load
from scripts.run_v2xvit_greedy005_full import (
    _baseline_candidate,
    _build_full_space,
    _formal_space,
)
from scripts.run_v2xvit_greedy005_stage2 import _export_candidate
from scripts.run_v2xvit_greedy005_weight_only_abs import _type_coverage_forward
from scripts.run_v2xvit_six_budget_proxy import FrozenTrainPrefix
from scripts.smoke_transformer_unified_search import _multi_agent_validation_batch
from search.candidate import CandidateGenotype
from search.canonicalization import canonicalize_candidate
from search.ga.stage12_v3 import (
    Stage2Result,
    StrictGAConfig,
    StrictStage12V3Runner,
    UnifiedTaylorStage1Evaluator,
    phenotype_identity,
    score_stage2,
    validate_genotype_schema,
)
from search.hashing import canonical_json_hash
from search.model_family.calibration_manifest import load_v2xvit_train_manifest
from search.model_family.evaluation import evaluate_v2xvit_engine_modelopt
from search.pruning_space.unified_physical_pruner import materialize_unified_widths
from search.proxy.conservative_gate_activation_taylor import (
    FunctionalGateTaylorProxy,
    build_activation_units,
    collect_activation_taylor_cache_multi,
    collect_functional_gate_scores_multi,
    rerank_domains_by_gate_scores,
)
from search.proxy.joint_weight_activation_taylor import taylor_units_from_transformer_precision
from search.proxy.joint_weight_taylor import JointWeightTaylorProxy


LABELS = ("010",)
TARGETS = {"010": 0.10}
FORMAL_SEEDS = (0,)
ALLOWED_PHYSICAL_GPUS = (2, 3)


def atomic_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _json_default(value: Any) -> Any:
    if isinstance(value, CandidateGenotype):
        return value.to_dict()
    if isinstance(value, Stage2Result):
        return stage2_payload(value)
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    return str(value)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stage2_payload(result: Stage2Result) -> dict[str, Any]:
    return {
        "complete_phenotype_hash": result.complete_phenotype_hash,
        "genotype": result.genotype.to_dict(),
        "status": result.status,
        "mAP": result.map,
        "p50_ms": result.p50_ms,
        "requested_realized_exact": result.requested_realized_exact,
        "evaluated": result.evaluated,
        "skipped": result.skipped,
        "metadata": dict(result.metadata),
    }


def stage2_from_payload(row: Mapping[str, Any]) -> Stage2Result:
    return Stage2Result(
        complete_phenotype_hash=str(row["complete_phenotype_hash"]),
        genotype=CandidateGenotype.from_dict(row["genotype"]),
        status=str(row["status"]),
        map=None if row.get("mAP") is None else float(row["mAP"]),
        p50_ms=None if row.get("p50_ms") is None else float(row["p50_ms"]),
        requested_realized_exact=bool(row.get("requested_realized_exact", False)),
        evaluated=int(row.get("evaluated", 0)),
        skipped=int(row.get("skipped", 0)),
        metadata=dict(row.get("metadata") or {}),
    )


def candidate_from_state(row: Mapping[str, Any], source: str) -> CandidateGenotype:
    state = dict(row["state"])
    return CandidateGenotype(
        pruning_width_genes={key: int(value) for key, value in state["widths"].items()},
        precision_genes={key: str(value) for key, value in state["precision"].items()},
        meta={"created_by": source, "repair_count": 0},
    )


def precision_realized_exact(destination: Path) -> tuple[bool, dict[str, Any]]:
    acceptance_path = destination / "engine_build_acceptance.json"
    if not acceptance_path.is_file():
        return False, {"failure": "engine_build_acceptance_missing"}
    acceptance = json.loads(acceptance_path.read_text())
    precision = dict(acceptance.get("precision_realization_validation") or {})
    functional_path = destination / "functional_precision_trt_audit.json"
    attention_path = destination / "trt_attention_fp32_audit.json"
    av_path = destination / "av_profile_trt_audit.json"
    if (
        not functional_path.is_file()
        or not attention_path.is_file()
        or not av_path.is_file()
    ):
        return False, {
            "failure": "functional_attention_or_av_precision_audit_missing",
            "engine_build_acceptance": acceptance,
        }
    functional = json.loads(functional_path.read_text())
    attention = json.loads(attention_path.read_text())
    av = json.loads(av_path.read_text())
    exact = bool(
        acceptance.get("status") == "ok"
        and precision.get("passed")
        and not precision.get("mismatches")
        and int(precision.get("unresolved_layer_count", 0)) == 0
        and functional.get("passed")
        and int(functional.get("conflict_count", 0)) == 0
        and int(functional.get("unmapped_count", 0)) == 0
        and int(functional.get("fallback_count", 0)) == 0
        and attention.get("passed")
        and av.get("passed")
        and int(av.get("conflict_count", 0)) == 0
        and int(av.get("unmapped_count", 0)) == 0
        and int(av.get("fallback_count", 0)) == 0
    )
    return exact, {
        "engine_build_acceptance": acceptance,
        "functional_precision": functional,
        "attention_precision": attention,
        "av_precision": av,
    }


def best_real_candidate(
    rows: list[Stage2Result], greedy: Stage2Result
) -> Stage2Result:
    eligible = [
        row for row in rows
        if score_stage2(
            row, greedy_map=float(greedy.map), greedy_p50_ms=float(greedy.p50_ms)
        )["eligible"]
    ]
    if not eligible:
        return greedy
    highest_map = max(float(row.map) for row in eligible)
    lowest_p50 = min(float(row.p50_ms) for row in eligible)
    dominant = [
        row for row in eligible
        if float(row.map) == highest_map and float(row.p50_ms) == lowest_p50
    ]
    if dominant:
        proposed = min(dominant, key=lambda row: row.complete_phenotype_hash)
    else:
        proposed = min(
            eligible,
            key=lambda row: (
                float(score_stage2(
                    row,
                    greedy_map=float(greedy.map),
                    greedy_p50_ms=float(greedy.p50_ms),
                )["F_S2"]),
                -float(row.map),
                float(row.p50_ms),
                row.complete_phenotype_hash,
            ),
        )
    greedy_f = float(score_stage2(
        greedy, greedy_map=float(greedy.map), greedy_p50_ms=float(greedy.p50_ms)
    )["F_S2"])
    proposed_f = float(score_stage2(
        proposed, greedy_map=float(greedy.map), greedy_p50_ms=float(greedy.p50_ms)
    )["F_S2"])
    return proposed if proposed_f < greedy_f else greedy


class RealStage2Evaluator:
    def __init__(
        self,
        *,
        root: Path,
        label: str,
        seed: int,
        model: Any,
        adapter: Any,
        hypes: Mapping[str, Any],
        representative: Any,
        identity: Mapping[str, Any],
        space: Any,
        qkv_paths: tuple[str, ...],
        request: Mapping[str, Any],
        evaluation_manifest: Path,
        evaluation_frames: int,
        evaluation_warmup_frames: int,
        evaluation_protocol: str,
        plugin: Path,
        tensorrt_root: Path,
        physical_gpu: int,
        stage2_gpus: tuple[int, ...] = (),
    ) -> None:
        self.root = root
        self.label = label
        self.seed = seed
        self.model = model
        self.adapter = adapter
        self.hypes = hypes
        self.representative = representative
        self.identity = identity
        self.space = space
        self.qkv_paths = qkv_paths
        self.request = request
        self.evaluation_manifest = evaluation_manifest
        self.evaluation_frames = int(evaluation_frames)
        self.evaluation_warmup_frames = int(evaluation_warmup_frames)
        self.evaluation_protocol = str(evaluation_protocol)
        self.plugin = plugin
        self.tensorrt_root = tensorrt_root
        self.physical_gpu = physical_gpu
        self.stage2_gpus = tuple(int(value) for value in stage2_gpus)

    def _load_cached_result(self, path: Path) -> Stage2Result:
        result = stage2_from_payload(json.loads(path.read_text()))
        frames = int(result.metadata.get("stage2_evaluation_frames", -1))
        protocol = str(result.metadata.get("evaluation_protocol", ""))
        if frames != self.evaluation_frames or protocol != self.evaluation_protocol:
            raise RuntimeError(
                "stage2_cache_evaluation_protocol_mismatch:"
                f"frames={frames}:expected={self.evaluation_frames}:protocol={protocol}"
            )
        return result

    def evaluate_many(
        self, genotypes: list[CandidateGenotype], generation: int
    ) -> list[Stage2Result]:
        """Evaluate one deterministic Stage-2 batch on isolated physical GPUs.

        The scheduler never changes candidate order: subprocess completion order
        is ignored and results are reloaded in the Stage-1 order.  Each job owns
        a distinct phenotype cache, CUDA context, calibration directory and
        engine directory.  When no GPU pool is configured this deliberately
        falls back to the established sequential evaluator.
        """

        if len(genotypes) <= 1 or not self.stage2_gpus:
            return [self(genotype, generation) for genotype in genotypes]
        if len(set(self.stage2_gpus)) != len(self.stage2_gpus):
            raise RuntimeError("ga_stage2_gpu_pool_contains_duplicates")

        results: list[Stage2Result | None] = [None] * len(genotypes)
        jobs: list[tuple[int, Path, Path, int, str]] = []
        for index, genotype in enumerate(genotypes):
            identity = phenotype_identity(genotype, self.space)
            complete_hash = str(identity["complete_phenotype_hash"])
            cache = self.root / f"ga/stage2_cache/budget_{self.label}/{complete_hash}"
            result_path = cache / "stage2_result.json"
            generation_dir = self.root / (
                f"ga/budget_{self.label}/seed_{self.seed}/generation_{generation:02d}/"
                f"candidate_{complete_hash}"
            )
            generation_dir.mkdir(parents=True, exist_ok=True)
            if result_path.is_file():
                results[index] = self._load_cached_result(result_path)
                atomic_write(generation_dir / "cache_reference.json", {
                    "stage2_cache": str(cache), "reused": True,
                    "complete_phenotype_hash": complete_hash,
                })
                print(json.dumps({"stage2": "cache_hit", "budget": self.label,
                                  "seed": self.seed, "generation": generation,
                                  "candidate": complete_hash}), flush=True)
                continue
            gpu = self.stage2_gpus[len(jobs) % len(self.stage2_gpus)]
            job_dir = generation_dir / "worker"
            job_dir.mkdir(parents=True, exist_ok=True)
            job_path = job_dir / "request.json"
            log_path = job_dir / "worker.log"
            atomic_write(job_path, {
                "root": str(self.root), "label": self.label, "seed": self.seed,
                "generation": generation, "genotype": genotype.to_dict(),
                "complete_phenotype_hash": complete_hash,
                "phenotype": canonicalize_candidate(genotype, self.space).to_dict(),
                "domains": [domain.to_dict() for domain in self.space.pruning_domains],
                "qkv_paths": list(self.qkv_paths), "request": dict(self.request),
                "evaluation_manifest": str(self.evaluation_manifest),
                "evaluation_frames": self.evaluation_frames,
                "evaluation_warmup_frames": self.evaluation_warmup_frames,
                "evaluation_protocol": self.evaluation_protocol,
                "plugin": str(self.plugin), "tensorrt_root": str(self.tensorrt_root),
                "physical_gpu": gpu,
            })
            jobs.append((index, job_path, log_path, gpu, complete_hash))

        def launch(job: tuple[int, Path, Path, int, str]) -> tuple[int, int, str]:
            index, job_path, log_path, gpu, complete_hash = job
            environment = dict(os.environ)
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
            command = [
                sys.executable,
                str(REPO / "scripts/run_v2xvit_ga_stage2_worker.py"),
                "--request", str(job_path),
            ]
            with log_path.open("ab", buffering=0) as log:
                completed = subprocess.run(
                    command, env=environment, stdout=log, stderr=subprocess.STDOUT,
                    check=False,
                )
            return index, int(completed.returncode), complete_hash

        if jobs:
            print(json.dumps({
                "stage2": "parallel_batch_start", "budget": self.label,
                "seed": self.seed, "generation": generation,
                "candidate_count": len(jobs),
                "physical_gpus": [job[3] for job in jobs],
            }), flush=True)
            gpu_queues: dict[int, list[tuple[int, Path, Path, int, str]]] = {}
            for job in jobs:
                gpu_queues.setdefault(job[3], []).append(job)

            def launch_queue(
                queue: list[tuple[int, Path, Path, int, str]],
            ) -> list[tuple[int, int, str]]:
                # One worker at a time per physical GPU.  Different GPUs run
                # concurrently, while calibration/build/evaluation on one GPU
                # can never overlap another candidate from this scheduler.
                return [launch(job) for job in queue]

            with ThreadPoolExecutor(max_workers=len(gpu_queues)) as pool:
                grouped = list(pool.map(launch_queue, gpu_queues.values()))
            completed_jobs = [item for group in grouped for item in group]
            completed_jobs.sort(key=lambda item: item[0])
            for index, returncode, complete_hash in completed_jobs:
                cache = self.root / f"ga/stage2_cache/budget_{self.label}/{complete_hash}"
                result_path = cache / "stage2_result.json"
                # A worker always persists its fail-closed result, including
                # the actionable exception.  Only synthesize a generic crash
                # when the subprocess produced no result at all.
                if not result_path.is_file():
                    genotype = genotypes[index]
                    failure = Stage2Result(
                        complete_hash, genotype, "failed", None, None, False, 0, 0,
                        {"generation": generation,
                         "failure": f"parallel_stage2_worker_no_result:{returncode}",
                         "precision_fallback": False,
                         "stage2_evaluation_frames": self.evaluation_frames,
                         "evaluation_protocol": self.evaluation_protocol},
                    )
                    atomic_write(result_path, stage2_payload(failure))
                results[index] = stage2_from_payload(json.loads(result_path.read_text()))
            print(json.dumps({
                "stage2": "parallel_batch_complete", "budget": self.label,
                "seed": self.seed, "generation": generation,
                "candidate_count": len(jobs),
            }), flush=True)
        if any(result is None for result in results):
            raise RuntimeError("ga_stage2_parallel_result_missing")
        return [result for result in results if result is not None]

    def __call__(self, genotype: CandidateGenotype, generation: int) -> Stage2Result:
        identity = phenotype_identity(genotype, self.space)
        complete_hash = identity["complete_phenotype_hash"]
        cache = self.root / f"ga/stage2_cache/budget_{self.label}/{complete_hash}"
        result_path = cache / "stage2_result.json"
        generation_dir = self.root / (
            f"ga/budget_{self.label}/seed_{self.seed}/generation_{generation:02d}/"
            f"candidate_{complete_hash}"
        )
        generation_dir.mkdir(parents=True, exist_ok=True)
        if result_path.is_file():
            result = self._load_cached_result(result_path)
            atomic_write(generation_dir / "cache_reference.json", {
                "stage2_cache": str(cache), "reused": True,
                "complete_phenotype_hash": complete_hash,
            })
            print(json.dumps({"stage2": "cache_hit", "budget": self.label,
                              "seed": self.seed, "generation": generation,
                              "candidate": complete_hash}), flush=True)
            return result
        print(json.dumps({"stage2": "start", "budget": self.label,
                          "seed": self.seed, "generation": generation,
                          "candidate": complete_hash}), flush=True)
        cache.mkdir(parents=True, exist_ok=True)
        if not (cache / "genotype.json").is_file():
            atomic_write(cache / "genotype.json", genotype.to_dict())
        phenotype = canonicalize_candidate(genotype, self.space)
        try:
            physical = materialize_unified_widths(
                self.model,
                self.identity["cnn_units"],
                self.space.pruning_domains,
                genotype.pruning_width_genes,
                model_name="lidar_v2xvit",
            )
            atomic_write(cache / "physical_report.json", physical.report.to_dict())
            if not physical.report.passed:
                raise RuntimeError("physical_materialization_failed")
            has_int8 = any(value == "INT8" for value in genotype.precision_genes.values())
            export_dir = cache / "JMIX-FRESH"
            export_result_path = cache / "export_result.json"
            if export_result_path.is_file() and (export_dir / "candidate.plan").is_file():
                exported = json.loads(export_result_path.read_text())
            else:
                if export_dir.exists():
                    incomplete = cache / "incomplete_export_before_resume"
                    if incomplete.exists():
                        raise RuntimeError("multiple_incomplete_stage2_export_attempts")
                    export_dir.rename(incomplete)
                export_dir.mkdir(parents=True, exist_ok=False)
                exported = _export_candidate(
                    export_dir,
                    physical.model,
                    self.adapter,
                    self.representative,
                    self.hypes,
                    phenotype,
                    complete_hash,
                    physical.report.structure_hash,
                    build_engine=True,
                    tensorrt_root=self.tensorrt_root,
                    plugin=self.plugin,
                    calibration_frames=200 if has_int8 else 0,
                    qkv_paths=self.qkv_paths,
                    fixed_k_override=int(self.request["fixed_k"]),
                    physical_gpu_id=self.physical_gpu,
                )
                atomic_write(export_result_path, exported)
            exact, acceptance = precision_realized_exact(export_dir)
            engine_path = export_dir / "candidate.plan"
            if not exported.get("passed") or not exact or not engine_path.is_file():
                result = Stage2Result(
                    complete_hash, genotype, "deployment_invalid", None, None,
                    exact, 0, 0,
                    {"generation": generation, "export": exported,
                     "precision_acceptance_status": acceptance.get("status"),
                     "precision_fallback": False,
                     "stage2_evaluation_frames": self.evaluation_frames,
                     "evaluation_protocol": self.evaluation_protocol},
                )
            else:
                evaluation = evaluate_v2xvit_engine_modelopt(
                    engine_path=engine_path,
                    model_config=self.request["model_config"],
                    heal_root=self.request["heal_root"],
                    output_dir=cache / "stage2_fixed500",
                    tensorrt_root=self.tensorrt_root,
                    plugin_path=self.plugin,
                    eval_manifest_path=self.evaluation_manifest,
                    physical_gpu_id=self.physical_gpu,
                    fixed_k=int(self.request["fixed_k"]),
                    max_agents=int(self.request["max_agents"]),
                    num_frames=self.evaluation_frames,
                    warmup_frames=self.evaluation_warmup_frames,
                    latency_rounds=1,
                    dataloader_num_workers=8,
                )
                ok = bool(
                    evaluation.get("status") == "ok"
                    and int(evaluation.get("num_evaluated_frames", -1))
                    == self.evaluation_frames
                    and int(evaluation.get("num_skipped_frames", -1)) == 0
                )
                result = Stage2Result(
                    complete_hash,
                    genotype,
                    "ok" if ok else "fixed500_failed",
                    float(evaluation["mAP"]) if ok else None,
                    float(evaluation["forward_p50_ms"]) if ok else None,
                    exact,
                    int(evaluation.get("num_evaluated_frames", 0)),
                    int(evaluation.get("num_skipped_frames", 0)),
                    {
                        "generation": generation,
                        "engine_sha256": sha256(engine_path),
                        "engine_path": str(engine_path),
                        "physical_structure_hash": physical.report.structure_hash,
                        "calibration_required": has_int8,
                        "calibration_manifest": str(export_dir / "calibration_manifest.json"),
                        "fresh_train200": has_int8,
                        "precision_fallback": False,
                        "stage2_fixed500_result": evaluation,
                        "stage2_evaluation_frames": self.evaluation_frames,
                        "evaluation_protocol": self.evaluation_protocol,
                    },
                )
            del physical
            torch.cuda.empty_cache()
        except Exception as exc:  # fail closed and allow the generation to continue
            result = Stage2Result(
                complete_hash, genotype, "failed", None, None, False, 0, 0,
                {"generation": generation, "failure": f"{type(exc).__name__}:{exc}",
                 "precision_fallback": False,
                 "stage2_evaluation_frames": self.evaluation_frames,
                 "evaluation_protocol": self.evaluation_protocol},
            )
        atomic_write(result_path, stage2_payload(result))
        atomic_write(generation_dir / "cache_reference.json", {
            "stage2_cache": str(cache), "reused": False,
            "complete_phenotype_hash": complete_hash,
        })
        print(json.dumps({"stage2": "complete", "budget": self.label,
                          "seed": self.seed, "generation": generation,
                          "candidate": complete_hash, "status": result.status,
                          "mAP": result.map, "p50_ms": result.p50_ms}), flush=True)
        return result

    def validate_generation_winner(
        self,
        screening_result: Stage2Result,
        generation: int,
        *,
        evaluation_frames: int = 500,
        evaluation_warmup_frames: int = 200,
        evaluation_protocol: str = "generation_winner_fixed500_warmup200",
    ) -> Stage2Result:
        """Re-evaluate one already-built generation winner without rebuilding it.

        Top-5 Stage-2 screening and generation-winner validation deliberately
        use separate result namespaces.  The engine is immutable and is reused
        only after its phenotype hash, file SHA and requested/realized audit
        have been checked.  This avoids repeating physical export/calibration/
        TensorRT build while preventing a 300-frame screening result from being
        mistaken for the authoritative 500-frame result.
        """

        complete_hash = str(screening_result.complete_phenotype_hash)
        expected_hash = str(
            phenotype_identity(screening_result.genotype, self.space)[
                "complete_phenotype_hash"
            ]
        )
        if complete_hash != expected_hash:
            raise RuntimeError(
                "generation_winner_phenotype_hash_mismatch:"
                f"{complete_hash}!={expected_hash}"
            )
        destination = self.root / (
            f"ga/generation_winner_validation/budget_{self.label}/{complete_hash}"
        )
        result_path = destination / "generation_winner_result.json"
        if result_path.is_file():
            result = stage2_from_payload(json.loads(result_path.read_text()))
            frames = int(result.metadata.get("stage2_evaluation_frames", -1))
            warmup = int(result.metadata.get("stage2_evaluation_warmup_frames", -1))
            protocol = str(result.metadata.get("evaluation_protocol", ""))
            if (
                frames != int(evaluation_frames)
                or warmup != int(evaluation_warmup_frames)
                or protocol != str(evaluation_protocol)
            ):
                raise RuntimeError(
                    "generation_winner_cache_protocol_mismatch:"
                    f"frames={frames}:warmup={warmup}:protocol={protocol}"
                )
            return result

        destination.mkdir(parents=True, exist_ok=True)
        try:
            if not screening_result.deployable:
                raise RuntimeError("generation_winner_screening_not_deployable")
            engine_text = str(screening_result.metadata.get("engine_path", ""))
            if not engine_text:
                raise RuntimeError("generation_winner_engine_path_missing")
            engine_path = Path(engine_text)
            if not engine_path.is_file():
                raise RuntimeError(f"generation_winner_engine_missing:{engine_path}")
            expected_engine_hash = str(
                screening_result.metadata.get("engine_sha256", "")
            )
            realized_engine_hash = sha256(engine_path)
            if expected_engine_hash and expected_engine_hash != realized_engine_hash:
                raise RuntimeError(
                    "generation_winner_engine_hash_mismatch:"
                    f"{expected_engine_hash}!={realized_engine_hash}"
                )
            exact, acceptance = precision_realized_exact(engine_path.parent)
            if not exact:
                raise RuntimeError(
                    "generation_winner_precision_not_exact:"
                    f"{acceptance.get('status')}"
                )
            evaluation = evaluate_v2xvit_engine_modelopt(
                engine_path=engine_path,
                model_config=self.request["model_config"],
                heal_root=self.request["heal_root"],
                output_dir=destination / "evaluation_fixed500",
                tensorrt_root=self.tensorrt_root,
                plugin_path=self.plugin,
                eval_manifest_path=self.evaluation_manifest,
                physical_gpu_id=self.physical_gpu,
                fixed_k=int(self.request["fixed_k"]),
                max_agents=int(self.request["max_agents"]),
                num_frames=int(evaluation_frames),
                warmup_frames=int(evaluation_warmup_frames),
                latency_rounds=1,
                dataloader_num_workers=8,
            )
            ok = bool(
                evaluation.get("status") == "ok"
                and int(evaluation.get("num_evaluated_frames", -1))
                == int(evaluation_frames)
                and int(evaluation.get("num_skipped_frames", -1)) == 0
            )
            result = Stage2Result(
                complete_hash,
                screening_result.genotype,
                "ok" if ok else "generation_winner_fixed500_failed",
                float(evaluation["mAP"]) if ok else None,
                float(evaluation["forward_p50_ms"]) if ok else None,
                exact,
                int(evaluation.get("num_evaluated_frames", 0)),
                int(evaluation.get("num_skipped_frames", 0)),
                {
                    "generation": int(generation),
                    "engine_path": str(engine_path),
                    "engine_sha256": realized_engine_hash,
                    "screening_evaluation_frames": int(
                        screening_result.metadata.get(
                            "stage2_evaluation_frames", -1
                        )
                    ),
                    "screening_result": stage2_payload(screening_result),
                    "stage2_evaluation_frames": int(evaluation_frames),
                    "stage2_evaluation_warmup_frames": int(
                        evaluation_warmup_frames
                    ),
                    "evaluation_protocol": str(evaluation_protocol),
                    "generation_winner_fixed500_result": evaluation,
                    "precision_acceptance_status": acceptance.get("status"),
                    "precision_fallback": False,
                    "engine_rebuilt_for_validation": False,
                },
            )
        except Exception as exc:
            result = Stage2Result(
                complete_hash,
                screening_result.genotype,
                "failed",
                None,
                None,
                False,
                0,
                0,
                {
                    "generation": int(generation),
                    "failure": f"{type(exc).__name__}:{exc}",
                    "stage2_evaluation_frames": int(evaluation_frames),
                    "stage2_evaluation_warmup_frames": int(
                        evaluation_warmup_frames
                    ),
                    "evaluation_protocol": str(evaluation_protocol),
                    "precision_fallback": False,
                    "engine_rebuilt_for_validation": False,
                },
            )
        atomic_write(result_path, stage2_payload(result))
        return result


def ensure_greedy_anchor_fixed50(
    *, root: Path, label: str, genotype: CandidateGenotype, complete_hash: str,
    request: Mapping[str, Any], fixed50_manifest: Path, tensorrt_root: Path,
    plugin: Path, physical_gpu: int,
) -> Stage2Result:
    destination = root / f"ga/greedy_anchors/budget_{label}/fixed50"
    result_path = destination / "anchor_result.json"
    if result_path.is_file():
        return stage2_from_payload(json.loads(result_path.read_text()))
    engine = root / f"engines/greedy_exact_winners/budget_{label}/JMIX-FRESH/candidate.plan"
    exact, _acceptance = precision_realized_exact(engine.parent)
    evaluation = evaluate_v2xvit_engine_modelopt(
        engine_path=engine,
        model_config=request["model_config"], heal_root=request["heal_root"],
        output_dir=destination / "evaluation", tensorrt_root=tensorrt_root,
        plugin_path=plugin, eval_manifest_path=fixed50_manifest,
        physical_gpu_id=physical_gpu, fixed_k=int(request["fixed_k"]),
        max_agents=int(request["max_agents"]), num_frames=50,
        warmup_frames=20, latency_rounds=1, dataloader_num_workers=8,
    )
    ok = bool(
        evaluation.get("status") == "ok"
        and int(evaluation.get("num_evaluated_frames", -1)) == 50
        and int(evaluation.get("num_skipped_frames", -1)) == 0
        and exact
    )
    result = Stage2Result(
        complete_hash, genotype, "ok" if ok else "greedy_anchor_failed",
        float(evaluation["mAP"]) if ok else None,
        float(evaluation["forward_p50_ms"]) if ok else None,
        exact, int(evaluation.get("num_evaluated_frames", 0)),
        int(evaluation.get("num_skipped_frames", 0)),
        {"greedy_anchor": True, "engine": str(engine), "fixed50_result": evaluation},
    )
    atomic_write(result_path, stage2_payload(result))
    if not result.deployable:
        raise RuntimeError(f"greedy_anchor_fixed50_failed:budget_{label}")
    return result


def run(args: argparse.Namespace) -> int:
    root = args.output_root.resolve()
    if args.seed != 0:
        raise RuntimeError(f"formal_ga_single_seed_zero_required:{args.seed}")
    physical_gpus = (args.physical_gpu, *args.stage2_gpus)
    if any(gpu not in ALLOWED_PHYSICAL_GPUS for gpu in physical_gpus):
        raise RuntimeError(
            "formal_ga_physical_gpu_outside_allowed_pool:"
            f"requested={physical_gpus}:allowed={ALLOWED_PHYSICAL_GPUS}"
        )
    if len(set(args.stage2_gpus)) > 2:
        raise RuntimeError("formal_ga_stage2_gpu_pool_exceeds_two")
    pre_ga = json.loads((root / "reports/pre_ga_acceptance.json").read_text())
    if not pre_ga.get("formal_search_allowed") or pre_ga.get("blockers"):
        raise RuntimeError(
            f"formal_ga_pre_acceptance_not_passed:{pre_ga.get('blockers')}"
        )
    admission = json.loads((root / "reports/ga_budget_admission.json").read_text())
    labels = [f"{int(round(float(value) * 100)):03d}" for value in admission["ga_admissible_budgets"]]
    if labels != ["010"]:
        raise RuntimeError(f"formal_ga_requires_only_budget_010:{labels}")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"formal_ga_requires_one_visible_gpu:{torch.cuda.device_count()}")
    search_spec = json.loads((root / "reports/final_search_space_spec.json").read_text())
    taylor_manifest = json.loads((root / "reports/taylor_cache_manifest.json").read_text())
    atomic_write(root / "reports/formal_ga_config.json", {
        "target_bops": 0.10,
        "bops_tolerance_abs": 0.005,
        "population_size": 64,
        "offspring_size": 64,
        "survivor_size": 64,
        "seed": 0,
        "seed_count": 1,
        "executed_seeds": [0],
        "formal_generations": 10,
        "generation_zero_counted": False,
        "formal_generation_ids": list(range(1, 11)),
        "stage2_new_candidate_quota_per_generation": 5,
        "main_physical_gpu": int(args.physical_gpu),
        "stage2_physical_gpus": list(args.stage2_gpus),
        "stage2_serial_per_gpu": True,
        "search_space_schema_version": search_spec["search_space_schema_version"],
        "search_space_schema_hash": search_spec["schema_hash"],
        "taylor_cache_schema_version": taylor_manifest["schema_version"],
        "legal_av_profiles": pre_ga["legal_av_profiles"],
        "black_box_surrogate": False,
        "repair_enabled": False,
        "full1789": False,
    })
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    request = json.loads((root / "evaluation_fixed500/B0/evaluation_request.json").read_text())
    pools = json.loads((root / "ga/initial_populations.json").read_text())
    model, adapter, hypes, _ = _load("v2xvit", device)
    representative, _, _ = _multi_agent_validation_batch(adapter, hypes, device)
    train_manifest_path = REPO / "search/model_family/manifests/heal_lidar_v2xvit_train200_fixed_k.json"
    train200 = load_v2xvit_train_manifest(train_manifest_path)
    train32 = FrozenTrainPrefix(adapter=adapter, hypes=hypes, device=device,
                                manifest=train200, count=32)
    calibration_hash = json.loads((root / "reports/input_provenance.json").read_text())["taylor_manifest_hash"]
    identity = _build_full_space(model, adapter, hypes, representative)
    print("[formal-ga] collect train32 Fisher E[g^2]", flush=True)
    formal = _formal_space(
        model, adapter, hypes, representative, identity, calibration_hash,
        fisher_forward_fn=lambda current_model, batch: _type_coverage_forward(adapter, current_model, batch),
        fisher_batches=train32,
    )
    print("[formal-ga] collect train32 functional gate Taylor", flush=True)
    gate32, gate_mapping = collect_functional_gate_scores_multi(
        model, formal["space"].pruning_domains,
        forward_fn=lambda current_model, batch: _type_coverage_forward(adapter, current_model, batch),
        loss_fn=adapter.compute_task_loss, calibration_batches=train32,
    )
    replay_domains = rerank_domains_by_gate_scores(
        formal["space"].pruning_domains, gate32
    )
    # The current contract is rebuilt after AV/merge closure.  Old six-budget
    # trajectories and their nested masks are intentionally invalidated.
    domains = replay_domains
    space = replace(
        formal["space"], pruning_domains=domains,
        pruning_unit_ids=[unit for domain in domains for unit in domain.ordered_unit_ids],
    )
    baseline = _baseline_candidate(space)
    transformer_units = taylor_units_from_transformer_precision(
        model, formal["components"].precision_units,
        active_module_paths=formal["active_paths"],
    )
    activation_units, group_to_units = build_activation_units(model, space, transformer_units)
    print("[formal-ga] collect train32 activation Q/DQ Taylor", flush=True)
    activation32 = collect_activation_taylor_cache_multi(
        model, activation_units, group_to_units,
        forward_fn=lambda current_model, batch: _type_coverage_forward(adapter, current_model, batch),
        loss_fn=adapter.compute_task_loss, calibration_batches=train32,
    )
    weight32 = JointWeightTaylorProxy(
        model, statistics=formal["fisher"], unit_to_parameter_slices=formal["slices"], strict=True,
    )
    qkv_paths = tuple(
        path for spec in formal["components"].attention_instances
        for path in (spec.q_projection_paths + spec.k_projection_paths)
    )
    atomic_write(root / "ga/formal_cache_audit.json", {
        "sample_count": 32, "calibration_hash": calibration_hash,
        "gate_domain_count": len(gate32), "gate_mapping_count": len(gate_mapping),
        "activation_observer_count": len(activation32.mapping),
        "fisher_formula": "E[g^2]", "search_loop_collection_calls": 0,
    })

    seed_summaries: list[dict[str, Any]] = []
    all_budget_results: dict[str, Any] = {}
    for label in labels:
        target = TARGETS[label]
        exact = json.loads((root / f"greedy/budget_{label}/exact_winner.json").read_text())
        greedy_genotype = CandidateGenotype.from_dict(exact["genotype"])
        validate_genotype_schema(greedy_genotype, space)
        greedy_identity = phenotype_identity(greedy_genotype, space)
        greedy_phenotype = canonicalize_candidate(greedy_genotype, space)
        if greedy_phenotype.to_dict() != exact["phenotype"]:
            raise RuntimeError(
                f"greedy_anchor_physical_phenotype_drift:budget_{label}"
            )
        if greedy_identity["complete_phenotype_hash"] != exact["candidate_hash"]:
            print(json.dumps({
                "budget": label,
                "greedy_anchor_source_hash": exact["candidate_hash"],
                "greedy_anchor_current_trace_hash": greedy_identity[
                    "complete_phenotype_hash"
                ],
                "physical_phenotype_exact": True,
                "candidate_hash_rebased_to_current_trace": True,
            }), flush=True)
        greedy = ensure_greedy_anchor_fixed50(
            root=root, label=label, genotype=greedy_genotype,
            complete_hash=greedy_identity["complete_phenotype_hash"], request=request,
            fixed50_manifest=args.fixed50_manifest, tensorrt_root=args.tensorrt_root,
            plugin=args.plugin, physical_gpu=args.physical_gpu,
        )
        budget_real: dict[str, Stage2Result] = {greedy.complete_phenotype_hash: greedy}
        budget_seed_rows = []
        for seed in FORMAL_SEEDS:
            config = StrictGAConfig(target_bops_retention=target, random_seed=seed)
            stage1 = UnifiedTaylorStage1Evaluator(
                space, baseline=baseline,
                structure_proxy=FunctionalGateTaylorProxy(gate32),
                weight_proxy=weight32, activation_cache=activation32,
                bops_evaluator=formal["bops"].evaluate_breakdown,
                size_evaluator=formal["size"].evaluate_breakdown,
                target=target, tolerance_abs=0.005,
            )
            initial = [
                candidate_from_state(row, f"budget_{label}_initial_pool")
                for row in pools[label]["candidates"]
            ]
            # Seed-specific deterministic ordering without changing candidates.
            random.Random(seed).shuffle(initial)
            real = RealStage2Evaluator(
                root=root, label=label, seed=seed, model=model, adapter=adapter,
                hypes=hypes, representative=representative, identity=identity,
                space=space, qkv_paths=qkv_paths, request=request,
                evaluation_manifest=args.fixed50_manifest,
                evaluation_frames=50,
                evaluation_warmup_frames=20,
                evaluation_protocol="legacy_fixed50_screening",
                plugin=args.plugin,
                tensorrt_root=args.tensorrt_root, physical_gpu=args.physical_gpu,
                stage2_gpus=args.stage2_gpus,
            )
            runner = StrictStage12V3Runner(
                space, config, stage1_evaluator=stage1, stage2_evaluator=real,
            )
            seed_root = root / f"ga/budget_{label}/seed_{seed}"
            seed_root.mkdir(parents=True, exist_ok=True)
            completed_seed = seed_root / "seed_summary.json"
            if completed_seed.is_file():
                row = json.loads(completed_seed.read_text())
                if int(row.get("completed_evolution_generations", -1)) != 10:
                    raise RuntimeError(f"incomplete_seed_summary:budget_{label}:seed_{seed}")
                for item in row.get("global_anchors", []):
                    restored = stage2_from_payload(item)
                    budget_real[restored.complete_phenotype_hash] = restored
                seed_summaries.append(row)
                budget_seed_rows.append(row)
                continue

            def callback(record: Mapping[str, Any]) -> None:
                generation = int(record["generation"])
                destination = seed_root / f"generation_{generation:02d}"
                destination.mkdir(parents=True, exist_ok=True)
                atomic_write(destination / "generation_summary.json", dict(record))
                print(json.dumps({"budget": label, "seed": seed,
                                  "generation": generation,
                                  "stage2_new_candidate_count": record.get("stage2_new_candidate_count", 0),
                                  "generation_winner_hash": record.get("generation_winner_hash")}),
                      flush=True)

            result = runner.run(initial, greedy_anchor=greedy, generation_callback=callback)
            for identity_hash, row in result["evaluated"].items():
                budget_real[identity_hash] = row
            seed_winner = best_real_candidate(list(result["evaluated"].values()), greedy)
            row = {
                "budget": target, "seed": seed,
                "termination_reason": result["termination_reason"],
                "completed_evolution_generations": result["completed_evolution_generations"],
                "generation_zero_counted": result["generation_zero_counted"],
                "greedy_anchor_hash": greedy.complete_phenotype_hash,
                "seed_winner": stage2_payload(seed_winner),
                "global_anchors": [stage2_payload(item) for item in result["anchors"].unique()],
                "stage2_real_evaluation_count": len(result["evaluated"]) - 1,
                "repair_counts": result["formal_ga_repair_counts"],
            }
            atomic_write(seed_root / "seed_summary.json", row)
            seed_summaries.append(row)
            budget_seed_rows.append(row)
            print(json.dumps({"budget": label, "seed": seed,
                              "completed_generations": row["completed_evolution_generations"],
                              "seed_winner": seed_winner.complete_phenotype_hash}), flush=True)
        final = best_real_candidate(list(budget_real.values()), greedy)
        all_budget_results[label] = {
            "budget": target,
            "greedy_anchor": stage2_payload(greedy),
            "seed_summaries": budget_seed_rows,
            "global_real_anchors": [stage2_payload(row) for row in budget_real.values()],
            "final_winner": stage2_payload(final),
            "ga_improved_greedy": final.complete_phenotype_hash != greedy.complete_phenotype_hash,
        }
        atomic_write(root / f"ga/budget_{label}/budget_summary.json", all_budget_results[label])

    atomic_write(root / "reports/ga_formal_results.json", {
        "configuration": {"seeds": len(FORMAL_SEEDS), "seed_ids": list(FORMAL_SEEDS),
                          "population_size": 64, "offspring_size": 64,
                          "generations": 10, "generation_zero_counted": False,
                          "stage2_new_candidate_quota": 5,
                          "stage2_parallel": bool(args.stage2_gpus),
                          "stage2_physical_gpus": list(args.stage2_gpus),
                          "stage2_result_merge_order": "stage1_rank_order",
                          "formal_fixed500_and_latency_parallel": False},
        "budgets": all_budget_results,
        "seed_summaries": seed_summaries,
    })
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, default=2)
    parser.add_argument(
        "--stage2-gpus", type=lambda value: tuple(
            int(item.strip()) for item in value.split(",") if item.strip()
        ), default=(),
        help="Comma-separated physical GPU pool for deterministic Stage-2 batches",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fixed50-manifest", type=Path, required=True)
    parser.add_argument("--plugin", type=Path, required=True)
    parser.add_argument("--tensorrt-root", type=Path,
                        default=Path("/home/lixingfeng/UniAD_examine/TensorRT-10.9_x86_cu118"))
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
