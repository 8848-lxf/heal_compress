#!/usr/bin/env python3
"""Build per-budget P-only/Q-only controls and repeat full1789 three times.

P-only uses the exact GA-final physical ONNX and forces every weighted and
derived boundary to FP32. Q-only keeps the original physical model while
replaying the GA-final precision genotype with fresh train200 calibration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from quantization.config import TensorRTBuildConfig
from quantization.types import (
    CanonicalPrecisionEntry,
    CanonicalPrecisionMappingResult,
    stable_json_hash,
)
from scripts.audit_heal_transformer_search_models import _load
from scripts.run_v2xvit_greedy005_full import _baseline_candidate, _build_full_space
from scripts.run_v2xvit_greedy005_stage2 import _export_candidate
from scripts.run_v2xvit_sixbudget_full1789_repeat3 import (
    LABELS,
    _parse_labels,
    _read,
    _sha256,
    _source_root,
    _write,
)
from scripts.smoke_transformer_unified_search import _multi_agent_validation_batch
from search.candidate import CandidateGenotype
from search.canonicalization import canonicalize_candidate
from search.model_family.evaluation import evaluate_v2xvit_engine_modelopt
from search.stage2.transformer_precision_export import (
    audit_trt_attention_fp32_contract,
)
from search.stage2.trt_modelopt import build_engine_modelopt


def all_fp32_mapping(payload: Mapping[str, Any]) -> CanonicalPrecisionMappingResult:
    entries = []
    for raw in payload["entries"]:
        row = dict(raw)
        row.update(
            {
                "requested_precision": "fp32",
                "realized_request_precision": "fp32",
                "realized_output_precision": "fp32",
                "fallback_reason": "",
                "protected_precision": "",
            }
        )
        entries.append(CanonicalPrecisionEntry(**row))
    auxiliary = {str(key): "fp32" for key in payload.get("auxiliary_layer_precisions", {})}
    outputs = {str(key): "fp32" for key in payload.get("auxiliary_layer_output_types", {})}
    profile = {entry.module_path: "fp32" for entry in entries}
    return CanonicalPrecisionMappingResult(
        entries=entries,
        profile_id="v2xvit_p_only_all_fp32",
        profile_hash=stable_json_hash(profile),
        origin_map_hash=str(payload.get("origin_map_hash", "")),
        policy_version="v2xvit-p-only-strongly-typed-fp32-v1",
        auxiliary_layer_precisions=auxiliary,
        auxiliary_layer_output_types=outputs,
    )


def q_only_genotype(
    baseline: CandidateGenotype,
    ga_final: CandidateGenotype,
    precision_gene_ids: tuple[str, ...],
) -> CandidateGenotype:
    expected = set(precision_gene_ids)
    actual = set(ga_final.precision_genes)
    if expected != actual:
        raise RuntimeError(
            f"q_only_precision_schema_mismatch:missing={sorted(expected-actual)}:"
            f"unknown={sorted(actual-expected)}"
        )
    return CandidateGenotype(
        pruning_width_genes=dict(baseline.pruning_width_genes),
        precision_genes=dict(ga_final.precision_genes),
        meta={"diagnostic_control": True, "control": "Q-only"},
    )


def _candidate_source(root: Path, label: str, candidate_hash: str) -> Path:
    return root / f"ga/stage2_cache/budget_{label}/{candidate_hash}"


def _build_p_only(
    *,
    source: Path,
    destination: Path,
    tensorrt_root: Path,
    plugin: Path,
    physical_gpu: int,
) -> dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=False)
    source_deploy = source / "JMIX-FRESH"
    onnx = source_deploy / "physical_fp32.onnx"
    mapping_path = source_deploy / "canonical_precision_mapping.json"
    request_path = source_deploy / "engine_build/trt_build_request.json"
    qk_path = source_deploy / "onnx_attention_fp32_audit.json"
    for path in (onnx, mapping_path, request_path, qk_path):
        if not path.is_file():
            raise RuntimeError(f"p_only_source_artifact_missing:{path}")
    source_request = _read(request_path)
    mapping = all_fp32_mapping(_read(mapping_path))
    trtexec = tensorrt_root / "bin/trtexec"
    if not trtexec.is_file():
        trtexec = tensorrt_root / "targets/x86_64-linux-gnu/bin/trtexec"
    build = build_engine_modelopt(
        qdq_onnx=onnx,
        engine_path=destination / "candidate.plan",
        precision_mapping=mapping,
        build_config=TensorRTBuildConfig(
            trtexec_path=trtexec,
            plugin_path=plugin,
            workspace_mib=4096,
            timeout_seconds=3600,
            no_tf32=True,
            skip_inference=True,
            export_layer_info=True,
            strongly_typed=True,
            enable_fp16=False,
            enable_int8=False,
            policy_version="v2xvit-p-only-strongly-typed-fp32-v1",
        ),
        physical_snapshot=source_request["physical_snapshot"],
        output_dir=destination / "engine_build",
        tensorrt_root=tensorrt_root,
        conda_env="modelopt",
        gpu_id=physical_gpu,
    )
    precision = dict(build.get("precision_realization_validation") or {})
    structure = dict(build.get("engine_structure_validation") or {})
    if not (
        build.get("status") == "ok"
        and precision.get("passed")
        and structure.get("passed")
        and not precision.get("mismatches")
        and int(precision.get("unresolved_layer_count", 0)) == 0
    ):
        raise RuntimeError(f"p_only_engine_build_failed:{build}")
    qk = audit_trt_attention_fp32_contract(
        destination / "engine_build/engine_layer_info.json", _read(qk_path)
    )
    if not qk.get("passed"):
        raise RuntimeError(f"p_only_qk_audit_failed:{qk}")
    report = {
        "control": "P-only",
        "diagnostic_control": True,
        "source_physical_onnx": str(onnx),
        "source_physical_onnx_sha256": _sha256(onnx),
        "candidate_plan": str(destination / "candidate.plan"),
        "engine_sha256": _sha256(destination / "candidate.plan"),
        "engine_size_bytes": (destination / "candidate.plan").stat().st_size,
        "requested_realized_exact": True,
        "precision_counts": {"FP32": len(mapping.entries), "FP16": 0, "INT8": 0},
        "calibration_required": False,
        "build": build,
        "qk_fp32_audit": qk,
    }
    _write(destination / "control_report.json", report)
    return report


def _summary(
    values: list[dict[str, Any]], *, repeat_count: int = 3
) -> dict[str, Any]:
    if len(values) != int(repeat_count):
        raise RuntimeError(f"pq_repeat_count_mismatch:{len(values)}")
    output: dict[str, Any] = {"repetitions": values}
    for metric in (
        "AP@0.3", "AP@0.5", "AP@0.7", "mAP", "forward_p50_ms",
        "forward_p90_ms", "forward_p99_ms",
    ):
        rows = [float(value[metric]) for value in values]
        output[f"{metric}_mean"] = statistics.fmean(rows)
        output[f"{metric}_std"] = statistics.stdev(rows)
    return output


def run(args: argparse.Namespace) -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible != str(int(args.physical_gpu)):
        raise RuntimeError(
            f"v2xvit_pq_only_gpu_binding_mismatch:{visible}:"
            f"{args.physical_gpu}"
        )
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"v2xvit_pq_only_requires_one_visible_gpu:{torch.cuda.device_count()}"
        )
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    for name in ("provenance", "engines", "evaluation_full1789", "reports", "logs"):
        (root / name).mkdir(parents=True, exist_ok=True)
    manifest = _read(args.full_manifest.resolve())
    if len(manifest["evaluation_frame_ids"]) != 1789:
        raise RuntimeError("pq_full1789_manifest_invalid")
    if len(manifest["warmup_frame_ids"]) < 200:
        raise RuntimeError("pq_full1789_warmup_invalid")
    request = _read(args.request_json.resolve())
    labels = _parse_labels(args.labels)
    if int(args.repetitions) < 2:
        raise ValueError("v2xvit_pq_repetitions_must_be_at_least_two")
    model, adapter, hypes, _ = _load("v2xvit", device)
    batch, _, _ = _multi_agent_validation_batch(adapter, hypes, device)
    identity = _build_full_space(model, adapter, hypes, batch)
    space = identity["space"]
    baseline = _baseline_candidate(space)
    qkv_paths = tuple(
        path
        for spec in identity["components"].attention_instances
        for path in (spec.q_projection_paths + spec.k_projection_paths)
    )
    controls: dict[str, dict[str, Any]] = {}
    expected_control_count = 4 * len(labels)
    for label in labels:
        source_root = _source_root(
            label, args.main_root.resolve(), args.frozen005_root.resolve()
        )
        summary_path = source_root / f"ga/budget_{label}/seed_0/budget_summary.json"
        summary = _read(summary_path)
        if int(summary.get("completed_evolution_generations", -1)) != int(
            args.formal_generations
        ):
            raise RuntimeError(
                f"v2xvit_pq_generation_count:{label}:"
                f"{summary.get('completed_evolution_generations')}:"
                f"{args.formal_generations}"
            )
        for method, key in (("Greedy", "greedy_anchor"), ("GA-final", "final_winner")):
            winner = summary[key]
            ga_final = CandidateGenotype.from_dict(winner["genotype"])
            candidate_hash = str(winner["complete_phenotype_hash"])
            source = _candidate_source(source_root, label, candidate_hash)

            p_dir = root / f"engines/budget_{label}/{method}/P-only"
            p_report_path = p_dir / "control_report.json"
            if p_report_path.is_file():
                p_report = _read(p_report_path)
            else:
                p_report = _build_p_only(
                    source=source,
                    destination=p_dir,
                    tensorrt_root=args.tensorrt_root.resolve(),
                    plugin=args.plugin.resolve(),
                    physical_gpu=args.physical_gpu,
                )
            controls[f"budget_{label}/{method}/P-only"] = {
                **p_report,
                "method": method,
                "budget": int(label) / 100.0,
                "source_candidate_hash": candidate_hash,
            }

            q_dir = root / f"engines/budget_{label}/{method}/Q-only"
            q_report_path = q_dir / "control_report.json"
            if q_report_path.is_file():
                q_report = _read(q_report_path)
            else:
                q_candidate = q_only_genotype(
                    baseline, ga_final, tuple(space.precision_gene_ids)
                )
                q_phenotype = canonicalize_candidate(q_candidate, space)
                q_hash = stable_json_hash(
                    {
                        "control": "Q-only",
                        "method": method,
                        "source_candidate_hash": candidate_hash,
                        "phenotype": q_phenotype.to_dict(),
                    }
                )
                q_dir.mkdir(parents=True, exist_ok=False)
                export = _export_candidate(
                    q_dir,
                    model,
                    adapter,
                    batch,
                    hypes,
                    q_phenotype,
                    q_hash,
                    stable_json_hash({"structure": "original", "control": "Q-only"}),
                    build_engine=True,
                    tensorrt_root=args.tensorrt_root.resolve(),
                    plugin=args.plugin.resolve(),
                    calibration_frames=(
                        200
                        if any(value == "INT8" for value in q_candidate.precision_genes.values())
                        else 0
                    ),
                    qkv_paths=qkv_paths,
                    fixed_k_override=int(request["fixed_k"]),
                    physical_gpu_id=args.physical_gpu,
                )
                if not export.get("passed") or not export.get("engine", {}).get("passed"):
                    raise RuntimeError(f"q_only_engine_build_failed:{label}:{method}:{export}")
                acceptance = _read(q_dir / "engine_build_acceptance.json")
                precision = dict(acceptance.get("precision_realization_validation") or {})
                functional = _read(q_dir / "functional_precision_trt_audit.json")
                av = _read(q_dir / "av_profile_trt_audit.json")
                if not (
                    precision.get("passed")
                    and not precision.get("mismatches")
                    and int(precision.get("unresolved_layer_count", 0)) == 0
                    and functional.get("passed")
                    and int(functional.get("unmapped_count", 0)) == 0
                    and int(functional.get("conflict_count", 0)) == 0
                    and av.get("passed")
                    and int(av.get("unmapped_count", 0)) == 0
                    and int(av.get("conflict_count", 0)) == 0
                    and int(av.get("fallback_count", 0)) == 0
                ):
                    raise RuntimeError(f"q_only_precision_audit_failed:{label}:{method}")
                q_report = {
                    "control": "Q-only",
                    "diagnostic_control": True,
                    "candidate_hash": q_hash,
                    "source_candidate_hash": candidate_hash,
                    "candidate_plan": str(q_dir / "candidate.plan"),
                    "engine_sha256": _sha256(q_dir / "candidate.plan"),
                    "engine_size_bytes": (q_dir / "candidate.plan").stat().st_size,
                    "requested_realized_exact": True,
                    "mutable_precision_counts": {
                        state: sum(value == state for value in q_candidate.precision_genes.values())
                        for state in ("FP32", "FP16", "INT8")
                    },
                    "calibration_required": any(
                        value == "INT8" for value in q_candidate.precision_genes.values()
                    ),
                    "export": export,
                }
                _write(q_report_path, q_report)
            controls[f"budget_{label}/{method}/Q-only"] = {
                **q_report,
                "method": method,
                "budget": int(label) / 100.0,
                "source_candidate_hash": candidate_hash,
            }
        _write(root / "reports/build_progress.json", {
            "status": "building",
            "completed_control_count": len(controls),
            "total_control_count": expected_control_count,
            "controls": controls,
        })
        torch.cuda.empty_cache()
    _write(root / "reports/build_summary.json", {
        "status": "complete", "controls": controls,
    })

    results: dict[str, list[dict[str, Any]]] = {}
    total_evaluations = expected_control_count * int(args.repetitions)
    for repeat in range(1, int(args.repetitions) + 1):
        for control_id, control in controls.items():
            destination = root / "evaluation_full1789" / f"repeat_{repeat}" / control_id
            result_path = destination / "evaluation.json"
            if result_path.is_file():
                evaluation = _read(result_path)
            else:
                evaluation = evaluate_v2xvit_engine_modelopt(
                    engine_path=control["candidate_plan"],
                    model_config=request["model_config"],
                    heal_root=request["heal_root"],
                    output_dir=destination,
                    tensorrt_root=args.tensorrt_root.resolve(),
                    plugin_path=request["plugin_path"],
                    eval_manifest_path=args.full_manifest.resolve(),
                    physical_gpu_id=args.physical_gpu,
                    fixed_k=int(request["fixed_k"]),
                    max_agents=int(request["max_agents"]),
                    num_frames=1789,
                    warmup_frames=200,
                    latency_rounds=1,
                    dataloader_num_workers=8,
                )
            if not (
                evaluation.get("status") == "ok"
                and int(evaluation.get("num_evaluated_frames", -1)) == 1789
                and int(evaluation.get("num_skipped_frames", -1)) == 0
                and str(evaluation.get("eval_manifest_hash"))
                == str(manifest.get("manifest_hash"))
            ):
                raise RuntimeError(f"pq_full1789_failed:{control_id}:repeat_{repeat}")
            row = {
                "repeat": repeat,
                "AP@0.3": float(evaluation["AP@0.3"]),
                "AP@0.5": float(evaluation["AP@0.5"]),
                "AP@0.7": float(evaluation["AP@0.7"]),
                "mAP": float(evaluation["mAP"]),
                "forward_p50_ms": float(evaluation["forward_p50_ms"]),
                "forward_p90_ms": float(evaluation["forward_p90_ms"]),
                "forward_p99_ms": float(evaluation["forward_p99_ms"]),
                "evaluated": 1789,
                "skipped": 0,
            }
            results.setdefault(control_id, [])
            existing = {int(item["repeat"]): item for item in results[control_id]}
            existing[repeat] = row
            results[control_id] = [existing[key] for key in sorted(existing)]
            _write(root / "reports/evaluation_progress.json", {
                "status": "running",
                "completed_evaluations": sum(len(value) for value in results.values()),
                "total_evaluations": total_evaluations,
                "last_control": control_id,
                "last_repeat": repeat,
                "results": results,
            })
            print(json.dumps({"event": "pq_full1789_complete", "control": control_id,
                              "repeat": repeat, "mAP": row["mAP"]}, sort_keys=True), flush=True)
    summaries = {
        control_id: _summary(rows, repeat_count=int(args.repetitions))
        for control_id, rows in results.items()
    }
    _write(root / f"reports/pq_only_full1789_repeat{int(args.repetitions)}.json", {
        "status": "complete",
        "manifest": str(args.full_manifest.resolve()),
        "manifest_hash": manifest.get("manifest_hash"),
        "controls": controls,
        "summaries": summaries,
    })
    _write(root / "reports/evaluation_progress.json", {
        "status": "complete",
        "completed_evaluations": total_evaluations,
        "total_evaluations": total_evaluations,
    })
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-root", type=Path, required=True)
    parser.add_argument("--frozen005-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--request-json", type=Path, required=True)
    parser.add_argument("--full-manifest", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--labels", default=",".join(LABELS))
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--formal-generations", type=int, default=10)
    parser.add_argument("--plugin", type=Path, required=True)
    parser.add_argument(
        "--tensorrt-root",
        type=Path,
        default=Path("/home/lixingfeng/UniAD_examine/TensorRT-10.9_x86_cu118"),
    )
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
