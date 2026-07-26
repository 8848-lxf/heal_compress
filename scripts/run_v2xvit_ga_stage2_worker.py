#!/usr/bin/env python3
"""Evaluate one V2X-ViT GA Stage-2 phenotype in an isolated GPU process."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.audit_heal_transformer_search_models import _load
from scripts.analyze_v2xvit_greedy005_bops_floor import _build_full_space
from scripts.run_v2xvit_formal_ga_gen10 import (
    atomic_write,
    precision_realized_exact,
    sha256,
    stage2_payload,
)
from scripts.run_v2xvit_greedy005_stage2 import _export_candidate
from scripts.smoke_transformer_unified_search import _multi_agent_validation_batch
from search.candidate import CandidateGenotype, CandidatePhenotype
from search.ga.stage12_v3 import Stage2Result
from search.model_family.evaluation import evaluate_v2xvit_engine_modelopt
from search.pruning_space.local_domains import (
    LocalPruningDomain,
    local_pruning_domain_from_dict,
)
from search.pruning_space.unified_physical_pruner import materialize_unified_widths


def _int_tuple_map(value: Mapping[str, Any]) -> dict[int, tuple[str, ...]]:
    return {
        int(key): tuple(str(item) for item in items)
        for key, items in value.items()
    }


def _nested_int_list_map(
    value: Mapping[str, Mapping[str, Any]],
) -> dict[int, dict[int, list[int]]]:
    return {
        int(width): {
            int(group): [int(item) for item in items]
            for group, items in mapping.items()
        }
        for width, mapping in value.items()
    }


def domain_from_payload(row: Mapping[str, Any]) -> LocalPruningDomain:
    """Restore the exact parent ranking/closure without recollecting Taylor."""

    return local_pruning_domain_from_dict(row)


def _gpu_snapshot(physical_gpu: int) -> dict[str, Any]:
    completed = subprocess.run(
        [
            "nvidia-smi", f"--id={physical_gpu}",
            "--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=False, capture_output=True, text=True,
    )
    return {
        "physical_gpu": physical_gpu,
        "cuda_visible_devices": __import__("os").environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_smi": completed.stdout.strip(),
        "nvidia_smi_returncode": completed.returncode,
    }


def run(request_path: Path) -> int:
    job = json.loads(request_path.read_text())
    root = Path(job["root"])
    label = str(job["label"])
    seed = int(job["seed"])
    generation = int(job["generation"])
    complete_hash = str(job["complete_phenotype_hash"])
    physical_gpu = int(job["physical_gpu"])
    genotype = CandidateGenotype.from_dict(job["genotype"])
    phenotype = CandidatePhenotype.from_dict(job["phenotype"])
    domains = tuple(domain_from_payload(row) for row in job["domains"])
    request = dict(job["request"])
    evaluation_frames = int(job["evaluation_frames"])
    evaluation_warmup_frames = int(job["evaluation_warmup_frames"])
    evaluation_protocol = str(job["evaluation_protocol"])
    if evaluation_protocol == "stage2_full_fixed500" and evaluation_frames != 500:
        raise RuntimeError(
            f"formal_ga_stage2_requires_fixed500:{evaluation_frames}"
        )
    cache = root / f"ga/stage2_cache/budget_{label}/{complete_hash}"
    result_path = cache / "stage2_result.json"
    generation_dir = root / (
        f"ga/budget_{label}/seed_{seed}/generation_{generation:02d}/"
        f"candidate_{complete_hash}"
    )
    cache.mkdir(parents=True, exist_ok=True)
    generation_dir.mkdir(parents=True, exist_ok=True)
    atomic_write(generation_dir / "gpu_assignment.json", _gpu_snapshot(physical_gpu))
    if result_path.is_file():
        cached = json.loads(result_path.read_text())
        cached_meta = dict(cached.get("metadata") or {})
        if (
            int(cached_meta.get("stage2_evaluation_frames", -1)) != evaluation_frames
            or str(cached_meta.get("evaluation_protocol", "")) != evaluation_protocol
        ):
            raise RuntimeError("stage2_worker_cache_evaluation_protocol_mismatch")
        atomic_write(generation_dir / "cache_reference.json", {
            "stage2_cache": str(cache), "reused": True,
            "complete_phenotype_hash": complete_hash,
        })
        return 0

    result: Stage2Result
    try:
        if torch.cuda.device_count() != 1:
            raise RuntimeError(
                f"stage2_worker_requires_one_visible_gpu:{torch.cuda.device_count()}"
            )
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        model, adapter, hypes, _ = _load("v2xvit", device)
        representative, _, _ = _multi_agent_validation_batch(adapter, hypes, device)
        identity = _build_full_space(model, adapter, hypes, representative)
        physical = materialize_unified_widths(
            model, identity["cnn_units"], domains,
            genotype.pruning_width_genes, model_name="lidar_v2xvit",
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
                export_dir, physical.model, adapter, representative,
                hypes, phenotype, complete_hash, physical.report.structure_hash,
                build_engine=True, tensorrt_root=Path(job["tensorrt_root"]),
                plugin=Path(job["plugin"]),
                calibration_frames=200 if has_int8 else 0,
                qkv_paths=tuple(str(value) for value in job["qkv_paths"]),
                fixed_k_override=int(request["fixed_k"]),
                physical_gpu_id=physical_gpu,
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
                 "physical_gpu": physical_gpu, "precision_fallback": False,
                 "stage2_evaluation_frames": evaluation_frames,
                 "evaluation_protocol": evaluation_protocol},
            )
        else:
            evaluation = evaluate_v2xvit_engine_modelopt(
                engine_path=engine_path,
                model_config=request["model_config"], heal_root=request["heal_root"],
                output_dir=cache / "stage2_fixed500",
                tensorrt_root=Path(job["tensorrt_root"]),
                plugin_path=Path(job["plugin"]),
                eval_manifest_path=Path(job["evaluation_manifest"]),
                physical_gpu_id=physical_gpu, fixed_k=int(request["fixed_k"]),
                max_agents=int(request["max_agents"]),
                num_frames=evaluation_frames,
                warmup_frames=evaluation_warmup_frames,
                latency_rounds=1, dataloader_num_workers=8,
            )
            ok = bool(
                evaluation.get("status") == "ok"
                and int(evaluation.get("num_evaluated_frames", -1))
                == evaluation_frames
                and int(evaluation.get("num_skipped_frames", -1)) == 0
            )
            result = Stage2Result(
                complete_hash, genotype, "ok" if ok else "fixed500_failed",
                float(evaluation["mAP"]) if ok else None,
                float(evaluation["forward_p50_ms"]) if ok else None,
                exact, int(evaluation.get("num_evaluated_frames", 0)),
                int(evaluation.get("num_skipped_frames", 0)),
                {"generation": generation, "engine_sha256": sha256(engine_path),
                 "engine_path": str(engine_path),
                 "physical_structure_hash": physical.report.structure_hash,
                 "calibration_required": has_int8,
                 "calibration_manifest": str(export_dir / "calibration_manifest.json"),
                 "fresh_train200": has_int8, "physical_gpu": physical_gpu,
                 "gpu_snapshot": _gpu_snapshot(physical_gpu),
                 "precision_fallback": False,
                 "stage2_fixed500_result": evaluation,
                 "stage2_evaluation_frames": evaluation_frames,
                 "evaluation_protocol": evaluation_protocol},
            )
        del physical
        torch.cuda.empty_cache()
    except Exception as exc:
        result = Stage2Result(
            complete_hash, genotype, "failed", None, None, False, 0, 0,
            {"generation": generation, "physical_gpu": physical_gpu,
             "failure": f"{type(exc).__name__}:{exc}",
             "precision_fallback": False,
             "stage2_evaluation_frames": evaluation_frames,
             "evaluation_protocol": evaluation_protocol},
        )
    atomic_write(result_path, stage2_payload(result))
    atomic_write(generation_dir / "cache_reference.json", {
        "stage2_cache": str(cache), "reused": False,
        "complete_phenotype_hash": complete_hash,
    })
    print(json.dumps({
        "stage2": "worker_complete", "budget": label, "seed": seed,
        "generation": generation, "candidate": complete_hash,
        "physical_gpu": physical_gpu, "status": result.status,
        "mAP": result.map, "p50_ms": result.p50_ms,
        "failure": result.metadata.get("failure"),
    }), flush=True)
    return 0 if result.status in {"ok", "deployment_invalid", "fixed500_failed"} else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    return run(parser.parse_args().request.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
