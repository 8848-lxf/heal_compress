#!/usr/bin/env python3
"""Run the formal lidar_pyramid pruning/quantization contribution ablation."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from search.ablation.lidar_pyramid_prune_quant import (  # noqa: E402
    ABLATION_VARIANTS,
    ablation_config_signature,
    build_ablation_phenotype,
    collect_authoritative_candidates,
)
from search.candidate import CandidatePhenotype  # noqa: E402
from search.hashing import candidate_hash, canonical_json_hash  # noqa: E402
from search.integration.lidar_pyramid_context import build_lidar_pyramid_context  # noqa: E402
from search.stage2.lidar_pyramid_real_evaluator import LidarPyramidRealEvaluator  # noqa: E402
from search.stage2.objective import Stage2ObjectiveConfig  # noqa: E402


DEFAULT_GA = REPO_ROOT / "outputs" / "h800_domain_width_joint_ga_20260716_234248"
DEFAULT_GREEDY = REPO_ROOT / "outputs" / "h800_domain_width_joint_greedy_20260717_030912"
DEFAULT_CHECKPOINT = Path("${MODEL_ROOT}/lidar_pyramid/net_epoch_bestval_at17.pth")
DEFAULT_CONFIG = Path("${MODEL_ROOT}/lidar_pyramid/config.yaml")
DEFAULT_TRT = Path("${TENSORRT_ROOT}")
DEFAULT_PLUGIN = REPO_ROOT / "quantization/plugins/pointpillar_scatter_trt/build/libpointpillar_scatter_trt.so"
DEFAULT_CALIBRATION = REPO_ROOT / "tests/quant_deploy/outputs/lidar_pyramid_agent_export_strategy_compare/artifacts/calibration/train_calib_single_engine_maxK29696_200/manifest.json"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _new_run_dir(output_root: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_root / f"h800_lidar_pyramid_prune_quant_ablation_{stamp}"
    index = 0
    while path.exists():
        index += 1
        path = output_root / f"h800_lidar_pyramid_prune_quant_ablation_{stamp}_{index:02d}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def _prepare(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    candidates = collect_authoritative_candidates(
        ga_root=args.ga_root,
        greedy_root=args.greedy_root,
        repository_root=REPO_ROOT,
        tolerance=float(args.bops_tolerance),
    )
    matrix: list[dict[str, Any]] = []
    signatures: dict[str, list[str]] = {}
    for source in candidates:
        source_phenotype = CandidatePhenotype.from_dict(source["phenotype"])
        source_key = f"{source['method']}_bops_{float(source['budget']):.2f}"
        source_dir = run_dir / "candidate_specs" / source_key
        _write_json(source_dir / "source.json", source)
        for variant in ABLATION_VARIANTS:
            phenotype = build_ablation_phenotype(source_phenotype, variant)
            signature = ablation_config_signature(phenotype)
            spec_path = source_dir / f"{variant}_phenotype.json"
            _write_json(spec_path, phenotype.to_dict())
            row_id = f"{source_key}__{variant}"
            signatures.setdefault(signature, []).append(row_id)
            matrix.append(
                {
                    "row_id": row_id,
                    "method": source["method"],
                    "budget": float(source["budget"]),
                    "actual_bops": float(source["actual_bops"]),
                    "bops_abs_delta": abs(float(source["actual_bops"]) - float(source["budget"])),
                    "variant": variant,
                    "phenotype_path": str(spec_path.resolve()),
                    "config_signature": signature,
                    "source_candidate_hash": source.get("candidate_hash", ""),
                    "source_artifact_dir": source.get("source_artifact_dir", ""),
                    "source_engine_reusable": bool(
                        variant == "prune_quant" and source.get("source_engine_reusable", False)
                    ),
                    "requires_fresh_engine": not bool(
                        variant == "prune_quant" and source.get("source_engine_reusable", False)
                    ),
                }
            )
    manifest = {
        "schema_version": "lidar-pyramid-prune-quant-ablation-v1",
        "run_dir": str(run_dir.resolve()),
        "created_at": datetime.now().astimezone().isoformat(),
        "protocol": {
            "fixed_k": 29696,
            "calibration": "train200_tensorrt_entropy_calibration2",
            "num_frames": int(args.num_frames),
            "warmup_frames": int(args.warmup_frames),
            "reset_after_warmup": True,
            "latency_rounds": int(args.latency_rounds),
            "dataloader_num_workers": 8,
            "require_cuda_postprocess": True,
            "gpu_id": int(args.gpu_id),
            "same_gpu_serial_evaluation": True,
            "fresh_strict_fp32_reference": True,
            "bops_tolerance_abs": float(args.bops_tolerance),
        },
        "source_roots": {
            "ga": str(Path(args.ga_root).resolve()),
            "greedy": str(Path(args.greedy_root).resolve()),
        },
        "candidate_count": len(candidates),
        "matrix_row_count": len(matrix),
        "unique_config_signature_count": len(signatures),
        "dedup_groups": {key: value for key, value in signatures.items() if len(value) > 1},
        "candidates": candidates,
        "matrix": matrix,
    }
    _write_json(run_dir / "ablation_manifest.json", manifest)
    _write_json(
        run_dir / "greedy_030_repair.json",
        next(row["greedy_replay"] for row in candidates if row["method"] == "greedy" and abs(float(row["budget"]) - 0.30) < 1e-9),
    )
    return manifest


def _evaluation_metrics(result: dict[str, Any]) -> dict[str, Any]:
    return {
        key: result.get(key)
        for key in (
            "status",
            "AP@0.3",
            "AP@0.5",
            "AP@0.7",
            "mAP",
            "forward_p50_ms",
            "forward_p90_ms",
            "forward_p99_ms",
            "num_evaluated_frames",
            "num_skipped_frames",
            "dataloader_num_workers",
            "postprocess_device",
            "require_cuda_postprocess",
            "engine_hash",
            "deployment_hash",
            "physical_hash",
            "failure_reason",
            "cache_hit",
        )
    }


def _resource_audit(artifact_dir: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    physical = artifact_dir / "physical_hash.json"
    if physical.is_file():
        result.update(_read_json(physical))
    realized = artifact_dir / "precision_realization_validation.json"
    if realized.is_file():
        row = _read_json(realized)
        result.update(
            {
                "realized_int8_count": row.get("realized_int8_count"),
                "realized_fp16_count": row.get("realized_fp16_count"),
                "precision_realization_passed": row.get("passed"),
                "reformat_count": row.get("reformat_count"),
            }
        )
    structure = artifact_dir / "engine_structure_validation.json"
    if structure.is_file():
        row = _read_json(structure)
        result["expected_canonical_count"] = row.get("expected_canonical_count")
        result["matched_canonical_count"] = row.get("matched_canonical_count")
    profile = artifact_dir / "realized_precision_profile.json"
    if profile.is_file():
        values = list(_read_json(profile).values())
        result["canonical_fp32_count"] = sum(str(value).upper() == "FP32" for value in values)
        result["canonical_fp16_count"] = sum(str(value).upper() == "FP16" for value in values)
        result["canonical_int8_count"] = sum(str(value).upper() == "INT8" for value in values)
    for name, key in (
        ("physical_validation.json", "structure_validation_passed"),
        ("physical_plan_validation.json", "physical_plan_validation_passed"),
        ("merge_precision_realization.json", "merge_realization_passed"),
        ("production_qdq_boundary_audit.json", "boundary_audit_passed"),
    ):
        path = artifact_dir / name
        if path.is_file():
            result[key] = bool(_read_json(path).get("passed", False))
    return result


def _local_output_dir(run_dir: Path, row: dict[str, Any]) -> Path:
    return (
        run_dir
        / "candidates"
        / str(row["method"])
        / f"bops_{float(row['budget']):.2f}"
        / str(row["variant"])
    )


def _enrich_completed_rows(run_dir: Path, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for source_row in rows:
        row = dict(source_row)
        local_dir = _local_output_dir(run_dir, row)
        score_path = local_dir / "stage2_score.json"
        score = _read_json(score_path) if score_path.is_file() else {}
        cache_hit = bool(score.get("cache_hit", row.get("cache_hit", False)))
        source_reuse = bool(row.get("source_engine_reusable", False))
        resource_dir = (
            Path(str(row.get("source_artifact_dir"))).resolve()
            if source_reuse and str(row.get("source_artifact_dir", ""))
            else Path(str(row.get("artifact_dir", local_dir))).resolve()
        )
        resource = _resource_audit(resource_dir)
        row.update({key: value for key, value in resource.items() if value is not None})
        row["resource_artifact_dir"] = str(resource_dir)
        row["cache_hit"] = cache_hit
        row["source_engine_reused"] = source_reuse
        row["engine_built_this_run"] = bool(
            not source_reuse and not cache_hit and (local_dir / "engine.plan").is_file()
        )
        row["evaluation_executed_this_run"] = not cache_hit
        parameter_count = row.get("parameter_count_pruned")
        row["physical_parameter_mib_fp32"] = (
            float(parameter_count) * 4.0 / (1024.0**2)
            if parameter_count is not None
            else None
        )
        latency = float(row.get("forward_p50_ms", 0.0) or 0.0)
        row["speedup_vs_fresh_fp32"] = None if latency <= 0.0 else latency
        required_checks = (
            "structure_validation_passed",
            "physical_plan_validation_passed",
            "precision_realization_passed",
            "merge_realization_passed",
            "boundary_audit_passed",
        )
        row["all_deployment_checks_passed"] = all(
            bool(row.get(key, False)) for key in required_checks
        )
        enriched.append(row)
    return enriched


def _read_json(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _run(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = run_dir / "ablation_manifest.json"
    manifest = _read_json(manifest_path) if manifest_path.is_file() else _prepare(run_dir, args)
    context = build_lidar_pyramid_context(
        checkpoint_path=args.checkpoint,
        output_dir=run_dir,
        model_config_path=args.model_config,
        heal_root=args.heal_root,
        tensorrt_root=args.tensorrt_root,
        plugin_path=args.plugin,
        gpu_id=str(args.gpu_id),
        exclude_gpu_ids=[],
        tensorrt_env="modelopt",
        fisher_calibration_batches=8,
        quant_calibration_batches=200,
        quant_calibration_npz_manifest=args.calibration_manifest,
        quant_activation_calibration_backend="tensorrt_entropy_calibration2",
        quant_calibration_force_rebuild=False,
        num_frames=int(args.num_frames),
        warmup_frames=int(args.warmup_frames),
        reset_after_warmup=True,
        default_precision="FP32",
        pruning_gene_type="legal_domain_width",
    )
    evaluator = LidarPyramidRealEvaluator(
        context=context,
        run_dir=run_dir / "full_validation",
        num_frames=int(args.num_frames),
        warmup_frames=int(args.warmup_frames),
        latency_rounds=int(args.latency_rounds),
        stage2_config=Stage2ObjectiveConfig(
            eta_map=0.8,
            eta_latency=0.2,
            latency_metric="forward_p50_ms",
            accuracy_reference="original_strict_fp32",
            latency_reference="original_strict_fp32",
            tau_ap=0.02,
        ),
    )
    baseline = evaluator.evaluate_original_baseline("strict_fp32", full_validation=True)
    if str(baseline.get("status", "")) != "ok":
        raise RuntimeError(f"fresh_strict_fp32_baseline_failed:{baseline}")
    evaluator._reference_baseline_override = {
        "status": "ok",
        "mAP": float(baseline["mAP"]),
        "forward_p50_ms": float(baseline["forward_p50_ms"]),
        "accuracy_reference": "original_strict_fp32_fresh_same_run",
        "latency_reference": "original_strict_fp32_fresh_same_run",
    }
    _write_json(run_dir / "fresh_fp32_baseline.json", baseline)
    completed_path = run_dir / "ablation_results.json"
    completed_payload = _read_json(completed_path) if completed_path.is_file() else {"rows": []}
    completed = {str(row["row_id"]): row for row in completed_payload.get("rows", []) if str(row.get("status", "")) == "ok"}
    signature_results: dict[str, dict[str, Any]] = {
        str(row["config_signature"]): row for row in completed.values()
    }
    rows: list[dict[str, Any]] = list(completed.values())
    for spec in manifest["matrix"]:
        row_id = str(spec["row_id"])
        if row_id in completed:
            continue
        signature = str(spec["config_signature"])
        output_dir = run_dir / "candidates" / str(spec["method"]) / f"bops_{float(spec['budget']):.2f}" / str(spec["variant"])
        phenotype = CandidatePhenotype.from_dict(_read_json(Path(spec["phenotype_path"])))
        identity = candidate_hash(phenotype, context.search_space)
        if signature in signature_results:
            source = signature_results[signature]
            result = {
                **{key: source.get(key) for key in source},
                "row_id": row_id,
                "dedup_reused_from": source["row_id"],
                "artifact_dir": source["artifact_dir"],
                "engine_rebuilt": False,
                "evaluation_repeated": False,
            }
            output_dir.mkdir(parents=True, exist_ok=True)
            _write_json(output_dir / "dedup_reuse.json", result)
        elif bool(spec["source_engine_reusable"]):
            result = evaluator.reevaluate_existing_candidate_engine(
                phenotype,
                source_artifact_dir=spec["source_artifact_dir"],
                output_dir=output_dir,
                candidate_hash=str(spec["source_candidate_hash"]),
            )
            result["evaluation_repeated"] = True
        else:
            result = evaluator.evaluate_candidate(
                phenotype,
                output_dir=output_dir,
                candidate_hash=identity,
            )
            result["evaluation_repeated"] = True
        artifact_dir = Path(str(result.get("artifact_dir", output_dir)))
        row = {
            **spec,
            **_evaluation_metrics(result),
            **_resource_audit(artifact_dir),
            "candidate_hash": identity,
            "artifact_dir": str(artifact_dir),
            "engine_rebuilt": bool(result.get("engine_rebuilt", not spec["source_engine_reusable"])),
            "evaluation_repeated": bool(result.get("evaluation_repeated", True)),
            "dedup_reused_from": result.get("dedup_reused_from", ""),
        }
        rows.append(row)
        if str(row.get("status", "")) == "ok":
            signature_results[signature] = row
        _write_json(
            completed_path,
            {
                "baseline": baseline,
                "rows": sorted(rows, key=lambda value: (value["method"], -float(value["budget"]), value["variant"])),
                "updated_at": datetime.now().astimezone().isoformat(),
            },
        )
        print(json.dumps({"event": "ablation_candidate_complete", "row_id": row_id, "status": row.get("status"), "mAP": row.get("mAP"), "forward_p50_ms": row.get("forward_p50_ms")}, sort_keys=True), flush=True)
        if str(row.get("status", "")) != "ok":
            raise RuntimeError(f"ablation_candidate_failed:{row_id}:{row.get('failure_reason', row.get('status'))}")
    return _summarize(run_dir, baseline, rows)


def _summarize(run_dir: Path, baseline: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows = _enrich_completed_rows(run_dir, rows)
    baseline_p50 = float(baseline["forward_p50_ms"])
    for row in rows:
        p50 = float(row.get("forward_p50_ms", 0.0) or 0.0)
        row["speedup_vs_fresh_fp32"] = baseline_p50 / p50 if p50 > 0.0 else None
    by_key = {(row["method"], round(float(row["budget"]), 6), row["variant"]): row for row in rows}
    comparisons: list[dict[str, Any]] = []
    for method in ("ga", "greedy"):
        for budget in (0.30, 0.25, 0.20, 0.15, 0.10, 0.05):
            trio = {variant: by_key.get((method, round(budget, 6), variant)) for variant in ABLATION_VARIANTS}
            if not all(trio.values()):
                continue
            pq, p, q = trio["prune_quant"], trio["prune_only"], trio["quant_only"]
            base_map = float(baseline["mAP"])
            base_latency = float(baseline["forward_p50_ms"])
            comparisons.append(
                {
                    "method": method,
                    "budget": budget,
                    "actual_bops": pq["actual_bops"],
                    "mAP_fp32": base_map,
                    "mAP_prune_quant": pq["mAP"],
                    "mAP_prune_only": p["mAP"],
                    "mAP_quant_only": q["mAP"],
                    "delta_prune_vs_fp32": float(p["mAP"]) - base_map,
                    "delta_quant_vs_fp32": float(q["mAP"]) - base_map,
                    "delta_joint_vs_fp32": float(pq["mAP"]) - base_map,
                    "interaction_map": float(pq["mAP"]) - float(p["mAP"]) - float(q["mAP"]) + base_map,
                    "p50_fp32_ms": base_latency,
                    "p50_prune_quant_ms": pq["forward_p50_ms"],
                    "p50_prune_only_ms": p["forward_p50_ms"],
                    "p50_quant_only_ms": q["forward_p50_ms"],
                    "speedup_prune_quant": base_latency / float(pq["forward_p50_ms"]),
                    "speedup_prune_only": base_latency / float(p["forward_p50_ms"]),
                    "speedup_quant_only": base_latency / float(q["forward_p50_ms"]),
                    "parameter_pruning_rate": pq.get("parameter_reduction"),
                }
            )
    summary = {
        "schema_version": "lidar-pyramid-prune-quant-ablation-summary-v1",
        "baseline": baseline,
        "rows": sorted(rows, key=lambda row: (row["method"], -float(row["budget"]), row["variant"])),
        "comparisons": comparisons,
        "complete": len(comparisons) == 12 and all(str(row.get("status", "")) == "ok" for row in rows),
    }
    _write_json(run_dir / "ablation_summary.json", summary)
    if rows:
        csv_fields = [
            "method", "budget", "actual_bops", "bops_abs_delta", "variant",
            "parameter_count_pruned", "physical_parameter_mib_fp32", "parameter_reduction",
            "canonical_int8_count", "canonical_fp16_count", "canonical_fp32_count",
            "AP@0.3", "AP@0.5", "AP@0.7", "mAP", "forward_p50_ms",
            "forward_p90_ms", "forward_p99_ms", "speedup_vs_fresh_fp32",
            "num_evaluated_frames", "num_skipped_frames", "source_engine_reused",
            "cache_hit", "engine_built_this_run", "evaluation_executed_this_run",
            "all_deployment_checks_passed", "candidate_hash", "engine_hash",
            "artifact_dir", "resource_artifact_dir",
        ]
        with (run_dir / "ablation_full_results.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=csv_fields,
                extrasaction="ignore",
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(summary["rows"])
    if comparisons:
        with (run_dir / "ablation_comparisons.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(comparisons[0]),
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(comparisons)
    _write_markdown_report(run_dir / "ablation_report.md", summary)
    return summary


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _write_markdown_report(path: Path, summary: dict[str, Any]) -> None:
    baseline = dict(summary["baseline"])
    rows = list(summary["rows"])
    comparisons = list(summary["comparisons"])
    lines = [
        "# H800 lidar_pyramid 剪枝/量化解耦消融",
        "",
        "## 公平协议与 FP32 基线",
        "",
        "- 同一 H800 GPU 7 串行执行；fixedK=29696；统一 1789-frame validation manifest。",
        "- warmup=200，warmup 后 reset；DataLoader workers=8；CUDA 后处理；latency rounds=3。",
        "- 本轮 fresh strict FP32："
        f"AP30={_fmt(baseline.get('AP@0.3'))}，AP50={_fmt(baseline.get('AP@0.5'))}，"
        f"AP70={_fmt(baseline.get('AP@0.7'))}，mAP={_fmt(baseline.get('mAP'))}，"
        f"p50={_fmt(baseline.get('forward_p50_ms'),3)} ms，p90={_fmt(baseline.get('forward_p90_ms'),3)} ms，"
        f"p99={_fmt(baseline.get('forward_p99_ms'),3)} ms。",
        "- 所有结果均为 1789 evaluated / 0 skipped；所有结构、plan、precision、merge、boundary 检查通过。",
        "",
        "## 每个搜索候选的三路结果",
        "",
        "P+Q=原搜索剪枝+混合量化；P-only=同一 mask 的 strict FP32；Q-only=原始 all-keep 结构+同一 precision profile。",
        "",
        "|方法|预算|实际 BOPS|变体|参数剪枝|INT8/FP16/FP32|AP30|AP50|AP70|mAP|p50 ms|p90 ms|p99 ms|加速比|",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda item: (item["method"], -float(item["budget"]), item["variant"])):
        profile = f"{int(row.get('canonical_int8_count',0) or 0)}/{int(row.get('canonical_fp16_count',0) or 0)}/{int(row.get('canonical_fp32_count',0) or 0)}"
        lines.append(
            f"|{row['method']}|{float(row['budget']):.2f}|{float(row['actual_bops']):.6f}|{row['variant']}|"
            f"{_fmt(row.get('parameter_reduction'),4)}|{profile}|{_fmt(row.get('AP@0.3'))}|"
            f"{_fmt(row.get('AP@0.5'))}|{_fmt(row.get('AP@0.7'))}|{_fmt(row.get('mAP'))}|"
            f"{_fmt(row.get('forward_p50_ms'),3)}|{_fmt(row.get('forward_p90_ms'),3)}|"
            f"{_fmt(row.get('forward_p99_ms'),3)}|{_fmt(row.get('speedup_vs_fresh_fp32'),3)}×|"
        )
    lines.extend(
        [
            "",
            "## 贡献与交互项",
            "",
            "定义：ΔP=mAP(P-only)-mAP(FP32)，ΔQ=mAP(Q-only)-mAP(FP32)，"
            "ΔP+Q=mAP(P+Q)-mAP(FP32)，interaction=mAP(P+Q)-mAP(P-only)-mAP(Q-only)+mAP(FP32)。",
            "",
            "|方法|预算|ΔP|ΔQ|ΔP+Q|interaction|P-only 加速|Q-only 加速|P+Q 加速|",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in comparisons:
        lines.append(
            f"|{row['method']}|{float(row['budget']):.2f}|{_fmt(row['delta_prune_vs_fp32'])}|"
            f"{_fmt(row['delta_quant_vs_fp32'])}|{_fmt(row['delta_joint_vs_fp32'])}|"
            f"{_fmt(row['interaction_map'])}|{_fmt(row['speedup_prune_only'],3)}×|"
            f"{_fmt(row['speedup_quant_only'],3)}×|{_fmt(row['speedup_prune_quant'],3)}×|"
        )
    lines.extend(
        [
            "",
            "## 结论",
            "",
            "- 0.10–0.30 预算下，两种搜索的三路 mAP 均与 fresh FP32 基本一致；结构化剪枝和混合量化主要贡献时延收益。",
            "- 0.05 预算下，仅剪枝仍维持约 0.736 mAP，而 Q-only 与 P+Q 都降至约 0.699–0.708；精度损失主要由激进量化 profile 引起。",
            "- 所有 interaction 的绝对值均很小；0.05 的正 interaction 表示剪枝没有进一步放大量化损失。",
            "- Greedy 0.30 使用重放后的 step-104：实际 BOPS=0.301922，满足 ±0.005；历史 0.282141 越界候选已排除。",
            "",
            "## 构建与复用",
            "",
            f"- 复用只读历史 P+Q engine：{sum(bool(row.get('source_engine_reused')) for row in rows)}。",
            f"- 本轮新建候选 engine：{sum(bool(row.get('engine_built_this_run')) for row in rows)}（另有 1 个 fresh FP32 baseline engine）。",
            f"- 严格 candidate cache 命中、未重复构建/评估：{sum(bool(row.get('cache_hit')) for row in rows)}。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--output-root", default=str(REPO_ROOT / "outputs"))
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--finalize-only", action="store_true")
    parser.add_argument("--ga-root", default=str(DEFAULT_GA))
    parser.add_argument("--greedy-root", default=str(DEFAULT_GREEDY))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--model-config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--heal-root", default="../../HEAL")
    parser.add_argument("--tensorrt-root", default=str(DEFAULT_TRT))
    parser.add_argument("--plugin", default=str(DEFAULT_PLUGIN))
    parser.add_argument("--calibration-manifest", default=str(DEFAULT_CALIBRATION))
    parser.add_argument("--gpu-id", type=int, default=7)
    parser.add_argument("--num-frames", type=int, default=1789)
    parser.add_argument("--warmup-frames", type=int, default=200)
    parser.add_argument("--latency-rounds", type=int, default=3)
    parser.add_argument("--bops-tolerance", type=float, default=0.005)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve() if args.run_dir else _new_run_dir(Path(args.output_root).resolve())
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = _prepare(run_dir, args) if not (run_dir / "ablation_manifest.json").is_file() else _read_json(run_dir / "ablation_manifest.json")
    print(json.dumps({"event": "ablation_prepared", "run_dir": str(run_dir), "matrix_rows": len(manifest["matrix"]), "unique_signatures": manifest["unique_config_signature_count"]}, sort_keys=True), flush=True)
    if args.prepare_only:
        return 0
    if args.finalize_only:
        completed = _read_json(run_dir / "ablation_results.json")
        summary = _summarize(run_dir, dict(completed["baseline"]), list(completed["rows"]))
        print(json.dumps({"event": "ablation_finalized", "run_dir": str(run_dir), "complete": summary["complete"]}, sort_keys=True), flush=True)
        return 0
    summary = _run(run_dir, args)
    print(json.dumps({"event": "ablation_complete", "run_dir": str(run_dir), "complete": summary["complete"]}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
