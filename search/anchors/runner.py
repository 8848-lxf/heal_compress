"""Production runner for the controlled BOPS-retention 0.21 anchor study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shlex
import subprocess
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import yaml

from .bops_021 import (
    QuantizationSensitivity,
    decompose_bops_retention,
    identify_low_damage_bops_path,
    make_all_fp16_genotype,
    make_all_fp16_pruning_genotype,
    make_mixed_no_prune_genotype,
    precision_identity_audit,
    quantization_perturbation_metrics,
    realized_bops_gate,
    select_mixed_precision_prefix,
    theoretical_bops_retention,
    validate_anchor_semantics,
    validate_manifest_consistency,
    validate_plugin_gene_exclusion,
    validate_strongly_typed_profile,
)
from ..candidate import CandidateGenotype, CandidatePhenotype
from ..canonicalization import canonicalize_candidate
from ..hashing import candidate_hash, canonical_json_hash
from ..integration.calibration_provider import (
    collect_or_load_fisher_statistics,
    fixed_k_calibration_npz_manifest_identity,
)
from ..integration.data_provider import load_split_frame_ids, write_eval_manifest
from ..integration.lidar_pyramid_context import build_lidar_pyramid_context
from ..integration.runtime_environment import require_gpu_isolation
from ..orchestration.lidar_pyramid_search import LidarPyramidTwoStageSearch
from ..proxy.gpu_batch_proxy import TorchBatchedProxyScorer
from ..proxy.normalization import NormalizationStats
from ..proxy.objective import ProxyObjectiveConfig
from ..proxy.parameter_slice_resolver import build_unit_parameter_slices
from ..proxy.runtime_shape_profiler import profile_runtime_layer_shapes
from ..pruning_space.action_catalog import build_pruning_action_catalog
from ..stage2.lidar_pyramid_real_evaluator import LidarPyramidRealEvaluator
from ..stage2.objective import Stage2ObjectiveConfig
from ..stage2.realized_bops import compute_realized_bops


def _write_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit(repo: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=10,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"git_commit_resolution_failed:{completed.stderr.strip()}")
    return completed.stdout.strip()


def _precision_counts(profile: Mapping[str, str], *, include_functional_fp16: bool) -> dict[str, int]:
    counts = Counter(str(value).upper() for value in profile.values())
    if include_functional_fp16:
        counts["FP16"] += 1
    return {name: int(counts.get(name, 0)) for name in ("INT8", "FP16", "FP32")}


def _unique_shapes(shapes: Sequence[Any]) -> list[Any]:
    result = []
    seen: set[tuple[str, int]] = set()
    for shape in shapes:
        key = (str(shape.module_path), int(shape.call_index))
        if key in seen:
            continue
        seen.add(key)
        result.append(shape)
    return result


def _total_macs(shapes: Sequence[Any]) -> float:
    return sum(float(shape.macs) for shape in _unique_shapes(shapes))


def _macs_by_module(shapes: Sequence[Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for shape in _unique_shapes(shapes):
        result[str(shape.module_path)] = result.get(str(shape.module_path), 0.0) + float(shape.macs)
    return result


def _plugin_dtype_audit(path: str | Path) -> dict[str, Any]:
    payload = _read_json(path)
    matches = [
        row
        for row in payload.get("Layers", [])
        if str(row.get("PluginType", "")) == "PointPillarScatterTRT"
        or "PointPillarScatterTRT" in str(row.get("Name", ""))
    ]
    if len(matches) != 1:
        return {
            "passed": False,
            "status": "plugin_layer_match_failure",
            "plugin_layer_count": len(matches),
        }
    row = matches[0]
    inputs = [
        str(tensor.get("Format/Datatype", ""))
        for tensor in row.get("Inputs", [])
        if str(tensor.get("Format/Datatype", "")) not in {"Int32", ""}
    ]
    outputs = [str(tensor.get("Format/Datatype", "")) for tensor in row.get("Outputs", [])]
    passed = bool(inputs) and all(value == "Float" for value in inputs) and outputs == ["Float"]
    return {
        "passed": passed,
        "status": "passed" if passed else "plugin_dtype_mismatch",
        "plugin_layer_count": 1,
        "floating_input_dtypes": inputs,
        "output_dtypes": outputs,
        "boundary": "FP32" if passed else "UNKNOWN",
    }


def _qdq_boundary_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"passed": False, "status": "qdq_boundary_audit_missing"}
    payload = _read_json(path)
    summary = dict(payload.get("summary") or payload.get("audit_summary") or payload)
    keys = (
        "pre_relu_qdq_count",
        "invalid_raw_conv_output_qdq_count",
        "orphan_q_count",
        "orphan_dq_count",
        "duplicate_qdq_count",
        "merge_contract_failure_count",
        "unexpected_fp32_fallback_count",
    )
    counts = {key: int(summary.get(key, 0) or 0) for key in keys}
    explicit_pass = payload.get("passed", summary.get("passed"))
    issues = list(payload.get("issues", []))
    layer_failures = [
        str(row.get("canonical_layer", ""))
        for row in payload.get("layers", [])
        if row.get("boundary_passed") is False
    ]
    passed = (explicit_pass is not False) and not issues and not layer_failures and all(value == 0 for value in counts.values())
    return {
        "passed": passed,
        "status": "passed" if passed else "invalid_QDQ_boundary",
        "issues": issues,
        "layer_failures": layer_failures,
        **counts,
    }


def _merge_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"passed": False, "status": "merge_realization_missing", "concat_9": {}}
    payload = _read_json(path)
    rows = list(payload.get("merges", []))
    concat = [row for row in rows if "/Concat_9" in str(row.get("merge_op_name", ""))]
    status = str(payload.get("status", ""))
    passed = status in {"passed", "ok"} or (not payload.get("issues") and all(str(row.get("status", "passed")) in {"passed", "ok"} for row in rows))
    return {
        "passed": passed,
        "status": "passed" if passed else "merge_realization_failure",
        "merge_count": len(rows),
        "concat_9": concat[0] if len(concat) == 1 else {"match_count": len(concat)},
    }


def _sensitivity_prior(module_paths: Sequence[str]) -> tuple[float, list[str]]:
    text = " ".join(str(value).lower() for value in module_paths)
    rules = (
        (("pillar_vfe", "encoder_m1"), 8.0, "pillar_vfe_protection"),
        (("backbone_m1", "resnet.layer0"), 6.0, "early_backbone_protection"),
        (("resnet.layer1", "stage1"), 3.0, "stage1_sensitivity"),
        (("shrink",), 6.0, "shrink_sensitivity"),
        (("cls_head", "reg_head", "dir_head", "single_head"), 8.0, "head_protection"),
    )
    multiplier = 1.0
    reasons = []
    for tokens, value, reason in rules:
        if any(token in text for token in tokens):
            multiplier = max(multiplier, value)
            reasons.append(reason)
    return multiplier, reasons


class Bops021AnchorStudy:
    """Plan and execute A/B/C without invoking the genetic search engine."""

    def __init__(self, *, config: dict[str, Any], output_root: str | Path) -> None:
        self.config = dict(config)
        self.output_root = Path(output_root).expanduser().resolve()
        self.repo = Path(__file__).resolve().parents[2]
        self.run_dir: Path | None = None

    def _new_run_dir(self) -> Path:
        output_cfg = dict(self.config.get("output", {}) or {})
        name = str(output_cfg.get("experiment_name", "4090_bops_021_anchor_study"))
        stamp = time.strftime("%Y%m%d_%H%M%S")
        path = self.output_root / f"{name}_{stamp}"
        suffix = 0
        while path.exists():
            suffix += 1
            path = self.output_root / f"{name}_{stamp}_{suffix:02d}"
        path.mkdir(parents=True, exist_ok=False)
        return path

    def _build_context(self, run_dir: Path) -> Any:
        model = dict(self.config.get("model", {}) or {})
        runtime = dict(self.config.get("runtime", {}) or {})
        proxy = dict(self.config.get("proxy", {}) or {})
        pruning = dict(self.config.get("pruning", {}) or {})
        precision = dict(self.config.get("precision", {}) or {})
        search = dict(self.config.get("search", {}) or {})
        stage2 = dict(self.config.get("stage2", {}) or {})
        grouped = dict(pruning.get("grouped_conv", {}) or {})
        return build_lidar_pyramid_context(
            checkpoint_path=model["checkpoint"],
            output_dir=run_dir,
            model_config_path=model.get("config"),
            heal_root=runtime.get("heal_root", "/home/lixingfeng/UniAD_examine/HEAL"),
            tensorrt_root=runtime["tensorrt_root"],
            plugin_path=runtime["plugin_path"],
            plugin_boundary_dtype=str(runtime.get("plugin_boundary_dtype", "fp32")),
            gpu_id=str(runtime.get("gpu_id", "4")),
            exclude_gpu_ids=[int(value) for value in runtime.get("exclude_gpu_ids", [])],
            tensorrt_env=str(runtime.get("tensorrt_env", "modelopt")),
            fisher_calibration_batches=int(proxy.get("fisher_calibration_batches", 8)),
            quant_calibration_batches=int(proxy.get("quant_calibration_batches", 200)),
            quant_calibration_npz_manifest=proxy.get("quant_calibration_npz_manifest"),
            quant_activation_calibration_backend=str(proxy.get("quant_activation_calibration_backend", "tensorrt_entropy_calibration2")),
            quant_activation_calibration_cache_path=proxy.get("quant_activation_calibration_cache_path"),
            quant_calibration_force_rebuild=bool(proxy.get("quant_calibration_force_rebuild", True)),
            num_frames=int(stage2.get("num_frames", 200)),
            warmup_frames=int(stage2.get("warmup_frames", 20)),
            reset_after_warmup=bool(stage2.get("reset_after_warmup", True)),
            default_precision=str(precision.get("default", "FP16")),
            max_pruning_units=int(search.get("max_pruning_units", 96)),
            grouped_conv_mode=str(grouped.get("position_mode", "independent_group_topk")),
            grouped_conv_align=int(grouped.get("default_channels_per_group", 8)),
            grouped_allowed_channels_per_group=[int(value) for value in grouped.get("allowed_channels_per_group", [4, 8, 16, 32, 64, 128, 256, 512])],
            pruning_gene_type=str(pruning.get("gene_type", "coupled_channel_keep_mask")),
            allow_foreign_gpu_processes=bool(runtime.get("allow_foreign_gpu_processes", False)),
            max_gpu_utilization_pct=int(runtime.get("max_gpu_utilization_pct", 20)),
        )

    def _expand_anchor_pruning_context(self, context: Any) -> Any:
        """Expose selected late local domains from the existing formal trace.

        The GA search space remains unchanged.  This study-only inventory is
        necessary because the formal GA context intentionally exposes only one
        dense 96-unit root plus grouped-conv domains, whose maximum all-FP16
        reduction is insufficient for BOPS 0.21.
        """

        anchor = dict(self.config.get("anchor_study", {}) or {})
        patterns = tuple(str(value) for value in anchor.get("pruning_root_patterns", []))
        if not patterns:
            return context
        modules = dict(context.model.named_modules())
        selected = []
        excluded_tokens = ("pillar_vfe", "scatter", "cls_head", "reg_head", "dir_head", "single_head")
        for unit in list(getattr(context.trace_result, "atomic_prune_units", []) or []):
            path = str(getattr(unit, "root_module_path", ""))
            lower = path.lower()
            constraints = dict(getattr(unit, "constraints", {}) or {})
            module = modules.get(path)
            if not any(path == pattern or path.startswith(pattern + ".") for pattern in patterns):
                continue
            if bool(getattr(unit, "protected", False)) or any(token in lower for token in excluded_tokens):
                continue
            if constraints.get("grouped_conv") and not constraints.get("depthwise"):
                continue
            if str(getattr(unit, "root_axis", "")) not in {"out", "channel"}:
                continue
            if not list(getattr(unit, "root_indices", []) or []):
                continue
            if not isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                continue
            selected.append(unit)
        if not selected:
            raise RuntimeError(f"anchor_pruning_root_patterns_unmatched:{patterns}")
        by_id = {str(getattr(unit, "stable_id", "")): unit for unit in selected}
        selected = [by_id[key] for key in sorted(by_id)]
        pruning_cfg = dict(self.config.get("pruning", {}) or {})
        grouped_cfg = dict(pruning_cfg.get("grouped_conv", {}) or {})
        catalog = build_pruning_action_catalog(
            selected,
            grouped_conv_mode=str(grouped_cfg.get("position_mode", "independent_group_topk")),
            grouped_conv_align=int(grouped_cfg.get("default_channels_per_group", 8)),
            grouped_allowed_channels_per_group=[
                int(value)
                for value in grouped_cfg.get("allowed_channels_per_group", [4, 8, 16, 32, 64, 128, 256, 512])
            ],
        )
        metadata = {
            str(getattr(unit, "stable_id", "")): {
                "scope_id": str(getattr(unit, "scope_id", "")),
                "root_module_path": str(getattr(unit, "root_module_path", "")),
                "root_axis": str(getattr(unit, "root_axis", "")),
                "root_indices": [int(value) for value in getattr(unit, "root_indices", []) or []],
                "constraints": dict(getattr(unit, "constraints", {}) or {}),
                "normalized_score": float(getattr(unit, "normalized_score", 0.0)),
            }
            for unit in selected
        }
        expanded_space = replace(
            context.search_space,
            pruning_unit_ids=list(by_id),
            pruning_unit_metadata=metadata,
            protected_pruning_unit_ids=set(),
        )
        domain_counts = Counter(str(getattr(unit, "root_module_path", "")) for unit in selected)
        _write_json(
            self.run_dir / "planning" / "anchor_pruning_inventory.json",
            {
                "source": "formal_trace_atomic_prune_units",
                "scope": "anchor_study_only_not_stage_a_search_space",
                "root_patterns": list(patterns),
                "unit_count": len(selected),
                "domain_counts": dict(sorted(domain_counts.items())),
                "early_backbone_protected": True,
                "detection_head_roots_protected": True,
                "grouped_conv_domains_modified": False,
            },
        )
        return replace(
            context,
            atomic_prune_units=selected,
            pruning_action_catalog=catalog,
            search_space=expanded_space,
        )

    def _make_evaluator(
        self,
        context: Any,
        *,
        frames: int,
        target: float | None,
    ) -> LidarPyramidRealEvaluator:
        stage2 = dict(self.config.get("stage2", {}) or {})
        return LidarPyramidRealEvaluator(
            context=context,
            run_dir=self.run_dir,
            num_frames=int(frames),
            warmup_frames=int(stage2.get("warmup_frames", 20)),
            latency_rounds=int(stage2.get("latency_rounds", 1)),
            target_bops_retention=target,
            bops_tolerance=float(stage2.get("bops_tolerance", 0.005)),
            stage2_config=Stage2ObjectiveConfig(
                eta_map=0.8,
                eta_latency=0.2,
                latency_metric=str(stage2.get("latency_metric", "forward_p50_ms")),
                tau_ap=None,
                max_map_drop=None,
            ),
        )

    @staticmethod
    def _proxy_metrics(scorer: TorchBatchedProxyScorer, phenotypes: Sequence[CandidatePhenotype]) -> list[dict[str, Any]]:
        return scorer.evaluate_batch(list(phenotypes), generation=0, outer_round=0).metrics

    def _repair(
        self,
        orchestrator: LidarPyramidTwoStageSearch,
        context: Any,
        raw: CandidateGenotype,
    ) -> tuple[CandidateGenotype, dict[str, Any]]:
        repaired, report = orchestrator._repair_raw_keep_mask(context, raw)
        if repaired is None or str(report.get("status")) != "ok":
            raise RuntimeError(f"anchor_channel_repair_failed:{report}")
        if raw.precision_genes != repaired.precision_genes:
            raise RuntimeError("anchor_channel_repair_modified_precision_genes")
        return repaired, report

    def _fp16_pruning_scan(
        self,
        *,
        context: Any,
        orchestrator: LidarPyramidTwoStageSearch,
        scorer: TorchBatchedProxyScorer,
    ) -> list[dict[str, Any]]:
        pruning = dict(self.config.get("pruning", {}) or {})
        alignment = int(dict(pruning.get("dense", {}) or {}).get("alignment", 4))
        minimum_ratio = float(pruning.get("minimum_retained_ratio", 0.10))
        units_by_id = {str(getattr(unit, "stable_id", "")): unit for unit in context.atomic_prune_units}
        domains: dict[tuple[str, str, str], list[str]] = {}
        for unit_id in context.search_space.pruning_unit_ids:
            unit = units_by_id[unit_id]
            constraints = dict(getattr(unit, "constraints", {}) or {})
            if constraints.get("grouped_conv") and not constraints.get("depthwise"):
                continue
            key = (
                str(getattr(unit, "root_module_path", "")),
                str(getattr(unit, "root_axis", "")),
                str(getattr(unit, "scope_id", "")),
            )
            domains.setdefault(key, []).append(unit_id)
        if not domains:
            raise RuntimeError("anchor_fp16_pruning_dense_domain_missing")
        action_index = {unit_id: index for index, unit_id in enumerate(scorer.action_ids)}
        action_costs = scorer.action_fisher_cost.detach().float().cpu()
        candidates: list[tuple[CandidateGenotype, CandidateGenotype, CandidatePhenotype, dict[str, Any]]] = []
        for key, unit_ids in sorted(domains.items()):
            ordered = sorted(
                unit_ids,
                key=lambda unit_id: (float(action_costs[action_index[unit_id]]), unit_id),
            )
            minimum = max(alignment, int(math.ceil(len(ordered) * minimum_ratio / alignment)) * alignment)
            for retained in range(minimum, len(ordered) + 1, alignment):
                prune_count = len(ordered) - retained
                if prune_count <= 0:
                    continue
                raw = make_all_fp16_pruning_genotype(
                    context.search_space,
                    pruned_unit_ids=ordered[:prune_count],
                )
                raw = CandidateGenotype(
                    raw.pruning_genes,
                    raw.precision_genes,
                    {
                        **raw.meta,
                        "local_domain": {"root_module_path": key[0], "root_axis": key[1], "scope_id": key[2]},
                        "selected_domain_width": len(ordered),
                        "retained_search_units": retained,
                        "importance": "second_order_fisher_low_to_high",
                    },
                )
                repaired, repair_report = self._repair(orchestrator, context, raw)
                phenotype = canonicalize_candidate(repaired, context.search_space)
                candidates.append((raw, repaired, phenotype, repair_report))
        metrics = self._proxy_metrics(scorer, [row[2] for row in candidates])
        rows = []
        for (raw, repaired, phenotype, repair_report), metric in zip(candidates, metrics):
            rows.append(
                {
                    "raw": raw,
                    "repaired": repaired,
                    "phenotype": phenotype,
                    "repair_report": repair_report,
                    "proxy": metric,
                    "proxy_gate": realized_bops_gate(
                        float(metric["R_bops_vs_fp32"]),
                        target=float(dict(self.config.get("stage2", {})).get("target_bops_retention", 0.21)),
                        tolerance=float(dict(self.config.get("stage2", {})).get("bops_tolerance", 0.005)),
                    ),
                }
            )
        return sorted(
            rows,
            key=lambda row: (
                0 if row["proxy_gate"]["passed"] else 1,
                abs(float(row["proxy"]["R_bops_vs_fp32"]) - 0.21),
                float(row["proxy"]["L_fisher"]),
                candidate_hash(row["phenotype"], context.search_space),
            ),
        )

    def _physical_pruning_selection(
        self,
        *,
        context: Any,
        evaluator: LidarPyramidRealEvaluator,
        baseline_shapes: Sequence[Any],
        scan: list[dict[str, Any]],
    ) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        anchor = dict(self.config.get("anchor_study", {}) or {})
        limit = int(anchor.get("planning_materialization_limit", 8))
        baseline_snapshot = evaluator.pruning.snapshot_fn(context.model)
        target = float(dict(self.config.get("stage2", {})).get("target_bops_retention", 0.21))
        tolerance = float(dict(self.config.get("stage2", {})).get("bops_tolerance", 0.005))
        planning_rows = []
        selected = None
        for index, row in enumerate(scan[:limit]):
            phenotype = row["phenotype"]
            key = candidate_hash(phenotype, context.search_space)
            destination = self.run_dir / "planning" / "anchor_b" / f"{index:03d}_{key}"
            physical = evaluator._materialize_physical(phenotype, destination)
            shapes = profile_runtime_layer_shapes(
                physical["model"],
                context.trace_example_inputs,
                forward_fn=context.model_bundle.adapter.forward_for_task,
            )
            profile = {str(shape.module_path): "FP16" for shape in _unique_shapes(shapes.shapes)}
            bops = compute_realized_bops(
                physical_runtime_shapes=shapes.shapes,
                baseline_runtime_shapes=baseline_shapes,
                realized_precision_profile=profile,
                physical_snapshot=physical["snapshot"],
                baseline_snapshot=baseline_snapshot,
                target_retention=target,
                tolerance=tolerance,
            )
            record = {
                "candidate_hash": key,
                "proxy_bops_retention": float(row["proxy"]["R_bops_vs_fp32"]),
                "physical_bops_retention": float(bops["bops_retention"]),
                "physical_gate_passed": bool(bops["passed"]),
                "pruning_loss": float(row["proxy"]["L_fisher"]),
                "pruned_unit_count": len(phenotype.pruned_unit_ids),
                "physical_hash": str(physical["physical_hash"]),
                "local_domain": dict(row["raw"].meta.get("local_domain") or {}),
                "retained_search_units": row["raw"].meta.get("retained_search_units"),
            }
            planning_rows.append(record)
            row["physical_planning"] = record
            if bops["passed"] and (
                selected is None
                or (float(row["proxy"]["L_fisher"]), abs(float(bops["bops_retention"]) - target), key)
                < (
                    float(selected["proxy"]["L_fisher"]),
                    abs(float(selected["physical_planning"]["physical_bops_retention"]) - target),
                    selected["physical_planning"]["candidate_hash"],
                )
            ):
                selected = row
        _write_json(self.run_dir / "planning" / "anchor_b_width_scan.json", planning_rows)
        return selected, planning_rows

    def _quantization_sensitivity(
        self,
        *,
        context: Any,
        fisher: Any,
        baseline_shapes: Sequence[Any],
    ) -> list[QuantizationSensitivity]:
        modules = dict(context.model.named_modules())
        macs = _macs_by_module(baseline_shapes)
        raw_rows = []
        for group in context.search_space.quantization_groups:
            if group.protected or "INT8" not in group.allowed_precisions:
                continue
            group_macs = sum(float(macs.get(module_path, 0.0)) for module_path in group.module_paths)
            if group_macs <= 0.0:
                raise RuntimeError(f"anchor_quantization_group_macs_missing:{group.group_id}")
            taylor = 0.0
            sqnr = 0.0
            saturated_values = 0.0
            parameter_values = 0.0
            for module_path in group.module_paths:
                module = modules.get(module_path)
                weight = getattr(module, "weight", None)
                if weight is None:
                    raise RuntimeError(f"anchor_quantization_group_weight_missing:{module_path}")
                output_axis = 1 if isinstance(module, nn.ConvTranspose2d) else 0
                metric = quantization_perturbation_metrics(
                    weight,
                    gradient=fisher.gradients.get(f"{module_path}.weight"),
                    fisher=fisher.fisher_diag.get(f"{module_path}.weight"),
                    output_axis=output_axis,
                )
                count = float(weight.numel())
                taylor += float(metric["taylor_loss"])
                sqnr += float(metric["sqnr_loss"])
                saturated_values += float(metric["saturation_ratio"]) * count
                parameter_values += count
            prior, reasons = _sensitivity_prior(group.module_paths)
            raw_rows.append(
                {
                    "group": group,
                    "macs": group_macs,
                    "taylor": taylor,
                    "sqnr": sqnr,
                    "prior": prior,
                    "prior_reasons": reasons,
                    "saturation": saturated_values / max(parameter_values, 1.0),
                }
            )
        taylor_total = sum(float(row["taylor"]) for row in raw_rows)
        sqnr_total = sum(float(row["sqnr"]) for row in raw_rows)
        result = []
        payload = []
        for row in raw_rows:
            group = row["group"]
            sensitivity = QuantizationSensitivity(
                group_id=group.group_id,
                module_paths=tuple(group.module_paths),
                macs=float(row["macs"]),
                taylor_loss=float(row["taylor"]) / max(taylor_total, 1.0e-30),
                sqnr_loss=float(row["sqnr"]) / max(sqnr_total, 1.0e-30),
                prior_multiplier=float(row["prior"]),
                saturation_ratio=float(row["saturation"]),
            )
            result.append(sensitivity)
            payload.append({**sensitivity.to_dict(), "prior_reasons": row["prior_reasons"]})
        _write_json(self.run_dir / "planning" / "anchor_c_quantization_sensitivity.json", payload)
        return result

    def _deploy_anchor(
        self,
        *,
        name: str,
        raw: CandidateGenotype,
        repaired: CandidateGenotype,
        phenotype: CandidatePhenotype,
        proxy: Mapping[str, Any],
        physical_planning: Mapping[str, Any],
        context: Any,
        smoke_evaluator: LidarPyramidRealEvaluator,
        full_evaluator: LidarPyramidRealEvaluator,
        calibration_manifest_hash: str,
        saturation_ratio: float,
    ) -> dict[str, Any]:
        target = None if name == "A" else float(dict(self.config.get("stage2", {})).get("target_bops_retention", 0.21))
        smoke_evaluator.target_bops_retention = target
        candidate_id = candidate_hash(phenotype, context.search_space)
        destination = self.run_dir / "anchors" / name
        deployment_dir = destination / "deployment_smoke10"
        raw_result = smoke_evaluator._deploy_and_evaluate(
            phenotype=phenotype,
            output_dir=deployment_dir,
            candidate_label=f"anchor_{name.lower()}",
            pruned_unit_ids=phenotype.pruned_unit_ids,
        )
        if str(raw_result.get("status")) != "ok":
            result = {
                "anchor": name,
                "status": str(raw_result.get("status", "deployment_failed")),
                "failure_reason": str(raw_result.get("failure_reason", "")),
                "candidate_hash": candidate_id,
            }
            _write_json(destination / "anchor_result.json", result)
            return result
        smoke = dict(raw_result["evaluation"])
        if int(smoke.get("num_evaluated_frames", 0)) != 10 or int(smoke.get("num_skipped_frames", -1)) != 0:
            raise RuntimeError(f"anchor_{name}_smoke_manifest_failure")
        full_dir = destination / "evaluation_200"
        full = full_evaluator._evaluate_engine(raw_result["engine_path"], full_dir)
        if str(full.get("status")) != "ok":
            result = {
                "anchor": name,
                "status": "evaluation_failed",
                "failure_reason": str(full.get("failure_reason", "")),
                "candidate_hash": candidate_id,
            }
            _write_json(destination / "anchor_result.json", result)
            return result
        if int(full.get("num_evaluated_frames", 0)) != 200 or int(full.get("num_skipped_frames", -1)) != 0:
            raise RuntimeError(f"anchor_{name}_full_manifest_failure")
        realized_bops = _read_json(deployment_dir / "realized_bops_audit.json")
        requested = _read_json(deployment_dir / "requested_precision_profile.json")
        realized = _read_json(deployment_dir / "engine_realized_precision_profile.json")["realized_precision_profile"]
        realized_parameterized = {key: realized[key] for key in requested if key in realized}
        identity = precision_identity_audit(
            raw=raw,
            repaired=repaired,
            requested_module_profile=requested,
            realized_module_profile=realized_parameterized,
            space=context.search_space,
        )
        typed_profile = validate_strongly_typed_profile(
            requested,
            realized_parameterized,
            builder_flags=context.search_space.builder_flags,
        )
        plugin = _plugin_dtype_audit(deployment_dir / "engine_layer_info.json")
        qdq = _qdq_boundary_summary(deployment_dir / "production_qdq_boundary_audit.json")
        merge = _merge_summary(deployment_dir / "merge_precision_realization.json")
        lineage = _read_json(deployment_dir / "deployment_lineage.json")
        physical_macs = sum(float(row["MACs"]) for row in realized_bops["breakdown"])
        baseline_macs = float(realized_bops["fp32_reference_bops"]) / (32.0 * 32.0)
        physical_bops = float(physical_planning.get("physical_bops_retention", proxy["R_bops_vs_fp32"]))
        decomposition = decompose_bops_retention(
            raw_proxy=float(proxy["R_bops_vs_fp32"]),
            repaired_proxy=float(proxy["R_bops_vs_fp32"]),
            physical=physical_bops,
            realized=float(realized_bops["bops_retention"]),
        )
        functional_precision = str(realized.get("pyramid_backbone.functional_affine_grid_matmul", ""))
        profile_counts = _precision_counts(realized, include_functional_fp16=False)
        result = {
            "anchor": name,
            "status": "ok",
            "candidate_hash": candidate_id,
            "candidate_semantics": validate_anchor_semantics(
                {"A": "fp16_baseline", "B": "fp16_pruning", "C": "mixed_no_prune"}[name],
                repaired,
                context.search_space,
            ),
            "AP@0.3": float(full["AP@0.3"]),
            "AP@0.5": float(full["AP@0.5"]),
            "AP@0.7": float(full["AP@0.7"]),
            "mAP": float(full["mAP"]),
            "forward_p50_ms": float(full["forward_p50_ms"]),
            "forward_p90_ms": float(full["forward_p90_ms"]),
            "forward_p95_ms": float(full["forward_p95_ms"]),
            "FPS": 1000.0 / max(float(full["forward_p50_ms"]), 1.0e-12),
            "physical_params": int(realized_bops["physical_params"]),
            "parameter_retention": float(realized_bops["parameter_retention"]),
            "MAC_retention": physical_macs / max(baseline_macs, 1.0),
            "INT8_MAC_share": sum(
                float(row["MACs"])
                for row in realized_bops["breakdown"]
                if str(row["realized_precision"]) == "INT8"
            ) / max(physical_macs, 1.0),
            "R_BOPS_theoretical": theoretical_bops_retention("FP16") if name in {"A", "B"} else None,
            "R_BOPS_proxy": float(proxy["R_bops_vs_fp32"]),
            "R_BOPS_physical": physical_bops,
            "R_BOPS_realized": float(realized_bops["bops_retention"]),
            "BOPS_decomposition": decomposition,
            "engine_sha256": _file_sha256(raw_result["engine_path"]),
            "engine_size_bytes": Path(raw_result["engine_path"]).stat().st_size,
            "physical_model_hash": str(raw_result["physical_hash"]),
            "deployment_hash": str(raw_result["deployment_hash"]),
            "requested_precision_profile_hash": identity["profile_hashes"]["requested_precision_profile_hash"],
            "realized_precision_profile_hash": identity["profile_hashes"]["realized_precision_profile_hash"],
            "calibration_manifest_hash": calibration_manifest_hash,
            "validation_manifest_hash": str(full["eval_manifest_hash"]),
            "plugin_io_dtype": plugin,
            "canonical_precision_counts": profile_counts,
            "canonical_compute_count": sum(profile_counts.values()),
            "unresolved_count": 0 if functional_precision == "FP16" and len(realized) == 70 else 1,
            "precision_identity_audit": identity,
            "strongly_typed_profile_audit": typed_profile,
            "QDQ_boundary_audit": qdq,
            "merge_audit": merge,
            "saturation_ratio": float(saturation_ratio),
            "saturation_definition": "INT8 per-output-channel weight code abs(code)>=127; zero when no INT8 weights",
            "evaluated": int(full["num_evaluated_frames"]),
            "skipped": int(full["num_skipped_frames"]),
            "smoke_evaluated": int(smoke["num_evaluated_frames"]),
            "smoke_skipped": int(smoke["num_skipped_frames"]),
            "qdq_topology_hash": str(lineage.get("qdq_topology_hash", "")),
            "artifact_dir": str(destination),
        }
        required_passes = (
            result["candidate_semantics"]["passed"],
            identity["passed"],
            typed_profile["passed"],
            plugin["passed"],
            qdq["passed"],
            merge["passed"],
        )
        result["passed"] = all(required_passes) and (
            name == "A"
            or realized_bops_gate(
                result["R_BOPS_realized"],
                target=float(dict(self.config.get("stage2", {})).get("target_bops_retention", 0.21)),
                tolerance=float(dict(self.config.get("stage2", {})).get("bops_tolerance", 0.005)),
            )["passed"]
        )
        _write_json(destination / "raw_genotype.json", raw.to_dict())
        _write_json(destination / "repaired_genotype.json", repaired.to_dict())
        _write_json(destination / "phenotype.json", phenotype.to_dict())
        _write_json(destination / "anchor_result.json", result)
        return result

    def run(self, *, planning_only: bool = False) -> dict[str, Any]:
        run_dir = self._new_run_dir()
        self.run_dir = run_dir
        (run_dir / "planning").mkdir(parents=True, exist_ok=True)
        _write_json(
            run_dir / "run_state.json",
            {
                "status": "running",
                "stage_a_allowed": False,
                "stage_a_started": False,
                "stage_b_allowed": False,
                "ga_invoked": False,
                "git_commit": _git_commit(self.repo),
            },
        )
        (run_dir / "resolved_config.yaml").write_text(
            yaml.safe_dump(self.config, sort_keys=False), encoding="utf-8"
        )
        context = self._expand_anchor_pruning_context(self._build_context(run_dir))
        runtime = dict(self.config.get("runtime", {}) or {})
        require_gpu_isolation(
            context.physical_gpu_id,
            report_path=run_dir / "gpu_preflight.json",
            allow_foreign_processes=bool(runtime.get("allow_foreign_gpu_processes", False)),
            max_gpu_utilization_pct=int(runtime.get("max_gpu_utilization_pct", 20)),
        )
        plugin_exclusion = validate_plugin_gene_exclusion(context.search_space)
        if not plugin_exclusion["passed"]:
            raise RuntimeError(f"plugin_quantization_gene_failure:{plugin_exclusion}")
        stage2 = dict(self.config.get("stage2", {}) or {})
        available = load_split_frame_ids(context.model_bundle.adapter, context.model_config, split="val")
        smoke_manifest = write_eval_manifest(
            run_dir / "manifests" / "smoke10.json",
            num_frames=int(dict(self.config.get("anchor_study", {})).get("smoke_frames", 10)),
            warmup_frames=int(stage2.get("warmup_frames", 20)),
            available_frame_ids=available,
            reset_after_warmup=True,
        )
        smoke_context = replace(
            context,
            eval_manifest_path=smoke_manifest.path,
            eval_frame_ids=smoke_manifest.frame_ids,
            eval_manifest_hash=smoke_manifest.manifest_hash,
        )
        unit_slices = build_unit_parameter_slices(context.model, context.atomic_prune_units)
        baseline_profile = profile_runtime_layer_shapes(
            context.model,
            context.trace_example_inputs,
            forward_fn=context.model_bundle.adapter.forward_for_task,
        )
        _write_json(run_dir / "planning" / "baseline_runtime_shapes.json", baseline_profile.to_dict())
        proxy = dict(self.config.get("proxy", {}) or {})
        fisher = collect_or_load_fisher_statistics(
            model=context.model,
            adapter=context.model_bundle.adapter,
            model_config_path=context.model_config,
            device=torch.device(context.runtime_device),
            cache_path=run_dir / "archives" / "fisher_statistics.pt",
            num_batches=int(proxy.get("fisher_calibration_batches", 8)),
            checkpoint_hash=context.checkpoint_hash,
            code_commit=context.code_commit,
        )
        scorer = TorchBatchedProxyScorer.from_components(
            model=context.model,
            space=context.search_space,
            unit_to_parameter_slices=unit_slices,
            fisher_statistics=fisher,
            runtime_shapes=baseline_profile.shapes,
            normalization=NormalizationStats(),
            config=ProxyObjectiveConfig(
                alpha_fisher=float(proxy.get("alpha_fisher", 0.55)),
                beta_sqnr=float(proxy.get("beta_sqnr", 0.25)),
                gamma_size=float(proxy.get("gamma_size", 0.05)),
                delta_bops=float(proxy.get("delta_bops_penalty", 0.15)),
                bops_threshold=float(stage2.get("target_bops_retention", 0.21)),
            ),
            device=context.runtime_device,
            batch_size=int(proxy.get("batch_size", 128)),
        )
        orchestrator = LidarPyramidTwoStageSearch(
            config=self.config,
            checkpoint=dict(self.config.get("model", {}))["checkpoint"],
            output_root=self.output_root,
        )
        smoke_evaluator = self._make_evaluator(
            smoke_context,
            frames=int(dict(self.config.get("anchor_study", {})).get("smoke_frames", 10)),
            target=float(stage2.get("target_bops_retention", 0.21)),
        )
        full_evaluator = self._make_evaluator(
            context,
            frames=int(stage2.get("num_frames", 200)),
            target=float(stage2.get("target_bops_retention", 0.21)),
        )

        raw_a = make_all_fp16_genotype(context.search_space)
        repaired_a, repair_a = self._repair(orchestrator, context, raw_a)
        phenotype_a = canonicalize_candidate(repaired_a, context.search_space)
        proxy_a = self._proxy_metrics(scorer, [phenotype_a])[0]

        pruning_scan = self._fp16_pruning_scan(
            context=context,
            orchestrator=orchestrator,
            scorer=scorer,
        )
        selected_b, physical_scan = self._physical_pruning_selection(
            context=context,
            evaluator=smoke_evaluator,
            baseline_shapes=baseline_profile.shapes,
            scan=pruning_scan,
        )

        sensitivity = self._quantization_sensitivity(
            context=context,
            fisher=fisher,
            baseline_shapes=baseline_profile.shapes,
        )
        mixed_selection = select_mixed_precision_prefix(
            sensitivity,
            target=float(stage2.get("target_bops_retention", 0.21)),
            tolerance=float(stage2.get("bops_tolerance", 0.005)),
            total_macs=_total_macs(baseline_profile.shapes),
        )
        _write_json(run_dir / "planning" / "anchor_c_profile_selection.json", mixed_selection)
        raw_c = make_mixed_no_prune_genotype(
            context.search_space,
            int8_group_ids=mixed_selection["selected_group_ids"],
        )
        repaired_c, repair_c = self._repair(orchestrator, context, raw_c)
        phenotype_c = canonicalize_candidate(repaired_c, context.search_space)
        proxy_c = self._proxy_metrics(scorer, [phenotype_c])[0]
        c_proxy_gate = realized_bops_gate(
            float(proxy_c["R_bops_vs_fp32"]),
            target=float(stage2.get("target_bops_retention", 0.21)),
            tolerance=float(stage2.get("bops_tolerance", 0.005)),
        )
        planning = {
            "anchor_A": {
                "raw": raw_a.to_dict(),
                "repaired": repaired_a.to_dict(),
                "repair_report": repair_a,
                "phenotype": phenotype_a.to_dict(),
                "proxy": proxy_a,
            },
            "anchor_B": {
                "found": selected_b is not None,
                "scan_candidate_count": len(pruning_scan),
                "physical_scan": physical_scan,
                "selected": None if selected_b is None else {
                    "raw": selected_b["raw"].to_dict(),
                    "repaired": selected_b["repaired"].to_dict(),
                    "repair_report": selected_b["repair_report"],
                    "phenotype": selected_b["phenotype"].to_dict(),
                    "proxy": selected_b["proxy"],
                    "physical_planning": selected_b["physical_planning"],
                },
            },
            "anchor_C": {
                "selection": mixed_selection,
                "proxy_gate": c_proxy_gate,
                "raw": raw_c.to_dict(),
                "repaired": repaired_c.to_dict(),
                "repair_report": repair_c,
                "phenotype": phenotype_c.to_dict(),
                "proxy": proxy_c,
            },
            "plugin_gene_exclusion": plugin_exclusion,
            "ga_invoked": False,
        }
        _write_json(run_dir / "anchor_planning.json", planning)
        if planning_only:
            _write_json(
                run_dir / "run_state.json",
                {
                    "status": "planning_complete",
                    "stage_a_allowed": False,
                    "stage_a_started": False,
                    "stage_b_allowed": False,
                    "ga_invoked": False,
                    "git_commit": context.code_commit,
                },
            )
            return {"run_dir": str(run_dir), "status": "planning_complete", "planning": planning}

        calibration_identity = fixed_k_calibration_npz_manifest_identity(
            context.quant_calibration_npz_manifest,
            num_batches=context.quant_calibration_batches,
            fixed_k=29696,
        )
        calibration_manifest_hash = str(
            calibration_identity.get("tensor_manifest_hash")
            or calibration_identity.get("manifest_sha256")
            or canonical_json_hash(calibration_identity)
        )
        strict_fp32 = full_evaluator.evaluate_original_baseline("strict_fp32", full_validation=False)
        if str(strict_fp32.get("status")) != "ok":
            raise RuntimeError(f"strict_fp32_anchor_reference_failed:{strict_fp32}")
        result_a = self._deploy_anchor(
            name="A",
            raw=raw_a,
            repaired=repaired_a,
            phenotype=phenotype_a,
            proxy=proxy_a,
            physical_planning={"physical_bops_retention": float(proxy_a["R_bops_vs_fp32"])},
            context=context,
            smoke_evaluator=smoke_evaluator,
            full_evaluator=full_evaluator,
            calibration_manifest_hash=calibration_manifest_hash,
            saturation_ratio=0.0,
        )
        if selected_b is None:
            result_b = {
                "anchor": "B",
                "status": "no_legal_physical_width_in_budget",
                "passed": False,
            }
        else:
            result_b = self._deploy_anchor(
                name="B",
                raw=selected_b["raw"],
                repaired=selected_b["repaired"],
                phenotype=selected_b["phenotype"],
                proxy=selected_b["proxy"],
                physical_planning=selected_b["physical_planning"],
                context=context,
                smoke_evaluator=smoke_evaluator,
                full_evaluator=full_evaluator,
                calibration_manifest_hash=calibration_manifest_hash,
                saturation_ratio=0.0,
            )
        selected_sensitivity = {
            row.group_id: row for row in sensitivity if row.group_id in set(mixed_selection["selected_group_ids"])
        }
        selected_weight_count = sum(
            int(group.parameter_count)
            for group in context.search_space.quantization_groups
            if group.group_id in selected_sensitivity
        )
        c_saturation = sum(
            selected_sensitivity[group.group_id].saturation_ratio * int(group.parameter_count)
            for group in context.search_space.quantization_groups
            if group.group_id in selected_sensitivity
        ) / max(selected_weight_count, 1)
        if not c_proxy_gate["passed"]:
            result_c = {
                "anchor": "C",
                "status": "no_legal_mixed_profile_in_budget",
                "passed": False,
                "proxy_gate": c_proxy_gate,
            }
        else:
            result_c = self._deploy_anchor(
                name="C",
                raw=raw_c,
                repaired=repaired_c,
                phenotype=phenotype_c,
                proxy=proxy_c,
                physical_planning={"physical_bops_retention": float(proxy_c["R_bops_vs_fp32"])},
                context=context,
                smoke_evaluator=smoke_evaluator,
                full_evaluator=full_evaluator,
                calibration_manifest_hash=calibration_manifest_hash,
                saturation_ratio=c_saturation,
            )
        anchors = {"A": result_a, "B": result_b, "C": result_c}
        manifest_audit = validate_manifest_consistency(anchors)
        stage2 = dict(self.config.get("stage2", {}) or {})
        low_damage_observation = identify_low_damage_bops_path(
            strict_fp32,
            {"B": result_b, "C": result_c},
            target=float(stage2.get("target_bops_retention", 0.21)),
            tolerance=float(stage2.get("bops_tolerance", 0.005)),
            expected_frames=int(stage2.get("num_frames", 200)),
        )
        summary = {
            "status": "complete",
            "git_commit": context.code_commit,
            "gpu": context.gpu_selection.to_dict(),
            "plugin_sha256": next(iter(context.search_space.plugin_hashes.values())),
            "calibration_manifest_hash": calibration_manifest_hash,
            "validation_manifest_hash": context.eval_manifest_hash,
            "strict_fp32": strict_fp32,
            "anchors": anchors,
            "manifest_consistency": manifest_audit,
            "low_damage_path_observation": low_damage_observation,
            "ANCHOR_FP16_BASELINE_PASS": bool(result_a.get("passed", False)),
            "ANCHOR_FP16_PRUNING_PASS": bool(result_b.get("passed", False)),
            "ANCHOR_MIXED_NO_PRUNE_PASS": bool(result_c.get("passed", False)),
            "LOW_DAMAGE_BOPS_021_PATH_IDENTIFIED": bool(low_damage_observation["identified"]),
            "MULTIGPU_TOP5_SMOKE_PASS": False,
            "STAGE_A_ALLOWED": False,
            "STAGE_A_STARTED": False,
            "STAGE_B_ALLOWED": False,
            "ga_invoked": False,
        }
        _write_json(run_dir / "anchor_study_summary.json", summary)
        self._write_csv(run_dir / "anchor_results.csv", anchors)
        _write_json(
            run_dir / "run_state.json",
            {
                "status": "complete",
                "stage_a_allowed": False,
                "stage_a_started": False,
                "stage_b_allowed": False,
                "ga_invoked": False,
                "git_commit": context.code_commit,
            },
        )
        return {"run_dir": str(run_dir), **summary}

    @staticmethod
    def _write_csv(path: Path, anchors: Mapping[str, Mapping[str, Any]]) -> None:
        fields = [
            "anchor",
            "status",
            "pruning",
            "MAC_retention",
            "INT8_MAC_share",
            "realized_profile",
            "R_BOPS_realized",
            "AP@0.3",
            "AP@0.5",
            "AP@0.7",
            "mAP",
            "forward_p50_ms",
            "forward_p90_ms",
            "forward_p95_ms",
        ]
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for name, row in anchors.items():
                writer.writerow(
                    {
                        "anchor": name,
                        "status": row.get("status"),
                        "pruning": int(row.get("candidate_semantics", {}).get("pruned_unit_count", 0) or 0),
                        "MAC_retention": row.get("MAC_retention"),
                        "INT8_MAC_share": row.get("INT8_MAC_share"),
                        "realized_profile": json.dumps(row.get("canonical_precision_counts", {}), sort_keys=True),
                        "R_BOPS_realized": row.get("R_BOPS_realized"),
                        "AP@0.3": row.get("AP@0.3"),
                        "AP@0.5": row.get("AP@0.5"),
                        "AP@0.7": row.get("AP@0.7"),
                        "mAP": row.get("mAP"),
                        "forward_p50_ms": row.get("forward_p50_ms"),
                        "forward_p90_ms": row.get("forward_p90_ms"),
                        "forward_p95_ms": row.get("forward_p95_ms"),
                    }
                )


def _load_config(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    return dict(yaml.safe_load(source.read_text(encoding="utf-8")) or {})


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deterministic BOPS 0.21 anchors without GA.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--gpu-id", default=None)
    parser.add_argument("--planning-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = _load_config(args.config)
    if args.gpu_id is not None:
        runtime = dict(config.get("runtime", {}) or {})
        runtime["gpu_id"] = str(args.gpu_id)
        config["runtime"] = runtime
    study = Bops021AnchorStudy(config=config, output_root=args.output_root)
    result = study.run(planning_only=bool(args.planning_only))
    command = "python -m search.anchors.runner " + " ".join(
        shlex.quote(value)
        for value in ["--config", str(Path(args.config).resolve()), "--output-root", str(Path(args.output_root).resolve()), *(["--gpu-id", str(args.gpu_id)] if args.gpu_id is not None else []), *(["--planning-only"] if args.planning_only else [])]
    )
    run_dir = Path(result["run_dir"])
    (run_dir / "commands.sh").write_text(command + "\n", encoding="utf-8")
    anchors = dict(result.get("anchors", {}) or {})
    print(
        json.dumps(
            {
                "run_dir": result["run_dir"],
                "status": result.get("status"),
                "anchor_status": {
                    name: {"status": row.get("status"), "passed": row.get("passed")}
                    for name, row in anchors.items()
                },
                "STAGE_A_STARTED": bool(result.get("STAGE_A_STARTED", False)),
                "STAGE_B_ALLOWED": bool(result.get("STAGE_B_ALLOWED", False)),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
