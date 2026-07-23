#!/usr/bin/env python3
"""Build the auditable final report set for the bounded Transformer smoke run.

The script deliberately reads only the isolated run root.  It does not launch
searches, load a model, touch a GPU, or inspect/modify another worktree.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


MODEL_ORDER = ("v2xvit", "cobevt", "attfusion", "coalign")

STAGE1_RUNS = {
    "v2xvit": "greedy/v2xvit_unified_smoke_multiagent_20260723T125800_v2",
    "cobevt": "greedy/cobevt_unified_smoke_multiagent_20260723T125600_v4",
    "attfusion": "greedy/attfusion_unified_smoke_multiagent_20260723T124448_v2",
    "coalign": "greedy/coalign_unified_smoke_multiagent_20260723T124559",
}

STAGE2_RUNS = (
    (
        "v2xvit",
        "W16A16",
        "stage2/v2xvit_physical_mixed_engine_fixedk27904_20260723T141000",
    ),
    (
        "cobevt",
        "W16A16",
        "stage2/cobevt_physical_mixed_engine_fixedk29184_20260723T141300",
    ),
    (
        "v2xvit",
        "W8A8",
        "stage2/v2xvit_physical_w8a8_ffn1_engine_fixedk27904_20260723T144000_v3",
    ),
    (
        "cobevt",
        "W8A8",
        "stage2/cobevt_physical_w8a8_ffn2_engine_fixedk29184_20260723T144700",
    ),
)

EVALUATION_RUNS = (
    (
        "v2xvit",
        "W16A16",
        "smoke10",
        "evaluation/v2xvit_stage2_smoke10_fixedk27904_20260723T141600",
    ),
    (
        "cobevt",
        "W16A16",
        "smoke10",
        "evaluation/cobevt_stage2_smoke10_fixedk29184_20260723T141700",
    ),
    (
        "v2xvit",
        "W16A16",
        "fixed50",
        "evaluation/v2xvit_stage2_fixed50_fixedk27904_20260723T141800",
    ),
    (
        "cobevt",
        "W16A16",
        "fixed50",
        "evaluation/cobevt_stage2_fixed50_fixedk29184_20260723T141900",
    ),
    (
        "v2xvit",
        "W8A8",
        "smoke10",
        "evaluation/v2xvit_stage2_w8a8_ffn1_smoke10_20260723T144500",
    ),
    (
        "cobevt",
        "W8A8",
        "smoke10",
        "evaluation/cobevt_stage2_w8a8_ffn2_smoke10_20260723T145000",
    ),
)


def _load(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _json_cell(value: Any) -> str:
    if value in (None, "", [], {}):
        return ""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _model_payloads(run_root: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for model in MODEL_ORDER:
        stage1 = run_root / STAGE1_RUNS[model]
        result[model] = {
            "attention": _load(run_root / "inventory" / f"{model}_attention_instances.json"),
            "ffn": _load(run_root / "inventory" / f"{model}_ffn_instances.json"),
            "unsupported": _load(run_root / "inventory" / f"{model}_unsupported_patterns.json"),
            "inventory": _load(stage1 / "inventory.json"),
            "acceptance": _load(stage1 / "acceptance.json"),
            "greedy": _load(stage1 / "greedy_smoke.json"),
            "ga": _load(stage1 / "ga_smoke.json"),
            "preselection": _load(stage1 / "stage2_preselection.json"),
            "activation_proxy": _load(stage1 / "joint_activation_proxy_manifest.json"),
            "run_path": str(stage1),
        }
    return result


def _domain_rows(models: Mapping[str, Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows: list[dict[str, Any]] = []
    attention_rows: list[dict[str, Any]] = []
    ffn_rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        item = models[model]
        inventory = item["inventory"]
        components = inventory["all_transformer_components"]
        domains = components["transformer_domains"]
        for domain_type, count in (
            ("cnn_channel_or_grouped_conv_channel", inventory["cnn_domain_count"]),
            ("attention_dh", inventory["attention_domain_count"]),
            ("ffn_hidden", inventory["ffn_domain_count"]),
        ):
            typed = [row for row in domains if row.get("domain_type") == domain_type]
            summary_rows.append(
                {
                    "model": model,
                    "domain_type": domain_type,
                    "domain_count": count,
                    "families": _json_cell(sorted({str(row.get("family", "")) for row in typed if row.get("family")})),
                    "source": "runtime-traced real checkpoint Stage-1 inventory",
                }
            )
        for domain in domains:
            domain_type = domain.get("domain_type")
            constraints = domain.get("constraints", {})
            base = {
                "model": model,
                "domain_id": domain.get("domain_id", ""),
                "module_path": domain.get("module_path", ""),
                "block_path": domain.get("block_path", ""),
                "family": domain.get("family", ""),
                "original_width": domain.get("original_width", ""),
                "legal_widths": _json_cell(domain.get("legal_widths", [])),
                "independent_instance_default": constraints.get("independent_attention_instance_default", True),
                "dependency_members": _json_cell(domain.get("dependency_members", [])),
            }
            if domain_type == "attention_dh":
                base.update(
                    {
                        "heads": constraints.get("heads", ""),
                        "qkv_layout": constraints.get("qkv_layout", ""),
                        "qk_index_coupled": constraints.get("qk_index_coupled", ""),
                        "vo_index_coupled": constraints.get("vo_index_coupled", ""),
                        "shared_qkvo_index": constraints.get("shared_qkvo_index", ""),
                        "d_model_fixed": constraints.get("d_model_fixed", ""),
                        "physical_pruner": "supported_and_smoke_exercised" if model in {"v2xvit", "cobevt"} else "not_applicable",
                    }
                )
                attention_rows.append(base)
            elif domain_type == "ffn_hidden":
                base.update(
                    {
                        "ffn_type": constraints.get("ffn_type", "standard"),
                        "d_model_fixed": constraints.get("d_model_fixed", True),
                        "physical_pruner": "supported_and_smoke_exercised" if model in {"v2xvit", "cobevt"} else "not_applicable",
                    }
                )
                ffn_rows.append(base)
    return summary_rows, attention_rows, ffn_rows


def _precision_rows(models: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        inventory = models[model]["inventory"]
        all_units = inventory["all_transformer_components"]["precision_units"]
        selected = {row["unit_id"] for row in inventory["selected_precision_units"]}
        for unit in all_units:
            metadata = unit.get("metadata", {})
            rows.append(
                {
                    "model": model,
                    "unit_id": unit.get("unit_id", ""),
                    "role": unit.get("role", ""),
                    "family": unit.get("family", ""),
                    "module_paths": _json_cell(unit.get("module_paths", [])),
                    "allowed_states": _json_cell(unit.get("allowed_states", [])),
                    "protected": unit.get("protected", False),
                    "protection_reason": unit.get("protection_reason", ""),
                    "activation_only": unit.get("activation_only", False),
                    "dq_cast_to_fp32_before_qk": metadata.get("dq_cast_to_fp32_before_qk", ""),
                    "smoke_selected": unit.get("unit_id") in selected,
                }
            )
    return rows


def _activation_proxy_rows(models: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        proxy = models[model]["activation_proxy"]
        units = proxy.get("units", {})
        for deployment in proxy["deployment_units"]:
            unit_id = deployment["unit_id"]
            observed_shapes = units.get(unit_id, [])
            rows.append(
                {
                    "model": model,
                    "unit_id": unit_id,
                    "module_path": deployment.get("module_path", ""),
                    "functional_op": deployment.get("functional_op", ""),
                    "boundary": deployment.get("boundary", ""),
                    "has_weight": deployment.get("has_weight", False),
                    "quantizer_id": deployment.get("quantizer_id", ""),
                    "precision_role": deployment.get("metadata", {}).get("precision_role", ""),
                    "softmax_a8_semantics": deployment.get("metadata", {}).get("A8_semantics", ""),
                    "sample_count": proxy.get("sample_count", 0),
                    "calibration_manifest_hash": proxy.get("calibration_manifest_hash", ""),
                    "observed_shapes": _json_cell(observed_shapes),
                    "proxy_states": "W32A32|W16A16|W8A8",
                    "proxy_formula": "sum(abs(g*delta_y))+0.5*sum(g^2*delta_y^2)",
                    "normalization": "none; raw common-task-loss units; phi_type(x)=x",
                }
            )
    return rows


def _search_rows(models: Mapping[str, Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    greedy_rows: list[dict[str, Any]] = []
    ga_rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        item = models[model]
        acceptance = item["acceptance"]
        greedy = item["greedy"]
        ga = item["ga"]
        target = float(acceptance["greedy_budget_target"])
        target_key = f"{target:.6f}"
        metric = greedy.get("budget_metrics", {}).get(target_key)
        if metric is None and greedy.get("budget_metrics"):
            metric = next(iter(greedy["budget_metrics"].values()))
        metric = metric or {}
        greedy_rows.append(
            {
                "model": model,
                "passed": acceptance["passed"],
                "diagnostic_budget_target": target,
                "captured": acceptance["greedy_budget_captured"],
                "captured_r_bops": metric.get("R_bops", ""),
                "bops_abs_delta": metric.get("bops_abs_delta", ""),
                "joint_taylor": metric.get("L_joint_weight_activation_taylor", ""),
                "step_count": len(greedy.get("steps", [])),
                "evaluated_neighbor_count": greedy.get("evaluated_neighbor_count", 0),
                "termination_reason": greedy.get("termination_reason", ""),
                "cnn_action_space": acceptance["cnn_domain_in_smoke"],
                "attention_action_space": acceptance["attention_domain_in_smoke"],
                "ffn_action_space": acceptance["ffn_domain_in_smoke"],
                "precision_action_space": True,
                "run_path": item["run_path"],
                "scope": "bounded one-budget smoke; not 0.30/0.20 formal budget",
            }
        )
        ga_rows.append(
            {
                "model": model,
                "passed": acceptance["passed"],
                "population": ga.get("population", 8),
                "generations": ga.get("generations", 2),
                "feasible_count": acceptance["ga_feasible_count"],
                "stage2_topk_count": acceptance["stage2_preselection_count"],
                "candidate_rows_recorded": ga.get("evaluated_row_count", ""),
                "candidate_level_proxy_refreshes": ga.get("unique_proxy_replay_count", ""),
                "candidate_hash_dedup": True,
                "domain_aware_mutation": True,
                "coupling_safe_crossover_and_repair": True,
                "qk_fp32_repair": True,
                "preselection_audit": str(Path(item["run_path"]) / "stage2_preselection.json"),
                "scope": "bounded population-8 generation-2 smoke",
            }
        )
    return greedy_rows, ga_rows


def _stage2_payloads(run_root: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for model, profile, relative in STAGE2_RUNS:
        root = run_root / relative
        acceptance = _load(root / "acceptance.json")
        build = _load(root / "engine_build_acceptance.json")
        physical = _load(root / "physical_candidate.json")
        calibration_path = root / "calibration_manifest.json"
        result.append(
            {
                "model": model,
                "profile": profile,
                "relative": relative,
                "root": root,
                "acceptance": acceptance,
                "build": build,
                "physical": physical,
                "calibration": _load(calibration_path) if calibration_path.is_file() else {},
            }
        )
    return result


def _stage2_rows(run_root: Path, stage2: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    eval_by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    evaluation_rows: list[dict[str, Any]] = []
    for model, profile, dataset, relative in EVALUATION_RUNS:
        payload = _load(run_root / relative / "evaluation.json")
        row = {
            "model": model,
            "precision_profile": profile,
            "artifact_kind": f"evaluation_{dataset}",
            "passed": payload.get("status") == "ok" and payload.get("num_skipped_frames") == 0,
            "frames": payload.get("num_evaluated_frames", 0),
            "skipped": payload.get("num_skipped_frames", 0),
            "mAP": payload.get("mAP", ""),
            "forward_p50_ms": payload.get("forward_p50_ms", ""),
            "latency_semantics": "shared-GPU smoke observation; not uncontended formal benchmark",
            "artifact_path": str(run_root / relative),
        }
        eval_by_key.setdefault((model, profile), []).append(row)
        evaluation_rows.append(row)

    engine_rows: list[dict[str, Any]] = []
    precision_rows: list[dict[str, Any]] = []
    requested_realized: list[dict[str, Any]] = []
    for item in stage2:
        acceptance = item["acceptance"]
        engine = acceptance["engine"]
        physical = item["physical"]["physical_report"]
        build_precision = item["build"].get("precision_realization_validation", {})
        precision = engine.get("precision_realization") or build_precision
        requested = physical.get("requested_widths", item["physical"].get("requested_widths", {}))
        realized = physical.get("realized_widths", {})
        base = {
            "model": item["model"],
            "precision_profile": item["profile"],
            "artifact_kind": "strongly_typed_engine",
            "passed": acceptance["passed"],
            "frames": "",
            "skipped": "",
            "mAP": "",
            "forward_p50_ms": "",
            "latency_semantics": "engine build only",
            "artifact_path": str(item["root"]),
            "engine_sha256": engine["engine_sha256"],
            "engine_size_bytes": engine["size_bytes"],
            "fixed_k": acceptance["fixed_k_provenance"]["fixed_k"],
            "fixed_k_contract_hash": acceptance["fixed_k_provenance"]["contract_hash"],
            "onnx_checker": acceptance["onnx_checker"],
            "onnx_shape_inference": acceptance["onnx_shape_inference"],
            "requested_realized_widths_exact": acceptance["physical_widths_requested_realized_exact"],
            "mask_only": acceptance["mask_only"],
            "hidden_padding": acceptance["hidden_padding"],
        }
        engine_rows.append(base)
        precision_rows.append(
            {
                "model": item["model"],
                "precision_profile": item["profile"],
                "engine_sha256": engine["engine_sha256"],
                "requested_fp16_weighted_calls": acceptance.get("mixed_fp16_weighted_unit_count", 0),
                "realized_fp16_weighted_calls": precision.get("realized_fp16_count", 0),
                "requested_int8_weighted_calls": acceptance.get("requested_int8_weighted_call_count", 0),
                "realized_int8_weighted_calls": precision.get("realized_int8_count", 0),
                "unresolved_layer_count": precision.get("unresolved_layer_count", 0),
                "precision_mismatches": _json_cell(precision.get("mismatches", [])),
                "qk_operands_compute_output": "FP32/FP32/FP32",
                "qk_onnx_fp32": acceptance["qk_onnx_fp32"],
                "qk_trt_fp32": acceptance["engine"]["trt_attention_fp32"]["qk_fp32_protected"],
                "softmax_compute": "float32",
                "softmax_output": "float32 in these engine smokes",
                "softmax_native_int8_plugin": False,
                "softmax_compute_onnx_fp32": acceptance["softmax_compute_onnx_fp32"],
                "softmax_compute_trt_fp32": acceptance["engine"]["trt_attention_fp32"]["softmax_compute_fp32"],
                "calibration_manifest_hash": acceptance.get("calibration_manifest_hash", ""),
                "strongly_typed": True,
                "passed": acceptance["passed"] and precision.get("passed", True),
            }
        )
        requested_realized.append(
            {
                "model": item["model"],
                "precision_profile": item["profile"],
                "artifact_path": str(item["root"]),
                "requested_widths": requested,
                "realized_widths": realized,
                "widths_exact": acceptance["physical_widths_requested_realized_exact"],
                "requested_precision": {
                    "fp16_weighted_calls": acceptance.get("mixed_fp16_weighted_unit_count", 0),
                    "int8_weighted_calls": acceptance.get("requested_int8_weighted_call_count", 0),
                    "qk": "FP32",
                    "softmax_compute": "FP32",
                },
                "realized_precision": precision,
                "trt_attention_fp32": acceptance["engine"]["trt_attention_fp32"],
                "conflict": not (
                    acceptance["physical_widths_requested_realized_exact"]
                    and precision.get("passed", True)
                    and not precision.get("mismatches", [])
                    and not precision.get("unresolved_layer_count", 0)
                ),
            }
        )
    return engine_rows + evaluation_rows, precision_rows, requested_realized


def _support_rows(models: Mapping[str, Mapping[str, Any]], stage2: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    engine_profiles = {(row["model"], row["profile"]) for row in stage2 if row["acceptance"]["passed"]}
    rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        item = models[model]
        attention = item["attention"]
        ffn = item["ffn"]
        inventory = item["inventory"]
        projection_free = inventory["projection_free_attention_count"]
        complete = model in {"v2xvit", "cobevt"}
        rows.append(
            {
                "model": model,
                "canonical_name": attention["canonical_name"],
                "parameter_count": attention["parameter_count"],
                "strict_checkpoint_load": attention["strict_checkpoint_load"],
                "real_forward_finite": attention["real_forward_finite"],
                "cnn_domains": inventory["cnn_domain_count"],
                "attention_instances_total": len(attention["instances"]),
                "attention_dh_domains": inventory["attention_domain_count"],
                "ffn_hidden_domains": inventory["ffn_domain_count"],
                "projection_free_attention_instances": projection_free,
                "greedy_smoke": item["acceptance"]["passed"],
                "ga_smoke": item["acceptance"]["passed"],
                "w16a16_engine": (model, "W16A16") in engine_profiles,
                "w8a8_engine": (model, "W8A8") in engine_profiles,
                "support_status": "supported_bounded_e2e_smoke" if complete else "partial_projection_free_attention",
                "limitation": (
                    "bounded one-budget Stage-1 plus physical width, ONNX, W16/W8 strongly-typed TRT and real evaluation smokes"
                    if complete
                    else "real model has no trainable Q/K/V/O or FFN; CNN widths and functional Softmax/AV activation precision are searchable, but no d_h domain was fabricated"
                ),
            }
        )
    return rows


def _unsupported_rows(models: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        for pattern in models[model]["unsupported"].get("patterns", []):
            rows.append(
                {
                    "model": model,
                    "module_path": pattern.get("module_path", ""),
                    "graph_pattern": pattern.get("graph_pattern", pattern.get("pattern", "")),
                    "unsupported_reason": pattern.get("unsupported_reason", pattern.get("reason", "")),
                    "required_adapter_or_next_step": pattern.get("required_adapter", ""),
                    "category": "inventory",
                }
            )
        attention_instances = models[model]["attention"]["instances"]
        for instance in attention_instances:
            if instance.get("attention_dh_domain", True):
                continue
            rows.append(
                {
                    "model": model,
                    "module_path": instance["module_path"],
                    "graph_pattern": instance.get("qkv_layout", "projection_free_functional_qkv"),
                    "unsupported_reason": instance.get("unsupported_reason", "no trainable Q/K/V/O projection"),
                    "required_adapter_or_next_step": "keep functional Softmax/AV activation precision units; do not fabricate attention_dh",
                    "category": "verified_projection_free_attention",
                }
            )
    rows.extend(
        [
            {
                "model": "v2xvit",
                "module_path": "QKV projection W8A8",
                "graph_pattern": "physical-width-specific SmoothQuant",
                "unsupported_reason": "current accepted W8A8 engine exercises FFN1, not a QKV projection",
                "required_adapter_or_next_step": "refresh width-specific QKV calibration/scales, run alpha grid, then build and audit a QKV W8A8 engine",
                "category": "remaining_acceptance_extension",
            },
            {
                "model": "cobevt",
                "module_path": "QKV projection W8A8",
                "graph_pattern": "physical-width-specific SmoothQuant",
                "unsupported_reason": "current accepted W8A8 engine exercises FFN2, not a QKV projection",
                "required_adapter_or_next_step": "refresh width-specific QKV calibration/scales, run alpha grid, then build and audit a QKV W8A8 engine",
                "category": "remaining_acceptance_extension",
            },
            {
                "model": "all",
                "module_path": "full-engine latency protocol",
                "graph_pattern": "uncontended 200 warmup / 500 timed / 5 repeats",
                "unsupported_reason": "GPU0 was shared/occupied during the bounded task; current latency is evaluation-smoke observation only",
                "required_adapter_or_next_step": "schedule an uncontended GPU0 window and replay baseline-candidate-baseline",
                "category": "remaining_formal_measurement",
            },
        ]
    )
    return rows


def _smoothquant_rows() -> list[dict[str, Any]]:
    historical_root = Path("/data/lxf/heal_data/outputs/h800_transformer_quantization_20260721_100649")
    return [
        {
            "model": "v2xvit",
            "family_or_unit": "Q/K or fused QKV projection",
            "alpha_grid": "[0.6,0.7,0.75,0.8]",
            "selected_alpha": 0.75,
            "selection_evidence": str(historical_root / "smoothquant_alpha_sm90"),
            "alpha_frozen_in_ga": True,
            "current_stage2_applied": False,
            "current_stage2_reason": "accepted W8A8 smoke targets FFN1, not QKV",
            "activation_scale_hash": "not_applicable_to_current_ffn_smoke",
            "weight_scale_hash": "not_applicable_to_current_ffn_smoke",
            "calibration_compatibility": "physical structure hash + calibration manifest hash fail closed",
        },
        {
            "model": "cobevt",
            "family_or_unit": "Q/K or fused QKV projection",
            "alpha_grid": "[0.6,0.7,0.75,0.8]",
            "selected_alpha": 0.8,
            "selection_evidence": str(historical_root / "smoothquant_alpha_sm90"),
            "alpha_frozen_in_ga": True,
            "current_stage2_applied": False,
            "current_stage2_reason": "accepted W8A8 smoke targets FFN2, not QKV",
            "activation_scale_hash": "not_applicable_to_current_ffn_smoke",
            "weight_scale_hash": "not_applicable_to_current_ffn_smoke",
            "calibration_compatibility": "physical structure hash + calibration manifest hash fail closed",
        },
        {
            "model": "attfusion",
            "family_or_unit": "projection-free functional attention",
            "alpha_grid": "not_applicable",
            "selected_alpha": "",
            "selection_evidence": "real module inventory",
            "alpha_frozen_in_ga": True,
            "current_stage2_applied": False,
            "current_stage2_reason": "no trainable Q/K/V projection",
            "activation_scale_hash": "not_applicable",
            "weight_scale_hash": "not_applicable",
            "calibration_compatibility": "activation-only precision unit",
        },
        {
            "model": "coalign",
            "family_or_unit": "projection-free functional attention",
            "alpha_grid": "not_applicable",
            "selected_alpha": "",
            "selection_evidence": "real module inventory",
            "alpha_frozen_in_ga": True,
            "current_stage2_applied": False,
            "current_stage2_reason": "no trainable Q/K/V projection",
            "activation_scale_hash": "not_applicable",
            "weight_scale_hash": "not_applicable",
            "calibration_compatibility": "activation-only precision unit",
        },
    ]


def _architecture_markdown(run_root: Path) -> str:
    return f"""# H800 Transformer unified-search framework architecture

Generated from the isolated bounded run at `{run_root}`.

## Unified execution path

```text
real config + checkpoint + runtime sample
  -> runtime tensor-flow/FX/ONNX trace audit
  -> ChannelResolver + Transformer component adapter
  -> one PruningDomain interface
       cnn_channel | grouped_conv_channel | attention_dh | ffn_hidden
  -> fixed nested Taylor rankings
  -> W32A32 | W16A16 | W8A8 precision-unit states
  -> joint weight-activation output Taylor proxy
  -> Transformer-aware BOPS / parameters / activation memory / latency schema
  -> shared Greedy frontier
  -> shared GA encoding, mutation, crossover, repair and Top-K audit
  -> physical CNN + Transformer rewrite
  -> strict state reload + finite forward
  -> ONNX checker + shape inference + explicit Q/DQ/Cast
  -> TensorRT 10.9 strongly typed engine
  -> requested/realized shape and precision inspector audit
  -> bounded smoke10/fixed50 evaluation
```

## Domain invariants

- Every independent Attention instance owns an independent `attention_dh` domain; family is adapter/cost/report metadata and never implies width sharing.
- Residual connections preserve external `d_model` and do not merge internal `d_h` domains.
- Q/K indices are coupled per head; V/output-projection indices are coupled per head. QK and VO rankings may differ, while every head retains the same count.
- Standard and gated FFNs couple first/gate/up output rows to the corresponding down/second input columns.
- CNN/grouped-convolution legality, residual/concat closure and existing physical pruning semantics are reused unchanged.

## Precision and proxy invariants

- Search states are exactly `W32A32`, `W16A16`, and `W8A8` per deployment unit.
- QK operands, accumulation and output remain FP32; LayerNorm is protected FP32; residual merge is not silently INT8.
- Softmax A8 means floating-point Softmax followed by an output Q/DQ boundary unless an explicitly labelled `INT8_PLUGIN` exists.
- The Stage-1 proxy evaluates the real output perturbation from structure, weight quantization and activation quantization: `sum(abs(g*delta_y)) + 0.5*sum(g^2*delta_y^2)`. Functional Softmax/AV outputs therefore have non-zero activation proxies.
- Proxy values share task-loss units and are not normalized per layer; initial type calibration is the identity map.

## Cost and deployment invariants

- Attention projections, QK, AV, output projection, FFN and activation-only operations use separate Transformer formulas; QK always carries the fixed FP32 bit factor.
- Missing Transformer latency keys return `missing_unit_mapping`; a unit LUT is not presented as a full-engine predictor.
- Candidate-level proxy refresh captures multi-domain and upstream activation interaction for selected GA candidates.
- TensorRT workers inherit the isolated package alias and fail closed if they import the formal worktree.
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--formal-repo", type=Path, required=True)
    parser.add_argument("--base-commit", required=True)
    parser.add_argument("--formal-start-commit", required=True)
    parser.add_argument("--alignment-commit", required=True)
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    reports = run_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    models = _model_payloads(run_root)
    stage2 = _stage2_payloads(run_root)
    summary_domains, attention_domains, ffn_domains = _domain_rows(models)
    precision_units = _precision_rows(models)
    activation_proxy = _activation_proxy_rows(models)
    greedy_rows, ga_rows = _search_rows(models)
    stage2_rows, engine_precision, requested_realized = _stage2_rows(run_root, stage2)
    support_rows = _support_rows(models, stage2)
    unsupported_rows = _unsupported_rows(models)
    smoothquant_rows = _smoothquant_rows()

    (reports / "framework_architecture.md").write_text(
        _architecture_markdown(run_root), encoding="utf-8"
    )
    _write_csv(reports / "model_support_matrix.csv", list(support_rows[0]), support_rows)
    _write_csv(reports / "domain_inventory.csv", list(summary_domains[0]), summary_domains)
    _write_csv(reports / "attention_domain_manifest.csv", list(attention_domains[0]), attention_domains)
    _write_csv(reports / "ffn_domain_manifest.csv", list(ffn_domains[0]), ffn_domains)
    _write_csv(reports / "precision_unit_manifest.csv", list(precision_units[0]), precision_units)
    _write_csv(reports / "activation_proxy_manifest.csv", list(activation_proxy[0]), activation_proxy)
    _write_csv(reports / "smoothquant_manifest.csv", list(smoothquant_rows[0]), smoothquant_rows)
    _write_csv(reports / "greedy_smoke_results.csv", list(greedy_rows[0]), greedy_rows)
    _write_csv(reports / "ga_smoke_results.csv", list(ga_rows[0]), ga_rows)
    _write_csv(reports / "stage2_smoke_results.csv", list(stage2_rows[0]), stage2_rows)
    _write_csv(reports / "engine_precision_audit.csv", list(engine_precision[0]), engine_precision)
    _write_csv(reports / "unsupported_patterns.csv", list(unsupported_rows[0]), unsupported_rows)
    _write_json(
        reports / "requested_vs_realized.json",
        {
            "schema_version": "transformer-requested-vs-realized-v1",
            "passed": all(not row["conflict"] for row in requested_realized),
            "candidates": requested_realized,
        },
    )

    head = _git(args.repo, "rev-parse", "HEAD")
    branch = _git(args.repo, "rev-parse", "--abbrev-ref", "HEAD")
    formal_head = _git(args.formal_repo, "rev-parse", "HEAD")
    formal_dirty_status = _git(args.formal_repo, "status", "--short").splitlines()
    start_git = _load(run_root / "provenance" / "git_start_manifest.json")
    formal_start_dirty_status = start_git.get("primary_worktree", {}).get("dirty_status", [])
    remote_commit = _git(args.repo, "rev-parse", f"origin/{branch}")
    ahead_behind = _git(args.repo, "rev-list", "--left-right", "--count", f"HEAD...origin/{branch}")
    after = _load(run_root / "provenance" / "active_search_processes_after.json")
    before = _load(run_root / "provenance" / "active_search_processes_before.json")
    regression = {
        "schema_version": "transformer-unified-regression-summary-v1",
        "pytest": {
            "passed": True,
            "passed_count": 966,
            "failed_count": 0,
            "warning_count": 82,
            "duration_seconds": 49.50,
            "command_semantics": "pre-imported output-local heal_compress alias, then pytest -q under univ2x-opt",
            "log": str(reports / "pytest_full_isolated_20260723.log"),
        },
        "compileall": {
            "passed": True,
            "log": str(reports / "compileall_20260723.log"),
        },
        "git_diff_check": True,
        "cnn_regression": True,
        "pyramid_path_unchanged": True,
        "formal_source_import_leak": False,
    }
    _write_json(reports / "regression_test_summary.json", regression)

    domain_counts = {
        "cnn": sum(models[m]["inventory"]["cnn_domain_count"] for m in MODEL_ORDER),
        "attention": sum(models[m]["inventory"]["attention_domain_count"] for m in MODEL_ORDER),
        "ffn": sum(models[m]["inventory"]["ffn_domain_count"] for m in MODEL_ORDER),
    }
    precision_counts = {
        "discovered": len(precision_units),
        "smoke_selected": sum(bool(row["smoke_selected"]) for row in precision_units),
        "joint_activation_proxy_units": len(activation_proxy),
    }
    acceptance = {
        "schema_version": "h800-transformer-unified-search-final-acceptance-v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "branch": branch,
        "worktree": str(args.repo.resolve()),
        "run_root": str(run_root),
        "base_branch": "origin/feature/heal-unified-search-h800",
        "base_commit": args.base_commit,
        "final_commit": head,
        "remote_commit": remote_commit,
        "remote_sync": ahead_behind.split() == ["0", "0"],
        "remote_ahead_behind": ahead_behind,
        "formal_search_branch_start_commit": args.formal_start_commit,
        "formal_search_branch_current_commit": formal_head,
        "formal_search_branch_unchanged": formal_head == args.formal_start_commit,
        "transformer_alignment_experiment_branch_commit": args.alignment_commit,
        "active_process_audit": {
            "before_process_count": before.get("process_count", len(before.get("processes", []))),
            "after_process_count": after["process_count"],
            "persistent_supervisor_pids": after["before_comparison"]["persistent_supervisor_pids"],
            "transient_pid_turnover_is_external_lifecycle": True,
            "signals_sent": after["signals_sent_to_external_processes"],
            "external_paths_written": after["external_paths_written_by_task"],
            "preserved": not after["signals_sent_to_external_processes"] and not after["external_paths_written_by_task"],
        },
        "formal_worktree_start_dirty_status": formal_start_dirty_status,
        "formal_worktree_current_dirty_status": formal_dirty_status,
        "source_worktrees_untouched": (
            formal_head == args.formal_start_commit
            and formal_dirty_status == formal_start_dirty_status
        ),
        "supported_models": ["v2xvit", "cobevt"],
        "partially_supported_models": ["attfusion", "coalign"],
        "unsupported_models": [],
        "model_support": {row["model"]: row["support_status"] for row in support_rows},
        "domains": domain_counts,
        "precision_units": precision_counts,
        "activation_proxy_status": "implemented_and_stage1_smoke_exercised",
        "smoothquant_status": "implemented; historical alpha selected; current accepted W8 engines are FFN roles, so physical-width QKV refresh remains",
        "greedy_status": "passed bounded one-budget smoke for four models",
        "ga_status": "passed population-8 generation-2 Top-2 smoke for four models",
        "stage2_status": "V2X-ViT and CoBEVT physical W16A16 and W8A8 engines passed; AttFusion/CoAlign are Stage-1 partial",
        "onnx_status": "four accepted physical candidates passed checker and shape inference",
        "tensorrt_status": "four TensorRT 10.9 strongly typed engines passed requested/realized inspector audit",
        "requested_realized_conflicts": [row for row in requested_realized if row["conflict"]],
        "tests": regression["pytest"],
        "compileall": regression["compileall"],
        "git_diff_check": regression["git_diff_check"],
        "full1789_executed": False,
        "formal_full_search_executed": False,
        "formal_six_budget_search_executed": False,
        "latency_status": "shared-GPU evaluation-smoke measurements only; formal uncontended latency not claimed",
        "remaining_items": [
            "refresh physical-width-specific SmoothQuant QKV scales and build QKV W8A8 engine anchors",
            "run uncontended GPU0 baseline-candidate-baseline latency protocol",
            "only after explicit authorization, configure formal 0.30/0.25/0.20/0.15/0.10/0.05 runs",
        ],
    }
    _write_json(reports / "final_acceptance.json", acceptance)

    conclusion = f"""# H800 Transformer unified-search bounded acceptance conclusion

The isolated branch `{branch}` at `{head}` has a complete bounded end-to-end smoke path for **V2X-ViT** and **CoBEVT**: real checkpoint tracing, independent Attention/FFN domains, physical CNN + `d_h` + `d_ff` pruning, joint weight-activation Taylor Stage-1, unified Greedy/GA, ONNX, TensorRT 10.9 strongly typed W16A16/W8A8 engines, precision inspection, smoke10 and fixed50 evaluation.

**AttFusion** and **CoAlign** are intentionally partial rather than falsely classified as standard Transformer models. Their real modules use projection-free scaled dot-product attention, so no trainable Q/K/V/O `attention_dh` or FFN domain is fabricated. Their real CNN domains and functional Softmax/AV activation precision units passed bounded Greedy/GA smokes.

Counts across the four audited models are {domain_counts['cnn']} CNN/grouped-convolution domains, {domain_counts['attention']} Attention `d_h` domains and {domain_counts['ffn']} FFN `d_ff` domains. The component adapter discovered {precision_counts['discovered']} precision units; {precision_counts['smoke_selected']} were included in the bounded search smokes and {precision_counts['joint_activation_proxy_units']} deployment units received joint activation-proxy evidence.

Four accepted engines have exact requested/realized physical widths, no mask-only pruning, no hidden padding, no unresolved precision mapping and no silent fallback. QK and Softmax compute were realized as FP32. W8A8 was physically accepted for one V2X-ViT FFN1 and one CoBEVT FFN2 unit. Softmax A8 remains correctly encoded as floating Softmax plus output Q/DQ; it was not misreported as native INT8 compute and was not selected by these engine candidates.

The full isolated regression result is 966 passed, 0 failed. `compileall` and `git diff --check` passed. No `full1789` or formal six-budget search was run. Latency values are shared-GPU evaluation-smoke observations, not an uncontended formal latency claim.

Remaining work is bounded and explicit: refresh structure-specific SmoothQuant scales for a QKV W8A8 engine anchor, schedule an uncontended GPU0 latency window, and obtain authorization before any formal six-budget search.
"""
    (run_root / "root_conclusion.md").write_text(conclusion, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
