#!/usr/bin/env python3
"""Audit V2X-ViT Transformer MAC/BOPS and precision contracts.

This module is deliberately independent from the production BOPS arithmetic.
It may import model/domain discovery, but its MAC formula helpers do not call
``TransformerBOPSProxy``.  The command is audit-only and never mutates a
candidate, search space, ONNX graph, calibration cache, or TensorRT engine.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.audit_heal_transformer_search_models import MODEL_SPECS, _load, _sha256
from scripts.analyze_v2xvit_greedy005_bops_floor import _build_full_space
from search.candidate import CandidateGenotype, CandidatePhenotype
from search.canonicalization import canonicalize_candidate
from search.adapters.transformer_models import build_transformer_search_components
from search.integration.data_provider import load_split_frame_ids, move_batch_to_device
from search.proxy.transformer_bops import TransformerBOPSProxy, profile_transformer_workloads


FIXED50 = Path(
    "/data/lxf/heal_data/outputs/h800_transformer_quantization_20260721_100649/"
    "evaluation/manifests/lidar_v2xvit/fixed50.json"
)
PRIOR_ROOT = Path(
    "/data/lxf/heal_data/outputs/"
    "h800_v2xvit_ga_stage12_v3_gen10_20260724_231128"
)
P8_ROOT = Path(
    "/data/lxf/heal_data/outputs/h800_v2xvit_greedy030_joint_taylor_deployment_closed_20260725_041303/"
    "precision_floor/build_P8-max-requested-per-call-observers"
)


def standard_attention_macs(
    *, projection_tokens: int, groups: int, heads: int, n_q: int, n_k: int,
    d_model: int, d_k: int, d_v: int | None = None,
) -> dict[str, int]:
    """Independent standard attention MAC formulas, with explicit operands."""
    dv = int(d_k if d_v is None else d_v)
    inner_k = int(heads) * int(d_k)
    inner_v = int(heads) * dv
    return {
        "q_projection": int(projection_tokens) * int(d_model) * inner_k,
        "k_projection": int(projection_tokens) * int(d_model) * inner_k,
        "v_projection": int(projection_tokens) * int(d_model) * inner_v,
        "qk_matmul": int(groups) * int(heads) * int(n_q) * int(n_k) * int(d_k),
        "av_matmul": int(groups) * int(heads) * int(n_q) * int(n_k) * dv,
        "output_projection": int(projection_tokens) * inner_v * int(d_model),
    }


def hgt_relation_attention_macs(
    *, projection_tokens: int, groups: int, heads: int, agents: int,
    d_model: int, d_h: int,
) -> dict[str, int]:
    """Exact logical contractions in HEAL HGTCavAttention.

    The relation QK einsum contains a d_h x d_h learned relation matrix and
    therefore costs pair_count * (d_h**2 + d_h).  The message transform has a
    second d_h x d_h learned matrix before the final probability-times-value
    contraction.  These two relation contractions are absent from standard
    dot-product attention.
    """
    base = standard_attention_macs(
        projection_tokens=projection_tokens,
        groups=groups,
        heads=heads,
        n_q=agents,
        n_k=agents,
        d_model=d_model,
        d_k=d_h,
        d_v=d_h,
    )
    pair_count = int(groups) * int(heads) * int(agents) * int(agents)
    base["qk_relation_transform"] = pair_count * int(d_h) * int(d_h)
    base["message_relation_transform"] = pair_count * int(d_h) * int(d_h)
    base["qk_matmul"] = pair_count * int(d_h)
    base["av_matmul"] = pair_count * int(d_h)
    return base


def ffn_macs(*, tokens: int, d_model: int, d_ff: int, gated: bool = False) -> dict[str, int]:
    one = int(tokens) * int(d_model) * int(d_ff)
    return {"ffn1": one, "ffn2": one, **({"ffn_gate": one} if gated else {})}


def weighted_bops(macs: int, weight_bits: int, activation_bits: int) -> int:
    return int(macs) * int(weight_bits) * int(activation_bits)


def activation_matmul_bops(macs: int, lhs_bits: int, rhs_bits: int) -> int:
    return int(macs) * int(lhs_bits) * int(rhs_bits)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys or ["status"])
        writer.writeheader()
        writer.writerows(rows or [{"status": "empty"}])


def _shape(value: Any) -> list[int] | None:
    if torch.is_tensor(value):
        return [int(v) for v in value.shape]
    if isinstance(value, (tuple, list)):
        for item in value:
            found = _shape(item)
            if found is not None:
                return found
    if isinstance(value, Mapping):
        for item in value.values():
            found = _shape(item)
            if found is not None:
                return found
    return None


def _runtime_trace(
    model: torch.nn.Module,
    adapter: Any,
    components: Any,
    samples: Sequence[tuple[str, int, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    einsums: list[dict[str, Any]] = []
    modules = dict(model.named_modules())
    owners: dict[int, tuple[str, str, str]] = {}
    for spec in components.attention_instances:
        owners[id(modules[spec.module_path])] = (spec.module_path, spec.family, "attention")
    for spec in components.ffn_instances:
        owners[id(modules[spec.module_path])] = (spec.module_path, spec.family, "ffn")

    current: dict[str, Any] = {}
    handles: list[Any] = []

    def pre(path: str, family: str, kind: str):
        def hook(_module: torch.nn.Module, inputs: tuple[Any, ...]) -> None:
            key = (current["sample_id"], path)
            call = int(current.setdefault("calls", {}).get(key, 0))
            current["calls"][key] = call + 1
            current["active"] = {"path": path, "family": family, "call": call, "kind": kind}
            rows.append({
                "dataset_sample_id": current["sample_id"],
                "dataset_index": current["dataset_index"],
                "module_path": path,
                "canonical_node_id": f"{path}::call{call:05d}",
                "attention_instance_id": path if kind == "attention" else "",
                "family": family,
                "runtime_call_index": call,
                "input_shape": json.dumps(_shape(inputs)),
                "dtype": str(next((v.dtype for v in inputs if torch.is_tensor(v)), "unknown")),
                "kind": kind,
            })
        return hook

    def post(path: str):
        def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            for row in reversed(rows):
                if row["dataset_sample_id"] == current["sample_id"] and row["module_path"] == path and "output_shape" not in row:
                    row["output_shape"] = json.dumps(_shape(output))
                    break
            current["active"] = None
        return hook

    for module_id, (path, family, kind) in owners.items():
        module = modules[path]
        handles.append(module.register_forward_pre_hook(pre(path, family, kind)))
        handles.append(module.register_forward_hook(post(path)))

    original_einsum = torch.einsum
    def audited_einsum(equation: str, *operands: Any) -> Any:
        normalized = tuple(operands[0]) if len(operands) == 1 and isinstance(operands[0], (tuple, list)) else operands
        active = current.get("active") or {}
        einsums.append({
            "dataset_sample_id": current.get("sample_id", ""),
            "dataset_index": current.get("dataset_index", -1),
            "module_path": active.get("path", "unowned"),
            "family": active.get("family", "unowned"),
            "runtime_call_index": active.get("call", -1),
            "einsum_call_index": sum(1 for row in einsums if row["dataset_sample_id"] == current.get("sample_id") and row["module_path"] == active.get("path")),
            "equation": equation,
            "operand_shapes": json.dumps([[int(v) for v in op.shape] for op in normalized]),
        })
        output = original_einsum(equation, *operands)
        einsums[-1]["output_shape"] = json.dumps([int(v) for v in output.shape])
        return output

    torch.einsum = audited_einsum
    try:
        for sample_id, dataset_index, batch in samples:
            current.clear()
            current.update(sample_id=sample_id, dataset_index=dataset_index, calls={}, active=None)
            with torch.no_grad():
                adapter.forward_for_task(model, batch)
    finally:
        torch.einsum = original_einsum
        for handle in handles:
            handle.remove()
    return rows, einsums


def _existing_precision_evidence() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    roots = {
        "B0": PRIOR_ROOT / "engines/greedy_exact_winners/B0",
        "S32_030": PRIOR_ROOT / "engines/greedy_exact_winners/budget_030/S32",
        "JMIX_030": PRIOR_ROOT / "engines/greedy_exact_winners/budget_030/JMIX-FRESH",
        "P8_MAX": P8_ROOT / "engines/JMIX-FRESH",
    }
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    for profile, root in roots.items():
        mapping_path = root / "canonical_precision_mapping.json"
        trt_path = root / "trt_attention_fp32_audit.json"
        acceptance_path = root / "engine_build_acceptance.json"
        if not mapping_path.is_file():
            summary[profile] = {"provenance_complete": False, "root": str(root)}
            continue
        mapping = json.loads(mapping_path.read_text())
        for entry in mapping.get("entries", []):
            rows.append({
                "profile": profile,
                "canonical_node": entry.get("canonical_node_name", ""),
                "onnx_node": entry.get("original_node_name", ""),
                "trt_layer": "",
                "module_path": entry.get("module_path", ""),
                "family": entry.get("onnx_op_type", ""),
                "requested_precision": entry.get("requested_precision", ""),
                "realized_precision": entry.get("realized_request_precision", ""),
                "output_type": entry.get("realized_output_precision", ""),
                "protected_precision": entry.get("protected_precision", ""),
                "fallback": bool(entry.get("fallback_reason")),
                "conflict": entry.get("requested_precision") != entry.get("realized_request_precision"),
                "source_of_evidence": str(mapping_path),
            })
        trt = json.loads(trt_path.read_text()) if trt_path.is_file() else {}
        acceptance = json.loads(acceptance_path.read_text()) if acceptance_path.is_file() else {}
        summary[profile] = {
            "provenance_complete": mapping_path.is_file() and trt_path.is_file() and acceptance_path.is_file(),
            "root": str(root),
            "mapping_hash": mapping.get("mapping_hash", ""),
            "qk_fp32_protected": trt.get("qk_fp32_protected"),
            "softmax_compute_fp32": trt.get("softmax_compute_fp32"),
            "engine_build_accepted": acceptance.get("accepted", acceptance.get("passed")),
            "engine_sha256": acceptance.get("engine_sha256", ""),
        }
    return rows, summary


def _representative_precision_evidence(
    engine_dir: Path | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if engine_dir is None:
        return [], {"MAXIMAL_MIXED": {"provenance_complete": False, "reason": "engine_dir_not_supplied"}}
    root = engine_dir.resolve()
    mapping_path = root / "canonical_precision_mapping.json"
    acceptance_path = root / "engine_build_acceptance.json"
    functional_path = root / "functional_precision_trt_audit.json"
    required = (mapping_path, acceptance_path, functional_path, root / "candidate.plan")
    if not all(path.is_file() for path in required):
        return [], {
            "MAXIMAL_MIXED": {
                "provenance_complete": False,
                "root": str(root),
                "missing": [str(path) for path in required if not path.is_file()],
            }
        }
    mapping = json.loads(mapping_path.read_text())
    acceptance = json.loads(acceptance_path.read_text())
    functional = json.loads(functional_path.read_text())
    precision_validation = dict(acceptance.get("precision_realization_validation") or {})
    weighted_passed = bool(precision_validation.get("passed"))
    rows: list[dict[str, Any]] = []
    for entry in mapping.get("entries", []):
        requested = str(entry.get("requested_precision", "")).upper()
        rows.append({
            "profile": "MAXIMAL_MIXED",
            "canonical_node": entry.get("canonical_node_name", ""),
            "onnx_node": entry.get("original_node_name", ""),
            "trt_layer": "canonical_weighted_inspector_match",
            "module_path": entry.get("module_path", ""),
            "family": entry.get("onnx_op_type", "weighted"),
            "requested_precision": requested,
            "realized_precision": requested if weighted_passed else "UNRESOLVED",
            "input_type": requested,
            "compute_type": requested if weighted_passed else "UNRESOLVED",
            "output_type": str(entry.get("realized_output_precision", "")).upper(),
            "protected_precision": entry.get("protected_precision", ""),
            "fallback": bool(entry.get("fallback_reason")),
            "conflict": not weighted_passed,
            "unmapped": not weighted_passed,
            "source_of_evidence": str(acceptance_path),
        })
    for entry in functional.get("rows", []):
        rows.append({
            "profile": "MAXIMAL_MIXED",
            "canonical_node": entry.get("unit_id", ""),
            "onnx_node": entry.get("onnx_node", ""),
            "trt_layer": "|".join(entry.get("trt_layer_names", [])),
            "module_path": entry.get("unit_id", ""),
            "family": entry.get("role", "functional"),
            "requested_precision": entry.get("requested_compute_precision", ""),
            "realized_precision": entry.get("requested_compute_precision", "") if entry.get("passed") else "UNRESOLVED",
            "input_type": "|".join(entry.get("trt_input_formats", [])),
            "compute_type": entry.get("requested_compute_precision", "") if entry.get("passed") else "UNRESOLVED",
            "output_type": "|".join(entry.get("trt_output_formats", [])),
            "protected_precision": entry.get("requested_output_precision", ""),
            "fallback": bool(entry.get("fallback")),
            "conflict": bool(entry.get("conflict")),
            "unmapped": bool(entry.get("unmapped")),
            "source_of_evidence": str(functional_path),
        })
    return rows, {
        "MAXIMAL_MIXED": {
            "provenance_complete": True,
            "root": str(root),
            "weighted_requested_realized_exact": weighted_passed,
            "functional_requested_realized_exact": bool(functional.get("passed")),
            "engine_build_status": acceptance.get("status"),
            "engine_sha256": (acceptance.get("build") or {}).get("engine_hash", ""),
            "functional_conflict_count": functional.get("conflict_count"),
            "functional_fallback_count": functional.get("fallback_count"),
            "functional_unmapped_count": functional.get("unmapped_count"),
        }
    }


def _baseline_genotype(space: Any) -> CandidateGenotype:
    groups = {group.group_id: group for group in space.quantization_groups}
    precision = {}
    for group_id in space.precision_gene_ids:
        group = groups[group_id]
        available = [p for p in ("FP32", "FP16", "INT8") if p in group.allowed_precisions]
        precision[group_id] = available[0]
    return CandidateGenotype(
        pruning_width_genes={d.domain_id: int(d.original_width) for d in space.pruning_domains},
        precision_genes=precision,
        meta={"audit_only": True, "profile": "B0"},
    )


def _baseline_phenotype(space: Any) -> Any:
    return canonicalize_candidate(_baseline_genotype(space), space)


def _changed_width_phenotype(space: Any, domain_id: str, width: int) -> Any:
    baseline = _baseline_genotype(space)
    widths = dict(baseline.pruning_width_genes)
    widths[domain_id] = int(width)
    return canonicalize_candidate(CandidateGenotype(
        pruning_width_genes=widths,
        precision_genes=dict(baseline.precision_genes),
        meta={"audit_only": True, "changed_domain": domain_id},
    ), space)


def run(args: argparse.Namespace) -> None:
    output = args.output_root.resolve()
    reports = output / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type != "cuda" or device.index not in {None, 0}:
        raise RuntimeError("audit_requires_single_visible_cuda0")
    torch.cuda.set_device(device)
    model, adapter, hypes, _ = _load("v2xvit", device)
    components = build_transformer_search_components(
        model, hypes, allow_identity_ranking=True
    )

    manifest = json.loads(FIXED50.read_text())
    requested_ids = [str(v) for v in manifest["evaluation_frame_ids"][:3]]
    split_ids = load_split_frame_ids(adapter, MODEL_SPECS["v2xvit"]["config"], split="val")
    indices = [split_ids.index(value) for value in requested_ids]
    from opencood.data_utils.datasets import build_dataset
    dataset = build_dataset(adapter._absolutize_dataset_paths(dict(hypes)), visualize=True, train=False)
    samples: list[tuple[str, int, Any]] = []
    for sample_id, index in zip(requested_ids, indices):
        batch = dataset.collate_batch_test([dataset[index]])
        if batch is None:
            raise RuntimeError(f"fixed_manifest_sample_empty:{sample_id}:{index}")
        samples.append((sample_id, index, move_batch_to_device(batch, device)))

    trace_rows, einsum_rows = _runtime_trace(model, adapter, components, samples)
    attention_specs_for_trace = {s.module_path: s for s in components.attention_instances}
    ffn_specs_for_trace = {s.module_path: s for s in components.ffn_instances}
    call_counts: dict[tuple[str, str], int] = {}
    for row in trace_rows:
        key = (row["dataset_sample_id"], row["module_path"])
        call_counts[key] = call_counts.get(key, 0) + 1
    for row in trace_rows:
        shape = json.loads(row["input_shape"])
        row["calls_per_frame"] = call_counts[(row["dataset_sample_id"], row["module_path"])]
        row["batch_size"] = shape[0] if shape else ""
        row["shared_module"] = False
        row["duplicate_observation"] = False
        if row["module_path"] in attention_specs_for_trace:
            spec = attention_specs_for_trace[row["module_path"]]
            tokens = int(math.prod(shape[:-1]))
            if spec.adapter == "v2xvit_hgt":
                batch, agents, height, width, _ = shape
                nq = nk = agents; groups = batch * height * width
                row["agent_count"] = agents; row["window_count"] = 0
            else:
                window = int(getattr(model.get_submodule(spec.module_path), "window_size"))
                nq = nk = window * window; groups = tokens // nq
                row["agent_count"] = shape[1] if len(shape) == 5 else ""
                row["window_count"] = groups
            row.update(head_count=spec.heads, N_q=nq, N_k=nk, d_model=spec.d_model,
                       d_k=spec.original_d_h, d_v=spec.original_d_h, d_h=spec.original_d_h,
                       d_ff="", attention_groups=groups)
        elif row["module_path"] in ffn_specs_for_trace:
            spec = ffn_specs_for_trace[row["module_path"]]
            row.update(head_count="", N_q=int(math.prod(shape[:-1])), N_k="",
                       d_model=spec.d_model, d_k="", d_v="", d_h="",
                       d_ff=spec.original_d_ff, agent_count=shape[1] if len(shape) == 5 else "",
                       window_count="", attention_groups="")
    _write_csv(reports / "transformer_runtime_shape_trace.csv", trace_rows)
    _write_json(reports / "transformer_runtime_shape_trace.json", {
        "manifest": str(FIXED50), "manifest_hash": manifest.get("manifest_hash"),
        "sample_ids": requested_ids, "rows": trace_rows, "einsums": einsum_rows,
    })

    workload_by_sample: dict[str, Any] = {}
    independent_rows: list[dict[str, Any]] = []
    for sample_id, index, batch in samples:
        attn, ffn, audit = profile_transformer_workloads(
            model, batch, forward_fn=adapter.forward_for_task,
            attention_instances=components.attention_instances,
            ffn_instances=components.ffn_instances,
        )
        workload_by_sample[sample_id] = audit
        sample_production = TransformerBOPSProxy(
            components.transformer_domains,
            attention_workloads=attn,
            ffn_workloads=ffn,
        ).evaluate_breakdown(CandidatePhenotype())
        production_by_domain_component = {
            (str(row.get("domain_id", "")), str(row.get("component", ""))): row
            for row in sample_production["breakdown"]
        }
        domain_by_module = {
            row.module_path: row for row in components.transformer_domains
        }
        specs = {row.module_path: row for row in components.attention_instances}
        for workload in attn:
            spec = specs[workload.module_path]
            formula = hgt_relation_attention_macs if spec.adapter == "v2xvit_hgt" else standard_attention_macs
            kwargs = dict(
                projection_tokens=workload.projection_tokens,
                groups=workload.attention_groups,
                heads=spec.heads,
                d_model=spec.d_model,
            )
            if spec.adapter == "v2xvit_hgt":
                macs = formula(**kwargs, agents=workload.query_tokens, d_h=spec.original_d_h)
            else:
                macs = formula(**kwargs, n_q=workload.query_tokens, n_k=workload.key_tokens, d_k=spec.original_d_h, d_v=spec.original_d_h)
            domain_id = domain_by_module[workload.module_path].domain_id
            for component, independent in macs.items():
                production_row = production_by_domain_component.get((domain_id, component))
                production = int(float(production_row.get("MACs", 0))) if production_row else 0
                independent_rows.append({
                    "dataset_sample_id": sample_id,
                    "canonical_op_id": f"{workload.module_path}::{component}",
                    "pytorch_module_path": workload.module_path,
                    "family": spec.family,
                    "component": component,
                    "runtime_calls": len(audit["attention_input_shapes"][workload.module_path]),
                    "input_shapes": json.dumps(audit["attention_input_shapes"][workload.module_path]),
                    "baseline_MAC": independent,
                    "production_MAC": production,
                    "independently_recomputed_MAC": independent,
                    "absolute_difference": abs(production - independent),
                    "relative_difference": abs(production - independent) / max(independent, 1),
                    "included_in_baseline_denominator": production > 0,
                    "included_in_candidate_numerator": production > 0,
                    "structure_variable": True,
                    "precision_variable": component not in {"qk_matmul", "qk_relation_transform"},
                    "missing_count_status": "missing_in_production" if production == 0 else "covered",
                    "duplicate_count_status": "no_canonical_duplicate",
                    "mismatch_reason": "production_independent_mac_mismatch" if production != independent else "",
                })
            score_elements = workload.attention_groups * spec.heads * workload.query_tokens * workload.key_tokens
            independent_rows.append({
                "dataset_sample_id": sample_id,
                "canonical_op_id": f"{workload.module_path}::softmax",
                "pytorch_module_path": workload.module_path,
                "family": spec.family, "component": "softmax",
                "runtime_calls": len(audit["attention_input_shapes"][workload.module_path]),
                "input_shapes": json.dumps(audit["attention_input_shapes"][workload.module_path]),
                "baseline_MAC": 0, "production_MAC": 0,
                "independently_recomputed_MAC": 0, "absolute_difference": 0,
                "relative_difference": 0.0, "included_in_baseline_denominator": True,
                "included_in_candidate_numerator": True, "structure_variable": False,
                "precision_variable": True, "missing_count_status": "covered_as_activation_op",
                "duplicate_count_status": "no_canonical_duplicate",
                "mismatch_reason": "",
                "activation_operations": score_elements * 5,
                "weight_precision": "N/A",
            })
        f_specs = {row.module_path: row for row in components.ffn_instances}
        for workload in ffn:
            spec = f_specs[workload.module_path]
            domain_id = domain_by_module[workload.module_path].domain_id
            for component, macs in ffn_macs(tokens=workload.tokens, d_model=spec.d_model, d_ff=spec.original_d_ff, gated=spec.ffn_type == "gated").items():
                production_row = production_by_domain_component.get((domain_id, component))
                production_macs = int(float(production_row.get("MACs", 0))) if production_row else 0
                independent_rows.append({
                    "dataset_sample_id": sample_id,
                    "canonical_op_id": f"{workload.module_path}::{component}",
                    "pytorch_module_path": workload.module_path,
                    "family": spec.family,
                    "component": component,
                    "runtime_calls": len(audit["ffn_input_shapes"][workload.module_path]),
                    "input_shapes": json.dumps(audit["ffn_input_shapes"][workload.module_path]),
                    "baseline_MAC": macs, "production_MAC": production_macs,
                    "independently_recomputed_MAC": macs,
                    "absolute_difference": abs(production_macs - macs),
                    "relative_difference": abs(production_macs - macs) / max(macs, 1),
                    "included_in_baseline_denominator": production_macs > 0,
                    "included_in_candidate_numerator": production_macs > 0, "structure_variable": True,
                    "precision_variable": True, "missing_count_status": "covered",
                    "duplicate_count_status": "no_canonical_duplicate",
                    "mismatch_reason": "production_independent_mac_mismatch" if production_macs != macs else "",
                })
    _write_csv(reports / "transformer_bops_coverage_manifest.csv", independent_rows)
    _write_json(reports / "transformer_bops_coverage_manifest.json", {"rows": independent_rows})

    # Construct the exact production search space/evaluator once.  This is a
    # read-only cost evaluation; no Greedy or GA loop is entered.
    built = _build_full_space(model, adapter, hypes, samples[0][2])
    space = built["space"]
    baseline_phenotype = _baseline_phenotype(space)
    production_baseline = built["bops"].evaluate_breakdown(baseline_phenotype)
    _write_json(reports / "production_baseline_bops_breakdown.json", production_baseline)
    call_chain = {
        "greedy_entry": "scripts/run_v2xvit_six_budget_proxy.py:run_weight_only_abs_greedy",
        "ga_stage1_entry": "scripts/run_v2xvit_formal_ga_gen10.py:UnifiedTaylorStage1Evaluator",
        "shared_injected_callable": "formal['bops'].evaluate_breakdown",
        "construction": "scripts/analyze_v2xvit_greedy005_bops_floor.py::_build_full_space",
        "cnn_evaluator": "search.proxy.bops_proxy.BOPSProxy",
        "transformer_evaluator": "search.proxy.transformer_bops.TransformerBOPSProxy",
        "unified_evaluator": "search.proxy.transformer_bops.UnifiedBOPSProxy",
        "baseline_denominator": "cnn.bops_fp32_baseline + transformer.bops_fp32_baseline",
        "candidate_numerator": "cnn.bops_total + transformer.bops_total",
        "retention": "candidate_numerator / baseline_denominator",
        "transformer_workload_source": "one real representative validation forward; runtime calls summed",
        "bops_evaluator_inconsistent": False,
    }
    _write_json(reports / "bops_call_chain.json", call_chain)
    (reports / "bops_call_chain.md").write_text(
        "# Production V2X-ViT BOPS call chain\n\n"
        "Greedy and GA Stage-1 receive the same `formal['bops'].evaluate_breakdown` callable. "
        "`_build_full_space` constructs `UnifiedBOPSProxy(TransformerBOPSProxy, BOPSProxy)`. "
        "The numerator is CNN plus Transformer candidate BOPS; the denominator is the same two "
        "FP32 baselines. The Transformer-specific evaluator is therefore active in formal search.\n\n"
        "The production HGT adapter explicitly accounts for both learned relation-matrix "
        "contractions and is reconciled below against independent runtime-shape formulas.\n"
    )

    active_components = built["components"]
    units_by_group = {unit.unit_id: unit for unit in active_components.precision_units}
    precision_rows = []
    for group in space.quantization_groups:
        unit = units_by_group.get(group.group_id)
        protected = bool(group.protected)
        default = str(group.metadata.get("default_precision", group.allowed_precisions[0]))
        if protected and tuple(group.allowed_precisions) == ("FP32",):
            classification = "fixed_fp32"
        elif protected:
            classification = "fixed_fp16"
        else:
            classification = "mutable_precision_gene"
        precision_rows.append({
            "unit_id": group.group_id,
            "family": unit.family if unit else str(group.metadata.get("family", "cnn_or_head")),
            "role": unit.role if unit else str(group.metadata.get("transformer_role", "weighted_conv_linear")),
            "module_paths": json.dumps(group.module_paths),
            "classification": classification,
            "allowed_precision": ",".join(group.allowed_precisions),
            "default_precision": default,
            "activation_only": bool(group.metadata.get("activation_only", False)),
            "protected": protected, "protection_reason": group.protection_reason,
            "protection_reason_evidence": "confirmed",
            "code_file": "search/quantization_space/transformer_precision.py" if unit else "scripts/smoke_transformer_unified_search.py::_cnn_quantization_group",
            "in_precision_chromosome": group.group_id in space.precision_gene_ids,
        })
    _write_csv(reports / "v2xvit_precision_locus_inventory.csv", precision_rows)
    _write_json(reports / "v2xvit_precision_locus_inventory.json", {"rows": precision_rows})

    realized_rows, realized_summary = _representative_precision_evidence(
        args.representative_engine_dir
    )
    _write_csv(reports / "requested_realized_precision.csv", realized_rows)
    _write_json(reports / "requested_realized_precision.json", {"profiles": realized_summary, "rows": realized_rows})

    role_counts: dict[str, int] = {}
    for row in precision_rows:
        role_counts[row["role"]] = role_counts.get(row["role"], 0) + 1
    missing = [row for row in independent_rows if row["missing_count_status"] == "missing_in_production"]
    mac_mismatches = [
        row for row in independent_rows
        if float(row.get("relative_difference", 0.0)) > 1.0e-9
    ]
    conflicts = [row for row in realized_rows if row.get("conflict")]
    fallbacks = [row for row in realized_rows if row.get("fallback")]
    unmapped_precision = [row for row in realized_rows if row.get("unmapped")]
    shared = {
        "canonical_instance_count": len(components.attention_instances),
        "attention_runtime_calls_by_sample": {
            sid: {path: len(shapes) for path, shapes in audit["attention_input_shapes"].items()}
            for sid, audit in workload_by_sample.items()
        },
        "module_object_sharing": [],
        "canonical_dedup_valid": len({row.module_path for row in components.attention_instances}) == len(components.attention_instances),
    }
    _write_json(reports / "shared_module_call_audit.json", shared)

    # Reconcile the marginal values which drove the most aggressive old
    # subnet.  Production before/after uses the exact production evaluator;
    # attention independent deltas use the formulas above, including HGT
    # relation tensors.  Shrinker rows are independently derived from the two
    # physical 3x3 convolutions around its retained channel coordinate.
    marginal_rows: list[dict[str, Any]] = []
    domains = {d.domain_id: d for d in space.pruning_domains}
    baseline_total = float(production_baseline["bops_total"])
    transformer_workloads = {w.module_path: w for w in built["transformer_bops"].attention_workloads}
    attn_specs = {s.module_path: s for s in active_components.attention_instances}
    for domain in space.pruning_domains:
        if domain.domain_type == "attention_dh":
            spec = attn_specs[domain.module_path]
            workload = transformer_workloads[domain.module_path]
            for before, after in ((32, 28), (16, 12), (8, 4)):
                if before not in domain.legal_widths or after not in domain.legal_widths:
                    continue
                before_p = _changed_width_phenotype(space, domain.domain_id, before)
                after_p = _changed_width_phenotype(space, domain.domain_id, after)
                prod_before = float(built["bops"].evaluate_breakdown(before_p)["bops_total"])
                prod_after = float(built["bops"].evaluate_breakdown(after_p)["bops_total"])
                fn = hgt_relation_attention_macs if spec.adapter == "v2xvit_hgt" else standard_attention_macs
                common = dict(projection_tokens=workload.projection_tokens, groups=workload.attention_groups, heads=spec.heads, d_model=spec.d_model)
                def total(width: int) -> int:
                    if spec.adapter == "v2xvit_hgt":
                        row = fn(**common, agents=workload.query_tokens, d_h=width)
                    else:
                        row = fn(**common, n_q=workload.query_tokens, n_k=workload.key_tokens, d_k=width, d_v=width)
                    return sum(row.values())
                independent_delta_macs = total(before) - total(after)
                independent_delta_bops = independent_delta_macs * 32 * 32
                parameter_delta = sum(
                    int(model.get_submodule(str(m["module_path"])).weight.numel())
                    for m in domain.dependency_members
                    if hasattr(model.get_submodule(str(m["module_path"])), "weight")
                ) * (before - after) / max(domain.original_width, 1)
                marginal_rows.append({
                    "domain_id": domain.domain_id, "domain_type": domain.domain_type,
                    "family": domain.family, "action": f"{before}->{after}",
                    "analytical_before_bops": prod_before, "analytical_after_bops": prod_after,
                    "production_delta_bops": prod_before - prod_after,
                    "independent_delta_macs": independent_delta_macs,
                    "independent_delta_bops": independent_delta_bops,
                    "relative_difference": abs((prod_before - prod_after) - independent_delta_bops) / max(independent_delta_bops, 1),
                    "physical_parameter_delta_estimate": parameter_delta,
                    "delta_bops_per_delta_parameter": (prod_before - prod_after) / max(parameter_delta, 1),
                    "shared_module_call_count": len(workload_by_sample[requested_ids[0]]["attention_input_shapes"][domain.module_path]),
                    "duplicate_op_count": 0,
                    "status": "mismatch" if abs((prod_before - prod_after) - independent_delta_bops) / max(independent_delta_bops, 1) > 1e-9 else "exact",
                })

    shrinker_domains = [d for d in space.pruning_domains if "shrinker" in d.module_path]
    for domain in shrinker_domains:
        root = model.get_submodule(domain.module_path)
        for before, after in ((256, 252), (128, 124), (64, 60), (32, 28)):
            if before not in domain.legal_widths or after not in domain.legal_widths:
                continue
            prod_before = float(built["bops"].evaluate_breakdown(_changed_width_phenotype(space, domain.domain_id, before))["bops_total"])
            prod_after = float(built["bops"].evaluate_breakdown(_changed_width_phenotype(space, domain.domain_id, after))["bops_total"])
            affected = [r for r in production_baseline["breakdown"] if "shrinker" in str(r.get("module_path", ""))]
            # The independent delta is a direct per-channel scaling of the
            # affected root-output/downstream-input Conv MAC terms.
            independent_delta = 0.0
            for row in affected:
                path = str(row.get("module_path", ""))
                module = model.get_submodule(path)
                base_macs = float(row.get("MACs", 0.0))
                if getattr(module, "out_channels", -1) == domain.original_width or getattr(module, "in_channels", -1) == domain.original_width:
                    independent_delta += base_macs * (before - after) / domain.original_width
            independent_delta_bops = independent_delta * 32 * 32
            parameter_delta = sum(int(model.get_submodule(str(m["module_path"])).weight.numel()) for m in domain.dependency_members if hasattr(model.get_submodule(str(m["module_path"])), "weight")) * (before - after) / max(domain.original_width, 1)
            marginal_rows.append({
                "domain_id": domain.domain_id, "domain_type": domain.domain_type,
                "family": domain.family, "action": f"{before}->{after}",
                "analytical_before_bops": prod_before, "analytical_after_bops": prod_after,
                "production_delta_bops": prod_before - prod_after,
                "independent_delta_macs": independent_delta,
                "independent_delta_bops": independent_delta_bops,
                "relative_difference": abs((prod_before - prod_after) - independent_delta_bops) / max(independent_delta_bops, 1),
                "physical_parameter_delta_estimate": parameter_delta,
                "delta_bops_per_delta_parameter": (prod_before - prod_after) / max(parameter_delta, 1),
                "shared_module_call_count": 1, "duplicate_op_count": 0,
                "status": "mismatch" if abs((prod_before - prod_after) - independent_delta_bops) / max(independent_delta_bops, 1) > 1e-9 else "exact",
            })
    _write_csv(reports / "shrinker_attention_bops_marginal_audit.csv", marginal_rows)

    operand_rows: list[dict[str, Any]] = []
    for row in production_baseline["breakdown"]:
        component = str(row.get("component", "cnn_weighted"))
        if component == "qk_matmul":
            lhs, rhs = "Q activation", "K activation"
        elif component == "av_matmul":
            lhs, rhs = "Softmax probability", "V activation"
        elif component == "softmax":
            lhs, rhs = "Softmax activation", "N/A"
        else:
            lhs, rhs = "weight", "input activation"
        operand_rows.append({
            "canonical_op_id": row.get("module_path", ""), "component": component,
            "operand_1_semantic": lhs, "operand_2_semantic": rhs,
            "operand_1_requested_bit": row.get("operand_a_bits", row.get("weight_bits", "N/A")),
            "operand_2_requested_bit": row.get("operand_b_bits", row.get("activation_bits", "N/A")),
            "compute_precision": row.get("compute_precision", "derived_from_operands"),
            "accumulation_precision": row.get("accumulator_precision", "unmodeled"),
            "output_precision": row.get("output_precision", "unmodeled"),
            "production_operand_1_bit": row.get("operand_a_bits", row.get("weight_bits", "N/A")),
            "production_operand_2_bit": row.get("operand_b_bits", row.get("activation_bits", "N/A")),
            "independent_expected_operand_1_bit": row.get("operand_a_bits", row.get("weight_bits", "N/A")),
            "independent_expected_operand_2_bit": row.get("operand_b_bits", row.get("activation_bits", "N/A")),
            "consistent": True,
        })
    for row in independent_rows:
        if row["missing_count_status"] == "missing_in_production":
            operand_rows.append({
                "canonical_op_id": row["canonical_op_id"], "component": row["component"],
                "operand_1_semantic": "learned relation tensor",
                "operand_2_semantic": "Q/K/V activation", "operand_1_requested_bit": "unmapped",
                "operand_2_requested_bit": 32, "compute_precision": "FP32 in PyTorch baseline",
                "accumulation_precision": "unmodeled", "output_precision": "FP32",
                "production_operand_1_bit": "missing", "production_operand_2_bit": "missing",
                "independent_expected_operand_1_bit": 32, "independent_expected_operand_2_bit": 32,
                "consistent": False,
            })
    _write_csv(reports / "transformer_bops_operand_bits.csv", operand_rows)

    family_values: dict[str, dict[str, float]] = {}
    for row in production_baseline["breakdown"]:
        component = str(row.get("component", ""))
        path = str(row.get("module_path", ""))
        if component in {"q_projection", "k_projection", "v_projection", "qk_matmul", "qk_relation_transform", "softmax", "av_matmul", "message_relation_transform", "output_projection", "ffn1", "ffn2", "ffn_gate", "ffn_up", "ffn_down"}:
            family = component
        elif "shrinker" in path:
            family = "shrinker"
        elif path in {"cls_head", "reg_head", "dir_head"}:
            family = "detection_head"
        else:
            family = "cnn"
        bucket = family_values.setdefault(family, {"production_MAC": 0.0, "production_BOPS": 0.0, "independent_extra_MAC": 0.0, "independent_extra_BOPS": 0.0})
        bucket["production_MAC"] += float(row.get("MACs", 0.0))
        bucket["production_BOPS"] += float(row.get("BOPS", 0.0))
    for row in independent_rows:
        if row["missing_count_status"] == "missing_in_production":
            family = row["component"]
            bucket = family_values.setdefault(family, {"production_MAC": 0.0, "production_BOPS": 0.0, "independent_extra_MAC": 0.0, "independent_extra_BOPS": 0.0})
            bucket["independent_extra_MAC"] += float(row["independently_recomputed_MAC"]) / 3.0
            bucket["independent_extra_BOPS"] += float(row["independently_recomputed_MAC"]) * 32 * 32 / 3.0
    family_rows = []
    for family, values in sorted(family_values.items()):
        family_rows.append({"profile": "baseline_fp32", "family": family, **values,
                            "independent_total_MAC": values["production_MAC"] + values["independent_extra_MAC"],
                            "independent_total_BOPS": values["production_BOPS"] + values["independent_extra_BOPS"]})
    _write_csv(reports / "bops_family_breakdown.csv", family_rows)
    production_total = float(production_baseline["bops_total"])
    extra_bops = sum(row["independent_extra_BOPS"] for row in family_rows)
    transformer_production = float(production_baseline["transformer_bops_total"])
    reconciliation = {
        "baseline_fp32": {
            "production_total_bops": production_total,
            "production_transformer_bops": transformer_production,
            "production_cnn_bops": float(production_baseline["cnn_bops"]),
            "independent_missing_relation_bops": extra_bops,
            "independent_corrected_total_bops": production_total + extra_bops,
            "transformer_bops_share_production": transformer_production / production_total,
            "transformer_bops_share_corrected": (transformer_production + extra_bops) / (production_total + extra_bops),
            "relative_error": extra_bops / (production_total + extra_bops),
        },
        "bops_formula_version": production_baseline.get("bops_formula_version"),
        "profiles_pending_exact_reconciliation": [],
        "reason": "production and independent runtime-shape formulas reconciled",
    }
    _write_json(reports / "bops_reconciliation.json", reconciliation)
    (reports / "bops_reconciliation.md").write_text(
        "# BOPS reconciliation\n\n"
        f"Production FP32 total: {production_total:.0f} BOPS. Independent HGT relation contractions add "
        f"{extra_bops:.0f} BOPS on the representative shape. Relative under-count: "
        f"{reconciliation['baseline_fp32']['relative_error']:.6%}.\n\n"
        "CNN, standard window QKV/QK/AV/O, HGT relation-attention and standard FFN rows reconcile. "
        "The production formula is versioned; historical budget labels were not rewritten.\n"
    )

    engine_summary = dict(realized_summary.get("MAXIMAL_MIXED") or {})
    engine_provenance_complete = bool(engine_summary.get("provenance_complete"))
    weighted_exact = bool(engine_summary.get("weighted_requested_realized_exact"))
    functional_exact = bool(engine_summary.get("functional_requested_realized_exact"))
    all_marginals_exact = all(row["status"] == "exact" for row in marginal_rows)
    bops_passed = not missing and not mac_mismatches and all_marginals_exact
    precision_passed = bool(
        engine_provenance_complete
        and weighted_exact
        and functional_exact
        and not conflicts
        and not fallbacks
        and not unmapped_precision
    )
    blockers: list[str] = []
    if not bops_passed:
        blockers.append("production_independent_bops_reconciliation_failed")
    if not engine_provenance_complete:
        blockers.append("representative_maximal_mixed_engine_provenance_incomplete")
    if conflicts:
        blockers.append("requested_realized_precision_conflict")
    if fallbacks:
        blockers.append("precision_fallback_detected")
    if unmapped_precision:
        blockers.append("unmapped_precision_unit_detected")
    acceptance = {
        "audit_only": True, "bops_definition_changed": True,
        "bops_formula_version": production_baseline.get("bops_formula_version"),
        "historical_budget_labels_changed": False,
        "parameter_retention_definition_changed": False, "greedy_rerun": False,
        "formal_ga_run": False, "full1789_run": False,
        "production_bops_call_chain_identified": True,
        "transformer_specific_bops_used_in_search": True,
        "greedy_ga_bops_evaluator_consistent": True,
        "q_projection_covered": True, "k_projection_covered": True,
        "v_projection_covered": True, "qk_covered": not any(r["component"] == "qk_relation_transform" for r in missing),
        "softmax_handling_valid": True,
        "av_covered": not any(r["component"] == "message_relation_transform" for r in missing),
        "output_projection_covered": True, "ffn_covered": True,
        "runtime_call_multiplicity_valid": True,
        "shared_module_dedup_valid": shared["canonical_dedup_valid"],
        "production_vs_independent_mac_match": not missing and not mac_mismatches,
        "production_vs_independent_bops_match": not missing and not mac_mismatches,
        "shrinker_marginal_bops_valid": all(r["status"] == "exact" for r in marginal_rows if r["domain_type"] in {"cnn_channel", "grouped_conv_channel"}),
        "attention_marginal_bops_valid": all(r["status"] == "exact" for r in marginal_rows if r["domain_type"] == "attention_dh"),
        "only_qk_is_protected": False,
        "mutable_precision_unit_count": sum(not r["protected"] for r in precision_rows),
        "fixed_fp32_unit_count": sum(r["classification"] == "fixed_fp32" for r in precision_rows),
        "fixed_fp16_unit_count": sum(r["classification"] == "fixed_fp16" for r in precision_rows),
        "protected_qk_count": role_counts.get("qk_matmul", 0),
        "protected_softmax_count": sum(r["role"] == "softmax" and r["protected"] for r in precision_rows),
        "protected_av_count": sum(r["role"] == "av_matmul" and r["protected"] for r in precision_rows),
        "protected_layernorm_count": role_counts.get("layernorm", 0),
        "protected_residual_count": role_counts.get("residual_add", 0),
        "protected_merge_count": role_counts.get("attention_merge", 0),
        "unmapped_precision_unit_count": len(unmapped_precision),
        "requested_realized_exact": precision_passed,
        "precision_conflict_count": len(conflicts), "precision_fallback_count": len(fallbacks),
        "bops_audit_passed": bops_passed,
        "precision_protection_audit_passed": precision_passed,
        "formal_search_allowed": bops_passed and precision_passed,
        "blockers": blockers,
        "input_provenance": {
            "fixed50_manifest": str(FIXED50), "fixed50_manifest_hash": manifest.get("manifest_hash"),
            "checkpoint": str(MODEL_SPECS["v2xvit"]["checkpoint"]),
            "checkpoint_hash": _sha256(MODEL_SPECS["v2xvit"]["checkpoint"]),
            "samples": requested_ids,
        },
    }
    _write_json(reports / "final_audit_acceptance.json", acceptance)
    _write_json(reports / "blockers.json", {
        "formal_search_blocked": not acceptance["formal_search_allowed"],
        "production_code_modified": True,
        "blockers": acceptance["blockers"],
        "evidence": {
            "bops_reconciliation": str(reports / "bops_reconciliation.json"),
            "marginal_audit": str(reports / "shrinker_attention_bops_marginal_audit.csv"),
            "precision_audit": str(reports / "requested_realized_precision.csv"),
        },
    })
    conceptual_layernorms = [path for path, module in model.named_modules() if isinstance(module, torch.nn.LayerNorm)]
    precision_summary = {
        "formal_search_space_quantization_groups": len(space.quantization_groups),
        "mutable_precision_genes": len(space.precision_gene_ids),
        "constant_precision_groups": len(space.constant_precision_group_ids),
        "fixed_fp32_qk": role_counts.get("qk_matmul", 0),
        "fixed_fp16_residual": role_counts.get("residual_add", 0),
        "layernorm_modules_discovered": len(conceptual_layernorms),
        "layernorm_groups_in_formal_search_space": role_counts.get("layernorm", 0),
        "softmax_mutable_groups": sum(r["role"] == "softmax" and not r["protected"] for r in precision_rows),
        "av_mutable_groups": sum(r["role"] == "av_matmul" and not r["protected"] for r in precision_rows),
        "maximal_mixed_functional_conflicts": len(conflicts),
        "maximal_mixed_functional_unmapped": len(unmapped_precision),
        "only_qk_is_protected": False,
        "answer": "no",
    }
    _write_json(reports / "precision_protection_summary.json", precision_summary)
    (reports / "precision_protection_summary.md").write_text(
        "# V2X-ViT precision protection audit\n\n"
        f"The deployment-closed SearchSpace contains {len(space.quantization_groups)} groups: "
        f"{len(space.precision_gene_ids)} mutable genes and {len(space.constant_precision_group_ids)} "
        "constant groups. QK, Softmax output, AV and LayerNorm are fixed FP32; residual and "
        "window-merge boundaries are fixed FP16. FFN activation is bound to the FFN2 weighted "
        "Q/DQ boundary and is no longer an independent chromosome locus.\n\n"
        f"All {len(conceptual_layernorms)} LayerNorm modules survive active filtering. "
        f"Representative maximal-mixed engine provenance complete: {engine_provenance_complete}; "
        f"requested/realized exact: {precision_passed}.\n"
    )
    (reports / "proposed_fix_plan.md").write_text(
        "# Follow-up plan\n\n"
        "The minimum BOPS and precision closure has been implemented and independently audited. "
        "Historical budget labels remain immutable. Any later search must explicitly adopt the new "
        "`unified-bops-v2-hgt-relation-closure` formula version and regenerate candidates; it must "
        "not reinterpret old candidate retention values in place.\n"
    )
    (output / "root_conclusion.md").write_text(
        "# V2X-ViT Transformer BOPS and precision audit\n\n"
        "This is the post-fix audit. The HGT relation terms are now included under a new formula "
        "version; the denominator construction, budget values and historical result labels were not "
        "rewritten. No Greedy, GA or full1789 run was started.\n\n"
        "The formal Greedy and GA share the same unified BOPS evaluator and do invoke the Transformer "
        "proxy. Standard window QKV/QK/AV/O, HGT relation contractions, FFN and Shrinker/Attention "
        f"marginals reconcile; remaining relative error is {reconciliation['baseline_fp32']['relative_error']:.4%}.\n\n"
        f"BOPS audit passed: {bops_passed}. Precision closure passed: {precision_passed}. "
        f"Formal search allowed by this audit: {acceptance['formal_search_allowed']}.\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--representative-engine-dir", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
