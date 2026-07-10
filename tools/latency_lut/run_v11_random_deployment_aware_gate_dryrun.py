#!/usr/bin/env python3
"""Materialize random deployment-aware subnets and run ONNX/QDQ/profile gates.

This is a pilot/dry-run helper. It intentionally does not call TensorRT
trtexec and does not run evaluator code unless the caller separately invokes
the pilot engine path.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import statistics
import sys
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[2]
_UNIAD = _ROOT.parent
for _p in (_UNIAD, _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from heal_compress.pruning.artifacts import save_v108_model_artifacts
from heal_compress.pruning.greedy_budget_selector import V108DomainSelection, V108GreedySelectionResult, V108PruningDomain, V108RankingUnit
from heal_compress.pruning.model_io import collect_module_structure, configure_grouped_conv_pruning_fns, load_heal_model, setup_logger
from heal_compress.pruning.propagation import GroupBuilder
from heal_compress.pruning.protection_policy import apply_v108_default_protection
from heal_compress.pruning.shape_invariants import check_model_shape_invariants, snapshot_model_shape_invariants
from heal_compress.tracer.generic_tracer import trace_model
from heal_compress.tracer.op_graph import build_op_graph
from tools.latency_lut.random_deployment_aware_subnet_sampler import (
    DEBLOCK_OUTPUT_PROTECTION_REASON,
    _bin_label,
    _nearest_safe_per_group,
    _parse_safe_set,
)
from tools.latency_lut.physical_structure_v2 import (
    atomic_write_json,
    build_physical_application_ledger,
    build_physical_structure_snapshot_v2,
    build_sampling_structure_request,
    compute_physical_hash_v2,
    validate_physical_application_ledger,
)
from tools.latency_lut.run_v108_complete_taylor_greedy_pruner import (
    _apply_grouped_input_legality_filter,
    _apply_structural_legality_skips,
    _build_global_physical_plan,
    _channels_for_item,
    _regular_grouped_output_item,
)
from tools.latency_lut.run_v11_mixed_precision_lut_dataset_builder import (
    _module_channel_rows,
    _module_shape_index_from_subnet_dir,
    _profile_index_row,
    _read_json_if_exists,
    _scale_table_for_profile,
    apply_deployment_aware_precision_legality,
    ensure_real_pruned_signal_maxk_onnx_for_subnet,
    insert_mixed_precision_qdq,
    module_int8_shape_eligibility,
    precision_constraint_specs_from_canonical_mapping,
    sample_stratified_mixed_precision_profile,
    write_csv,
    write_json,
)
from heal_compress.tracer.dependency_tracer import build_dependency_graph
from heal_compress.tracer.precision_coupling_tracer import build_precision_coupling_groups, precision_groups_to_json


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _round_down(value: int, align: int) -> int:
    align = max(int(align), 1)
    return max(align, (int(value) // align) * align)


def _round_up(value: int, align: int) -> int:
    align = max(int(align), 1)
    return ((int(value) + align - 1) // align) * align


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value).strip("_") or "domain"


def _str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _is_deblock_convtranspose_output_item(item: Any) -> bool:
    name = str(getattr(item, "name", "")).lower()
    module = getattr(item, "module", None)
    direction = str(getattr(item, "direction", "")).lower()
    return (
        isinstance(module, nn.ConvTranspose2d)
        and direction == "out"
        and "pyramid_backbone.deblocks" in name
        and name.endswith(".0")
    )


def _apply_deblock_output_domain_protection(groups: Sequence[Any], args: argparse.Namespace) -> dict[str, Any]:
    protect = _str2bool(getattr(args, "protect_deblock_output", True))
    allow = _str2bool(getattr(args, "allow_deblock_output_pruning", False))
    protected: list[str] = []
    skipped: list[dict[str, Any]] = []
    if not protect or allow:
        return {
            "enabled": bool(protect),
            "allow_deblock_output_pruning": bool(allow),
            "protected_deblock_output_domains": protected,
            "skipped_domains": skipped,
        }
    for group in groups:
        items = list(getattr(group, "items", []) or [])
        deblock_items = [item for item in items if _is_deblock_convtranspose_output_item(item)]
        if not deblock_items:
            continue
        group.protected = True
        group.protected_reason = "requires_deblock_output_pruning_but_deblock_output_protected"
        domain_id = str(getattr(group, "group_id", ""))
        names = [str(getattr(item, "name", "")) for item in deblock_items]
        protected.extend(names)
        skipped.append(
            {
                "pruning_domain_id": domain_id,
                "modules": names,
                "protected_reason": DEBLOCK_OUTPUT_PROTECTION_REASON,
                "skipped_reason": "requires_deblock_output_pruning_but_deblock_output_protected",
            }
        )
    return {
        "enabled": True,
        "allow_deblock_output_pruning": False,
        "protected_deblock_output_domains": sorted(set(protected)),
        "skipped_domains": skipped,
    }


def _target_bin_from_manifest(manifest: Mapping[str, Any]) -> tuple[float, float]:
    raw = str(manifest.get("target_global_prune_bin", "0.00:0.20"))
    left, right = raw.split(":", 1)
    return float(left), float(right)


def _sample_local_target(target: float, target_bin: tuple[float, float], rng: random.Random, max_prune: float) -> float:
    width = max(target_bin[1] - target_bin[0], 0.02)
    value = target + rng.uniform(-0.25 * width, 0.25 * width)
    return max(0.0, min(float(max_prune), min(target_bin[1], max(target_bin[0], value))))


def build_random_dependency_domains(model: nn.Module, adapter: Any, args: argparse.Namespace, out_dir: Path) -> tuple[list[Any], list[V108PruningDomain], dict[str, Any]]:
    sample = adapter.build_synthetic_batch(model)
    trace = trace_model(model, sample, forward_fn=adapter.forward_for_task)
    op_graph = build_op_graph(trace, model, protected_layers=[], det_head_keywords=(), protected_keywords=())
    groups = GroupBuilder(
        op_graph,
        align=int(args.round_to),
        grouped_conv_mode="independent_group_topk",
        protect_residual_add=False,
    ).build()
    configure_grouped_conv_pruning_fns(
        groups,
        argparse.Namespace(group_conv_selection_mode="independent_group_topk", allow_remove_groups=False),
    )
    protection_report = apply_v108_default_protection(
        groups,
        protect_fpn_output=bool(args.protect_fpn_output),
        protect_head_output=bool(args.protect_head_output),
    )
    deblock_protection_report = _apply_deblock_output_domain_protection(groups, args)
    domains: list[V108PruningDomain] = []
    for scope in groups:
        domain_id = str(getattr(scope, "group_id", ""))
        num_channels = int(getattr(scope, "num_channels", 0) or 0)
        if not domain_id or num_channels <= 0:
            continue
        grouped_item = _regular_grouped_output_item(scope)
        grouped_module = getattr(grouped_item, "module", None)
        is_grouped_output = isinstance(grouped_module, nn.Conv2d) and int(getattr(grouped_module, "groups", 1)) > 1
        groups_count = int(grouped_module.groups) if is_grouped_output else 1
        root_name = str(getattr(grouped_item, "name", "")) if grouped_item is not None else str(getattr(scope.items[0], "name", domain_id))
        per_group = num_channels // groups_count if is_grouped_output and groups_count and num_channels % groups_count == 0 else None
        units = [
            V108RankingUnit(
                pruning_domain_id=domain_id,
                root_module_name=root_name,
                root_dim="out",
                num_root_channels=num_channels,
                coupled_unit_id=f"{domain_id}::idx{idx}",
                root_channel_index=idx,
                is_grouped_conv=is_grouped_output,
                importance_raw=0.0,
                importance_normalized=0.0,
                grouped_local_unit_id=f"{domain_id}::g{idx // per_group}::l{idx % per_group}" if per_group else None,
                group_index=idx // per_group if per_group else None,
                local_channel_index=idx % per_group if per_group else None,
                protected_reason=str(getattr(scope, "protected_reason", "") if getattr(scope, "protected", False) else ""),
            )
            for idx in range(num_channels)
        ]
        domains.append(
            V108PruningDomain(
                pruning_domain_id=domain_id,
                root_module_name=root_name,
                root_dim="out",
                num_root_channels=num_channels,
                units=units,
                is_grouped_conv=is_grouped_output,
                groups=groups_count,
                per_group=per_group,
                protected_reason=str(getattr(scope, "protected_reason", "") if getattr(scope, "protected", False) else ""),
            )
        )
    structural_skip_rows = _apply_structural_legality_skips(groups, domains)
    report = {
        "trace_graph": {"num_nodes": len(op_graph.nodes), "num_edges": len(op_graph.edges), "warnings": op_graph.warnings},
        "protection_report": protection_report,
        "deblock_output_protection_report": deblock_protection_report,
        "structural_skip_rows": structural_skip_rows,
        "domain_count": len(domains),
        "uses_taylor_ranking": False,
        "importance_source": "none_random_dependency_domain_sampling",
    }
    write_json(out_dir / "random_dependency_domain_report.json", report)
    return groups, domains, report


def _random_keep_indices(count: int, keep_count: int, rng: random.Random) -> list[int]:
    keep_count = max(0, min(int(keep_count), int(count)))
    if keep_count >= int(count):
        return list(range(int(count)))
    return sorted(rng.sample(list(range(int(count))), keep_count))


def _grouped_keep_indices(groups: int, per_group: int, keep_per_group: int, rng: random.Random) -> list[int]:
    keep: list[int] = []
    for group_idx in range(int(groups)):
        start = group_idx * int(per_group)
        local = sorted(rng.sample(list(range(int(per_group))), int(keep_per_group)))
        keep.extend(start + idx for idx in local)
    return sorted(keep)


def random_selection_for_subnet(
    domains: Sequence[V108PruningDomain],
    manifest: Mapping[str, Any],
    *,
    seed: int,
    round_to: int,
    max_channel_prune_ratio: float,
    min_channel_keep_ratio: float,
    grouped_safe_per_group: set[int],
) -> tuple[V108GreedySelectionResult, list[dict[str, Any]]]:
    rng = random.Random(int(seed))
    target = _as_float(manifest.get("sampled_target_global_prune_ratio"), 0.2)
    target_bin = _target_bin_from_manifest(manifest)
    plans: dict[str, V108DomainSelection] = {}
    grouped_rows: list[dict[str, Any]] = []
    total_channels = 0
    total_pruned = 0
    selected_units: list[V108RankingUnit] = []
    skipped_units: list[V108RankingUnit] = []
    trace_rows: list[dict[str, Any]] = []
    for domain in domains:
        domain_id = str(domain.pruning_domain_id)
        current = int(domain.num_root_channels)
        total_channels += current
        plan = V108DomainSelection(pruning_domain_id=domain_id, keep_indices=list(range(current)))
        skip_reason = str(domain.protected_reason or domain.skipped_reason or "")
        if skip_reason:
            plan.skipped_reason = skip_reason
            plans[domain_id] = plan
            skipped_units.extend(domain.units)
            continue
        local_target = _sample_local_target(target, target_bin, rng, max_channel_prune_ratio)
        if domain.is_grouped_conv and int(domain.groups) > 1 and int(domain.per_group or 0) > 0:
            before_per = int(domain.per_group or 0)
            raw_target_per = max(1, int(round(before_per * max(min_channel_keep_ratio, 1.0 - local_target))))
            after_per, snap_reason = _nearest_safe_per_group(
                raw_target_per,
                before_per,
                grouped_safe_per_group,
                max_channel_prune_ratio=max_channel_prune_ratio,
                min_channel_keep_ratio=min_channel_keep_ratio,
            )
            if snap_reason.startswith("skipped"):
                plan.skipped_reason = snap_reason
                keep = list(range(current))
            else:
                keep = _grouped_keep_indices(int(domain.groups), before_per, after_per, rng)
            grouped_rows.append(
                {
                    "module_name": domain.root_module_name,
                    "pruning_domain_id": domain_id,
                    "groups": int(domain.groups),
                    "before_cout_per_group": before_per,
                    "after_cout_per_group": len(keep) // int(domain.groups) if int(domain.groups) else None,
                    "safe_per_group_set": sorted(grouped_safe_per_group),
                    "int8_shape_supported": (len(keep) // int(domain.groups) if int(domain.groups) else 0) in grouped_safe_per_group,
                    "snapped_from_target": {
                        "target_cout_per_group": raw_target_per,
                        "snapped_cout_per_group": after_per,
                    }
                    if after_per != raw_target_per
                    else {},
                    "snapping_policy": "nearest_tie_keeps_more_channels",
                    "unsupported_reason": "",
                    "whether_8_to_4": before_per == 8 and after_per == 4,
                    "whether_16_to_8_or_4": before_per == 16 and after_per in {4, 8},
                }
            )
        else:
            raw_keep = int(round(current * max(min_channel_keep_ratio, 1.0 - local_target)))
            min_keep = _round_up(max(1, int(round(current * min_channel_keep_ratio))), round_to)
            keep_count = min(current, max(min_keep, _round_down(raw_keep, round_to)))
            keep = _random_keep_indices(current, keep_count, rng)
        prune = [idx for idx in range(current) if idx not in set(keep)]
        plan.keep_indices = keep
        plan.prune_indices = prune
        plan.raw_selected_count = len(prune)
        plan.final_n_pruned = len(prune)
        total_pruned += len(prune)
        for unit in domain.units:
            if int(unit.root_channel_index) in set(prune):
                unit.selected_for_pruning = True
                selected_units.append(unit)
        trace_rows.append(
            {
                "pruning_domain_id": domain_id,
                "root_module_name": domain.root_module_name,
                "target_prune_ratio": local_target,
                "before_channels": current,
                "after_channels": len(keep),
                "pruned_channels": len(prune),
                "is_grouped_conv": domain.is_grouped_conv,
                "skipped_reason": plan.skipped_reason,
            }
        )
        plans[domain_id] = plan
    actual_channel = total_pruned / max(total_channels, 1)
    return (
        V108GreedySelectionResult(
            target_pruning_mode="channel",
            target_pruning_ratio=target,
            actual_channel_prune_ratio_on_searchable_surface=actual_channel,
            actual_param_prune_ratio=0.0,
            predicted_param_prune_ratio=0.0,
            param_prediction_error=0.0,
            param_budget_overshoot_ratio=0.0,
            actual_flops_proxy_ratio=None,
            rounding_overshoot_ratio=0.0,
            max_ch_sparsity_blocked_count=0,
            skipped_because_pergroup_below8_count=0,
            skipped_because_pergroup_below_round_to_count=0,
            unreachable=False,
            domain_plans=plans,
            selected_units=selected_units,
            skipped_units=skipped_units,
            trace_rows=trace_rows,
            grouped_shape_rows=grouped_rows,
        ),
        grouped_rows,
    )


def _hash_file(path: Path) -> str:
    import hashlib

    if not path.is_file():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _onnx_initializer_shape_hash(path: Path) -> str:
    import hashlib
    import onnx

    if not path.is_file():
        return ""
    model = onnx.load(str(path))
    rows = [(init.name, list(init.dims)) for init in model.graph.initializer]
    raw = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def materialize_subnet(
    *,
    baseline: nn.Module,
    adapter: Any,
    groups: Sequence[Any],
    domains: Sequence[V108PruningDomain],
    subnet_dir: Path,
    args: argparse.Namespace,
    structure_before: Mapping[str, Any],
    invariant_before: Mapping[str, Any],
    params_before: int,
) -> tuple[nn.Module | None, list[Any], dict[str, Any]]:
    manifest_path = subnet_dir / "pruning_manifest.json"
    manifest = _read_json_if_exists(manifest_path, {})
    if not manifest:
        return None, [], {"success": False, "failure_reason": "missing_pruning_manifest"}
    pruned = copy.deepcopy(baseline).to(args.device).eval()
    selection, grouped_rows = random_selection_for_subnet(
        domains,
        manifest,
        seed=_as_int(manifest.get("random_seed"), int(args.profile_seed)),
        round_to=int(args.round_to),
        max_channel_prune_ratio=float(args.max_channel_prune_ratio),
        min_channel_keep_ratio=float(args.min_channel_keep_ratio),
        grouped_safe_per_group=_parse_safe_set(args.grouped_conv_safe_per_group),
    )
    grouped_input_reports = _apply_grouped_input_legality_filter(groups, selection, align_channels=int(args.round_to))
    if grouped_input_reports:
        selection.grouped_shape_rows.extend(grouped_input_reports)
    physical_plan = _build_global_physical_plan(groups, selection, align_channels=int(args.round_to))
    try:
        surgery = physical_plan.apply_one_shot(pruned)
    except Exception as exc:  # noqa: BLE001
        return None, [], {"success": False, "failure_reason": f"physical_prune_failed:{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
    structure_after = collect_module_structure(pruned)
    shape_report = check_model_shape_invariants(invariant_before, snapshot_model_shape_invariants(pruned))
    if not shape_report.get("passed"):
        write_json(subnet_dir / "shape_invariant_report.json", shape_report.get("rows", []))
        return None, [], {"success": False, "failure_reason": f"shape_invariant_failed:{shape_report.get('non_channel_shape_violation_count')}"}
    params_after = int(sum(param.numel() for param in pruned.parameters()))
    channel_rows = _module_channel_rows(structure_before, structure_after)
    sample = adapter.build_synthetic_batch(pruned)
    try:
        with torch.no_grad():
            adapter.forward_for_task(pruned, sample)
        forward_ok = True
        forward_reason = ""
    except Exception as exc:  # noqa: BLE001
        forward_ok = False
        forward_reason = f"{type(exc).__name__}: {exc}"
    dep_graph = build_dependency_graph(pruned, sample)
    precision_groups = build_precision_coupling_groups(pruned, dep_graph, sample)
    structure_hash = str(manifest.get("structure_hash", ""))
    manifest.update(
        {
            "dry_run": False,
            "materialized_from_random_dependency_domains": True,
            "uses_taylor_ranking": False,
            "importance_source": "none_random_dependency_domain_sampling",
            "actual_param_prune_ratio": 1.0 - params_after / max(params_before, 1),
            "actual_channel_prune_ratio_on_searchable_surface": selection.actual_channel_prune_ratio_on_searchable_surface,
            "actual_channel_prune_ratio": selection.actual_channel_prune_ratio_on_searchable_surface,
            "shape_invariant_passed": True,
            "pytorch_forward_shape_sanity_passed": forward_ok,
            "pytorch_forward_shape_sanity_failure_reason": forward_reason,
            "physical_prune_surgery": surgery,
            "module_channel_before_after": channel_rows,
            "grouped_shape_report": selection.grouped_shape_rows,
            "global_physical_prune_plan_path": str(subnet_dir / "global_physical_prune_plan.json"),
        }
    )
    write_json(manifest_path, manifest)
    write_json(subnet_dir / "global_physical_prune_plan.json", physical_plan.to_json())
    write_json(subnet_dir / "shape_invariant_report.json", shape_report.get("rows", []))
    write_json(subnet_dir / "grouped_shape_report.json", selection.grouped_shape_rows)
    write_json(subnet_dir / "grouped_pergroup_round_to_shape_report.json", selection.grouped_shape_rows)
    write_json(subnet_dir / "module_channel_before_after.json", channel_rows)
    write_json(subnet_dir / "precision_coupling_groups.json", precision_groups_to_json(precision_groups))
    artifacts = save_v108_model_artifacts(
        model=pruned,
        models_dir=subnet_dir,
        manifest=manifest,
        model_config=str(args.model_config),
        checkpoint_source=str(args.checkpoint),
    )
    sampling_request = build_sampling_structure_request(manifest)
    snapshot = build_physical_structure_snapshot_v2(
        pruned,
        state_dict=pruned.state_dict(),
        generated_from="pruned_model_object.pth:model.named_modules+pruned_state_dict_with_manifest.pth",
    )
    ledger = build_physical_application_ledger(
        sampling_request,
        snapshot,
        physical_plan=physical_plan.to_json(),
    )
    ledger["validation"] = validate_physical_application_ledger(sampling_request, ledger)
    physical_hash = compute_physical_hash_v2(
        snapshot,
        legacy_structure_hash=str(manifest.get("structure_hash", "")),
        legacy_shape_hash=str(manifest.get("shape_hash", "")),
    )
    atomic_write_json(subnet_dir / "sampling_structure_request.json", sampling_request)
    atomic_write_json(subnet_dir / "physical_pruning_application_ledger.json", ledger)
    atomic_write_json(subnet_dir / "physical_structure_snapshot_v2.json", snapshot)
    atomic_write_json(subnet_dir / "physical_hash_v2.json", physical_hash)
    return pruned, precision_groups, {
        "success": bool(forward_ok),
        "failure_reason": forward_reason,
        "params_after": params_after,
        "structure_hash": structure_hash,
        "structure_hash_v2": physical_hash["structure_hash_v2"],
        "shape_hash_v2": physical_hash["shape_hash_v2"],
        "pruned_model_path": str(artifacts["model_object"]),
    }


def _grouped_eligibility_for_subnet(subnet_dir: Path) -> list[dict[str, Any]]:
    shapes = _module_shape_index_from_subnet_dir(subnet_dir)
    rows = []
    for shape in shapes.values():
        if _as_int(shape.get("groups"), 1) <= 1:
            continue
        eligibility = module_int8_shape_eligibility(shape)
        rows.append(eligibility)
    write_json(subnet_dir / "grouped_conv_int8_eligibility_report.json", rows)
    return rows


def _profile_gate_rows_for_subnet(subnet_dir: Path, subnet_id: str, structure_hash: str, groups: Sequence[Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    module_shapes = _module_shape_index_from_subnet_dir(subnet_dir)
    for profile_index in range(4):
        profile_id = f"profile_{profile_index:03d}"
        profile_dir = subnet_dir / profile_id
        profile_dir.mkdir(parents=True, exist_ok=True)
        profile = sample_stratified_mixed_precision_profile(
            subnet_id=subnet_id,
            structure_hash=structure_hash,
            groups=groups,
            profile_index=profile_index,
            profile_seed=int(args.profile_seed),
            subnet_index=_as_int(subnet_id.split("_")[-1], 0),
        )
        profile = apply_deployment_aware_precision_legality(profile, module_shapes)
        write_json(profile_dir / "mixed_precision_profile.json", profile)
        scale_table = _scale_table_for_profile(profile)
        write_json(profile_dir / "scale_table.json", scale_table)
        onnx_path, export_report = ensure_real_pruned_signal_maxk_onnx_for_subnet(subnet_dir, args)
        qdq_report: dict[str, Any] = {"success": False, "failure_reason": "onnx_export_failed"}
        constraint_specs: list[str] = []
        if onnx_path is not None and export_report.get("export_success"):
            input_onnx = profile_dir / "onnx" / "model.onnx"
            output_onnx = profile_dir / "onnx" / "model_mixed_qdq.onnx"
            input_onnx.parent.mkdir(parents=True, exist_ok=True)
            input_onnx.write_bytes(onnx_path.read_bytes())
            from tools.latency_lut.run_v11_mixed_precision_lut_dataset_builder import copy_onnx_origin_artifacts

            copy_onnx_origin_artifacts(onnx_path, input_onnx.parent)
            try:
                from tools.latency_lut.run_v11_mixed_precision_lut_dataset_builder import build_canonical_precision_mapping, make_qdq_insert_report, write_canonical_precision_mapping

                canonical_mapping = build_canonical_precision_mapping(input_onnx, profile)
                profile["canonical_precision_mapping"] = canonical_mapping
                write_canonical_precision_mapping(profile_dir, canonical_mapping)
                qdq_payload = insert_mixed_precision_qdq(input_onnx=input_onnx, output_onnx=output_onnx, profile=profile, scale_table=scale_table)
                canonical_mapping = profile.get("canonical_precision_mapping", canonical_mapping)
                write_canonical_precision_mapping(profile_dir, canonical_mapping)
                constraint_specs = precision_constraint_specs_from_canonical_mapping(canonical_mapping)
                qdq_report = make_qdq_insert_report(
                    input_onnx=str(input_onnx),
                    output_onnx=str(output_onnx),
                    profile=profile,
                    calibration_frame_ids=list(range(int(args.calib_train_frames))),
                    scale_table=scale_table,
                    success=True,
                    inserted_qdq_nodes=qdq_payload.get("inserted_qdq_nodes", []),
                    skipped_non_int8_layers=qdq_payload.get("skipped_non_int8_layers", []),
                    matched_int8_precision_groups=qdq_payload.get("matched_int8_precision_groups", []),
                    unmatched_int8_precision_groups=qdq_payload.get("unmatched_int8_precision_groups", []),
                    qdq_node_count=int(qdq_payload.get("qdq_node_count", 0)),
                    quantize_linear_count=int(qdq_payload.get("quantize_linear_count", 0)),
                    dequantize_linear_count=int(qdq_payload.get("dequantize_linear_count", 0)),
                )
                write_json(profile_dir / "qdq_insert_report.json", qdq_report)
                write_json(profile_dir / "precision_constraint_specs.json", constraint_specs)
                write_json(profile_dir / "onnx_check_report.json", {"success": True, "onnx_path": str(output_onnx), "status": "onnx_qdq_dryrun_passed"})
            except Exception as exc:  # noqa: BLE001
                qdq_report = {"success": False, "failure_reason": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
                write_json(profile_dir / "qdq_insert_report.json", qdq_report)
        fallback_reasons: dict[str, int] = {}
        for item in profile.get("fallback_layers", []):
            reason = str(item.get("fallback_reason", ""))
            fallback_reasons[reason] = fallback_reasons.get(reason, 0) + 1
        mapping = _read_json_if_exists(profile_dir / "canonical_precision_mapping.json", {})
        rows.append(
            {
                "subnet_id": subnet_id,
                "profile_id": profile_id,
                "onnx_export_success": bool(export_report.get("export_success")),
                "origin_map_success": bool(export_report.get("origin_map_success")),
                "origin_map_entry_count": int(export_report.get("origin_map_entry_count", 0) or 0),
                "onnx_hash": _hash_file(subnet_dir / "onnx" / "model_signal_maxk.onnx"),
                "initializer_shape_hash": _onnx_initializer_shape_hash(subnet_dir / "onnx" / "model_signal_maxk.onnx"),
                "canonical_mapping_entry_count": int(mapping.get("entry_count", 0) or len(mapping.get("entries", []))),
                "ambiguous_mapping_count": 0 if mapping.get("success") else 1,
                "unmatched_onnx_node_count": len(qdq_report.get("unmatched_int8_precision_groups", [])),
                "qdq_insert_success": bool(qdq_report.get("success")),
                "inserted_qdq_nodes_count": len(qdq_report.get("inserted_qdq_nodes", [])),
                "unmatched_int8_precision_groups": qdq_report.get("unmatched_int8_precision_groups", []),
                "concat_fp16_boundary_count": int(profile.get("concat_fp16_boundary_count", 0) or 0),
                "concat_requantize_after_count": int(profile.get("concat_requantize_after_count", 0) or 0),
                "requested_int8_group_ratio": profile.get("requested_int8_group_ratio"),
                "requested_int8_layer_count": sum(1 for value in (profile.get("layer_precision_assignment") or {}).values() if str(value).lower() == "int8"),
                "fallback_fp16_count": len(profile.get("fallback_layers", [])),
                "fallback_reasons": fallback_reasons,
                "precision_constraint_specs_count": len(constraint_specs),
            }
        )
    return rows


def _write_report(output_dir: Path, subnet_rows: Sequence[Mapping[str, Any]], profile_rows: Sequence[Mapping[str, Any]], recommended: Mapping[str, Any] | None) -> None:
    lines = [
        "# ONNX + QDQ + Profile Gate Dry Run",
        "",
        "trtexec_executed: false",
        "engine_build_executed: false",
        "eval_executed: false",
        "",
        "## Subnets",
        "",
        "| subnet_id | materialized | forward_ok | onnx | origin_map | origin_entries | grouped_unsupported | structure_hash | shape_hash | actual_param | actual_channel |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|---:|---:|",
    ]
    for row in subnet_rows:
        lines.append(
            f"| {row['subnet_id']} | {row['materialization_success']} | {row['pytorch_forward_shape_sanity_passed']} | "
            f"{row['onnx_export_success']} | {row.get('origin_map_success')} | {row.get('origin_map_entry_count', 0)} | "
            f"{row['grouped_conv_unsupported_int8_shape_count']} | {row.get('structure_hash','')} | "
            f"{row.get('shape_hash','')} | {float(row.get('actual_param_prune_ratio') or 0):.4f} | {float(row.get('actual_channel_prune_ratio') or 0):.4f} |"
        )
    lines.extend(["", "## Profiles", "", "| subnet_id | profile_id | origin_map | qdq | mapping_entries | int8_layers | fallback_fp16 | concat_boundary | specs |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"])
    for row in profile_rows:
        lines.append(
            f"| {row['subnet_id']} | {row['profile_id']} | {row.get('origin_map_success')} | {row['qdq_insert_success']} | {row['canonical_mapping_entry_count']} | "
            f"{row['requested_int8_layer_count']} | {row['fallback_fp16_count']} | {row['concat_fp16_boundary_count']} | {row['precision_constraint_specs_count']} |"
        )
    lines.extend(["", "## Recommendation", ""])
    if recommended:
        lines.append(
            f"recommended_candidate: {recommended['subnet_id']}/{recommended['profile_id']} "
            f"(requested_int8_layer_count={recommended['requested_int8_layer_count']})"
        )
    else:
        lines.append("recommended_candidate: none")
    (output_dir / "onnx_qdq_profile_gate_dryrun_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    subnet_root = output_dir / "subnets"
    subnet_dirs = sorted(subnet_root.glob("subnet_*"))[: int(args.max_subnets) or None]
    if not subnet_dirs:
        raise FileNotFoundError(f"no subnets under {subnet_root}")
    args.build_engines = False
    args.eval_engines = False
    args.overwrite_profiles = True
    args.require_onnx_origin_map = True
    args.device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    logger = setup_logger(output_dir / "random_gate_dryrun_logs", name="v11_random_gate_dryrun")
    baseline, adapter = load_heal_model(args, torch.device(args.device), logger)
    baseline.eval()
    params_before = int(sum(param.numel() for param in baseline.parameters()))
    structure_before = collect_module_structure(baseline)
    invariant_before = snapshot_model_shape_invariants(baseline)
    groups, domains, domain_report = build_random_dependency_domains(baseline, adapter, args, output_dir)
    subnet_rows: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []
    for subnet_dir in subnet_dirs:
        subnet_id = subnet_dir.name
        manifest = _read_json_if_exists(subnet_dir / "pruning_manifest.json", {})
        pruned, precision_groups, materialize_report = materialize_subnet(
            baseline=baseline,
            adapter=adapter,
            groups=groups,
            domains=domains,
            subnet_dir=subnet_dir,
            args=args,
            structure_before=structure_before,
            invariant_before=invariant_before,
            params_before=params_before,
        )
        grouped = _grouped_eligibility_for_subnet(subnet_dir) if materialize_report.get("success") else []
        export_report: dict[str, Any] = {}
        if materialize_report.get("success"):
            _onnx_path, export_report = ensure_real_pruned_signal_maxk_onnx_for_subnet(subnet_dir, args)
            profile_rows.extend(
                _profile_gate_rows_for_subnet(
                    subnet_dir,
                    subnet_id,
                    str((_read_json_if_exists(subnet_dir / "pruning_manifest.json", {}) or {}).get("structure_hash", "")),
                    precision_groups,
                    args,
                )
            )
        updated_manifest = _read_json_if_exists(subnet_dir / "pruning_manifest.json", {})
        subnet_rows.append(
            {
                "subnet_id": subnet_id,
                "materialization_success": bool(materialize_report.get("success")),
                "materialization_failure_reason": materialize_report.get("failure_reason", ""),
                "pytorch_forward_shape_sanity_passed": bool(updated_manifest.get("pytorch_forward_shape_sanity_passed")),
                "onnx_export_success": bool(export_report.get("export_success")),
                "origin_map_success": bool(export_report.get("origin_map_success")),
                "origin_map_entry_count": int(export_report.get("origin_map_entry_count", 0) or 0),
                "onnx_hash": _hash_file(subnet_dir / "onnx" / "model_signal_maxk.onnx"),
                "initializer_shape_hash": _onnx_initializer_shape_hash(subnet_dir / "onnx" / "model_signal_maxk.onnx"),
                "grouped_conv_unsupported_int8_shape_count": sum(1 for row in grouped if not row.get("int8_shape_supported")),
                "structure_hash": updated_manifest.get("structure_hash", manifest.get("structure_hash", "")),
                "shape_hash": updated_manifest.get("shape_hash", manifest.get("shape_hash", "")),
                "actual_param_prune_ratio": updated_manifest.get("actual_param_prune_ratio"),
                "actual_channel_prune_ratio": updated_manifest.get("actual_channel_prune_ratio"),
            }
        )
    write_csv(output_dir / "onnx_qdq_profile_gate_subnet_summary.csv", subnet_rows)
    write_csv(output_dir / "onnx_qdq_profile_gate_profile_summary.csv", profile_rows)
    write_json(output_dir / "onnx_qdq_profile_gate_dryrun_summary.json", {"subnets": subnet_rows, "profiles": profile_rows, "domain_report": domain_report})
    candidates = [
        row
        for row in profile_rows
        if row.get("qdq_insert_success")
        and not row.get("unmatched_int8_precision_groups")
        and "grouped_conv_int8_per_group_shape_not_supported" not in row.get("fallback_reasons", {})
    ]
    recommended = sorted(
        candidates,
        key=lambda row: (
            -int(row.get("requested_int8_layer_count", 0) or 0),
            str(row.get("profile_id", "")) != "profile_003",
            str(row.get("subnet_id", "")),
        ),
    )[0] if candidates else None
    if recommended:
        write_json(output_dir / "recommended_single_engine_candidate.json", recommended)
    _write_report(output_dir, subnet_rows, profile_rows, recommended)
    print(json.dumps({"subnet_count": len(subnet_rows), "profile_count": len(profile_rows), "recommended": recommended}, indent=2, default=str))
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-subnets", type=int, default=16)
    parser.add_argument("--round-to", type=int, default=4)
    parser.add_argument("--max-channel-prune-ratio", type=float, default=0.80)
    parser.add_argument("--min-channel-keep-ratio", type=float, default=0.20)
    parser.add_argument("--grouped-conv-safe-per-group", default="4,8,16,32")
    parser.add_argument("--profile-seed", type=int, default=20260708)
    parser.add_argument("--calib-train-frames", type=int, default=200)
    parser.add_argument("--fixed-k", "--fixed_k", dest="fixed_k", type=int, default=29696)
    parser.add_argument("--heal-root", default="/home/lixingfeng/UniAD_examine/HEAL")
    parser.add_argument("--model-config", default="/home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_pyramid/config.yaml")
    parser.add_argument("--checkpoint", default="/home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_pyramid/net_epoch_bestval_at17.pth")
    parser.add_argument("--plugin", default="quantization/plugins/pointpillar_scatter_trt/build/libpointpillar_scatter_trt.so")
    parser.add_argument("--trt-root", default="/home/lixingfeng/UniAD_examine/HEAL/prune_model/TensorRT-10.9_x86_cu118")
    parser.add_argument("--trtexec", default="")
    parser.add_argument("--device", default="")
    parser.add_argument("--protect-fpn-output", action="store_true", default=True)
    parser.add_argument("--protect-head-output", action="store_true", default=True)
    parser.add_argument("--allow-deblock-output-pruning", type=str, default="false")
    parser.add_argument("--protect-deblock-output", type=str, default="true")
    parser.add_argument("--num-calib-batches", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return run(parse_args(argv))
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"success": False, "failure_reason": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
