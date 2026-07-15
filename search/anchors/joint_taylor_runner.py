"""Production runner for the global joint-Taylor anchor sweep."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import yaml

from ..candidate import CandidatePhenotype, PrecisionDecision
from ..canonicalization import SearchSpaceSpec
from ..pruning_space.action_catalog import build_pruning_action_catalog
from ..quantization_space.legalizer import legalize_group_precision_genes
from ..hashing import candidate_hash, canonical_json_hash
from ..integration.calibration_provider import collect_or_load_fisher_statistics
from ..proxy.joint_taylor import JointTaylorProxy
from ..proxy.parameter_slice_resolver import build_unit_parameter_slices
from ..proxy.virtual_shape_resolver import resolve_virtual_shapes
from .joint_taylor_sweep import AnchorPruningUnit, plan_anchor_structures


def apply_global_anchor_pruning_context(
    context: Any,
    *,
    grouped_conv_mode: str,
    grouped_conv_align: int,
    grouped_allowed_channels_per_group: Sequence[int],
) -> tuple[Any, dict[str, Any]]:
    """Expose every legal trace atom without changing the tracer or graph."""

    modules = dict(context.model.named_modules())
    selected = []
    rejection_counts: dict[str, int] = {}
    for unit in list(getattr(context.trace_result, "atomic_prune_units", []) or []):
        module_path = str(getattr(unit, "root_module_path", ""))
        module = modules.get(module_path)
        reason = ""
        if bool(getattr(unit, "protected", False)):
            reason = "trace_protected"
        elif module is None or getattr(module, "weight", None) is None:
            reason = "root_not_weighted"
        elif str(getattr(unit, "root_axis", "")) not in {"out", "channel"}:
            reason = "root_axis_not_output_channel"
        elif not list(getattr(unit, "root_indices", []) or []):
            reason = "empty_root_indices"
        if reason:
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
            continue
        selected.append(unit)
    by_id = {str(getattr(unit, "stable_id")): unit for unit in selected}
    if not by_id:
        raise RuntimeError("global_anchor_trace_has_no_legal_pruning_units")
    selected = [by_id[key] for key in sorted(by_id)]
    catalog = build_pruning_action_catalog(
        selected,
        grouped_conv_mode=str(grouped_conv_mode),
        grouped_conv_align=int(grouped_conv_align),
        grouped_allowed_channels_per_group=[
            int(value) for value in grouped_allowed_channels_per_group
        ],
    )
    metadata = {
        str(getattr(unit, "stable_id")): {
            "scope_id": str(getattr(unit, "scope_id", "")),
            "root_module_path": str(getattr(unit, "root_module_path", "")),
            "root_axis": str(getattr(unit, "root_axis", "")),
            "root_indices": [int(value) for value in getattr(unit, "root_indices", [])],
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
    audit = {
        "source": "existing_formal_trace_atomic_prune_units",
        "tracer_modified": False,
        "dependency_graph_modified": False,
        "selected_unit_count": len(selected),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "plugin_pruning_gene_count": sum(
            "scatter" in str(getattr(unit, "root_module_path", "")).lower()
            for unit in selected
        ),
    }
    if audit["plugin_pruning_gene_count"]:
        raise RuntimeError("pointpillar_scatter_entered_pruning_gene")
    return (
        replace(
            context,
            atomic_prune_units=selected,
            pruning_action_catalog=catalog,
            search_space=expanded_space,
        ),
        audit,
    )


def build_anchor_precision_phenotype(
    space: SearchSpaceSpec,
    pruned_unit_ids: Sequence[str],
    precision_variant: str,
    *,
    pruning_metadata: Mapping[str, Any] | None = None,
) -> CandidatePhenotype:
    variant = str(precision_variant)
    if variant not in {"strict_fp32", "strict_fp16", "maximal_legal_int8"}:
        raise ValueError(f"unsupported_anchor_precision_variant:{variant}")
    if variant in {"strict_fp32", "strict_fp16"}:
        precision = "FP32" if variant == "strict_fp32" else "FP16"
        profile = {
            module_path: PrecisionDecision(precision, precision)
            for group in space.quantization_groups
            for module_path in group.module_paths
        } or {
            module_path: PrecisionDecision(precision, precision)
            for module_path in space.precision_layer_ids
        }
        return CandidatePhenotype(
            pruned_unit_ids=[str(value) for value in pruned_unit_ids],
            precision_profile=profile,
            metadata={
                **dict(pruning_metadata or {}),
                "anchor_precision_variant": variant,
                "fixed_diagnostic_profile": True,
            },
        )
    requested = {}
    for group in space.quantization_groups:
        precision = (
            "INT8"
            if "INT8" in group.allowed_precisions and not group.protected
            else "FP16"
        )
        requested[group.group_id] = precision
    if space.quantization_groups:
        legalization = legalize_group_precision_genes(
            requested,
            space.quantization_groups,
            default_precision=space.default_precision,
        )
        profile = legalization.expand_to_module_profile()
        metadata = legalization.to_dict()
    else:
        precision = "INT8"
        profile = {
            module_path: PrecisionDecision(precision, precision)
            for module_path in space.precision_layer_ids
        }
        metadata = {"requested_group_profile": {key: precision for key in profile}}
    return CandidatePhenotype(
        pruned_unit_ids=[str(value) for value in pruned_unit_ids],
        precision_profile=profile,
        metadata={
            **metadata,
            **dict(pruning_metadata or {}),
            "anchor_precision_variant": variant,
        },
    )


def stage2_result_to_tau_row(
    task: Mapping[str, Any],
    result: Mapping[str, Any],
    proxy: Mapping[str, Any],
    *,
    required_frames: int,
) -> dict[str, Any]:
    reasons: list[str] = []
    if str(result.get("status", "")) != "ok":
        reasons.append(f"stage2_status:{result.get('status', '')}")
    evaluated = int(result.get("evaluated", result.get("num_evaluated_frames", 0)) or 0)
    skipped = int(result.get("skipped", result.get("num_skipped_frames", -1)) or 0)
    if evaluated != int(required_frames) or skipped != 0:
        reasons.append("full_validation_incomplete")
    requested_hash = str(result.get("requested_precision_profile_hash", ""))
    realized_hash = str(result.get("realized_precision_profile_hash", ""))
    if (
        not bool(result.get("precision_identity_passed", False))
        or not requested_hash
        or requested_hash != realized_hash
    ):
        reasons.append("precision_identity_failed")
    if not bool(result.get("deployment_audits_passed", False)):
        reasons.append("deployment_audits_failed")
    map_value = float(result.get("mAP", result.get("map", float("nan"))))
    return {
        "anchor_id": str(task.get("anchor_id", "")),
        "precision_variant": str(task.get("precision_variant", "")),
        "candidate_hash": str(task.get("candidate_hash", "")),
        "physical_hash": str(result.get("physical_hash", "")),
        "precision_profile_hash": requested_hash,
        "engine_hash": str(result.get("engine_hash", "")),
        "mAP": map_value,
        "L_joint": float(proxy["total_importance"]),
        "L_joint_first_order": float(proxy["first_order_sum"]),
        "L_joint_second_order": float(proxy["second_order_fisher_sum"]),
        "R_prune": float(task.get("realized_prune_rate", 0.0)),
        "requested_prune_rate": float(task.get("requested_prune_rate", 0.0)),
        "R_BOPS": float(result.get("BOPS_retention", float("nan"))),
        "calibration_manifest_hash": str(result.get("calibration_manifest_hash", "")),
        "validation_manifest_hash": str(result.get("validation_manifest_hash", "")),
        "evaluated": evaluated,
        "skipped": skipped,
        "valid_for_tau": not reasons,
        "tau_rejection_reasons": reasons,
    }


def _write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({str(key) for row in rows for key in row})
    with target.open("w", encoding="utf-8", newline="") as handle:
        if not fields:
            return
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True)
                    if isinstance(value, (dict, list, tuple))
                    else value
                    for key, value in row.items()
                }
            )


class JointTaylorAnchorStudy:
    def __init__(self, *, config: Mapping[str, Any], output_root: str | Path) -> None:
        self.config = dict(config)
        self.output_root = Path(output_root).expanduser().resolve()
        self.run_dir: Path | None = None

    def _new_run_dir(self) -> Path:
        output = dict(self.config.get("output", {}) or {})
        name = str(
            output.get(
                "experiment_name", "4090_global_joint_taylor_anchor_sweep"
            )
        )
        stamp = time.strftime("%Y%m%d_%H%M%S")
        path = self.output_root / f"{name}_{stamp}"
        suffix = 0
        while path.exists():
            suffix += 1
            path = self.output_root / f"{name}_{stamp}_{suffix:02d}"
        path.mkdir(parents=True, exist_ok=False)
        return path

    def _build_context(self, run_dir: Path) -> Any:
        # Reuse the accepted context builder wrapper; this does not reuse any
        # old ONNX, engine, calibration cache, or experiment output.
        from .runner import Bops021AnchorStudy

        helper = Bops021AnchorStudy(config=self.config, output_root=self.output_root)
        helper.run_dir = run_dir
        context = helper._build_context(run_dir)
        pruning = dict(self.config.get("pruning", {}) or {})
        grouped = dict(pruning.get("grouped_conv", {}) or {})
        expanded, audit = apply_global_anchor_pruning_context(
            context,
            grouped_conv_mode=str(
                grouped.get("position_mode", "independent_group_topk")
            ),
            grouped_conv_align=int(grouped.get("default_channels_per_group", 4)),
            grouped_allowed_channels_per_group=[
                int(value)
                for value in grouped.get(
                    "allowed_channels_per_group",
                    [4, 8, 16, 32, 64, 128, 256, 512],
                )
            ],
        )
        _write_json(run_dir / "global_pruning_inventory.json", audit)
        return expanded

    def _grouped_specs(self, context: Any) -> dict[str, dict[str, Any]]:
        grouped: dict[str, list[Any]] = {}
        for unit in context.atomic_prune_units:
            constraints = dict(getattr(unit, "constraints", {}) or {})
            if constraints.get("grouped_conv") and not constraints.get("depthwise"):
                grouped.setdefault(str(getattr(unit, "scope_id", "")), []).append(unit)
        configured = tuple(
            int(value)
            for value in dict(
                dict(self.config.get("pruning", {}) or {}).get("grouped_conv", {})
                or {}
            ).get(
                "allowed_channels_per_group", [4, 8, 16, 32, 64, 128, 256, 512]
            )
        )
        specs: dict[str, dict[str, Any]] = {}
        for scope_id, units in sorted(grouped.items()):
            group_counts = {
                int(dict(getattr(unit, "constraints", {}) or {}).get("groups", 0))
                for unit in units
            }
            widths = {
                int(
                    dict(getattr(unit, "constraints", {}) or {}).get(
                        "channels_per_group", 0
                    )
                )
                for unit in units
            }
            if len(group_counts) != 1 or len(widths) != 1:
                raise RuntimeError(f"anchor_grouped_metadata_inconsistent:{scope_id}")
            groups = group_counts.pop()
            width = widths.pop()
            if groups <= 0 or width <= 0 or len(units) != groups * width:
                raise RuntimeError(f"anchor_grouped_inventory_incomplete:{scope_id}")
            physical_groups: dict[int, list[str]] = {group: [] for group in range(groups)}
            local_indices: dict[int, dict[str, int]] = {
                group: {} for group in range(groups)
            }
            for unit in units:
                indices = list(getattr(unit, "root_indices", []) or [])
                if len(indices) != 1:
                    raise RuntimeError(f"anchor_grouped_atom_not_single_channel:{scope_id}")
                physical_group, local = divmod(int(indices[0]), width)
                unit_id = str(getattr(unit, "stable_id"))
                physical_groups[physical_group].append(unit_id)
                local_indices[physical_group][unit_id] = local
            specs[scope_id] = {
                "physical_groups": {
                    group: tuple(sorted(unit_ids))
                    for group, unit_ids in physical_groups.items()
                },
                "local_indices": local_indices,
                "allowed_channels_per_group": configured,
            }
        return specs

    @staticmethod
    def _parameter_count_callback(
        context: Any, unit_slices: Mapping[str, list[Any]]
    ) -> tuple[int, Any]:
        original = sum(int(parameter.numel()) for parameter in context.model.parameters())
        all_keep = build_anchor_precision_phenotype(
            context.search_space, [], "strict_fp16"
        )
        baseline_shapes = resolve_virtual_shapes(context.model, all_keep, unit_slices)
        baseline_mapped = sum(row.parameter_count_before for row in baseline_shapes.values())

        def count(mask: dict[str, int]) -> int:
            phenotype = build_anchor_precision_phenotype(
                context.search_space,
                [unit_id for unit_id, keep in mask.items() if int(keep) == 0],
                "strict_fp16",
            )
            shapes = resolve_virtual_shapes(context.model, phenotype, unit_slices)
            mapped_after = sum(row.parameter_count_after for row in shapes.values())
            return int(original - baseline_mapped + mapped_after)

        return int(original), count

    def _importance_audit(
        self,
        context: Any,
        unit_slices: Mapping[str, list[Any]],
        costs: Mapping[str, Any],
        profile_hash: str,
    ) -> list[dict[str, Any]]:
        units = {str(getattr(unit, "stable_id")): unit for unit in context.atomic_prune_units}
        module_to_precision = {
            module_path: group.group_id
            for group in context.search_space.quantization_groups
            for module_path in group.module_paths
        }
        rows = []
        for unit_id in sorted(costs):
            cost = costs[unit_id]
            slices = list(unit_slices.get(unit_id, []))
            descriptors = [
                {
                    "parameter_name": row.parameter_name,
                    "module_path": row.module_path,
                    "axis": int(row.axis),
                    "indices": list(row.indices),
                    "operation": row.operation,
                }
                for row in slices
            ]
            unique = {
                canonical_json_hash(row): row for row in descriptors
            }
            unit = units[unit_id]
            involved = sorted(
                {
                    module_to_precision[row.module_path]
                    for row in slices
                    if row.module_path in module_to_precision
                }
            )
            rows.append(
                {
                    "group_id": unit_id,
                    "prune_domain_id": str(getattr(unit, "scope_id", "")),
                    "root_module": str(getattr(unit, "root_module_path", "")),
                    "dependent_modules": sorted({row.module_path for row in slices}),
                    "dependent_parameter_slices": list(unique.values()),
                    "unique_parameter_count": int(cost.unique_parameter_count),
                    "duplicate_slice_count": 0,
                    "input_duplicate_slice_count": len(descriptors) - len(unique),
                    "first_order_sum": float(cost.first_order_sum),
                    "second_order_fisher_sum": float(cost.second_order_fisher_sum),
                    "total_importance": float(cost.total_importance),
                    "involved_precision_groups": involved,
                    "current_precision_profile_hash": profile_hash,
                    "importance_mode": cost.importance_mode,
                    "finite": bool(cost.finite),
                    "failure_reason": str(cost.failure_reason),
                }
            )
        if any(row["duplicate_slice_count"] != 0 or not row["finite"] for row in rows):
            raise RuntimeError("anchor_importance_group_audit_failed")
        return rows

    def plan(self, run_dir: Path) -> dict[str, Any]:
        context = self._build_context(run_dir)
        unit_slices = build_unit_parameter_slices(
            context.model, context.atomic_prune_units
        )
        proxy_cfg = dict(self.config.get("proxy", {}) or {})
        fisher = collect_or_load_fisher_statistics(
            model=context.model,
            adapter=context.model_bundle.adapter,
            model_config_path=context.model_config,
            device=torch.device(context.runtime_device),
            cache_path=run_dir / "archives" / "fisher_statistics.pt",
            num_batches=int(proxy_cfg.get("fisher_calibration_batches", 32)),
            checkpoint_hash=context.checkpoint_hash,
            code_commit=context.code_commit,
        )
        _write_json(
            run_dir / "fisher_statistics_manifest.json",
            {
                **dict(fisher.manifest),
                "manifest_hash": fisher.manifest_hash,
                "statistics_version": fisher.statistics_version,
            },
        )
        fp16 = build_anchor_precision_phenotype(
            context.search_space, [], "strict_fp16"
        )
        joint = JointTaylorProxy(
            context.model,
            statistics=fisher,
            unit_to_parameter_slices=unit_slices,
            mode="conditional_joint_taylor_second_order_fisher_diag",
        )
        costs = joint.conditional_group_costs(fp16)
        profile_hash = canonical_json_hash(fp16.realized_precision_profile)
        audit = self._importance_audit(
            context, unit_slices, costs, profile_hash
        )
        _write_json(run_dir / "importance_group_audit.json", audit)
        _write_csv(run_dir / "importance_group_audit.csv", audit)
        units_by_id = {
            str(getattr(unit, "stable_id")): unit for unit in context.atomic_prune_units
        }
        anchor_units = [
            AnchorPruningUnit(
                unit_id=unit_id,
                prune_domain_id=str(getattr(units_by_id[unit_id], "scope_id", "")),
                importance=float(costs[unit_id].total_importance),
                root_module=str(
                    getattr(units_by_id[unit_id], "root_module_path", "")
                ),
                parameter_cost=int(
                    getattr(units_by_id[unit_id], "parameter_cost", 0)
                ),
                protected=bool(getattr(units_by_id[unit_id], "protected", False)),
                dependent_modules=tuple(
                    sorted({row.module_path for row in unit_slices.get(unit_id, [])})
                ),
            )
            for unit_id in sorted(units_by_id)
        ]
        original_params, count_fn = self._parameter_count_callback(context, unit_slices)
        pruning_cfg = dict(self.config.get("pruning", {}) or {})
        dense = dict(pruning_cfg.get("dense", {}) or {})
        grouped_specs = self._grouped_specs(context)
        domain_widths: dict[str, int] = {}
        for unit in anchor_units:
            domain_widths[unit.prune_domain_id] = domain_widths.get(unit.prune_domain_id, 0) + 1
        anchor_cfg = dict(self.config.get("joint_taylor_anchor_sweep", {}) or {})
        plan = plan_anchor_structures(
            anchor_units,
            requested_prune_rates=[
                float(value)
                for value in anchor_cfg.get(
                    "requested_prune_rates", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
                )
            ],
            original_params=original_params,
            parameter_count_fn=count_fn,
            dense_alignment_by_domain={
                domain_id: int(dense.get("alignment", 4))
                for domain_id in domain_widths
            },
            minimum_width_by_domain={
                domain_id: max(
                    int(dense.get("alignment", 4)),
                    int(
                        domain_widths[domain_id]
                        * float(pruning_cfg.get("minimum_retained_ratio", 0.10))
                    ),
                )
                for domain_id in domain_widths
            },
            per_domain_max_prune_rate=float(
                anchor_cfg.get("per_domain_max_prune_rate", 0.8)
            ),
            grouped_domain_specs=grouped_specs,
        )
        _write_json(run_dir / "anchor_structure_manifest.json", plan.to_dict())
        _write_json(run_dir / "global_group_ranking.json", list(plan.global_ranking))
        _write_csv(run_dir / "global_group_ranking.csv", list(plan.global_ranking))
        return {
            "context": context,
            "unit_slices": unit_slices,
            "fisher": fisher,
            "joint": joint,
            "plan": plan,
            "importance_audit": audit,
        }

    def run(self, *, planning_only: bool = False) -> dict[str, Any]:
        run_dir = self._new_run_dir()
        self.run_dir = run_dir
        (run_dir / "anchor_config.yaml").write_text(
            yaml.safe_dump(self.config, sort_keys=False), encoding="utf-8"
        )
        _write_json(
            run_dir / "run_state.json",
            {
                "status": "planning",
                "ANCHOR_SWEEP_COMPLETE": False,
                "TAU_CALIBRATION_PASS": False,
                "STAGE_A_STARTED": False,
                "STAGE_B_ALLOWED": False,
            },
        )
        planning = self.plan(run_dir)
        if not planning_only:
            raise RuntimeError("joint_taylor_anchor_deployment_not_yet_enabled")
        _write_json(
            run_dir / "run_state.json",
            {
                "status": "planning_complete",
                "ANCHOR_SWEEP_COMPLETE": False,
                "TAU_CALIBRATION_PASS": False,
                "STAGE_A_STARTED": False,
                "STAGE_B_ALLOWED": False,
            },
        )
        return {
            "run_dir": str(run_dir),
            "status": "planning_complete",
            "structure_count": len(planning["plan"].structures),
        }


def _load_config(path: str | Path) -> dict[str, Any]:
    return dict(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run global joint-Taylor anchors.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--planning-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    study = JointTaylorAnchorStudy(
        config=_load_config(args.config), output_root=args.output_root
    )
    result = study.run(planning_only=bool(args.planning_only))
    run_dir = Path(result["run_dir"])
    command = "python -m search.anchors.joint_taylor_runner " + " ".join(
        shlex.quote(value)
        for value in [
            "--config",
            str(Path(args.config).resolve()),
            "--output-root",
            str(Path(args.output_root).resolve()),
            *(["--planning-only"] if args.planning_only else []),
        ]
    )
    (run_dir / "reproduction_commands.sh").write_text(command + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
