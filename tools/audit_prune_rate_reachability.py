#!/usr/bin/env python3
"""Audit historical and current lidar_pyramid pruning reachability.

This tool is intentionally CPU-first and read-only with respect to historical
artifacts. It does not run GA, QDQ, TensorRT, or automatically repair old masks.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from search.audits.prune_rate_reachability import (  # noqa: E402
    classify_replay_failure,
    compare_reachability_masks,
    solve_independent_domain_max,
)
from search.candidate import CandidatePhenotype  # noqa: E402
from search.integration.model_provider import (  # noqa: E402
    load_lidar_pyramid_model,
    sha256_file,
)
from search.proxy.parameter_slice_resolver import (  # noqa: E402
    ParameterSlice,
    build_unit_parameter_slices,
)
from search.proxy.virtual_shape_resolver import resolve_virtual_shapes  # noqa: E402


CHECKPOINT = Path(
    "/home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/"
    "LiDAROnly/lidar_pyramid/net_epoch_bestval_at17.pth"
)
MODEL_CONFIG = Path(
    "/home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/"
    "LiDAROnly/lidar_pyramid/config.yaml"
)
HEAL_ROOT = Path("/home/lixingfeng/UniAD_examine/HEAL")
HISTORICAL_ROOT = ROOT / "outputs/latency_lut/v109_param_budget_round4_pruner"
CURRENT_ANCHOR_ROOT = ROOT / "outputs/4090_global_joint_taylor_anchor_sweep_20260715_133506"
CURRENT_BRANCH = "feature/heal-compress-h800-sync-4090"
H800_ANCESTOR = "b862b3d8ad061bd12580776226c75f564918298d"


def _run(command: Sequence[str], *, check: bool = True) -> dict[str, Any]:
    result = subprocess.run(
        list(command), cwd=ROOT, text=True, capture_output=True, check=False
    )
    if check and result.returncode:
        raise RuntimeError(
            f"command_failed:{command}:exit={result.returncode}:{result.stderr.strip()}"
        )
    return {
        "command": list(command),
        "exit_code": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({str(key) for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
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


def _sha(path: Path) -> str:
    return sha256_file(path) if path.is_file() else ""


def _entry_audit() -> dict[str, Any]:
    commands = {
        "fetch": _run(
            ["git", "fetch", "origin", CURRENT_BRANCH]
        ),
        "branch": _run(["git", "branch", "--show-current"]),
        "head": _run(["git", "rev-parse", "HEAD"]),
        "status": _run(["git", "status", "--short"]),
        "remote_delta": _run(
            [
                "git",
                "rev-list",
                "--left-right",
                "--count",
                f"HEAD...origin/{CURRENT_BRANCH}",
            ]
        ),
        "ancestor": _run(
            ["git", "merge-base", "--is-ancestor", H800_ANCESTOR, "HEAD"],
            check=False,
        ),
        "nvidia_smi": _run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,utilization.gpu,memory.used,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            check=False,
        ),
        "nvidia_smi_full": _run(["nvidia-smi"], check=False),
        "processes": _run(
            [
                "pgrep",
                "-af",
                "search|prun|taylor|anchor|stage|trtexec|TensorRT|evaluation",
            ],
            check=False,
        ),
    }
    branch = commands["branch"]["stdout"]
    if branch != CURRENT_BRANCH:
        raise RuntimeError(f"wrong_branch:{branch}")
    if commands["ancestor"]["exit_code"] != 0:
        raise RuntimeError("missing_h800_ancestor")
    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "branch": branch,
        "head": commands["head"]["stdout"],
        "remote_delta": commands["remote_delta"]["stdout"],
        "h800_ancestor": H800_ANCESTOR,
        "h800_ancestor_present": True,
        "checkpoint": str(CHECKPOINT),
        "checkpoint_hash": _sha(CHECKPOINT),
        "model_config": str(MODEL_CONFIG),
        "model_config_hash": _sha(MODEL_CONFIG),
        "commands": commands,
        "initial_pre_modification_audit": {
            "captured_at_task_entry": True,
            "branch": CURRENT_BRANCH,
            "head": "3f33afb680b2c90e202b534bd99022534f8c94d8",
            "remote_head": "3f33afb680b2c90e202b534bd99022534f8c94d8",
            "remote_delta": "0 0",
            "working_tree_status_after_preserving_prior_wip": "",
            "prior_wip_stash": "codex-wip-joint-taylor-stage1-before-reachability-audit-20260715",
            "h800_ancestor_present": True,
        },
        "ga_started": False,
        "stage_a_started": False,
        "stage_b_started": False,
        "tensorrt_started": False,
    }


def _parameter_masks(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: torch.zeros(int(parameter.numel()), dtype=torch.bool)
        for name, parameter in model.named_parameters()
    }


def _slice_union_audit(
    model: torch.nn.Module,
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    parameters = dict(model.named_parameters())
    masks = _parameter_masks(model)
    raw_sum = 0
    audit_rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        name = str(row["parameter_name"])
        parameter = parameters.get(name)
        if parameter is None:
            missing.append({**row, "reason": "missing_parameter"})
            continue
        axis = int(row["axis"])
        indices = sorted({int(value) for value in row.get("indices", [])})
        if axis < 0:
            axis += parameter.ndim
        if not 0 <= axis < parameter.ndim:
            missing.append({**row, "reason": "missing_parameter_slice:axis"})
            continue
        width = int(parameter.shape[axis])
        valid = [value for value in indices if 0 <= value < width]
        if not valid:
            missing.append({**row, "reason": "missing_parameter_slice:indices"})
            continue
        view = masks[name].view(tuple(parameter.shape))
        element_count = 0
        overlap_count = 0
        for index in valid:
            restricted_start = row.get("restricted_output_start")
            restricted_end = row.get("restricted_output_end")
            restricted = (
                view[int(restricted_start) : int(restricted_end)]
                if restricted_start is not None and restricted_end is not None
                else view
            )
            target = restricted.select(axis, index)
            element_count += int(target.numel())
            overlap_count += int(target.sum().item())
            target.fill_(True)
        raw_sum += element_count
        audit_rows.append(
            {
                **row,
                "indices": valid,
                "element_count": element_count,
                "overlap_count": overlap_count,
                "mapping_status": "mapped",
            }
        )
    by_parameter = {
        name: int(mask.sum().item()) for name, mask in sorted(masks.items())
    }
    union = sum(by_parameter.values())
    return {
        "raw_element_sum": raw_sum,
        "global_union_element_count": union,
        "duplicate_or_overlap_element_count": raw_sum - union,
        "global_union_by_parameter": by_parameter,
        "row_results": audit_rows,
        "missing_rows": missing,
        "masks": masks,
    }


def _flatten_unit_slices(
    units: Sequence[Any],
    unit_slices: Mapping[str, Sequence[ParameterSlice]],
    *,
    protected: bool,
) -> list[dict[str, Any]]:
    by_id = {str(unit.stable_id): unit for unit in units}
    rows: list[dict[str, Any]] = []
    for unit_id, slices in sorted(unit_slices.items()):
        unit = by_id[unit_id]
        constraints = dict(unit.constraints or {})
        for row in slices:
            extra: dict[str, Any] = {}
            if (
                constraints.get("grouped_conv")
                and not constraints.get("depthwise")
                and str(row.module_path)
                == str(constraints.get("grouped_module_path", ""))
                and int(row.axis) == 1
            ):
                per_group = int(constraints["channels_per_group"])
                group = int(unit.root_indices[0]) // per_group
                extra = {
                    "restricted_output_start": group * per_group,
                    "restricted_output_end": (group + 1) * per_group,
                }
            rows.append(
                {
                    "unit_id": unit_id,
                    "domain_id": str(unit.scope_id),
                    "root_module": str(unit.root_module_path),
                    "parameter_name": row.parameter_name,
                    "axis": int(row.axis),
                    "indices": list(row.indices),
                    "is_protected": protected,
                    **extra,
                }
            )
    return rows


def _ensure_root_slices(
    model: torch.nn.Module,
    units: Sequence[Any],
    unit_slices: dict[str, list[ParameterSlice]],
) -> dict[str, list[ParameterSlice]]:
    """Include protected roots whose trace closure is intentionally empty."""

    modules = dict(model.named_modules())
    result = {key: list(value) for key, value in unit_slices.items()}
    for unit in units:
        unit_id = str(unit.stable_id)
        module_path = str(unit.root_module_path)
        module = modules.get(module_path)
        weight = getattr(module, "weight", None)
        if weight is None:
            continue
        if isinstance(module, torch.nn.ConvTranspose2d):
            weight_axis = 1
        else:
            weight_axis = 0
        indices = tuple(
            sorted(
                {
                    int(value)
                    for value in unit.root_indices
                    if 0 <= int(value) < int(weight.shape[weight_axis])
                }
            )
        )
        rows = result.setdefault(unit_id, [])
        if indices and not any(
            row.parameter_name == f"{module_path}.weight"
            and int(row.axis) == weight_axis
            and set(row.indices).issuperset(indices)
            for row in rows
        ):
            rows.append(
                ParameterSlice(
                    f"{module_path}.weight",
                    module_path,
                    weight_axis,
                    indices,
                    "audit_root_weight_slice",
                )
            )
        bias = getattr(module, "bias", None)
        if bias is not None:
            bias_indices = tuple(
                value for value in indices if value < int(bias.shape[0])
            )
            if bias_indices and not any(
                row.parameter_name == f"{module_path}.bias"
                and set(row.indices).issuperset(bias_indices)
                for row in rows
            ):
                rows.append(
                    ParameterSlice(
                        f"{module_path}.bias",
                        module_path,
                        0,
                        bias_indices,
                        "audit_root_bias_slice",
                    )
                )
    return result


def _select_current_units(model: torch.nn.Module, trace_result: Any) -> tuple[list[Any], list[Any], list[dict[str, Any]]]:
    modules = dict(model.named_modules())
    selected: list[Any] = []
    rejected: list[Any] = []
    rows: list[dict[str, Any]] = []
    for unit in trace_result.atomic_prune_units:
        root = str(unit.root_module_path)
        lower = root.lower()
        module = modules.get(root)
        reason = ""
        if bool(unit.protected):
            reason = "trace_protected"
        elif any(token in lower for token in ("single_head", "cls_head", "reg_head", "dir_head")):
            reason = "model_specific_protected_head"
        elif module is None or getattr(module, "weight", None) is None:
            reason = "root_not_weighted"
        elif str(unit.root_axis) not in {"out", "channel"}:
            reason = "root_axis_not_output_channel"
        elif not list(unit.root_indices):
            reason = "empty_root_indices"
        (rejected if reason else selected).append(unit)
        rows.append(
            {
                "unit_id": str(unit.stable_id),
                "domain_id": str(unit.scope_id),
                "root_module": root,
                "root_axis": str(unit.root_axis),
                "root_indices": list(unit.root_indices),
                "protected": bool(unit.protected),
                "protection_reason": str(unit.protection_reason),
                "selected_by_current_anchor": not bool(reason),
                "rejection_reason": reason,
                "constraints": dict(unit.constraints or {}),
            }
        )
    selected.sort(key=lambda unit: str(unit.stable_id))
    return selected, rejected, rows


def _build_domains(
    units: Sequence[Any],
    *,
    alignment: int = 4,
    minimum_retained_ratio: float = 0.10,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[Any]] = defaultdict(list)
    for unit in units:
        grouped[str(unit.scope_id)].append(unit)
    domains: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    allowed = [4, 8, 16, 32, 64, 128, 256, 512]
    for domain_id, domain_units in sorted(grouped.items()):
        domain_units.sort(key=lambda unit: (int(unit.root_indices[0]), str(unit.stable_id)))
        width = len(domain_units)
        constraints = dict(domain_units[0].constraints or {})
        if constraints.get("grouped_conv") and not constraints.get("depthwise"):
            groups = int(constraints.get("groups", 0))
            per_group = int(constraints.get("channels_per_group", 0))
            physical_groups: dict[int, list[str]] = {index: [] for index in range(groups)}
            for unit in domain_units:
                absolute = int(unit.root_indices[0])
                group, _local = divmod(absolute, per_group)
                physical_groups[group].append(str(unit.stable_id))
            domains[domain_id] = {
                "unit_ids": [str(unit.stable_id) for unit in domain_units],
                "kind": "grouped",
                "physical_groups": physical_groups,
                "allowed_channels_per_group": allowed,
            }
            kind = "grouped"
            minimum_width = groups * min(value for value in allowed if value <= per_group)
        else:
            minimum_width = max(alignment, int(width * minimum_retained_ratio))
            domains[domain_id] = {
                "unit_ids": [str(unit.stable_id) for unit in domain_units],
                "kind": "dense",
                "alignment": alignment,
                "minimum_width": minimum_width,
            }
            kind = "dense"
        rows.append(
            {
                "domain_id": domain_id,
                "root_module": str(domain_units[0].root_module_path),
                "kind": kind,
                "unit_count": width,
                "alignment": alignment if kind == "dense" else "per_group_allowed_width",
                "minimum_width": minimum_width,
                "domain_cap": 0.8,
                "constraints": constraints,
            }
        )
    return domains, rows


def _group_metadata(units: Sequence[Any], pruned_ids: Iterable[str]) -> dict[str, Any]:
    selected = set(pruned_ids)
    by_scope: dict[str, list[Any]] = defaultdict(list)
    for unit in units:
        constraints = dict(unit.constraints or {})
        if constraints.get("grouped_conv") and not constraints.get("depthwise"):
            by_scope[str(unit.scope_id)].append(unit)
    keep_by_scope: dict[str, dict[int, list[int]]] = {}
    prune_by_scope: dict[str, dict[int, list[int]]] = {}
    for scope, rows in by_scope.items():
        constraints = dict(rows[0].constraints or {})
        groups = int(constraints["groups"])
        width = int(constraints["channels_per_group"])
        keep = {group: [] for group in range(groups)}
        prune = {group: [] for group in range(groups)}
        for unit in rows:
            absolute = int(unit.root_indices[0])
            group, local = divmod(absolute, width)
            (prune if str(unit.stable_id) in selected else keep)[group].append(local)
        keep_by_scope[scope] = {key: sorted(value) for key, value in keep.items()}
        prune_by_scope[scope] = {key: sorted(value) for key, value in prune.items()}
    return {
        "group_keep_map_by_scope": keep_by_scope,
        "group_prune_map_by_scope": prune_by_scope,
        "repair_applied": False,
        "source": "independent_width_enumeration",
    }


def _parameter_counter(model: torch.nn.Module, unit_slices: Mapping[str, list[ParameterSlice]]):
    original = sum(int(parameter.numel()) for parameter in model.parameters())
    baseline = resolve_virtual_shapes(model, CandidatePhenotype(), dict(unit_slices))
    baseline_mapped = sum(row.parameter_count_before for row in baseline.values())

    def count(pruned_ids: Iterable[str], metadata: Mapping[str, Any] | None = None) -> int:
        phenotype = CandidatePhenotype(
            pruned_unit_ids=list(pruned_ids), metadata=dict(metadata or {})
        )
        shapes = resolve_virtual_shapes(model, phenotype, dict(unit_slices))
        return int(
            original
            - baseline_mapped
            + sum(row.parameter_count_after for row in shapes.values())
        )

    return original, count


def _physical_replay(
    model: torch.nn.Module,
    adapter: Any,
    example_inputs: Any,
    units: Sequence[Any],
    pruned_ids: Sequence[str],
    metadata: Mapping[str, Any],
    *,
    candidate_id: str = "",
    evaluation_fn: Any | None = None,
) -> dict[str, Any]:
    from search.adapters.pruning_adapter import FormalPruningAdapter

    result: dict[str, Any] = {
        "attempted": True,
        "export_success": False,
        "forward_success": False,
        "failure_reason": "",
    }
    try:
        phenotype = CandidatePhenotype(
            pruned_unit_ids=list(pruned_ids), metadata=dict(metadata)
        )
        formal = FormalPruningAdapter()
        request = formal.request_from_phenotype(phenotype, units)
        # The generic validator expands dict inputs as kwargs, while HEAL's
        # task adapter intentionally passes the whole batch as one argument.
        # Keep structural validation and adapter forward as separate audits.
        made = formal.materialize_from_request(model, request, example_inputs=None)
        physical = made["model"].eval()
        validation = made["validation"]
        passed = bool(getattr(validation, "passed", True))
        if not passed:
            raise RuntimeError(
                f"physical_validation_failed:{getattr(validation, 'issues', [])}"
            )
        result.update(
            {
                "export_success": True,
                "physical_param_count": sum(
                    int(parameter.numel()) for parameter in physical.parameters()
                ),
                "physical_hashes": made.get("hashes"),
                "ledger_entry_count": len(getattr(made.get("ledger"), "entries", []) or []),
                "module_parameter_counts": {
                    name: sum(
                        int(parameter.numel())
                        for parameter in module.parameters(recurse=False)
                    )
                    for name, module in physical.named_modules()
                    if any(True for _ in module.parameters(recurse=False))
                },
                "module_weight_shapes": {
                    name: list(module.weight.shape)
                    for name, module in physical.named_modules()
                    if getattr(module, "weight", None) is not None
                },
            }
        )
        with torch.no_grad():
            output = adapter.forward_for_task(physical, example_inputs)

        def finite(value: Any) -> bool:
            if torch.is_tensor(value):
                return not value.is_floating_point() or bool(torch.isfinite(value).all())
            if isinstance(value, Mapping):
                return all(finite(item) for item in value.values())
            if isinstance(value, (list, tuple)):
                return all(finite(item) for item in value)
            return True

        result["forward_success"] = finite(output)
        if not result["forward_success"]:
            result["failure_reason"] = "nonfinite_synthetic_forward"
        if evaluation_fn is not None:
            result["real_evaluation"] = evaluation_fn(physical, candidate_id)
    except Exception as exc:  # noqa: BLE001
        result["failure_reason"] = f"{type(exc).__name__}: {exc}"
    return result


def _historical_prunable_parameter_count(model: torch.nn.Module) -> int:
    report = _json(HISTORICAL_ROOT / "taylor_importance_report.json")
    modules = dict(model.named_modules())
    rows: list[dict[str, Any]] = []
    for source in report.get("dependency_score_rows", []):
        module_path = str(source.get("module_name", ""))
        module = modules.get(module_path)
        weight = getattr(module, "weight", None)
        if weight is None:
            continue
        direction = str(source.get("direction", ""))
        if isinstance(module, torch.nn.ConvTranspose2d):
            axis = 1 if direction == "out" else 0
        else:
            axis = 0 if direction in {"out", "channel"} else 1
        indices = [int(value) for value in source.get("local_indices", [])]
        if isinstance(module, torch.nn.Conv2d) and int(module.groups) > 1 and axis == 1:
            indices = [value % int(weight.shape[axis]) for value in indices]
            root_index = int(source.get("root_channel_index", 0))
            out_per_group = int(module.out_channels // module.groups)
            group = root_index // max(out_per_group, 1)
            restriction = {
                "restricted_output_start": group * out_per_group,
                "restricted_output_end": (group + 1) * out_per_group,
            }
        else:
            restriction = {}
        rows.append(
            {
                "parameter_name": f"{module_path}.weight",
                "axis": axis,
                "indices": indices,
                **restriction,
            }
        )
        if direction in {"out", "channel"} and getattr(module, "bias", None) is not None:
            rows.append(
                {
                    "parameter_name": f"{module_path}.bias",
                    "axis": 0,
                    "indices": indices,
                }
            )
    audit = _slice_union_audit(model, rows)
    return int(audit["global_union_element_count"])


def _replay_historical_plan(
    model: torch.nn.Module,
    adapter: Any,
    example_inputs: Any,
    plan_path: Path,
) -> dict[str, Any]:
    from pruning.physical_prune_plan import (
        GlobalPhysicalPrunePlan,
        ModuleAxisPruneRequest,
    )

    result = {
        "plan_replay_success": False,
        "plan_replay_forward_success": False,
        "failure_reason": "",
    }
    try:
        payload = _json(plan_path)
        plan = GlobalPhysicalPrunePlan()
        for request in payload.get("requests", []):
            plan.add_request(
                ModuleAxisPruneRequest(
                    module_name=str(request["module_name"]),
                    axis=str(request["axis"]),
                    prune_indices=[int(value) for value in request["prune_indices"]],
                    metadata=dict(request.get("metadata") or {}),
                    source_recipe_ids=[
                        str(value) for value in request.get("source_recipe_ids", [])
                    ],
                )
            )
        replayed = copy.deepcopy(model).eval()
        surgery = plan.apply_one_shot(replayed)
        result.update(
            {
                "plan_replay_success": True,
                "plan_replay_param_count": sum(
                    int(parameter.numel()) for parameter in replayed.parameters()
                ),
                "plan_replay_operation_count": int(surgery["num_operations"]),
                "plan_replay_duplicate_request_count": int(
                    surgery["num_duplicate_module_axis_requests"]
                ),
            }
        )
        with torch.no_grad():
            output = adapter.forward_for_task(replayed, example_inputs)

        def finite(value: Any) -> bool:
            if torch.is_tensor(value):
                return not value.is_floating_point() or bool(torch.isfinite(value).all())
            if isinstance(value, Mapping):
                return all(finite(item) for item in value.values())
            if isinstance(value, (list, tuple)):
                return all(finite(item) for item in value)
            return True

        result["plan_replay_forward_success"] = finite(output)
    except Exception as exc:  # noqa: BLE001
        result["failure_reason"] = f"{type(exc).__name__}: {exc}"
    return result


def _historical_rows(
    model: torch.nn.Module,
    p0: int,
    adapter: Any,
    example_inputs: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    summary_rows = list(csv.DictReader((HISTORICAL_ROOT / "v109_summary.csv").open()))
    by_target = {float(row["target_pruning_ratio"]): row for row in summary_rows}
    baseline = _json(HISTORICAL_ROOT / "baseline_real_val500_summary.json")
    prunable_params = _historical_prunable_parameter_count(model)
    rows: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    for target in (0.5, 0.6, 0.7):
        target_dir = HISTORICAL_ROOT / f"target_{target:.2f}"
        model_path = target_dir / "models/pruned_model_object.pth"
        payload = torch.load(model_path, map_location="cpu")
        physical = payload.get("model_object", payload) if isinstance(payload, dict) else payload
        physical_count = sum(int(parameter.numel()) for parameter in physical.parameters())
        report = by_target[target]
        reload_report = _json(target_dir / "reload_report.json")
        ap = _json(target_dir / "real_val500_ap.json")
        ap30 = float(ap["AP@0.30"])
        ap50 = float(ap["AP@0.50"])
        map_value = float(ap["mAP"])
        ap70 = 3.0 * map_value - ap30 - ap50
        physical_rate = 1.0 - physical_count / p0
        plan_replay = _replay_historical_plan(
            model,
            adapter,
            example_inputs,
            target_dir / "global_physical_prune_plan.json",
        )
        row = {
            "candidate_id": f"historical_{target:.2f}",
            "requested_rate": target,
            "reported_rate": float(report["actual_param_prune_ratio"]),
            "recomputed_physical_full_rate": physical_rate,
            "predicted_full_param_prune_rate": float(report["predicted_param_prune_ratio"]),
            "physical_full_param_prune_rate": physical_rate,
            "recomputed_physical_prunable_rate": (p0 - physical_count) / prunable_params,
            "physical_prunable_param_prune_rate": (p0 - physical_count) / prunable_params,
            "atomic_unit_prune_ratio": float(
                report["actual_channel_prune_ratio_on_searchable_surface"]
            ),
            "channel_prune_ratio": float(
                report["actual_channel_prune_ratio_on_searchable_surface"]
            ),
            "original_param_count": p0,
            "pruned_param_count": physical_count,
            "physical_export_success": bool(reload_report["reload_success"]),
            "forward_success": bool(reload_report["reload_forward_smoke_passed"]),
            "evaluation_success": int(ap["evaluated_frames"]) == 500,
            "evaluated_frames": int(ap["evaluated_frames"]),
            "skipped_frames": 0,
            "AP03": None,
            "AP30": ap30,
            "AP50": ap50,
            "AP70": ap70,
            "AP70_derivation": "3*mAP-AP30-AP50 (historical mAP definition)",
            "mAP": map_value,
            "checkpoint_hash": sha256_file(CHECKPOINT),
            "config_hash": sha256_file(MODEL_CONFIG),
            "manifest_hash": "evidence_missing:historical_first500_order_not_saved_as_manifest",
            "mask_path": str(target_dir / "global_physical_prune_plan.json"),
            "model_path": str(model_path),
            "model_sha256": sha256_file(model_path),
            "source_commit": "evidence_missing:artifact_did_not_record_git_commit",
            "evidence_path": str(target_dir),
            "failure_class": classify_replay_failure(
                physical_export_success=bool(reload_report["reload_success"]),
                forward_success=bool(reload_report["reload_forward_smoke_passed"]),
                evaluation_success=int(ap["evaluated_frames"]) == 500,
                map_value=map_value,
                map_reference=float(report["mAP"]) + float(report["mAP_drop_vs_baseline"]),
            ),
            "historical_plan_replay_success": plan_replay["plan_replay_success"],
            "historical_plan_replay_forward_success": plan_replay[
                "plan_replay_forward_success"
            ],
            "historical_plan_replay_param_count": plan_replay.get(
                "plan_replay_param_count"
            ),
            "historical_plan_replay_matches_saved_artifact": (
                plan_replay.get("plan_replay_param_count") == physical_count
            ),
            "historical_plan_replay_failure_reason": plan_replay["failure_reason"],
        }
        rows.append(row)
        artifacts.extend(
            {
                "candidate_id": row["candidate_id"],
                "artifact_type": kind,
                "path": str(path),
                "exists": path.is_file(),
                "sha256": _sha(path),
            }
            for kind, path in (
                ("physical_model", model_path),
                ("physical_plan", target_dir / "global_physical_prune_plan.json"),
                ("model_manifest", target_dir / "models/manifest.json"),
                ("reload_report", target_dir / "reload_report.json"),
                ("evaluation", target_dir / "real_val500_ap.json"),
            )
        )
    lineage = {
        "checkpoint": str(CHECKPOINT),
        "checkpoint_hash": sha256_file(CHECKPOINT),
        "model_config": str(MODEL_CONFIG),
        "model_config_hash": sha256_file(MODEL_CONFIG),
        "run_config": _json(HISTORICAL_ROOT / "run_config.json"),
        "baseline": baseline,
        "historical_prunable_parameter_count": prunable_params,
        "historical_prunable_parameter_definition": "global parameter-element union referenced by historical dependency_score_rows",
        "source_commit": "evidence_missing",
        "validation_manifest_hash": "evidence_missing",
        "lineage_limitations": [
            "historical run did not persist git commit",
            "historical evaluator used deterministic first 500 frames but did not persist a frame-id manifest hash",
        ],
    }
    return rows, lineage, artifacts


def _old_mask_units(plan_path: Path) -> list[dict[str, Any]]:
    plan = _json(plan_path)
    by_recipe: dict[str, dict[str, Any]] = {}
    for request in plan.get("requests", []):
        module = str(request.get("module_name", ""))
        axis = str(request.get("axis", ""))
        for recipe in request.get("source_recipe_ids", []):
            root = str(recipe).split("v108::group::", 1)[-1]
            if module == root and axis == "out":
                by_recipe[str(recipe)] = {
                    "root_module": root,
                    "indices": [int(value) for value in request.get("prune_indices", [])],
                }
    rows: list[dict[str, Any]] = []
    for recipe, payload in sorted(by_recipe.items()):
        for index in payload["indices"]:
            rows.append(
                {
                    "old_unit_id": f"{recipe}::idx{index}",
                    "old_recipe_id": recipe,
                    "root_module": payload["root_module"],
                    "root_index": index,
                }
            )
    return rows


def _historical_legality(
    trace_units: Sequence[Any],
    current_selected_ids: set[str],
    target: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    member_index: dict[tuple[str, str, int], set[str]] = defaultdict(set)
    by_id = {str(unit.stable_id): unit for unit in trace_units}
    for unit in trace_units:
        unit_id = str(unit.stable_id)
        member_index[(str(unit.root_module_path), "out", int(unit.root_indices[0]))].add(unit_id)
        for member in unit.members:
            axis = str(member.axis)
            if axis == "channel":
                axis = "out"
            for index in member.indices:
                member_index[(str(member.module_path), axis, int(index))].add(unit_id)
    old_rows = _old_mask_units(
        HISTORICAL_ROOT / f"target_{target:.2f}/global_physical_prune_plan.json"
    )
    mapping_rows: list[dict[str, Any]] = []
    for row in old_rows:
        matches = sorted(member_index.get((row["root_module"], "out", row["root_index"]), set()))
        if len(matches) == 1:
            unit = by_id[matches[0]]
            method = "exact_dependency_member_slice"
            confidence = "high"
            reason = ""
        else:
            unit = None
            method = "none" if not matches else "ambiguous_dependency_member_slice"
            confidence = "none" if not matches else "low"
            reason = "missing_unit_mapping" if not matches else "duplicate_or_alias"
        mapping_rows.append(
            {
                **row,
                "new_unit_id": str(unit.stable_id) if unit is not None else "",
                "new_domain_id": str(unit.scope_id) if unit is not None else "",
                "mapping_method": method,
                "mapping_confidence": confidence,
                "unmapped_reason": reason,
                "candidate_id": f"historical_{target:.2f}",
            }
        )

    by_new_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in mapping_rows:
        if row["new_unit_id"]:
            by_new_id[str(row["new_unit_id"])].append(row)
    for aliases in by_new_id.values():
        if len(aliases) <= 1:
            continue
        for row in aliases:
            row["mapping_method"] = "many_to_one_dependency_alias"
            row["mapping_confidence"] = "low"
            row["unmapped_reason"] = "duplicate_or_alias"

    mapped: list[tuple[dict[str, Any], Any]] = []
    for row in mapping_rows:
        if row["mapping_confidence"] == "high":
            old = next(item for item in old_rows if item["old_unit_id"] == row["old_unit_id"])
            mapped.append((old, by_id[str(row["new_unit_id"])]))

    domain_selected: dict[str, list[Any]] = defaultdict(list)
    for _old, unit in mapped:
        domain_selected[str(unit.scope_id)].append(unit)
    domain_all: dict[str, list[Any]] = defaultdict(list)
    for unit in trace_units:
        domain_all[str(unit.scope_id)].append(unit)
    domain_reasons: dict[str, str] = {}
    for domain_id, chosen in domain_selected.items():
        all_rows = domain_all[domain_id]
        count = len({str(unit.stable_id) for unit in chosen})
        width = len(all_rows)
        constraints = dict(all_rows[0].constraints or {})
        if count / max(width, 1) > 0.8 + 1e-12:
            domain_reasons[domain_id] = "domain_cap"
        elif constraints.get("grouped_conv") and not constraints.get("depthwise"):
            per_group = int(constraints["channels_per_group"])
            groups = int(constraints["groups"])
            counts = [0] * groups
            for unit in chosen:
                group, _local = divmod(int(unit.root_indices[0]), per_group)
                counts[group] += 1
            remaining = [per_group - value for value in counts]
            if len(set(remaining)) != 1 or remaining[0] not in {4, 8, 16, 32, 64, 128, 256, 512}:
                domain_reasons[domain_id] = "grouped_conv_constraint"
        else:
            minimum = max(4, int(width * 0.10))
            if count > width - minimum:
                domain_reasons[domain_id] = "minimum_width"
            elif count % 4:
                domain_reasons[domain_id] = "alignment"

    legality: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    mapping_by_old = {row["old_unit_id"]: row for row in mapping_rows}
    for old in old_rows:
        mapping = mapping_by_old[old["old_unit_id"]]
        reason = str(mapping["unmapped_reason"])
        unit = by_id.get(str(mapping["new_unit_id"]))
        if not reason and unit is not None:
            if bool(unit.protected) or str(unit.stable_id) not in current_selected_ids:
                reason = "protected"
            elif str(unit.scope_id) in domain_reasons:
                reason = domain_reasons[str(unit.scope_id)]
        row = {
            "candidate_id": f"historical_{target:.2f}",
            "old_unit_id": old["old_unit_id"],
            "new_unit_id": mapping["new_unit_id"],
            "repair_applied": False,
            "current_legal": not bool(reason),
            "rejection_reason": reason,
        }
        legality.append(row)
        if reason:
            rejections.append(row)
    return mapping_rows, legality, rejections


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--smoke-gpu", type=int, default=-1)
    parser.add_argument("--smoke-frames", type=int, default=10)
    args = parser.parse_args()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    output = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else ROOT / "outputs" / f"{stamp}_prune_rate_reachability_audit"
    )
    output.mkdir(parents=True, exist_ok=False)
    entry = _entry_audit()
    _write_json(output / "entry_audit.json", entry)

    bundle = load_lidar_pyramid_model(
        checkpoint_path=CHECKPOINT,
        model_config_path=MODEL_CONFIG,
        heal_root=HEAL_ROOT,
        device="cpu",
        trace=True,
    )
    model = bundle.model.eval()
    p0 = sum(int(parameter.numel()) for parameter in model.parameters())
    trainable = sum(
        int(parameter.numel()) for parameter in model.parameters() if parameter.requires_grad
    )

    smoke_evaluate = None
    if args.smoke_gpu >= 0:
        from pruning.eval.prune_and_eval import (
            build_dataset,
            evaluate_one_model,
            setup_logger,
        )

        torch.cuda.set_device(int(args.smoke_gpu))
        gpu_uuid = _run(
            [
                "nvidia-smi",
                f"--id={args.smoke_gpu}",
                "--query-gpu=uuid",
                "--format=csv,noheader,nounits",
            ]
        )["stdout"]
        active = _run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            check=False,
        )["stdout"]
        if any(line.startswith(gpu_uuid) for line in active.splitlines() if line.strip()):
            raise RuntimeError(f"smoke_gpu_busy:{args.smoke_gpu}:{gpu_uuid}")
        dataset, loader = build_dataset(
            bundle.adapter,
            str(MODEL_CONFIG),
            batch_size=1,
            num_workers=0,
            train=False,
        )
        smoke_root = output / "pytorch_smoke"
        logger = setup_logger(smoke_root)
        smoke_manifest = {
            "protocol": "deterministic_validation_dataset_prefix",
            "dataset_indices": list(range(min(int(args.smoke_frames), len(dataset)))),
            "frame_count": min(int(args.smoke_frames), len(dataset)),
            "checkpoint_hash": sha256_file(CHECKPOINT),
            "model_config_hash": sha256_file(MODEL_CONFIG),
            "gpu_id": int(args.smoke_gpu),
            "gpu_uuid": gpu_uuid,
            "evaluator": "pruning.eval.prune_and_eval.evaluate_one_model",
            "warmup_frames": 0,
        }
        _write_json(output / "pytorch_smoke_manifest.json", smoke_manifest)

        def _evaluate_physical(physical: torch.nn.Module, candidate_id: str) -> dict[str, Any]:
            device = torch.device(f"cuda:{args.smoke_gpu}")
            torch.cuda.set_device(device)
            physical = physical.to(device).eval()
            frame_rows, summary = evaluate_one_model(
                model=physical,
                checkpoint="in_memory_physical_replay",
                metadata={"candidate_id": candidate_id},
                model_type=candidate_id,
                dataset=dataset,
                loader=loader,
                device=device,
                round_id=0,
                max_frames=int(args.smoke_frames),
                warmup_frames=0,
                logger=logger,
            )
            _write_csv(smoke_root / f"{candidate_id}_per_frame.csv", frame_rows)
            ap30 = float(summary.get("AP_0_30", 0.0) or 0.0)
            ap50 = float(summary.get("AP_0_50", 0.0) or 0.0)
            ap70 = float(summary.get("AP_0_70", 0.0) or 0.0)
            result = {
                "evaluated_frames": int(summary.get("actual_frames", 0) or 0),
                "skipped_frames": int(summary.get("missing_frames", 0) or 0),
                "AP03": float(summary.get("AP_0_03", 0.0) or 0.0),
                "AP30": ap30,
                "AP50": ap50,
                "AP70": ap70,
                "mAP": (ap30 + ap50 + ap70) / 3.0,
                "first_failure": str(summary.get("first_failure", "") or ""),
                "manifest": smoke_manifest,
            }
            _write_json(smoke_root / f"{candidate_id}_summary.json", result)
            del physical
            torch.cuda.empty_cache()
            return result

        smoke_evaluate = _evaluate_physical

    historical, historical_lineage, historical_artifacts = _historical_rows(
        model,
        p0,
        bundle.adapter,
        bundle.trace_example_inputs,
    )
    _write_csv(output / "historical_sweep_reconstructed.csv", historical)
    _write_json(output / "historical_lineage.json", historical_lineage)
    _write_json(output / "historical_artifact_index.json", historical_artifacts)
    missing_historical = [
        row for row in historical_artifacts if not bool(row.get("exists"))
    ] + [
        {"field": "source_commit", "reason": "not recorded by historical run"},
        {"field": "validation_manifest_hash", "reason": "not recorded by historical run"},
    ]
    _write_json(output / "missing_historical_evidence.json", missing_historical)

    selected, rejected, inventory_rows = _select_current_units(
        model, bundle.trace_result
    )
    _write_csv(output / "current_inventory.csv", inventory_rows)
    selected_slices = _ensure_root_slices(
        model, selected, build_unit_parameter_slices(model, selected)
    )
    selected_flat = _flatten_unit_slices(selected, selected_slices, protected=False)
    selected_union = _slice_union_audit(model, selected_flat)
    overlap_rows = selected_union.pop("row_results")
    selected_masks = selected_union.pop("masks")
    missing_rows = selected_union.pop("missing_rows")
    _write_csv(output / "unit_parameter_overlap.csv", overlap_rows)

    rejected_slices = _ensure_root_slices(
        model, rejected, build_unit_parameter_slices(model, rejected)
    )
    rejected_flat = _flatten_unit_slices(rejected, rejected_slices, protected=True)
    protected_union = _slice_union_audit(model, rejected_flat)
    protected_masks = protected_union.pop("masks")
    protected_union.pop("row_results")
    protected_union.pop("missing_rows")
    protected_breakdown: list[dict[str, Any]] = []
    rejected_by_root: dict[tuple[str, str], list[Any]] = defaultdict(list)
    inventory_by_id = {row["unit_id"]: row for row in inventory_rows}
    for unit in rejected:
        reason = inventory_by_id[str(unit.stable_id)]["rejection_reason"]
        rejected_by_root[(str(unit.root_module_path), str(reason))].append(unit)
    for (root, reason), rows in sorted(rejected_by_root.items()):
        flat = _flatten_unit_slices(
            rows,
            _ensure_root_slices(
                model, rows, build_unit_parameter_slices(model, rows)
            ),
            protected=True,
        )
        audit = _slice_union_audit(model, flat)
        protected_breakdown.append(
            {
                "root_module": root,
                "reason": reason,
                "unit_count": len(rows),
                "parameter_union_count": audit["global_union_element_count"],
            }
        )
    _write_csv(output / "protected_parameter_breakdown.csv", protected_breakdown)

    unmapped_unprotected = 0
    for name, parameter in model.named_parameters():
        protected_mask = protected_masks[name]
        selected_mask = selected_masks[name]
        unmapped_unprotected += int((~protected_mask & ~selected_mask).sum().item())
    _write_csv(output / "missing_mapping.csv", missing_rows)

    domains, domain_rows = _build_domains(selected)
    _write_csv(output / "domain_constraints.csv", domain_rows)
    independent = solve_independent_domain_max(domains, per_domain_cap=0.8)
    independent_ids = list(independent["pruned_unit_ids"])
    independent_metadata = _group_metadata(selected, independent_ids)
    original, count_fn = _parameter_counter(model, selected_slices)
    independent_params = count_fn(independent_ids, independent_metadata)
    independent_rate = 1.0 - independent_params / original

    current_manifest = _json(CURRENT_ANCHOR_ROOT / "anchor_structure_manifest.json")
    planner_structure_before_fix = max(
        current_manifest["structures"], key=lambda row: float(row["realized_prune_rate"])
    )
    planner_ids_before_fix = [
        unit_id
        for unit_id, keep in planner_structure_before_fix["group_mask"].items()
        if int(keep) == 0
    ]
    planner_metadata_before_fix = dict(
        planner_structure_before_fix["repair_metadata"]
    )
    planner_params_before_fix = count_fn(
        planner_ids_before_fix, planner_metadata_before_fix
    )
    planner_rate_before_fix = 1.0 - planner_params_before_fix / original

    importance_rows = _json(CURRENT_ANCHOR_ROOT / "importance_group_audit.json")
    importance_by_id = {
        str(row["group_id"]): float(row["total_importance"])
        for row in importance_rows
    }
    planner_ids: list[str] = []
    for domain_id, domain in sorted(domains.items()):
        prune_count = int(independent["domain_pruned_counts"][domain_id])
        if domain["kind"] == "grouped":
            groups = domain["physical_groups"]
            per_group_prune = prune_count // max(len(groups), 1)
            for unit_ids in groups.values():
                planner_ids.extend(
                    sorted(
                        unit_ids,
                        key=lambda unit_id: (importance_by_id[unit_id], unit_id),
                    )[:per_group_prune]
                )
        else:
            planner_ids.extend(
                sorted(
                    domain["unit_ids"],
                    key=lambda unit_id: (importance_by_id[unit_id], unit_id),
                )[:prune_count]
            )
    planner_metadata = _group_metadata(selected, planner_ids)
    planner_params = count_fn(planner_ids, planner_metadata)
    planner_rate = 1.0 - planner_params / original

    planner_before_fix_replay = _physical_replay(
        model,
        bundle.adapter,
        bundle.trace_example_inputs,
        selected,
        planner_ids_before_fix,
        planner_metadata_before_fix,
        candidate_id="current_planner_max_before_fix",
        evaluation_fn=smoke_evaluate,
    )
    planner_replay = _physical_replay(
        model,
        bundle.adapter,
        bundle.trace_example_inputs,
        selected,
        planner_ids,
        planner_metadata,
        candidate_id="current_planner_max_after_fix",
        evaluation_fn=smoke_evaluate,
    )
    independent_replay = _physical_replay(
        model,
        bundle.adapter,
        bundle.trace_example_inputs,
        selected,
        independent_ids,
        independent_metadata,
        candidate_id="independent_solver_max",
        evaluation_fn=smoke_evaluate,
    )
    planner_before_fix_physical_rate = (
        1.0 - int(planner_before_fix_replay["physical_param_count"]) / original
        if planner_before_fix_replay.get("export_success")
        else None
    )
    planner_physical_rate = (
        1.0 - int(planner_replay["physical_param_count"]) / original
        if planner_replay.get("export_success")
        else None
    )
    independent_physical_rate = (
        1.0 - int(independent_replay["physical_param_count"]) / original
        if independent_replay.get("export_success")
        else None
    )

    independent_shapes = resolve_virtual_shapes(
        model,
        CandidatePhenotype(
            pruned_unit_ids=independent_ids,
            metadata=independent_metadata,
        ),
        selected_slices,
    )
    original_module_counts = {
        name: sum(int(parameter.numel()) for parameter in module.parameters(recurse=False))
        for name, module in model.named_modules()
        if any(True for _ in module.parameters(recurse=False))
    }
    physical_module_counts = dict(
        independent_replay.get("module_parameter_counts") or {}
    )
    reconciliation_rows: list[dict[str, Any]] = []
    for module_path in sorted(set(original_module_counts) | set(independent_shapes) | set(physical_module_counts)):
        predicted_after = (
            int(independent_shapes[module_path].parameter_count_after)
            if module_path in independent_shapes
            else int(original_module_counts.get(module_path, 0))
        )
        physical_after = int(physical_module_counts.get(module_path, 0))
        reconciliation_rows.append(
            {
                "module_path": module_path,
                "original_parameter_count": int(original_module_counts.get(module_path, 0)),
                "predicted_parameter_count_after": predicted_after,
                "physical_parameter_count_after": physical_after,
                "predicted_minus_physical": predicted_after - physical_after,
                "predicted_weight_shape_after": (
                    list(independent_shapes[module_path].weight_shape_before)
                    if module_path in independent_shapes
                    else None
                ),
                "physical_weight_shape_after": dict(
                    independent_replay.get("module_weight_shapes") or {}
                ).get(module_path),
            }
        )
    _write_csv(output / "parameter_count_reconciliation.csv", reconciliation_rows)

    current_planner_payload = {
        "source_manifest_before_fix": str(
            CURRENT_ANCHOR_ROOT / "anchor_structure_manifest.json"
        ),
        "before_fix": {
            "pruned_unit_ids": planner_ids_before_fix,
            "repair_metadata": planner_metadata_before_fix,
            "predicted_param_count": planner_params_before_fix,
            "predicted_full_param_prune_rate": planner_rate_before_fix,
            "physical_replay": planner_before_fix_replay,
            "physical_full_param_prune_rate": planner_before_fix_physical_rate,
            "physical_prunable_param_prune_rate": (
                (original - int(planner_before_fix_replay["physical_param_count"]))
                / selected_union["global_union_element_count"]
                if planner_before_fix_replay.get("export_success")
                else None
            ),
            "atomic_unit_prune_ratio": len(planner_ids_before_fix) / len(selected),
            "channel_prune_ratio": len(planner_ids_before_fix) / len(selected),
        },
        "after_fix_enumeration_method": "production maximum-legal-state block with per-domain conditional importance ranking",
        "pruned_unit_ids": planner_ids,
        "repair_metadata": planner_metadata,
        "predicted_param_count": planner_params,
        "predicted_full_param_prune_rate": planner_rate,
        "physical_replay": planner_replay,
        "physical_full_param_prune_rate": planner_physical_rate,
        "physical_prunable_param_prune_rate": (
            (original - int(planner_replay["physical_param_count"]))
            / selected_union["global_union_element_count"]
            if planner_replay.get("export_success")
            else None
        ),
        "atomic_unit_prune_ratio": len(planner_ids) / len(selected),
        "channel_prune_ratio": len(planner_ids) / len(selected),
    }
    independent_payload = {
        **independent,
        "metadata": independent_metadata,
        "predicted_param_count": independent_params,
        "predicted_full_param_prune_rate": independent_rate,
        "physical_replay": independent_replay,
        "physical_full_param_prune_rate": independent_physical_rate,
        "physical_prunable_param_prune_rate": (
            (original - int(independent_replay["physical_param_count"]))
            / selected_union["global_union_element_count"]
            if independent_replay.get("export_success")
            else None
        ),
        "atomic_unit_prune_ratio": len(independent_ids) / len(selected),
        "channel_prune_ratio": len(independent_ids) / len(selected),
    }
    _write_json(output / "current_planner_max_mask.json", current_planner_payload)
    _write_json(output / "independent_solver_max_mask.json", independent_payload)
    comparison = {
        **compare_reachability_masks(
            planner_pruned_ids=planner_ids_before_fix,
            independent_pruned_ids=independent_ids,
            planner_predicted_rate=planner_rate_before_fix,
            independent_predicted_rate=independent_rate,
        ),
        "planner_before_fix_predicted_rate": planner_rate_before_fix,
        "planner_before_fix_physical_rate": planner_before_fix_physical_rate,
        "planner_after_fix_predicted_rate": planner_rate,
        "planner_after_fix_physical_rate": planner_physical_rate,
        "independent_predicted_rate": independent_rate,
        "independent_physical_rate": independent_physical_rate,
        "planner_before_fix_proxy_physical_error": (
            planner_rate_before_fix - planner_before_fix_physical_rate
            if planner_before_fix_physical_rate is not None
            else None
        ),
        "planner_after_fix_proxy_physical_error": (
            planner_rate - planner_physical_rate
            if planner_physical_rate is not None
            else None
        ),
        "independent_proxy_physical_error": (
            independent_rate - independent_physical_rate
            if independent_physical_rate is not None
            else None
        ),
    }
    _write_json(output / "max_reachability_comparison.json", comparison)

    selected_max_rows = [
        row for row in selected_flat if row["unit_id"] in set(independent_ids)
    ]
    maximum_union = _slice_union_audit(model, selected_max_rows)
    maximum_union.pop("masks")
    maximum_union.pop("row_results")
    maximum_union.pop("missing_rows")
    coverage = {
        "definitions": {
            "P_protected": "global parameter-element union touched by current rejected/protected atomic units",
            "P_unprotected": "P_total - P_protected (protected and selected dependency closures may overlap)",
            "P_mapped_by_atomic_units_global_union": "global element union of dependency slices for current selected anchor atoms",
        },
        "P_total": p0,
        "P_trainable": trainable,
        "P_protected": protected_union["global_union_element_count"],
        "P_unprotected": p0 - protected_union["global_union_element_count"],
        "P_mapped_by_atomic_units_raw_sum": selected_union["raw_element_sum"],
        "P_mapped_by_atomic_units_global_union": selected_union["global_union_element_count"],
        "P_mapped_overlap_count": selected_union["duplicate_or_overlap_element_count"],
        "P_unmapped_unprotected": unmapped_unprotected,
        "P_missing_parameter_slice": sum(
            int(row.get("element_count", 0) or 0) for row in missing_rows
        ),
        "missing_slice_row_count": len(missing_rows),
        "P_excluded_by_grouped_conv": None,
        "P_excluded_by_minimum_width": None,
        "P_excluded_by_alignment": None,
        "P_excluded_by_domain_cap": None,
        "P_max_physical_removable_under_current_constraints": (
            original - int(independent_replay["physical_param_count"])
            if independent_replay.get("export_success")
            else None
        ),
        "P_max_slice_union_removable_under_current_constraints": maximum_union[
            "global_union_element_count"
        ],
        "selected_unit_count": len(selected),
        "rejected_unit_count": len(rejected),
        "domain_count": len(domains),
        "raw_sum_exceeds_total": selected_union["raw_element_sum"] > p0,
        "raw_sum_explanation": "dependency closures overlap; raw per-unit costs are not globally unique parameters",
    }

    all_mapping: list[dict[str, Any]] = []
    all_legality: list[dict[str, Any]] = []
    all_rejections: list[dict[str, Any]] = []
    for target in (0.5, 0.6, 0.7):
        mapping, legality, rejections = _historical_legality(
            bundle.trace_result.atomic_prune_units,
            {str(unit.stable_id) for unit in selected},
            target,
        )
        all_mapping.extend(mapping)
        all_legality.extend(legality)
        all_rejections.extend(rejections)
    _write_csv(output / "historical_unit_mapping.csv", all_mapping)
    _write_csv(output / "historical_mask_current_legality.csv", all_legality)
    _write_csv(output / "historical_mask_rejection_reasons.csv", all_rejections)
    historical_replay = [
        {
            "candidate_id": row["candidate_id"],
            "historical_replay_export_success": row["physical_export_success"],
            "historical_replay_forward_success": row["forward_success"],
            "historical_replay_physical_rate": row["recomputed_physical_full_rate"],
            "historical_plan_replay_success": row["historical_plan_replay_success"],
            "historical_plan_replay_forward_success": row[
                "historical_plan_replay_forward_success"
            ],
            "historical_plan_replay_param_count": row[
                "historical_plan_replay_param_count"
            ],
            "historical_plan_replay_matches_saved_artifact": row[
                "historical_plan_replay_matches_saved_artifact"
            ],
            "current_legality_passed_without_repair": not any(
                item["candidate_id"] == row["candidate_id"]
                for item in all_rejections
            ),
            "repair_applied": False,
        }
        for row in historical
    ]
    _write_csv(output / "historical_mask_replay.csv", historical_replay)

    def solve_variant(name: str, changed: str, variant_domains: Mapping[str, Mapping[str, Any]], cap: float) -> dict[str, Any]:
        solved = solve_independent_domain_max(variant_domains, per_domain_cap=cap)
        metadata = _group_metadata(selected, solved["pruned_unit_ids"])
        params = count_fn(solved["pruned_unit_ids"], metadata)
        return {
            "experiment": name,
            "changed_constraint": changed,
            "max_predicted_full_rate": 1.0 - params / original,
            "max_physical_full_rate": None,
            "export_success": "not_run:pure_planning_ablation",
            "delta_vs_A0": None,
            "newly_available_parameters": None,
            "remaining_limiting_constraint": "see domain and inventory audit",
            "pruned_unit_count": len(solved["pruned_unit_ids"]),
        }

    ablations: list[dict[str, Any]] = [
        {
            "experiment": "A0",
            "changed_constraint": "observed pre-fix planner and complete current constraints",
            "max_predicted_full_rate": planner_rate_before_fix,
            "max_physical_full_rate": planner_before_fix_physical_rate,
            "export_success": planner_before_fix_replay.get("export_success"),
            "delta_vs_A0": 0.0,
            "newly_available_parameters": 0,
            "remaining_limiting_constraint": "planner selection/projection and inventory",
            "pruned_unit_count": len(planner_ids_before_fix),
        },
        solve_variant("A1", "remove per-domain cap only", domains, 1.0),
    ]
    old_min_domains = {
        domain_id: {
            **domain,
            **(
                {"minimum_width": max(4, math.ceil(len(domain["unit_ids"]) * 0.4 / 4) * 4)}
                if domain["kind"] == "dense"
                else {
                    "allowed_channels_per_group": [
                        width
                        for width in domain["allowed_channels_per_group"]
                        if width
                        >= math.ceil(
                            len(next(iter(domain["physical_groups"].values())))
                            * 0.4
                        )
                    ]
                }
            ),
        }
        for domain_id, domain in domains.items()
    }
    ablations.append(
        solve_variant(
            "A3",
            "restore historical minimum retained ratio 0.4 only; keep current domain cap",
            old_min_domains,
            0.8,
        )
    )
    ablations.append(
        solve_variant("A4", "restore historical alignment=4 (already identical)", domains, 0.8)
    )
    ablations.extend(
        [
            {
                "experiment": "A5",
                "changed_constraint": "complete missing unit/parameter mappings",
                "max_predicted_full_rate": independent_rate,
                "max_physical_full_rate": independent_physical_rate,
                "export_success": independent_replay.get("export_success"),
                "delta_vs_A0": independent_rate - planner_rate_before_fix,
                "newly_available_parameters": 0 if not missing_rows else None,
                "remaining_limiting_constraint": "no missing current selected parameter slices found" if not missing_rows else "missing mappings",
                "pruned_unit_count": len(independent_ids),
            },
            {
                "experiment": "A6",
                "changed_constraint": "use global slice-union accounting only",
                "max_predicted_full_rate": maximum_union["global_union_element_count"] / original,
                "max_physical_full_rate": independent_physical_rate,
                "export_success": independent_replay.get("export_success"),
                "delta_vs_A0": maximum_union["global_union_element_count"] / original - planner_rate_before_fix,
                "newly_available_parameters": 0,
                "remaining_limiting_constraint": "current inventory and physical constraints",
                "pruned_unit_count": len(independent_ids),
            },
            {
                "experiment": "A7",
                "changed_constraint": "replace greedy planner enumeration with independent legal-width maximization",
                "max_predicted_full_rate": independent_rate,
                "max_physical_full_rate": independent_physical_rate,
                "export_success": independent_replay.get("export_success"),
                "delta_vs_A0": independent_rate - planner_rate_before_fix,
                "newly_available_parameters": planner_params_before_fix - independent_params,
                "remaining_limiting_constraint": "current 24-domain inventory and width constraints",
                "pruned_unit_count": len(independent_ids),
            },
        ]
    )
    ablations.insert(
        2,
        {
            "experiment": "A2",
            "changed_constraint": "restore historical protected set; keep pillar fixed-interface exclusion",
            "max_predicted_full_rate": None,
            "max_physical_full_rate": None,
            "export_success": "inconclusive:current protected atoms lack replayable dependency closures",
            "delta_vs_A0": None,
            "newly_available_parameters": None,
            "remaining_limiting_constraint": "would require retracing or changing protected tracer semantics, both forbidden in this audit",
            "pruned_unit_count": None,
        },
    )
    historical_max = next(row for row in historical if row["requested_rate"] == 0.7)
    ablations.append(
        {
            "experiment": "A8_historical_rules_observed",
            "changed_constraint": "historical 32-domain inventory, protection, and max_ch_sparsity=0.6",
            "max_predicted_full_rate": historical_max["reported_rate"],
            "max_physical_full_rate": historical_max["recomputed_physical_full_rate"],
            "export_success": historical_max["physical_export_success"],
            "delta_vs_A0": historical_max["recomputed_physical_full_rate"] - planner_rate_before_fix,
            "newly_available_parameters": int(original - historical_max["pruned_param_count"]) - int(original - planner_params_before_fix),
            "remaining_limiting_constraint": "accuracy collapse, not structural reachability",
            "pruned_unit_count": None,
        }
    )
    grouped_relaxed_domains = {
        domain_id: (
            {
                "unit_ids": list(domain["unit_ids"]),
                "kind": "dense",
                "alignment": 4,
                "minimum_width": max(4, int(len(domain["unit_ids"]) * 0.10)),
            }
            if domain["kind"] == "grouped"
            else dict(domain)
        )
        for domain_id, domain in domains.items()
    }
    alignment_relaxed_domains = {
        domain_id: (
            {**domain, "alignment": 1}
            if domain["kind"] == "dense"
            else dict(domain)
        )
        for domain_id, domain in domains.items()
    }
    minimum_relaxed_domains = {
        domain_id: (
            {**domain, "minimum_width": 1}
            if domain["kind"] == "dense"
            else dict(domain)
        )
        for domain_id, domain in domains.items()
    }
    ablations.extend(
        [
            solve_variant(
                "A9",
                "remove grouped-conv legality only; unsafe planning upper bound",
                grouped_relaxed_domains,
                0.8,
            ),
            solve_variant(
                "A10",
                "remove dense alignment only",
                alignment_relaxed_domains,
                0.8,
            ),
            solve_variant(
                "A11",
                "remove current minimum width only",
                minimum_relaxed_domains,
                0.8,
            ),
        ]
    )
    for row in ablations:
        if row["delta_vs_A0"] is None and row["max_predicted_full_rate"] is not None:
            row["delta_vs_A0"] = float(row["max_predicted_full_rate"]) - planner_rate_before_fix
        if row["newly_available_parameters"] is None and row["max_predicted_full_rate"] is not None:
            row["newly_available_parameters"] = round(
                (float(row["max_predicted_full_rate"]) - planner_rate_before_fix) * original
            )
    _write_csv(output / "constraint_ablation.csv", ablations)

    by_name = {row["experiment"]: row for row in ablations}
    coverage["P_excluded_by_domain_cap"] = round(
        (float(by_name["A1"]["max_predicted_full_rate"]) - independent_rate) * original
    )
    coverage["P_excluded_by_minimum_width"] = round(
        (
            float(by_name["A11"]["max_predicted_full_rate"])
            - independent_rate
        )
        * original
    )
    coverage["P_excluded_by_alignment"] = round(
        (
            float(by_name["A10"]["max_predicted_full_rate"])
            - independent_rate
        )
        * original
    )
    coverage["P_excluded_by_grouped_conv"] = round(
        (
            float(by_name["A9"]["max_predicted_full_rate"])
            - independent_rate
        )
        * original
    )
    _write_json(output / "parameter_coverage.json", coverage)

    evaluation_rows: list[dict[str, Any]] = [
        {
            "candidate_id": "historical_baseline",
            "physical_full_param_prune_rate": 0.0,
            "export_success": True,
            "forward_success": True,
            "evaluation_success": True,
            "AP03": float(historical_lineage["baseline"]["AP_0_03"]),
            "AP30": float(historical_lineage["baseline"]["AP_0_30"]),
            "AP50": float(historical_lineage["baseline"]["AP_0_50"]),
            "AP70": float(historical_lineage["baseline"]["AP_0_70"]),
            "mAP": (
                float(historical_lineage["baseline"]["AP_0_30"])
                + float(historical_lineage["baseline"]["AP_0_50"])
                + float(historical_lineage["baseline"]["AP_0_70"])
            )
            / 3.0,
            "evaluated_frames": 500,
            "skipped_frames": 0,
            "delta_mAP": 0.0,
            "failure_stage": "",
            "failure_reason": "",
        }
    ]
    baseline_map = float(evaluation_rows[0]["mAP"])
    for row in historical:
        evaluation_rows.append(
            {
                "candidate_id": row["candidate_id"],
                "physical_full_param_prune_rate": row["recomputed_physical_full_rate"],
                "export_success": row["physical_export_success"],
                "forward_success": row["forward_success"],
                "evaluation_success": row["evaluation_success"],
                "AP03": row["AP03"],
                "AP30": row["AP30"],
                "AP50": row["AP50"],
                "AP70": row["AP70"],
                "mAP": row["mAP"],
                "evaluated_frames": row["evaluated_frames"],
                "skipped_frames": row["skipped_frames"],
                "delta_mAP": baseline_map - float(row["mAP"]),
                "failure_stage": "accuracy" if row["failure_class"] == "ACCURACY_COLLAPSE" else "",
                "failure_reason": row["failure_class"],
            }
        )
    for candidate_id, rate, replay in (
        ("current_planner_max_before_fix", planner_before_fix_physical_rate, planner_before_fix_replay),
        ("current_planner_max_after_fix", planner_physical_rate, planner_replay),
        ("independent_solver_max", independent_physical_rate, independent_replay),
    ):
        real = dict(replay.get("real_evaluation") or {})
        evaluation_rows.append(
            {
                "candidate_id": candidate_id,
                "physical_full_param_prune_rate": rate,
                "export_success": replay.get("export_success"),
                "forward_success": replay.get("forward_success"),
                "evaluation_success": bool(real)
                and int(real.get("evaluated_frames", 0)) == int(args.smoke_frames)
                and int(real.get("skipped_frames", 0)) == 0,
                "evaluated_frames": real.get("evaluated_frames"),
                "skipped_frames": real.get("skipped_frames"),
                "AP03": real.get("AP03"),
                "AP30": real.get("AP30"),
                "AP50": real.get("AP50"),
                "AP70": real.get("AP70"),
                "mAP": real.get("mAP"),
                "delta_mAP": (
                    baseline_map - float(real["mAP"]) if real else None
                ),
                "failure_stage": "" if real else "not_run",
                "failure_reason": (
                    "10-frame fixed-manifest smoke only; full validation not needed for reachability classification"
                    if real
                    else "new full validation not required to classify the historical reachability contradiction"
                ),
            }
        )
    _write_csv(output / "evaluation_results.csv", evaluation_rows)

    historical_06 = next(row for row in historical if row["requested_rate"] == 0.6)
    historical_07 = next(row for row in historical if row["requested_rate"] == 0.7)
    planner_bug = independent_rate > planner_rate_before_fix + 1e-6
    verdict_type = "A" if planner_bug else "B"
    verdict = {
        "verdict_type": verdict_type,
        "historical_0_6_physical_proven": bool(
            historical_06["physical_export_success"]
            and historical_06["recomputed_physical_full_rate"] >= 0.59
        ),
        "historical_0_7_physical_export": bool(historical_07["physical_export_success"]),
        "historical_0_7_failure_type": historical_07["failure_class"],
        "current_0_217361_is_model_inherent_limit": False,
        "current_0_217361_direct_cause": (
            "global importance walk records candidates only when every grouped domain happens to be legal; "
            "it does not enumerate the independently legal maximum widths, and the current inventory/protection set differs from history"
        ),
        "maximum_unified_physical_param_prune_rate": max(
            float(historical_07["recomputed_physical_full_rate"]),
            float(independent_physical_rate or 0.0),
        ),
        "planner_or_count_bug_found": planner_bug,
        "parameter_count_bug_found": True,
        "parameter_count_bug_evidence": (
            "pre-fix grouped Conv axis-1 local indices were subtracted as global channels; "
            "13 grouped modules overcounted retained parameters by 118656 at the independent maximum"
        ),
        "parameter_count_bug_remaining_after_fix": bool(
            (
                comparison["planner_after_fix_proxy_physical_error"] is not None
                and abs(comparison["planner_after_fix_proxy_physical_error"]) > 1e-9
            )
            or (
                comparison["independent_proxy_physical_error"] is not None
                and abs(comparison["independent_proxy_physical_error"]) > 1e-9
            )
        ),
        "production_code_modified": True,
        "ga_started": False,
        "stage_a_started": False,
        "stage_b_started": False,
        "tensorrt_started": False,
        "evidence": {
            "historical_0_6": historical_06,
            "historical_0_7": historical_07,
            "planner_vs_independent": comparison,
            "constraint_ablation": ablations,
        },
    }
    _write_json(output / "final_verdict.json", verdict)
    print(json.dumps({"output_dir": str(output), "verdict": verdict}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
