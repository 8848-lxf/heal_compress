"""Runtime helpers that delegate closure and materialization to formal APIs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from pruning.api import select_pruning_request
from pruning.config import (
    AlignmentConfig,
    GroupedConvConfig,
    GroupedConvSelectionPolicy,
    SelectionConfig,
)
from pruning.types import ImportanceResult, SamplingPruningRequest

from .contracts import filter_active_root_units
from .torch_pruning_strategy import torch_pruning_l2_channel_scores


def _stable_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_frame_manifest(
    frame_ids: Sequence[str],
    *,
    frame_count: int,
    dataset_config_hash: str,
    checkpoint_hash: str,
    evaluation_code_hash: str,
    split: str = "validation",
    seed: int = 0,
) -> dict[str, Any]:
    available = [str(value) for value in frame_ids]
    count = int(frame_count)
    if count <= 0 or len(available) < count:
        raise ValueError(f"requested {count} frames from split containing {len(available)}")
    selected = available[:count]
    return {
        "manifest_path": "validation_frame_manifest_500.json" if count == 500 else "",
        "frame_ids": selected,
        "frame_count": count,
        "frame_list_hash": _stable_hash(selected),
        "dataset_config_hash": str(dataset_config_hash),
        "checkpoint_hash": str(checkpoint_hash),
        "evaluation_code_hash": str(evaluation_code_hash),
        "split": str(split),
        "selection_policy": "ordered_split_prefix_v1",
        "random_seed": int(seed),
        "skipped_frame_policy": "fail_closed_no_replacement",
        "schema_version": "validation-frame-manifest-v2",
    }


def state_dict_content_hash(state_dict: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        value = state_dict[key]
        digest.update(key.encode("utf-8"))
        if torch.is_tensor(value):
            tensor = value.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        else:
            digest.update(json.dumps(value, sort_keys=True, default=str).encode("utf-8"))
    return digest.hexdigest()


def validate_core_freeze(
    before: dict[str, Any],
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root)
    after_rows: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    for row in before.get("frozen_files", []):
        relative = str(row["path"])
        path = root / relative
        current = _file_sha256(path) if path.is_file() else None
        after = {"path": relative, "exists": path.is_file(), "sha256": current}
        after_rows.append(after)
        if bool(row.get("exists")) != path.is_file() or row.get("sha256") != current:
            mismatches.append(
                {
                    "path": relative,
                    "before_exists": bool(row.get("exists")),
                    "after_exists": path.is_file(),
                    "before_sha256": row.get("sha256"),
                    "after_sha256": current,
                }
            )
    unchanged = not mismatches
    return {
        "core_files_unchanged": unchanged,
        "status": "valid" if unchanged else "invalid_due_to_core_file_modification",
        "frozen_files": after_rows,
        "mismatches": mismatches,
        "schema_version": "grouped-conv-core-freeze-validation-v2",
    }


def build_tp_importance_result(
    model: torch.nn.Module,
    atomic_units: Sequence[Any],
    *,
    allowed_roots: set[str],
) -> dict[str, Any]:
    filtered = filter_active_root_units(atomic_units, allowed_roots)
    roots_found = {str(unit.root_module_path) for unit in filtered}
    missing = sorted(set(allowed_roots) - roots_found)
    if missing:
        raise ValueError(f"missing grouped root units for TP importance: {missing}")
    modules = dict(model.named_modules())
    atoms_by_root: dict[str, list[Any]] = {}
    for unit in filtered:
        atoms_by_root.setdefault(str(unit.root_module_path), []).append(unit)
    raw: dict[str, float] = {}
    mapping_rows: list[dict[str, Any]] = []
    module_reports: list[dict[str, Any]] = []
    for root in sorted(atoms_by_root):
        module = modules.get(root)
        if module is None:
            raise ValueError(f"TP importance root module is missing: {root}")
        report = torch_pruning_l2_channel_scores(module, module_path=root)
        module_reports.append(report)
        scores = [float(value) for value in report["scores"]]
        seen_indices: set[int] = set()
        for atom in atoms_by_root[root]:
            indices = list(atom.root_indices)
            if len(indices) != 1:
                raise ValueError(f"TP grouped atomic unit must own one root channel: {atom.stable_id}")
            absolute = int(indices[0])
            if absolute in seen_indices or not 0 <= absolute < len(scores):
                raise ValueError(f"invalid TP root channel index for {root}: {absolute}")
            seen_indices.add(absolute)
            for source_id in atom.source_coupled_unit_ids:
                raw[str(source_id)] = scores[absolute]
            constraints = dict(atom.constraints)
            width = int(constraints.get("channels_per_group") or 0)
            group_id, local_position = divmod(absolute, width)
            mapping_rows.append(
                {
                    "module_path": root,
                    "scope_id": str(atom.scope_id),
                    "atomic_unit_id": str(atom.stable_id),
                    "source_coupled_unit_ids": list(atom.source_coupled_unit_ids),
                    "absolute_channel_index": absolute,
                    "group_id": group_id,
                    "local_position": local_position,
                    "raw_tp_l2_score": scores[absolute],
                }
            )
        if seen_indices != set(range(len(scores))):
            raise ValueError(
                f"TP importance did not map every root channel for {root}: "
                f"mapped={len(seen_indices)}, expected={len(scores)}"
            )
    importance = ImportanceResult(
        mode="torch_pruning_l2",
        normalization="none",
        aggregation="shared_local_position_mean",
        raw_scores=raw,
        normalized_scores=dict(raw),
        unit_parameter_costs={stable_id: 0 for stable_id in raw},
        unit_scores=mapping_rows,
        task_loss="not_applicable_weight_magnitude",
        implementation_version="torch-pruning-1.6.0-magnitude-p2-root-only-v1",
    )
    return {
        "importance": importance,
        "modules": module_reports,
        "channel_mapping": mapping_rows,
        "torch_pruning_api_used": bool(module_reports)
        and all(bool(row["torch_pruning_api_used"]) for row in module_reports),
        "tp_dependency_graph_used": False,
        "tp_physical_materialization_used": False,
    }


def build_group_score_records(
    atomic_units: Sequence[Any],
    importance: ImportanceResult,
    request: SamplingPruningRequest,
    *,
    strategy: str,
) -> list[dict[str, Any]]:
    """Record every local score, rank and final keep/prune decision."""

    root_entries = {
        row.module_path: row
        for row in request.entries
        if not bool(row.metadata.get("dependency_driven", False))
        and row.axis in {"out", "channel"}
        and row.group_keep_map
    }
    records_by_root: dict[str, list[dict[str, Any]]] = {}
    for atom in atomic_units:
        root = str(atom.root_module_path)
        entry = root_entries.get(root)
        if entry is None:
            continue
        if len(atom.root_indices) != 1:
            raise ValueError(f"group score atom must own one root index: {atom.stable_id}")
        constraints = dict(atom.constraints)
        width = int(constraints.get("channels_per_group") or 0)
        absolute = int(atom.root_indices[0])
        group_id, local_position = divmod(absolute, width)
        source_ids = [str(value) for value in atom.source_coupled_unit_ids]
        raw_scores = [float(importance.raw_scores[value]) for value in source_ids]
        normalized_scores = [float(importance.normalized_scores[value]) for value in source_ids]
        kept = local_position in set(entry.group_keep_map[group_id])
        records_by_root.setdefault(root, []).append(
            {
                "module_path": root,
                "scope_id": str(atom.scope_id),
                "atomic_unit_id": str(atom.stable_id),
                "source_coupled_unit_ids": source_ids,
                "group_id": group_id,
                "local_position": local_position,
                "absolute_channel_index": absolute,
                "score": sum(normalized_scores) / len(normalized_scores),
                "raw_score": sum(raw_scores) / len(raw_scores),
                "kept": kept,
                "pruned": not kept,
            }
        )
    output: list[dict[str, Any]] = []
    for root in sorted(records_by_root):
        rows = records_by_root[root]
        by_group: dict[int, list[dict[str, Any]]] = {}
        for row in rows:
            by_group.setdefault(int(row["group_id"]), []).append(row)
        local_positions = sorted({int(row["local_position"]) for row in rows})
        shared = {
            local: sum(
                next(float(row["score"]) for row in group_rows if int(row["local_position"]) == local)
                for group_rows in by_group.values()
            )
            / len(by_group)
            for local in local_positions
        }
        shared_order = sorted(local_positions, key=lambda local: (-shared[local], local))
        shared_rank = {local: rank for rank, local in enumerate(shared_order, start=1)}
        for group_id, group_rows in sorted(by_group.items()):
            order = sorted(
                group_rows,
                key=lambda row: (-float(row["score"]), int(row["local_position"])),
            )
            within_rank = {
                int(row["local_position"]): rank
                for rank, row in enumerate(order, start=1)
            }
            for row in sorted(group_rows, key=lambda item: int(item["local_position"])):
                local = int(row["local_position"])
                output.append(
                    {
                        **row,
                        "rank_within_group": within_rank[local],
                        "shared_local_mean_score": shared[local],
                        "shared_rank": shared_rank[local],
                        "strategy": strategy,
                        "torch_pruning_api_used": strategy
                        == "torch_pruning_l2_shared_position",
                    }
                )
    return output


def tensor_output_contract(value: Any) -> dict[str, Any]:
    shapes: dict[str, list[int]] = {}
    dtypes: dict[str, str] = {}
    non_finite: list[str] = []

    def visit(item: Any, path: str) -> None:
        if torch.is_tensor(item):
            shapes[path] = list(item.shape)
            dtypes[path] = str(item.dtype)
            if item.is_floating_point() or item.is_complex():
                if not bool(torch.isfinite(item).all().item()):
                    non_finite.append(path)
            return
        if isinstance(item, Mapping):
            for key in sorted(item, key=str):
                visit(item[key], f"{path}.{key}" if path else str(key))
            return
        if isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                visit(child, f"{path}.{index}" if path else str(index))

    visit(value, "")
    return {
        "tensor_shapes": shapes,
        "tensor_dtypes": dtypes,
        "tensor_count": len(shapes),
        "all_finite": not non_finite,
        "non_finite_tensor_paths": sorted(non_finite),
    }


def _snapshot_parameter_counts(snapshot: Mapping[str, Any]) -> dict[str, int]:
    modules = snapshot.get("modules", [])
    if isinstance(modules, Mapping):
        return {
            str(name): int(dict(row).get("parameter_count", 0))
            for name, row in modules.items()
        }
    return {
        str(row.get("canonical_module_name", row.get("module_path", ""))): int(
            row.get("parameter_count", 0)
        )
        for row in modules
    }


def parameter_reduction_breakdown(
    original_snapshot: Mapping[str, Any],
    candidate_snapshot: Mapping[str, Any],
    *,
    active_roots: set[str],
    closure_modules: set[str],
) -> dict[str, Any]:
    before = _snapshot_parameter_counts(original_snapshot)
    after = _snapshot_parameter_counts(candidate_snapshot)
    reductions = {
        module: int(before.get(module, 0)) - int(after.get(module, 0))
        for module in closure_modules
    }
    active = sum(value for module, value in reductions.items() if module in active_roots)
    dependency = sum(value for module, value in reductions.items() if module not in active_roots)
    return {
        "active_root_parameter_reduction": active,
        "dependency_driven_parameter_reduction": dependency,
        "total_parameter_reduction_in_closure": active + dependency,
        "closure_module_count": len(closure_modules),
        "active_root_count": len(active_roots),
        "module_parameter_reductions": reductions,
    }


def build_controlled_strategy_request(
    atomic_units: Sequence[Any],
    importance: ImportanceResult,
    *,
    allowed_roots: set[str],
    target_width: int,
    strategy: str,
) -> dict[str, Any]:
    """Select every whitelisted grouped scope at one exact per-group width."""

    filtered = filter_active_root_units(atomic_units, allowed_roots)
    roots_found = {str(unit.root_module_path) for unit in filtered}
    missing = sorted(set(allowed_roots) - roots_found)
    if missing:
        raise ValueError(f"missing grouped root units: {missing}")
    if not filtered:
        raise ValueError("root whitelist selected no atomic units")
    if strategy == "taylor_independent_group_ranking":
        selection_policy = GroupedConvSelectionPolicy.INDEPENDENT_GROUP_TOPK
    elif strategy == "torch_pruning_l2_shared_position":
        selection_policy = GroupedConvSelectionPolicy.SHARED_LOCAL_MEAN
    else:
        raise ValueError(f"unknown grouped-conv strategy: {strategy}")

    root_shapes: dict[str, tuple[int, int]] = {}
    for unit in filtered:
        constraints = dict(getattr(unit, "constraints", {}))
        if not bool(constraints.get("grouped_conv")) or bool(constraints.get("depthwise")):
            raise ValueError(f"whitelisted unit is not a non-depthwise grouped Conv: {unit.stable_id}")
        groups = int(constraints.get("groups") or 0)
        width = int(constraints.get("channels_per_group") or 0)
        root = str(unit.root_module_path)
        shape = (groups, width)
        if groups <= 0 or width <= int(target_width):
            raise ValueError(
                f"target width {target_width} is not smaller than grouped root {root} shape {shape}"
            )
        previous = root_shapes.setdefault(root, shape)
        if previous != shape:
            raise ValueError(f"inconsistent grouped metadata for {root}: {previous} vs {shape}")

    channel_budget = sum(
        groups * (width - int(target_width))
        for groups, width in root_shapes.values()
    )
    request = select_pruning_request(
        filtered,
        importance_result=importance,
        channel_budget=channel_budget,
        grouped_config=GroupedConvConfig(
            allowed_channels_per_group=(int(target_width),),
            selection_policy=selection_policy,
        ),
        selection_config=SelectionConfig(
            per_domain_max_sparsity=1.0,
            minimum_retained_channels=4,
        ),
        alignment_config=AlignmentConfig(),
    )
    active_roots = sorted(
        {
            row.module_path
            for row in request.entries
            if not bool(row.metadata.get("dependency_driven", False))
            and row.axis in {"out", "channel"}
        }
    )
    if active_roots != sorted(allowed_roots):
        raise ValueError(
            f"formal selector did not select every whitelisted root: expected={sorted(allowed_roots)}, "
            f"observed={active_roots}"
        )
    if int(request.requested_channel_cost) != channel_budget:
        raise ValueError(
            f"formal selector channel cost mismatch: expected={channel_budget}, "
            f"observed={request.requested_channel_cost}"
        )
    return {
        "request": request,
        "selected_active_roots": active_roots,
        "allowed_roots": sorted(allowed_roots),
        "filtered_atomic_unit_count": len(filtered),
        "target_width": int(target_width),
        "requested_channel_cost": channel_budget,
        "selection_policy": selection_policy.value,
        "formal_selector": "pruning.api.select_pruning_request",
        "formal_dependency_closure_preserved": True,
    }


__all__ = [
    "build_controlled_strategy_request",
    "build_frame_manifest",
    "build_group_score_records",
    "build_tp_importance_result",
    "parameter_reduction_breakdown",
    "state_dict_content_hash",
    "tensor_output_contract",
    "validate_core_freeze",
]
